"""FVM solver for hyperbolic PDE systems (numpy backend).

Uses the symbolic Riemann solver (riemann_solvers.py) for flux computation.
No dependency on legacy flux.py or nonconservative_flux.py.

Solver hierarchy:
    Solver (base: init, create_runtime, BCs)
      -> HyperbolicSolver (explicit time stepping + symbolic Riemann flux)
           -> setup_simulation(mesh, model)
           -> run_simulation() -> Q, Qaux
           -> step(dt): apply_bcs -> reconstruct -> flux -> ode_step -> update_state
"""

import os
from time import time as gettime

import numpy as np
import param
import sympy as sp

from zoomy_core.misc.logger_config import logger
import zoomy_core.misc.io as io
from zoomy_core.misc.misc import Zstruct, Settings
import zoomy_core.fvm.ode as ode
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel, NumpyRuntimeSymbolic
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.fvm.riemann_solvers import (
    PositiveNonconservativeRusanov,
    NonconservativeRusanov,
)


_EMPTY_AUX = np.array([])
_EMPTY_AUX2 = np.empty((0, 0))


# -- Field detection helpers ---------------------------------------------------

def _var_index(model, name, fallback=None):
    keys = list(model.variables.keys())
    if name in keys:
        return keys.index(name)
    if fallback is not None:
        return fallback
    raise KeyError(f"Variable '{name}' not found in model variables: {keys}")


def _param_value(model, name, default=None):
    # SystemModel keeps numeric values in ``parameter_values`` (``parameters``
    # holds Symbols); a Model keeps values in ``parameters``.
    pv = getattr(model, "parameter_values", None)
    if pv is not None and hasattr(pv, "contains") and pv.contains(name):
        return float(getattr(pv, name))
    params = getattr(model, "parameters", None)
    if params is not None and hasattr(params, "contains") and params.contains(name):
        try:
            return float(getattr(params, name))
        except (TypeError, ValueError):
            return default
    return default


def _detect_scaled_q_indices(model):
    if hasattr(model, "numerics_scaled_q_indices"):
        return model.numerics_scaled_q_indices
    keys = list(model.variables.keys())
    excluded = set()
    for name in ["b", "h"]:
        if name in keys:
            excluded.add(keys.index(name))
    return [i for i in range(model.n_variables) if i not in excluded]


# -- Boundary condition helpers ------------------------------------------------

def _compute_bf_face_gradients(Q, Qaux, bc_indices, bc_grad_fn, bf_cells,
                               bf_fidx, d_face, normals_arr, face_centers,
                               n_bf, n_vars, has_aux, time, parameters):
    """Compute per-variable boundary face-normal gradients via the
    indexed ``boundary_gradients(bc_idx, time, position, distance, Q,
    Qaux, p, normal)`` kernel.

    Returns dict ``{var_index: ndarray(n_bf,)}`` for use by
    ``DiffusionOperatorV2``."""
    bf_grads = {v: np.zeros(n_bf) for v in range(n_vars)}
    for i_bf in range(n_bf):
        q_inner = Q[:, bf_cells[i_bf]]
        qaux_inner = Qaux[:, bf_cells[i_bf]] if has_aux else np.array([])
        fidx = bf_fidx[i_bf]
        normal = normals_arr[:, fidx]
        position = face_centers[fidx, :]
        fg = bc_grad_fn(
            bc_indices[i_bf], time, position, d_face[i_bf],
            q_inner, qaux_inner, parameters, normal,
        )
        for v in range(n_vars):
            bf_grads[v][i_bf] = fg[v]
    return bf_grads


# -- Helpers -------------------------------------------------------------------

def _coerce_to_system_model(model):
    """Return a :class:`SystemModel` view of ``model``.

    Accepts a :class:`Model` (calls :meth:`SystemModel.from_model`),
    a :class:`SystemModel` (returned unchanged), or a
    :class:`NumericalSystemModel` (returns ``nsm.sm``). This is the
    canonical source of truth for ``state`` / ``aux_state`` sizing
    after :meth:`SystemModel.from_model.expose_aux_atoms` has run.
    """
    if isinstance(model, NumericalSystemModel):
        return model.sm
    if isinstance(model, SystemModel):
        return model
    return SystemModel.from_model(model)


# -- Base Solver ---------------------------------------------------------------

class Solver(param.Parameterized):
    """Base solver class: initialization, runtime creation, boundary conditions."""

    settings = param.Parameter(default=None, doc="Run configuration (Settings object)")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.settings is None:
            self.settings = Settings.default()
        else:
            defaults = Settings.default()
            defaults.update(self.settings)
            self.settings = defaults

    def initialize(self, mesh, model):
        """Allocate (Q, Qaux) of shape ``(n_state, n_inner_cells)``
        and ``(n_aux_state, n_inner_cells)``.

        The aux size MUST come from the SystemModel's ``aux_state``,
        not from the source Model's ``aux_variables``. Canonical
        derivation classes (SME, VAM, MLME, MLVAM) leave
        ``aux_variables`` empty and rely on
        :meth:`SystemModel.from_model` ``expose_aux_atoms`` to
        auto-promote bathymetry / derivative / function atoms into
        ``sm.aux_state``. The Model-side count is therefore wrong
        for any model that uses auto-promotion. We normalize to a
        SystemModel here so the allocation matches the runtime's
        view (which is also SystemModel-based).
        """
        nc = mesh.n_inner_cells
        # Resolve the SystemModel-side view of state + aux. Accepts a
        # Model (auto-promote), a SystemModel (use directly), or an NSM
        # (unwrap to ``nsm.sm``).
        sm = _coerce_to_system_model(model)
        n_variables = len(sm.state)
        n_aux_variables = len(sm.aux_state)
        Q = np.empty((n_variables, nc), dtype=float)
        Qaux = np.empty((n_aux_variables, nc), dtype=float)
        return Q, Qaux

    def create_runtime(self, Q, Qaux, mesh, model):
        """Build the numpy runtime from a :class:`SystemModel`.

        Contract: ``model`` is a :class:`SystemModel` — the
        self-contained numerical model.  Every ``setup_simulation``
        normalises its input to a SystemModel before reaching here.
        The runtime comes from
        :meth:`NumpyRuntimeModel.from_system_model`; the numeric
        parameter array from ``parameter_values``.  Numerical
        regularisation, if wanted, is a separate
        ``SystemModel → SystemModel`` pass applied *before* this point.
        """
        Q, Qaux = np.asarray(Q), np.asarray(Qaux)
        parameters = np.array(
            list(model.parameter_values.values()), dtype=float)
        runtime_model = NumpyRuntimeModel.from_system_model(model)
        return Q, Qaux, parameters, mesh, runtime_model

    def update_q(self, Q, Qaux, mesh, model, parameters):
        """Apply ``model.update_variables`` (h-clamp, momentum ramp) at
        each cell.

        ``update_variables`` is carried through the SystemModel and
        exposed on the runtime — the identity for models with no
        per-cell transform.  It is ``None`` only for SystemModels
        assembled directly without one (e.g. split sub-systems); then
        this is a genuine no-op, not a legacy fallback."""
        update = getattr(model, "update_variables", None)
        if update is None:
            return Q
        n_vars = Q.shape[0]
        for c in range(Q.shape[1]):
            aux = Qaux[:, c] if Qaux.shape[0] > 0 else _EMPTY_AUX
            Q[:, c] = np.asarray(
                update(Q[:, c], aux, parameters), dtype=float,
            ).ravel()[:n_vars]
        return Q

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        """Default: walk ``model._chain_systemmodel.aux_registry`` (or
        ``self._sm.aux_registry`` if set up that way) and fill every
        ``kind == 'derivative'`` row via ``LSQMesh.compute_derivatives``
        on the underlying source field (state Q for state derivatives,
        Qaux for derivatives of function-aux entries).

        Subclasses override to supply the ``kind == 'function'`` rows
        (e.g. user-supplied bathymetry, time-dependent forcing) and
        call ``super().update_qaux(...)`` to handle the derivative
        part.

        No-op if no SystemModel / registry is attached.
        """
        sm = self._sm_from_solver_or_model(model)
        if sm is None:
            return Qaux
        registry = getattr(sm, "aux_registry", None) or []
        if not registry:
            return Qaux
        nc = Q.shape[1]
        n_cells = mesh.n_cells
        u_full = np.zeros(n_cells)
        for entry in registry:
            if entry["kind"] != "derivative":
                continue
            row = entry["row"]
            mi = entry["multi_index"]
            tk = entry["target_kind"]
            if tk == "state":
                u_full[:nc] = Q[entry["state_index"], :]
            elif tk == "function":
                u_full[:nc] = Qaux[entry["function_row"], :]
            else:
                # Unknown target — leave whatever the caller put there.
                continue
            # TODO(boundary-aware-LSQ): supply BC-evaluated face values
            # for state-aux derivatives (analogous to
            # ChorinSplitVAMSolver._compute_boundary_face_state).  For
            # now the predictor's parent uses extrapolation = Neumann-
            # zero, matching the legacy behaviour at boundary cells.
            d = mesh.compute_derivatives(
                u_full, degree=max(mi), derivatives_multi_index=[mi],
                u_boundary_face="extrapolation",
            )
            Qaux[row, :] = d[:nc, 0]
        return Qaux

    def _sm_from_solver_or_model(self, model):
        """Look up the SystemModel: prefer ``self.sm`` (when the
        solver was built that way), else ``model._chain_systemmodel``
        if present, else None."""
        if hasattr(self, "sm") and self.sm is not None:
            return self.sm
        return getattr(model, "_chain_systemmodel", None)


# -- HyperbolicSolver ----------------------------------------------------------

class HyperbolicSolver(Solver):
    """Explicit time-stepping solver using the symbolic Riemann solver.

    Core methods:
        setup_simulation(mesh, model) -- build all operators once
        run_simulation()              -- time loop: compute_dt -> step -> output
        step(dt)                      -- one timestep (readable, no if-clauses)
        solve(mesh, model)            -- convenience: setup + run
    """

    time_end = param.Number(default=0.1, bounds=(0, None), doc="Simulation end time")
    min_dt = param.Number(default=1e-6, bounds=(0, None), doc="Minimum allowed timestep")
    compute_dt = param.Parameter(default=None, doc="Time-stepping strategy (callable)")
    _diffusion_in_flux = True  # explicit diffusion in flux operator
    # Numerical knobs (``reconstruction.order``, ``reconstruction.limiter``,
    # ``regularization.eigenvalue_eps``) live on the
    # :class:`NumericalSystemModel`; the solver reads ``self.nsm.*`` after
    # ``setup_simulation``.  Constructors no longer accept those kwargs —
    # pass an NSM (or a Model / SystemModel that gets auto-promoted to
    # one with default specs).

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.compute_dt is None:
            self.compute_dt = timestepping.adaptive(CFL=0.9)
        defaults = Settings.default()
        defaults.output.update(Zstruct(snapshots=10))
        defaults.update(self.settings)
        self.settings = defaults

    def initialize(self, mesh, model):
        Q, Qaux = super().initialize(mesh, model)
        nc = mesh.n_inner_cells
        Q = model.initial_conditions.apply(mesh.cell_centers[:, :nc], Q)
        Qaux = model.aux_initial_conditions.apply(mesh.cell_centers[:, :nc], Qaux)
        return Q, Qaux

    # -- NSM coercion --------------------------------------------------

    def _coerce_to_nsm(self, model):
        """Normalise ``model`` to a (NumericalSystemModel, source-Model)
        pair.  Auto-promotes Model / SystemModel by building an NSM
        with default specs (``ReconstructionSpec(order=1)``,
        ``RegularizationSpec(eigenvalue_eps=1e-8)``, etc.) — to override
        defaults, build the NSM explicitly and pass that in.

        ``source_model`` is the *original* :class:`Model` when one was
        passed (needed for ``resolve_periodic_bcs``, which walks
        ``boundary_conditions_list`` — a field that lives on the Model
        object and not on the lambdified SystemModel BC kernel).  When
        the caller supplied a bare SystemModel or NSM, ``source_model``
        is ``None``."""
        if isinstance(model, NumericalSystemModel):
            return model, None
        nsm = NumericalSystemModel.from_system_model(model)
        source_model = None if isinstance(model, SystemModel) else model
        return nsm, source_model

    # -- Symbolic model helpers ----------------------------------------

    def _get_symbolic_model(self, model):
        return model.model if hasattr(model, "model") else model

    def _get_dry_threshold(self, symbolic_model):
        return _param_value(symbolic_model, "eps_wet", default=1e-3)

    # -- Max wavespeed (for CFL + Rusanov dissipation) -----------------

    def _build_max_wavespeed(self, symbolic_model):
        eig_mode = getattr(symbolic_model, "eigenvalue_mode", "symbolic")
        n_vars = symbolic_model.n_variables
        n_aux = symbolic_model.n_aux_variables
        n_params = symbolic_model.n_parameters
        dim = symbolic_model.dimension

        if eig_mode != "numerical":
            rt = NumpyRuntimeModel.from_system_model(symbolic_model)
            compiled_eig = rt.eigenvalues
            def max_ws(*args):
                Q = np.array(args[:n_vars])
                Qaux = np.array(args[n_vars:n_vars + n_aux])
                p = np.array(args[n_vars + n_aux:n_vars + n_aux + n_params])
                n = np.array(args[n_vars + n_aux + n_params:])
                evs = np.asarray(compiled_eig(Q, Qaux, p, n), dtype=float).ravel()
                return float(np.max(np.abs(evs)))
            return max_ws

        rt = NumpyRuntimeModel.from_system_model(symbolic_model)
        ql_fn = rt.quasilinear_matrix
        keys = list(symbolic_model.variables.keys())
        fi_h = keys.index("h") if "h" in keys else None
        dry_thr = self._get_dry_threshold(symbolic_model) if fi_h is not None else 0.0
        eps_reg = self.nsm.regularization.eigenvalue_eps
        reg_diag = eps_reg * np.eye(n_vars)
        if fi_h is not None and "b" in keys:
            reg_diag[keys.index("b"), keys.index("b")] = 0.0

        def max_ws_numerical(*args):
            # BATCH-AWARE: scalar args → single 4×4 eigvals (legacy);
            # face-array args (the vectorized Riemann kernels broadcast the
            # whole flux expression over faces) → ONE stacked eigvals call.
            a0 = np.asarray(args[0])
            if a0.ndim == 0:
                Q = np.array(args[:n_vars])
                Qaux = np.array(args[n_vars:n_vars + n_aux])
                p = np.array(args[n_vars + n_aux:n_vars + n_aux + n_params])
                n_vec = np.array(args[n_vars + n_aux + n_params:])
                if fi_h is not None and Q[fi_h] < dry_thr:
                    return 0.0
                ql = np.asarray(ql_fn(Q, Qaux, p), dtype=float).reshape(
                    n_vars, n_vars, dim)
                A_n = sum(ql[:, :, d] * float(n_vec[d]) for d in range(dim))
                A_n += reg_diag
                evs = np.real(np.linalg.eigvals(A_n))
                return float(np.max(np.abs(evs)))

            nf = a0.shape[-1]
            Q = np.stack([np.broadcast_to(np.asarray(a, dtype=float), (nf,))
                          for a in args[:n_vars]])
            Qaux = (np.stack([np.broadcast_to(np.asarray(a, dtype=float), (nf,))
                              for a in args[n_vars:n_vars + n_aux]])
                    if n_aux else np.zeros((0, nf)))
            p = np.array([float(np.asarray(a).ravel()[0]) for a in
                          args[n_vars + n_aux:n_vars + n_aux + n_params]])
            n_vec = np.stack([np.broadcast_to(np.asarray(a, dtype=float), (nf,))
                              for a in args[n_vars + n_aux + n_params:]])
            ql = np.asarray(ql_fn(Q, Qaux, p), dtype=float)
            if ql.shape == (n_vars, n_vars, nf, dim):
                ql = np.transpose(ql, (0, 1, 3, 2))
            A_n = np.einsum("ijdk,dk->kij", ql, n_vec) + reg_diag[None]
            evs = np.real(np.linalg.eigvals(A_n))
            out = np.abs(evs).max(axis=1)
            if fi_h is not None:
                out[Q[fi_h] < dry_thr] = 0.0
            return out

        return max_ws_numerical

    def get_compute_max_abs_eigenvalue(self, mesh, model):
        symbolic_model = self._get_symbolic_model(model)
        max_ws = self._build_max_wavespeed(symbolic_model)
        keys = list(symbolic_model.variables.keys())
        fi_h = keys.index("h") if "h" in keys else None
        dry_thr = self._get_dry_threshold(symbolic_model) if fi_h is not None else 0.0
        dim = symbolic_model.dimension
        normals = mesh.face_normals[:dim, :]
        has_aux = symbolic_model.n_aux_variables > 0

        nc = mesh.n_inner_cells
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]

        # Split faces into interior and boundary
        int_mask = (iA < nc) & (iB < nc)
        bnd_mask = ~int_mask
        interior_faces = np.where(int_mask)[0]
        boundary_faces = np.where(bnd_mask)[0]
        iA_int = iA[interior_faces]
        iB_int = iB[interior_faces]
        # face_cells[0] is guaranteed to be the inner cell at boundary faces
        iInner_bnd = iA[boundary_faces]

        eig_mode = getattr(symbolic_model, "eigenvalue_mode", "symbolic")
        if eig_mode == "numerical":
            # BATCHED numerical wave speeds: one vectorized quasilinear
            # evaluation over all cells + one stacked np.linalg.eigvals over
            # all face-side matrices.  The per-face Python loop below costs
            # ~25 ms/step at 200 cells (it dominated the whole time loop);
            # the batched path is two orders of magnitude cheaper.
            rt = NumpyRuntimeModel.from_system_model(symbolic_model)
            ql_fn = rt.quasilinear_matrix
            n_vars = symbolic_model.n_variables
            eps_reg = self.nsm.regularization.eigenvalue_eps
            keys = list(symbolic_model.variables.keys())
            reg = eps_reg * np.eye(n_vars)
            if fi_h is not None and "b" in keys:
                reg[keys.index("b"), keys.index("b")] = 0.0
            n_int = normals[:, interior_faces]          # (dim, n_if)
            n_bnd = normals[:, boundary_faces]

            def compute_max_eigenvalue(Q, Qaux, parameters):
                ql = np.asarray(ql_fn(Q, Qaux, parameters), dtype=float)
                nc_ = Q.shape[1]
                if ql.shape == (n_vars, n_vars, nc_, dim):
                    # vectorize wrapper appends the cell axis before dim
                    ql = np.transpose(ql, (0, 1, 3, 2))
                elif ql.shape != (n_vars, n_vars, dim, nc_):
                    raise ValueError(
                        f"unexpected quasilinear shape {ql.shape}")
                # (n, n, dim, nc); A·n per face side → (n_faces_side, n, n)
                A_A = np.einsum("ijdk,dk->kij", ql[:, :, :, iA_int], n_int)
                A_B = np.einsum("ijdk,dk->kij", ql[:, :, :, iB_int], n_int)
                A_I = np.einsum("ijdk,dk->kij", ql[:, :, :, iInner_bnd], n_bnd)
                A_all = np.concatenate([A_A, A_B, A_I], axis=0) + reg[None]
                evs = np.real(np.linalg.eigvals(A_all))       # batched LAPACK
                m = np.abs(evs).max(axis=1)
                n_if = len(interior_faces)
                max_ev = np.zeros(mesh.n_faces)
                max_ev[interior_faces] = np.maximum(m[:n_if], m[n_if:2 * n_if])
                max_ev[boundary_faces] = m[2 * n_if:]
                if fi_h is not None:                          # dry-cell skip
                    both_dry = ((Q[fi_h, iA_int] < dry_thr)
                                & (Q[fi_h, iB_int] < dry_thr))
                    max_ev[interior_faces[both_dry]] = 0.0
                    max_ev[boundary_faces[Q[fi_h, iInner_bnd] < dry_thr]] = 0.0
                return max_ev
            return compute_max_eigenvalue

        def compute_max_eigenvalue(Q, Qaux, parameters):
            max_ev = np.zeros(mesh.n_faces)

            # Interior faces: evaluate at both cells
            for fi in range(len(interior_faces)):
                f = interior_faces[fi]
                if fi_h is not None:
                    if Q[fi_h, iA_int[fi]] < dry_thr and Q[fi_h, iB_int[fi]] < dry_thr:
                        continue
                n = normals[:, f]
                for i_cell in [iA_int[fi], iB_int[fi]]:
                    q = Q[:, i_cell]
                    qaux = Qaux[:, i_cell] if has_aux else _EMPTY_AUX
                    ev = max_ws(*q, *qaux, *parameters, *n)
                    max_ev[f] = max(max_ev[f], ev)

            # Boundary faces: evaluate at inner cell only
            for bi in range(len(boundary_faces)):
                f = boundary_faces[bi]
                cInner = iInner_bnd[bi]
                if fi_h is not None and Q[fi_h, cInner] < dry_thr:
                    continue
                n = normals[:, f]
                q = Q[:, cInner]
                qaux = Qaux[:, cInner] if has_aux else _EMPTY_AUX
                ev = max_ws(*q, *qaux, *parameters, *n)
                max_ev[f] = max(max_ev[f], ev)

            return max_ev
        return compute_max_eigenvalue

    # -- Flux operator (symbolic Riemann solver) -----------------------

    def _build_numerics(self, symbolic_model):
        """Build the symbolic Riemann solver. Override for SWE-specific variants."""
        return NonconservativeRusanov(symbolic_model)

    def _build_diffusion_operators(self, mesh, symbolic_model, dim, n_vars):
        """Build DiffusionOperatorV2 per variable when the SystemModel
        carries a non-zero ``diffusion_matrix`` and positive viscosity.

        ``diffusion_matrix`` is a stored SystemModel field — a rank-4
        tensor ``A(Q, Qaux, p)`` of shape ``(n_eq, n_state, n_dim, n_dim)``
        carried over from the Model by ``from_model``.  The numpy
        diffusion backend is the (legacy) scalar-viscosity path; it
        triggers when ``A`` has any symbolic non-zero entry and the
        model exposes a positive ``nu`` parameter."""
        sym_A = getattr(symbolic_model, 'diffusion_matrix', None)
        if sym_A is None:
            return None
        # Rank-4 NDimArray: scan every entry for a non-zero atom.
        if all(sp.simplify(e) == 0 for e in sp.flatten(sym_A)):
            return None
        from zoomy_core.fvm.reconstruction import DiffusionOperatorV2
        nu_val = _param_value(symbolic_model, "nu", default=0.0)
        if nu_val <= 0:
            return None
        return {v: DiffusionOperatorV2(mesh, dim, nu=nu_val) for v in range(n_vars)}

    def _build_reconstruction(self, mesh, symbolic_model):
        """Build ghost-cell-free face reconstruction."""
        from zoomy_core.fvm.reconstruction import (
            ConstantReconstructionV2, LSQMUSCLReconstruction,
        )
        dim = symbolic_model.dimension
        if self.nsm.reconstruction.order >= 2:
            return LSQMUSCLReconstruction(
                mesh, dim, limiter=self.nsm.reconstruction.limiter)
        return ConstantReconstructionV2(mesh, dim)

    def get_flux_operator(self, mesh, model):
        symbolic_model = self._get_symbolic_model(model)
        max_wavespeed_fn = self._build_max_wavespeed(symbolic_model)
        dim = symbolic_model.dimension

        numerics = self._build_numerics(symbolic_model)
        NumpyRuntimeSymbolic.module["max_wavespeed"] = max_wavespeed_fn
        # Let the numerics declare any extra backend kernels it needs (e.g.
        # the Roe |A| dissipation), merged in before lambdification.
        extra_kernels = getattr(numerics, "runtime_kernels", None)
        if callable(extra_kernels):
            for _kname, _kfn in extra_kernels(symbolic_model).items():
                NumpyRuntimeSymbolic.module[_kname] = _kfn
        runtime_numerics = numerics.to_runtime_numpy()
        runtime_numerics.local_max_abs_eigenvalue = (
            lambda Q, Qaux, p, n: max_wavespeed_fn(*Q, *Qaux, *p, *n)
        )
        use_batched = bool(getattr(numerics, "supports_batched_faces", False))

        nc = mesh.n_inner_cells
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        normals_arr = mesh.face_normals[:dim, :]
        face_volumes = mesh.face_volumes
        cell_volumes = mesh.cell_volumes
        n_vars = symbolic_model.n_variables
        has_aux = symbolic_model.n_aux_variables > 0

        # Build reconstruction (ghost-cell-free)
        reconstruct = self._build_reconstruction(mesh, symbolic_model)

        # Runtime non-conservative matrix B(Q) for the cell-interior NCP
        # integral (path-conservative consistency at order ≥ 2).  Built only
        # when a higher-order reconstruction carries a slope — order 1 has none.
        ncm = None
        if self.nsm.reconstruction.order >= 2:
            from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
            ncm = NumpyRuntimeModel.from_system_model(
                symbolic_model).nonconservative_matrix

        # Build diffusion operators (ghost-cell-free)
        self._diffusion_ops = self._build_diffusion_operators(mesh, symbolic_model, dim, n_vars)

        # Precompute face index sets — split into interior and boundary
        bf_face_set = set(mesh.boundary_face_face_indices)
        interior_faces = np.array(
            [f for f in range(mesh.n_faces) if f not in bf_face_set], dtype=int)
        boundary_faces = mesh.boundary_face_face_indices.copy()
        iA_int = iA[interior_faces]
        iB_int = iB[interior_faces]
        # THIS-side cell at boundary faces (for reconstruction + flux accumulation).
        # Use face_cells[0] (guaranteed by normalization), NOT boundary_face_cells
        # (which may be remapped to the opposite cell for periodic BCs).
        iInner_bnd = iA[boundary_faces]  # = face_cells[0] at boundary faces
        cvA_int = cell_volumes[iA_int]
        cvB_int = cell_volumes[iB_int]
        cvInner_bnd = cell_volumes[iInner_bnd]
        fv_int = face_volumes[interior_faces]
        fv_bnd = face_volumes[boundary_faces]

        # Per-boundary-face indices into the indexed BC function +
        # precompute d_face.  No Python BC-object list — the BC is
        # ``runtime_model.boundary_conditions(bc_idx, time, position,
        # distance, Q_cell, Qaux_cell, parameters, normal) → q_face``.
        n_bf = mesh.n_boundary_faces
        bf_cells = mesh.boundary_face_cells
        bf_fidx = mesh.boundary_face_face_indices
        bc_indices = np.asarray(
            mesh.boundary_face_function_numbers[:n_bf], dtype=int)
        bc_fn = model.boundary_conditions
        face_centers = mesh.face_centers

        d_face = np.array([
            np.linalg.norm(
                mesh.face_centers[bf_fidx[i], :dim] - mesh.cell_centers[:dim, bf_cells[i]]
            )
            for i in range(n_bf)
        ]) if n_bf > 0 else np.array([])

        # Store BC metadata for IMEX access — both kernels live on the
        # runtime; diffusion path uses the gradient one.
        self._bc_indices = bc_indices
        self._bc_fn = bc_fn
        self._bc_grad_fn = model.boundary_gradients
        self._bf_cells = bf_cells
        self._bf_fidx = bf_fidx
        self._d_face = d_face
        self._n_bf = n_bf

        def flux_operator(dt, time, Q, Qaux, parameters, dQ):
            dQ = np.zeros((n_vars, nc))

            # 1. Compute boundary face values via the indexed BC kernel.
            bf_values = np.zeros((n_vars, n_bf))
            for i_bf in range(n_bf):
                q_inner = Q[:, bf_cells[i_bf]]
                qaux_inner = Qaux[:, bf_cells[i_bf]] if has_aux else _EMPTY_AUX
                fidx = bf_fidx[i_bf]
                normal = normals_arr[:, fidx]
                position = face_centers[fidx, :]
                bf_values[:, i_bf] = bc_fn(
                    bc_indices[i_bf], time, position, d_face[i_bf],
                    q_inner, qaux_inner, parameters, normal,
                )

            # 2. Reconstruct (uses bf_values for limiter bounds + Q_R placeholder)
            Q_L, Q_R = reconstruct(Q, bf_values)

            # 3. Override Q_R at boundary faces with BC(Q_L).
            for i_bf in range(n_bf):
                fidx = bf_fidx[i_bf]
                qaux_inner = Qaux[:, bf_cells[i_bf]] if has_aux else _EMPTY_AUX
                normal = normals_arr[:, fidx]
                position = face_centers[fidx, :]
                Q_R[:, fidx] = bc_fn(
                    bc_indices[i_bf], time, position, d_face[i_bf],
                    Q_L[:, fidx], qaux_inner, parameters, normal,
                )

            # 3b. Cell-interior non-conservative integral (path-conservative,
            # order ≥ 2): ∫_cell B(Q)·∂_x Q dx ≈ B(Q_c)·s_c, with B evaluated
            # at the cell-centre state and s_c the limited reconstruction slope
            # (depth row = ∂_x η − ∂_x b for free-surface).  The face
            # fluctuations below carry only the inter-cell jump; this is the
            # intra-cell smooth part.  Skipped exactly at order 1 (slope = 0).
            # REQUIRED for well-balancing at order ≥ 2 — it telescopes the
            # reconstructed hydrostatic-pressure variation against the bed
            # slope.  The |cell| factor cancels the per-unit-volume residual
            # normalisation, so there is no division by cell volume.
            grad = getattr(reconstruct, "_limited_grad", None)
            if ncm is not None and grad is not None and grad.any():
                for c in range(nc):
                    qa_c = Qaux[:, c] if has_aux else _EMPTY_AUX
                    B_c = np.asarray(ncm(Q[:, c], qa_c, parameters), dtype=float)
                    if B_c.ndim == 2 and dim == 1:
                        B_c = B_c[..., None]
                    for d in range(dim):
                        dQ[:, c] -= B_c[:, :, d] @ grad[:, d, c]

            # 4. BATCHED face Riemann solves: the lambdified kernels (and the
            # batch-aware max_wavespeed backend) broadcast over a trailing
            # face axis — one call per kernel instead of one per face.
            # Scatter with np.add.at (unbuffered) so repeated cell indices
            # accumulate correctly on unstructured meshes.  Numerics classes
            # opt IN via ``supports_batched_faces`` (verified bit-identical
            # for the plain NCP-Rusanov; the HR variants keep the loop until
            # their kernels are batch-validated).

            # 4a. Interior faces — both cells valid
            if use_batched and len(interior_faces):
                qA = Q_L[:, interior_faces]
                qB = Q_R[:, interior_faces]
                qauxA = Qaux[:, iA_int] if has_aux else _EMPTY_AUX2
                qauxB = Qaux[:, iB_int] if has_aux else _EMPTY_AUX2
                n_f = normals_arr[:, interior_faces]
                fluct = np.asarray(runtime_numerics.numerical_fluctuations(
                    qA, qB, qauxA, qauxB, parameters, n_f), dtype=float
                ).reshape(2 * n_vars, -1)
                num_flux = np.asarray(runtime_numerics.numerical_flux(
                    qA, qB, qauxA, qauxB, parameters, n_f), dtype=float
                ).reshape(n_vars, -1)
                Dp = fluct[:n_vars]
                Dm = fluct[n_vars:]
                np.add.at(dQ.T, iA_int,
                          (-(num_flux + Dm) * fv_int / cvA_int).T)
                np.add.at(dQ.T, iB_int,
                          (-(-num_flux + Dp) * fv_int / cvB_int).T)

            # 4b. Boundary faces — one inner cell only
            if use_batched and len(boundary_faces):
                qA = Q_L[:, boundary_faces]
                qB = Q_R[:, boundary_faces]
                qauxI = Qaux[:, iInner_bnd] if has_aux else _EMPTY_AUX2
                n_f = normals_arr[:, boundary_faces]
                fluct = np.asarray(runtime_numerics.numerical_fluctuations(
                    qA, qB, qauxI, qauxI, parameters, n_f), dtype=float
                ).reshape(2 * n_vars, -1)
                num_flux = np.asarray(runtime_numerics.numerical_flux(
                    qA, qB, qauxI, qauxI, parameters, n_f), dtype=float
                ).reshape(n_vars, -1)
                Dm = fluct[n_vars:]
                np.add.at(dQ.T, iInner_bnd,
                          (-(num_flux + Dm) * fv_bnd / cvInner_bnd).T)

            # 4-loop. Per-face fallback for numerics without batch support.
            if not use_batched:
                for fi in range(len(interior_faces)):
                    f = interior_faces[fi]
                    qA = Q_L[:, f]
                    qB = Q_R[:, f]
                    qauxA = Qaux[:, iA_int[fi]] if has_aux else _EMPTY_AUX
                    qauxB = Qaux[:, iB_int[fi]] if has_aux else _EMPTY_AUX
                    n = normals_arr[:, f]
                    fluct = np.asarray(
                        runtime_numerics.numerical_fluctuations(
                            qA, qB, qauxA, qauxB, parameters, n
                        ), dtype=float,
                    ).reshape(-1)
                    num_flux = np.asarray(
                        runtime_numerics.numerical_flux(
                            qA, qB, qauxA, qauxB, parameters, n
                        ), dtype=float,
                    ).reshape(-1)
                    Dp = fluct[:n_vars]
                    Dm = fluct[n_vars:]
                    dQ[:, iA_int[fi]] -= (num_flux + Dm) * fv_int[fi] / cvA_int[fi]
                    dQ[:, iB_int[fi]] -= (-num_flux + Dp) * fv_int[fi] / cvB_int[fi]
                for bi in range(len(boundary_faces)):
                    f = boundary_faces[bi]
                    qA = Q_L[:, f]
                    qB = Q_R[:, f]
                    qauxA = Qaux[:, iInner_bnd[bi]] if has_aux else _EMPTY_AUX
                    n = normals_arr[:, f]
                    fluct = np.asarray(
                        runtime_numerics.numerical_fluctuations(
                            qA, qB, qauxA, qauxA, parameters, n
                        ), dtype=float,
                    ).reshape(-1)
                    num_flux = np.asarray(
                        runtime_numerics.numerical_flux(
                            qA, qB, qauxA, qauxA, parameters, n
                        ), dtype=float,
                    ).reshape(-1)
                    Dm = fluct[n_vars:]
                    dQ[:, iInner_bnd[bi]] -= (num_flux + Dm) * fv_bnd[bi] / cvInner_bnd[bi]

            # 5. Explicit diffusion with boundary contributions
            if self._diffusion_ops is not None and self._diffusion_in_flux:
                bf_grads = _compute_bf_face_gradients(
                    Q, Qaux, bc_indices, self._bc_grad_fn, bf_cells, bf_fidx,
                    d_face, normals_arr, face_centers, n_bf, n_vars, has_aux,
                    time, parameters,
                )
                for v, diff_op in self._diffusion_ops.items():
                    dQ[v, :] += diff_op.explicit_with_bc(Q[v, :], bf_grads[v])

            return dQ
        return flux_operator

    # -- Source operator -----------------------------------------------

    def get_compute_source(self, mesh, model):
        """Compound source operator: evaluates both the implicit
        ``source`` slot and the explicit ``source_explicit`` slot at
        the current state and sums them.

        This backend is explicit-only (FE/SSP-RK), so the IMEX split
        a Firedrake backend respects is collapsed here — *all* source
        contributions go to the RHS evaluated at the current state.
        Backends that genuinely support IMEX (e.g. Firedrake) keep
        ``source`` in the source-step Newton and ``source_explicit``
        in the convective step.
        """
        has_explicit = hasattr(model, "source_explicit")

        def compute_source(dt, Q, Qaux, parameters, dQ):
            dQ = model.source(Q[:, :], Qaux[:, :], parameters)
            if has_explicit:
                dQ = dQ + model.source_explicit(Q[:, :], Qaux[:, :], parameters)
            return dQ
        return compute_source

    def get_compute_source_jacobian_wrt_variables(self, mesh, model):
        def compute(dt, Q, Qaux, parameters, dQ):
            return model.source_jacobian_wrt_variables(Q, Qaux, parameters)
        return compute

    # -- Setup / Step / Run / Solve ------------------------------------

    def setup_simulation(self, mesh, model, write_output=True):
        """Build all operators once. Stores simulation state on self.

        ``model`` may be a :class:`Model`, a :class:`SystemModel`, or a
        :class:`NumericalSystemModel`.  Plain models are auto-promoted
        through Model → SystemModel → NSM internally; when an NSM is
        passed directly its numerical slots (reconstruction order,
        limiter, regularization) seed the solver attributes and its
        auto-resolved LSQ degree drives the mesh stencil.
        """
        nsm, source_model = self._coerce_to_nsm(model)
        self.nsm = nsm
        mesh = ensure_lsq_mesh(mesh, nsm)
        # Periodic-BC topology resolution needs the ``BoundaryConditions``
        # object (it iterates ``.boundary_conditions_list`` for tagged
        # periodic pairs).  Done here, *before* normalising to a
        # SystemModel — the SystemModel only carries the indexed BC
        # function, not the object.
        if hasattr(mesh, "resolve_periodic_bcs") and source_model is not None:
            mesh.resolve_periodic_bcs(source_model.boundary_conditions)
        model = nsm.sm
        self.sm = model
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)
        Qaux = self.update_qaux(Q, Qaux, Q, Qaux, mesh, model, parameters, 0.0, 1.0)

        # Store simulation state
        self._sim_mesh = mesh
        self._sim_model = model
        self._sim_parameters = parameters
        self._sim_Q = Q
        self._sim_Qaux = Qaux
        self._sim_time = 0.0

        # Build operators (once) — no ghost-cell BC operator needed
        self._sim_compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model)
        self._sim_flux_operator = self.get_flux_operator(mesh, model)
        self._sim_source_operator = self.get_compute_source(mesh, model)

        # Apply initial variable updates
        self._sim_Q = self.update_q(self._sim_Q, Qaux, mesh, model, parameters)

        # Precompute mesh constant for CFL
        nc = mesh.n_inner_cells
        inner_face_mask = (mesh.face_cells[0] < nc) & (mesh.face_cells[1] < nc)
        inner_inradii = np.minimum(
            mesh.cell_inradius[mesh.face_cells[0, inner_face_mask]],
            mesh.cell_inradius[mesh.face_cells[1, inner_face_mask]],
        )
        bnd_inradii = mesh.cell_inradius[
            mesh.face_cells[0, ~inner_face_mask]
        ] if (~inner_face_mask).any() else np.array([np.inf])
        self._sim_cell_inradius_face = min(
            inner_inradii.min() if len(inner_inradii) > 0 else np.inf,
            bnd_inradii.min() if len(bnd_inradii) > 0 else np.inf,
        )

        # Output setup
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            self._sim_save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            self._sim_save_fields = lambda time, time_stamp, i_snapshot, Q, Qaux: i_snapshot

    def step(self, dt):
        """One explicit timestep (ghost-cell-free).

        BCs are evaluated inline inside the flux operator — no separate
        ghost-cell filling step. Loops are split into interior/boundary.

        O1 (RK1): flux → advance
        O2 (RK2/Heun): flux → advance → flux → average
        """
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        parameters = self._sim_parameters
        time_now = self._sim_time
        mesh = self._sim_mesh
        model = self._sim_model
        flux = self._sim_flux_operator
        source = self._sim_source_operator

        def rhs(t, Q_in):
            """Total RHS = flux + source."""
            dQ_f = flux(dt, t, Q_in, Qaux, parameters, np.zeros_like(Q_in))
            dQ_s = source(dt, Q_in, Qaux, parameters, np.zeros_like(Q_in))
            return dQ_f + dQ_s

        if self.nsm.reconstruction.order >= 2:
            # SSP-RK2 (Heun) — flux + source per stage
            Q0 = np.array(Q)
            Q1 = Q + dt * rhs(time_now, Q)
            Q2 = Q1 + dt * rhs(time_now + dt, Q1)
            Qnew = 0.5 * (Q0 + Q2)
        else:
            # RK1
            Qnew = Q + dt * rhs(time_now, Q)

        # Variable updates (clamp, etc.)
        Qnew = self.update_q(Qnew, Qaux, mesh, model, parameters)

        # Auxiliary variable updates
        Qauxnew = self.update_qaux(Qnew, Qaux, Q, Qaux, mesh, model,
                                    parameters, time_now, dt)

        # Commit new state
        self._sim_Q = Qnew
        self._sim_Qaux = Qauxnew

    def run_simulation(self):
        """Time loop: compute_dt -> step -> post_step -> output."""
        t_start = gettime()
        time_now = 0.0
        next_write_at = 0.0
        i_snapshot = 0.0
        dt_snapshot = self.time_end / max(self.settings.output.snapshots - 1, 1)
        iteration = 0

        while time_now < self.time_end:
            dt = self.compute_dt(
                self._sim_Q, self._sim_Qaux, self._sim_parameters,
                self._sim_cell_inradius_face, self._sim_compute_max_abs_eigenvalue,
            )
            dt = min(float(dt), float(self.time_end - time_now))
            if not np.isfinite(dt) or dt <= 0.0:
                break

            self._sim_time = time_now
            self.step(dt)

            time_now += dt
            iteration += 1

            if time_now >= next_write_at:
                i_snapshot = self._sim_save_fields(
                    time_now, next_write_at, i_snapshot, self._sim_Q, self._sim_Qaux,
                )
                next_write_at += dt_snapshot

            if iteration % 10 == 0:
                logger.info(
                    f"iteration: {iteration}, time: {time_now:.6f}, "
                    f"dt: {dt:.6f}, next write at time: {next_write_at:.6f}"
                )

        self._sim_time = time_now
        logger.info(f"Finished simulation with in {gettime() - t_start:.3f} seconds")
        return self._sim_Q, self._sim_Qaux

    def solve(self, mesh, model, write_output=True):
        """Convenience: setup_simulation + run_simulation."""
        self.setup_simulation(mesh, model, write_output=write_output)
        return self.run_simulation()


# -- Free-surface variant ------------------------------------------------------

def _build_free_surface_numerics(symbolic_model):
    """Build positive Rusanov for free-surface models.  ``h`` and
    ``b`` are auto-located in ``model.variables`` or
    ``model.aux_variables`` via :class:`FieldHandle` — no explicit
    ``field_map`` is needed."""
    scaled_q_indices = _detect_scaled_q_indices(symbolic_model)
    return PositiveNonconservativeRusanov(
        symbolic_model,
        scaled_q_indices=scaled_q_indices,
    )


class _SurfaceReconAdapter:
    """Adapt :class:`SurfaceReconstruction` to the explicit flux operator's
    ``reconstruct(Q, bf) → (Q_L, Q_R)`` interface.

    ``SurfaceReconstruction`` reconstructs the free-surface elevation
    ``η = h + b`` (flat at lake-at-rest → zero limited slope) and returns
    ``(Q_L, Q_R, b_L, b_R)`` with the per-side depth recovered as ``h = η − b``;
    the explicit flux operator wants only ``(Q_L, Q_R)`` and reads the limited
    slope of the *actual* state (depth row ``∂η − ∂b``) off ``_limited_grad``
    for the cell-interior non-conservative integral.  ``b`` lives in the
    conservative state here, so no ``Qaux`` is needed at reconstruction time.
    """

    def __init__(self, surf):
        self.surf = surf
        self._limited_grad = None

    def __call__(self, Q, bf, phi=None):
        Q_L, Q_R, _b_L, _b_R = self.surf(Q, None, bf, phi=phi)
        self._limited_grad = self.surf._limited_grad
        return Q_L, Q_R


class FreeSurfaceFlowSolver(HyperbolicSolver):
    """Explicit FVM for free-surface flows (SWE, SME, VAM).

    Uses positive (hydrostatic reconstruction) Rusanov with wet/dry handling.
    Requires model variables 'b' and 'h'.

    At reconstruction order ≥ 2 it reconstructs the free-surface elevation
    ``η = h + b`` (so the limiter is well-balanced) and the base solver adds
    the cell-interior non-conservative integral — together these give exact
    well-balancing at lake-at-rest to machine precision.
    """

    def _build_numerics(self, symbolic_model):
        return _build_free_surface_numerics(symbolic_model)

    def _build_reconstruction(self, mesh, symbolic_model):
        dim = symbolic_model.dimension
        if self.nsm.reconstruction.order >= 2:
            h_idx = _var_index(symbolic_model, "h")
            b_idx = _var_index(symbolic_model, "b")
            # Well-balanced surface-elevation reconstruction needs b in the
            # conservative state (recover h = η − b per side).  When the model
            # has no in-state bathymetry, fall back to the wet/dry conservative
            # MUSCL (not WB at order 2, but the only sensible option there).
            if b_idx is not None:
                from zoomy_core.fvm.reconstruction import (
                    LSQMUSCLReconstruction, SurfaceReconstruction,
                )
                base = LSQMUSCLReconstruction(
                    mesh, dim, limiter=self.nsm.reconstruction.limiter)
                surf = SurfaceReconstruction(
                    base, h_index=h_idx, b_index=b_idx, b_in_state=True)
                return _SurfaceReconAdapter(surf)
            from zoomy_core.fvm.reconstruction import FreeSurfaceLSQMUSCL
            eps_wet = self._get_dry_threshold(symbolic_model)
            return FreeSurfaceLSQMUSCL(
                mesh, dim, h_index=h_idx, eps_wet=eps_wet,
                limiter=self.nsm.reconstruction.limiter)
        return super()._build_reconstruction(mesh, symbolic_model)


def _build_roe_numerics(symbolic_model):
    """Path-conservative Roe (matrix |A| dissipation) for free-surface models;
    auto-locates ``h``/``b`` and the depth-scaled momentum rows like the
    Rusanov variant."""
    from zoomy_core.fvm.riemann_solvers import NonconservativeRoe
    scaled_q_indices = _detect_scaled_q_indices(symbolic_model)
    return NonconservativeRoe(symbolic_model, scaled_q_indices=scaled_q_indices)


class RoeFreeSurfaceFlowSolver(FreeSurfaceFlowSolver):
    """:class:`FreeSurfaceFlowSolver` using the path-conservative Roe scheme
    (matrix ``|A|`` dissipation via runtime numerical eigendecomposition)
    instead of scalar Rusanov dissipation — sharper shocks/contacts at the
    same reconstruction order.  Everything else (hydrostatic reconstruction,
    wet/dry handling, time-stepping) is unchanged."""

    def _build_numerics(self, symbolic_model):
        return _build_roe_numerics(symbolic_model)
