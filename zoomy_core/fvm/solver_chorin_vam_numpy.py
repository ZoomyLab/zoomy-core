"""Chorin projection split for the VAM chain DAE (numpy backend).

Consumes the three sub-system models produced by
:func:`zoomy_core.model.splitter.split_for_pressure`:

* ``SM_pred``  — explicit predictor.  Evolution rows (mass + xmom_jk +
  zmom_jk) with the pressure modes implicitly frozen at the current
  step (read straight from ``Q[pressure_idx]``).  Solved with
  :class:`HyperbolicSolver`'s Rusanov + non-conservative path-integral
  + indexed-BC flux machinery — same path the SWE/SME solvers use.
  Mass-conservative by construction.
* ``SM_press`` — algebraic elliptic block on the pressure modes.
  Linear in ``(P_k, ∂_x P_k, ∂_xx P_k)``; solved via
  ``scipy.optimize.fsolve`` on the lambdified residual (converges in
  1–2 Newton iterations).  Matrix-free GMRES is the next refinement.
* ``SM_corr``  — closed-form algebraic corrector on the velocity modes
  via the ``state_update`` field.  Single lambdify call + in-place
  assignment, no solve.
"""
from __future__ import annotations

import time as _time
from typing import Optional

import numpy as np
import param
import sympy as sp

from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm.riemann_solvers import (
    PositiveNonconservativeRusanov,
    PositiveQuasilinearRusanov,
)
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.model.models.system_model import SystemModel, _to_zarray
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.misc.logger_config import logger
from zoomy_core.misc.misc import Zstruct, ZArray


_EMPTY_AUX = np.empty(0, dtype=float)


def _pad_to_square(sm: SystemModel) -> SystemModel:
    """Return a square SystemModel whose operators have ``n_state`` rows.

    Used by :class:`ChorinSplitVAMSolver` to make a rectangular sub-
    system consumable by :class:`HyperbolicSolver`'s flux machinery
    (which assumes ``n_equations == n_state``).  Rows at state indices
    *not* in ``sm.equation_to_state_index`` get all-zero operators
    (flux / source / NCP / etc.) — they contribute zero RHS to the RK,
    so the corresponding state slots stay frozen through the predictor
    substep.  This is precisely the Chorin semantics: predictor doesn't
    touch pressure modes.
    """
    n_eq = sm.n_equations
    n_st = sm.n_state
    if n_eq == n_st:
        return sm

    e2s = list(sm.equation_to_state_index)
    n_dim = sm.n_dim

    # Map: dest_state_row → src_eq_row.  Missing states get None.
    src_of_dest = {state_i: src_i for src_i, state_i in enumerate(e2s)}

    def _pad_rank2(arr, n_cols):
        """``(n_eq, n_cols)`` → ``(n_st, n_cols)`` ZArray with zero rows
        at missing dest indices.  ``arr`` may be ZArray or sp.Matrix."""
        if arr is None:
            return None
        out_rows = []
        for state_i in range(n_st):
            if state_i in src_of_dest:
                src = src_of_dest[state_i]
                row = [sp.sympify(arr[src, j]) for j in range(n_cols)]
            else:
                row = [sp.S.Zero] * n_cols
            out_rows.append(row)
        return ZArray(out_rows, shape=(n_st, n_cols))

    def _pad_rank1(arr):
        """``(n_eq,)`` → ``(n_st,)`` ZArray with zeros at missing indices."""
        if arr is None:
            return None
        out = []
        for state_i in range(n_st):
            if state_i in src_of_dest:
                src = src_of_dest[state_i]
                # arr may be rank-1 or rank-2 (n_eq, 1) — handle both.
                if len(arr.shape) == 1:
                    out.append(sp.sympify(arr[src]))
                else:
                    out.append(sp.sympify(arr[src, 0]))
            else:
                out.append(sp.S.Zero)
        return ZArray(out, shape=(n_st,))

    def _pad_rank3(arr):
        """``(n_eq, n_st_cols, n_dim)`` → ``(n_st, n_st_cols, n_dim)``."""
        if arr is None:
            return None
        n_st_cols = arr.shape[1]
        out = []
        for state_i in range(n_st):
            row = []
            if state_i in src_of_dest:
                src = src_of_dest[state_i]
                for j in range(n_st_cols):
                    inner = [sp.sympify(arr[src, j, d]) for d in range(n_dim)]
                    row.append(inner)
            else:
                for j in range(n_st_cols):
                    row.append([sp.S.Zero] * n_dim)
            out.append(row)
        return ZArray(out, shape=(n_st, n_st_cols, n_dim))

    # Pad each operator.  Source can be rank-1 (post-refactor) or (n_eq,1).
    new_flux = _pad_rank2(sm.flux, n_dim)
    new_P    = _pad_rank2(sm.hydrostatic_pressure, n_dim)
    new_B    = _pad_rank3(sm.nonconservative_matrix)
    new_S    = (_pad_rank2(sm.source, sm.source.shape[1])
                if len(sm.source.shape) == 2
                else _pad_rank1(sm.source))
    new_M    = _pad_rank2(sm.mass_matrix, n_st)
    new_E    = _pad_rank1(sm.eigenvalues) if sm.eigenvalues is not None else None

    sm_square = SystemModel(
        time=sm.time,
        space=list(sm.space),
        state=list(sm.state),
        aux_state=list(sm.aux_state),
        parameters=sm.parameters,
        parameter_values=sm.parameter_values,
        flux=new_flux,
        hydrostatic_pressure=new_P,
        nonconservative_matrix=new_B,
        source=new_S,
        mass_matrix=new_M,
        eigenvalues=new_E,
        equation_to_state_index=list(range(n_st)),
        boundary_conditions=sm.boundary_conditions,
        aux_boundary_conditions=sm.aux_boundary_conditions,
        boundary_gradients=sm.boundary_gradients,
        initial_conditions=sm.initial_conditions,
        aux_initial_conditions=sm.aux_initial_conditions,
        update_variables=sm.update_variables,
    )
    sm_square.aux_registry = list(getattr(sm, "aux_registry", []) or [])
    sm_square.equation_names = (
        [sm.equation_names[src_of_dest[i]] if i in src_of_dest
         else f"_pad_{i}"
         for i in range(n_st)]
        if getattr(sm, "equation_names", None) is not None
        else None
    )
    sm_square.history = list(sm.history) + [{
        "name": "pad_to_square",
        "description": (
            f"padded {n_eq} → {n_st} rows; "
            f"zero ops at state indices "
            f"{[i for i in range(n_st) if i not in src_of_dest]}"
        ),
    }]
    return sm_square


class ChorinSplitVAMSolver(HyperbolicSolver):
    """Chorin projection split for the VAM chain DAE.

    Consumes three pre-built sub-system models from
    :func:`split_for_pressure`.  Inherits :class:`HyperbolicSolver` so
    the predictor substep uses the same Rusanov flux + indexed-BC +
    well-balanced reconstruction machinery as every other hyperbolic
    solver in the codebase — mass-conservative by construction.

    The pressure projection and corrector substeps are Chorin-specific
    and don't duplicate any existing infrastructure (the
    ``state_update`` field on SystemModel + matrix-free linear solve
    are the new primitives this class introduces).
    """

    pressure_tol = param.Number(
        default=1e-10, bounds=(0, None),
        doc="Linear-solver tolerance for the elliptic pressure block")
    pressure_maxit = param.Integer(
        default=500, bounds=(1, None),
        doc="Maximum iterations for the elliptic pressure solve")

    def __init__(self, sm_pred, sm_press, sm_corr, **kwargs):
        if not isinstance(sm_pred, SystemModel):
            raise TypeError("sm_pred must be a SystemModel")
        if not isinstance(sm_press, SystemModel):
            raise TypeError("sm_press must be a SystemModel")
        if not isinstance(sm_corr, SystemModel):
            raise TypeError("sm_corr must be a SystemModel")
        super().__init__(**kwargs)
        self.sm_pred = sm_pred
        self.sm_press = sm_press
        self.sm_corr = sm_corr
        self.state = list(sm_pred.state)
        self.n_state = sm_pred.n_state

        self._dt_symbol = self._detect_dt_symbol()

        for name, sm in (("press", sm_press), ("corr", sm_corr)):
            if [str(s) for s in sm.state] != [str(s) for s in sm_pred.state]:
                raise ValueError(
                    f"SM_{name}.state disagrees with SM_pred.state — the "
                    "three sub-systems must share a common state vector."
                )

    # ------------------------------------------------------------------
    # Numerics — Audusse well-balanced Rusanov for free-surface flow.
    # ------------------------------------------------------------------

    def _build_numerics(self, symbolic_model):
        """Audusse hydrostatic-reconstruction Rusanov for the predictor.

        The plain ``NonconservativeRusanov`` (parent's default) fails
        lake-at-rest on a varying bathymetry: Rusanov's wavespeed-based
        dissipation breaks the discrete cancellation of
        ``g·h·∂_x η`` against the bottom-slope source.  Audusse–
        Bristeau–Klein hydrostatic reconstruction (
        :class:`PositiveNonconservativeRusanov`) restores well-
        balancing and handles wet/dry fronts via the ``h*`` clamp.

        ``scaled_q_indices``: the momentum-density state indices
        (auto-detected — excludes ``h`` and the pressure-mode state
        indices from ``self.sm_press``).  These are the rows that
        get rescaled by ``h*/h`` during the reconstruction (so the
        velocity stays constant when ``h`` is clipped).  Pressure
        modes ``P_k`` are NOT rescaled — they are pressure
        amplitudes, not momentum densities.
        """
        state_names = [str(s) for s in symbolic_model.state]
        excluded = set()
        if "h" in state_names:
            excluded.add(state_names.index("h"))
        if "b" in state_names:
            # Bathymetry as state (8-state b-promoted path): exclude
            # from h*/h rescaling — b is static topography, not a
            # momentum density that needs HR-mass-preservation scaling.
            excluded.add(state_names.index("b"))
        # Pressure-mode state indices come straight from the splitter.
        excluded.update(int(i) for i in self.sm_press.equation_to_state_index)
        scaled_q_indices = [
            i for i in range(symbolic_model.n_variables) if i not in excluded
        ]
        return PositiveNonconservativeRusanov(
            symbolic_model,
            scaled_q_indices=scaled_q_indices,
        )

    def _build_reconstruction(self, mesh, symbolic_model):
        """Wet/dry-aware MUSCL reconstruction on the predictor.

        At ``reconstruction_order >= 2`` use
        :class:`FreeSurfaceLSQMUSCL` (free-surface ``η = h + b`` slope
        limited, dry-cell clamp) instead of the parent's generic LSQ
        MUSCL.  Matches the choice made by
        :class:`FreeSurfaceFlowSolver`.
        """
        if self.reconstruction_order >= 2:
            from zoomy_core.fvm.reconstruction import FreeSurfaceLSQMUSCL
            from zoomy_core.fvm.solver_numpy import _var_index
            dim = symbolic_model.dimension
            h_idx = _var_index(symbolic_model, "h")
            eps_wet = self._get_dry_threshold(symbolic_model)
            return FreeSurfaceLSQMUSCL(
                mesh, dim, h_index=h_idx,
                eps_wet=eps_wet, limiter=self.limiter,
            )
        return super()._build_reconstruction(mesh, symbolic_model)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_simulation(self, mesh, write_output=False):
        """Set up the Chorin solver:

        1. Pad ``SM_pred`` to square so :class:`HyperbolicSolver` can
           drive it (the parent assumes ``n_equations == n_state``).
        2. Hand the padded predictor to the parent's
           ``setup_simulation`` — this builds the proper Rusanov flux
           operator, indexed-BC kernel, source operator, max-eigenvalue
           callable on the predictor SystemModel.
        3. Rename ``\\Delta t`` → ``dt`` in ``SM_press`` / ``SM_corr``
           (Python-safe Symbol), register as a parameter, build their
           runtimes via ``NumpyRuntimeModel.from_system_model``.
        """
        t0 = _time.time()

        # 1. Pad SM_pred to square and let the parent set up the
        #    predictor's flux machinery.
        sm_pred_square = _pad_to_square(self.sm_pred)
        self._sm_pred_square = sm_pred_square
        super().setup_simulation(mesh, sm_pred_square, write_output=write_output)

        # 2. dt rename + parameter registration on the dt-dependent subsystems.
        if self._dt_symbol is not None:
            dt_safe = sp.Symbol("dt", positive=True)
            self._dt_symbol_safe = dt_safe
            self.sm_press = _substitute_dt(self.sm_press, self._dt_symbol,
                                           dt_safe)
            self.sm_corr = _substitute_dt(self.sm_corr, self._dt_symbol,
                                          dt_safe)
            for sm in (self.sm_press, self.sm_corr):
                if not sm.parameters.contains("dt"):
                    new_params = Zstruct(**sm.parameters.as_dict())
                    new_params["dt"] = dt_safe
                    sm.parameters = new_params
                    new_pvals = Zstruct(**sm.parameter_values.as_dict())
                    new_pvals["dt"] = 0.0
                    sm.parameter_values = new_pvals
        else:
            self._dt_symbol_safe = None

        # 3. Build pressure + corrector runtimes (predictor runtime
        #    lives on ``self._sim_model`` from parent setup).
        self.rt_press = NumpyRuntimeModel.from_system_model(self.sm_press)
        self.rt_corr  = NumpyRuntimeModel.from_system_model(self.sm_corr)

        self._params_press_base = np.array(
            list(self.sm_press.parameter_values.values()), dtype=float,
        )
        self._params_corr_base = np.array(
            list(self.sm_corr.parameter_values.values()), dtype=float,
        )

        # State-index slices.
        self._pred_state_idx = np.asarray(
            self.sm_pred.equation_to_state_index, dtype=int)
        self._press_state_idx = np.asarray(
            self.sm_press.equation_to_state_index, dtype=int)
        self._corr_state_idx = np.asarray(
            self.sm_corr.equation_to_state_index, dtype=int)

        # 4. Per-subsystem aux pools.  Each sub-system has its own
        #    ``aux_state`` ordering (auto-scan picks up different
        #    atoms in different orders), so a single shared ``Qaux``
        #    breaks the row indexing — we maintain three arrays in
        #    lock-step via ``update_aux_variables``.  The predictor's
        #    Qaux is the parent's ``_sim_Qaux``.
        nc = self.nc
        self.Qaux_press = np.zeros(
            (len(self.sm_press.aux_state), nc), dtype=float)
        self.Qaux_corr = np.zeros(
            (len(self.sm_corr.aux_state), nc), dtype=float)

        # 5. Matrix-free pressure solver: identify which SM_press
        #    derivative-aux entries depend on the pressure modes
        #    being solved for.  These get recomputed via LSQ stencils
        #    on the current Krylov iterate during every matvec
        #    application — that's what makes the operator linear in P
        #    (the lambdified symbolic residual reads aux for ``P_x``,
        #    ``P_xx`` rather than embedding the LSQ stencil in the
        #    symbolic flow).
        self._press_state_idx_set = set(int(i) for i in self._press_state_idx)
        self._press_aux_recompute = []
        for entry in (self.sm_press.aux_registry or []):
            if (entry["kind"] != "derivative"
                    or entry.get("target_kind") != "state"
                    or entry.get("state_index") not in self._press_state_idx_set):
                continue
            self._press_aux_recompute.append({
                "row":         entry["row"],
                "state_index": entry["state_index"],
                "multi_index": entry["multi_index"],
            })

        logger.info(
            "ChorinSplitVAMSolver setup: pred → %s, press → %s, corr → %s "
            "in %.2fs",
            self._pred_state_idx.tolist(),
            self._press_state_idx.tolist(),
            self._corr_state_idx.tolist(),
            _time.time() - t0,
        )
        return self._sim_Q

    # ------------------------------------------------------------------
    # Conveniences
    # ------------------------------------------------------------------

    @property
    def nc(self):
        return self._sim_mesh.n_inner_cells

    @property
    def Q(self):
        return self._sim_Q

    @property
    def Qaux(self):
        return self._sim_Qaux

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def _aux_pool_for(self, sm):
        """Return the (Qaux array, registry) for a given sub-system."""
        if sm is self.sm_pred:
            return self._sim_Qaux, sm.aux_registry or []
        if sm is self.sm_press:
            return self.Qaux_press, sm.aux_registry or []
        if sm is self.sm_corr:
            return self.Qaux_corr, sm.aux_registry or []
        raise ValueError(f"Unknown sub-system {sm}")

    def update_aux_variables(self):
        """Refresh derivative-aux rows by LSQ in each sub-system's
        ``Qaux`` pool.  Each sub-system has its own ``aux_state``
        ordering (auto-scan ordering depends on which atoms appear
        in that sub-system's operators) — we walk all three
        registries and fill each one's Qaux array independently.

        Function-aux rows (``b`` topography) are left untouched —
        the user sets them once via :meth:`set_function_aux` and they
        stay static through the simulation.
        """
        Q = self._sim_Q
        nc = Q.shape[1]
        n_cells = self._sim_mesh.n_cells
        mesh = self._sim_mesh
        u_full = np.zeros(n_cells)

        for sm in (self.sm_pred, self.sm_press, self.sm_corr):
            Qaux, registry = self._aux_pool_for(sm)
            for entry in registry:
                if entry["kind"] != "derivative":
                    continue
                row = entry["row"]
                if row >= Qaux.shape[0]:
                    continue
                mi = entry["multi_index"]
                tk = entry["target_kind"]
                if tk == "state":
                    u_full[:nc] = Q[entry["state_index"], :]
                elif tk == "function":
                    u_full[:nc] = Qaux[entry["function_row"], :]
                else:
                    continue
                u_full[nc:] = 0.0
                d = mesh.compute_derivatives(
                    u_full, degree=max(mi),
                    derivatives_multi_index=[mi],
                )
                Qaux[row, :] = d[:nc, 0]

    def set_function_aux(self, name, values):
        """Set a function-aux row (e.g. static topography ``b``) on
        every sub-system's Qaux pool at its own row index.  Use this
        once at IC time to inject bathymetry; ``update_aux_variables``
        then refreshes the derived rows (``b_x``, ``b_xx``, …) from it.
        """
        for sm in (self.sm_pred, self.sm_press, self.sm_corr):
            Qaux, registry = self._aux_pool_for(sm)
            for entry in registry:
                if entry["kind"] == "function" and entry["name"] == name:
                    Qaux[entry["row"], :] = values
                    break

    def step(self, dt):
        """One Chorin step: predictor → pressure → corrector.

        Predictor: inherited :class:`HyperbolicSolver.step` — proper
        Rusanov + NCP + indexed-BC.  Mass-conservative.

        Pressure: scipy.optimize.fsolve on the lambdified elliptic
        residual (linear in P ⇒ 1–2 Newton iters from warm start).

        Corrector: one lambdify call to ``state_update`` + atomic
        in-place assignment.  Read-then-write is safe — Python
        evaluates the RHS fully before assigning to the LHS slice.
        """
        super().step(dt)              # predictor: parent's RK + flux + source
        self.update_aux_variables()
        self._step_pressure(dt)
        self.update_aux_variables()
        self._step_corrector(dt)
        self.update_aux_variables()
        # Parent's step advances self._sim_time internally? Check.
        # HyperbolicSolver.step doesn't update self._sim_time — that's
        # done in run_simulation.  Our step is the user-facing entry
        # point, so we track time here.
        self._sim_time += dt

    def _do_flux_only_substep(self, dt):
        """Explicit RK1 advance using ONLY the flux operator (no source).

        Helper for :class:`LegacyStyleChorinVAMSolver` which mirrors
        the old ``web/tutorials/vam/simple.ipynb`` algorithm: flux step
        first (without source), then implicit pressure solve, then
        source step using the freshly-solved pressure.
        """
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        parameters = self._sim_parameters
        time_now = self._sim_time
        mesh = self._sim_mesh
        model = self._sim_model
        flux = self._sim_flux_operator
        Qnew = Q + dt * flux(dt, time_now, Q, Qaux,
                             parameters, np.zeros_like(Q))
        Qnew = self.update_q(Qnew, Qaux, mesh, model, parameters)
        Qauxnew = self.update_qaux(Qnew, Qaux, Q, Qaux, mesh, model,
                                    parameters, time_now, dt)
        self._sim_Q = Qnew
        self._sim_Qaux = Qauxnew

    def _do_source_only_substep(self, dt):
        """Explicit RK1 advance using ONLY the source operator.

        Used by :class:`LegacyStyleChorinVAMSolver` after the implicit
        pressure solve, with the freshly-solved ``P_0, P_1`` exposed
        in the aux pool so the source-evaluation sees them.
        """
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        parameters = self._sim_parameters
        mesh = self._sim_mesh
        model = self._sim_model
        source = self._sim_source_operator
        Qnew = Q + dt * source(dt, Q, Qaux, parameters, np.zeros_like(Q))
        Qnew = self.update_q(Qnew, Qaux, mesh, model, parameters)
        self._sim_Q = Qnew

    def _params_with_dt(self, base, dt):
        """Splice numeric dt into the last slot of a parameter array
        (where ``dt`` lives by our setup_simulation convention)."""
        out = base.copy()
        out[-1] = float(dt)
        return out

    def _step_pressure(self, dt):
        """Solve the linear elliptic block for ``Q[press_e2s]``.

        Matrix-free GMRES.  The operator ``A·P`` is evaluated per
        Krylov iteration by:

        1. Stuffing the iterate ``p_vec`` into ``Q_try[press_e2s]``.
        2. **Recomputing the P-dependent aux entries** (``P_0_x``,
           ``P_1_x``, ``P_0_xx``, ``P_1_xx``) via LSQ stencils on
           ``p_vec`` — the bug fix vs. fsolve-with-frozen-aux.  The
           non-P-dependent aux entries (``h``, ``b``, ``h_x``, etc.)
           stay frozen at the start-of-step values.
        3. Evaluating ``rt.source(Q_try, Qaux_try, p_full)`` — the
           lambdified elliptic residual.

        ``b = R(P=0)`` once per step gives the data forcing; then
        ``A·P = R(P) − b`` is the linear operator.  GMRES warm-starts
        from the previous-step pressure.  No matrix assembled — same
        pattern jax-jits cleanly via ``jax.scipy.sparse.linalg.gmres``.
        """
        from scipy.sparse.linalg import LinearOperator, gmres
        rt    = self.rt_press
        Q     = self._sim_Q
        Qaux0 = self.Qaux_press
        nc    = self.nc
        mesh  = self._sim_mesh
        n_cells = mesh.n_cells
        p_full = self._params_with_dt(self._params_press_base, dt)
        e2s   = self._press_state_idx
        nP    = len(e2s)
        N     = nP * nc

        # Map state index → local pressure index (e.g. 5 → 0, 6 → 1).
        local_of_state = {int(s): k for k, s in enumerate(e2s)}

        def _refresh_pressure_aux(p_mat, Qaux_work):
            """LSQ-recompute the P-dependent derivative-aux rows from
            the current iterate ``p_mat`` (shape ``(nP, nc)``).  In-
            place on ``Qaux_work``."""
            u_full = np.zeros(n_cells)
            for entry in self._press_aux_recompute:
                state_i = entry["state_index"]
                k = local_of_state[state_i]
                u_full[:nc] = p_mat[k, :]
                u_full[nc:] = 0.0
                mi = entry["multi_index"]
                d = mesh.compute_derivatives(
                    u_full, degree=max(mi),
                    derivatives_multi_index=[mi],
                )
                Qaux_work[entry["row"], :] = d[:nc, 0]

        def _residual(p_vec):
            p_mat = p_vec.reshape(nP, nc)
            Q_try = Q.copy()
            Q_try[e2s, :] = p_mat
            Qaux_work = Qaux0.copy()
            _refresh_pressure_aux(p_mat, Qaux_work)
            R = np.asarray(rt.source(Q_try, Qaux_work, p_full), dtype=float)
            if R.ndim == 1:
                R = R[:, None] * np.ones((1, nc))
            return R.ravel()

        # Data forcing: ``b = R(P = 0)``.  Pure data — no P enters.
        b = _residual(np.zeros(N))

        # Early exit on near-zero forcing (e.g. exact lake-at-rest):
        # GMRES with only ``rtol`` set chases ``rtol·‖b‖`` which is
        # unreachable when ‖b‖ ~ machine epsilon, so it hits maxiter
        # and returns garbage.  Skip the solve and return P = warm-start
        # (which is the correct answer for ‖b‖ ≈ 0).
        b_norm = float(np.linalg.norm(b))
        if b_norm < self.pressure_tol:
            self._sim_Q = Q              # Q[e2s] unchanged (still p0)
            return

        # Linear operator: ``A·P = R(P) − b``.  Pure matvec callable.
        def apply_A(p_vec):
            return _residual(p_vec) - b

        A = LinearOperator((N, N), matvec=apply_A, dtype=float)
        p0 = Q[e2s, :].ravel()                      # warm start
        p_new, info = gmres(
            A, -b, x0=p0,
            rtol=self.pressure_tol,
            atol=self.pressure_tol,    # absolute floor — survives tiny ‖b‖
            maxiter=self.pressure_maxit,
        )
        if info != 0:
            logger.warning(
                "Chorin pressure GMRES did not fully converge "
                "(info=%d, ‖b‖=%.3e)", info, b_norm,
            )
        Q[e2s, :] = p_new.reshape(nP, nc)
        self._sim_Q = Q
        # Reflect the solved pressure into Qaux_press derivative rows
        # (so the corrector reads consistent ∂_x P, ∂_xx P).
        _refresh_pressure_aux(p_new.reshape(nP, nc), self.Qaux_press)

    def _step_corrector(self, dt):
        """``Q[corr_e2s] ← state_update(Q, Qaux_corr, p, dt)`` in place."""
        rt = self.rt_corr
        Q = self._sim_Q
        Qaux = self.Qaux_corr           # SM_corr's own aux pool
        p_full = self._params_with_dt(self._params_corr_base, dt)
        e2s = self._corr_state_idx

        new_vals = np.asarray(rt.state_update(Q, Qaux, p_full), dtype=float)
        Q[e2s, :] = new_vals
        self._sim_Q = Q

    def _detect_dt_symbol(self) -> Optional[sp.Symbol]:
        """Locate the symbolic time-step common to SM_press / SM_corr."""
        candidates = set()
        for sm in (self.sm_press, self.sm_corr):
            for atom in sm.source.free_symbols | sm.flux.free_symbols:
                if atom in sm.parameters.values():
                    continue
                if atom in sm.state:
                    continue
                if atom in sm.aux_state:
                    continue
                name = str(atom)
                if name in {"dt", "Delta t", r"\Delta t"}:
                    candidates.add(atom)
        if len(candidates) > 1:
            raise ValueError(
                "Multiple dt-like free Symbols in SM_press/SM_corr: "
                f"{[str(c) for c in candidates]}"
            )
        return next(iter(candidates)) if candidates else None


def _substitute_dt(sm, old_sym, new_sym):
    """Rename a Symbol throughout a sub-system's operator tensors.

    Used to convert the splitter's ``\\Delta t`` placeholder into the
    Python-safe ``dt`` Symbol before lambdify (the backslash is a line
    continuation in Python source).  Element-wise xreplace via ZArray.
    """
    sub = {old_sym: new_sym}
    def _xrepl(tensor):
        if tensor is None:
            return None
        return _to_zarray(tensor).xreplace(sub)
    sm.flux                   = _xrepl(sm.flux)
    sm.hydrostatic_pressure   = _xrepl(sm.hydrostatic_pressure)
    sm.source                 = _xrepl(sm.source)
    sm.mass_matrix            = _xrepl(sm.mass_matrix)
    sm.nonconservative_matrix = _xrepl(sm.nonconservative_matrix)
    sm.refresh_derived_operators(eigenvalues=False)
    sm.eigenvalues            = _xrepl(sm.eigenvalues)
    sm.state_update           = _xrepl(sm.state_update)
    return sm


class LegacyStyleChorinVAMSolver(ChorinSplitVAMSolver):
    """Reference solver: mirrors the algorithm of the old
    ``web/tutorials/vam/simple.ipynb`` (commit ``e1c91370``)
    ``PredictorCorrectorSolver`` exactly.  Kept as a checked-in
    reference so the splitter+chain pipeline can be A/B-compared
    against a known-working configuration.

    Differences vs the standard :class:`ChorinSplitVAMSolver`:

    * **Source step is decoupled from the predictor.**  In the
      standard solver, the parent ``HyperbolicSolver.step`` evaluates
      ``flux + source`` together using start-of-step (frozen)
      pressure.  Here, the cycle is:

      1. Explicit RK1 advance using ONLY the flux operator
         (no source) on the current state.
      2. Pressure-projection solve (implicit, dt-baked elliptic
         block, same as standard solver).
      3. Explicit RK1 advance using ONLY the source operator,
         which now sees the freshly-solved pressure modes
         ``(P_0, P_1)`` in state.
      4. Corrector update (``state_update``) closes the loop.

      The source contains pressure-dependent terms like
      ``S[xmom_j0] = −2·P_1·b_x/ρ`` — using fresh ``P_k`` after
      the projection is a strictly tighter coupling than using
      frozen ``P_k`` (start of step).

    * **SSPRK2 wrap on the FULL cycle**, not just the predictor.
      The old algorithm runs the four-step cycle above twice
      sequentially (each advancing by ``dt`` internally), then
      averages:

      .. code-block:: python

          Q^(1) = full_cycle(Q^n,   dt)
          Q^(2) = full_cycle(Q^(1), dt)
          Q^{n+1} = 0.5·(Q^n + Q^(2))

      Heun's method on the WHOLE pred-press-corr trio.  Matches
      Escalante 2024 JCP 504 (2024) 112882 §3.1's two-stage TVD-RK2
      structure exactly.

    Inherits all of :class:`ChorinSplitVAMSolver`'s setup
    (Audusse `PositiveNonconservativeRusanov`, wet/dry MUSCL at
    ``reconstruction_order ≥ 2``, BC kernel, aux pools, dt-symbol
    detection, pressure-GMRES, corrector).  Only ``step`` differs.
    """

    name = param.String(default="legacy_style_chorin_vam")

    def _build_numerics(self, symbolic_model):
        """Override with :class:`PositiveQuasilinearRusanov` — matches the
        OLD ``PredictorCorrectorSolver``'s flux/NCP choice:

        * ``numerical_flux`` returns zero (= OLD's ``flux.Zero()``).
        * The path-integral fluctuation uses the full *quasilinear
          matrix* ``A = ∂F/∂Q + ∂P/∂Q + B`` (= OLD's
          ``nc_flux.segmentpath()`` over the full quasilinear
          structure), rather than only the explicit NCP ``B``.
        * Audusse hydrostatic reconstruction is still active —
          ``b_*`` and ``h_*`` are reconstructed at every face before
          the path-integral evaluation, giving bit-exact lake-at-rest
          on a varying bathymetry.

        This is the predictor that the OLD ``simple.ipynb`` solver
        actually uses (modulo Python→numpy reimplementation).
        """
        state_names = [str(s) for s in symbolic_model.state]
        excluded = set()
        if "h" in state_names:
            excluded.add(state_names.index("h"))
        if "b" in state_names:
            excluded.add(state_names.index("b"))
        excluded.update(int(i) for i in self.sm_press.equation_to_state_index)
        scaled_q_indices = [
            i for i in range(symbolic_model.n_variables) if i not in excluded
        ]
        return PositiveQuasilinearRusanov(
            symbolic_model,
            scaled_q_indices=scaled_q_indices,
        )

    def step(self, dt):
        """SSPRK2-wrapped legacy-style full cycle (one effective dt)."""
        Q_n = self._sim_Q.copy()
        Qaux_n = self._sim_Qaux.copy()
        t_n = self._sim_time

        # Cycle 1: U^(1) ← full_cycle(U^n, dt).
        self._legacy_full_cycle(dt)
        Q1 = self._sim_Q.copy()
        Qaux1 = self._sim_Qaux.copy()

        # Cycle 2: U^(2) ← full_cycle(U^(1), dt).
        self._legacy_full_cycle(dt)
        Q2 = self._sim_Q.copy()
        Qaux2 = self._sim_Qaux.copy()

        # Heun average + time rewind.
        self._sim_Q = 0.5 * (Q_n + Q2)
        self._sim_Qaux = 0.5 * (Qaux_n + Qaux2)
        self._sim_time = t_n + dt
        # Refresh derivative-aux rows from the averaged state so the
        # next call sees consistent aux.  Function-aux (b) is static
        # and already unchanged.
        self.update_aux_variables()

    def _legacy_full_cycle(self, dt):
        """One full cycle in the OLD algorithm's order:
        flux (no source) → BC + aux → pressure solve → BC + aux →
        source (with fresh P) → BC + aux.  Advances ``_sim_time``
        by ``dt``.

        **No corrector step** — the OLD ``PredictorCorrectorSolver``
        applied the full pressure contribution via the source
        operator (with the freshly-solved ``P``), so the corrector
        / projection-step ``Q[corr_e2s] ← Q − (dt/h)·T_u(P)`` would
        double-count on top of the source's pressure terms.  This
        is the *structural* difference from
        :class:`ChorinSplitVAMSolver` whose predictor takes
        ``flux + source`` together with frozen ``P`` and then uses
        the corrector to replace pressure contributions with fresh
        ``P``.
        """
        # 1. EXPLICIT FLUX-ONLY advance (RK1).  Parent's flux operator
        #    already includes BC application (it evaluates the indexed
        #    BC kernel inline at boundary faces).
        self._do_flux_only_substep(dt)
        self.update_aux_variables()

        # 2. IMPLICIT PRESSURE SOLVE (matrix-free GMRES on the elliptic
        #    block, dt baked in via parameter slot).  Writes the new
        #    ``(P_0, P_1)`` into ``self._sim_Q[5:7]`` and refreshes the
        #    pressure-derivative aux rows.
        self._step_pressure(dt)
        self.update_aux_variables()

        # 3. EXPLICIT SOURCE-ONLY advance (RK1) using the FRESHLY
        #    solved pressure.
        self._do_source_only_substep(dt)
        self.update_aux_variables()

        self._sim_time += dt


__all__ = [
    "ChorinSplitVAMSolver",
    "LegacyStyleChorinVAMSolver",
]
