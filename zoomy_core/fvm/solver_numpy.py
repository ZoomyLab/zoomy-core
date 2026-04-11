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

from zoomy_core.misc.logger_config import logger
import zoomy_core.misc.io as io
from zoomy_core.misc.misc import Zstruct, Settings
import zoomy_core.fvm.ode as ode
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel, NumpyRuntimeSymbolic
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.fvm.riemann_solvers import (
    PositiveNonconservativeRusanov,
    NonconservativeRusanov,
)


_EMPTY_AUX = np.array([])


# -- Field detection helpers ---------------------------------------------------

def _var_index(model, name, fallback=None):
    keys = list(model.variables.keys())
    if name in keys:
        return keys.index(name)
    if fallback is not None:
        return fallback
    raise KeyError(f"Variable '{name}' not found in model variables: {keys}")


def _param_value(model, name, default=None):
    if hasattr(model.parameters, "contains") and model.parameters.contains(name):
        idx = list(model.parameters.keys()).index(name)
        return float(model.parameter_values[idx])
    return default


def _detect_field_map(model):
    if hasattr(model, "get_field_map"):
        return model.get_field_map()
    keys = list(model.variables.keys())
    field_map = {}
    if "h" in keys:
        field_map["h"] = {"container": "q", "index": keys.index("h")}
    if "b" in keys:
        field_map["b"] = {"container": "q", "index": keys.index("b")}
    return field_map


def _detect_scaled_q_indices(model):
    if hasattr(model, "numerics_scaled_q_indices"):
        return model.numerics_scaled_q_indices
    keys = list(model.variables.keys())
    excluded = set()
    for name in ["b", "h"]:
        if name in keys:
            excluded.add(keys.index(name))
    return [i for i in range(model.n_variables) if i not in excluded]


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
        n_variables = model.n_variables
        n_cells = mesh.n_cells
        n_aux_variables = model.aux_variables.length()
        Q = np.empty((n_variables, n_cells), dtype=float)
        Qaux = np.empty((n_aux_variables, n_cells), dtype=float)
        return Q, Qaux

    def create_runtime(self, Q, Qaux, mesh, model):
        if hasattr(mesh, "resolve_periodic_bcs"):
            mesh.resolve_periodic_bcs(model.boundary_conditions)
        Q, Qaux = np.asarray(Q), np.asarray(Qaux)
        parameters = np.asarray(model.parameter_values)
        from zoomy_core.kernel import Kernel
        kernel = Kernel(model)
        kernel.regularize(model)
        self._kernel = kernel  # store for numerics regularization
        runtime_model = NumpyRuntimeModel(model, kernel=kernel)
        return Q, Qaux, parameters, mesh, runtime_model

    def get_apply_boundary_conditions(self, mesh, model):
        dim = model.dimension
        def apply_boundary_conditions(time, Q, Qaux, parameters):
            for i in range(mesh.n_boundary_faces):
                i_face = mesh.boundary_face_face_indices[i]
                i_bc_func = mesh.boundary_face_function_numbers[i]
                q_cell = Q[:, mesh.boundary_face_cells[i]]
                qaux_cell = Qaux[:, mesh.boundary_face_cells[i]]
                normal = mesh.face_normals[:dim, i_face]
                position = mesh.face_centers[i_face, :]
                position_ghost = mesh.cell_centers[:, mesh.boundary_face_ghosts[i]]
                distance = np.linalg.norm(position - position_ghost)
                q_ghost = model.boundary_conditions(
                    i_bc_func, time, position, distance, q_cell, qaux_cell, parameters, normal
                )
                Q[:, mesh.boundary_face_ghosts[i]] = q_ghost
            return Q
        return apply_boundary_conditions

    def _build_gradient_operator(self, mesh, model):
        """Build a gradient function for all variables.  Returns callable(Q) → gradQ.

        gradQ shape: (n_vars, dim, n_cells).
        Method determined by ``self.gradient_method``.
        """
        dim = model.dimension
        method = getattr(self, 'gradient_method', 'green_gauss')

        if method == "lsq":
            from zoomy_core.mesh.lsq_reconstruction import compute_derivatives
            # LSQ first derivatives: multi_index [[1,0]] for dx, [[0,1]] for dy, etc.
            dim_indices = []
            for d in range(dim):
                idx = [0] * dim
                idx[d] = 1
                dim_indices.append(idx)

            def lsq_gradients(Q):
                n_vars = Q.shape[0]
                grad = np.zeros((n_vars, dim, mesh.n_cells))
                for v in range(n_vars):
                    derivs = compute_derivatives(Q[v, :], mesh,
                                                 derivatives_multi_index=dim_indices)
                    for d in range(dim):
                        grad[v, d, :] = derivs[:, d]
                return grad
            return lsq_gradients

        # Default: Green-Gauss
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        fn = mesh.face_normals[:dim, :]
        fv = mesh.face_volumes
        cv = mesh.cell_volumes

        def gg_gradients(Q):
            n_vars = Q.shape[0]
            grad = np.zeros((n_vars, dim, mesh.n_cells))
            for v in range(n_vars):
                u = Q[v, :]
                u_face = 0.5 * (u[iA] + u[iB])
                for d in range(dim):
                    contrib = u_face * fn[d, :] * fv
                    np.add.at(grad[v, d], iA, contrib / cv[iA])
                    np.subtract.at(grad[v, d], iB, contrib / cv[iB])
            return grad
        return gg_gradients

    def _build_bc_with_gradient(self, mesh, model):
        """Build BC application that uses gradient for 2nd-order ghost extrapolation.

        Returns callable(time, Q, Qaux, parameters, gradQ) → Q.
        """
        dim = model.dimension
        bf_faces = mesh.boundary_face_face_indices
        bf_cells = mesh.boundary_face_cells
        bf_ghosts = mesh.boundary_face_ghosts
        bf_funcs = mesh.boundary_face_function_numbers
        normals = mesh.face_normals
        fc = mesh.face_centers
        cc = mesh.cell_centers
        n_bf = mesh.n_boundary_faces

        def apply_bc_with_grad(time, Q, Qaux, parameters, gradQ):
            for i in range(n_bf):
                i_face = bf_faces[i]
                i_inner = bf_cells[i]
                i_ghost = bf_ghosts[i]

                q_cell = Q[:, i_inner]
                qaux_cell = Qaux[:, i_inner]
                normal = normals[:dim, i_face]
                position = fc[i_face, :]
                position_ghost = cc[:, i_ghost]
                distance = np.linalg.norm(position - position_ghost)

                # 1st-order BC: get wall/extrapolation state at face
                q_ghost = model.boundary_conditions(
                    bf_funcs[i], time, position, distance,
                    q_cell, qaux_cell, parameters, normal,
                )

                # 2nd-order: extrapolate from face to ghost using gradient
                if gradQ is not None:
                    dx = position_ghost[:dim] - position[:dim]
                    q_ghost = np.asarray(q_ghost, dtype=float)
                    q_ghost += gradQ[:, :, i_inner] @ dx

                Q[:, i_ghost] = q_ghost
            return Q
        return apply_bc_with_grad

    def update_q(self, Q, Qaux, mesh, model, parameters):
        """Apply model.update_variables (h clamp, momentum ramp) at each cell."""
        n_vars = Q.shape[0]
        for c in range(Q.shape[1]):
            aux = Qaux[:, c] if Qaux.shape[0] > 0 else _EMPTY_AUX
            Q[:, c] = np.asarray(
                model.update_variables(Q[:, c], aux, parameters), dtype=float,
            ).ravel()[:n_vars]
        return Q

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        return Qaux


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
    reconstruction_order = param.Integer(default=1, bounds=(1, 2),
        doc="Spatial reconstruction order: 1=piecewise-constant, 2=MUSCL")
    limiter = param.Selector(default="venkatakrishnan",
        objects=["venkatakrishnan", "barth_jespersen", "minmod"],
        doc="Slope limiter for MUSCL reconstruction")
    gradient_method = param.Selector(default="green_gauss",
        objects=["green_gauss", "lsq"],
        doc="Gradient reconstruction method for the whole mesh")
    eigenvalue_regularization = 1e-8
    _diffusion_in_flux = True  # explicit diffusion in flux operator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.compute_dt is None:
            self.compute_dt = timestepping.adaptive(CFL=0.45)
        defaults = Settings.default()
        defaults.output.update(Zstruct(snapshots=10))
        defaults.update(self.settings)
        self.settings = defaults

    def initialize(self, mesh, model):
        Q, Qaux = super().initialize(mesh, model)
        Q = model.initial_conditions.apply(mesh.cell_centers, Q)
        Qaux = model.aux_initial_conditions.apply(mesh.cell_centers, Qaux)
        return Q, Qaux

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
        kernel = getattr(self, '_kernel', None)

        if eig_mode != "numerical":
            rt = NumpyRuntimeModel(symbolic_model, kernel=kernel)
            compiled_eig = rt.eigenvalues
            def max_ws(*args):
                Q = np.array(args[:n_vars])
                Qaux = np.array(args[n_vars:n_vars + n_aux])
                p = np.array(args[n_vars + n_aux:n_vars + n_aux + n_params])
                n = np.array(args[n_vars + n_aux + n_params:])
                evs = np.asarray(compiled_eig(Q, Qaux, p, n), dtype=float).ravel()
                return float(np.max(np.abs(evs)))
            return max_ws

        rt = NumpyRuntimeModel(symbolic_model, kernel=kernel)
        ql_fn = rt.quasilinear_matrix
        keys = list(symbolic_model.variables.keys())
        fi_h = keys.index("h") if "h" in keys else None
        dry_thr = self._get_dry_threshold(symbolic_model) if fi_h is not None else 0.0
        eps_reg = self.eigenvalue_regularization
        reg_diag = eps_reg * np.eye(n_vars)
        if fi_h is not None and "b" in keys:
            reg_diag[keys.index("b"), keys.index("b")] = 0.0

        def max_ws_numerical(*args):
            Q = np.array(args[:n_vars])
            Qaux = np.array(args[n_vars:n_vars + n_aux])
            p = np.array(args[n_vars + n_aux:n_vars + n_aux + n_params])
            n_vec = np.array(args[n_vars + n_aux + n_params:])
            if fi_h is not None and Q[fi_h] < dry_thr:
                return 0.0
            ql = np.asarray(ql_fn(Q, Qaux, p), dtype=float).reshape(n_vars, n_vars, dim)
            A_n = sum(ql[:, :, d] * float(n_vec[d]) for d in range(dim))
            A_n += reg_diag
            evs = np.real(np.linalg.eigvals(A_n))
            return float(np.max(np.abs(evs)))

        return max_ws_numerical

    def get_compute_max_abs_eigenvalue(self, mesh, model):
        symbolic_model = self._get_symbolic_model(model)
        max_ws = self._build_max_wavespeed(symbolic_model)
        keys = list(symbolic_model.variables.keys())
        fi_h = keys.index("h") if "h" in keys else None
        dry_thr = self._get_dry_threshold(symbolic_model) if fi_h is not None else 0.0
        dim = symbolic_model.dimension
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        normals = mesh.face_normals[:dim, :]
        has_aux = symbolic_model.n_aux_variables > 0

        def compute_max_eigenvalue(Q, Qaux, parameters):
            max_ev = np.zeros(mesh.n_faces)
            for f in range(mesh.n_faces):
                if fi_h is not None and Q[fi_h, iA[f]] < dry_thr and Q[fi_h, iB[f]] < dry_thr:
                    continue
                n = normals[:, f]
                for i_cell in [iA[f], iB[f]]:
                    q = Q[:, i_cell]
                    qaux = Qaux[:, i_cell] if has_aux else _EMPTY_AUX
                    ev = max_ws(*q, *qaux, *parameters, *n)
                    max_ev[f] = max(max_ev[f], ev)
            return max_ev
        return compute_max_eigenvalue

    # -- Flux operator (symbolic Riemann solver) -----------------------

    def _build_numerics(self, symbolic_model):
        """Build the symbolic Riemann solver. Override for SWE-specific variants."""
        return NonconservativeRusanov(symbolic_model)

    def _build_diffusion_operators(self, mesh, symbolic_model, dim, n_vars):
        """Build DiffusionOperator for each variable (if model has diffusion)."""
        if not hasattr(symbolic_model, 'diffusive_flux'):
            return None
        sym_dflux = symbolic_model.diffusive_flux()
        is_zero = hasattr(sym_dflux, 'tolist') and all(
            e == 0 for row in sym_dflux.tolist()
            for e in (row if isinstance(row, list) else [row])
        )
        if is_zero:
            return None
        from zoomy_core.fvm.reconstruction import DiffusionOperator
        nu_val = _param_value(symbolic_model, "nu", default=0.0)
        if nu_val <= 0:
            return None
        return {v: DiffusionOperator(mesh, dim, nu=nu_val) for v in range(n_vars)}

    def _build_reconstruction(self, mesh, symbolic_model):
        """Build the face reconstruction. Override for free-surface variants."""
        from zoomy_core.fvm.reconstruction import ConstantReconstruction, MUSCLReconstruction
        dim = symbolic_model.dimension
        if self.reconstruction_order >= 2:
            return MUSCLReconstruction(mesh, dim, limiter=self.limiter)
        return ConstantReconstruction(mesh, dim)

    def get_flux_operator(self, mesh, model):
        symbolic_model = self._get_symbolic_model(model)
        max_wavespeed_fn = self._build_max_wavespeed(symbolic_model)
        dim = symbolic_model.dimension

        numerics = self._build_numerics(symbolic_model)
        NumpyRuntimeSymbolic.module["max_wavespeed"] = max_wavespeed_fn
        runtime_numerics = numerics.to_runtime_numpy()
        runtime_numerics.local_max_abs_eigenvalue = (
            lambda Q, Qaux, p, n: max_wavespeed_fn(*Q, *Qaux, *p, *n)
        )

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        normals_arr = mesh.face_normals[:dim, :]
        face_volumes = mesh.face_volumes
        cell_volumesA = mesh.cell_volumes[iA]
        cell_volumesB = mesh.cell_volumes[iB]
        n_vars = symbolic_model.n_variables
        has_aux = symbolic_model.n_aux_variables > 0

        # Build reconstruction (resolved at setup, not per-step)
        reconstruct = self._build_reconstruction(mesh, symbolic_model)

        # Build diffusion operators (explicit in HyperbolicSolver, implicit in IMEX)
        self._diffusion_ops = self._build_diffusion_operators(mesh, symbolic_model, dim, n_vars)

        def flux_operator(dt, Q, Qaux, parameters, dQ):
            dQ = np.zeros_like(dQ)
            Q_L, Q_R = reconstruct(Q)
            for f in range(mesh.n_faces):
                qA = Q_L[:, f]
                qB = Q_R[:, f]
                qauxA = Qaux[:, iA[f]] if has_aux else _EMPTY_AUX
                qauxB = Qaux[:, iB[f]] if has_aux else _EMPTY_AUX
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
                dQ[:, iA[f]] -= (num_flux + Dm) * face_volumes[f] / cell_volumesA[f]
                dQ[:, iB[f]] -= (-num_flux + Dp) * face_volumes[f] / cell_volumesB[f]

            # Explicit diffusion (skipped when IMEX handles it implicitly)
            if self._diffusion_ops is not None and self._diffusion_in_flux:
                for v, diff_op in self._diffusion_ops.items():
                    Lu = diff_op.explicit(Q[v, :])
                    dQ[v, :mesh.n_inner_cells] += Lu

            return dQ
        return flux_operator

    # -- Source operator -----------------------------------------------

    def get_compute_source(self, mesh, model):
        def compute_source(dt, Q, Qaux, parameters, dQ):
            dQ = model.source(Q[:, :], Qaux[:, :], parameters)
            return dQ
        return compute_source

    def get_compute_source_jacobian_wrt_variables(self, mesh, model):
        def compute(dt, Q, Qaux, parameters, dQ):
            return model.source_jacobian_wrt_variables(
                Q[:, :mesh.n_inner_cells], Qaux[:, :mesh.n_inner_cells], parameters
            )
        return compute

    # -- Setup / Step / Run / Solve ------------------------------------

    def setup_simulation(self, mesh, model, write_output=True):
        """Build all operators once. Stores simulation state on self."""
        mesh = ensure_lsq_mesh(mesh, model)
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

        # Build operators (once)
        self._sim_compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model)
        self._sim_boundary_operator = self.get_apply_boundary_conditions(mesh, model)
        self._sim_flux_operator = self.get_flux_operator(mesh, model)
        self._sim_ode_step = ode.RK2 if self.reconstruction_order >= 2 else ode.RK1

        # Gradient operator + gradient-aware BC (for O2 boundary accuracy)
        self._sim_use_gradient_bc = self.reconstruction_order >= 2
        if self._sim_use_gradient_bc:
            symbolic_model = self._get_symbolic_model(model)
            self._sim_compute_gradients = self._build_gradient_operator(mesh, symbolic_model)
            self._sim_bc_with_gradient = self._build_bc_with_gradient(mesh, model)

        # Apply initial BCs and update
        self._sim_Q = self._sim_boundary_operator(0.0, Q, Qaux, parameters)
        self._sim_Q = self.update_q(self._sim_Q, Qaux, mesh, model, parameters)

        # Precompute mesh constant
        self._sim_cell_inradius_face = np.minimum(
            mesh.cell_inradius[mesh.face_cells[0, :]],
            mesh.cell_inradius[mesh.face_cells[1, :]],
        ).min()

        # Output setup
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            self._sim_save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            self._sim_save_fields = lambda time, time_stamp, i_snapshot, Q, Qaux: i_snapshot

    def _apply_bcs(self, time, Q, Qaux, parameters):
        """Apply BCs — base uses the standard operator without gradient."""
        Q = self._sim_boundary_operator(time, Q, Qaux, parameters)
        return Q

    def step(self, dt):
        """One explicit timestep with per-stage BC application.

        O1 (RK1): BCs → flux → advance
        O2 (RK2/Heun): BCs → flux → advance → BCs → flux → average
        """
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        parameters = self._sim_parameters
        time_now = self._sim_time
        mesh = self._sim_mesh
        model = self._sim_model
        flux = self._sim_flux_operator

        if self.reconstruction_order >= 2:
            # SSP-RK2 (Heun) with BCs applied per stage
            Q0 = np.array(Q)

            # Stage 1: BCs → reconstruct → flux → advance
            Q = self._apply_bcs(time_now, Q, Qaux, parameters)
            dQ = np.zeros_like(Q)
            dQ = flux(dt, Q, Qaux, parameters, dQ)
            Q1 = Q + dt * dQ

            # Stage 2: BCs → reconstruct → flux → advance
            Q1 = self._apply_bcs(time_now + dt, Q1, Qaux, parameters)
            dQ = flux(dt, Q1, Qaux, parameters, dQ)
            Q2 = Q1 + dt * dQ

            # Average
            Qnew = 0.5 * (Q0 + Q2)
        else:
            # RK1: BCs → flux → advance
            Q = self._apply_bcs(time_now, Q, Qaux, parameters)
            dQ = np.zeros_like(Q)
            dQ = flux(dt, Q, Qaux, parameters, dQ)
            Qnew = Q + dt * dQ

        # Final BC application + variable updates
        Qnew = self._apply_bcs(time_now, Qnew, Qaux, parameters)
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
    """Build positive Rusanov for free-surface models (requires h/b)."""
    keys = list(symbolic_model.variables.keys())
    if "h" not in keys or "b" not in keys:
        raise ValueError(
            f"Free-surface solver requires 'h' and 'b' in model variables, "
            f"got {keys}. Use HyperbolicSolver for general models."
        )
    field_map = _detect_field_map(symbolic_model)
    scaled_q_indices = _detect_scaled_q_indices(symbolic_model)
    return PositiveNonconservativeRusanov(
        symbolic_model,
        field_map=field_map,
        scaled_q_indices=scaled_q_indices,
    )


class FreeSurfaceFlowSolver(HyperbolicSolver):
    """Explicit FVM for free-surface flows (SWE, SME, VAM).

    Uses positive (hydrostatic reconstruction) Rusanov with wet/dry handling.
    Requires model variables 'b' and 'h'.
    """

    def _build_numerics(self, symbolic_model):
        return _build_free_surface_numerics(symbolic_model)

    def _build_reconstruction(self, mesh, symbolic_model):
        dim = symbolic_model.dimension
        if self.reconstruction_order >= 2:
            from zoomy_core.fvm.reconstruction import FreeSurfaceMUSCL
            h_idx = _var_index(symbolic_model, "h")
            eps_wet = self._get_dry_threshold(symbolic_model)
            return FreeSurfaceMUSCL(mesh, dim, h_index=h_idx,
                                    eps_wet=eps_wet, limiter=self.limiter)
        return super()._build_reconstruction(mesh, symbolic_model)
