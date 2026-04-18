"""IMEX solver: explicit flux + implicit source (numpy backend).

Extends ``HyperbolicSolver`` with:
- Derivative-aware auxiliary variable updates (``DerivativeAwareSolverMixin``)
- Implicit source stepping: local (cell-wise Newton) or global (Newton-GMRES)

Solver hierarchy:
    IMEXSolver(HyperbolicSolver)
      -> step(dt):
            apply_bcs -> reconstruct -> flux -> ode_step
            -> implicit_diffusion -> implicit_source -> update_state
"""

import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm import ode
from zoomy_core.fvm.jvp_numpy import analytic_source_jvp
from zoomy_core.model.derivative_workflow import DerivativeAwareSolverMixin
import zoomy_core.misc.io as io
from zoomy_core.misc.logger_config import logger
from zoomy_core.mesh import ensure_lsq_mesh


@dataclass
class IMEXStats:
    n_steps: int = 0
    source_mode: str = "auto"
    implicit_calls: int = 0
    implicit_time_s: float = 0.0
    init_time_s: float = 0.0
    runtime_only_s: float = 0.0
    total_time_s: float = 0.0


class IMEXSolver(DerivativeAwareSolverMixin, HyperbolicSolver):
    """IMEX: explicit flux (Riemann solver) + implicit source (Newton/GMRES).

    Inherits the symbolic Riemann solver from ``HyperbolicSolver``.
    Adds derivative-aware Qaux updates and implicit source stepping.

    Parameters
    ----------
    source_mode : "auto", "local", or "global"
        "local": cell-wise Newton (no inter-cell coupling in source)
        "global": Newton-GMRES (for derivative-coupled sources)
        "auto": chooses based on model.derivative_specs
    jv_backend : "analytic" or "fd"
        Jacobian-vector product method for global mode.
    """

    source_mode = "auto"
    _diffusion_in_flux = False  # diffusion handled implicitly, not in flux operator
    implicit_tol = 1e-8
    implicit_maxiter = 8
    gmres_tol = 1e-7
    gmres_maxiter = 40
    jv_backend = "analytic"
    fd_eps = 1e-7

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "last_stats", IMEXStats(source_mode=self.source_mode))

    def _resolve_source_mode(self, model):
        if self.source_mode in ("local", "global"):
            return self.source_mode
        symbolic_model = model.model if hasattr(model, "model") else model
        has_derivatives = hasattr(symbolic_model, "derivative_specs") and bool(
            symbolic_model.derivative_specs
        )
        return "global" if has_derivatives else "local"

    # -- Implicit source solvers ---------------------------------------

    def _implicit_source_local(self, Qe, Qaux, model, parameters, dt):
        Q = np.array(Qe, copy=True)
        n_cells = Q.shape[1]
        n_vars = Q.shape[0]
        for _ in range(self.implicit_maxiter):
            S = model.source(Q, Qaux, parameters)
            Jq = model.source_jacobian_wrt_variables(Q, Qaux, parameters)
            max_corr = 0.0
            for c in range(n_cells):
                A = np.eye(n_vars) - dt * Jq[:, :, c]
                r = Q[:, c] - Qe[:, c] - dt * S[:, c]
                try:
                    d = np.linalg.solve(A, -r)
                except np.linalg.LinAlgError:
                    d = -r
                Q[:, c] += d
                max_corr = max(max_corr, float(np.linalg.norm(d)))
            if max_corr < self.implicit_tol:
                break
        return Q

    def _implicit_source_global(self, Qe, Qaux, Qold, Qauxold, mesh,
                                model, parameters, time_now, dt):
        Q = np.array(Qe, copy=True)
        runtime_model = model
        symbolic_model = model.model if hasattr(model, "model") else None

        def residual(Qstate):
            Qaux_state = self.update_qaux(
                Qstate, Qaux, Qold, Qauxold, mesh, runtime_model, parameters, time_now, dt
            )
            S = runtime_model.source(Qstate, Qaux_state, parameters)
            return Qstate - Qe - dt * S

        for _ in range(self.implicit_maxiter):
            R = residual(Q)
            rnorm = float(np.linalg.norm(R))
            if rnorm < self.implicit_tol:
                break

            q_shape = Q.shape
            n = Q.size

            def matvec(v_flat):
                V = v_flat.reshape(q_shape)
                jv_source = self._compute_source_jvp_global(
                    runtime_model, symbolic_model, residual,
                    Q, Qaux, Qold, Qauxold, V, mesh, parameters, time_now, dt, R,
                )
                return (V - dt * jv_source).reshape(-1)

            A = LinearOperator((n, n), matvec=matvec, dtype=float)
            delta_flat, info = gmres(A, (-R).reshape(-1), atol=0.0,
                                     rtol=self.gmres_tol, maxiter=self.gmres_maxiter)
            if info != 0:
                delta_flat = (-R).reshape(-1)
            Q += delta_flat.reshape(q_shape)
        return Q

    def _compute_source_jvp_global(self, runtime_model, symbolic_model, residual,
                                    Q, Qaux, Qold, Qauxold, V, mesh, parameters,
                                    time_now, dt, base_residual):
        if self.jv_backend == "analytic":
            if symbolic_model is None:
                raise RuntimeError("Analytic Jv requires symbolic model.")
            Qaux_state = self.update_qaux(
                Q, Qaux, Qold, Qauxold, mesh, runtime_model, parameters, time_now, dt
            )
            return analytic_source_jvp(
                runtime_model, symbolic_model, Q, Qaux_state, V, mesh, dt,
                include_chain_rule=True,
            )
        # Finite difference fallback
        Rp = residual(Q + self.fd_eps * V)
        return (Rp - base_residual) / self.fd_eps

    # -- Setup / Step / Run / Solve ------------------------------------

    def setup_simulation(self, mesh, model, write_output=False):
        """Build all operators. Extends HyperbolicSolver.setup_simulation."""
        t0 = time.time()
        self._imex_reconstruction_order = self.reconstruction_order
        super().setup_simulation(mesh, model, write_output=write_output)

        # Resolve source mode
        self._sim_source_mode = self._resolve_source_mode(self._sim_model)
        object.__setattr__(self, "last_stats", IMEXStats(source_mode=self._sim_source_mode))
        self.last_stats.init_time_s = time.time() - t0

        # Output setup for IMEX (override parent's to add HDF5 init)
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            self._sim_save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
            io.init_output_directory(
                self.settings.output.directory, self.settings.output.clean_directory
            )
            self._sim_mesh.write_to_hdf5(output_hdf5_path)
            io.save_settings(self.settings)

    def step(self, dt):
        """One IMEX timestep: explicit flux -> implicit diffusion -> implicit source.

        Ghost-cell-free: BCs are evaluated inside flux_operator.
        """
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        parameters = self._sim_parameters
        time_now = self._sim_time
        mesh = self._sim_mesh
        model = self._sim_model

        # Explicit flux step (convection) — BCs inside flux_operator
        if self.reconstruction_order >= 2:
            Q0 = np.array(Q)
            dQ = self._sim_flux_operator(dt, time_now, Q, Qaux, parameters, np.zeros_like(Q))
            Q1 = Q + dt * dQ
            dQ = self._sim_flux_operator(dt, time_now + dt, Q1, Qaux, parameters, np.zeros_like(Q))
            Q2 = Q1 + dt * dQ
            Qexp = 0.5 * (Q0 + Q2)
        else:
            dQ = self._sim_flux_operator(dt, time_now, Q, Qaux, parameters, np.zeros_like(Q))
            Qexp = Q + dt * dQ

        # Implicit diffusion step (with boundary face gradients)
        Qexp = self._apply_implicit_diffusion(Qexp, Qaux, dt, time_now)

        # Implicit source step
        Qimp = self._apply_implicit_source(Qexp, Q, Qaux, mesh, model, parameters, time_now, dt)

        # Variable + auxiliary updates
        Qnew = self.update_q(Qimp, Qaux, mesh, model, parameters)
        Qauxnew = self.update_qaux(
            Qnew, Qaux, Q, Qaux, mesh, model, parameters, time_now + dt, dt,
        )

        # Commit new state
        self._sim_Q = Qnew
        self._sim_Qaux = Qauxnew

    def _apply_implicit_diffusion(self, Qexp, Qaux, dt, time_now):
        """Apply implicit diffusion with boundary face gradients."""
        if not hasattr(self, '_diffusion_ops') or self._diffusion_ops is None:
            return Qexp

        # Compute boundary face gradients from BC objects
        from zoomy_core.fvm.solver_numpy import _compute_bf_face_gradients
        n_vars = Qexp.shape[0]
        has_aux = Qaux.shape[0] > 0
        normals_arr = self._sim_mesh.face_normals[:self._sim_mesh.dimension, :]

        # Compute face values for face_gradient
        bf_values = np.zeros((n_vars, self._n_bf))
        for i_bf in range(self._n_bf):
            q_inner = Qexp[:, self._bf_cells[i_bf]]
            qaux_inner = Qaux[:, self._bf_cells[i_bf]] if has_aux else np.array([])
            normal = normals_arr[:, self._bf_fidx[i_bf]]
            bf_values[:, i_bf] = self._bc_objects[i_bf].face_value(
                q_inner, qaux_inner, normal, self._d_face[i_bf],
                time_now, self._sim_parameters,
            )

        bf_grads = _compute_bf_face_gradients(
            Qexp, Qaux, bf_values, self._bc_objects, self._bf_cells,
            self._bf_fidx, self._d_face, normals_arr, self._n_bf,
            n_vars, has_aux, time_now, self._sim_parameters,
        )

        for v, diff_op in self._diffusion_ops.items():
            Qexp[v, :] = diff_op.implicit_solve_with_bc(Qexp[v, :], dt, bf_grads[v])
        return Qexp

    def _apply_implicit_source(self, Qexp, Qold, Qauxold, mesh, model, parameters, time_now, dt):
        """Apply implicit source stepping (local or global Newton)."""
        t_imp0 = time.time()
        source_mode = self._sim_source_mode

        if source_mode == "local":
            Qaux_imp = self.update_qaux(
                Qexp, Qauxold, Qold, Qauxold, mesh, model, parameters, time_now, dt,
            )
            Qimp = self._implicit_source_local(Qexp, Qaux_imp, model, parameters, dt)
        else:
            Qimp = self._implicit_source_global(
                Qexp, Qauxold, Qold, Qauxold, mesh, model, parameters,
                time_now, dt,
            )

        self.last_stats.implicit_time_s += time.time() - t_imp0
        self.last_stats.implicit_calls += 1
        return Qimp

    def run_simulation(self):
        """Time loop for IMEX solver."""
        time_now = 0.0
        dt_snapshot = self.time_end / max(self.settings.output.snapshots - 1, 1)
        i_snapshot = 0.0
        t_run0 = time.time()

        while time_now < self.time_end:
            dt = self.compute_dt(
                self._sim_Q, self._sim_Qaux, self._sim_parameters,
                self._sim_cell_inradius_face, self._sim_compute_max_abs_eigenvalue,
            )
            remaining = float(self.time_end - time_now)
            if remaining < 1e-10 * max(self.time_end, 1.0):
                break
            dt = min(float(dt), remaining)
            if not np.isfinite(dt) or dt <= 0.0:
                break
            if dt < self.min_dt:
                raise RuntimeError(f"IMEX dt={dt:.3e} < min_dt={self.min_dt:.3e}")

            self._sim_time = time_now
            self.step(dt)

            time_now += dt
            self.last_stats.n_steps += 1
            i_snapshot = self._sim_save_fields(
                time_now, i_snapshot * dt_snapshot, i_snapshot,
                self._sim_Q, self._sim_Qaux,
            )

            if self.last_stats.n_steps % 10 == 0:
                logger.info(
                    f"imex it={self.last_stats.n_steps}, t={time_now:.6f}, "
                    f"dt={dt:.6f}, mode={self._sim_source_mode}"
                )

        self._sim_time = time_now
        self.last_stats.runtime_only_s = time.time() - t_run0
        self.last_stats.total_time_s = self.last_stats.init_time_s + self.last_stats.runtime_only_s
        return self._sim_Q, self._sim_Qaux

    def solve(self, mesh, model, write_output=False):
        """Convenience: setup_simulation + run_simulation."""
        self.setup_simulation(mesh, model, write_output=write_output)
        return self.run_simulation()


# -- Free-surface IMEX variant -------------------------------------------------

from zoomy_core.fvm.solver_numpy import _build_free_surface_numerics


class FSFIMEXSolver(IMEXSolver):
    """IMEX solver for free-surface flows (SWE, SME, VAM).

    Combines:
    - Positive (hydrostatic reconstruction) Rusanov for explicit flux
    - Implicit source stepping (local/global Newton-GMRES)
    - Derivative-aware Qaux updates

    Requires model variables 'b' and 'h'.
    """

    def _build_numerics(self, symbolic_model):
        return _build_free_surface_numerics(symbolic_model)
