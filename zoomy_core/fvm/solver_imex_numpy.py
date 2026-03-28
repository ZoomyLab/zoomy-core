import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm import ode
from zoomy_core.fvm.jvp_numpy import analytic_source_jvp
from zoomy_core.model.derivative_workflow import DerivativeAwareSolverMixin
import os
import zoomy_core.misc.io as io
from zoomy_core.misc.logger_config import logger


@dataclass
class IMEXStats:
    n_steps: int = 0
    source_mode: str = "auto"
    implicit_calls: int = 0
    implicit_time_s: float = 0.0
    init_time_s: float = 0.0
    runtime_only_s: float = 0.0
    total_time_s: float = 0.0


class IMEXSourceSolver(DerivativeAwareSolverMixin, HyperbolicSolver):
    """
    IMEX solver:
      - flux / nonconservative terms explicit (existing path-conservative flux op)
      - source terms implicit:
          * local (cellwise) if source is local
          * global Newton-Krylov for derivative-coupled sources
    """

    source_mode = "auto"  # "auto" | "local" | "global"
    implicit_tol = 1e-8
    implicit_maxiter = 8
    gmres_tol = 1e-7
    gmres_maxiter = 40
    # jv_backend controls the Jacobian-vector product in global implicit source:
    # - "analytic": symbolic source Jacobians + chain rule through Qaux(Q)
    # - "fd": finite-difference Jv on the full residual
    jv_backend = "analytic"  # "analytic" | "fd"
    use_analytic_chain_jvp = True  # backward-compatible switch, mapped to jv_backend
    fd_eps = 1e-7

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "last_stats", IMEXStats(source_mode=self.source_mode))

    def _resolve_source_mode(self, model):
        if self.source_mode in ("local", "global"):
            return self.source_mode
        symbolic_model = model.model if hasattr(model, "model") else model
        has_derivative_specs = hasattr(symbolic_model, "derivative_specs") and bool(
            symbolic_model.derivative_specs
        )
        return "global" if has_derivative_specs else "local"

    def _resolve_jv_backend(self):
        if self.jv_backend in ("analytic", "fd"):
            return self.jv_backend
        # Compatibility mapping for existing scripts.
        return "analytic" if self.use_analytic_chain_jvp else "fd"

    def _implicit_source_local(self, Qe, Qaux, model, parameters, dt):
        Q = np.array(Qe, copy=True)
        n_cells = Q.shape[1]
        n_vars = Q.shape[0]
        for _ in range(self.implicit_maxiter):
            S = model.source(Q, Qaux, parameters)
            Jq = model.source_jacobian_wrt_variables(Q, Qaux, parameters)  # (nvar,nvar,nc)
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

    def _implicit_source_global(self, Qe, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt, boundary_operator):
        runtime_model = model
        symbolic_model = model.model if hasattr(model, "model") else None
        Q = np.array(Qe, copy=True)

        def residual(Qstate):
            Qaux_state = self.update_qaux(
                Qstate, Qaux, Qold, Qauxold, mesh, runtime_model, parameters, time, dt
            )
            Qstate_bc = boundary_operator(time, Qstate, Qaux_state, parameters)
            S = runtime_model.source(Qstate_bc, Qaux_state, parameters)
            return Qstate_bc - Qe - dt * S

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
                    runtime_model=runtime_model,
                    symbolic_model=symbolic_model,
                    residual=residual,
                    Q=Q,
                    Qaux=Qaux,
                    Qold=Qold,
                    Qauxold=Qauxold,
                    V=V,
                    mesh=mesh,
                    parameters=parameters,
                    time=time,
                    dt=dt,
                    base_residual=R,
                )
                jv = V - dt * jv_source
                return jv.reshape(-1)

            A = LinearOperator((n, n), matvec=matvec, dtype=float)
            b = (-R).reshape(-1)
            delta_flat, info = gmres(
                A,
                b,
                atol=0.0,
                rtol=self.gmres_tol,
                maxiter=self.gmres_maxiter,
            )
            if info != 0:
                delta_flat = b
            Q += delta_flat.reshape(q_shape)
            Q = boundary_operator(time, Q, Qaux, parameters)
        return Q

    def _compute_source_jvp_global(
        self,
        runtime_model,
        symbolic_model,
        residual,
        Q,
        Qaux,
        Qold,
        Qauxold,
        V,
        mesh,
        parameters,
        time,
        dt,
        base_residual=None,
    ):
        backend = self._resolve_jv_backend()
        if backend == "analytic":
            if symbolic_model is None:
                raise RuntimeError(
                    "Analytic Jv requested but symbolic model is unavailable."
                )
            Qaux_state = self.update_qaux(
                Q, Qaux, Qold, Qauxold, mesh, runtime_model, parameters, time, dt
            )
            return analytic_source_jvp(
                runtime_model,
                symbolic_model,
                Q,
                Qaux_state,
                V,
                mesh,
                dt,
                include_chain_rule=True,
            )

        if backend == "fd":
            eps = self.fd_eps
            if base_residual is None:
                base_residual = residual(Q)
            Rp = residual(Q + eps * V)
            return (Rp - base_residual) / eps

        raise ValueError(f"Unsupported jv_backend '{backend}'.")

    def solve(self, mesh, model, write_output=False):
        t0 = time.time()
        t_init0 = time.time()
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)
        Qaux = self.update_qaux(Q, Qaux, Q, Qaux, mesh, model, parameters, 0.0, 1.0)

        compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model)
        flux_operator = self.get_flux_operator(mesh, model)
        boundary_operator = self.get_apply_boundary_conditions(mesh, model)
        source_mode = self._resolve_source_mode(model)
        object.__setattr__(self, "last_stats", IMEXStats(source_mode=source_mode))
        self.last_stats.init_time_s = time.time() - t_init0

        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
            io.init_output_directory(
                self.settings.output.directory, self.settings.output.clean_directory
            )
            mesh.write_to_hdf5(output_hdf5_path)
            io.save_settings(self.settings)
            if self.settings.output.snapshots > 1:
                dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
            else:
                dt_snapshot = self.time_end
            i_snapshot = 0.0
            i_snapshot = save_fields(0.0, 0.0, i_snapshot, Q, Qaux)
        else:
            def save_fields(time, next_write_at, i_snapshot, Q, Qaux):
                return i_snapshot
            dt_snapshot = self.time_end
            i_snapshot = 0.0

        Qnew = boundary_operator(0.0, Q, Qaux, parameters)
        Qauxnew = Qaux
        time_now = 0.0
        cell_inradius_face = np.minimum(
            mesh.cell_inradius[mesh.face_cells[0, :]],
            mesh.cell_inradius[mesh.face_cells[1, :]],
        ).min()
        t_run0 = time.time()

        while time_now < self.time_end:
            Qold = Qnew
            Qauxold = Qauxnew
            dt = self.compute_dt(
                Qold, Qauxold, parameters, cell_inradius_face, compute_max_abs_eigenvalue
            )
            dt = min(float(dt), float(self.time_end - time_now))
            if not np.isfinite(dt) or dt <= 0.0:
                raise RuntimeError(
                    f"Invalid IMEX time step detected: dt={dt}. "
                    "Aborting to prevent unstable update."
                )
            if dt < self.min_dt:
                raise RuntimeError(
                    f"IMEX time step below stability floor: dt={dt:.3e} < min_dt={self.min_dt:.3e}. "
                    "Aborting to prevent unstable update."
                )

            # Explicit flux/nonconservative part
            Qexp = ode.RK1(flux_operator, Qold, Qauxold, parameters, dt)
            Qexp = boundary_operator(time_now, Qexp, Qauxold, parameters)

            t_imp0 = time.time()
            if source_mode == "local":
                Qaux_imp = self.update_qaux(
                    Qexp, Qauxold, Qold, Qauxold, mesh, model, parameters, time_now, dt
                )
                Qimp = self._implicit_source_local(Qexp, Qaux_imp, model, parameters, dt)
            else:
                Qimp = self._implicit_source_global(
                    Qexp, Qauxold, Qold, Qauxold, mesh, model, parameters, time_now, dt, boundary_operator
                )
            self.last_stats.implicit_time_s += time.time() - t_imp0
            self.last_stats.implicit_calls += 1

            Qnew = boundary_operator(time_now + dt, Qimp, Qauxold, parameters)
            Qauxnew = self.update_qaux(
                Qnew, Qauxold, Qold, Qauxold, mesh, model, parameters, time_now + dt, dt
            )

            time_now += dt
            self.last_stats.n_steps += 1
            time_stamp = i_snapshot * dt_snapshot
            i_snapshot = save_fields(time_now, time_stamp, i_snapshot, Qnew, Qauxnew)

            if self.last_stats.n_steps % 10 == 0:
                logger.info(
                    f"imex iteration: {int(self.last_stats.n_steps)}, "
                    f"time: {float(time_now):.6f}, dt: {float(dt):.6f}, "
                    f"implicit_calls: {int(self.last_stats.implicit_calls)}, "
                    f"source_mode: {self.last_stats.source_mode}"
                )

        self.last_stats.runtime_only_s = time.time() - t_run0
        self.last_stats.total_time_s = time.time() - t0
        return Qnew, Qauxnew
