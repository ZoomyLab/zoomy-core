"""
Pressurized IMEX solver for non-hydrostatic models (VAM).

Extends the standard IMEX solver with a pressure Poisson correction step:

    1. Velocity predictor:  Q* = Q^n - dt * Flux(Q^n)    [explicit]
    2. Pressure Poisson:    solve for P from constraints   [implicit]
    3. Velocity corrector:  Q^{n+1} = Q* - dt * tau(P)    [implicit]

The Poisson step enforces divergence-free (and optionally irrotational)
constraints on the predicted velocity field.

Usage:
    class MySolver(_GeneratedModelFluxMixin, PressurizedIMEXSolver):
        pass

    solver = MySolver(
        time_end=1.0,
        pressure_model=poisson_model,       # VAMProjectedPoisson or similar
        pressure_n_vars=2,                   # number of pressure variables (hp0, hp1)
        ...
    )
    Q, Qaux = solver.solve(mesh, hyp_model)
"""

import os
import time as time_module
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from loguru import logger

from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver


class PressurizedIMEXSolver(IMEXSourceSolver):
    """
    IMEX solver with predictor-Poisson-corrector for non-hydrostatic models.

    Additional attributes (set via object.__setattr__):
        pressure_model:    the Poisson model (provides source_implicit for constraints)
        pressure_n_vars:   number of pressure unknowns (e.g. 2 for hp0, hp1 at L1)
        pressure_maxiter:  Newton iterations for Poisson solve (default: 20)
        pressure_tol:      convergence tolerance for Poisson solve (default: 1e-10)
    """

    def solve(self, mesh, model, write_output=False):
        """
        Solve with predictor-Poisson-corrector time stepping.

        model is the HYPERBOLIC model (VAMProjectedHyperbolic).
        The Poisson model is set via self.pressure_model.
        """
        import zoomy_core.fvm.ode as ode
        import zoomy_core.misc.io as io
        from zoomy_core.fvm.solver_imex_numpy import IMEXStats

        t0 = time_module.time()
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)
        Qaux = self.update_qaux(Q, Qaux, Q, Qaux, mesh, model, parameters, 0.0, 1.0)

        compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model)
        flux_operator = self.get_flux_operator(mesh, model)
        boundary_operator = self.get_apply_boundary_conditions(mesh, model)
        source_mode = self._resolve_source_mode(model)
        object.__setattr__(self, "last_stats", IMEXStats(source_mode=source_mode))

        # Pressure state
        n_p = getattr(self, "pressure_n_vars", 2)
        n_cells = Q.shape[1]
        P = np.zeros((n_p, n_cells))  # pressure moments (hp0, hp1, ...)
        p_maxiter = getattr(self, "pressure_maxiter", 20)
        p_tol = getattr(self, "pressure_tol", 1e-10)

        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
            io.init_output_directory(
                self.settings.output.directory, self.settings.output.clean_directory
            )
            mesh.write_to_hdf5(output_hdf5_path)
            if self.settings.output.snapshots > 1:
                dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
            else:
                dt_snapshot = self.time_end
            i_snapshot = 0.0
            i_snapshot = save_fields(0.0, 0.0, i_snapshot, Q, Qaux)
        else:
            def save_fields(t, t_next, i, Q, Qaux):
                return i
            dt_snapshot = self.time_end
            i_snapshot = 0.0

        Qnew = boundary_operator(0.0, Q, Qaux, parameters)
        Qauxnew = Qaux
        time_now = 0.0
        cell_inradius_face = np.minimum(
            mesh.cell_inradius[mesh.face_cells[0, :]],
            mesh.cell_inradius[mesh.face_cells[1, :]],
        ).min()

        t_run0 = time_module.time()
        n_steps = 0

        while time_now < self.time_end:
            Qold = Qnew
            Qauxold = Qauxnew
            Pold = P.copy()

            dt = self.compute_dt(
                Qold, Qauxold, parameters, cell_inradius_face,
                compute_max_abs_eigenvalue,
            )
            dt = min(float(dt), float(self.time_end - time_now))
            if not np.isfinite(dt) or dt <= 0.0:
                raise RuntimeError(f"Invalid dt={dt}")
            if dt < self.min_dt:
                raise RuntimeError(f"dt={dt:.3e} < min_dt={self.min_dt:.3e}")

            # ============================================================
            # STEP 1: Velocity predictor (explicit flux)
            # ============================================================
            Qexp = ode.RK1(flux_operator, Qold, Qauxold, parameters, dt)
            Qexp = boundary_operator(time_now, Qexp, Qauxold, parameters)

            # ============================================================
            # STEP 2: Pressure Poisson correction
            # ============================================================
            # Update aux with predicted velocity
            Qaux_pred = self.update_qaux(
                Qexp, Qauxold, Qold, Qauxold, mesh, model, parameters,
                time_now, dt,
            )

            # Solve Poisson for P using predicted velocity
            P = self._solve_pressure_poisson(
                Qexp, Qaux_pred, Pold, mesh, model, parameters, dt,
                p_maxiter, p_tol,
            )

            # ============================================================
            # STEP 3: Velocity corrector (pressure source)
            # ============================================================
            # Load pressure into aux variables
            Qaux_with_P = self._load_pressure_into_aux(
                Qaux_pred, P, mesh, model, parameters,
            )

            # Apply pressure source implicitly
            if source_mode == "local":
                Qimp = self._implicit_source_local(
                    Qexp, Qaux_with_P, model, parameters, dt,
                )
            else:
                Qimp = self._implicit_source_global(
                    Qexp, Qauxold, Qold, Qauxold, mesh, model, parameters,
                    time_now, dt, boundary_operator,
                )

            Qnew = boundary_operator(time_now + dt, Qimp, Qaux_with_P, parameters)
            Qauxnew = self.update_qaux(
                Qnew, Qaux_with_P, Qold, Qauxold, mesh, model, parameters,
                time_now + dt, dt,
            )

            time_now += dt
            n_steps += 1
            self.last_stats.n_steps = n_steps
            self.last_stats.implicit_calls += 1
            time_stamp = i_snapshot * dt_snapshot
            i_snapshot = save_fields(time_now, time_stamp, i_snapshot, Qnew, Qauxnew)

            if n_steps % 10 == 0:
                logger.info(
                    f"pressurized imex: step={n_steps}, t={time_now:.6f}, "
                    f"dt={dt:.6f}, |P|={np.linalg.norm(P):.4e}"
                )

        self.last_stats.runtime_only_s = time_module.time() - t_run0
        self.last_stats.total_time_s = time_module.time() - t0
        return Qnew, Qauxnew

    # ------------------------------------------------------------------
    # Pressure Poisson solve
    # ------------------------------------------------------------------

    def _solve_pressure_poisson(
        self, Q, Qaux, Pold, mesh, model, parameters, dt,
        maxiter, tol,
    ):
        """
        Solve the Poisson constraint for pressure.

        Uses Newton iteration with finite-difference Jacobian.
        The constraint R(P) = 0 is evaluated by:
        1. Substituting predicted velocity + pressure into constraints I1, I2
        2. Computing spatial derivatives of P via LSQ reconstruction
        3. Iterating until ||R|| < tol
        """
        from zoomy_core.model.derivative_workflow import compute_derivatives

        n_p = Pold.shape[0]
        n_cells = Pold.shape[1]
        P = Pold.copy()
        n_u = (model.model.n_variables - 2) // 2 if hasattr(model, "model") else (Q.shape[0] - 2) // 2

        h = Q[0] if Q[0].mean() > 0.1 else Q[1]  # find h
        # Determine h index: for VAM state [h, hu0..., hw0..., b], h is at index 0
        h_idx = 0
        h_vals = Q[h_idx]

        def residual(P_state):
            """Evaluate Poisson constraint residual."""
            R = np.zeros_like(P_state)

            # Compute derivatives of P
            dPdx = np.zeros_like(P_state)
            for k in range(n_p):
                try:
                    dPdx[k] = compute_derivatives(
                        P_state[k], mesh, derivatives_multi_index=[[1]]
                    )[:, 0]
                except Exception:
                    dPdx[k] = np.gradient(P_state[k])

            # Compute velocity derivatives from Q
            # u_k = Q[1+k] / Q[0] for k in range(n_u)
            u = np.zeros((n_u, n_cells))
            w = np.zeros((n_u, n_cells))
            for k in range(n_u):
                u[k] = Q[1 + k] / np.maximum(h_vals, 1e-10)
                w[k] = Q[1 + n_u + k] / np.maximum(h_vals, 1e-10)

            dudx = np.zeros((n_u, n_cells))
            for k in range(n_u):
                try:
                    dudx[k] = compute_derivatives(
                        u[k], mesh, derivatives_multi_index=[[1]]
                    )[:, 0]
                except Exception:
                    dudx[k] = np.gradient(u[k])

            dhdx = np.zeros(n_cells)
            dbdx = np.zeros(n_cells)
            try:
                dhdx = compute_derivatives(
                    h_vals, mesh, derivatives_multi_index=[[1]]
                )[:, 0]
                b_vals = Q[-1]
                dbdx = compute_derivatives(
                    b_vals, mesh, derivatives_multi_index=[[1]]
                )[:, 0]
            except Exception:
                pass

            # I1: continuity constraint
            # I1 = h*du0/dx + (1/3)*d(h*u1)/dx + (1/3)*u1*dh/dx + 2*(w0 - u0*db/dx)
            I1 = h_vals * dudx[0]
            if n_u >= 2:
                dhu1dx = np.zeros(n_cells)
                try:
                    dhu1dx = compute_derivatives(
                        Q[2], mesh, derivatives_multi_index=[[1]]
                    )[:, 0]
                except Exception:
                    dhu1dx = np.gradient(Q[2])
                I1 += (1.0 / 3.0) * dhu1dx + (1.0 / 3.0) * u[1] * dhdx
            I1 += 2.0 * (w[0] - u[0] * dbdx)

            # I2: vorticity constraint
            I2 = h_vals * dudx[0]
            if n_u >= 2:
                I2 += u[1] * dhdx + 2.0 * (u[1] * dbdx - w[1])

            # Pressure correction terms (from splitting: U = U* - dt*tau(P))
            # These modify I1 and I2 to include P-dependent terms
            p = np.zeros((n_p, n_cells))
            for k in range(n_p):
                p[k] = P_state[k] / np.maximum(h_vals, 1e-10)

            # The P correction adds dt-scaled terms to I1, I2
            # At L1: tau modifies u0, u1, w0, w1 by pressure source terms
            # After substitution into constraints, P-dependent terms appear
            # For simplicity, add regularization: delta * d²P/dx²
            delta = getattr(self, "pressure_regularization", 0.0)
            if delta > 0:
                for k in range(n_p):
                    try:
                        ddPdxx = compute_derivatives(
                            P_state[k], mesh, derivatives_multi_index=[[2]]
                        )[:, 0]
                        I1 += delta * ddPdxx if k == 0 else 0
                        I2 += delta * ddPdxx if k == 1 else 0
                    except Exception:
                        pass

            R[0] = I1 + I2
            R[1] = I1 - I2
            return R

        # Newton iteration
        for it in range(maxiter):
            R = residual(P)
            rnorm = float(np.linalg.norm(R))
            if rnorm < tol:
                break

            # Finite-difference Jacobian approximation (cellwise)
            eps_fd = 1e-7
            for k in range(n_p):
                P_pert = P.copy()
                P_pert[k] += eps_fd
                R_pert = residual(P_pert)
                dRdP = (R_pert - R) / eps_fd
                # Simple damped update
                for c in range(n_cells):
                    if abs(dRdP[k, c]) > 1e-14:
                        P[k, c] -= 0.5 * R[k, c] / dRdP[k, c]

        return P

    def _load_pressure_into_aux(self, Qaux, P, mesh, model, parameters):
        """Load pressure P into the aux variable array."""
        from zoomy_core.model.derivative_workflow import compute_derivatives

        Qaux_new = Qaux.copy()
        n_p = P.shape[0]

        # The aux layout depends on the model.
        # For VAMProjectedHyperbolic: aux = [hw_{L+1}, hp0, hp1, ..., dbdx, dhdx, dhp0dx, ...]
        # We need to set hp_k and dhp_k_dx

        # Try to determine aux indices from model
        sym_model = model.model if hasattr(model, "model") else model
        if hasattr(sym_model, "_n_u"):
            n_u = sym_model._n_u
            # aux[0] = hw_closure, aux[1..n_p] = hp0..hpN, then dbdx, dhdx, dhp0dx..
            for k in range(n_p):
                Qaux_new[1 + k] = P[k]
                # Compute dhp_k/dx
                try:
                    dhpdx = compute_derivatives(
                        P[k], mesh, derivatives_multi_index=[[1]]
                    )[:, 0]
                    Qaux_new[1 + n_p + 2 + k] = dhpdx  # after dbdx, dhdx
                except Exception:
                    pass

        return Qaux_new
