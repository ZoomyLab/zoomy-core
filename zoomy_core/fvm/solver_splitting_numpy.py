"""Pressure-splitting solver: IMEX + Poisson constraint (numpy backend).

Extends ``IMEXSolver`` with a pressure Poisson correction step (Chorin
projection). Uses LSQ degree-2 reconstruction for the Laplacian and
divergence, and GMRES for the Poisson solve.

Solver steps per timestep:
1. Explicit flux (Riemann solver, from HyperbolicSolver)
2. Implicit source (Newton/GMRES, from IMEXSolver)
3. Divergence of velocity → Poisson RHS
4. Solve ∇²p = (1/Δt)∇·u*  via GMRES
5. Correct velocity: u = u* - Δt·∇p
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres

from zoomy_core.fvm.solver_imex_numpy import IMEXSolver, IMEXStats
from zoomy_core.fvm.solver_numpy import _build_free_surface_numerics
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.mesh.lsq_reconstruction import find_derivative_indices
import zoomy_core.fvm.ode as ode
import zoomy_core.misc.io as io
from zoomy_core.misc.logger_config import logger


import param


class SplittingSolver(IMEXSolver):
    """IMEX solver with pressure Poisson correction (Chorin projection).

    The model state Q holds velocity components.  Pressure is a separate
    field solved at each timestep via the incompressibility constraint.
    """

    pressure_gmres_tol = param.Number(default=1e-6, doc="GMRES tolerance for Poisson")
    pressure_gmres_maxiter = param.Integer(default=200, doc="Max GMRES iterations for Poisson")
    viscosity = param.Number(default=0.01, doc="Kinematic viscosity ν")

    def solve(self, mesh, model, write_output=False):
        t0 = time.time()
        mesh = ensure_lsq_mesh(mesh, model, lsq_degree=2)
        dim = model.dimension
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model_rt = self.create_runtime(Q, Qaux, mesh, model)

        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells
        n_vars = Q.shape[0]
        nu = self.viscosity

        # Pressure field
        p = np.zeros(n_cells, dtype=float)

        # Precompute face connectivity
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        face_normals = mesh.face_normals[:dim, :]
        face_volumes = mesh.face_volumes
        cell_vol = mesh.cell_volumes

        # ── Build sparse Laplacian matrix for Poisson ─────────────────
        L = lil_matrix((nc, nc), dtype=float)
        for f in range(mesh.n_faces):
            a, b = iA[f], iB[f]
            dist = np.linalg.norm(
                mesh.cell_centers[:dim, b] - mesh.cell_centers[:dim, a]
            )
            if dist < 1e-14:
                continue
            coeff = face_volumes[f] / dist
            if a < nc and b < nc:
                L[a, b] += coeff / cell_vol[a]
                L[a, a] -= coeff / cell_vol[a]
                L[b, a] += coeff / cell_vol[b]
                L[b, b] -= coeff / cell_vol[b]
            elif a < nc:
                # b is ghost → Neumann: p[b]=p[a], net contribution = 0
                pass
            elif b < nc:
                pass

        # Pin one pressure DOF to remove null space
        L[0, :] = 0
        L[0, 0] = 1.0
        L_csr = csr_matrix(L)

        # GMRES wrapper
        def solve_poisson(rhs):
            rhs_inner = rhs[:nc].copy()
            rhs_inner[0] = 0.0  # pinned DOF
            result = np.zeros(n_cells)

            def matvec(x):
                return L_csr @ x

            A_op = LinearOperator((nc, nc), matvec=matvec, dtype=float)
            sol, info = gmres(A_op, rhs_inner, atol=0.0,
                              rtol=self.pressure_gmres_tol,
                              maxiter=self.pressure_gmres_maxiter)
            result[:nc] = sol
            # Ghost cells: Neumann
            for i_bf in range(mesh.n_boundary_faces):
                ghost = mesh.boundary_face_ghosts[i_bf]
                inner = mesh.boundary_face_cells[i_bf]
                result[ghost] = result[inner]
            return result, info

        # ── Time stepping ─────────────────────────────────────────────
        compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model_rt)
        boundary_operator = self.get_apply_boundary_conditions(mesh, model_rt)
        flux_operator = self.get_flux_operator(mesh, model_rt)

        cell_inradius_face = np.minimum(
            mesh.cell_inradius[mesh.face_cells[0, :]],
            mesh.cell_inradius[mesh.face_cells[1, :]],
        ).min()

        time_now = 0.0
        iteration = 0
        Qnew = boundary_operator(0.0, Q, Qaux, parameters)

        while time_now < self.time_end:
            dt = self.compute_dt(Qnew, Qaux, parameters, cell_inradius_face,
                                 compute_max_abs_eigenvalue)
            dt = min(float(dt), float(self.time_end - time_now))
            if not np.isfinite(dt) or dt <= 0:
                break

            # Step 1: Explicit convective flux
            Q_star = ode.RK1(flux_operator, Qnew, Qaux, parameters, dt)
            Q_star = boundary_operator(time_now, Q_star, Qaux, parameters)

            # Step 2: Explicit viscous diffusion
            for d in range(min(n_vars, dim)):
                u_d = Q_star[d, :]
                lap = np.zeros(n_cells)
                for f in range(mesh.n_faces):
                    a, b = iA[f], iB[f]
                    diff = u_d[b] - u_d[a]
                    dist = np.linalg.norm(
                        mesh.cell_centers[:dim, b] - mesh.cell_centers[:dim, a]
                    )
                    if dist > 1e-14:
                        lap_contrib = diff / dist * face_volumes[f]
                        lap[a] += lap_contrib / cell_vol[a]
                        lap[b] -= lap_contrib / cell_vol[b]
                Q_star[d, :] += dt * nu * lap

            Q_star = boundary_operator(time_now, Q_star, Qaux, parameters)

            # Step 3: Compute divergence of u*
            div_u = np.zeros(n_cells)
            for f in range(mesh.n_faces):
                a, b = iA[f], iB[f]
                n = face_normals[:, f]
                u_face = 0.5 * (Q_star[:dim, a] + Q_star[:dim, b])
                flux_div = sum(u_face[d] * n[d] for d in range(dim))
                div_u[a] += flux_div * face_volumes[f] / cell_vol[a]
                div_u[b] -= flux_div * face_volumes[f] / cell_vol[b]

            # Step 4: Solve Poisson: ∇²p = (1/Δt)∇·u*
            rhs = div_u / dt
            p, gmres_info = solve_poisson(rhs)

            # Step 5: Correct velocity: u = u* - Δt·∇p
            grad_p = np.zeros((dim, n_cells))
            for f in range(mesh.n_faces):
                a, b = iA[f], iB[f]
                n = face_normals[:, f]
                p_face = 0.5 * (p[a] + p[b])
                for d in range(dim):
                    contrib = p_face * n[d] * face_volumes[f]
                    grad_p[d, a] += contrib / cell_vol[a]
                    grad_p[d, b] -= contrib / cell_vol[b]

            Qnew[:dim, :] = Q_star[:dim, :] - dt * grad_p
            Qnew = boundary_operator(time_now + dt, Qnew, Qaux, parameters)

            time_now += dt
            iteration += 1

            if iteration % self.log_every == 0 if hasattr(self, 'log_every') else iteration % 10 == 0:
                div_max = np.max(np.abs(div_u[:nc]))
                logger.info(
                    f"split it={iteration}, t={time_now:.4f}, dt={dt:.2e}, "
                    f"div_max={div_max:.2e}, gmres={gmres_info}"
                )

        logger.info(f"Splitting solver finished in {iteration} iterations, t={time_now:.4f}")
        return Qnew, p


class FSFSplittingSolver(SplittingSolver):
    """Splitting solver for free-surface flows (SWE, SME, VAM).

    Combines:
    - Positive Rusanov for explicit flux (requires h/b)
    - Implicit source stepping
    - Pressure Poisson correction

    Requires model variables 'b' and 'h'.
    """

    def _build_numerics(self, symbolic_model):
        return _build_free_surface_numerics(symbolic_model)
