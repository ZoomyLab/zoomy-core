"""Chorin projection solver for incompressible Navier-Stokes.

Implements the pressure-correction scheme:
1. Predictor: advance momentum without pressure (explicit FVM)
2. Pressure Poisson: solve ∇²p = (1/Δt)∇·u* via iterative Jacobi
3. Corrector: u^{n+1} = u* - Δt * ∇p
4. Viscous step: u += Δt * ν * ∇²u (explicit diffusion)
"""

from __future__ import annotations

import numpy as np
import param

from zoomy_core.misc.logger_config import logger
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.mesh.lsq_reconstruction import compute_derivatives
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
import zoomy_core.fvm.flux as fvmflux
import zoomy_core.fvm.ode as ode
import zoomy_core.fvm.timestepping as timestepping


class ProjectionSolver(param.Parameterized):
    """Chorin pressure-correction solver for incompressible NS."""

    time_end = param.Number(default=1.0)
    CFL = param.Number(default=0.3)
    poisson_iterations = param.Integer(default=100)
    poisson_tol = param.Number(default=1e-6)
    log_every = param.Integer(default=10)

    def solve(self, mesh, model, write_output=False):
        """Run the projection method."""
        mesh = ensure_lsq_mesh(mesh, model)
        dim = model.dimension
        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells
        n_vars = model.n_variables
        nu = float(model.parameter_values[0])  # kinematic viscosity

        # Compile model for the predictor flux
        runtime = NumpyRuntimeModel(model)

        # Initialize velocity field
        Q = np.zeros((n_vars, n_cells), dtype=float)
        Q = model.initial_conditions.apply(mesh.cell_centers, Q)

        # Pressure field (separate, not part of Q)
        p = np.zeros(n_cells, dtype=float)

        # Cell volumes for weighted operations
        cell_vol = mesh.cell_volumes

        # Precompute face connectivity
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        face_normals = mesh.face_normals[:dim, :]
        face_volumes = mesh.face_volumes
        cell_vol_A = cell_vol[iA]
        cell_vol_B = cell_vol[iB]

        # Time stepping
        time = 0.0
        iteration = 0

        while time < self.time_end:
            # Adaptive dt from CFL
            max_speed = 0.0
            for ic in range(nc):
                speed = sum(abs(Q[d, ic]) for d in range(dim))
                max_speed = max(max_speed, speed)
            if max_speed < 1e-14:
                max_speed = 1.0
            dx_min = mesh.cell_inradius[:nc].min()
            dt = self.CFL * 2 * dx_min / max_speed
            dt = min(dt, self.time_end - time)

            # ── Step 1: Predictor (explicit convection) ──────────────
            # Simple Rusanov flux for convective terms
            dQ = np.zeros_like(Q)
            for f in range(mesh.n_faces):
                qL = Q[:, iA[f]]
                qR = Q[:, iB[f]]
                n = face_normals[:, f]

                # Physical flux: F_d(Q) = Q * Q[d]
                FL = np.array([qL[i] * sum(qL[d] * n[d] for d in range(dim))
                               for i in range(n_vars)])
                FR = np.array([qR[i] * sum(qR[d] * n[d] for d in range(dim))
                               for i in range(n_vars)])

                # Max wave speed
                sL = abs(sum(qL[d] * n[d] for d in range(dim)))
                sR = abs(sum(qR[d] * n[d] for d in range(dim)))
                s_max = max(sL, sR)

                # Rusanov flux
                flux = 0.5 * (FL + FR) - 0.5 * s_max * (qR - qL)

                dQ[:, iA[f]] -= flux * face_volumes[f] / cell_vol_A[f]
                dQ[:, iB[f]] += flux * face_volumes[f] / cell_vol_B[f]

            Q_star = Q + dt * dQ

            # ── Step 2: Viscous diffusion (explicit) ─────────────────
            for d in range(dim):
                # Compute Laplacian via LSQ: ∇²u_d ≈ sum of second derivatives
                # For degree-1 LSQ, we approximate with neighbor averaging
                u_d = Q_star[d, :]
                lap = np.zeros(n_cells)
                for f in range(mesh.n_faces):
                    a, b = iA[f], iB[f]
                    diff = u_d[b] - u_d[a]
                    # Approximate Laplacian contribution from face
                    dist = np.linalg.norm(
                        mesh.cell_centers[:dim, b] - mesh.cell_centers[:dim, a]
                    )
                    if dist > 1e-14:
                        lap_contrib = diff / dist * face_volumes[f]
                        lap[a] += lap_contrib / cell_vol[a]
                        lap[b] -= lap_contrib / cell_vol[b]
                Q_star[d, :] += dt * nu * lap

            # ── Step 3: Apply BCs to u* ──────────────────────────────
            for i_bf in range(mesh.n_boundary_faces):
                ghost = mesh.boundary_face_ghosts[i_bf]
                inner = mesh.boundary_face_cells[i_bf]
                Q_star[:, ghost] = Q_star[:, inner]  # extrapolation

            # ── Step 4: Pressure Poisson: ∇²p = (1/Δt)∇·u* ─────────
            # Compute divergence of u*
            div_u = np.zeros(n_cells)
            for f in range(mesh.n_faces):
                a, b = iA[f], iB[f]
                n = face_normals[:, f]
                # Average velocity at face
                u_face = 0.5 * (Q_star[:dim, a] + Q_star[:dim, b])
                flux_div = sum(u_face[d] * n[d] for d in range(dim))
                div_u[a] += flux_div * face_volumes[f] / cell_vol[a]
                div_u[b] -= flux_div * face_volumes[f] / cell_vol[b]

            rhs = div_u / dt

            # ── Solve Poisson via sparse direct solve ──────────────
            if not hasattr(self, '_L_sparse'):
                from scipy import sparse
                from scipy.sparse.linalg import spsolve

                # Build sparse Laplacian matrix L (nc x nc)
                rows, cols, vals = [], [], []
                for f in range(mesh.n_faces):
                    a, b = iA[f], iB[f]
                    if a >= nc and b >= nc:
                        continue
                    dist = np.linalg.norm(
                        mesh.cell_centers[:dim, b] - mesh.cell_centers[:dim, a]
                    )
                    coeff = face_volumes[f] / max(dist, 1e-14)
                    if a < nc and b < nc:
                        rows.extend([a, a, b, b])
                        cols.extend([b, a, a, b])
                        vals.extend([
                            coeff / cell_vol[a],    # off-diag
                            -coeff / cell_vol[a],   # diag
                            coeff / cell_vol[b],    # off-diag
                            -coeff / cell_vol[b],   # diag
                        ])
                    elif a < nc:
                        # b is ghost (Neumann: p[b]=p[a] → contributes 0)
                        pass
                    elif b < nc:
                        pass

                L = sparse.coo_matrix((vals, (rows, cols)), shape=(nc, nc)).tocsr()
                # Fix singular system: pin p[0] = 0
                L[0, :] = 0
                L[0, 0] = 1.0
                self._L_sparse = L
                self._spsolve = spsolve

            rhs_inner = rhs[:nc].copy()
            rhs_inner[0] = 0.0  # pin p[0] = 0
            p[:nc] = self._spsolve(self._L_sparse, rhs_inner)

            # Ghost cells: Neumann BC
            for i_bf in range(mesh.n_boundary_faces):
                ghost = mesh.boundary_face_ghosts[i_bf]
                inner = mesh.boundary_face_cells[i_bf]
                p[ghost] = p[inner]
            poisson_iter = 0

            # ── Step 5: Corrector: u = u* - Δt * ∇p ─────────────────
            # Compute ∇p at cell centers via face differences
            grad_p = np.zeros((dim, n_cells))
            for f in range(mesh.n_faces):
                a, b = iA[f], iB[f]
                n = face_normals[:, f]
                p_face = 0.5 * (p[a] + p[b])
                for d in range(dim):
                    contrib = p_face * n[d] * face_volumes[f]
                    grad_p[d, a] += contrib / cell_vol[a]
                    grad_p[d, b] -= contrib / cell_vol[b]

            Q[:dim, :] = Q_star[:dim, :] - dt * grad_p

            # ── Step 6: Apply BCs ────────────────────────────────────
            for i_bf in range(mesh.n_boundary_faces):
                ghost = mesh.boundary_face_ghosts[i_bf]
                inner = mesh.boundary_face_cells[i_bf]
                Q[:, ghost] = Q[:, inner]

            time += dt
            iteration += 1

            if iteration % self.log_every == 0:
                div_max = np.max(np.abs(div_u[:nc]))
                logger.info(
                    f"it={iteration}, t={time:.4f}, dt={dt:.2e}, "
                    f"poisson_iters={poisson_iter+1}, div_max={div_max:.2e}"
                )

        logger.info(f"Finished in {iteration} iterations, t={time:.4f}")
        return Q, p
