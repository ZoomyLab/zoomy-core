"""Pressure-splitting solver: IMEX + Poisson constraint (numpy backend).

Extends ``IMEXSolver`` with a pressure Poisson correction step (Chorin
projection). Uses face-based divergence and gradient, and GMRES for the
Poisson solve.

Solver hierarchy:
    SplittingSolver(IMEXSolver)
      -> step(dt):
            [explicit flux] -> viscous diffusion -> pressure_correction -> update_state
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
    viscosity = param.Number(default=0.01, doc="Kinematic viscosity v")

    def setup_simulation(self, mesh, model, write_output=False):
        """Build all operators including Poisson infrastructure."""
        t0 = time.time()
        mesh = ensure_lsq_mesh(mesh, model, lsq_degree=2)
        dim = model.dimension
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model_rt = self.create_runtime(Q, Qaux, mesh, model)

        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells
        n_vars = Q.shape[0]

        # Store simulation state
        self._sim_mesh = mesh
        self._sim_model = model_rt
        self._sim_parameters = parameters
        self._sim_Q = Q
        self._sim_Qaux = Qaux
        self._sim_time = 0.0
        self._sim_dim = dim
        self._sim_nc = nc
        self._sim_n_cells = n_cells
        self._sim_n_vars = n_vars
        self._sim_nu = self.viscosity

        # Pressure field
        self._sim_pressure = np.zeros(n_cells, dtype=float)

        # Precompute face connectivity
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        self._sim_iA = iA
        self._sim_iB = iB
        self._sim_face_normals = mesh.face_normals[:dim, :]
        self._sim_face_volumes = mesh.face_volumes
        self._sim_cell_vol = mesh.cell_volumes

        # Build sparse Laplacian matrix for Poisson
        self._sim_poisson_solver = self._build_poisson_solver(mesh, dim, nc, n_cells, iA, iB)

        # Build operators from parent hierarchy
        self._sim_compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model_rt)
        self._sim_boundary_operator = self.get_apply_boundary_conditions(mesh, model_rt)
        self._sim_flux_operator = self.get_flux_operator(mesh, model_rt)
        self._sim_ode_step = ode.RK1

        # Precompute mesh constant
        self._sim_cell_inradius_face = np.minimum(
            mesh.cell_inradius[mesh.face_cells[0, :]],
            mesh.cell_inradius[mesh.face_cells[1, :]],
        ).min()

        # Apply initial BCs
        self._sim_Q = self._sim_boundary_operator(0.0, Q, Qaux, parameters)

        # Output setup
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            self._sim_save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            self._sim_save_fields = lambda time, time_stamp, i_snapshot, Q, Qaux: i_snapshot

    def _build_poisson_solver(self, mesh, dim, nc, n_cells, iA, iB):
        """Build sparse Laplacian and return a Poisson solve function."""
        face_volumes = mesh.face_volumes
        cell_vol = mesh.cell_volumes

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
                pass  # b is ghost: Neumann p[b]=p[a]
            elif b < nc:
                pass

        # Pin one pressure DOF to remove null space
        L[0, :] = 0
        L[0, 0] = 1.0
        L_csr = csr_matrix(L)

        tol = self.pressure_gmres_tol
        maxiter = self.pressure_gmres_maxiter

        def solve_poisson(rhs):
            rhs_inner = rhs[:nc].copy()
            rhs_inner[0] = 0.0  # pinned DOF
            result = np.zeros(n_cells)

            def matvec(x):
                return L_csr @ x

            A_op = LinearOperator((nc, nc), matvec=matvec, dtype=float)
            sol, info = gmres(A_op, rhs_inner, atol=0.0,
                              rtol=tol, maxiter=maxiter)
            result[:nc] = sol
            # Ghost cells: Neumann
            for i_bf in range(mesh.n_boundary_faces):
                ghost = mesh.boundary_face_ghosts[i_bf]
                inner = mesh.boundary_face_cells[i_bf]
                result[ghost] = result[inner]
            return result, info

        return solve_poisson

    def step(self, dt):
        """One splitting timestep: flux -> viscous diffusion -> pressure correction.

        Each line is one physics operation. No if-clauses.
        """
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        parameters = self._sim_parameters
        time_now = self._sim_time
        dim = self._sim_dim
        n_vars = self._sim_n_vars
        n_cells = self._sim_n_cells

        # Step 1: Explicit convective flux
        Q_star = self._sim_ode_step(self._sim_flux_operator, Q, Qaux, parameters, dt)
        Q_star = self._sim_boundary_operator(time_now, Q_star, Qaux, parameters)

        # Step 2: Explicit viscous diffusion
        Q_star = self._apply_viscous_diffusion(Q_star, dt)
        Q_star = self._sim_boundary_operator(time_now, Q_star, Qaux, parameters)

        # Step 3: Pressure correction (Poisson solve + velocity update)
        Qnew, p = self._pressure_correction(Q_star, dt)
        Qnew = self._sim_boundary_operator(time_now + dt, Qnew, Qaux, parameters)

        # Commit new state
        self._sim_Q = Qnew
        self._sim_pressure = p

    def _apply_viscous_diffusion(self, Q_star, dt):
        """Explicit viscous diffusion for velocity components."""
        dim = self._sim_dim
        n_vars = self._sim_n_vars
        n_cells = self._sim_n_cells
        nu = self._sim_nu
        iA = self._sim_iA
        iB = self._sim_iB
        mesh = self._sim_mesh
        face_volumes = self._sim_face_volumes
        cell_vol = self._sim_cell_vol

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
        return Q_star

    def _pressure_correction(self, Q_star, dt):
        """Compute divergence, solve Poisson, correct velocity."""
        dim = self._sim_dim
        n_cells = self._sim_n_cells
        iA = self._sim_iA
        iB = self._sim_iB
        mesh = self._sim_mesh
        face_normals = self._sim_face_normals
        face_volumes = self._sim_face_volumes
        cell_vol = self._sim_cell_vol

        # Compute divergence of u*
        div_u = np.zeros(n_cells)
        for f in range(mesh.n_faces):
            a, b = iA[f], iB[f]
            n = face_normals[:, f]
            u_face = 0.5 * (Q_star[:dim, a] + Q_star[:dim, b])
            flux_div = sum(u_face[d] * n[d] for d in range(dim))
            div_u[a] += flux_div * face_volumes[f] / cell_vol[a]
            div_u[b] -= flux_div * face_volumes[f] / cell_vol[b]

        # Solve Poisson
        rhs = div_u / dt
        p, gmres_info = self._sim_poisson_solver(rhs)

        # Correct velocity: u = u* - dt * grad(p)
        grad_p = np.zeros((dim, n_cells))
        for f in range(mesh.n_faces):
            a, b = iA[f], iB[f]
            n = face_normals[:, f]
            p_face = 0.5 * (p[a] + p[b])
            for d in range(dim):
                contrib = p_face * n[d] * face_volumes[f]
                grad_p[d, a] += contrib / cell_vol[a]
                grad_p[d, b] -= contrib / cell_vol[b]

        Qnew = Q_star.copy()
        Qnew[:dim, :] = Q_star[:dim, :] - dt * grad_p
        return Qnew, p

    def run_simulation(self):
        """Time loop for splitting solver."""
        time_now = 0.0
        iteration = 0
        nc = self._sim_nc

        while time_now < self.time_end:
            dt = self.compute_dt(
                self._sim_Q, self._sim_Qaux, self._sim_parameters,
                self._sim_cell_inradius_face, self._sim_compute_max_abs_eigenvalue,
            )
            dt = min(float(dt), float(self.time_end - time_now))
            if not np.isfinite(dt) or dt <= 0:
                break

            self._sim_time = time_now
            self.step(dt)

            time_now += dt
            iteration += 1

            if iteration % 10 == 0:
                div_max = 0.0  # logged for diagnostics
                logger.info(
                    f"split it={iteration}, t={time_now:.4f}, dt={dt:.2e}"
                )

        self._sim_time = time_now
        logger.info(f"Splitting solver finished in {iteration} iterations, t={time_now:.4f}")
        return self._sim_Q, self._sim_pressure

    def solve(self, mesh, model, write_output=False):
        """Convenience: setup_simulation + run_simulation."""
        self.setup_simulation(mesh, model, write_output=write_output)
        return self.run_simulation()


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
