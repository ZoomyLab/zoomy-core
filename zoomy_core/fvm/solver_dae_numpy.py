"""DAE solver (numpy backend) — index-1 DAE-PDE integrator via
Ascher-Ruuth-Spiteri IMEX-Runge-Kutta time stepping.

Sits next to :class:`HyperbolicSolver` and :class:`IMEXSolver` in
``zoomy_core.fvm`` and reuses every shared piece of the framework:

* :class:`SystemModel` is the single symbolic source-of-truth;
  ``SystemModel.from_model`` auto-scans all non-state Function and
  Derivative atoms into ``aux_state`` + a structured
  :attr:`SystemModel.aux_registry`.
* :class:`NumpyRuntimeModel` lambdifies the operator matrices into
  ``(Q, Qaux, p) → ndarray`` callables.
* :func:`ensure_lsq_mesh` promotes the input mesh to
  :class:`LSQMesh`, exposing ``compute_derivatives`` for the registry-
  driven ``update_qaux`` walker.
* Boundary-condition objects (``Extrapolation``, ``Lambda``,
  ``InflowOutflow``, …) supply face values via ``face_value``.
* ``Solver.update_q`` / ``Solver.update_qaux`` (registry-aware default)
  fire after every step.

Time integration: Ascher-Ruuth-Spiteri IMEX-Runge-Kutta (ARS232 / ARS343)
on the singular DAE ``M(Q)·∂_t Q = R(Q, Qaux, p)``.  Aux is updated
once per step (lagged through the Newton iteration), keeping the
Jacobian small and reusing the standard ``Solver.update_qaux`` hook.

Spatial scheme: Rusanov flux + non-conservative path-integral
fluctuations, mirroring :class:`NonconservativeRusanov` but driven by
the SystemModel runtime (so derivative-aux entries like ``∂_x h`` flow
through as ordinary Qaux components, no special-casing).  Boundary
faces use ``BC.face_value(Q_inner, Qaux_inner, normal, d_face, time,
parameters)``.
"""
from __future__ import annotations

import os
import time as _time

import numpy as np
import param

from zoomy_core.fvm.solver_numpy import Solver
from zoomy_core.fvm.imex_ark import ars232, ars343, imex_ark_step
from zoomy_core.fvm import timestepping
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.misc.logger_config import logger
import zoomy_core.misc.io as io


_EMPTY_AUX = np.empty(0, dtype=float)


class DAESolver(Solver):
    """Index-1 DAE-PDE solver with IMEX-RK time stepping."""

    time_end = param.Number(default=0.1, bounds=(0, None),
                            doc="Simulation end time")
    method = param.Selector(default="ars343",
                            objects=["ars232", "ars343"],
                            doc="IMEX-RK tableau name")
    compute_dt = param.Parameter(
        default=None,
        doc="Adaptive-timestep callable (see "
            "``zoomy_core.fvm.timestepping``); defaults to "
            "``adaptive(CFL=0.3)``")
    newton_tol = param.Number(default=1e-9, bounds=(0, None),
                              doc="Per-stage Newton residual tolerance")
    newton_maxit = param.Integer(default=30, bounds=(1, None),
                                 doc="Maximum Newton iterations per stage")
    h_index = param.Integer(default=0, bounds=(0, None),
                            doc="State index of the water-depth-like "
                                "variable used for max-eigenvalue / "
                                "CFL estimation (default 0 = ``h``)")
    nc_integration_order = param.Integer(default=3, bounds=(1, 8),
                            doc="Gauss-Legendre quadrature order for "
                                "the non-conservative path integral")
    jacobian_mode = param.Selector(default="sparse_fd",
                            objects=["sparse_fd", "dense_fd"],
                            doc="Newton Jacobian assembly strategy.  "
                                "``sparse_fd`` uses graph colouring of "
                                "the cell-dependency stencil for "
                                "``O(n_colors × n_state)`` f_I evals "
                                "per Jacobian; ``dense_fd`` is the "
                                "naive ``O((nc·n_state)²)`` path (only "
                                "viable for tiny grids).")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.compute_dt is None:
            self.compute_dt = timestepping.adaptive(CFL=0.3)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_simulation(self, mesh, model, *, write_output=False):
        """Build operators + state once, including BC plumbing,
        output infrastructure, and the registry-driven Qaux walker."""
        mesh = ensure_lsq_mesh(mesh, model)
        self._sim_mesh = mesh
        self._sim_model = model

        # SystemModel — auto-scan happens inside from_model.
        sm = SystemModel.from_model(model)
        self.sm = sm
        # Expose the SystemModel on the model too so
        # ``Solver._sm_from_solver_or_model`` and any user-provided
        # ``update_qaux`` override can pick it up generically.
        if not hasattr(model, "_chain_systemmodel"):
            model._chain_systemmodel = sm

        self.rt = NumpyRuntimeModel.from_system_model(sm)
        self.tab = ars232() if self.method == "ars232" else ars343()

        nc = mesh.n_inner_cells
        n_state = sm.n_state
        dim = mesh.dimension
        self.nc = nc
        self.n_state = n_state
        self.n_aux_total = len(sm.aux_state)
        self.dim = dim

        # DAE partition: rows with all-zero mass-matrix row are
        # algebraic; rest are evolution.
        Q_probe = np.zeros(n_state)
        Q_probe[0] = 1.0
        for k in range(1, n_state):
            Q_probe[k] = 0.1 * (k + 1)
        Qaux_probe = np.zeros(self.n_aux_total)
        p_arr = self._parameters_array()
        M_probe = np.asarray(
            self.rt.mass_matrix(Q_probe, Qaux_probe, p_arr), dtype=float,
        )
        row_norms = np.linalg.norm(M_probe, axis=1)
        self.dyn_mask_cell = row_norms > 1e-12
        self.evol_idx = np.where(self.dyn_mask_cell)[0]
        self.alg_idx = np.where(~self.dyn_mask_cell)[0]
        self.dyn_mask = np.tile(self.dyn_mask_cell, nc)

        # Cache scalars used per-cell inside the Rusanov flux loop.
        try:
            g_param = next(s for s in sm.parameters if str(s) == "g")
            self._g_cached = float(sm.parameters[g_param])
        except StopIteration:
            self._g_cached = 9.81
        self._u_idx_cached = None
        for i, s in enumerate(sm.state):
            name = str(s)
            if name.startswith("U_") or name == "u":
                self._u_idx_cached = i
                break

        # IC from model.
        Q = np.zeros((n_state, nc))
        if (hasattr(model, "initial_conditions")
                and model.initial_conditions is not None):
            try:
                Q = np.asarray(
                    model.initial_conditions.apply(
                        mesh.cell_centers[:, :nc], Q,
                    ),
                    dtype=float,
                )
            except Exception:
                pass

        # Permanent-aux IC (model.aux_variables-based).  Function-aux
        # and derivative-aux rows are filled by ``update_qaux``.
        Qaux = np.zeros((self.n_aux_total, nc))
        n_aux_permanent = (
            len(sm.aux_state) - len(getattr(sm, "aux_registry", []))
        )
        if (n_aux_permanent > 0
                and hasattr(model, "aux_initial_conditions")
                and model.aux_initial_conditions is not None):
            try:
                Qaux[:n_aux_permanent, :] = np.asarray(
                    model.aux_initial_conditions.apply(
                        mesh.cell_centers[:, :nc],
                        Qaux[:n_aux_permanent, :],
                    ),
                    dtype=float,
                )
            except Exception:
                pass
        self._n_aux_permanent = n_aux_permanent
        self._sim_Qaux = Qaux
        self._sim_parameters = p_arr

        # BC plumbing — uses Solver._setup_bc-style logic.
        self._setup_bc(mesh, model)

        # First Qaux fill: walk the registry once before the manifold
        # projection (so function-aux + derivative-aux entries have
        # sensible values at t = 0).
        self._sim_Qaux = self.update_qaux(
            Q, self._sim_Qaux, Q, self._sim_Qaux,
            mesh, model, p_arr, 0.0, 0.0,
        )

        # Jacobian colouring (sparse FD) — needed by
        # project_to_manifold below.
        if self.jacobian_mode == "sparse_fd":
            self._build_jacobian_coloring()

        # Project initial Q to the algebraic constraint manifold.
        Q = self.project_to_manifold(Q, time=0.0)

        # Mesh-CFL inradius (smallest min-inradius across face
        # neighbours).
        inner_face_mask = ((mesh.face_cells[0] < nc)
                           & (mesh.face_cells[1] < nc))
        if inner_face_mask.any():
            inradii = np.minimum(
                mesh.cell_inradius[mesh.face_cells[0, inner_face_mask]],
                mesh.cell_inradius[mesh.face_cells[1, inner_face_mask]],
            )
            self._sim_min_inradius = float(inradii.min())
        else:
            self._sim_min_inradius = float(mesh.cell_inradius[:nc].min())
        self._sim_max_eigenvalue = self._build_max_eigenvalue_callable()

        # Output setup (parallel to IMEXSolver.setup_simulation).
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory,
                f"{self.settings.output.filename}.h5",
            )
            io.init_output_directory(
                self.settings.output.directory,
                self.settings.output.clean_directory,
            )
            mesh.write_to_hdf5(output_hdf5_path)
            io.save_settings(self.settings)
            self._sim_save_fields = io.get_save_fields(
                output_hdf5_path, write_all=False,
            )
            self._output_hdf5_path = output_hdf5_path
        else:
            self._sim_save_fields = (
                lambda time, next_write_at, i_snap, Q, Qaux: i_snap
            )
            self._output_hdf5_path = None

        self._sim_Q = Q
        self._sim_time = 0.0
        return Q

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _setup_bc(self, mesh, model):
        """Resolve per-boundary-face BC objects from
        ``model.boundary_conditions``.  Defaults to ``Extrapolation``
        if none configured."""
        from zoomy_core.model.boundary_conditions import Extrapolation
        n_bf = mesh.n_boundary_faces
        self._n_bf = n_bf
        self._bf_cells = mesh.boundary_face_cells
        self._bf_fidx = mesh.boundary_face_face_indices
        bcs_obj = getattr(model, "boundary_conditions", None)
        bcs_list = (getattr(bcs_obj, "boundary_conditions_list", None)
                    if bcs_obj is not None else None)
        if bcs_list:
            self._bc_objects = [
                bcs_list[mesh.boundary_face_function_numbers[i]]
                for i in range(n_bf)
            ]
        else:
            self._bc_objects = [Extrapolation() for _ in range(n_bf)]
        if n_bf > 0:
            dim = mesh.dimension
            self._d_face = np.array([
                np.linalg.norm(
                    mesh.face_centers[self._bf_fidx[i], :dim]
                    - mesh.cell_centers[:dim, self._bf_cells[i]]
                ) for i in range(n_bf)
            ])
        else:
            self._d_face = np.array([])
        self._bf_face_to_idx = {
            int(self._bf_fidx[i]): i for i in range(n_bf)
        }

    # ------------------------------------------------------------------
    # Parameters helper
    # ------------------------------------------------------------------

    def _parameters_array(self):
        return np.array(
            [float(self.sm.parameters[s]) for s in self.sm.parameters],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Residual assembly (Rusanov flux + NC fluctuations + source)
    # ------------------------------------------------------------------

    def _residual(self, Q, Qaux, time):
        """Per-cell residual ``R = S − ∇·F − ∇·P − B·∇Q`` via Rusanov
        flux + linear-path-integral NC fluctuations.  Shape
        ``(n_state, nc)``.

        Aux is passed in (does NOT change inside f_I): this is the
        lagged-aux Newton convention so the Jacobian doesn't have to
        track LSQ-derivative sensitivities.  Qaux is refreshed once
        per timestep by ``update_qaux``.
        """
        p = self.sm_parameters_arr if hasattr(self, "sm_parameters_arr") \
            else self._parameters_array()
        nc = self.nc
        n_state = self.n_state
        dim = self.dim
        mesh = self._sim_mesh

        # Cell-centre source.
        Sv = np.empty((n_state, nc))
        for c in range(nc):
            Sv[:, c] = np.asarray(
                self.rt.source(Q[:, c], Qaux[:, c], p), dtype=float,
            ).ravel()

        # Boundary-face Q and Qaux.
        n_bf = self._n_bf
        bf_Q = np.zeros((n_state, n_bf))
        bf_Qaux = np.zeros((self.n_aux_total, n_bf))
        for i_bf in range(n_bf):
            c_in = int(self._bf_cells[i_bf])
            f_idx = int(self._bf_fidx[i_bf])
            normal = mesh.face_normals[:dim, f_idx]
            bf_Q[:, i_bf] = self._bc_objects[i_bf].face_value(
                Q[:, c_in], Qaux[:, c_in], normal,
                self._d_face[i_bf], time, p,
            )
            bf_Qaux[:, i_bf] = Qaux[:, c_in]

        # Face loop: Rusanov flux + NC fluctuations.
        cv = mesh.cell_volumes
        R = np.copy(Sv)
        xi_nodes, wi_nodes = np.polynomial.legendre.leggauss(
            self.nc_integration_order,
        )
        xi_nodes = 0.5 * (xi_nodes + 1)
        wi_nodes = 0.5 * wi_nodes

        for f in range(mesh.n_faces):
            cA = int(mesh.face_cells[0, f])
            cB = int(mesh.face_cells[1, f])
            normal = mesh.face_normals[:dim, f]
            area = float(mesh.face_volumes[f])

            if f in self._bf_face_to_idx:
                i_bf = self._bf_face_to_idx[f]
                c_in = int(self._bf_cells[i_bf])
                Q_L = Q[:, c_in]
                Qaux_L = Qaux[:, c_in]
                Q_R = bf_Q[:, i_bf]
                Qaux_R = bf_Qaux[:, i_bf]
                inner_is_A = (c_in == cA)
            else:
                Q_L = Q[:, cA]
                Qaux_L = Qaux[:, cA]
                Q_R = Q[:, cB]
                Qaux_R = Qaux[:, cB]
                inner_is_A = True

            # Rusanov conservative flux F + P.
            FL = (np.asarray(self.rt.flux(Q_L, Qaux_L, p), dtype=float)
                  .reshape(n_state, dim)
                  + np.asarray(self.rt.hydrostatic_pressure(
                      Q_L, Qaux_L, p), dtype=float).reshape(n_state, dim))
            FR = (np.asarray(self.rt.flux(Q_R, Qaux_R, p), dtype=float)
                  .reshape(n_state, dim)
                  + np.asarray(self.rt.hydrostatic_pressure(
                      Q_R, Qaux_R, p), dtype=float).reshape(n_state, dim))
            s_L = self._cell_max_eigenvalue(Q_L)
            s_R = self._cell_max_eigenvalue(Q_R)
            s_max = max(s_L, s_R)
            num_flux = (0.5 * (FL + FR) @ normal
                        - 0.5 * s_max * (Q_R - Q_L))

            # NC path-integral fluctuations.
            dQ = Q_R - Q_L
            dAux = Qaux_R - Qaux_L
            A_n = np.zeros((n_state, n_state))
            for xi, wi in zip(xi_nodes, wi_nodes):
                Q_path = Q_L + xi * dQ
                Qaux_path = Qaux_L + xi * dAux
                B_path = np.asarray(
                    self.rt.nonconservative_matrix(Q_path, Qaux_path, p),
                    dtype=float,
                )
                if B_path.ndim == 2 and dim == 1:
                    B_path = B_path[..., None]
                for d_ax in range(dim):
                    A_n += wi * B_path[:, :, d_ax] * normal[d_ax]
            adv = A_n @ dQ
            diss = s_max * dQ.copy()
            diss[0] = 0.0
            Dp = 0.5 * (adv + diss)
            Dm = 0.5 * (adv - diss)

            if f in self._bf_face_to_idx:
                c_in = cA if inner_is_A else cB
                sgn = +1.0 if inner_is_A else -1.0
                R[:, c_in] -= sgn * (num_flux + Dm) * area / cv[c_in]
            else:
                R[:, cA] -= (num_flux + Dm) * area / cv[cA]
                R[:, cB] += (num_flux - Dp) * area / cv[cB]

        return R

    def _cell_max_eigenvalue(self, Q_cell):
        """Scalar max wave-speed estimate at a single cell, used inside
        the face-flux loop for Rusanov dissipation."""
        h = max(float(Q_cell[self.h_index]), 1e-6)
        c = np.sqrt(self._g_cached * h)
        if self._u_idx_cached is not None:
            return abs(float(Q_cell[self._u_idx_cached])) + c
        return c

    def _build_max_eigenvalue_callable(self):
        """``(Q, Qaux, parameters) → max|λ|`` callable for adaptive dt:
        ``|U⃗₀| + √(g·h_max)`` Saint-Venant upper bound, with all
        leading-order horizontal velocity components included.
        """
        sm = self.sm
        h_idx = self.h_index
        velocity_indices = [
            i for i, s in enumerate(sm.state)
            if str(s) in {"U_0", "V_0", "u", "v"}
        ]
        g_val = self._g_cached

        def max_eig(Q, Qaux, parameters):
            h = Q[h_idx, :]
            c = np.sqrt(g_val * np.maximum(h, 1e-6))
            if velocity_indices:
                u_sq = np.zeros_like(h)
                for i in velocity_indices:
                    u_sq = u_sq + Q[i, :] ** 2
                return np.sqrt(u_sq) + c
            return c

        return max_eig

    # ------------------------------------------------------------------
    # IMEX RHS interface
    # ------------------------------------------------------------------

    def f_E(self, t, Y):
        return np.zeros_like(Y)

    def _Y_to_Q(self, Y):
        return Y.reshape(self.nc, self.n_state).T

    def _Q_to_Y(self, Q):
        return Q.T.ravel()

    def f_I(self, t, Y):
        """Implicit RHS: per-cell ``M_evol⁻¹·R_evol`` on evolution
        rows; constraint residual on algebraic rows.

        Aux is held fixed at ``self._sim_Qaux`` (refreshed once per
        timestep) — the lagged-aux Newton convention.
        """
        Q = self._Y_to_Q(Y)
        Qaux = self._sim_Qaux
        p = self._parameters_array()
        R = self._residual(Q, Qaux, t)
        out = np.zeros_like(R)
        out[self.alg_idx, :] = R[self.alg_idx, :]
        for c in range(self.nc):
            M_c = np.asarray(
                self.rt.mass_matrix(Q[:, c], Qaux[:, c], p),
                dtype=float,
            )
            M_evol = M_c[np.ix_(self.evol_idx, self.evol_idx)]
            out[self.evol_idx, c] = np.linalg.solve(
                M_evol, R[self.evol_idx, c],
            )
        return self._Q_to_Y(out)

    # ------------------------------------------------------------------
    # Sparse FD Jacobian via graph colouring
    # ------------------------------------------------------------------

    def J_I(self, t, Y):
        if self.jacobian_mode == "dense_fd":
            return self._dense_J_I(t, Y)
        return self._sparse_J_I(t, Y)

    def _dense_J_I(self, t, Y):
        n = len(Y)
        f0 = self.f_I(t, Y)
        J = np.zeros((n, n))
        eps = 1e-7
        for k in range(n):
            step = eps * max(1.0, abs(Y[k]))
            Yp = Y.copy(); Yp[k] += step
            J[:, k] = (self.f_I(t, Yp) - f0) / step
        return J

    def _sparse_J_I(self, t, Y):
        n = len(Y)
        n_state = self.n_state
        f0 = self.f_I(t, Y)
        J = np.zeros((n, n))
        eps = 1e-7
        for color, cells in enumerate(self._color_cells):
            for i_state in range(n_state):
                Yp = Y.copy()
                steps = {}
                for c in cells:
                    k = c * n_state + i_state
                    step = eps * max(1.0, abs(Y[k]))
                    Yp[k] += step
                    steps[c] = step
                df = self.f_I(t, Yp) - f0
                for c in cells:
                    k = c * n_state + i_state
                    step = steps[c]
                    for c_t in self._seen_by[c]:
                        base_r = c_t * n_state
                        J[base_r:base_r + n_state, k] = (
                            df[base_r:base_r + n_state] / step
                        )
        return J

    def _build_jacobian_coloring(self):
        """Greedy graph-colouring of the cell-dependency stencil
        (LSQ neighbours + face neighbours).  Cells of the same colour
        share no dependency cone — they can be perturbed
        simultaneously in FD Jacobian assembly."""
        mesh = self._sim_mesh
        nc = self.nc
        deps = [{c} for c in range(nc)]
        if (hasattr(mesh, "_lsq_neighbors")
                and mesh._lsq_neighbors is not None):
            for c in range(nc):
                for nbr in mesh._lsq_neighbors[c]:
                    nbr_i = int(nbr)
                    if 0 <= nbr_i < nc:
                        deps[c].add(nbr_i)
        for f in range(mesh.n_faces):
            cA = int(mesh.face_cells[0, f])
            cB = int(mesh.face_cells[1, f])
            if 0 <= cA < nc and 0 <= cB < nc:
                deps[cA].add(cB)
                deps[cB].add(cA)
        seen_by = [set() for _ in range(nc)]
        for c in range(nc):
            for d in deps[c]:
                seen_by[d].add(c)
        self._seen_by = [sorted(s) for s in seen_by]
        colors = [-1] * nc
        cells_with_color = []
        for c in range(nc):
            used = set()
            for c_prev in range(c):
                if colors[c_prev] < 0:
                    continue
                if seen_by[c] & seen_by[c_prev]:
                    used.add(colors[c_prev])
            k = 0
            while k in used:
                k += 1
            colors[c] = k
            while len(cells_with_color) <= k:
                cells_with_color.append([])
            cells_with_color[k].append(c)
        self._colors = colors
        self._color_cells = cells_with_color

    # ------------------------------------------------------------------
    # Manifold projection
    # ------------------------------------------------------------------

    def project_to_manifold(self, Q, *, time=0.0, tol=1e-10, maxit=20):
        """Newton-project ``Q`` onto the algebraic constraint manifold
        ``f_I[alg] = 0`` by adjusting algebraic state entries."""
        Y = self._Q_to_Y(Q)
        nc = self.nc
        n_state = self.n_state
        alg_idx = self.alg_idx
        if len(alg_idx) == 0:
            return Q
        alg_rows = np.concatenate([
            c * n_state + alg_idx for c in range(nc)
        ])
        for _ in range(maxit):
            f = self.f_I(time, Y)
            f_alg = f[alg_rows]
            if np.linalg.norm(f_alg) < tol:
                return self._Y_to_Q(Y)
            J = self.J_I(time, Y)
            sub = J[np.ix_(alg_rows, alg_rows)]
            try:
                delta = np.linalg.solve(sub, -f_alg)
            except np.linalg.LinAlgError:
                # Singular algebraic block — give up gracefully and
                # let the first IMEX Newton step do the projection.
                break
            Y[alg_rows] += delta
        return self._Y_to_Q(Y)

    # ------------------------------------------------------------------
    # Step / run loop
    # ------------------------------------------------------------------

    def step(self, dt):
        """One IMEX-ARK step on the stored state.

        Convention: refresh ``self._sim_Qaux`` via ``update_qaux``
        once per step (lagged through the Newton iteration), apply
        ``update_q`` to the new state after.
        """
        Q_old = self._sim_Q.copy()
        Qaux_old = self._sim_Qaux.copy()
        Y = self._Q_to_Y(Q_old)
        Y_new = imex_ark_step(
            self._sim_time, Y, dt, self.tab,
            self.f_E, self.f_I, self.J_I, self.dyn_mask,
            newton_tol=self.newton_tol, newton_maxit=self.newton_maxit,
        )
        Q_new = self._Y_to_Q(Y_new)
        # update_q + update_qaux post-step.  Some chain Models don't
        # implement the per-cell ``update_variables(Q_cell, aux, p)``
        # signature that SWE-style models expose, so we guard with
        # try/except — the default for those models is "no clamp".
        try:
            Q_new = self.update_q(Q_new, Qaux_old, self._sim_mesh,
                                  self._sim_model, self._sim_parameters)
        except TypeError:
            pass
        Qaux_new = self.update_qaux(
            Q_new, Qaux_old, Q_old, Qaux_old,
            self._sim_mesh, self._sim_model, self._sim_parameters,
            self._sim_time + dt, dt,
        )
        self._sim_Q = Q_new
        self._sim_Qaux = Qaux_new

    def run_simulation(self):
        t_start = _time.time()
        time_now = 0.0
        next_write_at = 0.0
        i_snapshot = 0.0
        n_snap = max(self.settings.output.snapshots - 1, 1)
        dt_snapshot = self.time_end / n_snap
        iteration = 0
        parameters = self._sim_parameters

        # Initial snapshot.
        i_snapshot = self._sim_save_fields(
            time_now, next_write_at, i_snapshot,
            self._sim_Q, self._sim_Qaux,
        )
        next_write_at += dt_snapshot

        while time_now < self.time_end:
            dt = self.compute_dt(
                self._sim_Q, self._sim_Qaux, parameters,
                self._sim_min_inradius, self._sim_max_eigenvalue,
            )
            dt = float(min(dt, self.time_end - time_now))
            if not np.isfinite(dt) or dt <= 0.0:
                break

            self._sim_time = time_now
            self.step(dt)
            time_now += dt
            iteration += 1

            if time_now >= next_write_at:
                i_snapshot = self._sim_save_fields(
                    time_now, next_write_at, i_snapshot,
                    self._sim_Q, self._sim_Qaux,
                )
                next_write_at += dt_snapshot

            if iteration % 10 == 0:
                logger.info(
                    f"[DAESolver] iter {iteration}  t={time_now:.6f}  "
                    f"dt={dt:.6f}  next-snap={next_write_at:.4f}"
                )

        self._sim_time = time_now
        elapsed = _time.time() - t_start
        logger.info(
            f"[DAESolver] finished in {elapsed:.2f}s "
            f"({iteration} steps)"
        )
        return self._sim_Q, self._sim_Qaux

    def solve(self, mesh, model, *, write_output=True):
        self.setup_simulation(mesh, model, write_output=write_output)
        return self.run_simulation()

    # ------------------------------------------------------------------
    # VTK / Paraview post-processing
    # ------------------------------------------------------------------

    def export_vtk(self, *, field_names=None, aux_field_names=None,
                   filename="dae_output", skip_aux=False):
        """Post-process HDF5 snapshots to a VTK time series for
        Paraview.  Requires ``write_output=True`` at setup time."""
        if self._output_hdf5_path is None:
            raise RuntimeError(
                "export_vtk requires write_output=True at setup time"
            )
        if field_names is None:
            field_names = [str(s) for s in self.sm.state]
        if aux_field_names is None:
            aux_field_names = [str(s) for s in self.sm.aux_state]
        io.generate_vtk(
            self._output_hdf5_path,
            field_names=field_names,
            aux_field_names=aux_field_names,
            skip_aux=skip_aux,
            filename=filename,
        )
