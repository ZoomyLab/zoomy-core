"""FVM solver for hyperbolic PDE systems (numpy backend).

Uses the symbolic Riemann solver (riemann_solvers.py) for flux computation.
No dependency on legacy flux.py or nonconservative_flux.py.

Solver hierarchy:
    Solver (base: init, create_runtime, BCs)
      └→ HyperbolicSolver (explicit time stepping + symbolic Riemann flux)
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


# ── Field detection helpers ───────────────────────────────────────────────────

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


# ── Base Solver ───────────────────────────────────────────────────────────────

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
        runtime_model = NumpyRuntimeModel(model)
        return Q, Qaux, parameters, mesh, runtime_model

    def get_apply_boundary_conditions(self, mesh, model):
        def apply_boundary_conditions(time, Q, Qaux, parameters):
            for i in range(mesh.n_boundary_faces):
                i_face = mesh.boundary_face_face_indices[i]
                i_bc_func = mesh.boundary_face_function_numbers[i]
                q_cell = Q[:, mesh.boundary_face_cells[i]]
                qaux_cell = Qaux[:, mesh.boundary_face_cells[i]]
                normal = mesh.face_normals[:, i_face]
                position = mesh.face_centers[i_face, :]
                position_ghost = mesh.cell_centers[:, mesh.boundary_face_ghosts[i]]
                distance = np.linalg.norm(position - position_ghost)
                q_ghost = model.boundary_conditions(
                    i_bc_func, time, position, distance, q_cell, qaux_cell, parameters, normal
                )
                Q[:, mesh.boundary_face_ghosts[i]] = q_ghost
            return Q
        return apply_boundary_conditions

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


# ── HyperbolicSolver ──────────────────────────────────────────────────────────

class HyperbolicSolver(Solver):
    """Explicit time-stepping solver using the symbolic Riemann solver.

    Uses PositiveNonconservativeRusanov from riemann_solvers.py.
    No dependency on legacy flux.py or nonconservative_flux.py.
    """

    time_end = param.Number(default=0.1, bounds=(0, None), doc="Simulation end time")
    min_dt = param.Number(default=1e-6, bounds=(0, None), doc="Minimum allowed timestep")
    compute_dt = param.Parameter(default=None, doc="Time-stepping strategy (callable)")
    eigenvalue_regularization = 1e-8

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

    # ── Symbolic model helpers ────────────────────────────────────────

    def _get_symbolic_model(self, model):
        return model.model if hasattr(model, "model") else model

    def _get_dry_threshold(self, symbolic_model):
        return _param_value(symbolic_model, "eps_wet", default=1e-3)

    # ── Max wavespeed (for CFL + Rusanov dissipation) ─────────────────

    def _build_max_wavespeed(self, symbolic_model):
        eig_mode = getattr(symbolic_model, "eigenvalue_mode", "symbolic")
        n_vars = symbolic_model.n_variables
        n_aux = symbolic_model.n_aux_variables
        n_params = symbolic_model.n_parameters
        dim = symbolic_model.dimension

        if eig_mode != "numerical":
            rt = NumpyRuntimeModel(symbolic_model)
            compiled_eig = rt.eigenvalues
            def max_ws(*args):
                Q = np.array(args[:n_vars])
                Qaux = np.array(args[n_vars:n_vars + n_aux])
                p = np.array(args[n_vars + n_aux:n_vars + n_aux + n_params])
                n = np.array(args[n_vars + n_aux + n_params:])
                evs = np.asarray(compiled_eig(Q, Qaux, p, n), dtype=float).ravel()
                return float(np.max(np.abs(evs)))
            return max_ws

        rt = NumpyRuntimeModel(symbolic_model)
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

    # ── Flux operator (symbolic Riemann solver) ───────────────────────

    def _build_numerics(self, symbolic_model):
        """Build the symbolic Riemann solver. Override for SWE-specific variants."""
        return NonconservativeRusanov(symbolic_model)

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

        def flux_operator(dt, Q, Qaux, parameters, dQ):
            dQ = np.zeros_like(dQ)
            for f in range(mesh.n_faces):
                qA = Q[:, iA[f]]
                qB = Q[:, iB[f]]
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
            return dQ
        return flux_operator

    # ── Source operator ───────────────────────────────────────────────

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

    # ── Solve loop ────────────────────────────────────────────────────

    def solve(self, mesh, model, write_output=True):
        mesh = ensure_lsq_mesh(mesh, model)
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)
        Qaux = self.update_qaux(Q, Qaux, Q, Qaux, mesh, model, parameters, 0.0, 1.0)

        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            def save_fields(time, time_stamp, i_snapshot, Q, Qaux):
                return i_snapshot

        def run(Q, Qaux, parameters, model):
            t_start = gettime()
            compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model)
            boundary_operator = self.get_apply_boundary_conditions(mesh, model)
            Q = boundary_operator(0.0, Q, Qaux, parameters)
            Q = self.update_q(Q, Qaux, mesh, model, parameters)

            time_now = 0.0
            next_write_at = 0.0
            i_snapshot = 0.0
            dt_snapshot = self.time_end / max(self.settings.output.snapshots - 1, 1)
            iteration = 0

            cell_inradius_face = np.minimum(
                mesh.cell_inradius[mesh.face_cells[0, :]],
                mesh.cell_inradius[mesh.face_cells[1, :]],
            ).min()

            flux_operator = self.get_flux_operator(mesh, model)

            while time_now < self.time_end:
                dt = self.compute_dt(Q, Qaux, parameters, cell_inradius_face,
                                     compute_max_abs_eigenvalue)
                dt = min(float(dt), float(self.time_end - time_now))
                if not np.isfinite(dt) or dt <= 0.0:
                    break

                Q1 = ode.RK1(flux_operator, Q, Qaux, parameters, dt)
                Qnew = Q1
                Qnew = boundary_operator(time_now, Qnew, Qaux, parameters)
                Qnew = self.update_q(Qnew, Qaux, mesh, model, parameters)
                Qauxnew = self.update_qaux(Qnew, Qaux, Q, Qaux, mesh, model,
                                           parameters, time_now, dt)
                Q = Qnew
                Qaux = Qauxnew

                time_now += dt
                iteration += 1

                if time_now >= next_write_at:
                    i_snapshot = save_fields(time_now, next_write_at, i_snapshot, Q, Qaux)
                    next_write_at += dt_snapshot

                if iteration % 10 == 0:
                    logger.info(
                        f"iteration: {iteration}, time: {time_now:.6f}, "
                        f"dt: {dt:.6f}, next write at time: {next_write_at:.6f}"
                    )

            logger.info(f"Finished simulation with in {gettime() - t_start:.3f} seconds")
            return Q, Qaux

        return run(Q, Qaux, parameters, model)


# ── Shallow water variant ─────────────────────────────────────────────────────

class FreeSurfaceFlowSolver(HyperbolicSolver):
    """HyperbolicSolver with positive (hydrostatic reconstruction) Rusanov.

    For free-surface flow models (SWE, SME, VAM) that have 'b' (bathymetry)
    and 'h' (water depth) variables.  Adds hydrostatic reconstruction at
    faces and wet/dry handling.
    """

    def _build_numerics(self, symbolic_model):
        keys = list(symbolic_model.variables.keys())
        if "h" not in keys or "b" not in keys:
            raise ValueError(
                f"ShallowWaterSolver requires 'h' and 'b' in model variables, "
                f"got {keys}. Use HyperbolicSolver for general models."
            )
        field_map = _detect_field_map(symbolic_model)
        scaled_q_indices = _detect_scaled_q_indices(symbolic_model)
        return PositiveNonconservativeRusanov(
            symbolic_model,
            field_map=field_map,
            scaled_q_indices=scaled_q_indices,
        )
