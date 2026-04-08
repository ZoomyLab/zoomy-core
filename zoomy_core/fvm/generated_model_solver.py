"""
General-purpose solver for NumericalModel or any Model with [b, h, ...] state.

Uses variable names (not hardcoded indices) for field detection.
Wet/dry threshold from model parameter 'eps_wet'.
"""

import numpy as np

from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.model.derivative_workflow import (
    DerivativeAwareSolver,
    DerivativeAwareSolverMixin,
)

_EMPTY_AUX = np.array([])


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


class _GeneratedModelFluxMixin:

    eigenvalue_regularization = 1e-8

    def _get_symbolic_model(self, model):
        return model.model if hasattr(model, "model") else model

    def _get_dry_threshold(self, symbolic_model):
        return _param_value(symbolic_model, "eps_wet", default=1e-3)

    def _field_indices(self, symbolic_model):
        return {
            "b": _var_index(symbolic_model, "b", fallback=0),
            "h": _var_index(symbolic_model, "h", fallback=1),
        }

    def _detect_field_map(self, symbolic_model):
        if hasattr(symbolic_model, "get_field_map"):
            return symbolic_model.get_field_map()
        fi = self._field_indices(symbolic_model)
        field_map = {
            "h": {"container": "q", "index": fi["h"]},
            "b": {"container": "q", "index": fi["b"]},
        }
        return field_map

    def _detect_scaled_q_indices(self, symbolic_model):
        if hasattr(symbolic_model, "numerics_scaled_q_indices"):
            return symbolic_model.numerics_scaled_q_indices
        fi = self._field_indices(symbolic_model)
        excluded = {fi["b"], fi["h"]}
        return [i for i in range(symbolic_model.n_variables) if i not in excluded]

    def _cell_aux(self, Qaux, c):
        return Qaux[:, c] if Qaux.shape[0] > 0 else _EMPTY_AUX

    def update_q(self, Q, Qaux, mesh, model, parameters):
        """Apply model.update_variables (h clamp, momentum ramp) at each cell."""
        n_vars = Q.shape[0]
        for c in range(Q.shape[1]):
            Q[:, c] = np.asarray(
                model.update_variables(Q[:, c], self._cell_aux(Qaux, c), parameters),
                dtype=float,
            ).ravel()[:n_vars]
        return Q

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        return DerivativeAwareSolverMixin.update_qaux(
            self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt
        )

    def get_compute_max_abs_eigenvalue(self, mesh, model):
        """
        CFL eigenvalue computation. Uses the same max_wavespeed function
        as the Rusanov flux — single path for both.
        """
        symbolic_model = self._get_symbolic_model(model)
        max_ws = self._build_max_wavespeed(symbolic_model)
        fi = self._field_indices(symbolic_model)
        i_h = fi["h"]
        dry_thr = self._get_dry_threshold(symbolic_model)
        dim = symbolic_model.dimension
        i_cellA = mesh.face_cells[0]
        i_cellB = mesh.face_cells[1]
        normals = mesh.face_normals[:dim, :]
        has_aux = symbolic_model.n_aux_variables > 0

        def compute_max_eigenvalue(Q, Qaux, parameters):
            max_ev = np.zeros(mesh.n_faces)
            for f in range(mesh.n_faces):
                if Q[i_h, i_cellA[f]] < dry_thr and Q[i_h, i_cellB[f]] < dry_thr:
                    continue
                n = normals[:, f]
                for i_cell in [i_cellA[f], i_cellB[f]]:
                    q = Q[:, i_cell]
                    qaux = Qaux[:, i_cell] if has_aux else _EMPTY_AUX
                    ev = max_ws(*q, *qaux, *parameters, *n)
                    max_ev[f] = max(max_ev[f], ev)
            return max_ev

        return compute_max_eigenvalue

    def _build_max_wavespeed(self, symbolic_model):
        """
        Build the max_wavespeed numpy function from the model's eigenvalues.

        Uses symbolic eigenvalues if available, otherwise numerical via
        np.linalg.eigvals on the quasilinear matrix.
        """
        from zoomy_core.transformation.to_numpy import NumpyRuntimeModel

        eig_mode = getattr(symbolic_model, "eigenvalue_mode", "symbolic")
        n_vars = symbolic_model.n_variables
        n_aux = symbolic_model.n_aux_variables
        n_params = symbolic_model.n_parameters
        dim = symbolic_model.dimension

        if eig_mode != "numerical":
            # Compile the symbolic eigenvalue expressions
            rt = NumpyRuntimeModel(symbolic_model)
            compiled_eig = rt.eigenvalues

            def max_ws_symbolic(*args):
                Q = np.array(args[:n_vars])
                Qaux = np.array(args[n_vars:n_vars + n_aux])
                p = np.array(args[n_vars + n_aux:n_vars + n_aux + n_params])
                n = np.array(args[n_vars + n_aux + n_params:])
                evs = np.asarray(compiled_eig(Q, Qaux, p, n), dtype=float).ravel()
                return float(np.max(np.abs(evs)))

            return max_ws_symbolic

        # Numerical: compile quasilinear matrix, use np.linalg.eigvals
        rt = NumpyRuntimeModel(symbolic_model)
        ql_fn = rt.quasilinear_matrix
        fi = self._field_indices(symbolic_model)
        dry_thr = self._get_dry_threshold(symbolic_model)
        eps_reg = self.eigenvalue_regularization
        reg_diag = eps_reg * np.eye(n_vars)
        reg_diag[fi["b"], fi["b"]] = 0.0

        def max_ws_numerical(*args):
            Q = np.array(args[:n_vars])
            Qaux = np.array(args[n_vars:n_vars + n_aux])
            p = np.array(args[n_vars + n_aux:n_vars + n_aux + n_params])
            n_vec = np.array(args[n_vars + n_aux + n_params:])
            if Q[fi["h"]] < dry_thr:
                return 0.0
            ql = np.asarray(ql_fn(Q, Qaux, p), dtype=float).reshape(n_vars, n_vars, dim)
            A_n = sum(ql[:, :, d] * float(n_vec[d]) for d in range(dim))
            A_n += reg_diag
            evs = np.real(np.linalg.eigvals(A_n))
            return float(np.max(np.abs(evs)))

        return max_ws_numerical

    def get_flux_operator(self, mesh, model):
        symbolic_model = self._get_symbolic_model(model)
        field_map = self._detect_field_map(symbolic_model)
        scaled_q_indices = self._detect_scaled_q_indices(symbolic_model)

        max_wavespeed_fn = self._build_max_wavespeed(symbolic_model)

        numerics = PositiveNonconservativeRusanov(
            symbolic_model,
            field_map=field_map,
            scaled_q_indices=scaled_q_indices,
        )
        # Inject the max_wavespeed implementation into the numpy module
        # BEFORE compilation so the lambdified flux/fluctuation expressions use it
        from zoomy_core.transformation.to_numpy import NumpyRuntimeSymbolic
        NumpyRuntimeSymbolic.module["max_wavespeed"] = max_wavespeed_fn
        runtime_numerics = numerics.to_runtime_numpy()
        # Also set the local_max_abs_eigenvalue for CFL
        runtime_numerics.local_max_abs_eigenvalue = lambda Q, Qaux, p, n: max_wavespeed_fn(*Q, *Qaux, *p, *n)

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        dim = symbolic_model.dimension
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
                    runtime_numerics.numerical_fluctuations(qA, qB, qauxA, qauxB, parameters, n),
                    dtype=float,
                ).reshape(-1)
                num_flux = np.asarray(
                    runtime_numerics.numerical_flux(qA, qB, qauxA, qauxB, parameters, n),
                    dtype=float,
                ).reshape(-1)
                Dp = fluct[:n_vars]
                Dm = fluct[n_vars:]
                dQ[:, iA[f]] -= (num_flux + Dm) * face_volumes[f] / cell_volumesA[f]
                dQ[:, iB[f]] -= (-num_flux + Dp) * face_volumes[f] / cell_volumesB[f]
            return dQ

        return flux_operator


class GeneratedModelSolver(_GeneratedModelFluxMixin, DerivativeAwareSolver):
    """
    Solver for NumericalModel or any Model with [b, h, ...] state layout.

    Wet/dry handling via model parameter 'eps_wet' (default 1e-3):
    - Eigenvalue computation skips cells with h < eps_wet
    - update_variables clamps h >= 0 and ramps momentum to zero
    """
    pass
