"""Symbolic Riemann solvers: Rusanov, positive, nonconservative variants."""

import numpy as np
import param
import sympy as sp

from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.basefunction import SymbolicRegistrar
from zoomy_core.model.basemodel import Model
from zoomy_core.model.kernel_functions import max_wavespeed
from zoomy_core.transformation.to_numpy import NumpyRuntimeSymbolic


class Numerics(param.Parameterized, SymbolicRegistrar):
    """Numerics. (class)."""
    name = param.String(default="NumericsV2")
    model = param.ClassSelector(class_=Model, is_instance=True)

    # Field map format:
    # {
    #   "h": {"container": "q", "index": 1},
    #   "b": {"container": "q", "index": 0},
    #   "hinv": {"container": "qaux", "index": 0},  # optional
    # }
    field_map = param.Dict(
        default=None, allow_None=True,
    )
    scaled_q_indices = param.List(default=None, allow_None=True)

    def __init__(self, model, **params):
        """Initialize the instance."""
        super().__init__(model=model, **params)
        self.functions, self.call = Zstruct(), Zstruct()

        self.variables = ZArray(self.model.variables.get_list())
        self.aux_variables = ZArray(self.model.aux_variables.get_list())
        self.parameters = ZArray(self.model.parameters.get_list())
        self.normal = ZArray(self.model.normal.get_list())

        self.variables_minus = self._create_v("Q_minus", self.model.n_variables)
        self.variables_plus = self._create_v("Q_plus", self.model.n_variables)
        self.aux_variables_minus = self._create_v("Qaux_minus", self.model.n_aux_variables)
        self.aux_variables_plus = self._create_v("Qaux_plus", self.model.n_aux_variables)

        self.flux_minus = self._create_v("flux_minus", self.model.n_variables)
        self.flux_plus = self._create_v("flux_plus", self.model.n_variables)
        self.source_term = self._create_v("source_term", self.model.n_variables)

        if self.field_map:
            self._field_map = self._normalize_and_validate_field_map(self.field_map)
        else:
            self._field_map = {}
        self._scaled_q_indices = self._resolve_scaled_q_indices(self.scaled_q_indices)

        self._initialize_functions()

    def _create_v(self, name, size):
        """Internal helper `_create_v`."""
        v = ZArray([sp.Symbol(f"{name}_{i}", real=True) for i in range(size)])
        v._symbolic_name = name
        return v

    def _normalize_and_validate_field_map(self, field_map):
        """Internal helper `_normalize_and_validate_field_map`."""
        if "h" not in field_map or "b" not in field_map:
            raise ValueError("field_map must define entries for both 'h' and 'b'.")

        out = {}
        for key, spec in field_map.items():
            if not isinstance(spec, dict):
                raise TypeError(f"field_map['{key}'] must be a dict.")
            if "container" not in spec or "index" not in spec:
                raise ValueError(
                    f"field_map['{key}'] must contain 'container' and 'index'."
                )
            container = str(spec["container"])
            index = int(spec["index"])
            if container not in {"q", "qaux"}:
                raise ValueError(
                    f"field_map['{key}']['container'] must be 'q' or 'qaux', got '{container}'."
                )
            upper = self.model.n_variables if container == "q" else self.model.n_aux_variables
            if index < 0 or index >= upper:
                raise IndexError(
                    f"field_map['{key}'] index {index} out of bounds for {container} (size={upper})."
                )
            out[key] = {"container": container, "index": index}
        return out

    def _resolve_scaled_q_indices(self, scaled_q_indices):
        """Internal helper `_resolve_scaled_q_indices`."""
        if scaled_q_indices is not None:
            cleaned = [int(i) for i in scaled_q_indices]
            for i in cleaned:
                if i < 0 or i >= self.model.n_variables:
                    raise IndexError(
                        f"scaled_q_indices contains out-of-bounds index {i}."
                    )
            return cleaned

        excluded = set()
        for key in ("h", "b"):
            if key in self._field_map:
                spec = self._field_map[key]
                if spec["container"] == "q":
                    excluded.add(spec["index"])
        return [i for i in range(self.model.n_variables) if i not in excluded]

    def _field_value(self, field_name, q_state, qaux_state):
        """Internal helper `_field_value`."""
        spec = self._field_map[field_name]
        arr = q_state if spec["container"] == "q" else qaux_state
        return arr[spec["index"]]

    def _set_field_value(self, field_name, q_state, qaux_state, value):
        """Internal helper `_set_field_value`."""
        spec = self._field_map[field_name]
        if spec["container"] == "q":
            q_state[spec["index"]] = value
        else:
            qaux_state[spec["index"]] = value

    def _eps_symbol(self):
        """Internal helper `_eps_symbol`."""
        if hasattr(self.model.parameters, "contains") and self.model.parameters.contains("eps"):
            return self.model.parameters.eps
        return sp.Float(1e-12)

    def _initialize_functions(self):
        """Internal helper `_initialize_functions`."""
        sig = Zstruct(
            q_minus=self.variables_minus,
            q_plus=self.variables_plus,
            aux_minus=self.aux_variables_minus,
            aux_plus=self.aux_variables_plus,
            p=self.parameters,
            normal=self.normal,
        )

        self.register_symbolic_function("numerical_flux", self.numerical_flux, sig)
        self.register_symbolic_function(
            "numerical_fluctuations", self.numerical_fluctuations, sig
        )

        eig_sig = Zstruct(
            Q=self.variables, Qaux=self.aux_variables, p=self.parameters, n=self.normal
        )
        self.register_symbolic_function(
            "local_max_abs_eigenvalue", self.local_max_eigenvalue_definition, eig_sig
        )

    def local_max_eigenvalue_definition(self):
        """
        Returns the opaque max_wavespeed function.
        The actual implementation is provided by the backend at runtime.
        """
        return max_wavespeed(
            *list(self.variables), *list(self.aux_variables),
            *list(self.parameters), *list(self.normal),
        )

    def local_max_abs_eigenvalue(self, Q=None, Qaux=None, p=None, n=None):
        """
        Called during symbolic Rusanov construction.
        Returns opaque max_wavespeed with the given state.
        """
        if Q is None:
            return self.local_max_eigenvalue_definition()
        return max_wavespeed(*list(Q), *list(Qaux), *list(p), *list(n))

    def _model_eval(self, function_name, Q, Qaux, p, n=None):
        """Internal helper `_model_eval`."""
        definition = self.model.functions[function_name].definition
        sub_map = {}
        for sym, val in zip(self.model.variables.get_list(), list(Q)):
            sub_map[sym] = val
        for sym, val in zip(self.model.aux_variables.get_list(), list(Qaux)):
            sub_map[sym] = val
        for sym, val in zip(self.model.parameters.get_list(), list(p)):
            sub_map[sym] = val
        if n is not None:
            for sym, val in zip(self.model.normal.get_list(), list(n)):
                sub_map[sym] = val

        if hasattr(definition, "subs"):
            out = definition.subs(sub_map)
        else:
            out = definition

        if isinstance(out, ZArray):
            return out
        if isinstance(out, (sp.NDimArray, sp.MatrixBase, list, tuple)):
            return ZArray(out)
        return out

    def to_runtime(self, backend="numpy"):
        """To runtime."""
        backend = backend.lower()
        if backend == "numpy":
            return NumpyRuntimeSymbolic(self)
        raise ValueError(f"Unsupported runtime backend '{backend}'.")

    def to_runtime_numpy(self):
        """To runtime numpy."""
        return self.to_runtime("numpy")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray(zeros)

    def numerical_fluctuations(self):
        """Numerical fluctuations."""
        zeros = ZArray.zeros(self.model.n_variables)
        return ZArray([zeros, zeros])


class Rusanov(Numerics):
    """Rusanov. (class)."""
    name = param.String(default="RusanovV2")

    def get_viscosity_identity_flux(self):
        """Get viscosity identity flux."""
        n = self.model.n_variables
        Id = ZArray.zeros(n, n)
        for i in range(n):
            Id[i, i] = 1
        # Exclude bottom-topography from Rusanov dissipation when b is part of
        # the conservative state. This preserves the stationary bed variable.
        b_spec = self._field_map["b"]
        if b_spec["container"] == "q":
            b_idx = b_spec["index"]
            if 0 <= b_idx < self.model.n_variables:
                Id[b_idx, b_idx] = 0
        return Id

    def get_viscosity_identity_fluctuations(self):
        # Conservative Rusanov variants do not use fluctuation viscosity.
        """Get viscosity identity fluctuations."""
        n = self.model.n_variables
        return ZArray.zeros(n, n)

    def numerical_flux(self):
        """Numerical flux."""
        return self._compute_flux(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        """Internal helper `_compute_flux`."""
        FL = self._model_eval("flux", qL, auxL, p)
        FR = self._model_eval("flux", qR, auxR, p)
        PL = self._model_eval("hydrostatic_pressure", qL, auxL, p)
        PR = self._model_eval("hydrostatic_pressure", qR, auxR, p)
        s_max = sp.Max(
            self.local_max_abs_eigenvalue(qL, auxL, p, n),
            self.local_max_abs_eigenvalue(qR, auxR, p, n),
        )
        Id = self.get_viscosity_identity_flux()
        return 0.5 * ((FL + PL) @ n + (FR + PR) @ n) - 0.5 * s_max * Id @ (qR - qL)


class PositiveRusanov(Rusanov):
    """PositiveRusanov. (class)."""
    name = param.String(default="PositiveRusanovV2")

    def hydrostatic_reconstruction(self, qL, qR, auxL, auxR):
        """Hydrostatic reconstruction."""
        qLs = ZArray(qL)
        qRs = ZArray(qR)
        qauxL = ZArray(auxL)
        qauxR = ZArray(auxR)

        bL = self._field_value("b", qL, auxL)
        bR = self._field_value("b", qR, auxR)
        hL = self._field_value("h", qL, auxL)
        hR = self._field_value("h", qR, auxR)

        b_star = sp.Max(bL, bR)
        hL_star = sp.Max(0.0, hL + bL - b_star)
        hR_star = sp.Max(0.0, hR + bR - b_star)

        eps = self._eps_symbol()
        hL_eff = sp.Max(hL, eps)
        hR_eff = sp.Max(hR, eps)

        self._set_field_value("b", qLs, qauxL, b_star)
        self._set_field_value("b", qRs, qauxR, b_star)
        self._set_field_value("h", qLs, qauxL, hL_star)
        self._set_field_value("h", qRs, qauxR, hR_star)

        for idx in self._scaled_q_indices:
            qLs[idx] = qL[idx] * hL_star / hL_eff
            qRs[idx] = qR[idx] * hR_star / hR_eff

        if "hinv" in self._field_map:
            self._set_field_value("hinv", qLs, qauxL, 1 / (hL_star + eps))
            self._set_field_value("hinv", qRs, qauxR, 1 / (hR_star + eps))

        return qLs, qRs, qauxL, qauxR

    def numerical_flux(self):
        """Numerical flux."""
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
        return self._compute_flux(qLs, qRs, qauxL, qauxR, self.parameters, self.normal)

    def numerical_fluctuations(self):
        """Numerical fluctuations."""
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )

        P_mat_L_raw = self._model_eval(
            "hydrostatic_pressure",
            self.variables_minus,
            self.aux_variables_minus,
            self.parameters,
        )
        P_mat_R_raw = self._model_eval(
            "hydrostatic_pressure",
            self.variables_plus,
            self.aux_variables_plus,
            self.parameters,
        )
        P_mat_L_star = self._model_eval(
            "hydrostatic_pressure", qLs, qauxL, self.parameters
        )
        P_mat_R_star = self._model_eval(
            "hydrostatic_pressure", qRs, qauxR, self.parameters
        )

        Dm_jump = (P_mat_L_raw - P_mat_L_star) @ self.normal
        Dp_jump = (P_mat_R_star - P_mat_R_raw) @ self.normal

        base_fluct = super().numerical_fluctuations()
        Dp_base = ZArray(base_fluct[0, :])
        Dm_base = ZArray(base_fluct[1, :])

        Dp = Dp_base + Dp_jump
        Dm = Dm_base + Dm_jump

        return ZArray([Dp, Dm])


class NonconservativeRusanov(Rusanov):
    """NonconservativeRusanov. (class)."""
    name = param.String(default="NonconservativeRusanovV2")
    integration_order = param.Integer(default=3)

    def get_path_integral_states(self):
        """Get path integral states."""
        return (
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )

    def get_viscosity_identity_flux(self):
        # Nonconservative/quasilinear variants place dissipation in
        # numerical_fluctuations to avoid double counting in flux + fluctuation
        # updates.
        """Get viscosity identity flux."""
        n = self.model.n_variables
        return ZArray.zeros(n, n)

    def get_viscosity_identity_fluctuations(self):
        """Get viscosity identity fluctuations."""
        n = self.model.n_variables
        Id = ZArray.zeros(n, n)
        for i in range(n):
            Id[i, i] = 1
        # Preserve stationary bed variable when b is part of conservative state.
        if "b" in self._field_map:
            b_spec = self._field_map["b"]
            if b_spec["container"] == "q":
                b_idx = b_spec["index"]
                if 0 <= b_idx < self.model.n_variables:
                    Id[b_idx, b_idx] = 0
        return Id

    def numerical_fluctuations(self):
        """Numerical fluctuations."""
        qLs, qRs, qauxL, qauxR = self.get_path_integral_states()
        nc_fluct = self._compute_fluctuations(
            qLs, qRs, qauxL, qauxR, self.parameters, self.normal
        )
        out = super().numerical_fluctuations()
        return out + nc_fluct

    def _call_model_matrix(self):
        """Internal helper `_call_model_matrix`."""
        return lambda Q, Qaux, p: self._model_eval("nonconservative_matrix", Q, Qaux, p)

    def _compute_fluctuations(self, qL, qR, auxL, auxR, p, n):
        """Internal helper `_compute_fluctuations`."""
        xi_np, wi_np = np.polynomial.legendre.leggauss(self.integration_order)
        xi_np = 0.5 * (xi_np + 1)
        wi_np = 0.5 * wi_np

        dQ = qR - qL
        dAux = auxR - auxL
        n_vars = self.model.n_variables
        dim = len(n)

        A_int = ZArray.zeros(n_vars, n_vars)

        for xi, wi in zip(xi_np, wi_np):
            q_path = qL + xi * dQ
            aux_path = auxL + xi * dAux
            A_tensor = self._call_model_matrix()(q_path, aux_path, p)

            A_n = ZArray.zeros(n_vars, n_vars)
            for i in range(n_vars):
                for j in range(n_vars):
                    val = 0
                    for d in range(dim):
                        val += A_tensor[i, j, d] * n[d]
                    A_n[i, j] = val
            A_int += wi * A_n

        s_max = sp.Max(
            self.local_max_abs_eigenvalue(qL, auxL, p, n),
            self.local_max_abs_eigenvalue(qR, auxR, p, n),
        )

        term_advection = A_int @ dQ
        Id = self.get_viscosity_identity_fluctuations()
        term_dissipation = s_max * (Id @ dQ)

        Dp_matrix = 0.5 * (term_advection + term_dissipation)
        Dm_matrix = 0.5 * (term_advection - term_dissipation)

        return ZArray([Dp_matrix, Dm_matrix])


class PositiveNonconservativeRusanov(PositiveRusanov, NonconservativeRusanov):
    """PositiveNonconservativeRusanov. (class)."""
    name = param.String(default="PositiveNonconservativeRusanovV2")

    def get_path_integral_states(self):
        """Get path integral states."""
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )


class QuasilinearRusanov(NonconservativeRusanov):
    """QuasilinearRusanov. (class)."""
    name = param.String(default="QuasilinearRusanovV2")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray(zeros)

    def _call_model_matrix(self):
        """Internal helper `_call_model_matrix`."""
        return lambda Q, Qaux, p: self._model_eval("quasilinear_matrix", Q, Qaux, p)


class PositiveQuasilinearRusanov(PositiveRusanov, QuasilinearRusanov):
    """PositiveQuasilinearRusanov. (class)."""
    name = param.String(default="PositiveQuasilinearRusanovV2")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray(zeros)

    def get_path_integral_states(self):
        """Get path integral states."""
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
