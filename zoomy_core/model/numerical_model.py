"""
NumericalModel: wraps any analytical Model for numerical simulation.

Two-level regularization:
  eps      (small, ~1e-8)  : regularizes 1/h → 1/(h+eps) in flux/source/quasilinear
  eps_wet  (larger, ~1e-3) : wet/dry threshold for eigenvalue skip and momentum deletion

Momentum ramp: update_variables zeros out momentum-like variables when h < eps_wet,
preventing unphysical velocities at near-dry cells from driving the solution.

Usage:
    analytical = GeneratedShallowModel(...)
    numerical = NumericalModel(analytical)
"""

import sympy as sp
from sympy import Symbol, Pow, S
import param
import numpy as np

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.custom_sympy_functions import conditional, clamp_positive, clamp_momentum
from zoomy_core.model.boundary_conditions import BoundaryConditions
from zoomy_core.model.initial_conditions import Constant, InitialConditions


def regularize_denominators(expr, h_sym, eps_sym):
    """Replace h^(-n) with (h + eps)^(-n) for n > 0."""
    if not isinstance(expr, sp.Basic):
        return expr
    h_reg = h_sym + eps_sym
    return expr.replace(
        lambda e: isinstance(e, Pow) and e.base == h_sym and e.exp.is_negative,
        lambda e: h_reg ** e.exp,
    )


def regularize_sqrt_arguments(expr, h_sym, eps_sym):
    """Replace h inside fractional powers with (h + eps) to prevent sqrt(negative)."""
    if not isinstance(expr, sp.Basic):
        return expr
    h_reg = h_sym + eps_sym
    return expr.replace(
        lambda e: isinstance(e, Pow) and e.exp.is_Rational and not e.exp.is_integer and e.base.has(h_sym),
        lambda e: Pow(e.base.subs(h_sym, h_reg), e.exp),
    )


class NumericalModel(Model):
    """
    Wraps an analytical Model for numerical simulation.

    Regularization:
    - 1/h → 1/(h + eps) in all compiled expressions
    - Momentum zeroed when h < eps_wet (via conditional in update_variables)
    - Eigenvalues use conditional(h > eps_wet, ev, 0) for symbolic path
    """

    def __init__(self, analytical_model, eigenvalue_proxy_level=None, **kwargs):
        """
        Parameters
        ----------
        analytical_model : Model
            The symbolic model to wrap.
        eigenvalue_proxy_level : int or None
            If set, uses eigenvalues from a lower-level model of the same basis
            as a CFL proxy. E.g., eigenvalue_proxy_level=1 for a level=2 model
            uses the L1 eigenvalues (fast symbolic) padded with zeros for
            the extra variables.
        """
        self._analytical = analytical_model
        self._eigenvalue_proxy_level = eigenvalue_proxy_level
        self.eigenvalue_mode = getattr(analytical_model, "eigenvalue_mode", "symbolic")

        var_keys = analytical_model.variables.keys()

        param_def = {}
        for k in analytical_model.parameters.keys():
            sym = analytical_model.parameters[k]
            default = analytical_model.parameter_defaults_map.get(k, 0.0)
            if sym.is_positive:
                param_def[k] = (default, "positive")
            else:
                param_def[k] = default
        if "eps" not in param_def:
            param_def["eps"] = (1e-8, "positive")
        if "eps_wet" not in param_def:
            param_def["eps_wet"] = (1e-3, "positive")

        super().__init__(
            init_functions=False,
            name=f"Numerical({analytical_model.name})",
            dimension=analytical_model.dimension,
            variables=list(var_keys),
            aux_variables=0,
            parameters=param_def,
            boundary_conditions=kwargs.get("boundary_conditions", analytical_model.boundary_conditions),
            initial_conditions=kwargs.get("initial_conditions", analytical_model.initial_conditions),
            **{k: v for k, v in kwargs.items() if k not in ("boundary_conditions", "initial_conditions")},
        )

        self._h_analytical = analytical_model.variables[1]
        # Keep positive=True for h — the analytical model needs it for
        # correct simplification. The clamp uses opaque clamp_positive()
        # which SymPy never simplifies away.
        if self._h_analytical.is_positive and not self.variables[1].is_positive:
            old_h = self.variables[1]
            new_h = sp.Symbol(old_h.name, positive=True, real=True)
            self.variables[1] = new_h
        self._h_numerical = self.variables[1]
        self._eps_sym = self.parameters[list(self.parameters.keys()).index("eps")]
        self._eps_wet_sym = self.parameters[list(self.parameters.keys()).index("eps_wet")]
        self._build_subs_map()

        self.numerics_scaled_q_indices = self._detect_scaled_indices()

        self._initialize_functions()

    def _build_subs_map(self):
        self._subs = {}
        for i in range(self._analytical.n_variables):
            old = self._analytical.variables[i]
            new = self.variables[i]
            if old != new:
                self._subs[old] = new
        for k in self._analytical.parameters.keys():
            old_p = self._analytical.parameters[k]
            if self.parameters.contains(k):
                new_p = self.parameters[k]
                if old_p != new_p:
                    self._subs[old_p] = new_p
        for i in range(len(self._analytical.normal.keys())):
            old_n = self._analytical.normal[i]
            new_n = self.normal[i]
            if old_n != new_n:
                self._subs[old_n] = new_n

    def _remap_symbols(self, expr):
        if expr is None or not isinstance(expr, sp.Basic):
            return expr
        if self._subs:
            return expr.subs(self._subs)
        return expr

    def _regularize_expr(self, expr):
        if expr is None:
            return expr
        result = self._remap_symbols(expr)
        return self._apply_elementwise(result, regularize_denominators, self._h_numerical, self._eps_sym)

    def _regularize_eigenvalue_expr(self, expr):
        if expr is None:
            return expr
        result = self._remap_symbols(expr)
        result = self._apply_elementwise(result, regularize_denominators, self._h_numerical, self._eps_sym)
        result = self._apply_elementwise(result, regularize_sqrt_arguments, self._h_numerical, self._eps_sym)
        return result

    @staticmethod
    def _apply_elementwise(expr, func, *args):
        if hasattr(expr, '_array'):
            new_elements = [func(e, *args) for e in expr._array]
            return ZArray(new_elements, shape=expr.shape)
        return func(expr, *args)

    def _detect_scaled_indices(self):
        return list(range(2, self.n_variables))

    def flux(self):
        raw = self._analytical.flux()
        return ZArray(self._regularize_expr(raw))

    def hydrostatic_pressure(self):
        raw = self._analytical.hydrostatic_pressure()
        return ZArray(self._regularize_expr(raw))

    def nonconservative_matrix(self):
        raw = self._analytical.nonconservative_matrix()
        return ZArray(self._regularize_expr(raw))

    def source(self):
        raw = self._analytical.source()
        return ZArray(self._regularize_expr(raw))

    def quasilinear_matrix(self):
        raw = self._analytical.quasilinear_matrix()
        return ZArray(self._regularize_expr(raw))

    def eigenvalues(self):
        """
        Eigenvalues for the model. Used by the backend's max_wavespeed
        implementation. No dry-cell guard here — that belongs in the
        backend function, not in the symbolic expressions.
        """
        if self.eigenvalue_mode == "numerical" and self._eigenvalue_proxy_level is None:
            return ZArray([sp.Integer(0)] * self.n_variables)

        if self._eigenvalue_proxy_level is not None:
            raw = self._compute_proxy_eigenvalues()
        else:
            raw = self._analytical.eigenvalues()

        return ZArray(self._regularize_eigenvalue_expr(raw))

    def _compute_proxy_eigenvalues(self):
        """Compute eigenvalues from a lower-level model of the same basis."""
        try:
            from zoomy_core.model.models.projected_model import ProjectedModel
            from zoomy_core.model.models.model_derivation import derive_shallow_moments
            from zoomy_core.model.models.ins_generator import StateSpace, Newtonian
            state = StateSpace(dimension=self._analytical.dimension + 1)
            pre = derive_shallow_moments(state, material=Newtonian(state))
            proxy = ProjectedModel(
                pre,
                basis_type=getattr(self._analytical, "basis_type", None) or type(self._analytical.basisfunctions),
                level=self._eigenvalue_proxy_level,
                n_layers=getattr(self._analytical, "n_layers", 1),
                eigenvalue_mode="symbolic",
            )
        except Exception:
            from zoomy_core.model.models.legacy.generated_shallow_model import GeneratedShallowModel
            proxy = GeneratedShallowModel(
                n_layers=getattr(self._analytical, "n_layers", 1),
                level=self._eigenvalue_proxy_level,
                dimension=self._analytical.dimension,
                basis_type=getattr(self._analytical, "basis_type", None) or type(self._analytical.basisfunctions),
                eigenvalue_mode="symbolic",
            )
        proxy_evs = list(proxy.eigenvalues())
        n_needed = self.n_variables
        n_have = len(proxy_evs)
        if n_have < n_needed:
            proxy_evs.extend([sp.Integer(0)] * (n_needed - n_have))
        return ZArray(proxy_evs[:n_needed])

    def update_variables(self):
        """
        Wet/dry treatment applied every timestep via opaque numerical functions:
        1. clamp_positive(h) — prevents negative water depth
        2. clamp_momentum(hu, h, u_max) — caps velocity at sqrt(g*h) + eps_wet
        3. Linear ramp via conditional — zeros momentum below eps_wet
        """
        h = self._h_numerical
        eps = self._eps_sym
        eps_wet = self._eps_wet_sym
        g = self.parameters[list(self.parameters.keys()).index("g")]

        result = ZArray(list(self.variables.values()))
        h_safe = clamp_positive(h)
        result[1] = h_safe

        u_max = sp.sqrt(g * (h_safe + eps)) + eps_wet
        ramp = sp.Min(h_safe / eps_wet, sp.Integer(1))

        for i in self._detect_scaled_indices():
            result[i] = clamp_momentum(result[i], h_safe, u_max) * ramp
        return result

    def get_field_map(self):
        var_keys = list(self.variables.keys())
        return {
            "h": {"container": "q", "index": var_keys.index("h") if "h" in var_keys else 1},
            "b": {"container": "q", "index": var_keys.index("b") if "b" in var_keys else 0},
        }

    @property
    def wet_dry_threshold(self):
        return self._eps_wet_sym

    @property
    def eigenvalue_regularization_eps(self):
        return self._eps_sym
