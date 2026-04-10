"""Kernel — SymbolicRegistrar for backend-agnostic helper functions.

The Kernel holds basefunctions (safe_denominator, clamp_positive, etc.)
that model and numerics expressions reference symbolically.  Each backend
(numpy, jax, C, UFL) provides concrete implementations during code
transformation, using the same NumpyRuntimeSymbolic / JaxRuntimeSymbolic
machinery that compiles Model and Numerics functions.

After code transformation the user gets three compiled artifacts:
    1. Model   — physics (flux, source, eigenvalues)
    2. Numerics — Riemann solver (numerical flux, fluctuations)
    3. Kernel   — helper functions shared by both

Usage::

    kernel = Kernel(model)
    runtime_kernel = NumpyRuntimeSymbolic(kernel)
    # merge kernel functions into the module dict used by model/numerics
"""

from __future__ import annotations

import sympy as sp
import param

from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basefunction import SymbolicRegistrar


class Kernel(param.Parameterized, SymbolicRegistrar):
    """Backend-agnostic registry of numerical helper functions.

    Mirrors the Model/Numerics pattern: each function is a ``Function``
    basefunction registered via ``register_symbolic_function``.  The
    symbolic definitions serve as blueprints; actual implementations
    are provided by the backend during compilation.
    """

    def __init__(self, model=None, **params):
        super().__init__(**params)
        self.functions = Zstruct()
        self.call = Zstruct()

        # Build convenient name → symbol maps from model
        if model is not None:
            self._var_map = {str(k): v for k, v in model.variables.items()}
            self._param_map = {str(k): v for k, v in model.parameters.items()}
        else:
            self._var_map = {}
            self._param_map = {}

        # Kernel function arguments: lightweight symbols, not full Q/Qaux/p
        self._x = sp.Symbol("x", real=True)
        self._y = sp.Symbol("y", real=True)
        self._z = sp.Symbol("z_arg", real=True)
        self._c = sp.Symbol("c")  # condition
        self._t = sp.Symbol("t_branch", real=True)  # true branch
        self._f = sp.Symbol("f_branch", real=True)  # false branch
        self._eps = self._param_map.get("eps", sp.Symbol("eps", positive=True))

        self._initialize_functions()

    # ── helpers for model parameter access by name ─────────────────────

    def var(self, name: str) -> sp.Symbol:
        """Get a model variable symbol by name."""
        return self._var_map[name]

    def par(self, name: str) -> sp.Symbol:
        """Get a model parameter symbol by name."""
        return self._param_map[name]

    # ── function registration ──────────────────────────────────────────

    def _initialize_functions(self):
        """Register all kernel functions as basefunctions."""
        x, y, z = self._x, self._y, self._z
        c, t, f = self._c, self._t, self._f

        # Each function gets a minimal signature: just the args it needs.
        # The backend fills in the actual implementation.

        self.register_symbolic_function(
            "safe_denominator",
            self.safe_denominator,
            Zstruct(x=x, eps=self._eps),
        )
        self.register_symbolic_function(
            "clamp_positive",
            self.clamp_positive,
            Zstruct(x=x),
        )
        self.register_symbolic_function(
            "clamp_momentum",
            self.clamp_momentum,
            Zstruct(hu=x, h=y, u_max=z),
        )
        self.register_symbolic_function(
            "conditional",
            self.conditional,
            Zstruct(condition=c, true_val=t, false_val=f),
        )

    # ── symbolic definitions (blueprints) ──────────────────────────────
    # Each returns a symbolic expression.  Backends override with
    # language-specific implementations during compilation.

    def safe_denominator(self):
        """safe_denominator(x) → x regularized to avoid 1/0."""
        return self._x + self._eps

    def clamp_positive(self):
        """clamp_positive(x) → max(x, 0)."""
        return sp.Max(self._x, 0)

    def clamp_momentum(self):
        """clamp_momentum(hu, h, u_max) → sign(hu) * min(|hu|, h * u_max)."""
        hu, h, u_max = self._x, self._y, self._z
        return sp.sign(hu) * sp.Min(sp.Abs(hu), h * u_max)

    def conditional(self):
        """conditional(c, t, f) → t if c else f."""
        return sp.Piecewise((self._t, self._c), (self._f, True))

    # ── automatic expression regularization ───────────────────────────

    def regularize(self, model):
        """Walk model expressions and regularize denominators involving
        positive variables (like h) that can reach zero.

        Finds variables declared with ``positive=True`` in the model,
        substitutes them to zero in each expression, and where the result
        is singular replaces ``1/v`` with ``1/(v + eps)`` in the original.

        This is called automatically by the solver at compilation time.
        The model itself stays analytical — regularization is a Kernel concern.
        """
        eps = self._eps

        # Find positive variables (candidates for zero-singularity)
        positive_vars = []
        for name, sym in model.variables.items():
            if getattr(sym, 'is_positive', False):
                positive_vars.append(sym)

        if not positive_vars:
            return  # nothing to regularize

        # Walk each model function and regularize
        for fn_name, fn_obj in model.functions.items():
            expr = fn_obj.definition
            if expr is None:
                continue
            fn_obj.definition = self._regularize_expr(expr, positive_vars, eps)

    @staticmethod
    def _regularize_expr(expr, positive_vars, eps):
        """Replace 1/v with 1/(v+eps) for positive vars where v=0 causes singularity."""
        from zoomy_core.misc.misc import ZArray

        # ZArray / NDimArray — process element-wise, return same type
        if isinstance(expr, (ZArray, sp.tensor.array.NDimArray)):
            flat = [Kernel._regularize_expr(e, positive_vars, eps) for e in sp.flatten(expr)]
            return ZArray(flat, expr.shape) if isinstance(expr, ZArray) else sp.Array(flat).reshape(*expr.shape)

        if isinstance(expr, (list, tuple)):
            return type(expr)(Kernel._regularize_expr(e, positive_vars, eps) for e in expr)

        if not isinstance(expr, sp.Basic):
            return expr

        # For each positive variable, replace v → v+eps in denominators
        for v in positive_vars:
            expr = expr.replace(
                lambda sub: (isinstance(sub, sp.Pow) and sub.base == v
                             and sub.exp.is_negative),
                lambda sub: sp.Pow(v + eps, sub.exp)
            )
        return expr
