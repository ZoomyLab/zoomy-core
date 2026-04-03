"""
Custom SymPy functions for numerical models.

These are opaque to SymPy (never simplified away) but have concrete
implementations in each backend (NumPy, JAX, C).

Each function must be registered in the backend's module dict:
  - NumPy: to_numpy.py → NumpyRuntimeModel.module
  - JAX:   to_jax.py
  - C:     generic_c.py
"""

import sympy as sp
import itertools
from zoomy_core.misc.misc import ZArray


class clamp_positive(sp.Function):
    """
    clamp_positive(x) = max(x, 0).

    Opaque to SymPy — never simplified, even if x has positive=True.
    At runtime: np.maximum(x, 0).
    """
    nargs = 1
    is_commutative = True

    @classmethod
    def eval(cls, x):
        # Only evaluate for explicit numeric values, never for symbols
        if x.is_Number:
            return sp.Max(x, 0)
        return None  # keep unevaluated

    def _eval_derivative(self, s):
        x = self.args[0]
        return conditional(x > 0, x.diff(s), sp.Integer(0))


class clamp_momentum(sp.Function):
    """
    clamp_momentum(hu, h, u_max) = sign(hu) * min(|hu|, h * u_max).

    Caps momentum so that |u| = |hu/h| <= u_max.
    Opaque to SymPy. At runtime: np.clip(hu, -h*u_max, h*u_max).
    """
    nargs = 3
    is_commutative = True

    @classmethod
    def eval(cls, hu, h, u_max):
        if hu.is_Number and h.is_Number and u_max.is_Number:
            bound = h * u_max
            return sp.Min(sp.Max(hu, -bound), bound)
        return None

    def _eval_derivative(self, s):
        hu, h, u_max = self.args
        return conditional(
            sp.Abs(hu) < h * u_max,
            hu.diff(s),
            sp.sign(hu) * (h * u_max).diff(s),
        )


class max_wavespeed(sp.Function):
    """
    max_wavespeed(Q, Qaux, p, n) — maximum absolute wave speed.

    Opaque to SymPy. Used in Rusanov dissipation and CFL computation.
    Backend implementations:
      - NumPy: provided by NumericalModel (symbolic eigenvalues compiled,
               or np.linalg.eigvals on quasilinear matrix)
      - JAX: same
      - C: compiled from symbolic eigenvalue expressions
    """
    is_commutative = True
    is_real = True
    is_nonnegative = True

    @classmethod
    def eval(cls, *args):
        return None  # always keep unevaluated


class conditional(sp.Function):
    """
    A Vector-Aware, Differentiable Conditional Function.

    Usage:
      conditional(c, t, f)

    Behavior:
      - If inputs are Scalars: Returns a symbolic `conditional` object.
      - If inputs are Vectors: Returns a `ZArray` of symbolic `conditional` objects.

    Differentiation:
      - Implements the 'Active Branch' rule:
        d/dx conditional(c, t, f) = conditional(c, dt/dx, df/dx)
    """

    nargs = 3
    is_commutative = (
        True  # <--- CRITICAL FIX: prevents PolynomialError in charpoly/eigenvals
    )

    def __new__(cls, condition, true_val, false_val, **kwargs):
        # 1. Helper to check for array-like inputs (excluding Symbols)
        """Hook `__new__`."""
        def is_vec(x):
            """Is vec."""
            return hasattr(x, "__getitem__") and not isinstance(
                x, (sp.Symbol, sp.Function)
            )

        is_array_t = is_vec(true_val)
        is_array_f = is_vec(false_val)

        # 2. Vector Case: Broadcast and return ZArray
        if is_array_t or is_array_f:
            # Flatten inputs
            t_flat = list(sp.flatten(true_val)) if is_array_t else [true_val]
            f_flat = list(sp.flatten(false_val)) if is_array_f else [false_val]

            # Determine shape
            if hasattr(true_val, "shape"):
                shape = true_val.shape
            elif hasattr(false_val, "shape"):
                shape = false_val.shape
            else:
                shape = (len(t_flat),)

            # Broadcast scalar '0' filler if lengths mismatch (standard python zip_longest behavior)
            result_list = []

            # Handling scalar vs vector broadcasting manually for safety
            max_len = max(len(t_flat), len(f_flat))
            if not is_array_t:
                t_flat = [true_val] * max_len
            if not is_array_f:
                f_flat = [false_val] * max_len

            for t, f in zip(t_flat, f_flat):
                # Recursively call __new__ (which will hit the Scalar Case below)
                result_list.append(cls(condition, t, f))

            # Return a ZArray containing the symbolic nodes
            return ZArray(result_list).reshape(*shape)

        # 3. Scalar Case: Create the actual Symbolic Node
        return super().__new__(cls, condition, true_val, false_val, **kwargs)

    def _eval_derivative(self, s):
        """
        Differentiates the branches, ignoring the jump at the condition boundary.
        This is standard for numerical Jacobians in FVM.
        """
        condition, true_expr, false_expr = self.args
        return conditional(condition, true_expr.diff(s), false_expr.diff(s))
