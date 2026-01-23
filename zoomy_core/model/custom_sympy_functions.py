import sympy as sp
import itertools
from zoomy_core.misc.misc import ZArray


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
        def is_vec(x):
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
