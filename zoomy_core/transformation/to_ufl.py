"""Module `zoomy_core.transformation.to_ufl`."""

import ufl
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel


def _ufl_conditional(condition, true_val, false_val):
    """Internal helper `_ufl_conditional`."""
    if condition is True:
        return true_val
    elif condition is False:
        return false_val
    elif isinstance(true_val, tuple):
        return ufl.conditional(condition, ufl.as_vector(true_val), ufl.as_vector(false_val))
    else:
        return ufl.conditional(condition, true_val, false_val)


def _ufl_from_nested_list(data):
    """Convert a nested list (from lambdify's ImmutableDenseMatrix) to UFL.

    - 2D list-of-lists -> ``ufl.as_tensor``
    - 1D list          -> ``ufl.as_vector``
    - scalar           -> pass-through
    """
    if isinstance(data, (list, tuple)):
        if data and isinstance(data[0], (list, tuple)):
            return ufl.as_tensor(data)
        return ufl.as_vector(data)
    return data


class UFLRuntimeModel(NumpyRuntimeModel):
    """UFLRuntimeModel. (class)."""
    printer = None
    use_cse = False  # Firedrake's SymPy can't CSE ImmutableDenseNDimArray

    # -----------------------------------------------------------------
    # Array -> Matrix conversion so that lambdify can serialise the expr
    # -----------------------------------------------------------------

    @staticmethod
    def _array_to_matrix(expr):
        """Convert ``ImmutableDenseNDimArray`` to ``ImmutableDenseMatrix``.

        SymPy's ``lambdify`` (without a code printer) cannot serialise
        ``ImmutableDenseNDimArray`` but *can* handle ``ImmutableDenseMatrix``.

        * 0-D  -> bare scalar symbol
        * 1-D  -> column vector  (n, 1)
        * 2-D  -> matrix         (r, c)
        * 3-D+ -> reshaped to 2-D (product-of-leading-dims, last-dim)
        """
        import sympy as sp
        from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray

        if not isinstance(expr, (ImmutableDenseNDimArray, sp.Array)):
            return expr

        shape = tuple(int(s) for s in expr.shape)
        flat = list(expr._array)  # always a flat list of SymPy exprs

        if len(shape) == 0:
            return flat[0] if flat else expr
        elif len(shape) == 1:
            return sp.ImmutableDenseMatrix([[e] for e in flat])
        elif len(shape) == 2:
            return sp.ImmutableDenseMatrix(shape[0], shape[1], flat)
        else:
            import functools, operator
            nrows = functools.reduce(operator.mul, shape[:-1], 1)
            ncols = shape[-1]
            return sp.ImmutableDenseMatrix(nrows, ncols, flat)

    def _vectorize_expression(self, expr, signature):
        """Override: vectorise, then convert Array -> Matrix for lambdify."""
        result = super()._vectorize_expression(expr, signature)

        from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
        import sympy as sp
        if isinstance(result, (ImmutableDenseNDimArray, sp.Array)):
            return self._array_to_matrix(result)
        return result

    def _lambdify_function(self, function_obj, modules):
        """Skip functions with zero-size output (e.g. Jacobian when n_aux=0)."""
        import sympy as sp
        expr = function_obj.definition
        if hasattr(expr, 'shape') and any(int(s) == 0 for s in expr.shape):
            import numpy as np
            shape = tuple(int(s) for s in expr.shape)
            return lambda *a, _s=shape: np.empty(_s)
        return super()._lambdify_function(function_obj, modules)
    
    module = {
        'ones_like': lambda x: 0*x + 1,
        'zeros_like': lambda x:  0*x,
        'array': ufl.as_vector,
        'squeeze': lambda x: x,
        'conditional': _ufl_conditional,
        
        # --- Elementary arithmetic ---
        "Abs": abs,
        "sign": ufl.sign,
        "Min": ufl.min_value,
        "Max": ufl.max_value,
        # --- Powers and roots ---
        "sqrt": ufl.sqrt,
        "exp": ufl.exp,
        "ln": ufl.ln,
        "pow": lambda x, y: x**y,
        # --- Trigonometric functions ---
        "sin": ufl.sin,
        "cos": ufl.cos,
        "tan": ufl.tan,
        "asin": ufl.asin,
        "acos": ufl.acos,
        "atan": ufl.atan,
        "atan2": ufl.atan2,
        # --- Hyperbolic functions ---
        "sinh": ufl.sinh,
        "cosh": ufl.cosh,
        "tanh": ufl.tanh,
        # --- Piecewise / conditional logic ---
        "Heaviside": lambda x: ufl.conditional(x >= 0, 1.0, 0.0),
        "Piecewise": ufl.conditional,
        "signum": ufl.sign,
        # --- Vector & tensor ops ---
        "dot": ufl.dot,
        "inner": ufl.inner,
        "outer": ufl.outer,
        "cross": ufl.cross,
        # --- Differential operators (used in forms) ---
        "grad": ufl.grad,
        "div": ufl.div,
        "curl": ufl.curl,
        # --- Common constants ---
        "pi": ufl.pi,
        "E": ufl.e,
        # --- Matrix and linear algebra ---
        "transpose": ufl.transpose,
        "det": ufl.det,
        "inv": ufl.inv,
        "tr": ufl.tr,
        # --- Python builtins (SymPy may emit these) ---
        "abs": abs,
        "min": ufl.min_value,
        "max": ufl.max_value,
        "sqrt": ufl.sqrt,
        "sum": lambda x: ufl.Constant(sum(x)) if isinstance(x, (list, tuple)) else x,
        "ImmutableDenseMatrix": _ufl_from_nested_list,
    }
