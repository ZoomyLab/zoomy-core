"""Module `zoomy_core.transformation.to_ufl`."""

import sympy as sp
import ufl
from zoomy_core.transformation.to_numpy import (
    NumpyRuntimeModel,
    NumpyRuntimeSymbolic,
)


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


class _ZoomyVector(sp.Function):
    """Symbolic wrapper so lambdify emits ``_ZoomyVector(e0, e1, ...)``
    which the UFL module maps to ``ufl.as_vector([e0, e1, ...])``.

    This distinguishes 1-D arrays (vectors) from 2-D arrays (tensors) at
    lambdify time, because both would otherwise be serialised as
    ``ImmutableDenseMatrix(...)`` and be indistinguishable.
    """
    pass


class UFLRuntimeModel(NumpyRuntimeModel):
    """UFLRuntimeModel. (class)."""
    printer = None
    use_cse = True

    # -----------------------------------------------------------------
    # Array -> Matrix conversion so that lambdify can serialise the expr
    # -----------------------------------------------------------------

    @staticmethod
    def _array_to_matrix(expr):
        """Convert ``ImmutableDenseNDimArray`` to a lambdify-friendly form.

        SymPy's ``lambdify`` (without a code printer) cannot serialise
        ``ImmutableDenseNDimArray`` but *can* handle ``ImmutableDenseMatrix``
        and custom ``Function`` subclasses.

        * 0-D  -> bare scalar symbol
        * 1-D  -> ``_ZoomyVector(e0, e1, ...)``  -> ``ufl.as_vector``
        * 2-D  -> ``ImmutableDenseMatrix``        -> ``ufl.as_tensor``
        * 3-D+ -> reshaped to 2-D Matrix (product-of-leading, last-dim)
        """
        from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray

        if not isinstance(expr, (ImmutableDenseNDimArray, sp.Array)):
            return expr

        shape = tuple(int(s) for s in expr.shape)
        flat = list(expr._array)  # always a flat list of SymPy exprs

        if len(shape) == 0:
            return flat[0] if flat else expr
        elif len(shape) == 1:
            return _ZoomyVector(*flat)
        elif len(shape) == 2:
            return sp.ImmutableDenseMatrix(shape[0], shape[1], flat)
        else:
            import functools, operator
            nrows = functools.reduce(operator.mul, shape[:-1], 1)
            ncols = shape[-1]
            return sp.ImmutableDenseMatrix(nrows, ncols, flat)

    @staticmethod
    def _distribute_piecewise_over_array(expr):
        """Pull ``Piecewise(NDimArray, …)`` out into an NDimArray of
        scalar ``Piecewise`` expressions.

        SymPy's ``PythonCodePrinter`` (which lambdify falls back to when
        no explicit printer is passed) does not know how to serialise an
        ``ImmutableDenseNDimArray`` *inside* a ``Piecewise`` branch.
        The fix is to swap the nesting: if every branch is an
        ``NDimArray`` of the same shape, return an ``Array`` of length
        ``n`` where each entry is the element-wise ``Piecewise``.

        Then :meth:`_array_to_matrix` (called from
        :meth:`_vectorize_expression`) turns the resulting flat array
        into the appropriate ``_ZoomyVector`` / ``ImmutableDenseMatrix``
        wrapper for the UFL module dict to lower.
        """
        from sympy.tensor.array.dense_ndim_array import (
            ImmutableDenseNDimArray,
        )

        if not isinstance(expr, sp.Piecewise):
            return expr

        branches = []
        for branch in expr.args:
            e, c = branch.expr, branch.cond
            if not isinstance(e, (ImmutableDenseNDimArray, sp.Array)):
                # mixed shape — bail and let the default printer try.
                return expr
            branches.append((list(sp.Array(e)._array), c))

        if not branches:
            return expr

        # All branches must agree on length.
        n = len(branches[0][0])
        if not all(len(b[0]) == n for b in branches):
            return expr

        shape = tuple(int(s) for s in sp.Array(expr.args[0].expr).shape)
        flat = []
        for i in range(n):
            per_branch = [(b[0][i], b[1]) for b in branches]
            flat.append(sp.Piecewise(*per_branch))
        return sp.Array(flat, shape)

    def _vectorize_expression(self, expr, signature):
        """Override: distribute any top-level ``Piecewise`` over its
        ``NDimArray`` branches, vectorise, then convert Array → Matrix
        for lambdify."""
        expr = self._distribute_piecewise_over_array(expr)
        result = super()._vectorize_expression(expr, signature)

        from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
        if isinstance(result, (ImmutableDenseNDimArray, sp.Array)):
            return self._array_to_matrix(result)
        return result

    def _lambdify_function(self, function_obj, modules):
        """Skip functions with zero-size output (e.g. Jacobian when n_aux=0)."""
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
        # SymPy's ``Min`` / ``Max`` accept N arguments; ``ufl.min_value`` /
        # ``ufl.max_value`` are strictly binary.  Reduce N-ary calls to a
        # pairwise chain so any arity lambdifies cleanly.
        "Min": lambda *a: (a[0] if len(a) == 1
                           else __import__("functools").reduce(ufl.min_value, a)),
        "Max": lambda *a: (a[0] if len(a) == 1
                           else __import__("functools").reduce(ufl.max_value, a)),
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
        "min": lambda *a: (a[0] if len(a) == 1
                           else __import__("functools").reduce(ufl.min_value, a)),
        "max": lambda *a: (a[0] if len(a) == 1
                           else __import__("functools").reduce(ufl.max_value, a)),
        "sqrt": ufl.sqrt,
        "sum": lambda x: ufl.Constant(sum(x)) if isinstance(x, (list, tuple)) else x,
        # --- Array / matrix serialisation ---
        "_ZoomyVector": lambda *args: ufl.as_vector(list(args)),
        "ImmutableDenseMatrix": lambda data: ufl.as_tensor(data),
    }


class UFLRuntimeSymbolic(UFLRuntimeModel, NumpyRuntimeSymbolic):
    """UFL runtime for symbolic registrars (e.g. :class:`Numerics`).

    The Firedrake counterpart to :class:`NumpyRuntimeSymbolic`.  It
    walks ``symbolic_obj.functions`` and lambdifies each registered
    symbolic function (``numerical_flux``, ``numerical_fluctuations``,
    ``local_max_abs_eigenvalue``, …) through the UFL module dict, so
    calling, e.g., ``runtime.numerical_flux(qL, qR, auxL, auxR, p, n)``
    returns a UFL expression suitable for direct assembly in a
    Firedrake form.

    The MRO is ``UFLRuntimeSymbolic → UFLRuntimeModel →
    NumpyRuntimeSymbolic → NumpyRuntimeModel``; ``__init__`` is
    inherited from ``NumpyRuntimeSymbolic`` (no other class defines
    one), and ``module`` / ``printer`` /
    ``_vectorize_expression`` / ``_lambdify_function`` come from
    ``UFLRuntimeModel``.
    """
    pass
