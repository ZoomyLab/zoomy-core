import sympy
import sympy as sp
import param
from sympy import lambdify, Tuple
from zoomy_core.misc.misc import Zstruct, ZArray

# --- Helper Functions ---


def listify(expr):
    if isinstance(expr, ZArray) or isinstance(expr, sp.NDimArray):
        return Tuple(*expr.tolist())  # convert list to sympy Tuple
    elif hasattr(expr, "args") and expr.args:
        return expr.func(*[listify(a) for a in expr.args])
    else:
        return expr


def substitute_expression(expr, subs_map):
    """
    Generic substitution for scalars, lists, matrices, and ZArrays.
    Ensures that ZArray inputs return ZArray outputs where possible.
    """
    # 1. Handle Lists (recursive)
    if isinstance(expr, list):
        return [substitute_expression(e, subs_map) for e in expr]

    # 2. Handle Tuples (recursive)
    if isinstance(expr, tuple):
        return tuple(substitute_expression(e, subs_map) for e in expr)

    # 3. Handle SymPy Objects (ZArray, Matrix, Expr)
    if hasattr(expr, "subs"):
        result = expr.subs(subs_map)

        # If input was ZArray (mutable) but output is ImmutableDenseNDimArray, convert back
        if isinstance(expr, ZArray) and not isinstance(result, ZArray):
            try:
                # ZArray is MutableDenseNDimArray. Construct from list and shape.
                if hasattr(result, "tolist") and hasattr(result, "shape"):
                    return ZArray(result.tolist(), result.shape)
                return ZArray(result)
            except Exception:
                pass
        return result

    return expr


def vectorize_constant_sympy_expressions(expr, Q, Qaux, vectorize=True):
    """
    Replace entries in `expr` that are constant w.r.t. Q and Qaux
    by entry * ones_like(Q[0]) so NumPy/JAX vectorization works.
    Handles scalars, lists, sympy.Matrix, sympy.Array, and sympy.Piecewise.
    """
    if not vectorize:
        return expr
    symbol_list = set(Q.get_list() + Qaux.get_list())
    q0 = Q[0]
    ones_like = sp.Function("ones_like")  # symbolic placeholder
    zeros_like = sp.Function("zeros_like")  # symbolic placeholder

    # convert matrices to nested lists (Array handles lists better)
    if isinstance(
        expr, (sp.MatrixBase, sp.ImmutableDenseMatrix, sp.MutableDenseMatrix)
    ):
        expr = expr.tolist()

    def vectorize_entry(entry):
        """Return entry multiplied by ones_like(q0) if it is constant."""
        # numeric zero
        if entry == 0:
            return zeros_like(q0)

        # numeric constant (int, float, Rational, pi, etc.)
        if getattr(entry, "is_number", False):
            return entry * ones_like(q0)

        # symbolic constant independent of Q and Qaux
        if hasattr(entry, "free_symbols") and entry.free_symbols.isdisjoint(
            symbol_list
        ):
            return entry * ones_like(q0)

        # otherwise, depends on variables
        return entry

    def recurse(e):
        """Recursively handle Array, Matrix, Piecewise, list, or scalar."""
        # Handle lists (possibly nested)
        if isinstance(e, list):
            return [recurse(sub) for sub in e]

        # Handle Matrices
        if isinstance(e, sp.MatrixBase):
            return sp.Matrix([[recurse(sub) for sub in row] for row in e.tolist()])

        # Handle Arrays (any rank)
        if isinstance(e, sp.Array) or isinstance(e, ZArray):
            return sp.Array([recurse(sub) for sub in e])

        # Handle Piecewise
        if isinstance(e, sp.Piecewise):
            # Recursively process all (expr, cond) pairs
            new_args = []
            for expr_i, cond_i in e.args:
                new_expr = recurse(expr_i)
                new_args.append((new_expr, cond_i))
            return sp.Piecewise(*new_args)

        # Scalar or atomic expression
        return vectorize_entry(e)

    # Recurse and then normalize into an N-dimensional array (if possible)
    result = recurse(expr)

    # Convert to Array if possible (this ensures uniform output type)
    if isinstance(result, list):
        try:
            return sp.Array(result)
        except Exception:
            # fall back if shapes inconsistent
            return result
    return result


# --- Refactored Class ---


class Function(param.Parameterized):
    """
    Generic (virtual) function implementation.
    Wraps symbolic definitions and handles substitution/lambdification.
    """

    name = param.String(default="Function")

    # We use default=None for mutable objects to avoid shared state bugs.
    # We will initialize them in __init__.
    args = param.ClassSelector(class_=Zstruct, default=None)
    definition = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)

        # Safe initialization of mutables
        if self.args is None:
            self.args = Zstruct()

        if self.definition is None:
            self.definition = sympy.zeros(1, 1)

    def __call__(self):
        """Allow calling the instance to get its symbolic definition."""
        return self.definition

    def subs(self, subs_map):
        """
        Perform substitution on the function definition using the new helper.
        Returns the substituted expression (does NOT modify self).
        """
        return substitute_expression(self.definition, subs_map)

    def lambdify(self, modules=None):
        """Return a lambdified version of the function."""

        # Create a symbolic placeholder to translate the output in the correct format
        make_array = sp.Function("array")

        func = lambdify(
            self.args.get_list(),
            make_array(
                listify(
                    vectorize_constant_sympy_expressions(
                        self.definition,
                        self.args.variables,
                        self.args.aux_variables,
                        vectorize=True,
                    )
                )
            ),
            modules=modules,
        )
        return func
