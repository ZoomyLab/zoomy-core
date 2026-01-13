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
    Simplified substitution for ZArray contents (nested lists).
    Assumes input 'expr' is a list (nested) or a scalar sympy expression.
    """
    # 1. Handle Lists (recursive)
    if isinstance(expr, list):
        return [substitute_expression(e, subs_map) for e in expr]

    # 2. Handle Scalar SymPy Expressions
    if hasattr(expr, "subs"):
        return expr.subs(subs_map)

    # 3. Fallback (literals)
    return expr


def vectorize_constant_sympy_expressions(expr, Q, Qaux, vectorize=True):
    """
    Replace entries in `expr` that are constant w.r.t. Q and Qaux
    by entry * ones_like(Q[0]) so NumPy/JAX vectorization works.
    """
    if not vectorize:
        return expr
    symbol_list = set(Q.get_list() + Qaux.get_list())
    q0 = Q[0]
    ones_like = sp.Function("ones_like")
    zeros_like = sp.Function("zeros_like")

    if isinstance(
        expr, (sp.MatrixBase, sp.ImmutableDenseMatrix, sp.MutableDenseMatrix)
    ):
        expr = expr.tolist()

    def vectorize_entry(entry):
        if entry == 0:
            return zeros_like(q0)
        if getattr(entry, "is_number", False):
            return entry * ones_like(q0)
        if hasattr(entry, "free_symbols") and entry.free_symbols.isdisjoint(
            symbol_list
        ):
            return entry * ones_like(q0)
        return entry

    def recurse(e):
        if isinstance(e, list):
            return [recurse(sub) for sub in e]
        if isinstance(e, sp.MatrixBase):
            return sp.Matrix([[recurse(sub) for sub in row] for row in e.tolist()])
        if isinstance(e, sp.Array) or isinstance(e, ZArray):
            return sp.Array([recurse(sub) for sub in e])
        if isinstance(e, sp.Piecewise):
            new_args = []
            for expr_i, cond_i in e.args:
                new_args.append((recurse(expr_i), cond_i))
            return sp.Piecewise(*new_args)
        return vectorize_entry(e)

    result = recurse(expr)
    if isinstance(result, list):
        try:
            return sp.Array(result)
        except Exception:
            return result
    return result


class Function(param.Parameterized):
    """
    Generic (virtual) function implementation.
    Wraps symbolic definitions and handles substitution/lambdification.
    """

    name = param.String(default="Function")
    args = param.ClassSelector(class_=Zstruct, default=None)
    definition = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        if self.args is None:
            self.args = Zstruct()
        # Default definition is a 1x1 ZArray
        if self.definition is None:
            self.definition = ZArray([0])

    def __call__(self):
        return self.definition
    
    def eval_symbolic(self, *input_args):
        """
        Evaluate the function symbolically by substituting the provided arguments.
        Arguments are mapped positionally to the fields in self.args (insertion order).

        Example:
            If self.args = Zstruct(variables=..., aux_variables=..., normal=...)
            Call: eval_symbolic(Q, Qaux, n)
        
        Args:
            *input_args: Variable length argument list. 
                         Must match the order and shape of self.args values.
        """
        # 1. Type Check: Enforce ZArray definition
        if not isinstance(self.definition, ZArray):
            raise TypeError(
                f"Function definition must be a ZArray, got {type(self.definition)}. "
                "Please ensure your model returns ZArray objects."
            )

        # 2. Build Substitution Map
        subs_map = {}
        
        # Retrieve defined symbolic arguments (order matters!)
        # Zstruct.keys()/values() return lists based on insertion order (Python 3.7+)
        defined_names = self.args.keys()
        defined_symbols = self.args.values()
        
        if len(input_args) > len(defined_symbols):
            raise ValueError(
                f"Too many arguments provided. Function expects at most {len(defined_symbols)} "
                f"({list(defined_names)}), but got {len(input_args)}."
            )

        # Zip defined arguments with provided inputs
        for i, (name, sym_obj, input_val) in enumerate(zip(defined_names, defined_symbols, input_args)):
            
            # --- A. Flatten Symbolic Definition ---
            if hasattr(sym_obj, "values") and callable(sym_obj.values):
                # Handle Zstruct of symbols (e.g. parameters)
                sym_list = sym_obj.values()
            elif hasattr(sym_obj, "__iter__") and not isinstance(sym_obj, (str, bytes)):
                # Handle Lists/ZArrays/Tuples
                sym_list = sym_obj
            else:
                # Handle Scalars
                sym_list = [sym_obj]

            # --- B. Flatten Input Value ---
            if hasattr(input_val, "values") and callable(input_val.values):
                # Handle Zstruct of values
                val_list = input_val.values()
            elif hasattr(input_val, "__iter__") and not isinstance(input_val, (str, bytes)):
                 # Handle Lists/ZArrays/Tuples
                val_list = input_val
            else:
                # Handle Scalars
                val_list = [input_val]

            # --- C. Validation & Mapping ---
            if len(sym_list) != len(val_list):
                raise ValueError(
                    f"Shape mismatch for argument '{name}' (position {i}). "
                    f"Function expects {len(sym_list)} items, got {len(val_list)}."
                )

            for s, v in zip(sym_list, val_list):
                subs_map[s] = v

        # 3. Perform Substitution on the underlying list structure
        #    (ZArray.tolist() returns nested lists, which is safe for recursion)
        def_as_list = self.definition.tolist()
        subbed_list = substitute_expression(def_as_list, subs_map)

        # 4. Return as ZArray
        return ZArray(subbed_list)

    def lambdify(self, modules=None):
        """Return a lambdified version of the function."""
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
