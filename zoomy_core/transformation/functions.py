import sympy
from zoomy_core.misc.misc import ZArray


def conditional(condition, true_val, false_val):
    """
    A Vector-Aware Conditional function (similar to UFL).

    - If inputs are Scalars: Returns conditional(c, t, f)
    - If inputs are Vectors (ZArray/Matrix): Returns ZArray([conditional(c, t[i], f[i]), ...])

    This ensures the return type is always compatible with Zoomy's ZArray requirements.
    """

    # Helper to check if input is array-like (ZArray, Matrix, list, tuple)
    # Exclude SymPy Symbols which might technically be iterable in some contexts
    def is_vec(x):
        return hasattr(x, "__getitem__") and not isinstance(
            x, (sympy.Symbol, sympy.Function)
        )

    is_array_t = is_vec(true_val)
    is_array_f = is_vec(false_val)

    if is_array_t or is_array_f:
        # Flatten inputs to lists for safe iteration
        # This handles mixed types (e.g. ZArray vs Matrix) if dimensions match
        t_flat = list(sympy.flatten(true_val)) if is_array_t else [true_val]
        f_flat = list(sympy.flatten(false_val)) if is_array_f else [false_val]

        # Determine target shape from the vector input
        if hasattr(true_val, "shape"):
            shape = true_val.shape
        elif hasattr(false_val, "shape"):
            shape = false_val.shape
        else:
            shape = (len(t_flat),)

        # Broadcast/Thread the conditional logic element-wise
        result_list = []
        import itertools

        # zip_longest allows broadcasting scalar 0 to vector length if needed
        for t, f in itertools.zip_longest(t_flat, f_flat, fillvalue=0):
            result_list.append(sympy.Function("conditional")(condition, t, f))

        # RETURN A ZARRAY: This satisfies basefunction.py
        return ZArray(result_list).reshape(*shape)

    else:
        # Scalar case: Return the raw symbolic function
        return sympy.Function("conditional")(condition, true_val, false_val)


# Add other unified functions here if needed (e.g., Min, Max wrappers)
