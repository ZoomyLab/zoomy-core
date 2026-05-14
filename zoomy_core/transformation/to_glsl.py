"""Module `zoomy_core.transformation.to_glsl`.

SymPy → GLSL ES 3.00 (WebGL2) code printer for generating kernels from
any Zoomy symbolic model.

Subclasses :class:`GenericCppBase` to reuse the Zoomy symbol-mapping,
CSE pipeline and expression-printing machinery; only the
language-specific output formatting is overridden.

Two GLSL ES 3.00 facts shape the output:

* **No array return types.**  Generated kernels return ``void`` and
  write their result through an ``out float res[N]`` parameter.  The
  result is laid out row-major in the natural shape of the symbolic
  definition (e.g. ``flux`` is ``(n_variables, dimension)`` →
  ``res[var * dimension + dim]``).
* **No implicit int→float conversion.**  Every numeric literal is
  emitted in float form (``0`` → ``0.0``).  The one exception is
  comparisons between integer quantities (e.g. a boundary-condition
  index), which are detected and printed with plain int literals.

Usage::

    from zoomy_core.transformation.to_glsl import GlslModel
    print(GlslModel(model).generate())
"""

import itertools
import textwrap

import sympy as sp

from zoomy_core.transformation.generic_c import (
    GenericCppBase,
    flatten_index,
    get_nested_shape,
)


# =========================================================================
#  1. GLSL BASE PRINTER
# =========================================================================


class GenericGlslBase(GenericCppBase):
    """GLSL ES 3.00 code printer extending :class:`GenericCppBase`.

    Inherits symbol-map registration, CSE, ``_expand_vector_conditionals``,
    ``_optimize_array_calls`` etc.  Overrides the output formatting to
    emit valid GLSL ES 3.00 with ``out``-parameter result passing.
    """

    math_namespace = ""
    real_type = "float"
    gpu_enabled = False

    # Symbols compared inside relational expressions print as plain ints
    # while this flag is set (see ``_print_Relational``).
    _int_literal_ctx = False

    c_functions = {
        "conditional": lambda p, c, t, f: (
            f"(({p.doprint(c)}) ? ({p.doprint(t)}) : ({p.doprint(f)}))"
        ),
        "clamp_positive": lambda p, x: f"max({p.doprint(x)}, 0.0)",
        "clamp_momentum": lambda p, hu, h, u_max: (
            f"clamp({p.doprint(hu)}, "
            f"-({p.doprint(h)}) * ({p.doprint(u_max)}), "
            f"({p.doprint(h)}) * ({p.doprint(u_max)}))"
        ),
        "Min": lambda p, *args: p._print_min_max("min", args),
        "Max": lambda p, *args: p._print_min_max("max", args),
        "Abs": lambda p, a: f"abs({p.doprint(a)})",
        "max_wavespeed": lambda p, *args: p._print_nested_max(
            [f"abs({p.doprint(a)})" for a in args]
        ),
    }

    # ── Numeric literals (GLSL has no implicit int→float) ────────────

    def _print_Integer(self, expr):
        """Internal helper `_print_Integer`."""
        if self._int_literal_ctx:
            return str(int(expr))
        return f"{int(expr)}.0"

    def _print_Rational(self, expr):
        """Internal helper `_print_Rational`."""
        return f"({expr.p}.0 / {expr.q}.0)"

    def _print_Float(self, expr):
        """Internal helper `_print_Float`."""
        s = super()._print_Float(expr)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s

    def _print_Relational(self, expr):
        """Print a comparison, choosing int vs float literal context."""
        int_ctx = bool(getattr(expr.lhs, "is_integer", False)) and bool(
            getattr(expr.rhs, "is_integer", False)
        )
        old = self._int_literal_ctx
        self._int_literal_ctx = int_ctx
        try:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
        finally:
            self._int_literal_ctx = old
        return f"({lhs} {expr.rel_op} {rhs})"

    # ── Pow (GLSL ``pow`` is undefined for negative base) ────────────

    def _print_Pow(self, expr):
        """Print powers — inline multiplication for integer exponents."""
        base, exp = expr.as_base_exp()
        b = self._print(base)
        if exp == sp.Rational(1, 2) or exp == 0.5:
            return f"sqrt({b})"
        if exp == sp.Rational(-1, 2):
            return f"inversesqrt({b})"
        if exp.is_Integer:
            n = int(exp)
            if n == 0:
                return "1.0"
            if n == 1:
                return f"({b})"
            if n > 0:
                return "(" + " * ".join([f"({b})"] * n) + ")"
            denom = " * ".join([f"({b})"] * (-n))
            return f"(1.0 / ({denom}))"
        return f"pow({b}, {self._print(exp)})"

    # ── Array / type formatting ──────────────────────────────────────

    def get_array_type(self, shape):
        """GLSL array type string, e.g. ``float[6]``."""
        total = 1
        for s in shape:
            total *= s
        return f"float[{total}]"

    def get_array_declaration(self, target_name, shape, init_zero=False):
        """Declare a local GLSL array."""
        total = 1
        for s in shape:
            total *= s
        if init_zero:
            zeros = ", ".join(["0.0"] * total)
            return f"float {target_name}[{total}] = float[{total}]({zeros});"
        return f"float {target_name}[{total}];"

    def format_array_initialization(self, sym_name, elements):
        """Initialize a local GLSL array from CSE-optimized values."""
        n = len(elements)
        init = ", ".join(self.doprint(e) for e in elements)
        return f"float {sym_name}[{n}] = float[{n}]({init});"

    def format_accessor(self, var_name, index):
        """Format accessor."""
        return f"{var_name}[{index}]"

    def format_assignment(self, target_name, indices, value, shape):
        """Format assignment."""
        idx = flatten_index(indices, shape)
        return f"{target_name}[{idx}] = {value};"

    # ── Body generation (out-parameter, no return) ───────────────────

    def convert_expression_body(self, expr, shape, target="res"):
        """Convert an expression into a GLSL body that writes ``target``.

        ``target`` is the function's ``out float target[N]`` parameter —
        the body assigns into it directly; there is no local declaration
        and no ``return``.
        """
        if isinstance(expr, sp.Piecewise):
            return self._print_piecewise_structure(expr, shape, target)

        flat_expr = (
            list(sp.flatten(expr))
            if hasattr(expr, "__iter__") and not isinstance(expr, sp.Matrix)
            else list(expr)
            if isinstance(expr, sp.Matrix)
            else [expr]
        )
        call_defs, optim_exprs = self._optimize_array_calls(flat_expr)
        if call_defs:
            raise NotImplementedError(
                "to_glsl: nested array-valued function calls are not supported"
            )

        total = 1
        for s in shape:
            total *= s
        lines = []
        if optim_exprs:
            temps, simplified = sp.cse(optim_exprs, symbols=sp.numbered_symbols("t"))
            for lhs, rhs in temps:
                lines.append(f"float {self.doprint(lhs)} = {self.doprint(rhs)};")
            result_array = sp.Array(simplified).reshape(*shape)
            for indices in itertools.product(*[range(s) for s in shape]):
                val = self.doprint(result_array[indices])
                lines.append(self.format_assignment(target, indices, val, shape))
        else:
            for i in range(total):
                lines.append(f"{target}[{i}] = 0.0;")
        return "\n".join("    " + ln for ln in lines)

    def _print_piecewise_structure(self, expr, shape, target):
        """Internal helper `_print_piecewise_structure`.

        Emits an if / else-if chain assigning into ``target``.  ``target``
        is zero-initialised first so any fall-through (a Piecewise with no
        final ``True`` branch, e.g. a boundary-condition dispatch) is
        well-defined.
        """
        total = 1
        for s in shape:
            total *= s
        lines = [f"    {target}[{i}] = 0.0;" for i in range(total)]
        for i, arg in enumerate(expr.args):
            val, cond = (
                (arg.expr, arg.cond) if hasattr(arg, "expr") else (arg[0], arg[1])
            )
            if cond is True or cond == sp.true:
                if i == 0:
                    return self.convert_expression_body(val, shape, target)
                lines.append("    } else {")
            elif i == 0:
                lines.append(f"    if ({self.doprint(cond)}) {{")
            else:
                lines.append(f"    }} else if ({self.doprint(cond)}) {{")
            lines.append(
                textwrap.indent(
                    self.convert_expression_body(val, shape, target), "    "
                )
            )
        lines.append("    }")
        return "\n".join(lines)

    # ── Function wrapping ────────────────────────────────────────────

    def wrap_function_signature(self, name, args_str, body_str, shape):
        """Wrap a GLSL body in a ``void`` function declaration."""
        return f"void {name}({args_str}) {{\n{body_str}\n}}"

    @staticmethod
    def _is_array_arg(obj):
        """True if a Function argument is a vector (vs a scalar symbol)."""
        if hasattr(obj, "get_list") or hasattr(obj, "_symbolic_name"):
            return True
        if hasattr(obj, "values") and callable(obj.values):
            return True
        return isinstance(obj, (list, tuple, sp.NDimArray, sp.MatrixBase))

    @staticmethod
    def _arg_length(obj):
        """Number of components of a vector Function argument."""
        if hasattr(obj, "length"):
            return obj.length()
        if hasattr(obj, "get_list"):
            return len(obj.get_list())
        try:
            return len(obj)
        except TypeError:
            return 1

    def _generate_glsl_args(self, func_obj, shape):
        """GLSL parameter list for a Function object, with the trailing
        ``out float res[N]`` result parameter appended.

        Zero-length array arguments (e.g. ``Qaux`` for a model with no
        auxiliary variables) are omitted — GLSL ES 3.00 forbids
        zero-sized arrays, and such a model has no symbols referencing
        the argument anyway.
        """
        parts = []
        for key, obj in func_obj.args.items():
            name = self.ARG_MAPPING.get(key, key)
            if key == "idx":
                # Boundary-condition dispatch index — integer scalar.
                parts.append("int bc_idx")
            elif self._is_array_arg(obj):
                length = self._arg_length(obj)
                if length == 0:
                    continue
                parts.append(f"float {name}[{length}]")
            else:
                parts.append(f"float {name}")
        total = 1
        for s in shape:
            total *= s
        parts.append(f"out float res[{total}]")
        return ", ".join(parts)

    # ── No includes / struct wrappers for GLSL ───────────────────────

    def get_includes(self):
        """No includes needed for GLSL."""
        return ""

    def get_simple_array_def(self):
        """No SimpleArray struct needed for GLSL."""
        return ""


# =========================================================================
#  2. GLSL MODEL GENERATOR
# =========================================================================


class GlslModel(GenericGlslBase):
    """Generate GLSL ES 3.00 kernels from a Zoomy symbolic model.

    Produces standalone ``void`` functions for ``flux``, ``source``,
    ``eigenvalues``, etc.  Each takes the model's flat ``float[]``
    argument arrays plus an ``out float res[N]`` result parameter.
    Mirrors :class:`zoomy_core.transformation.to_js.JsModel`.
    """

    # Functions to generate (in order).  Skipped if the definition is
    # missing or trivially zero.
    KERNEL_NAMES = [
        "flux",
        "source",
        "eigenvalues",
        "source_jacobian_wrt_variables",
        "quasilinear_matrix",
    ]

    def __init__(self, model):
        """Initialize with a Zoomy Model instance."""
        super().__init__()
        self.model = model
        self.n_dof_q = model.n_variables
        self.n_dof_qaux = model.n_aux_variables
        self.n_parameters = model.n_parameters

        self.register_map("Q", model.variables.values())
        self.register_map("Qaux", model.aux_variables.values())
        self.register_map("n", model.normal.values())
        if hasattr(model, "position"):
            self.register_map("X", model.position.values())
        # Map the *symbolic* parameters (``_parameter_symbols``); the
        # public ``model.parameters`` Zstruct holds numeric values.
        self.register_map("p", model._parameter_symbols.values())

    def generate(self):
        """Generate all GLSL kernels as a single code string."""
        blocks = [self._generate_metadata()]

        func_dict = (
            self.model.functions.as_dict()
            if hasattr(self.model.functions, "as_dict")
            else dict(self.model.functions.items())
        )
        for name in self.KERNEL_NAMES:
            if name not in func_dict:
                continue
            func_obj = func_dict[name]
            defn = func_obj.definition
            if defn is None:
                continue
            if hasattr(defn, "is_zero_matrix") and defn.is_zero_matrix:
                continue
            blocks.append(self._generate_kernel(name, func_obj))

        return "\n\n".join(blocks)

    # ── Internal ─────────────────────────────────────────────────────

    def _generate_metadata(self):
        """Generate a GLSL comment block with model metadata."""
        param_names = list(self.model.parameters.keys())
        param_vals = [float(v) for v in self.model.parameters.values()]
        lines = [
            f"// Generated GLSL ES 3.00 model: {self.model.__class__.__name__}",
            f"// Dimension: {self.model.dimension}",
            f"// Variables ({self.n_dof_q}): {list(self.model.variables.keys())}",
            f"// Aux variables ({self.n_dof_qaux}): "
            f"{list(self.model.aux_variables.keys())}",
            f"// Parameters ({self.n_parameters}): "
            f"{dict(zip(param_names, param_vals))}",
        ]
        return "\n".join(lines)

    def _generate_kernel(self, name, func_obj):
        """Generate one GLSL function from a Function object."""
        expr = func_obj.definition
        expr = self._expand_vector_conditionals(expr)
        if isinstance(expr, (list, tuple)):
            expr = self._flatten_ragged_list(expr)
            expr = sp.Array(expr)
        shape, expr = get_nested_shape(expr)
        body = self.convert_expression_body(expr, shape)
        args_str = self._generate_glsl_args(func_obj, shape)
        return self.wrap_function_signature(name, args_str, body, shape)
