"""Module `zoomy_core.transformation.to_js`.

SymPy → JavaScript code printer for generating Model2D-compatible
functions from any Zoomy symbolic model.

Subclasses GenericCppBase to reuse the Zoomy symbol-mapping,
CSE pipeline, and convert_expression_body() machinery.
Only the language-specific output formatting is overridden.

Usage::

    from zoomy_core.model.models.shallow_water import ShallowWaterEquations
    from zoomy_core.transformation.to_js import JsModel

    model = ShallowWaterEquations(dimension=2)
    printer = JsModel(model)
    print(printer.generate())
"""

import sympy as sp

from zoomy_core.transformation.generic_c import GenericCppBase, get_nested_shape


# =========================================================================
#  1. JS BASE PRINTER
# =========================================================================


class GenericJsBase(GenericCppBase):
    """JavaScript code printer extending GenericCppBase.

    Inherits symbol-map registration, CSE, convert_expression_body(),
    _expand_vector_conditionals(), _optimize_array_calls(), etc.
    Overrides only the output formatting to emit valid JavaScript.
    """

    math_namespace = "Math."
    real_type = "const"
    gpu_enabled = False

    c_functions = {
        "conditional": lambda p, c, t, f: (
            f"(({p.doprint(c)}) ? ({p.doprint(t)}) : ({p.doprint(f)}))"
        ),
        "clamp_positive": lambda p, x: f"Math.max({p.doprint(x)}, 0)",
        "clamp_momentum": lambda p, hu, h, u_max: (
            f"Math.max(Math.min({p.doprint(hu)}, "
            f"{p.doprint(h)} * {p.doprint(u_max)}), "
            f"-{p.doprint(h)} * {p.doprint(u_max)})"
        ),
        "Min": lambda p, *args: p._print_min_max("min", args),
        "Max": lambda p, *args: p._print_min_max("max", args),
        "Abs": lambda p, a: f"Math.abs({p.doprint(a)})",
    }

    # ── Array / type formatting ──────────────────────

    def get_array_type(self, shape):
        """Not needed for JS (no static type), but keep for API compat."""
        total = 1
        for s in shape:
            total *= s
        return f"Float64Array({total})"

    def get_array_declaration(self, target_name, shape, init_zero=False):
        """Declare a local result array."""
        total = 1
        for s in shape:
            total *= s
        return f"const {target_name} = new Float64Array({total});"

    def format_array_initialization(self, sym_name, elements):
        """Initialize a local array from CSE-optimized values."""
        init_str = ", ".join([self.doprint(e) for e in elements])
        return f"const {sym_name} = [{init_str}];"

    # ── Function wrapping ────────────────────────────

    def wrap_function_signature(self, name, args_str, body_str, shape):
        """Wrap a function body in a JS function declaration."""
        return f"function {name}({args_str}) {{\n{body_str}\n}}"

    def _generate_js_args(self, func_obj):
        """Generate JS function parameter list from a Function object."""
        parts = []
        for key, val in func_obj.args.items():
            js_name = self.ARG_MAPPING.get(key, key)
            parts.append(js_name)
        return ", ".join(parts)

    # ── Pow printing (JS uses ** or Math.pow) ────────

    def _print_Pow(self, expr):
        """Print power expressions using Math.pow / Math.sqrt / inline multiply."""
        base, exp = expr.as_base_exp()
        b = self._print(base)
        if exp == sp.Rational(1, 2) or exp == 0.5:
            return f"Math.sqrt({b})"
        if exp == sp.Rational(-1, 2):
            return f"(1.0 / Math.sqrt({b}))"
        if exp.is_Integer:
            n = int(exp)
            if n == 0:
                return "1.0"
            if n == 1:
                return b
            if n == 2:
                return f"(({b}) * ({b}))"
            if n == -1:
                return f"(1.0 / ({b}))"
            if n < 0:
                return f"(1.0 / Math.pow({b}, {abs(n)}))"
            return f"Math.pow({b}, {n})"
        return f"Math.pow({b}, {self._print(exp)})"

    # ── No includes / struct wrappers for JS ─────────

    def get_includes(self):
        """No includes needed for JS."""
        return ""

    def get_simple_array_def(self):
        """No SimpleArray struct needed for JS."""
        return ""


# =========================================================================
#  2. JS MODEL GENERATOR
# =========================================================================


class JsModel(GenericJsBase):
    """Generate JavaScript Model2D functions from a Zoomy symbolic model.

    Produces standalone JS functions for ``flux``, ``source``,
    ``eigenvalues``, ``source_jacobian_wrt_variables``, etc.
    Each function takes flat arrays (Q, Qaux, p, n) and returns
    a Float64Array result.
    """

    # Functions to generate (in order). Skipped if definition is None/zero.
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
        self.register_map("p", model.parameters.values())

    def generate(self):
        """Generate all JS functions as a single code string.

        Returns
        -------
        str
            JavaScript code containing function declarations for each
            registered model function that has a non-trivial definition.
        """
        blocks = []
        blocks.append(self._generate_metadata())

        func_dict = self.model.functions.as_dict() if hasattr(self.model.functions, 'as_dict') else dict(self.model.functions.items())
        for name in self.KERNEL_NAMES:
            if name not in func_dict:
                continue
            func_obj = func_dict[name]
            if func_obj.definition is None:
                continue
            # Skip trivially-zero definitions
            defn = func_obj.definition
            if hasattr(defn, "is_zero_matrix") and defn.is_zero_matrix:
                continue
            blocks.append(self._generate_kernel(name, func_obj))

        return "\n\n".join(blocks)

    def generate_model2d_object(self):
        """Generate a JS object literal implementing the Model2D interface.

        The object wraps the generated functions and adds metadata
        (nVars, parameter defaults).
        """
        funcs = self.generate()
        param_values = [
            float(v) for v in self.model.parameters.values()
        ]
        param_names = list(self.model.parameters.keys())

        obj = f"""// Auto-generated Model2D from {self.model.__class__.__name__}
// nVars={self.n_dof_q}, nAux={self.n_dof_qaux}, nParams={self.n_parameters}
// Parameters: {dict(zip(param_names, param_values))}

{funcs}

const generatedModel = {{
  nVars: {self.n_dof_q},
  nAux: {self.n_dof_qaux},
  parameterDefaults: [{', '.join(str(v) for v in param_values)}],
  parameterNames: {param_names},
  flux: flux,
  source: typeof source !== 'undefined' ? source : null,
  eigenvalues: typeof eigenvalues !== 'undefined' ? eigenvalues : null,
  sourceJacobian: typeof source_jacobian_wrt_variables !== 'undefined' ? source_jacobian_wrt_variables : null,
  quasilinearMatrix: typeof quasilinear_matrix !== 'undefined' ? quasilinear_matrix : null,
}};
"""
        return obj

    # ── Internal ─────────────────────────────────────

    def _generate_metadata(self):
        """Generate a JS comment block with model metadata."""
        param_names = list(self.model.parameters.keys())
        param_vals = [float(v) for v in self.model.parameters.values()]
        lines = [
            f"// Generated JS model: {self.model.__class__.__name__}",
            f"// Dimension: {self.model.dimension}",
            f"// Variables ({self.n_dof_q}): {list(self.model.variables.keys())}",
            f"// Aux variables ({self.n_dof_qaux}): {list(self.model.aux_variables.keys())}",
            f"// Parameters ({self.n_parameters}): {dict(zip(param_names, param_vals))}",
        ]
        return "\n".join(lines)

    def _generate_kernel(self, name, func_obj):
        """Generate one JS function from a Function object."""
        expr = func_obj.definition
        expr = self._expand_vector_conditionals(expr)
        if isinstance(expr, (list, tuple)):
            expr = self._flatten_ragged_list(expr)
            expr = sp.Array(expr)
        shape, expr = get_nested_shape(expr)
        body = self.convert_expression_body(expr, shape)
        args_str = self._generate_js_args(func_obj)
        return self.wrap_function_signature(name, args_str, body, shape)
