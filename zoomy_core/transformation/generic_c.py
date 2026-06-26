"""Module `zoomy_core.transformation.generic_c`."""

import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter
import re
import os
import itertools
import textwrap
from zoomy_core.misc import misc as misc
from zoomy_core.misc.misc import Zstruct, ZArray

# =========================================================================
#  1. HELPER FUNCTIONS
# =========================================================================


def flatten_index(indices, shape):
    """Flatten index."""
    flat_idx = 0
    stride = 1
    for i, size in zip(reversed(indices), reversed(shape)):
        flat_idx += i * stride
        stride *= size
    return flat_idx


def get_nested_shape(expr):
    """Get nested shape."""
    shape = (1,)
    if hasattr(expr, "shape"):
        shape = expr.shape
    elif hasattr(expr, "tomatrix"):
        shape = expr.shape
    if hasattr(expr, "definition"):
        expr = expr.definition

    if shape == (1,) and isinstance(expr, sp.Piecewise):
        try:
            first_arg = expr.args[0]
            val = first_arg.expr if hasattr(first_arg, "expr") else first_arg[0]
            if hasattr(val, "shape"):
                shape = val.shape
            elif isinstance(val, (list, tuple)):
                shape = (len(val),)
        except Exception:
            pass
    return shape, expr


# =========================================================================
#  2. GENERIC BASE
# =========================================================================


class GenericCppBase(CXX11CodePrinter):
    """GenericCppBase. (class)."""
    _output_subdir = "cpp_interface"
    _wrapper_name = "BaseWrapper"
    _is_template_class = False
    gpu_enabled = True
    real_type = "double"
    math_namespace = "std::"

    ARG_MAPPING = {
        "variables": "Q",
        "aux_variables": "Qaux",
        "gradient_variables": "gradQ",
        "p": "p",
        "normal": "n",
        "position": "X",
        "time": "time",
        "distance": "dX",
        "q_minus": "Q_minus",
        "q_plus": "Q_plus",
        "aux_minus": "Qaux_minus",
        "aux_plus": "Qaux_plus",
        "flux_minus": "flux_minus",
        "flux_plus": "flux_plus",
        "source": "source_term",
        "dt": "dt",
        "dx": "dx",
    }

    c_functions = {
        "conditional": lambda p,
        c,
        t,
        f: f"(({p.doprint(c)}) ? ({p.doprint(t)}) : ({p.doprint(f)}))",
        "Min": lambda p, *args: p._print_min_max("min", args),
        "Max": lambda p, *args: p._print_min_max("max", args),
        "Abs": lambda p, a: f"{p.math_namespace}abs({p.doprint(a)})",
        "clamp_positive": lambda p, x: f"{p.math_namespace}max({p.doprint(x)}, ({p.real_type})0.0)",
        "clamp_momentum": lambda p, hu, h, u_max: (
            f"{p.math_namespace}min({p.math_namespace}max({p.doprint(hu)}, "
            f"-{p.doprint(h)}*{p.doprint(u_max)}), "
            f"{p.doprint(h)}*{p.doprint(u_max)})"
        ),
        "max_wavespeed": lambda p, *args: p._print_nested_max(
            [f"{p.math_namespace}abs({p.doprint(a)})" for a in args]
        ),
        # Backend-provided free functions (the C++ printer only emits the
        # call; the backend supplies the implementation, exactly like
        # OpenFOAM's ``UserFunctions.H`` — see FoamNumericsPrinter
        # (``numerics::eigensystem``) and FoamUpdateAuxPrinter
        # (``numerics::compute_derivative``) in ``to_openfoam.py``).
        #
        #   compute_derivative(field_id, dx, dy, dz)
        #       -> the (dx,dy,dz)-order spatial derivative of the field
        #          identified by ``field_id``, evaluated on the mesh.
        #   eigensystem(idx, A...)
        #       -> the ``idx``-th eigen-quantity of the flattened matrix
        #          ``A`` (the opaque eigendecomposition a Roe scheme builds
        #          ``|A| = R|Lambda|L`` from).
        "compute_derivative": lambda p, *args: (
            "compute_derivative(" + ", ".join(p.doprint(a) for a in args) + ")"
        ),
        "eigensystem": lambda p, *args: (
            "eigensystem(" + ", ".join(p.doprint(a) for a in args) + ")"
        ),
    }

    def __init__(self, *args, **kwargs):
        """Initialize the instance."""
        super().__init__(*args, **kwargs)
        self.symbol_maps = []
        self._std_regex = re.compile(r"std::([A-Za-z_]\w*)")

    def register_map(self, name, keys):
        """Register map."""
        new_map = {k: self.format_accessor(name, i) for i, k in enumerate(keys)}
        self.symbol_maps.append(new_map)
        return new_map

    def _print_Symbol(self, s):
        """Internal helper `_print_Symbol`."""
        for m in self.symbol_maps:
            if s in m:
                return m[s]
        return super()._print_Symbol(s)

    def _print_Function(self, expr):
        """Internal helper `_print_Function`."""
        name = expr.func.__name__
        if name in self.c_functions:
            return self.c_functions[name](self, *expr.args)
        return f"{self._print(expr.func)}({', '.join(map(self._print, expr.args))})"

    def _print_IndexedBase(self, expr):
        """Internal helper `_print_IndexedBase`."""
        return self._print(expr.label) if hasattr(expr, "label") else str(expr)

    def _print_Indexed(self, expr):
        """Internal helper `_print_Indexed`."""
        base = self._print(expr.base)
        indices = [self._print(i) for i in expr.indices]
        return f"{base}[{']['.join(indices)}]"

    def _print_min_max(self, func_name, args):
        """Internal helper `_print_min_max`."""
        if len(args) == 1:
            return self._print(args[0])
        if len(args) == 2:
            return f"{self.math_namespace}{func_name}({self._print(args[0])}, {self._print(args[1])})"
        arg0 = self._print(args[0])
        rest = self._print_min_max(func_name, args[1:])
        return f"{self.math_namespace}{func_name}({arg0}, {rest})"

    def _print_nested_max(self, str_args):
        """Nest std::max calls for >2 arguments: max(a, max(b, max(c, d)))."""
        if len(str_args) == 1:
            return str_args[0]
        if len(str_args) == 2:
            return f"{self.math_namespace}max({str_args[0]}, {str_args[1]})"
        return f"{self.math_namespace}max({str_args[0]}, {self._print_nested_max(str_args[1:])})"

    def _expand_vector_conditionals(self, expr):
        """Internal helper `_expand_vector_conditionals`."""
        if isinstance(expr, (list, tuple)):
            return [self._expand_vector_conditionals(e) for e in expr]
        if hasattr(expr, "__getitem__") and not isinstance(expr, sp.Symbol):
            if hasattr(expr, "reshape") and hasattr(expr, "shape"):
                flat_args = [self._expand_vector_conditionals(e) for e in expr]
                return sp.Array(flat_args).reshape(*expr.shape)
            return [self._expand_vector_conditionals(e) for e in expr]
        if isinstance(expr, sp.Function) and expr.func.__name__ == "conditional":
            cond, true_val, false_val = expr.args
            is_vec_t = hasattr(true_val, "__getitem__") and not isinstance(
                true_val, sp.Symbol
            )
            is_vec_f = hasattr(false_val, "__getitem__") and not isinstance(
                false_val, sp.Symbol
            )
            if is_vec_t or is_vec_f:
                t_list = list(true_val) if is_vec_t else [true_val]
                f_list = list(false_val) if is_vec_f else [false_val]
                expanded_list = []
                for t, f in itertools.zip_longest(t_list, f_list, fillvalue=0):
                    expanded_list.append(
                        sp.Function("conditional")(
                            cond,
                            self._expand_vector_conditionals(t),
                            self._expand_vector_conditionals(f),
                        )
                    )
                shape = (
                    true_val.shape
                    if hasattr(true_val, "shape")
                    else (len(expanded_list),)
                )
                return sp.Array(expanded_list).reshape(*shape)
        return expr

    def _flatten_ragged_list(self, expr_list):
        """Internal helper `_flatten_ragged_list`."""
        flat = []
        for e in expr_list:
            if isinstance(e, (list, tuple, sp.NDimArray)):
                flat.extend(self._flatten_ragged_list(e))
            else:
                flat.append(e)
        return flat

    def _optimize_array_calls(self, expr_list):
        """Internal helper `_optimize_array_calls`."""
        definitions = []
        call_cache = {}

        def replace_logic(node):
            """Replace logic."""
            # Piecewise is a control structure, not a model-function
            # call — never extract it (its ExprCondPair args otherwise
            # look array-valued and get mangled).  It is printed inline
            # as a nested conditional by the language printer.
            if isinstance(node, sp.Piecewise):
                return node
            if isinstance(node, sp.Indexed):
                base = node.base
                label = base.label if hasattr(base, "label") else base
                if isinstance(label, sp.Function):
                    if label not in call_cache:
                        sym = sp.Symbol(f"v_call_{len(call_cache)}")
                        call_cache[label] = sym
                        definitions.append((sym, label))
                    return sp.Indexed(call_cache[label], *node.indices)
            if isinstance(node, sp.Function):
                name = node.func.__name__
                if name not in self.c_functions:
                    has_array_arg = any(
                        hasattr(a, "__getitem__") and not isinstance(a, sp.Symbol)
                        for a in node.args
                    )
                    if has_array_arg:
                        if node not in call_cache:
                            sym = sp.Symbol(f"v_call_{len(call_cache)}")
                            call_cache[node] = sym
                            definitions.append((sym, node))
                        return call_cache[node]
            return node

        new_exprs = []
        for e in expr_list:
            new_e = e.replace(
                lambda x: isinstance(x, (sp.Indexed, sp.Function)), replace_logic
            )
            new_exprs.append(new_e)
        return definitions, new_exprs

    def get_array_type(self, shape):
        """Returns the C++ type string for the array."""
        total_size = 1
        for s in shape:
            total_size *= s
        return f"SimpleArray<T, {total_size}>"

    def get_array_declaration(self, target_name, shape, init_zero=False):
        """Returns the C++ declaration string for the array."""
        total_size = 1
        for s in shape:
            total_size *= s
        arr_type = self.get_array_type(shape)
        if total_size == 0:
            return f"{arr_type} {target_name};"  # zero-size: no initializer
        if init_zero:
            return f"{arr_type} {target_name} = {{0}};"
        return f"{arr_type} {target_name};"

    def format_array_initialization(self, sym_name, elements):
        """Hook to format how local argument arrays are initialized."""
        init_str = ", ".join([self.doprint(e) for e in elements])
        return f"{self.real_type} {sym_name}[] = {{ {init_str} }};"

    def convert_expression_body(self, expr, shape, target="res"):
        """Convert expression body."""
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
        arg_defs = []
        clean_call_defs = []
        arg_cache = {}
        for call_sym, call_expr in call_defs:
            new_args = []
            for arg in call_expr.args:
                if hasattr(arg, "__getitem__") and not isinstance(arg, sp.Symbol):
                    if arg not in arg_cache:
                        sym = sp.Symbol(f"v_arg_{len(arg_cache)}")
                        arg_cache[arg] = sym
                        arg_defs.append((sym, arg))
                    new_args.append(arg_cache[arg])
                else:
                    new_args.append(arg)
            clean_call_defs.append((call_sym, call_expr.func(*new_args)))
        lines = []
        tmp_sym_gen = sp.numbered_symbols("t")
        arg_cse_inputs = []
        arg_sizes = {}
        for sym, arr in arg_defs:
            flat_arr = list(sp.flatten(arr))
            arg_cse_inputs.extend(flat_arr)
            arg_sizes[sym] = len(flat_arr)
        if arg_cse_inputs:
            temps_args, simplified_args = sp.cse(arg_cse_inputs, symbols=tmp_sym_gen)
            for lhs, rhs in temps_args:
                lines.append(
                    f"{self.real_type} {self.doprint(lhs)} = {self.doprint(rhs)};"
                )
            offset = 0
            for sym, _ in arg_defs:
                size = arg_sizes[sym]
                elements = simplified_args[offset : offset + size]
                offset += size
                lines.append(
                    self.format_array_initialization(self.doprint(sym), elements)
                )
        for sym, call in clean_call_defs:
            lines.append(f"auto {self.doprint(sym)} = {self.doprint(call)};")
        if optim_exprs:
            temps_res, simplified_res = sp.cse(optim_exprs, symbols=tmp_sym_gen)
            for lhs, rhs in temps_res:
                lines.append(
                    f"{self.real_type} {self.doprint(lhs)} = {self.doprint(rhs)};"
                )
            total_size = 1
            for s in shape:
                total_size *= s
            lines.append(self.get_array_declaration(target, shape, init_zero=False))
            result_array = sp.Array(simplified_res).reshape(*shape)
            ranges = [range(s) for s in shape]
            for indices in itertools.product(*ranges):
                val = self.doprint(result_array[indices])
                idx = flatten_index(indices, shape)
                lines.append(self.format_assignment(target, indices, val, shape))
        else:
            total_size = 1
            for s in shape:
                total_size *= s
            lines.append(self.get_array_declaration(target, shape, init_zero=True))
        lines.append(f"return {target};")
        return "\n".join(["    " + line for line in lines])

    def _print_piecewise_structure(self, expr, shape, target):
        """Internal helper `_print_piecewise_structure`."""
        lines = []
        for i, arg in enumerate(expr.args):
            val, cond = (
                (arg.expr, arg.cond) if hasattr(arg, "expr") else (arg[0], arg[1])
            )
            cond_str = self.doprint(cond)
            if i == 0:
                lines.append(f"    if ({cond_str}) {{")
            elif cond == True or cond == sp.true:
                lines.append("    } else {")
            else:
                lines.append(f"    }} else if ({cond_str}) {{")
            lines.append(
                textwrap.indent(
                    self.convert_expression_body(val, shape, target), "    "
                )
            )
        lines.append("    }")
        total_size = 1
        for s in shape:
            total_size *= s
        decl = self.get_array_declaration(f"default_{target}", shape, init_zero=True)
        lines.append(f"    {decl}")
        lines.append(f"    return default_{target};")
        return "\n".join(lines)

    def _generate_signature_from_function(self, func_obj):
        """Internal helper `_generate_signature_from_function`."""
        decls = []
        for key, obj in func_obj.args.items():
            cpp_name = self.ARG_MAPPING.get(key, key)
            is_pointer = (
                hasattr(obj, "_symbolic_name")
                or hasattr(obj, "values")
                or isinstance(obj, (ZArray, list, tuple, sp.MatrixBase, sp.NDimArray))
            )
            if key in [
                "variables",
                "aux_variables",
                "gradient_variables",
                "p",
                "normal",
                "position",
                "q_minus",
                "q_plus",
                "aux_minus",
                "aux_plus",
                "flux_minus",
                "flux_plus",
            ]:
                is_pointer = True
            type_prefix = "const T*" if is_pointer else "const T"
            decls.append(f"{type_prefix} {cpp_name}")
        return ",\n        ".join(decls)

    def _process_kernel_from_function(self, func_obj):
        """Internal helper `_process_kernel_from_function`."""
        name = func_obj.name
        expr = func_obj.definition
        expr = self._expand_vector_conditionals(expr)
        if isinstance(expr, (list, tuple)):
            expr = self._flatten_ragged_list(expr)
            expr = sp.Array(expr)
        shape, expr = get_nested_shape(expr)
        body = self.convert_expression_body(expr, shape)
        args_str = self._generate_signature_from_function(func_obj)
        return [self.wrap_function_signature(name, args_str, body, shape)]

    def get_includes(self):
        """Get includes."""
        return """#include <cmath>
#include <array>
"""

    def get_simple_array_def(self):
        """Get simple array def."""
        return """
#ifndef ZOOMY_SIMPLE_ARRAY
#define ZOOMY_SIMPLE_ARRAY
template <typename T, int N>
struct SimpleArray {
    T data[N];
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
    T* begin() { return data; }
    const T* begin() const { return data; }
    T* end() { return data + N; }
    const T* end() const { return data + N; }
};
#endif
"""

    def format_accessor(self, var_name, index):
        """Format accessor."""
        return f"{var_name}[{index}]"

    def format_assignment(self, target_name, indices, value, shape):
        """Format assignment."""
        idx = flatten_index(indices, shape)
        return f"{target_name}[{idx}] = {value};"

    def get_variable_declaration(self, variable_name):
        """Get variable declaration."""
        return f"const T* {variable_name}"

    def wrap_function_signature(self, name, args_str, body_str, shape):
        """Wrap function signature."""
        qualifier = "PORTABLE_FN " if self.gpu_enabled else ""
        arr_type = self.get_array_type(shape)
        return f"""    {qualifier}static inline {arr_type} {name}(
        {args_str})
    {{
{body_str}
    }}
"""

    def _print_Pow(self, expr):
        """Internal helper `_print_Pow`."""
        base, exp = expr.as_base_exp()
        if exp.is_Integer:
            n = int(exp)
            if n == 0:
                return "1.0"
            if n == 1:
                return self._print(base)
            pow_func = f"{self.math_namespace}pow"
            if n < 0:
                return f"(1.0 / {pow_func}({self._print(base)}, {abs(n)}))"
            return f"{pow_func}({self._print(base)}, {n})"
        return f"{self.math_namespace}pow({self._print(base)}, {self._print(exp)})"

    def doprint(self, expr, **settings):
        """Doprint."""
        code = super().doprint(expr, **settings)
        if self.math_namespace != "std::":

            def _repl(match):
                """Internal helper `_repl`."""
                return f"{self.math_namespace}{match.group(1)}"

            return self._std_regex.sub(_repl, code)
        return code

    @classmethod
    def _write_file(cls, code, settings, filename):
        """Internal helper `_write_file`."""
        main_dir = misc.get_main_directory()
        output_dir = os.path.join(
            main_dir, settings.output.directory, cls._output_subdir
        )
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "w+") as f:
            f.write(code)
        return file_path


# =========================================================================
#  2b. OUT-PARAMETER CODE PRINTER (shared JS / GLSL base)
# =========================================================================


class OutParamCodePrinter(GenericCppBase):
    """Shared base for the out-parameter code printers (JS, GLSL).

    Generated kernels return ``void`` / nothing and write their result
    through a trailing ``res`` array parameter the caller owns — no
    per-call heap allocation, which is what keeps the browser solver
    fast.  The body-generation, Piecewise handling and kernel/BC
    wrapping all live here; subclasses supply the language-specific
    bits via :meth:`_temp_decl`, :meth:`_generate_args` and
    :meth:`wrap_function_signature`.
    """

    # ``parameters`` is the BoundaryCondition Function's key for the
    # model parameter vector; map it onto the same ``p`` accessor.
    ARG_MAPPING = {**GenericCppBase.ARG_MAPPING, "parameters": "p"}

    @staticmethod
    def _shape_size(shape):
        """Total number of components for a (possibly nested) shape."""
        total = 1
        for s in shape:
            total *= s
        return total

    # ── Language hooks ───────────────────────────────────────────────

    def _temp_decl(self, lhs, rhs):
        """Declaration line for one CSE temporary.  Default is a
        JS-style ``const``; GLSL overrides to pick ``bool`` / ``float``."""
        return f"const {self.doprint(lhs)} = {self.doprint(rhs)};"

    def _generate_args(self, func_obj, shape):
        """Parameter list for a Function object, including the trailing
        ``res`` out-parameter.  Language-specific — must be overridden."""
        raise NotImplementedError

    # ── Piecewise ────────────────────────────────────────────────────

    def _print_Piecewise(self, expr):
        """Nested *scalar* Piecewise → a chained ternary.

        The top-level vector Piecewise (a boundary-condition dispatch)
        is handled by :meth:`_print_piecewise_structure`; this prints a
        scalar Piecewise appearing *inside* an expression — e.g. a
        piecewise-linear Q(t) interpolation.
        """
        args = list(expr.args)
        last = args[-1]
        if last.cond is True or last.cond == sp.true:
            result = f"({self._print(last.expr)})"
            rest = args[:-1]
        else:
            # Exhaustive-but-no-literal-True chains (e.g. the time
            # interpolation) — the fallback is never reached.
            result = "0.0"
            rest = args
        for pair in reversed(rest):
            cond = self.doprint(pair.cond)
            val = self._print(pair.expr)
            result = f"(({cond}) ? ({val}) : ({result}))"
        return result

    # ── Body generation (out-parameter, no return) ───────────────────

    def convert_expression_body(self, expr, shape, target="res"):
        """Convert an expression into a body that writes ``target`` — the
        function's out-parameter array.  No local declaration, no
        ``return``."""
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
                "out-parameter printer: nested array-valued function "
                "calls are not supported"
            )

        total = self._shape_size(shape)
        lines = []
        if optim_exprs:
            temps, simplified = sp.cse(
                optim_exprs, symbols=sp.numbered_symbols("t")
            )
            for lhs, rhs in temps:
                lines.append(self._temp_decl(lhs, rhs))
            result_array = sp.Array(simplified).reshape(*shape)
            for indices in itertools.product(*[range(s) for s in shape]):
                val = self.doprint(result_array[indices])
                lines.append(self.format_assignment(target, indices, val, shape))
        else:
            for i in range(total):
                lines.append(f"{target}[{i}] = 0.0;")
        return "\n".join("    " + ln for ln in lines)

    def _print_piecewise_structure(self, expr, shape, target):
        """Emit an if / else-if chain assigning into ``target``.

        ``target`` is zero-initialised first so any fall-through (a
        Piecewise with no final ``True`` branch — e.g. a
        boundary-condition dispatch) is well-defined."""
        total = self._shape_size(shape)
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

    # ── Kernel / boundary-condition generation ───────────────────────

    def _generate_kernel(self, name, func_obj):
        """Generate one out-parameter kernel from a Function object.

        Returns ``None`` for a zero-size result (a zero-component
        out-parameter is pointless and GLSL forbids it)."""
        expr = func_obj.definition
        expr = self._expand_vector_conditionals(expr)
        if isinstance(expr, (list, tuple)):
            expr = self._flatten_ragged_list(expr)
            expr = sp.Array(expr)
        shape, expr = get_nested_shape(expr)
        if self._shape_size(shape) == 0:
            return None
        body = self.convert_expression_body(expr, shape)
        args_str = self._generate_args(func_obj, shape)
        return self.wrap_function_signature(name, args_str, body, shape)

    def generate_boundary_conditions(self):
        """Generate the model's boundary-condition dispatch kernels —
        ``boundary_conditions`` and ``aux_boundary_conditions``, each a
        Piecewise over the integer ``bc_idx``."""
        blocks = []
        for attr in ("_boundary_conditions", "_aux_boundary_conditions"):
            func_obj = getattr(self.model, attr, None)
            if func_obj is None:
                continue
            kernel = self._generate_bc_kernel(func_obj)
            if kernel is not None:
                blocks.append(kernel)
        return "\n\n".join(blocks)

    def _generate_bc_kernel(self, func_obj):
        """Generate one boundary-condition kernel.

        Unlike :meth:`_generate_kernel` this does *not* run
        ``_expand_vector_conditionals`` — the BC definition is already a
        Piecewise of vector branches.  The scalar BC symbols (``bc_idx``,
        ``time``, ``distance``) are registered so they print as their
        parameter names."""
        scalar_map = {}
        for key, param_name in (
            ("idx", "bc_idx"),
            ("time", self.ARG_MAPPING.get("time", "time")),
            ("distance", self.ARG_MAPPING.get("distance", "dX")),
        ):
            if func_obj.args.contains(key):
                scalar_map[func_obj.args[key]] = param_name
        self.symbol_maps.append(scalar_map)
        try:
            shape, expr = get_nested_shape(func_obj.definition)
            if self._shape_size(shape) == 0:
                return None
            body = self.convert_expression_body(expr, shape)
            args_str = self._generate_args(func_obj, shape)
            return self.wrap_function_signature(
                func_obj.name, args_str, body, shape
            )
        finally:
            self.symbol_maps.pop()


# =========================================================================
#  3. GENERIC MODEL (Clean)
# =========================================================================


class GenericCppModel(GenericCppBase):
    """GenericCppModel. (class)."""
    _wrapper_name = "Model"

    def __init__(self, model, *args, **kwargs):
        """Initialize the instance."""
        super().__init__(*args, **kwargs)
        self.model = model
        # Declarative models carry their lowered interface (state, params,
        # BC Piecewise) on the SystemModel, not on the Model itself (e.g.
        # SME(level=0) has n_variables==1 but the SystemModel state is
        # [b, h, q_0] with 5 parameters).  Source the emitted interface
        # from the SystemModel when present — mirrors FoamSystemModelPrinter.
        # Legacy hand-written Models (no ``system_model``) fall back to the
        # Model's own counts.
        self.sm = getattr(model, "system_model", None)
        if self.sm is not None:
            self.n_dof_q = self.sm.n_equations
            self.n_dof_qaux = len(self.sm.aux_state)
            self.n_parameters = len(list(self.sm.parameters.keys()))
        else:
            self.n_dof_q = model.n_variables
            self.n_dof_qaux = model.n_aux_variables
            self.n_parameters = model.n_parameters
        self.register_map("Q", model.variables.values())
        self.register_map("Qaux", model.aux_variables.values())
        if hasattr(model, "gradient_variables") and model.gradient_variables.length() > 0:
            self.register_map("gradQ", model.gradient_variables.values())
        self.register_map("n", model.normal.values())
        if hasattr(model, "position"):
            self.register_map("X", model.position.values())
        self.register_map("p", self.model.parameters.values())

    @classmethod
    def write_code(cls, model, settings, filename="Model.H"):
        """Write code."""
        printer = cls(model)
        code = printer.create_code()
        return cls._write_file(code, settings, filename)

    def create_code(self):
        """Create code.

        Declarative models keep their operator matrices (flux, NCP,
        quasilinear, source, eigenvalues, diffusion) on the SystemModel —
        emit those when present, mirroring ``FoamSystemModelPrinter``.
        Legacy hand-written Models (no ``system_model``) fall back to
        iterating the Model's own ``functions``.
        """
        blocks = [self.get_file_header()]
        if self.sm is not None:
            blocks.extend(self._emit_operator_kernels())
            # Drop-in kernels the dmplex/C solver calls directly (REQ-44):
            # stacked all-directions flux / NCP, the source Jacobians, the
            # per-cell state / aux update + its Jacobian, and the initial
            # condition.  Each is read off the SystemModel; a None slot is
            # documented (``_doc_note``) instead of emitting a broken kernel.
            blocks.extend(self._emit_stacked_operators())
            blocks.extend(self._emit_source_jacobians())
            blocks.extend(self._emit_update_kernels())
            blocks.extend(self._emit_initial_conditions())
            blocks.extend(self._emit_reconstruction_kernels())
            blocks.extend(self._emit_projection_kernels())
        else:
            for name, func_obj in self.model.functions.items():
                blocks.extend(self._process_kernel_from_function(func_obj))

        blocks.extend(self._emit_boundary_conditions())

        blocks.append(self.get_file_footer())
        return "\n".join(blocks)

    # ── SystemModel operator-kernel emission ─────────────────────────────
    #
    # The operator matrices live on the SystemModel and reference its
    # state / aux / parameter / normal symbols (NOT the declarative
    # Model's own ``variables``).  These mirror
    # ``FoamSystemModelPrinter._per_direction`` / ``_slice`` / ``_kernel``
    # but emit the C++ pointer signatures the dmplex/C backend consumes.

    _AXIS = ("x", "y", "z")
    # Canonical 3D-field profile the coupling maps exchange (Phase 7):
    # ``interpolate_to_3d`` emits these slots in order; ``project_from_3d``
    # consumes them through fresh ``P3_<field>`` symbols mapped to
    # ``profile[i]``.  Mirrors ``FoamSystemModelPrinter._PROFILE_3D_FIELDS``.
    _PROFILE_3D_FIELDS = ("b", "h", "u", "v", "w", "p")
    # Emit the SystemModel's symbolic eigenvalue spectrum (default).  Set
    # False to emit a zero placeholder when the solver computes the
    # spectrum numerically from ``quasilinear_matrix``.
    analytical_eigenvalues = True

    def _sm_symbol_map(self):
        """Map the SystemModel state / aux / parameter / normal symbols to
        their C interface accessors (``Q[i]`` / ``Qaux[i]`` / ``p[i]`` /
        ``n[i]``).  Shared by the operator and boundary-condition kernels."""
        sm = self.sm
        smap = {}
        for i, s in enumerate(sm.state):
            smap[s] = self.format_accessor("Q", i)
        for i, s in enumerate(sm.aux_state):
            smap[s] = self.format_accessor("Qaux", i)
        for i, s in enumerate(sm.parameters.values()):
            smap[s] = self.format_accessor("p", i)
        for i, s in enumerate(sm.normal.values()):
            smap[s] = self.format_accessor("n", i)
        return smap

    def _sm_arg_decl(self, key):
        """C++ declaration for one operator-kernel argument (pointer
        form: ``const T* Q``).  Backends with a different calling
        convention override this."""
        return f"const {self.real_type}* {key}"

    def _sm_signature(self, args):
        """Operator-kernel parameter list from an ordered arg-key list."""
        return ",\n        ".join(self._sm_arg_decl(a) for a in args)

    def _kernel_with_symbol_map(self, name, expr, shape, args, smap):
        """Emit one kernel from a symbolic expression with ``smap`` pushed for
        the body (the shared lowering path; ``_sm_kernel`` uses the SystemModel
        state/aux/parameter/normal map, the IC kernels a position/parameter
        map)."""
        self.symbol_maps.append(smap)
        try:
            body = self.convert_expression_body(expr, shape)
        finally:
            self.symbol_maps.pop()
        return self.wrap_function_signature(
            name, self._sm_signature(args), body, shape
        )

    def _sm_kernel(self, name, expr, shape, args):
        """Emit one operator kernel from a SystemModel expression, with the
        SystemModel symbol map pushed for the body."""
        return self._kernel_with_symbol_map(
            name, expr, shape, args, self._sm_symbol_map()
        )

    def _doc_note(self, name, reason):
        """A struct-body comment standing in for a kernel the SystemModel does
        not carry, so the header stays a *documented contract* rather than a
        broken / zero kernel.  The backend must supply ``name`` solver-side
        (case directory / settings).  Mirrors the reconstruction skip but
        leaves a paper trail the dmplex consumer can grep for."""
        wrapped = textwrap.fill(
            reason,
            width=70,
            initial_indent="    //   ",
            subsequent_indent="    //   ",
        )
        return (
            f"    // {name}: NOT emitted from the SystemModel "
            f"(provide solver-side).\n{wrapped}\n"
        )

    def _sm_slice(self, tensor, axis_idx, out_shape):
        """``tensor[..., axis_idx]`` reshaped to ``out_shape`` — the
        per-direction slice of a direction-indexed operator tensor.  If
        ``out_shape`` carries a trailing ``1`` padding (the ``flux_x``
        column convention) walk one fewer axis when collecting values."""
        walk = (
            out_shape[:-1]
            if (len(out_shape) == len(tensor.shape) and out_shape[-1] == 1)
            else out_shape
        )
        flat = [
            tensor[(*idx, axis_idx)]
            for idx in itertools.product(*(range(s) for s in walk))
        ]
        return sp.Array(flat).reshape(*out_shape)

    def _sm_per_direction(self, base, tensor, out_shape, args):
        """Emit ``base_x`` / ``base_y`` / ``base_z`` kernels (one per
        spatial dimension) from a direction-indexed operator tensor."""
        return [
            self._sm_kernel(
                f"{base}_{self._AXIS[d]}",
                self._sm_slice(tensor, d, out_shape),
                out_shape,
                args,
            )
            for d in range(self.sm.dimension)
        ]

    def _emit_operator_kernels(self):
        """Emit the SystemModel operator kernels into ``Model.H``:
        per-direction conservative flux, nonconservative (NCP) matrix,
        quasilinear matrix, the eigenvalue spectrum, the source term, and
        diffusion when present.  Mirrors ``FoamSystemModelPrinter``."""
        sm = self.sm
        n_eq, n_state = sm.n_equations, len(sm.state)
        blocks = []
        blocks += self._sm_per_direction(
            "flux", sm.flux, (n_eq, 1), ["Q", "Qaux", "p"]
        )
        blocks += self._sm_per_direction(
            "nonconservative_matrix",
            sm.nonconservative_matrix,
            (n_eq, n_state),
            ["Q", "Qaux", "p"],
        )
        blocks += self._sm_per_direction(
            "quasilinear_matrix",
            sm.quasilinear_matrix,
            (n_eq, n_state),
            ["Q", "Qaux", "p"],
        )
        eig_expr = (
            sm.eigenvalues
            if self.analytical_eigenvalues and sm.eigenvalues is not None
            else sp.Array([[0]] * n_eq)
        )
        blocks.append(
            self._sm_kernel(
                "eigenvalues", eig_expr, (n_eq, 1), ["Q", "Qaux", "p", "n"]
            )
        )
        blocks.append(
            self._sm_kernel("source", sm.source, (n_eq, 1), ["Q", "Qaux", "p"])
        )
        if self._detect_has_diffusion():
            blocks += self._sm_per_direction(
                "diffusion_matrix",
                sm.diffusion_matrix,
                (n_eq, n_state),
                ["Q", "Qaux", "p"],
            )
        return blocks

    def _emit_stacked_operators(self):
        """Emit the *stacked* all-directions ``flux(Q,Qaux,p)`` and
        ``nonconservative_matrix(Q,Qaux,p)`` the dmplex solver / ``Numerics.H``
        call directly — in addition to the per-direction ``*_x`` / ``*_y``
        accessors ``_emit_operator_kernels`` already emits.

        Layout is the operator tensor flattened row-major (the convention the
        solver indexes): ``flux[eq, dim]`` → ``SimpleArray<n_eq*dim>`` and
        ``NCP[eq, state, dim]`` → ``SimpleArray<n_eq*n_state*dim>`` (see
        ``Model.H``'s ``flux`` / ``nonconservative_matrix`` and ``Numerics.H``'s
        ``Model<T>::flux`` / ``::nonconservative_matrix`` calls)."""
        sm = self.sm
        n_eq, n_state, dim = sm.n_equations, len(sm.state), sm.dimension
        return [
            self._sm_kernel(
                "flux", sm.flux, (n_eq, dim), ["Q", "Qaux", "p"]
            ),
            self._sm_kernel(
                "nonconservative_matrix",
                sm.nonconservative_matrix,
                (n_eq, n_state, dim),
                ["Q", "Qaux", "p"],
            ),
        ]

    def _emit_source_jacobians(self):
        """Emit the symbolic source Jacobians the implicit source step needs:

        * ``source_jacobian_wrt_variables(Q,Qaux,p)`` → ``(n_eq, n_state)`` —
          ``dS/dQ`` holding aux fixed (the SystemModel's ``source_jacobian``).
        * ``source_jacobian_wrt_aux_variables(Q,Qaux,p)`` → ``(n_eq, n_aux)`` —
          ``dS/dQaux``; the solver completes the chain rule
          ``dS/dQ + dS/dQaux . dQaux/dQ`` (see ``ModularSolver.hpp`` ~l.268).

        Row-major layout matches the solver's ``dS_dQ[i*n_dof+j]`` /
        ``dS_dAux[i*n_aux+k]`` indexing."""
        sm = self.sm
        n_eq, n_state, n_aux = sm.n_equations, len(sm.state), len(sm.aux_state)
        if sm.source is None:
            return [self._doc_note(
                "source_jacobian_wrt_variables / "
                "source_jacobian_wrt_aux_variables",
                "the SystemModel carries no source term.")]
        src = [sp.sympify(e) for e in sp.flatten(sm.source)]
        blocks = []
        jq = getattr(sm, "source_jacobian", None)
        if jq is None:
            jq = sp.derive_by_array(sp.Array(src), list(sm.state))
        blocks.append(self._sm_kernel(
            "source_jacobian_wrt_variables", jq, (n_eq, n_state),
            ["Q", "Qaux", "p"],
        ))
        if n_aux > 0:
            jaux = sp.Array(
                [[sp.diff(src[i], sm.aux_state[k]) for k in range(n_aux)]
                 for i in range(n_eq)]
            )
            blocks.append(self._sm_kernel(
                "source_jacobian_wrt_aux_variables", jaux, (n_eq, n_aux),
                ["Q", "Qaux", "p"],
            ))
        return blocks

    def _emit_update_kernels(self):
        """Emit the per-cell state / aux update kernels (mirrors the dmplex
        ``Model::update_variables`` / ``update_aux_variables`` /
        ``update_aux_variables_jacobian_wrt_variables`` the solver calls in
        ``TransportStep`` / ``ModularSolver``):

        * ``update_variables(Q,Qaux,p)`` → ``(n_eq,)`` — pointwise state remap.
        * ``update_aux_variables(Q,Qaux,p)`` → ``(n_aux,)`` — pointwise aux
          recompute, plus its Jacobian ``(n_aux, n_state)`` (row-major
          ``dAux_dQ[k*n_dof+j]``).

        These are *pointwise algebraic* updates.  A declarative model whose aux
        are mesh-derivative (computed by the solver's gradient) or user
        closures carries ``None`` here — documented, not emitted (it is the
        ``FoamUpdateAuxPrinter`` ``compute_derivative`` path, which cannot be a
        pointwise ``Model`` kernel)."""
        sm = self.sm
        n_eq, n_state, n_aux = sm.n_equations, len(sm.state), len(sm.aux_state)
        blocks = []

        uv = getattr(sm, "update_variables", None)
        if uv is not None and len(sp.flatten(uv)) > 0:
            blocks.append(self._sm_kernel(
                "update_variables", uv, (n_eq,), ["Q", "Qaux", "p"],
            ))
        else:
            blocks.append(self._doc_note(
                "update_variables",
                "no pointwise state remap on this SystemModel; the solver "
                "leaves Q unchanged after the transport step."))

        uav = getattr(sm, "update_aux_variables", None)
        if uav is not None and len(sp.flatten(uav)) > 0:
            blocks.append(self._sm_kernel(
                "update_aux_variables", uav, (n_aux,), ["Q", "Qaux", "p"],
            ))
            jac = sp.derive_by_array(
                sp.Array([sp.sympify(e) for e in sp.flatten(uav)]),
                list(sm.state),
            )
            blocks.append(self._sm_kernel(
                "update_aux_variables_jacobian_wrt_variables",
                jac, (n_aux, n_state), ["Q", "Qaux", "p"],
            ))
        else:
            blocks.append(self._doc_note(
                "update_aux_variables / "
                "update_aux_variables_jacobian_wrt_variables",
                "aux are mesh-derivative (solver gradient) and/or user "
                "closures; no pointwise algebraic aux update on this "
                "SystemModel — compute aux solver-side."))
        return blocks

    def _emit_initial_conditions(self):
        """Emit ``initial_condition(X,p)`` / ``initial_aux_condition(X,p)`` from
        the SystemModel's ``InitialConditions`` (``get_definition`` lowers them
        to a symbolic vector in the position symbols ``X[i]`` and parameters
        ``p[i]``, exactly the old MOOD-era header's IC).

        A SystemModel that carries no symbolic IC (IC is case-side, set from
        the case directory / ``settings.json``) is documented, not stubbed."""
        sm = self.sm
        return [
            self._ic_block(
                "initial_condition",
                getattr(sm, "initial_conditions", None),
                sm.n_equations,
            ),
            self._ic_block(
                "initial_aux_condition",
                getattr(sm, "aux_initial_conditions", None),
                len(sm.aux_state),
            ),
        ]

    def _ic_block(self, name, ic, n):
        """One IC kernel, or a doc-note when the SystemModel carries no IC."""
        sm = self.sm
        if ic is None or not hasattr(ic, "get_definition"):
            return self._doc_note(
                name,
                "the SystemModel carries no symbolic initial state; IC is "
                "case-side (case directory / settings.json).")
        X = sp.Array([sp.Symbol(f"_ICX{i}", real=True) for i in range(3)])
        params = list(sm.parameters.values())
        expr = ic.get_definition(X, sp.Array(params), n)
        if len(sp.flatten(expr)) == 0 or all(e == 0 for e in sp.flatten(expr)):
            return self._doc_note(
                name,
                "the SystemModel's IC is trivial / zero; set it case-side.")
        smap = {X[i]: self.format_accessor("X", i) for i in range(3)}
        for i, s in enumerate(params):
            smap[s] = self.format_accessor("p", i)
        return self._kernel_with_symbol_map(
            name, expr, (n,), ["X", "p"], smap,
        )

    def _emit_reconstruction_kernels(self):
        """Emit the MUSCL reconstruction variable change, mirroring
        ``FoamSystemModelPrinter._emit_reconstruction_kernels``:

        * ``reconstruction_variables(Q, Qaux, p) -> W[n_state]`` — forward
          map from the conservative state to the reconstruction variables
          (e.g. ``eta = b + h``).  Uses the shared SystemModel symbol scope.
        * ``state_from_reconstruction(W, Qaux, p) -> Q[n_state]`` — the
          inverse, parameterised by the fresh ``WB_<state>`` symbols
          ``invert_reconstruction`` created.  Push a temporary ``WB_* -> W[i]``
          map so each reads as ``W[i]`` for the matching state slot.

        A SystemModel that carries no reconstruction maps (default
        conservative reconstruction, e.g. VAM / Chorin sub-systems) emits
        nothing, exactly like the foam printer's skip.
        """
        sm = self.sm
        if (sm.reconstruction_variables is None
                or sm.state_from_reconstruction is None):
            return []
        shape = (len(sm.state),)

        fwd = self._sm_kernel(
            "reconstruction_variables",
            sm.reconstruction_variables,
            shape,
            ["Q", "Qaux", "p"],
        )

        # Inverse map — map the actual ``WB_*`` symbols (assumptions like
        # real=True mean a freshly-built ``Symbol("WB_b")`` would not match)
        # to ``W[i]`` of their matching state slot.  Pushed below the
        # ``_sm_kernel`` symbol map; state symbols never appear in the inverse
        # expressions, so there is no clash with the Q[i] scope.
        wb_map = {}
        free = set()
        for expr in sp.flatten(sm.state_from_reconstruction):
            if hasattr(expr, "free_symbols"):
                free |= expr.free_symbols
        wb_by_name = {str(s): s for s in free if str(s).startswith("WB_")}
        for i, state_sym in enumerate(sm.state):
            wb_name = f"WB_{state_sym}"
            if wb_name in wb_by_name:
                wb_map[wb_by_name[wb_name]] = self.format_accessor("W", i)
        self.symbol_maps.append(wb_map)
        try:
            inv = self._sm_kernel(
                "state_from_reconstruction",
                sm.state_from_reconstruction,
                shape,
                ["W", "Qaux", "p"],
            )
        finally:
            self.symbol_maps.pop()

        return [fwd, inv]

    def _emit_projection_kernels(self):
        """Emit the 3-D coupling maps from the SystemModel, mirroring
        ``FoamSystemModelPrinter._emit_projection_kernels`` but in the C++
        pointer-arg style (``const T*`` / ``res[...]``):

        * ``interpolate_to_3d(Q, Qaux, p, X) -> field[6]`` — the canonical
          3-D field ``[b,h,u,v,w,p]`` evaluated at one vertical position;
          ``sm.position[2]`` → ``X[2]`` via the position map the constructor
          already registered (the same ``X`` convention every C++ kernel
          uses, in place of foam's scalar ``z``).  The backend loops the
          column, calling this once per sample point.
        * ``project_from_3d(profile, p[, I]) -> Q[n_state]`` — the inverse:
          reduce one sampled column back to the 2-D state.  The depth-average
          enters through the ``P3_<field>`` symbols (``profile[i]``); any
          ``Integral(g(ζ), (ζ,0,1))`` row enters through a column-quadrature
          accumulator ``I[j]`` the backend fills (matching foam's
          ``project_from_3d_at`` per-profile signature).

        Both maps are read off the SystemModel slots filled by
        ``register_group("interpolate"/"project", …)``.  A model with the
        slot ``None`` / absent (default reconstruction, e.g. VAM / Chorin
        sub-systems) emits nothing — exactly like the reconstruction skip.
        """
        sm = self.sm
        blocks = []

        p2 = getattr(sm, "interpolate_to_3d", None)
        # The base model returns zeros(6); only emit a real reconstruction.
        if p2 is not None and any(e != 0 for e in sp.flatten(p2)):
            shape = (len(sp.flatten(p2)),)
            blocks.append(self._sm_kernel(
                "interpolate_to_3d", p2, shape, ["Q", "Qaux", "p", "X"],
            ))

        p3 = getattr(sm, "project_from_3d", None)
        if p3 is not None and len(sp.flatten(p3)) > 0:
            rows = [sp.sympify(e) for e in sp.flatten(p3)]
            shape = (len(rows),)

            # ζ-quadrature lowering: every ``Integral(g(ζ), (ζ,0,1))`` becomes
            # a fresh ``I[j]`` accumulator the backend fills from the sampled
            # column (the per-profile map only consumes the result).  Rows
            # without Integrals lower exactly as the depth-averaged
            # ``profile[]`` reduction.
            integral_atoms: list = []
            for e in rows:
                for a in e.atoms(sp.Integral):
                    if a not in integral_atoms:
                        integral_atoms.append(a)
            int_syms = {a: sp.Symbol(f"_ZINT{j}", real=True)
                        for j, a in enumerate(integral_atoms)}
            rows = [e.xreplace(int_syms) for e in rows]

            free = set()
            for expr in rows:
                if hasattr(expr, "free_symbols"):
                    free |= expr.free_symbols
            by_name = {str(s): s for s in free}
            prof_map = {}
            for i, field in enumerate(self._PROFILE_3D_FIELDS):
                sym = by_name.get(f"P3_{field}")
                if sym is not None:
                    prof_map[sym] = self.format_accessor("profile", i)
            for a, s in int_syms.items():
                prof_map[s] = self.format_accessor(
                    "I", int(str(s)[len("_ZINT"):]))
            at_args = (["profile", "p", "I"] if integral_atoms
                       else ["profile", "p"])
            self.symbol_maps.append(prof_map)
            try:
                blocks.append(self._sm_kernel(
                    "project_from_3d", sp.Matrix(rows), shape, at_args,
                ))
            finally:
                self.symbol_maps.pop()

        return blocks

    def _emit_boundary_conditions(self):
        """Lower the model's ``BoundaryConditions`` into the indexed
        ``boundary_conditions(bc_idx, …)`` / ``aux_boundary_conditions``
        Piecewise kernels.

        The indexed Piecewise (one branch per boundary tag, dispatched on
        ``bc_idx``) is owned by the SystemModel — built from the very
        ``BoundaryConditions`` the Model carries — and its branches
        reference the SystemModel state / aux / parameter / normal symbols.
        This mirrors ``FoamSystemModelPrinter._emit_boundary_conditions``;
        we read ``sm.boundary_conditions`` instead of the stale
        ``model._boundary_conditions`` that current models no longer expose.
        """
        sm = self.sm
        if sm is None:
            return []
        blocks = [
            self._emit_bc_kernel("boundary_conditions", sm.boundary_conditions)
        ]
        aux_bc = getattr(sm, "aux_boundary_conditions", None)
        if aux_bc is not None:
            blocks.append(
                self._emit_bc_kernel("aux_boundary_conditions", aux_bc)
            )
        return blocks

    def _emit_bc_kernel(self, func_name, bc_func):
        """Emit one boundary-condition Piecewise kernel.

        Pushes a temporary symbol map binding the SystemModel state /
        aux / parameter / normal symbols (and the BC-specific scalar
        ``bc_idx`` / ``time`` / ``distance`` / position symbols) to their
        C interface accessors, then prints the Piecewise body.
        """
        sm = self.sm
        smap = {}
        for i, s in enumerate(sm.state):
            smap[s] = self.format_accessor("Q", i)
        for i, s in enumerate(sm.aux_state):
            smap[s] = self.format_accessor("Qaux", i)
        for i, s in enumerate(sm.parameters.values()):
            smap[s] = self.format_accessor("p", i)
        for i, s in enumerate(sm.normal.values()):
            smap[s] = self.format_accessor("n", i)
        args = bc_func.args
        if args.contains("idx"):
            smap[args["idx"]] = "bc_idx"
        if args.contains("time"):
            smap[args["time"]] = self.ARG_MAPPING.get("time", "time")
        if args.contains("distance"):
            smap[args["distance"]] = self.ARG_MAPPING.get("distance", "dX")
        if args.contains("position"):
            pos = args["position"]
            if hasattr(pos, "values"):
                for i, s in enumerate(pos.values()):
                    smap[s] = self.format_accessor("X", i)
        self.symbol_maps.append(smap)
        try:
            shape, expr = get_nested_shape(bc_func.definition)
            body = self.convert_expression_body(expr, shape)
            return self.wrap_function_signature(
                func_name, self.get_bc_args(), body, shape
            )
        finally:
            self.symbol_maps.pop()

    def get_file_header(self):
        """Get file header."""
        sm = self.sm
        # Boundary tags and parameter names/defaults describe the emitted
        # C interface — take them from the SystemModel when present (the
        # declarative Model's own ``parameters`` may be empty; the lowered
        # operators and BCs reference the SystemModel parameters).
        if sm is not None:
            bc_names = sorted(sm._bc_source.boundary_conditions_list_dict.keys())
            param_keys = list(sm.parameters.keys())
            param_vals = list(sm.parameter_values.values())
            dimension = sm.dimension
        else:
            bc_names = sorted(
                self.model.boundary_conditions.boundary_conditions_list_dict.keys()
            )
            param_keys = list(self.model.parameters.keys())
            param_vals = list(self.model.parameters.values())
            dimension = self.model.dimension
        bc_str = ", ".join(f'"{item}"' for item in bc_names)
        if len(param_keys) != len(param_vals):
            raise ValueError(
                f"Parameter keys {param_keys} and values values {param_vals} do not match"
            )
        p_names_str = ", ".join(f'"{k}"' for k in param_keys)
        p_vals_str = ", ".join(f"{k}" for k in param_vals)
        tpl = "template <typename T>" if self._is_template_class else ""
        lines = [
            "#pragma once",
            self.get_includes().strip(),
            self.get_simple_array_def(),
            "#include <vector>",
            "#include <string>",
            "#include <algorithm>",
            "",
        ]
        if self.gpu_enabled:
            lines.extend(
                [
                    "#ifdef __CUDACC__",
                    "#define PORTABLE_FN __host__ __device__",
                    "#else",
                    "#define PORTABLE_FN",
                    "#endif",
                    "",
                ]
            )
        else:
            lines.append("#define PORTABLE_FN")
        # Detect whether the model has non-trivial diffusion.  Models that
        # do not expose ``gradient_variables`` (current declarative models)
        # have no gradQ dof — treat as 0 rather than crashing.
        n_dof_gradQ = self._n_dof_gradq()
        has_diffusion = self._detect_has_diffusion()
        has_free_surface = self._detect_has_free_surface()

        lines.extend(
            [
                tpl,
                f"struct {self._wrapper_name} {{",
                f"    static constexpr int n_dof_q    = {self.n_dof_q};",
                f"    static constexpr int n_dof_qaux = {self.n_dof_qaux};",
                f"    static constexpr int n_parameters = {self.n_parameters};",
                f"    static constexpr int dimension  = {dimension};",
                f"    static constexpr int n_dof_gradQ = {n_dof_gradQ};",
                f"    static constexpr bool has_diffusion = {'true' if has_diffusion else 'false'};",
                f"    static constexpr bool has_free_surface = {'true' if has_free_surface else 'false'};",
            ]
        )
        if has_free_surface:
            lines.extend(
                [
                    f"    static constexpr int idx_b = {self._var_index('b')};",
                    f"    static constexpr int idx_h = {self._var_index('h')};",
                ]
            )
        lines.extend(
            [
                f"    static constexpr int n_boundary_tags = {len(bc_names)};",
                f"    static const std::vector<std::string> get_boundary_tags() {{ return {{ {bc_str} }}; }}",
                f"    static const std::vector<std::string> parameter_names() {{ return {{ {p_names_str} }}; }}",
                f"    static const std::vector<T> default_parameters() {{ return {{ {p_vals_str} }}; }}",
            ]
        )
        return "\n".join(lines)

    def _detect_has_diffusion(self):
        """Check whether the model's diffusion_matrix is non-trivial (not all zeros).

        Declarative models keep ``diffusion_matrix`` on the SystemModel
        (``None`` when the model has no diffusion); legacy Models expose it
        as a ``functions`` entry.
        """
        if self.sm is not None:
            expr = getattr(self.sm, "diffusion_matrix", None)
            if expr is None:
                return False
        else:
            if "diffusion_matrix" not in self.model.functions.keys():
                return False
            expr = self.model.functions.diffusion_matrix.definition
        # Flatten the expression and check if every element is zero
        if hasattr(expr, "__iter__"):
            return not all(sp.simplify(e) == 0 for e in sp.flatten(expr))
        return sp.simplify(expr) != 0

    def _n_dof_gradq(self):
        """Number of gradient-variable dof, or 0 when the model has no
        ``gradient_variables`` (current declarative models)."""
        gv = getattr(self.model, "gradient_variables", None)
        if gv is None:
            return 0
        if hasattr(gv, "length"):
            return gv.length()
        return len(gv)

    def _state_names(self):
        """Names of the emitted state slots (SystemModel state when present,
        else the Model's own conserved variables)."""
        if self.sm is not None:
            return [str(s) for s in self.sm.state]
        return list(self.model.variables.keys())

    def _var_index(self, name):
        """Return the index of a state variable by name, or -1 if not found."""
        keys = self._state_names()
        return keys.index(name) if name in keys else -1

    def _detect_has_free_surface(self):
        """Check whether the model has free-surface variables (h and b).

        Free-surface models (SWE, SME, etc.) are identified by the presence
        of both 'h' (water depth) and 'b' (bathymetry) in either the
        conserved or auxiliary variable sets.
        """
        all_vars = set(self._state_names())
        if self.sm is not None:
            all_vars |= {str(s) for s in self.sm.aux_state}
        elif hasattr(self.model, "aux_variables"):
            all_vars |= set(self.model.aux_variables.keys())
        return "h" in all_vars and "b" in all_vars

    def get_file_footer(self):
        """Get file footer."""
        return "};\n"

    def get_bc_args(self):
        """Hook to define boundary condition arguments.

        ``p`` (the model parameter vector) is part of the interface so
        parameter-dependent boundary conditions lower correctly.
        """
        return "const int bc_idx,\n        const T* Q,\n        const T* Qaux,\n        const T* p,\n        const T* n,\n        const T* X,\n        const T time,\n        const T dX"


# =========================================================================
#  4. GENERIC NUMERICS (Clean)
# =========================================================================


class GenericCppNumerics(GenericCppBase):
    """GenericCppNumerics. (class)."""
    _wrapper_name = "Numerics"

    def __init__(self, numerics, gpu_enabled=False, *args, **kwargs):
        """Initialize the instance."""
        super().__init__(*args, **kwargs)
        self.numerics = numerics
        self.model = numerics.model               # a SystemModel
        self.gpu_enabled = gpu_enabled
        self.n_dof_q = self.model.n_equations
        self.n_dof_qaux = len(self.model.aux_state)
        self.register_map("Q", self.model.state)
        self.register_map("Qaux", self.model.aux_state)
        self.register_map("n", self.model.normal.values())
        self.register_map("Q_minus", numerics.variables_minus)
        self.register_map("Q_plus", numerics.variables_plus)
        self.register_map("Qaux_minus", numerics.aux_variables_minus)
        self.register_map("Qaux_plus", numerics.aux_variables_plus)
        self.register_map("flux_minus", numerics.flux_minus)
        self.register_map("flux_plus", numerics.flux_plus)
        self.register_map("source_term", numerics.source_term)
        self.register_map("p", self.model.parameters.values())

    @classmethod
    def write_code(cls, numerics, settings, filename="Numerics.H", gpu_enabled=False):
        """Write code."""
        printer = cls(numerics, gpu_enabled=gpu_enabled)
        code = printer.create_code()
        return cls._write_file(code, settings, filename)

    def create_code(self):
        """Create code."""
        blocks = [self.get_file_header()]
        for name, func_obj in self.numerics.functions.items():
            blocks.extend(self._process_kernel_from_function(func_obj))
        blocks.append(self.get_file_footer())
        return "\n".join(blocks)

    def get_file_header(self):
        """Get file header."""
        tpl = "template <typename T>" if self._is_template_class else ""
        lines = [
            "#pragma once",
            self.get_includes().strip(),
            '#include "Model.H"',
            self.get_simple_array_def(),
            "#include <vector>",
            "#include <algorithm>",
            "",
        ]
        if self.gpu_enabled:
            lines.extend(
                [
                    "#ifdef __CUDACC__",
                    "#define PORTABLE_FN __host__ __device__",
                    "#else",
                    "#define PORTABLE_FN",
                    "#endif",
                    "",
                ]
            )
        else:
            lines.append("#define PORTABLE_FN")
        lines.extend(
            [
                tpl,
                f"struct {self._wrapper_name} {{",
                f"    static constexpr int n_dof_q = {self.n_dof_q};",
            ]
        )
        return "\n".join(lines)

    def get_file_footer(self):
        """Get file footer."""
        return "};\n"


# =========================================================================
#  5. WRAPPERS
# =========================================================================


class CppModel(GenericCppModel):
    """CppModel. (class)."""
    _output_subdir = ".c_interface"
    _is_template_class = True

    def __init__(self, model, *args, **kwargs):
        """Initialize the instance."""
        super().__init__(model, *args, **kwargs)
        self.real_type = "T"
        self.math_namespace = "std::"


class CppNumerics(GenericCppNumerics):
    """CppNumerics. (class)."""
    _output_subdir = ".c_interface"
    _is_template_class = True

    def __init__(self, numerics, *args, **kwargs):
        """Initialize the instance."""
        super().__init__(numerics, *args, **kwargs)
        self.real_type = "T"
        self.math_namespace = "std::"
        self.gpu_enabled = True
