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
    flat_idx = 0
    stride = 1
    for i, size in zip(reversed(indices), reversed(shape)):
        flat_idx += i * stride
        stride *= size
    return flat_idx


def get_nested_shape(expr):
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
#  2. GENERIC BASE (The Engine)
# =========================================================================


class GenericCppBase(CXX11CodePrinter):
    _output_subdir = "cpp_interface"
    _wrapper_name = "BaseWrapper"
    _is_template_class = False
    gpu_enabled = True
    real_type = "double"
    math_namespace = "std::"

    ARG_MAPPING = {
        "variables": "Q",
        "aux_variables": "Qaux",
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
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbol_maps = []
        self._std_regex = re.compile(r"std::([A-Za-z_]\w*)")

    def register_map(self, name, keys):
        new_map = {k: self.format_accessor(name, i) for i, k in enumerate(keys)}
        self.symbol_maps.append(new_map)
        return new_map

    def _print_Symbol(self, s):
        for m in self.symbol_maps:
            if s in m:
                return m[s]
        return super()._print_Symbol(s)

    def _print_Function(self, expr):
        name = expr.func.__name__
        if name in self.c_functions:
            return self.c_functions[name](self, *expr.args)
        return f"{self._print(expr.func)}({', '.join(map(self._print, expr.args))})"

    def _print_IndexedBase(self, expr):
        return self._print(expr.label) if hasattr(expr, "label") else str(expr)

    def _print_Indexed(self, expr):
        base = self._print(expr.base)
        indices = [self._print(i) for i in expr.indices]
        return f"{base}[{']['.join(indices)}]"

    def _print_min_max(self, func_name, args):
        if len(args) == 1:
            return self._print(args[0])
        if len(args) == 2:
            return f"{self.math_namespace}{func_name}({self._print(args[0])}, {self._print(args[1])})"
        arg0 = self._print(args[0])
        rest = self._print_min_max(func_name, args[1:])
        return f"{self.math_namespace}{func_name}({arg0}, {rest})"

    def _expand_vector_conditionals(self, expr):
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
        flat = []
        for e in expr_list:
            if isinstance(e, (list, tuple, sp.NDimArray)):
                flat.extend(self._flatten_ragged_list(e))
            else:
                flat.append(e)
        return flat

    def _optimize_array_calls(self, expr_list):
        definitions = []
        call_cache = {}

        def replace_logic(node):
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

    def convert_expression_body(self, expr, shape, target="res"):
        if isinstance(expr, sp.Piecewise):
            return self._print_piecewise_structure(expr, shape, target)
        flat_expr = (
            list(sp.flatten(expr))
            if hasattr(expr, "__iter__") and not isinstance(expr, sp.Matrix)
            else list(expr)
            if isinstance(expr, sp.Matrix)
            else [expr]
        )

        # 1. Extract Calls
        call_defs, optim_exprs = self._optimize_array_calls(flat_expr)

        # 2. Extract Arguments
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

        # --- STAGE A: COMPUTE INPUTS ---
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
                init_str = ", ".join([self.doprint(e) for e in elements])
                lines.append(
                    f"{self.real_type} {self.doprint(sym)}[] = {{ {init_str} }};"
                )

        # --- STAGE B: EXECUTE CALLS ---
        for sym, call in clean_call_defs:
            lines.append(f"auto {self.doprint(sym)} = {self.doprint(call)};")

        # --- STAGE C: COMPUTE OUTPUTS ---
        if optim_exprs:
            temps_res, simplified_res = sp.cse(optim_exprs, symbols=tmp_sym_gen)
            for lhs, rhs in temps_res:
                lines.append(
                    f"{self.real_type} {self.doprint(lhs)} = {self.doprint(rhs)};"
                )

            total_size = 1
            for s in shape:
                total_size *= s
            lines.append(f"SimpleArray<T, {total_size}> {target};")

            result_array = sp.Array(simplified_res).reshape(*shape)
            ranges = [range(s) for s in shape]
            for indices in itertools.product(*ranges):
                val = self.doprint(result_array[indices])
                idx = flatten_index(indices, shape)
                lines.append(f"{target}[{idx}] = {val};")
        else:
            total_size = 1
            for s in shape:
                total_size *= s
            lines.append(f"SimpleArray<T, {total_size}> {target} = {{0}};")

        lines.append(f"return {target};")
        return "\n".join(["    " + line for line in lines])

    def _print_piecewise_structure(self, expr, shape, target):
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
        lines.append(f"    SimpleArray<T, {total_size}> default_{target} = {{0}};")
        lines.append(f"    return default_{target};")
        return "\n".join(lines)

    def _generate_signature_from_function(self, func_obj):
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
        name = func_obj.name
        expr = func_obj.definition
        expr = self._expand_vector_conditionals(expr)
        if isinstance(expr, (list, tuple)):
            # FIX: Ensure we have a flat list before creating Array to avoid ambiguous shapes
            expr = self._flatten_ragged_list(expr)
            expr = sp.Array(expr)
        shape, expr = get_nested_shape(expr)
        body = self.convert_expression_body(expr, shape)
        args_str = self._generate_signature_from_function(func_obj)
        return [self.wrap_function_signature(name, args_str, body, shape)]

    def get_includes(self):
        return """#include <cmath>
#include <array>
"""

    def get_simple_array_def(self):
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
        return f"{var_name}[{index}]"

    def format_assignment(self, target_name, indices, value, shape):
        idx = flatten_index(indices, shape)
        return f"{target_name}[{idx}] = {value};"

    def get_variable_declaration(self, variable_name):
        return f"const T* {variable_name}"

    def wrap_function_signature(self, name, args_str, body_str, shape):
        qualifier = "PORTABLE_FN " if self.gpu_enabled else ""
        total_size = 1
        for s in shape:
            total_size *= s
        return f"""    {qualifier}static inline SimpleArray<T, {total_size}> {name}(
        {args_str})
    {{
{body_str}
    }}
"""

    def _print_Pow(self, expr):
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
        code = super().doprint(expr, **settings)
        if self.math_namespace != "std::":

            def _repl(match):
                return f"{self.math_namespace}{match.group(1)}"

            return self._std_regex.sub(_repl, code)
        return code

    @classmethod
    def _write_file(cls, code, settings, filename):
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
#  3. GENERIC MODEL
# =========================================================================


class GenericCppModel(GenericCppBase):
    _wrapper_name = "Model"

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.n_dof_q = model.n_variables
        self.n_dof_qaux = model.n_aux_variables
        self.register_map("Q", model.variables.values())
        self.register_map("Qaux", model.aux_variables.values())
        self.register_map("n", model.normal.values())
        if hasattr(model, "position"):
            self.register_map("X", model.position.values())
        self.symbol_maps.append(
            {k: f"p[{i}]" for i, k in enumerate(model.parameters.values())}
        )

    @classmethod
    def write_code(cls, model, settings, filename="Model.H"):
        printer = cls(model)
        code = printer.create_code()
        return cls._write_file(code, settings, filename)

    def create_code(self):
        blocks = [self.get_file_header()]
        for name, func_obj in self.model.functions.items():
            blocks.extend(self._process_kernel_from_function(func_obj))

        bc_wrapper = self.model.boundary_conditions.get_boundary_condition_function(
            self.model.time,
            self.model.position,
            self.model.distance,
            self.model.variables,
            self.model.aux_variables,
            self.model.parameters,
            self.model.normal,
        )
        bc_args = "const int bc_idx,\n        const T* Q,\n        const T* Qaux,\n        const T* n,\n        const T* X,\n        const T time,\n        const T dX"
        shape, expr = get_nested_shape(bc_wrapper.definition)
        body = self.convert_expression_body(expr, shape)
        blocks.extend(
            [self.wrap_function_signature("boundary_conditions", bc_args, body, shape)]
        )
        blocks.append(self.get_file_footer())
        return "\n".join(blocks)

    def get_file_header(self):
        bc_names = sorted(
            self.model.boundary_conditions.boundary_conditions_list_dict.keys()
        )
        bc_str = ", ".join(f'"{item}"' for item in bc_names)

        # --- Extract Parameter Metadata ---
        # 1. Get ordered list of keys from the symbolic dictionary (used for p[i] indexing)
        param_keys = list(self.model.parameters.keys())

        # 2. Retrieve default values from the class param definition
        # Assumes model.param.parameters.default is available and populated
        defaults = self.model.param.parameters.default
        if callable(defaults):
            defaults = defaults()

        param_vals = []
        for k in param_keys:
            if k not in defaults:
                # Fallback or Error: Ideally this shouldn't happen if structure is sound
                raise ValueError(
                    f"C++ Gen Error: Parameter '{k}' found in symbols but missing default value in param definition."
                )

            val = defaults[k]
            # Handle tuple format e.g. (9.81, "positive") -> extract value
            if isinstance(val, (list, tuple)):
                val = val[0]

            param_vals.append(str(val))

        p_names_str = ", ".join(f'"{k}"' for k in param_keys)
        p_vals_str = ", ".join(param_vals)
        # ----------------------------------

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

        lines.extend(
            [
                tpl,
                f"struct {self._wrapper_name} {{",
                f"    static constexpr int n_dof_q    = {self.n_dof_q};",
                f"    static constexpr int n_dof_qaux = {self.n_dof_qaux};",
                f"    static constexpr int dimension  = {self.model.dimension};",
                f"    static constexpr int n_boundary_tags = {len(bc_names)};",
                f"    static const std::vector<std::string> get_boundary_tags() {{ return {{ {bc_str} }}; }}",
                # New Parameter Accessors
                f"    static const std::vector<std::string> parameter_names() {{ return {{ {p_names_str} }}; }}",
                f"    static const std::vector<T> default_parameters() {{ return {{ {p_vals_str} }}; }}",
            ]
        )
        return "\n".join(lines)

    def get_file_footer(self):
        return "};\n"


# =========================================================================
#  4. GENERIC NUMERICS
# =========================================================================


class GenericCppNumerics(GenericCppBase):
    _wrapper_name = "Numerics"

    def __init__(self, numerics, gpu_enabled=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numerics = numerics
        self.model = numerics.model
        self.gpu_enabled = gpu_enabled
        self.n_dof_q = self.model.n_variables
        self.register_map("Q", self.model.variables.values())
        self.register_map("Qaux", self.model.aux_variables.values())
        self.register_map("n", self.model.normal.values())
        self.register_map("Q_minus", numerics.variables_minus)
        self.register_map("Q_plus", numerics.variables_plus)
        self.register_map("Qaux_minus", numerics.aux_variables_minus)
        self.register_map("Qaux_plus", numerics.aux_variables_plus)
        self.register_map("flux_minus", numerics.flux_minus)
        self.register_map("flux_plus", numerics.flux_plus)
        self.register_map("source_term", numerics.source_term)
        self.symbol_maps.append(
            {k: f"p[{i}]" for i, k in enumerate(self.model.parameters.values())}
        )

    @classmethod
    def write_code(cls, numerics, settings, filename="Numerics.H", gpu_enabled=False):
        printer = cls(numerics, gpu_enabled=gpu_enabled)
        code = printer.create_code()
        return cls._write_file(code, settings, filename)

    def create_code(self):
        blocks = [self.get_file_header()]
        for name, func_obj in self.numerics.functions.items():
            blocks.extend(self._process_kernel_from_function(func_obj))
        blocks.append(self.get_file_footer())
        return "\n".join(blocks)

    def get_file_header(self):
        tpl = "template <typename T>" if self._is_template_class else ""
        lines = [
            "#pragma once",
            self.get_includes().strip(),
            '#include "Model.H"',  # <--- CORRECTLY ADDED INCLUDE HERE
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
        return "};\n"
