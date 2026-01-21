import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter
import re
import os
import itertools
import textwrap
from zoomy_core.misc import misc as misc


# =========================================================================
#  1. HELPER FUNCTIONS (Pure Logic)
# =========================================================================


def flatten_index(indices, shape):
    """
    Converts N-dimensional tuple index to 1D flat index (Row-Major).
    Example: shape=(3,2), index=(1,1) -> 1*2 + 1 = 3
    """
    flat_idx = 0
    stride = 1
    for i, size in zip(reversed(indices), reversed(shape)):
        flat_idx += i * stride
        stride *= size
    return flat_idx


def get_nested_shape(expr):
    """
    Robustly determines the shape of a SymPy expression, handling
    Piecewise branches and Lists.
    """
    shape = (1,)

    # 1. Check direct attributes
    if hasattr(expr, "shape"):
        shape = expr.shape
    elif hasattr(expr, "tomatrix"):
        shape = expr.shape

    # 2. Unwrap Zoomy definitions
    if hasattr(expr, "definition"):
        expr = expr.definition

    # 3. Detect shape from Piecewise branches if top-level is scalar
    if shape == (1,) and isinstance(expr, sp.Piecewise):
        try:
            # Peek at the first branch
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
    """
    The Core C++ Translation Engine.

    Features:
    - Module-based Function Dispatch (c_functions dict)
    - Automatic Vector Conditional Expansion
    - CSE Optimization
    """

    _output_subdir = "cpp_interface"
    _wrapper_name = "BaseWrapper"
    _is_template_class = False

    gpu_enabled = True
    real_type = "double"
    math_namespace = "std::"

    # --- THE MODULE CATALOG (Unified Solution) ---
    # Define handlers that take (printer, *args) and return a C++ string.
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
        self.symbol_maps = []  # List of dicts to resolve symbols
        self._std_regex = re.compile(r"std::([A-Za-z_]\w*)")

    # --- Abstract Interface ---
    def get_includes(self):
        raise NotImplementedError

    def format_accessor(self, var_name, index):
        raise NotImplementedError

    def format_assignment(self, target_name, indices, value, shape):
        raise NotImplementedError

    def get_variable_declaration(self, variable_name):
        raise NotImplementedError

    def wrap_function_signature(self, name, args_str, body_str, shape):
        raise NotImplementedError

    # --- Symbol Resolution ---
    def register_map(self, name, keys):
        new_map = {k: self.format_accessor(name, i) for i, k in enumerate(keys)}
        self.symbol_maps.append(new_map)
        return new_map

    def _print_Symbol(self, s):
        for m in self.symbol_maps:
            if s in m:
                return m[s]
        return super()._print_Symbol(s)

    # --- Unified Function Dispatcher ---
    def _print_Function(self, expr):
        """
        Looks up the function name in self.c_functions.
        If found, executes the handler. Otherwise, falls back to default C++ call.
        """
        name = expr.func.__name__

        if name in self.c_functions:
            handler = self.c_functions[name]
            # Pass (self, *args) to the handler
            return handler(self, *expr.args)

        # Default: Print as standard C++ function call: name(arg1, arg2)
        return f"{self._print(expr.func)}({', '.join(map(self._print, expr.args))})"

    # --- Helper for Min/Max (Recursive) ---
    def _print_min_max(self, func_name, args):
        if len(args) == 1:
            return self._print(args[0])
        if len(args) == 2:
            return f"{self.math_namespace}{func_name}({self._print(args[0])}, {self._print(args[1])})"
        # Recurse for >2 args: min(a, b, c) -> min(a, min(b, c))
        # We assume the symbolic function allows reconstruction via func(*rest)
        # For simple recursion we can just chain strings since we know the operator
        arg0 = self._print(args[0])
        rest = self._print_min_max(func_name, args[1:])
        return f"{self.math_namespace}{func_name}({arg0}, {rest})"

    # --- Smart Expansion (Vector Logic Fix) ---
    def _expand_vector_conditionals(self, expr):
        """
        Detects 'conditional(c, VecA, VecB)' and transforms it into
        'Matrix([conditional(c, a1, b1), conditional(c, a2, b2), ...])'.
        This allows the C++ printer to generate valid ternary operators for scalars.
        """
        # 1. Handle Lists/Arrays/Matrices recursively
        if isinstance(expr, (list, tuple)):
            return [self._expand_vector_conditionals(e) for e in expr]

        if hasattr(expr, "__getitem__") and not isinstance(expr, sp.Symbol):
            # It's a SymPy Matrix/Array - iterate and expand elements
            if hasattr(expr, "reshape") and hasattr(expr, "shape"):
                flat_args = [self._expand_vector_conditionals(e) for e in expr]
                return sp.Array(flat_args).reshape(*expr.shape)
            return [self._expand_vector_conditionals(e) for e in expr]

        # 2. Handle the 'conditional' Function specifically
        if isinstance(expr, sp.Function) and expr.func.__name__ == "conditional":
            cond, true_val, false_val = expr.args

            # Check if branches are vectors
            is_vec_t = hasattr(true_val, "__getitem__") and not isinstance(
                true_val, sp.Symbol
            )
            is_vec_f = hasattr(false_val, "__getitem__") and not isinstance(
                false_val, sp.Symbol
            )

            if is_vec_t or is_vec_f:
                t_list = list(true_val) if is_vec_t else [true_val]
                f_list = list(false_val) if is_vec_f else [false_val]

                # Expand element-wise
                expanded_list = []
                import itertools

                for t, f in itertools.zip_longest(t_list, f_list, fillvalue=0):
                    t_ex = self._expand_vector_conditionals(t)
                    f_ex = self._expand_vector_conditionals(f)
                    expanded_list.append(sp.Function("conditional")(cond, t_ex, f_ex))

                # Return wrapped in Array to preserve shape
                shape = (
                    true_val.shape
                    if hasattr(true_val, "shape")
                    else (len(expanded_list),)
                )
                return sp.Array(expanded_list).reshape(*shape)

        return expr

    # --- Core Generation Logic ---
    def convert_expression_body(self, expr, shape, target="res"):
        if isinstance(expr, sp.Piecewise):
            return self._print_piecewise_structure(expr, shape, target)

        if hasattr(expr, "__iter__") and not isinstance(expr, sp.Matrix):
            flat_expr = list(sp.flatten(expr))
        elif isinstance(expr, sp.Matrix):
            flat_expr = list(expr)
        else:
            flat_expr = [expr]

        tmp_sym = sp.numbered_symbols("t")
        temps, simplified_flat = sp.cse(flat_expr, symbols=tmp_sym)
        lines = []

        for lhs, rhs in temps:
            lines.append(f"{self.real_type} {self.doprint(lhs)} = {self.doprint(rhs)};")

        result_array = sp.Array(simplified_flat).reshape(*shape)
        ranges = [range(s) for s in shape]

        for indices in itertools.product(*ranges):
            val = self.doprint(result_array[indices])
            lines.append(self.format_assignment(target, indices, val, shape))

        return "\n".join(["    " + line for line in lines])

    def _print_piecewise_structure(self, expr, shape, target):
        lines = []
        for i, arg in enumerate(expr.args):
            val = arg.expr if hasattr(arg, "expr") else arg[0]
            cond = arg.cond if hasattr(arg, "cond") else arg[1]
            cond_str = self.doprint(cond)

            if i == 0:
                lines.append(f"    if ({cond_str}) {{")
            elif cond == True or cond == sp.true:
                lines.append("    } else {")
            else:
                lines.append(f"    }} else if ({cond_str}) {{")

            branch_body = self.convert_expression_body(val, shape, target)
            lines.append(textwrap.indent(branch_body, "    "))

        lines.append("    }")
        return "\n".join(lines)

    def _process_kernel(self, name, expr, required_vars):
        """
        Handles expressions (Scalar or Vector).
        Generates a SINGLE function writing to 'res'.
        """
        # 1. Expand Vector Conditionals (Flip conditional(vec) -> vec(conditional))
        expr = self._expand_vector_conditionals(expr)

        # 2. Unify Input: Ensure we have a SymPy Array/Matrix structure
        # If it's a raw Python list, convert it to a SymPy Array so get_nested_shape works
        if isinstance(expr, (list, tuple)):
            import sympy as sp
            expr = sp.Array(expr)
        
        # 3. Determine Shape
        # get_nested_shape works on SymPy objects (Matrix, Array, Piecewise)
        shape, expr = get_nested_shape(expr)

        # 4. Generate Body
        # convert_expression_body handles flattening, CSE, and writing assignments based on 'shape'
        body = self.convert_expression_body(expr, shape)

        # 5. Generate Signature
        decls = [self.get_variable_declaration(v) for v in required_vars]
        args_str = ",\n        ".join([d for d in decls if d])

        return [self.wrap_function_signature(name, args_str, body, shape)]

    def _process_scalar_kernel(self, name, expression, required_vars):
        expression = self._expand_vector_conditionals(expression)
        import sympy as sp

        expr_array = sp.Array([expression])
        shape = (1,)
        body = self.convert_expression_body(expr_array, shape)
        decls = [self.get_variable_declaration(v) for v in required_vars]
        args_str = ",\n        ".join([d for d in decls if d])
        return [self.wrap_function_signature(name, args_str, body, shape)]

    # --- Printers ---
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

    # --- File IO ---
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
#  3. GENERIC MODEL (The Physics)
# =========================================================================


class GenericCppModel(GenericCppBase):
    _wrapper_name = "Model"

    KERNEL_ARGUMENTS = {
        "physics": ["Q", "Qaux", "res"],
        "geometric": ["Q", "Qaux", "n", "res"],
        "interpolate": ["Q", "Qaux", "X", "res"],
        "boundary": ["bc_idx", "Q", "Qaux", "n", "X", "time", "dX", "res"],
        "residual": ["Q", "Qaux", "X", "time", "dX", "res"],
    }

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.n_dof_q = model.n_variables
        self.n_dof_qaux = model.n_aux_variables

        self.register_map("Q", model.variables.values())
        self.register_map("Qaux", model.aux_variables.values())
        self.register_map("n", model.normal.values())
        self.symbol_maps.append(
            {
                k: str(float(model.parameter_values[i]))
                for i, k in enumerate(model.parameters.values())
            }
        )
        if hasattr(model, "position"):
            self.register_map("X", model.position.values())

    @classmethod
    def write_code(cls, model, settings, filename="Model.H"):
        printer = cls(model)
        code = printer.create_code()
        return cls._write_file(code, settings, filename)

    def create_code(self):
        blocks = [self.get_file_header()]

        # Physics Kernels
        for name, expr in [
            ("flux", self.model.flux()),
            ("dflux", self.model.dflux()),
            ("nonconservative_matrix", self.model.nonconservative_matrix()),
            ("quasilinear_matrix", self.model.quasilinear_matrix()),
            ("source", self.model.source()),
            (
                "source_jacobian_wrt_variables",
                self.model.source_jacobian_wrt_variables(),
            ),
            (
                "source_jacobian_wrt_aux_variables",
                self.model.source_jacobian_wrt_aux_variables(),
            ),
        ]:
            blocks.extend(
                self._process_kernel(name, expr, self.KERNEL_ARGUMENTS["physics"])
            )

        # Geometric Kernels
        for name, expr in [
            ("eigenvalues", self.model.eigenvalues()),
            ("left_eigenvectors", self.model.left_eigenvectors()),
            ("right_eigenvectors", self.model.right_eigenvectors()),
        ]:
            blocks.extend(
                self._process_kernel(name, expr, self.KERNEL_ARGUMENTS["geometric"])
            )

        # Residual & Interpolate
        blocks.extend(
            self._process_kernel(
                "residual", self.model.residual(), self.KERNEL_ARGUMENTS["residual"]
            )
        )
        blocks.extend(
            self._process_kernel(
                "interpolate",
                self.model.project_2d_to_3d(),
                self.KERNEL_ARGUMENTS["interpolate"],
            )
        )

        # Boundary
        bc_wrapper = self.model.boundary_conditions.get_boundary_condition_function(
            self.model.time,
            self.model.position,
            self.model.distance,
            self.model.variables,
            self.model.aux_variables,
            self.model.parameters,
            self.model.normal,
        )
        blocks.extend(
            self._process_kernel(
                "boundary_conditions", bc_wrapper, self.KERNEL_ARGUMENTS["boundary"]
            )
        )

        blocks.append(self.get_file_footer())
        return "\n".join(blocks)

    def get_file_header(self):
        bc_names = sorted(
            self.model.boundary_conditions.boundary_conditions_list_dict.keys()
        )
        bc_str = ", ".join(f'"{item}"' for item in bc_names)
        tpl = "template <typename T>" if self._is_template_class else ""
        return f"""#pragma once
{self.get_includes().strip()}
#include <vector>
#include <string>
#include <algorithm>

{tpl}
struct {self._wrapper_name} {{
    // --- Constants ---
    static constexpr int n_dof_q    = {self.n_dof_q};
    static constexpr int n_dof_qaux = {self.n_dof_qaux};
    static constexpr int dimension  = {self.model.dimension};
    static constexpr int n_boundary_tags = {len(bc_names)};

    // --- Helpers ---
    static const std::vector<std::string> get_boundary_tags() {{ return {{ {bc_str} }}; }}

    // --- Kernels ---"""

    def get_file_footer(self):
        return "};\n"


# =========================================================================
#  4. GENERIC NUMERICS (The Solver)
# =========================================================================


class GenericCppNumerics(GenericCppBase):
    _wrapper_name = "Numerics"

    KERNEL_ARGUMENTS = {
        "numerical_flux": ["Q_minus", "Q_plus", "Qaux_minus", "Qaux_plus", "n", "res"],
        "local_max_abs_eigenvalue": ["Q", "Qaux", "n", "res"],
        "update_Q": ["Q", "Qaux", "res"],
        "update_Qaux": ["Q", "Qaux", "res"],
    }

    def __init__(self, numerics, gpu_enabled=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numerics = numerics
        self.model = numerics.model
        self.gpu_enabled = gpu_enabled

        self.n_dof_q = self.model.n_variables
        self.n_dof_qaux = self.model.n_aux_variables

        self.register_map("Q", self.model.variables.values())
        self.register_map("Qaux", self.model.aux_variables.values())
        self.register_map("n", self.model.normal.values())
        self.register_map("Q_minus", numerics.variables_minus)
        self.register_map("Q_plus", numerics.variables_plus)
        self.register_map("Qaux_minus", numerics.aux_variables_minus)
        self.register_map("Qaux_plus", numerics.aux_variables_plus)
        self.symbol_maps.append(
            {
                k: str(float(self.model.parameter_values[i]))
                for i, k in enumerate(self.model.parameters.values())
            }
        )

    @classmethod
    def write_code(cls, numerics, settings, filename="Numerics.H", gpu_enabled=False):
        printer = cls(numerics, gpu_enabled=gpu_enabled)
        code = printer.create_code()
        return cls._write_file(code, settings, filename)

    def create_code(self):
        blocks = [self.get_file_header()]
        blocks.extend(
            self._process_kernel(
                "numerical_flux",
                self.numerics.numerical_flux(),
                self.KERNEL_ARGUMENTS["numerical_flux"],
            )
        )

        blocks.extend(
            self._process_scalar_kernel(
                "local_max_abs_eigenvalue",
                self.numerics.local_max_abs_eigenvalue(),
                self.KERNEL_ARGUMENTS["local_max_abs_eigenvalue"],
            )
        )
        
        blocks.extend(
            self._process_kernel(
                "update_Q",
                self.numerics.update_q(),
                self.KERNEL_ARGUMENTS["update_Q"],
            )
        )
        
        blocks.extend(
            self._process_kernel(
                "update_Qaux",
                self.numerics.update_qaux(),
                self.KERNEL_ARGUMENTS["update_Qaux"],
            )
        )

        blocks.append(self.get_file_footer())
        return "\n".join(blocks)

    def get_file_header(self):
        tpl = "template <typename T>" if self._is_template_class else ""
        return f"""#pragma once
{self.get_includes().strip()}
#include <vector>
#include <algorithm>

{tpl}
struct {self._wrapper_name} {{
    // --- Numerics Constants ---
    static constexpr int n_dof_q = {self.n_dof_q};

    // --- Kernels ---"""

    def get_file_footer(self):
        return "};\n"
