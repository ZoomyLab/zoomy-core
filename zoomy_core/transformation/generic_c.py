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
    Responsibilities:
    - Expression traversal (CSE, Flattening)
    - Control Flow generation (If/Else for Piecewise)
    - Formatting hooks (Accessors, Assignments)
    - File IO (Header/Footer construction)
    """

    _output_subdir = "cpp_interface"
    _wrapper_name = "BaseWrapper"
    _is_template_class = False
    
    gpu_enabled=True
    real_type = "double"
    math_namespace = "std::"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbol_maps = []  # List of dicts to resolve symbols
        self._std_regex = re.compile(r"std::([A-Za-z_]\w*)")

    # --- Abstract Interface (Backends must implement) ---
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
        """Creates a map {Symbol: "AccessorString"} and adds it to resolution list."""
        new_map = {k: self.format_accessor(name, i) for i, k in enumerate(keys)}
        self.symbol_maps.append(new_map)
        return new_map

    def _print_Symbol(self, s):
        for m in self.symbol_maps:
            if s in m:
                return m[s]
        return super()._print_Symbol(s)

    # --- Core Generation Logic ---
    def convert_expression_body(self, expr, shape, target="res"):
        # 1. Control Flow
        if isinstance(expr, sp.Piecewise):
            return self._print_piecewise_structure(expr, shape, target)

        # 2. Flatten for CSE
        if hasattr(expr, "__iter__") and not isinstance(expr, sp.Matrix):
            flat_expr = list(sp.flatten(expr))
        elif isinstance(expr, sp.Matrix):
            flat_expr = list(expr)
        else:
            flat_expr = [expr]

        # 3. CSE Optimization
        tmp_sym = sp.numbered_symbols("t")
        temps, simplified_flat = sp.cse(flat_expr, symbols=tmp_sym)
        lines = []

        for lhs, rhs in temps:
            lines.append(f"{self.real_type} {self.doprint(lhs)} = {self.doprint(rhs)};")

        # 4. Reconstruction & Assignment
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
        """Recursively handles lists of expressions (e.g., [flux_x, flux_y])."""
        if isinstance(expr, list):
            results = []
            dims = ["x", "y", "z"]
            for i, sub_expr in enumerate(expr):
                suffix = f"_{dims[i]}" if i < 3 else f"_{i}"
                results.extend(
                    self._process_kernel(f"{name}{suffix}", sub_expr, required_vars)
                )
            return results

        shape, expr = get_nested_shape(expr)
        if isinstance(expr, list):
            expr = sp.Array(expr)

        body = self.convert_expression_body(expr, shape)

        decls = [self.get_variable_declaration(v) for v in required_vars]
        args_str = ",\n        ".join([d for d in decls if d])

        return [self.wrap_function_signature(name, args_str, body, shape)]
    
    def _process_scalar_kernel(self, name, expression, required_vars):
        """
        Generates code for a single scalar expression without appending suffixes.
        """
        # 1. Wrap the scalar expression in a list/Array so it has shape (1,)
        #    This matches how _process_kernel prepares 'expr' before calling convert
        import sympy as sp
        expr_array = sp.Array([expression])
        shape = (1,)

        # 2. Convert Body (using the standard signature: expr, shape)
        #    The base class likely defaults to assigning to 'res' if not specified
        body = self.convert_expression_body(expr_array, shape)
        
        # 3. Generate arguments string
        #    We filter out empty declarations just like _process_kernel does
        decls = [self.get_variable_declaration(v) for v in required_vars]
        args_str = ",\n        ".join([d for d in decls if d])
        
        # 4. Wrap signature using the EXACT name provided (No _x suffix)
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

        # Register Physics Maps
        self.register_map("Q", model.variables.values())
        self.register_map("Qaux", model.aux_variables.values())
        self.register_map("n", model.normal.values())

        # Parameters need special handling (float conversion)
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

        # 1. Physics
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

        # 2. Geometric
        for name, expr in [
            ("eigenvalues", self.model.eigenvalues()),
            ("left_eigenvectors", self.model.left_eigenvectors()),
            ("right_eigenvectors", self.model.right_eigenvectors()),
        ]:
            blocks.extend(
                self._process_kernel(name, expr, self.KERNEL_ARGUMENTS["geometric"])
            )

        # 3. Residual & Interpolation
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

        # 4. Boundary
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

        lines = [
            "#pragma once",
            self.get_includes().strip(),
            "#include <vector>",
            "#include <string>",
            "",
            tpl,
            f"struct {self._wrapper_name} {{",
            "",
            f"    // --- Constants ---",
            f"    static constexpr int n_dof_q    = {self.n_dof_q};",
            f"    static constexpr int n_dof_qaux = {self.n_dof_qaux};",
            f"    static constexpr int dimension  = {self.model.dimension};",
            f"    static constexpr int n_boundary_tags = {len(bc_names)};",
            "",
            f"    // --- Helpers ---",
            f"    static const std::vector<std::string> get_boundary_tags() {{ return {{ {bc_str} }}; }}",
            "",
            f"    // --- Kernels ---",
        ]
        return "\n".join(lines)

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
        "dt": ["Q", "Qaux", "h", "cfl", "res"],
        "update": ["Q", "Qaux", "res"],
    }

    def __init__(self, numerics, gpu_enabled=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numerics = numerics
        self.model = numerics.model  # Reference to physics model
        self.gpu_enabled = gpu_enabled # Store GPU flag
        
        self.n_dof_q = self.model.n_variables
        self.n_dof_qaux = self.model.n_aux_variables

        # Register Model Maps (for shared concepts like Normal vector)
        self.register_map("Q", self.model.variables.values())
        self.register_map("Qaux", self.model.aux_variables.values())
        self.register_map("n", self.model.normal.values())

        # Register Numerics Maps (Left/Right states)
        self.register_map("Q_minus", numerics.variables_minus)
        self.register_map("Q_plus", numerics.variables_plus)
        self.register_map("Qaux_minus", numerics.aux_variables_minus)
        self.register_map("Qaux_plus", numerics.aux_variables_plus)

        # Parameters
        self.symbol_maps.append(
            {
                k: str(float(self.model.parameter_values[i]))
                for i, k in enumerate(self.model.parameters.values())
            }
        )

    @classmethod
    def write_code(cls, numerics, settings, filename="Numerics.H", gpu_enabled=False):
        # We pass gpu_enabled to the constructor here
        printer = cls(numerics, gpu_enabled=gpu_enabled)
        code = printer.create_code()
        return cls._write_file(code, settings, filename)

    def create_code(self):
        blocks = [self.get_file_header()]

        # 1. Numerical Flux
        expr_flux = self.numerics.numerical_flux()
        blocks.extend(
            self._process_kernel(
                "numerical_flux", expr_flux, self.KERNEL_ARGUMENTS["numerical_flux"]
            )
        )

        # 2. Local Max Abs Eigenvalue
        # We need the symbolic variables from the model to pass to the numerics method
        q_sym = list(self.model.variables.values())
        aux_sym = list(self.model.aux_variables.values())
        n_sym = list(self.model.normal.values())

        # Evaluate the symbolic expression
        expr_lambda = self.numerics.local_max_abs_eigenvalue(q_sym, aux_sym, n_sym)

        # Wrap result in a list because _process_kernel expects iterable return values (like a vector)
        blocks.extend(
            self._process_scalar_kernel(
                "local_max_abs_eigenvalue", 
                expr_lambda, 
                self.KERNEL_ARGUMENTS["local_max_abs_eigenvalue"]
            )
        )

        blocks.append(self.get_file_footer())
        return "\n".join(blocks)

    def get_file_header(self):
        tpl = "template <typename T>" if self._is_template_class else ""
        lines = [
            "#pragma once",
            self.get_includes().strip(),
            "#include <vector>",
            "",
            tpl,
            f"struct {self._wrapper_name} {{",
            "",
            f"    // --- Numerics Constants ---",
            f"    static constexpr int n_dof_q = {self.n_dof_q};",
            "",
            f"    // --- Kernels ---",
        ]
        return "\n".join(lines)

    def get_file_footer(self):
        return "};\n"