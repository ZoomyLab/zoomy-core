"""Module `zoomy_core.transformation.to_amrex`."""

from zoomy_core.transformation.generic_c import (
    GenericCppModel,
    GenericCppNumerics,
    flatten_index,
)
import functools


class AmrexCore:
    """
    Provides all AMReX-specific syntax rules, data types, and macros.
    Designed to be mixed in with GenericCppModel or GenericCppNumerics.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the instance."""
        self.real_type = "amrex::Real"
        self.math_namespace = "amrex::Math::"
        super().__init__(*args, **kwargs)

    def _print_Indexed(self, expr):
        """Forces SymPy Indexed objects to use AMReX matrix indexing."""
        base = self._print(expr.base)
        indices = [self._print(i) for i in expr.indices]
        # We assume 1D vectors mapped to column matrices
        return f"{base}({indices[0]}, 0)"

    def _print_Function(self, expr):
        """AST node visitor to properly format function calls."""
        name = expr.func.__name__

        # 1. Handle pre-registered C functions (Min, Max, conditional, etc.)
        if name in self.c_functions:
            return self.c_functions[name](self, *expr.args)

        # 2. Since AMReX wrappers are not templated, safely strip the <T>
        #    from the function namespace (e.g., 'Model<T>::flux' -> 'Model::flux')
        if not getattr(self, "_is_template_class", True):
            name = name.replace("<T>", "")

        # Print the arguments recursively
        args_str = ", ".join(map(self._print, expr.args))
        return f"{name}({args_str})"
    
    def _print_Symbol(self, s):
        """Overrides symbol printing to match by string name, bypassing SymPy assumptions."""
        s_name = s.name
        
        # --- FIX: Map SymPy's 't' directly to AMReX's 'time' ---
        if s_name == "t":
            return "time"
            
        for m in self.symbol_maps:
            for sym, val in m.items():
                # Compare string names safely
                if hasattr(sym, "name") and sym.name == s_name:
                    return val
                elif str(sym) == s_name:
                    return val
        return super()._print_Symbol(s)

    def doprint(self, expr, **settings):
        """Doprint."""
        code = super().doprint(expr, **settings)
        # AMReX uses std::pow for floating point powers
        code = code.replace("amrex::Math::pow(", "std::pow(")
        # AMReX min/max are in the amrex namespace, not amrex::Math
        code = code.replace("amrex::Math::max(", "amrex::max(")
        code = code.replace("amrex::Math::min(", "amrex::min(")
        # SymPy's Abs generates fabs, AMReX prefers std::abs
        code = code.replace("amrex::Math::fabs(", "std::abs(")
        
        # --- FIX: Map standard trigonometric functions to std:: ---
        code = code.replace("amrex::Math::cos(", "std::cos(")
        code = code.replace("amrex::Math::sin(", "std::sin(")
        code = code.replace("amrex::Math::tan(", "std::tan(")
        code = code.replace("amrex::Math::exp(", "std::exp(")
        
        return code

    def get_includes(self):
        """Get includes."""
        return """#include <AMReX_Array4.H>
#include <AMReX_Vector.H>
#include <AMReX_SmallMatrix.H>"""

    def get_simple_array_def(self):
        # We don't need SimpleArray in AMReX, so we return an empty string.
        """Get simple array def."""
        return ""

    def get_array_type(self, shape):
        """Hook to use amrex::SmallMatrix natively for all mathematical arrays."""
        import functools
        total_size = functools.reduce(lambda x, y: x * y, shape)
        
        # ALWAYS force a flat column vector! 
        # This perfectly safely matches the res(k, 0) SymPy assignment generation.
        return f"amrex::SmallMatrix<{self.real_type},{total_size},1>"

    def get_array_declaration(self, target_name, shape, init_zero=False):
        """Hook to declare amrex::SmallMatrix variables correctly."""
        arr_type = self.get_array_type(shape)
        if init_zero:
            return f"{arr_type} {target_name}{{}};"
        return f"{arr_type} {target_name};"

    def format_accessor(self, var_name, index):
        # Access elements using matrix index notation (row, col)
        """Format accessor."""
        return f"{var_name}({index}, 0)"

    def format_assignment(self, target_name, indices, value, shape):
        """Format assignment."""
        idx = flatten_index(indices, shape)
        return f"{target_name}({idx}, 0) = {value};"

    def format_array_initialization(self, sym_name, elements):
        """Initializes an amrex::SmallMatrix instead of a raw C-array."""
        arr_type = self.get_array_type((len(elements),))
        lines = [f"{arr_type} {sym_name}{{}};"]
        for i, e in enumerate(elements):
            lines.append(f"{sym_name}({i}, 0) = {self.doprint(e)};")
        return "\n".join(lines)

    def _generate_signature_from_function(self, func_obj):
        """Overrides base class to use const references instead of raw pointers."""
        decls = []
        for key, obj in func_obj.args.items():
            cpp_name = self.ARG_MAPPING.get(key, key)

            # Group standard keys used in both Model and Numerics
            if cpp_name in ["Q", "Q_minus", "Q_plus"]:
                t_val = self.get_array_type((self.n_dof_q,))
                decls.append(f"{t_val} const& {cpp_name}")
            elif cpp_name in ["Qaux", "Qaux_minus", "Qaux_plus"]:
                t_val = self.get_array_type((self.n_dof_qaux,))
                decls.append(f"{t_val} const& {cpp_name}")
            elif cpp_name == "n":
                t_val = self.get_array_type((self.model.dimension,))
                decls.append(f"{t_val} const& {cpp_name}")
            elif cpp_name == "X":
                t_val = self.get_array_type((3,))
                decls.append(f"{t_val} const& {cpp_name}")
            elif cpp_name == "p":
                # We dynamically get the size of the parameters array
                p_len = self.model.parameters.length()
                t_val = self.get_array_type((p_len,))
                decls.append(f"{t_val} const& {cpp_name}")
            elif cpp_name in ["time", "dX", "dt", "dx", "bc_idx"]:
                type_prefix = (
                    "const int" if cpp_name == "bc_idx" else f"{self.real_type} const"
                )
                decls.append(f"{type_prefix} {cpp_name}")
            else:
                decls.append(f"{self.real_type} const& {cpp_name}")

        return ",\n        ".join(decls)

    def wrap_function_signature(self, name, args_str, body_str, shape):
        """Overrides the generated function wrapper to include AMReX specific macros."""
        ret_type = self.get_array_type(shape)
        return f"""
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static {ret_type} {name}(
        {args_str}) noexcept
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
            if n < 0:
                return f"(1.0 / amrex::Math::powi<{abs(n)}>({self._print(base)}))"
            return f"amrex::Math::powi<{n}>({self._print(base)})"
        return super()._print_Pow(expr)


# =========================================================================
#  AMREX WRAPPERS
# =========================================================================


class AmrexModel(AmrexCore, GenericCppModel):
    """
    Generates an AMReX compatible Model.H file natively using amrex::SmallMatrix.
    """

    _output_subdir = ".amrex_interface"
    _is_template_class = False

    def get_file_header(self):
        # We inject a 'using T = amrex::Real;' just in case anything falls back to T
        """Get file header."""
        header = super().get_file_header()
        struct_decl = f"struct {self._wrapper_name} {{"
        replacement = f"{struct_decl}\n    using T = {self.real_type};"
        return header.replace(struct_decl, replacement)

    def get_bc_args(self):
        """Get bc args."""
        t_q = self.get_array_type((self.n_dof_q,))
        t_aux = self.get_array_type((self.n_dof_qaux,))
        t_n = self.get_array_type((self.model.dimension,))
        t_x = self.get_array_type((3,))
        return f"const int bc_idx,\n        {t_q} const& Q,\n        {t_aux} const& Qaux,\n        {t_n} const& n,\n        {t_x} const& X,\n        {self.real_type} const time,\n        {self.real_type} const dX"


class AmrexNumerics(AmrexCore, GenericCppNumerics):
    """
    Generates an AMReX compatible Numerics.H file natively using amrex::SmallMatrix.
    """

    _output_subdir = ".amrex_interface"
    _is_template_class = False

    def __init__(self, numerics, *args, **kwargs):
        """Initialize the instance."""
        super().__init__(numerics, *args, **kwargs)
        self.gpu_enabled = True  # Ensure GPU macros are enabled for AMReX numerics

    def get_file_header(self):
        """Get file header."""
        header = super().get_file_header()
        struct_decl = f"struct {self._wrapper_name} {{"
        replacement = f"{struct_decl}\n    using T = {self.real_type};"
        return header.replace(struct_decl, replacement)
