from zoomy_core.transformation.generic_c import GenericCppModel, flatten_index
import functools


class AmrexModel(GenericCppModel):
    _output_subdir = ".amrex_interface"
    _is_template_class = False  # AMReX wrapper is not a template class

    def __init__(self, model, *args, **kwargs):
        self.real_type = "amrex::Real"
        self.math_namespace = "amrex::Math::"
        super().__init__(model, *args, **kwargs)

    def get_includes(self):
        return """#include <AMReX_Array4.H>
#include <AMReX_Vector.H>
#include <AMReX_SmallMatrix.H>"""

    def format_accessor(self, var_name, index):
        return f"{var_name}({index})"

    def format_assignment(self, target_name, indices, value, shape):
        # Handle Standard 1D/2D
        if len(shape) <= 2:
            if len(indices) == 1:
                return f"{target_name}({indices[0]}) = {value};"
            return f"{target_name}({indices[0]},{indices[1]}) = {value};"

        # Handle Rank > 2 (Flatten to 1D vector)
        flat_idx = flatten_index(indices, shape)
        return f"{target_name}({flat_idx}) = {value};"

    def _get_matrix_type(self, shape):
        # Standard cases
        if len(shape) == 1:
            rows, cols = shape[0], 1
        elif len(shape) == 2:
            rows, cols = shape[0], shape[1]
        else:
            # Rank > 2: Flatten into a column vector
            total_size = functools.reduce(lambda x, y: x * y, shape)
            rows, cols = total_size, 1

        return f"amrex::SmallMatrix<{self.real_type},{rows},{cols}>"

    def get_variable_declaration(self, v):
        if v == "res":
            return ""
        t_q = self._get_matrix_type((self.n_dof_q,))
        t_aux = self._get_matrix_type((self.n_dof_qaux,))
        t_n = self._get_matrix_type((self.model.dimension,))
        t_x = self._get_matrix_type((3,))

        mapping = {
            "Q": f"{t_q} const& Q",
            "Qaux": f"{t_aux} const& Qaux",
            "n": f"{t_n} const& n",
            "X": f"{t_x} const& X",
            "time": f"{self.real_type} const& time",
            "dX": f"{self.real_type} const& dX",
            "bc_idx": "const int bc_idx",
        }
        return mapping.get(v, "")

    def wrap_function_signature(self, name, args_str, body_str, shape):
        ret_type = self._get_matrix_type(shape)
        return f"""
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static {ret_type} {name}(
        {args_str}) noexcept
    {{
        auto res = {ret_type}{{}};
{body_str}
        return res;
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
            if n < 0:
                return f"(1.0 / amrex::Math::powi<{abs(n)}>({self._print(base)}))"
            return f"amrex::Math::powi<{n}>({self._print(base)})"
        return super()._print_Pow(expr)
