"""Module `zoomy_core.transformation.to_openfoam`."""

from zoomy_core.transformation.generic_c import GenericCppModel


class FoamModel(GenericCppModel):
    """FoamModel. (class)."""
    _output_subdir = ".foam_interface"
    _is_template_class = False

    def __init__(self, model, *args, **kwargs):
        """Initialize the instance."""
        self.real_type = "Foam::scalar"
        self.math_namespace = "Foam::"
        super().__init__(model, *args, **kwargs)

    def get_includes(self):
        """Get includes."""
        return """#include "List.H"
#include "vector.H"
#include "scalar.H" """

    def format_accessor(self, var_name, index):
        """Format accessor."""
        if var_name in ["n", "X"] and index < 3:
            return f"{var_name}.{['x()', 'y()', 'z()'][index]}"
        return f"{var_name}[{index}]"

    def format_assignment(self, target_name, indices, value, shape):
        # Recursive access [i][j]
        """Format assignment."""
        access_str = "".join([f"[{i}]" for i in indices])
        return f"{target_name}{access_str} = {value};"

    def get_variable_declaration(self, v):
        """Get variable declaration."""
        if v == "res":
            return ""
        mapping = {
            "Q": "const Foam::List<Foam::scalar>& Q",
            "Qaux": "const Foam::List<Foam::scalar>& Qaux",
            "n": "const Foam::vector& n",
            "X": "const Foam::vector& X",
            "time": "const Foam::scalar& time",
            "dX": "const Foam::scalar& dX",
            "bc_idx": "const int bc_idx",
        }
        return mapping.get(v, "")

    def _get_foam_type(self, dims):
        """Internal helper `_get_foam_type`."""
        if len(dims) == 0:
            return "Foam::scalar"
        return f"Foam::List<{self._get_foam_type(dims[1:])}>"

    def wrap_function_signature(self, name, args_str, body_str, shape):
        """Wrap function signature."""
        def gen_init(dims):
            """Gen init."""
            if len(dims) == 1:
                return f"Foam::List<Foam::scalar>({dims[0]}, 0.0)"
            return f"Foam::List<{self._get_foam_type(dims[1:])}>({dims[0]}, {gen_init(dims[1:])})"

        ret_type = self._get_foam_type(shape)
        # We manually indent the auto res = ... line to 4 spaces
        return f"""
    static inline {ret_type} {name}(
        {args_str})
    {{
        auto res = {gen_init(shape)};
{body_str}
        return res;
    }}
"""
