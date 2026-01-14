from zoomy_core.transformation.generic_c import (
    GenericCppModel,
    GenericCppNumerics,
    flatten_index,
)

class CppModel(GenericCppModel):
    _output_subdir = ".c_interface"
    _is_template_class = True

    def __init__(self, model, *args, **kwargs):
        self.real_type = "T"
        self.math_namespace = "std::"
        super().__init__(model, *args, **kwargs)

    def get_includes(self):
        return "#include <cmath>\n#include <algorithm>"

    def format_accessor(self, var_name, index):
        return f"{var_name}[{index}]"

    def format_assignment(self, target_name, indices, value, shape):
        flat_idx = flatten_index(indices, shape)
        return f"{target_name}[{flat_idx}] = {value};"

    def get_variable_declaration(self, v):
        mapping = {
            "Q": "const T* Q",
            "Qaux": "const T* Qaux",
            "n": "const T* n",
            "X": "const T* X",
            "time": "const T time",
            "dX": "const T dX",
            "res": "T* res",
            "bc_idx": "const int bc_idx",
        }
        return mapping.get(v, "")

    def wrap_function_signature(self, name, args_str, body_str, shape):
        return f"""
    static inline void {name}(
        {args_str})
    {{
{body_str}
    }}
"""


class CppNumerics(GenericCppNumerics):
    _output_subdir = ".c_interface"
    _is_template_class = True  # We want template <typename T>

    def __init__(self, numerics, *args, **kwargs):
        self.real_type = "T"
        self.math_namespace = "std::"
        super().__init__(numerics, *args, **kwargs)

    def get_includes(self):
        return "#include <cmath>\n#include <algorithm>"

    def format_accessor(self, var_name, index):
        return f"{var_name}[{index}]"

    def format_assignment(self, target_name, indices, value, shape):
        flat_idx = flatten_index(indices, shape)
        return f"{target_name}[{flat_idx}] = {value};"

    def get_variable_declaration(self, v):
        mapping = {
            # Shared variables
            "n": "const T* n",
            "res": "T* res",
            "h": "const T h",  
            "cfl": "const T cfl",  
            # Numerics specific (Left/Right states)
            "Q_minus": "const T* Q_minus",
            "Q_plus": "const T* Q_plus",
            "Qaux_minus": "const T* Qaux_minus",
            "Qaux_plus": "const T* Qaux_plus",
            # For update/dt/eigenvalue kernels
            "Q": "const T* Q",
            "Qaux": "const T* Qaux",
        }
        return mapping.get(v, "")

    def get_file_header(self):
        tpl = "template <typename T>" if self._is_template_class else ""
        
        lines = [
            "#pragma once",
            self.get_includes().strip(),
            "#include <vector>",
            ""
        ]

        # --- GPU Logic ---
        if self.gpu_enabled:
            lines.extend([
                "// --- GPU Compatibility Macros ---",
                "#ifdef __CUDACC__",
                "#define PORTABLE_FN __host__ __device__",
                "#else",
                "#define PORTABLE_FN",
                "#endif",
                ""
            ])
        else:
            lines.append("#define PORTABLE_FN")

        lines.extend([
            tpl,
            f"struct {self._wrapper_name} {{",
            "",
            f"    // --- Numerics Constants ---",
            f"    static constexpr int n_dof_q = {self.n_dof_q};",
            "",
            f"    // --- Kernels ---",
        ])
        return "\n".join(lines)

    def wrap_function_signature(self, name, args_str, body_str, shape):
        # We prepend PORTABLE_FN so it compiles on both Host and Device
        return f"""
    PORTABLE_FN static inline void {name}(
        {args_str})
    {{
{body_str}
    }}
"""