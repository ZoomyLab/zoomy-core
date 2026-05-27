"""Foam printer for the SystemModel contract.

Trusts that the incoming :class:`SystemModel` is complete and well-shaped
(emitted by Zoomy's own pipeline) — no defensive checks, no fallback
machinery.  Options that affect the *content* of the emitted C++ live as
printer flags, not as branching in the printer's plumbing.
"""

from __future__ import annotations

import itertools

import sympy as sp

from zoomy_core.transformation.generic_c import GenericCppBase, GenericCppModel


# ── Legacy printer (unchanged) ───────────────────────────────────────────


class FoamModel(GenericCppModel):
    """Legacy Foam printer — consumes a pre-SystemModel ``Model``."""

    _output_subdir = ".foam_interface"
    _is_template_class = False

    def __init__(self, model, *args, **kwargs):
        self.real_type = "Foam::scalar"
        self.math_namespace = "Foam::"
        super().__init__(model, *args, **kwargs)

    def get_includes(self):
        return '#include "List.H"\n#include "vector.H"\n#include "scalar.H" '

    def format_accessor(self, var_name, index):
        if var_name in ("n", "X") and index < 3:
            return f"{var_name}.{('x()', 'y()', 'z()')[index]}"
        return f"{var_name}[{index}]"

    def format_assignment(self, target_name, indices, value, shape):
        return f"{target_name}{''.join(f'[{i}]' for i in indices)} = {value};"

    def get_variable_declaration(self, v):
        return {
            "Q": "const Foam::List<Foam::scalar>& Q",
            "Qaux": "const Foam::List<Foam::scalar>& Qaux",
            "n": "const Foam::vector& n",
            "X": "const Foam::vector& X",
            "time": "const Foam::scalar& time",
            "dX": "const Foam::scalar& dX",
            "bc_idx": "const int bc_idx",
        }.get(v, "")

    def _get_foam_type(self, dims):
        if not dims:
            return "Foam::scalar"
        return f"Foam::List<{self._get_foam_type(dims[1:])}>"

    def wrap_function_signature(self, name, args_str, body_str, shape):
        def init(dims):
            if len(dims) == 1:
                return f"Foam::List<Foam::scalar>({dims[0]}, 0.0)"
            return (
                f"Foam::List<{self._get_foam_type(dims[1:])}>"
                f"({dims[0]}, {init(dims[1:])})"
            )

        return (
            f"\n    static inline {self._get_foam_type(shape)} {name}(\n"
            f"        {args_str})\n"
            f"    {{\n"
            f"        auto res = {init(shape)};\n"
            f"{body_str}\n"
            f"        return res;\n"
            f"    }}\n"
        )


# ── SystemModel printer ──────────────────────────────────────────────────


_FOAM_ARG = {
    "Q": "const Foam::List<Foam::scalar>& Q",
    "Qaux": "const Foam::List<Foam::scalar>& Qaux",
    "p": "const Foam::List<Foam::scalar>& p",
    "n": "const Foam::vector& n",
    "X": "const Foam::vector& X",
    "time": "const Foam::scalar& time",
    "dX": "const Foam::scalar& dX",
    "bc_idx": "const int bc_idx",
}

_AXIS = ("x", "y", "z")


class FoamSystemModelPrinter(GenericCppBase):
    """Foam printer for a frozen :class:`SystemModel`.

    Emits ``Model.H`` with one kernel per operator matrix.  Per-direction
    kernels (``flux_x`` / ``_y`` / ``_z`` etc.) match the calling
    convention of the existing hand-written ``numerics.H``.

    Options
    -------
    analytical_eigenvalues : bool, default False
        If True, emit the SystemModel's symbolic eigenvalue spectrum.
        If False, emit a zero placeholder — the solver computes
        eigenvalues numerically from ``quasilinear_matrix``.
    """

    _output_subdir = ".foam_interface"
    real_type = "Foam::scalar"
    math_namespace = "Foam::"
    analytical_eigenvalues = False

    def __init__(self, sm, **opts):
        super().__init__()
        self.sm = sm
        self.register_map("Q", list(sm.state))
        self.register_map("Qaux", list(sm.aux_state))
        self.register_map("n", list(sm.normal.values()))
        self.register_map("p", list(sm.parameters.values()))
        for k, v in opts.items():
            setattr(self, k, v)

    # ── Foam syntax hooks ────────────────────────────────────────────────

    def format_accessor(self, var, idx):
        if var in ("n", "X") and idx < 3:
            return f"{var}.{('x()', 'y()', 'z()')[idx]}"
        return f"{var}[{idx}]"

    def format_assignment(self, target, indices, value, shape):
        return f"{target}{''.join(f'[{i}]' for i in indices)} = {value};"

    def _print_Abs(self, expr):
        """Foam has no ``Foam::fabs``; canonical abs is ``Foam::mag``."""
        return f"Foam::mag({self._print(expr.args[0])})"

    def _foam_type(self, shape):
        if not shape:
            return self.real_type
        return f"Foam::List<{self._foam_type(shape[1:])}>"

    def _foam_init(self, shape):
        if len(shape) == 1:
            return f"Foam::List<{self.real_type}>({shape[0]}, 0.0)"
        return (
            f"Foam::List<{self._foam_type(shape[1:])}>"
            f"({shape[0]}, {self._foam_init(shape[1:])})"
        )

    def get_array_declaration(self, target, shape, init_zero=False):
        """Foam-flavoured ``auto res = Foam::List<...>(...);`` decl, used
        by the inherited :meth:`convert_expression_body` in place of the
        base's ``SimpleArray<T, N>`` declaration."""
        return f"auto {target} = {self._foam_init(shape)};"

    def wrap_function_signature(self, name, args_str, body_str, shape):
        """Signature wrapper — the body already declares ``res`` and
        returns it, so we only emit the surrounding function."""
        return (
            f"\ninline {self._foam_type(shape)} {name}(\n"
            f"    {args_str})\n"
            f"{{\n"
            f"{body_str}\n"
            f"}}\n"
        )

    # ── Emission ─────────────────────────────────────────────────────────

    def _kernel(self, name, expr, shape, args):
        body = self.convert_expression_body(expr, shape)
        sig = ",\n    ".join(_FOAM_ARG[a] for a in args)
        return self.wrap_function_signature(name, sig, body, shape)

    def _slice(self, tensor, axis_idx, out_shape):
        """``tensor[..., axis_idx]`` reshaped to ``out_shape``.  If
        ``out_shape`` has a trailing ``1`` padding (the ``flux_x`` column
        convention), walk one fewer axis when collecting source values."""
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

    def _per_direction(self, base, tensor, out_shape, args):
        return [
            self._kernel(
                f"{base}_{_AXIS[d]}",
                self._slice(tensor, d, out_shape),
                out_shape,
                args,
            )
            for d in range(self.sm.dimension)
        ]

    def create_code(self):
        sm = self.sm
        n_eq, n_state = sm.n_equations, len(sm.state)
        bc_tags = sorted(sm._bc_source.boundary_conditions_list_dict.keys())
        bc_str = ", ".join(f'"{t}"' for t in bc_tags)
        p_names = ", ".join(f'"{k}"' for k in sm.parameters.keys())
        p_vals = ", ".join(str(v) for v in sm.parameter_values.values())

        blocks = [
            "#pragma once",
            '#include "List.H"',
            '#include "vector.H"',
            '#include "scalar.H"',
            '#include "word.H"',
            "",
            "namespace Model",
            "{",
            f"constexpr int n_dof_q    = {n_eq};",
            f"constexpr int n_dof_qaux = {len(sm.aux_state)};",
            f"constexpr int n_parameters = {len(list(sm.parameters.keys()))};",
            f"constexpr int dimension  = {sm.dimension};",
            f"const Foam::List<Foam::word> map_boundary_tag_to_function_index{{ {bc_str} }};",
            f"const Foam::List<Foam::word> parameter_names{{ {p_names} }};",
            f"inline Foam::List<Foam::scalar> default_parameters() {{ return {{ {p_vals} }}; }}",
        ]

        # Every operator takes (Q, Qaux, p, …) — parameters are always in the interface.
        blocks += self._per_direction(
            "flux", sm.flux, (n_eq, 1), ["Q", "Qaux", "p"]
        )
        blocks += self._per_direction(
            "nonconservative_matrix",
            sm.nonconservative_matrix,
            (n_eq, n_state),
            ["Q", "Qaux", "p"],
        )
        blocks += self._per_direction(
            "quasilinear_matrix",
            sm.quasilinear_matrix,
            (n_eq, n_state),
            ["Q", "Qaux", "p"],
        )

        eig_expr = (
            sm.eigenvalues
            if self.analytical_eigenvalues
            else sp.Array([[0]] * n_eq)
        )
        blocks.append(
            self._kernel(
                "eigenvalues", eig_expr, (n_eq, 1), ["Q", "Qaux", "p", "n"]
            )
        )

        blocks.append(
            self._kernel("source", sm.source, (n_eq, 1), ["Q", "Qaux", "p"])
        )

        blocks.append(self._emit_boundary_conditions())

        blocks.append("} // namespace Model")
        return "\n".join(blocks)

    def _emit_boundary_conditions(self):
        """Emit ``Model::boundary_conditions(bc_idx, time, X, dX, Q,
        Qaux, p, n)`` from the SystemModel's symbolic Piecewise kernel.

        Returns a ``Foam::List<scalar>`` of size ``n_eq`` — the
        boundary state for the branch matching ``bc_idx``.
        """
        bc = self.sm.boundary_conditions
        # The Q / Qaux / p / n symbols are already mapped via __init__'s
        # register_map (they share Symbol identity with sm.state etc.).
        # Add scalar / position symbols specific to the BC kernel.
        extra_map = {}
        if bc.args.contains("idx"):
            extra_map[bc.args["idx"]] = "bc_idx"
        if bc.args.contains("time"):
            extra_map[bc.args["time"]] = "time"
        if bc.args.contains("distance"):
            extra_map[bc.args["distance"]] = "dX"
        if bc.args.contains("position"):
            pos = bc.args["position"]
            for axis in ("x", "y", "z"):
                if hasattr(pos, axis):
                    extra_map[getattr(pos, axis)] = f"X.{axis}()"

        self.symbol_maps.append(extra_map)
        try:
            shape = (self.sm.n_equations,)
            body = self.convert_expression_body(bc.definition, shape)
            sig = ",\n    ".join([
                "const int bc_idx",
                "const Foam::scalar& time",
                "const Foam::vector& X",
                "const Foam::scalar& dX",
                "const Foam::List<Foam::scalar>& Q",
                "const Foam::List<Foam::scalar>& Qaux",
                "const Foam::List<Foam::scalar>& p",
                "const Foam::vector& n",
            ])
            return self.wrap_function_signature(
                "boundary_conditions", sig, body, shape
            )
        finally:
            self.symbol_maps.pop()

    @classmethod
    def write_code(cls, sm, output_path, **opts):
        with open(output_path, "w") as f:
            f.write(cls(sm, **opts).create_code())
        return output_path


# ── Numerics (Riemann) printer ───────────────────────────────────────────


# Args carried by symbolic Riemann functions → Foam parameter declaration.
# Keys match ``func_obj.args.keys()`` for the Numerics-registered functions.
_FOAM_NUMERICS_ARG = {
    "q_minus": "const Foam::List<Foam::scalar>& Q_minus",
    "q_plus": "const Foam::List<Foam::scalar>& Q_plus",
    "aux_minus": "const Foam::List<Foam::scalar>& Qaux_minus",
    "aux_plus": "const Foam::List<Foam::scalar>& Qaux_plus",
    "Q": "const Foam::List<Foam::scalar>& Q",
    "Qaux": "const Foam::List<Foam::scalar>& Qaux",
    "p": "const Foam::List<Foam::scalar>& p",
    "normal": "const Foam::vector& n",
    "n": "const Foam::vector& n",
}


class FoamNumericsPrinter(GenericCppBase):
    """Foam printer for a symbolic :class:`Numerics` object (Rusanov,
    HLL, NonconservativeRusanov, …).

    Emits ``Numerics.H`` with one kernel per entry in
    ``numerics.functions`` — typically ``numerical_flux``,
    ``numerical_fluctuations``, ``local_max_abs_eigenvalue``.  Body
    expressions are CSE-optimised by the inherited
    :meth:`convert_expression_body`; signatures use the Foam type
    aliases above.
    """

    _output_subdir = ".foam_interface"
    real_type = "Foam::scalar"
    math_namespace = "Foam::"
    # Override the inherited expansion of max_wavespeed (which prints
    # nested ``max(abs(args))``).  In the Foam backend max_wavespeed
    # is opaque — the solver provides the C++ implementation in
    # numerics.H, mirroring numpy/jax (``max_wavespeed: None``).
    #
    # The symbolic ``max_wavespeed`` always receives the flattened
    # (Q, Qaux, p, normal) for a given side.  We detect the side from
    # the first arg's symbol name and emit a fixed-signature call
    # ``numerics::max_wavespeed(Q_side, Qaux_side, p, n)`` that the
    # solver implements once (wrapping ``Model::eigenvalues``).
    # Named without the ``_print_`` prefix to avoid sympy's
    # automatic ``_print_<funcname>`` dispatch (which would call this
    # with the whole Function expr instead of via c_functions).
    @staticmethod
    def _emit_max_wavespeed(printer, *args):
        first = str(args[0])
        if first.startswith("Q_minus"):
            return "numerics::max_wavespeed(Q_minus, Qaux_minus, p, n)"
        if first.startswith("Q_plus"):
            return "numerics::max_wavespeed(Q_plus, Qaux_plus, p, n)"
        return "numerics::max_wavespeed(Q, Qaux, p, n)"

    c_functions = {
        **GenericCppBase.c_functions,
        "max_wavespeed": _emit_max_wavespeed.__func__,
    }

    def __init__(self, numerics, **opts):
        super().__init__()
        self.numerics = numerics
        sm = numerics.model
        self.sm = sm
        # State / aux / parameter / normal symbol maps.
        self.register_map("Q", list(sm.state))
        self.register_map("Qaux", list(sm.aux_state))
        self.register_map("n", list(sm.normal.values()))
        self.register_map("p", list(sm.parameters.values()))
        # Face-state symbols carried by the symbolic Numerics — wired
        # into the printer so they print as ``Q_minus[i]`` etc.
        self.register_map("Q_minus", list(numerics.variables_minus))
        self.register_map("Q_plus", list(numerics.variables_plus))
        self.register_map("Qaux_minus", list(numerics.aux_variables_minus))
        self.register_map("Qaux_plus", list(numerics.aux_variables_plus))
        self.register_map("flux_minus", list(numerics.flux_minus))
        self.register_map("flux_plus", list(numerics.flux_plus))
        for k, v in opts.items():
            setattr(self, k, v)

    # ── Foam syntax (shared with the SystemModel printer) ────────────────

    def format_accessor(self, var, idx):
        if var in ("n", "X") and idx < 3:
            return f"{var}.{('x()', 'y()', 'z()')[idx]}"
        return f"{var}[{idx}]"

    def format_assignment(self, target, indices, value, shape):
        return f"{target}{''.join(f'[{i}]' for i in indices)} = {value};"

    def _print_Abs(self, expr):
        """Foam has no ``Foam::fabs``; canonical abs is ``Foam::mag``."""
        return f"Foam::mag({self._print(expr.args[0])})"

    def _foam_type(self, shape):
        if not shape:
            return self.real_type
        return f"Foam::List<{self._foam_type(shape[1:])}>"

    def _foam_init(self, shape):
        if len(shape) == 1:
            return f"Foam::List<{self.real_type}>({shape[0]}, 0.0)"
        return (
            f"Foam::List<{self._foam_type(shape[1:])}>"
            f"({shape[0]}, {self._foam_init(shape[1:])})"
        )

    def get_array_declaration(self, target, shape, init_zero=False):
        return f"auto {target} = {self._foam_init(shape)};"

    def wrap_function_signature(self, name, args_str, body_str, shape):
        return (
            f"\ninline {self._foam_type(shape)} {name}(\n"
            f"    {args_str})\n"
            f"{{\n"
            f"{body_str}\n"
            f"}}\n"
        )

    # ── Emission ─────────────────────────────────────────────────────────

    def _generate_signature_from_function(self, func_obj):
        """Foam-typed parameter list built from ``func_obj.args.keys()``."""
        return ",\n    ".join(_FOAM_NUMERICS_ARG[k] for k in func_obj.args.keys())

    def create_code(self):
        sm = self.sm
        blocks = [
            "#pragma once",
            '#include "List.H"',
            '#include "vector.H"',
            '#include "scalar.H"',
            '#include "Model.H"',
            "",
            "namespace Numerics",
            "{",
            f"constexpr int n_dof_q = {sm.n_equations};",
        ]
        for _name, func_obj in self.numerics.functions.items():
            blocks.extend(self._process_kernel_from_function(func_obj))
        blocks.append("} // namespace Numerics")
        return "\n".join(blocks)

    @classmethod
    def write_code(cls, numerics, output_path, **opts):
        with open(output_path, "w") as f:
            f.write(cls(numerics, **opts).create_code())
        return output_path
