"""DerivedModel — chainable base class for equation-derived models.

The derivation graph is the Python class hierarchy.  Each subclass
overrides :meth:`derive_model` and uses ``self.apply(...)`` to transform
the equations inherited from its parent (via ``super().derive_model()``).

Every ``apply`` call auto-registers the operation so that ``describe()``
can reconstruct the full derivation path without any manual bookkeeping.

Example::

    class INSModel(DerivedModel):
        def derive_model(self):
            state = StateSpace(dimension=2)
            ins = FullINS(state)
            self._init_system("INS", {
                "continuity": ins.continuity,
                "x_momentum": ins.x_momentum,
            }, state)

    class SMEModel(INSModel):
        projectable = True
        def derive_model(self):
            super().derive_model()
            self.apply(HydrostaticPressure(self.state))
            self.apply(Newtonian(self.state))
            self.depth_integrate()

    model = SMEModel(level=2)
    model.describe(derivation='mermaid')
"""

from __future__ import annotations

from typing import Optional, List

import sympy as sp
from sympy import Matrix, MutableDenseNDimArray, Rational, S, Symbol
import param
import numpy as np

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Basisfunction
from zoomy_core.model.models.symbolic_integrator import SymbolicIntegrator
from zoomy_core.model.models.derived_system import DerivedSystem


# ── Basis matrix cache ────────────────────────────────────────────────────────

_basis_matrix_cache: dict = {}


def _cache_key(basis_type, level):
    if isinstance(basis_type, type):
        name = getattr(basis_type, "name", basis_type.__name__)
    else:
        name = getattr(basis_type, "name", str(basis_type))
    return (name, level)


def get_cached_matrices(basis, level, integrator):
    key = _cache_key(basis, level)
    if key not in _basis_matrix_cache:
        _basis_matrix_cache[key] = integrator.compute_all_matrices(level)
    return _basis_matrix_cache[key]


def clear_matrix_cache():
    _basis_matrix_cache.clear()


class DerivedModel(Model):
    """Base class for models derived via a chain of symbolic operations.

    **Derivation graph = class hierarchy.**
    - Each class is a node.
    - ``derive_model()`` is the edge: it calls ``super().derive_model()``
      to get the parent's equations, then ``self.apply(...)`` to refine.
    - ``apply()`` auto-registers every operation for ``describe()``.

    Set ``projectable = True`` on the final class to make it solver-ready.
    """

    projectable = False

    level = param.Integer(default=0, doc="Vertical basis function order")
    n_layers = param.Integer(default=1, doc="Number of vertical layers")
    basis_type = param.ClassSelector(
        class_=Basisfunction, default=Legendre_shifted, is_instance=False,
        doc="Vertical basis function family"
    )

    def __init__(self, level=0, n_layers=1, basis_type=Legendre_shifted,
                 eigenvalue_mode="symbolic", dimension=None, **kwargs):

        self._system: Optional[DerivedSystem] = None
        self._applied: List[dict] = []  # auto-filled by apply()

        # Run the derivation graph (calls super().derive_model() + apply()s)
        self.derive_model()

        # Name the system after this class
        if self._system is not None:
            self._system.name = type(self).__name__

        # Infer dimension from system state if not given
        if dimension is None and self._system is not None:
            state = self._system.state
            dimension = getattr(state, "dimension", 2) - 1
            if dimension < 1:
                dimension = 1

        if self.projectable and self._system is not None:
            n_mom = level + 1
            hdim = dimension or 1
            n_vars = 2 + hdim * n_layers * n_mom
            var_names = ["b", "h"] + [f"q{i}" for i in range(2, n_vars)]
            param_dict = {
                "g": (9.81, "positive"),
                "eps": (1e-6, "positive"),
                "ez": (1.0, "positive"),
                "rho": (1000.0, "positive"),
                "lamda": (0.1, "positive"),
                "nu": (1e-6, "positive"),
            }
            super().__init__(
                init_functions=False,
                dimension=hdim,
                variables=var_names,
                parameters=param_dict,
                eigenvalue_mode=eigenvalue_mode,
                level=level,
                n_layers=n_layers,
                basis_type=basis_type,
                **kwargs,
            )
            self._initialize_functions()
        else:
            super().__init__(
                init_functions=False,
                dimension=dimension or 1,
                variables=0,
                parameters={},
                level=level,
                n_layers=n_layers,
                basis_type=basis_type,
                **kwargs,
            )

    # ── derive_model: user overrides this ─────────────────────────────

    def derive_model(self):
        """Override to build/refine the equation system.

        Root classes: call ``self._init_system(...)`` to create equations.
        Child classes: call ``super().derive_model()`` then ``self.apply(...)``
        to transform.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement derive_model()"
        )

    # ── system init (for root classes) ────────────────────────────────

    def _init_system(self, name, equations, state, assumptions=None):
        """Set the initial equation system (used by root classes)."""
        self._system = DerivedSystem(name, state, equations, assumptions)

    # ── apply: mutate system + auto-register ──────────────────────────

    def apply(self, operation):
        """Apply a relation or operation to all equations.

        Accepts two kinds of arguments:

        1. **Relation** (Assumption, Material): substitutes symbols
           in all equations (``lhs → rhs``).
        2. **Operation** (DepthIntegrate, etc.): applies a callable
           transformation to each equation.

        Mutates ``self._system`` in place and auto-registers the
        operation for ``describe()``.
        """
        from zoomy_core.model.models.ins_generator import Relation, Operation

        if self._system is None:
            raise RuntimeError(
                "No system to apply to. Root classes must call "
                "self._init_system() before apply()."
            )

        # Apply to all equations in place
        self._system.apply(operation)

        # Track for describe()
        display_name = getattr(operation, 'description', None) or getattr(operation, 'name', str(operation))
        latex = operation._repr_latex_() if hasattr(operation, '_repr_latex_') else None
        if latex is not None and not latex.strip():
            latex = None
        self._applied.append({"name": display_name, "latex": latex})

    # ── public API ────────────────────────────────────────────────────

    @property
    def system(self) -> Optional[DerivedSystem]:
        """The current equation system."""
        return self._system

    @property
    def state(self):
        """The shared symbolic StateSpace."""
        if self._system is not None:
            return self._system.state
        return None

    @property
    def applied_operations(self) -> List[dict]:
        """Operations applied by this class's derive_model()."""
        return list(self._applied)

    def _parent_derived_class(self):
        """Find the parent DerivedModel class in the MRO (not self's class)."""
        for cls in type(self).__mro__[1:]:
            if cls is DerivedModel:
                return None  # reached the base — no parent
            if issubclass(cls, DerivedModel) and cls is not DerivedModel:
                return cls
        return None

    # ── describe ──────────────────────────────────────────────────────

    def describe(self, header=True, derivation=False, assumptions=True,
                 final_equation=True, parameters=False, strip_args=False,
                 verbose=False):
        """Composable description of this model.

        Parameters
        ----------
        header : bool
            Model class name + system info.
        derivation : False, 'mermaid', or 'markdown'
            Show the derivation path from parent to self.
            Compact (default): parent → one edge → self.
            Verbose (``verbose=True``): parent → edge per apply → self.
        assumptions : bool
            List assumptions.
        final_equation : bool
            Show final equations.
        parameters : bool
            List free symbols.
        strip_args : bool
            Display ``u`` instead of ``u(t, x, z)``.
        verbose : bool
            In derivation mode, show one edge per apply() call
            instead of summarizing all on a single edge.
        """
        from zoomy_core.misc.description import Description

        parts = []

        # ── Header ────────────────────────────────────────────────────
        if header and self._system is not None:
            cls_name = type(self).__name__
            eq_names = ", ".join(self._system.equations.keys())
            proj = "solver-ready" if self.projectable else "intermediate"
            parts.append(f"**{cls_name}** ({eq_names}) — {proj}\n")

        # ── Derivation graph ──────────────────────────────────────────
        if derivation and self._applied:
            parent_cls = self._parent_derived_class()

            if derivation == "mermaid":

                def _mermaid_node(sys, cls_name, strip):
                    """Build a mermaid node label with class name + multiline equations."""
                    lines = [f"**{cls_name}**"]
                    if sys:
                        for eq_name, eq in sys.equations.items():
                            tex = eq.latex(strip_args=strip, multiline=True)
                            tex = tex.replace("\n", " ")
                            # multiline already includes = 0, don't add another
                            if "\\end{aligned}" in tex:
                                lines.append(f"*{eq_name}*: $${tex}$$")
                            else:
                                lines.append(f"*{eq_name}*: $${tex} = 0$$")
                    return "<br/>".join(lines)

                def _mermaid_edge(ops):
                    """Build edge label from applied operations."""
                    edge_parts = []
                    for op in ops:
                        name = op["name"]
                        latex = op.get("latex")
                        if latex:
                            # Strip outer $...$ delimiters from _repr_latex_
                            inner = latex.strip("$").strip()
                            edge_parts.append(f"**{name}**<br/>$${inner}$$")
                        else:
                            edge_parts.append(f"**{name}**")
                    return "<br/>".join(edge_parts)

                parts.append("```mermaid")
                parts.append("graph TD")

                # Parent node
                if parent_cls:
                    parent_inst = parent_cls(level=self.level)
                    psys = parent_inst._system
                    p_label = _mermaid_node(psys, parent_cls.__name__, strip_args)
                    parts.append(f'    P["{p_label}"]')
                else:
                    parts.append(f'    P["(root)"]')

                # Edge(s) + final node
                s_label = _mermaid_node(self._system, type(self).__name__, strip_args)

                if verbose and len(self._applied) > 1:
                    prev = "P"
                    for i, op in enumerate(self._applied):
                        mid = f"M{i}"
                        elabel = _mermaid_edge([op])
                        parts.append(f'    {prev} -->|"{elabel}"| {mid}[" "]')
                        prev = mid
                    parts.append(f'    {prev} --> S["{s_label}"]')
                else:
                    elabel = _mermaid_edge(self._applied)
                    parts.append(f'    P -->|"{elabel}"| S["{s_label}"]')

                parts.append("```\n")

            elif derivation == "markdown":
                parts.append("**Derivation path:**\n")

                # Parent
                if parent_cls:
                    parent_inst = parent_cls(level=self.level)
                    psys = parent_inst._system
                    parts.append(f"#### {parent_cls.__name__}\n")
                    if psys:
                        for eq_name, eq in psys.equations.items():
                            tex = eq.latex(strip_args=strip_args)
                            parts.append(f"**{eq_name}:** ${tex} = 0$\n")

                # Operations
                parts.append("**Operations applied:**\n")
                for op in self._applied:
                    line = f"- **{op['name']}**"
                    if op.get("latex"):
                        line += f"  {op['latex']}"
                    parts.append(line)

                parts.append(f"\n#### {type(self).__name__}\n")

        # ── Delegate remaining sections to DerivedSystem ──────────────
        if self._system is not None:
            sys_desc = self._system.describe(
                header=False,
                assumptions=assumptions,
                final_equation=final_equation,
                parameters=parameters,
                strip_args=strip_args,
            )
            parts.append(str(sys_desc))

        if not parts:
            return Description(f"**{type(self).__name__}**: no system derived")

        return Description("\n".join(parts))

    def _repr_markdown_(self):
        return self.describe()._repr_markdown_()

    # ── projection machinery (for projectable models) ─────────────────

    def _initialize_derived_properties(self):
        super()._initialize_derived_properties()

        if not self.projectable or self._system is None:
            return

        old_h = self.variables[1]
        new_h = Symbol(old_h.name, positive=True, real=True)
        self.variables[1] = new_h

        self.basisfunctions = self.basis_type(level=self.level)
        self._integrator = SymbolicIntegrator(self.basisfunctions)
        matrices = get_cached_matrices(self.basis_type, self.level, self._integrator)
        self._M = matrices["M"]
        self._A = matrices["A"]
        self._D = matrices["D"]
        self._D1 = matrices["D1"]
        self._B = matrices.get("B", np.zeros_like(self._A))
        self._phib = matrices["phib"]
        self._c_mean = self.basisfunctions.mean_coefficients()
        n = self.level + 1
        self._phi_int = [sum(self._M[l, j] * self._c_mean[j] for j in range(n))
                         for l in range(n)]

        M = self._M
        is_diag = all(M[i, j] == 0 for i in range(n) for j in range(n) if i != j)
        if is_diag:
            self._Minv = [[Rational(1) / M[i, i] if i == j else S.Zero
                           for j in range(n)] for i in range(n)]
        else:
            M_sp = sp.Matrix([[M[i, j] for j in range(n)] for i in range(n)])
            Minv_sp = M_sp.inv()
            self._Minv = [[Minv_sp[i, j] for j in range(n)] for i in range(n)]

    def _apply_Minv(self, raw_vec, k):
        n = self.level + 1
        return sum(self._Minv[k][l] * raw_vec[l] for l in range(n))

    def get_primitives(self):
        n_mom = self.level + 1
        n_layers = self.n_layers
        b = self.variables[0]
        h = self.variables[1]
        hinv = 1 / h
        moments_u = []
        idx = 2
        for lk in range(n_layers):
            layer = [self.variables[idx + j] * hinv for j in range(n_mom)]
            moments_u.append(layer)
            idx += n_mom
        moments_v = []
        if self.dimension > 1:
            for lk in range(n_layers):
                layer = [self.variables[idx + j] * hinv for j in range(n_mom)]
                moments_v.append(layer)
                idx += n_mom
        else:
            moments_v = [[S.Zero] * n_mom for _ in range(n_layers)]
        return b, h, moments_u, moments_v, hinv

    def mass_matrix(self):
        return self._M

    def mass_matrix_inverse(self):
        return self._Minv

    # ── Model interface (flux, source, NC, etc.) ──────────────────────

    def flux(self):
        dim = self.dimension
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        b, h, moments_u, moments_v, hinv = self.get_primitives()
        A = self._A
        c_mean = self._c_mean
        F = Matrix.zeros(n_vars, dim)
        for lk in range(n_layers):
            w_k = Rational(1, n_layers)
            alpha = moments_u[lk]
            beta = moments_v[lk]
            F[1, 0] += h * sum(c_mean[k] * alpha[k] for k in range(n_mom)) * w_k
            row_base_u = 2 + lk * n_mom
            raw_adv = [S.Zero] * n_mom
            for l in range(n_mom):
                for i in range(n_mom):
                    for j in range(n_mom):
                        raw_adv[l] += h * w_k * alpha[i] * alpha[j] * A[l, i, j]
            for k in range(n_mom):
                F[row_base_u + k, 0] += self._apply_Minv(raw_adv, k)
            if dim == 2:
                F[1, 1] += h * sum(c_mean[k] * beta[k] for k in range(n_mom)) * w_k
                row_base_v = 2 + n_layers * n_mom + lk * n_mom
                raw_vv = [S.Zero] * n_mom
                raw_vu = [S.Zero] * n_mom
                raw_uv = [S.Zero] * n_mom
                for l in range(n_mom):
                    for i in range(n_mom):
                        for j in range(n_mom):
                            raw_vv[l] += h * w_k * beta[i] * beta[j] * A[l, i, j]
                            raw_vu[l] += h * w_k * beta[i] * alpha[j] * A[l, i, j]
                            raw_uv[l] += h * w_k * alpha[i] * beta[j] * A[l, i, j]
                for k in range(n_mom):
                    F[row_base_v + k, 1] += self._apply_Minv(raw_vv, k)
                    F[row_base_v + k, 0] += self._apply_Minv(raw_vu, k)
                    F[row_base_u + k, 1] += self._apply_Minv(raw_uv, k)
        return ZArray(F)

    def hydrostatic_pressure(self):
        dim = self.dimension
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        p = self.parameters
        b, h, moments_u, moments_v, hinv = self.get_primitives()
        F = Matrix.zeros(n_vars, dim)
        phi_int = self._phi_int
        raw_p = [p.g * p.ez * h**2 / 2 * phi_int[l] for l in range(n_mom)]
        for k in range(n_mom):
            F[2 + k, 0] = self._apply_Minv(raw_p, k)
        if dim == 2:
            row_v0 = 2 + n_layers * n_mom
            for k in range(n_mom):
                F[row_v0 + k, 1] = self._apply_Minv(raw_p, k)
        return ZArray(F)

    def nonconservative_matrix(self):
        dim = self.dimension
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        p = self.parameters
        B_mat = self._B
        b, h, moments_u, moments_v, hinv = self.get_primitives()
        nc_x = Matrix.zeros(n_vars, n_vars)
        nc_y = Matrix.zeros(n_vars, n_vars)
        for lk in range(n_layers):
            w_k = Rational(1, n_layers)
            alpha = moments_u[lk]
            row_base_u = 2 + lk * n_mom
            phi_int = self._phi_int
            raw_topo = [p.g * p.ez * h * phi_int[l] for l in range(n_mom)]
            for k in range(n_mom):
                nc_x[row_base_u + k, 0] += self._apply_Minv(raw_topo, k)
            for col in range(n_vars):
                ci = col - row_base_u
                if ci < 1 or ci >= n_mom:
                    continue
                raw_nc = [S.Zero] * n_mom
                for l in range(n_mom):
                    for j in range(1, n_mom):
                        raw_nc[l] += alpha[j] * B_mat[l, j, ci]
                for k in range(n_mom):
                    nc_x[row_base_u + k, col] += self._apply_Minv(raw_nc, k)
            um = alpha[0]
            for k in range(1, n_mom):
                raw_um = [S.Zero] * n_mom
                raw_um[k] = -um
                nc_x[row_base_u + k, row_base_u + k] += self._apply_Minv(raw_um, k)
            if dim == 2:
                row_base_v = 2 + n_layers * n_mom + lk * n_mom
                raw_topo_y = [p.g * p.ez * h * phi_int[l] for l in range(n_mom)]
                for k in range(n_mom):
                    nc_y[row_base_v + k, 0] += self._apply_Minv(raw_topo_y, k)
        A_tensor = MutableDenseNDimArray.zeros(n_vars, n_vars, dim)
        for r in range(n_vars):
            for c in range(n_vars):
                A_tensor[r, c, 0] = nc_x[r, c]
                if dim > 1:
                    A_tensor[r, c, 1] = nc_y[r, c]
        return ZArray(A_tensor)

    def source(self):
        return ZArray.zeros(self.n_variables)

    def newtonian_viscosity(self):
        p = self.parameters
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        D = self._D
        out = ZArray.zeros(n_vars)
        b, h, moments_u, moments_v, hinv = self.get_primitives()
        for lk in range(n_layers):
            w_k = Rational(1, n_layers)
            alpha = moments_u[lk]
            row_base_u = 2 + lk * n_mom
            raw_u = [sum(-p.nu * alpha[i] * hinv * D[i, l] / w_k
                         for i in range(n_mom)) for l in range(n_mom)]
            for k in range(n_mom):
                out[row_base_u + k] += self._apply_Minv(raw_u, k)
            if self.dimension == 2:
                beta = moments_v[lk]
                row_base_v = 2 + n_layers * n_mom + lk * n_mom
                raw_v = [sum(-p.nu * beta[i] * hinv * D[i, l] / w_k
                             for i in range(n_mom)) for l in range(n_mom)]
                for k in range(n_mom):
                    out[row_base_v + k] += self._apply_Minv(raw_v, k)
        return out

    def navier_slip(self):
        p = self.parameters
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        phib = self._phib
        out = ZArray.zeros(n_vars)
        b, h, moments_u, moments_v, hinv = self.get_primitives()
        for lk in range(n_layers):
            w_k = Rational(1, n_layers)
            alpha = moments_u[lk]
            row_base_u = 2 + lk * n_mom
            u_bottom = sum(alpha[i] * phib[i] for i in range(n_mom))
            raw_slip = [-u_bottom * phib[l] / (p.lamda * w_k) for l in range(n_mom)]
            for k in range(n_mom):
                out[row_base_u + k] += self._apply_Minv(raw_slip, k)
            if self.dimension == 2:
                beta = moments_v[lk]
                row_base_v = 2 + n_layers * n_mom + lk * n_mom
                v_bottom = sum(beta[i] * phib[i] for i in range(n_mom))
                raw_slip_v = [-v_bottom * phib[l] / (p.lamda * w_k)
                              for l in range(n_mom)]
                for k in range(n_mom):
                    out[row_base_v + k] += self._apply_Minv(raw_slip_v, k)
        return out
