"""SystemModel — operator-form symbolic PDE for analysis + transformation.

`SystemModel` is the runtime/operator-form representation of a PDE
system: a flat container of symbolic ``sp.Matrix`` / sympy-tensor
operators (flux, non-conservative-matrix, source, mass matrix,
hydrostatic pressure) plus state, parameters, coordinates, and
boundary-condition kernels.  It carries no derivation history and no
equation tree — that's the `Model` class's job.

The two are **independent siblings**, not inherited.  `Model` owns
derivation; `SystemModel` owns the operator surface that solvers and
analysis consume.  ``SystemModel.from_model(m)`` extracts the
operators once by calling the model's API methods and freezes the
result; subsequent calls on the SystemModel never re-walk.

Operations on a SystemModel modify the stored matrices in place via
``apply(operation)`` — the most important being ``InvertMassMatrix``
which left-multiplies the system by ``M⁻¹`` to reach canonical
``∂_t Q + ∂_x F + B·∂_x Q − S = 0`` form.

Solvers and analysis routines accept either a ``Model`` or a
``SystemModel`` directly; internally they normalize via
``SystemModel.from_model(m)`` if a Model is passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sympy as sp


def _iter_indices(shape):
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for rest in _iter_indices(shape[1:]):
            yield (i,) + rest


@dataclass
class SystemModel:
    """Symbolic operator-form PDE system.

    Stored matrices follow the contract:

    .. math::

        M(Q) \\; \\partial_t Q
            + \\nabla \\cdot \\big(F(Q) + P(Q)\\big)
            + \\sum_d B(Q)[:,:,d] \\, \\partial_d Q
            - S(Q) = 0

    Operators are indexed ``(equation_row, state_col[, dim])``.  In
    general the system is **rectangular** — there can be fewer (or
    more) equations than state entries; this is the case for the
    splitter sub-systems, where each stage updates only a subset of
    the shared state vector.  ``equation_to_state_index[r]`` records
    which state entry equation ``r`` updates; for square systems the
    default is the identity map ``[0, 1, …, n_state-1]``.

    Shape contract:

    * ``flux``                  — ``(n_eq, n_dim)``
    * ``hydrostatic_pressure``  — ``(n_eq, n_dim)``
    * ``nonconservative_matrix``— ``(n_eq, n_state, n_dim)``
    * ``source``                — ``(n_eq, 1)``
    * ``mass_matrix``           — ``(n_eq, n_state)``
    """

    time: sp.Symbol
    space: List[sp.Symbol]
    state: List[Any]
    aux_state: List[Any]
    parameters: Dict[Any, Any]
    flux: sp.Matrix
    hydrostatic_pressure: sp.Matrix
    nonconservative_matrix: Any              # sp.MutableDenseNDimArray
    source: sp.Matrix
    mass_matrix: sp.Matrix
    equation_to_state_index: Optional[List[int]] = None
    boundary_conditions: Optional[Any] = None
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if self.equation_to_state_index is None:
            self.equation_to_state_index = list(range(self.n_equations))

    # ── Shape accessors ────────────────────────────────────────────────

    @property
    def n_equations(self) -> int:
        return self.flux.shape[0]

    @property
    def n_state(self) -> int:
        return len(self.state)

    @property
    def n_dim(self) -> int:
        return len(self.space)

    @property
    def is_square(self) -> bool:
        """True when n_equations == n_state and equations map identity-style."""
        return (self.n_equations == self.n_state
                and self.equation_to_state_index == list(range(self.n_state)))

    # ── Operator-API methods (mirror Model) ─────────────────────────────

    def quasilinear_matrix(self):
        """``∂F/∂Q + ∂P/∂Q + B`` — shape ``(n_eq, n_state, n_dim)``."""
        n_eq = self.n_equations
        n_st = self.n_state
        d = self.n_dim
        Q = sp.MutableDenseNDimArray.zeros(n_eq, n_st, d)
        for i in range(n_eq):
            for j in range(n_st):
                for k in range(d):
                    djF = sp.diff(self.flux[i, k], self.state[j])
                    djP = sp.diff(self.hydrostatic_pressure[i, k],
                                  self.state[j])
                    Q[i, j, k] = djF + djP + self.nonconservative_matrix[i, j, k]
        return Q

    def source_jacobian_wrt_state(self):
        """``∂S/∂Q`` — shape ``(n_eq, n_state)``."""
        n_eq = self.n_equations
        n_st = self.n_state
        out = sp.zeros(n_eq, n_st)
        for i in range(n_eq):
            for j in range(n_st):
                out[i, j] = sp.diff(self.source[i, 0], self.state[j])
        return out

    # ── apply / row view ───────────────────────────────────────────────

    def apply(self, operation, *, name: Optional[str] = None,
              description: Optional[str] = None):
        """Apply a system-level operation in place.  Returns self."""
        op_name = name or getattr(operation, "name", None) or \
                  type(operation).__name__
        op_desc = description or getattr(operation, "description", op_name)
        operation(self)
        self.history.append({"name": op_name, "description": op_desc})
        return self

    def __getitem__(self, i: int) -> "SystemModelRow":
        return SystemModelRow(self, i)

    # ── from_model factory ────────────────────────────────────────────

    @classmethod
    def from_model(cls, model) -> "SystemModel":
        """Build a SystemModel from a Model by reading its operator API.

        Calls ``model.flux()``, ``model.nonconservative_matrix()``,
        ``model.source()``, ``model.hydrostatic_pressure()`` once and
        freezes the matrices.  ``state`` / ``aux_state`` /
        ``parameters`` come from the model's Zstructs.
        """
        time_sym = getattr(model, "time", sp.Symbol("t", real=True))
        dim = getattr(model, "dimension", 1)
        coord_names = ["x", "y", "z"]
        space = [sp.Symbol(coord_names[d], real=True) for d in range(dim)]

        state = [model.variables[k] for k in model.variables.keys()]
        aux_state = [model.aux_variables[k]
                     for k in model.aux_variables.keys()]
        parameters: Dict[Any, Any] = {}
        for k in model._parameter_symbols.keys():
            sym = model._parameter_symbols[k]
            if hasattr(model.parameters, k):
                parameters[sym] = getattr(model.parameters, k)

        n_eq = model.n_variables

        def _to_matrix(z, n_rows, n_cols):
            """Coerce a model ZArray / sp.Matrix / 1-D ZArray into an
            sp.Matrix with the requested shape."""
            # Try direct tomatrix() first (rank-2 ZArrays).
            if hasattr(z, "tomatrix"):
                try:
                    m = z.tomatrix()
                except (ValueError, AttributeError):
                    m = None
            else:
                m = z
            if m is None:
                # Rank-1 ZArray: extract via tolist() / iteration.
                if hasattr(z, "tolist"):
                    items = list(z.tolist())
                else:
                    items = [z[i] for i in range(n_rows)]
                # Reshape to (n_rows, n_cols).
                if n_cols == 1:
                    m = sp.Matrix(items)
                else:
                    m = sp.Matrix(n_rows, n_cols,
                                  lambda i, j: items[i * n_cols + j])
                return m
            if not isinstance(m, sp.Matrix):
                m = sp.Matrix(m)
            if m.shape == (n_rows, n_cols):
                return m
            if m.shape == (n_rows * n_cols, 1) or m.shape == (1, n_rows * n_cols):
                return m.reshape(n_rows, n_cols)
            return m

        F = _to_matrix(model.flux(), n_eq, dim)
        P = _to_matrix(model.hydrostatic_pressure(), n_eq, dim)

        ncp_z = model.nonconservative_matrix()
        if hasattr(ncp_z, "todense"):
            B = sp.MutableDenseNDimArray(ncp_z.todense())
        elif hasattr(ncp_z, "tolist"):
            B = sp.MutableDenseNDimArray(ncp_z.tolist())
        else:
            B = sp.MutableDenseNDimArray(ncp_z)

        S_z = model.source()
        S_mat = _to_matrix(S_z, n_eq, 1)

        # Mass matrix: identity by default (canonical form).  Subclasses
        # or system-level transforms (InvertMassMatrix) can override.
        M_mat = sp.eye(n_eq)

        bcs = getattr(model, "_boundary_conditions", None)

        sm = cls(
            time=time_sym,
            space=space,
            state=state,
            aux_state=aux_state,
            parameters=parameters,
            flux=F,
            hydrostatic_pressure=P,
            nonconservative_matrix=B,
            source=S_mat,
            mass_matrix=M_mat,
            boundary_conditions=bcs,
        )
        sm.history.append({"name": "from_model",
                            "description": f"extracted from {type(model).__name__}"})
        return sm

    # ── from_pdesystem factory ────────────────────────────────────────

    @classmethod
    def from_pdesystem(cls, pdesys) -> "SystemModel":
        """Build a SystemModel from a :class:`PDESystem` (e.g. the chain
        DAE returned by ``VAMModelGalerkin._chain_dae``).

        The mass matrix ``M`` is extracted via symbolic linearisation
        around the state itself (``base = f`` for each field).  For
        evolution rows ``M`` carries state-dependent entries
        (e.g. ``u_0`` on the ``h`` column when the equation is
        ``∂_t(h u_0)``); for algebraic rows ``M`` is all-zero — that is
        the DAE form the user wants.

        ``flux``, ``hydrostatic_pressure``, ``nonconservative_matrix``,
        ``source`` are populated by walking the PDESystem equations'
        ``solver_tags`` via :func:`collect_solver_tag`.  If the
        equations carry no tags (plain ``sp.Expr`` instances), those
        slots are left as zero placeholders.
        """
        from zoomy_core.analysis.linearisation import linearise
        from zoomy_core.analysis.pencil import extract_quasilinear_pencil
        from zoomy_core.model.models.tag_extraction import collect_solver_tag

        # Linearise around the symbolic state to extract M_t.
        base_state = {f: f for f in pdesys.fields}
        sys_lin = linearise(pdesys, base_state)
        M_t, _, _ = extract_quasilinear_pencil(sys_lin)

        # State exposed as Symbols (SystemModel convention) — map
        # field Functions ``h(t, x)`` etc. to ``Symbol("h")``.
        state_syms = [sp.Symbol(f.func.__name__, real=True)
                      for f in pdesys.fields]
        func_to_sym = dict(zip(pdesys.fields, state_syms))

        params = (dict(pdesys.parameters)
                  if hasattr(pdesys, "parameters") else {})

        n_eq = len(pdesys.equations)
        n_state = len(pdesys.fields)
        n_dim = len(pdesys.space)
        coords = list(pdesys.space)

        # Decide whether the PDESystem rows are tagged: every equation
        # must be an Expression with a non-empty _solver_groups dict.
        from zoomy_core.model.models.ins_generator import Expression
        tagged = all(
            isinstance(eq, Expression) and bool(eq._solver_groups)
            for eq in pdesys.equations
        )

        if tagged:
            equation_names = list(getattr(pdesys, "equation_names",
                                          [f"eq_{i}" for i in range(n_eq)]))

            class _NamedSystem:
                pass
            sys_obj = _NamedSystem()
            sys_obj.equations = dict(zip(equation_names, pdesys.equations))
            variable_map = {name: [i] for i, name in enumerate(equation_names)}

            def _func_to_sym(M):
                if isinstance(M, sp.Matrix):
                    return M.xreplace(func_to_sym)
                out = sp.MutableDenseNDimArray.zeros(*M.shape)
                for idx in _iter_indices(M.shape):
                    entry = sp.sympify(M[idx])
                    out[idx] = entry.xreplace(func_to_sym)
                return out

            F_func = collect_solver_tag(
                sys_obj, "flux",
                variable_map=variable_map, n_variables=n_eq,
                n_directions=n_dim, coords=coords,
                state_variables=pdesys.fields, policy="strict",
            )
            P_func = collect_solver_tag(
                sys_obj, "hydrostatic_pressure",
                variable_map=variable_map, n_variables=n_eq,
                n_directions=n_dim, coords=coords,
                state_variables=pdesys.fields, policy="strict",
            )
            B_func = collect_solver_tag(
                sys_obj, "nonconservative_flux",
                variable_map=variable_map, n_variables=n_eq,
                n_directions=n_dim, coords=coords,
                state_variables=pdesys.fields, policy="strict",
            )
            S_list = collect_solver_tag(
                sys_obj, "source",
                variable_map=variable_map, n_variables=n_eq,
                policy="strict",
            )
            S_func = sp.Matrix(n_eq, 1, lambda i, _j: S_list[i])
            # The NC slab is (n_eq, n_state, n_dim); collect_solver_tag
            # returned (n_eq, n_eq, n_dim) on the assumption n_eq == n_state.
            # Reshape to (n_eq, n_state, n_dim) — for chain-DAE-derived
            # systems they happen to coincide.
            B_resized = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim)
            for i in range(n_eq):
                for j in range(min(n_state, B_func.shape[1])):
                    for k in range(n_dim):
                        B_resized[i, j, k] = B_func[i, j, k]
            F = _func_to_sym(F_func)
            P = _func_to_sym(P_func)
            B = _func_to_sym(B_resized)
            S_mat = _func_to_sym(S_func)
            mass_descr = (f"chain-DAE PDESystem, {n_eq} equations / "
                          f"{n_state} fields, F/P/B/S extracted via "
                          f"solver_tags")
        else:
            F = sp.zeros(n_eq, n_dim)
            P = sp.zeros(n_eq, n_dim)
            B = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim)
            S_mat = sp.zeros(n_eq, 1)
            mass_descr = (f"PDESystem, {n_eq} equations / {n_state} "
                          f"fields, untagged (F/P/B/S = 0 placeholders)")

        # M_t lives in Function form from linearise; convert to Symbols
        # so the SystemModel state ordering matches.
        M_t_sym = M_t.xreplace(func_to_sym)

        sm = cls(
            time=pdesys.time,
            space=coords,
            state=state_syms,
            aux_state=[],
            parameters=params,
            flux=F,
            hydrostatic_pressure=P,
            nonconservative_matrix=B,
            source=S_mat,
            mass_matrix=M_t_sym,
        )
        sm.history.append({
            "name": "from_pdesystem",
            "description": mass_descr,
        })
        return sm

    # ── describe ──────────────────────────────────────────────────────

    def describe(self, full: bool = False) -> "SystemModelDescription":
        """Return a Description rendering the operator form.  ``full=True``
        includes symbolic flux/NCP/source/mass-matrix entries."""
        return SystemModelDescription(self, full=full)


class SystemModelRow:
    """Row view of a :class:`SystemModel` — exposes row ``i`` of
    every stored matrix."""

    def __init__(self, parent: SystemModel, i: int):
        self._parent = parent
        self._i = i

    @property
    def flux(self):
        return self._parent.flux[self._i, :]

    @property
    def hydrostatic_pressure(self):
        return self._parent.hydrostatic_pressure[self._i, :]

    @property
    def source(self):
        return self._parent.source[self._i, 0]

    @property
    def mass_matrix_row(self):
        return self._parent.mass_matrix[self._i, :]

    @property
    def nonconservative_matrix_row(self):
        n = self._parent.n_equations
        d = self._parent.n_dim
        out = sp.MutableDenseNDimArray.zeros(n, d)
        for j in range(n):
            for k in range(d):
                out[j, k] = self._parent.nonconservative_matrix[self._i, j, k]
        return out


class SystemModelDescription:
    """Markdown / plaintext description of a SystemModel.

    Renders via ``_repr_markdown_`` in Jupyter; ``str(...)`` gives a
    plain-text fallback.
    """

    def __init__(self, sm: SystemModel, *, full: bool = False):
        self._sm = sm
        self._full = full

    def _operator_block(self) -> str:
        sm = self._sm
        parts = []

        # Canonical equation form.
        parts.append("**System form:**")
        parts.append(
            r"$$"
            r"M(Q)\,\partial_t Q "
            r"+ \nabla\cdot\!\big(F(Q) + P(Q)\big)"
            r" + \sum_{d} B_{d}(Q)\,\partial_{d} Q "
            r"- S(Q) = 0"
            r"$$"
        )

        # State vector.
        Q_vec = sp.Matrix(list(sm.state))
        parts.append("**State $Q$:**")
        parts.append(f"$$\n{sp.latex(Q_vec)}\n$$")

        def _render(label, mat):
            if mat.is_zero_matrix:
                parts.append(f"**{label} $= 0$**")
            else:
                parts.append(f"**{label}:**")
                parts.append(f"$$\n{sp.latex(mat)}\n$$")

        _render("Mass matrix $M$", sm.mass_matrix)
        _render("Flux $F$", sm.flux)
        _render("Hydrostatic pressure $P$", sm.hydrostatic_pressure)
        for d in range(sm.n_dim):
            slab = sp.Matrix(sm.n_equations, sm.n_equations,
                             lambda i, j, _d=d:
                                 sm.nonconservative_matrix[i, j, _d])
            label = (f"NCP $B_{{{d}}}$ "
                     f"(direction ${sp.latex(sm.space[d])}$)")
            _render(label, slab)
        _render("Source $S$", sm.source)
        return "\n\n".join(parts)

    def _repr_markdown_(self) -> str:
        sm = self._sm
        parts = [
            f"**SystemModel** — {sm.n_equations} equations, "
            f"{sm.n_dim} spatial dimension{'s' if sm.n_dim != 1 else ''}",
            f"**State:** {', '.join(str(s) for s in sm.state)}",
        ]
        if sm.parameters:
            parts.append(
                "**Parameters:** "
                + ", ".join(f"${sp.latex(s)} = {v}$"
                            for s, v in sm.parameters.items())
            )
        if sm.history:
            parts.append(
                "**Operations:** "
                + ", ".join(h["name"] for h in sm.history)
            )
        if self._full:
            parts.append(self._operator_block())
        return "\n\n".join(parts)

    def __str__(self) -> str:
        md = self._repr_markdown_()
        return md.replace("$$", "").replace("$", "").replace("**", "")

    def __repr__(self) -> str:
        return self.__str__()


# ── System-level operations ───────────────────────────────────────────

class InvertMassMatrix:
    """System-level op: left-multiply the system by ``M⁻¹``.

    Verifies the mass matrix is constant (no state / aux-state atoms),
    inverts symbolically, and applies ``M⁻¹`` to ``flux``,
    ``hydrostatic_pressure``, ``source``, and each per-direction slab
    of ``nonconservative_matrix``.  Sets ``mass_matrix`` to identity.
    """

    name = "invert_mass_matrix"
    description = "Left-multiply by M⁻¹ to reach canonical ∂_t Q form."

    def __call__(self, sm: SystemModel):
        if not sm.is_square:
            raise ValueError(
                "InvertMassMatrix: requires a square system "
                f"(n_eq == n_state); got n_eq={sm.n_equations}, "
                f"n_state={sm.n_state}, "
                f"equation_to_state_index={sm.equation_to_state_index}."
            )
        state_atoms = set(sm.state) | set(sm.aux_state)
        if state_atoms and sm.mass_matrix.has(*state_atoms):
            raise ValueError(
                "InvertMassMatrix: mass_matrix has state-dependent "
                "entries — non-constant mass matrices are unsupported.\n"
                f"Mass matrix:\n{sm.mass_matrix}"
            )

        if sm.mass_matrix == sp.eye(sm.n_equations):
            return

        M_inv = sm.mass_matrix.inv()
        sm.flux = M_inv * sm.flux
        sm.hydrostatic_pressure = M_inv * sm.hydrostatic_pressure
        sm.source = M_inv * sm.source

        n = sm.n_equations
        d = sm.n_dim
        new_B = sp.MutableDenseNDimArray.zeros(n, n, d)
        for k in range(d):
            slab = sp.Matrix(n, n,
                             lambda i, j, _k=k:
                                 sm.nonconservative_matrix[i, j, _k])
            slab = M_inv * slab
            for i in range(n):
                for j in range(n):
                    new_B[i, j, k] = slab[i, j]
        sm.nonconservative_matrix = new_B
        sm.mass_matrix = sp.eye(n)


__all__ = [
    "SystemModel",
    "SystemModelRow",
    "SystemModelDescription",
    "InvertMassMatrix",
]
