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


@dataclass
class SystemModel:
    """Symbolic operator-form PDE system.

    Stored matrices follow the contract:

    .. math::

        M(Q) \\; \\partial_t Q
            + \\nabla \\cdot \\big(F(Q) + P(Q)\\big)
            + \\sum_d B(Q)[:,:,d] \\, \\partial_d Q
            - S(Q) = 0

    Attributes are sympy ``Matrix`` / ``MutableDenseNDimArray``
    objects; row/column indexing follows ``state`` order.
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
    boundary_conditions: Optional[Any] = None
    history: List[Dict[str, str]] = field(default_factory=list)

    # ── Shape accessors ────────────────────────────────────────────────

    @property
    def n_equations(self) -> int:
        return self.flux.shape[0]

    @property
    def n_dim(self) -> int:
        return len(self.space)

    # ── Operator-API methods (mirror Model) ─────────────────────────────

    def quasilinear_matrix(self):
        """``∂F/∂Q + ∂P/∂Q + B`` — shape ``(n_eq, n_eq, n_dim)``."""
        n = self.n_equations
        d = self.n_dim
        Q = sp.MutableDenseNDimArray.zeros(n, n, d)
        for i in range(n):
            for j in range(n):
                for k in range(d):
                    djF = sp.diff(self.flux[i, k], self.state[j])
                    djP = sp.diff(self.hydrostatic_pressure[i, k],
                                  self.state[j])
                    Q[i, j, k] = djF + djP + self.nonconservative_matrix[i, j, k]
        return Q

    def source_jacobian_wrt_state(self):
        """``∂S/∂Q`` — shape ``(n_eq, n_eq)``."""
        n = self.n_equations
        out = sp.zeros(n, n)
        for i in range(n):
            for j in range(n):
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
        if any(sm.flux):
            parts.append("**Flux** $F$:")
            parts.append(f"$$\n{sp.latex(sm.flux)}\n$$")
        if any(sm.hydrostatic_pressure):
            parts.append("**Hydrostatic pressure** $P$:")
            parts.append(f"$$\n{sp.latex(sm.hydrostatic_pressure)}\n$$")
        for d in range(sm.n_dim):
            slab = sp.Matrix(sm.n_equations, sm.n_equations,
                             lambda i, j, _d=d:
                                 sm.nonconservative_matrix[i, j, _d])
            if any(slab):
                parts.append(
                    f"**NCP** $B_{{{d}}}$ (direction ${sp.latex(sm.space[d])}$):"
                )
                parts.append(f"$$\n{sp.latex(slab)}\n$$")
        if any(sm.source):
            parts.append("**Source** $S$:")
            parts.append(f"$$\n{sp.latex(sm.source)}\n$$")
        if sm.mass_matrix != sp.eye(sm.n_equations):
            parts.append("**Mass matrix** $M$ (non-identity):")
            parts.append(f"$$\n{sp.latex(sm.mass_matrix)}\n$$")
        else:
            parts.append("**Mass matrix** $M = I$ (canonical form).")
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
