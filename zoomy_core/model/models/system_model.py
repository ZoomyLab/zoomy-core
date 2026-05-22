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

import numpy as np
import sympy as sp

from zoomy_core.misc.misc import Zstruct, ZArray


def _to_zarray(obj):
    """Coerce a sympy Matrix / NDimArray / list into ``ZArray``.

    Accepts ``None`` (returns None) and existing ``ZArray`` (returns
    as-is).  Shape is preserved exactly — column vectors stay
    ``(n, 1)`` Matrix → ``(n, 1)`` ZArray.  Construction sites are
    responsible for choosing the rank they want; this normaliser
    just unifies the storage type.
    """
    if obj is None or isinstance(obj, ZArray):
        return obj
    if isinstance(obj, sp.MatrixBase):
        return ZArray(obj.tolist(), shape=tuple(obj.shape))
    if isinstance(obj, sp.NDimArray):
        return ZArray(obj.tolist(), shape=tuple(obj.shape))
    return ZArray(obj)


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

    Every operator the system needs is a **stored field**, frozen at
    construction — there are no derived-on-demand methods.  Alongside
    the five primaries (flux, P, NCP, source, mass matrix) the system
    carries the derived operators ``quasilinear_matrix``,
    ``source_jacobian`` and ``eigenvalues``, plus the ``normal``
    symbols and the parameter Zstructs.  ``quasilinear_matrix`` and
    ``source_jacobian`` are cheap derivatives — ``__post_init__``
    computes them from the primaries when not supplied, and any
    in-place operation refreshes them via
    :meth:`refresh_derived_operators`.  ``eigenvalues`` is the one
    operator that may be expensive (a symbolic spectral derivation) or
    deliberately skipped — :meth:`from_model` simply carries over
    whatever ``Model.eigenvalues()`` produced, and ``None`` means
    "skipped" (numerical-eigenvalue mode).

    Shape contract:

    * ``flux``                  — ``(n_eq, n_dim)``
    * ``hydrostatic_pressure``  — ``(n_eq, n_dim)``
    * ``nonconservative_matrix``— ``(n_eq, n_state, n_dim)``
    * ``source``                — ``(n_eq, 1)``
    * ``mass_matrix``           — ``(n_eq, n_state)``
    * ``quasilinear_matrix``    — ``(n_eq, n_state, n_dim)``
    * ``source_jacobian``       — ``(n_eq, n_state)``
    * ``eigenvalues``           — ``(n_eq, 1)`` or ``None`` if skipped

    ``parameters`` is a Zstruct mapping parameter name → symbol (the
    symbolic side the operators reference); ``parameter_values`` maps
    name → numeric default.  ``normal`` is a Zstruct of the face-normal
    component symbols ``n0, n1, …``.
    """

    time: sp.Symbol
    space: List[sp.Symbol]
    state: List[Any]
    aux_state: List[Any]
    parameters: Zstruct                      # name -> Symbol
    # Every operator tensor is a :class:`ZArray` — the project's
    # unified symbolic tensor type.  ZArray inherits from
    # :class:`sp.MutableDenseNDimArray`, so it is a drop-in for any
    # sympy NDimArray consumer (lambdify, xreplace, atoms, etc.) while
    # carrying our matrix-algebra extensions (``@``, ``+``, ``-`` with
    # lists / Zstructs).  Eigenvalue computation (the one place that
    # genuinely needs ``sp.Matrix``) wraps via ``zarray.tomatrix()`` →
    # ``M.eigenvals()`` → ``ZArray(...)``.
    flux: ZArray                             # (n_eq, n_dim)
    hydrostatic_pressure: ZArray             # (n_eq, n_dim)
    nonconservative_matrix: ZArray           # (n_eq, n_state, n_dim)
    source: ZArray                           # (n_eq,)
    mass_matrix: ZArray                      # (n_eq, n_state)
    # Derived operators — frozen alongside the primaries.  ``None`` ⇒
    # ``__post_init__`` computes quasilinear_matrix / source_jacobian
    # from the primaries; ``eigenvalues`` stays ``None`` when skipped.
    quasilinear_matrix: Optional[ZArray] = None    # (n_eq, n_state, n_dim)
    source_jacobian: Optional[ZArray] = None       # (n_eq, n_state)
    eigenvalues: Optional[ZArray] = None           # (n_eq,)
    normal: Optional[Zstruct] = None
    parameter_values: Optional[Zstruct] = None   # name -> numeric default
    equation_to_state_index: Optional[List[int]] = None
    # ``boundary_conditions`` / ``aux_boundary_conditions`` are the
    # *indexed symbolic BC functions* (``Function`` objects with
    # ``Piecewise(…, Eq(idx,i))`` definition).  The runtime
    # (``from_system_model``) lambdifies them; solvers call
    # ``runtime.boundary_conditions(bc_idx, time, position, distance,
    # Q, Qaux, p, normal)`` per face.
    # ``boundary_gradients`` is the parallel kernel for the
    # face-normal gradient ``∂Q/∂n`` — consumed by the diffusion path.
    boundary_conditions: Optional[Any] = None
    aux_boundary_conditions: Optional[Any] = None
    boundary_gradients: Optional[Any] = None
    # Application-logic carried over from the source Model so the
    # SystemModel is the self-contained numerical model a solver
    # consumes — no Model needed alongside it.
    initial_conditions: Optional[Any] = None
    aux_initial_conditions: Optional[Any] = None
    # ``update_variables``: per-cell state transform (h-clamp etc.) — an
    # ``(n_eq, 1)`` expression in ``(Q, Qaux, p)``.
    #
    # Two-slot IMEX operator design — solvers route by treatment:
    #
    # - ``source``                — implicit (default).  Added to the
    #   source-step Newton residual at ``Qnp1``; appropriate for stiff
    #   friction / reaction.
    # - ``source_explicit``       — explicit.  Added to the convective
    #   step at ``Qn``; appropriate for body forces / gravity / cheap
    #   non-stiff RHS.
    # - ``diffusion_matrix``      — implicit (default).  IP-DG/TPFA
    #   evaluated at ``Qnp1`` inside the source step.  No parabolic CFL.
    # - ``diffusion_matrix_explicit`` — explicit.  IP-DG/TPFA evaluated
    #   at ``Qn`` inside the convective step.  Subject to
    #   ``dt ≤ h²/(2ν)`` on top of the hyperbolic CFL.
    #
    # The constitutive tensor ``A(Q, Qaux, p)`` has shape
    # ``(n_eq, n_state, n_dim, n_dim)``; the diffusive flux is recovered
    # downstream as ``F_diff[i, d] = Σ_{j, e} A[i, j, d, e] · ∂_e Q[j]``,
    # and the residual contributes ``-∇·(A:∇Q)``.  Derivatives enter
    # ``A`` only via ``Qaux`` (auto-derived as needed) so ``A`` itself
    # is a pure ``(Q, Qaux, p)`` expression — same call convention as
    # every other operator.
    #
    # Backends that support only one treatment (e.g. explicit-only FV)
    # may *compound* the two: ``F_total = F_explicit + F_implicit``.
    update_variables: Optional[ZArray] = None        # (n_eq,)
    diffusion_matrix: Optional[ZArray] = None         # (n_eq, n_state, n_dim, n_dim) — implicit
    diffusion_matrix_explicit: Optional[ZArray] = None  # (n_eq, n_state, n_dim, n_dim) — explicit
    source_explicit: Optional[ZArray] = None          # (n_eq,) — explicit
    # ``reconstruction_variables``: symbolic forward map state → primitive
    # well-balanced variables used by MUSCL-style reconstruction
    # (η = h+b, u = q_U/h, …).  ``state_from_reconstruction`` is the
    # symbolic inverse in terms of fresh ``WB_<state_name>`` symbols,
    # auto-derived from the forward via
    # :func:`zoomy_core.model.reconstruction_inverse.invert_reconstruction`.
    # See ``thesis/chapters/30_numerics.md`` "Primitive-variable MUSCL
    # reconstruction".  The pair is refreshed whenever the state vector
    # changes (CoV via :meth:`change_state_variables`).
    reconstruction_variables: Optional[ZArray] = None      # (n_state,)
    state_from_reconstruction: Optional[ZArray] = None     # (n_state,) in WB_<name> symbols
    # ``state_update``: rank-1 ZArray of length ``len(equation_to_state_index)``
    # encoding an explicit-update operator ``Q[e2s] ← state_update(Q, Qaux, p, dt)``.
    # When set, the solver dispatches this substep as in-place assignment
    # rather than residual semantics (mass_matrix is implicitly zero).
    state_update: Optional[ZArray] = None
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        # Normalise every operator tensor field to ZArray.  Construction
        # sites that still produce sympy types are wrapped transparently.
        self.flux                   = _to_zarray(self.flux)
        self.hydrostatic_pressure   = _to_zarray(self.hydrostatic_pressure)
        self.nonconservative_matrix = _to_zarray(self.nonconservative_matrix)
        self.source                 = _to_zarray(self.source)
        self.mass_matrix            = _to_zarray(self.mass_matrix)
        self.quasilinear_matrix     = _to_zarray(self.quasilinear_matrix)
        self.source_jacobian        = _to_zarray(self.source_jacobian)
        self.eigenvalues            = _to_zarray(self.eigenvalues)
        self.update_variables           = _to_zarray(self.update_variables)
        self.diffusion_matrix           = _to_zarray(self.diffusion_matrix)
        self.diffusion_matrix_explicit  = _to_zarray(self.diffusion_matrix_explicit)
        self.source_explicit            = _to_zarray(self.source_explicit)
        self.state_update               = _to_zarray(self.state_update)
        self.reconstruction_variables   = _to_zarray(self.reconstruction_variables)
        self.state_from_reconstruction  = _to_zarray(self.state_from_reconstruction)

        if self.equation_to_state_index is None:
            self.equation_to_state_index = list(range(self.n_equations))
        if self.normal is None:
            self.normal = Zstruct(
                **{f"n{d}": sp.Symbol(f"n{d}", real=True)
                   for d in range(self.n_dim)}
            )
            self.normal._symbolic_name = "n"
        if self.parameter_values is None:
            self.parameter_values = Zstruct()
        if self.quasilinear_matrix is None:
            self.quasilinear_matrix = _to_zarray(self._compute_quasilinear_matrix())
        if self.source_jacobian is None:
            self.source_jacobian = _to_zarray(self._compute_source_jacobian())

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

    # ── Model-compatible facade ───────────────────────────────────────
    # So a solver can consume a SystemModel exactly where it used to
    # consume a Model — same accessor surface, SystemModel internals.

    @property
    def n_variables(self) -> int:
        return self.n_equations

    @property
    def n_aux_variables(self) -> int:
        return len(self.aux_state)

    @property
    def n_parameters(self) -> int:
        return self.parameters.length()

    @property
    def dimension(self) -> int:
        return self.n_dim

    @property
    def variables(self) -> Zstruct:
        """Zstruct view ``name → state Symbol`` (mirrors ``Model.variables``)."""
        v = Zstruct(**{str(s): s for s in self.state})
        v._symbolic_name = "Q"
        return v

    @property
    def aux_variables(self) -> Zstruct:
        """Zstruct view ``name → aux Symbol`` (mirrors ``Model.aux_variables``)."""
        v = Zstruct(**{str(s): s for s in self.aux_state})
        v._symbolic_name = "Qaux"
        return v

    @property
    def _parameter_symbols(self) -> Zstruct:
        """Alias of ``parameters`` (name → Symbol) for Model-API parity."""
        return self.parameters

    @property
    def eigenvalue_mode(self) -> str:
        """``"symbolic"`` when a symbolic spectrum was carried over,
        ``"numerical"`` when it was skipped (``eigenvalues is None``) —
        mirrors ``Model.eigenvalue_mode`` so a solver's wavespeed build
        branches the same way."""
        return "numerical" if self.eigenvalues is None else "symbolic"

    @property
    def is_square(self) -> bool:
        """True when n_equations == n_state and equations map identity-style."""
        return (self.n_equations == self.n_state
                and self.equation_to_state_index == list(range(self.n_state)))

    @property
    def stationary_indices(self) -> frozenset:
        """State indices whose evolution is identically zero by
        construction.

        A field ``Q[i]`` is **stationary** iff its row vanishes in
        every transport / source / diffusion slot:

        - ``flux[i, :] = 0``
        - ``hydrostatic_pressure[i, :] = 0``
        - ``nonconservative_matrix[i, :, :] = 0``
        - ``source[i] = 0``
        - ``source_explicit[i] = 0`` (if present)
        - ``diffusion_matrix[i, :, :, :] = 0`` (if present)
        - ``diffusion_matrix_explicit[i, :, :, :] = 0`` (if present)

        The discrete time-step then writes nothing to ``Q[i]``, so any
        post-step hook (slope limiter, positivity projection, ...)
        that does is introducing **spurious** modification — the
        canonical example is a vertex-based DG slope limiter clipping
        the bathymetry ``b`` row in a shallow-water model, which
        breaks rank-boundary consistency under MPI and silently
        creates or destroys topography.

        Solvers should consult this set to skip such fields when
        applying post-step state transforms.  Users can override (in
        either direction) via per-solver kwargs that resolve handles
        through :meth:`field_index`.
        """
        n = self.n_variables
        slots = (
            self.flux, self.hydrostatic_pressure,
            self.nonconservative_matrix,
            self.source, self.source_explicit,
            self.diffusion_matrix, self.diffusion_matrix_explicit,
        )
        return frozenset(
            i for i in range(n)
            if all(self._row_is_identically_zero(t, i) for t in slots)
        )

    @staticmethod
    def _row_is_identically_zero(tensor, i) -> bool:
        """Return ``True`` if ``tensor[i, ...]`` is identically zero.

        ``None`` (slot absent) counts as zero.  Entries are checked
        with ``== 0`` only (catches ``sp.S.Zero`` / ``sp.Integer(0)``
        / ``0`` — the canonical "stationary row" produced by
        ``Matrix.zeros`` / ``sp.MutableDenseNDimArray.zeros`` plus the
        model author's explicit zero assignment).  We deliberately do
        **NOT** fall back to ``sp.simplify`` here — the source / NCP /
        diffusion rows of non-stationary fields routinely contain
        deeply nested expressions (Manning friction = nested
        ``sqrt`` ∘ ``Max`` ∘ ``Rational``) on which ``sp.simplify``
        runs for many minutes.  Stationary detection runs at solver
        setup, so any pessimisation hits the user-facing startup
        latency.  If a model author writes a non-trivially-simplified
        expression that *happens* to evaluate to zero, they should
        either simplify it in the model or list the field manually
        via ``solver.limiter_exclude_fields``.
        """
        if tensor is None:
            return True
        try:
            row = tensor[i]
        except (IndexError, TypeError):
            return True
        try:
            flat = np.asarray(row, dtype=object).ravel()
        except Exception:
            flat = [row]
        for entry in flat:
            if entry != 0:
                return False
        return True

    def field_index(self, field) -> int:
        """Resolve a state-field handle to its integer index.

        Accepts:

        - ``sp.Symbol`` (e.g. ``self.variables.b``) — matched by name.
        - ``str`` (e.g. ``"b"``) — matched by name.
        - ``int`` — returned as-is (with bounds check).

        Raises ``ValueError`` if the handle does not correspond to a
        registered state variable.  This is the canonical way for
        solvers to translate the *symbolic* field handles that the
        model author writes (``[model.variables.b]``) into the
        *integer* indices the post-processing pipeline uses
        internally — without ever forcing the model author to think
        about state layout.
        """
        names = list(self.variables.keys())
        if isinstance(field, str):
            try:
                return names.index(field)
            except ValueError as e:
                raise ValueError(
                    f"Unknown state field {field!r}; available: {names}"
                ) from e
        if isinstance(field, (int, np.integer)):
            idx = int(field)
            if not 0 <= idx < len(names):
                raise ValueError(
                    f"State index {idx} out of range [0, {len(names)})"
                )
            return idx
        name = getattr(field, "name", None)
        if name is None:
            raise ValueError(
                f"Cannot resolve field handle {field!r} (type "
                f"{type(field).__name__}); expected sp.Symbol, str, "
                "or int."
            )
        try:
            return names.index(name)
        except ValueError as e:
            raise ValueError(
                f"Unknown state field {name!r}; available: {names}"
            ) from e

    # ── Derived-operator computation ────────────────────────────────────

    def _compute_quasilinear_matrix(self):
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

    def _compute_source_jacobian(self):
        """``∂S/∂Q`` — shape ``(n_eq, n_state)``."""
        n_eq = self.n_equations
        n_st = self.n_state
        out = sp.zeros(n_eq, n_st)
        for i in range(n_eq):
            for j in range(n_st):
                out[i, j] = sp.diff(self.source[i, 0], self.state[j])
        return out

    def _compute_eigenvalues(self):
        """Eigenvalues of the normal-projected quasilinear matrix —
        ``(n_eq, 1)``.  This is the (potentially expensive) symbolic
        spectral derivation; callers decide when to pay it."""
        n_eq = self.n_equations
        n = list(self.normal.values())
        A = sp.zeros(n_eq, n_eq)
        for k in range(self.n_dim):
            for i in range(n_eq):
                for j in range(n_eq):
                    A[i, j] += n[k] * self.quasilinear_matrix[i, j, k]
        lam = sp.Symbol("lam")
        evs = sp.solve(A.charpoly(lam), lam)
        return sp.Matrix(n_eq, 1, lambda i, _j: sp.simplify(evs[i]))

    def refresh_derived_operators(self, *, eigenvalues: bool = False):
        """Recompute ``quasilinear_matrix`` and ``source_jacobian`` from
        the (possibly just-mutated) primary operators, keeping the
        derived operators consistent.  ``eigenvalues=True`` also
        re-derives the spectrum — only request it when an operation
        genuinely changes the system's characteristic structure and the
        spectrum was not skipped."""
        self.quasilinear_matrix = self._compute_quasilinear_matrix()
        self.source_jacobian = self._compute_source_jacobian()
        if eigenvalues and self.eigenvalues is not None:
            self.eigenvalues = self._compute_eigenvalues()

    def assert_diagonal_mass_matrix(self):
        """Consistency check: verify ``mass_matrix`` is diagonal on
        every evolution row and zero on every algebraic row.

        - Evolution row ``i`` (with ``equation_to_state_index[i] = j``):
          must have ``M[i, k] = 0`` for every ``k != j``, and
          ``M[i, j] != 0``.
        - Algebraic row ``i``: ``M[i, :] = 0``.

        Raises ``ValueError`` with a precise location report on
        violation.  Run this **after** ``change_state_variables`` to
        confirm the variable choice produces a clean diagonal mass
        matrix — otherwise the variable map is wrong for the system
        at hand.

        Once this passes, ``InvertMassMatrix`` is a trivial per-row
        division by the diagonal entry; downstream consumers see
        ``M = I`` on evolution rows.
        """
        n_eq = self.n_equations
        n_st = self.n_state
        e2s = list(self.equation_to_state_index)
        offenders = []
        for i in range(n_eq):
            diag_col = e2s[i]
            for k in range(n_st):
                if k == diag_col:
                    continue
                entry = sp.simplify(self.mass_matrix[i, k])
                if entry != 0:
                    offenders.append((i, k, entry))
        if offenders:
            msg_lines = [
                "Mass matrix is not diagonal on evolution rows:",
            ]
            for (i, k, entry) in offenders:
                msg_lines.append(
                    f"  M[row={i}, state_col={k}={self.state[k]}] = {entry}"
                )
            msg_lines.append(
                "Pick a state-variable transformation in "
                "``change_state_variables`` that diagonalises M, or "
                "address the system structurally."
            )
            raise ValueError("\n".join(msg_lines))

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
        parameters = Zstruct(
            **{k: model._parameter_symbols[k]
               for k in model._parameter_symbols.keys()}
        )
        parameters._symbolic_name = "p"
        parameter_values = Zstruct(
            **{k: getattr(model.parameters, k, 0.0)
               for k in model._parameter_symbols.keys()}
        )
        normal = Zstruct(
            **{k: model.normal[k] for k in model.normal.keys()}
        )
        normal._symbolic_name = "n"

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

        def _extract_A(name):
            """Pull a rank-4 ``(n_eq, n_state, n_dim, n_dim)`` ZArray
            from ``model.<name>()`` if the model defines it; otherwise
            return an explicit zero tensor of that shape."""
            if hasattr(model, name) and callable(getattr(model, name)):
                A_z = getattr(model, name)()
                if hasattr(A_z, "todense"):
                    return sp.MutableDenseNDimArray(A_z.todense())
                if hasattr(A_z, "tolist"):
                    return sp.MutableDenseNDimArray(A_z.tolist())
                return sp.MutableDenseNDimArray(A_z)
            return sp.MutableDenseNDimArray.zeros(n_eq, n_eq, dim, dim)

        A_diff_impl = _extract_A("diffusion_matrix")
        A_diff_expl = _extract_A("diffusion_matrix_explicit")

        S_z = model.source()
        S_mat = _to_matrix(S_z, n_eq, 1)
        # Explicit source slot — defaults to zero when the model does
        # not opt in by overriding ``source_explicit()``.
        if (hasattr(model, "source_explicit")
                and callable(model.source_explicit)):
            S_expl_mat = _to_matrix(model.source_explicit(), n_eq, 1)
        else:
            S_expl_mat = sp.zeros(n_eq, 1)

        # Mass matrix: prefer ``model.mass_matrix()`` if the model
        # exposes one (chain-derived models do); otherwise default to
        # identity (canonical form for operator-API-only models).
        if hasattr(model, "mass_matrix") and callable(model.mass_matrix):
            M_mat = _to_matrix(model.mass_matrix(), n_eq, n_eq)
        else:
            M_mat = sp.eye(n_eq)

        equation_names = getattr(model, "equation_names", None)
        # Carry the *indexed symbolic BC function* ``_boundary_conditions``
        # — a ``Function(name="boundary_conditions",
        #              args=(idx, time, position, distance, Q, Qaux, p, normal),
        #              definition=Piecewise((res_i, Eq(idx, i)), …))``.
        # This is the representation a SystemModel-driven runtime
        # lambdifies and the solver flux operators call per face.
        bcs = model._boundary_conditions
        aux_bcs = model._aux_boundary_conditions
        bgrads = model._boundary_gradients

        # Eigenvalues — carry over whatever ``Model.eigenvalues()``
        # produced.  In numerical-eigenvalue mode the model deliberately
        # skips the spectral derivation, so the SystemModel records
        # ``None`` ("skipped") rather than a zero placeholder.
        eigenvalues = None
        if getattr(model, "eigenvalue_mode", "symbolic") != "numerical":
            ev_def = model.functions["eigenvalues"].definition
            if ev_def is not None:
                eigenvalues = _to_matrix(ev_def, n_eq, 1)

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
            eigenvalues=eigenvalues,
            normal=normal,
            parameter_values=parameter_values,
            boundary_conditions=bcs,
            aux_boundary_conditions=aux_bcs,
            boundary_gradients=bgrads,
            initial_conditions=getattr(model, "initial_conditions", None),
            aux_initial_conditions=getattr(
                model, "aux_initial_conditions", None),
            update_variables=_to_matrix(model.update_variables(), n_eq, 1),
            diffusion_matrix=A_diff_impl,
            diffusion_matrix_explicit=A_diff_expl,
            source_explicit=S_expl_mat,
            reconstruction_variables=_to_zarray(
                model.reconstruction_variables()),
            state_from_reconstruction=_to_zarray(
                model.state_from_reconstruction()),
        )
        if equation_names is not None:
            sm.equation_names = list(equation_names)
        sm.history.append({"name": "from_model",
                            "description": f"extracted from {type(model).__name__}"})
        # Auto-scan: every non-state Function atom and every Derivative
        # atom in the operator matrices becomes an aux Symbol with a
        # structured registry entry.  Solvers walk ``sm.aux_registry``
        # to compute aux values per step.
        sm.expose_aux_atoms()
        return sm

    # ── reconstruct_residuals ─────────────────────────────────────────

    def reconstruct_residuals(self):
        """Re-assemble the row-wise residuals ``M·∂_t Q + ∂_x F + ∂_x P
        + B·∂_x Q − S`` for every equation.  Returns a list of sympy
        expressions, indexed by row.

        Aux Symbols generated by ``expose_aux_atoms`` are back-
        substituted to their original ``Function`` / ``Derivative``
        atoms so the residuals display in their natural form for
        bit-for-bit fixture comparisons.

        Used in tests to compare the operator-form back against the
        original equation residuals (e.g. against Escalante eq (4)).
        """
        n_eq = self.n_equations
        n_st = self.n_state
        n_dim = self.n_dim
        t = self.time
        coords = self.space

        # Build per-state time- and spatial-derivative Functions if the
        # state is held as Symbols.  We need a coordinate-dependent
        # representation to take symbolic derivatives.
        def _as_function(sym):
            if isinstance(sym, sp.Function):
                return sym
            args = [t] + list(coords)
            return sp.Function(str(sym), real=True)(*args)

        state_funcs = [_as_function(s) for s in self.state]
        sym_to_func = dict(zip(self.state, state_funcs))
        aux_reverse = self._aux_reverse_map()

        def _restore(expr):
            """First put back the original aux atoms (which still
            reference state as Symbols), then upgrade state Symbols
            to Functions so derivatives display naturally."""
            return (sp.sympify(expr)
                    .xreplace(aux_reverse)
                    .xreplace(sym_to_func))

        residuals = []
        for i in range(n_eq):
            res = sp.S.Zero
            # M · ∂_t Q
            for j in range(n_st):
                m_ij = _restore(self.mass_matrix[i, j])
                if m_ij != 0:
                    res = res + m_ij * sp.Derivative(state_funcs[j], t)
            # ∂_x F[i] (and ∂_y if 2D)
            for d in range(n_dim):
                f_id = _restore(self.flux[i, d])
                if f_id != 0:
                    res = res + sp.Derivative(f_id, coords[d])
                p_id = _restore(self.hydrostatic_pressure[i, d])
                if p_id != 0:
                    res = res + sp.Derivative(p_id, coords[d])
            # B · ∂_x Q
            for j in range(n_st):
                for d in range(n_dim):
                    b_ijd = _restore(self.nonconservative_matrix[i, j, d])
                    if b_ijd != 0:
                        res = res + b_ijd * sp.Derivative(state_funcs[j], coords[d])
            # − S[i]
            s_i = _restore(self.source[i, 0])
            res = res - s_i
            residuals.append(sp.expand(res))
        return residuals

    # ── from_pdesystem factory (REMOVED) ──────────────────────────────


    # ── expose_aux_atoms (auto-scan) ──────────────────────────────────

    def expose_aux_atoms(self):
        """Auto-scan the operator matrices and route every
        non-state symbolic input into ``aux_state``.

        For each unique:

        * **bare** ``Function`` atom whose name is **not** a state
          variable (e.g. topography ``b(t, x, y)``, externally
          prescribed forcing fields) — substitute with an aux Symbol
          of the same name;
        * ``Derivative`` atom — substitute with an aux Symbol named
          ``{target}_{axes}`` (e.g. ``b_x``, ``h_x``, ``b_x_x``);

        and populate :attr:`aux_registry` with one structured dict per
        new aux entry:

        .. code-block:: python

            {"kind":   "function" | "derivative",
             "name":   <aux symbol name>,
             "row":    <index in self.aux_state>,
             "atom":   <original sympy atom>,
             "aux_symbol": <Symbol>,
             # derivative entries also carry:
             "target_name":  <name of the field being differentiated>,
             "target_kind":  "state" | "function" | "unknown",
             "state_index":  <int>          # if target_kind == "state"
             "function_row": <int>          # if target_kind == "function"
             "multi_index":  <tuple>        # spatial-derivative orders
                                            # in (axis_0, axis_1, …)}

        ``multi_index`` follows the SAME convention as
        :meth:`LSQMesh.compute_derivatives` so a solver can compute
        every derivative-aux entry in one call.  Time derivatives are
        skipped (they live in the mass matrix).

        Idempotent — if ``aux_registry`` is already non-empty (or all
        atoms are state) this is a no-op.
        """
        import sympy as sp
        from itertools import product as _product

        if getattr(self, "aux_registry", None) is not None:
            return

        state_names = {str(s) for s in self.state}
        space_names = {str(s): d for d, s in enumerate(self.space)}
        n_dim = len(self.space)

        def _iter_entries(M):
            if isinstance(M, sp.Matrix):
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        yield M[i, j]
            else:
                for idx in _product(*[range(s) for s in M.shape]):
                    yield M[idx]

        matrices = [self.flux, self.hydrostatic_pressure,
                    self.nonconservative_matrix, self.source,
                    self.mass_matrix]
        if self.state_update is not None:
            matrices.append(self.state_update)
        # State-dependent diffusion (e.g. MY-2.5 K_M(ℓ, q, G_H)) can
        # carry ``Derivative(state, x)`` atoms via the Galperin G_H
        # argument; auto-expose them so the rank-4 A tensor lambdifies
        # cleanly through the Firedrake-DG / NumpyRuntimeModel path.
        if self.diffusion_matrix is not None:
            matrices.append(self.diffusion_matrix)
        if self.diffusion_matrix_explicit is not None:
            matrices.append(self.diffusion_matrix_explicit)

        # ── Pass 1: collect *user-defined* Function atoms (excl. state).
        # ``atoms(sp.Function)`` matches every Function subclass — including
        # built-ins like ``Piecewise``, ``Min``, ``Max``, ``conditional``.
        # Those should stay inline; only undefined / user-named function
        # atoms (e.g. topography ``b(t, x, y)`` declared via
        # ``sp.Function("b")``) are candidates for promotion to aux state.
        # Filter via ``sp.AppliedUndef`` which is precisely "user-defined
        # function application".
        function_atoms = {}     # name → [atoms]
        for M in matrices:
            for entry in _iter_entries(M):
                for atom in sp.sympify(entry).atoms(sp.core.function.AppliedUndef):
                    name = atom.func.__name__
                    if name in state_names:
                        continue
                    function_atoms.setdefault(name, []).append(atom)

        # ── Pass 1.5: collect ``limit(Derivative(...), scheme)`` atoms.
        # These flag a derivative for runtime TVD limiting via the
        # numerics layer (see ``zoomy_core.model.numerics.limit``).  We
        # substitute each with a dedicated aux Symbol BEFORE the plain
        # Derivative scan below — so the limited derivative gets a
        # ``limited_derivative`` aux entry with a ``limiter_scheme``
        # field, and the un-wrapped Derivative atom is no longer present
        # to be picked up by Pass 2.
        from zoomy_core.model.numerics import limit as _limit_fn
        limit_atoms = {}   # (target_name, multi_index_tuple, scheme) → [atoms]
        for M in matrices:
            for ent in _iter_entries(M):
                for atom in sp.sympify(ent).atoms(_limit_fn):
                    inner = atom.args[0]
                    scheme = str(atom.args[1])
                    if not isinstance(inner, sp.Derivative):
                        continue
                    target = inner.args[0]
                    target_name = (target.func.__name__
                                   if isinstance(target, sp.Function)
                                   else str(target))
                    mi = [0] * n_dim
                    has_time = False
                    for var, n in inner.variable_count:
                        vn = str(var)
                        if vn in space_names:
                            mi[space_names[vn]] += int(n)
                        else:
                            has_time = True
                            break
                    if has_time or all(o == 0 for o in mi):
                        continue
                    key = (target_name, tuple(mi), scheme)
                    limit_atoms.setdefault(key, []).append(atom)

        # ── Pass 2: collect Derivative atoms (skip time-derivs). ────
        # Note: any Derivative that was inside a ``limit(...)`` wrapper
        # is collected in Pass 1.5 above and skipped here (the .atoms
        # scan walks the tree, so it WOULD pick up the inner derivative
        # too — we filter by checking whether the parent expression
        # contains a wrapping ``limit`` atom).
        deriv_atoms = {}    # (target_name, multi_index_tuple) → [atoms]
        limited_inner_atoms = set()
        for lst in limit_atoms.values():
            for a in lst:
                limited_inner_atoms.add(a.args[0])
        for M in matrices:
            for entry in _iter_entries(M):
                for d in sp.sympify(entry).atoms(sp.Derivative):
                    if d in limited_inner_atoms:
                        continue
                    target = d.args[0]
                    target_name = (target.func.__name__
                                   if isinstance(target, sp.Function)
                                   else str(target))
                    mi = [0] * n_dim
                    has_time = False
                    for var, n in d.variable_count:
                        vn = str(var)
                        if vn in space_names:
                            mi[space_names[vn]] += int(n)
                        else:
                            has_time = True
                            break
                    if has_time or all(o == 0 for o in mi):
                        continue
                    key = (target_name, tuple(mi))
                    deriv_atoms.setdefault(key, []).append(d)

        if not function_atoms and not deriv_atoms:
            self.aux_registry = []
            return

        registry = []
        sub_dict = {}
        n_aux_before = len(self.aux_state)
        new_syms = []
        function_row_of_name = {}

        # ── Function entries first (so derivative entries can ─────
        # reference them by row in the registry). ──────────────────
        for name, atoms in function_atoms.items():
            sym = sp.Symbol(name, real=True)
            for a in atoms:
                sub_dict[a] = sym
            row = n_aux_before + len(new_syms)
            new_syms.append(sym)
            function_row_of_name[name] = row
            registry.append({
                "kind": "function",
                "name": name,
                "row": row,
                "atom": atoms[0],
                "aux_symbol": sym,
            })

        # ── Limited-derivative entries (from Pass 1.5). ──────────
        # Substitute the ``limit(D, scheme)`` atom with a dedicated aux
        # Symbol, and register the limiter scheme so the solver knows to
        # apply it after the LSQ gradient computation.
        for (target_name, mi, scheme), atoms in limit_atoms.items():
            suffix = "_".join(
                str(self.space[d])
                for d in range(n_dim) for _ in range(mi[d])
            )
            aux_name = f"{target_name}_{suffix}__{scheme}"
            sym = sp.Symbol(aux_name, real=True)
            for a in atoms:
                sub_dict[a] = sym
            row = n_aux_before + len(new_syms)
            new_syms.append(sym)
            entry = {
                "kind": "limited_derivative",
                "name": aux_name,
                "row": row,
                "atom": atoms[0],
                "aux_symbol": sym,
                "target_name": target_name,
                "multi_index": tuple(mi),
                "limiter_scheme": scheme,
            }
            if target_name in state_names:
                entry["target_kind"] = "state"
                entry["state_index"] = next(
                    i for i, s in enumerate(self.state)
                    if str(s) == target_name
                )
            elif target_name in function_row_of_name:
                entry["target_kind"] = "function"
                entry["function_row"] = function_row_of_name[target_name]
            else:
                entry["target_kind"] = "unknown"
            registry.append(entry)

        # ── Derivative entries. ───────────────────────────────────
        for (target_name, mi), atoms in deriv_atoms.items():
            suffix = "_".join(
                str(self.space[d])
                for d in range(n_dim) for _ in range(mi[d])
            )
            aux_name = f"{target_name}_{suffix}"
            sym = sp.Symbol(aux_name, real=True)
            for d in atoms:
                sub_dict[d] = sym
            row = n_aux_before + len(new_syms)
            new_syms.append(sym)
            entry = {
                "kind": "derivative",
                "name": aux_name,
                "row": row,
                "atom": atoms[0],
                "aux_symbol": sym,
                "target_name": target_name,
                "multi_index": tuple(mi),
            }
            if target_name in state_names:
                entry["target_kind"] = "state"
                entry["state_index"] = next(
                    i for i, s in enumerate(self.state)
                    if str(s) == target_name
                )
            elif target_name in function_row_of_name:
                entry["target_kind"] = "function"
                entry["function_row"] = function_row_of_name[target_name]
            else:
                entry["target_kind"] = "unknown"
            registry.append(entry)

        # ── Apply substitutions to every primary matrix. ──────────
        # All operator fields are ZArray (or None); ZArray.xreplace
        # is element-wise and handles non-sympy entries (e.g. Python
        # int 0 from a zero-initialised array) gracefully.
        self.flux = self.flux.xreplace(sub_dict)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(
            sub_dict)
        self.source = self.source.xreplace(sub_dict)
        self.mass_matrix = self.mass_matrix.xreplace(sub_dict)
        self.nonconservative_matrix = self.nonconservative_matrix.xreplace(
            sub_dict)
        if self.state_update is not None:
            self.state_update = self.state_update.xreplace(sub_dict)
        if self.diffusion_matrix is not None:
            self.diffusion_matrix = self.diffusion_matrix.xreplace(sub_dict)
        if self.diffusion_matrix_explicit is not None:
            self.diffusion_matrix_explicit = (
                self.diffusion_matrix_explicit.xreplace(sub_dict))

        # Keep the derived operators consistent: recompute
        # quasilinear_matrix / source_jacobian from the substituted
        # primaries, and push the substitution through eigenvalues
        # (cheap — avoids re-paying the spectral derivation).
        self.refresh_derived_operators(eigenvalues=False)
        if self.eigenvalues is not None:
            self.eigenvalues = self.eigenvalues.xreplace(sub_dict)

        self.aux_state = list(self.aux_state) + new_syms
        self.aux_registry = registry
        self.history.append({
            "name": "expose_aux_atoms",
            "description": (
                f"auto-scan: {sum(1 for r in registry if r['kind']=='function')} "
                f"function-aux, "
                f"{sum(1 for r in registry if r['kind']=='derivative')} "
                f"derivative-aux"
            ),
        })

    # ── reverse-aux substitution helper (for tests / inspection) ──

    def _aux_reverse_map(self):
        """Return ``{aux_Symbol: original_atom}`` for every entry in
        ``aux_registry`` so reconstructed residuals can be displayed in
        their original ``Derivative(…)`` form."""
        registry = getattr(self, "aux_registry", None) or []
        return {entry["aux_symbol"]: entry["atom"] for entry in registry}

    # ── (legacy) expose_functions_as_aux / expose_derivatives_as_aux ─

    def expose_functions_as_aux(self, function_names=None):
        """Replace bare :class:`~sympy.Function` atoms with auxiliary
        Symbols of the same name added to :attr:`aux_state`.

        Companion to :meth:`expose_derivatives_as_aux`.  For each
        unique Function atom whose name is in ``function_names``
        (or every free Function if ``function_names is None``), a
        fresh aux Symbol named ``{function_name}`` is substituted in
        every operator matrix.

        Use for free Functions like topography ``b(t, x, y)`` so they
        live as proper aux state (visible in VTK output, fed from a
        per-cell callable at runtime) instead of being symbolically
        deleted via ``_zero_functions_by_name``.

        The mapping ``aux_function_map: Dict[Symbol, Function]`` is
        attached to ``self`` so solvers know which aux entries are
        free-function values (vs state derivatives, vs permanent
        Zstruct aux).
        """
        import sympy as sp
        from itertools import product as _product

        names = (set(function_names)
                 if function_names is not None else None)

        def _matches(a):
            if not isinstance(a, sp.Function):
                return False
            n = a.func.__name__
            return names is None or n in names

        atoms = set()
        for M in (self.flux, self.hydrostatic_pressure,
                  self.nonconservative_matrix, self.source,
                  self.mass_matrix):
            if isinstance(M, sp.Matrix):
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        for a in M[i, j].atoms(sp.Function):
                            if _matches(a):
                                atoms.add(a)
            else:
                for idx in _product(*[range(s) for s in M.shape]):
                    for a in M[idx].atoms(sp.Function):
                        if _matches(a):
                            atoms.add(a)

        if not atoms:
            return

        # One aux Symbol per distinct function name (atoms with
        # different dummy args still map to the same conceptual field).
        sub_dict = {}
        name_to_sym = {}
        aux_function_map = getattr(self, "aux_function_map", {})
        for a in atoms:
            name = a.func.__name__
            if name not in name_to_sym:
                name_to_sym[name] = sp.Symbol(name, real=True)
            sub_dict[a] = name_to_sym[name]
            aux_function_map[name_to_sym[name]] = a

        # Apply substitution.
        self.flux = self.flux.xreplace(sub_dict)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(
            sub_dict)
        self.source = self.source.xreplace(sub_dict)
        self.mass_matrix = self.mass_matrix.xreplace(sub_dict)
        B = self.nonconservative_matrix
        new_B = sp.MutableDenseNDimArray.zeros(*B.shape)
        for idx in _product(*[range(s) for s in B.shape]):
            new_B[idx] = B[idx].xreplace(sub_dict)
        self.nonconservative_matrix = new_B

        self.refresh_derived_operators(eigenvalues=False)
        if self.eigenvalues is not None:
            self.eigenvalues = self.eigenvalues.xreplace(sub_dict)

        self.aux_state = list(self.aux_state) + list(name_to_sym.values())
        self.aux_function_map = aux_function_map
        self.history.append({
            "name": "expose_functions_as_aux",
            "description": (
                f"intercepted {len(atoms)} Function atom(s) → aux "
                f"Symbols {sorted(name_to_sym.keys())}"
            ),
        })


    # ── expose_derivatives_as_aux ─────────────────────────────────────

    def expose_derivatives_as_aux(self):
        """Replace ``Derivative(target, axes...)`` atoms in every
        operator matrix with auxiliary Symbols added to
        ``self.aux_state``.

        Each unique Derivative atom found in ``flux``,
        ``hydrostatic_pressure``, ``nonconservative_matrix``,
        ``source``, or ``mass_matrix`` is replaced with a fresh aux
        Symbol named ``{target}_{axes}``  (e.g. ``h_x``, ``U_0_x_x``,
        ``b_x``).  The Symbols are appended to ``self.aux_state`` and
        recorded in ``self.aux_derivative_map: Dict[Symbol,
        Derivative]`` so a solver can compute their per-cell values
        (state derivatives via mesh + LSQ reconstruction; parameter
        Function derivatives via analytical / tabulated evaluation).

        This makes spatial derivatives — at any order — first-class
        runtime inputs without changing the
        ``(Q, Qaux, p) → ndarray`` operator-API surface.

        Mutates ``self`` in place; no-op if the matrices contain no
        Derivative atoms.
        """
        from itertools import product

        def _matrix_entries(M):
            if isinstance(M, sp.Matrix):
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        yield (i, j), M[i, j]
            else:
                shape = M.shape
                for idx in product(*[range(s) for s in shape]):
                    yield idx, M[idx]

        derivative_subs = {}
        aux_derivative_map = {}

        def _aux_name(d):
            target = d.args[0]
            target_name = (target.func.__name__
                           if isinstance(target, sp.Function)
                           else str(target))
            axes = []
            for var, n in d.variable_count:
                axes.extend([str(var)] * int(n))
            return f"{target_name}_" + "_".join(axes)

        for M in (self.flux, self.hydrostatic_pressure,
                  self.nonconservative_matrix, self.source,
                  self.mass_matrix):
            for _, entry in _matrix_entries(M):
                for d in sp.sympify(entry).atoms(sp.Derivative):
                    if d in derivative_subs:
                        continue
                    aux_sym = sp.Symbol(_aux_name(d), real=True)
                    derivative_subs[d] = aux_sym
                    aux_derivative_map[aux_sym] = d

        if not derivative_subs:
            return

        # Apply substitution to every matrix.  sp.Matrix has xreplace;
        # NDimArray (the NCP) does not, so iterate entries.
        self.flux = self.flux.xreplace(derivative_subs)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(
            derivative_subs)
        self.source = self.source.xreplace(derivative_subs)
        self.mass_matrix = self.mass_matrix.xreplace(derivative_subs)
        B = self.nonconservative_matrix
        shape = B.shape
        new_B = sp.MutableDenseNDimArray.zeros(*shape)
        for idx in product(*[range(s) for s in shape]):
            new_B[idx] = B[idx].xreplace(derivative_subs)
        self.nonconservative_matrix = new_B

        self.refresh_derived_operators(eigenvalues=False)
        if self.eigenvalues is not None:
            self.eigenvalues = self.eigenvalues.xreplace(derivative_subs)

        # Extend aux_state with the new derivative Symbols.
        self.aux_state = list(self.aux_state) + list(
            derivative_subs.values())
        self.aux_derivative_map = aux_derivative_map

        self.history.append({
            "name": "expose_derivatives_as_aux",
            "description": (
                f"intercepted {len(derivative_subs)} Derivative atom(s) "
                f"→ aux Symbols "
                f"{[str(s) for s in derivative_subs.values()]}"
            ),
        })


    # ── change_state_variables ────────────────────────────────────────

    def change_state_variables(self, new_state, transform):
        """Apply a state-variable change of variables in place.

        Parameters
        ----------
        new_state : list
            The new state vector (same length as ``self.state``).
            Entries that already appear in the old state stay; entries
            that replace an old state must appear in ``transform``.
        transform : dict
            Maps each OLD state entry that is being replaced to its
            expression in the NEW state variables.  Old states not
            mentioned are assumed to map identically to themselves.

        Updates ``flux``, ``hydrostatic_pressure``,
        ``nonconservative_matrix``, ``source``, ``mass_matrix``, and
        ``state`` in place.

        Mechanics.  Let ``T_i(Q_new) = transform[Q_old[i]]`` (identity
        if not in ``transform``) and ``J[i, j] = ∂T_i/∂Q_new[j]``.  In
        operator form the system is invariant under the substitution
        ``Q_old → T(Q_new)`` together with the time-derivative
        identity ``∂_t Q_old = J · ∂_t Q_new`` and the spatial-
        derivative identity ``∂_d Q_old = J · ∂_d Q_new``:

          * ``F_new = F_old(T(Q_new))``,
            ``P_new = P_old(T(Q_new))``,
            ``S_new = S_old(T(Q_new))``.
          * ``B_new[i, k, d] = Σ_j B_old[i, j, d](T) · J[j, k]``.
          * ``M_new = M_old(T(Q_new)) · J``.

        This contract preserves the residual ``M ∂_t Q + ∂_x F + ∂_x P
        + B · ∂_x Q − S`` under the change of variables.
        """
        n = self.n_state
        if len(new_state) != n:
            raise ValueError(
                f"new_state has length {len(new_state)}, expected {n}"
            )

        old_state = list(self.state)
        full_transform = {}
        for old_s in old_state:
            if old_s in transform:
                full_transform[old_s] = sp.sympify(transform[old_s])
            else:
                if old_s not in new_state:
                    raise ValueError(
                        f"old state {old_s!r} is not in new_state and "
                        f"has no entry in transform — cannot map it"
                    )
                full_transform[old_s] = old_s

        # Jacobian J[i, j] = ∂T_i / ∂new_state[j]
        J = sp.zeros(n, n)
        for i, old_s in enumerate(old_state):
            T_i = full_transform[old_s]
            for j, new_s in enumerate(new_state):
                J[i, j] = sp.diff(T_i, new_s)

        # Substitute T into operators.  ZArray.xreplace returns ZArray;
        # ZArray.expand applies sp.expand element-wise.
        new_flux = self.flux.xreplace(full_transform).expand()
        new_P    = self.hydrostatic_pressure.xreplace(full_transform).expand()
        new_S    = self.source.xreplace(full_transform).expand()

        # NCP: B_new[i, k, d] = Σ_j B_old[i, j, d](T) · J[j, k].  Per-d
        # slab is a Matrix-multiply; convert each slab through sp.Matrix
        # for the multiply and wrap back at the end.
        n_eq = self.n_equations
        n_dim = self.n_dim
        new_B = sp.MutableDenseNDimArray.zeros(n_eq, n, n_dim)
        for d in range(n_dim):
            B_slab = sp.Matrix(
                n_eq, n,
                lambda i, j, _d=d: sp.sympify(
                    self.nonconservative_matrix[i, j, _d]
                ),
            )
            B_slab_sub = B_slab.xreplace(full_transform)
            B_slab_new = sp.expand(B_slab_sub * J)
            for i in range(n_eq):
                for k in range(n):
                    new_B[i, k, d] = B_slab_new[i, k]
        new_B = _to_zarray(new_B)

        # Mass matrix: M_new = M_old(T) · J.  Same Matrix-multiply
        # pattern — ZArray.xreplace → sp.Matrix for multiply → ZArray.
        M_old_M = sp.Matrix(self.mass_matrix.tolist())
        M_old_sub = M_old_M.xreplace(full_transform)
        new_M = _to_zarray(sp.expand(M_old_sub * J))

        self.state = list(new_state)
        self.flux = new_flux
        self.hydrostatic_pressure = new_P
        self.nonconservative_matrix = new_B
        self.source = new_S
        self.mass_matrix = new_M
        # Push the transform through the secondary fields too so they
        # reference the new state, not the old: ``update_variables``
        # (per-cell state remap), ``state_update`` (Chorin corrector's
        # explicit update), and ``eigenvalues`` (preserved under
        # invertible change-of-vars — xreplace, don't re-solve).
        if self.update_variables is not None:
            # ``update_variables`` is a per-cell state-to-state map.
            # Naively xreplacing the OLD state symbols turns
            # ``identity_in_old(Q_old) = Q_old`` into the inverse
            # transform on the NEW state — e.g. ``[h, U_0, U_1, ...]``
            # becomes ``[h, q_U0/h, 3·q_U1/h, ...]`` which actively
            # destroys the conservative-form state at every step.
            # If the old map was the identity (default for Models with
            # no custom clamp), rebuild the identity in the new state.
            old_uv_list = [sp.sympify(self.update_variables[i, 0])
                           for i in range(self.update_variables.shape[0])]
            was_identity = (
                len(old_uv_list) == len(old_state)
                and all(sp.simplify(old_uv_list[i] - old_state[i]) == 0
                        for i in range(len(old_state)))
            )
            if was_identity:
                n_uv = len(new_state)
                self.update_variables = _to_zarray(sp.Matrix(
                    n_uv, 1, lambda r, _c: new_state[r]
                ))
            else:
                self.update_variables = self.update_variables.xreplace(
                    full_transform
                )
        if self.state_update is not None:
            self.state_update = self.state_update.xreplace(full_transform)
        # ``reconstruction_variables`` / ``state_from_reconstruction``:
        # the forward map xreplaces straight through under CoV (it's an
        # expression in state symbols).  The inverse is re-derived from
        # the post-CoV forward + new state — its WB-symbol expressions
        # depend on the *new* state slot names, so a plain xreplace
        # wouldn't suffice.
        if self.reconstruction_variables is not None:
            from zoomy_core.model.reconstruction_inverse import (
                invert_reconstruction,
            )
            self.reconstruction_variables = (
                self.reconstruction_variables.xreplace(full_transform))
            self.state_from_reconstruction = invert_reconstruction(
                self.reconstruction_variables, list(new_state),
            )
        # Chain-rule propagation for aux derivative entries that target
        # OLD state variables.  Their meaning (∂_x of OLD state) must be
        # rewritten in terms of NEW-state derivatives, otherwise the
        # operator expressions disagree with the runtime aux pipeline
        # (which computes ``∂_x state[i]`` using the NEW state).
        self._propagate_chain_rule_through_aux(
            full_transform, J, old_state, new_state,
        )
        # Re-derive the quasilinear matrix + source jacobian from the new
        # primaries.
        self.refresh_derived_operators(eigenvalues=False)
        if self.eigenvalues is not None:
            self.eigenvalues = self.eigenvalues.xreplace(full_transform)
        self.history.append({
            "name": "change_state_variables",
            "description": (
                "state: ["
                + ", ".join(str(s) for s in old_state)
                + "] → ["
                + ", ".join(str(s) for s in new_state)
                + "]; transform: "
                + ", ".join(f"{k}↦{v}" for k, v in transform.items())
            ),
        })
        return self

    def _propagate_chain_rule_through_aux(self, full_transform, J,
                                          old_state, new_state):
        """After ``change_state_variables`` substitutes state Symbols
        in the operator matrices, the aux-registry entries that
        represent ``∂_x state[i]`` (where ``state[i]`` is an OLD
        variable being substituted) carry stale meaning: the symbol
        in the operator was ``∂_x U_0`` but the runtime computes
        ``∂_x state[i] = ∂_x q_U0`` after the substitution.  Chain-rule
        propagation fixes this:

        ``∂_x T_i(Q_new) = Σ_j (∂T_i/∂Q_new[j]) · ∂_x Q_new[j]``

        For every aux entry whose target was an OLD state variable
        that got substituted, substitute the aux Symbol in the
        operators with the chain-rule expansion, and add new aux
        entries for ``∂_x Q_new[j]`` if not already present.
        """
        if not getattr(self, "aux_registry", None):
            return

        from itertools import product

        old_state_names = {str(s): s for s in old_state}
        new_state_names = {str(s): (j, s) for j, s in enumerate(new_state)}
        space = list(self.space)
        if not space:
            return
        axis = space[0]   # only handle 1D for now (matches multi_index = (1,))

        aux_subs = {}
        new_aux_entries = []
        entries_to_remove = []

        for entry in self.aux_registry:
            if entry.get("kind") != "derivative":
                continue
            if entry.get("target_kind") != "state":
                continue
            if entry.get("multi_index") != (1,):
                continue
            target_name = entry["target_name"]
            if target_name not in old_state_names:
                continue
            old_sym = old_state_names[target_name]
            T = full_transform.get(old_sym, old_sym)
            if T == old_sym:
                # Identity transform — state retained; just sync state_index
                # in case the order changed.
                if target_name in new_state_names:
                    entry["state_index"] = new_state_names[target_name][0]
                continue
            # State substituted; build chain-rule expansion.
            old_aux_sym = entry["aux_symbol"]
            chain_expr = sp.S.Zero
            for j, new_sym in enumerate(new_state):
                partial = sp.diff(T, new_sym)
                if partial == 0:
                    continue
                new_aux_name = f"{str(new_sym)}_x"
                new_aux_sym = sp.Symbol(new_aux_name, real=True)
                # Only add if not already in registry or pending.
                already = (any(e.get("name") == new_aux_name
                               for e in self.aux_registry)
                           or any(e["name"] == new_aux_name
                                  for e in new_aux_entries))
                if not already:
                    new_aux_entries.append({
                        "kind": "derivative",
                        "name": new_aux_name,
                        "row": -1,
                        "atom": sp.Derivative(new_sym, axis, evaluate=False),
                        "aux_symbol": new_aux_sym,
                        "target_name": str(new_sym),
                        "multi_index": (1,),
                        "target_kind": "state",
                        "state_index": j,
                    })
                chain_expr += partial * new_aux_sym
            aux_subs[old_aux_sym] = sp.expand(chain_expr)
            entries_to_remove.append(entry["name"])

        if not aux_subs:
            return

        # Apply substitution to all operator matrices.
        self.flux = self.flux.xreplace(aux_subs)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(aux_subs)
        self.source = self.source.xreplace(aux_subs)
        self.mass_matrix = self.mass_matrix.xreplace(aux_subs)
        B = self.nonconservative_matrix
        shape = B.shape
        new_B = sp.MutableDenseNDimArray.zeros(*shape)
        for idx in product(*[range(s) for s in shape]):
            new_B[idx] = sp.sympify(B[idx]).xreplace(aux_subs)
        self.nonconservative_matrix = _to_zarray(new_B)

        # Update aux_state list + aux_registry: drop the substituted
        # entries, append the new ones, and reindex `row` so they form
        # a contiguous block.
        keep = [e for e in self.aux_registry
                if e["name"] not in entries_to_remove]
        new_aux_state = []
        new_registry = []
        for i, entry in enumerate(keep + new_aux_entries):
            entry = dict(entry)
            entry["row"] = i
            new_registry.append(entry)
            new_aux_state.append(entry["aux_symbol"])
        self.aux_state = new_aux_state
        self.aux_registry = new_registry
        # Stash chain-rule info; ``change_state_variables`` folds it
        # into its single history entry so callers see one event per
        # CoV (matches what tests expect).
        self._last_cov_aux_chain_rule = {
            "removed": list(entries_to_remove),
            "added": [e["name"] for e in new_aux_entries],
        }

    def remove_non_diagonal_h(self, **kwargs):
        """Convenience wrapper around :class:`RemoveNonDiagonalH`."""
        return RemoveNonDiagonalH()(self, **kwargs)

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
        if sm.parameters.length():
            parts.append(
                "**Parameters:** "
                + ", ".join(
                    f"${sp.latex(sm.parameters[k])} = "
                    f"{getattr(sm.parameter_values, k, '?')}$"
                    for k in sm.parameters.keys()
                )
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

class RemoveNonDiagonalH:
    """System-level op: substitute the mass equation into every row
    whose mass-matrix ``h``-column entry is non-zero, pushing the
    ``M[i, h] · ∂_t h`` term into the NCP matrix ``B`` and the source
    ``S``.  Leaves ``M[i, k]`` untouched for ``k != h``.

    Motivation.  The Galerkin chain for free-surface models produces
    j ≥ 1 momentum rows that carry a state-dependent ``∂_t h``
    cross-term — e.g. ``M[xmom_j1, h] = (q_U1 − q_U0)/h`` in the
    conservative state of VAM(1, 2, 2).  The cross-term reflects the
    test function's time-dependence through ``ζ = (z − b)/h`` and is
    genuine, not a bug.  Substituting the mass equation
    ``∂_t h = −∂_x F[m, :] − ∂_x P[m, :] − Σ_k B[m, k, :] ∂_x Q_k +
    S[m]`` into each affected row pushes the term out of the
    time-derivative slot, where state-dependent ``M`` would otherwise
    silently mislead explicit solvers that assume ``M = I``.

    Scope.  This op clears only the ``h`` column of ``M``.  Other
    diagonal / off-diagonal entries are left intact — normalising
    state-dependent diagonal entries (e.g. ``M[xmom_j1, U_1] = h/3`` in
    primitive form) is the job of an extended
    :class:`InvertMassMatrix`, not this pass.

    Preconditions.  The mass row ``m`` (auto-detected, or supplied via
    ``mass_equation_index``) must satisfy ``M[m, h] = 1`` and
    ``M[m, k] = 0`` for ``k != h``.  Other rows may have arbitrary
    ``M[i, h]`` content.

    Post-conditions.  ``M[i, h_state_index] = 0`` for every previously
    affected row ``i``.  Per affected row, ``B`` and ``S`` are updated
    so the residual is exact modulo the mass equation:

    ``B[i, k, d] ← B[i, k, d] − M_old[i, h] · (∂F[m, d]/∂Q_k
                                                + ∂P[m, d]/∂Q_k
                                                + B[m, k, d])``

    ``S[i] ← S[i] − M_old[i, h] · S[m]``

    The mass row ``m`` itself is never modified.
    """

    name = "remove_non_diagonal_h"
    description = (
        "Substitute mass equation into rows with non-zero ∂_t h "
        "coupling; clear M[i, h] for those rows."
    )

    def __call__(self, sm: "SystemModel", *, h_state_index=None,
                 mass_equation_index=None):
        n_eq = sm.n_equations
        n_st = sm.n_state
        n_dim = sm.n_dim

        if h_state_index is None:
            h_state_index = self._find_h_state_index(sm)
        if mass_equation_index is None:
            mass_equation_index = self._find_mass_row(sm, h_state_index)

        h_col = h_state_index
        m = mass_equation_index

        # Precondition: M[m, :] is e_{h_col}.
        M_mh = sp.simplify(sm.mass_matrix[m, h_col])
        if M_mh != 1:
            raise ValueError(
                f"RemoveNonDiagonalH: mass row {m} has M[{m}, {h_col}] = "
                f"{M_mh}, expected 1.  Supply mass_equation_index "
                f"explicitly or normalise the mass row first."
            )
        for k in range(n_st):
            if k == h_col:
                continue
            entry = sp.simplify(sm.mass_matrix[m, k])
            if entry != 0:
                raise ValueError(
                    f"RemoveNonDiagonalH: mass row {m} is not pure: "
                    f"M[{m}, {k}] = {entry}, expected 0."
                )

        M = sp.Matrix(sm.mass_matrix.tolist())
        S = sp.Matrix(sm.source.tolist())
        B = sp.MutableDenseNDimArray(
            sm.nonconservative_matrix.tolist(),
            shape=tuple(sm.nonconservative_matrix.shape),
        )

        # Pre-compute mass-row spatial pieces.
        mass_F = [sm.flux[m, d] for d in range(n_dim)]
        mass_P = [sm.hydrostatic_pressure[m, d] for d in range(n_dim)]
        # ∂F[m, d]/∂Q_k and ∂P[m, d]/∂Q_k for every (k, d).
        dFmd_dk = [
            [sp.diff(mass_F[d], sm.state[k]) for d in range(n_dim)]
            for k in range(n_st)
        ]
        dPmd_dk = [
            [sp.diff(mass_P[d], sm.state[k]) for d in range(n_dim)]
            for k in range(n_st)
        ]
        mass_B = [
            [sm.nonconservative_matrix[m, k, d] for d in range(n_dim)]
            for k in range(n_st)
        ]
        mass_S = sm.source[m, 0]

        affected = []
        for i in range(n_eq):
            if i == m:
                continue
            coeff = sp.simplify(M[i, h_col])
            if coeff == 0:
                continue
            affected.append(i)
            for k in range(n_st):
                for d in range(n_dim):
                    B[i, k, d] = sp.expand(
                        B[i, k, d]
                        - coeff * (dFmd_dk[k][d]
                                   + dPmd_dk[k][d]
                                   + mass_B[k][d])
                    )
            S[i, 0] = sp.expand(S[i, 0] - coeff * mass_S)
            M[i, h_col] = sp.S.Zero

        sm.mass_matrix = _to_zarray(M)
        sm.nonconservative_matrix = _to_zarray(B)
        sm.source = _to_zarray(S)
        sm.refresh_derived_operators(eigenvalues=False)
        sm.history.append({
            "name": self.name,
            "description": (
                f"substituted mass equation (row {m}) into rows "
                f"{affected}; cleared M[:, {h_col}] on those rows"
            ),
        })
        return sm

    @staticmethod
    def _find_h_state_index(sm: "SystemModel") -> int:
        for i, s in enumerate(sm.state):
            if getattr(s, "name", None) == "h":
                return i
        raise ValueError(
            "RemoveNonDiagonalH: no state entry named 'h' found; "
            "pass h_state_index explicitly."
        )

    @staticmethod
    def _find_mass_row(sm: "SystemModel", h_col: int) -> int:
        """Find the row m with M[m, h_col] = 1 and M[m, k] = 0 elsewhere."""
        candidates = []
        for i in range(sm.n_equations):
            entry = sp.simplify(sm.mass_matrix[i, h_col])
            if entry != 1:
                continue
            others_zero = all(
                sp.simplify(sm.mass_matrix[i, k]) == 0
                for k in range(sm.n_state) if k != h_col
            )
            if others_zero:
                candidates.append(i)
        if not candidates:
            raise ValueError(
                f"RemoveNonDiagonalH: no row found with M[i, {h_col}] = 1 "
                f"and zero elsewhere; pass mass_equation_index explicitly."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"RemoveNonDiagonalH: multiple candidate mass rows "
                f"{candidates}; pass mass_equation_index explicitly."
            )
        return candidates[0]




class HydrostaticReconstruction:
    """System-level op: repack `g·h·∂_x η` from the chain's
    ``(P, B)`` form to standard SWE ``P = g·h²/2`` form, for use
    with Audusse hydrostatic-reconstruction Riemann solvers.

    **What this does**

    The chain's ``quadratic_form="escalante"`` output represents
    the depth-integrated gravity term ``g·h·∂_x η`` as

    ``P[i, d] = g·h·(b + h)``  and  ``B[i, h, d] = −g·(b + h)``,

    because ``∂_x(g·h·η) − g·η·h_x ≡ g·h·∂_x η`` (product rule).
    This packaging makes the bathymetry ``b`` appear *inside* the
    hydrostatic-pressure flux, which is correct as a continuous
    PDE — but Audusse's WB cancellation works most cleanly when
    ``P`` has the *standard SWE* form ``g·h²/2`` (no ``b``) and
    the bathymetry-on-momentum contribution ``−g·h·b_x`` is left
    to be supplied at runtime by the Audusse fluctuation
    ``(P_raw − P_star) @ n`` (see
    :meth:`PositiveRusanov.numerical_fluctuations`).

    This op rewrites for every row with that packaging:

    * ``P[i, d]``      →  ``P[i, d] − g·h·(b + h) + g·h²/2``
                       =  remainder + ``g·h²/2``
    * ``B[i, h, d]``   →  ``B[i, h, d] + g·(b + h)``
                       =  ``0`` (chain's escalante form had only
                          the gravity term here on the j=0 row)

    **Trade-off**

    The resulting symbolic SystemModel is *missing* the
    ``−g·h·b_x`` term — its residual no longer represents the
    full PDE on its own.  Audusse HR provides this term via the
    flux fluctuation at runtime; a non-HR Riemann solver (plain
    Rusanov) will be inconsistent on a varying bathymetry.  Only
    apply this op when the runtime is configured to use a
    ``PositiveRusanov``-family flux (which the
    :class:`ChorinSplitVAMSolver` does by default).

    **Detection**

    The op auto-detects every row ``i`` whose
    ``P[i, d]`` contains a ``g·h·(b + h)`` summand AND whose
    ``B[i, h_state_index, d]`` contains a matching ``−g·(b + h)``.
    These together signal the chain's gravity-on-η packaging.
    For VAM(1, 2, 2) this fires on the ``xmom_j0`` row only.
    """

    name = "hydrostatic_reconstruction"
    description = (
        "Convert chain's g·h·η in (P, B) to standard SWE P = g·h²/2 "
        "for Audusse HR compatibility (drops -g·h·b_x; HR fluctuation "
        "supplies it at runtime)."
    )

    def __call__(self, sm: "SystemModel", *, h_state_index=None):
        # Locate h state index.
        if h_state_index is None:
            for i, s in enumerate(sm.state):
                if getattr(s, "name", None) == "h":
                    h_state_index = i
                    break
            if h_state_index is None:
                raise ValueError(
                    "PrepareForAudusseHR: no state entry named 'h' found"
                )
        h_sym = sm.state[h_state_index]

        # Locate g (parameter) and b (aux Symbol).
        g_sym = None
        for k in sm.parameters.keys():
            sym = sm.parameters[k]
            if getattr(sym, "name", None) == "g":
                g_sym = sym
                break
        if g_sym is None:
            raise ValueError(
                "PrepareForAudusseHR: no 'g' parameter found"
            )

        # ``b`` may live in either ``aux_state`` (b-as-static-aux) or
        # ``state`` (b-promoted-to-state with ``∂_t b = 0`` row).  HR
        # only needs the Symbol to match against the gravity packaging
        # ``P[i, d]`` and ``B[i, h, d]``; it does not care whether
        # ``b`` evolves.
        b_sym = None
        for s in sm.aux_state:
            if getattr(s, "name", None) == "b":
                b_sym = s
                break
        if b_sym is None:
            for s in sm.state:
                if getattr(s, "name", None) == "b":
                    b_sym = s
                    break
        if b_sym is None:
            raise ValueError(
                "HydrostaticReconstruction: no 'b' symbol found in state "
                "or aux_state"
            )

        n_eq = sm.n_equations
        n_dim = sm.n_dim

        # Pattern we expect on rows with gravity-on-η packaging:
        # P[i, d] contains  +g·h·(b + h) = g·h·b + g·h²
        # B[i, h, d] contains  -g·(b + h) = -g·b - g·h.
        # Repackage = subtract  g·h·b + g·h²/2  from P
        #             add  g·(b + h)  to B[i, h, d].
        # Detection: the *coefficient* of g·h·b in P[i, d] is +1 AND
        # the coefficient of g·b in B[i, h, d] is -1.  Then the
        # row has the gravity-on-η packaging.

        P = sp.Matrix(sm.hydrostatic_pressure.tolist())
        B = sp.MutableDenseNDimArray(
            sm.nonconservative_matrix.tolist(),
            shape=tuple(sm.nonconservative_matrix.shape),
        )

        affected = []
        for i in range(n_eq):
            for d in range(n_dim):
                P_id = sp.expand(P[i, d])
                B_ihd = sp.expand(B[i, h_state_index, d])
                # Coefficient of g·h·b in P[i, d].
                coeff_in_P = P_id.coeff(g_sym * h_sym * b_sym)
                # Coefficient of g·b in B[i, h, d].
                coeff_in_B = B_ihd.coeff(g_sym * b_sym)
                if coeff_in_P == 1 and coeff_in_B == -1:
                    # Repackage: P -= g·h·(b+h) - g·h²/2 = g·h·b + g·h²/2
                    P[i, d] = sp.expand(
                        P_id - g_sym * h_sym * b_sym
                        - g_sym * h_sym**2 / sp.Integer(2)
                    )
                    # B[i, h, d] -= -g·(b+h) = +g·(b+h)
                    B[i, h_state_index, d] = sp.expand(
                        B_ihd + g_sym * (b_sym + h_sym)
                    )
                    affected.append((i, d))

        if not affected:
            sm.history.append({
                "name": self.name,
                "description": "no rows with gravity-on-η packaging found",
            })
            return sm

        sm.hydrostatic_pressure = _to_zarray(P)
        sm.nonconservative_matrix = _to_zarray(B)
        sm.refresh_derived_operators(eigenvalues=False)
        sm.history.append({
            "name": self.name,
            "description": (
                f"repackaged gravity g·h·(b+h) → g·h²/2 on rows/dims "
                f"{affected}; -g·h·b_x source dropped — Audusse HR "
                f"must supply it at runtime"
            ),
        })
        return sm


class InvertMassMatrix:
    """System-level op: left-multiply each evolution row by ``1/M_ii``.

    Operates on a SystemModel whose mass matrix is **diagonal** on
    every evolution row (with arbitrary state-dependent diagonal
    entries) and zero on every algebraic row.  Use
    :meth:`SystemModel.assert_diagonal_mass_matrix` first to verify
    this precondition.

    Per evolution row ``i`` (with ``equation_to_state_index[i] = j``):
    divide ``flux[i, :]``, ``hydrostatic_pressure[i, :]``,
    ``source[i, 0]``, and ``nonconservative_matrix[i, :, :]`` by the
    diagonal entry ``M[i, j]``.  Set ``M[i, j] = 1``.

    Diagonal entries may be state- or aux-dependent — the division is
    symbolic and produces a ``1/M_ii`` factor in the resulting
    operators (which gets lambdified per-cell at runtime).
    """

    name = "invert_mass_matrix"
    description = (
        "Divide each evolution row by its diagonal mass entry to reach "
        "canonical ∂_t Q form."
    )

    def __call__(self, sm: SystemModel):
        # Precondition: mass matrix is diagonal on evolution rows.
        sm.assert_diagonal_mass_matrix()

        n_eq = sm.n_equations
        n_st = sm.n_state
        n_dim = sm.n_dim
        e2s = list(sm.equation_to_state_index)

        # Mutable working copies of the operator tensors.
        M = sp.Matrix(sm.mass_matrix.tolist())
        F = sp.Matrix(sm.flux.tolist())
        P = sp.Matrix(sm.hydrostatic_pressure.tolist())
        S = sp.Matrix(sm.source.tolist())
        B = sp.MutableDenseNDimArray(
            sm.nonconservative_matrix.tolist(),
            shape=tuple(sm.nonconservative_matrix.shape),
        )

        n_flux_cols = F.shape[1]
        n_P_cols = P.shape[1]
        n_S_cols = S.shape[1]

        normalised = []
        for i in range(n_eq):
            diag = sp.simplify(M[i, e2s[i]])
            if diag == 0:
                # Algebraic row — nothing to invert.
                continue
            if diag == 1:
                continue
            inv = 1 / diag
            for k in range(n_flux_cols):
                F[i, k] = sp.simplify(F[i, k] * inv)
            for k in range(n_P_cols):
                P[i, k] = sp.simplify(P[i, k] * inv)
            for k in range(n_S_cols):
                S[i, k] = sp.simplify(S[i, k] * inv)
            for k in range(n_st):
                for d in range(n_dim):
                    B[i, k, d] = sp.simplify(B[i, k, d] * inv)
            M[i, e2s[i]] = sp.S.One
            normalised.append(i)

        sm.mass_matrix = _to_zarray(M)
        sm.flux = _to_zarray(F)
        sm.hydrostatic_pressure = _to_zarray(P)
        sm.source = _to_zarray(S)
        sm.nonconservative_matrix = _to_zarray(B)
        # Derived operators depend on the primaries; refresh.
        sm.refresh_derived_operators(eigenvalues=False)
        sm.history.append({
            "name": self.name,
            "description": (
                f"divided evolution rows {normalised} by their diagonal "
                f"mass entries"
            ),
        })


__all__ = [
    "SystemModel",
    "SystemModelRow",
    "SystemModelDescription",
    "InvertMassMatrix",
    "RemoveNonDiagonalH",
    "HydrostaticReconstruction",
]
