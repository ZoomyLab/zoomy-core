"""Modal separation-of-variables for the clean-redesign framework.

The pieces here turn a single field ``ũ(t, x, ζ)`` into an UNEXPANDED modal
sum ``Σ_i a(i, t, x)·φ(i, ζ)`` and keep the bookkeeping that lets a later
``ResolveBasis`` recover the trial index and basis per field:

:class:`ModalIndexRegistry`
    Hands out a **distinct** real summation index per coefficient family
    (``a → i``, ``aw → j``, ``ap → k`` …) drawn from ``_INDEX_NAMES``, and
    records ``coeff_name → (index, basis)`` plus a reverse
    ``field_head → coeff_name`` map.  Distinct indices are essential: when two
    expansions multiply (``u·w``) their dummies must differ, else ``sympy``
    keeps the two ``Sum``s sharing one index and silently drops the cross
    terms.  Lives on every :class:`~zoomy_core.model.derivation.model.Model` as
    ``model._modal_registry``.

:func:`separation_of_variables`
    A :class:`~zoomy_core.model.derivation.model.Model` op (``whole_model_op``).  It
    looks up the field's DECORATED head (post-PDE, via
    ``model._field_decoration``), builds the unexpanded ``sp.Sum`` with a fresh
    distinct registry index, substitutes the decorated head → that Sum across
    every equation, registers ``coeff → (index, basis)`` and
    ``field_head → coeff``, and calls ``model.redeclare_unknown(field →
    coeff-family)`` so ``Q`` tracks the modal coefficients.

:func:`modal_bound`
    A fresh abstract truncation-bound Symbol ``N`` (integer, non-negative).
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation


__all__ = [
    "ModalIndexRegistry",
    "reset_modal_indices",
    "separation_of_variables",
    "SeparationOfVariables",
    "TensorSeparationOfVariables",
    "build_modal_sum",
    "modal_bound",
    "modal_index",
    "test_index",
]


# Distinct mode letters, in order.  ``l`` is skipped — it is the Galerkin
# test index.  Indices are memoised by coefficient name, so re-running a
# notebook cell (or expanding the same field twice) reuses the same index
# instead of running off the alphabet.
_INDEX_NAMES = ["i", "j", "k", "m", "n", "o", "p", "q", "r", "s"]


def modal_index(name="i"):
    """A single integer, non-negative summation-index Symbol (a *trial*
    summation dummy ``i`` / ``j`` in ``Σ_i a_i φ_i``)."""
    return sp.Symbol(name, integer=True, nonnegative=True)


def test_index(name="l"):
    """A Galerkin **test / moment** index Symbol (integer, non-negative).

    Distinct in intent from :func:`modal_index` (a trial summation dummy) and
    :func:`modal_bound` (a truncation bound), though the Symbol shape is
    identical.  Use it for the projection test weight ``c(ζ)·φ(l, ζ)`` so the
    derivation reads ``l = test_index()`` instead of a raw ``sp.Symbol``.
    Default ``"l"`` matches the codebase's test-index name (``_INDEX_NAMES``
    reserves ``l`` for exactly this)."""
    return sp.Symbol(name, integer=True, nonnegative=True)


def modal_bound(name="N_u"):
    """A fresh abstract truncation-bound Symbol (integer, non-negative)."""
    return sp.Symbol(name, integer=True, nonnegative=True)


def _coeff_key(coeff):
    """Resolve any coefficient handle (name / head class / applied) to its
    family name string (``a(i, t, x) → "a"``)."""
    if isinstance(coeff, str):
        return coeff
    if isinstance(coeff, type):
        return coeff.__name__
    func = getattr(coeff, "func", None)
    if func is not None:
        return func.__name__
    return str(coeff)


# ── ModalIndexRegistry ────────────────────────────────────────────────────


class ModalIndexRegistry:
    """Per-model registry of ``coeff_name → (index, basis)`` with a reverse
    ``field_head → coeff_name`` map.

    ``separation_of_variables`` calls :meth:`assign` with the field head, the
    coefficient name, and the (opaque) basis; ``ResolveBasis(u, …)`` later
    looks the index + basis up by FIELD or by COEFFICIENT via
    :meth:`index_for` / :meth:`basis_for`.
    """

    def __init__(self):
        self._entries = {}          # coeff_name → {"index": Symbol, "basis": …}
        self._field_to_coeff = {}   # field head class → coeff_name
        self._order = []            # mode letters handed out, in order
        self._registered = {}       # user-registered test/moment index → Symbol

    def _mint_index(self):
        nm = next((n for n in _INDEX_NAMES if n not in self._order), None)
        if nm is None:
            nm = f"i_{len(self._order)}"
        self._order.append(nm)
        return sp.Symbol(nm, integer=True, nonnegative=True)

    def assign(self, coeff_name, *, basis=None, field_head=None, index=None):
        """Register or update an entry; return the (possibly memoised) index."""
        if coeff_name in self._entries:
            if basis is not None:
                self._entries[coeff_name]["basis"] = basis
        else:
            self._entries[coeff_name] = {
                "index": index if index is not None else self._mint_index(),
                "basis": basis,
            }
        if field_head is not None:
            self._field_to_coeff[field_head] = coeff_name
        return self._entries[coeff_name]["index"]

    def fresh(self, key=None):
        """Hand out (or recall) an index keyed by coefficient name."""
        if key is None:
            return self._mint_index()
        return self.assign(key)

    def coeff_name_for(self, target):
        """Resolve ``target`` (name / coeff head / applied coeff / field head /
        applied field) to its coefficient name in this registry, or ``None``."""
        if isinstance(target, str):
            return target if target in self._entries else None
        if isinstance(target, type):                    # head class
            if target in self._field_to_coeff:
                return self._field_to_coeff[target]
            nm = target.__name__
            return nm if nm in self._entries else None
        if getattr(target, "args", None) is not None:   # applied instance
            head = target.func
            if head in self._field_to_coeff:
                return self._field_to_coeff[head]
            nm = head.__name__
            return nm if nm in self._entries else None
        return None

    def index_for(self, target):
        name = self.coeff_name_for(target)
        return self._entries[name]["index"] if name else None

    def basis_for(self, target):
        name = self.coeff_name_for(target)
        return self._entries[name]["basis"] if name else None

    def register(self, name):
        """Mint and register a user test/moment index Symbol, raising if the
        name is already a registered index OR an auto-minted trial index — so
        the user cannot accidentally create two colliding indices."""
        if name in self._registered:
            return self._registered[name]
        if name in self._order:
            raise ValueError(
                f"index {name!r} is already an auto-assigned modal (trial) "
                f"index on this model; pick another test-index name.")
        sym = sp.Symbol(name, integer=True, nonnegative=True)
        self._registered[name] = sym
        return sym

    def reset(self):
        self._entries.clear()
        self._field_to_coeff.clear()
        self._order.clear()
        self._registered.clear()


def reset_modal_indices(model):
    """Clear ``model``'s modal-index registry so index assignment is
    deterministic (call at the top of a derivation / notebook cell)."""
    model._modal_registry.reset()


# ── build the unexpanded modal sum ─────────────────────────────────────────


def build_modal_sum(field, coeff, basis, order, index):
    """Build the unexpanded modal sum ``Σ_i coeff(i, *coords)·φ(i, ζ)``.

    Parameters
    ----------
    field : sympy applied Function
        The field application being approximated, e.g. ``ũ(t, x, ζ)``.  Its
        LAST argument is the separation coordinate ``ζ`` (read off here).
    coeff : sympy Function head | applied
        The modal coefficient.  Applied (``a(t, x)``) supplies its own
        horizontal coords; a bare head defaults to ``field``'s leading args.
    basis : Basis | sympy Function head
        Supplies the opaque ``φ`` head (``basis.phi`` / ``basis.phi_fn``) or
        is itself a bare ``φ`` head.
    order : sympy.Expr | int
        The (abstract or concrete) truncation bound ``N``.
    index : sympy.Symbol
        The distinct summation index.
    """
    zeta = field.args[-1]
    if isinstance(coeff, type):
        coeff_head, coords = coeff, tuple(field.args[:-1])
    elif getattr(coeff, "args", None):
        coeff_head, coords = coeff.func, tuple(coeff.args)
    else:
        coeff_head, coords = getattr(coeff, "func", coeff), tuple(field.args[:-1])
    phi = getattr(basis, "phi", None) or getattr(basis, "phi_fn", basis)
    return sp.Sum(coeff_head(index, *coords) * phi(index, zeta),
                  (index, 0, order))


# ── separation_of_variables (the Model op) ─────────────────────────────────


class SeparationOfVariables(Operation):
    """Replace a field's decorated head with its unexpanded modal sum and
    swap the unknown family in ``Q``.

    Built by :func:`separation_of_variables`.  This is a model-level op
    (``whole_model_op``): it rewrites every equation, registers
    ``coeff → (index, basis)`` + ``field_head → coeff`` on
    ``model._modal_registry``, and redeclares the unknown
    ``field → coeff-family`` so ``Q`` tracks the modal coefficients.
    """

    whole_model_op = True

    def __init__(self, field, coeff, basis, order, *, index=None,
                 name="separation_of_variables"):
        self._field = field
        self._coeff = coeff
        self._basis = basis
        self._order = order
        self._index = index
        super().__init__(
            name=name,
            description=(f"separation of variables {field} → "
                         f"Σ {_coeff_key(coeff)}·φ"),
        )

    def apply_to_model(self, model):
        # The field's DECORATED head (post-PDE).  Read it off the model's
        # ``_field_decoration`` map; if the model never ran a PDE transform
        # the original head is used.
        deco = model._field_decoration or {}
        head = deco.get(self._field.func, self._field.func)

        coeff_name = _coeff_key(self._coeff)
        idx = model._modal_registry.assign(
            coeff_name, basis=self._basis, field_head=head, index=self._index)
        # Register the ORIGINAL (pre-PDE) field head too, so a later
        # ``model.modal_index(u)`` keyed on the physical-z field resolves to
        # the same index as the decorated head.
        if self._field.func is not head:
            model._modal_registry.assign(
                coeff_name, field_head=self._field.func)

        # The coefficient family head + its (horizontal) coords.
        coeff_head = (self._coeff if isinstance(self._coeff, type)
                      else getattr(self._coeff, "func", self._coeff))
        coords = (tuple(self._coeff.args) if getattr(self._coeff, "args", None)
                  else tuple(self._field.args[:-1]))
        phi = (getattr(self._basis, "phi", None)
               or getattr(self._basis, "phi_fn", self._basis))
        order = self._order

        # Replace EVERY application of the (decorated) field head — at ANY
        # argument list — with the modal sum ``Σ_i a_i(coords)·φ(i, <basis arg>)``.
        # Using a HEAD-level ``replace`` (not ``xreplace`` on a single ``ũ(t,x,ζ)``
        # target) is what lets the ansatz be inserted AFTER a field-level
        # ``KinematicBC``: the BC substitutes boundary evaluations ``ũ(t,x,0)`` /
        # ``ũ(t,x,1)`` whose last arg is the concrete boundary value, and those
        # must expand to ``Σ_i a_i·φ(i, 0/1)`` too.  ``call_args[-1]`` is the
        # separation (basis) coordinate of each application — ζ in the bulk, 0/1
        # at a boundary.
        #
        # FRESH DUMMY PER FIELD OCCURRENCE.  ``Basic.replace`` fires the callback
        # once per DISTINCT subexpression, so two structurally different
        # applications of the same head — e.g. ``u(t,x,ζ)`` in the bulk and
        # ``u(t,x,0)`` at a bed BC, or a nested ``u·u`` whose factors differ by
        # argument — used to collapse onto ONE shared summation dummy ``idx``.
        # ``sympy`` then keeps the two ``Sum``s sharing that index and SILENTLY
        # DROPS the cross terms (``i ≠ j``) when they multiply.  We hand the FIRST
        # occurrence the canonical registry index (so single-field expansions and
        # the ``Q`` redeclaration are byte-identical to before) and mint a fresh
        # DISTINCT index for every subsequent occurrence, drawn from the same
        # ``i, j, k, …`` pool.  A ``u·u`` that ``sympy`` has already folded to a
        # ``Pow`` (one base node) still relies on :class:`ExpandSums` to relabel —
        # this fix removes the collision for the DISTINCT-node cross terms that
        # ``ExpandSums`` never sees (they are handed to it pre-multiplied).
        def _make_expander():
            """A fresh per-equation expander: the FIRST distinct occurrence keeps
            the canonical registry index ``idx`` (byte-identical to the old
            single-dummy behaviour when a field occurs once per equation); every
            LATER distinct occurrence gets a fresh index from the ``i, j, k, …``
            pool.  Resetting per equation keeps cross-equation dummy names stable
            (they never interact) while removing the within-equation collision."""
            used = {idx}
            first = {"done": False}

            def _next_index():
                if not first["done"]:
                    first["done"] = True
                    return idx
                for nm in _INDEX_NAMES:
                    cand = sp.Symbol(nm, integer=True, nonnegative=True)
                    if cand not in used:
                        used.add(cand)
                        return cand
                n = 1
                while True:
                    cand = sp.Symbol(f"{idx.name}_{n}", integer=True,
                                     nonnegative=True)
                    if cand not in used:
                        used.add(cand)
                        return cand
                    n += 1

            def _expand(*call_args):
                j = _next_index()
                return sp.Sum(coeff_head(j, *coords) * phi(j, call_args[-1]),
                              (j, 0, order))
            return _expand

        for eq in model._equations.values():
            _expand = _make_expander()
            eq.expr = eq.expr.replace(head, _expand)
            # An ORIENTED relation (e.g. ``ω = …`` after ``SolveFor``) keeps a
            # separate ``_as_relation`` dict; rewrite its lhs/rhs too so the
            # stored relation stays consistent with ``expr`` and a later
            # ``apply(eq)`` substitutes the ansatz-expanded form, not the stale
            # field head.
            rel = getattr(eq, "_as_relation", None)
            if rel:
                eq._as_relation = {k.replace(head, _expand): v.replace(head, _expand)
                                   for k, v in rel.items()}

        # Swap the unknown family: the field's head leaves Q, the coeff
        # family (``a(i, *coords)``) enters.
        coeff_applied = coeff_head(idx, *coords)
        # A genuine family rename (u → a): the Q key follows the new coeff family.
        model.redeclare_unknown(head, coeff_applied, rename_key=True)
        model._refresh_unknowns()
        return model


class TensorSeparationOfVariables(Operation):
    r"""Direct **tensor-product** Galerkin ansatz — the N-direction generalization
    of :func:`separation_of_variables`.

    Approximates a field by a full tensor product over TWO (or, via the generic
    ``slots`` form, N) separation coordinates at once:

    .. math::

        u(\dots,\eta,\zeta)\;\approx\;
            \sum_{i=0}^{N_z}\sum_{j=0}^{N_y}
            q_{ji}(t,x)\,\varphi_i(\zeta)\,\psi_j(\eta).

    Unlike a *chain* of single-coordinate expansions, this is inserted
    **concretely expanded** (integer orders), so every occurrence is
    collision-free — no shared summation dummies to collapse the cross terms of a
    product.  Boundary evaluations (a separation slot carrying a concrete ``0``/
    ``scale`` instead of the reference coord) resolve through φ/ψ naturally.

    Heads are matched **BY NAME** (the field head *and* its σ-decorated
    ``\tilde{field}`` sibling): a ``PDETransformation`` σ-map produces a decorated
    head that is a custom :class:`sympy.Function` subclass, and object identity is
    unreliable across maps/assumptions — so name matching is the robust key.

    Parameters
    ----------
    field : sympy applied Function
        The ORIGINAL field application (e.g. ``u(t, x, y, z)``) whose head (and
        decorated sibling) is expanded.
    coeff_head : sympy Function head
        The tensor coefficient family ``q``.  Applied as
        ``coeff_head(j, i, *kept_args)`` — lateral index ``j`` OUTER, vertical
        index ``i`` inner (matches the nested resolution order).
    pos_z, basis_z, order_z : int, Basis, int
        Argument position of the vertical (inner) separation coordinate, its
        opaque basis, and truncation order ``N_z``.
    pos_y, basis_y, order_y : int, Basis, int
        Same for the lateral (outer) separation coordinate.
    scale_y : sympy.Expr | int, optional
        The lateral slot carries the PHYSICAL coordinate ``y = scale_y·η``
        textually (a ``PDETransformation`` maps ``y → W·η``); the coefficient's
        η is recovered as ``arg / scale_y`` (so ``W·η → η``, ``0 → 0``,
        ``W → 1``).  Default ``1``.
    """

    whole_model_op = True

    def __init__(self, field, coeff_head, *, pos_z, basis_z, order_z,
                 pos_y, basis_y, order_y, scale_y=1, name="tensor_sov"):
        self._field, self._coeff = field, coeff_head
        self._pz, self._bz, self._oz = pos_z, basis_z, order_z
        self._py, self._by, self._oy = pos_y, basis_y, order_y
        self._sy = scale_y
        super().__init__(
            name=name,
            description=(f"tensor separation of variables {field} → "
                        f"Σ_ji {_coeff_key(coeff_head)}·φ_i·ψ_j"))

    def apply_to_model(self, model):
        # Match heads BY NAME — field + its σ-decorated ``\tilde{field}`` sibling
        # (decorated heads are custom Function subclasses; identity is unreliable).
        names = {self._field.func.__name__,
                 "\\tilde{%s}" % self._field.func.__name__}
        heads = {f.func for eq in model._equations.values()
                 for f in eq.expr.atoms(sp.Function) if f.func.__name__ in names}
        pz, py = self._pz, self._py
        phi = getattr(self._bz, "phi", None) or getattr(self._bz, "phi_fn",
                                                         self._bz)
        psi = getattr(self._by, "phi", None) or getattr(self._by, "phi_fn",
                                                         self._by)

        def _expand(*call_args):
            kept = tuple(a for p, a in enumerate(call_args) if p not in (pz, py))
            eta_val = sp.cancel(call_args[py] / self._sy)   # W·η→η, 0→0, W→1
            return sp.Add(*[
                self._coeff(j, i, *kept)
                * phi(i, call_args[pz]) * psi(j, eta_val)
                for i in range(self._oz + 1) for j in range(self._oy + 1)])

        for eq in model._equations.values():
            for head in heads:
                eq.expr = eq.expr.replace(head, _expand)
                rel = getattr(eq, "_as_relation", None)
                if rel:
                    eq._as_relation = {k.replace(head, _expand):
                                       v.replace(head, _expand)
                                       for k, v in rel.items()}
        return model


def separation_of_variables(field, coeff, basis, order, *, index=None):
    """Modal **separation of variables** — a :class:`Model` op.

    Returns a :class:`SeparationOfVariables` ready for ``model.apply(...)``.
    It approximates ``field`` by the UNEXPANDED sum ``Σ_i coeff(i, *coords)·
    φ(i, ζ)`` with a FRESH DISTINCT registry index, substitutes the field's
    decorated head everywhere, registers the index + basis, and redeclares
    the unknown ``field → coeff-family`` in ``Q``.

    .. math::

        u(t,x,\\zeta)\\;\\approx\\;\\sum_{i=0}^{N} a(i,t,x)\\,\\phi(i,\\zeta)

    Parameters
    ----------
    field : sympy applied Function
        The ORIGINAL field application (``u(t, x, z)``) — its decorated head
        is looked up on the model.
    coeff : sympy Function head | applied
        The modal coefficient family, given applied to its horizontal coords
        (``a(t, x)``) or as a bare head (``a``).
    basis : Basis
        The opaque basis (``basis.phi`` / ``basis.weight``).
    order : sympy.Expr | int
        Truncation bound ``N``.
    index : sympy.Symbol, optional
        Override the summation index.  By default a fresh DISTINCT index is
        drawn from the model's registry.
    """
    return SeparationOfVariables(field, coeff, basis, order, index=index)
