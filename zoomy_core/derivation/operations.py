"""Derivation operations for the clean-redesign framework.

These ops slot into the existing :class:`zoomy_core.model.operations.Operation`
spine — same ``_leaf_sp`` / ``apply_to`` / ``whole_model_op`` /
``transforms_bcs`` protocol — so an :class:`~zoomy_core.derivation.Equation`'s
``.apply`` and a :class:`~zoomy_core.derivation.Model`'s broadcast both dispatch
through them unchanged.

Two ops live here:

``Substitution``
    A general dict substitution that matches by *bare Function head* (rewrite
    every application of ``u`` regardless of arguments) for head keys, and by
    exact structural match for applied/expression keys.  It carries the
    outer-product specialisation (``over=`` / ``for_=`` / ``target=`` /
    ``into=`` / ``consume_parent=``) recovered from the lost session, which
    clones one equation row per index value.

``ChangeOfVariables``
    A coefficient-family rename (``a_i -> relation(q_i)``) that ALSO swaps the
    unknown family in the model's ``Q`` (``a -> q``) via
    ``model.redeclare_unknown``.  A bare ``Substitution`` deliberately does
    *not* do the Q-swap — that is the whole reason ``ChangeOfVariables`` is a
    distinct op rather than a flag.

``granularity``
    Every op exposes a ``granularity`` attribute (one of ``LEAF`` / ``TERM`` /
    ``EQUATION`` / ``SYSTEM``) computed from the reused ``single_term_only`` /
    ``whole_model_op`` / ``system_level`` flags, so callers can introspect at
    which structural level an op acts.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation, Relation


__all__ = ["Substitution", "ChangeOfVariables", "Granularity", "granularity_of"]


# ── granularity ──────────────────────────────────────────────────────────


class Granularity:
    """Structural level an :class:`Operation` acts on.

    Mapped onto the reused operation flags:

      * ``SYSTEM``   — ``system_level`` (couples every equation; e.g. an
                        ``InvertMassMatrix``-style whole-DAE op).
      * ``EQUATION`` — ``whole_model_op`` (grows / reshapes the equation
                        dict; e.g. an outer-product ``Substitution``).
      * ``TERM``     — ``single_term_only`` (must target one additive term).
      * ``LEAF``     — the default (per-additive-term broadcast inside one
                        equation).
    """

    LEAF = "LEAF"
    TERM = "TERM"
    EQUATION = "EQUATION"
    SYSTEM = "SYSTEM"


def granularity_of(op) -> str:
    """Return the :class:`Granularity` level an op acts on, reading the
    reused operation flags (``system_level`` / ``whole_model_op`` /
    ``single_term_only``)."""
    if getattr(op, "system_level", False):
        return Granularity.SYSTEM
    if getattr(op, "whole_model_op", False):
        return Granularity.EQUATION
    if getattr(op, "single_term_only", False):
        return Granularity.TERM
    return Granularity.LEAF


# ── Substitution ─────────────────────────────────────────────────────────


class Substitution(Relation):
    """General dict substitution ``{lhs: rhs}`` applied across a model.

    Rule kinds (split once at construction):

      * **head rule** — ``lhs`` is a bare ``sp.Function`` *class* (``a`` from
        ``sp.Function("a")``).  Every application ``a(...)`` at any argument
        list is rewritten via ``expr.replace(head, lambda *a: rhs)``.
      * **exact rule** — ``lhs`` is an applied Function / Derivative / any
        concrete expression.  Rewritten via ``expr.subs(lhs, rhs)``.

    Outer-product mode (``over=`` / ``for_=`` + ``target=`` + ``into=``):
        clone ``model._equations[target]`` once per value in ``over``, each
        clone with the single placeholder specialised to that value.  This is
        a model-level op (``whole_model_op = True``).  ``consume_parent=True``
        (default) drops the template row after the clones are created.

    A ``Substitution`` *is* a change of representation, so by default it
    rewrites boundary conditions too (``transforms_bcs = True``); the
    outer-product mode suppresses that (index specialisation is not a change
    of variable).
    """

    # A substitution IS a change of variables → it transforms BCs.
    transforms_bcs = True
    # Default: per-equation broadcast (whole_model_op flipped on only for the
    # outer-product specialisation).
    whole_model_op = False

    def __init__(self, substitutions, name="substitution", *,
                 over=None, for_=None, target=None, into=None,
                 consume_parent=True):
        # A solved-for Relation may arrive as the mapping itself — accept any
        # object exposing ``subs_map``.
        if hasattr(substitutions, "subs_map") and not isinstance(
                substitutions, (dict, list, tuple)):
            substitutions = dict(substitutions.subs_map)

        # ``for_`` is a sugar alias for ``over``; passing both is an ordering
        # footgun, so reject early.
        if for_ is not None and over is not None:
            raise ValueError(
                "Substitution: pass either `over=` or `for_=` (alias), "
                "not both."
            )
        if for_ is not None:
            over = for_

        # Sympify both sides so a raw ``0`` rhs never leaks a Python int into
        # the substitution (which would break the downstream ``.has`` checks).
        substitutions = {sp.sympify(k): sp.sympify(v)
                         for k, v in dict(substitutions).items()}

        super().__init__(substitutions, name=name)

        self._over = None
        self._target = None
        self._into = None
        self._consume_parent = True
        if over is not None:
            if target is None:
                raise ValueError(
                    "Substitution(..., over=...) requires `target=` "
                    "(the equation name to clone for each value)."
                )
            if len(self.subs_map) != 1:
                raise ValueError(
                    "Substitution(..., over=...) requires exactly one "
                    "placeholder in the substitution map "
                    f"(got {len(self.subs_map)})."
                )
            self._over = list(over)
            self._target = target
            self._into = into
            self._consume_parent = bool(consume_parent)
            # The instance grows the equation dict (one row per value), so it
            # is a model-level op.  Per-row index specialisation is NOT a
            # change of variable, so the BC-rewrite broadcast is suppressed.
            self.whole_model_op = True
            self.transforms_bcs = False

        # Split the rules into head-keyed (replace every application) and
        # applied/exact-keyed (structural match) once, up front.
        self._head_rules: dict = {}
        self._exact_rules: dict = {}
        for lhs, rhs in self.subs_map.items():
            if self._is_function_head(lhs):
                self._head_rules[lhs] = rhs
            else:
                self._exact_rules[lhs] = rhs

    @staticmethod
    def _is_function_head(key) -> bool:
        """True when ``key`` is a bare ``sp.Function`` head (an
        undefined-function *class*, e.g. ``sp.Function("u")``) rather than an
        application ``u(t, x, z)`` or any other expression."""
        return isinstance(key, sp.FunctionClass)

    # ── leaf substitution ────────────────────────────────────────────────

    def apply_to(self, expr):
        """Substitute every rule into ``expr``.

        Head rules use ``expr.replace(head, lambda *a: rhs)`` so every
        application of the head — at any argument list — is rewritten; applied
        / exact rules use ``subs``."""
        result = expr
        for head, rhs in self._head_rules.items():
            result = result.replace(head, lambda *a, _rhs=rhs: _rhs)
        for lhs, rhs in self._exact_rules.items():
            result = result.subs(lhs, rhs)
        return result

    # ── outer-product expansion (over=/for_=/target=/into=) ──────────────

    def apply_to_model(self, model):
        """Clone ``model._equations[target]`` once per value in ``over``, each
        clone with the placeholder specialised to that value.

        ``target`` may be a bare equation name (``"foo"``) or a dotted path
        into a vector equation (``"momentum.x"``); the dotted form is resolved
        by ``getattr``-walking the model.  Returns ``model``.
        """
        target = self._target
        if "." in target:
            head, *rest = target.split(".")
            src = getattr(model, head)
            for attr in rest:
                src = getattr(src, attr)
        else:
            src = model._equations[target]

        placeholder = next(iter(self.subs_map.keys()))
        name_pattern = self._into or "{target}_{value}"
        flat_target = target.replace(".", "_")

        for v in self._over:
            row_name = name_pattern.format(target=flat_target, value=v)
            model.add_equation(row_name, src.expr)
            row_sub = Substitution(
                {placeholder: v}, name=f"{self.name}[{placeholder}={v}]")
            model._equations[row_name].apply(row_sub, _no_history=True)

        # Drop the parent template — the specialised rows ARE the state rows.
        if self._consume_parent:
            parent_key = flat_target
            if parent_key in model._equations:
                model._remove_equation(parent_key)

        model._refresh_unknowns()
        return model

    @property
    def bc_substitution_map(self) -> dict:
        """The ``{lhs: rhs}`` rules used to rewrite BC expressions — the same
        rules applied to the bulk equations."""
        return dict(self.subs_map)

    def __repr__(self):
        body = ", ".join(f"{k} -> {v}" for k, v in self.subs_map.items())
        return f"Substitution({self.name!r}: {body})"


# ── ChangeOfVariables ────────────────────────────────────────────────────


class ChangeOfVariables(Operation):
    """Coefficient-family change of variables that swaps the unknown family.

    ``ChangeOfVariables("a", "q", relation)`` rewrites every application
    ``a(i, *coords) -> relation(q(i, *coords))`` across the whole model AND
    calls ``model.redeclare_unknown(a -> q)`` so the state vector ``Q`` swaps
    the ``a`` family for the ``q`` family.

    The Q-swap is exactly what distinguishes this from a bare
    :class:`Substitution`: a substitution rewrites expressions but leaves the
    declared-unknown bookkeeping alone, whereas a change of variables *is* a
    redeclaration of the unknowns.

    Parameters
    ----------
    old, new : str
        Coefficient family head names (``"a"``, ``"q"``).  ``old(...)`` is the
        family currently in ``Q``; ``new`` is what replaces it.
    relation : callable
        ``relation(new_applied) -> sympy expr`` giving the substituted form,
        e.g. ``lambda q_i: q_i / h``.  Called with the *applied* ``new``
        Function so the arguments (mode index + coords) carry through.
    """

    # A change of variables grows / rewrites the whole equation set and swaps
    # Q — it is a model-level op routed through ``apply_to_model``.
    whole_model_op = True
    transforms_bcs = True

    def __init__(self, old, new, relation, name=None):
        self._old = old
        self._new = new
        self._relation = relation
        super().__init__(
            name=name or f"change_of_variables[{old}->{new}]",
            description=f"Change of variables {old}_i -> relation({new}_i)",
        )

    def apply_to_model(self, model):
        old_head = sp.Function(self._old)
        new_head = sp.Function(self._new)

        def _rewrite(expr):
            return expr.replace(
                old_head,
                lambda *args, _new=new_head: self._relation(_new(*args)),
            )

        for eq in model._equations.values():
            eq.expr = _rewrite(eq.expr)

        # Swap the unknown family in Q (and let the model recompute Qaux).
        model.redeclare_unknown(old_head, new_head)
        model._refresh_unknowns()
        return model

    def __repr__(self):
        return (f"ChangeOfVariables({self._old!r} -> {self._new!r}, "
                f"{self._relation!r})")
