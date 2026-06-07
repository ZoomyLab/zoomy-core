"""Derivation operations for the clean-redesign framework.

These ops slot into the existing :class:`zoomy_core.model.operations.Operation`
spine — same ``_leaf_sp`` / ``apply_to`` / ``whole_model_op`` /
``transforms_bcs`` protocol — so an :class:`~zoomy_core.derivation.Equation`'s
``.apply`` and a :class:`~zoomy_core.derivation.Model`'s broadcast both dispatch
through them unchanged.

Plain substitutions are now just oriented :class:`~zoomy_core.model.equation.Equation`
objects (or bare ``{lhs: rhs}`` dicts) consumed by ``.apply`` — there is no
``Substitution`` class.  What remains here is the one op that does MORE than a
substitution:

``ChangeOfVariables``
    A coefficient-family rename (``a_i -> relation(q_i)``) that ALSO swaps the
    unknown family in the model's ``Q`` (``a -> q``) via
    ``model.redeclare_unknown``.  That Q-swap is exactly why it is an Operation
    and not a substitution: a substitution rewrites expressions but leaves the
    declared-unknown bookkeeping alone.

``granularity``
    Every op exposes a ``granularity`` attribute (one of ``LEAF`` / ``TERM`` /
    ``EQUATION`` / ``SYSTEM``) computed from the reused ``single_term_only`` /
    ``whole_model_op`` / ``system_level`` flags, so callers can introspect at
    which structural level an op acts.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation


__all__ = ["SolveFor", "ChangeOfVariables", "Granularity", "granularity_of"]


# ── SolveFor ───────────────────────────────────────────────────────────────


class SolveFor(Operation):
    """Reorient the equation it is applied to into ``variable = solution``,
    IN PLACE — the pipeline-style counterpart of ``equation.solve_for(var)``
    (which returns a NEW equation and leaves the original untouched).

    After ``eq.apply(SolveFor(x))`` the row IS the oriented relation ``x = …``:
    it renders as ``x = …`` in ``describe`` and applies as the substitution
    ``x → …`` wherever the equation is handed to ``.apply``.  Use it when you
    want to rewrite a model row into solved form and keep it in the model::

        m.add_equation("omega", h*omega - (wt - ...))
        m.omega.apply(SolveFor(omega))      # m.omega is now  ω = (...)/h
    """

    def __init__(self, variable, name=None):
        self._var = variable
        super().__init__(name=name or f"solve_for[{variable}]")

    def apply_to_equation(self, eq):
        solved = eq.solve_for(self._var)        # oriented Equation
        eq.expr = solved.expr                   # residual  var - solution
        eq._as_relation = dict(solved._as_relation)
        return eq

    def __repr__(self):
        return f"SolveFor({self._var})"


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
