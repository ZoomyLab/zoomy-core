"""Substitution + solver primitives.

Four primitives, all mathematically transparent:

* :func:`subst` — purely-structural ``xreplace``-based substitution.
  Accepts a dict ``{lhs: rhs}`` or any object exposing
  ``_as_relation`` (e.g. a Relation produced by :func:`solve_for`).
* :func:`function_expand` — function-level rewrite ``f(args) →
  rhs_callable(*args)`` everywhere ``f`` appears as a function call.
  Lifted from ``_FieldExpansion`` (ins_generator.py:3764).
* :func:`subs_at_point` — produce a held ``Subs(expr, var, value)``
  rendered as ``expr|_{var=value}`` by the latex printer.
* :func:`solve_for` — solve an equation for a single variable, with
  ``Derivative(Integral(...))`` atoms protected against sympy's
  Leibniz expansion (the bug-1 fix lifted verbatim from
  ``_NodeProxy.solve_for``, derived_system.py:326).
"""

from __future__ import annotations

import sympy as sp
from sympy import Derivative, Integral

from zoomy_core.symbolic.errors import PrimitiveDoesNotMatch

__all__ = [
    "subst",
    "function_expand",
    "subs_at_point",
    "solve_for",
]


def subst(expr, rule):
    """Apply ``rule`` (dict or Relation) via structural ``xreplace``.

    Pure ``xreplace`` — no sympy ``.subs()`` mass-substitution
    behaviour, no Subs wrapping when the substitution lands inside a
    ``Derivative`` whose differentiation variable matches a key.
    Sympy's ``xreplace`` is pure structural matching — what you ask
    for is what you get.

    Accepted ``rule`` shapes:

    * ``dict`` of ``{lhs: rhs}``.
    * Any object with an ``_as_relation`` attribute (a dict),
      typically produced by :func:`solve_for`.
    * Any object with an ``apply_to(expr)`` method, in which case the
      method is delegated to (matches the legacy ``Relation`` /
      ``Material`` protocol).
    """
    if hasattr(rule, "apply_to"):
        return rule.apply_to(expr)
    if hasattr(rule, "_as_relation"):
        rule = rule._as_relation
    if not isinstance(rule, dict):
        raise TypeError(
            f"subst: rule must be a dict, Relation, or object with "
            f"apply_to(); got {type(rule).__name__}"
        )
    return expr.xreplace(rule)


def function_expand(expr, field_fn, rhs_callable):
    """Rewrite every call of ``field_fn`` everywhere in ``expr``.

    ``rhs_callable(*args)`` is invoked with the call's positional
    arguments and must return the replacement expression.  Used to
    insert a velocity ansatz like

        u(t, x, arg) → α_0 + α_1·φ_1((arg − b)/h)

    The walk goes through ``Derivative``, ``Subs``, ``Integral``
    integrands — anywhere a ``field_fn`` call lives.

    Lifted verbatim from ``_FieldExpansion.apply_to``
    (ins_generator.py:3785) — already a clean primitive.
    """
    return expr.replace(field_fn, rhs_callable)


def subs_at_point(expr, var, value):
    """Produce a held ``Subs(expr, (var,), (value,))``.

    The held form means the latex printer renders it as
    ``expr|_{var=value}``.  Only :func:`un_subs` (in
    :mod:`zoomy_core.symbolic.primitives_canonical`) may unwrap it,
    and only when the safety guards pass.
    """
    return sp.Subs(expr, (var,), (value,))


def solve_for(eq_expr, variable):
    """Solve ``eq_expr == 0`` for ``variable``; return a relation
    dict ``{variable: solution}``.

    ``Derivative(Integral(...))`` atoms in ``eq_expr`` are protected
    as Dummies before ``sp.solve`` sees them.  Without this protection
    sympy eagerly Leibniz-expands such atoms — the legacy bug-1 source
    in ``_NodeProxy.solve_for`` (derived_system.py:326), which
    produced a substitution rule like

        ∂_t h = ∫∂_x u dẑ − u(b+h)·∂_x(b+h) + u(b)·∂_x b

    instead of the conservative form

        ∂_t h = −∂_x[Integral(u, (ẑ, b, b+h))]

    that pen-and-paper uses.  When applied to an equation containing
    ``α·∂_t h`` and then composed with ``_FieldExpansion``, the
    Leibniz-expanded form double-counts the moving-frame contribution.
    The Dummy-protect/unprotect bracket avoids the issue entirely.

    Returns a plain ``dict`` so callers can pass it directly to
    :func:`subst`.  Use :class:`PrimitiveDoesNotMatch` semantics: if
    no solution exists, raise.

    Lifted verbatim from ``_NodeProxy.solve_for`` (derived_system.py:326).
    """
    if not isinstance(eq_expr, sp.Basic):
        raise TypeError(
            f"solve_for: eq_expr must be a sympy Basic, got "
            f"{type(eq_expr).__name__}"
        )

    # Protect every Derivative(Integral(...), *) atom as a Dummy.
    protect_map: dict[sp.Dummy, sp.Basic] = {}

    def _protect(e):
        if isinstance(e, Derivative) and e.args[0].has(Integral):
            key = sp.Dummy(f"_solveprot_{len(protect_map)}")
            protect_map[key] = e
            return key
        if e.args:
            new_args = tuple(_protect(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    protected_expr = _protect(eq_expr)
    solutions = sp.solve(protected_expr, variable)
    if not solutions:
        raise PrimitiveDoesNotMatch(
            "solve_for", eq_expr,
            f"no solution for {variable}",
        )
    if len(solutions) > 1:
        # Caller is responsible for picking; we return the first like
        # the legacy code did but flag the ambiguity.
        import warnings
        warnings.warn(
            f"solve_for: multiple solutions for {variable}, using first: "
            f"{solutions[0]}"
        )
    solution = solutions[0].xreplace(protect_map)
    return {variable: solution}
