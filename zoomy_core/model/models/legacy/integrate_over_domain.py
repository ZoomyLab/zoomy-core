"""``IntegrateOverDomain`` and ``SplitIntegralOverAdd`` — the small
glue Operations that, with :class:`DivergenceTheorem` and
:class:`MapToReferenceElement`, let a multi-D weak form be derived
through a sequence of ``.apply(...)`` calls (no raw sympy).
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.models.legacy.ins_generator import Operation
from zoomy_core.symbolic.domains import Domain


class IntegrateOverDomain(Operation):
    """Wrap an expression in ``∫_Ω (·) d**x**`` over a :class:`Domain`.

    Each leaf's expression ``e`` is replaced with
    ``sp.Integral(e, *domain.coords)`` — i.e. an iterated integral over
    every coordinate of the domain (indefinite limits, since the domain
    bounds are carried implicitly via the ``Domain`` object).  This is
    the multi-D analogue of "wrap in an integral" that you'd otherwise
    write by hand as ``sp.Integral(expr, x, y)``.
    """

    whole_leaf_op = True

    def __init__(self, domain: Domain, *,
                 name: str | None = None,
                 description: str | None = None):
        super().__init__(
            name=name or f"integrate_over_{domain.name}",
            description=(description or
                         f"Wrap leaf in ∫_{domain.name} (·) d{domain.coords}"),
        )
        self._domain = domain

    def _leaf_sp(self, expr: sp.Expr) -> sp.Expr:
        return sp.Integral(expr, *self._domain.coords)


class SplitIntegralOverAdd(Operation):
    """Distribute ``∫(a + b) → ∫a + ∫b`` across every ``Integral`` atom.

    Used after :class:`IntegrateOverDomain` so each integrable
    subterm ends up in its own ``Integral`` — that's the form
    :class:`DivergenceTheorem` expects, since each Integral atom
    is then a single-term integrand and the user can target one
    via ``apply_to_term(idx, ...)``.
    """

    whole_leaf_op = True

    def __init__(self, *, name: str | None = None,
                 description: str | None = None):
        super().__init__(
            name=name or "split_integral_over_add",
            description=(description or
                         "∫(a + b + …) → ∫a + ∫b + …"),
        )

    def _leaf_sp(self, expr: sp.Expr) -> sp.Expr:
        def _walk(e):
            if isinstance(e, sp.Integral):
                # Recurse first so nested Integrals get split too.
                integrand = _walk(e.args[0])
                limits = e.args[1:]
                # Expand to surface any Add hidden inside a Mul (e.g.
                # ``φ · (Δu + f)`` → ``φ Δu + φ f``).  Without this,
                # the ``isinstance(..., Add)`` check below never fires
                # when the user multiplied by a test function before
                # integrating.
                expanded = sp.expand(integrand)
                if isinstance(expanded, sp.Add):
                    return sp.Add(*[sp.Integral(t, *limits)
                                    for t in expanded.args])
                return sp.Integral(expanded, *limits)
            if e.args:
                new_args = tuple(_walk(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e
        return _walk(expr)
