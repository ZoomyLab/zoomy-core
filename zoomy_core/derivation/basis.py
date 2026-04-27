"""Shifted Legendre basis on [0, 1] + polynomial integration helpers.

Paper convention (Escalante et al. 2024, K&T 2019, Aguillon et al.
2026, …):

    φ_n(ξ) = P_n(1 − 2ξ)        with   φ_n(0) = 1, φ_n(1) = (-1)^n,
    ∫_0^1 φ_n² dξ = 1 / (2n + 1).

So ``φ_0 = 1, φ_1 = 1 − 2ξ, φ_2 = 6ξ² − 6ξ + 1, …``.
"""
from __future__ import annotations

from typing import List

import sympy as sp


def shifted_legendre_basis(n_max: int, xi: sp.Symbol) -> List[sp.Expr]:
    """Return ``[φ_0, φ_1, …, φ_{n_max}]`` in the ``ξ`` variable."""
    s = sp.Symbol("__shifted_legendre_arg__")
    return [sp.expand(sp.legendre(i, s).subs(s, 1 - 2 * xi))
            for i in range(n_max + 1)]


def polynomial_integrate(integrand, var: sp.Symbol, lo=0, hi=1):
    """``∫_lo^hi p(var) d(var)`` for polynomial ``p`` (with arbitrary
    opaque coefficients).  Falls back to the constant-integrand rule if
    the integrand has no ``var`` dependence."""
    expr = sp.expand(integrand)
    if not expr.has(var):
        return (hi - lo) * expr
    poly = sp.Poly(expr, var)
    anti = poly.integrate().as_expr()
    return sp.expand(anti.subs(var, hi) - anti.subs(var, lo))
