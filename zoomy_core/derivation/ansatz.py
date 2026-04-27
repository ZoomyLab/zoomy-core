"""Polynomial ansatz in physical (t, x, z): basis evaluated at ``ζ(z)``.

A ``PolynomialAnsatz(M, N_w, N_p)`` builds:

    u(t, x, z) = Σ_{i=0}^{M}   u_i(t, x) · φ_i(ζ(z))
    w(t, x, z) = Σ_{i=0}^{N_w} w_i(t, x) · φ_i(ζ(z))
    p(t, x, z) = Σ_{i=0}^{N_p} p_i(t, x) · φ_i(ζ(z))

where ``ζ(z) = (z - b(x))/h(t, x)`` is just an argument to the basis
polynomials — NOT a coordinate of the PDE.  Models that don't need
``w`` or ``p`` as state pass ``N_w = -1`` or ``N_p = -1``.

The basis polynomials φ_i are kept as polynomials in a separate
"reference" symbol ``ξ_ref``; ``basis_at_z(i)`` returns φ_i evaluated
at ζ(z) by ``xreplace({ξ_ref: ζ(z)})``.

Sympy's ``sp.diff`` then handles all chain rules through ``ζ(z)``
correctly when the caller takes derivatives ``∂_t``, ``∂_x``, ``∂_z``
of the resulting expressions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import sympy as sp

from .basis import shifted_legendre_basis


@dataclass
class PolynomialAnsatz:
    t: sp.Symbol
    x: sp.Symbol
    z: sp.Symbol
    h: sp.Function
    b: sp.Function
    M: int
    N_w: int
    N_p: int
    xi_ref: sp.Symbol = field(default_factory=lambda: sp.Symbol("xi", real=True))
    basis_xi: List[sp.Expr] = field(default_factory=list)

    u_coeffs: List[sp.Function] = field(default_factory=list)
    w_coeffs: List[sp.Function] = field(default_factory=list)
    p_coeffs: List[sp.Function] = field(default_factory=list)

    def __post_init__(self):
        n_max = max(self.M, self.N_w, self.N_p, 0)
        if not self.basis_xi:
            self.basis_xi = shifted_legendre_basis(n_max, self.xi_ref)
        self.u_coeffs = [sp.Function(f"u_{i}", real=True)(self.t, self.x)
                         for i in range(self.M + 1)]
        if self.N_w >= 0:
            self.w_coeffs = [sp.Function(f"w_{i}", real=True)(self.t, self.x)
                             for i in range(self.N_w + 1)]
        if self.N_p >= 0:
            self.p_coeffs = [sp.Function(f"p_{i}", real=True)(self.t, self.x)
                             for i in range(self.N_p + 1)]

    @property
    def zeta_of_z(self) -> sp.Expr:
        """``ζ(z) = (z - b(x)) / h(t, x)``."""
        return (self.z - self.b) / self.h

    def basis_at_z(self, i: int) -> sp.Expr:
        """φ_i evaluated at ζ(z).  Returns a sympy expression in
        (t, x, z) on which ``sp.diff`` applies chain rules correctly."""
        return self.basis_xi[i].xreplace({self.xi_ref: self.zeta_of_z})

    @property
    def u(self) -> sp.Expr:
        """``u(t, x, z) = Σ u_i(t, x) φ_i(ζ(z))``."""
        return sum((self.u_coeffs[i] * self.basis_at_z(i)
                    for i in range(self.M + 1)), sp.S.Zero)

    @property
    def w(self) -> sp.Expr:
        if not self.w_coeffs:
            return sp.S.Zero
        return sum((self.w_coeffs[i] * self.basis_at_z(i)
                    for i in range(self.N_w + 1)), sp.S.Zero)

    @property
    def p(self) -> sp.Expr:
        if not self.p_coeffs:
            return sp.S.Zero
        return sum((self.p_coeffs[i] * self.basis_at_z(i)
                    for i in range(self.N_p + 1)), sp.S.Zero)

    @property
    def all_fields(self) -> List[sp.Function]:
        return list(self.u_coeffs) + list(self.w_coeffs) + list(self.p_coeffs)
