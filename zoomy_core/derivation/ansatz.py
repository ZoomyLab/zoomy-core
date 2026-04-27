"""Polynomial ansatzes for the velocity / pressure profiles in σ.

A ``PolynomialAnsatz(M, N, basis_fn)`` builds:

    u(t, x, ξ) = Σ_{i=0}^{M} u_i(t, x) · φ_i(ξ)
    w(t, x, ξ) = Σ_{i=0}^{N} w_i(t, x) · φ_i(ξ)
    p(t, x, ξ) = Σ_{i=0}^{N} p_i(t, x) · φ_i(ξ)

Hydrostatic models pass ``N = -1`` for ``w`` (no z-momentum, no w
state polynomial — ``w(ξ)`` is set by depth-integrated continuity in
the projection module) and pass ``N = -1`` for ``p`` (no
non-hydrostatic remainder).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import sympy as sp

from .basis import shifted_legendre_basis


@dataclass
class PolynomialAnsatz:
    """Holds the coefficient functions ``u_i(t, x)``, ``w_i(t, x)``,
    ``p_i(t, x)`` at user-chosen degrees, plus the basis polynomials
    ``φ_i(ξ)``.  Default basis is shifted Legendre.
    """
    t: sp.Symbol
    x: sp.Symbol
    xi: sp.Symbol
    M: int                                # u degree
    N_w: int                              # w degree (-1 = "no state w")
    N_p: int                              # p degree (-1 = "no non-hydro p")
    basis: List[sp.Expr] = field(default_factory=list)

    # Filled in __post_init__:
    u_coeffs: List[sp.Function] = field(default_factory=list)
    w_coeffs: List[sp.Function] = field(default_factory=list)
    p_coeffs: List[sp.Function] = field(default_factory=list)

    def __post_init__(self):
        n_max = max(self.M, self.N_w, self.N_p, 0)
        if not self.basis:
            self.basis = shifted_legendre_basis(n_max, self.xi)
        elif len(self.basis) <= n_max:
            raise ValueError(
                f"basis has only {len(self.basis)} entries but needs "
                f"{n_max + 1} (max(M={self.M}, N_w={self.N_w}, N_p={self.N_p}) + 1)."
            )
        self.u_coeffs = [sp.Function(f"u_{i}", real=True)(self.t, self.x)
                         for i in range(self.M + 1)]
        if self.N_w >= 0:
            self.w_coeffs = [sp.Function(f"w_{i}", real=True)(self.t, self.x)
                             for i in range(self.N_w + 1)]
        if self.N_p >= 0:
            self.p_coeffs = [sp.Function(f"p_{i}", real=True)(self.t, self.x)
                             for i in range(self.N_p + 1)]

    # ---- expanded polynomials ----

    @property
    def u(self) -> sp.Expr:
        return sum((self.u_coeffs[i] * self.basis[i]
                    for i in range(self.M + 1)), sp.S.Zero)

    @property
    def w(self) -> sp.Expr:
        if not self.w_coeffs:
            return sp.S.Zero
        return sum((self.w_coeffs[i] * self.basis[i]
                    for i in range(self.N_w + 1)), sp.S.Zero)

    @property
    def p(self) -> sp.Expr:
        if not self.p_coeffs:
            return sp.S.Zero
        return sum((self.p_coeffs[i] * self.basis[i]
                    for i in range(self.N_p + 1)), sp.S.Zero)

    @property
    def all_fields(self) -> List[sp.Function]:
        """Ordered list of every coefficient function: ``[u_0, …, u_M,
        w_0, …, w_{N_w}, p_0, …, p_{N_p}]``."""
        return list(self.u_coeffs) + list(self.w_coeffs) + list(self.p_coeffs)
