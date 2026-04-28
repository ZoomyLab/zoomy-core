"""Physical-z Galerkin projection (no σ-coord transformation).

The σ-coord-first approach in ``flow.py`` + ``projection.py``
introduces extra terms via the chain rule of ``∂_x`` in (t, x, ξ)
coordinates.  Those terms appear inside the IBP integrand of the
σ-momentum equations and don't get cleanly eliminated even when the
algebraic constraints are solved.

The standard SME / VAM derivations (K&T 2019, slim_walkthrough,
Escalante 2024 eq 4 in its compact form) work in **physical (t, x, z)**
coordinates throughout.  The basis ``φ_k`` is evaluated at
``ζ = (z - b)/h`` but only *as an argument* — the PDE itself stays
in (t, x, z), and ``∂_x`` is the *physical* ∂_x at fixed z.

Workflow this module supports:

  1. Equations in physical (t, x, z): ``∂_t u + u ∂_x u + w ∂_z u + …``.
  2. Hydrostatic models: depth-integrate continuity to express
     ``w(t, x, z) = u|_b · ∂_x b − ∫_b^z ∂_x u(t, x, z') dz'``
     using KBC at the bottom.
  3. Multiply momentum by basis ``φ_k(ζ(z))`` and integrate
     ``dz`` from ``b`` to ``b + h``.
  4. For ``∂_x ∫(...) dz`` terms use **Leibniz rule**:
     ``∂_x ∫_b^{b+h} F dz = ∫_b^{b+h} ∂_x F dz + F|_{b+h} ∂_x(b+h)
                            − F|_b ∂_x b``.
     Boundary values evaluated via KBCs.
  5. The integration over ``z`` uses the affine substitution
     ``z = ζh + b`` only at integration time (so the basis becomes a
     polynomial in ``ζ ∈ [0, 1]`` for ``polynomial_integrate``).

This module is a clean rewrite that produces the standard K&T 2019
SME equations literally (matching ``kt2019_verification.py``) and
the standard VAM eq (4) form for VAM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import sympy as sp

from .ansatz import PolynomialAnsatz
from .basis import polynomial_integrate
from .coords import default_coords, default_h, default_b


# ---------------------------------------------------------------------------
# Coordinate helper
# ---------------------------------------------------------------------------

def make_physical_z_coords():
    """Return ``(t, x, z, ζ, g, h, b)`` with ``ζ = (z - b)/h``.

    z is the physical vertical coordinate; ζ is its scaled image on
    [0, 1] used only as the argument to the basis polynomials.
    """
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    z = sp.Symbol("z", real=True)
    g = sp.Symbol("g", positive=True)
    h = sp.Function("h", real=True)(t, x)
    b = sp.Function("b", real=True)(x)
    zeta = (z - b) / h
    return t, x, z, zeta, g, h, b


# ---------------------------------------------------------------------------
# Physical-z polynomial ansatz
# ---------------------------------------------------------------------------

@dataclass
class PhysicalZAnsatz:
    """Polynomial ansatz in physical z.  Each component is

        u(t, x, z) = Σ_i u_i(t, x) · φ_i((z − b)/h),
        w(t, x, z) = Σ_i w_i(t, x) · φ_i((z − b)/h),
        p(t, x, z) = Σ_i p_i(t, x) · φ_i((z − b)/h)

    where ``φ_i`` are basis polynomials in their argument (shifted
    Legendre on [0, 1] by default).
    """
    t: sp.Symbol
    x: sp.Symbol
    z: sp.Symbol
    h: sp.Function
    b: sp.Function
    M: int                                       # u degree
    N_w: int                                     # w degree (-1 = no w state)
    N_p: int                                     # p degree (-1 = no non-hydro p)
    basis_xi: List[sp.Expr] = field(default_factory=list)
    xi: sp.Symbol = field(default_factory=lambda: sp.Symbol("xi", real=True))

    u_coeffs: List[sp.Function] = field(default_factory=list)
    w_coeffs: List[sp.Function] = field(default_factory=list)
    p_coeffs: List[sp.Function] = field(default_factory=list)

    def __post_init__(self):
        from .basis import shifted_legendre_basis
        n_max = max(self.M, self.N_w, self.N_p, 0)
        if not self.basis_xi:
            self.basis_xi = shifted_legendre_basis(n_max, self.xi)
        self.u_coeffs = [sp.Function(f"u_{i}", real=True)(self.t, self.x)
                         for i in range(self.M + 1)]
        if self.N_w >= 0:
            self.w_coeffs = [sp.Function(f"w_{i}", real=True)(self.t, self.x)
                             for i in range(self.N_w + 1)]
        if self.N_p >= 0:
            self.p_coeffs = [sp.Function(f"p_{i}", real=True)(self.t, self.x)
                             for i in range(self.N_p + 1)]

    @property
    def zeta(self) -> sp.Expr:
        return (self.z - self.b) / self.h

    def basis_at_z(self, i: int) -> sp.Expr:
        """φ_i evaluated at ``ζ((z - b)/h)``."""
        return self.basis_xi[i].xreplace({self.xi: self.zeta})

    @property
    def u(self) -> sp.Expr:
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


# ---------------------------------------------------------------------------
# Depth-integrated w from continuity
# ---------------------------------------------------------------------------

def w_from_continuity_physical_z(ansatz: PhysicalZAnsatz) -> sp.Expr:
    """Express ``w(t, x, z)`` via depth-integrated continuity.

    Continuity (incompressible, 2D): ``∂_x u + ∂_z w = 0``.

    Integrate from ``b`` to ``z`` and apply KBC at bottom
    (``w(b) = u(b) ∂_x b``):

        w(z) = u(b) · ∂_x b  −  ∫_b^z ∂_x u(t, x, z') dz'

    Here ``u(t, x, z) = Σ_i u_i(t, x) · φ_i((z - b)/h)``.  The
    derivative ``∂_x u`` at fixed z follows the chain rule:

        ∂_x u(t, x, z)|_z = Σ_i (∂_x u_i)·φ_i  +  Σ_i u_i · φ_i'(ζ) · ∂_x ζ,
        ∂_x ζ = - (∂_x b + ζ ∂_x h) / h.

    The integral ``∫_b^z (...) dz'`` is computed by changing variable
    ``z' = ζ' h + b``, ``dz' = h dζ'`` (since ``h, b`` are independent
    of z'), running ζ' from 0 to ``ζ = (z-b)/h``.
    """
    t, x, z = ansatz.t, ansatz.x, ansatz.z
    h, b = ansatz.h, ansatz.b
    xi = ansatz.xi
    zeta = ansatz.zeta

    # u as polynomial in xi (after substituting back z = xi·h + b in
    # the basis argument).
    u_xi = sum((ansatz.u_coeffs[i] * ansatz.basis_xi[i]
                for i in range(ansatz.M + 1)), sp.S.Zero)

    # ∂_x u at fixed z: chain rule.
    dx_u = sp.S.Zero
    for i in range(ansatz.M + 1):
        u_i = ansatz.u_coeffs[i]
        phi_i_xi = ansatz.basis_xi[i]
        dphi_i_xi = sp.diff(phi_i_xi, xi)
        # ∂_x u_i term:
        dx_u += sp.Derivative(u_i, x) * phi_i_xi
        # u_i · φ_i'(ζ) · ∂_x ζ term, where ∂_x ζ at fixed z is
        # -(∂_x b + ζ ∂_x h)/h.  Express in terms of xi via ζ → xi
        # (since we're inside the integral).
        dx_zeta = -(sp.Derivative(b, x) + xi * sp.Derivative(h, x)) / h
        dx_u += u_i * dphi_i_xi * dx_zeta

    # ∫_0^ζ (∂_x u as polynomial in xi) · h · dxi → polynomial in ζ.
    # First treat h as constant (it doesn't depend on z'), so the
    # integral is h times a polynomial integral in xi.
    # We integrate from xi=0 to xi=zeta (substituting z'=xi·h+b → ζ').
    # Use indefinite-integral approach: get Φ(xi) = ∫₀^xi dx_u dxi',
    # then evaluate at xi=ζ and multiply by h.
    integrand = sp.expand(dx_u)
    # Convert to polynomial in xi.
    poly = sp.Poly(integrand, xi)
    anti = poly.integrate().as_expr()                # ∫dx_u dxi' from 0 to xi
    # Multiply by h (since dz' = h dxi').
    integral_b_to_z = h * sp.expand(anti.xreplace({xi: zeta}))

    # u at the bottom: u(z=b) → ζ=0 → φ_i(0)=1 for all i.
    u_at_b = sum(ansatz.u_coeffs[i] for i in range(ansatz.M + 1))

    w_z = u_at_b * sp.Derivative(b, x) - integral_b_to_z
    return sp.expand(w_z)


# ---------------------------------------------------------------------------
# Physical-z Galerkin projection
# ---------------------------------------------------------------------------

@dataclass
class PhysicalZProjection:
    """Project an equation ``LHS(t, x, z) = 0`` against ``φ_j(ζ(z))``
    by integrating ``dz`` from ``b`` to ``b + h``.

    Methods:
      ``project(eq, j)`` — multiply by φ_j(ζ), integrate dz over the
      column.  Affine transform z = ζh + b is applied internally for
      the integration step (so the integrand becomes a polynomial in
      ζ).
    """
    ansatz: PhysicalZAnsatz

    def project(self, eq: sp.Expr, j: int) -> sp.Expr:
        """``∫_b^{b+h} φ_j(ζ(z)) · eq dz``.

        Affine transform z = ζ·h + b, dz = h dζ.  Integrand becomes
        a polynomial in ζ ∈ [0, 1].
        """
        z = self.ansatz.z
        xi = self.ansatz.xi
        h = self.ansatz.h
        b = self.ansatz.b
        # Substitute z = xi·h + b in the equation + multiply by φ_j(xi).
        eq_xi = eq.xreplace({z: xi * h + b})
        integrand = sp.expand(self.ansatz.basis_xi[j] * eq_xi)
        # Multiply by h (the dz = h dxi Jacobian) and integrate.
        return polynomial_integrate(integrand * h, xi, 0, 1)
