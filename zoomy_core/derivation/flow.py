"""σ-coordinate Navier–Stokes setup, with hydrostatic / non-hydrostatic
specialisations.

Reference: Escalante, Morales de Luna, Cantero-Chinchilla, Castro-Orgaz
(2024) eq (3) — the σ-coord form of incompressible NS over a varying
bottom ``b(x)`` of depth ``h(t, x)`` with hydrostatic / non-hydrostatic
pressure split ``p_total = p_H + p`` where ``p_H = -g h (ξ - 1)``.

The classes here are **stateless containers for sympy expressions** —
they don't manage their own derivations.  Callers wire them into a
``GalerkinProjection`` once they've chosen a ``PolynomialAnsatz``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp

from .coords import default_coords, default_h, default_b


@dataclass
class FlowSetup:
    """Symbolic σ-coord NS setup with stresses retained as opaque
    user-supplied functions (default: zero, i.e. inviscid).

    The methods return symbolic LHS expressions for each conservation
    law; callers project them onto a basis via ``GalerkinProjection``.

    The non-hydrostatic pressure ``p`` is treated symbolically — for
    the hydrostatic specialisation we'll fix ``p`` to zero in the
    momentum equations, but the *expression* shape is identical.
    """
    t: sp.Symbol
    x: sp.Symbol
    xi: sp.Symbol
    g: sp.Symbol
    h: sp.Function
    b: sp.Function
    # Stresses retained as opaque sympy expressions; default: 0
    sigma_xz: sp.Expr = sp.S.Zero          # viscous shear stress
    tau:       sp.Expr = sp.S.Zero         # bottom-friction term

    @classmethod
    def with_defaults(cls):
        t, x, xi, g = default_coords()
        h = default_h(t, x)
        b = default_b(x)
        return cls(t=t, x=x, xi=xi, g=g, h=h, b=b)

    @property
    def eta(self) -> sp.Expr:
        return self.h + self.b

    # ---- Equations (eq 3 of Escalante et al. 2024) ----
    #
    # All equations are written so that LHS = 0 holds.  ``u``, ``w``,
    # ``p`` are the (t, x, ξ)-polynomials provided by the caller's
    # ansatz; for VAM these are independent state polynomials, for SME
    # ``w`` is the depth-integrated combination derived from continuity
    # (set by the projection module).

    def continuity_lhs(self, u, w):
        """``∂_t h + ∂_x(h u) + ∂_ξ(w − ∂_t(ξh+b) − u ∂_x(ξh+b))``.

        This is the σ-coord continuity equation, rewritten with the
        ``∂_ξ ω = ∂_ξ w − ∂_ξ ∂_t(ξh+b) − ∂_ξ(u ∂_x(ξh+b))`` identity
        and using ``∂_ξ(ξh+b) = h``.  KBCs ω(0)=ω(1)=0 take care of
        the boundary terms when this is later projected.
        """
        return (sp.Derivative(self.h, self.t)
                + sp.Derivative(self.h * u, self.x)
                + sp.Derivative(self._omega_argument(u, w), self.xi))

    def x_momentum_lhs(self, u, w, p):
        """``∂_t(hu) + ∂_x(h u² + h p) + g h ∂_x η + ∂_ξ(ω u − p ∂_x(ξh+b)) − ∂_ξ σ_xz = 0``."""
        omega = self._omega(u, w)
        return (sp.Derivative(self.h * u, self.t)
                + sp.Derivative(self.h * u**2 + self.h * p, self.x)
                + self.g * self.h * sp.Derivative(self.eta, self.x)
                + sp.Derivative(omega * u - p * self._d_xH_b(), self.xi)
                - sp.Derivative(self.sigma_xz, self.xi))

    def z_momentum_lhs(self, u, w, p):
        """``∂_t(hw) + ∂_x(h u w) + ∂_ξ(ω w + p) − ∂_x(h σ_zx) + ∂_ξ(σ_zx ∂_x(ξh+b)) = 0``.

        Hydrostatic models drop this equation; non-hydrostatic models
        project it.
        """
        omega = self._omega(u, w)
        sigma_zx = self.sigma_xz                                # symmetric
        return (sp.Derivative(self.h * w, self.t)
                + sp.Derivative(self.h * u * w, self.x)
                + sp.Derivative(omega * w + p, self.xi)
                - sp.Derivative(self.h * sigma_zx, self.x)
                + sp.Derivative(sigma_zx * self._d_xH_b(), self.xi))

    # --------- ω helpers ---------

    def _d_xH_b(self) -> sp.Expr:
        """``∂_x(ξ h + b) = ξ ∂_x h + ∂_x b``."""
        return self.xi * sp.Derivative(self.h, self.x) + sp.Derivative(self.b, self.x)

    def _omega(self, u, w) -> sp.Expr:
        """σ-vertical velocity ``ω = w − ∂_t(ξh+b) − u ∂_x(ξh+b)``.

        For models where ``w`` is opaque (depth-integrated continuity,
        SME) the projection module substitutes the polynomial form for
        ``ω`` directly; this method is mainly used by the σ-momentum
        equations.
        """
        return (w
                - self.xi * sp.Derivative(self.h, self.t)
                - u * self._d_xH_b())

    def _omega_argument(self, u, w) -> sp.Expr:
        """The argument of ``∂_ξ`` in the σ-coord continuity equation
        — slightly different from ``ω`` itself: ``w − u ∂_x(ξh+b)``.
        Used by ``continuity_lhs`` to keep the equation in the form
        eq (3) row 1 of the paper."""
        return w - u * self._d_xH_b()


@dataclass
class HydrostaticFlow(FlowSetup):
    """Hydrostatic specialisation: non-hydrostatic remainder ``p ≡ 0``;
    z-momentum equation is dropped (the depth-integrated ``w`` is
    determined by continuity, not by an evolution equation)."""
    pass


@dataclass
class NonHydrostaticFlow(FlowSetup):
    """Non-hydrostatic specialisation: keeps z-momentum + non-zero
    ``p`` polynomial.  No specialisation needed beyond the base class —
    callers project the z-momentum and treat ``p`` as a state
    polynomial."""
    pass
