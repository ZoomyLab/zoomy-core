"""Physical-z Navier–Stokes setup.

Reference: Escalante, Morales de Luna, Cantero-Chinchilla, Castro-Orgaz
(2024) **eq (1)** — incompressible NS in physical (t, x, z)
coordinates with constant density.  Equivalently the form K&T 2019
starts from for SME.

The PDEs stay in physical (t, x, z); the change of variable
``ζ = (z - b)/h`` is used **only** as the argument to basis polynomials
in the projection step.  No σ-coordinate ∂_ξ derivatives appear in the
PDEs themselves — those would introduce trace terms via the chain
rule that have to be balanced out by extra algebraic identities.

Pressure split: ``p_total = p_H + p`` with ``p_H = ρg(η - z)``
(hydrostatic; ``∂_z p_H = -ρg``, so the hydrostatic part exactly
cancels gravity in z-momentum).  ``p`` is the *non-hydrostatic*
remainder.  For density ρ = 1: ``p_H = g(η - z)``, ``∂_x p_H = g ∂_x η``.

After substitution the equations read:

    Continuity:    ∂_x u + ∂_z w = 0
    x-momentum:    ∂_t u + u ∂_x u + w ∂_z u + g ∂_x η + ∂_x p
                                                      = ∂_x σ_xx + ∂_z σ_xz
    z-momentum:    ∂_t w + u ∂_x w + w ∂_z w + ∂_z p = ∂_x σ_zx + ∂_z σ_zz

For inviscid models (σ ≡ 0) the right-hand sides are zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def make_physical_z_coords():
    """Return ``(t, x, z, g, h, b)`` with ``z`` the physical vertical."""
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    z = sp.Symbol("z", real=True)
    g = sp.Symbol("g", positive=True)
    h = sp.Function("h", real=True)(t, x)
    b = sp.Function("b", real=True)(x)
    return t, x, z, g, h, b


@dataclass
class FlowSetup:
    """Physical-z NS in (t, x, z) with stresses retained as opaque
    user-supplied functions (default 0 — inviscid).

    The methods return symbolic LHS expressions for each conservation
    law in physical-z coordinates; callers project them onto a basis
    via ``GalerkinProjection`` (which handles the affine map
    ``z = ξh + b`` only at integration time).
    """
    t: sp.Symbol
    x: sp.Symbol
    z: sp.Symbol
    g: sp.Symbol
    h: sp.Function
    b: sp.Function
    sigma_xx: sp.Expr = sp.S.Zero
    sigma_xz: sp.Expr = sp.S.Zero
    sigma_zz: sp.Expr = sp.S.Zero

    @classmethod
    def with_defaults(cls):
        t, x, z, g, h, b = make_physical_z_coords()
        return cls(t=t, x=x, z=z, g=g, h=h, b=b)

    @property
    def eta(self) -> sp.Expr:
        return self.h + self.b

    # ---- Physical-z PDE LHS expressions (eq 1 of Escalante 2024) ----

    def continuity_lhs(self, u, w):
        """``∂_x u + ∂_z w = 0``."""
        return sp.Derivative(u, self.x) + sp.Derivative(w, self.z)

    def x_momentum_lhs(self, u, w, p):
        """``∂_t u + u ∂_x u + w ∂_z u + g ∂_x η + ∂_x p
              − ∂_x σ_xx − ∂_z σ_xz = 0``."""
        return (sp.Derivative(u, self.t)
                + u * sp.Derivative(u, self.x)
                + w * sp.Derivative(u, self.z)
                + self.g * sp.Derivative(self.eta, self.x)
                + sp.Derivative(p, self.x)
                - sp.Derivative(self.sigma_xx, self.x)
                - sp.Derivative(self.sigma_xz, self.z))

    def z_momentum_lhs(self, u, w, p):
        """``∂_t w + u ∂_x w + w ∂_z w + ∂_z p
              − ∂_x σ_zx − ∂_z σ_zz = 0``.

        Note: gravity is already absorbed via the hydrostatic split
        (``p_H = g(η - z)``, ``∂_z p_H = -g`` cancels the body force).
        Hydrostatic models drop this equation entirely.
        """
        sigma_zx = self.sigma_xz
        return (sp.Derivative(w, self.t)
                + u * sp.Derivative(w, self.x)
                + w * sp.Derivative(w, self.z)
                + sp.Derivative(p, self.z)
                - sp.Derivative(sigma_zx, self.x)
                - sp.Derivative(self.sigma_zz, self.z))

    # ---- Convenience: ζ(z) for downstream callers ----

    def zeta(self):
        """``ζ = (z - b)/h`` — the basis-argument map.  NOT a coordinate
        of the PDE — used only by the projection module as an argument
        to the basis polynomials φ_i(ζ)."""
        return (self.z - self.b) / self.h


@dataclass
class HydrostaticFlow(FlowSetup):
    """Hydrostatic specialisation: drop z-momentum (w determined by
    depth-integrated continuity).  Non-hydrostatic ``p`` is identically
    zero."""
    pass


@dataclass
class NonHydrostaticFlow(FlowSetup):
    """Non-hydrostatic: keep z-momentum and the non-hydrostatic
    pressure remainder ``p`` as state."""
    pass
