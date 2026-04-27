"""Galerkin projection in physical (t, x, z) coordinates.

The PDE stays in physical-z; the projection multiplies by ``φ_j(ζ(z))``
and integrates ``dz`` from ``b`` to ``b + h``.  The affine map
``z = ξh + b`` (``dz = h dξ``) is applied **only at integration time**,
so ``ξ`` never appears in the PDE itself.

Why physical-z: the σ-coord rewrite of the PDE (Escalante eq 3) carries
chain-rule "trace terms" inside ω = w − ∂_t(ξh+b) − u·∂_x(ξh+b).  Those
have to be balanced by extra algebraic identities for the eigenvalue
analysis to come out right.  Working in physical-z, sympy's chain rule
on ``∂_t / ∂_x / ∂_z`` of basis-arg compositions produces the exact
same final equations that the standard derivations (K&T 2019,
Aguillon 2026, Escalante 2024 eq 4) use.

Algorithm for ``project(eq, j)``:

  1. ``eq`` is a sympy expression in (t, x, z) with state polynomials
     u, w, p (each Σ q_i(t,x) φ_i(ζ(z))) substituted in.  Any
     ``Derivative`` atoms are propagated by sympy (chain rules through
     ζ).
  2. Apply the affine map ``z → ξh + b`` to remove z from the
     integrand.  This is NOT a plain ``xreplace``: derivatives of
     z-dependent things must be evaluated to their analytic form FIRST
     (via ``.doit()``), then the substitution is safe.
  3. Multiply by ``φ_j(ξ) · h`` (the basis evaluated at ζ=ξ on [0,1],
     and the Jacobian h from dz = h dξ).
  4. Integrate ξ from 0 to 1 via ``polynomial_integrate``.

Step 2 is the math-aware substitution: differentiate first, then
substitute.  Equivalently: ``eq.doit()`` flattens all
``Derivative(thing-of-z, z)`` to explicit z-dependence; then ``xreplace
({z: ξh+b})`` is just a coordinate substitution on a polynomial.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .ansatz import PolynomialAnsatz
from .basis import polynomial_integrate
from .flow import FlowSetup


@dataclass
class GalerkinProjection:
    """Project an equation against the basis on a column ``[b, b+h]``.

    Args:
        flow:    the FlowSetup providing (t, x, z, g, h, b) symbols.
        ansatz:  the polynomial ansatz that defines u, w, p as
                 functions of (t, x, z) via the basis evaluated at ζ.
    """
    flow: FlowSetup
    ansatz: PolynomialAnsatz

    def project(self, eq: sp.Expr, j: int) -> sp.Expr:
        """Project ``eq`` (a physical-z PDE, LHS = 0) against ``φ_j``
        and integrate ``dz`` over the column.

        Returns ``∫_{b}^{b+h} φ_j(ζ(z)) · eq(t, x, z) dz``.
        """
        z = self.flow.z
        h = self.flow.h
        xi = self.ansatz.xi_ref

        # Step 1: evaluate any z-derivatives in the equation analytically.
        # ``.doit()`` makes sympy apply the chain rule for
        # Derivative(stuff(z), z) where stuff is a polynomial in
        # ζ(z) — turning the derivative into an explicit polynomial
        # in z.
        eq_evaluated = eq.doit()

        # Step 2: substitute z → ξh + b.  After ``.doit()`` there are
        # no Derivative-w.r.t.-z atoms left, so plain xreplace is safe.
        eq_at_xi = eq_evaluated.xreplace({z: xi * h + self.flow.b})

        # Step 3: multiply by φ_j(ξ) and the Jacobian h, integrate over ξ ∈ [0, 1].
        integrand = sp.expand(self.ansatz.basis_xi[j] * eq_at_xi * h)
        return polynomial_integrate(integrand, xi, 0, 1)
