"""Galerkin projection of the σ-coord NS equations against ``φ_j(ξ)``.

Two source-of-w modes:

  ``w_mode='state'`` (VAM-type) — ``w`` is an independent state
  polynomial (the user's ``PolynomialAnsatz``) and the projection just
  uses ``ω = w − ∂_t(ξh+b) − u ∂_x(ξh+b)`` as-is.

  ``w_mode='from_continuity'`` (SME-type) — ``w`` is determined by
  depth-integrating continuity from the bottom, which gives

    ω(ξ) = - ξ ∂_t h - Σ_i ∂_x(h u_i) Φ_i(ξ),
    Φ_i(ξ) = ∫_0^ξ φ_i(ξ') dξ'.

  KBC ω(0) = 0 is then automatic; KBC ω(1) = 0 reduces to
  continuity j = 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import sympy as sp

from .ansatz import PolynomialAnsatz
from .basis import polynomial_integrate
from .flow import FlowSetup, HydrostaticFlow, NonHydrostaticFlow


@dataclass
class GalerkinProjection:
    """Wires ``FlowSetup`` + ``PolynomialAnsatz`` together to produce
    the j-th projected equations.

    Args:
        flow:    the σ-coord NS setup (Hydro or NonHydro).
        ansatz:  velocity / w / p polynomial degrees and basis.
        w_mode:  ``'state'`` (independent w state) or
                 ``'from_continuity'`` (w determined by depth-integrated
                 continuity); default chosen automatically:
                 ``'from_continuity'`` if the ansatz provides no w
                 coefficients, otherwise ``'state'``.
    """
    flow: FlowSetup
    ansatz: PolynomialAnsatz
    w_mode: Optional[str] = None
    _Phi: List[sp.Expr] = field(default_factory=list, repr=False)
    _omega_from_cont: sp.Expr = field(default=sp.S.Zero, repr=False)

    def __post_init__(self):
        if self.w_mode is None:
            self.w_mode = ("from_continuity" if not self.ansatz.w_coeffs
                           else "state")
        if self.w_mode == "from_continuity":
            self._build_omega_from_continuity()

    # ---- ω construction (continuity-derived branch) ----

    def _build_omega_from_continuity(self):
        """``ω(ξ) = - ξ ∂_t h - Σ_i ∂_x(h u_i) Φ_i(ξ)``."""
        xi = self.ansatz.xi
        h = self.flow.h
        t = self.flow.t
        x = self.flow.x
        # Φ_i(ξ) = ∫_0^ξ φ_i(ξ') dξ'.
        self._Phi = []
        for i in range(self.ansatz.M + 1):
            phi_i_anti = sp.Poly(self.ansatz.basis[i], xi).integrate().as_expr()
            self._Phi.append(sp.expand(phi_i_anti))             # Φ_i(0) = 0
        omega = -xi * sp.Derivative(h, t)
        for i, ui in enumerate(self.ansatz.u_coeffs):
            omega -= sp.Derivative(h * ui, x) * self._Phi[i]
        self._omega_from_cont = omega

    # ---- effective ω used in the IBP integrand ----

    @property
    def _omega_eff(self) -> sp.Expr:
        if self.w_mode == "from_continuity":
            return self._omega_from_cont
        # 'state' mode: build ω directly from the w polynomial in the
        # ansatz.
        return (self.ansatz.w
                - self.ansatz.xi * sp.Derivative(self.flow.h, self.flow.t)
                - self.ansatz.u * self.flow._d_xH_b())

    # ---- per-projection methods ----

    def _polynomial_part_of_omega(self) -> sp.Expr:
        """``ω`` polynomial part (drops opaque-w if w is a state
        polynomial — but in 'state' mode w *is* polynomial in ξ via the
        ansatz, so the full ω is polynomial; in 'from_continuity' mode
        we already have ω as a polynomial)."""
        return self._omega_eff

    def project_continuity(self, j: int) -> sp.Expr:
        """Project the σ-coord continuity equation against ``φ_j``.

        Continuity (in the ``∂_t h`` form): ``∂_t h + ∂_x(h u) + ∂_ξ ω = 0``.
        Using KBCs ω(0) = ω(1) = 0:

            ∫ φ_j (∂_t h + ∂_x(h u) + ∂_ξ ω) dξ
              = (∫φ_j dξ) · ∂_t h + ∂_x(h · ∫φ_j u dξ) − ∫φ_j' ω dξ.

        Result is the LHS of the j-th continuity projection (= 0).
        """
        xi = self.ansatz.xi
        phi_j = self.ansatz.basis[j]
        int_phi_j = polynomial_integrate(phi_j, xi)
        int_phi_j_u = polynomial_integrate(phi_j * self.ansatz.u, xi)
        dphi_j = sp.diff(phi_j, xi)
        omega_poly = self._polynomial_part_of_omega()
        int_dphi_omega = polynomial_integrate(dphi_j * omega_poly, xi)
        return (int_phi_j * sp.Derivative(self.flow.h, self.flow.t)
                + sp.Derivative(self.flow.h * int_phi_j_u, self.flow.x)
                - int_dphi_omega)

    def project_x_momentum(self, j: int) -> sp.Expr:
        """Project the σ-coord x-momentum equation against ``φ_j``.

        Boundary terms in ``∂_ξ(...)`` integrate via IBP; KBCs
        ω(0)=ω(1)=0 kill the ω·u boundary, and the surface BC
        ``p|_{ξ=1}=0`` (when the user applies it later) kills the
        surface part of the ``p ∂_x(ξh+b)`` boundary.
        """
        xi = self.ansatz.xi
        phi_j = self.ansatz.basis[j]
        u = self.ansatz.u
        p = self.ansatz.p
        int_phi_j = polynomial_integrate(phi_j, xi)
        int_phi_j_u = polynomial_integrate(phi_j * u, xi)
        int_phi_j_u2 = polynomial_integrate(phi_j * u**2, xi)
        int_phi_j_p = polynomial_integrate(phi_j * p, xi)
        # Boundary term [φ_j (ω u − p ∂_x(ξh+b))]_0^1; ω(0,1) = 0
        # ⇒ only −p ∂_x(ξh+b) survives.
        p_at_0 = sp.expand(p.subs(xi, 0))
        p_at_1 = sp.expand(p.subs(xi, 1))
        phi_j_at_0 = sp.expand(phi_j.subs(xi, 0))
        phi_j_at_1 = sp.expand(phi_j.subs(xi, 1))
        eta_x = sp.Derivative(self.flow.eta, self.flow.x)
        b_x = sp.Derivative(self.flow.b, self.flow.x)
        boundary = (-phi_j_at_1 * p_at_1 * eta_x
                    + phi_j_at_0 * p_at_0 * b_x)
        # Interior IBP: -∫φ_j' (ω u - p ∂_x(ξh+b)) dξ.
        dphi_j = sp.diff(phi_j, xi)
        omega_poly = self._polynomial_part_of_omega()
        inner = omega_poly * u - p * self.flow._d_xH_b()
        int_dphi_inner = polynomial_integrate(dphi_j * inner, xi)
        return (sp.Derivative(self.flow.h * int_phi_j_u, self.flow.t)
                + sp.Derivative(self.flow.h * int_phi_j_u2
                                + self.flow.h * int_phi_j_p, self.flow.x)
                + self.flow.g * self.flow.h * eta_x * int_phi_j
                + boundary - int_dphi_inner)

    def project_z_momentum(self, j: int) -> sp.Expr:
        """Project the σ-coord z-momentum equation against ``φ_j``
        (only meaningful for non-hydrostatic flows where ``w`` is a
        state polynomial).  Refuses if the ansatz has no w coefficients."""
        if not self.ansatz.w_coeffs:
            raise ValueError(
                "z-momentum projection requested but the ansatz has no "
                "w coefficients (w_mode='from_continuity' / hydrostatic). "
                "Either provide a w polynomial in the ansatz or skip "
                "this projection.")
        xi = self.ansatz.xi
        phi_j = self.ansatz.basis[j]
        u = self.ansatz.u
        w = self.ansatz.w
        p = self.ansatz.p
        int_phi_j_w = polynomial_integrate(phi_j * w, xi)
        int_phi_j_uw = polynomial_integrate(phi_j * u * w, xi)
        p_at_0 = sp.expand(p.subs(xi, 0))
        p_at_1 = sp.expand(p.subs(xi, 1))
        phi_j_at_0 = sp.expand(phi_j.subs(xi, 0))
        phi_j_at_1 = sp.expand(phi_j.subs(xi, 1))
        # Boundary: φ_j(1)·p(1) − φ_j(0)·p(0) (ω·w boundary kills via KBC).
        boundary = phi_j_at_1 * p_at_1 - phi_j_at_0 * p_at_0
        dphi_j = sp.diff(phi_j, xi)
        omega_poly = self._polynomial_part_of_omega()
        inner = omega_poly * w + p
        int_dphi_inner = polynomial_integrate(dphi_j * inner, xi)
        return (sp.Derivative(self.flow.h * int_phi_j_w, self.flow.t)
                + sp.Derivative(self.flow.h * int_phi_j_uw, self.flow.x)
                + boundary - int_dphi_inner)
