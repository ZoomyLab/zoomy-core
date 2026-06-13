"""KESME — the two-equation k–ε Shallow Moment Equations, a full dynamical model.

The dimension-agnostic :class:`SME` moment dynamics PLUS two transported,
depth-averaged turbulence fields — turbulent kinetic energy ``k`` and its
dissipation ``ε`` — coupled to the momentum through the eddy viscosity
``ν_t = C_μ k²/ε`` (the :class:`KEpsilonViscosity` bulk closure).

Design (Rastogi & Rodi 1978, shallow k–ε):
* ``k`` and ``ε`` are DEPTH-AVERAGED scalars (functions of ``(t, x[, y])``), so
  ``ν_t`` is constant over the column ⇒ the bulk stress ``ρ ν_t ∂_z u`` is
  polynomial in ζ and the moment projection closes ANALYTICALLY (no quadrature).
  They are declared before the closure step via the SME ``_declare_turbulence_
  fields`` hook so :class:`KEpsilonViscosity` can read them.
* Their balances are added after the moment system via the
  ``_add_turbulence_transport`` hook:

      ∂_t k + Σ_d u_d ∂_d k = P_k − ε
      ∂_t ε + Σ_d u_d ∂_d ε = (ε/k)(C_1 P_k − C_2 ε)

  with the depth-mean velocity ``u_d = q_{d,0}/h`` and the SHEAR PRODUCTION
  ``P_k = ν_t Σ_d ∫₀¹ (∂_z u_d)² dζ`` built from the resolved moment profile
  ``u_d(ζ) = Σ_i (q_{d,i}/h) φ_i(ζ)`` — i.e. the moment hierarchy feeds the
  turbulence production directly.

Dimension-agnostic via the inherited ``dimension`` (1-D / 2-D horizontal).
``KESME(level=2, dimension=3).system_model``.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.closures import (
    KEpsilonViscosity, RoughWall, StressFree)

zeta = sp.Symbol("zeta", real=True)
_DERIV = {C.x: d.x, C.y: d.y}


class KESME(SME):
    """k–ε Shallow Moment Equations.  SME moments + depth-averaged ``k``, ``ε``
    transported and coupled via ``ν_t = C_μ k²/ε``.  Default closures:
    ``[KEpsilonViscosity(), RoughWall(), StressFree()]`` (turbulent eddy
    viscosity, turbulent rough-wall bed, stress-free surface)."""

    def __init__(self, **params):
        params.setdefault(
            "closures", [KEpsilonViscosity(), RoughWall(), StressFree()])
        super().__init__(**params)

    def _declare_turbulence_fields(self, m, t, horiz):
        # depth-averaged turbulence state — read by KEpsilonViscosity in §5.
        # Register PLACEHOLDER balances now so k, ε appear in m.functions (the
        # closure resolves its required fields against it); the real transport
        # overwrites them in _add_turbulence_transport once the moments exist.
        k = sp.Function("k", positive=True)(t, *horiz)
        eps = sp.Function("varepsilon", positive=True)(t, *horiz)
        m.declare_state(k, eps)
        m.add_equation("k", d.t(k))
        m.add_equation("varepsilon", d.t(eps))
        m.parameter("C_mu", 0.09)          # eddy-viscosity constant
        m.parameter("C_1", 1.44)           # ε-production constant
        m.parameter("C_2", 1.92)           # ε-dissipation constant
        self._k, self._eps = k, eps

    def _add_turbulence_transport(self, m, t, horiz, h, q_heads):
        Nu = int(self.level)
        k, eps = self._k, self._eps
        C_mu, C_1, C_2 = m.parameters.C_mu, m.parameters.C_1, m.parameters.C_2
        nu_t = C_mu * k ** 2 / eps
        # depth-mean velocity per direction and the shear-production integral
        # P_k = ν_t Σ_d ∫₀¹ (∂_z u_d)² dζ, u_d(ζ) = Σ_i (q_{d,i}/h) φ_i(ζ).
        um, Pk = [], sp.S.Zero
        for qh in q_heads:
            um.append(qh(0, t, *horiz) / h)
            u_zeta = sum((qh(i, t, *horiz) / h) * sp.legendre(i, 2 * zeta - 1)
                         for i in range(Nu + 1))
            dz_u = sp.diff(u_zeta, zeta) / h          # ∂_z = (1/h) ∂_ζ
            Pk += sp.integrate(dz_u ** 2, (zeta, 0, 1))
        Pk = nu_t * Pk
        adv_k = sum(um[i] * _DERIV[xd](k) for i, xd in enumerate(horiz))
        adv_e = sum(um[i] * _DERIV[xd](eps) for i, xd in enumerate(horiz))
        # overwrite the placeholder balances with the real transport + sources
        m._equations["k"].expr = sp.expand(d.t(k) + adv_k - (Pk - eps))
        m._equations["varepsilon"].expr = sp.expand(
            d.t(eps) + adv_e - (eps / k) * (C_1 * Pk - C_2 * eps))


__all__ = ["KESME"]
