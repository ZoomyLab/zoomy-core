"""Turbulent Shallow Moment Equations — named full dynamical models.

Two turbulence closures on top of the dimension-agnostic :class:`SME`:

* :class:`ElderSME` — Elder parabolic eddy viscosity ``ν_t = κ u_⋆ h ζ(1−ζ)``
  (Elder 1959).  ALGEBRAIC: polynomial in ζ, closes analytically, adds NO new
  state — it is the SME dynamical system with the Elder bulk stress + a
  turbulent rough-wall bed.  Inherits SME's ``dimension`` (1-D / 2-D horizontal).

* :class:`KESME` — the two-equation k–ε model (see ``ke_sme.py``): SME moments
  PLUS depth-averaged transported turbulent kinetic energy ``k`` and dissipation
  ``ε``, coupled through the eddy viscosity ``ν_t = C_μ k²/ε``.

Both are real ``Model`` classes:
``SystemModel.from_model(ElderSME(level=2, dimension=3))``.
"""
from __future__ import annotations

from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.closures import (
    ElderViscosity, RoughWall, StressFree)


class ElderSME(SME):
    """SME with the Elder parabolic eddy-viscosity closure + a turbulent
    rough-wall bed.  Full dynamical SME (h + velocity moments evolve); the
    turbulence enters as the algebraic bulk stress ``ρ ν_t ∂_z u`` with
    ``ν_t = κ u_⋆ h ζ(1−ζ)`` (closes analytically — no quadrature, no new
    state).  Dimension-agnostic via the inherited ``dimension`` param.

    Parameters of note (``parameters={...}``): ``u_star`` (friction velocity
    scaling the Elder profile), ``kappa`` (von Kármán, default 0.41), and the
    rough-wall ``k_s`` / ``z_p`` / ``kappa`` for the bed drag.  Pass an explicit
    ``closures=[...]`` to override the default turbulent set."""

    def __init__(self, **params):
        params.setdefault(
            "closures", [ElderViscosity(), RoughWall(), StressFree()])
        super().__init__(**params)


__all__ = ["ElderSME"]
