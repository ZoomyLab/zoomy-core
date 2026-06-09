"""SigmaReference — the σ-mapped reference Model.

Inherits :class:`zoomy_core.model.basemodel.Model` directly.  Builds the
shared starting point of the SME / VAM (and future ML-SME / ML-VAM)
derivations:

* :class:`StateSpace` (coordinates only: ``t, x, [y,] z``).
* :class:`MassMomentum` (the free-surface mass + momentum balance —
  owns ``u, v, w, p, tau, h, b, eta`` and uses the Model's
  ``parameters.g`` / ``parameters.rho``).
* Adds the four reference equations
  (``continuity / momentum_x / momentum_z / bottom``).
* Drops ``tau_xx`` (no horizontal extensional viscous stress).
* Applies ``SigmaTransform`` (z → b + ζ_ref·h; registers
  ``state.zeta_ref`` on the state).
* Applies two ``KinematicBC``s (bottom σ=0, free surface σ=1).

SigmaReference standalone is informational: subsequent derivations
(SME, VAM, ML*) extend `derive_model` with closure / projection steps
that produce the dynamic system.  Until those subclasses run their
extension, the σ-mapped reference equations may contain unresolved
atoms (``tau_xz`` BCs, raw σ-integrals) — that is expected.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.basemodel import Model
from zoomy_core.model.state import StateSpace, MassMomentum
from zoomy_core.model.operations import (
    SigmaTransform,
    KinematicBC,
)


__all__ = ["SigmaReference"]


class SigmaReference(Model):
    """σ-mapped INS reference Model.

    Constructor takes ``**kwargs`` forwarded to :class:`Model`
    (`parameters`, `boundary_conditions`, `initial_conditions`,
    `eigenvalue_mode`, etc.).  Sensible defaults are set for
    `variables` (= ``["h"]``), `parameters` (= ``{"g": 9.81,
    "rho": 1.0}``), `eigenvalue_mode` (= ``"numerical"``).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("name", type(self).__name__)
        kwargs.setdefault("dimension", 1)
        kwargs.setdefault("variables", ["h"])
        kwargs.setdefault("parameters", {"g": 9.81, "rho": 1.0})
        kwargs.setdefault("eigenvalue_mode", "numerical")
        super().__init__(**kwargs)

    # ── Derivation hook ─────────────────────────────────────────────
    def derive_model(self):
        """Build the σ-mapped reference equation set."""
        # Coordinates only.
        self.state = StateSpace(dimension=2)
        # Physics — flow fields + h/b/eta + tau + the balance.
        # Pass the Model's parameters Zstruct (Symbols) so g, rho on
        # the balance equations have the SAME Symbol identity as
        # `self.parameters.g`, `self.parameters.rho`.  No translation
        # / xreplace needed downstream.
        self.src = MassMomentum(self.state, self.parameters)

        s, src = self.state, self.src

        self.add_equation("continuity", src.continuity.expr)
        self.add_equation("momentum_x", src.momentum.x.expr)
        self.add_equation("momentum_z", src.momentum.z.expr)
        self.add_equation("bottom",     sp.Derivative(src.b, s.t))

        # Drop horizontal extensional viscous stress.
        self.apply({src.tau.xx: 0})

        # Subclass pre-σ hook (SME overrides for hydrostatic + p-elim).
        self._pre_sigma_hook()

        # σ-transform: registers state.zeta_ref + substitutes
        # z → b + zeta_ref·h everywhere; drops bare z from coordinates.
        self.apply(SigmaTransform(s, src))

        # KinematicBCs at bottom (σ=0, interface=b) and surface
        # (σ=1, interface=eta).  Pass ``src`` so KinematicBC can
        # read ``u, v, w`` from the MassMomentum (they're not on
        # the new minimal StateSpace).
        self.apply(KinematicBC(s, src.b,   src, at=sp.S.Zero))
        self.apply(KinematicBC(s, src.eta, src, at=sp.S.One))

    # Override hook: SME-specific pre-σ (hydrostatic + p-elim).  VAM
    # leaves this no-op (keeps w, p as state).
    def _pre_sigma_hook(self):
        return None
