"""SigmaRef — the σ-mapped mass+momentum reference system.

Builds the shared starting point of the SME, VAM, ML-SME, ML-VAM
derivations: incompressible Navier-Stokes (mass + 2D momentum) with

* ``state.tau.xx`` dropped (no horizontal extensional viscous stress);
* ``SigmaTransform`` applied (z → σ ∈ [0, 1]);
* ``KinematicBC`` at the bottom (σ=0, interface=``state.b``) and the
  top (σ=1, interface=``state.eta``).

Everything downstream (``×h``, conservative-form ProductRule folds,
modal ansatz, Galerkin projection, …) is left to the subclass.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.models.model import Model
from zoomy_core.model.models.operations import (
    StateSpace,
    MassMomentum,
    SigmaTransform,
    KinematicBC,
)


__all__ = ["SigmaRef"]


class SigmaRef:
    """Base class.  After ``__init__`` we have ``self.state``,
    ``self.src`` and ``self.model`` populated with the σ-mapped
    reference equations:

    * ``model.equations["continuity"]``
    * ``model.equations["momentum_x"]``
    * ``model.equations["momentum_z"]``
    * ``model.equations["bottom"]`` — placeholder ``∂_t b = 0`` so the
      bottom topography lives in the model alongside the dynamic
      equations (the SystemModel matrix extractor ignores it because
      ``b(t, x)`` is a parameter, not a state).

    Subclasses extend ``self.model`` by calling further operations
    in their own ``__init__``.
    """

    def __init__(self, *, name=None):
        self.state = StateSpace(dimension=2)
        self.src = MassMomentum(self.state)
        self.model = Model(name or type(self).__name__)
        self._build_reference()

    # Override hook: subclasses can do extra pre-σ work here.
    def _pre_sigma_hook(self):
        return None

    def _build_reference(self):
        s = self.state
        m = self.model

        # Equations exactly as in the notebooks.
        m.add_equation("continuity",  self.src.continuity.expr)
        m.add_equation("momentum_x",  self.src.momentum.x.expr)
        m.add_equation("momentum_z",  self.src.momentum.z.expr)
        m.add_equation("bottom",      sp.Derivative(s.b, s.t))

        # Drop tau_xx (no horizontal extensional viscous stress) — kept
        # consistent across SME and VAM derivations.
        m.apply({s.tau.xx: 0})

        # Hook for subclass-specific pre-σ modifications (e.g. SME's
        # hydrostatic reduction of momentum_z + p-elimination).
        self._pre_sigma_hook()

        # σ-transform + boundary KBCs (the σ-domain values are 0 / 1).
        m.apply(SigmaTransform(s))
        m.apply(KinematicBC(s, interface=s.b,   at=sp.S.Zero))
        m.apply(KinematicBC(s, interface=s.eta, at=sp.S.One))
