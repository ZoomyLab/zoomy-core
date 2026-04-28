"""DepthIntegrated — INS + depth-integration + KBCs.

Applies (in order, on top of INS):
  - DepthIntegrate    : ∫_b^{b+h} (·) dz, held as ``Integral`` atoms.
  - ApplyKinematicBCs : substitute w(b) and w(η) via KBCs.
  - StressFreeSurface : drop stress contribution at the free surface.
  - ZeroAtmosphericPressure : p(η) = 0.
  - SimplifyIntegrals : split / merge / cancel residual integral atoms.

The held ``Integral`` atoms remain symbolic.  Concrete subclasses
(``SME``, ``MLSME``, ``VAM``, …) replace them with either polynomial
Galerkin evaluations or numerical 3D-mesh quadrature.
"""
from __future__ import annotations

from zoomy_core.model.models.ins_generator import (
    DepthIntegrate, ApplyKinematicBCs, StressFreeSurface,
    ZeroAtmosphericPressure, SimplifyIntegrals,
)
from .ins import INS


class DepthIntegrated(INS):
    step_description = (
        "Depth-integrate every equation over [b, b+h]; apply kinematic "
        "boundary conditions at z=b and z=η; stress-free surface; "
        "atmospheric pressure gauge p(η)=0.  Held ``Integral`` atoms "
        "stay symbolic (concretised by SME / VAM / MLSME / ML-VAM)."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cache lookup at *this* level.
        cached = DepthIntegrated._cache_get(self._init_kwargs)
        if cached is not None:
            self._adopt_cached(cached)
            return
        self._derive_step()
        DepthIntegrated._cache_put(self._init_kwargs, self._snapshot_for_cache())

    def _derive_step(self):
        s = self.state
        sys_ = self._system
        # The order here matches what SMEInviscid does today (modulo the
        # hydrostatic split, which now lives in Hydrostatic).
        sys_.apply(DepthIntegrate(s))
        sys_.apply(ApplyKinematicBCs(s))
        sys_.apply(StressFreeSurface(s))
        sys_.apply(ZeroAtmosphericPressure(s))
        sys_.apply(SimplifyIntegrals(s))
