"""INS — full incompressible Navier-Stokes (root of the chain).

State: ``(t, x, y, z, u, v, w, p, τ_xx, τ_xz, …)`` via
:class:`StateSpace`.  Equations: continuity + momentum (x, y, z).

For 2D (x-z slice): ``y, v`` are absent.  For 3D: full state.
"""
from __future__ import annotations

from zoomy_core.model.models.ins_generator import StateSpace, FullINS
from ._base import DerivationStep


class INS(DerivationStep):
    """Full INS PDE system.  No simplifications applied yet."""

    step_description = (
        "Full incompressible Navier-Stokes (continuity, x-momentum, "
        "y-momentum, z-momentum) on (t, x, [y,] z).  No closures applied."
    )

    def __init__(self, *, dimension: int = 2, **kwargs):
        # Cache lookup
        self._init_kwargs = {"dimension": dimension, **kwargs}
        cached = INS._cache_get(self._init_kwargs)
        if cached is not None:
            self._adopt_cached(cached)
            return
        # Build fresh
        self._derive_step(dimension=dimension, **kwargs)
        INS._cache_put(self._init_kwargs, self._snapshot_for_cache())

    def _derive_step(self, *, dimension: int, **kwargs):
        self.state = StateSpace(dimension=dimension)
        self._system = FullINS(self.state)
        self._system.name = "INS"

    def _snapshot_for_cache(self) -> dict:
        return {"_system": self._system, "state": self.state}
