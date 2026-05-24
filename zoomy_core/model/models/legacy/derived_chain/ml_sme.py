"""MLSME — multi-layer Shallow Moment Equations.

Stage-1 stub.  Inherits from :class:`Hydrostatic`; per-layer Galerkin
polynomial projection planned in a follow-up commit.

The per-layer derivation pattern is documented in
``tutorials/multilayer/ml_sme_prototype.py`` (Aguillon-Hörnschemeyer
2026 eq (5)).
"""
from __future__ import annotations

from zoomy_core.model.models.basisfunctions import Legendre_shifted
from .hydrostatic import Hydrostatic


class MLSME(Hydrostatic):
    step_description = (
        "Multi-layer per-layer Galerkin polynomial projection at order "
        "``level`` per layer; jump conditions at layer interfaces."
    )

    def __init__(self, *, level: int = 0, n_layers: int = 2,
                 basis_type=Legendre_shifted, **kwargs):
        super().__init__(**kwargs)
        self._init_kwargs = {**self._init_kwargs,
                             "level": level, "n_layers": n_layers,
                             "basis_type": basis_type.__name__}
        cached = MLSME._cache_get(self._init_kwargs)
        if cached is not None:
            self._adopt_cached(cached)
            return
        self._derive_step(level=level, n_layers=n_layers, basis_type=basis_type)
        MLSME._cache_put(self._init_kwargs, self._snapshot_for_cache())

    def _derive_step(self, *, level, n_layers, basis_type):
        # Stage-1 stub: leave system at Hydrostatic state; record params.
        self.level = level
        self.n_layers = n_layers
        self.basis = basis_type(level=level)
