"""MLVAM — multi-layer non-hydrostatic shallow moments.

Stage-1 stub.  Inherits from :class:`SimplifyStress` (skips
:class:`Hydrostatic`); per-layer non-hydrostatic projection planned in
a follow-up commit.
"""
from __future__ import annotations

from zoomy_core.model.models.basisfunctions import Legendre_shifted
from .simplify_stress import SimplifyStress


class MLVAM(SimplifyStress):
    step_description = (
        "Multi-layer non-hydrostatic Galerkin projection per layer; "
        "interface jump conditions; non-hydrostatic pressure as state."
    )

    def __init__(self, *, M: int = 1, N_w: int | None = None,
                 N_p: int | None = None, n_layers: int = 2,
                 basis_type=Legendre_shifted, **kwargs):
        super().__init__(**kwargs)
        if N_w is None:
            N_w = M + 1
        if N_p is None:
            N_p = M + 1
        self._init_kwargs = {**self._init_kwargs,
                             "M": M, "N_w": N_w, "N_p": N_p,
                             "n_layers": n_layers,
                             "basis_type": basis_type.__name__}
        cached = MLVAM._cache_get(self._init_kwargs)
        if cached is not None:
            self._adopt_cached(cached)
            return
        self._derive_step(M=M, N_w=N_w, N_p=N_p, n_layers=n_layers,
                          basis_type=basis_type)
        MLVAM._cache_put(self._init_kwargs, self._snapshot_for_cache())

    def _derive_step(self, *, M, N_w, N_p, n_layers, basis_type):
        self.M = M
        self.N_w = N_w
        self.N_p = N_p
        self.n_layers = n_layers
        self.basis = basis_type(level=max(M, N_w, N_p))
