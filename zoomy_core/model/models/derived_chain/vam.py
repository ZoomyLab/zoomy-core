"""VAM — non-hydrostatic shallow moments.

Inherits from :class:`SimplifyStress` (skips :class:`Hydrostatic`).
Pressure ``p`` and vertical velocity ``w`` remain in the state, with
their own polynomial ansätze at orders ``N_w`` and ``N_p``.

Stage 1: minimal stub.  The Galerkin polynomial projection step is not
yet implemented here (planned for a follow-up); for now the class
delivers the depth-integrated + stress-simplified system intact, which
downstream code can project as it sees fit (see ``tutorials/vam/`` for
existing manual projection paths).
"""
from __future__ import annotations

from zoomy_core.model.models.basisfunctions import Legendre_shifted
from .simplify_stress import SimplifyStress


class VAM(SimplifyStress):
    step_description = (
        "Non-hydrostatic Galerkin polynomial projection at orders "
        "(M, N_w, N_p) for u, w, p ansätze.  Surface BC p(η)=0 enforced."
    )

    def __init__(self, *, M: int = 1, N_w: int | None = None,
                 N_p: int | None = None, basis_type=Legendre_shifted, **kwargs):
        super().__init__(**kwargs)
        if N_w is None:
            N_w = M + 1
        if N_p is None:
            N_p = M + 1
        self._init_kwargs = {**self._init_kwargs,
                             "M": M, "N_w": N_w, "N_p": N_p,
                             "basis_type": basis_type.__name__}
        cached = VAM._cache_get(self._init_kwargs)
        if cached is not None:
            self._adopt_cached(cached)
            return
        self._derive_step(M=M, N_w=N_w, N_p=N_p, basis_type=basis_type)
        VAM._cache_put(self._init_kwargs, self._snapshot_for_cache())

    def _derive_step(self, *, M, N_w, N_p, basis_type):
        # Stage-1 stub: leave system as-is from SimplifyStress + record
        # the requested projection orders for downstream consumers.
        self.M = M
        self.N_w = N_w
        self.N_p = N_p
        self.basis = basis_type(level=max(M, N_w, N_p))
