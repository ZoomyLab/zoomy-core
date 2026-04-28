"""SimplifyStress — apply a constitutive choice to the stress tensor.

Two choices supported (``stress=`` kwarg):
  - ``"inviscid"`` : drop all stress components (τ ≡ 0).
  - ``"newtonian"``: τ = 2μ ε(u) (Newtonian viscous fluid).
"""
from __future__ import annotations

from zoomy_core.model.models.ins_generator import Inviscid, Newtonian
from .depth_integrated import DepthIntegrated


_STRESS_OPS = {
    "inviscid": Inviscid,
    "newtonian": Newtonian,
}


class SimplifyStress(DepthIntegrated):
    step_description = (
        "Apply constitutive choice to the stress tensor (inviscid → "
        "τ=0; newtonian → τ=2μ ε(u))."
    )

    def __init__(self, *, stress: str = "inviscid", **kwargs):
        super().__init__(**kwargs)
        self._init_kwargs = {**self._init_kwargs, "stress": stress}
        cached = SimplifyStress._cache_get(self._init_kwargs)
        if cached is not None:
            self._adopt_cached(cached)
            return
        self._derive_step(stress=stress)
        SimplifyStress._cache_put(self._init_kwargs, self._snapshot_for_cache())

    def _derive_step(self, *, stress: str):
        if stress not in _STRESS_OPS:
            raise ValueError(
                f"unknown stress={stress!r}; pick one of {list(_STRESS_OPS)}"
            )
        op_cls = _STRESS_OPS[stress]
        self._system.apply(op_cls(self.state))
        self._stress_choice = stress
