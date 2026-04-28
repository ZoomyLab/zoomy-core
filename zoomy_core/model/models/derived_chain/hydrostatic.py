"""Hydrostatic — drop z-momentum and substitute hydrostatic pressure.

Applies the standard hydrostatic-pressure split:
  - ``∂_z p = -ρg`` ⇒  ``p(z) = p_atm + ρg(η - z)``.
  - Substitute that ``p`` everywhere in the (depth-integrated) system.
  - Drop the z-momentum equation (now identically satisfied).
  - Set ``w = 0`` and zero all z-row stresses (closure consistent
    with hydrostatic shallow flow).

After this step the system has no pressure / vertical-velocity / z-stress
unknowns.  Suitable basis for SME and MLSME (which then add a
polynomial Galerkin ansatz).
"""
from __future__ import annotations

from sympy import S
from zoomy_core.model.models.ins_generator import HydrostaticPressure
from .simplify_stress import SimplifyStress


def _hydrostatic_scaling(state):
    """Drop w and all z-row/z-column stresses."""
    sub = {state.w: S.Zero}
    for key in state.tau.keys():
        if "z" in key:
            sub[state.tau[key]] = S.Zero
    return sub


class Hydrostatic(SimplifyStress):
    step_description = (
        "Hydrostatic pressure split: derive p(z) from z-momentum, drop "
        "z-momentum, set w=0 and zero all z-row stresses."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cached = Hydrostatic._cache_get(self._init_kwargs)
        if cached is not None:
            self._adopt_cached(cached)
            return
        self._derive_step()
        Hydrostatic._cache_put(self._init_kwargs, self._snapshot_for_cache())

    def _derive_step(self):
        s = self.state
        sys_ = self._system
        # Same operations as SMEInviscid does today, only re-ordered to
        # apply *after* DepthIntegrate + KBCs.  HydrostaticPressure is an
        # apply-able op that substitutes p(z) into all other equations.
        sys_.momentum.z.apply(_hydrostatic_scaling(s)).simplify()
        sys_.apply(HydrostaticPressure(s))
        sys_.remove_equation("momentum.z")
