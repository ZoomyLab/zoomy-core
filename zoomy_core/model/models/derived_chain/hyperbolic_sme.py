"""HyperbolicSME — SME with a hyperbolicity-fixing modification.

Inherits from :class:`SME`.  Applies one of the regularisation
strategies discovered in the L=2/L=3 mode-pattern analysis
(see ``thesis/notebooks/modeling/sme/sme_l2_regularization_findings.md``):

  - ``"none"``       : passthrough.
  - ``"minimal"``    : zero a single coupling entry of the principal
                       symbol (the minimum-norm strict-hyperbolicity
                       fix; e.g. ``A[1, 2] = 0`` at L=2).
  - ``"koellermeier"``: drop the entire Schur-feedback term
                       (Koellermeier-Torrilhon style).
  - ``"viscous"``    : add τ_xx Newtonian shear stress (parabolic
                       regularisation; not strictly hyperbolic but
                       dispersively well-posed).

Stage-1 stub.  The actual flux mutation is left as a follow-up; for
now this records the requested ``hyperbolic_fix`` choice on the
instance so downstream code can branch.
"""
from __future__ import annotations

from .sme import SME


class HyperbolicSME(SME):
    step_description = (
        "Apply hyperbolicity-fixing regularisation to the SME flux "
        "(``hyperbolic_fix``: none / minimal / koellermeier / viscous)."
    )

    _ALLOWED_FIXES = {"none", "minimal", "koellermeier", "viscous"}

    def __init__(self, *, hyperbolic_fix: str = "minimal", **kwargs):
        super().__init__(**kwargs)
        if hyperbolic_fix not in self._ALLOWED_FIXES:
            raise ValueError(
                f"hyperbolic_fix={hyperbolic_fix!r}; pick one of "
                f"{sorted(self._ALLOWED_FIXES)}"
            )
        self._init_kwargs = {**self._init_kwargs,
                             "hyperbolic_fix": hyperbolic_fix}
        cached = HyperbolicSME._cache_get(self._init_kwargs)
        if cached is not None:
            self._adopt_cached(cached)
            return
        self._derive_step(hyperbolic_fix=hyperbolic_fix)
        HyperbolicSME._cache_put(self._init_kwargs, self._snapshot_for_cache())

    def _derive_step(self, *, hyperbolic_fix: str):
        # Stage-1 stub: just record the choice; flux mutation TBD.
        self.hyperbolic_fix = hyperbolic_fix
