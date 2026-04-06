"""
Legacy model files — hand-derived, pre-3-phase pipeline.

These models are preserved for reference but are no longer actively
maintained.  New development should use the 3-phase pipeline:

    from zoomy_core.model.models.ins_generator import StateSpace, FullINS
    from zoomy_core.model.models.model_derivation import derive_shallow_moments
    from zoomy_core.model.models.projected_model import ProjectedModel

For VAM (non-hydrostatic):
    from zoomy_core.model.models.vam_derivation import derive_vam_moments
    from zoomy_core.model.models.vam_projected_model import VAMProjectedHyperbolic
"""

import warnings

warnings.warn(
    "zoomy_core.model.models.legacy contains deprecated model files. "
    "Use the 3-phase pipeline (ins_generator → model_derivation → projected_model) instead.",
    DeprecationWarning,
    stacklevel=2,
)
