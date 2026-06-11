"""Canonical model derivation classes (lazy-imported to avoid cycles).

After the Model-class rewrite, primitives moved out of this package
into ``zoomy_core/model/{basemodel,equation,state,operations}.py``.
This package retains only the derivation subclasses
(``SigmaReference``, ``SME``, ``VAM``).

Lazy import (PEP 562) is used so that importing
``zoomy_core.model.derivation.basisfunctions`` (e.g. from
``zoomy_core.model.operations``) does not trigger eager loading of
the derivation classes — which themselves depend on
``zoomy_core.model.operations`` and would create a cycle.

Users get the eager-looking surface (``from zoomy_core.model.models
import SME``) without the cycle.
"""

__all__ = ["SigmaReference", "SME", "VAM", "MLSWE", "MLSME", "MLVAM",
           "MaterialModel", "newtonian_navier_slip"]


def __getattr__(name):
    if name == "SigmaReference":
        from zoomy_core.model.models.legacy.sigmaref import SigmaReference
        return SigmaReference
    if name == "SME":
        from zoomy_core.model.models.sme import SME
        return SME
    if name == "VAM":
        from zoomy_core.model.models.vam import VAM
        return VAM
    if name == "MLSWE":
        from zoomy_core.model.models.ml_swe import MLSWE
        return MLSWE
    if name == "MLSME":
        from zoomy_core.model.models.ml_sme import MLSME
        return MLSME
    if name == "MLVAM":
        from zoomy_core.model.models.ml_vam import MLVAM
        return MLVAM
    if name == "MaterialModel":
        from zoomy_core.model.models.material import MaterialModel
        return MaterialModel
    if name == "newtonian_navier_slip":
        from zoomy_core.model.models.material import newtonian_navier_slip
        return newtonian_navier_slip
    raise AttributeError(f"module 'zoomy_core.model.models' has no attribute {name!r}")
