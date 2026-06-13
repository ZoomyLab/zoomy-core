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

__all__ = ["SigmaReference", "SWE", "MalpassetSWE", "SME", "VAM",
           "MLSWE", "MLSME", "MLVAM",
           "MaterialModel", "ClosureState",
           # composable stress / interface closures (closures.py)
           "Closure", "Newtonian", "NavierSlip", "StressFree", "RoughWall",
           "KEpsilonViscosity", "InterfaceFlux", "MeanInterface",
           "UpwindInterface",
           # legacy MaterialModel factories (deprecated)
           "newtonian_navier_slip", "bingham_navier_slip"]


def __getattr__(name):
    if name == "SigmaReference":
        from zoomy_core.model.models.legacy.sigmaref import SigmaReference
        return SigmaReference
    if name == "SWE":
        from zoomy_core.model.models.swe import SWE
        return SWE
    if name == "MalpassetSWE":
        from zoomy_core.model.models.malpasset import MalpassetSWE
        return MalpassetSWE
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
    if name in ("ClosureState", "newtonian_navier_slip", "bingham_navier_slip"):
        from zoomy_core.model.models import material as _mat
        return getattr(_mat, name)
    if name in ("Closure", "Newtonian", "NavierSlip", "StressFree", "RoughWall",
                "KEpsilonViscosity", "InterfaceFlux", "MeanInterface",
                "UpwindInterface"):
        from zoomy_core.model.models import closures as _cl
        return getattr(_cl, name)
    raise AttributeError(f"module 'zoomy_core.model.models' has no attribute {name!r}")
