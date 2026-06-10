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

__all__ = ["SigmaReference", "SME", "VAM"]


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
    raise AttributeError(f"module 'zoomy_core.model.models' has no attribute {name!r}")
