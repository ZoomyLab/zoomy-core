"""Operations + Relations — the canonical set used by the
``sigmaref / sme / vam / mlme / mlvam`` derivation files.

This is the only place new code should import operations from.  The
underlying library implementation still lives in ``ins_generator.py``;
we re-export the small subset we actually use to keep the new
derivation files free of the legacy 5800-line module's surface area.
"""

from __future__ import annotations

# Re-exports from the library backend.
from zoomy_core.model.models.ins_generator import (
    # Core symbolic / dispatch layer.
    Expression,
    Operation,
    Relation,
    Assumption,
    StateSpace,
    # Mass + momentum starting system.
    FullINS as MassMomentum,
    # Operations actually used by the derivation classes.
    Multiply,
    ProductRule,
    Integrate,
    EvaluateIntegrals,
    SigmaTransform,
    AffineProjection,          # kept for future SME-via-affine path.
    # Boundary conditions.
    KinematicBC,
)
from zoomy_core.model.models.basisfunctions import Legendre_shifted

# Symmetrize lives on the new Model side (depends on Expression +
# Operation but conceptually a derivation-level helper).
from zoomy_core.model.models.model import Symmetrize


__all__ = [
    "Expression",
    "Operation",
    "Relation",
    "Assumption",
    "StateSpace",
    "MassMomentum",
    "Multiply",
    "ProductRule",
    "Integrate",
    "EvaluateIntegrals",
    "SigmaTransform",
    "AffineProjection",
    "KinematicBC",
    "Legendre_shifted",
    "Symmetrize",
]
