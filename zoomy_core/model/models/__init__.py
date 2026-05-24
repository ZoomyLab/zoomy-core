"""Canonical model derivations.

The five derivation classes — :class:`SigmaRef`, :class:`SME`,
:class:`VAM`, :class:`MLME`, :class:`MLVAM` — are progressively built
on top of one another: ``SigmaRef`` is the common mass+momentum
σ-mapped reference; ``SME`` and ``VAM`` add level-N Galerkin
projection (hydrostatic vs. non-hydrostatic); ``MLME`` and ``MLVAM``
stack ``N_layers`` such derivations with appropriate interface
topography and pressure.

All five reuse the same :class:`Model` /:class:`Equation`
/:class:`Term` framework and the same operations re-exported by
``operations`` (``Multiply``, ``ProductRule``, ``Integrate``,
``EvaluateIntegrals``, ``SigmaTransform``, ``AffineProjection``,
``KinematicBC``) — those operations themselves still live in the
library backend ``ins_generator``.

Legacy app-level model files have been moved to the ``legacy/``
sub-package.
"""

from zoomy_core.model.models.model import (
    Term,
    Equation,
    Model,
    Symmetrize,
)
from zoomy_core.model.models.operations import (
    Expression,
    Operation,
    Relation,
    Assumption,
    StateSpace,
    MassMomentum,
    Multiply,
    ProductRule,
    Integrate,
    EvaluateIntegrals,
    SigmaTransform,
    AffineProjection,
    KinematicBC,
    Legendre_shifted,
)
from zoomy_core.model.models.sigmaref import SigmaRef
from zoomy_core.model.models.sme   import SME
from zoomy_core.model.models.vam   import VAM
from zoomy_core.model.models.mlme  import MLME
from zoomy_core.model.models.mlvam import MLVAM


__all__ = [
    # Framework.
    "Term", "Equation", "Model", "Symmetrize",
    # Operations + symbolic primitives.
    "Expression", "Operation", "Relation", "Assumption",
    "StateSpace", "MassMomentum",
    "Multiply", "ProductRule", "Integrate", "EvaluateIntegrals",
    "SigmaTransform", "AffineProjection", "KinematicBC",
    "Legendre_shifted",
    # Derivation classes.
    "SigmaRef", "SME", "VAM", "MLME", "MLVAM",
]
