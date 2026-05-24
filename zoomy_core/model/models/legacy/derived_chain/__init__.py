"""Convenience class hierarchy for shallow-water-family models.

A linear-inheritance chain where each class adds **one derivation
step** to its parent's symbolic system:

    INS                    full incompressible Navier-Stokes (3D)
     └── DepthIntegrated   depth-integrate cont + momentum + KBCs;
                           held ``Integral`` atoms preserved
          └── SimplifyStress     constitutive choice (inviscid / Newtonian)
                ├── Hydrostatic   drop z-momentum + substitute hydrostatic p
                │    ├── SME(level=L)            Galerkin polynomial ansatz
                │    └── MLSME(level=L, n_layers=N)  per-layer Galerkin
                ├── VAM(M, N_w, N_p)              non-hydrostatic Galerkin
                └── MLVAM(...)                    multi-layer non-hydrostatic

The chain is purely additive: SME / VAM keep `_system` as a sympy
``DerivedSystem`` that the rest of the runtime can consume.
``describe(full_hierarchy=True)`` walks ``type(self).__mro__`` to
trace the derivation back to INS.

A class-level cache keyed on (qualname, frozen kwargs) avoids
re-deriving the same model twice.

This is a **convenience layer** — the existing ``SMEInviscid``,
``SMEModel``, ``VAMModel`` (in ``sme_model.py`` / ``vam_model.py``)
remain valid one-shot derivations off ``INSModel``.  Manual chains
are always allowed.
"""
from ._base import describe
from .ins import INS
from .depth_integrated import DepthIntegrated
from .simplify_stress import SimplifyStress
from .hydrostatic import Hydrostatic
from .sme import SME
from .ml_sme import MLSME
from .vam import VAM
from .ml_vam import MLVAM
from .hyperbolic_sme import HyperbolicSME

# Inject describe() as a method on every chain class.
for _cls in (INS, DepthIntegrated, SimplifyStress, Hydrostatic,
             SME, MLSME, VAM, MLVAM, HyperbolicSME):
    _cls.describe = describe

__all__ = [
    "INS",
    "DepthIntegrated",
    "SimplifyStress",
    "Hydrostatic",
    "SME",
    "MLSME",
    "VAM",
    "MLVAM",
    "HyperbolicSME",
    "describe",
]
