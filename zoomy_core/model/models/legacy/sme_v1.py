"""SME — Shallow Moment Equations (hydrostatic, level-generic, layer-generic).

Thin canonical surface over :class:`SMEModel` from
``sme_model.py``.  Reasonable defaults: shifted-Legendre basis,
single layer, level ``L=0`` (≡ standard SWE on the moment hierarchy).

The underlying derivation chain (FullINS → hydrostatic-pressure
substitution → depth integration → kinematic BCs → Newtonian /
inviscid closure) lives in ``sme_model.py``.  In a follow-up refactor
this body will be lifted into the shared
``library/zoomy_core/zoomy_core/model/sigma_balance.py`` builder so
SME and VAM share a single source of truth, but the user-visible
``SME(...)`` API is stable as of v1.

Multi-layer (``n_layers ≥ 2``) is forwarded to the underlying chain,
which already supports ``LayeredBasis`` constructions through the
generic projection path.  The Aguillon et al. 2026 ML-SWE form is
recovered at ``level=0`` with a per-layer ``Monomials`` inner basis.
"""

from __future__ import annotations

from zoomy_core.model.models.legacy.sme_model import SMEModel


class SME(SMEModel):
    """Hydrostatic shallow-moment model.

    Parameters
    ----------
    level : int, default 0
        Vertical basis order ``L`` (number of moments − 1).  ``L = 0``
        recovers SWE shape on the moment hierarchy.
    n_layers : int, default 1
        Number of vertical layers.  ``≥ 2`` activates the multilayer
        chain (canonical ML-SWE at ``level=0``, ML-SME at ``level ≥ 1``).
    basis_type : :class:`Basisfunction` class, default ``Legendre_shifted``.
    dimension : int, optional
        Horizontal dimension (1 or 2).  If omitted, inferred from the
        chain ``StateSpace``.

    Notes
    -----
    The class exposes the full param surface of :class:`SMEModel`; this
    subclass exists primarily to provide the canonical name on the
    model surface and to anchor the planned future refactor that will
    route construction through ``mass_momentum_galerkin_system``.
    """


__all__ = ["SME"]
