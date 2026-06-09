"""VAM — vertically-averaged non-hydrostatic moment model.

Thin canonical surface over the Galerkin-chain implementation in
:class:`VAMModelGalerkin`.  Adds the :meth:`split` method that returns
``(predictor, pressure, corrector)`` sub-:class:`SystemModel` instances
for the Chorin-style pressure-projection numerics.

Single-layer VAM is fully supported; multi-layer (``n_layers ≥ 2``)
raises ``NotImplementedError`` pending the inter-layer pressure-
continuity derivation.

The underlying mass + momentum Galerkin chain (with the canonical
``cantero_chinchilla`` quadratic form) lives in
``vam_galerkin.py``.  In a follow-up refactor this body will be lifted
into the shared
``library/zoomy_core/zoomy_core/model/sigma_balance.py`` builder so
SME and VAM share a single source of truth, but the user-visible
``VAM(...)`` API is stable as of v1.
"""

from __future__ import annotations

import param

from zoomy_core.model.models.legacy.vam_galerkin import VAMModelGalerkin
from zoomy_core.systemmodel.system_model import SystemModel


class VAM(VAMModelGalerkin):
    """Non-hydrostatic vertically-averaged moment model.

    Parameters
    ----------
    level : int
        Default basis order; sets ``M`` (U-modes) and seeds ``N_w`` and
        ``N_p`` if not given explicitly.
    M : int, optional
        Number of U-modes − 1 (defaults to ``level``).
    N_w : int, optional
        Number of active W-modes (defaults to ``level + 1``).
    N_p : int, optional
        Number of active P-modes (defaults to ``level + 1``).
    dimension : int, default 2
        StateSpace dimension (2 = 1D-horizontal + σ-vertical;
        3 = 2D-horizontal + σ-vertical).
    n_layers : int, default 1
        Number of vertical layers.  ``≥ 2`` raises ``NotImplementedError``
        until the inter-layer pressure-continuity derivation lands.
    quadratic_form : {"cantero_chinchilla", "escalante"}
        Symbolic form of the j ≥ 1 momentum rows.  Default
        ``cantero_chinchilla`` keeps W² content pointwise.
    eigenvalue_mode : {"symbolic", "numerical"}, default "symbolic".
    """

    n_layers = param.Integer(1, bounds=(1, None),
                             doc="1 = single-layer VAM; >= 2 reserved for MLVAM (not yet implemented)")

    def __init__(self, level=0, *, M=None, N_w=None, N_p=None,
                 dimension=2, n_layers=1,
                 quadratic_form="cantero_chinchilla",
                 eigenvalue_mode="symbolic", **kwargs):
        if n_layers != 1:
            raise NotImplementedError(
                "VAM(n_layers >= 2) — multilayer VAM is not yet implemented. "
                "The inter-layer non-hydrostatic-pressure continuity condition "
                "must first be derived in a thesis notebook."
            )
        super().__init__(
            level=level, M=M, N_w=N_w, N_p=N_p,
            dimension=dimension,
            quadratic_form=quadratic_form,
            eigenvalue_mode=eigenvalue_mode,
            n_layers=n_layers,
            **kwargs,
        )

    # ── Split for pressure projection ──────────────────────────────────

    def split(self, dt, *, bottom=None):
        """Return ``(SM_pred, SM_press, SM_corr)`` sub-SystemModels.

        Convenience wrapper around
        :func:`zoomy_core.model.splitter.split_for_pressure`.  Internally
        freezes ``self`` to a :class:`SystemModel`, collects the
        pressure-mode state Symbols ``[P_0, P_1, …, P_{N_p-1}]`` and
        dispatches the chain-DAE row partition.

        Parameters
        ----------
        dt : sympy Symbol
            Symbolic time-step (used by the elliptic block to scale the
            pressure source).
        bottom : sympy Function, optional
            Bottom-topography Function ``b(t, x)``.  If ``None`` the
            splitter scans residuals for a Function named ``b``.

        Returns
        -------
        :class:`SplitForPressureResult` dataclass with attributes
        ``SM_pred``, ``SM_press``, ``SM_corr`` — three
        :class:`SystemModel` instances sharing the same state vector.
        """
        from zoomy_core.model.splitter import split_for_pressure

        sm = SystemModel.from_model(self)
        pressure_vars = [
            sm.state[i]
            for i, s in enumerate(sm.state)
            if str(s).startswith("P_")
        ]
        if not pressure_vars:
            raise ValueError(
                "VAM.split: no pressure-mode state entries found "
                "(expected names like P_0, P_1, …)."
            )
        return split_for_pressure(sm, pressure_vars, dt, bottom=bottom)


__all__ = ["VAM"]
