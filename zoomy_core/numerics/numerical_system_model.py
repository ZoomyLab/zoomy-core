"""NumericalSystemModel — numerical sibling of :class:`SystemModel`.

The NSM bundles a frozen :class:`SystemModel` with everything a solver
needs that is *not* symbolic-PDE-shaped:

    - the Riemann-solver class (a :class:`Numerics` subclass)
    - the LSQ reconstruction spec (order, limiter)
    - the diffusion-scheme spec (Crank-Nicolson, ν override, …)
    - numerical regularization (eigenvalue eps, …)
    - the LSQ polynomial degree (auto-derived from the SystemModel's
      ``aux_registry`` when not pinned explicitly)

This means a solver constructor no longer takes a pile of
``(model, riemann=, reconstruction_order=, eigenvalue_regularization=,
…)`` kwargs.  Solvers consume the NSM directly; everything else is a
runtime knob (dt control, end time, IO, GMRES tolerances).

The class is **read-only** on the contained :class:`SystemModel` — it
never mutates the symbolic operators.  Two reasons:

1. Other agents may be revisiting ``SystemModel`` concurrently; we keep
   the seam clean by leaving the symbolic frozen-form untouched.
2. Numerical operations (regularization, field-inversion-into-aux,
   splitting into sub-systems) belong on the NSM, not on the symbolic
   sibling — mirrors the user's mental model that *numerical*
   transformations operate on a *numerical* container.

Pipeline:  Model → SystemModel → NumericalSystemModel → Solver
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Type

from zoomy_core.model.models.system_model import SystemModel


# ── Slot dataclasses ────────────────────────────────────────────────


@dataclass
class ReconstructionSpec:
    """Numerical face-state reconstruction configuration.

    ``order``: 1 = piecewise-constant; 2 = LSQ-MUSCL with ``limiter``.
    ``free_surface_aware``: when True, use the wet-dry-aware MUSCL
    variant (clamps ``h ≥ 0`` at faces, falls back to first order in
    dry cells).
    """
    order: int = 1
    limiter: str = "venkatakrishnan"
    free_surface_aware: bool = False


@dataclass
class DiffusionSpec:
    """Diffusion / viscous-flux configuration.

    ``enabled`` is honoured by the solver; it is auto-set to False
    in :meth:`NumericalSystemModel.from_system_model` when the
    SystemModel's ``diffusion_matrix`` is identically zero.  ``nu``
    overrides the value pulled from ``sm.parameter_values['nu']``.
    """
    enabled: bool = True
    scheme: str = "crank_nicolson"
    nu: Optional[float] = None


@dataclass
class RegularizationSpec:
    """Numerical regularization knobs.

    ``eigenvalue_eps`` is added to the diagonal of the local
    quasi-linear matrix before eigenvalue decomposition in the
    numerical-eigenvalue path; without it, dry/near-dry SWE cells
    yield ``A·n`` matrices with repeated zero eigenvalues and the
    LAPACK eigensolve can spike spurious large modes.
    """
    eigenvalue_eps: float = 1e-8


# ── The NSM itself ───────────────────────────────────────────────────


@dataclass
class NumericalSystemModel:
    """Numerical sibling of :class:`SystemModel`.

    Always constructed via :meth:`from_system_model`.  Direct
    instantiation is allowed (for tests, manual overrides) but the
    classmethod is the documented entry point.
    """

    sm: SystemModel
    riemann: Optional[Type[Any]] = None
    reconstruction: ReconstructionSpec = field(default_factory=ReconstructionSpec)
    diffusion: DiffusionSpec = field(default_factory=DiffusionSpec)
    regularization: RegularizationSpec = field(default_factory=RegularizationSpec)
    # Captured at promotion time so SDM-style derivative declarations
    # survive Model → SystemModel.  SystemModel itself has no back-
    # reference to its source Model and ``derivative_specs`` is a
    # StructuredDerivativeModel attribute — without this snapshot the
    # NSM would lose the lift-to-degree-2 signal for SDM models whose
    # ``D.dxx(Q.h)`` calls are substituted by Symbols before
    # ``SystemModel.from_model`` runs (so the aux_registry's
    # ``kind=="derivative"`` scan misses them).
    source_derivative_specs: Optional[list] = None
    # Co-running SystemModels whose derivative requirements must be
    # max'd into the LSQ stencil sizing.  For Chorin VAM the predictor
    # alone declares only first-order spatial derivatives, but the
    # pressure block's elliptic operator carries ``∂_xx P`` — the LSQ
    # stencil at every cell needs degree 2 to fit them.  Without
    # ``additional_systems`` the predictor-only NSM would silently
    # pick degree 1 and the pressure GMRES residual would zero the
    # curvature contribution (yesterday's root-cause for the dam-break
    # blow-up).  Solvers that compose sub-systems set this slot;
    # single-system solvers leave it empty.
    additional_systems: list = field(default_factory=list)

    # ── Constructors ──────────────────────────────────────────────

    @classmethod
    def from_system_model(
        cls,
        sm,
        *,
        riemann=None,
        reconstruction: Optional[ReconstructionSpec] = None,
        diffusion: Optional[DiffusionSpec] = None,
        regularization: Optional[RegularizationSpec] = None,
        additional_systems: Optional[list] = None,
    ) -> "NumericalSystemModel":
        """Build an NSM from a :class:`SystemModel` (or a :class:`Model`,
        auto-promoted via :meth:`SystemModel.from_model`).

        Defaults:
            - ``riemann`` → :class:`NonconservativeRusanov`
            - ``reconstruction`` → first-order constant
            - ``diffusion`` → enabled if the SystemModel carries a
              non-zero ``diffusion_matrix`` and a positive ``nu``;
              otherwise disabled.
            - ``regularization`` → ``eigenvalue_eps=1e-8``

        LSQ polynomial degree is **always** auto-derived (from
        ``sm.aux_registry`` plus any ``additional_systems``) — it is
        no longer a hand-adjustable knob.  Composite solvers pass
        ``additional_systems=[sm_press, sm_corr, ...]`` so the
        predictor's mesh stencil is large enough for the co-running
        sub-systems' derivatives.
        """
        source_specs = getattr(sm, "derivative_specs", None)
        if not isinstance(sm, SystemModel):
            sm = SystemModel.from_model(sm)
        if riemann is None:
            # Imported lazily — fvm/riemann_solvers.py imports from
            # zoomy_core.transformation.to_numpy which transitively
            # imports zoomy_core.model; doing it at module level here
            # creates a cycle on first import of the numerics package.
            from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
            riemann = NonconservativeRusanov
        if reconstruction is None:
            reconstruction = ReconstructionSpec()
        if diffusion is None:
            diffusion = DiffusionSpec(enabled=_diffusion_auto_enabled(sm))
        if regularization is None:
            regularization = RegularizationSpec()
        return cls(
            sm=sm,
            riemann=riemann,
            reconstruction=reconstruction,
            diffusion=diffusion,
            regularization=regularization,
            source_derivative_specs=(
                list(source_specs) if source_specs else None),
            additional_systems=list(additional_systems or []),
        )

    # ── LSQ-degree resolution ─────────────────────────────────────

    def resolved_lsq_degree(self) -> int:
        """Return the LSQ polynomial degree the mesh should use.

        Computed as the max spatial-derivative order across:

        - ``self.sm.aux_registry`` (every SystemModel carries this,
          populated by :meth:`SystemModel.from_model`),
        - every entry in ``self.additional_systems`` (composite
          sub-systems that share the same mesh; both
          ``aux_registry`` and ``derivative_specs`` are consulted —
          entries may be either Models or SystemModels), and
        - ``self.source_derivative_specs`` (captured at promotion
          when the source was a :class:`StructuredDerivativeModel`,
          whose ``D.dxx(...)`` calls are substituted by Symbols
          before ``SystemModel.from_model`` runs — so the source-side
          declaration is the only signal that survives).

        Falls back to 1 when no signal exists.  **Never user-set.**
        """
        candidates = [1, _lsq_degree_from_aux_registry(self.sm)]
        candidates.append(
            _lsq_degree_from_derivative_specs(self.source_derivative_specs))
        for sm in self.additional_systems:
            candidates.append(_lsq_degree_from_aux_registry(sm))
            candidates.append(_lsq_degree_from_derivative_specs(
                getattr(sm, "derivative_specs", None)))
        return max(candidates)

    # ── Numerics + runtime construction ───────────────────────────

    def build_numerics(self):
        """Instantiate the symbolic Riemann numerics over ``sm``."""
        return self.riemann(self.sm)

    def build_runtime_numpy(self):
        """Lambdify ``sm`` into a runtime model the NumPy solvers
        consume (callable ``.flux`` / ``.source`` / ``.eigenvalues`` /
        ``.boundary_conditions`` etc.)."""
        from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
        return NumpyRuntimeModel.from_system_model(self.sm)


# ── Helpers ─────────────────────────────────────────────────────────


def _diffusion_auto_enabled(sm) -> bool:
    """True iff the SystemModel carries a structurally non-zero
    ``diffusion_matrix``."""
    import sympy as sp
    A = getattr(sm, "diffusion_matrix", None)
    if A is None:
        return False
    try:
        flat = list(sp.flatten(A))
    except Exception:
        return False
    return any(sp.simplify(e) != 0 for e in flat)


def _lsq_degree_from_aux_registry(sm) -> int:
    """Max spatial-derivative order across ``sm.aux_registry``
    derivative entries.  Returns 1 when none are present (the LSQ
    stencil always at least supports first derivatives)."""
    registry = getattr(sm, "aux_registry", None) or []
    orders = [
        sum(int(k) for k in entry["multi_index"])
        for entry in registry
        if entry.get("kind") in ("derivative", "limited_derivative")
        and entry.get("multi_index") is not None
    ]
    return max(orders) if orders else 1


def _lsq_degree_from_derivative_specs(specs) -> int:
    """Max spatial-axes count across ``StructuredDerivativeModel``
    derivative specs.  Returns 1 when ``specs`` is empty."""
    if not specs:
        return 1
    orders = [sum(1 for a in spec.axes if a != "t") for spec in specs]
    return max(orders) if orders else 1
