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
    lsq_degree: Optional[int] = None
    # Captured at promotion time so SDM-style derivative declarations
    # survive Model → SystemModel.  SystemModel itself has no back-
    # reference to its source Model and ``derivative_specs`` is a
    # StructuredDerivativeModel attribute — without this snapshot the
    # NSM would lose the lift-to-degree-2 signal for SDM models whose
    # ``D.dxx(Q.h)`` calls are substituted by Symbols before
    # ``SystemModel.from_model`` runs (so the aux_registry's
    # ``kind=="derivative"`` scan misses them).
    source_derivative_specs: Optional[list] = None

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
        lsq_degree: Optional[int] = None,
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
            - ``lsq_degree`` → auto from ``sm.aux_registry`` (or, when
              the source was a :class:`StructuredDerivativeModel`, from
              its captured ``derivative_specs``).
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
            lsq_degree=lsq_degree,
            source_derivative_specs=(
                list(source_specs) if source_specs else None),
        )

    # ── LSQ-degree resolution ─────────────────────────────────────

    def resolved_lsq_degree(self) -> int:
        """Return the LSQ polynomial degree the mesh should use.

        Order of precedence:

        1. ``self.lsq_degree`` (explicit override) wins.
        2. The maximum spatial-derivative order across
           ``sm.aux_registry`` entries of ``kind == "derivative"``.
           ``aux_registry`` is populated by
           :meth:`SystemModel.from_model` for *every* SystemModel
           (more general than ``derivative_specs`` which only
           exists on :class:`StructuredDerivativeModel`).
        3. Fallback: 1.
        """
        if self.lsq_degree is not None:
            return self.lsq_degree
        registry = getattr(self.sm, "aux_registry", None) or []
        spatial_orders = [
            sum(int(k) for k in entry["multi_index"])
            for entry in registry
            if entry.get("kind") == "derivative"
            and entry.get("multi_index") is not None
        ]
        if spatial_orders:
            return max(spatial_orders)
        # Fallback to ``source_derivative_specs`` captured at
        # promotion (StructuredDerivativeModel path).  The SDM
        # substitutes ``D.dxx(Q.h)`` calls by Symbols *before* the
        # SystemModel is built, so the aux_registry derivative scan
        # never sees them — the source-side declaration is the only
        # signal that survives.
        specs = self.source_derivative_specs or []
        if specs:
            spatial = [
                sum(1 for a in spec.axes if a != "t") for spec in specs
            ]
            if spatial:
                return max(spatial) or 1
        return 1

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
