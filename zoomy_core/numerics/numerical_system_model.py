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

from zoomy_core.systemmodel.system_model import SystemModel

# Operator arrays a code printer lowers — the complete set of symbolic-PDE
# operators carried on a (Numerical)SystemModel.  ``quasilinear_matrix`` is a
# lazy cached property derived from ``flux``/``nonconservative_matrix``; it is
# guarded (read) but never assigned back to.
_OPERATOR_ATTRS = (
    "flux", "hydrostatic_pressure", "nonconservative_matrix", "source",
    "source_explicit", "mass_matrix", "eigenvalues",
    "source_jacobian_wrt_variables", "source_jacobian_wrt_aux_variables",
    "diffusion_matrix", "update_variables", "update_aux_variables",
    "state_update",
)


def _op_flat(arr):
    """Flat list of scalar entries of an operator array (ZArray / MatrixBase /
    nested list), or ``[]`` for non-array operands."""
    import sympy as sp
    if arr is None:
        return []
    if hasattr(arr, "_array"):                 # ZArray — flat backing store
        return list(arr._array)
    if isinstance(arr, sp.Basic):
        try:
            return list(sp.flatten(arr))
        except Exception:
            return [arr]
    if isinstance(arr, (list, tuple)):
        return list(sp.flatten(arr))
    return []


def _resolve_boundary_traces(nsm) -> None:
    """REQ-130 — collapse every bed/surface-trace ``Subs(f(ζ), ζ, 0|1)`` node
    that survived symbolic lowering into its concrete ``Σ αₖ φₖ(0|1)`` value on
    ALL operator arrays, then GUARD that none remain.

    A boundary trace (bottom KBC ``w(0)=u(0)·∂ₓb``, bed traction, …) is a
    ``Subs`` of a basis-expanded vertical profile at ζ∈{0,1}; ``.doit()`` turns
    it into a plain ``Σ αₖ φₖ(0|1)`` with no residual ζ.  ``sp.lambdify``
    evaluates ``Subs`` transparently, so the NumPy path never noticed a leftover
    node — but every symbolic-tree printer (generic_c / amrex / ufl) emits
    ``Subs``/``zeta`` into code that will not compile.  Resolving here — the
    single ``SystemModel → NumericalSystemModel`` lowering seam every printer
    routes through — makes this backend-agnostic rather than a per-printer
    workaround, and the guard mirrors the unresolved-Galerkin-integral raise in
    ``system_extract`` (``model/derivation/system_extract.py``): a lazy node
    that cannot be lowered must fail LOUD here, never leak to a printer."""
    import sympy as sp
    is_subs = lambda n: isinstance(n, sp.Subs)
    resolve = lambda n: n.doit()

    for attr in _OPERATOR_ATTRS:
        arr = getattr(nsm, attr, None)
        if arr is None or not hasattr(arr, "replace"):
            continue
        if any(sp.sympify(e).has(sp.Subs) for e in _op_flat(arr)):
            setattr(nsm, attr, arr.replace(is_subs, resolve))

    # Guard (defense-in-depth, mirrors system_extract's unresolved-integral
    # raise): after the resolve NO ``Subs`` atom and NO free vertical ζ symbol
    # may remain in ANY lowered operator (including the derived
    # ``quasilinear_matrix``).  A leftover means a boundary trace was genuinely
    # unresolvable (e.g. free ζ after ``.doit()``) — no symbolic-tree printer
    # can lower it, so fail here rather than emit uncompilable code.
    for attr in _OPERATOR_ATTRS + ("quasilinear_matrix",):
        arr = getattr(nsm, attr, None)
        for e in _op_flat(arr):
            e = sp.sympify(e)
            if e.atoms(sp.Subs):
                raise ValueError(
                    f"REQ-130: unresolved Subs boundary trace survives in "
                    f"operator {attr!r}: {e} — a bed/surface trace could not be "
                    "collapsed by .doit() (free-ζ profile?).  A symbolic-tree "
                    "printer (generic_c / amrex / ufl) cannot lower Subs; fix "
                    "the trace at its birthplace in the model derivation.")
            stray_zeta = {s for s in e.free_symbols
                          if getattr(s, "name", None) == "zeta"}
            if stray_zeta:
                raise ValueError(
                    f"REQ-130: free vertical symbol {stray_zeta} survives in "
                    f"operator {attr!r}: {e} — the σ-reference coordinate ζ must "
                    "not reach the lowered operators (it is an integration / "
                    "evaluation dummy).  Fix at the model-derivation birthplace.")


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


def _linearize_source(sm):
    """Rewrite ``sm.source`` into the point-implicit (linearized) update

        S_lin(Q, dt) = (I − dt·J)⁻¹ · S,

    with the **consistent** source jacobian

        J = ∂S/∂Q + ∂S/∂aux · ∂aux/∂Q
          = source_jacobian_wrt_variables
            + source_jacobian_wrt_aux_variables · (∂ update_aux_variables / ∂Q).

    The per-cell linear solve is emitted as one opaque
    :class:`zoomy_core.model.kernel_functions.solve` call per row (``A = I −
    dt·J``, ``b = S``); every backend then consumes the transformed ``source``
    as an ordinary source, the only backend-specific atom being ``solve`` →
    ``np``/``jnp.linalg.solve`` / Eigen.  ``dt`` is the canonical ``DT_SYMBOL``
    the solvers already thread into the per-cell update kernels.

    A point-implicit step treats the *local* algebraic source dependence
    implicitly; spatial-derivative aux (``compute_derivative`` rows, which carry
    no ``update_aux_variables`` formula) stay explicit — their ∂aux/∂Q is zero
    here, which is correct (a non-local derivative cannot be inverted per cell).

    Requires a square system (``n_equations == n_state``).  Mutates ``sm`` in
    place; the now-stale channeled source jacobians are irrelevant downstream
    (the jacobian is baked into ``S_lin`` and consumed explicitly)."""
    import sympy as sp
    from zoomy_core.misc.misc import ZArray
    from zoomy_core.model.kernel_functions import solve as _solve
    from zoomy_core.transformation.to_numpy import DT_SYMBOL

    n = sm.n_equations
    if sm.n_state != n:
        raise ValueError(
            "source_treatment='linearized' requires a square system "
            f"(n_equations == n_state); got n_equations={n}, "
            f"n_state={sm.n_state}.")
    if sm.source is None:
        return
    state = list(sm.state)
    aux = list(sm.aux_state)

    Jvar = sp.Matrix(
        n, n, lambda i, j: sp.sympify(sm.source_jacobian_wrt_variables[i, j]))
    # ∂aux/∂Q from the local aux-update map; None / no aux ⇒ aux is independent
    # of Q in the point-implicit step ⇒ the channel contributes nothing.
    uav = getattr(sm, "update_aux_variables", None)
    if uav is not None and aux:
        Jaux = sp.Matrix(
            n, len(aux),
            lambda i, j: sp.sympify(sm.source_jacobian_wrt_aux_variables[i, j]))
        Jauxupd = sp.zeros(len(aux), n)
        nrows = uav.shape[0]
        for k in range(min(nrows, len(aux))):
            fk = sp.sympify(uav[k, 0])
            for j in range(n):
                Jauxupd[k, j] = sp.diff(fk, state[j])
        J = Jvar + Jaux * Jauxupd
    else:
        J = Jvar

    A = sp.eye(n) - DT_SYMBOL * J
    A_flat = [A[i, j] for i in range(n) for j in range(n)]
    b_flat = [sp.sympify(sm.source[i, 0]) for i in range(n)]
    rows = [[_solve(sp.Integer(i), *A_flat, *b_flat)] for i in range(n)]
    sm.source = ZArray(rows)


# ── The NSM itself ───────────────────────────────────────────────────


@dataclass
class NumericalSystemModel(SystemModel):
    """Numerical sibling of :class:`SystemModel` — now a SUBCLASS of it.

    The NSM IS-A SystemModel: it inherits the entire physics/codegen
    surface (flux, source, eigenvalues, the source jacobians,
    reconstruction_variables / state_from_reconstruction,
    interpolate_to_3d / project_from_3d, update_variables /
    update_aux_variables, the lambdified BC *kernel*
    ``boundary_conditions``, state / aux_state / parameters, the
    shape properties n_equations / n_state / dimension /
    equation_to_state_index, …) and ADDS the numerical knobs below.

    There is no ``sm`` field anymore — the ``sm`` property returns
    ``self`` so legacy ``nsm.sm.<x>`` accesses keep resolving, and the
    BC kernel/container split is preserved: ``nsm.boundary_conditions``
    is the inherited lambdified BC kernel; the ``BoundaryConditions``
    container stays at ``nsm._bc_source`` (== ``nsm.sm._bc_source``).

    Always constructed via :meth:`from_system_model`, which PROMOTES a
    frozen SystemModel instance in place (re-class + attach numerics)
    so ``nsm is sm`` and the SystemModel's full state is preserved with
    no copy.  Direct dataclass instantiation is possible (it accepts the
    full SystemModel field signature followed by the numerics fields)
    but the classmethod is the documented entry point.
    """

    riemann: Optional[Type[Any]] = None
    reconstruction: ReconstructionSpec = field(default_factory=ReconstructionSpec)
    diffusion: DiffusionSpec = field(default_factory=DiffusionSpec)
    # Numerical eigenvalue regularization — added to the diagonal of the local
    # quasi-linear matrix before the LAPACK eigensolve in the numerical-wave-
    # speed path (without it, dry/near-dry SWE cells yield ``A·n`` matrices with
    # repeated zero eigenvalues and the eigensolve can spike spurious large
    # modes).  This is solver config, not a system operation, so it lives as a
    # slim NSM attribute rather than driving a ``.apply`` op.
    eigenvalue_eps: float = 1e-8
    # Opt-in system operations run on this NSM by :meth:`derive`, in addition to
    # :meth:`default_operations`.  Each entry is a reusable ``op(sm)`` callable
    # from :mod:`zoomy_core.systemmodel.operations` (e.g.
    # ``desingularize_hinv()``); the old ``desingularize=True`` regularization
    # opt-in is now ``extra_operations=[desingularize_hinv()]``.
    extra_operations: list = field(default_factory=list)
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
    # Indices (into ``sm.state``) of state rows that the Audusse HR /
    # ``PositiveRusanov``-style Riemann should rescale by ``h*/h`` at
    # face states.  Default (``None``) lets the Riemann class pick —
    # which excludes only ``h`` and ``b``.  For VAM-Chorin we must
    # exclude pressure modes ``P_k`` too (they are amplitudes, not
    # depth-scaled momenta).  ``ChorinSplitVAMSolverJax`` computes the
    # right list at setup time and passes it here.
    scaled_q_indices: Optional[list] = None
    # How the NSM supplies the source to every backend solver:
    #   "explicit"   — raw ``S`` (default, unchanged behaviour);
    #   "linearized" — the point-implicit update ``S_lin = (I − dt·J)⁻¹ S`` with
    #                  the consistent ``J = J_var + J_aux · (∂aux/∂Q)``.
    # The transform happens ONCE at construction (``_linearize_source``); every
    # backend then consumes ``source`` as an ordinary source, the only new atom
    # being the per-cell opaque ``solve``.  This is an NSM-level knob, NOT a
    # per-backend solver mode.
    source_treatment: str = "explicit"

    # ── SystemModel identity bridge ───────────────────────────────
    @property
    def sm(self) -> "SystemModel":
        """Return ``self``.

        The NSM *is* a SystemModel (it subclasses it), so the entire
        physics/codegen interface is INHERITED — no allowlist, no
        ``__getattr__`` delegation, and a bogus ``nsm.__typo__`` raises
        ``AttributeError`` for free.  This property keeps the historical
        ``nsm.sm.<x>`` indirection resolving and preserves the BC
        kernel/container split: ``nsm.boundary_conditions`` is the
        inherited lambdified BC *kernel* (a sympy ``Function``), while
        the ``BoundaryConditions`` *container* lives at
        ``nsm._bc_source`` (== ``nsm.sm._bc_source``)."""
        return self

    # ── Derivation pipeline ───────────────────────────────────────

    def _is_transport_system(self) -> bool:
        """True iff this NSM is a hyperbolic TRANSPORT system (carries a
        structurally non-zero ``flux`` or ``nonconservative_matrix``).

        The Chorin split produces sub-systems that are NOT transport:

        * the pressure stage (``source_only`` elliptic block) has
          ``flux == NCP == 0`` — it is a Poisson/constraint solve, and
        * the corrector is update-only (``flux == NCP == source == 0``;
          everything rides on ``update_variables``).

        Neither carries wavespeeds, so the wet/dry desingularization /
        eigenvalue gate is meaningless there — worse, registering the
        ``hinv`` aux on the pressure stage adds a ``dt``-bearing
        ``update_aux_variables`` row that collides with the split solver's
        own ``dt`` parameter (``duplicate argument 'dt'``).  Gating
        :meth:`default_operations` on this predicate keeps the default ops
        on the predictor (a real flux system) and off the elliptic /
        projection sub-systems, even though they share the same state
        vector (so an ``h``-presence test alone would not separate them).
        """
        import sympy as sp
        for attr in ("flux", "nonconservative_matrix"):
            M = getattr(self, attr, None)
            if M is None:
                continue
            if any(sp.sympify(e) != 0 for e in sp.flatten(M)):
                return True
        return False

    def default_operations(self) -> list:
        """Return the system operations applied to EVERY NSM by
        :meth:`derive`, before :attr:`extra_operations`.

        Shallow-water transport systems (state carrying a depth ``h``) get the
        wet/dry-safe defaults — the KP ``1/h`` desingularization plus the dry
        eigenvalue gate — so every depth-based FVM run is positivity-robust at
        the wet/dry front without a per-case opt-in.  The list is gated on:

        * :meth:`_is_transport_system` — the Chorin pressure/corrector
          sub-systems share the same ``h``-bearing state but are NOT transport
          (no flux/NCP); they must stay clean (see :meth:`_is_transport_system`),
          and
        * the presence of a depth state ``h`` — non-shallow-water systems
          (no ``h``) get nothing.

        Non-default behaviour still rides on opt-in :attr:`extra_operations`."""
        if not self._is_transport_system():
            return []
        if not any(str(s) == "h" for s in self.state):
            return []
        from zoomy_core.systemmodel.operations import (
            desingularize_hinv, gate_eigenvalues_dry)
        return [desingularize_hinv(), gate_eigenvalues_dry()]

    def derive(self) -> "NumericalSystemModel":
        """Run the operation pipeline on this NSM in place and return ``self``.

        Applies ``self.default_operations() + self.extra_operations``, each a
        reusable ``op(sm)`` callable, through the inherited
        :meth:`SystemModel.apply` hook (so every step records its own history
        entry and owns its derived-operator refresh)."""
        for op in self.default_operations() + list(self.extra_operations):
            self.apply(op)
        return self

    # ── Constructors ──────────────────────────────────────────────

    @classmethod
    def from_system_model(
        cls,
        sm,
        *,
        riemann=None,
        reconstruction: Optional[ReconstructionSpec] = None,
        diffusion: Optional[DiffusionSpec] = None,
        eigenvalue_eps: float = 1e-8,
        extra_operations: Optional[list] = None,
        additional_systems: Optional[list] = None,
        scaled_q_indices: Optional[list] = None,
        source_treatment: str = "explicit",
    ) -> "NumericalSystemModel":
        """Build an NSM from a :class:`SystemModel` (or a :class:`Model`,
        auto-promoted via :meth:`SystemModel.from_model`).

        Defaults:
            - ``riemann`` → :class:`NonconservativeRusanov`
            - ``reconstruction`` → first-order constant
            - ``diffusion`` → enabled if the SystemModel carries a
              non-zero ``diffusion_matrix`` and a positive ``nu``;
              otherwise disabled.
            - ``eigenvalue_eps`` → ``1e-8``
            - ``extra_operations`` → ``[]`` (opt-in system ops run by
              :meth:`derive`, e.g. ``[desingularize_hinv()]``)

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
            source_specs = getattr(sm, "derivative_specs", source_specs)
        _assert_bc_kernels_match_state(sm)
        for sub in (additional_systems or []):
            if isinstance(sub, SystemModel):
                _assert_bc_kernels_match_state(sub)
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
        if source_treatment not in ("explicit", "linearized"):
            raise ValueError(
                "source_treatment must be 'explicit' or 'linearized'; "
                f"got {source_treatment!r}.")
        if source_treatment == "linearized":
            _linearize_source(sm)

        # The NSM IS-A SystemModel, so PROMOTE the frozen ``sm`` instance
        # in place — re-class it to the NSM and attach the numerics
        # fields — rather than copy its field values into a separate
        # object.  Both classes are plain (``__dict__``-backed) dataclasses
        # so the re-class is layout-compatible, and the promotion keeps the
        # SystemModel's full state (every dataclass field PLUS the dynamic
        # attributes ``_bc_source`` / ``_aux_bc_source`` (the
        # BoundaryConditions CONTAINER — REQ-16, reached as
        # ``nsm._bc_source``), ``aux_registry`` (LSQ-degree sizing),
        # ``positive_state``, ``derivative_specs``, the lazy
        # ``_quasilinear_matrix`` cache, …) intact with NO copy.  Crucially
        # it preserves object identity: ``nsm is sm`` and ``nsm.sm is nsm``,
        # so callers that built the SystemModel, then keep mutating it
        # (e.g. ``sm.initial_conditions = …`` between successive solves)
        # see those mutations through the NSM — the faithful
        # ``SystemModel → NumericalSystemModel`` pipeline step, not a
        # snapshot.
        sm.__class__ = cls
        sm.riemann = riemann
        sm.reconstruction = reconstruction
        sm.diffusion = diffusion
        sm.eigenvalue_eps = eigenvalue_eps
        sm.extra_operations = list(extra_operations or [])
        sm.source_derivative_specs = (
            list(source_specs) if source_specs else None)
        sm.additional_systems = list(additional_systems or [])
        sm.scaled_q_indices = (
            list(scaled_q_indices) if scaled_q_indices is not None
            else None)
        sm.source_treatment = source_treatment
        # Run the operation pipeline (default_operations() + extra_operations)
        # on the now-promoted NSM.  Phase A1: default_operations() is empty, so
        # this is a no-op unless the caller opted in via ``extra_operations``.
        sm.derive()
        # REQ-130: collapse any surviving bed/surface-trace ``Subs(f(ζ),ζ,0|1)``
        # nodes on every operator array — the single backend-agnostic lowering
        # seam — and GUARD that none (nor a free ζ) remain, so no symbolic-tree
        # printer ever receives a node it cannot compile.
        _resolve_boundary_traces(sm)
        return sm

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
        """Instantiate the symbolic Riemann numerics over ``sm``.

        Threads ``self.scaled_q_indices`` into the Riemann constructor
        when set — the Audusse-HR variants (``PositiveRusanov`` family)
        use it to control which state rows are rescaled by ``h*/h`` at
        face states.  Default (``None``) lets the Riemann class fall
        back to its own heuristic (excluding only ``h`` and ``b``)."""
        if self.scaled_q_indices is not None:
            return self.riemann(self.sm,
                                scaled_q_indices=self.scaled_q_indices)
        return self.riemann(self.sm)

    def build_runtime_numpy(self):
        """Lambdify ``sm`` into a runtime model the NumPy solvers
        consume (callable ``.flux`` / ``.source`` / ``.eigenvalues`` /
        ``.boundary_conditions`` etc.)."""
        from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
        return NumpyRuntimeModel.from_system_model(self.sm)


# ── Canonical coercion (the printers' front door) ───────────────────


def to_numerical_system_model(obj) -> "NumericalSystemModel":
    """Coerce ``obj`` to a :class:`NumericalSystemModel`.

    The single front door every CORE code printer routes its entry
    through, so a printer accepts a :class:`Model`, a
    :class:`SystemModel`, OR an already-built NSM and always operates on
    an NSM:

    - already a :class:`NumericalSystemModel` → returned unchanged;
    - a :class:`SystemModel` → promoted in place via
      :meth:`NumericalSystemModel.from_system_model`;
    - a :class:`zoomy_core.model` ``Model`` (exposes ``.system_model``)
      → its SystemModel is promoted;
    - anything else → :class:`TypeError`.

    The NSM-first check matters because ``NumericalSystemModel`` *is* a
    ``SystemModel`` (subclass) — an NSM must short-circuit before the
    SystemModel branch so it is never re-promoted.
    """
    if isinstance(obj, NumericalSystemModel):
        return obj
    if isinstance(obj, SystemModel):
        return NumericalSystemModel.from_system_model(obj)
    sm = getattr(obj, "system_model", None)
    if sm is not None:
        return NumericalSystemModel.from_system_model(sm)
    raise TypeError(
        f"to_numerical_system_model: cannot coerce {type(obj).__name__!r} "
        "to a NumericalSystemModel — expected a NumericalSystemModel, a "
        "SystemModel, or a Model exposing `.system_model`."
    )


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


def _assert_bc_kernels_match_state(sm) -> None:
    """Fail loudly if any of the SystemModel's BC kernels was built
    against state symbols that no longer match ``sm.state``.

    This catches the silent-wrong-result trap where
    ``sm.change_state_variables(...)`` remaps state in
    ``flux`` / ``source`` / ``NCP`` / ``mass_matrix`` but the
    BC kernel ``Function`` objects (signature + definition) keep
    their original state Zstruct.  The runtime then evaluates the
    BC with a Q vector whose slot ``i`` carries the NEW symbol value
    but the BC body computes against the OLD symbol — Lambda inflow
    values land on the wrong scaled state, mass drifts, etc.

    Raised at NSM construction so the trap can't reach a solver.

    Checks ``sm.boundary_conditions``, ``sm.boundary_gradients`` and
    ``sm.aux_boundary_conditions`` when present.  Each is a
    ``zoomy_core.model.basefunction.Function`` exposing
    ``.args.variables`` (a Zstruct whose values are the state symbols
    bound at kernel-construction time)."""
    expected = list(sm.state)
    for attr in ("boundary_conditions",
                 "boundary_gradients",
                 "aux_boundary_conditions"):
        kernel = getattr(sm, attr, None)
        if kernel is None:
            continue
        args = getattr(kernel, "args", None)
        if args is None or not hasattr(args, "contains") \
                or not args.contains("variables"):
            continue
        bound = list(args.variables.values())
        if bound != expected:
            raise ValueError(
                f"SystemModel.{attr} is stale: its state signature\n"
                f"  {[str(s) for s in bound]}\n"
                f"does not match sm.state\n"
                f"  {[str(s) for s in expected]}.\n"
                "This typically means sm.change_state_variables(...) "
                "remapped the state symbols in flux/source/NCP but did "
                "NOT rebuild the BC kernel.  The BC body still "
                "references the original state symbols and will read "
                "the wrong slot of Q at runtime — Lambda-inflow values "
                "land on the wrong (scaled) state, mass drifts.\n\n"
                "Either rebuild the BC kernel against the new state "
                "(see vam_chorin_bump_8state_chain.py for the canonical "
                "pattern) — or, once SystemModel.change_state_variables "
                "is fixed upstream, the BC remap will happen "
                "automatically and this check becomes a no-op."
            )


def _lsq_degree_from_derivative_specs(specs) -> int:
    """Max spatial-axes count across ``StructuredDerivativeModel``
    derivative specs.  Returns 1 when ``specs`` is empty."""
    if not specs:
        return 1
    orders = [sum(1 for a in spec.axes if a != "t") for spec in specs]
    return max(orders) if orders else 1
