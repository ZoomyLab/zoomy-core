"""Stress closures as composable, self-describing operations.

A :class:`Closure` is a single constitutive relation for one component of the
shear stress ``τ_xz`` — the *bulk* field, the *bottom* (bed) trace, or the
*surface* trace.  You compose a model's closure by listing the pieces::

    SME(level=2, closures=[Newtonian(), NavierSlip(), StressFree()])
    SME(level=2, closures=[KEpsilonViscosity(), RoughWall()])   # turbulent

Each closure is a small object that

* declares **which component** it closes — ``closes ∈ {"bulk","bottom","surface"}``;
* declares **which fields it needs** — ``requires`` (asserted against the model);
* **registers its parameters** — ``register(model)`` (register-or-query, with a
  default), so a closure is self-contained and never depends on a constant the
  model author forgot to declare;
* returns its **symbolic relation** — ``expression(s)`` over a full-access
  :class:`~zoomy_core.model.models.material.ClosureState` ``s`` (``s.u``, ``s.k``,
  ``s.dz(...)``, ``s.par.rho`` …), with NO side effects; the model performs the
  substitution.

Closures only *consume* the state a model already declares — they never mint a
new Q/Qaux field.  New physics (k, ε transport) is a new *equation* on a new
model class, not a closure (see the k–ε derivation notebook).

Class-level metadata (``closes``, ``requires``) is IMMUTABLE (str / tuple) so two
instances can never contaminate each other's defaults.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation
from zoomy_core.model.models.material import ClosureState   # the full-access state


def _solve_traction(P, s, nh):
    """Solve the boundary frame system ``t_α·(τ·n) = P_α`` for the shear-stress
    traces ``τ_iz`` (``i = 0…nh−1``), given the prescribed tangential tractions
    ``P`` and the local frame on the :class:`ClosureState` ``s``.

    Our stress tensor carries only the shear components ``τ_xz, τ_yz`` (the
    in-plane components were dropped by ``moment_scaling``), so
    ``(τ·n)_horiz_i = τ_iz n_z`` and ``(τ·n)_z = Σ_i τ_iz n_i``.  Returns the
    ``τ_iz`` as expressions in the OPAQUE frame slopes; under
    ``small_slope_scaling`` (slopes → 0, ``n → ẑ``) they collapse to
    ``τ_iz = P_i`` — the flat-boundary trace.  In 1-D, exactly
    ``τ_0 = P_0·(1+σ²)/(1−σ²)`` → ``P_0``."""
    nvec, tans = s.normal, s.tangents
    tau = [sp.Symbol(f"__tauz{i}") for i in range(nh)]
    nz = nvec[nh]
    taudotn = sp.Matrix([tau[i] * nz for i in range(nh)]
                        + [sum(tau[i] * nvec[i] for i in range(nh))])
    eqs = [sp.Eq((tans[a].T * taudotn)[0], P[a]) for a in range(nh)]
    sol = sp.solve(eqs, tau, dict=True)
    if not sol:
        raise ValueError("boundary traction frame solve failed for "
                         f"P={P} (nh={nh})")
    return [sp.simplify(sol[0][tau[i]]) for i in range(nh)]


def apply_stress_closures(closures, m, axes, state, horiz):
    """Inject stress closures at the projected momentum (shared by SME / VAM /
    the multilayer models) — frame-aware, dimension-agnostic.

    Boundary closures prescribe the traction in the local frame ``{n, t_α}``
    (:meth:`Closure.traction` → ``{"normal", "tangent"}``) and
    :func:`_solve_traction` recovers the per-direction shear traces; bulk
    closures stay diagonal (one constitutive :meth:`Closure.expression` per
    direction, ``s.u`` aliased to that direction's velocity).

    ``axes`` is a list of ``{"mx": eq, "tau": FieldHandle, "velname": "u"|"v"}``
    (one per horizontal momentum component) and ``state`` is
    ``state(at, *, alias=None, btag=None) -> ClosureState``.  Boundary traces
    (surface/bottom) are substituted BEFORE the bulk field.  Returns True iff a
    BULK closure was applied (else the caller leaves the bulk stress free /
    modally expanded)."""
    nh = len(axes)
    for c in closures:
        c.register(m)
    for c in closures:
        c.check(m)
    surface = [c for c in closures if c.closes == "surface"]
    bottom = [c for c in closures if c.closes == "bottom"]
    bulk = [c for c in closures if c.closes == "bulk"]
    for at, btag, pieces in ((1, "eta", surface), (0, "b", bottom)):
        if not pieces:
            continue
        P = [sp.S.Zero] * nh
        for c in pieces:
            tang = c.traction(state(at, btag=btag)).get("tangent") or []
            for i in range(min(nh, len(tang))):
                P[i] += tang[i]
        traces = _solve_traction(P, state(at, btag=btag), nh)
        for i, ax in enumerate(axes):
            ax["mx"].apply({ax["tau"].at(at): traces[i]})
    for c in bulk:                                # diagonal constitutive
        for ax in axes:
            ax["mx"].apply({ax["tau"].expr:
                            c.expression(state(None, alias={"u": ax["velname"]}))})
    return bool(bulk)


class Closure(Operation):
    """Base for a one-component stress closure (see module docstring).

    Subclasses set the immutable class attributes ``closes`` / ``requires`` and
    implement :meth:`expression`; :meth:`register` is optional."""

    closes = None          # "bulk" | "bottom" | "surface"   (immutable)
    requires = ()          # field names the closure consumes (immutable tuple)

    def __init__(self, name=None):
        super().__init__(name=name or type(self).__name__,
                         description=f"{type(self).__name__} ({self.closes})")

    def register(self, model):
        """Register-or-query the parameters this closure needs (override)."""

    def check(self, model):
        """Assert every required field is present in the model."""
        have = set(model.functions.keys())
        for f in self.requires:
            assert f in have, (
                f"{type(self).__name__} closure needs field {f!r}, which this "
                f"model does not define (has: {sorted(have)})")

    def expression(self, s):
        """Return the symbolic relation for this stress component (override)."""
        raise NotImplementedError

    def traction(self, s):
        """BOUNDARY traction in the local frame ``{n, t_α}``:
        ``{"normal": <n·σn or None>, "tangent": [t_α·σn, …]}`` (one entry per
        horizontal direction).  Default wraps the scalar :meth:`expression` as a
        single tangential component — the 1-D / diagonal fallback.  Boundary
        closures whose components COUPLE (friction magnitude ``|U|``, a
        prescribed wind/tension direction) override this to return the full
        vector; ``"normal"=None`` leaves the normal traction (pressure) to the
        hydrostatic step.  Only consulted for ``closes ∈ {"bottom","surface"}``
        under the frame-aware caller."""
        return {"normal": None, "tangent": [self.expression(s)]}


# ── bulk closures ──────────────────────────────────────────────────────────


class Newtonian(Closure):
    """Newtonian bulk stress  τ = ρ ν ∂_z u."""
    closes = "bulk"; requires = ("u",)

    def register(self, m):
        m.parameter("nu", 0.0)

    def expression(self, s):
        return s.par.rho * s.par.nu * s.dz(s.u)


class KEpsilonViscosity(Closure):
    """Turbulent bulk stress with the standard k–ε eddy viscosity
    ``τ = ρ ν_t ∂_z u``,  ``ν_t = C_μ k²/ε`` — reads the transported turbulence
    fields ``k`` and ``ε`` (only available on a k–ε model class).  The Galerkin
    projection is rational in ζ → build with ``quadrature_order > 0``.

    ``wall_floor`` (default on) adds the WALL-FUNCTION-consistent near-wall eddy
    viscosity ``ν_t,wall = C_μ^{1/4}√k · κ z_p`` (the log-layer ``κ u_⋆ z_p`` with
    ``u_⋆=C_μ^{1/4}√k``).  Why: ``C_μ k²/ε`` VANISHES at the wall, so in the moment
    projection it provides almost no damping of the velocity-shape moments
    (their stress-trace damping scales with the viscosity AT the wall — see
    :class:`QRViscosity`).  The mixing-length floor keeps ν_t non-zero there and
    restores that damping.  Set ``wall_floor=False`` for the bare ``C_μ k²/ε``."""
    closes = "bulk"; requires = ("u", "k", "varepsilon")

    def __init__(self, name=None, wall_floor=True):
        super().__init__(name=name)
        self.wall_floor = bool(wall_floor)

    def register(self, m):
        m.parameter("C_mu", 0.09)
        m.parameter("nu", 0.0)        # molecular part: ν_eff = ν + ν_t
        if self.wall_floor:
            m.parameter("kappa", 0.41); m.parameter("z_p", 0.1)

    def expression(self, s):
        nu_t = s.par.C_mu * s.k ** 2 / s.varepsilon
        if self.wall_floor:
            # offset log-layer mixing length ℓ=κ(z+z_p) (see QRViscosity)
            nu_t += (s.par.C_mu ** sp.Rational(1, 4) * sp.sqrt(s.k) * s.par.kappa
                     * (s.par.z_p + s.depth * s.zeta))
        return s.par.rho * (s.par.nu + nu_t) * s.dz(s.u)


class QRViscosity(Closure):
    """Turbulent bulk stress for the q–r (positivity-by-construction) k–ε model.

    Same eddy viscosity ``ν_t = C_μ k²/ε`` as :class:`KEpsilonViscosity`, but the
    transported state is ``sk = √k`` and ``se = √ε`` (Fe et al. 2009, k=q², ε=r²),
    so ``ν_t = C_μ sk⁴/se²`` — k=sk²≥0 and ε=se²≥0 hold by construction.  Reads
    the transported ``sk``, ``se`` fields (only on a q–r k–ε model class); the
    Galerkin projection is rational in ζ → build with ``quadrature_order > 0``."""
    closes = "bulk"; requires = ("u", "sk", "se")

    def __init__(self, name=None, wall_floor=True):
        super().__init__(name=name)
        self.wall_floor = bool(wall_floor)

    def register(self, m):
        m.parameter("C_mu", 0.09)
        m.parameter("nu", 0.0)        # molecular part: ν_eff = ν + ν_t
        m.parameter("eps_min", 0.0)   # realizability floor: ε → ε + eps_min
        if self.wall_floor:
            m.parameter("kappa", 0.41); m.parameter("z_p", 0.1)

    def expression(self, s):
        # ν_t = C_μ k²/ε = C_μ sk⁴/(se²+ε_min)  (sk=√k, se=√ε); the ε_min
        # realizability floor keeps the denominator off zero (see QRKESME).
        nu_t = s.par.C_mu * s.sk ** 4 / (s.se ** 2 + s.par.eps_min)
        if self.wall_floor:
            # WALL-FUNCTION-consistent near-wall eddy viscosity — the
            # Prandtl–Kolmogorov form with the log-layer mixing length, OFFSET by
            # z_p so it stays finite at the wall:  ℓ = κ(z + z_p) = κ(h ζ + z_p).
            #   ν_t,wall = C_μ^{1/4}√k · κ(z+z_p)   (= κ u_⋆ (z+z_p), u_⋆=C_μ^{1/4}√k).
            # SME is valid only ABOVE the wall layer, so the wall distance is
            # offset by z_p (the reference height): ℓ→κz_p at the bed (the
            # wall-function value) and GROWS with height like the physical κz —
            # the opposite of the bed-heavy ν_t=C_μ sk⁴/se² (which is inverted
            # because ε is under-resolved near the wall), so it straightens the
            # velocity profile.  C_μ sk⁴/se² also VANISHES at the wall, so this
            # term additionally restores the stress-trace damping of the velocity
            # moments (q₁'s eddy damping ∝ the √k gradient, else too weak).
            nu_t += (s.par.C_mu ** sp.Rational(1, 4) * s.sk * s.par.kappa
                     * (s.par.z_p + s.depth * s.zeta))
        return s.par.rho * (s.par.nu + nu_t) * s.dz(s.u)


class Bingham(Closure):
    """Regularized BINGHAM (viscoplastic) bulk stress
    ``τ = (ρν + τ_y/√((∂_z u)² + ε²))·∂_z u`` — rigid below the yield stress
    ``τ_y`` in the ``ε→0`` limit; the ``ε²`` floor keeps the root argument
    positive in floating point (the ``|γ̇|+ε`` form NaN-ed at plug formation).
    Needs ``tau_y`` (yield stress) and ``eps_reg`` (regularization scale).  The
    Galerkin projection is not analytically integrable → build the model with
    ``quadrature_order > 0`` (Gauss–Legendre)."""
    closes = "bulk"; requires = ("u",)

    def register(self, m):
        m.parameter("nu", 0.0); m.parameter("tau_y", 0.0); m.parameter("eps_reg", 1e-3)

    def expression(self, s):
        return (s.par.rho * s.par.nu
                + s.par.tau_y / sp.sqrt(s.dz(s.u) ** 2 + s.par.eps_reg ** 2)) * s.dz(s.u)


class ElderViscosity(Closure):
    """Elder / parabolic eddy viscosity  ν_t = κ u_⋆ h ζ(1−ζ)  (Elder 1959) —
    the classical ALGEBRAIC turbulence closure for free-surface flow:
    ``τ = ρ ν_t ∂_z u``.  Unlike k–ε it is a POLYNOMIAL in ζ, so the Galerkin
    projection closes analytically (no quadrature).  ``u_star`` is the friction
    velocity (a parameter here; set it from the bed law / RoughWall in practice)."""
    closes = "bulk"; requires = ("u",)

    def register(self, m):
        m.parameter("kappa", 0.41)
        m.parameter("u_star", 0.0)
        m.parameter("nu", 0.0)        # molecular part: ν_eff = ν + ν_t

    def expression(self, s):
        nu_t = s.par.kappa * s.par.u_star * s.depth * s.zeta * (1 - s.zeta)
        return s.par.rho * (s.par.nu + nu_t) * s.dz(s.u)


# ── bottom (bed) closures ──────────────────────────────────────────────────


class NavierSlip(Closure):
    """Navier slip at the bed  τ_b = λ_s · u_b  (linear; the "easy" closure)."""
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        m.parameter("lambda_s", 0.0)

    def expression(self, s):
        return s.par.lambda_s * s.u

    def traction(self, s):
        """Tangential traction ∝ slip velocity, per axis (linear — no coupling;
        the per-axis form is just the slip components in the frame).  Reduces to
        ``τ_iz|_b = λ_s u_i`` under small-slope scaling."""
        return {"normal": None,
                "tangent": [s.par.lambda_s * ut for ut in s.u_tangent]}


class RoughWall(Closure):
    """Turbulent ROUGH-WALL bed drag (OpenFOAM ``nutkRoughWallFunction`` family):

        τ_b = ρ C_f · u_b |u_b|,   C_f = (κ / ln(z_p / z_0))²,   z_0 = k_s/30,

    the physically-correct turbulent replacement for Navier slip.  ``u_b`` is the
    bed velocity trace; ``k_s`` the Nikuradse roughness, ``z_p`` the reference
    height.  The recovered friction velocity ``u_⋆ = √(C_f)|u_b|`` is what feeds
    the k–ε bed sources (Rastogi & Rodi 1978)."""
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        m.parameter("kappa", 0.41); m.parameter("k_s", 1e-3); m.parameter("z_p", 0.1)

    def expression(self, s):
        Cf = (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2
        return s.par.rho * Cf * s.u * sp.Abs(s.u)

    def traction(self, s):
        """Quadratic bed drag as a VECTOR traction opposing the slip velocity:
        ``τ_b,i = ρ C_f |U_t| u_{t,i}`` — the magnitude ``|U_t| = √(Σ u_{t,α}²)``
        couples the x/y components (a single scalar per axis would wrongly use
        ``|u_i|``).  Reduces to ``ρ C_f u_b|u_b|`` in 1-D under small-slope."""
        Cf = (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2
        speed = sp.sqrt(sum(ut ** 2 for ut in s.u_tangent))
        return {"normal": None,
                "tangent": [s.par.rho * Cf * speed * ut for ut in s.u_tangent]}


class WallFunctionBed(Closure):
    """k–ε ROUGH-WALL-FUNCTION bed stress — the standard turbulent momentum sink.

    Same log-law drag coefficient as :class:`RoughWall`,
    ``C_f = (κ / ln(z_p/z_0))²``, ``z_0 = k_s/30``, giving the wall shear
    ``τ_b,i = ρ C_f |U_p| U_{p,i}`` (= ρ u_*² in the flow direction, with
    ``u_* = κ|U_p|/ln(z_p/z_0)``).  The crucial difference: the reference
    velocity ``U_p`` is read at the first NEAR-WALL height ``ζ_p = z_p/h`` —
    NOT at the bed trace ``ζ=0``.

    Why this matters for a moment model: the bed trace ``u(0) = Σ û_i(−1)^i`` is
    where the log-law velocity is formally singular, and in a truncated moment
    expansion it COLLAPSES as the higher velocity moments grow (the bulk eddy
    viscosity ν_t correctly vanishes at the wall and cannot damp those moments),
    so ``RoughWall``'s ``u(0)``-based drag drops toward zero and the slope-driven
    flow accelerates without bound.  Evaluating ``U_p`` at ``ζ_p`` instead drains
    momentum at the physically-correct rate ρu_*² regardless of the profile — the
    wall-function momentum sink, consistent with the k/ε wall-function BCs.

    Needs ``k_s`` (Nikuradse roughness), ``z_p`` (reference height), ``kappa``."""
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        m.parameter("kappa", 0.41); m.parameter("k_s", 1e-3); m.parameter("z_p", 0.1)

    def _Cf(self, s):
        return (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2

    def expression(self, s):
        # scalar 1-D fallback: drag from the near-wall reference velocity U_p
        Up = s.velocity_at(s.par.z_p / s.depth)[0]
        return s.par.rho * self._Cf(s) * Up * sp.Abs(Up)

    def traction(self, s):
        """Wall-function bed drag as a VECTOR traction ``τ_b,i = ρ C_f |U_p| U_{p,i}``
        with ``U_p`` the tangential velocity at the near-wall height ``ζ_p=z_p/h``
        (couples x/y via ``|U_p|``).  Reduces to ``ρ C_f U_p|U_p|`` in 1-D."""
        Cf = self._Cf(s)
        ut = s.u_tangent_at(s.par.z_p / s.depth)
        speed = sp.sqrt(sum(u ** 2 for u in ut))
        return {"normal": None,
                "tangent": [s.par.rho * Cf * speed * u for u in ut]}


# ── depth-averaged SWE closures (bed friction + horizontal mixing) ──────────
# These close the DEPTH-AVERAGED shallow-water momentum, not the vertical
# moment hierarchy: a depth-averaged model has no resolved vertical profile,
# so its dissipation is (i) a bed-stress trace and (ii) a HORIZONTAL turbulent
# mixing τ_xx/τ_xy.  They are consumed by the SWE model's own closure hook
# (``apply_swe_closures``), which supplies the operator STRUCTURE (2-D vector
# source / rank-4 diffusion tensor) while the closure supplies the
# CONSTITUTIVE COEFFICIENT — same "compose the physics from named pieces"
# philosophy as the moment-hierarchy closures above.


class ManningFriction(Closure):
    """Manning bed friction as a BOTTOM (bed-trace) closure.

    Returns the friction RATE (per unit velocity)
    ``-g n² |u| / max(h, h_floor)^{1/3}`` so the depth-averaged momentum sink
    on each component is ``rate · u_i`` (the conservative
    ``-g n² u_i |u| / h^{1/3}``).  ``|u|`` is the full horizontal speed
    (``s.speed``) so x/y momenta couple correctly; ``h_floor`` keeps friction
    finite at the wet/dry interface.  No ``Piecewise``/``sign`` — the source
    Jacobian is auto-derived by sp.diff.
    """
    closes = "bottom"; requires = ("u", "h")

    def __init__(self, name=None, h_floor=0.0):
        super().__init__(name=name)
        self.h_floor = float(h_floor)

    def register(self, m):
        m.parameter("g", 9.81); m.parameter("n", 0.0)

    def expression(self, s):
        h_eff = sp.Max(s.h, sp.Float(self.h_floor)) if self.h_floor > 0 else s.h
        return -s.par.g * s.par.n ** 2 * s.speed / h_eff ** sp.Rational(1, 3)

    def traction(self, s):
        """Bed-shear-stress VECTOR for the moment hierarchy (SME / VAM bottom
        trace): ``τ_b,i = ρ g n² |U_t| u_{t,i} / h^{1/3}`` — the Manning analog
        of :meth:`RoughWall.traction`, with ``C_f = g n²/h^{1/3}`` and the speed
        ``|U_t| = √(Σ u_{t,α}²)`` from the frame tangents (couples x/y).  This is
        the depth-resolved STRESS form; the depth-averaged SWE path consumes the
        scalar :meth:`expression` (a rate) instead.  Uses the untraced column
        depth ``s.depth`` (NOT ``s.h`` — depth has no vertical profile to trace
        at the bed)."""
        h = s.depth
        h_eff = sp.Max(h, sp.Float(self.h_floor)) if self.h_floor > 0 else h
        Cf = s.par.g * s.par.n ** 2 / h_eff ** sp.Rational(1, 3)
        speed = sp.sqrt(sum(ut ** 2 for ut in s.u_tangent))
        return {"normal": None,
                "tangent": [s.par.rho * Cf * speed * ut for ut in s.u_tangent]}


class EddyViscosity(Closure):
    """Horizontal eddy viscosity as a HORIZONTAL-stress closure.

    The depth-averaged turbulent mixing ``∇·(ν h ∇u)`` (the τ_xx/τ_xy stress
    divergence — a horizontal diffusion, NOT the vertical τ_xz shear).
    Returns the isotropic kinematic eddy viscosity ``ν``; the SWE model builds
    the velocity-diffusion tensor (diagonal on momentum, chain-rule cross term
    on h) from it.  Vanishes at lake-at-rest (u=0) so well-balancing holds.
    """
    closes = "horizontal"; requires = ("u",)

    def register(self, m):
        m.parameter("nu", 0.0)

    def expression(self, s):
        return s.par.nu


# ── surface closures ───────────────────────────────────────────────────────


class StressFree(Closure):
    """Stress-free free surface  τ(1) = 0  (the usual top BC)."""
    closes = "surface"; requires = ()

    def expression(self, s):
        return sp.S.Zero

    def traction(self, s):
        """Zero tangential traction on every axis (the normal traction — the
        free-surface pressure — is left to the hydrostatic step, as before)."""
        return {"normal": None, "tangent": [sp.S.Zero for _ in s.tangents]}


# ── interface-transfer closures (multilayer) ───────────────────────────────
# The shared transfer velocity u* at an INTERNAL layer interface — the
# numerical scheme for the advection trace exchanged across the interface.
# A distinct closure family: ``expression(below, above, G)`` (the two one-sided
# modal interface traces + the interface mass flux G), not the stress ``(s)``.


class InterfaceFlux(Closure):
    """Base for the multilayer interface transfer-velocity scheme."""
    closes = "interface"; requires = ()

    def expression(self, below, above, G):    # noqa: D401 - distinct signature
        raise NotImplementedError


class MeanInterface(InterfaceFlux):
    """Central interface velocity  u* = (u_below + u_above) / 2."""

    def expression(self, below, above, G):
        return (below + above) / 2


class UpwindInterface(InterfaceFlux):
    """Donor interface velocity by the sign of the interface flux G
    (Hörnschemeyer Eq. 9): u* = u_below if G ≥ 0 else u_above."""

    def expression(self, below, above, G):
        return sp.Piecewise((below, G >= 0), (above, True))


def apply_layer_stress_closures(closures, m, axes, state, *, is_top, is_bottom):
    """Per-layer stress-closure injection for the multilayer models — frame-aware.

    Same contract as :func:`apply_stress_closures` (boundary closures prescribe
    the traction in the local frame {n, t_α}, bulk closures stay diagonal), but
    the *surface* closure fires only on the TOP layer and the *bottom* closure
    only on the BED layer (internal interfaces carry no dynamic stress BC — they
    use the InterfaceFlux family).  ``axes`` = ``[{"mx","tau","velname"}]`` per
    horizontal component, ``state(at, *, alias=None, btag=None)``.  No
    ``c.check(m)`` — the per-layer state maps the generic name ("u") to the
    layer field (u_ℓ) explicitly, so the model-namespace check does not apply.
    Returns True iff a BULK closure was applied."""
    nh = len(axes)
    stress = [c for c in closures if c.closes in ("bulk", "bottom", "surface")]
    for c in stress:
        c.register(m)
    surface = [c for c in stress if c.closes == "surface"] if is_top else []
    bottom = [c for c in stress if c.closes == "bottom"] if is_bottom else []
    bulk = [c for c in stress if c.closes == "bulk"]
    for at, btag, pieces in ((1, "eta", surface), (0, "b", bottom)):
        if not pieces:
            continue
        P = [sp.S.Zero] * nh
        for c in pieces:
            tang = c.traction(state(at, btag=btag)).get("tangent") or []
            for i in range(min(nh, len(tang))):
                P[i] += tang[i]
        traces = _solve_traction(P, state(at, btag=btag), nh)
        for i, ax in enumerate(axes):
            ax["mx"].apply({ax["tau"].at(at): traces[i]})
    for c in bulk:
        for ax in axes:
            ax["mx"].apply({ax["tau"].expr:
                            c.expression(state(None, alias={"u": ax["velname"]}))})
    return bool(bulk)


def interface_closure(closures):
    """Return the InterfaceFlux closure in ``closures`` (or None)."""
    return next((c for c in (closures or []) if c.closes == "interface"), None)


# ── depth-averaged SWE closure consumption ──────────────────────────────────
# SWE states its operators directly (no moment derivation that calls
# apply_stress_closures), so it consumes closures through this hook instead:
# the closure supplies the constitutive COEFFICIENT, the model the operator
# STRUCTURE (2-D vector source / rank-4 velocity-diffusion tensor).


class _SWEField:
    """Minimal FunctionFamily stand-in for a depth-averaged SWE field: a single
    bulk expression with no vertical-trace distinction (depth-averaged)."""

    def __init__(self, expr):
        self.expr = expr

    def at(self, loc):              # bottom/surface trace == the depth average
        return self.expr


def swe_closure_state(model):
    """Build a :class:`ClosureState` over a depth-averaged SWE model so its
    closures can read ``s.u``/``s.w``/``s.h``/``s.speed`` and ``s.par.*``.
    Velocity is desingularised through the model's ``hinv`` aux (``u=hu·hinv``);
    ``speed`` is the floored horizontal magnitude (couples x/y friction)."""
    v = model.variables
    a = model.aux_variables
    hinv = a.hinv
    u = v.hu * hinv
    w = v.hv * hinv if hasattr(v, "hv") else sp.S.Zero
    speed = sp.sqrt(u * u + w * w + 1e-12)
    fields = {"u": _SWEField(u), "w": _SWEField(w),
              "h": _SWEField(v.h), "speed": _SWEField(speed)}
    return ClosureState(fields, params=model.parameters)


__all__ = ["Closure", "ClosureState", "apply_stress_closures",
           "apply_layer_stress_closures", "interface_closure",
           "Newtonian", "KEpsilonViscosity", "QRViscosity", "NavierSlip", "RoughWall",
           "WallFunctionBed",
           "StressFree", "InterfaceFlux", "MeanInterface", "UpwindInterface",
           "ManningFriction", "EddyViscosity", "swe_closure_state"]
