"""Balance-equation blueprints — composable templates that declare the state /
closure variables they introduce and add their residual to the model.

A blueprint is the equation-level analogue of a :mod:`closures` `Closure`: you
build a model by *listing* its balances, and each balance owns the variables it
introduces (state → Q, closure vars → Qaux if unsubstituted)::

    m = Model(coords=(t, x, z), parameters={...})
    m.declare_state(h)                 # geometry
    m.add_equation("bottom", d.t(b))
    m.add_equation(Mass(m))            # registers u, w (state); ∂_x u + ∂_z w
    m.add_equation(Momentum(m))        # registers p (state), tau_xz (closure)

Only the model class mints new state; closures merely consume it (see
``closures.py``).  These templates reproduce the hand-built SME 3-D balances
exactly (term-by-term), so existing models can adopt them with no change to the
derived system.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d

# horizontal coord symbol → its directional derivative operator, velocity-field
# name, and coordinate name string
_DERIV = {C.x: d.x, C.y: d.y}
_HNAME = {C.x: "u", C.y: "v"}
_CNAME = {C.x: "x", C.y: "y"}


class Equation:
    """Base balance blueprint.  ``add_to(model)`` declares variables and adds
    the residual; ``model.add_equation(BlueprintInstance)`` dispatches here."""

    _is_blueprint = True

    def __init__(self, model=None):
        self.model = model

    def add_to(self, model):
        raise NotImplementedError


class Mass(Equation):
    """Incompressible 3-D mass balance ``Σ_d ∂_d u_d + ∂_z w = 0``.

    Registers the horizontal velocity(ies) and the vertical velocity ``w`` as
    state.  For the canonical SME shape ``coords=(t, x, z)`` this is
    ``∂_x u + ∂_z w``.  ``suffix`` scopes the field names for a multilayer
    column (e.g. ``"_1"`` → ``u_1, w_1``)."""

    def __init__(self, model=None, suffix=""):
        super().__init__(model)
        self.suffix = suffix

    def add_to(self, model):
        coords = model.coords
        horiz = coords[1:-1]                       # (x,) or (x, y)
        s = self.suffix
        uvel = [sp.Function(_HNAME[xd] + s, real=True)(*coords) for xd in horiz]
        w = sp.Function("w" + s, real=True)(*coords)
        model.declare_state(*uvel, w)
        expr = sum(_DERIV[xd](uvel[i]) for i, xd in enumerate(horiz)) + d.z(w)
        model.add_equation("mass", expr)
        self.uvel, self.w = uvel, w
        return model._equations["mass"]


class Momentum(Equation):
    """GENERAL 3-D momentum balance — the FULL stress tensor, no shallow
    simplification baked in:

        ∂_t u_i + Σ_j ∂_j(u_i u_j) + ∂_i p/ρ − Σ_j ∂_j τ_ij /ρ − g e_i = 0

    The complete symmetric stress tensor ``τ_ij`` is declared as closure
    variables — 1-D: ``tau_xx, tau_xz``; 2-D: ``tau_xx, tau_xy, tau_xz, tau_yy,
    tau_yz`` (``j`` runs over the horizontals AND z).  The shallow / thin-layer
    simplification that drops the in-plane components (``tau_xx`` …) is NOT built
    in — it is applied explicitly and tracked by :func:`moment_scaling`.  The
    pressure ``p`` is a state variable (eliminated hydrostatically).
    """

    def add_to(self, model):
        coords = model.coords
        t, zc = coords[0], coords[-1]
        horiz = coords[1:-1]
        g, rho = model.parameters.g, model.parameters.rho
        e_x = model.parameter("e_x", 0.0)
        uvel = [sp.Function(_HNAME[xd], real=True)(*coords) for xd in horiz]
        w = sp.Function("w", real=True)(*coords)
        p = sp.Function("p", real=True)(*coords)
        dname = lambda cc: "z" if cc == zc else _CNAME[cc]
        dop = lambda cc: d.z if cc == zc else _DERIV[cc]
        tau = {}                               # symmetric stress tensor, by name

        def stress(a, b):
            key = "".join(sorted([dname(a), dname(b)]))   # τ_ij = τ_ji
            if key not in tau:
                tau[key] = sp.Function(f"tau_{key}", real=True)(*coords)
            return tau[key]

        comps = []
        for i, xi in enumerate(horiz):
            adv = (sum(dop(xj)(uvel[i] * uvel[j]) for j, xj in enumerate(horiz))
                   + d.z(uvel[i] * w))
            # FULL stress divergence Σ_j ∂_j τ_ij, j over horizontals + z
            sdiv = (sum(dop(xj)(stress(xi, xj)) for xj in horiz)
                    + d.z(stress(xi, zc)))
            incline = g * e_x if xi == C.x else sp.S.Zero
            comps.append(d.t(uvel[i]) + adv + dop(xi)(p) / rho - sdiv / rho - incline)
        # vertical momentum (eliminated hydrostatically)
        comps.append(d.t(w) + sum(dop(xj)(uvel[j] * w) for j, xj in enumerate(horiz))
                     + d.z(w * w) + d.z(p) / rho + g)
        model.declare_state(p)
        model.declare_closure(*tau.values())
        model.add_equation("momentum", (len(comps),), comps)
        self.uvel, self.w, self.p, self.tau = uvel, w, p, tau
        return model.momentum


def moment_scaling(model, momentum):
    """Shallow / thin-layer MOMENT SCALING — the explicit, TRACKED derivation
    step that drops the IN-PLANE stress components (both indices horizontal,
    e.g. ``tau_xx, tau_xy, tau_yy``) from the full-tensor :class:`Momentum`.

    Under the shallow scaling ``∂_x ~ ε ∂_z`` the in-plane stress divergences are
    O(ε) smaller than the vertical shears ``∂_z τ_iz`` and are dropped.  The
    vertical shears ``tau_xz, tau_yz`` survive (the moment models' closure
    target).  This is the ONE simplification that turns the general momentum
    balance into the shallow-moment form; recording it keeps the derivation
    honest (and 1-D recovers K&T exactly because only ``tau_xz`` survives)."""
    drop = {s: sp.S.Zero for k, s in momentum.tau.items() if "z" not in k}
    if drop:
        model.apply(drop)
        model._history("moment_scaling", "momentum",
                       description=f"shallow scaling: drop in-plane stresses "
                                   f"{sorted(k for k in momentum.tau if 'z' not in k)}")
    return model


def add_inplane_viscous(model, momentum_eqs, uvel, horiz, nu):
    """KEEP-ALL complement of :func:`moment_scaling`: ADD the incompressible-
    Newtonian IN-PLANE deviatoric stress divergence

        −(1/ρ) Σ_e ∂_e[ ρ ν (∂_d u_e + ∂_e u_d) ]

    to each horizontal momentum residual ``momentum_eqs[d]`` (same −div/ρ sign
    the momentum blueprints carry).  Applied BEFORE the σ-map on the 3-D
    velocity fields ``uvel`` — exactly as the ``ml_fullsme`` / ``fullvam``
    recipes inline ``tau_inplane`` — so the retained divergence σ-maps and
    projects with the shear, then :func:`package_viscous` routes it into
    conservative diffusion.

    ``moment_scaling`` (dropping the opaque in-plane stress) is applied FIRST in
    every family, so with ``ν=0`` this term vanishes and the model reduces
    byte-identically to the shallow-moment default; only a retain-viscous
    closure list turns it on.  Registering it on the model history keeps the
    derivation honest."""
    rho = model.parameters.rho
    dop = {C.x: d.x, C.y: d.y}
    for i, xd in enumerate(horiz):
        # τ_de = ρ ν (∂_d u_e + ∂_e u_d);  sdiv_d = Σ_e ∂_e τ_de
        sdiv = sum(dop[xe](rho * nu * (dop[xd](uvel[horiz.index(xe)]) + dop[xe](uvel[i])))
                   for xe in horiz)
        eq = momentum_eqs[i]
        eq.expr = eq.expr - sdiv / rho
    model._history("add_inplane_viscous", "momentum",
                   description="keep in-plane deviatoric stress (incompressible "
                               "Newtonian τ_de = ρν(∂_d u_e + ∂_e u_d))")
    return model


def package_viscous(expr, nu, states, space):
    """Route a KEEP-ALL viscous residual into CONSERVATIVE diffusion.

    After projection the retained in-plane stress lands as BARE second
    derivatives of state ``C·∂_d∂_e Q`` (``Q ∈ states``, ``d,e ∈ space``); a
    conservative FVM needs ``∂_d(D·∂_e Q)``.  Fully expand the ν-bearing terms
    and rewrite each such compound via the EXACT identity

        C·∂_d∂_e Q = ∂_d(C·∂_e Q) − (∂_d C)·∂_e Q

    (→ a diffusive flux ``A`` the extractor routes into ``diffusion_matrix`` +
    a first-derivative NCP remainder that includes every topography-coupled
    ``ν·q·∂b`` product — ``b`` is a STATE, so those evaluate from the same stage
    state, no flat-bed / frozen-b assumption).  Non-viscous terms pass through
    untouched.  This is the single-source form of the identical helper copied
    into the ``ml_fullsme`` / ``fullvam`` / ``ml_fullvam`` notebooks."""
    expr = sp.expand(expr)
    visc = sum((tt for tt in sp.Add.make_args(expr) if tt.has(nu)), sp.S.Zero)
    rest = expr - visc
    out = sp.S.Zero
    for term in sp.Add.make_args(sp.expand(visc.doit())):
        d2 = [dd for dd in term.atoms(sp.Derivative)
              if len(dd.variables) == 2 and all(v in space for v in dd.variables)
              and dd.args[0] in states]
        if d2:
            D2 = d2[0]; Q = D2.args[0]; v = D2.variables
            Coef = sp.cancel(term / D2); de = sp.Derivative(Q, v[1])
            out += (sp.Derivative(Coef * de, v[0], evaluate=False)
                    - sp.Derivative(Coef, v[0]) * de)
        else:
            out += term
    return rest + out


# ── opaque boundary frame ────────────────────────────────────────────────────
# A boundary stress closure prescribes the traction in the local frame
# ``{n, t_α}`` (see closures.py).  The frame is built from OPAQUE slope symbols
# — one per (boundary interface, horizontal direction) — NOT from the physical
# ``∂_d(interface)``, so resolving the frame (``small_slope_scaling`` → 0, or the
# exact ``∂_d I``) never aliases the topographic source ``g h ∂_x b``.  This is
# the inject-opaque / resolve-late pattern: a closure references the frame
# abstractly and is therefore independent of WHEN the frame is resolved.

def frame_slope(tag, direction):
    """Opaque geometric-slope symbol of a boundary interface (``tag`` ∈
    ``{"b", "eta"}`` — bed / free surface) in a horizontal ``direction`` ∈
    ``{"x", "y"}``.  Distinct from the physical ``∂_d(interface)`` so a frame
    resolution touches only the boundary geometry, never the body force."""
    return sp.Symbol(f"frameslope_{tag}_{direction}", real=True)


def small_slope_scaling(model):
    """Shallow-moment BOUNDARY scaling — the tracked, removable step that
    resolves the opaque boundary frame to its small-slope (``n → ẑ``) limit:
    every :func:`frame_slope` symbol → 0.

    The geometric sibling of :func:`moment_scaling` (which drops the in-plane
    stress divergence): together they turn the geometrically-exact column model
    into the shallow-moment (Kowalski & Torrilhon) form.  Applied AFTER the
    boundary closures are injected — because the frame is opaque, the order is
    immaterial; resolving the frame symbols to 0 reduces the projected-traction
    closures to their flat-boundary traces (``τ_iz|_bc`` = the prescribed
    tangential traction) while leaving the physical bed slope ``∂_x b`` in the
    body force untouched.  SKIP this step to keep the slope-aware traction (the
    higher-order model)."""
    slopes = set()
    for eq in model._equations.values():
        slopes |= {s for s in eq.expr.free_symbols
                   if str(s).startswith("frameslope_")}
    if slopes:
        model.apply({s: sp.S.Zero for s in slopes})
        model._history("small_slope_scaling", "momentum",
                       description="resolve boundary frame to small-slope limit "
                                   "n→ẑ (drop O(slope) traction corrections)")
    return model


class MomentumNonHydrostatic(Equation):
    """Non-hydrostatic momentum (VAM): the hydrostatic pressure is PRE-ABSORBED,
    so the horizontal balance carries ``g·∂_x(b+h)`` and ``p`` is only the
    NON-hydrostatic part; the vertical balance keeps ``∂_z p`` with NO ``+g``.
    Registered as separate scalar rows ``momentum_x`` / ``momentum_z`` (the VAM
    convention), since the vertical balance is not eliminated::

        ∂_t u + ∂_x(u u) + ∂_z(u w) + g ∂_x(η) + ∂_x p/ρ − ∂_z τ/ρ   (momentum_x)
        ∂_t w + ∂_x(u w) + ∂_z(w w) + ∂_z p/ρ                        (momentum_z)

    ``suffix`` scopes the field names (multilayer column), ``tau_name`` overrides
    the stress field name (default ``tau_xz``; multilayer uses ``tau_<ell>``),
    and ``free_surface`` overrides η (default ``b + h``; a multilayer column uses
    the TOTAL free surface ``b + H``).
    """

    def __init__(self, model=None, suffix="", tau_name="tau_xz", free_surface=None):
        super().__init__(model)
        self.suffix = suffix
        self.tau_name = tau_name
        self.free_surface = free_surface

    def add_to(self, model):
        coords = model.coords
        t = coords[0]
        horiz = coords[1:-1]
        s = self.suffix
        g, rho = model.parameters.g, model.parameters.rho
        uvel = [sp.Function(_HNAME[xd] + s, real=True)(*coords) for xd in horiz]
        w = sp.Function("w" + s, real=True)(*coords)
        p = sp.Function("p" + s, real=True)(*coords)
        # one shear stress PER horizontal direction.  1 horizontal → keep the
        # ``tau_name`` (default ``tau_xz``; multilayer overrides to ``tau_<ell>``)
        # so 1-D / per-layer is byte-identical; 2 horizontals → ``tau_xz, tau_yz``
        # (+ suffix) so momentum_y closes its OWN shear, not momentum_x's.
        def _sname(xd):
            return self.tau_name if len(horiz) == 1 else f"tau_{_CNAME[xd]}z" + s
        tau = {xd: sp.Function(_sname(xd), real=True)(*coords) for xd in horiz}
        if self.free_surface is not None:
            eta = self.free_surface
        else:
            h = sp.Function("h", positive=True)(t, *horiz)
            b = sp.Function("b", real=True)(t, *horiz)
            eta = b + h
        model.declare_state(p)
        model.declare_closure(*tau.values())
        for i, xd in enumerate(horiz):
            adv = sum(_DERIV[xe](uvel[i] * uvel[j]) for j, xe in enumerate(horiz))
            model.add_equation(
                f"momentum_{_CNAME[xd]}",
                d.t(uvel[i]) + adv + d.z(uvel[i] * w) + g * _DERIV[xd](eta)
                + _DERIV[xd](p) / rho - d.z(tau[xd]) / rho)
        model.add_equation(
            "momentum_z",
            d.t(w) + sum(_DERIV[xd](uvel[i] * w) for i, xd in enumerate(horiz))
            + d.z(w * w) + d.z(p) / rho)
        self.uvel, self.w, self.p, self.tau = uvel, w, p, tau
        self.txz = tau[horiz[0]]
        return model


class Transport(Equation):
    """Conservative transport of a scalar field ``name``:

        ∂_t c + Σ_d ∂_d(u_d c) + ∂_z(w c) = source

    Registers ``c`` as a state field (a new transported quantity: TKE ``k``,
    dissipation ``ε``, a passive tracer, temperature, …).  ``source`` is an
    optional free source field (its modal moments become Qaux unless a closure
    substitutes it) — pass a sympy expression / field, or leave ``None`` for a
    passive scalar.  The velocity ``u``/``w`` are reused from the Mass blueprint.
    """

    def __init__(self, model=None, name="c", source=None):
        super().__init__(model)
        self.name = name
        self.source = source

    def add_to(self, model):
        coords = model.coords
        horiz = coords[1:-1]
        c = sp.Function(self.name, real=True)(*coords)
        uvel = [sp.Function(_HNAME[xd], real=True)(*coords) for xd in horiz]
        w = sp.Function("w", real=True)(*coords)
        model.declare_state(c)
        expr = (d.t(c) + sum(_DERIV[xd](uvel[i] * c) for i, xd in enumerate(horiz))
                + d.z(w * c))
        if self.source is not None:
            expr = expr - self.source
        model.add_equation(self.name, expr)
        self.c = c
        return model._equations[self.name]


__all__ = ["Equation", "Mass", "Momentum", "MomentumNonHydrostatic", "Transport",
           "moment_scaling", "small_slope_scaling", "frame_slope",
           "add_inplane_viscous", "package_viscous"]
