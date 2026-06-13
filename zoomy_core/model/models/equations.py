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
        txz = sp.Function(self.tau_name, real=True)(*coords)
        if self.free_surface is not None:
            eta = self.free_surface
        else:
            h = sp.Function("h", positive=True)(t, *horiz)
            b = sp.Function("b", real=True)(t, *horiz)
            eta = b + h
        model.declare_state(p)
        model.declare_closure(txz)
        for i, xd in enumerate(horiz):
            adv = sum(_DERIV[xe](uvel[i] * uvel[j]) for j, xe in enumerate(horiz))
            model.add_equation(
                f"momentum_{_CNAME[xd]}",
                d.t(uvel[i]) + adv + d.z(uvel[i] * w) + g * _DERIV[xd](eta)
                + _DERIV[xd](p) / rho - d.z(txz) / rho)
        model.add_equation(
            "momentum_z",
            d.t(w) + sum(_DERIV[xd](uvel[i] * w) for i, xd in enumerate(horiz))
            + d.z(w * w) + d.z(p) / rho)
        self.uvel, self.w, self.p, self.txz = uvel, w, p, txz
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
           "moment_scaling"]
