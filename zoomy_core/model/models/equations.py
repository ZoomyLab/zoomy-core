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
    """3-D momentum balance (horizontal + vertical), with the pressure ``p`` as a
    state variable (eliminated hydrostatically later) and the shear stress
    ``tau_xz`` as a closure variable (free unless a bulk closure substitutes it).

    Reproduces the SME momentum exactly for ``coords=(t, x, z)``::

        ∂_t u + ∂_x(u u) + ∂_z(u w) + ∂_x p/ρ − ∂_z τ/ρ − g e_x      (x)
        ∂_t w + ∂_x(u w) + ∂_z(w w) + ∂_z p/ρ + g                    (z)
    """

    def add_to(self, model):
        coords = model.coords
        horiz = coords[1:-1]
        g, rho = model.parameters.g, model.parameters.rho
        e_x = model.parameter("e_x", 0.0)
        uvel = [sp.Function(_HNAME[xd], real=True)(*coords) for xd in horiz]
        w = sp.Function("w", real=True)(*coords)
        p = sp.Function("p", real=True)(*coords)
        # PER-DIRECTION shear stress τ_{d}z (one closure variable per horizontal
        # direction): 1-D → just tau_xz (byte-identical); 2-D → tau_xz + tau_yz.
        txz = [sp.Function(f"tau_{_CNAME[xd]}z", real=True)(*coords) for xd in horiz]
        model.declare_state(p)
        model.declare_closure(*txz)
        # horizontal momenta — each uses ITS OWN stress τ_{d}z
        comps = []
        for i, xd in enumerate(horiz):
            adv = sum(_DERIV[xe](uvel[i] * uvel[j]) for j, xe in enumerate(horiz))
            incline = g * e_x if xd == C.x else sp.S.Zero
            comps.append(d.t(uvel[i]) + adv + d.z(uvel[i] * w)
                         + _DERIV[xd](p) / rho - d.z(txz[i]) / rho - incline)
        # vertical momentum (for hydrostatic elimination)
        comps.append(d.t(w) + sum(_DERIV[xd](uvel[i] * w) for i, xd in enumerate(horiz))
                     + d.z(w * w) + d.z(p) / rho + g)
        model.add_equation("momentum", (len(comps),), comps)
        self.uvel, self.w, self.p, self.txz = uvel, w, p, txz
        return model.momentum


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


__all__ = ["Equation", "Mass", "Momentum", "MomentumNonHydrostatic", "Transport"]
