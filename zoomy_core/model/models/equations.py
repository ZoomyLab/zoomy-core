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

# horizontal coord symbol → its directional derivative operator and field name
_DERIV = {C.x: d.x, C.y: d.y}
_HNAME = {C.x: "u", C.y: "v"}


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
    ``∂_x u + ∂_z w``."""

    def add_to(self, model):
        coords = model.coords
        horiz = coords[1:-1]                       # (x,) or (x, y)
        uvel = [sp.Function(_HNAME[xd], real=True)(*coords) for xd in horiz]
        w = sp.Function("w", real=True)(*coords)
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
        txz = sp.Function("tau_xz", real=True)(*coords)
        model.declare_state(p)
        model.declare_closure(txz)
        # horizontal momenta
        comps = []
        for i, xd in enumerate(horiz):
            adv = sum(_DERIV[xe](uvel[i] * uvel[j]) for j, xe in enumerate(horiz))
            incline = g * e_x if xd == C.x else sp.S.Zero
            comps.append(d.t(uvel[i]) + adv + d.z(uvel[i] * w)
                         + _DERIV[xd](p) / rho - d.z(txz) / rho - incline)
        # vertical momentum (for hydrostatic elimination)
        comps.append(d.t(w) + sum(_DERIV[xd](uvel[i] * w) for i, xd in enumerate(horiz))
                     + d.z(w * w) + d.z(p) / rho + g)
        model.add_equation("momentum", (len(comps),), comps)
        self.uvel, self.w, self.p, self.txz = uvel, w, p, txz
        return model.momentum


__all__ = ["Equation", "Mass", "Momentum"]
