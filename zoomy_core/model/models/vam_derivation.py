"""
VAM (Vorticity-Assisted Model) Derivation: Phase 1.

Derives the non-hydrostatic shallow water moment equations from the full 3D INS,
keeping BOTH x-momentum AND z-momentum and treating pressure as an unknown.

This differs from ``derive_shallow_moments()`` (which applies hydrostatic assumption
and eliminates z-momentum) in three ways:

1. **Non-hydrostatic**: z-momentum is kept, not eliminated
2. **Both u and w moments**: velocity ansatz includes w(zeta) = sum gamma_k phi_k
3. **Pressure unknown**: p(zeta) = sum pi_k phi_k, solved via Poisson constraint

Usage:
    state = StateSpace(dimension=2)
    vam = derive_vam_moments(state, material=Inviscid(state))
    # vam contains tagged terms for x-mom, z-mom, pressure, and continuity
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

import sympy as sp
from sympy import Symbol, Function, Derivative, Integral, Rational, S

from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Expression, IBPResult,
    materials, Newtonian, Inviscid,
    KinematicBCBottom, KinematicBCSurface,
)
from zoomy_core.model.models.model_derivation import TaggedTerm


# ---------------------------------------------------------------------------
# VAMPreProjectedEquations: Phase 1 output for VAM
# ---------------------------------------------------------------------------

@dataclass
class VAMPreProjectedEquations:
    """
    Basis-independent non-hydrostatic shallow water equations (VAM).

    Unlike ``PreProjectedEquations`` (hydrostatic SME), this keeps the
    z-momentum equation and treats pressure as an unknown field.

    Fields:
        continuity: depth-integrated continuity (dh/dt + du/dx terms)
        x_momentum: x-momentum terms (advection, gravity) — NO pressure
        z_momentum: z-momentum terms (advection, gravity) — NO pressure
        pressure_x: pressure gradient terms for x-momentum (implicit)
        pressure_z: pressure gradient terms for z-momentum (implicit)
        continuity_closure: closure relation for higher w-modes from continuity
    """
    state: StateSpace
    continuity: List[TaggedTerm] = field(default_factory=list)
    x_momentum: List[TaggedTerm] = field(default_factory=list)
    z_momentum: List[TaggedTerm] = field(default_factory=list)
    pressure_x: List[TaggedTerm] = field(default_factory=list)
    pressure_z: List[TaggedTerm] = field(default_factory=list)
    continuity_closure: List[TaggedTerm] = field(default_factory=list)
    assumptions_applied: List[str] = field(default_factory=list)
    dimension: int = 2

    @property
    def horizontal_dim(self):
        return self.dimension - 1

    def all_equations(self) -> Dict[str, List[TaggedTerm]]:
        eqs = {
            "continuity": self.continuity,
            "x_momentum": self.x_momentum,
            "z_momentum": self.z_momentum,
            "pressure_x": self.pressure_x,
            "pressure_z": self.pressure_z,
        }
        if self.continuity_closure:
            eqs["continuity_closure"] = self.continuity_closure
        return eqs

    def summary(self) -> str:
        lines = [
            f"VAMPreProjectedEquations (dim={self.dimension}, "
            f"hdim={self.horizontal_dim})",
            f"  Assumptions: {self.assumptions_applied}",
        ]
        for name, terms in self.all_equations().items():
            roles = {}
            for t in terms:
                roles[t.role] = roles.get(t.role, 0) + 1
            lines.append(f"  {name}: {len(terms)} terms {roles}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# derive_vam_moments: Phase 1 derivation
# ---------------------------------------------------------------------------

def derive_vam_moments(
    state: StateSpace,
    material=None,
    slip_length: Optional[Symbol] = None,
) -> VAMPreProjectedEquations:
    """
    Derive the non-hydrostatic VAM equations from the full 3D INS.

    Unlike ``derive_shallow_moments()``, this function:
    - Does NOT apply the hydrostatic assumption
    - Keeps the z-momentum equation (for vertical velocity moments w_k)
    - Treats pressure p as an unknown (separated into pressure_x, pressure_z)

    Steps:
    1. Start from FullINS (continuity, x/z momentum)
    2. Apply material model (Inviscid by default for VAM)
    3. Separate pressure gradient from advection in x-momentum and z-momentum
    4. Tag advective terms (flux/NC/source)
    5. Tag pressure terms (source_implicit)
    6. Extract continuity closure for auxiliary w-modes

    Returns VAMPreProjectedEquations ready for Phase 2 (zeta projection).
    """
    ins = FullINS(state)
    dim = state.dim

    if material is None:
        material = Inviscid(state)

    if slip_length is None:
        slip_length = Symbol("lamda", positive=True)

    # Apply material model to all momentum equations
    xm = ins.x_momentum.apply(material)
    zm = ins.z_momentum.apply(material)
    cont = ins.continuity

    assumptions_applied = [f"material={material.name}", "non_hydrostatic"]

    result = VAMPreProjectedEquations(
        state=state,
        dimension=dim,
        assumptions_applied=assumptions_applied,
    )

    t, x, z = state.t, state.x, state.z
    u, w, p = state.u, state.w, state.p
    rho, g = state.rho, state.g
    b, H, eta = state.b, state.H, state.eta

    # ===== Continuity =====================================================
    # du/dx + dw/dz = 0
    # After depth integration + Leibniz + kinematic BCs:
    #   dh/dt + d(h*u_mean)/dx = 0
    # The w-terms vanish via kinematic BCs, but produce the closure for w.

    result.continuity.append(TaggedTerm(
        expr=Expression(Derivative(H, t), "dH/dt"),
        role="temporal",
        origin="mass_conservation",
    ))

    u_sym = Function("u", real=True)(t, x, z)
    result.continuity.append(TaggedTerm(
        expr=Expression(u_sym, "u"),
        role="flux",
        origin="mass_flux",
    ))

    # ===== x-Momentum (advective part, no pressure) =======================
    # du/dt + d(u^2)/dx + d(uw)/dz = -(1/rho)*dp/dx + stress terms
    #
    # We separate the pressure gradient from the advective terms.
    # The advective + gravity part goes into x_momentum.
    # The pressure gradient goes into pressure_x.

    vel_name = "u"
    _tag_vam_momentum(
        result.x_momentum, state, "x", vel_name, slip_length,
        include_gravity=True,
    )

    # Pressure gradient for x-momentum: (1/rho) * dp/dx
    # After depth integration against phi_l:
    #   int phi_l * dp/dx dz = d(h*p_mean)/dx + boundary terms
    # These become dhp0dx + 2*p1*dbdx etc. after basis expansion.
    result.pressure_x.append(TaggedTerm(
        expr=Expression(
            Rational(1, 1) / rho * Derivative(p, x),
            "dp/dx"
        ),
        role="source",
        origin="pressure_gradient_x",
    ))

    # ===== z-Momentum (advective part, no pressure) =======================
    # dw/dt + d(uw)/dx + d(w^2)/dz + g = -(1/rho)*dp/dz + stress terms
    #
    # Advective + gravity goes into z_momentum.
    # Pressure gradient dp/dz goes into pressure_z.

    vel_name_w = "w"
    _tag_vam_momentum(
        result.z_momentum, state, "z", vel_name_w, slip_length,
        include_gravity=True,
    )

    # Pressure gradient for z-momentum: (1/rho) * dp/dz
    # After depth integration with IBP against phi_l:
    #   int phi_l * dp/dz dz = [p*phi_l]_b^eta - int p * dphi_l/dz dz
    # The boundary terms at z=eta give atmospheric pressure (=0 gauge).
    # The boundary terms at z=b give bottom pressure.
    # The volume integral gives the algebraic pressure coupling:
    #   at L1: 6*(p0-p1) for mode k=1 etc.
    result.pressure_z.append(TaggedTerm(
        expr=Expression(
            Rational(1, 1) / rho * Derivative(p, z),
            "dp/dz"
        ),
        role="source",
        origin="pressure_gradient_z",
    ))

    # ===== Continuity closure (w2 from divergence constraint) =============
    # The depth-integrated continuity + basis expansion gives:
    #   w_{L+1} = f(u_k, w_k, db/dx, dh/dx)
    # At L1: w2 is determined by du/dx + dw/dz = 0 integrated against phi_2.
    result.continuity_closure.append(TaggedTerm(
        expr=Expression(
            Derivative(u_sym, x) + Derivative(
                Function("w", real=True)(t, x, z), z
            ),
            "continuity_closure"
        ),
        role="source",
        origin="w_closure",
    ))

    return result


# ---------------------------------------------------------------------------
# Helper: tag momentum terms for VAM
# ---------------------------------------------------------------------------

def _tag_vam_momentum(
    tagged_list: List[TaggedTerm],
    state: StateSpace,
    component: str,
    vel_name: str,
    slip_length: Symbol,
    include_gravity: bool = True,
):
    """
    Tag the advective (non-pressure) terms for a VAM momentum equation.

    For x-momentum (component='x', vel_name='u'):
        - Temporal: du/dt
        - Advective flux: d(u^2)/dx (same direction)
        - Cross advective: d(uw)/dx (x-flux of w, appears in z-momentum)
        - Vertical advection: d(uw)/dz → IBP → boundary + NC coupling
        - Gravity: g (for z-momentum only)
        - Viscous/slip: if material is Newtonian

    For z-momentum (component='z', vel_name='w'):
        - Temporal: dw/dt
        - Cross advective flux: d(uw)/dx
        - Vertical advection: d(w^2)/dz → IBP
        - Gravity: g
    """
    t, x, z_coord = state.t, state.x, state.z
    u, w = state.u, state.w
    rho, g = state.rho, state.g
    nu = Symbol("nu", positive=True)

    # The primary velocity for this equation
    if component == "x":
        vel_func = Function("u", real=True)(t, x, z_coord)
        cross_vel = Function("w", real=True)(t, x, z_coord)
    else:  # z
        vel_func = Function("w", real=True)(t, x, z_coord)
        cross_vel = Function("u", real=True)(t, x, z_coord)

    # -- Temporal --
    tagged_list.append(TaggedTerm(
        expr=Expression(Derivative(vel_func, t), f"d{vel_name}/dt"),
        role="temporal",
        origin="inertia",
    ))

    if component == "x":
        # -- Advective flux: d(u*u)/dx --
        tagged_list.append(TaggedTerm(
            expr=Expression(
                Function("u", real=True)(t, x, z_coord) *
                Function("u", real=True)(t, x, z_coord),
                "u*u_advection"
            ),
            role="flux",
            origin="advection_uu",
        ))

        # -- Cross flux: d(u*w)/dx (appears in z-momentum row) --
        # Not needed here; z-momentum handles its own d(uw)/dx.

    else:  # z-momentum
        # -- Cross advective flux: d(u*w)/dx --
        tagged_list.append(TaggedTerm(
            expr=Expression(
                Function("u", real=True)(t, x, z_coord) *
                Function("w", real=True)(t, x, z_coord),
                "u*w_cross_flux"
            ),
            role="flux",
            origin="advection_uw",
        ))

    # -- Vertical advection: d(vel*w)/dz → IBP boundary + NC volume --
    tagged_list.append(TaggedTerm(
        expr=Expression(
            vel_func * Function("w", real=True)(t, x, z_coord),
            f"{vel_name}*w_vertical_advection"
        ),
        role="source",
        origin="vertical_advection_ibp",
    ))

    # -- Gravity (z-momentum only) --
    if component == "z" and include_gravity:
        tagged_list.append(TaggedTerm(
            expr=Expression(state.g, "gravity"),
            role="source",
            origin="gravity",
        ))

    # -- Topography NC: g*h*db/dx (x-momentum only, from depth integration) --
    if component == "x" and include_gravity:
        tagged_list.append(TaggedTerm(
            expr=Expression(state.g * state.H, "g*H_topography"),
            role="nonconservative",
            origin="topography",
        ))
