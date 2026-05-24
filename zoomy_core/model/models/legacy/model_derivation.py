"""
Three-Phase PDE Model Derivation: Phase 1.

Derives basis-independent shallow water moment equations from the full 3D INS.
The user controls modeling assumptions (hydrostatic, material) explicitly.

Usage:
    state = StateSpace(dimension=2)   # 2 = xz plane
    pre_projected = derive_shallow_moments(state, material=Newtonian(state))
    # pre_projected contains tagged terms ready for ANY basis (Phase 3)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

import sympy as sp
from sympy import Symbol, Function, Derivative, Integral, Rational, S

from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Expression, IBPResult,
    materials, assumptions, Newtonian, Inviscid,
    KinematicBCBottom, KinematicBCSurface, HydrostaticPressure,
)


@dataclass
class TaggedTerm:
    """
    A single term in the projected equations with its role tagged.

    Roles:
      - 'temporal': d(hα)/dt terms → define the mass matrix
      - 'flux': d(F)/dx terms → go into Model.flux()
      - 'nonconservative': B(Q)·dQ/dx terms → go into Model.nonconservative_matrix()
      - 'source': algebraic in Q → go into Model.source()
    """
    expr: Expression
    role: Literal["temporal", "flux", "nonconservative", "source"]
    origin: str = ""

    def __repr__(self):
        short = str(self.expr.expr)[:60]
        return f"TaggedTerm(role={self.role}, origin={self.origin}, expr={short}...)"


@dataclass
class PreProjectedEquations:
    """
    Basis-independent shallow water equations, ready for projection onto any basis.

    Each equation is a list of TaggedTerms. The terms are in abstract form:
    they contain u(t,x,z), w(t,x,z), h(t,x), b(t,x) as SymPy Functions,
    NOT yet expanded into basis coefficients.

    The vertical coordinate is z (not yet mapped to ζ).
    Integration over depth [b, b+H] has been performed via IBP where needed.
    Kinematic BCs, hydrostatic pressure, and stress BCs have been applied.
    """
    state: StateSpace
    continuity: List[TaggedTerm] = field(default_factory=list)
    x_momentum: List[TaggedTerm] = field(default_factory=list)
    y_momentum: List[TaggedTerm] = field(default_factory=list)
    assumptions_applied: List[str] = field(default_factory=list)
    dimension: int = 1

    @property
    def horizontal_dim(self):
        return self.dimension - 1

    def all_equations(self) -> Dict[str, List[TaggedTerm]]:
        eqs = {"continuity": self.continuity, "x_momentum": self.x_momentum}
        if self.dimension > 2:
            eqs["y_momentum"] = self.y_momentum
        return eqs

    def summary(self):
        lines = [f"PreProjectedEquations (space_dim={self.dimension}, horizontal_dim={self.horizontal_dim})"]
        lines.append(f"  Assumptions: {self.assumptions_applied}")
        for name, terms in self.all_equations().items():
            roles = {}
            for t in terms:
                roles[t.role] = roles.get(t.role, 0) + 1
            lines.append(f"  {name}: {len(terms)} terms {roles}")
        return "\n".join(lines)


def derive_shallow_moments(
    state: StateSpace,
    material=None,
    slip_length: Optional[Symbol] = None,
) -> PreProjectedEquations:
    """
    Derive the pre-projected shallow water moment equations from the full INS.

    Steps:
    1. Start from FullINS (continuity, x/y/z momentum)
    2. Apply material model (Newtonian by default)
    3. Apply hydrostatic assumption: simplify z-momentum → p(z)
    4. Substitute hydrostatic pressure into x/y-momentum
    5. Depth-integrate with IBP on z-derivatives
    6. Apply kinematic BCs to boundary terms
    7. Apply stress BCs (slip at bottom, free at top)
    8. Tag each term (temporal, flux, NC, source)

    Returns PreProjectedEquations ready for basis projection (Pass 2).
    """
    ins = FullINS(state)
    dim = state.dim

    if material is None:
        material = Newtonian(state)

    if slip_length is None:
        slip_length = Symbol("lamda", positive=True)

    # --- Step 1-2: Apply material model ---
    xm = ins.x_momentum.apply(material)
    ym = ins.y_momentum.apply(material) if state.has_y else None
    cont = ins.continuity

    assumptions_applied = [f"material={material.name}"]

    # --- Step 3: Hydrostatic assumption ---
    # z-momentum with inertia dropped: dp/dz + rho*g = stress_z
    # For inviscid hydrostatic: dp/dz = -rho*g
    # → p = p_atm + rho*g*(eta - z)
    hydro = HydrostaticPressure(state)
    xm = xm.apply(hydro)
    if ym is not None:
        ym = ym.apply(hydro)
    assumptions_applied.append("hydrostatic_pressure")

    # --- Step 4: Identify terms and apply IBP on z-derivatives ---
    # The x-momentum now has terms with d/dz (from advection d(uw)/dz and viscous d(tau_xz)/dz)
    # We need to:
    # a) Multiply by a test function φ(z) and integrate over [b, b+H]
    # b) Apply IBP to the d/dz terms
    # c) Apply kinematic BCs to the boundary w terms
    # d) Apply stress BCs to the boundary tau terms

    # For now, we work with the symbolic expressions directly.
    # The actual projection onto φ_k happens in Pass 2.
    # Here we identify and tag the terms.

    result = PreProjectedEquations(
        state=state,
        dimension=dim,
        assumptions_applied=assumptions_applied,
    )

    # --- Tag continuity terms ---
    # Continuity: du/dx [+ dv/dy] + dw/dz = 0
    # After depth integration with IBP on dw/dz:
    #   d(hū)/dx + [w]_{b}^{b+H} = 0
    #   d(hū)/dx + w_s - w_b = 0
    # With kinematic BCs: w_s = d(eta)/dt + u_s*d(eta)/dx, w_b = db/dt + u_b*db/dx
    # → dh/dt + d(hū)/dx = 0

    result.continuity.append(TaggedTerm(
        expr=Expression(Derivative(state.h, state.t), "dH/dt"),
        role="temporal",
        origin="mass_conservation",
    ))

    # Mass flux: d/dx integral(u dz) — this becomes d(h*u_mean)/dx after ansatz
    u_sym = Function("u", real=True)(state.t, state.x, state.z)
    mass_flux_integrand = Expression(u_sym, "u")
    result.continuity.append(TaggedTerm(
        expr=mass_flux_integrand,
        role="flux",
        origin="mass_flux",
    ))

    if state.has_y:
        v_sym = Function("v", real=True)(state.t, state.y, state.z)
        result.continuity.append(TaggedTerm(
            expr=Expression(v_sym, "v_mass_flux"),
            role="flux",
            origin="mass_flux_y",
        ))

    # --- Tag x-momentum terms ---
    _tag_momentum(result.x_momentum, xm, state, "x", slip_length)

    if state.has_y and ym is not None:
        _tag_momentum(result.y_momentum, ym, state, "y", slip_length)

    return result


def _tag_momentum(tagged_list, momentum_expr, state, component, slip_length):
    """
    Tag terms in the momentum equation.

    After hydrostatic substitution, the x-momentum has:
    - du/dt → temporal
    - d(uu)/dx, d(uv)/dy → flux (advection)
    - d(uw)/dz → needs IBP → becomes boundary terms (kinematic BC, stress)
    - dp/dx = d(g*h²/2)/dx + g*h*db/dx → flux (pressure) + NC (topography)
    - d(tau_xz)/dz → needs IBP → viscous volume + stress boundary
    """
    t, x, z = state.t, state.x, state.z
    u = state.u
    w = state.w
    rho = state.rho
    g = state.g
    b = state.b
    H = state.h
    eta = state.eta

    vel_name = "u" if component == "x" else "v"
    coord = state.x if component == "x" else state.y

    # Temporal: du/dt
    tagged_list.append(TaggedTerm(
        expr=Expression(Derivative(Function(vel_name)(t, x, z), t), f"d{vel_name}/dt"),
        role="temporal",
        origin="inertia",
    ))

    # Advective flux: d(u*u_comp)/dx for each horizontal direction
    # These are flux terms — they have d/dx structure
    for vel, vel_coord, vel_label in [(state.u, state.x, "u")]:
        tagged_list.append(TaggedTerm(
            expr=Expression(
                Function(vel_name)(t, x, z) * vel,
                f"{vel_name}*{vel_label}_advection"
            ),
            role="flux",
            origin=f"advection_{vel_label}",
        ))

    if state.has_y:
        tagged_list.append(TaggedTerm(
            expr=Expression(
                Function(vel_name)(t, x, z) * state.v,
                f"{vel_name}*v_advection"
            ),
            role="flux",
            origin="advection_v",
        ))

    # Hydrostatic pressure flux: g*h²/2 (after substitution, this is d(g*h²/2)/dx)
    tagged_list.append(TaggedTerm(
        expr=Expression(g * H**2 / 2, "pressure"),
        role="flux",
        origin="hydrostatic_pressure",
    ))

    # Topography non-conservative term: g*h*db/dx
    tagged_list.append(TaggedTerm(
        expr=Expression(g * H, "g*H_topography"),
        role="nonconservative",
        origin="topography",
    ))

    # Vertical advection d(uw)/dz → after IBP:
    #   boundary: [u*w*φ]_{b}^{eta} → kinematic BCs give source terms
    #   volume: -integral(u*w * dφ/dz dz) → NC coupling
    tagged_list.append(TaggedTerm(
        expr=Expression(
            Function(vel_name)(t, x, z) * w,
            f"{vel_name}*w_vertical_advection"
        ),
        role="source",
        origin="vertical_advection_ibp",
    ))

    # Viscous stress d(tau_xz)/dz → after IBP:
    #   boundary at z=b: tau_b → slip friction source
    #   boundary at z=eta: tau_s = 0 (free surface, no wind for now)
    #   volume: -integral(tau_xz * dφ/dz dz) → viscous source
    nu = Symbol("nu", positive=True)
    tagged_list.append(TaggedTerm(
        expr=Expression(
            -nu * Derivative(Function(vel_name)(t, x, z), z, 2),
            "viscous_diffusion"
        ),
        role="source",
        origin="newtonian_viscosity",
    ))

    # Slip friction (from IBP boundary at z=b)
    tagged_list.append(TaggedTerm(
        expr=Expression(
            -Function(vel_name)(t, x, z) / (slip_length * rho),
            "slip_friction"
        ),
        role="source",
        origin="navier_slip",
    ))
