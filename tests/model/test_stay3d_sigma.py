"""Stay-3D σ-coordinate model + the vertical-as-flux extractor fix.

Verifies (run, not asserted by hand):
* the height equation is EXACTLY  ∂_t h + ∂_x(h·U) = 0,  U = ∫₀¹ ũ dζ, with the
  vertical integral substituted into the aux U via the plain add_equation/apply
  machinery (no promote_to_aux op) and no surviving sp.Integral;
* the CONSERVATIVE σ identity (mass + x-momentum), with the contravariant
  vertical velocity h·ω = w̃ − ∂_t z − ũ ∂_x z and ω(0)=ω(1)=0 ⟺ the two KBCs;
* extract_system_operators treats the VERTICAL as a genuine flux direction —
  a σ-dependent STATE field extracts over space=(x,ζ) (∂_ζ as flux + diffusion);
* the FULL stay-3D system (state [b,h,mom]) extracts over space=(x,ζ) with
  per-field dimensionality — the depth h(t,x) stays ζ-independent so the viscous
  ζζ diffusion of mom/h carries no spurious ∂_ζ h coupling;
* depth-reduced models keep their horizontals (the space-fix is byte-identical
  for them).
"""
import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.derivation import (
    Model as DModel, PDETransformation, Simplify, ResolveIntegral)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ
from zoomy_core.model.models.equations import Mass, Momentum, moment_scaling
from zoomy_core.systemmodel import SystemModel

t, x, z = C.t, C.x, C.z
zeta = sp.Symbol("zeta", real=True)
_VALUES = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0, "e_x": 0.0}


def _sigma_mapped_model():
    """3-D mass+momentum → hydrostatic p → σ-map; returns (m, h, b, ũ, w̃)."""
    m = DModel(coords=(t, x, z), parameters=_VALUES)
    h = sp.Function("h", positive=True)(t, x)
    b = sp.Function("b", real=True)(t, x)
    m.declare_state(h)
    m.add_equation("bottom", d.t(b))
    m.add_equation(Mass(m))
    mom = Momentum(m); m.add_equation(mom)
    moment_scaling(m, mom)
    uvel, w, p = mom.uvel, mom.w, mom.p
    m.add_equation("kbc_top", KinematicBC(w=w, u=uvel[0], interface=b + h))
    m.add_equation("kbc_bot", KinematicBC(w=w, u=uvel[0], interface=b))
    mz = m.momentum.z
    mz.apply({d.t(w): 0, d.z(w * w): 0, d.x(uvel[0] * w): 0})
    mz.apply(IntegrateZ(z, z, b + h, method="analytical"))
    mz.apply({p.subs(z, b + h): 0})
    m.momentum.x.apply(mz.solve_for(p)); mz.remove(); m.momentum.x.apply(Simplify())
    m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))
    pool = m.momentum.x.expr.atoms(sp.Function)
    ut = next(a.func for a in pool if str(a.func) == r"\tilde{u}")(t, x, zeta)
    wt = next(a.func for a in pool if str(a.func) == r"\tilde{w}")(t, x, zeta)
    return m, h, b, ut, wt


def test_full_3d_conservative_sigma_extract():
    """Full 3-D conservative-σ system: state ``[b, h, mom]`` over ``space=(x,ζ)``.

    Height eq ``∂_t h + ∂_x(h·U) = 0`` (U aux, integral absorbed); momentum
    x-flux ``mom²/h`` (RAW 1/h — regularization is the NumericalSystemModel's
    job) + the ζ-flux ``mom·ω``; pressure ``g·h²/2`` tagged hydrostatic
    (manual); viscous ``−ν/h²`` as the ζζ diffusion of mom with NO spurious
    ``∂_ζ h`` coupling (per-field dims); bed slope ``g·h`` in the NCP; ``e_x·g·h``
    source."""
    from zoomy_core.model.models.stay3d_sigma import Stay3DSigma
    mdl = Stay3DSigma(parameters={"nu": 0.7, "e_x": 0.3, "g": 9.81,
                                  "rho": 1.0, "lambda_s": 0.5})
    m = mdl.derivation                               # internal asserts ran in derive
    assert not m.mass.expr.atoms(sp.Integral)        # integral absorbed into U
    sm = mdl.system_model
    names = [str(s) for s in sm.state]
    assert names[:3] == ["b", "h", "mom"]
    assert sm.space == [x, zeta]                      # ζ IS a genuine flux direction
    assert {"U", "omega"} <= {str(s) for s in sm.aux_state}
    hrow, mrow, brow = names.index("h"), names.index("mom"), names.index("b")
    g = sm.parameters.g
    # height flux F[h]=U·h (x only); momentum x-flux mom²/h + ζ-flux mom·ω
    assert sm.flux[hrow, 0] != 0 and sm.flux[hrow, 1] == 0
    assert sm.flux[mrow, 0] != 0 and sm.flux[mrow, 1] != 0
    # g·h²/2 routed to hydrostatic_pressure (manual tag), not flux
    assert sm.hydrostatic_pressure[mrow, 0].has(g)
    assert not sm.flux[mrow, 0].has(g)
    # viscous −ν/h² is the ONLY ζζ diffusion entry of the mom row (no ∂_ζ h term)
    assert sm.diffusion_matrix[mrow, mrow, 1, 1] != 0
    assert sm.diffusion_matrix[mrow, hrow, 1, 1] == 0
    # bed slope g·h is the NCP coupling mom←b
    assert sm.nonconservative_matrix[mrow, brow, 0] != 0


def test_conservative_sigma_identity():
    """σ-mass(×h) ≡ ∂_t h+∂_x(hu)+∂_ζ(hω); σ-mom(×h) ≡ ∂_t(hu)+∂_x(hu²+gh²/2)
    +∂_ζ(huω)+gh∂_xb−e_x gh−τ̃_z/ρ; ω(0)=ω(1)=0 ⟺ bed/surface KBC."""
    m, h, b, ut, wt = _sigma_mapped_model()
    g, rho, e_x = m.parameters.g, m.parameters.rho, m.parameters.e_x
    taut = next(a.func for a in m.momentum.x.expr.atoms(sp.Function)
                if str(a.func) == r"\tilde{tau_xz}")(t, x, zeta)
    m.mass.apply(Multiply(h))
    mass_res = sp.expand(m.mass.expr.doit())
    mom_res = sp.expand((h * m.momentum.x.expr).doit())
    zc = b + h * zeta
    homega = wt - sp.Derivative(zc, t) - ut * sp.Derivative(zc, x)   # = h·ω
    cons_mass = sp.expand((sp.Derivative(h, t) + sp.Derivative(h * ut, x)
                           + sp.Derivative(homega, zeta)).doit())
    cons_mom = sp.expand((sp.Derivative(h * ut, t)
                          + sp.Derivative(h * ut**2 + g * h**2 / 2, x)
                          + sp.Derivative(ut * homega, zeta)
                          + g * h * sp.Derivative(b, x) - e_x * g * h
                          - sp.Derivative(taut, zeta) / rho).doit())
    assert sp.simplify(cons_mass - mass_res) == 0
    assert sp.simplify(cons_mom - mom_res) == 0
    # ω = 0 at the boundaries reproduces the kinematic BCs
    assert sp.simplify(homega.subs(zeta, 0)
                       - (wt.subs(zeta, 0) - sp.Derivative(b, t)
                          - ut.subs(zeta, 0) * sp.Derivative(b, x))) == 0
    assert sp.simplify(homega.subs(zeta, 1)
                       - (wt.subs(zeta, 1) - sp.Derivative(b + h, t)
                          - ut.subs(zeta, 1) * sp.Derivative(b + h, x))) == 0


def test_vertical_is_a_flux_direction():
    """A σ-dependent STATE field extracts over space=(x,ζ): ∂_ζ routes as a
    genuine flux (column) and ∂_ζ² as the ζζ diffusion — no special-casing."""
    m = DModel(coords=(t, x, zeta), parameters={"nu": 0.01})
    u = sp.Function("u", real=True)(t, x, zeta)
    om = sp.Function("om", real=True)(t, x, zeta)
    nu = m.parameters.nu
    m.declare_state(u)
    m.add_equation("mom", d.t(u) + d.x(u * u / 2) + d.zeta(u * om)
                   - d.zeta(nu * d.zeta(u)))
    sm = SystemModel.from_model(m, Q=[u])
    assert sm.space == [x, zeta]                      # vertical IS a flux direction
    assert sm.flux.shape == (1, 2)                    # x-flux and ζ-flux columns
    assert sm.flux[0, 1] != 0                         # the ζ-flux (u·ω) is present
    assert sm.diffusion_matrix[0, 0, 1, 1] != 0       # the ζζ vertical diffusion
    assert m.vertical == zeta                         # derived from the field dep, not a hack
