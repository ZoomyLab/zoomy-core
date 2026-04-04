"""
VAM Derivation: Phase 2 -- Abstract zeta-space projection for VAM.

Takes VAMPreProjectedEquations and writes the formal Galerkin projection
in normalized zeta-space [0,1] with abstract matrix symbols.

Unlike the SME projection, VAM has:
- x-momentum AND z-momentum projections
- Cross-momentum terms (u*w coupling)
- Pressure terms kept separate (implicit, not substituted)
- Poisson constraint equations I1, I2

Usage:
    from zoomy_core.model.models.vam_derivation import derive_vam_moments
    from zoomy_core.model.models.vam_zeta_projection import project_vam_to_zeta

    state = StateSpace(dimension=2)
    vam = derive_vam_moments(state)
    zeta = project_vam_to_zeta(vam)
    print(zeta.summary())
"""

from dataclasses import dataclass, field
from typing import List, Dict, Literal
from enum import Enum

from zoomy_core.model.models.vam_derivation import VAMPreProjectedEquations
from zoomy_core.model.models.zeta_projection import ZetaTerm


# ---------------------------------------------------------------------------
# VAM-specific term types
# ---------------------------------------------------------------------------

class VAMTermType(str, Enum):
    """Term types for VAM projected equations."""

    # -- Continuity --
    MASS_TEMPORAL = "mass_temporal"
    MASS_FLUX = "mass_flux"

    # -- x-Momentum (u-equation) --
    U_INERTIA = "u_inertia"                    # h M_lk d(alpha_k)/dt
    U_ADVECTIVE_FLUX = "u_advective_flux"      # h A_lij alpha_i alpha_j
    U_TOPOGRAPHY_NC = "u_topography_nc"        # g h Phi_l db/dx
    U_VERTICAL_ADV_NC = "u_vertical_adv_nc"    # B_lij alpha_i (NC coupling)
    U_MEAN_VEL_NC = "u_mean_vel_nc"            # -alpha_0 coupling

    # -- z-Momentum (w-equation) --
    W_INERTIA = "w_inertia"                    # h M_lk d(gamma_k)/dt
    W_CROSS_FLUX = "w_cross_flux"              # h A_lij alpha_i gamma_j (u*w flux)
    W_VERTICAL_ADV = "w_vertical_adv"          # B-type terms from d(w^2)/dz IBP
    W_GRAVITY = "w_gravity"                    # g Phi_l

    # -- Pressure (implicit, separate from advection) --
    PRESSURE_X_GRADIENT = "pressure_x_gradient"  # d(hp_k)/dx terms in x-mom
    PRESSURE_X_BOUNDARY = "pressure_x_boundary"  # p * db/dx boundary terms
    PRESSURE_Z_VOLUME = "pressure_z_volume"      # -int p dphi/dz dz (after IBP)
    PRESSURE_Z_BOUNDARY = "pressure_z_boundary"  # [p phi]_b^eta boundary terms

    # -- Poisson constraints --
    POISSON_I1 = "poisson_I1"                  # continuity constraint
    POISSON_I2 = "poisson_I2"                  # vorticity constraint

    # -- Closure --
    W_CLOSURE = "w_closure"                    # w_{L+1} from continuity


# ---------------------------------------------------------------------------
# VAMZetaProjectedEquations: Phase 2 output
# ---------------------------------------------------------------------------

@dataclass
class VAMZetaProjectedEquations:
    r"""
    VAM equations projected into abstract zeta-space [0, 1].

    Contains the formal Galerkin projection for non-hydrostatic shallow water:

    **x-momentum** (mode l, raw, before M^{-1}):

    .. math::

        h M_{lk} \frac{\partial \alpha_k}{\partial t}
        + \frac{\partial}{\partial x}(h A_{lij} \alpha_i \alpha_j)
        + g h \Phi_l \frac{\partial b}{\partial x}
        + \text{NC terms}
        = -\frac{\partial(h\pi_0)}{\partial x} \Phi_l - \text{pressure BCs}

    **z-momentum** (mode l, raw):

    .. math::

        h M_{lk} \frac{\partial \gamma_k}{\partial t}
        + \frac{\partial}{\partial x}(h A_{lij} \alpha_i \gamma_j)
        + g \Phi_l
        = \int_0^1 p \frac{d\varphi_l}{d\zeta} d\zeta + \text{pressure BCs}

    Abstract basis matrices:
        M_{lk}, A_{lij}, D_{lk}, B_{lij}, Phi_l, phib_l
    """
    state: object
    continuity: List[ZetaTerm] = field(default_factory=list)
    x_momentum: List[ZetaTerm] = field(default_factory=list)
    z_momentum: List[ZetaTerm] = field(default_factory=list)
    pressure_x: List[ZetaTerm] = field(default_factory=list)
    pressure_z: List[ZetaTerm] = field(default_factory=list)
    poisson: List[ZetaTerm] = field(default_factory=list)
    assumptions_applied: List[str] = field(default_factory=list)
    dimension: int = 2

    @property
    def horizontal_dim(self):
        return self.dimension - 1

    def all_equations(self) -> Dict[str, List[ZetaTerm]]:
        eqs = {
            "continuity": self.continuity,
            "x_momentum": self.x_momentum,
            "z_momentum": self.z_momentum,
            "pressure_x": self.pressure_x,
            "pressure_z": self.pressure_z,
        }
        if self.poisson:
            eqs["poisson"] = self.poisson
        return eqs

    def summary(self) -> str:
        lines = [
            f"VAMZetaProjectedEquations (dim={self.dimension}, "
            f"hdim={self.horizontal_dim})",
            f"  Assumptions: {self.assumptions_applied}",
            f"  Basis matrices: abstract (not yet evaluated)",
        ]
        for name, terms in self.all_equations().items():
            roles = {}
            deps = set()
            for t in terms:
                roles[t.role] = roles.get(t.role, 0) + 1
                deps.update(t.matrix_deps)
            lines.append(
                f"  {name}: {len(terms)} terms {roles}, needs {sorted(deps)}"
            )
        return "\n".join(lines)

    def latex_system(self) -> str:
        """Return LaTeX for the projected VAM PDE system."""
        parts = []

        parts.append(r"\textbf{Continuity:}")
        parts.append(
            " + ".join(t.latex_str for t in self.continuity) + " = 0"
        )

        for eq_name, terms, label in [
            ("x_momentum", self.x_momentum, r"$x$-momentum"),
            ("z_momentum", self.z_momentum, r"$z$-momentum"),
            ("pressure_x", self.pressure_x, r"Pressure ($x$)"),
            ("pressure_z", self.pressure_z, r"Pressure ($z$)"),
        ]:
            if terms:
                parts.append(rf"\textbf{{{label} (mode $l$, raw):}}")
                parts.append(" + ".join(t.latex_str for t in terms) + " = 0")

        return "\n\n".join(parts)

    def __repr__(self):
        return self.summary()


# ---------------------------------------------------------------------------
# project_vam_to_zeta: Phase 1 -> Phase 2
# ---------------------------------------------------------------------------

def project_vam_to_zeta(
    pre: VAMPreProjectedEquations,
) -> VAMZetaProjectedEquations:
    """
    Phase 2: Formal Galerkin projection of VAM equations into abstract zeta-space.

    Produces abstract matrix symbols for all terms. No basis or level chosen yet.
    """
    state = pre.state
    dim = pre.dimension

    result = VAMZetaProjectedEquations(
        state=state,
        dimension=dim,
        assumptions_applied=pre.assumptions_applied + ["vam_zeta_projection"],
    )

    # ===== Continuity =====================================================
    result.continuity.append(ZetaTerm(
        role="temporal",
        origin="mass_conservation",
        term_type=VAMTermType.MASS_TEMPORAL,
        matrix_deps=(),
        latex_str=r"\frac{\partial h}{\partial t}",
    ))
    result.continuity.append(ZetaTerm(
        role="flux",
        origin="mass_flux",
        term_type=VAMTermType.MASS_FLUX,
        matrix_deps=("c_mean",),
        latex_str=r"\frac{\partial}{\partial x}\!\left(h \sum_k c_k \alpha_k\right)",
    ))

    # ===== x-Momentum =====================================================
    _build_vam_x_momentum(result.x_momentum, state)

    # ===== z-Momentum =====================================================
    _build_vam_z_momentum(result.z_momentum, state)

    # ===== Pressure terms (implicit) ======================================
    _build_pressure_terms(result.pressure_x, result.pressure_z, state)

    # ===== Poisson constraints ============================================
    result.poisson.append(ZetaTerm(
        role="source",
        origin="continuity_constraint",
        term_type=VAMTermType.POISSON_I1,
        matrix_deps=("M", "D1", "c_mean"),
        latex_str=(
            r"I_1 = h \frac{\partial \bar{u}}{\partial x}"
            r" + \tfrac{1}{3}\frac{\partial(h\,\alpha_1)}{\partial x}"
            r" + 2(\gamma_0 - \bar{u}\,\frac{\partial b}{\partial x})"
        ),
    ))
    result.poisson.append(ZetaTerm(
        role="source",
        origin="vorticity_constraint",
        term_type=VAMTermType.POISSON_I2,
        matrix_deps=("M", "D1"),
        latex_str=(
            r"I_2 = h \frac{\partial \bar{u}}{\partial x}"
            r" + \alpha_1 \frac{\partial h}{\partial x}"
            r" + 2(\alpha_1\,\frac{\partial b}{\partial x} - \gamma_1)"
        ),
    ))

    return result


# ---------------------------------------------------------------------------
# Builders for momentum and pressure terms
# ---------------------------------------------------------------------------

def _build_vam_x_momentum(terms: List[ZetaTerm], state):
    """Build abstract x-momentum terms for VAM."""

    # Temporal: h M_lk d(alpha_k)/dt
    terms.append(ZetaTerm(
        role="temporal",
        origin="inertia",
        term_type=VAMTermType.U_INERTIA,
        matrix_deps=("M",),
        latex_str=r"h\,M_{lk}\,\frac{\partial \alpha_k}{\partial t}",
        component="x",
    ))

    # Advective flux: d/dx(h A_lij alpha_i alpha_j)
    terms.append(ZetaTerm(
        role="flux",
        origin="advection_uu",
        term_type=VAMTermType.U_ADVECTIVE_FLUX,
        matrix_deps=("A",),
        latex_str=r"\frac{\partial}{\partial x}\!\left(h\,A_{lij}\,\alpha_i\,\alpha_j\right)",
        component="x",
    ))

    # Topography NC: g h Phi_l db/dx
    terms.append(ZetaTerm(
        role="nonconservative",
        origin="topography",
        term_type=VAMTermType.U_TOPOGRAPHY_NC,
        matrix_deps=("phi_int",),
        latex_str=r"g\,h\,\Phi_l\,\frac{\partial b}{\partial x}",
        component="x",
    ))

    # Vertical advection NC: B_lij alpha_j (from IBP of d(uw)/dz)
    terms.append(ZetaTerm(
        role="nonconservative",
        origin="vertical_advection",
        term_type=VAMTermType.U_VERTICAL_ADV_NC,
        matrix_deps=("B",),
        latex_str=r"B_{lij}\,\alpha_i\,\frac{\partial(h\,\alpha_j)}{\partial x}",
        component="x",
    ))

    # Mean velocity NC coupling
    terms.append(ZetaTerm(
        role="nonconservative",
        origin="mean_velocity_coupling",
        term_type=VAMTermType.U_MEAN_VEL_NC,
        matrix_deps=(),
        latex_str=r"-\alpha_0\,\frac{\partial(h\,\alpha_k)}{\partial x}",
        component="x",
    ))


def _build_vam_z_momentum(terms: List[ZetaTerm], state):
    """Build abstract z-momentum terms for VAM."""

    # Temporal: h M_lk d(gamma_k)/dt
    terms.append(ZetaTerm(
        role="temporal",
        origin="inertia",
        term_type=VAMTermType.W_INERTIA,
        matrix_deps=("M",),
        latex_str=r"h\,M_{lk}\,\frac{\partial \gamma_k}{\partial t}",
        component="z",
    ))

    # Cross advective flux: d/dx(h A_lij alpha_i gamma_j)
    terms.append(ZetaTerm(
        role="flux",
        origin="advection_uw",
        term_type=VAMTermType.W_CROSS_FLUX,
        matrix_deps=("A",),
        latex_str=r"\frac{\partial}{\partial x}\!\left(h\,A_{lij}\,\alpha_i\,\gamma_j\right)",
        component="z",
    ))

    # Vertical advection: d(w^2)/dz → IBP → B-type terms
    terms.append(ZetaTerm(
        role="nonconservative",
        origin="vertical_advection_ww",
        term_type=VAMTermType.W_VERTICAL_ADV,
        matrix_deps=("B",),
        latex_str=r"B_{lij}\,\gamma_i\,\frac{\partial(h\,\gamma_j)}{\partial x}",
        component="z",
    ))

    # Gravity source: g * Phi_l
    terms.append(ZetaTerm(
        role="source",
        origin="gravity",
        term_type=VAMTermType.W_GRAVITY,
        matrix_deps=("phi_int",),
        latex_str=r"g\,\Phi_l",
        component="z",
    ))


def _build_pressure_terms(pressure_x: List[ZetaTerm], pressure_z: List[ZetaTerm], state):
    """Build abstract pressure projection terms for VAM."""

    # Pressure gradient in x-momentum:
    # After depth integration: d(h*pi_k)/dx projected onto phi_l
    # At L1: dhp0dx + 2*p1*dbdx (mode 0), -2*p1 (mode 1)
    pressure_x.append(ZetaTerm(
        role="source",
        origin="pressure_gradient_x",
        term_type=VAMTermType.PRESSURE_X_GRADIENT,
        matrix_deps=("M", "phi_int", "phib"),
        latex_str=(
            r"\frac{\partial(h\pi_k)}{\partial x}\,\Phi_l"
            r" + \text{boundary terms}"
        ),
        component="x",
    ))

    # Pressure in z-momentum:
    # After IBP of dp/dz against phi_l:
    #   [p*phi_l]_b^eta - int p * dphi_l/dz dz
    # Volume: uses D1 matrix (int phi_k * dphi_l/dz dz)
    # Boundary: uses phib values
    pressure_z.append(ZetaTerm(
        role="source",
        origin="pressure_ibp_volume",
        term_type=VAMTermType.PRESSURE_Z_VOLUME,
        matrix_deps=("D1",),
        latex_str=r"-\int_0^1 p\,\frac{d\varphi_l}{d\zeta}\,d\zeta",
        component="z",
    ))
    pressure_z.append(ZetaTerm(
        role="source",
        origin="pressure_ibp_boundary",
        term_type=VAMTermType.PRESSURE_Z_BOUNDARY,
        matrix_deps=("phib",),
        latex_str=r"[p\,\varphi_l]_0^1",
        component="z",
    ))
