"""
Three-Phase PDE Model Derivation: Phase 2 -- Abstract zeta-space projection.

Takes PreProjectedEquations from Phase 1 and performs formal Galerkin projection
in normalized zeta-space [0,1] WITHOUT choosing a basis or level.

The result contains abstract matrix symbols (M, A, D, B, Phi, phi_b) that are
evaluated in Phase 3 with a specific basis.

Mathematical steps performed here:
    1. Coordinate transform  z -> zeta = (z - b) / H,  zeta in [0, 1]
    2. Ansatz substitution:  u(t,x,zeta) = sum_k alpha_k(t,x) phi_k(zeta)
    3. Galerkin projection:  multiply by phi_l(zeta), integrate int_0^1 ... dzeta
    4. Integration by parts on d/dzeta terms
    5. Kinematic BCs at zeta=0 (bottom) and zeta=1 (surface)
    6. Stress BCs: free surface at top, Navier-slip at bottom
    7. Term tagging: temporal, flux, nonconservative, source

Usage:
    from zoomy_core.model.models.model_derivation import derive_shallow_moments
    from zoomy_core.model.models.zeta_projection import project_to_zeta

    state = StateSpace(dimension=2)
    pre = derive_shallow_moments(state)
    zeta = project_to_zeta(pre)

    # Inspect abstract equations:
    print(zeta.summary())
    print(zeta.latex_system())

    # Phase 3: evaluate with a basis
    model = ProjectedModel(zeta, basis_type=Legendre_shifted, level=2)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Literal
from enum import Enum

from zoomy_core.model.models.model_derivation import PreProjectedEquations


# ---------------------------------------------------------------------------
# Term classification
# ---------------------------------------------------------------------------

class TermType(str, Enum):
    """Classification of projected term types for Phase 3 assembly."""

    # -- Continuity --
    MASS_TEMPORAL = "mass_temporal"          # dh/dt
    MASS_FLUX = "mass_flux"                  # d(h * sum c_k alpha_k)/dx
    MASS_FLUX_Y = "mass_flux_y"              # d(h * sum c_k beta_k)/dy

    # -- Momentum (single-component, same-direction) --
    INERTIA = "inertia"                      # h M_{lk} d(alpha_k)/dt
    ADVECTIVE_FLUX = "advective_flux"        # h A_{lij} alpha_i alpha_j
    PRESSURE_FLUX = "pressure_flux"          # g h^2/2 Phi_l
    TOPOGRAPHY_NC = "topography_nc"          # g h Phi_l  (acts on db/dx)
    VERTICAL_ADVECTION_NC = "vertical_advection_nc"  # B_{lij} alpha_j (NC coupling)
    MEAN_VELOCITY_NC = "mean_velocity_nc"    # -alpha_0 (mode coupling)
    VISCOUS_SOURCE = "viscous_source"        # nu/h D_{lk} alpha_k
    SLIP_SOURCE = "slip_source"              # -1/(lam rho) u_b phi_l(0)

    # -- 2D cross-direction flux (only when has_y) --
    ADVECTIVE_FLUX_CROSS = "advective_flux_cross"  # h A_{lij} alpha_i beta_j (or beta_i alpha_j)


# ---------------------------------------------------------------------------
# ZetaTerm: single projected term with abstract matrices
# ---------------------------------------------------------------------------

@dataclass
class ZetaTerm:
    """
    A single term in the zeta-projected PDE system.

    Carries physical classification, matrix dependencies, and LaTeX
    representation.  Phase 3 dispatches on ``term_type`` to evaluate
    the term with concrete basis matrices.
    """
    role: Literal["temporal", "flux", "nonconservative", "source"]
    origin: str
    term_type: TermType
    matrix_deps: tuple          # e.g. ('M',), ('A',), ('D',), ('phib', 'phi_int')
    latex_str: str
    component: str = "x"       # velocity component this term belongs to

    def _repr_latex_(self):
        return f"${self.latex_str}$"

    def __repr__(self):
        return f"ZetaTerm({self.role}, {self.origin}, deps={self.matrix_deps})"


# ---------------------------------------------------------------------------
# ZetaProjectedEquations: Phase 2 output
# ---------------------------------------------------------------------------

@dataclass
class ZetaProjectedEquations:
    r"""
    Galerkin-projected shallow water equations in abstract zeta-space [0, 1].

    Phase 2 output.  Contains the formal Galerkin projection using abstract
    basis matrix symbols:

    .. math::

        M_{lk}   &= \int_0^1 \varphi_l\,\varphi_k\, d\zeta              \\
        A_{lij}  &= \int_0^1 \varphi_l\,\varphi_i\,\varphi_j\, d\zeta    \\
        D_{lk}   &= \int_0^1 \varphi'_l\,\varphi'_k\, d\zeta             \\
        B_{lij}  &= \int_0^1 \varphi'_l\,\psi_j\,\varphi_i\, d\zeta
                    \quad (\psi_j = \textstyle\int_0^\zeta \varphi_j\,d\zeta') \\
        \Phi_l   &= (M \cdot c)_l = \int_0^1 \varphi_l\, d\zeta          \\
        \varphi^b_l &= \varphi_l(0)

    These symbols are NOT evaluated -- the actual numerical values depend on
    the choice of basis and level (Phase 3).
    """
    state: object               # StateSpace
    continuity: List[ZetaTerm] = field(default_factory=list)
    x_momentum: List[ZetaTerm] = field(default_factory=list)
    y_momentum: List[ZetaTerm] = field(default_factory=list)
    assumptions_applied: List[str] = field(default_factory=list)
    dimension: int = 2

    @property
    def horizontal_dim(self):
        return self.dimension - 1

    def all_equations(self) -> Dict[str, List[ZetaTerm]]:
        eqs = {"continuity": self.continuity, "x_momentum": self.x_momentum}
        if self.dimension > 2:
            eqs["y_momentum"] = self.y_momentum
        return eqs

    # -- Display ---------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"ZetaProjectedEquations (dim={self.dimension}, "
            f"hdim={self.horizontal_dim})",
            f"  Assumptions: {self.assumptions_applied}",
            f"  Basis matrices: abstract (not yet evaluated)",
        ]
        for name, terms in self.all_equations().items():
            roles: Dict[str, int] = {}
            deps: set = set()
            for t in terms:
                roles[t.role] = roles.get(t.role, 0) + 1
                deps.update(t.matrix_deps)
            lines.append(
                f"  {name}: {len(terms)} terms {roles}, needs {sorted(deps)}"
            )
        return "\n".join(lines)

    def latex_system(self) -> str:
        """
        Return LaTeX string for the full projected PDE system.

        Shows the raw (before M^{-1}) form of each equation with abstract
        matrix symbols.  Summation convention on repeated indices.
        """
        parts: List[str] = []

        # -- Continuity --------------------------------------------------------
        parts.append(r"\textbf{Continuity:}")
        cont = " + ".join(t.latex_str for t in self.continuity)
        parts.append(cont + " = 0")

        # -- x-momentum --------------------------------------------------------
        parts.append(r"\textbf{$x$-momentum (mode $l$, raw -- before $M^{-1}$):}")
        parts.append(self._latex_momentum(self.x_momentum))

        # -- y-momentum (if 3D) ------------------------------------------------
        if self.dimension > 2 and self.y_momentum:
            parts.append(
                r"\textbf{$y$-momentum (mode $l$, raw):}"
            )
            parts.append(self._latex_momentum(self.y_momentum))

        # -- Matrix legend -----------------------------------------------------
        parts.append(r"\textbf{Abstract basis matrices:}")
        parts.append(
            r"M_{lk} = \int_0^1 \varphi_l\,\varphi_k\, d\zeta"
        )
        parts.append(
            r"A_{lij} = \int_0^1 \varphi_l\,\varphi_i\,\varphi_j\, d\zeta"
        )
        parts.append(
            r"D_{lk} = \int_0^1 \varphi'_l\,\varphi'_k\, d\zeta"
        )
        parts.append(
            r"B_{lij} = \int_0^1 \varphi'_l\,\psi_j\,\varphi_i\, d\zeta"
            r"\quad (\psi_j = \int_0^\zeta \varphi_j\,d\zeta')"
        )
        parts.append(
            r"\Phi_l = (M \cdot c)_l = \int_0^1 \varphi_l\, d\zeta"
        )
        parts.append(
            r"\varphi^b_l = \varphi_l(0)"
        )

        return "\n\n".join(parts)

    @staticmethod
    def _latex_momentum(terms: List[ZetaTerm]) -> str:
        by_role: Dict[str, List[str]] = {
            "temporal": [], "flux": [], "nonconservative": [], "source": [],
        }
        for t in terms:
            by_role[t.role].append(t.latex_str)

        role_label = {
            "temporal": "temporal",
            "flux": "flux",
            "nonconservative": "NC",
            "source": "source",
        }
        pieces = []
        for role in ("temporal", "flux", "nonconservative", "source"):
            if not by_role[role]:
                continue
            inner = " + ".join(by_role[role])
            pieces.append(
                r"\underbrace{" + inner + r"}_{\text{" + role_label[role] + "}}"
            )
        return " + ".join(pieces) + " = 0"

    def _repr_latex_(self):
        return f"$${self.latex_system()}$$"

    def __repr__(self):
        return self.summary()


# ---------------------------------------------------------------------------
# project_to_zeta: Phase 1 -> Phase 2
# ---------------------------------------------------------------------------

def project_to_zeta(pre: PreProjectedEquations) -> ZetaProjectedEquations:
    r"""
    Phase 2: Formal Galerkin projection into abstract zeta-space.

    Takes the Phase 1 output (basis-independent shallow water equations)
    and writes down the Galerkin projection in terms of abstract basis
    matrix symbols.

    No numerical integration is performed -- the result is purely symbolic.
    A specific basis and level are chosen in Phase 3.

    Returns
    -------
    ZetaProjectedEquations
        Abstract projected equations ready for Phase 3 evaluation.
    """
    state = pre.state
    dim = pre.dimension
    has_y = dim > 2

    result = ZetaProjectedEquations(
        state=state,
        dimension=dim,
        assumptions_applied=pre.assumptions_applied + ["zeta_projection"],
    )

    # ===== Continuity =====================================================
    # After depth integration + Leibniz + kinematic BCs:
    #   dh/dt + d/dx[ h sum_k c_k alpha_k ] = 0
    # (+ d/dy[ h sum_k c_k beta_k ] for 3D)

    result.continuity.append(ZetaTerm(
        role="temporal",
        origin="mass_conservation",
        term_type=TermType.MASS_TEMPORAL,
        matrix_deps=(),
        latex_str=r"\frac{\partial h}{\partial t}",
    ))

    result.continuity.append(ZetaTerm(
        role="flux",
        origin="mass_flux",
        term_type=TermType.MASS_FLUX,
        matrix_deps=("c_mean",),
        latex_str=(
            r"\frac{\partial}{\partial x}"
            r"\!\left(h \sum_k c_k \alpha_k\right)"
        ),
    ))

    if has_y:
        result.continuity.append(ZetaTerm(
            role="flux",
            origin="mass_flux_y",
            term_type=TermType.MASS_FLUX_Y,
            matrix_deps=("c_mean",),
            latex_str=(
                r"\frac{\partial}{\partial y}"
                r"\!\left(h \sum_k c_k \beta_k\right)"
            ),
        ))

    # ===== x-Momentum =====================================================
    _build_momentum_terms(result.x_momentum, state, "x", has_y)

    # ===== y-Momentum (3D only) ===========================================
    if has_y:
        _build_momentum_terms(result.y_momentum, state, "y", has_y)

    return result


# ---------------------------------------------------------------------------
# Momentum term builder (shared by x and y)
# ---------------------------------------------------------------------------

def _build_momentum_terms(
    term_list: List[ZetaTerm],
    state,
    component: str,
    has_y: bool,
):
    r"""
    Build the abstract projected terms for one momentum equation.

    For x-momentum, mode l (raw -- before M^{-1} application):

    .. math::

        h\,M_{lk}\,\frac{\partial \alpha_k}{\partial t}
        + \frac{\partial}{\partial x}\!\bigl(h\,A_{lij}\,\alpha_i\,\alpha_j\bigr)
        + \frac{\partial}{\partial x}\!\bigl(\tfrac{g}{2}\,h^2\,\Phi_l\bigr)
        + g\,h\,\Phi_l\,\frac{\partial b}{\partial x}
        + B_{lij}\,\alpha_i\,\frac{\partial(h\,\alpha_j)}{\partial x}
        - \alpha_0\,\frac{\partial(h\,\alpha_k)}{\partial x}
        + \frac{\nu}{h}\,D_{lk}\,\alpha_k
        - \frac{1}{\lambda\rho}\,u_b\,\varphi_l(0)
        = 0
    """
    vel = r"\alpha" if component == "x" else r"\beta"
    coord = component

    # -- Temporal: h M_{lk} d(alpha_k)/dt ---------------------------------
    term_list.append(ZetaTerm(
        role="temporal",
        origin="inertia",
        term_type=TermType.INERTIA,
        matrix_deps=("M",),
        latex_str=(
            rf"h\,M_{{lk}}\,"
            rf"\frac{{\partial {vel}_k}}{{\partial t}}"
        ),
        component=component,
    ))

    # -- Advective flux: d/dx( h A_{lij} alpha_i alpha_j ) ----------------
    term_list.append(ZetaTerm(
        role="flux",
        origin="advection",
        term_type=TermType.ADVECTIVE_FLUX,
        matrix_deps=("A",),
        latex_str=(
            rf"\frac{{\partial}}{{\partial {coord}}}"
            rf"\!\left(h\,A_{{lij}}\,{vel}_i\,{vel}_j\right)"
        ),
        component=component,
    ))

    # -- Cross-direction advective flux (3D only) --------------------------
    if has_y:
        other_vel = r"\beta" if component == "x" else r"\alpha"
        other_coord = "y" if component == "x" else "x"
        term_list.append(ZetaTerm(
            role="flux",
            origin="advection_cross",
            term_type=TermType.ADVECTIVE_FLUX_CROSS,
            matrix_deps=("A",),
            latex_str=(
                rf"\frac{{\partial}}{{\partial {other_coord}}}"
                rf"\!\left(h\,A_{{lij}}\,{vel}_i\,{other_vel}_j\right)"
            ),
            component=component,
        ))

    # -- Hydrostatic pressure flux: d/dx( g h^2/2 Phi_l ) -----------------
    term_list.append(ZetaTerm(
        role="flux",
        origin="hydrostatic_pressure",
        term_type=TermType.PRESSURE_FLUX,
        matrix_deps=("phi_int",),
        latex_str=(
            rf"\frac{{\partial}}{{\partial {coord}}}"
            rf"\!\left(\frac{{g\,h^2}}{{2}}\,\Phi_l\right)"
        ),
        component=component,
    ))

    # -- Topography NC: g h Phi_l db/dx ------------------------------------
    term_list.append(ZetaTerm(
        role="nonconservative",
        origin="topography",
        term_type=TermType.TOPOGRAPHY_NC,
        matrix_deps=("phi_int",),
        latex_str=(
            rf"g\,h\,\Phi_l\,"
            rf"\frac{{\partial b}}{{\partial {coord}}}"
        ),
        component=component,
    ))

    # -- Vertical advection NC: B_{lij} alpha_i d(h alpha_j)/dx -----------
    term_list.append(ZetaTerm(
        role="nonconservative",
        origin="vertical_advection",
        term_type=TermType.VERTICAL_ADVECTION_NC,
        matrix_deps=("B",),
        latex_str=(
            rf"B_{{lij}}\,{vel}_i\,"
            rf"\frac{{\partial(h\,{vel}_j)}}{{\partial {coord}}}"
        ),
        component=component,
    ))

    # -- Mean velocity NC: -alpha_0 d(h alpha_k)/dx -----------------------
    term_list.append(ZetaTerm(
        role="nonconservative",
        origin="mean_velocity_coupling",
        term_type=TermType.MEAN_VELOCITY_NC,
        matrix_deps=(),
        latex_str=(
            rf"-{vel}_0\,"
            rf"\frac{{\partial(h\,{vel}_k)}}{{\partial {coord}}}"
        ),
        component=component,
    ))

    # -- Viscous source: nu/h D_{lk} alpha_k  (from IBP of d^2u/dzeta^2) --
    term_list.append(ZetaTerm(
        role="source",
        origin="newtonian_viscosity",
        term_type=TermType.VISCOUS_SOURCE,
        matrix_deps=("D",),
        latex_str=(
            rf"\frac{{\nu}}{{h}}\,D_{{lk}}\,{vel}_k"
        ),
        component=component,
    ))

    # -- Slip source: -1/(lam rho) u_b phi_l(0) ---------------------------
    term_list.append(ZetaTerm(
        role="source",
        origin="navier_slip",
        term_type=TermType.SLIP_SOURCE,
        matrix_deps=("phib",),
        latex_str=(
            rf"-\frac{{1}}{{\lambda\rho}}"
            rf"\left(\sum_i {vel}_i\,\varphi_i(0)\right)"
            rf"\varphi_l(0)"
        ),
        component=component,
    ))
