"""SME — Shallow Moment Equations (hydrostatic).

Derivation::

    system = FullINS(state).system()
    system.equations["z_momentum"].apply({w: 0, τ_zz: 0, τ_zx: 0})
    → ∂p/∂z/ρ + g = 0  →  p = p_atm + ρg(η - z)
    system.apply({p: p_hydro})
    del system.equations["z_momentum"]
    system.apply(DepthIntegrate)
    system.apply(ApplyKinematicBCs)
    ...
"""

import sympy as sp
from sympy import Function, S

from zoomy_core.model.models.derived_model import DerivedModel


def hydrostatic_scaling(state):
    """Dict that drops w and all z-row/z-column stresses (hydrostatic assumption).

    Dimension-agnostic: works for 2D (x-z) and 3D (x-y-z).
    """
    scaling = {state.w: S.Zero}
    for key in state.tau.keys():
        if "z" in key:
            scaling[state.tau[key]] = S.Zero
    return scaling


class INSModel(DerivedModel):
    """Root: the full incompressible Navier-Stokes (all equations).

    Dimension-agnostic: works for 2D (x-z) and 3D (x-y-z).
    The ``ins_dimension`` parameter controls the INS state space:
      - 2: u, w (standard 2D vertical slice)
      - 3: u, v, w (full 3D)

    Uses numerical eigenvalues (np.linalg.eigvals on quasilinear matrix)
    because the symbolic Cardano formula produces complex intermediates
    for higher-moment models (casus irreducibilis).
    """

    eigenvalue_mode = "numerical"
    ins_dimension = 2  # override to 3 for 3D derivation

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import StateSpace, FullINS
        self._system = FullINS(StateSpace(dimension=self.ins_dimension))


class SMEModel(INSModel):
    """Shallow Moment Equations — hydrostatic, depth-integrated, Newtonian."""

    projectable = True

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            HydrostaticPressure, Newtonian, DepthIntegrate,
            ApplyKinematicBCs, StressFreeSurface,
            ZeroAtmosphericPressure, SimplifyIntegrals,
        )
        super().derive_model()
        s = self.state

        # Hydrostatic: scale z-momentum, derive pressure, apply to system
        self._system.equations["z_momentum"] = (
            self._system.equations["z_momentum"]
            .apply(hydrostatic_scaling(s))
            .simplify()
        )
        self.apply(HydrostaticPressure(s))
        self._system.remove_equation("z_momentum")

        # Depth integrate + closures
        self.apply(DepthIntegrate(s))
        self.apply(ApplyKinematicBCs(s))
        self.apply(StressFreeSurface(s))
        self.apply(ZeroAtmosphericPressure(s))
        self.apply(SimplifyIntegrals(s))
        self.apply(Newtonian(s))

    def source(self):
        return (self.newtonian_viscosity() + self.navier_slip()
                + self.gravity_body_force())


class SMEInviscid(INSModel):
    """SME without viscosity."""

    projectable = True

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            HydrostaticPressure, Inviscid, DepthIntegrate,
            ApplyKinematicBCs, StressFreeSurface,
            ZeroAtmosphericPressure, SimplifyIntegrals,
        )
        super().derive_model()
        s = self.state

        self._system.equations["z_momentum"] = (
            self._system.equations["z_momentum"]
            .apply(hydrostatic_scaling(s))
            .simplify()
        )
        self.apply(HydrostaticPressure(s))
        self._system.remove_equation("z_momentum")

        self.apply(DepthIntegrate(s))
        self.apply(ApplyKinematicBCs(s))
        self.apply(StressFreeSurface(s))
        self.apply(ZeroAtmosphericPressure(s))
        self.apply(SimplifyIntegrals(s))
        self.apply(Inviscid(s))

    def source(self):
        return self.gravity_body_force()


# =============================================================================
# SMEModelTagged  — REMOVED.
#
# An earlier iteration of this file implemented SMEModelTagged by reading
# DerivedModel.flux / hydrostatic_pressure / nonconservative_matrix and
# SMEModel.source (all hand-coded basis-matrix operators) and wrapping
# them symbolically, which violated the "everything comes from
# incompressible Navier-Stokes" constraint.  It has been removed.
#
# A proper implementation must derive tagged scalar equations from the
# symbolic INS chain (FullINS → hydrostatic → DepthIntegrate → ApplyKinematicBCs
# → StressFreeSurface → ZeroAtmosphericPressure → SimplifyIntegrals → Newtonian)
# followed by ``Expression.project_onto_basis`` per test mode and the
# subsequent substitutions required to land in state-symbol space
# (surface/bottom velocity evaluations, Function → Symbol for h, b, etc.).
# That work is tracked; it is not a drop-in of the hand-coded reference.
# =============================================================================
