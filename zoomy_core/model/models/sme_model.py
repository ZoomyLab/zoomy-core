"""SME — Shallow Moment Equations (hydrostatic).

Derivation graph::

    INSModel
      |  apply(HydrostaticPressure)
      |  apply(DepthIntegrate)        — Leibniz rule, boundary values remain
      |  apply(ApplyKinematicBCs)     — w terms cancel with Leibniz terms
      |  apply(StressFreeSurface)     — τ·n|_surface = 0
      |  apply(ZeroAtmosphericPressure)
      |  apply(SimplifyIntegrals)     — evaluate constant/zero integrals
      |  apply(Newtonian)             — τ → ν expressions
      v
    SMEModel  (solver-ready)
"""

from zoomy_core.model.models.derived_model import DerivedModel


class INSModel(DerivedModel):
    """Root: the full 2D incompressible Navier-Stokes."""

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import StateSpace, FullINS
        state = StateSpace(dimension=2)
        ins = FullINS(state)
        self._init_system("INS", {
            "continuity": ins.continuity,
            "x_momentum": ins.x_momentum,
        }, state)


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
        self.apply(HydrostaticPressure(self.state))
        self.apply(DepthIntegrate(self.state))
        self.apply(ApplyKinematicBCs(self.state))
        self.apply(StressFreeSurface(self.state))
        self.apply(ZeroAtmosphericPressure(self.state))
        self.apply(SimplifyIntegrals(self.state))
        self.apply(Newtonian(self.state))

    def source(self):
        return self.newtonian_viscosity() + self.navier_slip()


class SMEInviscid(INSModel):
    """SME without viscosity — pure hyperbolic shallow water moments."""

    projectable = True

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            HydrostaticPressure, Inviscid, DepthIntegrate,
            ApplyKinematicBCs, StressFreeSurface,
            ZeroAtmosphericPressure, SimplifyIntegrals,
        )
        super().derive_model()
        self.apply(HydrostaticPressure(self.state))
        self.apply(DepthIntegrate(self.state))
        self.apply(ApplyKinematicBCs(self.state))
        self.apply(StressFreeSurface(self.state))
        self.apply(ZeroAtmosphericPressure(self.state))
        self.apply(SimplifyIntegrals(self.state))
        self.apply(Inviscid(self.state))
