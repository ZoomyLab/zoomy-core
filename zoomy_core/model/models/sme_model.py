"""SME — Shallow Moment Equations (hydrostatic).

Derivation graph::

    INSModel
      |  apply(HydrostaticPressure)
      |  apply(DepthIntegrate)
      |  apply(KinematicBCSurface)
      |  apply(KinematicBCBottom)
      |  apply(ZeroAtmosphericPressure)
      |  apply(Newtonian)           — material last: keeps τ symbols in intermediate steps
      v
    SMEModel  (solver-ready)

Usage::

    model = SMEModel(level=2)
    model.describe(derivation='mermaid')
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
            KinematicBCBottom, KinematicBCSurface,
            ZeroAtmosphericPressure,
        )
        super().derive_model()
        self.apply(HydrostaticPressure(self.state))
        self.apply(DepthIntegrate(self.state))
        self.apply(KinematicBCSurface(self.state))
        self.apply(KinematicBCBottom(self.state))
        self.apply(ZeroAtmosphericPressure(self.state))
        self.apply(Newtonian(self.state))

    def source(self):
        return self.newtonian_viscosity() + self.navier_slip()


class SMEInviscid(INSModel):
    """SME without viscosity — pure hyperbolic shallow water moments."""

    projectable = True

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            HydrostaticPressure, Inviscid, DepthIntegrate,
            KinematicBCBottom, KinematicBCSurface,
            ZeroAtmosphericPressure,
        )
        super().derive_model()
        self.apply(HydrostaticPressure(self.state))
        self.apply(DepthIntegrate(self.state))
        self.apply(KinematicBCSurface(self.state))
        self.apply(KinematicBCBottom(self.state))
        self.apply(ZeroAtmosphericPressure(self.state))
        self.apply(Inviscid(self.state))
