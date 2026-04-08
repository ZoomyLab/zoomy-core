"""SME — Shallow Moment Equations (hydrostatic).

Derivation graph::

    INSModel
      |  apply(hydrostatic), apply(Newtonian), apply(DepthIntegrate)
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
    """Shallow Moment Equations — hydrostatic + Newtonian, depth-integrated."""

    projectable = True

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            HydrostaticPressure, Newtonian, DepthIntegrate,
        )
        super().derive_model()
        self.apply(HydrostaticPressure(self.state))
        self.apply(Newtonian(self.state))
        self.apply(DepthIntegrate(self.state))

    def source(self):
        return self.newtonian_viscosity() + self.navier_slip()


class SMEInviscid(INSModel):
    """SME without viscosity — pure hyperbolic shallow water moments."""

    projectable = True

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            HydrostaticPressure, Inviscid, DepthIntegrate,
        )
        super().derive_model()
        self.apply(HydrostaticPressure(self.state))
        self.apply(Inviscid(self.state))
        self.apply(DepthIntegrate(self.state))
