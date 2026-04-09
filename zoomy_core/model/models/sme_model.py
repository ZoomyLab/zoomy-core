"""SME — Shallow Moment Equations (hydrostatic).

Derivation::

    INSModel (continuity + x_momentum + z_momentum)
      |  z_momentum.apply({w: 0, τ_zz: 0, τ_zx: 0})  — hydrostatic scaling
      |  → z_momentum reduces to ∂p/∂z/ρ + g = 0
      |  → integrate: p = p_atm + ρg(η - z)
      |  system.apply({p: p_hydro})  — substitute into x_momentum
      |  remove z_momentum (consumed)
      |  apply(DepthIntegrate + KinematicBCs + ...)
      v
    SMEModel  (solver-ready: continuity + x_momentum)
"""

import sympy as sp
from sympy import Function, S

from zoomy_core.model.models.derived_model import DerivedModel


class INSModel(DerivedModel):
    """Root: the full 2D incompressible Navier-Stokes (all equations)."""

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import StateSpace, FullINS
        state = StateSpace(dimension=2)
        ins = FullINS(state)
        self._init_system("INS", {
            "continuity": ins.continuity,
            "x_momentum": ins.x_momentum,
            "z_momentum": ins.z_momentum,
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
        s = self.state

        # 1. Hydrostatic scaling on z-momentum only
        self._system.equations["z_momentum"] = (
            self._system.equations["z_momentum"]
            .apply({s.w: S.Zero, s.tau["zz"]: S.Zero, s.tau["zx"]: S.Zero})
            .simplify()
        )
        # z-momentum is now: ∂p/∂z/ρ + g = 0 → p = p_atm + ρg(η - z)

        # 2. Apply derived pressure to all equations
        self.apply(HydrostaticPressure(s))

        # 3. Remove z-momentum (consumed by the derivation)
        del self._system.equations["z_momentum"]

        # 4. Depth integrate + BCs + closures
        self.apply(DepthIntegrate(s))
        self.apply(ApplyKinematicBCs(s))
        self.apply(StressFreeSurface(s))
        self.apply(ZeroAtmosphericPressure(s))
        self.apply(SimplifyIntegrals(s))
        self.apply(Newtonian(s))

    def source(self):
        return self.newtonian_viscosity() + self.navier_slip()


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
            .apply({s.w: S.Zero, s.tau["zz"]: S.Zero, s.tau["zx"]: S.Zero})
            .simplify()
        )
        self.apply(HydrostaticPressure(s))
        del self._system.equations["z_momentum"]

        self.apply(DepthIntegrate(s))
        self.apply(ApplyKinematicBCs(s))
        self.apply(StressFreeSurface(s))
        self.apply(ZeroAtmosphericPressure(s))
        self.apply(SimplifyIntegrals(s))
        self.apply(Inviscid(s))
