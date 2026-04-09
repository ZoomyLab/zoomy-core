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
    """Dict that drops w, τ_zz, τ_zx from an equation (hydrostatic assumption)."""
    return {state.w: S.Zero, state.tau["zz"]: S.Zero, state.tau["zx"]: S.Zero}


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

        # Hydrostatic: scale z-momentum, derive pressure, apply to system
        self._system.equations["z_momentum"] = (
            self._system.equations["z_momentum"]
            .apply(hydrostatic_scaling(s))
            .simplify()
        )
        self.apply(HydrostaticPressure(s))
        del self._system.equations["z_momentum"]

        # Depth integrate + closures
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
            .apply(hydrostatic_scaling(s))
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
