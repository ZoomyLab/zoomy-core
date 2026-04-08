"""3D Incompressible Navier-Stokes with Chorin pressure splitting.

The model holds the full INS and provides separate interfaces for
predictor (momentum without pressure), Poisson (pressure), and corrector.

Usage::

    model = INS3DChorin(dimension=3, nu=0.01)
    solver = ProjectionSolver(time_end=1.0, ...)
    Q, p = solver.solve(mesh, model)
"""

import sympy as sp
from sympy import Matrix, Symbol
import param

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray


class INS3DChorin(Model):
    """3D incompressible Navier-Stokes with Chorin projection splitting.

    State vector Q = [u, v, w] (velocity components).
    Pressure is a separate field solved via Poisson equation.

    The model registers:
    - flux() — convective flux (for the predictor)
    - source() — zero (viscosity handled via LSQ Laplacian in solver)
    - eigenvalues() — max(|u|, |v|, |w|) for CFL

    The solver uses LSQ derivatives for:
    - Divergence: ∇·u* → Poisson RHS
    - Laplacian: ∇²p → Poisson LHS
    - Gradient: ∇p → correction
    - Viscous: ν∇²u → predictor source
    """

    nu = param.Number(default=0.01, doc="Kinematic viscosity")

    def __init__(self, dimension=3, nu=0.01, **kwargs):
        n_vars = dimension  # u, v, w (or u, v for 2D)
        var_names = ["u", "v", "w"][:dimension]

        super().__init__(
            dimension=dimension,
            variables=var_names,
            parameters={"nu": nu},
            eigenvalue_mode="symbolic",
            nu=nu,
            **kwargs,
        )

    def flux(self):
        """Convective flux F_d = u_d * Q (advection)."""
        dim = self.dimension
        n = self.n_variables
        F = Matrix.zeros(n, dim)
        q = [self.variables[i] for i in range(n)]
        for i in range(n):
            for d in range(dim):
                F[i, d] = q[i] * q[d]
        return ZArray(F)

    def eigenvalues(self):
        """Eigenvalues for CFL: max velocity component along normal."""
        q = [self.variables[i] for i in range(self.n_variables)]
        n = self.normal
        ev = sum(q[d] * n[d] for d in range(self.dimension))
        # All components have the same wave speed
        return ZArray([ev] * self.n_variables)
