"""Simple scalar advection: du/dt + a · ∇u = 0.

Works in 1D, 2D, and 3D. Advection velocity `a` is a parameter vector.
"""

import sympy as sp
from sympy import Matrix
import param

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray


class ScalarAdvection(Model):
    """Scalar advection du/dt + a · ∇u = 0."""

    def __init__(self, dimension=1, **kwargs):
        a_names = ["a_x", "a_y", "a_z"][:dimension]
        param_dict = {name: (1.0, "positive") for name in a_names}

        super().__init__(
            dimension=dimension,
            variables=["u"],
            parameters=param_dict,
            eigenvalue_mode="symbolic",
            **kwargs,
        )

    def flux(self):
        u = self.variables[0]
        p = self.parameters
        dim = self.dimension
        F = Matrix.zeros(1, dim)
        a_syms = list(p.values())
        for d in range(dim):
            F[0, d] = a_syms[d] * u
        return ZArray(F)

    def eigenvalues(self):
        p = self.parameters
        n = self.normal
        a_syms = list(p.values())
        ev = sum(a_syms[d] * n[d] for d in range(self.dimension))
        return ZArray([ev])
