"""Scalar advection and advection-diffusion models.

Works in 1D, 2D, and 3D.
"""

import sympy as sp
from sympy import Matrix
import param

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray


class Advection(Model):
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
        p = self._parameter_symbols
        dim = self.dimension
        F = Matrix.zeros(1, dim)
        a_syms = list(p.values())
        for d in range(dim):
            F[0, d] = a_syms[d] * u
        return ZArray(F)

    def eigenvalues(self):
        p = self._parameter_symbols
        n = self.normal
        a_syms = list(p.values())
        ev = sum(a_syms[d] * n[d] for d in range(self.dimension))
        return ZArray([ev])


class AdvectionDiffusion(Model):
    """Scalar advection-diffusion: du/dt + a · ∇u = ∇ · (ν ∇u).

    The diffusive flux is F_diff = -ν * ∇u (isotropic).
    """

    def __init__(self, dimension=1, nu=1e-3, **kwargs):
        a_names = ["a_x", "a_y", "a_z"][:dimension]
        param_dict = {name: (1.0, "positive") for name in a_names}
        param_dict["nu"] = (nu, "positive")

        super().__init__(
            dimension=dimension,
            variables=["u"],
            parameters=param_dict,
            eigenvalue_mode="symbolic",
            **kwargs,
        )

    def flux(self):
        u = self.variables[0]
        p = self._parameter_symbols
        dim = self.dimension
        F = Matrix.zeros(1, dim)
        a_syms = [p[k] for k in list(p.keys()) if k.startswith("a_")]
        for d in range(dim):
            F[0, d] = a_syms[d] * u
        return ZArray(F)

    def diffusion_matrix(self):
        """Isotropic Fickian diffusion: ``A[0, 0, d, d] = ν``.

        Shape ``(1, 1, dim, dim)``.  Together with the operator-form
        contraction ``F_diff[i, d] = Σ_{j, e} A[i, j, d, e] · ∂_e Q[j]``
        this gives ``F_diff[0, d] = ν · ∂_d u`` and the PDE residual
        contribution ``-ν · Δu`` — i.e. classical advection-diffusion.
        """
        p = self._parameter_symbols
        dim = self.dimension
        nu = p.nu
        A = sp.MutableDenseNDimArray.zeros(1, 1, dim, dim)
        for d in range(dim):
            A[0, 0, d, d] = nu
        return ZArray(A)

    def eigenvalues(self):
        p = self._parameter_symbols
        n = self.normal
        a_syms = [p[k] for k in list(p.keys()) if k.startswith("a_")]
        ev = sum(a_syms[d] * n[d] for d in range(self.dimension))
        return ZArray([ev])
