"""Poisson model — solver-facing operator form.

State ``Q = (u)``, residual

    M ∂_t Q + ∇·(F + P) + B·∂Q − ∇·(A:∇Q) − S = 0

with ``M = 0`` (purely algebraic), ``F = P = B = 0``, ``A[0,0,d,d] = 1``
(isotropic Laplacian), ``S = -f(x)``.  This reduces to ``Δu = f``.

A pure operator-form companion to :class:`ScalarPoissonGalerkin` (which
is the symbolic-FEM weak-form variant); this class is the one solvers
that consume the SystemModel operator surface dispatch on.

Dimension-generic in 1D, 2D, 3D.  ``f`` is a free :class:`~sympy.Function`
on the spatial coordinates, exposed as an aux variable so a runtime can
plug in a per-cell callable.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray


class Poisson(Model):
    """Poisson :math:`\\Delta u = f`.

    State ``Q = (u)``, source ``f(x)`` carried as an aux Function so the
    runtime can plug in a per-cell callable.  Mass matrix is identically
    zero — the equation is purely algebraic, dispatched as an elliptic
    solve.
    """

    def __init__(self, dimension=1, **kwargs):
        super().__init__(
            dimension=dimension,
            variables=["u"],
            parameters={},
            eigenvalue_mode="symbolic",
            **kwargs,
        )

    def mass_matrix(self):
        """``M = 0``: no time derivative, purely algebraic."""
        return ZArray.zeros(self.n_variables, self.n_variables)

    def diffusion_matrix(self):
        """Isotropic Laplacian: ``A[0, 0, d, d] = 1``."""
        dim = self.dimension
        A = sp.MutableDenseNDimArray.zeros(1, 1, dim, dim)
        for d in range(dim):
            A[0, 0, d, d] = sp.S.One
        return ZArray(A)

    def source(self):
        """``S = -f(x)`` so the residual reads ``-Δu + f = 0`` ⇒
        ``Δu = f``.  ``f`` is a free symbolic Function on the spatial
        coordinates.
        """
        coords = self._coords()
        f = sp.Function("f", real=True)(*coords)
        return ZArray([-f])

    def eigenvalues(self):
        """No hyperbolic transport ⇒ spectrum is identically zero."""
        return ZArray([sp.S.Zero])

    def _coords(self):
        names = ("x", "y", "z")[: self.dimension]
        return [sp.Symbol(n, real=True) for n in names]
