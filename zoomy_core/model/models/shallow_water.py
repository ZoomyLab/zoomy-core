"""2D shallow-water-equation model on the SystemModel pipeline.

State vector ``Q = (h, hu, hv)`` (no bathymetry in the state — it is
carried as an aux field together with its gradient).  Friction and
diffusion are switchable from the constructor.  The Model is index-
agnostic: every name resolution goes through ``self.variables`` /
``self.aux_variables`` rather than hardcoded slots, so the SystemModel
extracted via :meth:`SystemModel.from_model` carries no SWE-specific
field-ordering assumptions and is consumable by any structure-agnostic
solver (numpy FVM, Firedrake DG, …).
"""

from __future__ import annotations

import sympy as sp
from sympy import Matrix, sqrt

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray


class ShallowWater2D(Model):
    """2D shallow-water with Manning friction and depth-weighted eddy viscosity.

    Variables
    ---------
    h, hu, hv : conservative state.

    Aux variables
    -------------
    b      : bathymetry (a free Function on the mesh).
    b_x    : bathymetry gradient (``∂_x b``); equivalent for ``b_y``.

    Parameters
    ----------
    g            : gravity (default 9.81)
    manning_n    : Manning roughness (default 0.0 — frictionless)
    nu           : kinematic viscosity (default 0.0 — non-viscous)
    """

    def __init__(self, *, g=9.81, manning_n=0.0, nu=0.0, **kwargs):
        param_dict = {
            "g":         (float(g),         "positive"),
            "manning_n": (float(manning_n), "non-negative"),
            "nu":        (float(nu),        "non-negative"),
        }
        super().__init__(
            dimension=2,
            variables=["h", "hu", "hv"],
            aux_variables=["b", "b_x", "b_y"],
            parameters=param_dict,
            eigenvalue_mode="symbolic",
            **kwargs,
        )

    # ── Operators ──────────────────────────────────────────────────────

    def flux(self):
        """``F = [ hu                       hv                      ]
                  [ hu²/h + g h²/2          hu hv / h               ]
                  [ hu hv / h               hv²/h + g h²/2          ]``"""
        v = self.variables
        p = self._parameter_symbols
        h, hu, hv = v.h, v.hu, v.hv
        F = Matrix.zeros(3, 2)
        # mass
        F[0, 0] = hu
        F[0, 1] = hv
        # x-momentum
        F[1, 0] = hu * hu / h + sp.Rational(1, 2) * p.g * h ** 2
        F[1, 1] = hu * hv / h
        # y-momentum
        F[2, 0] = hu * hv / h
        F[2, 1] = hv * hv / h + sp.Rational(1, 2) * p.g * h ** 2
        return ZArray(F)

    def source(self):
        """Source = bathymetry slope + Manning bed friction.

        - Slope: ``S[hu] = -g h ∂_x b``, ``S[hv] = -g h ∂_y b``.
        - Manning: ``S[hu_i] -= g · n² · u_i · ‖u‖ / h^(7/3)``
          (zero when ``manning_n = 0``, the default).

        Returned as a rank-1 ``ZArray`` of length ``n_variables`` to
        match the operator-API convention shared with the basemodel
        default and the existing chain-derived models.
        """
        v = self.variables
        a = self.aux_variables
        p = self._parameter_symbols
        h, hu, hv = v.h, v.hu, v.hv
        # Bathymetry slope contributions.
        S_h = sp.S.Zero
        S_hu = -p.g * h * a.b_x
        S_hv = -p.g * h * a.b_y
        # Manning friction (only active when manning_n != 0).
        u = hu / h
        w = hv / h
        u_mag = sqrt(u * u + w * w)
        manning_coeff = -p.g * p.manning_n ** 2 * u_mag / h ** sp.Rational(7, 3)
        S_hu = S_hu + manning_coeff * hu
        S_hv = S_hv + manning_coeff * hv
        return ZArray([S_h, S_hu, S_hv])

    def diffusion_matrix(self):
        """Depth-weighted eddy viscosity on momentum rows only.

        ``A[hu, hu, d, d] = ν · h`` and ``A[hv, hv, d, d] = ν · h``;
        every other entry of the ``(3, 3, 2, 2)`` tensor is zero.  The
        contraction with ``∇Q`` produces ``F_diff[hu, d] = ν h · ∂_d hu``
        which vanishes naturally at wet/dry interfaces (``h → 0``).
        """
        v = self.variables
        p = self._parameter_symbols
        h = v.h
        nu = p.nu
        A = sp.MutableDenseNDimArray.zeros(3, 3, 2, 2)
        for i_row in (1, 2):              # hu, hv
            for d in (0, 1):              # x, y
                A[i_row, i_row, d, d] = nu * h
        return ZArray(A)

    def eigenvalues(self):
        """Eigenvalues of the normal-projected quasilinear matrix.

        For the 2D SWE these are
        ``λ = u·n ± √(g h)``, ``u·n`` (contact).
        """
        v = self.variables
        p = self._parameter_symbols
        n = self.normal
        h, hu, hv = v.h, v.hu, v.hv
        u = hu / h
        w = hv / h
        un = u * n.n0 + w * n.n1
        c = sqrt(p.g * h)
        return ZArray([un - c, un, un + c])
