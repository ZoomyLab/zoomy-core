"""Dimension-generic shallow-water-equation model on the SystemModel pipeline.

State vector ``Q = (h, hu_0, hu_1, …, hu_{D-1})`` for ``D`` horizontal
dimensions.  Bathymetry ``b`` and its gradient ``b_x, [b_y]`` are aux
variables (carried per-cell, not in the state).  Friction (Manning) and
isotropic eddy viscosity are switchable kwargs.

The Model is index-agnostic: every name resolution goes through
``self.variables`` / ``self.aux_variables`` so the SystemModel extracted
via ``SystemModel.from_model`` carries no SWE-specific field-ordering
assumptions and is consumable by any structure-agnostic solver (numpy
FVM, Firedrake DG, …).

This replaces the 2D-hardcoded ``ShallowWater2D`` from
``shallow_water.py``.  Conventional state names are kept (``hu``, ``hv``,
``hw`` in 2D / 3D) for readability and to match the existing test
fixtures.
"""

from __future__ import annotations

import sympy as sp
from sympy import sqrt

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray


_MOM_NAMES = ("hu", "hv", "hw")
_COORD_NAMES = ("x", "y", "z")


class SWE(Model):
    """Dim-generic shallow-water with Manning friction and depth-weighted eddy viscosity.

    Variables
    ---------
    h, hu, [hv], [hw] : conservative state (1 + ``dimension`` entries).

    Aux variables
    -------------
    b      : bathymetry (a free Function on the mesh).
    b_x, [b_y], [b_z] : bathymetry gradient components.

    Parameters
    ----------
    dimension : 1 or 2 (3D unsupported — physical SWE is depth-averaged).
    g            : gravity (default 9.81)
    manning_n    : Manning roughness (default 0.0 — frictionless)
    nu           : kinematic viscosity (default 0.0 — non-viscous)
    """

    def __init__(self, *, dimension=2, g=9.81, manning_n=0.0, nu=0.0,
                 **kwargs):
        if dimension not in (1, 2):
            raise ValueError(
                f"SWE: dimension must be 1 or 2 (3D shallow water is not "
                f"depth-averaged); got {dimension}"
            )

        mom_names = list(_MOM_NAMES[:dimension])
        b_grad_names = [f"b_{c}" for c in _COORD_NAMES[:dimension]]
        var_names = ["h"] + mom_names
        aux_names = ["b"] + b_grad_names

        param_dict = {
            "g":         (float(g),         "positive"),
            "manning_n": (float(manning_n), "non-negative"),
            "nu":        (float(nu),        "non-negative"),
        }
        super().__init__(
            dimension=dimension,
            variables=var_names,
            aux_variables=aux_names,
            parameters=param_dict,
            eigenvalue_mode="symbolic",
            **kwargs,
        )

    # ── Operators ──────────────────────────────────────────────────────

    def _state_names(self):
        """[``h``, ``hu``, [``hv``], [``hw``]] in canonical order."""
        return ["h"] + list(_MOM_NAMES[: self.dimension])

    def flux(self):
        """``F[0, d] = hu_d``, ``F[i, d] = hu_{i-1} hu_d / h + δ ½ g h²``."""
        v = self.variables
        p = self._parameter_symbols
        D = self.dimension
        names = self._state_names()
        h = v.h
        momenta = [v[name] for name in names[1:]]

        F = sp.zeros(D + 1, D)
        # Mass row.
        for d in range(D):
            F[0, d] = momenta[d]
        # Momentum rows.
        for i in range(D):
            for d in range(D):
                F[i + 1, d] = momenta[i] * momenta[d] / h
                if i == d:
                    F[i + 1, d] += sp.Rational(1, 2) * p.g * h ** 2
        return ZArray(F)

    def source(self):
        """Bathymetry slope + Manning bed friction.

        ``S[0] = 0``,
        ``S[i+1] = -g h ∂_{x_i} b - g n²_M ·  ‖u‖ · u_i / h^{7/3}`` for
        ``i = 0..D-1``.  When ``manning_n = 0`` (default) only the slope
        survives.
        """
        v = self.variables
        a = self.aux_variables
        p = self._parameter_symbols
        D = self.dimension
        names = self._state_names()
        h = v.h
        momenta = [v[name] for name in names[1:]]
        b_grad = [a[f"b_{c}"] for c in _COORD_NAMES[:D]]

        # Velocity magnitude for Manning.
        u_components = [m / h for m in momenta]
        u_mag = sqrt(sum(u * u for u in u_components))
        manning_coeff = (-p.g * p.manning_n ** 2 * u_mag
                         / h ** sp.Rational(7, 3))

        S = [sp.S.Zero]
        for i in range(D):
            S_i = -p.g * h * b_grad[i] + manning_coeff * momenta[i]
            S.append(S_i)
        return ZArray(S)

    def diffusion_matrix(self):
        """Depth-weighted eddy viscosity on momentum rows only.

        ``A[i+1, i+1, d, d] = ν · h`` for ``i = 0..D-1``, ``d = 0..D-1``;
        every other entry is zero.  Contracting with ``∇Q`` gives
        ``F_diff[i+1, d] = ν h · ∂_d hu_i`` which vanishes at wet/dry
        interfaces (``h → 0``).
        """
        v = self.variables
        p = self._parameter_symbols
        D = self.dimension
        h = v.h
        nu = p.nu

        A = sp.MutableDenseNDimArray.zeros(D + 1, D + 1, D, D)
        for i in range(D):                     # momentum row
            for d in range(D):                 # spatial direction
                A[i + 1, i + 1, d, d] = nu * h
        return ZArray(A)

    def eigenvalues(self):
        """1D: ``u·n ± c``.  2D: ``u·n − c, u·n, u·n + c``.

        ``c = √(g h)``, ``u·n = Σ_d (hu_d/h) · n_d``.
        """
        v = self.variables
        p = self._parameter_symbols
        n = self.normal
        D = self.dimension
        h = v.h
        momenta = [v[name] for name in self._state_names()[1:]]
        un = sum(momenta[d] / h * n[f"n{d}"] for d in range(D))
        c = sqrt(p.g * h)

        if D == 1:
            return ZArray([un - c, un + c])
        # D == 2: include contact wave.
        return ZArray([un - c, un, un + c])
