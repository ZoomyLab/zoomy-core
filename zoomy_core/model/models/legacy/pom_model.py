"""POM (Princeton Ocean Model) on the SystemModel pipeline.

Depth-averaged (level-0) POM with every Δ vs. K&T (2019) except
Coriolis (Δ 1) and Salinity (Δ 3) wired through.  State vector::

    Q = (h, hu, [hv,] hT, hq2, hq2l)

``hT`` is the depth-weighted potential temperature; ``hq2`` and
``hq2l`` are the prognostic MY-2.5 turbulence states (``q²`` and
``q²ℓ``, both depth-weighted).  Bathymetry ``b`` and its gradient
``b_x[, b_y]`` live in ``aux_variables``.

The four enabled deltas (§3 of `008_pom_3d_reference.py`) populate
the standard SystemModel slots:

============== ===========================================================
Δ              SystemModel slot
============== ===========================================================
Δ 2 (T)        flux (T advection), diffusion_matrix (K_H · h on hT)
Δ 4 (MY-2.5)   flux (q²/q²ℓ advection), source (production / dissipation),
               diffusion_matrix (K_q · h on hq2/hq2l), and state-dependent
               K_M · h coupled into the hu diffusion row.
Δ 5 (Smag.)    diffusion_matrix (constant A_M · h on hu, [hv])
Δ 6 (baro.)    flux (gh²/2 augmented by g h² α_T (T − T_ref)/2)
============== ===========================================================

Bathymetry slope -g h ∂_x b enters the source slot (consistent with
``ShallowWater2D``).  Eigenvalues are numerical (the L=0 5-state
quasilinear matrix has no useful closed form).

This depth-averaged limit collapses the MY-2.5 closure under
neutral stratification (``ρ ≡ ρ_o``) to closed-form Galperin
constants S_M = A1·(1 − 3 C1 − 6 A1/B1), S_H = A2·(1 − 6 A1/B1)
(`Mellor04 (42a, 42b)`).  The full G_H ≠ 0 branch is recovered when
∂_z ρ is restored — which is a higher-level (L≥1) extension.

Coriolis (Δ 1) and Salinity (Δ 3) are deliberately not declared.
They become single-method additions on a future ``POMFull`` subclass.
"""

from __future__ import annotations

import sympy as sp
from sympy import Matrix, MutableDenseNDimArray, Rational, sqrt

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray


class POMDepthAveraged(Model):
    """Depth-averaged POM (level-0 SME limit) with T, MY-2.5,
    Smagorinsky and linear-EOS Boussinesq baroclinic pressure.

    Parameters
    ----------
    g          : gravity (9.81)
    rho_o      : reference density (1025 kg/m³)
    alpha_T    : thermal expansion coefficient (2e-4 / K, ocean default)
    T_ref      : reference temperature (15 °C)
    A_M        : Smagorinsky-style horizontal eddy viscosity baseline (m²/s)
    nu         : background molecular viscosity (m²/s) — keeps the
                 diffusion path live even with A_M = K_* = 0
    B1, A1, A2, C1, E1, Sq, kappa
                : Mellor–Yamada level 2½ closure constants
                  (BM87 (18), (19))
    l_floor    : floor on the master length scale ℓ (m)
    q2_floor   : floor on q²
    dimension  : 1 or 2; declares (hv, b_y) when 2.
    """

    def __init__(
        self,
        *,
        g=9.81, rho_o=1025.0,
        alpha_T=2.0e-4, T_ref=15.0,
        A_M=1.0e-3, nu=1.0e-6,
        B1=16.6, A1=0.92, A2=0.74, C1=0.08,
        E1=1.8, Sq=0.20, kappa=0.4,
        l_floor=1.0e-2, q2_floor=1.0e-8,
        dimension=1,
        **kwargs,
    ):
        param_dict = {
            "g":        (float(g),       "positive"),
            "rho_o":    (float(rho_o),   "positive"),
            "alpha_T":  (float(alpha_T), "non-negative"),
            "T_ref":    (float(T_ref),   "real"),
            "A_M":      (float(A_M),     "non-negative"),
            "nu":       (float(nu),      "non-negative"),
            "B1":       (float(B1),      "positive"),
            "A1":       (float(A1),      "positive"),
            "A2":       (float(A2),      "positive"),
            "C1":       (float(C1),      "positive"),
            "E1":       (float(E1),      "positive"),
            "Sq":       (float(Sq),      "positive"),
            "kappa":    (float(kappa),   "positive"),
            "l_floor":  (float(l_floor), "positive"),
            "q2_floor": (float(q2_floor),"positive"),
        }
        if dimension == 1:
            variables = ["h", "hu", "hT", "hq2", "hq2l"]
            aux_variables = ["b", "b_x"]
        elif dimension == 2:
            variables = ["h", "hu", "hv", "hT", "hq2", "hq2l"]
            aux_variables = ["b", "b_x", "b_y"]
        else:
            raise ValueError(f"dimension must be 1 or 2, got {dimension!r}")

        super().__init__(
            dimension=dimension,
            variables=variables,
            aux_variables=aux_variables,
            parameters=param_dict,
            eigenvalue_mode="numerical",
            **kwargs,
        )

    # ── Closure helpers ────────────────────────────────────────────────

    def _stability_funcs(self):
        """Galperin (1988) neutral-stratification stability functions.

        With ρ ≡ ρ_o the buoyancy argument G_H vanishes and the
        Galperin 2×2 quasi-equilibrium system collapses to closed-form
        constants (Mellor04 §11; canonical values S_M^0 ≈ 0.39,
        S_H^0 ≈ 0.49 quoted in MY82 and Mellor (1989))."""
        p = self._parameter_symbols
        S_M = p.A1 * (1 - 3 * p.C1 - 6 * p.A1 / p.B1)
        S_H = p.A2 * (1 - 6 * p.A1 / p.B1)
        return S_M, S_H

    def _length_and_q(self):
        """Recover ℓ and q from the prognostic state, clamped to keep
        the mixing coefficients well defined when (h, q²) approach zero.

        ``q² = hq²/h``, ``ℓ = hq²ℓ / hq²``; both clamped below by the
        floors ``q2_floor`` and ``l_floor`` (sympy ``Max`` lowers to a
        ``np.maximum`` per cell)."""
        p = self._parameter_symbols
        v = self.variables
        h, hq2, hq2l = v.h, v.hq2, v.hq2l
        q2 = hq2 / h
        l = sp.Max(hq2l / sp.Max(hq2, p.q2_floor * h), p.l_floor)
        q = sqrt(sp.Max(q2, p.q2_floor))
        return l, q

    def _mixing_coefficients(self):
        """K_M, K_H, K_q = ℓ q (S_M, S_H, S_q) (Mellor04 (12))."""
        p = self._parameter_symbols
        S_M, S_H = self._stability_funcs()
        l, q = self._length_and_q()
        K_M = l * q * S_M
        K_H = l * q * S_H
        K_q = l * q * p.Sq
        return K_M, K_H, K_q

    # ── Operators ──────────────────────────────────────────────────────

    def flux(self):
        """Conservative flux: SWE + baroclinic-augmented pressure + tracers.

        Mass:           hu  (and hv in 2D).
        x-momentum:     hu·u + g h²/2 + (g/2) α_T (T − T_ref) h²
                                            ↑ Δ 6 baroclinic (L=0)
        y-momentum:     hu·v   /   hv·v + g h²/2 + baro    (only in 2D)
        Tracers (hT, hq², hq²ℓ): each advected as ``state · u_d``.
        """
        v = self.variables
        p = self._parameter_symbols
        n_eq = self.n_variables
        d = self.dimension
        h, hu = v.h, v.hu
        u = hu / h
        hv = getattr(v, "hv", None)
        w = hv / h if hv is not None else None
        hT, hq2, hq2l = v.hT, v.hq2, v.hq2l
        T_anom = hT / h - p.T_ref
        bp = Rational(1, 2) * p.g * h**2 * p.alpha_T * T_anom  # Δ 6
        gp = Rational(1, 2) * p.g * h**2                       # barotropic
        F = Matrix.zeros(n_eq, d)
        # ── x-direction ──
        F[0, 0] = hu
        F[1, 0] = hu * u + gp + bp
        if hv is None:
            F[2, 0] = hT * u
            F[3, 0] = hq2 * u
            F[4, 0] = hq2l * u
        else:
            F[2, 0] = hu * w
            F[3, 0] = hT * u
            F[4, 0] = hq2 * u
            F[5, 0] = hq2l * u
            # ── y-direction (only in 2D) ──
            F[0, 1] = hv
            F[1, 1] = hu * w
            F[2, 1] = hv * w + gp + bp
            F[3, 1] = hT * w
            F[4, 1] = hq2 * w
            F[5, 1] = hq2l * w
        return ZArray(F)

    def source(self):
        """Bathymetry slope (-g h ∂_x b) + MY-2.5 production / dissipation.

        Tagged into the **implicit** source slot (default) so the
        IMEX backend treats the stiff q³/(B₁ℓ) dissipation
        implicitly at Q^{n+1}.

        Shear estimate at L=0 follows the column-averaged form
        ``∂_z u ≈ u/h`` (linear-profile assumption, consistent with
        Mellor04 §10's depth-averaged MY-2.5 closure).
        """
        v = self.variables
        a = self.aux_variables
        p = self._parameter_symbols
        h, hu = v.h, v.hu
        hv = getattr(v, "hv", None)
        u = hu / h
        w = (hv / h) if hv is not None else sp.S.Zero
        hq2, hq2l = v.hq2, v.hq2l
        l, q = self._length_and_q()
        K_M, K_H, K_q = self._mixing_coefficients()
        shear_sq = (u / h)**2 + (w / h)**2
        # Production
        prod_q2  = 2 * K_M * shear_sq * h
        prod_q2l = p.E1 * l * K_M * shear_sq * h
        # Dissipation (MY-2.5 ε term; W̃ → 1 in depth-averaged form)
        dissip_q2  = -2 * h * q**3 / (p.B1 * l)
        dissip_q2l = -h * q**3 / p.B1
        # Bathymetry slope
        S = [sp.S.Zero] * self.n_variables
        S[1] = -p.g * h * a.b_x                      # hu row
        if hv is not None:
            S[2] = -p.g * h * a.b_y                  # hv row
            S[4] = prod_q2 + dissip_q2               # hq2 row
            S[5] = prod_q2l + dissip_q2l             # hq2l row
        else:
            S[3] = prod_q2 + dissip_q2               # hq2 row
            S[4] = prod_q2l + dissip_q2l             # hq2l row
        return ZArray(S)

    def diffusion_matrix(self):
        """Horizontal eddy diffusion — Smagorinsky baseline A_M plus
        MY-2.5 state-dependent K_*, multiplied by ``h`` so the flux
        vanishes naturally at wet/dry interfaces (h → 0).

        - Momentum rows (hu, [hv]):   ν_mom · h  with  ν_mom = A_M + nu + K_M
        - Temperature row (hT):       (K_H + nu) · h
        - Turbulence rows (hq², hq²ℓ): (K_q + nu) · h

        State-dependent entries flow through ``SystemModel.from_model``
        intact; the numpy FVM solver currently lowers to a scalar
        ``nu`` per variable (degenerating K_M → constant) — the IMEX
        and Firedrake backends consume the full state-dependent
        ``A(Q, Qaux, p)`` directly.
        """
        v = self.variables
        p = self._parameter_symbols
        n = self.n_variables
        d = self.dimension
        h = v.h
        K_M, K_H, K_q = self._mixing_coefficients()
        A = MutableDenseNDimArray.zeros(n, n, d, d)
        nu_mom = (p.A_M + p.nu + K_M) * h
        nu_T   = (K_H + p.nu) * h
        nu_q   = (K_q + p.nu) * h
        # Map the row index of each variable.
        hv_present = hasattr(v, "hv")
        if hv_present:
            i_hu, i_hv, i_hT, i_hq2, i_hq2l = 1, 2, 3, 4, 5
        else:
            i_hu, i_hT, i_hq2, i_hq2l = 1, 2, 3, 4
        for axis in range(d):
            A[i_hu, i_hu, axis, axis] = nu_mom
            if hv_present:
                A[i_hv, i_hv, axis, axis] = nu_mom
            A[i_hT, i_hT, axis, axis] = nu_T
            A[i_hq2, i_hq2, axis, axis] = nu_q
            A[i_hq2l, i_hq2l, axis, axis] = nu_q
        return ZArray(A)

    def eigenvalues(self):
        """Numerical eigenvalues — declared but never used because
        ``eigenvalue_mode="numerical"`` skips the symbolic path in
        ``SystemModel.from_model`` and routes to
        ``np.linalg.eigvals(QL_n + ε I)`` at every face evaluation."""
        return ZArray.zeros(self.n_variables)
