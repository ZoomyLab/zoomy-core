"""1D vertical Mellor–Yamada level 2½ water column.

Canonical POM physics test: a single 1D vertical column with **no
horizontal extent**, driven by a surface wind stress, with an initial
linear stratification.  This is exactly the configuration of the
Kato & Phillips (1969) annular-tank entrainment experiment that
every MY-closure paper has used for validation since
Mellor & Durbin (1975).

The PDE system on ``z ∈ [0, H]`` (z = 0 bottom, z = H surface) is::

    ∂_t U     = ∂_z (K_M ∂_z U)
    ∂_t T     = ∂_z (K_H ∂_z T)
    ∂_t q²    = ∂_z (K_q ∂_z q²)
              + 2 K_M (∂_z U)²
              + 2 g α_T K_H ∂_z T              ← buoyancy production
              - 2 q³ / (B₁ ℓ)                   ← dissipation
    ∂_t q²ℓ   = ∂_z (K_q ∂_z q²ℓ)
              + E₁ ℓ K_M (∂_z U)²
              + E₁ E₃ ℓ g α_T K_H ∂_z T
              - W̃ q³ / B₁                       ← wall-amplified dissipation

with the canonical Mellor–Yamada closure::

    K_M = ℓ q S_M(G_H),   K_H = ℓ q S_H(G_H),   K_q = ℓ q S_q
    q   = sqrt(q²),       ℓ   = q²ℓ / q²
    G_H = (ℓ² / q²)·(g α_T / 1)·∂_z T            ← stratification arg
    S_M, S_H : Galperin (1988) quasi-equilibrium pair, closed-form
               rational in G_H once the BM87 constants are fixed.
    W̃   = 1 + E₂·(ℓ/(κ L))²                       ← wall proximity

with ``L⁻¹ = (H - z)⁻¹ + z⁻¹`` the harmonically-averaged distance to
the nearer rigid wall.

Boundary conditions (BM87 §2.15 / §2.16, depth-coordinate flavour):

- z = H (surface):  K_M ∂_z U = u_*²,  ∂_z T = 0,
                     q² = B₁^(2/3) u_*²,  q²ℓ = 0.
- z = 0 (bottom):   U = 0 (no-slip),    ∂_z T = 0,
                     q² = B₁^(2/3) u_τb², q²ℓ = 0,
                     with u_τb² recovered self-consistently from
                     the near-bottom shear at the lowest cell.

There is **no horizontal advection** — this is a 1D PARABOLIC
problem (pure diffusion + algebraic source).  Backends that consume
the full state-dependent ``A(Q, Qaux, p)`` diffusion tensor
(``zoomy_firedrake``'s IP-DG/TPFA, ``NumpyRuntimeModel``'s
runtime tensor) propagate the state-dependent K_M, K_H, K_q
correctly.  The legacy numpy FVM ``_build_diffusion_operators``
scalar-``nu`` shortcut is not appropriate for this problem.

Coordinate convention.  We use the depth-coordinate flavour
``z ∈ [0, H]`` (bottom at 0, surface at H), matching zoomy's
ζ-convention up to a domain rescaling.  POM uses
``σ ∈ [-1, 0]`` (surface at 0); the relabel ``z = H·(σ + 1)``
recovers POM's σ for diagnostic post-processing.
"""

from __future__ import annotations

import sympy as sp
from sympy import MutableDenseNDimArray, Rational, sqrt

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray


class MY25Column1D(Model):
    """1D vertical Mellor–Yamada 2½ water column.

    State::

        Q = (U, T, q2, q2l)

    on a 1D mesh in z.  Aux state (auto-exposed by
    ``SystemModel.expose_aux_atoms``) carries the
    derivative atoms ``∂_z U``, ``∂_z T``, ``∂_z q²`` and
    the wall-proximity input position ``z``.

    Parameters
    ----------
    g           : gravity (9.81 m/s²)
    rho_o       : reference density (1025 kg/m³, ocean default)
    alpha_T     : thermal expansion (2e-4 / K — water near 20°C)
    T_ref       : reference temperature (15 °C)
    column_depth: total water depth H (used inside the wall-proximity
                  expression — *not* a coordinate; the mesh sets ``z``)
    u_star      : surface friction velocity (m/s); driven into the
                  surface BC via ``K_M ∂_z U = u_*²``
    B1, A1, A2, C1, E1, E2, E3, Sq, kappa : BM87 (18)/(19) MY-2.5 constants
    l_floor, q2_floor : positivity floors

    The model has no flux (no convective transport in 1D vertical);
    ``flux()`` returns zero.  All physics lives in
    ``diffusion_matrix()`` and ``source()``.
    """

    def __init__(
        self,
        *,
        g=9.81, rho_o=1025.0,
        alpha_T=2.0e-4, T_ref=15.0,
        column_depth=50.0,
        u_star=0.01,
        B1=16.6, A1=0.92, A2=0.74, C1=0.08,
        E1=1.8, E2=1.33, E3=1.0, Sq=0.20, kappa=0.4,
        l_floor=1.0e-2, q2_floor=1.0e-8,
        **kwargs,
    ):
        param_dict = {
            "g":            (float(g),            "positive"),
            "rho_o":        (float(rho_o),        "positive"),
            "alpha_T":      (float(alpha_T),      "non-negative"),
            "T_ref":        (float(T_ref),        "real"),
            "column_depth": (float(column_depth), "positive"),
            "u_star":       (float(u_star),       "non-negative"),
            "B1":           (float(B1),           "positive"),
            "A1":           (float(A1),           "positive"),
            "A2":           (float(A2),           "positive"),
            "C1":           (float(C1),           "positive"),
            "E1":           (float(E1),           "positive"),
            "E2":           (float(E2),           "positive"),
            "E3":           (float(E3),           "real"),
            "Sq":           (float(Sq),           "positive"),
            "kappa":        (float(kappa),        "positive"),
            "l_floor":      (float(l_floor),      "positive"),
            "q2_floor":     (float(q2_floor),     "positive"),
            "nu":           (1.0e-6,              "non-negative"),
        }
        super().__init__(
            dimension=1,
            variables=["U", "T", "q2", "q2l"],
            aux_variables=["z"],
            parameters=param_dict,
            eigenvalue_mode="numerical",
            **kwargs,
        )

    # ── Closure helpers ───────────────────────────────────────────────

    def _length_and_q(self):
        """ℓ = q²ℓ / q²  (clamped); q = sqrt(q²) (clamped)."""
        p = self._parameter_symbols
        v = self.variables
        q2 = v.q2
        q2l = v.q2l
        l = sp.Max(q2l / sp.Max(q2, p.q2_floor), p.l_floor)
        q = sqrt(sp.Max(q2, p.q2_floor))
        return l, q

    def _G_H(self):
        """Buoyancy stratification argument

            G_H = (ℓ² / q²) · (g/ρ_o) · ∂_z ρ
                = -(ℓ² / q²) · g α_T · ∂_z T

        With our linear EOS ρ = ρ_o (1 - α_T (T - T_ref)).  Positive
        in convectively unstable columns (T decreases with depth,
        ∂_z T > 0 ⇒ ρ decreases upward ⇒ stable, G_H < 0).
        """
        p = self._parameter_symbols
        v = self.variables
        space = sp.Symbol("x", real=True)
        T_z = sp.Derivative(v.T, space)
        l, q = self._length_and_q()
        return -(l ** 2 / sp.Max(v.q2, p.q2_floor)) * p.g * p.alpha_T * T_z

    def _stability_funcs(self):
        """Galperin (1988) quasi-equilibrium closure for S_M, S_H —
        rational functions of G_H.  Mellor04 (42a, 42b)::

            S_H [1 - (3 A2 B2 + 18 A1 A2) G_H] = A2 [1 - 6 A1/B1]
            S_M [1 - 9 A1 A2 G_H]
                - S_H (18 A1² + 9 A1 A2) G_H   = A1 [1 - 3 C1 - 6 A1/B1]

        Solving sequentially for S_H then S_M gives closed-form
        rational expressions in G_H.  We clamp G_H from above at
        0.028 (the Galperin instability bound) to stay on the
        physical branch.
        """
        p = self._parameter_symbols
        # B2 isn't a runtime parameter — use the BM87 constant directly.
        B2 = sp.Rational(101, 10)
        G_H = sp.Min(self._G_H(), sp.Rational(28, 1000))  # ≤ 0.028
        # Solve algebraic 2×2:
        S_H = (p.A2 * (1 - 6 * p.A1 / p.B1)) / (
            1 - (3 * p.A2 * B2 + 18 * p.A1 * p.A2) * G_H
        )
        S_M = (
            p.A1 * (1 - 3 * p.C1 - 6 * p.A1 / p.B1)
            + S_H * (18 * p.A1 ** 2 + 9 * p.A1 * p.A2) * G_H
        ) / (1 - 9 * p.A1 * p.A2 * G_H)
        return S_M, S_H

    def _wall_proximity(self):
        """W̃ = 1 + E₂ (ℓ/(κ L))²,  L⁻¹ = z⁻¹ + (H - z)⁻¹."""
        p = self._parameter_symbols
        a = self.aux_variables
        l, _ = self._length_and_q()
        L_inv = 1 / sp.Max(a.z, p.l_floor) + 1 / sp.Max(
            p.column_depth - a.z, p.l_floor)
        L = 1 / L_inv
        return 1 + p.E2 * (l / (p.kappa * L)) ** 2

    def _mixing_coefficients(self):
        """K_M = ℓq S_M(G_H),  K_H = ℓq S_H(G_H),  K_q = ℓq S_q."""
        p = self._parameter_symbols
        l, q = self._length_and_q()
        S_M, S_H = self._stability_funcs()
        K_M = l * q * S_M
        K_H = l * q * S_H
        K_q = l * q * p.Sq
        return K_M, K_H, K_q

    # ── Operators ─────────────────────────────────────────────────────

    def flux(self):
        """No horizontal advection in a 1D vertical column."""
        return ZArray.zeros(self.n_variables, self.dimension)

    def diffusion_matrix(self):
        """Diagonal state-dependent diffusion tensor::

            A[U, U, 0, 0]     = K_M(ℓ, q, G_H)
            A[T, T, 0, 0]     = K_H(ℓ, q, G_H)
            A[q2, q2, 0, 0]   = K_q(ℓ, q)
            A[q2l, q2l, 0, 0] = K_q(ℓ, q)

        Backend contract: ``F_diff[i, d] = Σ_{j, e} A[i, j, d, e]·∂_e Q[j]``;
        Firedrake-DG (``firedrake_solver._get_weak_form_diffusion``) and
        ``NumpyRuntimeModel.diffusion_matrix(Q, Qaux, p)`` consume this
        as a full state-dependent rank-4 tensor.
        """
        n = self.n_variables
        K_M, K_H, K_q = self._mixing_coefficients()
        A = MutableDenseNDimArray.zeros(n, n, 1, 1)
        A[0, 0, 0, 0] = K_M
        A[1, 1, 0, 0] = K_H
        A[2, 2, 0, 0] = K_q
        A[3, 3, 0, 0] = K_q
        return ZArray(A)

    def source(self):
        """MY-2.5 production / dissipation, tagged implicit.

        - Shear production    :  2 K_M (∂_z U)²            → q² row
                                  E₁ ℓ K_M (∂_z U)²        → q²ℓ row
        - Buoyancy production :  2 g α_T K_H (∂_z T)        → q² row
                                  E₁ E₃ ℓ g α_T K_H (∂_z T) → q²ℓ row
        - Dissipation          : -2 q³/(B₁ ℓ)              → q² row
                                  -W̃ q³/B₁                 → q²ℓ row

        Derivative atoms ``∂_z U``, ``∂_z T`` enter as
        ``sp.Derivative(state, z)`` which ``SystemModel.expose_aux_atoms``
        auto-promotes to aux Symbols.  The solver computes them from
        the current Q each timestep via LSQ-mesh derivatives.
        """
        p = self._parameter_symbols
        v = self.variables
        space = sp.Symbol("x", real=True)
        U_z = sp.Derivative(v.U, space)
        T_z = sp.Derivative(v.T, space)
        l, q = self._length_and_q()
        K_M, K_H, _ = self._mixing_coefficients()
        Wt = self._wall_proximity()
        shear_sq = U_z ** 2
        buoy = p.g * p.alpha_T * K_H * T_z
        prod_q2  = 2 * K_M * shear_sq + 2 * buoy
        prod_q2l = p.E1 * l * (K_M * shear_sq + p.E3 * buoy)
        dissip_q2  = -2 * q ** 3 / (p.B1 * l)
        dissip_q2l = -Wt * q ** 3 / p.B1
        S = [sp.S.Zero] * self.n_variables
        S[2] = prod_q2 + dissip_q2     # q²
        S[3] = prod_q2l + dissip_q2l   # q²ℓ
        return ZArray(S)

    def eigenvalues(self):
        """Pure parabolic system — no hyperbolic eigenvalues.  Returns
        zeros; ``eigenvalue_mode="numerical"`` skips the symbolic path
        in ``SystemModel.from_model``."""
        return ZArray.zeros(self.n_variables)
