"""SME vs Kowalski & Torrilhon 2019 ("Moment approximations and model
cascades for shallow flow") — TERM-BY-TERM reference pinning.

Every row of ``SME(level=N)`` is asserted SYMBOLICALLY against the
published general moment system, INCLUDING the Newtonian/Navier-slip
friction:

* mass:      ∂t h + ∂x(h u_m) = 0
* mean (4.7, 1D, e_x=0, e_z=1, h_b=b):
             ∂t(hu_m) + ∂x(h(u_m² + Σ α_j²/(2j+1)) + g h²/2) + g h ∂x b
             = −(ν/λ)(u_m + Σ α_j)
* moment i (4.10, 1D):
             ∂t(hα_i) + ∂x(h(2u_m α_i + Σ_{jk} A_ijk α_j α_k))
             = u_m D_i − Σ_{jk} B_ijk D_j α_k
               − (2i+1)(ν/λ)(u_m + Σ_j (1 + (λ/h) C_ij) α_j)
  with D_j = ∂x(h α_j) and the appendix-B Legendre integrals
  A_ijk = (2i+1)∫φ_iφ_jφ_k,  B_ijk = (2i+1)∫φ_i′(∫₀^ζφ_j)φ_k,
  C_ij = ∫φ_i′φ_j′  — all computed here in sympy, not hand-copied.

Mapping (theirs ← ours):

* their basis is normalized φ_j(0)=1 (φ_1 = 1−2ζ) = (−1)^j·(our shifted
  Legendre) ⇒ α_j = (−1)^j q_j/h, u_m = q_0/h, and our moment row i equals
  (−1)^i × their row (the test function itself flips);
* friction: our Navier-slip closure τ_b = λ_s·u_b (dynamic, /ρ in the
  momentum balance) ⇒ their kinematic ν/λ = our λ_s/ρ; the slip-length λ
  cancels in the C-term: (2i+1)(ν/λ)(λ/h)C_ij = (2i+1)(ν/h)C_ij, so the
  reference needs only OUR parameters (λ_s, ν, ρ).

Shape, count and smoke checks are NEVER sufficient — pin every term.
"""
import pytest
import sympy as sp

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel.system_model import SystemModel


def _kt_reference_rows(sm, N):
    """K&T general 1-D moment system in OUR variables: returns the list of
    row residuals [b-row, mass, mean, moment_1 … moment_N] aligned with
    sm.state = [b, h, q_0 … q_N], each pre-multiplied by the row-sign that
    maps it onto OUR rows ((−1)^i for moment i)."""
    t, x = sm.time, sm.space[0]
    Fn = lambda n: sp.Function(n, real=True)(t, x)
    h, b = Fn("h"), Fn("b")
    q = [Fn(f"q_{i}") for i in range(N + 1)]
    g = sm.parameters.g
    # K&T's friction is KINEMATIC (their ν = kinematic viscosity); our basal
    # stress is dynamic (τ_b = λ_s·u_b, divided by ρ in the momentum balance)
    # ⇒ their ν/λ = our λ_s/ρ.  The interior C-term is kinematic on both
    # sides (our τ = ρν/h·∂ζu cancels its ρ) ⇒ their ν = our ν.
    lam_s = sm.parameters.lambda_s / sm.parameters.rho
    nu = sm.parameters.nu
    Dx = lambda e: sp.Derivative(e, x)
    Dt = lambda e: sp.Derivative(e, t)

    zeta, s_ = sp.symbols("zeta s_", nonnegative=True)
    phi = [sp.legendre(j, 1 - 2 * zeta) for j in range(N + 1)]   # THEIR basis

    def I01(e):
        return sp.integrate(sp.expand(e), (zeta, 0, 1))

    A = lambda i, j, k: (2 * i + 1) * I01(phi[i] * phi[j] * phi[k])
    B = lambda i, j, k: (2 * i + 1) * I01(
        sp.diff(phi[i], zeta)
        * sp.integrate(phi[j].subs(zeta, s_), (s_, 0, zeta)) * phi[k])
    Cf = lambda i, j: I01(sp.diff(phi[i], zeta) * sp.diff(phi[j], zeta))

    um = q[0] / h
    alpha = {i: (-1) ** i * q[i] / h for i in range(1, N + 1)}
    D = {j: Dx(h * alpha[j]) for j in range(1, N + 1)}
    J = range(1, N + 1)

    rows = [Dt(b)]                                               # inert bed
    rows.append(Dt(h) + Dx(h * um))                              # mass
    e_x = sm.parameters.e_x                       # K&T 4.7 downslope body force
    rows.append(                                                 # mean (4.7)
        Dt(h * um)
        + Dx(h * (um**2 + sum(alpha[j]**2 / (2 * j + 1) for j in J))
             + g * h**2 / 2)
        + g * h * Dx(b)
        - g * h * e_x
        + lam_s * (um + sum(alpha[j] for j in J)))
    for i in J:                                                  # moments (4.10)
        row = (Dt(h * alpha[i])
               + Dx(h * (2 * um * alpha[i]
                         + sum(A(i, j, k) * alpha[j] * alpha[k]
                               for j in J for k in J)))
               - um * D[i]
               + sum(B(i, j, k) * D[j] * alpha[k] for j in J for k in J)
               + (2 * i + 1) * (lam_s * (um + sum(alpha[j] for j in J))
                                + (nu / h) * sum(Cf(i, j) * alpha[j]
                                                 for j in J)))
        rows.append((-1) ** i * row)        # their row → our row sign
    return rows


@pytest.mark.parametrize("level", [1, 2, 3])
def test_sme_rows_match_kowalski_torrilhon(level):
    sm = SystemModel.from_model(SME(closures=[Newtonian(), NavierSlip(), StressFree()], level=level))
    assert [str(s) for s in sm.state] == (
        ["b", "h"] + [f"q_{i}" for i in range(level + 1)])
    rv = sm.reconstruct_residuals()
    refs = _kt_reference_rows(sm, level)
    for i, (mine, ref) in enumerate(zip(rv, refs)):
        diff = sp.simplify(sp.expand(sp.sympify(mine).doit())
                           - sp.expand(ref.doit()))
        name = str(sm.state[i])
        assert diff == 0, (
            f"SME({level}) row {i} ({name}) differs from K&T: {diff}")
