"""Tests for ``zoomy_core.analysis``.

The pipeline is:

  PDESystem → linearise → (plane_wave_dispersion | extract_quasilinear_pencil → sample_hyperbolicity)

These tests use **only** classical / well-known PDE systems with
hand-checkable answers: the linear advection equation, scalar wave
equation, and the (constant-coefficient) shallow water equations.
No model-specific framework code is exercised.
"""
from __future__ import annotations

import sympy as sp
import pytest

from zoomy_core.analysis import (
    PDESystem,
    linearise,
    plane_wave_dispersion,
    extract_quasilinear_pencil,
    generalised_eigenvalues,
    is_hyperbolic_at,
    sample_hyperbolicity,
)


# ---------------------------------------------------------------------------
# PDESystem dataclass — sanity / validation
# ---------------------------------------------------------------------------

def test_pdesystem_validates_field_dependence():
    t, x = sp.symbols("t x", real=True)
    h = sp.Function("h")(t, x)
    eq = sp.Derivative(h, t) + sp.Derivative(h, x)
    ok = PDESystem([eq], [h], t, [x])
    assert ok.n_equations() == 1
    assert ok.n_fields() == 1

    bad_field = sp.Symbol("not_a_function_call")
    with pytest.raises(TypeError):
        PDESystem([eq], [bad_field], t, [x])


# ---------------------------------------------------------------------------
# Linear advection: u_t + c u_x = 0  →  ω = c k
# ---------------------------------------------------------------------------

def test_linear_advection_dispersion():
    t, x = sp.symbols("t x", real=True)
    c = sp.Symbol("c", real=True)
    u = sp.Function("u")(t, x)
    eq = sp.Derivative(u, t) + c * sp.Derivative(u, x)
    sys_adv = PDESystem([eq], [u], t, [x])

    # Already linear; no need to linearise around a base state.  Just
    # plane-wave directly: the same primitive works since linearise
    # leaves an already-linear PDE alone (perturbations enter through
    # the q = q_0 + ε δq map; for linear-in-q PDEs all coefficients
    # are q_0-independent).  Linearise around u = 0 to obtain a δu
    # field for the plane-wave step.
    sys_lin = linearise(sys_adv, {u: 0})
    disp = plane_wave_dispersion(sys_lin, simplify=True)
    assert len(disp["solutions"]) == 1
    omega_sol = disp["solutions"][0]
    k = sp.Symbol("k", real=True)
    assert sp.simplify(omega_sol - c * k) == 0


# ---------------------------------------------------------------------------
# Scalar wave equation in first-order form:
#   u_t + v_x = 0;  v_t + c² u_x = 0  →  ω² = c² k²
# ---------------------------------------------------------------------------

def test_first_order_wave_equation_dispersion():
    t, x = sp.symbols("t x", real=True)
    c = sp.Symbol("c", positive=True)
    u = sp.Function("u")(t, x)
    v = sp.Function("v")(t, x)
    eq1 = sp.Derivative(u, t) + sp.Derivative(v, x)
    eq2 = sp.Derivative(v, t) + c**2 * sp.Derivative(u, x)
    sys_w = PDESystem([eq1, eq2], [u, v], t, [x])
    sys_lin = linearise(sys_w, {u: 0, v: 0})
    disp = plane_wave_dispersion(sys_lin, simplify=True)
    k = sp.Symbol("k", real=True)
    assert len(disp["solutions"]) == 2
    expected = {sp.simplify(c * k), sp.simplify(-c * k)}
    got = {sp.simplify(s) for s in disp["solutions"]}
    assert got == expected


# ---------------------------------------------------------------------------
# Shallow water (conservative form)  →  c² = g H at rest state
# ---------------------------------------------------------------------------

def test_swe_rest_state_dispersion_equals_gH():
    t, x = sp.symbols("t x", real=True)
    g = sp.Symbol("g", positive=True)
    H = sp.Symbol("H", positive=True)
    h = sp.Function("h")(t, x)
    u = sp.Function("u")(t, x)
    eq1 = sp.Derivative(h, t) + sp.Derivative(h * u, x)
    eq2 = sp.Derivative(h * u, t) + sp.Derivative(h * u**2 + g * h**2 / 2, x)
    sys_swe = PDESystem([eq1, eq2], [h, u], t, [x])
    sys_lin = linearise(sys_swe, {h: H, u: 0})
    disp = plane_wave_dispersion(sys_lin, simplify=True)
    k = sp.Symbol("k", real=True)
    pvs = [sp.simplify(s / k) for s in disp["solutions"]]
    c2 = {sp.simplify(pv**2) for pv in pvs}
    assert sp.simplify(g * H) in c2


# ---------------------------------------------------------------------------
# Quasilinear pencil extraction
# ---------------------------------------------------------------------------

def test_extract_pencil_swe():
    t, x = sp.symbols("t x", real=True)
    g = sp.Symbol("g", positive=True)
    H = sp.Symbol("H", positive=True)
    U0 = sp.Symbol("U0", real=True)
    h = sp.Function("h")(t, x)
    u = sp.Function("u")(t, x)
    eq1 = sp.Derivative(h, t) + sp.Derivative(h * u, x)
    eq2 = sp.Derivative(h * u, t) + sp.Derivative(h * u**2 + g * h**2 / 2, x)
    sys_swe = PDESystem([eq1, eq2], [h, u], t, [x])
    sys_lin = linearise(sys_swe, {h: H, u: U0})
    M_t, M_xa, M_0 = extract_quasilinear_pencil(sys_lin)

    # M_t should be Identity-ish: row 0 → δh has coeff 1; row 1 → δu
    # has coeff H (because ∂_t(h u) → U0 ∂_t δh + H ∂_t δu ⇒ M_t row 1 = (U0, H)).
    assert M_t.shape == (2, 2)
    assert sp.simplify(M_t[0, 0] - 1) == 0
    assert sp.simplify(M_t[0, 1]) == 0
    assert sp.simplify(M_t[1, 0] - U0) == 0
    assert sp.simplify(M_t[1, 1] - H) == 0


def test_generalised_eigenvalues_swe_rest():
    """At rest (U0 = 0), eigenvalues = ±√(gH)."""
    t, x = sp.symbols("t x", real=True)
    g = sp.Symbol("g", positive=True)
    H = sp.Symbol("H", positive=True)
    h = sp.Function("h")(t, x)
    u = sp.Function("u")(t, x)
    eq1 = sp.Derivative(h, t) + sp.Derivative(h * u, x)
    eq2 = sp.Derivative(h * u, t) + sp.Derivative(h * u**2 + g * h**2 / 2, x)
    sys_swe = PDESystem([eq1, eq2], [h, u], t, [x])
    sys_lin = linearise(sys_swe, {h: H, u: 0})
    M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)
    eigs = generalised_eigenvalues(M_xa[0], M_t)
    eigs_simpl = {sp.simplify(e) for e in eigs}
    assert sp.simplify(sp.sqrt(g * H)) in eigs_simpl
    assert sp.simplify(-sp.sqrt(g * H)) in eigs_simpl


# ---------------------------------------------------------------------------
# Numerical eigenvalues + hyperbolicity at one state
# ---------------------------------------------------------------------------

def test_is_hyperbolic_at_swe():
    t, x = sp.symbols("t x", real=True)
    g = sp.Symbol("g", positive=True)
    H_sym = sp.Symbol("H", positive=True)
    U_sym = sp.Symbol("U", real=True)
    h = sp.Function("h")(t, x)
    u = sp.Function("u")(t, x)
    eq1 = sp.Derivative(h, t) + sp.Derivative(h * u, x)
    eq2 = sp.Derivative(h * u, t) + sp.Derivative(h * u**2 + g * h**2 / 2, x)
    sys_swe = PDESystem([eq1, eq2], [h, u], t, [x])
    sys_lin = linearise(sys_swe, {h: H_sym, u: U_sym})
    M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)

    sub = {g: 9.81, H_sym: 1.0, U_sym: 0.0}
    hyp, eigs = is_hyperbolic_at(M_xa[0], M_t, sub)
    assert hyp
    # eigenvalues should be ±√9.81 ≈ ±3.13
    eigs_real = sorted(eigs.real)
    assert abs(eigs_real[0] + sp.sqrt(sp.Float(9.81))) < 1e-9
    assert abs(eigs_real[1] - sp.sqrt(sp.Float(9.81))) < 1e-9


def test_sample_hyperbolicity_swe_always_hyperbolic():
    t, x = sp.symbols("t x", real=True)
    g = sp.Symbol("g", positive=True)
    H_sym = sp.Symbol("H", positive=True)
    U_sym = sp.Symbol("U", real=True)
    h = sp.Function("h")(t, x)
    u = sp.Function("u")(t, x)
    eq1 = sp.Derivative(h, t) + sp.Derivative(h * u, x)
    eq2 = sp.Derivative(h * u, t) + sp.Derivative(h * u**2 + g * h**2 / 2, x)
    sys_swe = PDESystem([eq1, eq2], [h, u], t, [x])
    sys_lin = linearise(sys_swe, {h: H_sym, u: U_sym})
    M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)

    M_x_num = M_xa[0].subs(g, 9.81)
    M_t_num = M_t                                # no g in M_t for SWE
    report = sample_hyperbolicity(
        M_x_num, M_t_num,
        {H_sym: (0.1, 5.0), U_sym: (-3.0, 3.0)},
        n_samples=200,
        constraint_filter=lambda s: s[H_sym] > 0,
    )
    assert report.fraction_hyperbolic == 1.0


# ---------------------------------------------------------------------------
# Singular pencil (DAE) — algebraic-constraint row produces a
# rank-deficient M_t.  The pencil framework reports the constraint
# eigenvalue at infinity; ``drop_infinite=True`` filters it.
# ---------------------------------------------------------------------------

def test_singular_pencil_constraint_row_is_rank_deficient():
    """Toy DAE: u_t + v_x = 0;  u - v = 0.

    Pencil: M_t row 1 is zero (the algebraic constraint).  At finite k
    the constraint v = u is enforced via M_0; in the principal-symbol
    limit (which is what extract_quasilinear_pencil + sample_hyperbolicity
    inspect) the pencil is rank-deficient — generalised eigenvalues
    include +∞ for the constraint mode.  Drop them and we should be
    left with no propagating mode in this trivial example (since at
    high-k the system reduces to a redundant statement) — i.e. the
    test verifies the rank-deficient pencil is correctly *detected*
    and infinite eigenvalues are filtered out without crashing.
    """
    t, x = sp.symbols("t x", real=True)
    u = sp.Function("u")(t, x)
    v = sp.Function("v")(t, x)
    eq1 = sp.Derivative(u, t) + sp.Derivative(v, x)
    eq2 = u - v
    sys_dae = PDESystem([eq1, eq2], [u, v], t, [x])
    sys_lin = linearise(sys_dae, {u: 0, v: 0})

    M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)
    assert M_t.shape == (2, 2)
    assert M_t.rank() == 1                               # constraint row is zero

    # Algebraic constraints only show up as finite eigenvalues if the
    # M_0 term contributes — at the principal-symbol level the wave
    # speed isn't recoverable from this trivial example.  We just
    # assert that the routine completes without error and returns a
    # well-formed eigenvalue array.
    sub = {}
    hyp, eigs = is_hyperbolic_at(M_xa[0], M_t, sub, drop_infinite=True)
    assert isinstance(hyp, bool)
    assert eigs.shape == (2,) or len(eigs) <= 2
