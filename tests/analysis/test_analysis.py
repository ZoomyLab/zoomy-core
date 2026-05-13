"""Tests for ``zoomy_core.analysis``.

The pipeline is:

  SystemModel → linearise → (plane_wave_dispersion |
                              extract_quasilinear_pencil →
                              sample_hyperbolicity)

These tests use **only** classical / well-known PDE systems with
hand-checkable answers: the linear advection equation, scalar wave
equation, and the (constant-coefficient) shallow water equations.
No model-specific framework code is exercised.
"""
from __future__ import annotations

import sympy as sp
import pytest

from zoomy_core.analysis import (
    linearise,
    plane_wave_dispersion,
    extract_quasilinear_pencil,
    generalised_eigenvalues,
    is_hyperbolic_at,
    sample_hyperbolicity,
)
from zoomy_core.model.models.system_model import SystemModel


def _swe(g_sym=None):
    """Standard SWE SystemModel ``(h, u)`` with mass matrix
    ``[[1, 0], [U, h]]`` and flux ``[hU, hU² + gh²/2]``."""
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    h, u = sp.symbols("h u", real=True)
    g = g_sym if g_sym is not None else sp.Symbol("g", positive=True)
    return SystemModel(
        time=t,
        space=[x],
        state=[h, u],
        aux_state=[],
        parameters={g: 9.81},
        flux=sp.Matrix([[h * u], [h * u**2 + g * h**2 / 2]]),
        hydrostatic_pressure=sp.zeros(2, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(2, 2, 1),
        source=sp.zeros(2, 1),
        mass_matrix=sp.Matrix([[1, 0], [u, h]]),
    )


# ---------------------------------------------------------------------------
# Linear advection: u_t + c u_x = 0  →  ω = c k
# ---------------------------------------------------------------------------

def test_linear_advection_dispersion():
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    c = sp.Symbol("c", real=True)
    u = sp.Symbol("u", real=True)
    sm = SystemModel(
        time=t,
        space=[x],
        state=[u],
        aux_state=[],
        parameters={},
        flux=sp.Matrix([[c * u]]),
        hydrostatic_pressure=sp.zeros(1, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(1, 1, 1),
        source=sp.zeros(1, 1),
        mass_matrix=sp.eye(1),
    )
    sm_lin = linearise(sm, {u: 0})
    disp = plane_wave_dispersion(sm_lin, simplify=True)
    assert len(disp["solutions"]) == 1
    omega_sol = disp["solutions"][0]
    k = sp.Symbol("k", real=True)
    assert sp.simplify(omega_sol - c * k) == 0


# ---------------------------------------------------------------------------
# Scalar wave equation in first-order form:
#   u_t + v_x = 0;  v_t + c² u_x = 0  →  ω² = c² k²
# ---------------------------------------------------------------------------

def test_first_order_wave_equation_dispersion():
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    c = sp.Symbol("c", positive=True)
    u, v = sp.symbols("u v", real=True)
    sm = SystemModel(
        time=t,
        space=[x],
        state=[u, v],
        aux_state=[],
        parameters={},
        flux=sp.Matrix([[v], [c**2 * u]]),
        hydrostatic_pressure=sp.zeros(2, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(2, 2, 1),
        source=sp.zeros(2, 1),
        mass_matrix=sp.eye(2),
    )
    sm_lin = linearise(sm, {u: 0, v: 0})
    disp = plane_wave_dispersion(sm_lin, simplify=True)
    assert len(disp["solutions"]) == 2
    k = sp.Symbol("k", real=True)
    expected = {sp.simplify(c * k), sp.simplify(-c * k)}
    got = {sp.simplify(s) for s in disp["solutions"]}
    assert got == expected


# ---------------------------------------------------------------------------
# Shallow water (conservative form)  →  c² = g H at rest state
# ---------------------------------------------------------------------------

def test_swe_rest_state_dispersion_equals_gH():
    g = sp.Symbol("g", positive=True)
    H = sp.Symbol("H", positive=True)
    sm = _swe(g_sym=g)
    h, u = sm.state
    sm_lin = linearise(sm, {h: H, u: 0})
    disp = plane_wave_dispersion(sm_lin, simplify=True)
    k = sp.Symbol("k", real=True)
    pvs = [sp.simplify(s / k) for s in disp["solutions"]]
    c2 = {sp.simplify(pv**2) for pv in pvs}
    assert sp.simplify(g * H) in c2


# ---------------------------------------------------------------------------
# Quasilinear pencil extraction
# ---------------------------------------------------------------------------

def test_extract_pencil_swe():
    g = sp.Symbol("g", positive=True)
    H = sp.Symbol("H", positive=True)
    U0 = sp.Symbol("U0", real=True)
    sm = _swe(g_sym=g)
    h, u = sm.state
    sm_lin = linearise(sm, {h: H, u: U0})
    M_t, M_xa, M_0 = extract_quasilinear_pencil(sm_lin)

    # M_t row 0 (mass eq) = (1, 0).  M_t row 1 (momentum eq) = (U0, H).
    assert M_t.shape == (2, 2)
    assert sp.simplify(M_t[0, 0] - 1) == 0
    assert sp.simplify(M_t[0, 1]) == 0
    assert sp.simplify(M_t[1, 0] - U0) == 0
    assert sp.simplify(M_t[1, 1] - H) == 0


def test_generalised_eigenvalues_swe_rest():
    """At rest (U0 = 0), eigenvalues = ±√(gH)."""
    g = sp.Symbol("g", positive=True)
    H = sp.Symbol("H", positive=True)
    sm = _swe(g_sym=g)
    h, u = sm.state
    sm_lin = linearise(sm, {h: H, u: 0})
    M_t, M_xa, _ = extract_quasilinear_pencil(sm_lin)
    eigs = generalised_eigenvalues(M_xa[0], M_t)
    eigs_simpl = {sp.simplify(e) for e in eigs}
    assert sp.simplify(sp.sqrt(g * H)) in eigs_simpl
    assert sp.simplify(-sp.sqrt(g * H)) in eigs_simpl


# ---------------------------------------------------------------------------
# Numerical eigenvalues + hyperbolicity
# ---------------------------------------------------------------------------

def test_is_hyperbolic_at_swe():
    g = sp.Symbol("g", positive=True)
    H_sym = sp.Symbol("H", positive=True)
    U_sym = sp.Symbol("U", real=True)
    sm = _swe(g_sym=g)
    h, u = sm.state
    sm_lin = linearise(sm, {h: H_sym, u: U_sym})
    M_t, M_xa, _ = extract_quasilinear_pencil(sm_lin)

    sub = {g: 9.81, H_sym: 1.0, U_sym: 0.0}
    hyp, eigs = is_hyperbolic_at(M_xa[0], M_t, sub)
    assert hyp
    eigs_real = sorted(eigs.real)
    assert abs(eigs_real[0] + sp.sqrt(sp.Float(9.81))) < 1e-9
    assert abs(eigs_real[1] - sp.sqrt(sp.Float(9.81))) < 1e-9


def test_sample_hyperbolicity_swe_always_hyperbolic():
    g = sp.Symbol("g", positive=True)
    H_sym = sp.Symbol("H", positive=True)
    U_sym = sp.Symbol("U", real=True)
    sm = _swe(g_sym=g)
    h, u = sm.state
    sm_lin = linearise(sm, {h: H_sym, u: U_sym})
    M_t, M_xa, _ = extract_quasilinear_pencil(sm_lin)

    M_x_num = M_xa[0].subs(g, 9.81)
    M_t_num = M_t
    report = sample_hyperbolicity(
        M_x_num, M_t_num,
        {H_sym: (0.1, 5.0), U_sym: (-3.0, 3.0)},
        n_samples=200,
        constraint_filter=lambda s: s[H_sym] > 0,
    )
    assert report.fraction_hyperbolic == 1.0
