"""SME closed-form wavespeeds: the β-HSWME spectrum registration.

Koellermeier & Rominger 2020 (Thm 3.5): the β-HSWME eigenvalues are
λ_{1,2} = u_m ± √(gh + α₁²) and λ_{i+2} = u_m + c_{i,N}·α₁ with c_{i,N}
the roots of the Legendre polynomial P_N — all inside
[u_m − √(gh+α₁²), u_m + √(gh+α₁²)].  SME registers them as the symbolic
``SystemModel.eigenvalues`` so every backend (numpy dt + Riemann, jax,
codegen) uses closed-form wavespeeds instead of per-face LAPACK/geev
eigensolves (the CUDA blocker).

Pins:
* the registration (mode 'symbolic', exact λ formulas at level 1, where
  HSWME ≡ SME so the spectrum is EXACT);
* the in-regime bound quality: over sampled hyperbolic states with
  |α_j| ≤ 0.5·√(gh), the estimate under-shoots the true SME spectrum by
  at most 8% (measured ≤7%; near the hyperbolicity margin the truncated
  HSWME matrix loses the α_{≥2} dependence — documented trade-off per
  the directive: sharp, no extra diffusion, unlike Gershgorin).
"""
import numpy as np
import sympy as sp
import pytest

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel


def test_sme_registers_symbolic_hswme_spectrum_exact_at_level1():
    sm = SME(level=1, closures=[Newtonian(), NavierSlip(), StressFree()]).system_model
    assert sm.eigenvalue_mode == "symbolic"
    by = {str(s): s for s in sm.state}
    g = sm.parameters.g
    u = by["q_0"] / by["h"]
    c = sp.sqrt(g * by["h"] + (by["q_1"] / by["h"]) ** 2)
    n0 = sm.normal[0]
    got = [sp.simplify(e / n0) if e != 0 else sp.S.Zero
           for e in sm.eigenvalues]
    # level 1: HSWME ≡ SME — spectrum {0(bed), u±√(gh+α₁²), u}, K&T (4.16).
    # Compare the eigenvalue SETS numerically over random positive states:
    # the symbolic forms agree only under h>0 (the spectrum carries
    # √(g h³+q_1²)/h, equal to √(g h+(q_1/h)²) but not pulled out of the
    # radical without the assumption), so a symbolic `simplify(a-b)==0` is too
    # brittle — a numeric set match is the honest, h>0-safe check.
    expected = [sp.S.Zero, u - c, u + c, u]
    rng = np.random.default_rng(3)
    for _ in range(20):
        sub = {by["h"]: rng.uniform(0.1, 3.0), by["q_0"]: rng.normal(),
               by["q_1"]: rng.normal(), g: 9.81}
        got_v = sorted(float(e.subs(sub)) for e in got)
        exp_v = sorted(float(e.subs(sub)) for e in expected)
        assert np.allclose(got_v, exp_v), f"{got_v} != {exp_v}"


@pytest.mark.parametrize("level", [2, 3])
def test_hswme_bound_quality_in_hyperbolic_regime(level):
    sm = SME(level=level, closures=[Newtonian(), NavierSlip(), StressFree()],
             parameters={"lambda_s": 0.5, "nu": 1e-3}).system_model
    rt = NumpyRuntimeModel.from_system_model(sm)
    n_v = sm.n_state
    g = 9.81
    rng = np.random.default_rng(11)
    worst_rel = -1e9
    for _ in range(500):
        h = rng.uniform(0.05, 2.0)
        c = np.sqrt(g * h)
        um = rng.normal(0, 1.5 * c)
        alph = rng.uniform(-0.5, 0.5, size=level) * c / (1 + np.arange(level))
        q = np.zeros(n_v)
        q[1] = h
        q[2] = h * um
        q[3:3 + level] = h * alph * [(-1) ** (j + 1)
                                     for j in range(1, level + 1)]
        Q = q.reshape(-1, 1)
        Qaux = np.zeros((len(sm.aux_state), 1))
        p = np.array(list(sm.parameter_values.values()), float)
        A = np.asarray(rt.quasilinear_matrix(Q, Qaux, p),
                       float).reshape(n_v, n_v, -1)[:, :, 0]
        evs_t = np.linalg.eigvals(A)
        if np.abs(evs_t.imag).max() > 1e-8:
            continue                       # outside the hyperbolic region
        true_max = np.abs(evs_t.real).max()
        est = np.abs(np.asarray(
            rt.eigenvalues(Q, Qaux, p, np.array([1.0])), float)).max()
        worst_rel = max(worst_rel, (true_max - est) / true_max)
    assert worst_rel < 0.08, f"HSWME bound degraded: {worst_rel:.2%}"


@pytest.mark.parametrize("level", [6, 7])
def test_high_level_spectrum_falls_back_to_truncated_twin(level):
    """Level ≥ 6: the truncated moment block is Abel-unsolvable in radicals, so
    SME splices in the level-5 twin's exact spectrum and pads the interior with
    n·u_m.  The β-HSWME gravity waves are level-INDEPENDENT and bound the whole
    spectrum, so the registered spectral RADIUS must still equal the gravity
    wave |u_m| + √(g h + α₁²) exactly — a sharp closed-form wavespeed at any
    level, no per-face eigensolve."""
    sm = SME(level=level).system_model
    assert sm.eigenvalue_mode == "symbolic"
    assert len(sm.eigenvalues) == sm.n_equations
    by = {str(s): s for s in sm.state}
    g_val, h_val, q0_val, a_val = 9.81, 1.3, 0.8, 0.15
    sub = {by["h"]: h_val, by["q_0"]: q0_val, sm.parameters.g: g_val,
           sm.normal[0]: 1.0, by["b"]: 0.0}
    sub.update({s: a_val for s in sm.state
                if str(s) not in ("h", "q_0", "b")})
    vals = [float(sp.sympify(e).subs(sub)) for e in sm.eigenvalues]
    grav = q0_val / h_val + np.sqrt(g_val * h_val + (a_val / h_val) ** 2)
    assert np.isclose(max(abs(v) for v in vals), grav, rtol=1e-6), \
        f"radius {max(abs(v) for v in vals)} != gravity {grav}"
