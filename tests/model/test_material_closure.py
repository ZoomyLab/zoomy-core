"""Material-closure injection — one class per model, three paths:

* UNCLOSED default (``material=None``): no substitution; the stress is
  expanded in the modal basis and its moments σ̂_j stay free functions
  (routed to aux by the extraction); no friction parameter anywhere.
* standard ``newtonian_navier_slip()``: reproduces the historical
  hard-coded closure exactly — the term-by-term reference tests pin the
  full systems, here we only spot-check the friction term.
* a CUSTOM material (quadratic Chézy bed drag) flows through the same
  injection path and lands in the projected source.
"""
import pytest
import sympy as sp

from zoomy_core.model.models import (
    SME, MLSWE, MLSME, VAM, MLVAM, MaterialModel, newtonian_navier_slip)

CASES = [
    (SME, dict(level=1)),
    (MLSWE, dict(n_layers=2)),
    (MLSME, dict(n_layers=2, level=1)),
    (VAM, dict(level=1)),
    (MLVAM, dict(n_layers=2, level=1)),
]


@pytest.mark.parametrize("cls,kw", CASES, ids=[c.__name__ for c, _ in CASES])
def test_unclosed_default_keeps_stress_moments_free(cls, kw):
    sm = cls(**kw).system_model
    lam, nu = sm.parameters.lambda_s, sm.parameters.nu
    for i in range(sm.n_equations):
        src = sp.sympify(sm.source[i, 0])
        assert not src.has(lam) and not src.has(nu), (
            f"{cls.__name__} row {i}: friction parameters in the UNCLOSED "
            "model — a closure leaked in")
    assert any("sigma" in str(a) for a in sm.aux_state), (
        f"{cls.__name__}: no free stress moments in the unclosed system")


@pytest.mark.parametrize("cls,kw", CASES, ids=[c.__name__ for c, _ in CASES])
def test_standard_closure_produces_friction(cls, kw):
    sm = cls(material=newtonian_navier_slip(), **kw).system_model
    lam = sm.parameters.lambda_s
    assert any(sp.sympify(sm.source[i, 0]).has(lam)
               for i in range(sm.n_equations)), (
        f"{cls.__name__}: standard closure produced no bed friction")
    assert not any("sigma" in str(a) for a in sm.aux_state), (
        f"{cls.__name__}: closed model still carries free stress moments")


def test_custom_chezy_material_lands_in_source():
    chezy = MaterialModel(
        bulk=lambda s: s.par.rho * s.par.nu * s.dz(s.u),
        bottom=lambda s: s.par.lambda_s * s.u * s.u,
        surface=lambda s: 0,
        name="newtonian+chezy",
    )
    sm = SME(level=1, material=chezy).system_model
    names = [str(s) for s in sm.state]
    lam = sm.parameters.lambda_s
    src = sp.sympify(sm.source[names.index("q_0"), 0])
    fric = sp.simplify(src.coeff(lam, 1))
    # τ_b = λ·u_b² with u_b = (q_0 − q_1)/h  →  source −λ(q_0−q_1)²/(ρh²)
    h, q0, q1 = (s for s in sm.state if str(s) in ("h", "q_0", "q_1"))
    by = {str(s): s for s in sm.state}
    expected = -(by["q_0"] - by["q_1"]) ** 2 / (sm.parameters.rho * by["h"] ** 2)
    assert sp.simplify(fric - expected) == 0, f"chezy friction: {fric}"


def test_bingham_requires_explicit_quadrature():
    """A non-polynomial closure (Bingham) leaves Galerkin integrals the
    bracket machinery cannot resolve — extraction must REFUSE loudly and
    point at the numerical-integration escape hatch."""
    from zoomy_core.model.models.material import bingham_navier_slip
    par = {"nu": 0.1, "lambda_s": 50.0, "tau_y": 0.3, "eps_reg": 1e-2}
    with pytest.raises(ValueError, match="quadrature"):
        SME(level=2, material=bingham_navier_slip(),
            parameters=par).system_model


def test_bingham_with_gauss_quadrature_builds_and_lambdifies():
    """With quadrature_order set, the surviving integrals become
    Gauss–Legendre sums: no Integral atoms reach the operators, the yield
    stress lands in the source, and the runtime lambdification works."""
    import numpy as np
    from zoomy_core.model.models.material import bingham_navier_slip
    from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
    par = {"nu": 0.1, "lambda_s": 50.0, "tau_y": 0.3, "eps_reg": 1e-2}
    sm = SME(level=2, material=bingham_navier_slip(), quadrature_order=8,
             parameters=par).system_model
    for i in range(sm.n_equations):
        assert not sp.sympify(sm.source[i, 0]).atoms(sp.Integral)
    names = [str(s) for s in sm.state]
    assert sp.sympify(sm.source[names.index("q_1"), 0]).has(
        sm.parameters.tau_y)
    rt = NumpyRuntimeModel.from_system_model(sm)
    Q = np.array([0.0, 1.0, 0.3, -0.1, 0.02]).reshape(-1, 1)
    Qaux = np.zeros((len(sm.aux_state), 1))
    p = np.array(list(sm.parameter_values.values()), float)
    s_val = np.asarray(rt.source(Q, Qaux, p), float)
    assert np.all(np.isfinite(s_val))
