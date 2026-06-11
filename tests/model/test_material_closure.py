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
        bulk=lambda u, dz, par: par.rho * par.nu * dz(u),
        bottom=lambda u_b, par: par.lambda_s * u_b * u_b,
        surface=lambda u_s, par: 0,
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


def test_newtonian_variants_equal_explicit_injection():
    """The pre-closed convenience factories (NewtonianSME, …) must build
    the SAME system as the bare class with the standard closure injected."""
    from zoomy_core.model.models import NewtonianSME, NewtonianVAM

    for mk, cls, kw in ((NewtonianSME, SME, dict(level=1)),
                        (NewtonianVAM, VAM, dict(level=1))):
        sm_a = mk(**kw).system_model
        sm_b = cls(material=newtonian_navier_slip(), **kw).system_model
        assert [str(s) for s in sm_a.state] == [str(s) for s in sm_b.state]
        for i in range(sm_a.n_equations):
            d_ = sp.simplify(sp.sympify(sm_a.source[i, 0])
                             - sp.sympify(sm_b.source[i, 0]))
            assert d_ == 0, f"{mk.__name__} row {i} source differs"
            d_ = sp.simplify(sp.sympify(sm_a.flux[i, 0])
                             - sp.sympify(sm_b.flux[i, 0]))
            assert d_ == 0, f"{mk.__name__} row {i} flux differs"
