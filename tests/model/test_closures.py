"""Composable stress closures (closures.py) on SME.

The contract: a closure LIST reproduces the legacy MaterialModel system
term-by-term, the new turbulent pieces build, and a boundary-only closure
leaves the bulk stress free.
"""
import sympy as sp

from zoomy_core.model.models import SME
from zoomy_core.model.models.material import newtonian_navier_slip
from zoomy_core.model.models.closures import (
    Newtonian, NavierSlip, StressFree, RoughWall)


def test_closures_list_equals_material_term_by_term():
    pars = {"nu": 0.1, "lambda_s": 0.5}
    legacy = SME(level=2, material=newtonian_navier_slip(), parameters=pars).system_model
    composed = SME(level=2, parameters=pars,
                   closures=[Newtonian(), NavierSlip(), StressFree()]).system_model
    assert [str(s) for s in legacy.state] == [str(s) for s in composed.state]
    for i in range(legacy.n_equations):
        d = sp.simplify(sp.sympify(legacy.source[i, 0]) - sp.sympify(composed.source[i, 0]))
        assert d == 0, f"row {i}: closures differ from material: {d}"


def test_empty_closures_leave_stress_unclosed():
    sm = SME(level=2).system_model
    assert any("sigma" in str(s) for s in sm.aux_state)


def test_rough_wall_is_bottom_only_bulk_free():
    sm = SME(level=2, closures=[RoughWall()], parameters={"k_s": 0.01}).system_model
    # bed drag closed, bulk left free → free stress moments survive
    assert any("sigma" in str(s) for s in sm.aux_state)
    lam = sm.parameters
    src = sp.sympify(sm.source[[str(s) for s in sm.state].index("q_0"), 0])
    assert src.has(getattr(lam, "kappa")), "rough-wall bed drag absent from q_0 source"
