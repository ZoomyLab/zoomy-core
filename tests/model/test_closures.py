"""Composable stress closures (closures.py) on SME.

The contract: a closure LIST reproduces the legacy MaterialModel system
term-by-term, the new turbulent pieces build, and a boundary-only closure
leaves the bulk stress free.
"""
import sympy as sp
import numpy as np

from zoomy_core.model.models import SME
from zoomy_core.model.models.material import newtonian_navier_slip
from zoomy_core.model.models.closures import (
    Newtonian, NavierSlip, StressFree, RoughWall)

# Operators that make up the full SystemModel — "term-by-term identical" means
# EVERY entry of EVERY one of these matches (not just `source`).
_OPERATORS = ("flux", "nonconservative_matrix", "quasilinear_matrix",
              "source", "eigenvalues")


def _flat(z):
    """Flatten a ZArray / Matrix / list operator to scalar sympy exprs."""
    a = np.array(z, dtype=object).reshape(-1)
    out = []
    for e in a:
        while not isinstance(e, sp.Basic) and hasattr(e, "__len__"):
            e = np.array(e, dtype=object).reshape(-1)
            e = e[0] if len(e) == 1 else e
            if not isinstance(e, sp.Basic) and not hasattr(e, "__len__"):
                break
        out.append(sp.sympify(e))
    return out


def test_closures_list_equals_material_term_by_term():
    """closures=[...] reproduces material=... in EVERY operator entry."""
    pars = {"nu": 0.1, "lambda_s": 0.5}
    legacy = SME(level=2, material=newtonian_navier_slip(), parameters=pars).system_model
    composed = SME(level=2, parameters=pars,
                   closures=[Newtonian(), NavierSlip(), StressFree()]).system_model
    assert [str(s) for s in legacy.state] == [str(s) for s in composed.state]
    for name in _OPERATORS:
        fa, fb = _flat(getattr(legacy, name)), _flat(getattr(composed, name))
        assert len(fa) == len(fb), f"{name}: length {len(fa)} != {len(fb)}"
        for k, (a, b) in enumerate(zip(fa, fb)):
            assert sp.simplify(a - b) == 0, f"{name}[{k}] differs: {sp.simplify(a - b)}"


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
