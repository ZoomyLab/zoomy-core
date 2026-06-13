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


def _termwise_equal(A, B):
    if [str(s) for s in A.state] != [str(s) for s in B.state]:
        return False
    for name in _OPERATORS:
        a, b = getattr(A, name, None), getattr(B, name, None)
        if a is None or b is None:
            continue
        fa, fb = _flat(a), _flat(b)
        if any(sp.simplify((p or 0) - (q or 0)) != 0 for p, q in zip(fa, fb)):
            return False
    return True


def test_ml_stress_closures_equal_material():
    from zoomy_core.model.models import MLSME
    pars = {"nu": 0.1, "lambda_s": 0.5}
    a = MLSME(n_layers=2, level=1, material=newtonian_navier_slip(), parameters=pars).system_model
    b = MLSME(n_layers=2, level=1, parameters=pars,
              closures=[Newtonian(), NavierSlip(), StressFree()]).system_model
    assert _termwise_equal(a, b)


def test_ml_interface_closures_equal_selector():
    from zoomy_core.model.models import MLSME
    from zoomy_core.model.models.closures import UpwindInterface, MeanInterface
    up_sel = MLSME(n_layers=2, level=1, interface_velocity="upwind").system_model
    up_cl = MLSME(n_layers=2, level=1, closures=[UpwindInterface()]).system_model
    assert _termwise_equal(up_sel, up_cl)
    mn_sel = MLSME(n_layers=2, level=1, interface_velocity="mean").system_model
    mn_cl = MLSME(n_layers=2, level=1, closures=[MeanInterface()]).system_model
    assert _termwise_equal(mn_sel, mn_cl)


def test_empty_closures_leave_stress_unclosed():
    sm = SME(level=2).system_model
    assert any("sigma" in str(s) for s in sm.aux_state)


def test_bingham_bulk_closes_via_quadrature():
    """Bingham (viscoplastic) bulk material: rational/√ in ζ → no analytic
    Galerkin bracket; closes with quadrature_order>0 (Gauss-Legendre)."""
    from zoomy_core.model.models.closures import Bingham
    sm = SME(level=2, quadrature_order=8,
             parameters={"nu": 0.1, "tau_y": 0.5, "lambda_s": 0.5},
             closures=[Bingham(), NavierSlip(), StressFree()]).system_model
    surv = sum(len(sp.sympify(sm.source[i, 0]).atoms(sp.Integral))
               for i in range(sm.n_equations))
    assert surv == 0, "Bingham bulk must close via quadrature (no loose integrals)"


def test_rough_wall_is_bottom_only_bulk_free():
    sm = SME(level=2, closures=[RoughWall()], parameters={"k_s": 0.01}).system_model
    # bed drag closed, bulk left free → free stress moments survive
    assert any("sigma" in str(s) for s in sm.aux_state)
    lam = sm.parameters
    src = sp.sympify(sm.source[[str(s) for s in sm.state].index("q_0"), 0])
    assert src.has(getattr(lam, "kappa")), "rough-wall bed drag absent from q_0 source"
