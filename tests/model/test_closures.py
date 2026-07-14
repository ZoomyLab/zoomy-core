"""Composable stress closures (closures.py) on SME / MLSME.

The contract: the closure LIST is the ONLY stress-closure path (no legacy
MaterialModel), the interface-velocity selector equals its closure form, the
turbulent pieces build, and a boundary-only closure leaves the bulk free.
"""
import sympy as sp
import numpy as np
import pytest

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import (
    Newtonian, NavierSlip, StressFree, RoughWall, ElderViscosity)
from zoomy_core.systemmodel.system_model import SystemModel

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


def test_ml_interface_closures_equal_selector():
    from zoomy_core.model.models import MLSME
    from zoomy_core.model.models.closures import UpwindInterface, MeanInterface
    up_sel = SystemModel.from_model(MLSME(n_layers=2, level=1, interface_velocity="upwind"))
    up_cl = SystemModel.from_model(MLSME(n_layers=2, level=1, closures=[UpwindInterface()]))
    assert _termwise_equal(up_sel, up_cl)
    mn_sel = SystemModel.from_model(MLSME(n_layers=2, level=1, interface_velocity="mean"))
    mn_cl = SystemModel.from_model(MLSME(n_layers=2, level=1, closures=[MeanInterface()]))
    assert _termwise_equal(mn_sel, mn_cl)


def test_empty_closures_leave_stress_unclosed():
    sm = SystemModel.from_model(SME(level=2))
    assert any("sigma" in str(s) for s in sm.aux_state)


def test_bingham_bulk_closes_via_quadrature():
    """Bingham (viscoplastic) bulk material: rational/√ in ζ → no analytic
    Galerkin bracket; closes with quadrature_order>0 (Gauss-Legendre)."""
    from zoomy_core.model.models.closures import Bingham
    sm = SystemModel.from_model(SME(level=2, quadrature_order=8,
             parameters={"nu": 0.1, "tau_y": 0.5, "lambda_s": 0.5},
             closures=[Bingham(), NavierSlip(), StressFree()]))
    surv = sum(len(sp.sympify(sm.source[i, 0]).atoms(sp.Integral))
               for i in range(sm.n_equations))
    assert surv == 0, "Bingham bulk must close via quadrature (no loose integrals)"


@pytest.mark.parametrize("dim", [2, 3])
def test_elder_turbulence_closes_any_dimension(dim):
    """Elder parabolic eddy viscosity ν_t = κ u_⋆ h ζ(1−ζ) is polynomial in ζ →
    the Galerkin projection closes analytically (no quadrature) in 1-D AND 2-D
    (the dimension-agnostic SME applies it diagonally per velocity component)."""
    lvl = 2 if dim == 2 else 1            # dim=3 level=2 is ~90s; keep level 1
    sm = SystemModel.from_model(SME(level=lvl, dimension=dim, parameters={"u_star": 0.3, "kappa": 0.41},
             closures=[ElderViscosity(), NavierSlip(), StressFree()]))
    surv = sum(len(sp.sympify(sm.source[i, 0]).atoms(sp.Integral))
               for i in range(sm.n_equations))
    assert surv == 0, "Elder ν_t(ζ) must close analytically (no surviving integrals)"


@pytest.mark.parametrize("dim", [2, 3])
def test_rough_wall_dynamic_bed_any_dimension(dim):
    """RoughWall turbulent bed drag (vector traction τ_b = ρ C_f |U| U, the
    magnitude couples x/y in 2-D) builds on the dimension-agnostic SME and the
    bed drag (∝ κ) reaches the momentum source."""
    lvl = 2 if dim == 2 else 1
    sm = SystemModel.from_model(SME(level=lvl, dimension=dim, parameters={"k_s": 0.01},
             closures=[RoughWall()]))
    kappa = sm.parameters.kappa
    assert any(sp.sympify(sm.source[i, 0]).has(kappa)
               for i in range(sm.n_equations)), "rough-wall bed drag absent from source"


def test_rough_wall_is_bottom_only_bulk_free():
    sm = SystemModel.from_model(SME(level=2, closures=[RoughWall()], parameters={"k_s": 0.01}))
    # bed drag closed, bulk left free → free stress moments survive
    assert any("sigma" in str(s) for s in sm.aux_state)
    lam = sm.parameters
    src = sp.sympify(sm.source[[str(s) for s in sm.state].index("q_0"), 0])
    assert src.has(getattr(lam, "kappa")), "rough-wall bed drag absent from q_0 source"
