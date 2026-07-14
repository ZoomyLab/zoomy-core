"""Multilayer models, dimension-agnostic: the 2-D (two-horizontal) versions.

`MLSME(dimension=3)` / `MLSWE(dimension=3)` / `MLVAM(dimension=3)` derive each
layer with the SME/VAM 2-D pipeline (mass from the dimension-agnostic Mass
blueprint, per-direction momentum) and generalise the interface coupling — the
Hörnschemeyer mass-flux closure (∇·-based) and the transfer velocity u* — per
horizontal direction.

Pinned: 2-D state layout (q_x_ℓ_i, q_y_ℓ_i) and x↔y flux rotational symmetry.
"""
import sympy as sp
import pytest

from zoomy_core.model.models import MLSME, MLSWE, MLVAM
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel.system_model import SystemModel

PARS = {"nu": 0.1, "lambda_s": 0.5}
CLO = [Newtonian(), NavierSlip(), StressFree()]


def _flux_symmetry(sm, layers, modes):
    """x-flux of q_x_ℓ_k == y-flux of q_y_ℓ_k under q_x↔q_y."""
    st = [str(s) for s in sm.state]
    by = {n: s for n, s in zip(st, sm.state)}
    swap = {}
    for ell in layers:
        for k in modes:
            swap[by[f"q_x_{ell}_{k}"]] = by[f"q_y_{ell}_{k}"]
            swap[by[f"q_y_{ell}_{k}"]] = by[f"q_x_{ell}_{k}"]
    idx = st.index
    for ell in layers:
        for k in modes:
            a = sp.sympify(sm.flux[idx(f"q_x_{ell}_{k}"), 0]).xreplace(swap)
            b = sp.sympify(sm.flux[idx(f"q_y_{ell}_{k}"), 1])
            assert sp.simplify(a - b) == 0, f"layer {ell} mode {k} x/y asymmetry"


def test_mlsme_2d():
    sm = SystemModel.from_model(MLSME(n_layers=2, level=1, dimension=3, parameters=PARS,
               closures=CLO))
    assert sm.n_dim == 2
    st = [str(s) for s in sm.state]
    assert "q_x_1_0" in st and "q_y_2_1" in st
    _flux_symmetry(sm, layers=(1, 2), modes=(0, 1))


def test_mlswe_2d():
    sm = SystemModel.from_model(MLSWE(n_layers=2, dimension=3, parameters=PARS,
               closures=CLO))
    assert sm.n_dim == 2
    st = [str(s) for s in sm.state]
    assert "q_x_1_0" in st and "q_y_2_0" in st
    _flux_symmetry(sm, layers=(1, 2), modes=(0,))


def test_mlvam_2d():
    sm = SystemModel.from_model(MLVAM(n_layers=2, level=1, dimension=3, parameters=PARS,
               closures=CLO))
    assert sm.n_dim == 2
    st = [str(s) for s in sm.state]
    # 2-D: q_x/q_y per layer; r (w-moments) and P (pressure) stay scalar
    assert "q_x_1_0" in st and "q_y_2_1" in st
    assert "r_1_0" in st and "P_2_1" in st
    _flux_symmetry(sm, layers=(1, 2), modes=(0, 1))
