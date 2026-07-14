"""SME dimension-agnostic: the 2-D (two-horizontal) model.

`SME(dimension=3)` derives the (t, x, y, z) shallow-moment system with the SAME
declarative pipeline as the 1-D model — full stress tensor → moment_scaling →
σ-map → per-direction Galerkin projection → ŵ-closure (coupling û, v̂) →
conservative CoV û_d → q_d/h.  This is the canonical 2-D test case for agents
building on the framework (Steffler, turbulence, coupling).

Pinned here:
* the state layout (q_x_i, q_y_i) and that the system assembles;
* x↔y rotational symmetry of the FLUX (the hyperbolic operator) — the strongest
  structural correctness check for the 2-D derivation;
* a real, bounded HSWME spectrum (truncated-matrix wavespeeds, dimension-
  agnostic by construction — A·n carries n_x, n_y).

NB build cost: SME(dimension=3, level=2) is ~90 s to derive (the 3-family
ŵ-closure + bracket resolution); models are built ONCE per level here.
"""
from functools import lru_cache

import numpy as np
import sympy as sp
import pytest

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel.system_model import SystemModel


@lru_cache(maxsize=None)
def _sm(level):
    return SystemModel.from_model(SME(level=level, dimension=3, parameters={"nu": 0.1, "lambda_s": 0.5},
               closures=[Newtonian(), NavierSlip(), StressFree()]))


@pytest.mark.parametrize("level", [1, 2])
def test_2d_state_layout(level):
    sm = _sm(level)
    expected = (["b", "h"]
                + [f"q_x_{i}" for i in range(level + 1)]
                + [f"q_y_{i}" for i in range(level + 1)])
    assert [str(s) for s in sm.state] == expected
    assert sm.n_dim == 2


@pytest.mark.parametrize("level", [1, 2])
def test_2d_flux_x_y_rotational_symmetry(level):
    """x-flux of q_x_i == y-flux of q_y_i under the swap q_x_i↔q_y_i.  The
    defining correctness property of the 2-D hyperbolic operator.  Numeric over
    random states (sp.simplify on the level-2 2-D flux takes minutes)."""
    sm = _sm(level)
    st = [str(s) for s in sm.state]
    by = {n: s for n, s in zip(st, sm.state)}
    swap = {}
    for i in range(level + 1):
        swap[by[f"q_x_{i}"]] = by[f"q_y_{i}"]
        swap[by[f"q_y_{i}"]] = by[f"q_x_{i}"]
    idx = st.index
    rng = np.random.default_rng(7)
    pars = {p: rng.uniform(0.2, 1.0) for p in sm.parameters.values()}

    def _num(expr, stvals):
        return float(sp.sympify(expr).xreplace({**pars, **stvals}))
    for _ in range(5):
        stvals = {by[n]: rng.uniform(0.3, 1.5) for n in st}
        for i in range(level + 1):
            a = sp.sympify(sm.flux[idx(f"q_x_{i}"), 0]).xreplace(swap)
            b = sp.sympify(sm.flux[idx(f"q_y_{i}"), 1])
            assert np.isclose(_num(a, stvals), _num(b, stvals)), \
                f"q_{i} x/y flux asymmetry"


def test_2d_hswme_spectrum_real_and_bounded():
    sm = _sm(2)
    assert sm.eigenvalue_mode == "symbolic"
    st = [str(s) for s in sm.state]
    by = {n: s for n, s in zip(st, sm.state)}
    rng = np.random.default_rng(0)
    for _ in range(20):
        th = rng.uniform(0, 2 * np.pi)
        sub = {by["h"]: rng.uniform(0.2, 2.0), by["b"]: 0.0, sm.parameters.g: 9.81,
               sm.normal[0]: np.cos(th), sm.normal[1]: np.sin(th)}
        sub.update({s: rng.uniform(-0.3, 0.3) for s in sm.state if s not in sub})
        sub.update({p: 0.1 for p in sm.parameters.values() if p not in sub})
        vals = [complex(sp.sympify(e).xreplace(sub)) for e in sm.eigenvalues]
        assert all(abs(v.imag) < 1e-9 for v in vals), "spectrum must be real (hyperbolic)"
        assert all(np.isfinite(v.real) for v in vals)
