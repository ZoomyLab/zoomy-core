"""VAM dimension-agnostic: the 2-D (two-horizontal) non-hydrostatic model.

`VAM(dimension=3)` derives the (t,x,y,z) Escalante moment system with the same
pipeline as 1-D — per-direction Galerkin projection, the ŵ/p̂ top-mode closures
(2-D bottom kinematic w(0)=Σ_d u_d(0)·∂_d b), the σ-mass-flux ω̃ correction with
the ∂_y terms, the conservative flux DERIVED per direction from our ansatz, and
the bed-curvature ∂²b terms rewritten as conservative compounds.

Pinned: the 2-D state layout (q_x_i, q_y_i + scalar r_i, P_i), and x↔y flux
rotational symmetry (the defining 2-D correctness check).
"""
import sympy as sp
import numpy as np
import pytest

from zoomy_core.model.models import VAM
from zoomy_core.model.models.closures import NavierSlip, StressFree


@pytest.fixture(scope="module")
def vam2d():
    return VAM(level=1, dimension=3,
               closures=[NavierSlip(), StressFree()]).system_model


def test_vam_2d_state_layout(vam2d):
    st = [str(s) for s in vam2d.state]
    assert st == ["b", "h", "q_x_0", "q_x_1", "q_y_0", "q_y_1",
                  "r_0", "r_1", "P_0", "P_1"]
    assert vam2d.n_dim == 2


def test_vam_2d_flux_x_y_symmetry(vam2d):
    """x-flux of q_x_i == y-flux of q_y_i under q_x_i↔q_y_i (r, P scalar)."""
    st = [str(s) for s in vam2d.state]
    by = {n: s for n, s in zip(st, vam2d.state)}
    swap = {}
    for i in range(2):
        swap[by[f"q_x_{i}"]] = by[f"q_y_{i}"]
        swap[by[f"q_y_{i}"]] = by[f"q_x_{i}"]
    idx = st.index
    for i in range(2):
        a = sp.sympify(vam2d.flux[idx(f"q_x_{i}"), 0]).xreplace(swap)
        b = sp.sympify(vam2d.flux[idx(f"q_y_{i}"), 1])
        assert sp.simplify(a - b) == 0, f"q_{i} x/y flux asymmetry"
