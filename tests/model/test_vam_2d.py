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
from zoomy_core.systemmodel.system_model import SystemModel


@pytest.fixture(scope="module")
def vam2d():
    return SystemModel.from_model(VAM(level=1, dimension=3,
               closures=[NavierSlip(), StressFree()]))


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


def test_vam_2d_pressure_couples_into_momentum(vam2d):
    """The non-hydrostatic pressure MUST couple into BOTH horizontal momenta
    (q_x_i, q_y_i) and the vertical (r_i) — exactly as in dim=2.  Regression
    for the hard-coded single-horizontal P_head(j,t,x) in `system_model`, which
    dropped every P mode from the dim=3 momentum rows (empty corrector →
    chorin_split ZeroDivisionError)."""
    st = [str(s) for s in vam2d.state]
    e2s = list(vam2d.equation_to_state_index)
    Ps = {s for s in vam2d.state if str(s).startswith("P_")}
    Pn = {str(p) for p in Ps}
    res = vam2d.reconstruct_residuals()
    carry = {
        st[e2s[i]] for i in range(vam2d.n_equations)
        if (sp.sympify(res[i]).free_symbols & Ps)
        or any(getattr(a, "func", a).__name__ in Pn
               for a in sp.sympify(res[i]).atoms(sp.Function))
    }
    assert carry == {"q_x_0", "q_x_1", "q_y_0", "q_y_1", "r_0", "r_1"}


def test_vam_2d_chorin_split(vam2d):
    """The dim=3 Chorin split builds a non-empty corrector / pressure block."""
    from zoomy_core.model.models import VAM
    from zoomy_core.model.models.closures import NavierSlip, StressFree
    v = VAM(level=1, dimension=3, closures=[NavierSlip(), StressFree()])
    out = v.chorin_split(sp.Symbol("dt", positive=True))
    assert out.SM_corr.n_equations == 6      # 2 q_x + 2 q_y + 2 r rows
    assert out.SM_press.n_equations == 2     # 2 pressure constraints
