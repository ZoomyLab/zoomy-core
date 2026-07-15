"""REQ-166: plain SWE's Manning source Jacobian must be FINITE at rest.

The 2-D term ``hu·sqrt(hu²+hv²)`` has derivative ``sqrt(...)+hu²/sqrt(...)`` →
0/0 at zero velocity, which NaNs the implicit/IMEX Newton on step 0 of any
from-rest run.  The ``vel_eps`` regularizer keeps ∂S/∂Q finite everywhere.
"""
import numpy as np
import sympy as sp

from zoomy_core.systemmodel import SystemModel
from zoomy_core.model.models.swe import SWE


def _jac_nonfinite_at_rest(dim):
    sm = SystemModel.from_model(
        SWE(dimension=dim, parameters={"g": 9.81, "n_m": 0.033}))
    J = sm.source_jacobian_wrt_variables
    subs = {}
    for s in sm.state:
        subs[s] = 2.0 if str(s) == "h" else 0.0            # rest: hu=hv=0
    for a in sm.aux_state:
        subs[a] = 0.5 if str(a) == "hinv" else 0.0
    for row in range(J.shape[0]):
        for col in range(J.shape[1]):
            e = sp.sympify(J[row, col])
            if e == 0:
                continue
            for p in e.free_symbols:
                nm = str(p)
                subs.setdefault(p, {"g": 9.81, "n_m": 0.033,
                                    "vel_eps": 1e-12}.get(nm, 0.0))
            val = complex(sp.N(e.subs(subs)))
            if not np.isfinite(val.real):
                return True
    return False


def test_swe_2d_manning_jacobian_finite_at_rest():
    assert not _jac_nonfinite_at_rest(2)


def test_swe_1d_manning_jacobian_finite_at_rest():
    assert not _jac_nonfinite_at_rest(1)
