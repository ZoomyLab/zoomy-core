"""REQ-167/168: with no closed-form spectrum the wave speed must be the
DIMENSIONALLY CORRECT numerical spectrum ``max|λ(A_n)|`` (via the opaque
``eigenvalues`` kernel), not the Gershgorin row-sum of a dimensionally-
inhomogeneous matrix (which returned ~g·h, ~8.9× too large at h=2).
"""
import math

import sympy as sp

from zoomy_core.systemmodel import SystemModel
from zoomy_core.model.models.swe import SWE
from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
from zoomy_core.fvm import userfunctions as uf


def _wave_speed(sm, Q, Qaux, p, n):
    sm.eigenvalues = None                     # force the numerical path
    expr = NonconservativeRusanov(sm).local_max_abs_eigenvalue(Q, Qaux, p, n)
    return float(sp.lambdify([], expr, modules=[uf.numpy_module()])())


def _swe2d_setup():
    sm = SystemModel.from_model(SWE(dimension=2, parameters={"g": 9.81}))
    names = [str(s) for s in sm.state]
    pars = [str(pp) for pp in sm.parameters.values()]
    pvals = {"g": 9.81, "n_m": 0.0, "vel_eps": 1e-12, "h_in": 0.0, "q_in": 0.0}
    p = [pvals.get(pp, 0.0) for pp in pars]
    Qaux = [0.0 for _ in sm.aux_state]
    return sm, names, p, Qaux


def test_swe2d_rest_wave_speed_is_sqrt_gh():
    sm, names, p, Qaux = _swe2d_setup()
    Q = [2.0 if nm == "h" else 0.0 for nm in names]
    got = _wave_speed(sm, Q, Qaux, p, [1.0, 0.0])
    assert abs(got - math.sqrt(9.81 * 2.0)) < 1e-9   # 4.4294, was 39.24


def test_swe2d_moving_wave_speed_is_u_plus_sqrt_gh():
    sm, names, p, Qaux = _swe2d_setup()
    Q = [0.0] * len(names)
    Q[names.index("h")] = 2.0
    Q[names.index("hu")] = 2.0                        # u = 1
    got = _wave_speed(sm, Q, Qaux, p, [1.0, 0.0])
    assert abs(got - (1.0 + math.sqrt(9.81 * 2.0))) < 1e-9
