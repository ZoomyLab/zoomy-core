"""Phase 3 — end-to-end smoke of the new closure/blueprint structure:
derive a full model (SME via Mass()/Momentum() blueprints + composable
closures) → SystemModel → solve a dam break with the numpy FVM solver.

This is the integration gate other agents watch for: it proves the Phase-1
closure interface and the Phase-2 equation blueprints carry all the way through
to a running numerical solution.
"""
import numpy as np

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import Extrapolation
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec


def _build_sme(level=2):
    """SME built entirely through the NEW structure: Mass()/Momentum()
    blueprints (Phase 2) closed by the composable closure list (Phase 1), with
    the NEW flat per-field boundary-condition list."""
    return SME(
        level=level,
        parameters={"nu": 1e-3, "lambda_s": 0.0},
        closures=[Newtonian(), NavierSlip(), StressFree()],
        boundary_conditions=[Extrapolation("left"), Extrapolation("right")],
    ).system_model


def _solve_dambreak(sm, nc=100, t_end=0.3):
    def _ic(xv):
        h = 1.5 if float(xv[0]) < 5.0 else 0.75
        return np.array([0.0, h] + [0.0] * (sm.n_equations - 2))

    sm.initial_conditions = IC.UserFunction(function=_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=nc)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=t_end,
                              compute_dt=timestepping.adaptive(CFL=0.45))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    return np.asarray(Q, float), mesh, nc


def test_phase3_sme_closures_blueprints_numpy_dambreak():
    sm = _build_sme(level=2)
    assert [str(s) for s in sm.state] == ["b", "h", "q_0", "q_1", "q_2"]

    Q, mesh, nc = _solve_dambreak(sm)
    h = Q[1, :nc]
    # finite, strictly positive depth; bounded; mass between the two dam levels
    assert np.all(np.isfinite(Q)), "solution went non-finite"
    assert np.all(h > 0.0), "depth went non-positive"
    assert h.max() <= 1.6 and h.min() >= 0.7, f"depth out of band: {h.min()}..{h.max()}"
    # mass conservation (Extrapolation BC ~ closed-ish over short time): the
    # mean depth stays close to the IC mean (small outflow over 0.3 s).
    mean0 = 0.5 * (1.5 + 0.75)
    assert abs(h.mean() - mean0) < 0.15, f"mean depth drifted: {h.mean()} vs {mean0}"
