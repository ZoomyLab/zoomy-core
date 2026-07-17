"""End-to-end test: declarative multilayer SME (Hörnschemeyer closure) →
SystemModel → numpy solver, sheared two-layer moment dam break."""
import numpy as np
import pytest
import sympy as sp

from zoomy_core.model.models import MLSME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel


def test_mlsme_structure():
    mod = MLSME(closures=[Newtonian(), NavierSlip(), StressFree()], n_layers=2, level=1, interface_velocity="mean")
    sm = SystemModel.from_model(mod)
    assert [str(s) for s in sm.state] == [
        "b", "h", "q_1_0", "q_1_1", "q_2_0", "q_2_1"]
    M = sp.Matrix(sm.mass_matrix.tolist())
    # Gram-normalized at extraction (runtime integrates ∂_t Q = RHS)
    assert M == sp.eye(6)
    # same fraction-multiplier G as the level-0 MLSWE (k=0 continuities)
    G = sp.sympify(list(mod.derivation.G_closed.values())[0])
    rho, l1 = sp.Symbol("rho", positive=True), sp.Symbol("l_1", positive=True)
    t_, x_ = sm.time, sm.space[0]
    q1 = sp.Function("q_1", real=True)(0, t_, x_)
    q2 = sp.Function("q_2", real=True)(0, t_, x_)
    expected = rho * (l1 * sp.Derivative(q1 + q2, x_) - sp.Derivative(q1, x_))
    assert sp.simplify(G - expected.doit()) == 0


def _build(ustar, nc=100):
    """Shared sheared two-layer MLSME dam-break setup for the large march and
    its 1-step twin."""
    sm = SystemModel.from_model(MLSME(closures=[Newtonian(), NavierSlip(), StressFree()], n_layers=2, level=1, interface_velocity=ustar,
               boundary_conditions=BoundaryConditions(
                   [Extrapolation(tag="left"), Extrapolation(tag="right")])
               ))

    def _ic(xv):
        htv = 1.5 if float(xv[0]) < 5.0 else 0.9
        out = np.zeros(len(sm.state))
        out[1] = htv
        out[4] = 0.1 * htv          # q_2_0: top layer moving
        return out

    sm.initial_conditions = IC.UserFunction(function=_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=nc)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    return sm, mesh, nsm, nc


@pytest.mark.parametrize("ustar", ["upwind", "mean"])
def test_mlsme_dambreak_one_step_twin(ustar, one_hyperbolic_step):
    """Default-tier canary: identical MLSME setup, exactly ONE step. Cheap
    invariants only (finite, positive depth, bounded mass change)."""
    _, mesh, nsm, nc = _build(ustar)
    solver = HyperbolicSolver(time_end=0.3,
                              compute_dt=timestepping.adaptive(CFL=0.45))
    Q = one_hyperbolic_step(solver, mesh, nsm)
    h = Q[1, :nc]
    assert np.all(np.isfinite(Q[:, :nc]))
    assert h.min() > 0.0
    assert abs(h.sum() * (10.0 / nc) - 12.0) < 0.05


@pytest.mark.large
@pytest.mark.parametrize("ustar", ["upwind", "mean"])
def test_mlsme_dambreak_with_shear(ustar):
    _, mesh, nsm, nc = _build(ustar)
    solver = HyperbolicSolver(time_end=0.3,
                              compute_dt=timestepping.adaptive(CFL=0.45))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    h = np.asarray(Q[1, :nc], float)
    assert np.all(np.isfinite(Q[:, :nc]))
    assert h.min() > 0.0
    assert abs(h.sum() * (10.0 / nc) - 12.0) < 0.05
    # the inter-layer shear excites the in-layer moments, boundedly
    q11 = np.abs(np.asarray(Q[3, :nc], float)).max()
    q21 = np.abs(np.asarray(Q[5, :nc], float)).max()
    assert 1e-7 < q11 < 0.5 and 1e-7 < q21 < 0.5
