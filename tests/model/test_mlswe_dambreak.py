"""End-to-end test: declarative multilayer SWE (Hörnschemeyer closure) →
SystemModel → numpy solver, two-layer dam break with shear.

The internal-interface mass flux is the Lagrange multiplier of the
layer-fraction constraint, ``G_1 = ρ(l_1·∂_x(q_1+q_2) − ∂_x q_1)``; the
momentum-transfer velocity ``u*`` is shared by the adjacent rows
(sign-of-G upwind by default, mean as option).
"""
import numpy as np
import pytest
import sympy as sp

from zoomy_core.model.models import MLSWE
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec


def test_mlswe_structure_and_closed_G():
    mod = MLSWE(closures=[Newtonian(), NavierSlip(), StressFree()], n_layers=2, interface_velocity="mean")
    sm = mod.system_model
    assert [str(s) for s in sm.state] == ["b", "h", "q_1_0", "q_2_0"]
    M = sp.Matrix(sm.mass_matrix.tolist())
    assert M == sp.eye(4)
    # the Hörnschemeyer multiplier, pinned: G_1 = ρ(l_1·∂x(q_1+q_2) − ∂x q_1)
    G = sp.sympify(list(mod._G_closed.values())[0])
    rho, l1 = sp.Symbol("rho", positive=True), sp.Symbol("l_1", positive=True)
    t_, x_ = sm.time, sm.space[0]
    q1 = sp.Function("q_1", real=True)(0, t_, x_)
    q2 = sp.Function("q_2", real=True)(0, t_, x_)
    expected = rho * (l1 * sp.Derivative(q1 + q2, x_) - sp.Derivative(q1, x_))
    assert sp.simplify(G - expected.doit()) == 0


@pytest.mark.parametrize("ustar", ["upwind", "mean"])
def test_mlswe_dambreak_with_shear(ustar):
    """Sheared two-layer dam break: finite, positive depth, mass conserved,
    shear bounded — for both interface-velocity variants."""
    nc = 100
    sm = MLSWE(closures=[Newtonian(), NavierSlip(), StressFree()], n_layers=2, interface_velocity=ustar,
               boundary_conditions=BoundaryConditions(
                   [Extrapolation(tag="left"), Extrapolation(tag="right")])
               ).system_model

    def _ic(xv):
        htv = 1.5 if float(xv[0]) < 5.0 else 0.9
        return np.array([0.0, htv, 0.0, 0.1 * htv])   # top layer moving

    sm.initial_conditions = IC.UserFunction(function=_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=nc)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=0.3,
                              compute_dt=timestepping.adaptive(CFL=0.45))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    h = np.asarray(Q[1, :nc], float)
    q1 = np.asarray(Q[2, :nc], float)
    q2 = np.asarray(Q[3, :nc], float)
    assert np.all(np.isfinite(Q[:, :nc]))
    assert h.min() > 0.0
    dx = 10.0 / nc
    mass = h.sum() * dx
    assert abs(mass - 12.0) < 0.05          # boundary outflow only
    shear = np.abs(q2 - q1).max()
    assert 1e-4 < shear < 1.0               # shear alive and bounded
