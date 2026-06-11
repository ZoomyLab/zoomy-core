"""End-to-end test: declarative multilayer non-hydrostatic VAM →
square DAE → structural Chorin split → numpy split solver, dam break
over a bump.

Per-layer VAM columns (hydrostatic/non-hydrostatic split, downward
pressure-trace convention), Hörnschemeyer fraction-multiplier G, shared
u*/w* transfers.
"""
import numpy as np
import pytest
import sympy as sp

from zoomy_core.model.models import MLVAM
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver


@pytest.fixture(scope="module")
def mlvam():
    return MLVAM(n_layers=2, level=1)


def test_mlvam_is_a_square_dae(mlvam):
    sm = mlvam.system_model
    assert [str(s) for s in sm.state] == [
        "b", "h", "q_1_0", "q_1_1", "q_2_0", "q_2_1",
        "r_1_0", "r_1_1", "r_2_0", "r_2_1",
        "P_1_0", "P_1_1", "P_2_0", "P_2_1"]
    assert sm.n_equations == sm.n_state
    M = sp.Matrix(sm.mass_matrix.tolist())
    zero_rows = [i for i in range(sm.n_equations)
                 if all(M[i, j] == 0 for j in range(sm.n_state))]
    # one divergence constraint per pressure mode, per layer
    assert zero_rows == [10, 11, 12, 13]


def test_mlvam_split_has_all_correctors(mlvam):
    """The downward pressure-trace convention keeps EVERY velocity row
    pressure-coupled — 8 corrector rows (the upward convention loses
    corr_r_1_1 and the elliptic block goes singular)."""
    split = mlvam.chorin_split(system_model=mlvam.system_model)
    assert split.SM_press.equation_to_state_index == [10, 11, 12, 13]
    assert split.SM_corr.equation_names == [
        "corr_q_1_0", "corr_q_1_1", "corr_q_2_0", "corr_q_2_1",
        "corr_r_1_0", "corr_r_1_1", "corr_r_2_0", "corr_r_2_1"]


def test_mlvam_dambreak_over_bump(mlvam):
    sm = mlvam.system_model

    def _bump_ic(xv):
        xx = float(xv[0])
        bv = 0.20 * np.exp(-(xx**2) / (2 * 0.20**2))
        hv = max((0.34 - bv) if xx < 0.0 else 0.015, 1e-6)
        out = np.zeros(len(sm.state)); out[0] = bv; out[1] = hv
        return out

    sm.initial_conditions = IC.UserFunction(function=_bump_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    bcs = BoundaryConditions([Extrapolation(tag="left"),
                              Extrapolation(tag="right")])
    sm.attach_boundary_conditions(bcs)
    split = mlvam.chorin_split(system_model=sm)
    split.SM_pred.attach_boundary_conditions(bcs)

    nc = 60
    mesh = BaseMesh.create_1d(domain=(-1.5, 1.5), n_inner_cells=nc)
    solver = ChorinSplitVAMSolver(split.SM_pred, split.SM_press,
                                  split.SM_corr, pressure_solver="lu",
                                  riemann_solver="hr")
    solver.setup_simulation(mesh)
    sn = [str(s) for s in sm.state]
    ih = sn.index("h")
    dx = 3.0 / nc
    mass0 = solver._sim_Q[ih, :nc].sum() * dx
    for _ in range(40):
        solver.step(2e-4)
    Q = solver._sim_Q
    h = Q[ih, :nc]
    assert np.all(np.isfinite(Q[:, :nc]))
    assert h.min() > 0.0
    assert abs(h.sum() * dx - mass0) < 1e-4 * mass0
    # all four pressure modes active and bounded
    for p in ("P_1_0", "P_1_1", "P_2_0", "P_2_1"):
        pm = np.abs(Q[sn.index(p), :nc]).max()
        assert 1e-6 < pm < 5.0
