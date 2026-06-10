"""End-to-end pipeline test: declarative VAM → square DAE SystemModel →
structural Chorin split → numpy split solver, on a dam break over a bump.

Exercises the non-hydrostatic chain the way a model USER drives it:
  * ``VAM(level=1).system_model`` — square 8×8 DAE
    (state ``[b, h, q_0, q_1, r_0, r_1, P_0, P_1]``; the pressure rows are
    divergence constraints with zero mass-matrix rows);
  * ``split_for_pressure_structural`` via ``VAM.chorin_split`` — row roles
    from the OPERATORS, no equation-name conventions;
  * ``ChorinSplitVAMSolver`` (predictor / LU-elliptic pressure / corrector).
"""
import numpy as np
import pytest
import sympy as sp

from zoomy_core.model.models import VAM
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver


@pytest.fixture(scope="module")
def vam_sm():
    return VAM(level=1).system_model


def test_vam_is_a_square_dae_with_pressure_constraints(vam_sm):
    sm = vam_sm
    assert [str(s) for s in sm.state] == [
        "b", "h", "q_0", "q_1", "r_0", "r_1", "P_0", "P_1"]
    assert sm.n_equations == sm.n_state
    M = sp.Matrix(sm.mass_matrix.tolist())
    zero_rows = [i for i in range(sm.n_equations)
                 if all(M[i, j] == 0 for j in range(sm.n_state))]
    # exactly the two divergence constraints (the P_0 / P_1 rows)
    assert zero_rows == [6, 7]
    # the dynamic rows carry their ∂_t
    for i in range(6):
        assert M[i, i] != 0


def test_vam_chorin_split_roles(vam_sm):
    split = VAM(level=1).chorin_split(system_model=vam_sm)
    assert split.SM_pred.equation_names == [
        "pred_b", "pred_h", "pred_q_0", "pred_q_1", "pred_r_0", "pred_r_1"]
    assert split.SM_press.equation_names == ["elliptic_P_0", "elliptic_P_1"]
    assert split.SM_corr.equation_names == [
        "corr_q_0", "corr_q_1", "corr_r_0", "corr_r_1"]
    assert split.SM_press.equation_to_state_index == [6, 7]
    assert split.SM_corr.state_update is not None


def test_vam_dambreak_over_bump():
    """Short dam break over a Gaussian bump: finite, positive depth, mass
    conserved, bounded non-hydrostatic pressure."""
    nc = 60
    model = VAM(level=1)
    sm = model.system_model

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

    split = model.chorin_split(system_model=sm)
    # the predictor CALLS its BC kernel → rebuild against its own aux
    split.SM_pred.attach_boundary_conditions(bcs)

    mesh = BaseMesh.create_1d(domain=(-1.5, 1.5), n_inner_cells=nc)
    solver = ChorinSplitVAMSolver(split.SM_pred, split.SM_press,
                                  split.SM_corr, pressure_solver="lu",
                                  riemann_solver="hr")
    solver.setup_simulation(mesh)
    state_names = [str(s) for s in sm.state]
    ih, ip0 = state_names.index("h"), state_names.index("P_0")
    dx = 3.0 / nc
    mass0 = solver._sim_Q[ih, :nc].sum() * dx
    dt = 2e-4
    for _ in range(40):
        solver.step(dt)
    Q = solver._sim_Q
    h = Q[ih, :nc]
    assert np.all(np.isfinite(Q[:, :nc]))
    assert h.min() > 0.0
    mass1 = h.sum() * dx
    assert abs(mass1 - mass0) < 1e-4 * mass0
    # non-hydrostatic pressure active but bounded (no Chorin drift)
    P0 = np.abs(Q[ip0, :nc]).max()
    assert 1e-4 < P0 < 2.0
