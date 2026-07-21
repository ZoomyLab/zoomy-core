"""The emitted restart actually reaches the Chorin pressure solve.

``test_elliptic_constants.py`` pins the RULE (the restart is a function of the
block size).  This pins the WIRING: the solver resolves that rule against its
own block and runs GMRES with the result, instead of scipy's fixed default of
20.  Without this, the rule could be perfectly correct and entirely unused —
which is the state this change found the code in.
"""

import numpy as np
import pytest

import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation)
from zoomy_core.model.models import VAM
from zoomy_core.model.models.closures import NavierSlip, Newtonian, StressFree
from zoomy_core.solver.constants import elliptic_restart
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.solver, pytest.mark.small, pytest.mark.gate]

NC = 24
PAR = {"lambda_s": 0.5, "nu": 1e-3}


def _bcs():
    return BoundaryConditions([Extrapolation(tag="left"),
                               Extrapolation(tag="right")])


def _solver(mesh, restart):
    vam = VAM(closures=[Newtonian(), NavierSlip(), StressFree()], level=1,
              parameters=dict(PAR), boundary_conditions=_bcs())
    sm = SystemModel.from_model(vam)

    n_state = len(sm.state)

    def ic(xv):
        out = np.zeros(n_state)
        out[1] = 0.5 + 0.05 * np.exp(-((float(xv[0]) - 0.5) ** 2) / 0.045)
        return out

    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda k: np.zeros(k))
    bcs = _bcs()
    sm.attach_boundary_conditions(bcs)
    split = vam.chorin_split(system_model=sm)
    split.SM_pred.attach_boundary_conditions(bcs)
    sol = ChorinSplitVAMSolver(split.SM_pred, split.SM_press, split.SM_corr,
                               pressure_solver="gmres",
                               pressure_restart=restart,
                               riemann_solver="hr")
    sol.setup_simulation(mesh)
    return sol


@pytest.fixture(scope="module")
def mesh():
    return BaseMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=NC)


def test_default_restart_is_resolved_from_the_block_not_scipy_default(mesh):
    """The default must be the SIZE-RESOLVED value.  scipy's own default is a
    fixed 20 — landing on that for a block this size would mean the parameter
    is not plumbed through at all."""
    sol = _solver(mesh, None)
    sol.step(1e-4)
    n_modes = len(sol._press_state_idx)
    expected = elliptic_restart(n_modes * NC)
    assert sol.last_elliptic_restart == expected
    assert sol.last_elliptic_restart != 20, (
        "resolved restart coincides with scipy's fixed default — pick a mesh "
        "size where the two differ, or the wiring is untested")


def test_a_pinned_restart_is_honoured(mesh):
    """Pinning is how the measured degradation is reproduced deliberately; it
    must reach the solve unchanged."""
    sol = _solver(mesh, 20)
    sol.step(1e-4)
    assert sol.last_elliptic_restart == 20
