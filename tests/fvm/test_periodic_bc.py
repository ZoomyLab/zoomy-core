"""Periodic BCs on the declarative SystemModel path (numpy solver).

Pins the two breaks that made ``Periodic`` silently act as extrapolation:

1. ``setup_simulation`` only resolved periodic topology when handed a
   production ``Model`` (``source_model``); the declarative path (NSM /
   SystemModel) must fall back to ``sm._bc_source``.
2. The flux operator overrode Q_R at boundary faces with ``bc_fn(Q_L)``;
   the Periodic kernel is a pass-through, so the wrap degraded to
   extrapolation — the face value must be the opposite-side cell state.

Symptom of either break: a bump advects OUT of the domain instead of
wrapping (mass leaks / perturbations decay instead of re-entering).
"""

import numpy as np

from zoomy_core.model.models import SME
from zoomy_core.model.boundary_conditions import BoundaryConditions, Periodic
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm import timestepping
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC


def test_bump_wraps_and_conserves_mass():
    NC, XMAX = 100, 10.0
    bcs = BoundaryConditions([
        Periodic(tag="left", periodic_to_physical_tag="right"),
        Periodic(tag="right", periodic_to_physical_tag="left"),
    ])
    sm = SME(level=0, parameters={"nu": 1e-6, "lambda_s": 0.0},
             boundary_conditions=bcs).system_model
    n_state = len(sm.state)

    def ic(xv):
        x = float(xv[0])
        out = np.zeros(n_state)
        out[1] = 1.0 + 0.3 * np.exp(-((x - 9.0) ** 2) / 0.1)
        out[2] = out[1] * 1.0          # u = 1: bump moves right, must wrap
        return out

    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, XMAX), n_inner_cells=NC)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))

    solver = HyperbolicSolver(
        time_end=1.5,
        compute_dt=timestepping.adaptive(CFL=0.9, dimension=1))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    Q = np.asarray(Q[:, :NC], float)
    xc = np.linspace(XMAX / NC / 2, XMAX - XMAX / NC / 2, NC)
    dx = XMAX / NC

    # boundary_face_cells remapped to the opposite side
    assert solver._bf_cells[0] == NC - 1 and solver._bf_cells[1] == 0

    # mass exactly conserved (periodic: no boundary flux imbalance)
    m_ic = sum(ic([x])[1] for x in xc) * dx
    np.testing.assert_allclose(Q[1].sum() * dx, m_ic, rtol=1e-12)

    # the bump crossed the right boundary and re-entered on the left:
    # at u+c ≈ 4.1 the peak (from x=9, t=1.5) must sit mid-domain,
    # far from where extrapolation BCs would leave it (flat ≈ 1.0).
    i_pk = int(np.argmax(Q[1]))
    assert 2.0 < xc[i_pk] < 8.0
    assert Q[1].max() > 1.02
