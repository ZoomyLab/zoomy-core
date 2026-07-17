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
import pytest

from zoomy_core.model.models import SME
from zoomy_core.model.boundary_conditions import BoundaryConditions, Periodic
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm import timestepping
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.systemmodel.system_model import SystemModel


def test_resolve_periodic_bcs_is_idempotent():
    """One setup_simulation per output frame re-resolves on the SAME
    mesh; a non-idempotent resolve swaps the remap back every second
    call (boundary alternates periodic/open, mass pumps in and out)."""
    NC = 50
    bcs = BoundaryConditions([
        Periodic(tag="left", periodic_to_physical_tag="right"),
        Periodic(tag="right", periodic_to_physical_tag="left"),
    ])
    mesh = BaseMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=NC)
    for _ in range(3):
        mesh.resolve_periodic_bcs(bcs)
        assert (mesh.boundary_face_cells[0] == NC - 1
                and mesh.boundary_face_cells[1] == 0)


def _build_bump(NC=100, XMAX=10.0):
    """Right-moving surface bump on a periodic SME(0) channel — shared by the
    large wrap-around march and its 1-step twin."""
    bcs = BoundaryConditions([
        Periodic(tag="left", periodic_to_physical_tag="right"),
        Periodic(tag="right", periodic_to_physical_tag="left"),
    ])
    sm = SystemModel.from_model(SME(level=0, parameters={"nu": 1e-6, "lambda_s": 0.0},
             boundary_conditions=bcs))
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

    xc = np.linspace(XMAX / NC / 2, XMAX - XMAX / NC / 2, NC)
    dx = XMAX / NC
    m_ic = sum(ic([x])[1] for x in xc) * dx
    return sm, mesh, nsm, NC, dx, m_ic


def test_bump_periodic_one_step_twin(one_hyperbolic_step):
    """Default-tier canary: identical periodic-bump setup, exactly ONE step.
    Periodic BCs mean no boundary-flux imbalance, so mass is conserved to
    machine precision and the internal mesh is remapped to the opposite side
    even after a single step (the wrap-around itself stays in the large tier)."""
    _, mesh, nsm, NC, dx, m_ic = _build_bump()
    solver = HyperbolicSolver(
        time_end=0.5, compute_dt=timestepping.adaptive(CFL=0.9, dimension=1))
    Q = one_hyperbolic_step(solver, mesh, nsm)
    assert solver._bf_cells[0] == NC - 1 and solver._bf_cells[1] == 0
    assert np.all(np.isfinite(Q[:, :NC]))
    np.testing.assert_allclose(Q[1, :NC].sum() * dx, m_ic, rtol=1e-12)


@pytest.mark.large
def test_bump_wraps_and_conserves_mass():
    sm, mesh, nsm, NC, dx, m_ic = _build_bump()
    XMAX = 10.0
    xc = np.linspace(XMAX / NC / 2, XMAX - XMAX / NC / 2, NC)

    # THREE consecutive solves on the same script-level mesh: each
    # setup_simulation takes a fresh internal mesh copy that SHARES the
    # boundary_face_cells array — a remap derived from the current array
    # state (instead of from face_cells[0]) alternates periodic/open
    # across restarts (wrap, leak, wrap, ...).
    Qprev = None
    for _ in range(3):
        if Qprev is not None:
            Qp = Qprev.copy()
            sm.initial_conditions = IC.UserFunction(
                function=lambda xv, Qp=Qp: Qp[
                    :, min(int(float(xv[0]) / dx), NC - 1)])
        solver = HyperbolicSolver(
            time_end=0.5,
            compute_dt=timestepping.adaptive(CFL=0.9, dimension=1))
        Q, _ = solver.solve(mesh, nsm, write_output=False)
        Qprev = np.asarray(Q[:, :NC], float).copy()

        # internal mesh remapped to the opposite side, every restart
        assert solver._bf_cells[0] == NC - 1 and solver._bf_cells[1] == 0
        # mass exactly conserved (periodic: no boundary flux imbalance)
        np.testing.assert_allclose(Qprev[1].sum() * dx, m_ic, rtol=1e-12)
    Q = Qprev

    # the bump crossed the right boundary and re-entered on the left:
    # at u+c ≈ 4.1 the peak (from x=9, t=1.5) must sit mid-domain,
    # far from where extrapolation BCs would leave it (flat ≈ 1.0).
    i_pk = int(np.argmax(Q[1]))
    assert 2.0 < xc[i_pk] < 8.0
    assert Q[1].max() > 1.02
