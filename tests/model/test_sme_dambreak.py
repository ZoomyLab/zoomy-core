"""End-to-end pipeline test: the new declarative SME → SystemModel → numpy FVM
solver, on a 1-D dam break.

Exercises the full tool chain the way a model USER drives it:
  * `SME(level=…).system_model` — the declarative model → operator form;
  * `attach_boundary_conditions(...)` — inject the SAME old `BoundaryConditions`
    objects (the hook; the model is not bound to them);
  * numerical eigenvalues by DEFAULT (no symbolic spectrum needed — the
    SystemModel reports `eigenvalue_mode == "numerical"` when `eigenvalues is
    None`, and the solver builds wavespeeds from the numerical Jacobian);
  * `BaseMesh.create_1d(...)` — simple 1-D mesh generation;
  * `RP` Riemann-problem initial condition — the dam break.

A passing run = the new SME/SWE model solves a real test case.
"""
import numpy as np
import pytest

from zoomy_core.model.models import SME
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.initial_conditions import RP, Constant
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec


def _run_dambreak(level, *, hL=2.0, hR=1.0, n_cells=100, t_end=0.3):
    """Build the new SME at `level` with BCs as always, run a 1-D dam break."""
    # the NORMAL interface — BoundaryConditions in the model constructor,
    # exactly as the production models always took them (transmissive here)
    sm = SME(level=level, boundary_conditions=BoundaryConditions(
        [Extrapolation(tag="left"), Extrapolation(tag="right")])).system_model
    n_state = len(sm.state)
    # state = [b, h, q_0, …]; bed b=0, dam: h=hL left of x=5, hR right, momenta 0
    high = np.zeros(n_state); high[1] = hL
    low = np.zeros(n_state);  low[1] = hR
    sm.initial_conditions = RP(high=lambda n, hi=high: hi,
                               low=lambda n, lo=low: lo, jump_position_x=5.0)
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))

    assert sm.eigenvalue_mode == "numerical"          # numerical by default

    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=n_cells)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=t_end,
                              compute_dt=timestepping.adaptive(CFL=0.9))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    return np.asarray(Q[1, :n_cells], dtype=float)


@pytest.mark.parametrize("level", [0, 2])      # SWE (level 0) and SME (level 2)
def test_sme_dambreak_1d(level):
    hL, hR = 2.0, 1.0
    h = _run_dambreak(level, hL=hL, hR=hR)
    # finite, mass-positive, bounded by the two initial states
    assert np.all(np.isfinite(h))
    assert h.min() > 0.0
    assert hR - 1e-6 <= h.min() and h.max() <= hL + 1e-6
    # the dam actually broke: an intermediate state propagated inward
    intermediate = np.count_nonzero((h > hR + 0.05) & (h < hL - 0.05))
    assert intermediate >= 5, f"no dam-break wave (level={level}): {intermediate} cells"
