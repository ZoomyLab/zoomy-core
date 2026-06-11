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

from zoomy_core.model.models import SME, newtonian_navier_slip
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.initial_conditions import RP, Constant
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation, FromModel)
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec


def _run_dambreak(level, *, hL=2.0, hR=1.0, n_cells=100, t_end=0.3):
    """Build the new SME at `level` with BCs as always, run a 1-D dam break."""
    # the NORMAL interface — BoundaryConditions in the model constructor,
    # exactly as the production models always took them (transmissive here)
    sm = SME(material=newtonian_navier_slip(), level=level, boundary_conditions=BoundaryConditions(
        [Extrapolation(tag="left"), Extrapolation(tag="right")])).system_model
    n_state = len(sm.state)
    # state = [b, h, q_0, …]; bed b=0, dam: h=hL left of x=5, hR right, momenta 0
    high = np.zeros(n_state); high[1] = hL
    low = np.zeros(n_state);  low[1] = hR
    sm.initial_conditions = RP(high=lambda n, hi=high: hi,
                               low=lambda n, lo=low: lo, jump_position_x=5.0)
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))

    # symbolic since the beta-HSWME spectrum registration (closed-form
    # wavespeeds, no per-face LAPACK eigensolves)
    assert sm.eigenvalue_mode == "symbolic"

    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=n_cells)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=t_end,
                              compute_dt=timestepping.adaptive(CFL=0.9))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    return np.asarray(Q[1, :n_cells], dtype=float)


def test_wall_bc_comes_from_the_model():
    """The wall is DEFINED in the derivation (register_group('boundary:wall'))
    and accessed at runtime via FromModel — same signature as every other BC."""
    sm = SME(material=newtonian_navier_slip(), level=2).system_model
    bc = FromModel(tag="left", definition="wall").resolve(sm)
    ghost = bc.compute_boundary_condition(
        sm.time, sm.position, None, sm.variables,
        sm.aux_variables, sm.parameters, sm.normal)
    b, h, q0, q1, q2 = sm.state
    # mirror state u(ζ) → −u(ζ): every moment flips, b and h extrapolate
    assert list(ghost) == [b, h, -q0, -q1, -q2]


def test_sme_dambreak_between_walls_conserves_mass():
    """Dam break in a closed box: both walls are the MODEL-derived wall BC.
    The mirrored ghost gives an exactly zero mass flux at each wall, so total
    mass is conserved to machine precision while the wave keeps reflecting."""
    n_cells, hL, hR = 50, 2.0, 1.0
    sm = SME(material=newtonian_navier_slip(), level=2, boundary_conditions=BoundaryConditions(
        [FromModel(tag="left", definition="wall"),
         FromModel(tag="right", definition="wall")])).system_model
    n_state = len(sm.state)
    high = np.zeros(n_state); high[1] = hL
    low = np.zeros(n_state);  low[1] = hR
    sm.initial_conditions = RP(high=lambda n, hi=high: hi,
                               low=lambda n, lo=low: lo, jump_position_x=1.0)
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 2.0), n_inner_cells=n_cells)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=1.0,
                              compute_dt=timestepping.adaptive(CFL=0.9))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    h = np.asarray(Q[1, :n_cells], dtype=float)
    assert np.all(np.isfinite(h)) and h.min() > 0.0
    mass0 = (hL + hR) / 2 * 2.0                 # ∫h dx at t=0
    mass1 = h.sum() * (2.0 / n_cells)
    assert abs(mass1 - mass0) < 1e-10 * mass0, "wall leaks mass"
    # the box is closed and t_end ≫ first wall hit: the state must NOT be the
    # plain dam-break similarity solution anymore (reflections happened)
    assert h.max() < hL - 0.05 and h.min() > hR + 0.05


def test_friction_decays_uniform_flow():
    """Coupling acceptance (chat_model_coupling.md 21:50): slip friction must
    DAMP.  Uniform flow h=0.2, q_0=0.1, λ_s=0.5 ⇒ ḣ=0 and
    q̇_0 = −λ q_0/(ρh), so q_0(t=1) = 0.1·e^{−2.5} ≈ 0.00821.  The wrong
    (pre-fix) sign grows this to ≈1.12 — far outside any tolerance."""
    sm = SME(material=newtonian_navier_slip(), level=0, parameters={"lambda_s": 0.5},
             boundary_conditions=BoundaryConditions(
                 [Extrapolation(tag="left"), Extrapolation(tag="right")])
             ).system_model
    ic = np.array([0.0, 0.2, 0.1])
    sm.initial_conditions = Constant(constants=lambda n, v=ic: v)
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=1.0,
                              compute_dt=timestepping.adaptive(CFL=0.9))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    q0 = np.asarray(Q[2, :20], dtype=float)
    assert q0.std() < 1e-12, "uniform flow stopped being uniform"
    assert abs(q0.mean() - 0.1 * np.exp(-2.5)) < 1e-3


def test_friction_excites_q1_boundedly():
    """Level-1 dam break with bottom friction must EXCITE shear (q_1 ≠ 0 —
    the physics the inter-level coupling wants to see) and stay bounded."""
    n_cells = 100
    sm = SME(material=newtonian_navier_slip(), level=1, parameters={"lambda_s": 0.5, "nu": 1e-3},
             boundary_conditions=BoundaryConditions(
                 [Extrapolation(tag="left"), Extrapolation(tag="right")])
             ).system_model
    n_state = len(sm.state)
    high = np.zeros(n_state); high[1] = 2.0
    low = np.zeros(n_state);  low[1] = 1.0
    sm.initial_conditions = RP(high=lambda n, hi=high: hi,
                               low=lambda n, lo=low: lo, jump_position_x=5.0)
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=n_cells)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=0.3,
                              compute_dt=timestepping.adaptive(CFL=0.9))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    h = np.asarray(Q[1, :n_cells], dtype=float)
    q1 = np.asarray(Q[3, :n_cells], dtype=float)
    assert np.all(np.isfinite(h)) and h.min() > 0.0
    assert np.abs(q1).max() > 1e-6, "friction failed to excite shear (q_1)"
    assert np.abs(q1).max() < 1.0, "q_1 unbounded — friction sign regressed?"


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
