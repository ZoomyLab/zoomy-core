"""REQ-173 — canonical stage-list construction of ``ChorinSplitVAMSolver``.

The split is *data*: ``model.chorin_split(...)`` returns three SystemModels
and ``.stages`` exposes them as a canonical list of
``Stage(label, kind, sm)`` — ``predictor``/``hyperbolic``,
``pressure``/``elliptic``, ``corrector``/``pointwise``.  A solver marches
over that list, binding each SystemModel to the executor for its ``kind``.

This file pins three things (additive, non-breaking change):

* (a) constructing the solver from ``stages=split.stages`` reproduces the
  legacy positional ``(SM_pred, SM_press, SM_corr)`` construction
  BIT-FOR-BIT on the escalante-style VAM(1) dam break — same objects, same
  order, so anything but bit-identical is a refactor bug;
* (b) ``SplitForPressureResult.stages`` has the documented shape / labels /
  kinds and points at the same three SystemModel objects;
* (c) the elliptic stage surfaces ``last_elliptic_rel_resid =
  ‖b − A x‖/‖b‖`` and it is finite and small (jax's binding contract:
  an unreporting elliptic solve is indistinguishable between solved and
  gave-up).
"""
import numpy as np
import pytest

from zoomy_core.model.models import VAM
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.model.splitter import Stage
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver
from zoomy_core.systemmodel.system_model import SystemModel


NC = 40
N_STEPS = 8
DT = 2e-4
DOMAIN = (-1.5, 1.5)


def _build_split():
    """VAM(1) dam break over a Gaussian bump — the escalante-style
    non-hydrostatic reference the dambreak pipeline test uses."""
    model = VAM(closures=[Newtonian(), NavierSlip(), StressFree()], level=1)
    sm = SystemModel.from_model(model)

    def _bump_ic(xv):
        xx = float(xv[0])
        bv = 0.20 * np.exp(-(xx**2) / (2 * 0.20**2))
        hv = max((0.34 - bv) if xx < 0.0 else 0.015, 1e-6)
        out = np.zeros(len(sm.state))
        out[0] = bv
        out[1] = hv
        return out

    sm.initial_conditions = IC.UserFunction(function=_bump_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    bcs = BoundaryConditions([Extrapolation(tag="left"),
                              Extrapolation(tag="right")])
    sm.attach_boundary_conditions(bcs)

    split = model.chorin_split(system_model=sm)
    # The predictor calls its own BC kernel → attach against its own aux.
    split.SM_pred.attach_boundary_conditions(bcs)
    return split


def _run(solver):
    mesh = BaseMesh.create_1d(domain=DOMAIN, n_inner_cells=NC)
    solver.setup_simulation(mesh)
    for _ in range(N_STEPS):
        solver.step(DT)
    return solver


# ── (b) stages property shape / label / kind ────────────────────────────────

def test_stages_property_shape_labels_kinds():
    split = _build_split()
    stages = split.stages

    assert [s.label for s in stages] == ["predictor", "pressure", "corrector"]
    assert [s.kind for s in stages] == ["hyperbolic", "elliptic", "pointwise"]

    # Same three objects as the primary attributes — ``.stages`` is a view.
    assert stages[0].sm is split.SM_pred
    assert stages[1].sm is split.SM_press
    assert stages[2].sm is split.SM_corr

    # Stage is a NamedTuple: positional AND named access agree.
    label, kind, sm = stages[1]
    assert (label, kind, sm) == (stages[1].label, stages[1].kind, stages[1].sm)
    assert isinstance(stages[1], Stage)


# ── (a) stages-vs-positional bit-for-bit equivalence ────────────────────────

def test_stages_construction_matches_positional_bitforbit():
    split = _build_split()  # ONE split — both solvers use the SAME SM objects

    solver_pos = ChorinSplitVAMSolver(
        split.SM_pred, split.SM_press, split.SM_corr,
        pressure_solver="lu", riemann_solver="hr")
    solver_stg = ChorinSplitVAMSolver(
        stages=split.stages,
        pressure_solver="lu", riemann_solver="hr")

    _run(solver_pos)
    _run(solver_stg)

    assert np.array_equal(solver_pos._sim_Q, solver_stg._sim_Q)
    assert np.array_equal(solver_pos._sim_Qaux, solver_stg._sim_Qaux)


# ── (c) elliptic residual contract ──────────────────────────────────────────

def test_elliptic_rel_resid_is_finite_and_small():
    split = _build_split()
    solver = ChorinSplitVAMSolver(
        stages=split.stages, pressure_solver="lu", riemann_solver="hr")
    _run(solver)

    r = solver.last_elliptic_rel_resid
    assert r is not None
    assert np.isfinite(r)
    # LU is a direct solve of the elliptic block ⇒ machine-precision residual.
    assert 0.0 <= r < 1e-6


# ── constructor guardrails ──────────────────────────────────────────────────

def test_stages_and_positional_are_mutually_exclusive():
    split = _build_split()
    with pytest.raises(TypeError):
        ChorinSplitVAMSolver(split.SM_pred, split.SM_press, split.SM_corr,
                             stages=split.stages)


def test_stages_bind_by_kind_not_position():
    """Binding is by ``kind``; reordering the list must not change it."""
    split = _build_split()
    shuffled = [split.stages[2], split.stages[0], split.stages[1]]
    solver = ChorinSplitVAMSolver(stages=shuffled, pressure_solver="lu")
    assert solver.sm_pred is split.SM_pred    # hyperbolic
    assert solver.sm_press is split.SM_press  # elliptic
    assert solver.sm_corr is split.SM_corr    # pointwise


def test_unknown_and_missing_kinds_raise():
    split = _build_split()
    with pytest.raises(ValueError):
        ChorinSplitVAMSolver(
            stages=[Stage("predictor", "hyperbolic", split.SM_pred),
                    Stage("pressure", "elliptic", split.SM_press),
                    Stage("weird", "spectral", split.SM_corr)])
    with pytest.raises(ValueError):
        ChorinSplitVAMSolver(
            stages=[Stage("predictor", "hyperbolic", split.SM_pred),
                    Stage("pressure", "elliptic", split.SM_press)])
