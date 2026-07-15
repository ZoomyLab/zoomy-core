"""REQ-174 — the Chorin elliptic stage must CONSUME its declared BCs.

Before this fix ``_step_pressure`` built the pressure derivative-aux with a
hardcoded ``u_boundary_face="extrapolation"`` (homogeneous Neumann /
``zeroGradient``) and never read ``SM_press.boundary_conditions``.  A
user-declared Dirichlet P pin at a boundary was therefore SILENTLY replaced by
``∂ₙP = 0`` — a well-posed solve of the WRONG problem (@jax measured the
operator full-rank without the pin, so this is not a singularity).  Downstream
the 2-D extruded VAM exploded four steps after the dam front reached the
P-pinned outflow, while the 1-D twin (adequate Neumann) sailed through.

This file pins the fix:

* (a) a declared Dirichlet P → the boundary cells carry the pinned value
  EXACTLY after the solve (real Dirichlet rows in the operator/residual);
* (b) NO declared P BC → the new code path is never entered
  (``solver._press_dir is None``) ⇒ the elliptic stage stays bit-identical to
  the pre-REQ-174 homogeneous-Neumann path;
* (c) the surfaced ``last_elliptic_rel_resid`` (REQ-173) is finite and small
  with the BC rows active — it is the residual of the ACTUAL solved system.
"""
import numpy as np
import pytest
import sympy as sp

from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Dirichlet, Extrapolation)
from zoomy_core.model.models import VAM
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
import zoomy_core.model.initial_conditions as IC
from zoomy_core.systemmodel.system_model import SystemModel


G, H_RES, H_DRY, Q_IN = 9.81, 0.34, 0.015, 0.11197
DOMAIN = (-1.5, 1.5)
PIN = 0.5                                   # non-zero ⇒ crisp exactness + no early-exit
BUMP = lambda x: 0.20 * np.exp(-(x ** 2) / (2 * 0.20 ** 2))


# ── shared derivations (expensive) built once per module ─────────────────────

@pytest.fixture(scope="module")
def dir_split():
    """VAM(1, dim=3) with a Dirichlet P pin (value ``PIN``) at the x-hi outflow,
    inflow discharge on the left, lateral extrapolation — the extruded escalante
    reproducer, shrunk."""
    bcs = [Dirichlet("left", on="q_x_0", value=Q_IN),
           Dirichlet("left", on="q_x_1", value=0.0),
           Dirichlet("left", on="q_y_0", value=0.0),
           Dirichlet("left", on="q_y_1", value=0.0),
           Dirichlet("left", on="r_0", value=0.0),
           Dirichlet("left", on="r_1", value=0.0),
           Dirichlet("right", on="P_0", value=PIN),
           Dirichlet("right", on="P_1", value=PIN),
           Extrapolation(tag="bottom"), Extrapolation(tag="top")]
    model = VAM(level=1, dimension=3, boundary_conditions=bcs,
                closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return model.chorin_split(sp.Symbol("dt", positive=True), system_model=sm), sm


@pytest.fixture(scope="module")
def nopbc_split():
    """VAM(1) dam break, EXTRAPOLATION-only BCs — the REQ-173 setup, no P
    Dirichlet anywhere (so the elliptic stage keeps homogeneous Neumann)."""
    model = VAM(closures=[Newtonian(), NavierSlip(), StressFree()], level=1)
    sm = SystemModel.from_model(model)

    def _bump_ic(xv):
        xx = float(xv[0])
        bv = 0.20 * np.exp(-(xx ** 2) / (2 * 0.20 ** 2))
        hv = max((0.34 - bv) if xx < 0.0 else 0.015, 1e-6)
        out = np.zeros(len(sm.state))
        out[0], out[1] = bv, hv
        return out

    sm.initial_conditions = IC.UserFunction(function=_bump_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    bcs = BoundaryConditions([Extrapolation(tag="left"),
                              Extrapolation(tag="right")])
    sm.attach_boundary_conditions(bcs)
    split = model.chorin_split(system_model=sm)
    split.SM_pred.attach_boundary_conditions(bcs)
    return split, sm


def _setup_dir_solver(dir_split, solver_kind, nx=30, ny=4):
    split, sm = dir_split
    solver = ChorinSplitVAMSolver(
        split.SM_pred, split.SM_press, split.SM_corr,
        pressure_tol=1e-9, pressure_maxit=200, pressure_solver=solver_kind)
    mesh = BaseMesh.create_2d((DOMAIN[0], DOMAIN[1], 0.0, 0.4), nx, ny)
    Q = np.asarray(solver.setup_simulation(mesh))
    idx = {str(s): k for k, s in enumerate(sm.state)}
    x = np.asarray(solver._sim_mesh.cell_centers)[0, :solver.nc]
    b = BUMP(x)
    Q[idx["b"], :len(x)] = b
    Q[idx["h"], :len(x)] = np.maximum(np.where(x < 1.0, H_RES - b, H_DRY), H_DRY)
    solver.update_aux_variables()
    return solver, x


# ── (a) declared Dirichlet-P → boundary cells carry the pinned value ─────────

def test_declared_dirichlet_pins_boundary_cells_exactly(dir_split):
    solver, x = _setup_dir_solver(dir_split, "lu")

    pd = solver._press_dir
    assert pd is not None, "a declared Dirichlet P must populate _press_dir"

    # The pinned mode-0 cells are exactly the x-hi (outflow) column.
    pinned = np.nonzero(pd["cell_mask"][0])[0]
    assert pinned.size > 0
    assert np.allclose(x[pinned], x.max()), "pin must land on the outflow column"

    dx = (DOMAIN[1] - DOMAIN[0]) / 30
    dt = 0.3 * dx / np.sqrt(G * H_RES)
    solver.step(dt)                          # predictor → pressure → corrector

    Qn = np.asarray(solver._sim_Q, dtype=float)
    for k, s in enumerate(solver._press_state_idx):
        cells = np.nonzero(pd["cell_mask"][k])[0]
        # LU is a direct solve of the operator with real Dirichlet rows ⇒ the
        # pinned cells carry PIN to machine precision.  The corrector touches
        # only the velocity modes, so P stays pinned across the full step.
        assert np.allclose(Qn[s, cells], PIN, atol=1e-9), (
            f"mode {k}: pinned cells = {Qn[s, cells]}, expected {PIN}")


# ── (c) rel_resid finite/small against the BC-carrying operator ──────────────

def test_rel_resid_finite_small_with_bcs(dir_split):
    solver, _ = _setup_dir_solver(dir_split, "lu")
    dx = (DOMAIN[1] - DOMAIN[0]) / 30
    dt = 0.3 * dx / np.sqrt(G * H_RES)
    solver.step(dt)

    r = solver.last_elliptic_rel_resid
    assert r is not None and np.isfinite(r)
    # Residual of the ACTUAL solved system (PDE rows + Dirichlet rows); LU ⇒
    # machine precision.
    assert 0.0 <= r < 1e-6


# ── (b) no declared P BC → default path (bit-identical to pre-REQ-174) ───────

def test_no_pressure_bc_keeps_default_neumann_path(nopbc_split):
    split, _ = nopbc_split
    solver = ChorinSplitVAMSolver(
        split.SM_pred, split.SM_press, split.SM_corr, pressure_solver="lu")
    mesh = BaseMesh.create_1d(domain=DOMAIN, n_inner_cells=40)
    solver.setup_simulation(mesh)

    # No pressure mode carries a Dirichlet ⇒ _press_dir is None ⇒ neither the
    # BC-aware gradient branch nor the Dirichlet-row branch is reachable, so the
    # elliptic stage executes the identical homogeneous-Neumann statements as
    # before the fix (bit-identical by construction).
    assert solver._press_dir is None

    # And it still marches (the default Chorin pressure BC is homogeneous
    # Neumann, which is adequate here — the REQ-173 dam break).
    for _ in range(8):
        solver.step(2e-4)
    assert np.isfinite(np.asarray(solver._sim_Q, dtype=float)).all()


def test_extrapolation_only_builds_no_dirichlet_descriptor(nopbc_split):
    """`_build_pressure_dirichlet` returns None when every pressure-mode BC is
    Extrapolation — the guard that keeps the default path bit-identical."""
    split, _ = nopbc_split
    solver = ChorinSplitVAMSolver(
        split.SM_pred, split.SM_press, split.SM_corr, pressure_solver="lu")
    mesh = BaseMesh.create_1d(domain=DOMAIN, n_inner_cells=20)
    solver.setup_simulation(mesh)
    assert solver._build_pressure_dirichlet(solver._sim_mesh) is None
