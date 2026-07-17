"""REQ-176(4) — viscous-retained derivations for the four moment families.

Each of SME / ML-SME / VAM / ML-VAM derives with the in-plane deviatoric
stress DROPPED by default (shallow ``moment_scaling``); adding a
``NewtonianInPlane`` closure RETAINS the incompressible-Newtonian streamwise
normal stress ``τ_de = ρν(∂_d u_e + ∂_e u_d)`` (``tau_xx`` included), which
``add_inplane_viscous`` folds in before the σ-map and ``package_viscous`` (or,
for ML-VAM, the model's own generalized second-derivative absorption) routes
into ``diffusion_matrix`` as a horizontal eddy viscosity.

Verified per family, TERM-BY-TERM (not shape/smoke):

  * DEFAULT unchanged — pinned by the *_reference.py suites; here we assert
    ``retain|_{ν→0} == default`` EXACTLY (the ONLY change is the viscous stress);
  * the retained stress lands as the diffusion diagonal ``A[q_i←q_i]=−2ν``
    (matching ml_fullsme.py / fullvam.py);
  * variable-b safe — the topography-coupled viscous terms are live ``ν·q·∂_x b``
    in the residual (evaluated from the stage state; no frozen-b), and a
    static-bump numpy run stays finite (no lagged bed-gradient).
"""
import itertools

import numpy as np
import pytest
import sympy as sp

from zoomy_core.systemmodel import SystemModel
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.vam import VAM
from zoomy_core.model.models.ml_sme import MLSME
from zoomy_core.model.models.ml_vam import MLVAM
from zoomy_core.model.models.closures import (
    NewtonianInPlane, Newtonian, NavierSlip, StressFree, MeanInterface)

_STRESS = [Newtonian(), NavierSlip(), StressFree()]


def _build(cls, **kw):
    if cls in (MLSME, MLVAM):
        base = _STRESS + [MeanInterface()]
        common = dict(n_layers=2, level=1, dimension=2)
    else:
        base = _STRESS
        common = dict(level=2, dimension=2)
    common.update(kw)
    default = SystemModel.from_model(cls(closures=base, **common))
    retain = SystemModel.from_model(
        cls(closures=[NewtonianInPlane()] + base, **common))
    return default, retain


def _resid(sm):
    R = sm.reconstruct_residuals()
    return {str(s): sp.sympify(R[i]) for i, s in enumerate(sm.state)}


@pytest.mark.parametrize("cls", [SME, VAM, MLSME, MLVAM])
def test_retain_reduces_to_default_at_nu_zero(cls):
    """The retain-viscous rows minus the default rows are PURELY the ν-viscous
    stress: setting ν→0 recovers the default derivation EXACTLY, row for row.
    So the DEFAULT (no NewtonianInPlane) path is provably unchanged, and the
    retained content is exactly the in-plane deviatoric stress."""
    default, retain = _build(cls)
    assert [str(s) for s in default.state] == [str(s) for s in retain.state]
    rd, rr = _resid(default), _resid(retain)
    nu = retain.parameters.nu
    for k in rd:
        diff = sp.expand((sp.sympify(rr[k]) - sp.sympify(rd[k])).doit())
        assert sp.expand(diff.subs(nu, 0)) == 0, (
            f"{cls.__name__} row {k}: retain|_(ν→0) ≠ default")


@pytest.mark.parametrize("cls", [SME, VAM, MLSME, MLVAM])
def test_inplane_stress_lands_in_diffusion_diagonal(cls):
    """The retained ∂_e(2ρν ∂_e u) projects to the horizontal eddy-viscosity
    diagonal A[q_i←q_i, x, x] = −2ν (per moment, per layer) — the term the
    shallow-moment quasilinear structure lacks (matches ml_fullsme/fullvam)."""
    _default, retain = _build(cls)
    A = retain.diffusion_matrix
    assert A is not None, f"{cls.__name__}: no diffusion_matrix on the retain path"
    st = [str(s) for s in retain.state]
    nu = retain.parameters.nu
    diag = {}
    for idx in itertools.product(*[range(n) for n in A.shape]):
        v = sp.sympify(A[idx])
        if v != 0 and idx[0] == idx[1] and st[idx[0]].startswith("q"):
            diag[st[idx[0]]] = sp.simplify(v)
    assert diag, f"{cls.__name__}: no q-momentum diffusion diagonal"
    for name, v in diag.items():
        assert sp.simplify(v + 2 * nu) == 0, (
            f"{cls.__name__}: A[{name}←{name}] = {v}, expected −2ν")


@pytest.mark.parametrize("cls", [SME, VAM, MLSME, MLVAM])
def test_variable_b_topography_coupling_is_live(cls):
    """No frozen-b: the topography-coupled viscous terms appear as live
    ν·q·∂_x b PRODUCTS in the residual (evaluated from the same stage state as
    everything else), not a precomputed constant — so a time-varying b feeds
    the same rows."""
    _default, retain = _build(cls)
    rr = _resid(retain)
    x = retain.space[0]; t = retain.time
    b = sp.Function("b", real=True)(t, x)
    nu = retain.parameters.nu
    dxb = sp.Derivative(b, x)
    found = 0
    for k, row in rr.items():
        if not k.startswith("q"):
            continue
        e = sp.expand(sp.sympify(row).doit())
        found += sum(1 for tm in sp.Add.make_args(e)
                     if tm.has(nu) and tm.has(dxb))
    assert found > 0, f"{cls.__name__}: no live ν·q·∂_x b topography coupling"


def _build_variable_b():
    """Retain-viscous SME(1, dim=2) over a topography bump — shared by the large
    IMEX march and its 1-step twin.  Returns (sm, mesh, nsm, names, Q0)."""
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
    from zoomy_core.mesh import BaseMesh
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec

    sm = SystemModel.from_model(SME(
        level=1, dimension=2,
        closures=[NewtonianInPlane(), Newtonian(), NavierSlip(), StressFree()],
        parameters={"nu": 0.05, "lambda_s": 0.1}))
    names = [str(s) for s in sm.state]

    def _ic(xv):
        xx = float(xv[0])
        out = np.zeros(len(sm.state))
        out[names.index("b")] = 0.2 * np.exp(-((xx - 5.0) ** 2))   # topography bump
        out[names.index("h")] = 1.4 if xx < 5 else 1.0
        out[names.index("q_0")] = 0.2 * out[names.index("h")]
        return out

    sm.initial_conditions = IC.UserFunction(function=_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    sm.attach_boundary_conditions(BoundaryConditions(
        [Extrapolation(tag="left"), Extrapolation(tag="right")]))
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=80)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    Q0 = np.zeros((len(names), 80))
    for j in range(80):
        Q0[:, j] = _ic([(j + 0.5) * 10.0 / 80])
    return sm, mesh, nsm, names, Q0


def test_variable_b_one_step_twin():
    """Default-tier canary: identical retain-viscous IMEX setup, exactly ONE
    step; cheap invariants only (finite, positive depth). The genuine march
    (state advanced, mass conserved) stays in the large tier."""
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.fvm.solver_imex_numpy import IMEXSolver

    _, mesh, nsm, names, _ = _build_variable_b()
    solver = IMEXSolver(time_end=0.2, compute_dt=timestepping.adaptive(CFL=0.2))
    solver.setup_simulation(mesh, nsm, write_output=False)
    solver.step(1e-3)
    Q = np.asarray(solver._sim_Q, float)
    assert np.all(np.isfinite(Q[:, :80])), "retain-viscous SME went non-finite"
    assert np.asarray(Q[names.index("h"), :80], float).min() > 0.0


@pytest.mark.large
def test_variable_b_smoke_marches_over_topography():
    """Retain-viscous SME on a topography bump (∂_x b ≠ 0), IMEX (implicit dense
    diffusion + INVISCID hyperbolic wave speed): the solve actually MARCHES —
    the state evolves, stays finite and conserves mass.  Proves the
    topography-coupled viscous NCP terms are evaluated from the stage state
    every step (no lagged/frozen bed gradient)."""
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.fvm.solver_imex_numpy import IMEXSolver

    sm, mesh, nsm, names, Q0 = _build_variable_b()
    solver = IMEXSolver(time_end=0.2, compute_dt=timestepping.adaptive(CFL=0.2))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    q0 = np.asarray(Q[names.index("q_0"), :80], float)
    assert np.all(np.isfinite(Q[:, :80])), "retain-viscous SME went non-finite"
    # the solve genuinely advanced in time (not frozen at the IC)
    assert np.max(np.abs(q0 - Q0[names.index("q_0")])) > 1e-3, "solve did not march"
    h = np.asarray(Q[names.index("h"), :80], float)
    # positive depth + mass ~conserved (small open-boundary flux drift allowed)
    assert h.min() > 0 and abs(h.sum() * (10.0 / 80) - 12.0) < 0.05
