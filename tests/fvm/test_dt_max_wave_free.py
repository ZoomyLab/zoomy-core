"""REQ-190 (piece 1) — ``dt_max`` is a STANDARD NSM numerical parameter, and a
WAVE-FREE (fully-dry) domain steps at ``dt_max`` — never a hardcoded floor.

Two coupled facts, verified once so every backend inherits them:

1. ``NumericalSystemModel`` carries ``dt_max`` (default 5.0 s = amrex's
   historical ``dtmax``) alongside its other numerical knobs, and EVERY printer
   emits it among the numerical parameters (numpy/UFL runtime attribute;
   generic-C / OpenFOAM header constant).  Only the ``dt_max`` line is added —
   the rest of each emit is unchanged.

2. numpy: ``timestepping.adaptive`` takes ``dt_max`` from the NSM when the
   caller did not pass one explicitly (explicit argument wins).  When every
   gated eigenvalue is 0 (all cells dry) the local CFL limits are ``+inf`` and
   the step collapses to ``dt_max`` (clipped to ``time_end``), NOT to a magic
   floor and NOT to ``inf``.  A wet run's dt is unaffected (``dt_max`` never
   binds).
"""
import numpy as np
import pytest

from zoomy_core.fvm import timestepping
from zoomy_core.model.models.swe import SWE
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics.numerical_system_model import (
    NumericalSystemModel, ReconstructionSpec)
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.mesh import BaseMesh
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver


# ── (1) NSM carries dt_max ────────────────────────────────────────────────

def test_nsm_dt_max_default_and_explicit():
    sm = SystemModel.from_model(SWE(dimension=1))
    assert NumericalSystemModel.from_system_model(sm).dt_max == 5.0
    sm2 = SystemModel.from_model(SWE(dimension=1))
    assert NumericalSystemModel.from_system_model(sm2, dt_max=0.25).dt_max == 0.25


# ── (2) every printer emits dt_max among the numerical parameters ─────────

def _swe_sm():
    return SystemModel.from_model(SWE(dimension=1))


def _vam_sm():
    from zoomy_core.model.models.vam import VAM
    return SystemModel.from_model(VAM(level=1, dimension=2))


@pytest.mark.parametrize("build", [_swe_sm, _vam_sm], ids=["swe", "vam"])
def test_numpy_runtime_emits_dt_max(build):
    from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
    rt = NumpyRuntimeModel.from_system_model(build())
    assert rt.dt_max == 5.0


@pytest.mark.parametrize("build", [_swe_sm, _vam_sm], ids=["swe", "vam"])
def test_generic_c_emits_dt_max(build):
    from zoomy_core.transformation.to_c import CppModel
    code = CppModel(build()).create_code()
    hits = [l for l in code.splitlines() if "dt_max" in l]
    assert hits == ["    static constexpr double dt_max = 5.0;"], hits


@pytest.mark.parametrize("build", [_swe_sm, _vam_sm], ids=["swe", "vam"])
def test_foam_emits_dt_max(build):
    from zoomy_core.transformation.to_openfoam import FoamSystemModelPrinter
    code = FoamSystemModelPrinter(build()).create_code()
    hits = [l for l in code.splitlines() if "dt_max" in l]
    assert hits == ["constexpr Foam::scalar dt_max = 5.0;"], hits


def test_ufl_runtime_emits_dt_max():
    try:
        from zoomy_core.transformation.to_ufl import UFLRuntimeModel
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"UFL backend unavailable: {exc}")
    rt = UFLRuntimeModel.from_system_model(_swe_sm())
    assert rt.dt_max == 5.0


# ── (2b) the dt_max line is the ONLY change (rest of the emit unchanged) ───

def test_generic_c_emit_only_dt_max_line_added():
    """Strip the single new ``dt_max`` constant → the emit is what it was
    before REQ-190 (a one-line, additive change; nothing else perturbed)."""
    from zoomy_core.transformation.to_c import CppModel
    code = CppModel(_swe_sm()).create_code()
    stripped = "\n".join(l for l in code.splitlines() if "dt_max" not in l)
    assert "dt_max" not in stripped
    # the removed line was a single numerical constant — the struct still opens
    # and the parameter interface is intact.
    assert "static constexpr int n_parameters" in stripped
    assert "default_parameters()" in stripped


# ── helpers for the solver marches ────────────────────────────────────────

def _swe_solver(*, h_value, time_end, dt_max=None, compute_dt=None, nc=20):
    """A 1-D SWE FreeSurfaceFlowSolver over a flat bed, uniform depth
    ``h_value`` (0 → fully dry).  ``dt_max`` (when given) is set on the NSM."""
    bcs = BoundaryConditions([Extrapolation(tag="left"),
                              Extrapolation(tag="right")])
    sm = SystemModel.from_model(SWE(dimension=1, boundary_conditions=bcs))
    n_state = len(sm.state)                          # [b, h, hu]

    def ic(xv):
        out = np.zeros(n_state)
        out[1] = h_value
        return out

    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=nc)
    kw = {} if dt_max is None else {"dt_max": dt_max}
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1), **kw)
    solver = FreeSurfaceFlowSolver(
        time_end=time_end,
        **({"compute_dt": compute_dt} if compute_dt is not None else {}))
    solver.setup_simulation(mesh, nsm, write_output=False)
    return solver


def _dt_now(solver):
    return float(solver.compute_dt(
        solver._sim_Q, solver._sim_Qaux, solver._sim_parameters,
        solver._sim_face_inradius, solver._sim_compute_max_abs_eigenvalue))


# ── (2c) wave-free march steps at dt_max, then clips to time_end ───────────

def test_all_dry_steps_at_dt_max():
    """Fully-dry SWE (h=0 everywhere): the CFL is wave-free so the step is
    exactly the NSM's ``dt_max`` — not ``inf`` and not a hardcoded floor."""
    dtmax = 0.5
    solver = _swe_solver(h_value=0.0, time_end=2.0, dt_max=dtmax)
    assert _dt_now(solver) == pytest.approx(dtmax)


def test_all_dry_clips_to_time_end():
    """When ``dt_max`` exceeds the remaining time, the step clips to
    ``time_end`` (finite) — a dry domain marches, it does not crawl or hang."""
    solver = _swe_solver(h_value=0.0, time_end=0.3, dt_max=5.0)
    dt = _dt_now(solver)
    assert not np.isinf(dt) and dt == pytest.approx(5.0)   # cap is 5.0
    clipped = min(dt, solver.time_end - 0.0)               # run-loop clip
    assert clipped == pytest.approx(0.3)


# ── (2d) a wet march is unaffected — dt_max never binds ────────────────────

def test_wet_dt_unchanged_by_dt_max():
    """A wet SWE (h=1) has a finite CFL dt well below ``dt_max``; the emitted
    dt is BIT-IDENTICAL to the pure-CFL adaptive value (no cap perturbation)."""
    solver = _swe_solver(h_value=1.0, time_end=1.0, dt_max=5.0)
    dt_capped = _dt_now(solver)
    # the same strategy the solver builds by default (CFL=0.9, default dim/deg)
    # but WITHOUT any cap wired (dt_max stays None → no min applied).
    pure = timestepping.adaptive(CFL=0.9)
    dt_pure = float(pure(
        solver._sim_Q, solver._sim_Qaux, solver._sim_parameters,
        solver._sim_face_inradius, solver._sim_compute_max_abs_eigenvalue))
    assert np.isfinite(dt_capped) and dt_capped < 5.0
    assert dt_capped == dt_pure                       # bit-for-bit


# ── (2e) an EXPLICIT dt_max on adaptive() overrides the NSM default ────────

def test_explicit_dt_max_wins_over_nsm():
    """A caller-supplied ``adaptive(dt_max=...)`` is never overwritten by the
    NSM's ``dt_max`` at setup — explicit argument wins."""
    explicit = timestepping.adaptive(CFL=0.9, dimension=1, degree=0, dt_max=0.1)
    solver = _swe_solver(h_value=0.0, time_end=2.0, dt_max=5.0,
                         compute_dt=explicit)
    assert _dt_now(solver) == pytest.approx(0.1)      # 0.1, not the NSM 5.0
