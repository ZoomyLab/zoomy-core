"""Gate-tier tests for the EMITTED march (``zoomy_core.solver.march``).

Two contracts, and nothing else — the numbers are the twin gate's job
(``zoomy_jax``), the STRUCTURE and the CONSTANTS are core's:

* **structure** — dt at the step head and frozen across stages; the stage
  sequence unrolled from the tableau; every build-time decision resolved AWAY
  (no ``IfStatic`` survives, no flag name survives); the two honesty guards
  present; the Shu-Osher average taken against the STEP BASE while the
  residual is added to the STAGE state.
* **emitted constants (v7 / mandate 6a)** — every constant the march compares
  against is a NAMED value with a provenance string, ``eps_h`` is READ BACK
  out of the derived NSM rather than restated, and no bare float literal
  reaches a comparison site.

Wall time: one SME(level=0) derivation off the cache plus IR walks — well
inside the gate tier, no march.
"""
import sympy as sp
import pytest

from zoomy_core.solver import Assign, Call, IfStatic, Procedure, While
from zoomy_core.solver.constants import (
    ConstantResolutionError,
    MarchConstants,
    eigen_wave_speed_floor,
    march_constants,
)
from zoomy_core.solver.external import REQUIRED_PROCEDURES
from zoomy_core.solver.march import (
    MARCH_FLAGS,
    TABLEAU_EULER,
    TABLEAU_SSPRK2,
    emit_march,
    tableau_for,
)
from zoomy_core.transformation.procedure_python import ProcedureBuilder

pytestmark = [pytest.mark.gate, pytest.mark.small, pytest.mark.printer]


# ── the NSM under test: DERIVED SWE = SME(level=0), capless ────────────────

@pytest.fixture(scope="module")
def swe_nsm():
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel
    from zoomy_core.model.models import SME
    from zoomy_core.systemmodel.operations import desingularize_hinv

    def _build(order):
        sm = SystemModel.from_model(SME(level=0, dimension=2))
        return NumericalSystemModel.from_system_model(
            sm, extra_operations=[desingularize_hinv()],
            reconstruction=ReconstructionSpec(order=order)).derive()
    return _build


@pytest.fixture(scope="module")
def prog_o1(swe_nsm):
    return emit_march(swe_nsm(1), cfl=0.9, dimension=2)


@pytest.fixture(scope="module")
def prog_o2(swe_nsm):
    return emit_march(swe_nsm(2), cfl=0.9, dimension=2, mood=True)


# ── helpers ────────────────────────────────────────────────────────────────

def _walk(stmts):
    for s in stmts:
        yield s
        for child in s.children():
            yield from _walk(child)


def _calls(proc):
    return [s for s in _walk(proc.stmts) if isinstance(s, Call)]


def _assigns(proc):
    return [s for s in _walk(proc.stmts) if isinstance(s, Assign)]


# ── (1) structure ──────────────────────────────────────────────────────────

def test_dt_is_computed_at_the_step_head_and_frozen(prog_o2):
    """v6 §1: dt_pass -> reduce_dt -> guard, ONCE, before the step; and no
    stage writes dt."""
    names = [c.procedure for c in _calls(prog_o2.march)]
    assert names.index("solver_dt_pass") < names.index("solver_reduce_dt")
    assert names.index("solver_reduce_dt") < names.index(
        "solver_assert_dt_admissible")
    assert names.index("solver_assert_dt_admissible") < names.index(
        "solver_hyperbolic_step")
    assert names.count("solver_dt_pass") == 1, "dt is computed once per step"
    assert names.count("solver_reduce_dt") == 1

    # FROZEN: nothing inside the step produces or overwrites dt.
    assert "dt" not in [a.target for a in _assigns(prog_o2.step)]
    for c in _calls(prog_o2.step):
        assert "dt" not in c.results, (
            f"{c.procedure} rebinds dt inside the step — dt is frozen")


def test_no_lambda_row_comes_back_from_the_flux(prog_o2):
    """v6 deleted the face-wavespeed row from numerical_flux; the flux pass
    returns exactly the three stored face arrays."""
    flux = [c for c in _calls(prog_o2.step)
            if c.procedure == "solver_flux_pass"]
    assert flux, "the step must call solver_flux_pass"
    for c in flux:
        assert c.results == ("flux_face", "d_plus", "d_minus")


def test_stage_loop_is_unrolled_from_the_tableau(prog_o1, prog_o2):
    assert prog_o1.tableau == TABLEAU_EULER
    assert prog_o2.tableau == TABLEAU_SSPRK2
    for prog in (prog_o1, prog_o2):
        n = len(prog.tableau)
        names = [c.procedure for c in _calls(prog.step)]
        for block in ("solver_halo_bc", "solver_reconstruct",
                      "solver_flux_pass", "solver_gather_update"):
            assert names.count(block) == n, (
                f"{block} appears {names.count(block)}x for {n} stage(s)")
        # the stage weights are emitted, one alpha/beta pair per stage
        targets = [a.target for a in _assigns(prog.step)]
        for k, (alpha, beta) in enumerate(prog.tableau):
            assert f"c_alpha_{k}" in targets and f"c_beta_{k}" in targets


def test_shu_osher_averages_the_step_base_not_the_stage_state(prog_o2):
    """The residual is evaluated at and added to the STAGE state; only the
    average uses the step base ``Q0``.  Conflating the two silently turns
    SSP-RK2 into two Euler half-steps."""
    gathers = [c for c in _calls(prog_o2.step)
               if c.procedure == "solver_gather_update"]
    assert all(c.args[0] == "variables" for c in gathers), (
        "gather_update must start from the CURRENT stage state")
    combines = [a for a in _assigns(prog_o2.step) if a.target == "Q"]
    assert len(combines) == len(prog_o2.tableau)
    for k, a in enumerate(combines):
        expr = sp.sympify(a.expr)
        alpha = sp.Symbol(f"c_alpha_{k}")
        assert sp.simplify(
            expr - (alpha * sp.Symbol("Q0")
                    + (1 - alpha) * sp.Symbol("Q_cand"))) == 0
    # the step base is snapshotted exactly once, before any stage
    base = [c for c in _calls(prog_o2.step)
            if c.procedure == "solver_stage_base"]
    assert len(base) == 1 and base[0].results == ("q0",)


def test_build_time_branches_never_become_runtime_branches(prog_o1, prog_o2):
    for prog in (prog_o1, prog_o2):
        for proc in (prog.march, prog.step, prog.should_write):
            assert proc.is_resolved(), f"{proc.name} still carries an IfStatic"
            assert not any(isinstance(s, IfStatic) for s in _walk(proc.stmts))
            blob = repr(proc)
            for flag in MARCH_FLAGS:
                assert f"'{flag}'" not in blob, (
                    f"build flag {flag!r} survived into {proc.name}")


def test_mood_and_ncp_are_build_time_decisions(prog_o1, prog_o2):
    o1_names = [c.procedure for c in _calls(prog_o1.step)]
    o2_names = [c.procedure for c in _calls(prog_o2.step)]
    assert "solver_mood_resolve" not in o1_names
    assert "solver_mood_resolve" in o2_names
    # SME(level=0) carries a nonconservative matrix, so order 2 takes the
    # amendment-10 interior-NCP term and order 1 takes the identity.
    assert "solver_no_cell_term" in o1_names and "solver_cell_ncp" not in o1_names
    assert "solver_cell_ncp" in o2_names and "solver_no_cell_term" not in o2_names


def test_both_honesty_guards_are_present(prog_o2):
    names = [c.procedure for c in _calls(prog_o2.march)]
    assert "solver_assert_dt_admissible" in names, "the FATAL dt guard"
    assert "solver_assert_march_progress" in names, (
        "the second guard: a dt collapsing towards zero while staying "
        "strictly positive is invisible to the dt guard")


def test_the_march_is_the_one_while(prog_o2):
    whiles = [s for s in _walk(prog_o2.march.stmts) if isinstance(s, While)]
    assert len(whiles) == 1
    assert sp.sympify(whiles[0].condition) == sp.Lt(sp.Symbol("time"),
                                                    sp.Symbol("t_end"))
    assert "variables" in whiles[0].carry and "time" in whiles[0].carry


def test_dt_floor_is_off_by_default_and_emits_nothing(prog_o2):
    assert prog_o2.flags["dt_floor"] is False
    assert prog_o2.constants.values["c_dt_floor"] is None
    names = [c.procedure for c in _calls(prog_o2.march)]
    assert "solver_apply_dt_floor" not in names, (
        "the sanctioned march has NO dt floor — only the FATAL guard")


def test_every_called_body_is_a_declared_external(prog_o1, prog_o2):
    """Every ``Call`` target is either a REQUIRED_PROCEDURES body or one of
    the two procedures core itself emits."""
    emitted = {"solver_hyperbolic_step", "solver_should_write"}
    for prog in (prog_o1, prog_o2):
        for proc in (prog.march, prog.step, prog.should_write):
            for name in proc.calls():
                assert name in REQUIRED_PROCEDURES or name in emitted, (
                    f"{proc.name} calls undeclared block {name!r}")


def test_unknown_build_flag_raises(swe_nsm):
    with pytest.raises(KeyError, match="unknown march build flag"):
        emit_march(swe_nsm(1), cfl=0.9, dimension=2, flags={"turbo": True})


def test_unmapped_tableau_order_raises():
    with pytest.raises(KeyError, match="no stage tableau"):
        tableau_for(3)


# ── (2) the emitted-constant contract (v7 / mandate 6a) ────────────────────

#: Every constant the design requires the march to EMIT rather than bake in.
REQUIRED_CONSTANTS = (
    "c_mood_h_bound", "c_mood_require_finite", "c_kp_eps", "c_eps_h",
    "c_cfl", "c_cfl_dimension", "c_cfl_degree_factor", "c_dt_max",
    "c_write_eps", "c_dt_floor",
)


def test_all_required_constants_are_emitted_with_provenance(prog_o2):
    consts = prog_o2.constants
    for name in REQUIRED_CONSTANTS:
        assert name in consts.values, f"{name} is not emitted"
        assert consts.provenance[name].strip(), f"{name} has no provenance"


def test_mood_bound_is_strictly_zero(prog_o2):
    """The user law: ``h < 0`` is DETECTED, never repaired — so the detector
    bound is exactly zero and a backend cannot widen it."""
    assert prog_o2.constants.values["c_mood_h_bound"] == 0.0
    assert prog_o2.constants.values["c_mood_require_finite"] == 1


def test_eps_h_is_read_back_out_of_the_derived_nsm(swe_nsm):
    """``eps_h`` is not restated here: ``regularize_pow`` writes
    ``1/Max(eps_h, h)`` into the eigenvalue slot and the emit reads it back."""
    nsm = swe_nsm(1)
    value, provenance = eigen_wave_speed_floor(nsm)
    assert value is not None, (
        "the derived, desingularised SWE must carry a wave-speed floor in "
        "its eigenvalue slot")
    assert "nsm.eigenvalues" in provenance
    prog = emit_march(nsm, cfl=0.9, dimension=2)
    assert prog.constants.values["c_eps_h"] == value


def test_contradictory_eps_h_raises_rather_than_picking_one(swe_nsm):
    class _Two:
        state = list(swe_nsm(1).state)
        h = next(s for s in state if str(s) == "h")
        eigenvalues = sp.Matrix([
            1 / sp.Max(sp.Float(1e-14), h),
            1 / sp.Max(sp.Float(1e-10), h)])
    with pytest.raises(ConstantResolutionError, match="different wave-speed"):
        eigen_wave_speed_floor(_Two())


def test_cfl_is_a_pure_safety_factor(swe_nsm):
    nsm = swe_nsm(1)
    prog = emit_march(nsm, cfl=0.9, dimension=2)
    assert prog.constants.values["c_cfl"] == 0.9
    assert prog.constants.values["c_cfl_dimension"] == 2
    assert prog.constants.values["c_cfl_degree_factor"] == 1
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ConstantResolutionError, match="safety factor"):
            emit_march(nsm, cfl=bad, dimension=2)


def test_dt_max_comes_from_the_nsm(swe_nsm):
    nsm = swe_nsm(1)
    prog = emit_march(nsm, cfl=0.9, dimension=2)
    assert prog.constants.values["c_dt_max"] == float(nsm.dt_max)
    assert "nsm.dt_max" in prog.constants.provenance["c_dt_max"]


def test_no_float_literal_reaches_a_comparison_site(prog_o1, prog_o2):
    """Mandate 6a.  The ONLY ``Assign`` allowed to hold a bare number is the
    one DEFINING an emitted constant; everything else must be built from
    names.  Small integers (loop seeds, the ``1 -`` of the Shu-Osher average,
    the ``i_snapshot + 1`` of the write gate) are structure, not tuned
    numbers, so only non-integer Floats are policed.
    """
    for prog in (prog_o1, prog_o2):
        emitted = set(prog.constants.values) | {
            f"c_{w}_{k}" for k in range(len(prog.tableau))
            for w in ("alpha", "beta")}
        for proc in (prog.march, prog.step, prog.should_write):
            for a in _assigns(proc):
                if a.target in emitted:
                    continue          # this IS the definition site
                floats = [f for f in sp.sympify(a.expr).atoms(sp.Float)]
                assert not floats, (
                    f"{proc.name}: assignment to {a.target!r} carries the "
                    f"literal(s) {floats} — emit them as named constants")


def test_constants_report_names_every_value_and_source(prog_o2):
    report = prog_o2.constants.report()
    for name in REQUIRED_CONSTANTS:
        assert name in report
    assert "user law" in report        # the CFL / MOOD-bound rulings
    assert "nsm.eigenvalues" in report  # eps_h really was read back


def test_inactive_constant_cannot_be_used_at_a_comparison_site(prog_o2):
    with pytest.raises(ConstantResolutionError, match="inactive"):
        prog_o2.constants.symbol("c_dt_floor")
    assert prog_o2.constants.active("c_cfl")
    assert not prog_o2.constants.active("c_dt_floor")


def test_provenance_is_mandatory():
    with pytest.raises(ConstantResolutionError, match="provenance"):
        MarchConstants(values={"c_x": 1.0}, provenance={})


# ── (3) lowering: the emitted march must actually RUN ──────────────────────

def test_emitted_march_lowers_and_runs_on_numpy(prog_o1):
    """Structure-only execution with recording stub bodies: the point is the
    ORDER of the blocks and the arithmetic of the loop, not the physics."""
    import numpy as np

    trace = []

    def gather(Qk, F, Dp, Dm, dt, beta, cell_term, Qaux, p, t):
        trace.append(("gather", float(beta), float(dt), float(t)))
        return Qk - beta * dt * np.ones_like(Qk), np.zeros(1, bool)

    bodies = {
        "solver_stage_base": lambda Q: Q,
        "solver_clear_troubled": lambda Q: np.zeros(1, bool),
        "solver_merge_troubled": lambda a, b: a | b,
        "solver_halo_bc": lambda Q, Qaux, p, t: (Q, Qaux, np.zeros(1)),
        "solver_reconstruct": lambda Q, Qaux, bf, o1: (Q, Q, None),
        "solver_no_cell_term": lambda Q: None,
        "solver_flux_pass": lambda Q, Qaux, p, t, QL, QR: (
            np.zeros_like(Q),) * 3,
        "solver_gather_update": gather,
        "solver_update_variables": lambda Q, Qaux, p, t, dt: Q,
        "solver_update_aux": lambda Q, Qaux, p, t, dt: Qaux,
        "solver_dt_pass": lambda Q, Qaux, p, eps: (np.array([-1.0]),
                                                   np.array([1.0])),
        "solver_reduce_dt": lambda lo, hi, r, nf, cfl, d, degf, dtmax, clamp: (
            float(min(cfl * 2 * r.min()
                      / (d * degf * max(abs(lo).max(), abs(hi).max())),
                      dtmax, clamp))),
        "solver_assert_dt_admissible": lambda dt, lo, hi, r, t, it: float(dt),
        "solver_assert_march_progress": lambda t, it, dt, te: None,
        "solver_write_fields": lambda Q, Qaux, t, i, dw: trace.append(
            ("write", int(i), bool(dw))),
    }
    sw = ProcedureBuilder("numpy", externals=bodies).build(prog_o1.should_write)
    bodies["solver_should_write"] = lambda t, dt, i, wi: (
        lambda e: (e["do_write"], int(e["i_snapshot"])))(
            sw(time=t, dt=dt, i_snapshot=i, write_interval=wi))
    step = ProcedureBuilder("numpy", externals=bodies).build(prog_o1.step)
    bodies["solver_hyperbolic_step"] = lambda Q, Qaux, p, t, it, dt: (
        lambda e: (e["Q"], e["Qaux"], e["time"], e["iteration"],
                   e["troubled"]))(
            step(Q=Q, Qaux=Qaux, p=p, time=t, iteration=it, dt=dt))
    march = ProcedureBuilder("numpy", externals=bodies).build(prog_o1.march)

    env = march(Q=np.array([10.0]), Qaux=np.zeros(1), p=np.zeros(1),
                time=0.0, iteration=0, i_snapshot=0, t_end=1.0,
                dt_window=None, write_interval=0.5,
                inradius_f=np.array([0.1]), n_faces=1)

    # dt = 0.9 * 2*0.1 / (2 * 1 * 1) = 0.09; the last step is clamped to t_end
    assert env["time"] == 1.0, "the t_end clamp must land the march exactly"
    gathers = [rec for rec in trace if rec[0] == "gather"]
    dts = [rec[2] for rec in gathers]
    assert dts[0] == pytest.approx(0.09)
    assert dts[-1] == pytest.approx(1.0 - 0.09 * (len(dts) - 1))
    # Euler tableau: one gather per step at beta = 1
    assert all(rec[1] == 1.0 for rec in gathers)
    # Q decreased by exactly the elapsed time (sum of beta*dt)
    assert float(env["Q"][0]) == pytest.approx(9.0)
    # drift-free cadence at write_interval 0.5 over [0, 1]: stamps 0, 0.5, 1.0
    writes = [rec[1] for rec in trace if rec[0] == "write" and rec[2]]
    assert writes == [1, 2, 3], writes


def test_stage_time_is_t_then_t_plus_dt(prog_o2):
    """SSP-RK2 evaluates stage 2 at ``t + dt`` — the BCs of stage 2 are the
    end-of-step BCs (amendment 11)."""
    stage_times = [sp.sympify(a.expr) for a in _assigns(prog_o2.step)
                   if a.target == "time_stage"]
    t, dt = sp.Symbol("time"), sp.Symbol("dt")
    assert stage_times == [t, t + dt]


def test_march_ir_is_fast(swe_nsm):
    """Test law: record the wall time so this file cannot creep into the
    time-march tier."""
    import time as _time
    t0 = _time.perf_counter()
    emit_march(swe_nsm(2), cfl=0.9, dimension=2, mood=True)
    elapsed = _time.perf_counter() - t0
    assert elapsed < 2.0, f"emit_march took {elapsed:.3f} s"
