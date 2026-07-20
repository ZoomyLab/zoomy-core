"""Gate-tier tests for the solver Procedure/Statement IR and its two walkers.

The IR (``zoomy_core.solver.ir``) is the ONE core extension the
solver-unification design needs; these tests are its contract:

* **Q01 golden** — the C-family walker's emitted text over the shared example
  procedures (``goldenlib.solver_procedure_examples``).
* **execution** — the SAME procedures built through the python
  ``ProcedureBuilder`` must actually compute the right numbers (numpy), and
  the march must lower to ``lax.while_loop`` on jax.
* **guards** — the ``solver_`` prefix, the collision assert against the
  ``c_functions`` / ``_NON_RESOLVABLE`` / operator-kernel namespaces, the
  unknown-slot raise, and the refusal to lower an unresolved ``IfStatic``.

Wall time: this whole file runs in well under a second (no derivation, no
march) — it is a small/gate member by construction.
"""
import time as _time

import pytest
import sympy as sp

import goldenlib

from zoomy_core.solver import (
    Assign,
    Call,
    ForEachFace,
    IfStatic,
    Procedure,
    ProcedureNameError,
    While,
    REQUIRED_PROCEDURES,
    assert_procedure_bodies,
    required_procedure_gaps,
)
from zoomy_core.solver.external import ExternalProcedure, MissingProcedureBody
from zoomy_core.transformation.procedure_c import CProcedurePrinter
from zoomy_core.transformation.procedure_python import ProcedureBuilder

pytestmark = [pytest.mark.gate, pytest.mark.small, pytest.mark.printer]


# ── (1) the golden: C-family walker ────────────────────────────────────────

@pytest.mark.parametrize("name", goldenlib.golden_params("procedure"))
def test_procedure_golden(name):
    builder, _family, _tier = goldenlib.GOLDENS[name]
    goldenlib.assert_matches_golden(name, builder())


def test_ifstatic_is_resolved_away_not_emitted():
    """A build-time branch must never reach the emitted march.  The two
    resolutions differ ONLY in the dt seed, and neither contains an ``if`` on
    the flag."""
    reduce_dt = goldenlib.solver_procedure_examples()["solver_reduce_dt"]
    printer = CProcedurePrinter()
    free = printer.print_procedure(reduce_dt.resolve({"coupled": False}))
    coupled = printer.print_procedure(reduce_dt.resolve({"coupled": True}))
    assert "double dt = dt_max;" in free
    assert "double dt = dt_window;" in coupled

    def code_only(text):
        return [ln for ln in text.splitlines()
                if not ln.lstrip().startswith("//")]

    for text in (free, coupled):
        assert not any("coupled" in ln for ln in code_only(text)), (
            "the build flag survived into emitted code")
    with pytest.raises(ValueError, match="IfStatic"):
        printer.print_procedure(reduce_dt)
    with pytest.raises(KeyError, match="build flag"):
        reduce_dt.resolve({})


def test_written_arguments_are_const_correct():
    """A face array is ``T*`` in the block that WRITES it and ``const T*`` in
    the block that only reads it — from one declaration table."""
    ex = goldenlib.solver_procedure_examples()
    printer = CProcedurePrinter()
    dt_pass = printer.print_procedure(ex["solver_dt_pass"])
    reduce_dt = printer.print_procedure(
        ex["solver_reduce_dt"].resolve({"coupled": False}))
    assert "double* lam_lo_f" in dt_pass and "const double* lam_lo_f" not in dt_pass
    assert "const double* lam_lo_f" in reduce_dt
    assert "const double* Q" in dt_pass          # read-only state
    assert "const int* face_cells" in dt_pass    # read-only connectivity
    assert "const int n_faces" in dt_pass        # by-value scalar


def test_opaque_kernel_is_emitted_as_a_call():
    """``eigenvalues`` is in ``_NON_RESOLVABLE``: the printer emits the call
    and the backend supplies the body — the shield still holds inside a
    statement-position procedure."""
    ex = goldenlib.solver_procedure_examples()
    code = CProcedurePrinter().print_procedure(ex["solver_dt_pass"])
    assert "eigenvalues(0, Q, Qaux, p, face_normal, c_own)" in code
    assert "eigenvalues(1, Q, Qaux, p, face_normal, c_own)" in code


# ── (2) execution: the python walker ───────────────────────────────────────

def test_reduce_dt_executes_numpy():
    import numpy as np

    ex = goldenlib.solver_procedure_examples()
    fn = ProcedureBuilder("numpy").build(
        ex["solver_reduce_dt"].resolve({"coupled": False}))
    env = fn(lam_lo_f=np.array([1.0, 2.0, 0.0]),
             lam_hi_f=np.array([1.5, 0.5, 0.0]),
             inradius_f=np.array([0.1, 0.2, 0.3]),
             n_faces=3, dt_max=1.0, dt_window=5.0)
    # face 0: 0.9*0.1/1.5, face 1: 0.9*0.2/2.0, face 2 is DRY (lam == 0) and
    # drops out of the min rather than producing dt = 0.
    expected = min(1.0, 0.9 * 0.1 / 1.5, 0.9 * 0.2 / 2.0)
    assert env["dt"] == pytest.approx(expected, rel=1e-14)


def test_reduce_dt_all_dry_gives_dt_max():
    """March-honesty guard (v6 §1): an all-dry domain must yield dt = dt_max,
    never dt = 0 — a zero-progress march that exits 0 is the exact failure
    mode this design forbids."""
    import numpy as np

    ex = goldenlib.solver_procedure_examples()
    fn = ProcedureBuilder("numpy").build(
        ex["solver_reduce_dt"].resolve({"coupled": False}))
    env = fn(lam_lo_f=np.zeros(4), lam_hi_f=np.zeros(4),
             inradius_f=np.full(4, 0.25), n_faces=4,
             dt_max=0.5, dt_window=9.0)
    assert env["dt"] == pytest.approx(0.5)
    assert env["dt"] > 0.0


def test_coupled_resolution_uses_dt_window_only():
    """D7 exclusivity through the python walker: in coupled mode the seed is
    dt_window and dt_max never enters as a second clamp."""
    import numpy as np

    ex = goldenlib.solver_procedure_examples()
    fn = ProcedureBuilder("numpy").build(
        ex["solver_reduce_dt"].resolve({"coupled": True}))
    env = fn(lam_lo_f=np.zeros(2), lam_hi_f=np.zeros(2),
             inradius_f=np.full(2, 1.0), n_faces=2,
             dt_max=100.0, dt_window=0.02)
    assert env["dt"] == pytest.approx(0.02)


def test_march_executes_and_calls_the_external_body():
    ex = goldenlib.solver_procedure_examples()
    written = []

    def write_fields(Q, Qaux, t, i):
        written.append((float(t), int(i)))

    fn = ProcedureBuilder(
        "numpy", externals={"solver_write_fields": write_fields}
    ).build(ex["solver_march"])
    env = fn(Q=None, Qaux=None, t_end=1.0, dt=0.25, write_interval=0.5)
    assert env["time"] == pytest.approx(1.0)
    assert env["iteration"] == 4
    # write_interval 0.5 over t_end 1.0 at dt 0.25 -> snapshots at t=0.5, 1.0
    assert env["i_snapshot"] == 2
    assert written == [(1.0, 2)], "the post-loop final write runs exactly once"


def test_missing_external_body_raises_at_build_time():
    """Missing backend body = a RED TEST at build, never a NameError
    mid-march."""
    ex = goldenlib.solver_procedure_examples()
    with pytest.raises(KeyError, match="solver_write_fields"):
        ProcedureBuilder("numpy").build(ex["solver_march"])


def test_indexed_write_numpy_and_jax_agree():
    """``ForEachFace`` + an indexed lvalue: numpy mutates in place, jax goes
    through ``.at[].set`` inside ``lax.fori_loop`` — same numbers."""
    import numpy as np

    f = sp.Symbol("f")
    lo, hi = sp.IndexedBase("lam_lo_f"), sp.IndexedBase("lam_hi_f")
    proc = Procedure(
        name="solver_face_bound_demo",
        args=("lam_lo_f", "lam_hi_f", "n_faces"),
        stmts=(ForEachFace("f", "n_faces", body=(
            Assign("lam_hi_f[f]", sp.Max(lo[f], hi[f]), declare=False),
        )),),
    )
    lo_v = np.array([1.0, 5.0, -2.0])
    hi_v = np.array([2.0, 3.0, -7.0])
    out_np = ProcedureBuilder("numpy").build(proc)(
        lam_lo_f=lo_v.copy(), lam_hi_f=hi_v.copy(), n_faces=3)["lam_hi_f"]

    jnp = pytest.importorskip("jax.numpy")
    out_jax = ProcedureBuilder("jax").build(proc)(
        lam_lo_f=jnp.asarray(lo_v), lam_hi_f=jnp.asarray(hi_v),
        n_faces=3)["lam_hi_f"]
    assert np.allclose(out_np, [2.0, 5.0, -2.0])
    assert np.allclose(np.asarray(out_jax), out_np)


def test_while_lowers_to_lax_while_loop_and_is_jittable():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    t, t_end, dt, it = sp.symbols("time t_end dt iteration")
    proc = Procedure(
        name="solver_march_pure",
        args=("t_end", "dt"),
        stmts=(
            Assign("time", sp.Integer(0)),
            Assign("iteration", sp.Integer(0), ctype="int"),
            While(t < t_end, carry=("time", "iteration"), body=(
                Assign("time", t + dt, declare=False),
                Assign("iteration", it + 1, declare=False),
            )),
        ),
    )
    fn = ProcedureBuilder("jax").build(proc)
    jitted = jax.jit(lambda te, d: fn(t_end=te, dt=d)["iteration"])
    assert int(jitted(jnp.float32(1.0), jnp.float32(0.25))) == 4
    # the loop really is a traced lax loop, not python unrolling
    txt = jax.make_jaxpr(lambda te, d: fn(t_end=te, dt=d)["time"])(
        jnp.float32(1.0), jnp.float32(0.25)).pretty_print()
    assert "while" in txt


# ── (3) guards ─────────────────────────────────────────────────────────────

def test_solver_prefix_is_mandatory():
    with pytest.raises(ProcedureNameError, match="must start with 'solver_'"):
        Procedure(name="reduce_dt")


@pytest.mark.parametrize("name", ["solver_", "Min", "flux"])
def test_names_that_would_collide_are_rejected(name):
    with pytest.raises(ProcedureNameError):
        Procedure(name=name if name.startswith("solver_") else name)


def test_kernel_and_printmap_names_cannot_be_shadowed():
    """The registration assert covers ``c_functions`` U ``_NON_RESOLVABLE`` U
    the operator slots — a solver block can never shadow a model kernel."""
    from zoomy_core.solver.ir import _reserved_names
    reserved = _reserved_names()
    assert {"Min", "Max", "conditional", "eigenvalues", "solve", "flux",
            "source", "update_aux_variables"} <= reserved
    assert not any(n.startswith("solver_") for n in reserved)


def test_unknown_argument_slot_raises():
    """No silent default for a state row / march object: an unmapped slot
    RAISES rather than being positionally guessed."""
    with pytest.raises(ProcedureNameError, match="unknown argument slot"):
        Procedure(name="solver_bogus", args=("not_a_slot",))
    with pytest.raises(ProcedureNameError, match="repeats argument"):
        Procedure(name="solver_dupe", args=("dt_max", "dt_max"))


def test_non_scalar_return_is_rejected():
    with pytest.raises(ProcedureNameError, match="not a declared SCALAR"):
        Procedure(name="solver_bad_return", args=("lam_f",), returns="lam_f")


# ── (4) the REQUIRED_PROCEDURES contract ───────────────────────────────────

def test_required_procedures_are_well_formed():
    assert REQUIRED_PROCEDURES, "the opaque-block contract must not be empty"
    for name, decl in REQUIRED_PROCEDURES.items():
        assert isinstance(decl, ExternalProcedure)
        assert decl.name == name and name.startswith("solver_")
        assert decl.doc.strip(), f"{name} declares no contract text"


def test_missing_backend_body_is_a_red_test_not_a_link_error():
    complete = {n: (lambda *a: None) for n in REQUIRED_PROCEDURES}
    assert required_procedure_gaps(complete) == ()
    assert_procedure_bodies(complete, "fake_backend")

    incomplete = dict(complete)
    incomplete.pop("solver_mood_resolve")
    assert required_procedure_gaps(incomplete) == ("solver_mood_resolve",)
    with pytest.raises(MissingProcedureBody, match="solver_mood_resolve"):
        assert_procedure_bodies(incomplete, "fake_backend")


def test_ir_walkers_are_fast():
    """Wall time recorded per the test law: building + printing the whole
    example set is milliseconds, so this file can never creep into the
    time-march tier."""
    t0 = _time.perf_counter()
    ex = goldenlib.solver_procedure_examples()
    CProcedurePrinter().print_procedure(
        ex["solver_reduce_dt"].resolve({"coupled": False}))
    ProcedureBuilder("numpy").build(
        ex["solver_reduce_dt"].resolve({"coupled": False}))
    elapsed = _time.perf_counter() - t0
    assert elapsed < 5.0, f"IR walkers took {elapsed:.2f}s — expected << 1s"
