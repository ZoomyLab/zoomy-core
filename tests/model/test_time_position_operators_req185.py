"""REQ-185 — TIME ``t`` and POSITION ``x`` (a length-3 VECTOR) as first-class
arguments of the ``source`` and ``update_aux_variables`` operators.

Convention (mirrors the BC path's ``time`` / ``X`` args):

* ``source(Q, Qaux, p, t, dt, x)``            — t scalar, dt scalar, x ∈ R³
* ``update_aux_variables(Q, Qaux, p, t, x)``  — t scalar, x ∈ R³ (NO dt)

The position ``x`` is a single length-3 vector REGARDLESS of model dimension;
symbol binding is unconditional ``x→x[0], y→x[1], z→x[2]``.  A model that does
not reference the coordinate/time args pays nothing (bit-identical) — the
lambdify/compiler simply ignores the unused trailing args.

Acceptance:
  (a) a rain model whose ``update_aux_variables`` sets ``r_o = Piecewise((rate,
      t<T_rain),(0,True))`` and whose ``source`` adds ``r_o`` to the continuity
      equation: a numpy march plateaus in total volume after ``T_rain`` (it
      would keep rising if ``t`` were unbound / ignored);
  (b) a source referencing ``dt`` emits with ``dt`` bound and matches a hand
      evaluation;
  (c) the C-family emit for ``source`` + ``update_aux_variables`` carries the
      ``time`` / ``dt`` (source only) / ``X`` args, and the numpy runtime
      binds a length-3 position vector regardless of dimension.
"""
import numpy as np
import sympy as sp
import pytest

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.model.models.swe import SWE
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel, DT_SYMBOL


# ── a minimal "rain" model: a passive tracer c with a time-gated rain source ─
# (a plain tracer field ``c`` rather than the free-surface ``h`` so the NSM's
# KP-``hinv`` desingularisation is not auto-attached — the test isolates the
# time-dependent aux mechanism, nothing else.)
class RainTracer(StructuredDerivativeModel):
    """∂_t c + a·∂_x c = r_o,  r_o = rain rate active only while t < T_rain."""

    dimension = 1
    variables = ["c"]
    parameters = {
        "a":         (1.0, "real"),          # advection speed (finite CFL dt)
        "rain_rate": (0.5, "nonnegative"),
        "T_rain":    (0.05, "positive"),
    }

    def __init__(self, **kw):
        self._coupling_bcs = kw.pop("boundary_conditions", None)
        super().__init__(aux_variables=["r_o"], **kw)

    def _build_function_groups(self):
        return {}

    def flux(self):
        F = sp.Matrix.zeros(1, 1)
        F[0, 0] = self.parameters.a * self.Q.c
        return ZArray(F)

    def eigenvalues(self):
        return ZArray([self.parameters.a])

    def update_aux_variables(self):
        t = sp.Symbol("t", real=True)
        p = self.parameters
        return ZArray([sp.Piecewise((p.rain_rate, t < p.T_rain),
                                    (sp.S.Zero, True))])

    def source(self):
        S = sp.Matrix.zeros(1, 1)
        S[0, 0] = self.aux_variables.r_o
        return ZArray(S)


# ── a free-surface SWE-with-rain (state [b, h, hu]) for the solver march ─────
# The numpy free-surface solver keys on an ``h`` field; the NSM auto-attaches
# its KP ``hinv`` aux alongside the declared ``r_o`` (aux → [r_o, hinv]) and
# extends ``update_aux_variables`` to [r_o(t), hinv(h)] — so the rain aux binds
# ``t`` and ``hinv`` its usual desingularised depth.
class RainSWE(SWE):
    def __init__(self, **kw):
        params = {"rain_rate": (0.5, "nonnegative"), "T_rain": (0.05, "positive")}
        params.update(kw.pop("parameters", None) or {})
        super().__init__(dimension=1, aux_variables=["r_o"],
                         parameters=params, **kw)

    def update_aux_variables(self):
        t = sp.Symbol("t", real=True)
        p = self.parameters
        return ZArray([sp.Piecewise((p.rain_rate, t < p.T_rain),
                                    (sp.S.Zero, True))])

    def source(self):
        S = sp.Matrix(SWE.source(self))     # Manning friction (0 for n_m=0)
        S[1, 0] = S[1, 0] + self.aux_variables.r_o   # rain into continuity
        return ZArray(S)


# ── (a) rain: volume rises before T_rain, FLAT after (solver march) ──────────

@pytest.mark.large
def test_rain_volume_plateaus_after_T_rain():
    """Full numpy-solver march: with a periodic domain and a uniform tracer,
    ``∂_t c = r_o`` everywhere; total volume must rise while t < T_rain and be
    FLAT afterwards.  If the solver dropped ``t`` (rain never turns off) the
    volume would keep climbing — this test discriminates."""
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Periodic
    from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
    from zoomy_core.fvm import timestepping
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.mesh import BaseMesh
    import zoomy_core.model.initial_conditions as IC

    NC, XMAX = 40, 1.0
    T_RAIN, RATE = 0.05, 0.5
    bcs = BoundaryConditions([
        Periodic(tag="left", periodic_to_physical_tag="right"),
        Periodic(tag="right", periodic_to_physical_tag="left"),
    ])
    sm = SystemModel.from_model(
        RainSWE(parameters={"rain_rate": (RATE, "nonnegative"),
                            "T_rain": (T_RAIN, "positive")},
                boundary_conditions=bcs))
    n_state = len(sm.state)          # [b, h, hu]

    def ic(xv):
        out = np.zeros(n_state)
        out[1] = 1.0                 # uniform wet depth h=1 (lake at rest)
        return out

    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, XMAX), n_inner_cells=NC)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))

    solver = FreeSurfaceFlowSolver(
        time_end=0.12, compute_dt=timestepping.adaptive(CFL=0.4))
    solver.setup_simulation(mesh, nsm, write_output=False)

    nc = solver._sim_mesh.n_inner_cells
    vol = solver._sim_mesh.cell_volumes[:nc]
    c_idx = int(solver._free_surface_h_index)

    def total_volume():
        return float((solver._sim_Q[c_idx, :nc] * vol).sum())

    v0 = total_volume()
    times, vols = [0.0], [v0]
    t = 0.0
    while t < solver.time_end:
        dt = solver.compute_dt(
            solver._sim_Q, solver._sim_Qaux, solver._sim_parameters,
            solver._sim_face_inradius,
            solver._sim_compute_max_abs_eigenvalue)
        dt = min(float(dt), float(solver.time_end - t))
        if not np.isfinite(dt) or dt <= 0.0:
            break
        solver._sim_time = t
        solver.step(dt)
        t += dt
        times.append(t)
        vols.append(total_volume())

    times, vols = np.array(times), np.array(vols)
    # BEFORE T_rain: volume strictly rises (rain is on).
    pre = vols[times <= T_RAIN]
    assert pre[-1] > pre[0] + 1e-6, (
        f"volume did not rise during rain: {pre[0]} -> {pre[-1]}")
    # AFTER T_rain (+ a couple of steps of lag): volume is FLAT.
    post = vols[times > T_RAIN + 2 * (times[1] - times[0])]
    assert post.size >= 3, "not enough post-rain samples"
    assert np.ptp(post) < 1e-9, (
        f"volume kept changing after T_rain (ptp={np.ptp(post):.2e}); "
        "t is not bound in update_aux_variables")
    # And the plateau is the expected accumulated rain (rate·T_rain·area),
    # NOT an ever-growing ramp.
    expected_gain = RATE * T_RAIN * XMAX
    assert abs((post[-1] - v0) - expected_gain) < 5e-3, (
        f"accumulated rain {post[-1]-v0:.4f} != rate·T_rain·area "
        f"{expected_gain:.4f}")


# ── runtime-level: update_aux binds t; source reads the aux ──────────────────

def test_update_aux_binds_time_directly():
    """The lowered ``update_aux_variables(Q, Qaux, p, t, x)`` returns the rain
    rate for t < T_rain and zero after — proving ``t`` reaches the kernel."""
    sm = SystemModel.from_model(RainTracer())
    rt = NumpyRuntimeModel.from_system_model(sm)
    p = rt.parameters
    Q = np.array([1.0])
    Qaux = np.zeros(rt.n_aux_variables)
    x = np.zeros(3)                    # position vector, length 3
    r_before = float(np.ravel(rt.update_aux_variables(Q, Qaux, p, 0.02, x))[0])
    r_after = float(np.ravel(rt.update_aux_variables(Q, Qaux, p, 0.20, x))[0])
    assert r_before == pytest.approx(0.5)
    assert r_after == pytest.approx(0.0)


# ── (b) a source referencing dt: emitted kernel matches hand evaluation ──────

def test_source_binds_dt_matches_hand_eval():
    """Inject a ``dt``-dependent term into the source and verify the lowered
    numpy kernel binds ``dt`` (``source(Q, Qaux, p, t, dt, x)``) and reproduces
    the symbolic value at a known ``dt``."""
    sm = SystemModel.from_model(RainTracer())
    c = sm.state[0]
    # source row 0 := r_o + dt·c  (a dt-scaled reaction term)
    src = sp.Matrix([[sm.source[0, 0] + DT_SYMBOL * c]])
    sm.source = ZArray(src)
    rt = NumpyRuntimeModel.from_system_model(sm)
    assert rt.source_needs_dt is True          # dt is a first-class source arg

    p = rt.parameters
    cval = 2.0
    Q = np.array([[cval]])                      # (n_state, n_cells=1)
    Qaux = np.zeros((rt.n_aux_variables, 1))    # r_o = 0 → isolate dt·c
    x = np.zeros((3, 1))
    dt = 0.3
    val = float(np.asarray(rt.source(Q, Qaux, p, 0.0, dt, x)).ravel()[0])
    assert val == pytest.approx(dt * cval), f"dt not bound: {val} != {dt*cval}"


# ── (c-i) position is a length-3 vector regardless of dimension ──────────────

def test_source_binds_position_length3_regardless_of_dim():
    """A manufactured source ``S = x`` (position[0]) evaluated per cell equals
    the cell's x-coordinate — and the position argument is length 3 even for a
    1-D model (y, z present, unused)."""
    sm = SystemModel.from_model(RainTracer())
    x_sym, y_sym, z_sym = (sm.position.x, sm.position.y, sm.position.z)
    # source row 0 := x + 10·y + 100·z  → reads all three position components
    sm.source = ZArray(sp.Matrix([[x_sym + 10 * y_sym + 100 * z_sym]]))
    rt = NumpyRuntimeModel.from_system_model(sm)

    p = rt.parameters
    ncell = 4
    Q = np.ones((1, ncell))
    Qaux = np.zeros((rt.n_aux_variables, ncell))
    # a length-3 position array (3, ncell) — 1-D model still gets 3 rows.
    pos = np.array([[0.1, 0.2, 0.3, 0.4],       # x
                    [1.0, 2.0, 3.0, 4.0],       # y
                    [0.5, 0.5, 0.5, 0.5]])      # z
    out = np.asarray(rt.source(Q, Qaux, p, 0.0, 0.0, pos)).ravel()
    expected = pos[0] + 10 * pos[1] + 100 * pos[2]
    np.testing.assert_allclose(out, expected)


# ── (c-ii) C-family signature contract: source + update_aux carry time/dt/X ──

def test_c_family_emit_source_and_update_aux_signatures():
    """Both the OpenFOAM and the generic-C (dmplex/amrex) SystemModel emit
    carry the REQ-185 convention: ``source`` gains ``time`` + ``dt`` + one
    position vector ``X``; ``update_aux_variables`` gains ``time`` + ``X`` (no
    dt)."""
    # A model with BOTH a source and an algebraic aux closure so both kernels
    # are emitted.  RainTracer's update_aux is pointwise (r_o), source reads it.
    sm = SystemModel.from_model(RainTracer())

    # -- OpenFOAM (per-cell source kernel) --------------------------------
    from zoomy_core.transformation.to_openfoam import FoamSystemModelPrinter
    foam = FoamSystemModelPrinter(sm).create_code()
    src_sig = foam[foam.index("Model::source") if "Model::source" in foam
                   else foam.index(" source("):]
    src_sig = foam.split(" source(", 1)[1].split(")", 1)[0]
    assert "time" in src_sig and "dt" in src_sig and "X" in src_sig, (
        f"foam source signature missing time/dt/X: {src_sig!r}")

    # -- generic C (dmplex/amrex path) ------------------------------------
    from zoomy_core.transformation.to_c import CppModel
    printer = CppModel(sm)
    src_block = "\n".join(printer._emit_operator_kernels())
    upd_block = "\n".join(printer._emit_update_kernels())

    c_src_sig = src_block.split(" source(", 1)[1].split(")", 1)[0]
    assert "time" in c_src_sig and "dt" in c_src_sig and "X" in c_src_sig, (
        f"C source signature missing time/dt/X: {c_src_sig!r}")

    c_upd_sig = upd_block.split(" update_aux_variables(", 1)[1].split(")", 1)[0]
    assert "time" in c_upd_sig and "X" in c_upd_sig, (
        f"C update_aux signature missing time/X: {c_upd_sig!r}")
    # update_aux carries NO dt (algebraic aux closures are dt-independent).
    assert "dt" not in c_upd_sig, (
        f"C update_aux unexpectedly carries dt: {c_upd_sig!r}")
    # exactly ONE position argument ``X`` (a length-3 vector, not 3 scalars).
    assert c_src_sig.count(" X") == 1 and c_upd_sig.count(" X") == 1


# ── numpy lambdify accepts the trailing coordinate args on source ------------

def test_numpy_source_accepts_time_dt_position_args():
    sm = SystemModel.from_model(RainTracer())
    rt = NumpyRuntimeModel.from_system_model(sm)
    p = rt.parameters
    Q = np.ones((1, 3))
    Qaux = np.zeros((rt.n_aux_variables, 3))
    x = np.zeros((3, 3))
    # full arg form (Q, Qaux, p, t, dt, x) must not raise
    out = rt.source(Q, Qaux, p, 0.0, 0.01, x)
    assert np.asarray(out).shape[-1] == 3
