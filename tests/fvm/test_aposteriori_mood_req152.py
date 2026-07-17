"""REQ-152 — a-posteriori MOOD positivity corrector (numpy backend).

Port of the jax ``_explicit_hyperbolic_step`` a-posteriori MOOD re-run
(``solver_jax.py`` ~L846-880) to the numpy solver.  At order 2 the SSP-RK2
candidate is taken, troubled cells are flagged (PAD ``h < 0`` + CAD
non-finite), and — only if any cell is troubled — the whole step is re-run
with those cells forced to 1st order (constant reconstruction) via a per-cell
``force_o1`` mask threaded into the reconstruction.  Positivity of a demoted
cell follows from the order-1 Xing–Zhang lemma; conservation from the shared
face flux.

HARD INVARIANT (user mandate): **no solver / reconstruction path clamps,
floors or truncates the depth ``h``.**  Positivity is achieved ONLY by the
masked order-1 re-step.  ``test_mood_output_is_exactly_the_masked_o1_restep``
and ``test_force_o1_reconstruction_is_constant_not_clamped`` pin this: the
committed depth equals the masked O1 re-step bit-for-bit (no post-processing),
and a negative reconstruction value passes through a demoted cell unchanged.

Wiring: ``ReconstructionSpec(positivity="mood")`` selects the a-posteriori
corrector (matching jax / dmplex semantics).  The former a-priori Xing–Zhang–Shu
deviation cap remains reachable under the distinct name ``positivity="zhang_shu"``.

Coverage note: a FLAT dry dam break gets MORE benign as the mesh refines
(finer mesh ⇒ smaller dt ⇒ the plain order-2 candidate stops overshooting:
plain minstep h is −2.8e-4 at 8×8, −1.7e-6 at 10×10, 0 at 12×12 and 32×32).
The strict h≥0 assertion is therefore exercised on the 8×8 slab, where plain
reaches −2.8e-4 and the corrector cures it to EXACTLY 0 with ~1e-16 mass drift.
A violent 32×32 *radial* dam break DOES drive plain negative (−9e-5); there
single-pass a-posteriori MOOD cuts it ~1600× to ~−6e-8 but does NOT reach
exactly 0 — a real a-posteriori-only residual (the Heun 2nd stage reuses the
Q0-based dt at a freshly-thinned front; jax closes this with its a-priori
``front_theta_tol`` pre-detector, out of scope for this port).  Per the
no-clamp mandate that residual is left as-is (reportable), never truncated.
"""
import numpy as np
import pytest

from zoomy_core.model.models.swe import SWE
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.initial_conditions import UserFunction, Constant
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.operations import gate_eigenvalues_dry


# ── setup helper ─────────────────────────────────────────────────────────

def _make_solver(positivity, nx=8, cfl=0.45, t_end=0.1, ic="slab", H=1.0):
    """Closed-box (wall) 2-D SWE dry dam break, order-2, given positivity."""
    sm = SystemModel.from_model(SWE(dimension=2, boundary_conditions=BoundaryConditions(
        [FromModel(tag="left", definition="wall_x"),
         FromModel(tag="right", definition="wall_x"),
         FromModel(tag="bottom", definition="wall_y"),
         FromModel(tag="top", definition="wall_y")])))
    n_state = len(sm.state)              # [b, h, hu, hv]

    def _ic(x):
        q = np.zeros(n_state)
        if ic == "radial":
            q[1] = H if np.hypot(x[0] - 0.5, x[1] - 0.5) < 0.15 else 0.0
        elif ic == "wet_smooth":
            q[1] = 1.0 + 0.2 * np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])
        else:                            # slab dry dam break
            q[1] = H if x[0] < 0.5 else 0.0
        return q

    sm.initial_conditions = UserFunction(function=_ic)
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_2d(domain=(0.0, 1.0, 0.0, 1.0), nx=nx, ny=nx)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, positivity=positivity),
        # REQ-181: the dry eigenvalue gate is no longer an NSM default; this
        # wet/dry dam break opts in explicitly (byte-identical to the old
        # default) so the recorded dry-front h_min values still hold.
        extra_operations=[gate_eigenvalues_dry()])
    solver = FreeSurfaceFlowSolver(
        time_end=t_end, compute_dt=timestepping.adaptive(CFL=cfl))
    solver.setup_simulation(mesh, nsm, write_output=False)
    return solver


def _h_index(solver):
    return int(solver._free_surface_h_index)


def _drive(solver, record=False):
    """Run to ``time_end``; return (min h over all steps, mass drift)."""
    nc = solver._sim_mesh.n_inner_cells
    vol = solver._sim_mesh.cell_volumes[:nc]
    h_idx = _h_index(solver)
    mass0 = float((solver._sim_Q[h_idx, :nc] * vol).sum())
    hmin = np.inf
    t = 0.0
    while t < solver.time_end:
        dt = solver.compute_dt(
            solver._sim_Q, solver._sim_Qaux, solver._sim_parameters,
            solver._sim_cell_inradius_face,
            solver._sim_compute_max_abs_eigenvalue)
        dt = min(float(dt), float(solver.time_end - t))
        if not np.isfinite(dt) or dt <= 0.0:
            break
        solver._sim_time = t
        solver.step(dt)
        hmin = min(hmin, float(solver._sim_Q[h_idx, :nc].min()))
        t += dt
    mass1 = float((solver._sim_Q[h_idx, :nc] * vol).sum())
    return hmin, abs(mass1 - mass0) / abs(mass0)


# ── the acceptance ───────────────────────────────────────────────────────

def test_plain_order2_goes_negative_control():
    """Control: without the corrector the plain order-2 candidate drives the
    depth negative on the dry dam break (so the corrector below has real
    work to do — not a vacuous pass)."""
    hmin, _ = _drive(_make_solver(""))
    assert hmin < 0.0, f"plain order-2 stayed non-negative (hmin={hmin}); test is vacuous"


def test_mood_keeps_depth_nonnegative_and_conserves_mass():
    """positivity='mood': the a-posteriori corrector keeps h ≥ 0 at EVERY
    step and mass is conserved to machine precision (closed box, shared face
    flux).  NO clamp — positivity is the order-1 Xing–Zhang lemma acting on
    the demoted troubled cells."""
    solver = _make_solver("mood")
    assert solver._mood is True and solver._support_o1 is True
    hmin, drift = _drive(solver)
    assert hmin >= 0.0, f"a-posteriori MOOD still yields h<0 (hmin={hmin})"
    assert drift < 1e-13, f"mass drifted {drift:.2e} (corrector must be conservative)"


def test_mood_is_strict_noop_on_wet_smooth_run():
    """On a fully-wet smooth run no cell is ever troubled, so the mood step is
    bit-identical to the plain order-2 step at every step — proving (a) the
    corrector is a strict no-op when nothing is flagged, and (b) it adds NO
    depth clamping / post-processing on the mood path."""
    s_plain = _make_solver("", ic="wet_smooth", t_end=0.04)
    s_mood = _make_solver("mood", ic="wet_smooth", t_end=0.04)
    t = 0.0
    n_steps = 0
    while t < s_plain.time_end:
        dt = s_plain.compute_dt(
            s_plain._sim_Q, s_plain._sim_Qaux, s_plain._sim_parameters,
            s_plain._sim_cell_inradius_face,
            s_plain._sim_compute_max_abs_eigenvalue)
        dt = min(float(dt), float(s_plain.time_end - t))
        if not np.isfinite(dt) or dt <= 0.0:
            break
        for s in (s_plain, s_mood):
            s._sim_time = t
            s.step(dt)
        assert np.array_equal(s_plain._sim_Q, s_mood._sim_Q), (
            f"mood diverged from plain on a smooth wet run at t={t}")
        t += dt
        n_steps += 1
    assert n_steps > 0


def test_mood_output_is_exactly_the_masked_o1_restep():
    """Mandate: the committed depth on a troubled step equals the masked
    order-1 re-step EXACTLY — no clamp, no floor, no post-processing.

    At each step we independently reproduce the corrector with the solver's
    OWN flux+source operators (candidate → troubled mask → masked re-step) and
    compare the depth row of the committed state bit-for-bit.  ``update_q``
    never touches ``h`` (it only caps momentum), so equality on the depth row
    isolates exactly the corrector output."""
    # Violent radial dry dam break so a troubled step fires early.
    solver = _make_solver("mood", nx=16, t_end=0.06, ic="radial", H=2.0)
    flux = solver._sim_flux_operator
    source = solver._sim_source_operator
    params = solver._sim_parameters
    h_idx = _h_index(solver)
    troubled_ever = False
    t = 0.0
    n = 0
    while t < solver.time_end and n < 20:
        dt = solver.compute_dt(
            solver._sim_Q, solver._sim_Qaux, solver._sim_parameters,
            solver._sim_cell_inradius_face,
            solver._sim_compute_max_abs_eigenvalue)
        dt = min(float(dt), float(solver.time_end - t))
        if not np.isfinite(dt) or dt <= 0.0:
            break
        Q0 = np.array(solver._sim_Q)
        Qaux = solver._sim_Qaux

        def _rhs(tt, Qin, f):
            z = np.zeros_like(Qin)
            return flux(dt, tt, Qin, Qaux, params, z, f) + source(dt, Qin, Qaux, params, z)

        def _rk2(f):
            Q1 = Q0 + dt * _rhs(t, Q0, f)
            Q2 = Q1 + dt * _rhs(t + dt, Q1, f)
            return 0.5 * (Q0 + Q2)

        cand = _rk2(None)
        troubled = (cand[h_idx, :] < 0.0) | ~np.isfinite(cand).all(axis=0)
        expected = _rk2(troubled) if troubled.any() else cand
        troubled_ever = troubled_ever or bool(troubled.any())

        solver._sim_time = t
        solver.step(dt)
        # h row is never touched by update_q → exact equality isolates the
        # corrector; NO clamp anywhere on the path.
        assert np.array_equal(solver._sim_Q[h_idx, :], expected[h_idx, :]), (
            f"committed depth != masked O1 re-step at step {n}")
        t += dt
        n += 1
    assert troubled_ever, "no troubled step occurred — corrector path not exercised"


def test_force_o1_reconstruction_is_constant_not_clamped():
    """Unit: a ``force_o1`` cell reconstructs to its cell MEAN on every face
    (zero slope) and any NEGATIVE state value passes through unchanged — i.e.
    the demotion is a slope kill, NOT a depth clamp."""
    from zoomy_core.fvm.reconstruction import LSQMUSCLReconstruction
    solver = _make_solver("mood", nx=8, t_end=0.0)
    mesh = solver._sim_mesh
    recon = LSQMUSCLReconstruction(mesh, dim=2)
    nc = mesh.n_inner_cells
    n_vars = 4
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((n_vars, nc))       # deliberately includes negatives
    ifaces = recon._interior_faces
    c = int(recon._iA_int[0])                    # A-cell of the 1st interior face
    Q[1, c] = -0.37                              # a clearly-negative depth entry
    n_bf = mesh.n_boundary_faces
    bf = np.zeros((n_vars, n_bf))

    QL0, QR0 = recon(Q, bf)                                   # order 2
    mask = np.ones(nc, dtype=bool)
    QLm, QRm = recon(Q, bf, force_o1=mask)                    # all cells demoted

    # demoted → piecewise constant: face value == cell mean, exactly
    assert np.array_equal(QLm[:, ifaces], Q[:, recon._iA_int])
    assert np.array_equal(QRm[:, ifaces], Q[:, recon._iB_int])
    # order-2 genuinely carried slopes (so the demotion is not a no-op)
    assert not np.array_equal(QL0[:, ifaces], Q[:, recon._iA_int])
    # negative value survived the demotion untouched — NOT floored to 0
    assert QLm.min() < 0.0
    assert QLm[1, ifaces[0]] == -0.37


def test_zhang_shu_does_not_engage_mood_corrector():
    """The former a-priori Xing–Zhang–Shu cap stays reachable under the
    distinct name ``positivity='zhang_shu'`` and must NOT turn on the
    a-posteriori corrector (``_mood`` stays False)."""
    solver = _make_solver("zhang_shu")
    assert solver._mood is False and solver._support_o1 is True


def test_base_solver_positivity_routing():
    """Base ``HyperbolicSolver._build_reconstruction`` routes ``zhang_shu`` to
    the a-priori XZS cap and ``mood`` to the plain (force_o1-aware) LSQ-MUSCL —
    the a-posteriori path carries NO deviation cap."""
    from zoomy_core.fvm.reconstruction import (
        LSQMUSCLReconstruction, PositivityPreservingLSQMUSCL)
    solver = _make_solver("mood", nx=8, t_end=0.0)
    mesh = solver._sim_mesh
    sm = solver._get_symbolic_model(solver._sim_model)
    base = FreeSurfaceFlowSolver.__mro__[1]      # base HyperbolicSolver class
    for pos, cls in (("zhang_shu", PositivityPreservingLSQMUSCL),
                     ("mood", LSQMUSCLReconstruction)):
        solver.nsm.reconstruction.positivity = pos
        r = base._build_reconstruction(solver, mesh, sm)
        assert type(r) is cls, f"{pos} → {type(r).__name__}, expected {cls.__name__}"
