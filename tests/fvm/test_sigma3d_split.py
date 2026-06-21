"""Stay-3D σ reference solver (:class:`Sigma3DSplitSolver`) — the face-consistent
ω makes the inviscid 3-D solver CONVERGENT.

The lagged-ω prototype was non-convergent on the inviscid BBSM13/AHS26
stationary solution (error GREW under refinement at the standard CFL, |ω|
unbounded — a dispersive ω↔mom instability).  The split solver reconstructs ω
at ζ-faces from the shared horizontal mass fluxes (telescoping, discretely
divergence-free), so:

* the error DECAYS at ≈O(Δx) at the standard CFL=0.4, at all times, and
* |ω|max stays bounded (its true physical value), grid-independent.

These are run, not asserted by hand.
"""
import numpy as np

from zoomy_core.model.models.sigma3d import Sigma3D
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree, ShallowInPlane
from zoomy_core.model.boundary_conditions import Extrapolation
from zoomy_core.fvm.sigma3d_split_solver import Sigma3DSplitSolver

# BBSM13 / AHS26 §3.1: stationary inviscid hydrostatic-Euler solution.
_A, _B, _ZBAR, _G = 0.1, 1.0, 2.0, 9.81
def _h(x): return 2.0 - np.exp(-x**2)
def _zb(x): h = _h(x); return _ZBAR - h - (_A**2*_B**2)/(2*_G*np.sin(_B*h)**2)
def _u(x, z): h = _h(x); return (_A*_B)/np.sin(_B*h)*np.cos(_B*(z - _zb(x)))


def _bbsm13_model():
    m = Sigma3D(closures=[Newtonian(), NavierSlip(), StressFree(), ShallowInPlane()],
                    parameters={"nu": 0.0, "e_x": 0.0, "g": _G,
                                "rho": 1.0, "lambda_s": 0.0})
    m.boundary_conditions = [Extrapolation(tag="left"), Extrapolation(tag="right")]
    return m


def _ic(x, zeta):
    h = _h(x); zb = _zb(x)
    return zb, h, h * _u(x, zb + zeta * h)


def _l1_error(NX, NY, t_end):
    s = Sigma3DSplitSolver(NX, NY, domain=(-5., 5., 0., 1.), cfl=0.4, rk=2)
    r = s.solve(_bbsm13_model(), _ic, t_end)
    xc, zc, h, b, uu = r["x"], r["zeta"], r["h"], r["b"], r["u"]
    uref = np.array([[_u(xc[i], _zb(xc[i]) + zc[k]*_h(xc[i]))
                      for k in range(NY)] for i in range(NX)])
    L1u = float(np.mean(np.abs(uu - uref)))
    dEta = float(np.max(np.abs(b + h - (_zb(xc) + _h(xc)))))
    return L1u, dEta, s.omega_max


def test_bbsm13_converges_under_refinement():
    """Inviscid BBSM13 at standard CFL: error decays ≈O(Δx); |ω| bounded."""
    e0, eta0, om0 = _l1_error(80, 8, 1.0)
    e1, eta1, om1 = _l1_error(160, 16, 1.0)
    # the OLD lagged-ω solver GREW (0.12 → 0.26); the fix must DECAY
    assert e1 < 0.7 * e0, f"u error did not decay: {e0:.4e} -> {e1:.4e}"
    assert eta1 < eta0, f"η error did not decay: {eta0:.4e} -> {eta1:.4e}"
    # the OLD |ω|max blew up (0.15 -> 0.39 -> 1.0); the fix stays bounded & flat
    assert om0 < 0.1 and om1 < 0.1, f"|ω| not bounded: {om0:.3e}, {om1:.3e}"
    assert abs(om1 - om0) < 0.2 * om0, "|ω| not grid-independent"


def test_bbsm13_stays_accurate_long_time():
    """At t=5 the steady state is held and STILL converges (not a transient)."""
    e0, eta0, _ = _l1_error(80, 8, 5.0)
    e1, eta1, _ = _l1_error(160, 16, 5.0)
    assert e1 < 0.7 * e0, f"t=5 u error did not decay: {e0:.4e} -> {e1:.4e}"
    assert eta1 < 0.05, f"t=5 surface drift too large: {eta1:.4e}"


def test_relaxes_to_steady_state_from_perturbed_ic():
    """Different IC → SAME steady state: a localized depth-bump perturbation of
    BBSM13 radiates out the transmissive boundaries and the solution RETURNS to
    the cosine steady state (relaxes to the truncation floor of the exact-IC
    run).  The bump carries the BBSM13 vertical structure, so ω≠0 and the
    baroclinic state is recovered — unlike a vertically-uniform IC, which has no
    inviscid shear-generation mechanism and would settle to the SWE state."""
    model = _bbsm13_model()
    DELTA, X0, W = 0.15, -1.5, 0.6                     # initial |Δη| ≈ 0.15
    def ic_bump(x, zeta):
        h = _h(x) + DELTA * np.exp(-((x - X0) / W) ** 2)
        zb = _zb(x)
        return zb, h, h * _u(x, zb + zeta * h)

    NX, NY, dom = 120, 12, (-5., 5., 0., 1.)
    rp = Sigma3DSplitSolver(NX, NY, domain=dom, cfl=0.4, rk=2).solve(model, ic_bump, 3.0)
    re = Sigma3DSplitSolver(NX, NY, domain=dom, cfl=0.4, rk=2).solve(model, _ic, 3.0)
    xc, zc = rp["x"], rp["zeta"]

    dEta = float(np.max(np.abs(rp["b"] + rp["h"] - (_zb(xc) + _h(xc)))))
    assert dEta < 0.02, f"did not relax (initial bump 0.15): surface drift {dEta:.3e}"
    # different IC reaches the SAME steady state as the exact-IC run
    assert float(np.max(np.abs(rp["h"] - re["h"]))) < 5e-3, "did not converge to BBSM13 depth"
    uref = np.array([[_u(xc[i], _zb(xc[i]) + zc[k] * _h(xc[i]))
                      for k in range(NY)] for i in range(NX)])
    assert float(np.mean(np.abs(rp["u"] - uref))) < 0.04, "velocity did not return to cosine"


def test_viscous_navierslip_dambreak_stable():
    """Viscous + Navier-slip dam break: stable, mass-conserving, sheared."""
    model = Sigma3D(closures=[Newtonian(), NavierSlip(), StressFree(), ShallowInPlane()],
                        parameters={"nu": 1e-2, "e_x": 0.0, "g": 9.81,
                                    "rho": 1.0, "lambda_s": 1.0})
    model.boundary_conditions = [Extrapolation(tag="left"),
                                 Extrapolation(tag="right")]

    def ic(x, zeta):
        return 0.0, (2.0 if x < 5.0 else 1.0), 0.0

    NX, NY = 200, 16
    s = Sigma3DSplitSolver(NX, NY, domain=(0., 10., 0., 1.), cfl=0.4, rk=2)
    r = s.solve(model, ic, 0.5)
    h, mom, u = r["h"], r["mom"], r["u"]
    dx = 10.0 / NX
    mass = float(np.sum(h) * dx)
    mass0 = (2.0 * (NX // 2) + 1.0 * (NX - NX // 2)) * dx
    assert np.all(np.isfinite(h)) and np.all(np.isfinite(mom))
    assert abs(mass - mass0) / mass0 < 1e-6, f"mass drift {mass-mass0:.2e}"
    prof = u[np.argmin(np.abs(r["x"] - 5.5))]
    assert prof[-1] > prof[0], "no bed-to-surface shear from Navier-slip"
