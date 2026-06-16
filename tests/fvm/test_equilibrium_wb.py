"""Well-balanced equilibrium reconstruction wiring (Model.equilibrium_reconstruction).

The unified ``reconstruct_equilibrium`` hook (solver_numpy.get_flux_operator) +
conservative-flux-jump source applies, on the standard NonconservativeRusanov
numerics, the equilibrium reconstruction selected by the model keyword:

* ``'audusse'`` — lake-at-rest (Audusse hydrostatic reconstruction).  This is the
  100% check on the wiring: through the hook it must hold lake-at-rest to machine
  precision AND match the existing FreeSurfaceFlowSolver (PositiveNonconservative-
  Rusanov) bit-for-bit, confirming the hook + source are correct on our solver.
* ``'bernoulli'`` — moving-equilibrium reconstruction (preserve discharge q and
  the per-streamline Bernoulli head H(s)); valid for single-signed velocity
  profiles.  Here we guard the reconstruction kernel itself: it round-trips a
  state to its own bed (identity) and preserves the discharge exactly.
  (Note: profiles that REVERSE sign over depth — e.g. BBSM13 for h>π/2 — are
  outside this recipe; the discharge-fraction streamline label is then not
  monotone.)
"""
import numpy as np

from zoomy_core.model.models import SME
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.mesh import BaseMesh
from zoomy_core.fvm.solver_numpy import HyperbolicSolver, FreeSurfaceFlowSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.model.initial_conditions as IC

_ETA0, _LEVEL, _NX = 2.0, 2, 100
def _bed(x): return 0.5 * np.exp(-x ** 2)


def _lake_at_rest_model(eqr):
    sm = SME(level=_LEVEL, equilibrium_reconstruction=eqr,
             boundary_conditions=BoundaryConditions(
                 [Extrapolation(tag="left"), Extrapolation(tag="right")])).system_model
    def ic(xv):
        x = float(xv[0]); b = _bed(x)
        return np.array([b, _ETA0 - b] + [0.0] * (_LEVEL + 1))
    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return sm


def _run(SolverClass, eqr, tend=2.0):
    sm = _lake_at_rest_model(eqr)
    mesh = BaseMesh.create_1d(domain=(-5., 5.), n_inner_cells=_NX)
    nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=1))
    solver = SolverClass(time_end=tend, compute_dt=timestepping.adaptive(CFL=0.45))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    return Q[0, :_NX] + Q[1, :_NX], np.asarray(Q)[:, :_NX]


def test_audusse_hook_holds_lake_at_rest_and_matches_reference():
    """Audusse via the unified hook on a plain HyperbolicSolver holds lake-at-rest
    to machine precision and is bit-identical to FreeSurfaceFlowSolver."""
    eta_none, _ = _run(HyperbolicSolver, "none")
    eta_aud, Qa = _run(HyperbolicSolver, "audusse")
    eta_ref, Qr = _run(FreeSurfaceFlowSolver, "none")
    assert np.max(np.abs(eta_none - _ETA0)) > 1e-4, "non-WB baseline should drift"
    assert np.max(np.abs(eta_aud - _ETA0)) < 1e-11, "audusse-hook must hold lake-at-rest"
    assert np.max(np.abs(Qa - Qr)) < 1e-10, "audusse-hook must match the reference Audusse"


def test_bernoulli_reconstruction_roundtrips_and_preserves_discharge():
    """The Bernoulli kernel reconstructs a (single-signed) sheared column to its
    own bed as the identity and preserves the discharge q=h·α_0 exactly."""
    from zoomy_core.fvm.bernoulli_wb import build_bernoulli_config, reconstruct
    sm = SME(level=2, boundary_conditions=BoundaryConditions(
        [Extrapolation(tag="left"), Extrapolation(tag="right")])).system_model
    cfg = build_bernoulli_config(sm, mode="bernoulli")
    # a sheared but single-signed column: u(σ) = 0.6 + 0.2·P1 (stays > 0)
    h, b = 1.5, 0.3
    a0, a1 = 0.6, 0.2
    Q = np.array([[b], [h], [h * a0], [h * a1], [0.0]])
    Qs = reconstruct(Q, np.array([b]), cfg)                       # own bed -> identity
    # round-trip is quadrature-limited (n_sigma=200); the discharge below is exact
    assert np.max(np.abs(Qs - Q)) < 5e-4, "round-trip to own bed must be ~identity"
    Qs2 = reconstruct(Q, np.array([b + 0.1]), cfg)               # shifted bed
    assert abs(Qs2[2, 0] - Q[2, 0]) < 1e-12, "discharge q=h·α_0 must be preserved exactly"
