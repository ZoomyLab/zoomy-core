"""REQ-174c — the Chorin predictor must preserve depth positivity at the 2-D
wet/dry front (BC tag-routing regression).

The extruded escalante reproducer (``thesis/cases/speed/vam_extruded.py``) drove
``h`` NEGATIVE at step 6 (blow-up ~step 44) in 2-D, while the 1-D twin marched
indefinitely.  The seed was NOT the elliptic stage, NOT the LSQ boundary
stencil, and NOT a missing positivity limiter (the predictor already runs
Audusse ``PositiveNonconservativeRusanov`` + KP-desingularised ``hinv``).  It
was a BOUNDARY-CONDITION TAG-ROUTING SWAP that exists only in ≥2-D:

``_pad_to_square`` (which makes the rectangular predictor sub-model square for
:class:`HyperbolicSolver`) rebuilt the model through ``SystemModel(...)`` and
copied the lambdified BC *kernel* (``boundary_conditions``) but silently dropped
the ``_bc_source`` *container*.  The flux operator's boundary-tag remap
(``solver_numpy.get_flux_operator``) needs ``_bc_source.list_sorted_function_names``
to align the mesh's POSITIONAL tag order (``create_2d``: left, right, bottom,
top) with the BC kernel's ALPHABETICAL Piecewise-branch order (bottom, left,
right, top).  With the container gone the remap was skipped and every 2-D
boundary face routed by mesh position against alphabetical branches — the
outflow face inherited the INFLOW Dirichlet ``q = Q_IN``, leaking mass out of
the dry-shelf boundary column at ``-Q_IN/(2·dx)`` per step until ``h < 0``.
1-D is accidentally safe (mesh order == alphabetical for {left, right}).

The fix propagates ``_bc_source`` / ``_aux_bc_source`` in ``_pad_to_square``.

This file pins:
* (a) the padded predictor carries the BC container (root-cause guard);
* (b) the outflow boundary ghost extrapolates (``q_x ≈ 0``), NOT the inflow
      discharge — the mis-route is gone;
* (c) ``h`` stays ``>= 0`` over the first steps (spurious drain gone) and the
      solution stays y-invariant (the extruded IC/BC must remain extruded).
"""
import numpy as np
import pytest
import sympy as sp

from zoomy_core.fvm.solver_chorin_vam_numpy import (
    ChorinSplitVAMSolver, _pad_to_square)
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.boundary_conditions import Dirichlet, Extrapolation
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.models import VAM
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.systemmodel.system_model import SystemModel


G, H_RES, H_DRY, Q_IN = 9.81, 0.34, 0.015, 0.11197
DOMAIN = (-1.5, 1.5)
BUMP = lambda x: 0.20 * np.exp(-(x ** 2) / (2 * 0.20 ** 2))


@pytest.fixture(scope="module")
def extruded_split():
    """VAM(1, dim=3) extruded escalante: inflow discharge on the left, a
    Dirichlet-P pin at the x-hi outflow, lateral extrapolation — the reproducer,
    shrunk (built once per module; the derivation is expensive)."""
    bcs = [Dirichlet("left", on="q_x_0", value=Q_IN),
           Dirichlet("left", on="q_x_1", value=0.0),
           Dirichlet("left", on="q_y_0", value=0.0),
           Dirichlet("left", on="q_y_1", value=0.0),
           Dirichlet("left", on="r_0", value=0.0),
           Dirichlet("left", on="r_1", value=0.0),
           Dirichlet("right", on="P_0", value=0.0),
           Dirichlet("right", on="P_1", value=0.0),
           Extrapolation(tag="bottom"), Extrapolation(tag="top")]
    model = VAM(level=1, dimension=3, boundary_conditions=bcs,
                closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    split = model.chorin_split(sp.Symbol("dt", positive=True), system_model=sm)
    return split, sm


def _setup(extruded_split, nx=30, ny=4):
    split, sm = extruded_split
    solver = ChorinSplitVAMSolver(
        split.SM_pred, split.SM_press, split.SM_corr,
        pressure_tol=1e-9, pressure_maxit=200)
    mesh = BaseMesh.create_2d((DOMAIN[0], DOMAIN[1], 0.0, 0.4), nx, ny)
    Q = np.asarray(solver.setup_simulation(mesh))
    idx = {str(s): k for k, s in enumerate(sm.state)}
    cc = np.asarray(solver._sim_mesh.cell_centers)
    xc = cc[0, :solver.nc]
    b = BUMP(xc)
    Q[idx["b"], :solver.nc] = b
    Q[idx["h"], :solver.nc] = np.maximum(
        np.where(xc < 1.0, H_RES - b, H_DRY), H_DRY)
    solver.update_aux_variables()
    return solver, idx, xc, cc


# ── (a) the padded predictor carries the BC container ────────────────────────

def test_pad_to_square_propagates_bc_container(extruded_split):
    """Root-cause guard: ``_pad_to_square`` must keep ``_bc_source`` so the flux
    operator's tag remap fires.  Without it the 2-D boundary tags rotate."""
    split, _ = extruded_split
    padded = _pad_to_square(split.SM_pred)
    src = getattr(padded, "_bc_source", None)
    assert src is not None, (
        "_pad_to_square dropped the BC container — the flux operator's "
        "boundary-tag remap will be skipped and 2-D tags mis-route.")
    # container names present (alphabetical) — the remap key
    assert list(src.list_sorted_function_names) == ["bottom", "left", "right", "top"]


# ── (b) the outflow ghost extrapolates, not the inflow discharge ─────────────

def test_outflow_ghost_is_not_the_inflow_discharge(extruded_split):
    solver, idx, xc, cc = _setup(extruded_split)
    mesh = solver._sim_mesh
    fc = mesh.face_centers
    normals = np.asarray(mesh.face_normals)
    qx0 = idx["q_x_0"]
    Q = solver._sim_Q
    Qaux = solver._sim_Qaux
    x_hi = float(fc[:, 0].max())
    x_lo = float(fc[:, 0].min())
    seen_out = seen_in = False
    for i in range(solver._n_bf):
        fidx = solver._bf_fidx[i]
        fx = fc[fidx, 0]
        gh = np.asarray(solver._bc_fn(
            solver._bc_indices[i], 0.0, fc[fidx, :], solver._d_face[i],
            Q[:, solver._bf_cells[i]], Qaux[:, solver._bf_cells[i]],
            solver._sim_parameters, normals[:, fidx]), float).reshape(-1)
        if abs(fx - x_hi) < 1e-9:            # outflow: q_x extrapolates → 0
            assert abs(gh[qx0]) < 1e-9, (
                f"outflow ghost q_x_0 = {gh[qx0]} (expected extrapolation ≈ 0); "
                "the outflow inherited the inflow Dirichlet q=Q_IN (mis-route).")
            seen_out = True
        elif abs(fx - x_lo) < 1e-9:          # inflow: q_x = Q_IN (declared)
            assert abs(gh[qx0] - Q_IN) < 1e-9, (
                f"inflow ghost q_x_0 = {gh[qx0]} (expected Q_IN={Q_IN}); "
                "the inflow lost its declared Dirichlet discharge (mis-route).")
            seen_in = True
    assert seen_out and seen_in


# ── (c) positivity + y-invariance over the march ─────────────────────────────

def test_predictor_keeps_h_positive_and_y_invariant(extruded_split):
    solver, idx, xc, cc = _setup(extruded_split)
    yc = cc[1, :solver.nc]
    hi = idx["h"]
    nx = 30
    dx = (DOMAIN[1] - DOMAIN[0]) / nx
    dt = 0.08 * dx / np.sqrt(G * H_RES)      # the reproducer's fragile CFL
    x_cols = np.unique(np.round(xc, 6))

    # Pre-fix: h_min crossed 0 by step 6 in the 60x8 case; the front reaches the
    # outflow only ~step 120, so these steps are a clean boundary-positivity test.
    for k in range(15):
        solver.step(dt)
        h = np.asarray(solver._sim_Q, float)[hi, :solver.nc]
        assert np.isfinite(h).all(), f"non-finite h at step {k}"

        # (1) POSITIVITY — the REQ-174c deliverable.  The pre-fix outflow mass
        # leak drove this negative; with correct BC routing the dry shelf holds.
        assert h.min() >= 0.0, (
            f"h went negative at step {k}: h_min={h.min():.3e} "
            "(spurious outflow-boundary mass leak).")

        # (2) TOP/BOTTOM SYMMETRY to machine precision — the sharp routing guard.
        # Pre-fix the bottom lateral wrongly inherited the 'right' outflow's
        # Dirichlet-P while the top kept extrapolation, so the two lateral walls
        # carried DIFFERENT BCs and the field was y-ASYMMETRIC.  Correct routing
        # gives identical extrapolation walls ⇒ h(y) mirrors exactly.
        tb = 0.0
        for xv in x_cols:
            col = h[np.abs(xc - xv) < 1e-9]
            ys = yc[np.abs(xc - xv) < 1e-9]
            colo = col[np.argsort(ys)]
            tb = max(tb, float(np.max(np.abs(colo - colo[::-1]))))
        assert tb < 1e-11, (
            f"top/bottom asymmetry {tb:.3e} at step {k} — the lateral walls "
            "carry different BCs (tag mis-route).")

        # (3) y-invariance BOUNDED.  It is NOT machine-zero: the lateral-row LSQ
        # ∂ₓ of the non-polynomial bump differs from interior rows by O(dx²) at
        # the wet/dry front — a SEPARATE, resolution-convergent boundary-stencil
        # truncation (the REQ-174b tangential-quality track), not this bug.  It
        # must stay small and must not run away.
        ystd = max(h[np.abs(xc - xv) < 1e-9].std() for xv in x_cols)
        assert ystd < 5e-3, f"y-variation runaway at step {k}: max y-std={ystd:.3e}"
