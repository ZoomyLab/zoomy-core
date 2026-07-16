"""REQ-174b — boundary-cell tangential / mixed second-derivative decoupling.

CONTEXT / SCOPE (read before trusting this as a "fix"):
    This module pins the *stencil-level* behaviour of the REQ-174b boundary
    decoupling in ``least_squares_reconstruction_local``.  The worker verified
    that this decoupling — while a genuine, correct improvement to the boundary
    LSQ — does **NOT** cure the extruded-VAM 2-D instability the diagnosis
    attributed to it.  The dominant seed is an *interior* wet/dry-front
    phenomenon (the VAM 1/h source amplifying an O(h²) boundary-vs-interior
    truncation mismatch in ∂ₓb into the Chorin elliptic RHS), not the boundary
    second derivative.  See the worker report for the disproof.  These tests
    therefore document the boundary-stencil contract, not a closed REQ-174b.

The decoupling itself:
    The extrapolation/Neumann ghost the stencil places at 2·(face − cell)
    carries an *off-curve* value.  It is genuine boundary information ONLY for
    the *pure boundary-normal* derivatives (∂ₙ, ∂ₙₙ, … — the Neumann/Dirichlet
    BC, which the Chorin pressure Poisson NEEDS kept for well-posedness).  Left
    in the coupled fit it also corrupts the *tangential* and *mixed*
    derivatives via cross terms (on f=1+2x+3x² a boundary cell reported
    ∂_yy≈21, ∂_xy≈14 vs exact 0).

    The fix: fit tangential + mixed derivatives from the interior stencil only
    (exact), and restore each pure-normal derivative from a fit carrying ONLY
    its own aligned ghost (Neumann coupling preserved, corners decoupled).
    Consequence: the pure-normal 2nd derivative is Neumann-encoded, hence NOT
    polynomial-exact on a field that violates ∂ₙ=0 — this is deliberate (a
    "drop the ghost entirely" variant makes it exact but gives the elliptic
    operator a near-null transverse mode that blows the exact solve).
"""

import numpy as np
import pytest

from zoomy_core.mesh import BaseMesh
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.mesh.lsq_mesh import LSQMesh
from zoomy_core.mesh.lsq_reconstruction import build_vandermonde


DERIVS = [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]


def _mesh_2d(domain=(-1.5, 1.5, 0.0, 0.4), nx=30, ny=8, degree=2):
    m = LSQMesh.from_fvm(FVMMesh.from_base(BaseMesh.create_2d(domain, nx, ny)))
    m._build_lsq_stencil(degree)
    return m


def _boundary_mask(m):
    bcells = set(int(c) for c in np.asarray(m.boundary_face_cells))
    return np.array([i in bcells for i in range(m.n_inner_cells)])


def _dd(m, f, u_boundary_face="extrapolation"):
    nc = m.n_inner_cells
    cc = np.asarray(m.cell_centers)
    u = np.zeros(m.n_cells)
    u[:nc] = f(cc[0, :nc], cc[1, :nc])
    return m.compute_derivatives(u, degree=2, derivatives_multi_index=DERIVS,
                                 u_boundary_face=u_boundary_face)[:nc]


# ── (1) tangential + mixed 2nd derivs (and the Neumann-consistent normal)
#        are exact at boundary, both orientations ─────────────────────────────

@pytest.mark.parametrize("transpose", [False, True])
def test_tangential_and_mixed_second_derivatives_exact(transpose):
    """On a y-invariant quadratic the tangential and mixed second derivatives
    (and the normal one, which the field satisfies ∂ₙ=0 for) are machine-exact
    at boundary AND interior cells."""
    m = _mesh_2d()
    isb = _boundary_mask(m)
    col = {tuple(mi): k for k, mi in enumerate(DERIVS)}
    if not transpose:
        d = _dd(m, lambda X, Y: 1.0 + 2.0 * X + 3.0 * X ** 2)   # ∂ₙ=0 in y
        clean = {(0, 2): 0.0, (1, 1): 0.0, (0, 1): 0.0}         # tangential/mixed
    else:
        d = _dd(m, lambda X, Y: 1.0 + 2.0 * Y + 3.0 * Y ** 2)   # ∂ₙ=0 in x
        clean = {(2, 0): 0.0, (1, 1): 0.0, (1, 0): 0.0}
    for mi, tgt in clean.items():
        err = np.abs(d[:, col[mi]] - tgt)
        assert err[isb].max() < 1e-10, (
            f"∂{mi} boundary max|err|={err[isb].max():.2e}")
        assert err[~isb].max() < 1e-10, (
            f"∂{mi} interior max|err|={err[~isb].max():.2e}")


def test_normal_second_derivative_retains_neumann_ghost():
    """The pure boundary-normal 2nd derivative is Neumann-encoded, NOT
    polynomial-exact, on a field that violates ∂ₙ=0.  This is deliberate: the
    ghost supplies the elliptic Neumann coupling; dropping it (to make ∂ₙₙ
    exact) gives the Chorin pressure operator a near-null transverse mode."""
    m = _mesh_2d()
    isb = _boundary_mask(m)
    col = {tuple(mi): k for k, mi in enumerate(DERIVS)}
    # f=1+2x+3x^2 violates ∂ₙ=0 at the x-normal (left/right) walls ⇒ ∂ₓₓ there
    # keeps the (BC-encoding) ghost and is not 6.
    d = _dd(m, lambda X, Y: 1.0 + 2.0 * X + 3.0 * X ** 2)
    assert np.abs(d[:, col[(2, 0)]][isb] - 6.0).max() > 1.0


# ── (2) compact y-invariance mechanism (elliptic + corrector seeds) ──────────

def test_y_invariant_field_seeds_no_transverse_derivative():
    """A y-invariant field with x-curvature produces ~0 tangential ∂_yy
    (elliptic seed) and ∂_y (corrector seed) at boundary cells."""
    m = _mesh_2d()
    isb = _boundary_mask(m)
    d = _dd(m, lambda X, Y: 0.7 + 1.3 * X + 2.1 * X ** 2)
    col = {tuple(mi): k for k, mi in enumerate(DERIVS)}
    assert np.abs(d[:, col[(0, 2)]])[isb].max() < 1e-10
    assert np.abs(d[:, col[(0, 1)]])[isb].max() < 1e-10


# ── (3) BC-aware first derivative not regressed (well-balancing) ─────────────

def test_boundary_normal_first_derivative_recovers_linear_slope():
    """With Dirichlet face values the first derivative recovers a linear
    field's slope exactly at boundary cells, edge AND corner — the REQ-174
    BC-aware gradient / well-balancing is preserved."""
    m = _mesh_2d()
    nc = m.n_inner_cells
    isb = _boundary_mask(m)
    B, C = 1.9, -0.8
    lin = lambda X, Y: 0.37 + B * X + C * Y
    cc = np.asarray(m.cell_centers)
    u = np.zeros(m.n_cells)
    u[:nc] = lin(cc[0, :nc], cc[1, :nc])
    bfi = np.asarray(m.boundary_face_face_indices)
    fcx = m.face_centers_computed()[bfi, :2]
    u_bf = lin(fcx[:, 0], fcx[:, 1])
    d = m.compute_derivatives(u, degree=2, derivatives_multi_index=DERIVS,
                              u_boundary_face=u_bf)[:nc]
    col = {tuple(mi): k for k, mi in enumerate(DERIVS)}
    assert np.abs(d[:, col[(1, 0)]][isb] - B).max() < 1e-10
    assert np.abs(d[:, col[(0, 1)]][isb] - C).max() < 1e-10


# ── (4) 1-D path is bit-identical to the legacy coupled fit ──────────────────

def test_1d_boundary_fit_is_legacy_coupled_pinv():
    """In 1-D there is no tangential direction, so the boundary fit must be
    exactly the legacy coupled ``pinv(V)`` (ghost kept) — a strict no-op."""
    m = LSQMesh.from_fvm(FVMMesh.from_base(BaseMesh.create_1d((-1.5, 1.5), 40)))
    m._build_lsq_stencil(2)
    cc = np.asarray(m.cell_centers)
    fc = m.face_centers_computed()
    bfi = np.asarray(m.boundary_face_face_indices)
    bdy_fc = fc[bfi, :1]
    mon = m.lsq_monomial_multi_index
    neigh = m.lsq_neighbors
    bfn = m.lsq_boundary_face_neighbors
    max_nb, max_bd = neigh.shape[1], bfn.shape[1]
    Ag = np.asarray(m.lsq_gradQ)
    bcells = sorted(set(int(c) for c in np.asarray(m.boundary_face_cells)))
    assert bcells
    for ic in bcells:
        dX = np.zeros((max_nb + max_bd, 1))
        for j, n in enumerate(neigh[ic]):
            dX[j] = cc[:1, n] - cc[:1, ic]
        active = [int(b) for b in bfn[ic] if b >= 0]
        for j, b in enumerate(active):
            dX[max_nb + j] = 2 * (bdy_fc[b] - cc[:1, ic])
        V = build_vandermonde(dX, mon)
        for j in range(len(active), max_bd):
            V[max_nb + j] = 0.0
        A_legacy = np.linalg.pinv(V).T
        assert np.array_equal(Ag[ic], A_legacy), (
            f"1-D boundary cell {ic} deviates (max|Δ|="
            f"{np.abs(Ag[ic] - A_legacy).max():.2e})")
