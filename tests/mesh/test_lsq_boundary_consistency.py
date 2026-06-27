"""REQ-46 / task 0033 — boundary-cell LSQ gradient consistency.

The LSQ stencil places a boundary virtual neighbour at the GHOST-CELL
offset ``2·(face - cell)``.  For a prescribed-value (Dirichlet) face the
value carried there must be the linear extrapolation through the face,
``u_ghost = 2·u_face - u_cell`` — NOT the bare face value.  With the bare
face value the boundary gradient of a linear field is wrong at any finite
``dx`` (it does not converge away), capping the boundary reconstruction at
1st order while the interior is clean 2nd order.

These tests pin the fix in both code paths:

* mesh-derivative path —
  :meth:`zoomy_core.mesh.lsq_mesh.LSQMesh.compute_derivatives`
  (``_resolve_u_boundary_face``), and
* FVM-MUSCL path —
  :meth:`zoomy_core.fvm.reconstruction.LSQMUSCLReconstruction._lsq_gradient`.

A linear field ``u = a + b·x + c·y`` must reconstruct ``(b, c)`` exactly at
boundary cells, resolution-independent, for a Dirichlet boundary.
"""

import numpy as np
import pytest

from zoomy_core.mesh.lsq_mesh import LSQMesh
from zoomy_core.fvm.reconstruction import LSQMUSCLReconstruction


A, B, C = 0.37, 1.9, -0.8          # u(x,y) = A + B*x + C*y


def _linear_at(coords_xy):
    """coords_xy: (n, 2) -> linear field values."""
    return A + B * coords_xy[:, 0] + C * coords_xy[:, 1]


def _build(nx, ny):
    mesh = LSQMesh.create_2d((0.0, 1.0, 0.0, 1.0), nx, ny)
    centers = np.asarray(mesh.cell_centers)[:2, :].T      # (n_cells, 2)
    n_inner = mesh.n_inner_cells
    u = np.zeros(mesh.n_cells)
    u[:n_inner] = _linear_at(centers[:n_inner])

    bf_idx = mesh.boundary_face_face_indices
    face_centers = np.asarray(mesh.face_centers)[bf_idx, :2]   # (n_bf, 2)
    u_bf = _linear_at(face_centers)                           # Dirichlet face vals

    # Inner cells adjacent to a boundary face (face_cells[0] = this-side).
    bcells = np.unique(np.asarray(mesh.face_cells)[0, bf_idx])
    bcells = bcells[bcells < n_inner]
    return mesh, u, u_bf, bcells


@pytest.mark.parametrize("nx,ny", [(4, 4), (8, 8)])
def test_mesh_path_boundary_gradient_exact(nx, ny):
    """LSQMesh.compute_derivatives recovers (B, C) at boundary cells."""
    mesh, u, u_bf, bcells = _build(nx, ny)

    grad = mesh.compute_derivatives(
        u, degree=1, derivatives_multi_index=[(1, 0), (0, 1)],
        u_boundary_face=u_bf,
    )                                                  # (n_cells, 2)

    dudx = grad[bcells, 0]
    dudy = grad[bcells, 1]
    assert np.allclose(dudx, B, atol=1e-12), (
        f"max|dudx-B| = {np.max(np.abs(dudx - B)):.2e}")
    assert np.allclose(dudy, C, atol=1e-12), (
        f"max|dudy-C| = {np.max(np.abs(dudy - C)):.2e}")


@pytest.mark.parametrize("nx,ny", [(4, 4), (8, 8)])
def test_fvm_path_boundary_gradient_exact(nx, ny):
    """LSQMUSCLReconstruction._lsq_gradient recovers (B, C) at boundary cells
    and matches the mesh-derivative path bit-for-bit."""
    mesh, u, u_bf, bcells = _build(nx, ny)
    n_inner = mesh.n_inner_cells

    recon = LSQMUSCLReconstruction(mesh, dim=2)
    grad_fvm = recon._lsq_gradient(u[:n_inner], u_bf=u_bf)   # (2, n_inner)

    dudx = grad_fvm[0, bcells]
    dudy = grad_fvm[1, bcells]
    assert np.allclose(dudx, B, atol=1e-12), (
        f"max|dudx-B| = {np.max(np.abs(dudx - B)):.2e}")
    assert np.allclose(dudy, C, atol=1e-12), (
        f"max|dudy-C| = {np.max(np.abs(dudy - C)):.2e}")

    # Twin paths must agree where both are defined (inner cells).
    grad_mesh = mesh.compute_derivatives(
        u, degree=1, derivatives_multi_index=[(1, 0), (0, 1)],
        u_boundary_face=u_bf,
    )[:n_inner].T                                            # (2, n_inner)
    assert np.allclose(grad_fvm, grad_mesh, atol=1e-12)


def test_neumann_extrapolation_sanity():
    """Neumann-zero / extrapolation branch (ghost = inner cell): a constant
    field has zero gradient everywhere, including boundary cells, in both
    paths."""
    mesh, _, _, bcells = _build(6, 6)
    n_inner = mesh.n_inner_cells
    u = np.full(mesh.n_cells, 2.71828)

    grad_mesh = mesh.compute_derivatives(
        u, degree=1, derivatives_multi_index=[(1, 0), (0, 1)],
        u_boundary_face="extrapolation",
    )
    assert np.allclose(grad_mesh[bcells], 0.0, atol=1e-12)

    recon = LSQMUSCLReconstruction(mesh, dim=2)
    grad_fvm = recon._lsq_gradient(u[:n_inner], u_bf=None)
    assert np.allclose(grad_fvm[:, bcells], 0.0, atol=1e-12)
