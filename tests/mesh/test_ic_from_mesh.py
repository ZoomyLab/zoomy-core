"""REQ-134 — core initial-conditions-from-mesh generalises per-backend build_ic.

``zoomy_core.mesh.initial_conditions_from_mesh`` must reproduce the jax
Malpasset case's hand-rolled ``build_ic`` (thesis/cases/malpasset_jax/model.py,
``build_ic`` at lines 58-82): read gmsh ``$NodeData`` (B/H/U/V), nearest-node
interpolate onto cell centres, and assemble the state ``[b, h, h·u, h·v]``.
"""
import os

import numpy as np
import pytest

from zoomy_core.mesh import (
    BaseMesh,
    initial_conditions_from_mesh,
    interpolate_node_data_to_cells,
)

meshio = pytest.importorskip("meshio")
pytest.importorskip("scipy")

# Zoomy-root/data/malpasset/geo_malpasset-small.msh (this file:
# Zoomy/library/zoomy_core/tests/mesh/test_ic_from_mesh.py)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
MSH = os.path.join(_ROOT, "data", "malpasset", "geo_malpasset-small.msh")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MSH), reason=f"malpasset mesh fixture not found: {MSH}")

# Malpasset SME state layout (mode-0 momentum); higher moments start at 0.
STATE = ["b", "h", "q_x_0", "q_y_0"]
FIELD_MAP = {
    "b": "B",
    "h": "H",
    "q_x_0": lambda f: f["H"] * f["U"],
    "q_y_0": lambda f: f["H"] * f["V"],
}


def _reference_build_ic(mesh):
    """Verbatim algorithm of malpasset_jax/model.py build_ic, evaluated at the
    core mesh's inner cell centres (what the solver passes to the IC callable)."""
    from scipy.spatial import cKDTree

    mio = meshio.read(MSH)
    B, H = mio.point_data["B"], mio.point_data["H"]
    U, V = mio.point_data["U"], mio.point_data["V"]
    tree = cKDTree(mio.points[:, :2])
    centers = mesh.cell_centers_computed()[:2, : mesh.n_inner_cells].T

    Q = np.zeros((len(STATE), mesh.n_inner_cells))
    for c, xv in enumerate(centers):
        _, i = tree.query([float(xv[0]), float(xv[1])])
        hi = float(H[i])
        Q[0, c] = float(B[i])
        Q[1, c] = hi
        Q[2, c] = hi * float(U[i])
        Q[3, c] = hi * float(V[i])
    return Q


def test_from_msh_carries_node_data():
    mesh = BaseMesh.from_msh(MSH)
    assert mesh.node_data, "from_msh dropped $NodeData"
    for f in ("B", "H", "U", "V"):
        assert f in mesh.node_data
        assert mesh.node_data[f].shape == (mesh.n_vertices,)


def test_ic_from_mesh_matches_handrolled_build_ic():
    mesh = BaseMesh.from_msh(MSH)
    ref = _reference_build_ic(mesh)
    got = initial_conditions_from_mesh(mesh, FIELD_MAP, state_names=STATE)

    assert got.shape == ref.shape == (4, mesh.n_inner_cells)
    # Nearest-node interpolation reproduces the hand-rolled reader bit-for-bit.
    assert np.array_equal(got, ref), (
        f"max|Δ| = {np.max(np.abs(got - ref)):.3e}")


def test_ic_from_path_equals_ic_from_mesh():
    """The path overload loads the mesh itself and gives the same result."""
    mesh = BaseMesh.from_msh(MSH)
    from_mesh = initial_conditions_from_mesh(mesh, FIELD_MAP, state_names=STATE)
    from_path = initial_conditions_from_mesh(MSH, FIELD_MAP, state_names=STATE)
    assert np.allclose(from_mesh, from_path, atol=1e-10)


def test_cell_average_method_runs_and_is_close():
    """Connectivity average is a valid alternative node→cell interpolation."""
    mesh = BaseMesh.from_msh(MSH)
    h_nearest = interpolate_node_data_to_cells(
        mesh, mesh.node_data["H"], method="nearest")
    h_avg = interpolate_node_data_to_cells(
        mesh, mesh.node_data["H"], method="cell_average")
    assert h_avg.shape == h_nearest.shape == (mesh.n_inner_cells,)
    # Both sample the same smooth survey field; means agree to ~cell scale.
    assert np.mean(np.abs(h_avg - h_nearest)) < 0.5 * np.mean(np.abs(h_nearest) + 1e-9) + 1.0
