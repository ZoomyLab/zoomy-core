"""BaseMesh — topology-only mesh loaded from .msh via meshio (no PETSc).

All geometric quantities (volumes, normals, centers, …) are computed on the
fly from the stored topology.  Derived classes (``FVMMesh``, ``LSQMesh``, …)
cache selected quantities for performance.
"""

from __future__ import annotations

import os
from copy import deepcopy
from math import factorial
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import param

import zoomy_core.mesh.mesh_util as mesh_util
from zoomy_core.mesh.mesh_util import compute_subvolume

# Optional deps
try:
    import h5py
    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False

try:
    import meshio
    _HAVE_MESHIO = True
except ImportError:
    _HAVE_MESHIO = False


# ── helpers: meshio cell-type → zoomy type ──────────────────────────────────

_MESHIO_TO_ZOOMY = {
    "line": "line",
    "triangle": "triangle",
    "quad": "quad",
    "tetra": "tetra",
    "hexahedron": "hexahedron",
    "wedge": "wface",
}

_ZOOMY_TO_MESHIO = {v: k for k, v in _MESHIO_TO_ZOOMY.items()}

_FACES_PER_CELL = {
    "line": 2, "triangle": 3, "quad": 4,
    "tetra": 4, "hexahedron": 6, "wface": 5,
}

_DIM_OF_TYPE = {
    "line": 1, "triangle": 2, "quad": 2,
    "tetra": 3, "hexahedron": 3, "wface": 3,
}


# ── topology builder: extract faces from cell-vertex connectivity ───────────

def _local_faces_for_type(mesh_type: str):
    """Return a list of tuples: for each cell, the local vertex indices that
    form each face.  E.g. for a triangle: [(0,1), (1,2), (2,0)]."""
    if mesh_type == "line":
        return [(0,), (1,)]
    if mesh_type == "triangle":
        return [(0, 1), (1, 2), (2, 0)]
    if mesh_type == "quad":
        return [(0, 1), (1, 2), (2, 3), (3, 0)]
    if mesh_type == "tetra":
        return [(0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    if mesh_type == "hexahedron":
        return [
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (1, 2, 6, 5),
            (2, 3, 7, 6), (3, 0, 4, 7),
        ]
    if mesh_type == "wface":
        return [
            (0, 1, 2), (3, 4, 5),
            (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0),
        ]
    raise ValueError(f"Unsupported mesh type: {mesh_type}")


def _build_face_topology(cell_vertices: np.ndarray, mesh_type: str):
    """Build face-cell connectivity from cell-vertex data.

    Returns
    -------
    cell_faces : (n_faces_per_cell, n_cells) int
    face_cells : (2, n_faces) int  — second entry is -1 for boundary faces
    face_vertex_indices : list of tuple — global vertex indices per face (sorted)
    """
    local_faces = _local_faces_for_type(mesh_type)
    n_faces_per_cell = len(local_faces)
    n_cells = cell_vertices.shape[1]

    # Map: sorted vertex tuple → (face_index, [cell0, cell1])
    face_map: dict[tuple, list] = {}
    face_list: list[tuple] = []  # ordered list of face keys
    cell_faces = np.empty((n_faces_per_cell, n_cells), dtype=int)

    for ic in range(n_cells):
        verts = cell_vertices[:, ic]
        for lf_idx, lf in enumerate(local_faces):
            global_verts = tuple(sorted(verts[list(lf)]))
            if global_verts not in face_map:
                face_idx = len(face_list)
                face_list.append(global_verts)
                face_map[global_verts] = [face_idx, ic, -1]
            else:
                face_idx = face_map[global_verts][0]
                face_map[global_verts][2] = ic
            cell_faces[lf_idx, ic] = face_idx

    n_faces = len(face_list)
    face_cells = np.empty((2, n_faces), dtype=int)
    for fkey in face_list:
        fidx, c0, c1 = face_map[fkey]
        face_cells[0, fidx] = c0
        face_cells[1, fidx] = c1

    return cell_faces, face_cells, face_list


def _build_cell_neighbors(face_cells: np.ndarray, cell_faces: np.ndarray,
                          n_cells: int, n_inner_cells: int):
    """Build cell_neighbors from face_cells.

    For boundary faces (face_cells[1] == -1), the neighbor is the ghost cell.
    Ghost cells are indexed starting at n_inner_cells.
    """
    n_faces_per_cell = cell_faces.shape[0]
    cell_neighbors = (n_cells + 1) * np.ones((n_cells, n_faces_per_cell), dtype=int)

    for ic in range(n_inner_cells):
        for lf in range(n_faces_per_cell):
            fidx = cell_faces[lf, ic]
            c0, c1 = face_cells[0, fidx], face_cells[1, fidx]
            neighbor = c1 if c0 == ic else c0
            cell_neighbors[ic, lf] = neighbor

    return cell_neighbors


def _identify_boundary_faces(face_cells: np.ndarray, n_inner_cells: int,
                             mesh_type: str, cell_vertices: np.ndarray,
                             vertex_coordinates: np.ndarray,
                             meshio_mesh):
    """Identify boundary faces and assign ghost cells.

    Returns boundary arrays and updated face_cells with ghost cell indices.
    """
    boundary_mask = face_cells[1, :] == -1
    boundary_face_indices = np.where(boundary_mask)[0]
    n_boundary_faces = len(boundary_face_indices)

    # Ghost cells: one per boundary face, starting at n_inner_cells
    ghost_start = n_inner_cells
    n_ghost = n_boundary_faces
    n_cells = n_inner_cells + n_ghost

    boundary_face_cells = np.empty(n_boundary_faces, dtype=int)
    boundary_face_ghosts = np.empty(n_boundary_faces, dtype=int)
    boundary_face_face_indices = np.empty(n_boundary_faces, dtype=int)
    boundary_face_physical_tags = np.empty(n_boundary_faces, dtype=int)
    boundary_face_function_numbers = np.empty(n_boundary_faces, dtype=int)

    # Build physical tag lookup from meshio
    boundary_dict = {name: data[0] for name, data in meshio_mesh.field_data.items()}
    # Reverse: tag_id → name
    tag_to_name = {v: k for k, v in boundary_dict.items()}

    # Build a vertex-set → physical tag map from meshio boundary cells
    dim = _DIM_OF_TYPE[mesh_type]
    boundary_element_types = {
        1: "line",    # 1D boundaries are vertices (point), but gmsh uses "vertex"
        2: "line",    # 2D boundaries are edges (lines)
        3: "triangle",  # 3D boundaries are faces (triangles or quads)
    }

    # Collect boundary elements and their physical tags from meshio
    face_verts_to_tag: dict[tuple, int] = {}
    for cb, cell_data_block in zip(meshio_mesh.cells, meshio_mesh.cell_data.get("gmsh:physical", [])):
        cell_type = cb.type
        # Only boundary elements (lower-dimensional)
        if cell_type in _MESHIO_TO_ZOOMY and _DIM_OF_TYPE.get(_MESHIO_TO_ZOOMY.get(cell_type, ""), 0) >= dim:
            continue
        for i_elem, elem in enumerate(cb.data):
            sorted_verts = tuple(sorted(elem))
            tag = cell_data_block[i_elem]
            face_verts_to_tag[sorted_verts] = tag

    # Assign ghost cells and tags to boundary faces
    new_face_cells = face_cells.copy()
    for i_bf, fidx in enumerate(boundary_face_indices):
        inner_cell = face_cells[0, fidx]
        ghost_idx = ghost_start + i_bf

        boundary_face_cells[i_bf] = inner_cell
        boundary_face_ghosts[i_bf] = ghost_idx
        boundary_face_face_indices[i_bf] = fidx
        new_face_cells[1, fidx] = ghost_idx

        # Look up physical tag from face vertices
        # Reconstruct face vertices from cell_faces
        # We need the actual face vertex set for this face
        # Get it from the face topology we built
        pass  # will be filled below

    return (n_cells, n_ghost, new_face_cells,
            boundary_face_cells, boundary_face_ghosts,
            boundary_face_face_indices, boundary_face_physical_tags,
            boundary_face_function_numbers,
            face_verts_to_tag, boundary_face_indices, boundary_dict)


# ── BaseMesh ────────────────────────────────────────────────────────────────

class BaseMesh(param.Parameterized):
    """Topology-only mesh.  Geometry computed on the fly."""

    # Metadata
    dimension = param.Integer(default=2)
    type = param.String(default="triangle")
    n_cells = param.Integer(default=0)
    n_inner_cells = param.Integer(default=0)
    n_faces = param.Integer(default=0)
    n_vertices = param.Integer(default=0)
    n_boundary_faces = param.Integer(default=0)
    n_faces_per_cell = param.Integer(default=0)

    # Topology
    vertex_coordinates = param.Array(default=np.empty((0, 0)))
    cell_vertices = param.Array(default=np.empty((0, 0), dtype=int))
    cell_faces = param.Array(default=np.empty((0, 0), dtype=int))
    face_cells = param.Array(default=np.empty((0, 0), dtype=int))
    cell_neighbors = param.Array(default=np.empty((0, 0), dtype=int))

    # Boundary
    boundary_face_cells = param.Array(default=np.empty(0, dtype=int))
    boundary_face_ghosts = param.Array(default=np.empty(0, dtype=int))
    boundary_face_function_numbers = param.Array(default=np.empty(0, dtype=int))
    boundary_face_physical_tags = param.Array(default=np.empty(0, dtype=int))
    boundary_face_face_indices = param.Array(default=np.empty(0, dtype=int))
    boundary_conditions_sorted_physical_tags = param.Array(default=np.empty(0, dtype=int))
    boundary_conditions_sorted_names = param.List(default=[])

    # Optional
    z_ordering = param.Array(default=np.array([-1]))

    # ── on-the-fly geometry ─────────────────────────────────────────────

    def cell_centers_computed(self) -> np.ndarray:
        """Compute cell centers from vertex coordinates. Shape (3, n_cells)."""
        dim = self.dimension
        n_inner = self.n_inner_cells
        n_cells = self.n_cells
        verts_per_cell = self.cell_vertices.shape[0]

        centers = np.zeros((3, n_cells), dtype=float)
        for ic in range(n_inner):
            vert_ids = self.cell_vertices[:, ic]
            coords = self.vertex_coordinates[:dim, vert_ids]  # (dim, verts_per_cell)
            centers[:dim, ic] = coords.mean(axis=1)

        # Ghost cell centers: reflect inner cell across boundary face
        all_face_centers = self.face_centers_computed()
        all_face_normals = self.face_normals_computed() if dim > 1 else None

        for i_bf in range(self.n_boundary_faces):
            inner = self.boundary_face_cells[i_bf]
            ghost = self.boundary_face_ghosts[i_bf]
            fidx = self.boundary_face_face_indices[i_bf]
            fc = all_face_centers[fidx, :dim]
            inner_center = centers[:dim, inner]

            if dim == 1:
                centers[:dim, ghost] = 2 * fc - inner_center
            else:
                fn = all_face_normals[:dim, fidx]
                fn = fn / np.linalg.norm(fn)
                offset = fc - inner_center
                centers[:dim, ghost] = inner_center + 2 * np.dot(offset, fn) * fn

        return centers

    def cell_volumes_computed(self) -> np.ndarray:
        """Compute cell volumes. Shape (n_cells,)."""
        n_inner = self.n_inner_cells
        n_cells = self.n_cells
        dim = self.dimension
        volumes = np.ones(n_cells, dtype=float)
        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T

        for ic in range(n_inner):
            vert_ids = list(self.cell_vertices[:, ic])
            volumes[ic] = mesh_util.volume(coords_3d, vert_ids, self.type)

        # Ghost cell volumes = same as their inner cell
        for i_bf in range(self.n_boundary_faces):
            inner = self.boundary_face_cells[i_bf]
            ghost = self.boundary_face_ghosts[i_bf]
            volumes[ghost] = volumes[inner]

        return volumes

    def cell_inradius_computed(self) -> np.ndarray:
        """Compute cell inradius (min distance from center to face plane).

        Uses the same projection-based approach as PETSc's
        ``computeCellGeometryFVM`` for consistency.
        """
        from zoomy_core.mesh.lsq_reconstruction import compute_inradius_generic as _compute_inradius_generic

        n_inner = self.n_inner_cells
        n_cells = self.n_cells
        dim = self.dimension
        inradii = np.zeros(n_cells, dtype=float)

        if dim == 1:
            # 1D: inradius = half cell length
            vols = self.cell_volumes_computed()
            inradii[:] = vols / 2.0
        else:
            centers = self.cell_centers_computed()
            face_ctrs = self.face_centers_computed()
            face_norms = self.face_normals_computed()

            for ic in range(n_inner):
                fc_list = []
                fn_list = []
                for lf_idx in range(self.n_faces_per_cell):
                    fidx = self.cell_faces[lf_idx, ic]
                    fc_list.append(face_ctrs[fidx, :dim])
                    fn_list.append(face_norms[:dim, fidx])
                inradii[ic] = _compute_inradius_generic(
                    centers[:dim, ic], fc_list, fn_list
                )

        for i_bf in range(self.n_boundary_faces):
            inner = self.boundary_face_cells[i_bf]
            ghost = self.boundary_face_ghosts[i_bf]
            inradii[ghost] = inradii[inner]

        return inradii

    def face_normals_computed(self) -> np.ndarray:
        """Compute face normals. Shape (3, n_faces).

        For 1D meshes all normals point in the +x direction (convention).
        """
        dim = self.dimension
        n_faces = self.n_faces
        normals = np.zeros((3, n_faces), dtype=float)

        if dim == 1:
            normals[0, :] = 1.0
            return normals

        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T

        for ic in range(self.n_inner_cells):
            vert_ids = list(self.cell_vertices[:, ic])
            cell_normals = mesh_util.face_normals(coords_3d, vert_ids, self.type)
            cell_center = coords_3d[vert_ids].mean(axis=0)

            for lf in range(self.n_faces_per_cell):
                fidx = self.cell_faces[lf, ic]
                if np.linalg.norm(normals[:, fidx]) > 0:
                    continue  # already set by another cell

                n = cell_normals[lf]
                # Orient: normal should point from face_cells[0] to face_cells[1]
                c0 = self.face_cells[0, fidx]
                if c0 == ic:
                    face_verts = self._face_vertex_ids(fidx, ic, lf)
                    face_center = coords_3d[face_verts].mean(axis=0)
                    if np.dot(n, face_center - cell_center) < 0:
                        n = -n
                normals[:, fidx] = n

        return normals

    def face_volumes_computed(self) -> np.ndarray:
        """Compute face areas/lengths. Shape (n_faces,).

        For 1D meshes, face 'volumes' are 1.0 (point has no area).
        """
        dim = self.dimension
        n_faces = self.n_faces

        if dim == 1:
            return np.ones(n_faces, dtype=float)

        fvols = np.zeros(n_faces, dtype=float)
        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T

        for ic in range(self.n_inner_cells):
            vert_ids = list(self.cell_vertices[:, ic])
            face_areas_arr = mesh_util.face_areas(coords_3d, vert_ids, self.type)
            for lf in range(self.n_faces_per_cell):
                fidx = self.cell_faces[lf, ic]
                if fvols[fidx] == 0:
                    fvols[fidx] = face_areas_arr[lf]

        return fvols

    def face_centers_computed(self) -> np.ndarray:
        """Compute face centers. Shape (n_faces, 3)."""
        dim = self.dimension
        n_faces = self.n_faces
        fcenters = np.zeros((n_faces, 3), dtype=float)
        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T
        local_faces = _local_faces_for_type(self.type)
        done = np.zeros(n_faces, dtype=bool)

        for ic in range(self.n_inner_cells):
            verts = self.cell_vertices[:, ic]
            for lf_idx, lf in enumerate(local_faces):
                fidx = self.cell_faces[lf_idx, ic]
                if done[fidx]:
                    continue
                face_vert_ids = verts[list(lf)]
                fcenters[fidx] = coords_3d[face_vert_ids].mean(axis=0)
                done[fidx] = True

        return fcenters

    def face_subvolumes_computed(self) -> np.ndarray:
        """Compute face subvolumes. Shape (n_faces, 2)."""
        dim = self.dimension
        n_faces = self.n_faces
        subvols = np.zeros((n_faces, 2), dtype=float)

        if dim == 1:
            # 1D: subvolume = half cell length on each side
            vols = self.cell_volumes_computed()
            for fidx in range(n_faces):
                c0, c1 = self.face_cells[:, fidx]
                subvols[fidx, 0] = vols[c0] / 2
                subvols[fidx, 1] = vols[c1] / 2
            return subvols

        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T
        local_faces = _local_faces_for_type(self.type)
        centers = self.cell_centers_computed()

        for ic in range(self.n_inner_cells):
            verts = self.cell_vertices[:, ic]
            for lf_idx, lf in enumerate(local_faces):
                fidx = self.cell_faces[lf_idx, ic]
                face_vert_ids = verts[list(lf)]
                face_verts_coords = coords_3d[face_vert_ids, :dim]
                cell_center = centers[:dim, ic]
                sv = compute_subvolume(face_verts_coords, cell_center, dim)

                c0, c1 = self.face_cells[:, fidx]
                if ic == c0:
                    subvols[fidx, 0] = sv
                else:
                    subvols[fidx, 1] = sv

        # For boundary ghost cells, copy inner cell subvolume
        for i_bf in range(self.n_boundary_faces):
            fidx = self.boundary_face_face_indices[i_bf]
            inner = self.boundary_face_cells[i_bf]
            c0 = self.face_cells[0, fidx]
            if inner == c0:
                subvols[fidx, 1] = subvols[fidx, 0]
            else:
                subvols[fidx, 0] = subvols[fidx, 1]

        return subvols

    # ── internal helpers ────────────────────────────────────────────────

    def _face_center_single(self, fidx: int) -> np.ndarray:
        """Compute center of a single face."""
        dim = self.dimension
        local_faces = _local_faces_for_type(self.type)
        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T

        # Find a cell that has this face
        for ic in range(self.n_inner_cells):
            for lf_idx, lf in enumerate(local_faces):
                if self.cell_faces[lf_idx, ic] == fidx:
                    verts = self.cell_vertices[:, ic]
                    face_vert_ids = verts[list(lf)]
                    return coords_3d[face_vert_ids].mean(axis=0)[:dim]
        raise ValueError(f"Face {fidx} not found in any cell")

    def _face_normal_single(self, fidx: int) -> np.ndarray:
        """Compute normal of a single face (unit, dim-dimensional)."""
        dim = self.dimension
        coords_3d = np.zeros((self.n_vertices, 3), dtype=float)
        coords_3d[:, :dim] = self.vertex_coordinates[:dim, :].T
        local_faces = _local_faces_for_type(self.type)

        for ic in range(self.n_inner_cells):
            for lf_idx, lf in enumerate(local_faces):
                if self.cell_faces[lf_idx, ic] == fidx:
                    verts = self.cell_vertices[:, ic]
                    vert_ids = list(verts)
                    all_normals = mesh_util.face_normals(coords_3d, vert_ids, self.type)
                    n = all_normals[lf_idx, :dim]
                    n /= np.linalg.norm(n)
                    return n
        raise ValueError(f"Face {fidx} not found")

    def _face_vertex_ids(self, fidx: int, ic: int, lf_idx: int) -> list:
        """Get global vertex IDs for a face."""
        local_faces = _local_faces_for_type(self.type)
        verts = self.cell_vertices[:, ic]
        return list(verts[list(local_faces[lf_idx])])

    # ── gradient computation (on-the-fly LSQ) ───────────────────────────

    def compute_derivatives(self, u: np.ndarray, degree: int = 1,
                            derivatives_multi_index=None) -> np.ndarray:
        """Compute derivatives using on-the-fly LSQ reconstruction."""
        from zoomy_core.mesh.lsq_reconstruction import (
            least_squares_reconstruction_local,
            scale_lsq_derivative,
            find_derivative_indices,
            compute_derivatives as _compute_derivs,
        )
        centers = self.cell_centers_computed()
        dim = self.dimension
        lsq_gradQ, lsq_neighbors, lsq_monomial_multi_index = (
            least_squares_reconstruction_local(
                self.n_cells, dim, self.cell_neighbors,
                centers[:dim, :].T, degree,
            )
        )
        lsq_scale_factors = scale_lsq_derivative(lsq_monomial_multi_index)

        # Build a lightweight namespace that looks like a mesh to _compute_derivs
        class _Stencil:
            pass
        s = _Stencil()
        s.lsq_gradQ = lsq_gradQ
        s.lsq_neighbors = lsq_neighbors
        s.lsq_monomial_multi_index = lsq_monomial_multi_index
        s.lsq_scale_factors = lsq_scale_factors
        return _compute_derivs(u, s, derivatives_multi_index)

    # ── I/O ─────────────────────────────────────────────────────────────

    @classmethod
    def from_msh(cls, filepath: str) -> "BaseMesh":
        """Load a .msh file via meshio and build topology (no PETSc)."""
        if not _HAVE_MESHIO:
            raise RuntimeError("BaseMesh.from_msh() requires meshio")

        mio = meshio.read(filepath)
        boundary_dict = {name: data[0] for name, data in mio.field_data.items()}
        tag_to_name = {v: k for k, v in boundary_dict.items()}

        # Find the primary (highest-dim) cell block
        dim = 0
        primary_block = None
        primary_type = None
        for cb in mio.cells:
            mtype = _MESHIO_TO_ZOOMY.get(cb.type)
            if mtype and _DIM_OF_TYPE.get(mtype, 0) > dim:
                dim = _DIM_OF_TYPE[mtype]
                primary_block = cb
                primary_type = mtype

        if primary_block is None:
            raise ValueError("No supported cell type found in mesh")

        mesh_type = primary_type
        n_faces_per_cell = _FACES_PER_CELL[mesh_type]
        cell_data = primary_block.data  # (n_inner_cells, verts_per_cell)
        n_inner_cells = cell_data.shape[0]
        n_vertices = mio.points.shape[0]

        vertex_coordinates = mio.points[:, :dim].T  # (dim, n_vertices)
        cell_vertices_arr = cell_data.T  # (verts_per_cell, n_inner_cells)

        # Build face topology
        cell_faces_arr, face_cells_arr, face_list = _build_face_topology(
            cell_vertices_arr, mesh_type
        )
        n_faces = len(face_list)

        # Identify boundary faces (face_cells[1] == -1)
        boundary_mask = face_cells_arr[1, :] == -1
        boundary_face_indices = np.where(boundary_mask)[0]
        n_boundary_faces = len(boundary_face_indices)
        n_ghost = n_boundary_faces
        n_cells = n_inner_cells + n_ghost

        # Assign ghost cells
        boundary_face_cells_arr = np.empty(n_boundary_faces, dtype=int)
        boundary_face_ghosts_arr = np.empty(n_boundary_faces, dtype=int)
        boundary_face_face_indices_arr = np.empty(n_boundary_faces, dtype=int)
        boundary_face_physical_tags_arr = np.zeros(n_boundary_faces, dtype=int)
        boundary_face_function_numbers_arr = np.zeros(n_boundary_faces, dtype=int)

        # Build face_verts → physical tag from meshio boundary elements
        face_verts_to_tag: dict[tuple, int] = {}
        phys_data = mio.cell_data.get("gmsh:physical", [])
        for i_block, cb in enumerate(mio.cells):
            mtype = _MESHIO_TO_ZOOMY.get(cb.type)
            if mtype is None or _DIM_OF_TYPE.get(mtype, 0) >= dim:
                continue
            if i_block < len(phys_data):
                tags = phys_data[i_block]
                for i_elem, elem in enumerate(cb.data):
                    key = tuple(sorted(elem))
                    face_verts_to_tag[key] = tags[i_elem]

        # Assign ghosts and tags
        updated_face_cells = face_cells_arr.copy()
        for i_bf, fidx in enumerate(boundary_face_indices):
            ghost_idx = n_inner_cells + i_bf
            boundary_face_cells_arr[i_bf] = face_cells_arr[0, fidx]
            boundary_face_ghosts_arr[i_bf] = ghost_idx
            boundary_face_face_indices_arr[i_bf] = fidx
            updated_face_cells[1, fidx] = ghost_idx

            # Look up physical tag
            face_key = face_list[fidx]
            tag = face_verts_to_tag.get(face_key, 0)
            boundary_face_physical_tags_arr[i_bf] = tag

        # Build sorted tags and names
        unique_tags = sorted(set(boundary_face_physical_tags_arr))
        tag_to_func_num = {t: i for i, t in enumerate(unique_tags)}
        for i_bf in range(n_boundary_faces):
            t = boundary_face_physical_tags_arr[i_bf]
            boundary_face_function_numbers_arr[i_bf] = tag_to_func_num[t]

        sorted_tags = np.array(unique_tags, dtype=int)
        sorted_names = [tag_to_name.get(t, f"tag_{t}") for t in unique_tags]

        # Build cell neighbors
        cell_neighbors_arr = _build_cell_neighbors(
            updated_face_cells, cell_faces_arr, n_cells, n_inner_cells
        )
        # Ghost cell neighbors: copy from their inner cell's neighbors
        for i_bf in range(n_boundary_faces):
            inner = boundary_face_cells_arr[i_bf]
            ghost = boundary_face_ghosts_arr[i_bf]
            cell_neighbors_arr[ghost, :] = cell_neighbors_arr[inner, :]

        return cls(
            dimension=dim,
            type=mesh_type,
            n_cells=n_cells,
            n_inner_cells=n_inner_cells,
            n_faces=n_faces,
            n_vertices=n_vertices,
            n_boundary_faces=n_boundary_faces,
            n_faces_per_cell=n_faces_per_cell,
            vertex_coordinates=vertex_coordinates,
            cell_vertices=cell_vertices_arr,
            cell_faces=cell_faces_arr,
            face_cells=updated_face_cells,
            cell_neighbors=cell_neighbors_arr,
            boundary_face_cells=boundary_face_cells_arr,
            boundary_face_ghosts=boundary_face_ghosts_arr,
            boundary_face_function_numbers=boundary_face_function_numbers_arr,
            boundary_face_physical_tags=boundary_face_physical_tags_arr,
            boundary_face_face_indices=boundary_face_face_indices_arr,
            boundary_conditions_sorted_physical_tags=sorted_tags,
            boundary_conditions_sorted_names=sorted_names,
        )

    @classmethod
    def create_1d(cls, domain: tuple, n_inner_cells: int) -> "BaseMesh":
        """Build a uniform 1D interval mesh."""
        xL, xR = domain
        n_cells = n_inner_cells + 2
        n_vertices = n_inner_cells + 1
        n_faces = n_inner_cells + 1
        n_boundary_faces = 2
        n_faces_per_cell = 2
        dx = (xR - xL) / n_inner_cells

        vertex_coordinates = np.linspace(xL, xR, n_vertices, dtype=float).reshape(1, -1)

        cell_vertices = np.zeros((2, n_inner_cells), dtype=int)
        cell_vertices[0, :] = np.arange(n_vertices - 1)
        cell_vertices[1, :] = np.arange(1, n_vertices)

        cell_faces = np.zeros((2, n_inner_cells), dtype=int)
        cell_faces[0, :] = np.arange(n_faces - 1)
        cell_faces[1, :] = np.arange(1, n_faces)

        face_cells = np.empty((2, n_faces), dtype=int)
        face_cells[0, 1:-1] = np.arange(n_inner_cells - 1)
        face_cells[1, 1:-1] = np.arange(1, n_inner_cells)
        face_cells[0, 0] = n_inner_cells      # ghost left
        face_cells[1, 0] = 0
        face_cells[0, -1] = n_inner_cells - 1
        face_cells[1, -1] = n_inner_cells + 1  # ghost right

        cell_neighbors = (n_cells + 1) * np.ones((n_cells, 2), dtype=int)
        for ic in range(n_cells):
            cell_neighbors[ic, :] = [ic - 1, ic + 1]
        cell_neighbors[0, 0] = n_inner_cells
        cell_neighbors[n_inner_cells - 1, 1] = n_inner_cells + 1
        cell_neighbors[n_inner_cells, 0] = 1
        cell_neighbors[n_inner_cells, 1] = 0
        cell_neighbors[n_inner_cells + 1, 0] = n_inner_cells - 1
        cell_neighbors[n_inner_cells + 1, 1] = n_inner_cells - 2

        boundary_face_cells = np.array([0, n_inner_cells - 1], dtype=int)
        boundary_face_ghosts = np.array([n_inner_cells, n_inner_cells + 1], dtype=int)
        boundary_face_function_numbers = np.array([0, 1], dtype=int)
        boundary_face_physical_tags = np.array([0, 1], dtype=int)
        boundary_face_face_indices = np.array([0, n_faces - 1], dtype=int)

        return cls(
            dimension=1,
            type="line",
            n_cells=n_cells,
            n_inner_cells=n_inner_cells,
            n_faces=n_faces,
            n_vertices=n_vertices,
            n_boundary_faces=n_boundary_faces,
            n_faces_per_cell=n_faces_per_cell,
            vertex_coordinates=vertex_coordinates,
            cell_vertices=cell_vertices,
            cell_faces=cell_faces,
            face_cells=face_cells,
            cell_neighbors=cell_neighbors,
            boundary_face_cells=boundary_face_cells,
            boundary_face_ghosts=boundary_face_ghosts,
            boundary_face_function_numbers=boundary_face_function_numbers,
            boundary_face_physical_tags=boundary_face_physical_tags,
            boundary_face_face_indices=boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=np.array([0, 1], dtype=int),
            boundary_conditions_sorted_names=["left", "right"],
        )

    @classmethod
    def create_2d(cls, domain: tuple, nx: int, ny: int) -> "BaseMesh":
        """Build a uniform 2D quad mesh.

        Parameters
        ----------
        domain : (x_min, x_max, y_min, y_max)
        nx, ny : number of inner cells in x and y directions
        """
        x_min, x_max, y_min, y_max = domain
        n_inner_cells = nx * ny

        # ── vertices on a regular grid ──
        xs = np.linspace(x_min, x_max, nx + 1)
        ys = np.linspace(y_min, y_max, ny + 1)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")  # (nx+1, ny+1)
        n_vertices = (nx + 1) * (ny + 1)
        # vertex_coordinates: (3, n_vertices), row-major over (ix, iy)
        vertex_coordinates = np.zeros((3, n_vertices), dtype=float)
        vertex_coordinates[0] = gx.ravel()
        vertex_coordinates[1] = gy.ravel()

        def vid(ix, iy):
            return ix * (ny + 1) + iy

        # ── cell-vertex connectivity: (4, n_inner_cells) ──
        cell_vertices = np.empty((4, n_inner_cells), dtype=int)
        for ix in range(nx):
            for iy in range(ny):
                ic = ix * ny + iy
                cell_vertices[0, ic] = vid(ix, iy)
                cell_vertices[1, ic] = vid(ix + 1, iy)
                cell_vertices[2, ic] = vid(ix + 1, iy + 1)
                cell_vertices[3, ic] = vid(ix, iy + 1)

        # ── build face topology via _build_face_topology ──
        cell_faces, face_cells_raw, face_list = _build_face_topology(
            cell_vertices, "quad"
        )
        n_faces = len(face_list)

        # ── identify boundary faces and assign ghost cells ──
        boundary_mask = face_cells_raw[1, :] == -1
        boundary_face_indices = np.where(boundary_mask)[0]
        n_boundary_faces = len(boundary_face_indices)
        n_cells = n_inner_cells + n_boundary_faces

        face_cells = face_cells_raw.copy()
        boundary_face_cells = np.empty(n_boundary_faces, dtype=int)
        boundary_face_ghosts = np.empty(n_boundary_faces, dtype=int)
        boundary_face_face_indices = np.empty(n_boundary_faces, dtype=int)
        boundary_face_physical_tags = np.empty(n_boundary_faces, dtype=int)
        boundary_face_function_numbers = np.empty(n_boundary_faces, dtype=int)

        # Classify boundary faces by position of face center
        tag_map = {"left": 0, "right": 1, "bottom": 2, "top": 3}
        tol = 1e-12 * max(x_max - x_min, y_max - y_min)

        for i_bf, fidx in enumerate(boundary_face_indices):
            ghost_idx = n_inner_cells + i_bf
            inner_cell = face_cells_raw[0, fidx]

            boundary_face_cells[i_bf] = inner_cell
            boundary_face_ghosts[i_bf] = ghost_idx
            boundary_face_face_indices[i_bf] = fidx
            face_cells[1, fidx] = ghost_idx

            # Compute face center from vertices
            fverts = face_list[fidx]
            fc = vertex_coordinates[:2, list(fverts)].mean(axis=1)

            if abs(fc[0] - x_min) < tol:
                tag = tag_map["left"]
            elif abs(fc[0] - x_max) < tol:
                tag = tag_map["right"]
            elif abs(fc[1] - y_min) < tol:
                tag = tag_map["bottom"]
            elif abs(fc[1] - y_max) < tol:
                tag = tag_map["top"]
            else:
                tag = 0  # fallback

            boundary_face_physical_tags[i_bf] = tag
            boundary_face_function_numbers[i_bf] = tag

        # ── cell neighbors ──
        cell_neighbors = _build_cell_neighbors(
            face_cells, cell_faces, n_cells, n_inner_cells
        )

        # Ghost cell neighbors: each ghost mirrors its inner cell's stencil
        for i_bf in range(n_boundary_faces):
            ghost_idx = n_inner_cells + i_bf
            inner_cell = boundary_face_cells[i_bf]
            # Ghost cell's neighbors = inner cell's neighbors
            cell_neighbors[ghost_idx, :] = cell_neighbors[inner_cell, :]

        sorted_tags = np.array([0, 1, 2, 3], dtype=int)
        sorted_names = ["left", "right", "bottom", "top"]

        return cls(
            dimension=2,
            type="quad",
            n_cells=n_cells,
            n_inner_cells=n_inner_cells,
            n_faces=n_faces,
            n_vertices=n_vertices,
            n_boundary_faces=n_boundary_faces,
            n_faces_per_cell=4,
            vertex_coordinates=vertex_coordinates,
            cell_vertices=cell_vertices,
            cell_faces=cell_faces,
            face_cells=face_cells,
            cell_neighbors=cell_neighbors,
            boundary_face_cells=boundary_face_cells,
            boundary_face_ghosts=boundary_face_ghosts,
            boundary_face_function_numbers=boundary_face_function_numbers,
            boundary_face_physical_tags=boundary_face_physical_tags,
            boundary_face_face_indices=boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=sorted_tags,
            boundary_conditions_sorted_names=sorted_names,
        )

    @classmethod
    def create_3d(cls, domain: tuple, nx: int, ny: int, nz: int) -> "BaseMesh":
        """Build a uniform 3D hexahedral mesh.

        Parameters
        ----------
        domain : (x_min, x_max, y_min, y_max, z_min, z_max)
        nx, ny, nz : number of inner cells in each direction
        """
        x_min, x_max, y_min, y_max, z_min, z_max = domain
        n_inner_cells = nx * ny * nz

        # ── vertices on a regular grid ──
        xs = np.linspace(x_min, x_max, nx + 1)
        ys = np.linspace(y_min, y_max, ny + 1)
        zs = np.linspace(z_min, z_max, nz + 1)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        n_vertices = (nx + 1) * (ny + 1) * (nz + 1)
        vertex_coordinates = np.zeros((3, n_vertices), dtype=float)
        vertex_coordinates[0] = gx.ravel()
        vertex_coordinates[1] = gy.ravel()
        vertex_coordinates[2] = gz.ravel()

        def vid(ix, iy, iz):
            return ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz

        # ── cell-vertex connectivity: (8, n_inner_cells) ──
        # Hexahedron vertex ordering (VTK convention):
        #   bottom face: 0,1,2,3   top face: 4,5,6,7
        #   0=(ix,iy,iz), 1=(ix+1,iy,iz), 2=(ix+1,iy+1,iz), 3=(ix,iy+1,iz)
        #   4=(ix,iy,iz+1), 5=(ix+1,iy,iz+1), 6=(ix+1,iy+1,iz+1), 7=(ix,iy+1,iz+1)
        cell_vertices = np.empty((8, n_inner_cells), dtype=int)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    ic = ix * ny * nz + iy * nz + iz
                    cell_vertices[0, ic] = vid(ix, iy, iz)
                    cell_vertices[1, ic] = vid(ix + 1, iy, iz)
                    cell_vertices[2, ic] = vid(ix + 1, iy + 1, iz)
                    cell_vertices[3, ic] = vid(ix, iy + 1, iz)
                    cell_vertices[4, ic] = vid(ix, iy, iz + 1)
                    cell_vertices[5, ic] = vid(ix + 1, iy, iz + 1)
                    cell_vertices[6, ic] = vid(ix + 1, iy + 1, iz + 1)
                    cell_vertices[7, ic] = vid(ix, iy + 1, iz + 1)

        # ── build face topology ──
        cell_faces, face_cells_raw, face_list = _build_face_topology(
            cell_vertices, "hexahedron"
        )
        n_faces = len(face_list)

        # ── boundary faces + ghost cells ──
        boundary_mask = face_cells_raw[1, :] == -1
        boundary_face_indices = np.where(boundary_mask)[0]
        n_boundary_faces = len(boundary_face_indices)
        n_cells = n_inner_cells + n_boundary_faces

        face_cells = face_cells_raw.copy()
        boundary_face_cells = np.empty(n_boundary_faces, dtype=int)
        boundary_face_ghosts = np.empty(n_boundary_faces, dtype=int)
        boundary_face_face_indices = np.empty(n_boundary_faces, dtype=int)
        boundary_face_physical_tags = np.empty(n_boundary_faces, dtype=int)
        boundary_face_function_numbers = np.empty(n_boundary_faces, dtype=int)

        tag_map = {"left": 0, "right": 1, "front": 2, "back": 3,
                   "bottom": 4, "top": 5}
        tol = 1e-12 * max(x_max - x_min, y_max - y_min, z_max - z_min)

        for i_bf, fidx in enumerate(boundary_face_indices):
            ghost_idx = n_inner_cells + i_bf
            inner_cell = face_cells_raw[0, fidx]

            boundary_face_cells[i_bf] = inner_cell
            boundary_face_ghosts[i_bf] = ghost_idx
            boundary_face_face_indices[i_bf] = fidx
            face_cells[1, fidx] = ghost_idx

            fverts = face_list[fidx]
            fc = vertex_coordinates[:, list(fverts)].mean(axis=1)

            if abs(fc[0] - x_min) < tol:
                tag = tag_map["left"]
            elif abs(fc[0] - x_max) < tol:
                tag = tag_map["right"]
            elif abs(fc[1] - y_min) < tol:
                tag = tag_map["front"]
            elif abs(fc[1] - y_max) < tol:
                tag = tag_map["back"]
            elif abs(fc[2] - z_min) < tol:
                tag = tag_map["bottom"]
            elif abs(fc[2] - z_max) < tol:
                tag = tag_map["top"]
            else:
                tag = 0

            boundary_face_physical_tags[i_bf] = tag
            boundary_face_function_numbers[i_bf] = tag

        # ── cell neighbors ──
        cell_neighbors = _build_cell_neighbors(
            face_cells, cell_faces, n_cells, n_inner_cells
        )
        for i_bf in range(n_boundary_faces):
            ghost_idx = n_inner_cells + i_bf
            inner_cell = boundary_face_cells[i_bf]
            cell_neighbors[ghost_idx, :] = cell_neighbors[inner_cell, :]

        sorted_tags = np.array(list(tag_map.values()), dtype=int)
        sorted_names = list(tag_map.keys())

        return cls(
            dimension=3,
            type="hexahedron",
            n_cells=n_cells,
            n_inner_cells=n_inner_cells,
            n_faces=n_faces,
            n_vertices=n_vertices,
            n_boundary_faces=n_boundary_faces,
            n_faces_per_cell=6,
            vertex_coordinates=vertex_coordinates,
            cell_vertices=cell_vertices,
            cell_faces=cell_faces,
            face_cells=face_cells,
            cell_neighbors=cell_neighbors,
            boundary_face_cells=boundary_face_cells,
            boundary_face_ghosts=boundary_face_ghosts,
            boundary_face_function_numbers=boundary_face_function_numbers,
            boundary_face_physical_tags=boundary_face_physical_tags,
            boundary_face_face_indices=boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=sorted_tags,
            boundary_conditions_sorted_names=sorted_names,
        )

    def write_to_hdf5(self, filepath: str):
        """Serialize topology-only fields to HDF5."""
        if not _HAVE_H5PY:
            raise RuntimeError("write_to_hdf5 requires h5py")
        with h5py.File(filepath, "w") as f:
            g = f.create_group("mesh")
            g.attrs["format_version"] = 2

            g.create_dataset("dimension", data=self.dimension)
            g.create_dataset("type", data=self.type)
            g.create_dataset("n_cells", data=self.n_cells)
            g.create_dataset("n_inner_cells", data=self.n_inner_cells)
            g.create_dataset("n_faces", data=self.n_faces)
            g.create_dataset("n_vertices", data=self.n_vertices)
            g.create_dataset("n_boundary_faces", data=self.n_boundary_faces)
            g.create_dataset("n_faces_per_cell", data=self.n_faces_per_cell)

            g.create_dataset("vertex_coordinates", data=self.vertex_coordinates)
            g.create_dataset("cell_vertices", data=self.cell_vertices)
            g.create_dataset("cell_faces", data=self.cell_faces)
            g.create_dataset("face_cells", data=self.face_cells)
            g.create_dataset("cell_neighbors", data=self.cell_neighbors)

            g.create_dataset("boundary_face_cells", data=self.boundary_face_cells)
            g.create_dataset("boundary_face_ghosts", data=self.boundary_face_ghosts)
            g.create_dataset("boundary_face_function_numbers", data=self.boundary_face_function_numbers)
            g.create_dataset("boundary_face_physical_tags", data=self.boundary_face_physical_tags)
            g.create_dataset("boundary_face_face_indices", data=self.boundary_face_face_indices)
            g.create_dataset("boundary_conditions_sorted_physical_tags",
                             data=self.boundary_conditions_sorted_physical_tags)
            g.create_dataset("boundary_conditions_sorted_names",
                             data=np.array(self.boundary_conditions_sorted_names, dtype="S"))
            g.create_dataset("z_ordering", data=self.z_ordering)

    @classmethod
    def from_hdf5(cls, filepath: str) -> "BaseMesh":
        """Load a BaseMesh from HDF5."""
        if not _HAVE_H5PY:
            raise RuntimeError("from_hdf5 requires h5py")
        with h5py.File(filepath, "r") as f:
            g = f["mesh"]
            return cls(
                dimension=int(g["dimension"][()]),
                type=g["type"][()].decode("utf-8") if isinstance(g["type"][()], bytes) else str(g["type"][()]),
                n_cells=int(g["n_cells"][()]),
                n_inner_cells=int(g["n_inner_cells"][()]),
                n_faces=int(g["n_faces"][()]),
                n_vertices=int(g["n_vertices"][()]),
                n_boundary_faces=int(g["n_boundary_faces"][()]),
                n_faces_per_cell=int(g["n_faces_per_cell"][()]),
                vertex_coordinates=g["vertex_coordinates"][()],
                cell_vertices=g["cell_vertices"][()],
                cell_faces=g["cell_faces"][()],
                face_cells=g["face_cells"][()],
                cell_neighbors=g["cell_neighbors"][()],
                boundary_face_cells=g["boundary_face_cells"][()],
                boundary_face_ghosts=g["boundary_face_ghosts"][()],
                boundary_face_function_numbers=g["boundary_face_function_numbers"][()],
                boundary_face_physical_tags=g["boundary_face_physical_tags"][()],
                boundary_face_face_indices=g["boundary_face_face_indices"][()],
                boundary_conditions_sorted_physical_tags=g["boundary_conditions_sorted_physical_tags"][()],
                boundary_conditions_sorted_names=list(
                    np.array(g["boundary_conditions_sorted_names"][()], dtype="str")
                ),
                z_ordering=g["z_ordering"][()] if "z_ordering" in g else np.array([-1]),
            )

    def resolve_periodic_bcs(self, bcs):
        """Patch boundary_face_cells for periodic boundary conditions."""
        from zoomy_core.model.boundary_conditions import Periodic

        dict_physical_name_to_index = {
            v: i for i, v in enumerate(self.boundary_conditions_sorted_names)
        }
        dict_function_index_to_physical_tag = {
            i: v for i, v in enumerate(self.boundary_conditions_sorted_physical_tags)
        }

        boundary_face_cells_copy = deepcopy(self.boundary_face_cells)

        for bc in bcs.boundary_conditions_list:
            if type(bc) == Periodic:
                from_physical_tag = dict_function_index_to_physical_tag[
                    dict_physical_name_to_index[bc.periodic_to_physical_tag]
                ]
                to_physical_tag = dict_function_index_to_physical_tag[
                    dict_physical_name_to_index[bc.tag]
                ]

                mask_face_from = self.boundary_face_physical_tags == from_physical_tag
                mask_face_to = self.boundary_face_physical_tags == to_physical_tag

                from_cells = self.boundary_face_cells[mask_face_from]
                to_cells = self.boundary_face_ghosts[mask_face_to]

                centers = self.cell_centers_computed()
                from_coords = centers[:, from_cells]
                to_coords = centers[:, to_cells]

                significance_per_dimension = [
                    from_coords[d, :].max() - from_coords[d, :].min()
                    for d in range(self.dimension)
                ]
                sort_order_significance = np.lexsort([significance_per_dimension])

                from_cells_sort_order = np.lexsort(
                    [from_coords[d, :] for d in sort_order_significance]
                )
                to_cells_sort_order = np.lexsort(
                    [to_coords[d, :] for d in sort_order_significance]
                )

                indices = np.arange(mask_face_to.shape[0])
                indices_to = indices[mask_face_to]
                indices_to_sort = indices_to[to_cells_sort_order]
                indices_from = indices[mask_face_from]
                indices_from_sort = indices_from[from_cells_sort_order]

                self.boundary_face_cells[indices_to_sort] = boundary_face_cells_copy[
                    indices_from_sort
                ]
