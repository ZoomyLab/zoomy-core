"""LSQMesh — FVMMesh with precomputed least-squares reconstruction stencils.

This is the highest-fidelity mesh class: it caches the LSQ derivative
operators so that ``compute_derivatives`` is a single matrix–vector product
per cell instead of building the stencil on every call.
"""

from __future__ import annotations

import numpy as np
import param

from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.mesh.lsq_reconstruction import (
    least_squares_reconstruction_local,
    scale_lsq_derivative,
    find_derivative_indices,
    build_monomial_indices,
)


def _build_face_neighbors(face_cells, cell_neighbors, n_faces, n_faces_per_cell, n_cells):
    """Build face neighbor stencil for flux reconstruction."""
    polynomial_degree = 1
    n_face_neighbors = (2 * (n_faces_per_cell + 1) - 2) * polynomial_degree
    face_neighbors = (n_cells + 1) * np.ones((n_faces, n_face_neighbors), dtype=int)

    for i_f in range(n_faces):
        c0, c1 = face_cells[:, i_f]
        nbrs = set()
        for c in [c0, c1]:
            if c < len(cell_neighbors):
                for n in cell_neighbors[c]:
                    if n < n_cells:
                        nbrs.add(n)
        nbrs.discard(c0)
        nbrs.discard(c1)
        nbr_list = [c0, c1] + sorted(nbrs)
        n_avail = min(len(nbr_list), n_face_neighbors)
        face_neighbors[i_f, :n_avail] = nbr_list[:n_avail]

    return face_neighbors


class LSQMesh(FVMMesh):
    """FVMMesh + precomputed LSQ derivative operators."""

    _lsq_gradQ = param.Array(default=None, allow_None=True)
    _lsq_neighbors = param.Array(default=None, allow_None=True)
    _lsq_monomial_multi_index = param.Parameter(default=None)
    _lsq_scale_factors = param.Array(default=None, allow_None=True)
    _face_neighbors = param.Array(default=None, allow_None=True)

    @property
    def lsq_gradQ(self):
        return self._lsq_gradQ

    @property
    def lsq_neighbors(self):
        return self._lsq_neighbors

    @property
    def lsq_monomial_multi_index(self):
        return self._lsq_monomial_multi_index

    @property
    def lsq_scale_factors(self):
        return self._lsq_scale_factors

    @property
    def face_neighbors(self):
        return self._face_neighbors

    def compute_derivatives(self, u: np.ndarray, degree: int = 1,
                            derivatives_multi_index=None) -> np.ndarray:
        """Compute derivatives using precomputed LSQ stencil."""
        A_glob = self._lsq_gradQ
        neighbors = self._lsq_neighbors
        mon_indices = self._lsq_monomial_multi_index
        sf = self._lsq_scale_factors

        if derivatives_multi_index is None:
            derivatives_multi_index = mon_indices
        indices = find_derivative_indices(mon_indices, derivatives_multi_index)

        out = np.zeros((A_glob.shape[0], len(derivatives_multi_index)), dtype=float)
        for i in range(A_glob.shape[0]):
            A_loc = A_glob[i]
            nbr_idx = neighbors[i]
            u_neighbors = u[nbr_idx]
            delta_u = u_neighbors - u[i]
            out[i, :] = (sf * (A_loc.T @ delta_u))[indices]

        return out

    @classmethod
    def from_fvm(cls, fvm: FVMMesh, lsq_degree: int = 1) -> "LSQMesh":
        """Build an LSQMesh by computing LSQ operators from an FVMMesh."""
        lsq = cls(
            dimension=fvm.dimension,
            type=fvm.type,
            n_cells=fvm.n_cells,
            n_inner_cells=fvm.n_inner_cells,
            n_faces=fvm.n_faces,
            n_vertices=fvm.n_vertices,
            n_boundary_faces=fvm.n_boundary_faces,
            n_faces_per_cell=fvm.n_faces_per_cell,
            vertex_coordinates=fvm.vertex_coordinates,
            cell_vertices=fvm.cell_vertices,
            cell_faces=fvm.cell_faces,
            face_cells=fvm.face_cells,
            cell_neighbors=fvm.cell_neighbors,
            boundary_face_cells=fvm.boundary_face_cells,
            boundary_face_ghosts=fvm.boundary_face_ghosts,
            boundary_face_function_numbers=fvm.boundary_face_function_numbers,
            boundary_face_physical_tags=fvm.boundary_face_physical_tags,
            boundary_face_face_indices=fvm.boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=fvm.boundary_conditions_sorted_physical_tags,
            boundary_conditions_sorted_names=fvm.boundary_conditions_sorted_names,
            z_ordering=fvm.z_ordering,
            _cell_centers=fvm._cell_centers,
            _cell_volumes=fvm._cell_volumes,
            _cell_inradius=fvm._cell_inradius,
            _face_normals=fvm._face_normals,
            _face_volumes=fvm._face_volumes,
            _face_centers=fvm._face_centers,
        )

        dim = fvm.dimension
        centers = fvm.cell_centers_computed()

        lsq._lsq_gradQ, lsq._lsq_neighbors, lsq._lsq_monomial_multi_index = (
            least_squares_reconstruction_local(
                fvm.n_cells, dim, fvm.cell_neighbors,
                centers[:dim, :].T, lsq_degree,
                n_inner_cells=fvm.n_inner_cells,
            )
        )
        lsq._lsq_scale_factors = scale_lsq_derivative(lsq._lsq_monomial_multi_index)

        lsq._face_neighbors = _build_face_neighbors(
            fvm.face_cells, fvm.cell_neighbors,
            fvm.n_faces, fvm.n_faces_per_cell, fvm.n_cells,
        )

        return lsq

    @classmethod
    def from_msh(cls, filepath: str, lsq_degree: int = 1) -> "LSQMesh":
        fvm = FVMMesh.from_msh(filepath)
        return cls.from_fvm(fvm, lsq_degree)

    @classmethod
    def from_hdf5(cls, filepath: str, lsq_degree: int = 1) -> "LSQMesh":
        fvm = FVMMesh.from_hdf5(filepath)
        return cls.from_fvm(fvm, lsq_degree)

    @classmethod
    def create_1d(cls, domain: tuple, n_inner_cells: int, lsq_degree: int = 1) -> "LSQMesh":
        fvm = FVMMesh.create_1d(domain, n_inner_cells)
        return cls.from_fvm(fvm, lsq_degree)

    @classmethod
    def create_2d(cls, domain: tuple, nx: int, ny: int, lsq_degree: int = 1) -> "LSQMesh":
        fvm = FVMMesh.create_2d(domain, nx, ny)
        return cls.from_fvm(fvm, lsq_degree)

    @classmethod
    def create_3d(cls, domain: tuple, nx: int, ny: int, nz: int,
                  lsq_degree: int = 1) -> "LSQMesh":
        fvm = FVMMesh.create_3d(domain, nx, ny, nz)
        return cls.from_fvm(fvm, lsq_degree)
