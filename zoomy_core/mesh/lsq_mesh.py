"""LSQMesh — FVMMesh with precomputed least-squares reconstruction stencils.

This is the highest-fidelity mesh class: it caches the LSQ derivative
operators so that ``compute_derivatives`` is a single matrix–vector product
per cell instead of building the stencil on every call.
"""

from __future__ import annotations

import numpy as np
import param
from scipy.sparse import lil_matrix, csr_matrix

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
    _lsq_boundary_face_neighbors = param.Array(default=None, allow_None=True)
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
    def lsq_boundary_face_neighbors(self):
        return self._lsq_boundary_face_neighbors

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
                            derivatives_multi_index=None, *,
                            u_boundary_face) -> np.ndarray:
        """Compute derivatives using precomputed LSQ stencil.

        Parameters
        ----------
        u_boundary_face : ndarray | ``'extrapolation'``
            **Required.**  Either ``(n_boundary_faces,)`` array of face
            values (from a prescribed BC kernel), or the string
            ``'extrapolation'`` for Neumann-zero / face = inner-cell.
            See :func:`zoomy_core.mesh.lsq_reconstruction.compute_derivatives`
            for the rationale: silent extrapolation was deprecated
            because it masks Dirichlet BCs as Neumann-zero.
        """
        from zoomy_core.mesh.lsq_reconstruction import (
            _resolve_u_boundary_face,
        )
        A_glob = self._lsq_gradQ
        neighbors = self._lsq_neighbors
        bdy_neighbors = self._lsq_boundary_face_neighbors
        mon_indices = self._lsq_monomial_multi_index
        sf = self._lsq_scale_factors

        has_bdy = bdy_neighbors is not None and bdy_neighbors.size > 0
        if has_bdy:
            u_boundary_face = _resolve_u_boundary_face(
                u, u_boundary_face, self)

        if derivatives_multi_index is None:
            derivatives_multi_index = mon_indices
        indices = find_derivative_indices(mon_indices, derivatives_multi_index)

        out = np.zeros((A_glob.shape[0], len(derivatives_multi_index)), dtype=float)
        for i in range(A_glob.shape[0]):
            A_loc = A_glob[i]
            nbr_idx = neighbors[i]
            u_cells = u[nbr_idx]
            if has_bdy:
                bf = bdy_neighbors[i]
                u_bdy = np.where(
                    bf >= 0, u_boundary_face[np.maximum(bf, 0)], u[i]
                )
                u_full = np.concatenate([u_cells, u_bdy])
            else:
                u_full = u_cells
            delta_u = u_full - u[i]
            out[i, :] = (sf * (A_loc.T @ delta_u))[indices]

        return out

    def derivative_operator(self, multi_index) -> csr_matrix:
        """Sparse-matrix realisation of the LSQ derivative stencil.

        Returns a ``(n_inner_cells, n_inner_cells)`` sparse matrix
        ``D`` such that ``D @ u`` equals the cell-wise estimate of the
        derivative ``∂^multi_index u`` — the *same* quantity
        :meth:`compute_derivatives` produces for that ``multi_index``,
        but as an explicit **linear operator**.

        ``compute_derivatives`` applies the stencil to a *known* field;
        ``derivative_operator`` exposes the stencil itself, for
        assembling implicit / elliptic systems where the field is the
        unknown (e.g. the Chorin pressure-projection ``A·P = rhs``,
        where ``∂_xx P`` must enter the matrix, not be evaluated).

        Per cell ``i`` the stencil is
        ``deriv[i] = Σ_j w_j·(u[nbr_j] − u[i])`` with
        ``w_j = sf[idx]·A_loc[j, idx]`` — so
        ``D[i, nbr_j] += w_j`` and ``D[i, i] -= w_j``.

        Parameters
        ----------
        multi_index : tuple[int]
            Spatial-derivative orders per axis, e.g. ``(1,)`` for
            ``∂_x`` or ``(2,)`` for ``∂_xx`` in 1D — the same
            convention as :meth:`compute_derivatives` and the
            ``aux_registry`` ``multi_index`` field.

        Raises
        ------
        ValueError
            If ``multi_index`` is not in the mesh's monomial set —
            i.e. the mesh was built with too low an ``lsq_degree``.
        """
        multi_index = tuple(int(o) for o in multi_index)
        idx = int(find_derivative_indices(
            self._lsq_monomial_multi_index, [multi_index])[0])
        if idx < 0:
            raise ValueError(
                f"multi_index {multi_index} is not in the LSQ monomial "
                f"set {self._lsq_monomial_multi_index} — rebuild the "
                f"mesh with a higher lsq_degree."
            )
        A_glob = self._lsq_gradQ
        neighbors = self._lsq_neighbors
        sf = self._lsq_scale_factors
        nc = self.n_inner_cells
        D = lil_matrix((nc, nc), dtype=float)
        for i in range(nc):
            w = sf[idx] * A_glob[i][:, idx]          # stencil weights
            for j, n in enumerate(neighbors[i]):
                if n >= nc:
                    continue
                D[i, n] += w[j]
                D[i, i] -= w[j]
        return csr_matrix(D)

    @classmethod
    def from_fvm(cls, fvm: FVMMesh) -> "LSQMesh":
        """Build an LSQMesh shell from an FVMMesh, populating the LSQ
        stencil at degree 1 (the minimum that supports any derivative
        reconstruction).

        The LSQ polynomial degree is **not** a hand-adjustable knob
        here — use :func:`zoomy_core.mesh.ensure_lsq_mesh(mesh, model)`
        for any solver setup, which sizes the stencil from the model's
        NumericalSystemModel.  This factory exists for low-level mesh
        construction; the degree is set by the model in the higher
        layers."""
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

        lsq._build_lsq_stencil(degree=1)

        lsq._face_neighbors = _build_face_neighbors(
            fvm.face_cells, fvm.cell_neighbors,
            fvm.n_faces, fvm.n_faces_per_cell, fvm.n_cells,
        )

        return lsq

    def _build_lsq_stencil(self, degree: int) -> None:
        """Rebuild the LSQ stencil at the given polynomial ``degree``.

        Called from :meth:`from_fvm` (with ``degree=1``) and from
        :func:`zoomy_core.mesh.ensure_lsq_mesh` (with the model-derived
        degree).  Mutates the LSQ-cache slots on ``self`` in place.

        The ``degree`` parameter is **internal**: user code never sets
        it.  The right entry point for solver setup is
        ``ensure_lsq_mesh(mesh, model)``."""
        dim = self.dimension
        centers = self.cell_centers_computed()
        face_centers = self.face_centers_computed()
        bdy_face_centers = face_centers[
            self.boundary_face_face_indices, :dim
        ]
        cell_boundary_faces = [[] for _ in range(self.n_cells)]
        for i_bf, inner_cell in enumerate(self.boundary_face_cells):
            cell_boundary_faces[int(inner_cell)].append(i_bf)

        (self._lsq_gradQ,
         self._lsq_neighbors,
         self._lsq_boundary_face_neighbors,
         self._lsq_monomial_multi_index) = (
            least_squares_reconstruction_local(
                self.n_cells, dim, self.cell_neighbors,
                centers[:dim, :].T, degree,
                n_inner_cells=self.n_inner_cells,
                boundary_face_centers=bdy_face_centers,
                cell_boundary_faces=cell_boundary_faces,
                cell_vertices=self.cell_vertices,
            )
        )
        self._lsq_scale_factors = scale_lsq_derivative(
            self._lsq_monomial_multi_index)

    def _current_lsq_degree(self) -> int:
        """Polynomial degree currently represented by the cached
        stencil.  Used by ``ensure_lsq_mesh`` to decide if a rebuild
        is needed."""
        if self._lsq_monomial_multi_index is None:
            return 0
        return max(sum(mi) for mi in self._lsq_monomial_multi_index)

    @classmethod
    def from_msh(cls, filepath: str) -> "LSQMesh":
        fvm = FVMMesh.from_msh(filepath)
        return cls.from_fvm(fvm)

    @classmethod
    def from_hdf5(cls, filepath: str) -> "LSQMesh":
        fvm = FVMMesh.from_hdf5(filepath)
        return cls.from_fvm(fvm)

    @classmethod
    def create_1d(cls, domain: tuple, n_inner_cells: int) -> "LSQMesh":
        fvm = FVMMesh.create_1d(domain, n_inner_cells)
        return cls.from_fvm(fvm)

    @classmethod
    def create_2d(cls, domain: tuple, nx: int, ny: int) -> "LSQMesh":
        fvm = FVMMesh.create_2d(domain, nx, ny)
        return cls.from_fvm(fvm)

    @classmethod
    def create_3d(cls, domain: tuple, nx: int, ny: int, nz: int) -> "LSQMesh":
        fvm = FVMMesh.create_3d(domain, nx, ny, nz)
        return cls.from_fvm(fvm)
