"""FVM face reconstruction and diffusion operators.

Reconstruction classes
---------------------
All reconstruction classes share the same interface: ``recon(Q) → (Q_L, Q_R)``.
The solver calls this once per flux evaluation — no if-clauses in the flux loop.

- ``ConstantReconstruction``: 1st-order piecewise-constant (identity).
- ``MUSCLReconstruction``: 2nd-order piecewise-linear with slope limiting
  (Barth-Jespersen or Venkatakrishnan).
- ``FreeSurfaceMUSCL``: MUSCL with wet-dry fallback and h-positivity.

Diffusion operators
-------------------
- ``FaceGradient``: corrected face-normal gradient for diffusion.
- ``DiffusionOperator``: sparse discrete Laplacian (explicit + implicit).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres


# ── Reconstruction classes ───────────────────────────────────────────────────

class ConstantReconstruction:
    """First-order piecewise-constant reconstruction.

    Returns cell-center values at each face: Q_L = Q[:, iA], Q_R = Q[:, iB].
    """

    def __init__(self, mesh, dim):
        self.iA = mesh.face_cells[0]
        self.iB = mesh.face_cells[1]

    def __call__(self, Q):
        return Q[:, self.iA], Q[:, self.iB]


class MUSCLReconstruction:
    """Second-order MUSCL reconstruction with slope limiting.

    Green-Gauss gradients + slope limiter.
    Call signature matches ``ConstantReconstruction``: ``recon(Q) → (Q_L, Q_R)``.

    Parameters
    ----------
    mesh : FVMMesh or LSQMesh
    dim : int
    limiter : str
        "venkatakrishnan" — smooth, 2nd-order at extrema (default).
        "barth_jespersen" — strict DMP, clips to 1st order at extrema.
        "minmod" — classic TVD, most diffusive, clips to 1st order at extrema.
    """

    def __init__(self, mesh, dim, limiter="venkatakrishnan"):
        self.dim = dim
        self.n_faces = mesh.n_faces
        self.n_cells = mesh.n_cells
        self.nc = mesh.n_inner_cells
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        self.iA = iA
        self.iB = iB
        self._limiter_type = limiter

        centers = mesh.cell_centers[:dim, :]       # (dim, n_cells)
        face_ctrs = mesh.face_centers[:, :dim].T   # (n_faces, 3) → (dim, n_faces)

        # Cell-center → face-center displacement vectors
        self.r_Af = face_ctrs - centers[:, iA]     # (dim, n_faces)
        self.r_Bf = face_ctrs - centers[:, iB]

        # Mesh data for Green-Gauss
        self._normals = mesh.face_normals[:dim, :]
        self._face_volumes = mesh.face_volumes
        self._cell_volumes = mesh.cell_volumes

        # Venkatakrishnan smoothing: eps² = (K·h)²
        # Quadratic scaling preserves 2nd-order accuracy at smooth extrema.
        # (Cubic scaling (K·h)³ from the original paper gives φ→0 at extrema.)
        h = np.zeros(self.n_cells)
        h[:self.nc] = self._cell_volumes[:self.nc] ** (1.0 / max(dim, 1))
        self._eps_v2 = (1.0 * h) ** 2

    def __call__(self, Q):
        """Reconstruct face states. Returns (Q_L, Q_R), each (n_vars, n_faces)."""
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars)
        return self._reconstruct(Q, grads, phi)

    # ── Green-Gauss gradient ─────────────────────────────────────────

    def _green_gauss_gradient(self, u):
        """Cell-center gradient of scalar u via Green-Gauss. Returns (dim, n_cells)."""
        dim = self.dim
        grad = np.zeros((dim, self.n_cells))
        iA, iB = self.iA, self.iB
        u_face = 0.5 * (u[iA] + u[iB])
        for d in range(dim):
            contrib = u_face * self._normals[d, :] * self._face_volumes
            np.add.at(grad[d], iA, contrib / self._cell_volumes[iA])
            np.subtract.at(grad[d], iB, contrib / self._cell_volumes[iB])
        return grad

    # ── Neighbor bounds ──────────────────────────────────────────────

    def _neighbor_bounds(self, u):
        """Per-cell min/max over immediate face-neighbors."""
        u_max = u.copy()
        u_min = u.copy()
        np.maximum.at(u_max, self.iA, u[self.iB])
        np.maximum.at(u_max, self.iB, u[self.iA])
        np.minimum.at(u_min, self.iA, u[self.iB])
        np.minimum.at(u_min, self.iB, u[self.iA])
        return u_min, u_max

    # ── Slope limiters ───────────────────────────────────────────────

    def _face_deltas(self, grad):
        """Reconstructed increment at face center. Returns (delta_A, delta_B)."""
        dA = np.zeros(self.n_faces)
        dB = np.zeros(self.n_faces)
        for d in range(self.dim):
            dA += grad[d, self.iA] * self.r_Af[d, :]
            dB += grad[d, self.iB] * self.r_Bf[d, :]
        return dA, dB

    def _limit_bj(self, u, grad, u_min, u_max):
        """Barth-Jespersen limiter → φ ∈ [0, 1] per cell."""
        phi = np.ones(self.n_cells)
        eps = 1e-30
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps
            cand = np.ones(self.n_faces)
            cand[pos] = np.minimum(1.0, (u_max[cell_ids[pos]] - uc[pos]) / deltas[pos])
            cand[neg] = np.minimum(1.0, (u_min[cell_ids[neg]] - uc[neg]) / deltas[neg])
            np.minimum.at(phi, cell_ids, cand)

        return np.clip(phi, 0.0, 1.0)

    def _limit_vk(self, u, grad, u_min, u_max):
        """Venkatakrishnan limiter → φ ∈ [0, 1] per cell (smooth)."""
        phi = np.ones(self.n_cells)
        eps = 1e-30
        ev2 = self._eps_v2
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps

            if pos.any():
                dm = u_max[cell_ids[pos]] - uc[pos]
                df = deltas[pos]
                num = dm ** 2 + ev2[cell_ids[pos]] + 2 * df * dm
                den = dm ** 2 + 2 * df ** 2 + df * dm + ev2[cell_ids[pos]]
                np.minimum.at(phi, cell_ids[pos],
                              np.minimum(1.0, num / np.maximum(den, eps)))

            if neg.any():
                dm = u_min[cell_ids[neg]] - uc[neg]
                df = deltas[neg]
                num = dm ** 2 + ev2[cell_ids[neg]] + 2 * df * dm
                den = dm ** 2 + 2 * df ** 2 + df * dm + ev2[cell_ids[neg]]
                np.minimum.at(phi, cell_ids[neg],
                              np.minimum(1.0, num / np.maximum(den, eps)))

        return np.clip(phi, 0.0, 1.0)

    def _limit_minmod(self, u, grad, u_min, u_max):
        """Minmod limiter → φ ∈ [0, 1] per cell. Most diffusive TVD limiter."""
        phi = np.ones(self.n_cells)
        eps = 1e-30
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps

            # Upwind slope ratio r = (u_upwind - u_cell) / delta
            # minmod(1, r) = max(0, min(1, r))
            cand = np.ones(self.n_faces)
            if pos.any():
                # delta > 0 → the reconstruction goes up → compare against neighbor max
                r = (u_max[cell_ids[pos]] - uc[pos]) / deltas[pos]
                cand[pos] = np.maximum(0.0, np.minimum(1.0, r))
            if neg.any():
                r = (u_min[cell_ids[neg]] - uc[neg]) / deltas[neg]
                cand[neg] = np.maximum(0.0, np.minimum(1.0, r))

            np.minimum.at(phi, cell_ids, cand)

        return np.clip(phi, 0.0, 1.0)

    # ── Core pipeline ────────────────────────────────────────────────

    def _compute_limited_gradients(self, Q, n_vars):
        """Gradient + limiter for all variables. Override for wet-dry."""
        _limiter_map = {
            "barth_jespersen": self._limit_bj,
            "venkatakrishnan": self._limit_vk,
            "minmod": self._limit_minmod,
        }
        limiter_fn = _limiter_map[self._limiter_type]

        grads = np.zeros((n_vars, self.dim, self.n_cells))
        phi = np.ones((n_vars, self.n_cells))

        for v in range(n_vars):
            u = Q[v, :]
            grads[v] = self._green_gauss_gradient(u)
            u_min, u_max = self._neighbor_bounds(u)
            phi[v] = limiter_fn(u, grads[v], u_min, u_max)

        return grads, phi

    def _reconstruct(self, Q, grads, phi):
        """Linear reconstruction at face centers."""
        iA, iB = self.iA, self.iB
        Q_L = Q[:, iA].copy()
        Q_R = Q[:, iB].copy()
        for d in range(self.dim):
            Q_L += phi[:, iA] * grads[:, d, iA] * self.r_Af[d, :][np.newaxis, :]
            Q_R += phi[:, iB] * grads[:, d, iB] * self.r_Bf[d, :][np.newaxis, :]
        return Q_L, Q_R


class FreeSurfaceMUSCL(MUSCLReconstruction):
    """MUSCL with wet-dry fallback for free-surface flows.

    In dry cells (h < eps_wet), falls back to 1st order (φ = 0).
    Clamps h ≥ 0 at face states after reconstruction.

    Parameters
    ----------
    mesh, dim, limiter : same as MUSCLReconstruction
    h_index : int
        Index of water depth h in the Q array.
    eps_wet : float
        Dry-cell threshold.
    """

    def __init__(self, mesh, dim, h_index, eps_wet=1e-3, limiter="venkatakrishnan"):
        super().__init__(mesh, dim, limiter=limiter)
        self._h_idx = h_index
        self._eps_wet = eps_wet

    def __call__(self, Q):
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars)
        Q_L, Q_R = self._reconstruct(Q, grads, phi)
        # Clamp water depth at faces
        np.maximum(Q_L[self._h_idx, :], 0.0, out=Q_L[self._h_idx, :])
        np.maximum(Q_R[self._h_idx, :], 0.0, out=Q_R[self._h_idx, :])
        return Q_L, Q_R

    def _compute_limited_gradients(self, Q, n_vars):
        grads, phi = super()._compute_limited_gradients(Q, n_vars)
        dry = Q[self._h_idx, :] < self._eps_wet
        phi[:, dry] = 0.0
        return grads, phi


# ── Ghost-cell-free reconstruction classes ───────────────────────────────────
#
# These classes operate on Q of shape (n_vars, n_inner_cells) — no ghost cells.
# Boundary-face states come from BC.face_value(), passed in as bf_face_values.
# Loops are split into interior faces and boundary faces — no if-branches.


class ConstantReconstructionV2:
    """First-order piecewise-constant reconstruction (ghost-cell-free).

    Q has shape (n_vars, n_inner_cells). Boundary face values are supplied
    externally via ``bf_face_values``.
    """

    def __init__(self, mesh, dim):
        nc = mesh.n_inner_cells
        fc0 = mesh.face_cells[0]
        fc1 = mesh.face_cells[1]

        # Identify boundary faces via mesh metadata
        bf_face_set = set(mesh.boundary_face_face_indices)

        # Split faces into interior and boundary — no mixing
        self._interior_faces = np.array(
            [f for f in range(mesh.n_faces) if f not in bf_face_set], dtype=int)
        self._boundary_faces = mesh.boundary_face_face_indices.copy()
        self._iA_int = fc0[self._interior_faces]
        self._iB_int = fc1[self._interior_faces]
        # Inner cell at boundary faces: use face_cells[0] which is always the
        # THIS-side cell (BaseMesh.__init__ normalizes this). Do NOT use
        # mesh.boundary_face_cells which may be remapped for periodic BCs.
        self._iInner_bnd = mesh.face_cells[0, mesh.boundary_face_face_indices].copy()
        self._n_faces = mesh.n_faces

    def __call__(self, Q, bf_face_values=None):
        """Reconstruct face states.

        Parameters
        ----------
        Q : ndarray, shape (n_vars, n_inner_cells)
        bf_face_values : ndarray, shape (n_vars, n_boundary_faces)
            BC-provided boundary face values.

        Returns (Q_L, Q_R) each (n_vars, n_faces).
        """
        n_vars = Q.shape[0]
        Q_L = np.zeros((n_vars, self._n_faces))
        Q_R = np.zeros((n_vars, self._n_faces))

        # Interior faces: both sides from Q
        Q_L[:, self._interior_faces] = Q[:, self._iA_int]
        Q_R[:, self._interior_faces] = Q[:, self._iB_int]

        # Boundary faces: L = inner cell (face_cells[0]), R = BC value
        Q_L[:, self._boundary_faces] = Q[:, self._iInner_bnd]
        if bf_face_values is not None:
            Q_R[:, self._boundary_faces] = bf_face_values

        return Q_L, Q_R


class LSQMUSCLReconstruction:
    """Second-order MUSCL reconstruction with LSQ gradients (ghost-cell-free).

    Uses precomputed LSQ stencil from ``LSQMesh`` for gradient computation.
    Only interior cells participate in the LSQ stencil — boundary cells grow
    their stencil inward automatically.

    Q has shape (n_vars, n_inner_cells). Boundary face values are supplied
    externally. All loops are split into interior and boundary — no if-branches.

    Parameters
    ----------
    mesh : LSQMesh
    dim : int
    limiter : str
        "venkatakrishnan" (default), "barth_jespersen", or "minmod".
    """

    def __init__(self, mesh, dim, limiter="venkatakrishnan"):
        self.dim = dim
        self._limiter_type = limiter
        nc = mesh.n_inner_cells
        self._nc = nc
        self._n_faces = mesh.n_faces

        fc0 = mesh.face_cells[0]
        fc1 = mesh.face_cells[1]

        # Identify boundary faces via mesh metadata
        bf_face_set = set(mesh.boundary_face_face_indices)

        # Split faces into interior and boundary
        self._interior_faces = np.array(
            [f for f in range(mesh.n_faces) if f not in bf_face_set], dtype=int)
        self._boundary_faces = mesh.boundary_face_face_indices.copy()

        # Interior face connectivity (both cells are valid inner cells)
        self._iA_int = fc0[self._interior_faces]
        self._iB_int = fc1[self._interior_faces]

        # Inner cell at boundary faces: use face_cells[0] which is always the
        # THIS-side cell. Do NOT use mesh.boundary_face_cells which may be
        # remapped for periodic BCs (pointing to the opposite boundary).
        self._iInner_bnd = mesh.face_cells[0, mesh.boundary_face_face_indices].copy()

        # Cell→face displacement vectors for reconstruction extrapolation
        centers = mesh.cell_centers[:dim, :]
        face_ctrs = mesh.face_centers[:, :dim].T  # (dim, n_faces)

        # Interior faces: both A and B displacements
        self._r_Af_int = face_ctrs[:, self._interior_faces] - centers[:, self._iA_int]
        self._r_Bf_int = face_ctrs[:, self._interior_faces] - centers[:, self._iB_int]

        # Boundary faces: displacement from inner cell to face center
        self._r_Af_bnd = face_ctrs[:, self._boundary_faces] - centers[:, self._iInner_bnd]

        # Face normals and volumes for face_deltas (only interior + boundary A-side)
        self._normals_int = mesh.face_normals[:dim, self._interior_faces]
        self._fv_int = mesh.face_volumes[self._interior_faces]

        # Boundary face metadata (for limiter bounds)
        self._bf_cells = mesh.boundary_face_cells
        self._n_bf = mesh.n_boundary_faces

        # LSQ gradient data (precomputed by LSQMesh)
        self._lsq_gradQ = mesh.lsq_gradQ          # (n_cells, max_nbr, n_monomials)
        self._lsq_neighbors = mesh.lsq_neighbors    # (n_cells, max_nbr)
        self._lsq_scale = mesh.lsq_scale_factors    # (n_monomials,)

        # Venkatakrishnan smoothing: eps² = (K·h)²
        cell_vols = mesh.cell_volumes[:nc]
        h = cell_vols ** (1.0 / max(dim, 1))
        self._eps_v2 = h ** 2

    def __call__(self, Q, bf_face_values):
        """Reconstruct face states.

        Parameters
        ----------
        Q : ndarray, shape (n_vars, n_inner_cells)
        bf_face_values : ndarray, shape (n_vars, n_boundary_faces)
            BC-provided boundary face values (for limiter bounds and Q_R at boundary).

        Returns (Q_L, Q_R) each (n_vars, n_faces).
        Q_R at boundary faces is set to bf_face_values (placeholder; overwritten
        by face_value(Q_L) in the flux operator).
        """
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars, bf_face_values)
        return self._reconstruct(Q, grads, phi, bf_face_values)

    # ── LSQ gradient ────────────────────────────────────────────────

    def _lsq_gradient(self, u):
        """Cell-center gradient via precomputed LSQ stencil.

        Only interior cells are used in the stencil (guaranteed by LSQMesh).
        Returns (dim, nc).
        """
        nc = self._nc
        dim = self.dim
        grad = np.zeros((dim, nc))
        A_glob = self._lsq_gradQ
        neighbors = self._lsq_neighbors
        scale = self._lsq_scale

        for i in range(nc):
            A_loc = A_glob[i]                     # (max_nbr, n_monomials)
            nbr_idx = neighbors[i]                # (max_nbr,)
            delta_u = u[nbr_idx] - u[i]           # (max_nbr,)
            coeffs = scale * (A_loc.T @ delta_u)  # (n_monomials,)
            # First `dim` monomials are the gradient components for degree=1
            grad[:, i] = coeffs[:dim]

        return grad

    # ── Neighbor bounds (split loops) ───────────────────────────────

    def _neighbor_bounds(self, u, bf_values):
        """Per-cell min/max over face-neighbors.

        Two separate passes: interior faces (both neighbors valid),
        boundary faces (use BC-provided values).
        """
        u_max = u.copy()
        u_min = u.copy()

        # Pass 1: interior faces
        iA, iB = self._iA_int, self._iB_int
        np.maximum.at(u_max, iA, u[iB])
        np.maximum.at(u_max, iB, u[iA])
        np.minimum.at(u_min, iA, u[iB])
        np.minimum.at(u_min, iB, u[iA])

        # Pass 2: boundary faces — use BC face values
        bf_cells = self._bf_cells
        np.maximum.at(u_max, bf_cells, bf_values)
        np.minimum.at(u_min, bf_cells, bf_values)

        return u_min, u_max

    # ── Face deltas (split loops) ───────────────────────────────────

    def _face_deltas_interior(self, grad):
        """Reconstructed increment at interior face centers.

        Returns (delta_A, delta_B) each shape (n_interior_faces,).
        """
        dA = np.zeros(len(self._interior_faces))
        dB = np.zeros(len(self._interior_faces))
        for d in range(self.dim):
            dA += grad[d, self._iA_int] * self._r_Af_int[d]
            dB += grad[d, self._iB_int] * self._r_Bf_int[d]
        return dA, dB

    def _face_deltas_boundary(self, grad):
        """Reconstructed increment at boundary face centers (A-side only).

        Returns delta_A shape (n_boundary_faces,).
        """
        dA = np.zeros(len(self._boundary_faces))
        for d in range(self.dim):
            dA += grad[d, self._iInner_bnd] * self._r_Af_bnd[d]
        return dA

    # ── Slope limiters ──────────────────────────────────────────────

    def _limit_vk(self, u, grad, u_min, u_max):
        """Venkatakrishnan limiter (smooth)."""
        nc = self._nc
        phi = np.ones(nc)
        eps = 1e-30
        ev2 = self._eps_v2

        # Interior face deltas
        dA_int, dB_int = self._face_deltas_interior(grad)
        # Boundary face deltas (A-side only)
        dA_bnd = self._face_deltas_boundary(grad)

        def _apply_vk(cell_ids, deltas):
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps
            if pos.any():
                dm = u_max[cell_ids[pos]] - uc[pos]
                df = deltas[pos]
                num = dm ** 2 + ev2[cell_ids[pos]] + 2 * df * dm
                den = dm ** 2 + 2 * df ** 2 + df * dm + ev2[cell_ids[pos]]
                np.minimum.at(phi, cell_ids[pos],
                              np.minimum(1.0, num / np.maximum(den, eps)))
            if neg.any():
                dm = u_min[cell_ids[neg]] - uc[neg]
                df = deltas[neg]
                num = dm ** 2 + ev2[cell_ids[neg]] + 2 * df * dm
                den = dm ** 2 + 2 * df ** 2 + df * dm + ev2[cell_ids[neg]]
                np.minimum.at(phi, cell_ids[neg],
                              np.minimum(1.0, num / np.maximum(den, eps)))

        # Interior faces: constrain both A and B sides
        _apply_vk(self._iA_int, dA_int)
        _apply_vk(self._iB_int, dB_int)
        # Boundary faces: constrain A-side only
        _apply_vk(self._iInner_bnd, dA_bnd)

        return np.clip(phi, 0.0, 1.0)

    def _limit_bj(self, u, grad, u_min, u_max):
        """Barth-Jespersen limiter (strict DMP)."""
        nc = self._nc
        phi = np.ones(nc)
        eps = 1e-30

        dA_int, dB_int = self._face_deltas_interior(grad)
        dA_bnd = self._face_deltas_boundary(grad)

        def _apply_bj(cell_ids, deltas):
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps
            cand = np.ones(len(deltas))
            cand[pos] = np.minimum(1.0, (u_max[cell_ids[pos]] - uc[pos]) / deltas[pos])
            cand[neg] = np.minimum(1.0, (u_min[cell_ids[neg]] - uc[neg]) / deltas[neg])
            np.minimum.at(phi, cell_ids, cand)

        _apply_bj(self._iA_int, dA_int)
        _apply_bj(self._iB_int, dB_int)
        _apply_bj(self._iInner_bnd, dA_bnd)

        return np.clip(phi, 0.0, 1.0)

    def _limit_minmod(self, u, grad, u_min, u_max):
        """Minmod limiter (most diffusive TVD)."""
        nc = self._nc
        phi = np.ones(nc)
        eps = 1e-30

        dA_int, dB_int = self._face_deltas_interior(grad)
        dA_bnd = self._face_deltas_boundary(grad)

        def _apply_minmod(cell_ids, deltas):
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps
            cand = np.ones(len(deltas))
            if pos.any():
                r = (u_max[cell_ids[pos]] - uc[pos]) / deltas[pos]
                cand[pos] = np.maximum(0.0, np.minimum(1.0, r))
            if neg.any():
                r = (u_min[cell_ids[neg]] - uc[neg]) / deltas[neg]
                cand[neg] = np.maximum(0.0, np.minimum(1.0, r))
            np.minimum.at(phi, cell_ids, cand)

        _apply_minmod(self._iA_int, dA_int)
        _apply_minmod(self._iB_int, dB_int)
        _apply_minmod(self._iInner_bnd, dA_bnd)

        return np.clip(phi, 0.0, 1.0)

    # ── Core pipeline ───────────────────────────────────────────────

    def _compute_limited_gradients(self, Q, n_vars, bf_face_values):
        """LSQ gradient + slope limiter for all variables."""
        _limiter_map = {
            "barth_jespersen": self._limit_bj,
            "venkatakrishnan": self._limit_vk,
            "minmod": self._limit_minmod,
        }
        limiter_fn = _limiter_map[self._limiter_type]

        nc = self._nc
        grads = np.zeros((n_vars, self.dim, nc))
        phi = np.ones((n_vars, nc))

        for v in range(n_vars):
            u = Q[v, :]
            grads[v] = self._lsq_gradient(u)
            u_min, u_max = self._neighbor_bounds(u, bf_face_values[v, :])
            phi[v] = limiter_fn(u, grads[v], u_min, u_max)

        return grads, phi

    def _reconstruct(self, Q, grads, phi, bf_face_values):
        """Linear reconstruction at face centers (split loops)."""
        n_vars = Q.shape[0]
        Q_L = np.zeros((n_vars, self._n_faces))
        Q_R = np.zeros((n_vars, self._n_faces))

        # Interior faces: both sides reconstructed from Q
        Q_L[:, self._interior_faces] = Q[:, self._iA_int]
        Q_R[:, self._interior_faces] = Q[:, self._iB_int]
        for d in range(self.dim):
            Q_L[:, self._interior_faces] += (
                phi[:, self._iA_int] * grads[:, d, self._iA_int]
                * self._r_Af_int[d][np.newaxis, :])
            Q_R[:, self._interior_faces] += (
                phi[:, self._iB_int] * grads[:, d, self._iB_int]
                * self._r_Bf_int[d][np.newaxis, :])

        # Boundary faces: inner cell reconstructed, BC on the other side
        # Compute the reconstructed inner state at each boundary face
        Q_inner_recon = Q[:, self._iInner_bnd].copy()
        for d in range(self.dim):
            Q_inner_recon += (
                phi[:, self._iInner_bnd] * grads[:, d, self._iInner_bnd]
                * self._r_Af_bnd[d][np.newaxis, :])

        # L = reconstructed inner cell, R = BC value
        Q_L[:, self._boundary_faces] = Q_inner_recon
        Q_R[:, self._boundary_faces] = bf_face_values

        return Q_L, Q_R


class FreeSurfaceLSQMUSCL(LSQMUSCLReconstruction):
    """LSQ MUSCL with wet-dry fallback for free-surface flows (ghost-cell-free).

    In dry cells (h < eps_wet), falls back to 1st order (phi = 0).
    Clamps h >= 0 at face states after reconstruction.
    """

    def __init__(self, mesh, dim, h_index, eps_wet=1e-3, limiter="venkatakrishnan"):
        super().__init__(mesh, dim, limiter=limiter)
        self._h_idx = h_index
        self._eps_wet = eps_wet

    def __call__(self, Q, bf_face_values):
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars, bf_face_values)
        # Wet-dry: zero limiter in dry cells
        dry = Q[self._h_idx, :] < self._eps_wet
        phi[:, dry] = 0.0
        Q_L, Q_R = self._reconstruct(Q, grads, phi, bf_face_values)
        # Clamp h >= 0
        np.maximum(Q_L[self._h_idx, :], 0.0, out=Q_L[self._h_idx, :])
        np.maximum(Q_R[self._h_idx, :], 0.0, out=Q_R[self._h_idx, :])
        return Q_L, Q_R


# ── Diffusion operators ──────────────────────────────────────────────────────

class FaceGradient:
    """Precomputes face geometry and provides face gradient reconstruction.

    This class is instantiated once per mesh and reused across timesteps.
    """

    def __init__(self, mesh, dim):
        self.mesh = mesh
        self.dim = dim
        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        centers = mesh.cell_centers[:dim, :]
        normals = mesh.face_normals[:dim, :]

        self.iA = iA
        self.iB = iB
        self.n_faces = n_faces
        self.n_cells = n_cells
        self.nc = nc

        self.d_AB = np.zeros(n_faces)
        self.e_AB = np.zeros((dim, n_faces))

        for f in range(n_faces):
            dx = centers[:, iB[f]] - centers[:, iA[f]]
            dist = np.linalg.norm(dx)
            self.d_AB[f] = max(dist, 1e-30)
            self.e_AB[:, f] = dx / self.d_AB[f]

        self.n_dot_e = np.zeros(n_faces)
        for f in range(n_faces):
            self.n_dot_e[f] = np.dot(normals[:, f], self.e_AB[:, f])
            self.n_dot_e[f] = max(abs(self.n_dot_e[f]), 0.1) * np.sign(self.n_dot_e[f] + 1e-30)

    def compute_cell_gradients(self, u, cell_volumes, face_volumes, normals):
        """Green-Gauss cell-center gradients."""
        dim = self.dim
        grad = np.zeros((dim, self.n_cells))
        for f in range(self.n_faces):
            a, b = self.iA[f], self.iB[f]
            u_face = 0.5 * (u[a] + u[b])
            for d in range(dim):
                contrib = u_face * normals[d, f] * face_volumes[f]
                grad[d, a] += contrib / cell_volumes[a]
                grad[d, b] -= contrib / cell_volumes[b]
        return grad

    def face_normal_gradient(self, u):
        """Corrected face-normal gradient: (u_B - u_A) / d_AB / (n·e).

        Returns shape (n_faces,).
        """
        grad_n = np.zeros(self.n_faces)
        for f in range(self.n_faces):
            grad_n[f] = (u[self.iB[f]] - u[self.iA[f]]) / self.d_AB[f] / self.n_dot_e[f]
        return grad_n


class DiffusionOperator:
    """Sparse discrete diffusion operator: L(u) = nabla·(nu nabla u).

    Assembled once per mesh + viscosity. Can be used for:
    - Explicit diffusion: dQ += dt * L @ Q
    - Implicit diffusion: solve (I - dt * L) @ Q^{n+1} = Q^*
    """

    def __init__(self, mesh, dim, nu=1.0):
        self.mesh = mesh
        self.dim = dim
        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        centers = mesh.cell_centers[:dim, :]
        normals = mesh.face_normals[:dim, :]
        face_vol = mesh.face_volumes
        cell_vol = mesh.cell_volumes

        L = lil_matrix((nc, nc), dtype=float)

        for f in range(mesh.n_faces):
            a, b = iA[f], iB[f]
            dx = centers[:, b] - centers[:, a]
            dist = np.linalg.norm(dx)
            if dist < 1e-30:
                continue

            n = normals[:, f]
            n_dot_e = np.dot(n, dx / dist)
            n_dot_e = max(abs(n_dot_e), 0.1) * np.sign(n_dot_e + 1e-30)
            coeff = nu * face_vol[f] / dist / abs(n_dot_e)

            if a < nc and b < nc:
                L[a, b] += coeff / cell_vol[a]
                L[a, a] -= coeff / cell_vol[a]
                L[b, a] += coeff / cell_vol[b]
                L[b, b] -= coeff / cell_vol[b]
            elif a < nc:
                pass
            elif b < nc:
                pass

        self.L = csr_matrix(L)
        self.nc = nc
        self.n_cells = n_cells

    def explicit(self, u):
        """Compute L @ u (for explicit stepping). Returns shape (nc,)."""
        return self.L @ u[:self.nc]

    def implicit_solve(self, u_star, dt, tol=1e-8, maxiter=100):
        """Crank-Nicolson: (I - dt/2 * L) u^{n+1} = (I + dt/2 * L) u*.

        Second-order in time for diffusion (backward Euler was first-order).
        """
        nc = self.nc
        rhs = u_star[:nc] + 0.5 * dt * (self.L @ u_star[:nc])

        def matvec(x):
            return x - 0.5 * dt * (self.L @ x)

        A = LinearOperator((nc, nc), matvec=matvec, dtype=float)
        sol, info = gmres(A, rhs, x0=u_star[:nc], atol=0.0, rtol=tol, maxiter=maxiter)

        result = np.zeros(self.n_cells)
        result[:nc] = sol
        mesh = self.mesh
        for i_bf in range(mesh.n_boundary_faces):
            ghost = mesh.boundary_face_ghosts[i_bf]
            inner = mesh.boundary_face_cells[i_bf]
            result[ghost] = result[inner]
        return result


class DiffusionOperatorV2:
    """Sparse discrete diffusion with boundary face contributions (ghost-cell-free).

    Interior faces are assembled into the sparse matrix L (same as DiffusionOperator).
    Boundary faces are stored separately — their contributions come from
    ``BC.face_gradient()`` and are added as an explicit RHS correction.

    Parameters
    ----------
    mesh : FVMMesh or LSQMesh
    dim : int
    nu : float
        Kinematic viscosity.
    """

    def __init__(self, mesh, dim, nu=1.0):
        nc = mesh.n_inner_cells
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        centers = mesh.cell_centers[:dim, :]
        normals = mesh.face_normals[:dim, :]
        face_vol = mesh.face_volumes
        cell_vol = mesh.cell_volumes

        L = lil_matrix((nc, nc), dtype=float)

        # Interior faces only
        for f in range(mesh.n_faces):
            a, b = iA[f], iB[f]
            if not (a < nc and b < nc):
                continue
            dx = centers[:, b] - centers[:, a]
            dist = np.linalg.norm(dx)
            if dist < 1e-30:
                continue
            n = normals[:, f]
            n_dot_e = np.dot(n, dx / dist)
            n_dot_e = max(abs(n_dot_e), 0.1) * np.sign(n_dot_e + 1e-30)
            coeff = nu * face_vol[f] / dist / abs(n_dot_e)
            L[a, b] += coeff / cell_vol[a]
            L[a, a] -= coeff / cell_vol[a]
            L[b, a] += coeff / cell_vol[b]
            L[b, b] -= coeff / cell_vol[b]

        self.L = csr_matrix(L)
        self.nc = nc
        self._tol = 1e-8

        # Boundary face data: (inner_cell, coefficient, dist)
        # coefficient = nu * face_vol / dist / |n_dot_e| / cell_vol
        self._bf_data = []
        for i_bf in range(mesh.n_boundary_faces):
            fidx = mesh.boundary_face_face_indices[i_bf]
            inner = mesh.boundary_face_cells[i_bf]
            a, b = iA[fidx], iB[fidx]
            dx = centers[:, b] - centers[:, a]
            dist = np.linalg.norm(dx)
            if dist < 1e-30:
                self._bf_data.append((inner, 0.0, dist))
                continue
            n = normals[:, fidx]
            n_dot_e = np.dot(n, dx / dist)
            n_dot_e = max(abs(n_dot_e), 0.1) * np.sign(n_dot_e + 1e-30)
            coeff = nu * face_vol[fidx] / abs(n_dot_e) / cell_vol[inner]
            self._bf_data.append((inner, coeff, dist))

    def explicit(self, u):
        """Interior-only diffusion: L @ u. Returns shape (nc,)."""
        return self.L @ u[:self.nc]

    def explicit_with_bc(self, u, bf_face_normal_grads):
        """Diffusion with boundary face contributions.

        Parameters
        ----------
        u : ndarray, shape (nc,)
            Scalar field on inner cells.
        bf_face_normal_grads : ndarray, shape (n_boundary_faces,)
            Face-normal gradient dQ/dn at each boundary face,
            from ``BC.face_gradient()``.

        Returns
        -------
        ndarray, shape (nc,)
        """
        Lu = self.L @ u[:self.nc]
        for i_bf, (inner, coeff, dist) in enumerate(self._bf_data):
            # Diffusive flux contribution: coeff * (du/dn) * dist
            # coeff = nu * fv / |n_dot_e| / cv, times dist gives
            # nu * fv * dist / |n_dot_e| / cv * (du/dn)
            # But face_gradient already returns (Q_face - Q_inner) / d_face,
            # and the standard FVM contribution is nu * (Q_B - Q_A) / dist * fv / cv / |n_dot_e|
            # = coeff * (Q_B - Q_A) = coeff * face_gradient * dist
            # Wait — let me re-derive:
            # Interior: contribution = nu * fv * (u_B - u_A) / dist / |n_dot_e| / cv
            #         = (nu * fv / |n_dot_e| / cv) * (u_B - u_A) / dist
            #         = coeff * (u_B - u_A) / dist
            # But coeff already includes / dist? No:
            # Original code: coeff = nu * face_vol[f] / dist / abs(n_dot_e)
            # then: L[a,b] += coeff / cell_vol[a], L[a,a] -= coeff / cell_vol[a]
            # So L[a,b] = nu * fv / dist / |n_dot_e| / cv
            # L @ u gives: sum_j L[i,j] * u[j]
            # For face f: L[a,b]*u[b] + L[a,a]*u[a] = coeff/cv * (u[b] - u[a])
            # = nu * fv / dist / |n_dot_e| / cv * (u_B - u_A)
            #
            # For boundary: same formula but u_B is not a DOF.
            # contribution = nu * fv / dist / |n_dot_e| / cv * (u_face - u_inner)
            # face_gradient = (u_face - u_inner) / d_face
            # So contribution = (nu * fv / |n_dot_e| / cv) * face_gradient
            # Our stored coeff = nu * fv / |n_dot_e| / cv  (no /dist this time)
            Lu[inner] += coeff * bf_face_normal_grads[i_bf]
        return Lu

    def implicit_solve_with_bc(self, u_star, dt, bf_face_normal_grads,
                               tol=1e-8, maxiter=100):
        """Crank-Nicolson with boundary contributions in RHS.

        Boundary terms are treated explicitly (in the RHS), not in the operator.

        Parameters
        ----------
        u_star : ndarray, shape (nc,) or larger (only [:nc] used)
            State after explicit step.
        dt : float
        bf_face_normal_grads : ndarray, shape (n_boundary_faces,)
            From ``BC.face_gradient()``.

        Returns
        -------
        ndarray, shape (nc,)
        """
        nc = self.nc
        u = u_star[:nc]

        # Boundary flux contribution (explicit)
        bf_contrib = np.zeros(nc)
        for i_bf, (inner, coeff, dist) in enumerate(self._bf_data):
            bf_contrib[inner] += coeff * bf_face_normal_grads[i_bf]

        # RHS: (I + dt/2 * L) u* + dt/2 * bf_contrib
        rhs = u + 0.5 * dt * (self.L @ u + bf_contrib)

        def matvec(x):
            return x - 0.5 * dt * (self.L @ x)

        A = LinearOperator((nc, nc), matvec=matvec, dtype=float)
        sol, info = gmres(A, rhs, x0=u, atol=0.0, rtol=tol, maxiter=maxiter)

        return sol
