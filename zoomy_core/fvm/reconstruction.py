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
        """Solve (I - dt * L) @ u = u_star via GMRES."""
        nc = self.nc
        rhs = u_star[:nc]

        def matvec(x):
            return x - dt * (self.L @ x)

        A = LinearOperator((nc, nc), matvec=matvec, dtype=float)
        sol, info = gmres(A, rhs, x0=rhs, atol=0.0, rtol=tol, maxiter=maxiter)

        result = np.zeros(self.n_cells)
        result[:nc] = sol
        mesh = self.mesh
        for i_bf in range(mesh.n_boundary_faces):
            ghost = mesh.boundary_face_ghosts[i_bf]
            inner = mesh.boundary_face_cells[i_bf]
            result[ghost] = result[inner]
        return result
