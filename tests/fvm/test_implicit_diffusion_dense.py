r"""Dense, state-dependent implicit diffusion (:class:`DenseDiffusionOperator`).

The IMEX implicit-diffusion step now assembles + solves the model's GENERAL
rank-4 ``diffusion_matrix`` ``A(Q, Qaux, p)`` — the diffusive flux
``F_diff[i,d] = Σ_{j,e} A[i,j,d,e] ∂_e Q[j]`` — instead of a scalar ``ν`` ×
per-variable Laplacian.  Three properties are pinned here:

1. **Scalar-ν regression** — a constant diagonal ``A = ν·I·δ_de`` reproduces the
   existing :class:`DiffusionOperatorV2` operator AND its Crank–Nicolson implicit
   update bit-for-bit.
2. **Dense consumption** — a non-zero off-diagonal ``A[0,1]`` cross-couples the
   variables (result differs from diagonal-only) and matches an independent
   two-point ``∇·(A:∇Q)``.
3. **State dependence** — an ``A`` that depends on ``Q`` drives a Newton solve
   whose Crank–Nicolson residual converges to ~0.

Two-point (normal-gradient) contraction: ``T[i,j] = Σ_{d,e} A[i,j,d,e] n_d n_e``,
face-averaged — this is the numpy reference the jax mirror must match.
"""
import numpy as np
import pytest

from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.fvm.reconstruction import (
    DiffusionOperatorV2, DenseDiffusionOperator,
)


@pytest.fixture
def mesh():
    return FVMMesh.create_2d((0.0, 1.0, 0.0, 1.0), 6, 5)


def _const_tensor(A_slab, nc):
    """Broadcast a constant ``(n_eq, n_st, dim, dim)`` tensor to per-cell."""
    return np.broadcast_to(A_slab[:, :, None, :, :],
                           A_slab.shape[:2] + (nc,) + A_slab.shape[2:])


def _independent_divergence(op, Q, A_cells, bf_grads=None):
    """Reference ``∇·(A:∇Q)`` computed by an explicit per-face loop — an
    independent re-derivation of ``DenseDiffusionOperator._divergence``."""
    n_eq = A_cells.shape[0]
    D = np.zeros((n_eq, op.nc))
    for f in range(op._ia.size):
        a, b = op._ia[f], op._ib[f]
        n = op._n_int[:, f]
        Aface = 0.5 * (A_cells[:, :, a] + A_cells[:, :, b])
        T = np.einsum('ijde,d,e->ij', Aface, n, n)
        flux = T @ (Q[:, b] - Q[:, a])
        D[:, a] += flux * op._gA[f]
        D[:, b] -= flux * op._gB[f]
    if bf_grads is not None:
        for k in range(op._bf_inner.size):
            inn = op._bf_inner[k]
            n = op._n_bf[:, k]
            T = np.einsum('ijde,d,e->ij', A_cells[:, :, inn], n, n)
            D[:, inn] += (T @ bf_grads[:, k]) * op._bf_g[k]
    return D


def test_scalar_nu_regression(mesh):
    """Constant diagonal ``A = ν·I`` ⇒ operator and CN update match
    :class:`DiffusionOperatorV2` bit-for-bit."""
    dim, nc, nbf = mesh.dimension, mesh.n_inner_cells, mesh.n_boundary_faces
    nu = 0.73
    rng = np.random.default_rng(0)
    u = rng.standard_normal(nc)
    bf_grad = rng.standard_normal(nbf)

    v2 = DiffusionOperatorV2(mesh, dim, nu=nu)
    dense = DenseDiffusionOperator(mesh, dim, n_vars=1, state_dependent=False)
    A_slab = np.zeros((1, 1, dim, dim))
    for d in range(dim):
        A_slab[0, 0, d, d] = nu
    A_cells = _const_tensor(A_slab, nc)

    # Interior operator: L @ u.
    assert np.allclose(dense._divergence(u[None], A_cells)[0],
                       v2.explicit(u), atol=1e-12)
    # With boundary face gradients.
    assert np.allclose(dense._divergence(u[None], A_cells, bf_grads=bf_grad[None])[0],
                       v2.explicit_with_bc(u, bf_grad), atol=1e-12)

    # Full Crank–Nicolson implicit update (both solved tightly).
    dt = 0.04
    ref = v2.implicit_solve_with_bc(u, dt, bf_grad, tol=1e-13, maxiter=800)
    got = dense.implicit_solve(u[None], dt, lambda Qs: A_cells,
                               bf_grad[None], tol=1e-13, maxiter=800)[0]
    assert np.allclose(got, ref, atol=1e-10)


def test_dense_off_diagonal_cross_couples(mesh):
    """A non-zero ``A[0,1]`` couples variable 1 into equation 0's diffusion,
    changing the result and matching an independent hand-computed divergence."""
    dim, nc = mesh.dimension, mesh.n_inner_cells
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((2, nc))
    op = DenseDiffusionOperator(mesh, dim, n_vars=2)

    Adiag = np.zeros((2, 2, dim, dim))
    for d in range(dim):
        Adiag[0, 0, d, d] = 1.0
        Adiag[1, 1, d, d] = 1.0
    Aoff = Adiag.copy()
    for d in range(dim):
        Aoff[0, 1, d, d] = 0.5           # eq 0 diffuses along ∇(variable 1)

    Adiag_c = _const_tensor(Adiag, nc)
    Aoff_c = _const_tensor(Aoff, nc)
    D_diag = op._divergence(Q, Adiag_c)
    D_off = op._divergence(Q, Aoff_c)

    # Off-diagonal genuinely changes the update (cross-coupling is live).
    assert not np.allclose(D_diag, D_off)
    # Matches an independent two-point ∇·(A:∇Q).
    assert np.allclose(D_off, _independent_divergence(op, Q, Aoff_c), atol=1e-12)
    # The extra term in equation 0 is exactly 0.5 · Laplacian(variable 1).
    A01 = np.zeros((2, 2, dim, dim))
    for d in range(dim):
        A01[0, 1, d, d] = 1.0
    lap_Q1 = _independent_divergence(op, Q, _const_tensor(A01, nc))
    assert np.allclose(D_off[0] - D_diag[0], 0.5 * lap_Q1[0], atol=1e-12)


def test_state_dependent_newton_residual(mesh):
    """A state-dependent ``A(Q)`` drives Newton; the Crank–Nicolson residual
    ``Q^{n+1}−Q* − dt/2(𝒟(Q^{n+1})+𝒟(Q*))`` converges to ~0."""
    dim, nc, nbf = mesh.dimension, mesh.n_inner_cells, mesh.n_boundary_faces
    rng = np.random.default_rng(2)
    Q_star = 0.4 * rng.standard_normal((2, nc))
    bf_grad = 0.1 * rng.standard_normal((2, nbf))
    op = DenseDiffusionOperator(mesh, dim, n_vars=2, state_dependent=True)

    def A_fn(Qs):
        A = np.zeros((2, 2, Qs.shape[1], dim, dim))
        for d in range(dim):
            A[0, 0, :, d, d] = 1.0 + 0.3 * Qs[0] ** 2   # nonlinear in Q
            A[1, 1, :, d, d] = 0.8
            A[0, 1, :, d, d] = 0.2 * np.tanh(Qs[1])     # dense + state-dep
        return A

    dt = 0.02
    Q_np1 = op.implicit_solve(Q_star, dt, A_fn, bf_grad,
                              tol=1e-11, maxiter=500,
                              newton_maxiter=20, newton_tol=1e-11)
    assert op.residual_norm(Q_np1, Q_star, dt, A_fn, bf_grad) < 1e-8
    # The solve actually moved the state (diffusion acted).
    assert not np.allclose(Q_np1, Q_star)
