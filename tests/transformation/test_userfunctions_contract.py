"""REQ-168: the numpy UserFunctions table must supply EVERY backend-supplied
kernel core declares in ``kernel_functions.REQUIRED_KERNELS``.

This is numpy's copy of the per-backend contract test (foam / dmplex / jax ship
their own): a new opaque kernel added to the registry goes RED here instead of
surfacing as a silent ``NameError`` at lambdify time.  ``compute_derivative`` is
present with a ``None`` placeholder — the solver injects the mesh-bound impl.
"""
import numpy as np

from zoomy_core.model.kernel_functions import REQUIRED_KERNELS
from zoomy_core.fvm import userfunctions as uf


def test_numpy_table_covers_every_required_kernel():
    table = uf.numpy_module()
    missing = REQUIRED_KERNELS - set(table)
    assert not missing, f"numpy UserFunctions missing kernels: {sorted(missing)}"


def test_conditional_is_not_in_the_required_contract():
    # printer-lowered to np.where — must NOT be demanded of any backend table
    assert "conditional" not in REQUIRED_KERNELS


def test_eigenvalues_kernel_matches_numpy_eig():
    # 2x2 SWE-like A_n at rest h=2: [[0,1],[g*h,0]] -> lambda = +-sqrt(g*h)
    a = [0.0, 1.0, 19.62, 0.0]
    ev = sorted(abs(uf.eigenvalues(i, *a)) for i in range(2))
    assert np.allclose(ev, [np.sqrt(19.62), np.sqrt(19.62)])


def test_eigenvalues_kernel_is_batched():
    # each flat entry a (n_cells,) row -> per-cell spectral radius
    g_h = np.array([19.62, 9.81, 4.905])
    a = [np.zeros(3), np.ones(3), g_h, np.zeros(3)]
    got = np.maximum(np.abs(uf.eigenvalues(0, *a)), np.abs(uf.eigenvalues(1, *a)))
    assert np.allclose(got, np.sqrt(g_h))


def test_eigensystem_reconstructs_the_matrix():
    # |A| = R |Lambda| L must reproduce A from ONE shared eigenbasis (cache).
    a = [0.0, 1.0, 19.62, 0.0]
    n = 2
    stack = np.array([uf.eigensystem(i, *a) for i in range(n + 2 * n * n)])
    lam = stack[:n]
    R = stack[n:n + n * n].reshape(n, n)
    L = stack[n + n * n:].reshape(n, n)
    A = np.array(a, float).reshape(n, n)
    assert np.allclose(R @ np.diag(lam) @ L, A, atol=1e-10)


def test_solve_kernel_matches_linalg_solve():
    # A x = b, 2x2:  A = [[2,1],[1,3]], b = [1, 2]
    args = [2.0, 1.0, 1.0, 3.0, 1.0, 2.0]
    x = np.array([uf.solve(0, *args), uf.solve(1, *args)]).ravel()
    assert np.allclose(x, np.linalg.solve([[2, 1], [1, 3]], [1, 2]))
