"""REQ-168 ADDENDA 1/2 — the opaque wave-speed kernels MUST be inf-tolerant.

LAPACK raises ``LinAlgError`` on non-finite input, but the order-2 MOOD path
feeds not-yet-corrected candidate face states BY DESIGN (non-finite at a dry
front) and transient non-finite BC-ghost states reach the kernel at order 1
too.  Contract: eig the finite batch members, **+inf** eigenvalues for the
rest — an infinite wave speed is what a garbage face state should report (dt
clamps, MOOD flags).  ``eigensystem`` additionally returns an identity
eigenbasis (R = L = I) for the flagged members.

The 1-slot caches are ID-keyed: freeing one test's argument arrays lets the
next test's fresh arrays reuse the same ids and silently hit the stale cache
entry.  ``_keep`` pins every argument pack for the module's lifetime so ids
are never recycled.
"""
import numpy as np

from zoomy_core.fvm import userfunctions as uf

_KEEP = []


def _keep(arrs):
    _KEEP.append(arrs)
    return arrs


def _swe_jacobian(u, c):
    """SWE quasilinear matrix [[0, 1], [c²−u², 2u]] — known spectrum u∓c."""
    return [0.0, 1.0, c * c - u * u, 2.0 * u]


def test_eigenvalues_finite_input_unchanged():
    a = _keep([np.array(x) for x in _swe_jacobian(2.0, 3.0)])
    lams = sorted(float(uf.eigenvalues(i, *a)) for i in range(2))
    assert np.allclose(lams, [-1.0, 5.0])


def test_eigenvalues_nonfinite_returns_inf_not_raise():
    a = _keep([np.array(x) for x in [np.inf, 1.0, 5.0, 4.0]])
    lams = [float(uf.eigenvalues(i, *a)) for i in range(2)]
    assert all(np.isposinf(lams))


def test_eigenvalues_mixed_batch_masks_per_member():
    # Two cells: cell 0 finite (u=2, c=3 → λ = −1, 5), cell 1 poisoned.
    a = _keep([np.array([x, np.nan]) for x in _swe_jacobian(2.0, 3.0)])
    lam0 = np.array([float(np.asarray(uf.eigenvalues(i, *a))[0])
                     for i in range(2)])
    a2 = _keep([np.array([x, np.nan]) for x in _swe_jacobian(2.0, 3.0)])
    lam1 = np.array([float(np.asarray(uf.eigenvalues(i, *a2))[1])
                     for i in range(2)])
    assert np.allclose(sorted(lam0), [-1.0, 5.0])
    assert np.isposinf(lam1).all()


def test_eigensystem_nonfinite_identity_basis():
    n = 2
    a = _keep([np.array(x) for x in [np.inf, 0.0, 0.0, np.inf]])
    out = np.array([float(uf.eigensystem(i, *a)) for i in range(n + 2 * n * n)])
    lam, R, L = out[:n], out[n:n + n * n].reshape(n, n), out[n + n * n:].reshape(n, n)
    assert np.isposinf(lam).all()
    assert np.array_equal(R, np.eye(n))
    assert np.array_equal(L, np.eye(n))


def test_eigensystem_finite_input_unchanged():
    n = 2
    a = _keep([np.array(x) for x in _swe_jacobian(2.0, 3.0)])
    out = np.array([float(uf.eigensystem(i, *a)) for i in range(n + 2 * n * n)])
    lam, R, L = out[:n], out[n:n + n * n].reshape(n, n), out[n + n * n:].reshape(n, n)
    A = np.array(_swe_jacobian(2.0, 3.0)).reshape(n, n)
    assert np.allclose(sorted(lam), [-1.0, 5.0])
    assert np.allclose(R @ np.diag(lam) @ L, A, atol=1e-12)
    assert np.allclose(L @ R, np.eye(n), atol=1e-12)


def test_cache_pins_argument_refs():
    """REQ-168 ADDENDUM 3: the 1-slot caches hold STRONG refs to the argument
    arrays, so a gc'd pack's ids can never be recycled into a stale hit."""
    a = [np.array(x) for x in _swe_jacobian(1.0, 2.0)]
    uf.eigenvalues(0, *a)
    assert all(x is y for x, y in zip(uf._EIGENVALUES_CACHE["args"], a))
    uf.eigensystem(0, *a)
    assert all(x is y for x, y in zip(uf._EIGENSYSTEM_CACHE["args"], a))
