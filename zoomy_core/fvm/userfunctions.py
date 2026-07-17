"""NumPy backend **UserFunctions** — the concrete implementations of the opaque
:mod:`zoomy_core.model.kernel_functions` kernels for the numpy runtime.

This is numpy's ``UserFunctions`` table, the python mirror of the C++ backends'
``UserFunctions.H`` (REQ-168).  Every backend-supplied kernel named in
:data:`zoomy_core.model.kernel_functions.REQUIRED_KERNELS` has an entry in
:data:`KERNELS` here, so a missing kernel is a caught omission (see the contract
test) rather than a silent ``NameError`` at lambdify time.  ``to_numpy.module``
is built from these tables via :func:`numpy_module`.

Kept a leaf module (imports only numpy) so ``to_numpy`` can import it without a
``fvm`` ↔ ``transformation`` cycle.

The 1-slot caches share ONE decomposition/solve across the many component calls
lambdify emits for a single evaluation: the printer turns the ``n + 2n²``
component reads of one face into ``n + 2n²`` separate ``eigensystem(i, …)``
calls with the SAME argument objects (hence same ``id``), and likewise ``n``
``solve(i, …)`` calls.  R, |Λ| and L MUST come from the same eigenbasis or
``|A| = R|Λ|L`` is silently inconsistent — the cache is a CORRECTNESS
requirement, not just an optimization.  Valid only WITHIN one lambdified
evaluation (same argument objects); a fresh evaluation gets fresh arrays.
"""

import numpy as np

_EIGENSYSTEM_CACHE = {"key": None, "out": None}
_EIGENVALUES_CACHE = {"key": None, "out": None}
_SOLVE_CACHE = {"key": None, "out": None}


def _stack_square(a_flat, n):
    """Broadcast the flat row-major args into an ``(..., n, n)`` array.

    Each arg is a scalar or a ``(n_cells,)`` grid row; ``np.broadcast_arrays``
    lifts scalars to the common grid shape so the eig / solve is batched over
    every cell at once.  Scalar-only inputs give a ``(n, n)`` matrix (0-d
    broadcast), so the scalar paths still work."""
    bcast = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in a_flat])
    return np.stack(bcast, axis=-1).reshape(bcast[0].shape + (n, n))


def _scalarize(x):
    """Return a python float for a 0-d result (scalar-input path), else the
    array — so a scalar evaluation matches the old ``float(...)`` return."""
    return float(x) if np.ndim(x) == 0 else x


# ── compute-once pack helpers (REQ-179) ──────────────────────────────────────
# The per-component ``eigensystem(idx, *A_flat)`` calling convention forces the
# printer to inline the full ``A_flat`` argument list into EACH of the ``n+2n²``
# component calls — 93% of the lowered eigensystem tree and the ~20-min
# single-core lambdify at ML-FullVAM scale (fully-inlined source is 10-15×
# larger and compiles 13× slower — REQ-179 measurements).  The numpy printer
# (:meth:`to_numpy.NumpyRuntimeModel._lower_opaque_kernels`) instead rewrites
# each ``K(idx, *A_flat)`` into ``pick(K_pack(*A_flat), idx)``.  Every component
# of one face shares the SAME ``A_flat`` objects, so all ``K_pack(*A_flat)`` are
# ONE sympy node — cse hoists it to a single temp, emitting ``A_flat`` ONCE.
# The ``*_pack`` bodies are the exact (inf-guarded) numerics the cached kernels
# below use, so ``pick(K_pack(a), i)`` is bit-identical to ``K(i, *a)``.

def _eigensystem_pack(a_flat):
    """Full ``[λ(n), R(n·n), L=R⁻¹(n·n)]`` stack of the row-major ``n×n`` matrix
    ``a_flat`` (``n = round(√len)``), batched over the grid.  REQ-168 inf-guard:
    LAPACK raises on non-finite input, so non-finite batch members get λ = +inf
    with an identity eigenbasis (R = L = I) — the +inf wave speed flags the
    garbage state; the identity basis keeps ``|A| = R|Λ|L`` well-defined without
    inventing an inverse of a garbage matrix."""
    n = int(round(len(a_flat) ** 0.5))
    A = _stack_square(a_flat, n)
    flat = A.reshape(-1, n, n)
    m = flat.shape[0]
    ok = np.isfinite(flat).all(axis=(1, 2))
    w = np.full((m, n), np.inf)
    V = np.broadcast_to(np.eye(n), (m, n, n)).copy()
    L = V.copy()
    if ok.any():
        wk, Vk = np.linalg.eig(flat[ok])
        V[ok] = np.real(Vk)
        w[ok] = np.real(wk)
        try:
            L[ok] = np.linalg.inv(V[ok])
        except np.linalg.LinAlgError:
            L[ok] = np.linalg.pinv(V[ok])
    return np.concatenate(
        [w, V.reshape(m, n * n), L.reshape(m, n * n)],
        axis=-1).reshape(A.shape[:-2] + (n + 2 * n * n,))


def _eigenvalues_pack(a_flat):
    """λ-only spectrum (n,) of the row-major ``n×n`` matrix ``a_flat``, batched
    over the grid.  REQ-168 inf-guard: eig the finite batch members, +inf for
    the rest (order-2 MOOD / transient BC-ghost feed non-finite candidate face
    states BY DESIGN — an infinite wave speed is exactly what a garbage face
    should report; dt clamps, MOOD flags the candidate)."""
    n = int(round(len(a_flat) ** 0.5))
    A = _stack_square(a_flat, n)
    flat = A.reshape(-1, n, n)
    ok = np.isfinite(flat).all(axis=(1, 2))
    out = np.full((flat.shape[0], n), np.inf)
    if ok.any():
        out[ok] = np.real(np.linalg.eigvals(flat[ok]))
    return out.reshape(A.shape[:-2] + (n,))


def _solve_pack(args):
    """Per-cell linear solve ``A⁻¹ b`` (ncells, n).  ``args`` = row-major ``A``
    (n·n) followed by ``b`` (n); ``n`` inferred from the count ``n·n + n``."""
    m = len(args)
    n = int(round((-1.0 + (1.0 + 4.0 * m) ** 0.5) / 2.0))
    arrs = [np.asarray(a, dtype=float) for a in args]
    ncells = max((a.shape[0] for a in arrs if a.ndim > 0), default=1)

    def _col(a):
        if a.ndim > 0 and a.shape[0] == ncells:
            return a
        return np.full(ncells, float(a))

    cols = [_col(a) for a in arrs]
    A = np.stack(cols[:n * n], axis=-1).reshape(ncells, n, n)
    # Column RHS ``(ncells, n, 1)`` — NumPy 2.0 reads a 2-D ``b`` as a matrix,
    # so the vector RHS must be an explicit column.
    b = np.stack(cols[n * n:], axis=-1).reshape(ncells, n, 1)
    return np.linalg.solve(A, b)[..., 0]   # (ncells, n)


def eigensystem_pack(*a_flat):
    """Compute-once companion of :func:`eigensystem` (no ``idx``): the full
    packed stack, consumed by :func:`pick`.  See the pack-helper note above."""
    return _eigensystem_pack(a_flat)


def eigenvalues_pack(*a_flat):
    """Compute-once companion of :func:`eigenvalues` (no ``idx``)."""
    return _eigenvalues_pack(a_flat)


def solve_pack(*args):
    """Compute-once companion of :func:`solve` (no ``idx``)."""
    return _solve_pack(args)


def pick(packed, idx):
    """idx-th component of a packed opaque-kernel result (``*_pack``).  Cheap
    trailing-axis index read — the numpy printer emits ``n+2n²`` of these
    against ONE cse-hoisted ``K_pack(*A_flat)`` temp (REQ-179).  ``_scalarize``
    keeps the scalar-input path returning a float, matching the cached
    kernels."""
    return _scalarize(packed[..., int(idx)])


def eigensystem(idx, *a_flat):
    """Opaque ``eigensystem`` kernel: idx-th component of the flat stack
    ``[eigenvalues(n), R(n·n), L=R⁻¹(n·n)]`` of the row-major ``n×n`` matrix
    ``a_flat`` (``n = round(√len)``).  Batched over the grid; one ``eig`` is
    shared across the ``n + 2n²`` component calls via the 1-slot cache.  (The
    numpy printer normally rewrites these into ``pick(eigensystem_pack(*a), i)``
    per REQ-179; this cached form remains the cross-backend contract kernel and
    the direct-call path.)"""
    key = tuple(map(id, a_flat))
    c = _EIGENSYSTEM_CACHE
    if c["key"] != key:
        c["args"] = a_flat   # REQ-168 ADD.3: pin refs — ids of gc'd arrays get recycled
        c["key"] = key
        c["out"] = _eigensystem_pack(a_flat)
    return _scalarize(c["out"][..., int(idx)])


def eigenvalues(idx, *a_flat):
    """Opaque ``eigenvalues`` kernel (λ-only): idx-th eigenvalue (real part) of
    the row-major ``n×n`` matrix ``a_flat`` (``n = round(√len)``).  The light
    companion of :func:`eigensystem` — no eigenvectors — for the wave-speed /
    CFL bound.  Batched over the grid; one ``eigvals`` shared via the cache."""
    key = tuple(map(id, a_flat))
    c = _EIGENVALUES_CACHE
    if c["key"] != key:
        c["args"] = a_flat   # REQ-168 ADD.3: pin refs — ids of gc'd arrays get recycled
        c["key"] = key
        c["out"] = _eigenvalues_pack(a_flat)
    return _scalarize(c["out"][..., int(idx)])


def solve(idx, *args):
    """Opaque ``solve`` kernel: idx-th component of the per-cell linear solve
    ``A⁻¹ b``.  ``args`` = row-major ``A`` (n·n) followed by ``b`` (n); ``n``
    inferred from the count ``n·n + n``.  Batched over the grid — the NSM
    point-implicit ``source`` lowers to one batched ``np.linalg.solve`` per
    step, shared across the ``n`` component calls via the 1-slot cache."""
    key = tuple(id(a) for a in args)
    c = _SOLVE_CACHE
    if c["key"] != key:
        c["args"] = args     # REQ-168 ADD.3: pin refs — ids of gc'd arrays get recycled
        c["key"] = key
        c["out"] = _solve_pack(args)
    return c["out"][:, int(idx)]


# Backend-supplied opaque kernels (kernel_functions.REQUIRED_KERNELS).
# ``compute_derivative`` is None here: the SOLVER injects the mesh-bound impl
# (``mesh.compute_derivatives``) before the ``update_aux_variables`` slot is
# compiled — same seam as the other backends.
KERNELS = {
    "compute_derivative": None,
    "eigensystem": eigensystem,
    "eigenvalues": eigenvalues,
    "solve": solve,
    # REQ-179 compute-once lowering targets (numpy-internal, not part of the
    # cross-backend REQUIRED_KERNELS contract — like the ARITHMETIC helpers).
    "eigensystem_pack": eigensystem_pack,
    "eigenvalues_pack": eigenvalues_pack,
    "solve_pack": solve_pack,
    "pick": pick,
}

# Arithmetic / printer-lowered helpers the numpy printer emits — NOT part of the
# backend-supplied kernel contract (``conditional`` lowers to ``np.where``).
ARITHMETIC = {
    "ones_like": np.ones_like,
    "zeros_like": np.zeros_like,
    "array": np.array,
    "squeeze": np.squeeze,
    "conditional": lambda c, t, f: np.where(c, t, f),
}


def numpy_module():
    """The full numpy runtime module dict = arithmetic helpers + the
    UserFunctions kernel table.  ``to_numpy.NumpyRuntimeModel.module`` is this."""
    return dict(ARITHMETIC, **KERNELS)
