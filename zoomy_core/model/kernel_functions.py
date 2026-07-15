"""
Custom SymPy functions for numerical models.

These are opaque to SymPy (never simplified away) but have concrete
implementations in each backend (NumPy, JAX, C).

Each function must be registered in the backend's module dict:
  - NumPy: to_numpy.py → NumpyRuntimeModel.module
  - JAX:   to_jax.py
  - C:     generic_c.py
"""

import sympy as sp
import itertools
from zoomy_core.misc.misc import ZArray


class eigensystem(sp.Function):
    """eigensystem(idx, *A_flat) — idx-th component of the eigendecomposition of
    the row-major n*n matrix A_flat, laid out as
      [ eigenvalues(n), right_eigenvectors R (n*n, row-major), left L=R^{-1} (n*n) ].
    Opaque to SymPy; numerical in the backends (np.linalg.eig / Eigen).  The Roe
    scheme builds |A| = R|Lambda|L from this."""

    is_commutative = True
    is_real = True

    @classmethod
    def eval(cls, *args):
        return None  # always keep unevaluated


class eigenvalues(sp.Function):
    """eigenvalues(idx, *A_flat) — idx-th eigenvalue (real part) of the
    row-major n*n matrix A_flat, with n inferred from ``len(A_flat) = n*n``.

    The λ-only companion of :class:`eigensystem` (no right/left eigenvectors),
    used for the wave-speed / CFL bound ``max|λ_i(A_n)|`` when the model has no
    closed-form spectrum (SME / VAM).  A full ``eigensystem`` returns λ **and**
    R **and** L=R⁻¹ — for a wave speed that needs only ``max|λ|`` that is ~2/3
    waste per face per step, and the R⁻¹ is the part that can fail near wet/dry;
    this kernel skips the eigenvector solve entirely and stays cheap /
    on-device.  Opaque to SymPy; realised numerically per backend
    (``np.linalg.eigvals`` / Eigen / on-device iteration)."""

    is_commutative = True
    is_real = True

    @classmethod
    def eval(cls, *args):
        return None  # always keep unevaluated (opaque)


class solve(sp.Function):
    """``solve(idx, *A_flat, *b_flat)`` — idx-th component of the per-cell
    linear solve ``A⁻¹ b``.  ``A`` is the row-major ``n*n`` matrix (the first
    ``n*n`` args after ``idx``) and ``b`` the length-``n`` RHS (the last ``n``
    args); ``n`` is inferred from the arg count (``n*n + n``).

    Opaque to SymPy; numerical in each backend (``np``/``jnp.linalg.solve`` /
    Eigen ``A.colPivHouseholderQr().solve(b)``).  The single backend-specific
    atom of the NSM point-implicit source treatment: the linearized source
    ``S_lin = (I − dt·J)⁻¹ S`` emits one ``solve(i, A_flat, b_flat)`` per row,
    with ``A = I − dt·J`` and ``b = S`` — every backend then consumes the
    transformed ``source`` as an ordinary source, the only new op being this
    per-cell solve (mirrors the ``eigensystem`` opaque-op pattern)."""

    is_commutative = True
    is_real = True

    @classmethod
    def eval(cls, *args):
        return None  # always keep unevaluated (opaque)


class compute_derivative(sp.Function):
    """``compute_derivative(field, *multi_index)`` — the NON-LOCAL spatial
    derivative of ``field`` of order ``multi_index`` (e.g.
    ``compute_derivative(h, 1, 0)`` = ∂ₓh, ``compute_derivative(b, 0, 1)`` =
    ∂_yb).  Opaque to SymPy; the numeric impl is **injected by the solver**
    into the backend module dict (like the ``eigensystem`` kernel), bound
    to that backend's LSQ stencil + mesh — numpy ``mesh.compute_derivatives``,
    jax ``lsq_gradient_per_field``.

    This is the single code-printed source for derivative aux: ``expose_aux``
    emits it into ``update_aux_variables`` so every backend computes the
    derivative from ONE symbolic object, given only a ``compute_derivative``
    entry in its user-functions map.  The carrying slot must be lowered
    WHOLE-GRID (``field`` arrives as the full row, not a per-cell scalar); the
    trailing integer args are the static multi-index.
    """
    is_commutative = True
    is_real = True

    @classmethod
    def eval(cls, *args):
        return None  # always keep unevaluated (opaque)


class conditional(sp.Function):
    """
    A Vector-Aware, Differentiable Conditional Function.

    Usage:
      conditional(c, t, f)

    Behavior:
      - If inputs are Scalars: Returns a symbolic `conditional` object.
      - If inputs are Vectors: Returns a `ZArray` of symbolic `conditional` objects.

    Differentiation:
      - Implements the 'Active Branch' rule:
        d/dx conditional(c, t, f) = conditional(c, dt/dx, df/dx)
    """

    nargs = 3
    is_commutative = (
        True  # <--- CRITICAL FIX: prevents PolynomialError in charpoly/eigenvals
    )

    def __new__(cls, condition, true_val, false_val, **kwargs):
        # 1. Helper to check for array-like inputs (excluding Symbols)
        """Hook `__new__`."""
        def is_vec(x):
            """Is vec."""
            return hasattr(x, "__getitem__") and not isinstance(
                x, (sp.Symbol, sp.Function)
            )

        is_array_t = is_vec(true_val)
        is_array_f = is_vec(false_val)

        # 2. Vector Case: Broadcast and return ZArray
        if is_array_t or is_array_f:
            # Flatten inputs
            t_flat = list(sp.flatten(true_val)) if is_array_t else [true_val]
            f_flat = list(sp.flatten(false_val)) if is_array_f else [false_val]

            # Determine shape
            if hasattr(true_val, "shape"):
                shape = true_val.shape
            elif hasattr(false_val, "shape"):
                shape = false_val.shape
            else:
                shape = (len(t_flat),)

            # Broadcast scalar '0' filler if lengths mismatch (standard python zip_longest behavior)
            result_list = []

            # Handling scalar vs vector broadcasting manually for safety
            max_len = max(len(t_flat), len(f_flat))
            if not is_array_t:
                t_flat = [true_val] * max_len
            if not is_array_f:
                f_flat = [false_val] * max_len

            for t, f in zip(t_flat, f_flat):
                # Recursively call __new__ (which will hit the Scalar Case below)
                result_list.append(cls(condition, t, f))

            # Return a ZArray containing the symbolic nodes
            return ZArray(result_list).reshape(*shape)

        # 3. Scalar Case: Create the actual Symbolic Node
        return super().__new__(cls, condition, true_val, false_val, **kwargs)

    def _eval_derivative(self, s):
        """
        Differentiates the branches, ignoring the jump at the condition boundary.
        This is standard for numerical Jacobians in FVM.
        """
        condition, true_expr, false_expr = self.args
        return conditional(condition, true_expr.diff(s), false_expr.diff(s))


# ── The UserFunctions contract (REQ-168) ─────────────────────────────────────
# The opaque kernels every backend must SUPPLY (numpy `module` /
# `UserFunctions.H` / jax `userfunctions.py`).  A backend whose table is missing
# one of these has a silent hole — each backend ships a contract test asserting
# its table covers this set, so the next missing kernel is a RED test rather
# than a `NameError` at lambdify (python) or a link error mid-build (C++).
#
# ``conditional`` is DELIBERATELY EXCLUDED: it is printer-lowered to an inline
# ternary (``generic_c.py`` → ``((c) ? (t) : (f))``; numpy → ``np.where``), so
# it never reaches a backend UserFunctions table — a flat "all kernel classes"
# list would wrongly demand every C++ backend implement a function that must not
# exist (dmplex REQ-168 item 5).
REQUIRED_KERNELS = frozenset({
    "compute_derivative",
    "eigensystem",
    "eigenvalues",
    "solve",
})
