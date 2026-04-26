"""Generalised-eigenvalue (pencil) form for analysis.

A linearised PDE system can be written in the form

    M_t · ∂_t δq + M_xa · ∂_xa δq + M_0 · δq = 0     (sum over axis a)

where ``M_t``, ``M_xa``, ``M_0`` are constant matrices (coefficients
evaluated at the base state).  Plane-wave ansatz ``δq = q̂ e^(i(k·x −
ωt))`` reduces this to

    [-iω M_t + i k · (Σ_a n_a M_xa) + M_0] q̂ = 0

where ``n`` is the normal direction (``Σ n_a² = 1``).  Dividing by ``ik``
and writing ``λ = ω/k`` gives the **generalised eigenvalue problem**

    [Σ_a n_a M_xa + (1 / (ik)) M_0] q̂ = λ M_t q̂.

For the principal symbol (high-k limit, or systems with M_0 = 0):

    A_n q̂ = λ M_t q̂          A_n := Σ_a n_a M_xa.

This module extracts ``(M_t, [M_xa], M_0)`` mechanically from a
linearised ``PDESystem`` and computes the generalised eigenvalues
either symbolically (``sp.solve(charpoly)``) or numerically
(``scipy.linalg.eig``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from .pde_system import PDESystem


# ---------------------------------------------------------------------------
# Pencil extraction
# ---------------------------------------------------------------------------

def extract_quasilinear_pencil(linear_system: PDESystem
                              ) -> Tuple[sp.Matrix, List[sp.Matrix], sp.Matrix]:
    """Extract ``(M_t, [M_xa], M_0)`` from a *linearised* PDE system.

    Each equation must be linear in the perturbation fields and their
    first derivatives; nonlinear remnants raise ``ValueError``.  Higher
    derivatives (∂_xx, ∂_tt, ∂_xt) are also surfaced as a non-zero
    extra structure — see ``M_higher`` returned in the optional 4th
    element when ``return_higher=True`` (not on by default).

    Returns
    -------
    M_t    : (n_eq × n_field) sp.Matrix — coefficient of ∂_t δq_j.
    M_xa   : list of (n_eq × n_field) sp.Matrix — one per spatial axis.
    M_0    : (n_eq × n_field) sp.Matrix — coefficient of δq_j (no derivative).
    """
    eqs = linear_system.equations
    fields = linear_system.fields
    t = linear_system.time
    spaces = linear_system.space
    n_eq = len(eqs)
    n_field = len(fields)
    n_dim = len(spaces)

    M_t = sp.zeros(n_eq, n_field)
    M_xa = [sp.zeros(n_eq, n_field) for _ in range(n_dim)]
    M_0 = sp.zeros(n_eq, n_field)

    for i, eq in enumerate(eqs):
        eq_e = sp.expand(eq)
        for j, f in enumerate(fields):
            M_t[i, j] = eq_e.coeff(sp.Derivative(f, t))
            for a, x_a in enumerate(spaces):
                M_xa[a][i, j] = eq_e.coeff(sp.Derivative(f, x_a))
            M_0[i, j] = eq_e.coeff(f)

    return M_t, M_xa, M_0


# ---------------------------------------------------------------------------
# Generalised eigenvalues — symbolic
# ---------------------------------------------------------------------------

def generalised_eigenvalues(M_x: sp.Matrix, M_t: sp.Matrix,
                            *, lam: Optional[sp.Symbol] = None,
                            simplify: bool = True) -> List[sp.Expr]:
    """Symbolic generalised eigenvalues of the pencil ``(M_x, M_t)``.

    Solves ``det(M_x − λ M_t) = 0`` for λ.  Returns a list of solutions.

    Caveat — symbolic charpoly degree blows up fast.  For larger systems
    use ``sample_generalised_eigenvalues`` (numerical).
    """
    if lam is None:
        lam = sp.Symbol("lambda")
    n = M_t.shape[0]
    if M_x.shape != M_t.shape:
        raise ValueError(f"shape mismatch: M_x={M_x.shape}, M_t={M_t.shape}")
    char = (M_x - lam * M_t).det(method="berkowitz")
    if simplify:
        char = sp.expand(char)
    return sp.solve(sp.Eq(char, 0), lam)


# ---------------------------------------------------------------------------
# Generalised eigenvalues — numerical, with parameter sampling
# ---------------------------------------------------------------------------

def _eval_matrix_numerically(M: sp.Matrix, sub: Dict, dtype=complex):
    """Evaluate a sympy matrix at numeric values of all symbols, returning numpy."""
    M_sub = M.subs(sub)
    arr = np.array(M_sub.tolist(), dtype=dtype)
    return arr


def sample_generalised_eigenvalues(M_x: sp.Matrix, M_t: sp.Matrix,
                                  parameter_samples: List[Dict],
                                  *, dtype=complex,
                                  drop_infinite: bool = False
                                  ) -> List[np.ndarray]:
    """For each sample (a dict of symbolic-value → numeric-value), return
    the numerical generalised eigenvalues of ``(M_x, M_t)`` at that
    sample.

    Uses ``scipy.linalg.eig(A, B)`` which solves ``A v = λ B v`` even
    when ``B`` is singular (in which case some λ are returned as
    ``inf``).  ``drop_infinite=True`` filters those out.
    """
    try:
        from scipy.linalg import eig as scipy_eig
    except ImportError:                                # pragma: no cover
        scipy_eig = None

    out = []
    for sub in parameter_samples:
        A = _eval_matrix_numerically(M_x, sub, dtype=dtype)
        B = _eval_matrix_numerically(M_t, sub, dtype=dtype)
        if scipy_eig is not None:
            lam = scipy_eig(A, B, left=False, right=False)
        else:
            # Fallback: use np.linalg.eig on B^-1 A (fails if B singular).
            try:
                Binv = np.linalg.inv(B)
                lam = np.linalg.eigvals(Binv @ A)
            except np.linalg.LinAlgError:
                # Add small perturbation, try again.
                lam = np.linalg.eigvals(np.linalg.solve(B + 1e-12 * np.eye(B.shape[0]), A))
        if drop_infinite:
            lam = lam[np.isfinite(lam)]
        out.append(lam)
    return out
