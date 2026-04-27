"""Algebraic closures common to non-hydrostatic projected models.

  ``kbc_bottom_solve_w_N(ansatz, flow)`` — returns ``w_N = …`` from the
  KBC at ξ=0 (``ω(0) = 0``).  Solves the algebraic equation in
  ``w_N``.

  ``surface_bc_solve_p_N(ansatz, flow)`` — returns ``p_N = …`` from the
  non-hydrostatic surface condition ``p|_{ξ=1} = 0``.

Both expressions are returned **as the RHS of the closure relation**;
callers typically wire ``w_N - rhs = 0`` and ``p_N - rhs = 0`` into
their PDESystem as algebraic equations, OR substitute them into the
differential equations to eliminate ``w_N`` and ``p_N`` entirely.
"""
from __future__ import annotations

import sympy as sp


def kbc_bottom_solve_w_N(ansatz, flow):
    """Solve ``ω(ξ=0) = 0`` for ``w_N``.  ``ω(0) = w(0) − u(0)·∂_x b``
    after killing the constant-in-ξ ``∂_t b`` term (b is fixed)."""
    if not ansatz.w_coeffs:
        raise ValueError("ansatz has no w coefficients; nothing to close.")
    omega = (ansatz.w
             - ansatz.xi * sp.Derivative(flow.h, flow.t)
             - ansatz.u * flow._d_xH_b())
    omega_at_0 = sp.expand(omega.subs(ansatz.xi, 0))
    omega_at_0 = omega_at_0.subs({sp.Derivative(flow.b, flow.t): 0})
    sol = sp.solve(omega_at_0, ansatz.w_coeffs[-1])
    if not sol:
        raise ValueError(
            "KBC at ξ=0 doesn't solve for w_N — check that w_N appears "
            "linearly with non-zero coefficient (it should, via φ_N(0)=1)."
        )
    return sol[0]


def surface_bc_solve_p_N(ansatz, flow):
    """Solve ``p(ξ=1) = 0`` for ``p_N``.  Surface BC for the
    non-hydrostatic pressure."""
    if not ansatz.p_coeffs:
        raise ValueError("ansatz has no p coefficients; nothing to close.")
    p_at_1 = sp.expand(ansatz.p.subs(ansatz.xi, 1))
    sol = sp.solve(p_at_1, ansatz.p_coeffs[-1])
    if not sol:
        raise ValueError(
            "Surface BC at ξ=1 doesn't solve for p_N (φ_N(1) ≠ 0?)."
        )
    return sol[0]
