"""Algebraic closures for projected models in physical-z.

  ``kbc_bottom_solve_w_N(ansatz, flow)`` — solve the kinematic BC at
  the bottom (``w(z=b) = u(z=b) · ∂_x b``) for ``w_N``.  Substitutes
  ``z → b`` in the polynomial ansatz, which makes ζ → 0, so all basis
  values become 1 and ``u(b) = Σ u_i``, ``w(b) = Σ w_i``.

  ``surface_bc_solve_p_N(ansatz, flow)`` — solve the non-hydrostatic
  surface BC ``p(z = b + h) = 0`` for ``p_N``.  At z = b + h: ζ → 1,
  so ``φ_i(1) = (-1)^i`` and ``p|_{η} = Σ (-1)^i p_i``.
"""
from __future__ import annotations

import sympy as sp


def kbc_bottom_solve_w_N(ansatz, flow):
    """``w(z=b) − u(z=b) · ∂_x b = 0`` solved for ``w_N``.

    With ``∂_t b = 0`` (b fixed in time), this reduces to a purely
    spatial relation: at z = b, ζ = 0, basis values are 1.
    """
    if not ansatz.w_coeffs:
        raise ValueError("ansatz has no w coefficients; nothing to close.")
    z = flow.z
    b = flow.b
    # Evaluate u(z=b), w(z=b) by direct substitution into the
    # polynomial expressions.  doit() to flatten any derivatives.
    u_at_b = sp.expand(ansatz.u.doit().xreplace({z: b}))
    w_at_b = sp.expand(ansatz.w.doit().xreplace({z: b}))
    kbc = w_at_b - u_at_b * sp.Derivative(b, flow.x)
    sol = sp.solve(kbc, ansatz.w_coeffs[-1])
    if not sol:
        raise ValueError("KBC at z=b doesn't solve linearly for w_N.")
    return sol[0]


def surface_bc_solve_p_N(ansatz, flow):
    """``p(z = b+h) = 0`` solved for ``p_N``."""
    if not ansatz.p_coeffs:
        raise ValueError("ansatz has no p coefficients; nothing to close.")
    z = flow.z
    eta = flow.eta
    p_at_eta = sp.expand(ansatz.p.doit().xreplace({z: eta}))
    sol = sp.solve(p_at_eta, ansatz.p_coeffs[-1])
    if not sol:
        raise ValueError("Surface BC at z=η doesn't solve for p_N.")
    return sol[0]
