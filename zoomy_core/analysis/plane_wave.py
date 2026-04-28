"""Plane-wave dispersion analysis.

Inserts the 1D plane-wave ansatz ``δq(t, x) = q̂ exp(i(k x − ω t))``
into a linearised ``PDESystem`` and reduces to an algebraic system in
the amplitudes ``q̂``.  Solves ``det M(ω, k) = 0`` for ``ω(k)`` (or
``k(ω)``).  Works for arbitrary linear PDE systems; algebraic
constraint rows just contribute amplitude-only relations to the
matrix.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import sympy as sp

from .pde_system import PDESystem


def plane_wave_matrix(linear_system: PDESystem, *,
                      k: Optional[sp.Symbol] = None,
                      omega: Optional[sp.Symbol] = None,
                      axis: int = 0):
    """Insert ``δq → q̂ exp(i(kx_axis − ωt))`` and reduce to a matrix M.

    The ``axis`` parameter selects which spatial direction the plane
    wave propagates along (default 0 ⇒ ``x``); other directions get
    set to constants in the ansatz (no spatial dependence).

    Returns
    -------
    M : sp.Matrix
        Coefficient matrix such that ``M · q̂_vector = 0``.
    amplitudes : list[sp.Symbol]
        Amplitude symbols, in the same order as ``linear_system.fields``.
    """
    if k is None:
        k = sp.Symbol("k", real=True)
    if omega is None:
        omega = sp.Symbol("omega", real=True)

    t_sym = linear_system.time
    x_axes = linear_system.space
    if not (0 <= axis < len(x_axes)):
        raise ValueError(f"axis={axis} out of range for space={x_axes}")
    x_sym = x_axes[axis]
    I = sp.I
    E = sp.exp(I * (k * x_sym - omega * t_sym))

    # Build amplitude symbols and replacement map.
    amplitudes = []
    repl = {}
    for f in linear_system.fields:
        head = f.func
        name = head.__name__ if hasattr(head, "__name__") else str(head)
        # Strip TeX delta-prefix if present so the amplitude symbol
        # reads cleanly.
        amp_name = name.replace(r"\delta ", "").replace("delta ", "") + "_hat"
        amp = sp.Symbol(amp_name)
        amplitudes.append(amp)
        repl[f] = amp * E

    # Substitute and divide by E.
    pw_eqs = []
    for eq in linear_system.equations:
        eq_sub = eq.xreplace(repl).doit()
        # Each non-zero term in eq_sub contains the factor E (or is 0).
        eq_div = sp.simplify(eq_sub / E)
        pw_eqs.append(eq_div)

    # Build the coefficient matrix.
    M, _ = sp.linear_eq_to_matrix(pw_eqs, amplitudes)
    return M, amplitudes


def plane_wave_dispersion(linear_system: PDESystem, *,
                          k: Optional[sp.Symbol] = None,
                          omega: Optional[sp.Symbol] = None,
                          axis: int = 0,
                          solve_for: str = "omega",
                          simplify: bool = True,
                          factor_in_target: bool = True):
    """Full dispersion solve.

    Returns a dict with::

        matrix       — sp.Matrix M(ω, k)
        amplitudes   — list of q̂ symbols
        determinant  — det M(ω, k)
        solutions    — list of ω(k) (or k(ω)) solutions
        phase_velocity_solutions — [ω/k for ω in solutions] (omega-mode only)

    ``factor_in_target=True`` factors the determinant before solving so
    that trivial ω = 0 or k = 0 roots come out as separate factors and
    don't multiply through.
    """
    if k is None:
        k = sp.Symbol("k", real=True)
    if omega is None:
        omega = sp.Symbol("omega", real=True)

    M, amps = plane_wave_matrix(linear_system, k=k, omega=omega, axis=axis)
    det = M.det(method="berkowitz")
    if simplify:
        det = sp.simplify(det)
    if factor_in_target:
        det = sp.factor(det, omega if solve_for == "omega" else k)

    target = omega if solve_for == "omega" else k
    sols = sp.solve(sp.Eq(det, 0), target)

    out = {
        "matrix": M,
        "amplitudes": amps,
        "determinant": det,
        "solutions": sols,
    }
    if solve_for == "omega":
        out["phase_velocity_solutions"] = [sp.simplify(s / k) for s in sols]
    return out
