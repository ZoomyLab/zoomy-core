"""Standard coordinate symbols + canonical state functions.

Every model in this family uses the same coordinate quartet
``(t, x, ξ, g)`` with ``h(t, x)`` for the depth and ``b(x)`` for the
bottom topography (function of ``x`` only — the bottom is fixed in
time for hyperbolic shallow-water-family analyses).

Callers can either use these defaults or pass their own symbols; all
the helpers in this package accept arbitrary symbols.
"""
from __future__ import annotations

from typing import Tuple

import sympy as sp


def default_coords() -> Tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]:
    """Return ``(t, x, ξ, g)`` with the conventional sign / positivity
    constraints (g positive)."""
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    xi = sp.Symbol("xi", real=True)
    g = sp.Symbol("g", positive=True)
    return t, x, xi, g


def default_h(t: sp.Symbol, x: sp.Symbol):
    return sp.Function("h", real=True)(t, x)


def default_b(x: sp.Symbol):
    return sp.Function("b", real=True)(x)
