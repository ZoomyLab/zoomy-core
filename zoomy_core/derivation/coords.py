"""Standard coordinate symbols for the physical-z workflow.

Every model in this family uses the same coordinate quartet
``(t, x, z, g)`` with ``h(t, x)`` for the depth and ``b(x)`` for the
bottom topography.  ``ξ`` is reserved for the *projection-step* basis
argument (after the affine map z = ξh + b), and is created locally in
the projection module — not part of the default coordinate set.

Callers can use ``default_coords()`` and ``default_h``/``default_b``
or pass their own symbols; all helpers in this package accept
arbitrary symbols.
"""
from __future__ import annotations

from typing import Tuple

import sympy as sp


def default_coords() -> Tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]:
    """Return ``(t, x, z, g)`` with the conventional sign / positivity
    constraints (g positive)."""
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    z = sp.Symbol("z", real=True)
    g = sp.Symbol("g", positive=True)
    return t, x, z, g


def default_h(t: sp.Symbol, x: sp.Symbol):
    return sp.Function("h", real=True)(t, x)


def default_b(x: sp.Symbol):
    return sp.Function("b", real=True)(x)
