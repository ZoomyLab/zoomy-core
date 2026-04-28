"""Integration tests for the canonicalise pass + bug-N regression tests."""

from __future__ import annotations

import pytest
import sympy as sp

from zoomy_core.symbolic import (
    D,
    Int,
    AutoEvalGuard,
    AutoEvalForbidden,
    canonicalise,
    solve_for,
)


t, x, z = sp.symbols("t x z", real=True)
b = sp.Function("b")(t, x)
h = sp.Function("h")(t, x)
u = sp.Function("u")
a0 = sp.Function("alpha_0")(t, x)
a1 = sp.Function("alpha_1")(t, x)


# ---------------------------------------------------------------------------
# Canonicalise pass
# ---------------------------------------------------------------------------

def test_canonicalise_idempotent():
    f = sp.Function("f")(t, x, z)
    g = sp.Function("g")(t, x, z)
    e = sp.Symbol("c") * (f + g) + Int(0, (z, 0, h)) + sp.Derivative(z, x)
    once = canonicalise(e)
    twice = canonicalise(once)
    assert once == twice


def test_canonicalise_kills_alpha_equivalent_pair():
    f = sp.Function("f")(t, x, z)
    zh = sp.Symbol(r"\hat{z}", real=True)
    f_zh = f.subs(z, zh)
    e = Int(f, (z, 0, h)) - Int(f_zh, (zh, 0, h))
    assert canonicalise(e) == 0


def test_canonicalise_drops_zero_integrand():
    e = Int(0, (z, 0, h)) + sp.Symbol("y")
    assert canonicalise(e) == sp.Symbol("y")


def test_canonicalise_kills_zero_length_integral():
    f = sp.Function("f")(t, x, z)
    e = Int(f, (z, b, b)) + sp.Symbol("y")
    assert canonicalise(e) == sp.Symbol("y")


def test_canonicalise_drops_free_symbol_derivative():
    e = sp.Derivative(z, x) + 1
    assert canonicalise(e) == 1


# ---------------------------------------------------------------------------
# Bug-1 regression: solve_for protects Derivative(Integral) from Leibniz expansion
# ---------------------------------------------------------------------------

def test_bug1_solve_for_returns_conservative_form():
    """``∂_t h + ∂_x[Integral(u dẑ, b, b+h)] = 0`` solved for ∂_t h
    must return ``-Derivative(Integral(...), x)`` — the conservative
    form — NOT the Leibniz-expanded ``∫∂_x u dẑ - u(b+h)·∂_x(b+h) +
    u(b)·∂_x b`` form that sp.solve emits without protection.
    """
    zh = sp.Symbol(r"\hat{z}", real=True)
    cont = D(h, t) + D(Int(u(t, x, zh), (zh, b, b + h)), x)
    rel = solve_for(cont, D(h, t))
    rhs = rel[D(h, t)]
    # Structural assertion: rhs is a single Derivative-of-Integral atom (with
    # a leading ``-``).
    assert isinstance(rhs, sp.Mul)
    factors = rhs.args
    has_neg = any(f == -1 for f in factors)
    has_deriv_int = any(
        isinstance(f, sp.Derivative) and isinstance(f.args[0], sp.Integral)
        for f in factors
    )
    assert has_neg and has_deriv_int


# ---------------------------------------------------------------------------
# Auto-eval guard
# ---------------------------------------------------------------------------

def test_auto_eval_guard_bans_doit():
    e = D(sp.Symbol("x")**2, sp.Symbol("x"))
    with AutoEvalGuard():
        with pytest.raises(AutoEvalForbidden):
            e.doit()


def test_auto_eval_guard_bans_simplify():
    e = sp.Symbol("a") + sp.Symbol("a")
    with AutoEvalGuard():
        with pytest.raises(AutoEvalForbidden):
            sp.simplify(e)


def test_auto_eval_guard_restores_on_exit():
    e = D(sp.Symbol("x")**2, sp.Symbol("x"))
    with AutoEvalGuard():
        pass
    # Outside the context, .doit() works normally.
    assert e.doit() == 2 * sp.Symbol("x")
