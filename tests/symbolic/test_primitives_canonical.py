"""Per-primitive tests for ``zoomy_core.symbolic.primitives_canonical``.

Each primitive gets:

* Pen-and-paper canonical case — known input, known output.
* Refusal case — input with no matching atom, primitive is a no-op.
* Idempotence — applying twice gives the same result as once.
"""

from __future__ import annotations

import pytest
import sympy as sp

from zoomy_core.symbolic import D, Int
from zoomy_core.symbolic.primitives_canonical import (
    alpha_rename,
    constant_integrand,
    distribute_mul_over_add,
    drop_zero_derivative_inner,
    drop_zero_integrand,
    kill_zero_length_integral,
    merge_integrals_over_add,
    split_integral_over_add,
    un_subs,
    zero_derivative_of_free_symbol,
)


# Common symbols.
t, x, z = sp.symbols("t x z", real=True)
b = sp.Function("b")(t, x)
h = sp.Function("h")(t, x)
f = sp.Function("f")(t, x, z)
g = sp.Function("g")(t, x, z)


# ---------------------------------------------------------------------------
# alpha_rename
# ---------------------------------------------------------------------------

def test_alpha_rename_noop_on_no_integral():
    assert alpha_rename(t + x) == t + x


def test_alpha_rename_renames_bound_var():
    e = Int(f, (z, 0, h))
    out = alpha_rename(e)
    # The bound variable was renamed to a Dummy named '\hat{z}'.
    assert isinstance(out, sp.Integral)
    new_var = out.args[1][0]
    assert isinstance(new_var, sp.Dummy)
    assert new_var.name == r"\hat{z}"


def test_alpha_rename_structurally_canonical():
    # Each call generates fresh Dummies, so ``once == twice`` would be
    # False (different Dummy identities).  What we DO check: each call
    # produces an Integral over a single Dummy bound variable named
    # ``\hat{z}`` (same canonical name).
    e = Int(f, (z, 0, h))
    once = alpha_rename(e)
    twice = alpha_rename(once)
    for out in (once, twice):
        assert isinstance(out, sp.Integral)
        bound_var = out.args[1][0]
        assert isinstance(bound_var, sp.Dummy)
        assert bound_var.name == r"\hat{z}"


# ---------------------------------------------------------------------------
# distribute_mul_over_add
# ---------------------------------------------------------------------------

def test_distribute_mul_over_add_canonical():
    c = sp.Symbol("c")
    e = c * (f + g)
    assert distribute_mul_over_add(e) == c * f + c * g


def test_distribute_mul_over_add_does_not_split_integrals():
    e = Int(f + g, (z, 0, h))
    # The integrand stays intact (split is a separate primitive).
    out = distribute_mul_over_add(e)
    assert isinstance(out, sp.Integral)


def test_distribute_mul_over_add_idempotent():
    c = sp.Symbol("c")
    e = c * (f + g)
    once = distribute_mul_over_add(e)
    twice = distribute_mul_over_add(once)
    assert once == twice


# ---------------------------------------------------------------------------
# split_integral_over_add
# ---------------------------------------------------------------------------

def test_split_integral_over_add_canonical():
    e = Int(f + g, (z, 0, h))
    out = split_integral_over_add(e)
    args = sp.Add.make_args(out)
    assert len(args) == 2


def test_split_integral_noop_single_term():
    e = Int(f, (z, 0, h))
    assert split_integral_over_add(e) == e


# ---------------------------------------------------------------------------
# merge_integrals_over_add
# ---------------------------------------------------------------------------

def test_merge_integrals_alpha_equivalent():
    """Two integrals over the same range with different bound dummy names
    must merge under alpha-equivalence."""
    zh = sp.Symbol(r"\hat{z}", real=True)
    a = Int(f, (z, 0, h))
    b_eq = Int(f.subs(z, zh), (zh, 0, h))
    out = merge_integrals_over_add(a + b_eq)
    # Merged form: ``2 * Integral(f, ...)`` (sympy's Add canonicalisation
    # collects the like-Integral atoms and pulls the coefficient out).
    # Either form (single Integral with 2 inside, or coeff·Integral) is
    # mathematically correct; the test asserts a single-atom Integral
    # remains.
    integrals = list(out.atoms(sp.Integral))
    assert len(integrals) == 1


def test_merge_integrals_alpha_rename_makes_pre_existing_factors_var_independent():
    """After ``alpha_rename`` runs, every outer factor is
    automatically independent of the (now-canonical, fresh-Dummy) bound
    variable.  This means the legacy "var-dependent coeff blocks
    merge" scenario is unreachable in normal usage — the merge primitive
    always succeeds when limits and Derivative-wrappers match.

    We assert this property directly: an outer factor that names the
    *pre-rename* bound symbol ``z`` is no longer the same as the
    *post-rename* bound Dummy, so it merges fine.
    """
    e = z * Int(f, (z, 0, h)) + Int(g, (z, 0, h))
    out = merge_integrals_over_add(e)
    integrals = list(out.atoms(sp.Integral))
    assert len(integrals) == 1   # pre-existing ``z`` merged in


# ---------------------------------------------------------------------------
# kill_zero_length_integral — STRUCTURAL ONLY
# ---------------------------------------------------------------------------

def test_kill_zero_length_canonical():
    e = Int(f, (z, b, b))
    assert kill_zero_length_integral(e) == 0


def test_kill_zero_length_does_not_use_simplify():
    """A non-trivially-equal lo/hi pair (e.g. one that requires sp.simplify
    to detect equality) is NOT collapsed.  This is the deliberate
    tightening over the legacy version."""
    a, b_sym = sp.symbols("a b_sym")
    # lo - hi = (a + b_sym) - (b_sym + a) — sympy's Add canonicalisation
    # actually reduces this structurally.  Use a case that needs simplify:
    expr_with_complex_eq = Int(f, (z, sp.sin(x)**2 + sp.cos(x)**2, 1))
    # sin² + cos² = 1, but only sp.simplify reveals it; structural
    # comparison says they differ.
    out = kill_zero_length_integral(expr_with_complex_eq)
    # Stays as the Integral — primitive does NOT use sp.simplify.
    assert isinstance(out, sp.Integral)


# ---------------------------------------------------------------------------
# zero_derivative_of_free_symbol
# ---------------------------------------------------------------------------

def test_zero_derivative_of_free_symbol_canonical():
    # ∂_x z = 0 (z and x are distinct free symbols)
    e = D(z, x) + 5
    assert zero_derivative_of_free_symbol(e) == 5


def test_zero_derivative_of_free_symbol_noop_on_function():
    e = D(f, x) + 5
    out = zero_derivative_of_free_symbol(e)
    assert out == e   # f is a function call, not a free symbol


def test_zero_derivative_of_free_symbol_noop_self_derivative():
    e = D(z, z) + 5
    # ∂_z z = 1 (sympy auto-evaluates this in canonical construction;
    # the held form D(z, z) might keep it.)
    out = zero_derivative_of_free_symbol(e)
    # The check is: same diff-var as inner ⇒ skip.  We don't fire.
    # (Sympy may auto-eval D(z,z) → 1 elsewhere; that's fine, just
    #  ensure our primitive doesn't fire on it.)
    if isinstance(e, sp.Add) and any(isinstance(t_, sp.Derivative)
                                     for t_ in e.args):
        # Auto-eval didn't fire — verify our primitive doesn't reduce to 0.
        assert out != 5


# ---------------------------------------------------------------------------
# un_subs
# ---------------------------------------------------------------------------

def test_un_subs_safe_unwraps():
    e = sp.Subs(f, z, b)
    # f has z as the third argument; the substitution into z is safe
    # (no conflicting Derivative-of-z or nested Integral over z).
    out = un_subs(e)
    assert out == f.subs(z, b)


def test_un_subs_unsafe_keeps_wrapped():
    """Subs whose inner has a Derivative w.r.t. the binding variable
    must NOT be unwrapped — would commit to chain-rule-through-val."""
    e = sp.Subs(D(f, z), z, b)
    out = un_subs(e)
    assert out == e


def test_un_subs_unsafe_inner_integral_over_var():
    """Subs binding ``z`` whose inner has ``Integral(_, (z, ..., ...))``
    — would shadow the binder."""
    e = sp.Subs(Int(f, (z, 0, h)), z, b)
    out = un_subs(e)
    assert out == e


# ---------------------------------------------------------------------------
# drop_zero_integrand / drop_zero_derivative_inner
# ---------------------------------------------------------------------------

def test_drop_zero_integrand():
    e = Int(0, (z, 0, h)) + 7
    assert drop_zero_integrand(e) == 7


def test_drop_zero_derivative_inner():
    e = sp.Derivative(0, x) + 7
    assert drop_zero_derivative_inner(e) == 7


def test_constant_integrand():
    # b doesn't depend on z, so ∫_0^h b dz = b · h.
    e = Int(b, (z, 0, h))
    assert constant_integrand(e) == b * h
