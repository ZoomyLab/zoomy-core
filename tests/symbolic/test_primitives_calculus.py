"""Per-primitive tests for ``zoomy_core.symbolic.primitives_calculus``."""

from __future__ import annotations

import pytest
import sympy as sp

from zoomy_core.symbolic import D, Int
from zoomy_core.symbolic.primitives_calculus import (
    chain_rule,
    distribute_derivative_over_add,
    fundamental_theorem,
    integration_by_parts,
    leibniz,
    leibniz_general,
    polynomial_integrate,
    product_rule_forward,
    product_rule_inverse,
    pull_scalar_out_of_derivative,
    push_scalar_into_derivative,
)


t, x, z = sp.symbols("t x z", real=True)
b = sp.Function("b")(t, x)
h = sp.Function("h")(t, x)
u = sp.Function("u")(t, x, z)
a0 = sp.Function("alpha_0")(t, x)
a1 = sp.Function("alpha_1")(t, x)


# ---------------------------------------------------------------------------
# fundamental_theorem
# ---------------------------------------------------------------------------

def test_ft_canonical():
    # ∫_0^h ∂_z u dz = u|_h - u|_0
    out = fundamental_theorem(D(u, z), z, 0, h)
    expected = u.subs(z, h) - u.subs(z, 0)
    assert out == expected


def test_ft_refuses_non_derivative():
    # Integrand is u, not a Derivative → return None.
    assert fundamental_theorem(u, z, 0, h) is None


def test_ft_refuses_wrong_diff_var():
    # ∫_0^h ∂_x u dz — fundamental theorem doesn't apply (var = z, diff = x).
    assert fundamental_theorem(D(u, x), z, 0, h) is None


# ---------------------------------------------------------------------------
# leibniz (poly-specialised)
# ---------------------------------------------------------------------------

def test_leibniz_poly_canonical():
    # ∫_0^h ∂_x(α_0 + α_1·z) dz, α-functions of (t, x); poly in z.
    out = leibniz(D(a0 + a1 * z, x), z, 0, h)
    # Closed form: ∂_x[α_0·h + α_1·h²/2] - (α_0 + α_1·h)·∂_x h
    expected = (sp.Derivative(a0 * h + a1 * h**2 / 2, x)
                - (a0 + a1 * h) * sp.Derivative(h, x))
    assert sp.simplify(out - expected) == 0


def test_leibniz_refuses_diff_var_eq_int_var():
    # ∂_z fired, but var = z too → fundamental_theorem handles it.
    assert leibniz(D(a0 + a1 * z, z), z, 0, h) is None


# ---------------------------------------------------------------------------
# leibniz_general (opaque integrand)
# ---------------------------------------------------------------------------

def test_leibniz_general_canonical():
    # ∫_0^h ∂_x u dz = ∂_x[∫u dz] - u|_h · ∂_x h
    out = leibniz_general(D(u, x), z, 0, h)
    expected = (sp.Derivative(Int(u, (z, 0, h)), x)
                - u.subs(z, h) * sp.Derivative(h, x))
    assert out == expected


def test_leibniz_general_keeps_volume_held():
    out = leibniz_general(D(u, x), z, 0, h)
    # Volume is a held Integral — primitive never evaluates it.
    integrals = list(sp.preorder_traversal(out))
    has_held_integral = any(
        isinstance(node, sp.Derivative)
        and isinstance(node.args[0], sp.Integral)
        for node in integrals
    )
    assert has_held_integral


# ---------------------------------------------------------------------------
# polynomial_integrate
# ---------------------------------------------------------------------------

def test_polynomial_integrate_constant():
    # ∫_0^h c dz = c · h
    out = polynomial_integrate(sp.Symbol("c"), z, 0, h)
    assert out == sp.Symbol("c") * h


def test_polynomial_integrate_canonical():
    out = polynomial_integrate(a0 + a1 * z, z, 0, h)
    expected = a0 * h + a1 * h**2 / 2
    assert sp.simplify(out - expected) == 0


def test_polynomial_integrate_refuses_non_polynomial():
    # u(t, x, z) is opaque — sp.Poly fails.
    assert polynomial_integrate(u, z, 0, h) is None


# ---------------------------------------------------------------------------
# integration_by_parts
# ---------------------------------------------------------------------------

def test_ibp_canonical():
    # ∫_0^h ∂_z u · h_t dz where h_t = h(t,x) (constant in z).
    # Volume = -∫ u · ∂_z h_t dz.  Boundary upper = u|_h · h, lower = u|_0 · h.
    volume, b_up, b_lo = integration_by_parts(u, h, z, 0, h)
    assert b_up == (u * h).subs(z, h)
    assert b_lo == (u * h).subs(z, 0)


# ---------------------------------------------------------------------------
# product_rule_forward
# ---------------------------------------------------------------------------

def test_product_rule_forward_canonical():
    # ∂_x(a0 · h) → ∂_x a0 · h + a0 · ∂_x h
    out = product_rule_forward(D(a0 * h, x), x)
    # Order may differ — test as Add canonical equality.
    expected = sp.Derivative(a0, x) * h + a0 * sp.Derivative(h, x)
    assert sp.simplify(out - expected) == 0


def test_product_rule_forward_pow():
    # ∂_x(α_0²) → 2·α_0·∂_x α_0
    out = product_rule_forward(D(a0**2, x), x)
    expected = 2 * a0 * sp.Derivative(a0, x)
    assert out == expected


def test_product_rule_forward_noop_on_non_derivative():
    assert product_rule_forward(a0, x) == a0


def test_product_rule_forward_noop_on_wrong_var():
    out = product_rule_forward(D(a0 * h, x), t)
    # var is t, derivative is in x → no-op.
    assert out == D(a0 * h, x)


# ---------------------------------------------------------------------------
# product_rule_inverse
# ---------------------------------------------------------------------------

def test_product_rule_inverse_canonical():
    # a0 · ∂_x f → ∂_x(a0 · f) − ∂_x(a0) · f
    f_fn = sp.Function("f")(t, x)
    out = product_rule_inverse(a0 * D(f_fn, x), x)
    expected = (sp.Derivative(a0 * f_fn, x)
                - sp.Derivative(a0, x) * f_fn)
    assert out == expected


def test_product_rule_inverse_does_NOT_doit_residual():
    """Bug-3 source: legacy ProductRule did ``Derivative(coeff,
    var).doit()`` on the residual, firing chain-rule against
    moving-frame at the wrong pipeline position.  The redesigned
    primitive leaves the residual as a held atom.
    """
    coeff = a1   # purely (t,x)-dependent — sympy could .doit() it but we don't.
    f_fn = sp.Function("f")(t, x)
    out = product_rule_inverse(coeff * D(f_fn, x), x)
    # The output must contain a HELD ``Derivative(coeff, x)`` atom.
    derivs_of_coeff = [
        node for node in sp.preorder_traversal(out)
        if isinstance(node, sp.Derivative) and node.args[0] == coeff
    ]
    assert len(derivs_of_coeff) >= 1


# ---------------------------------------------------------------------------
# chain_rule
# ---------------------------------------------------------------------------

def test_chain_rule_canonical():
    phi_1 = sp.Function("phi_1")
    arg = (z - b) / h

    def is_basis(call):
        return getattr(call.func, "__name__", "").startswith("phi_")

    out = chain_rule(D(phi_1(arg), x), is_basis, x)
    # Should produce f'(arg) · ∂_x arg shape.
    # Structural check: presence of Subs(Derivative(phi_1(_xi), _xi), _xi, arg).
    has_subs_deriv = any(
        isinstance(node, sp.Subs) and isinstance(node.args[0], sp.Derivative)
        for node in sp.preorder_traversal(out)
    )
    assert has_subs_deriv


def test_chain_rule_noop_when_predicate_false():
    g = sp.Function("g")
    arg = (z - b) / h

    def never(call):
        return False

    e = D(g(arg), x)
    assert chain_rule(e, never, x) == e


# ---------------------------------------------------------------------------
# distribute_derivative_over_add
# ---------------------------------------------------------------------------

def test_distribute_derivative_canonical():
    out = distribute_derivative_over_add(D(a0 + a1, x))
    expected = sp.Derivative(a0, x) + sp.Derivative(a1, x)
    assert out == expected


def test_distribute_derivative_zero_when_no_diff_var():
    # ∂_t(a0(t,x)) — a0 depends on t, so DOES NOT zero out.
    out = distribute_derivative_over_add(D(a0, t))
    assert out == D(a0, t)
    # But ∂_t z (z is just a free Symbol) — does zero out.
    out2 = distribute_derivative_over_add(D(z, t))
    assert out2 == 0


# ---------------------------------------------------------------------------
# pull_scalar_out_of_derivative / push_scalar_into_derivative
# ---------------------------------------------------------------------------

def test_pull_scalar_out():
    out = pull_scalar_out_of_derivative(D(2 * a0, x))
    expected = 2 * sp.Derivative(a0, x)
    assert out == expected


def test_push_scalar_in():
    out = push_scalar_into_derivative(2 * D(a0, x))
    expected = sp.Derivative(2 * a0, x)
    assert out == expected


def test_pull_then_push_is_identity():
    e = D(sp.Rational(3, 5) * a0, x)
    pulled = pull_scalar_out_of_derivative(e)
    pushed = push_scalar_into_derivative(pulled)
    assert pushed == e


def test_push_then_pull_is_identity():
    e = sp.Rational(3, 5) * D(a0, x)
    pushed = push_scalar_into_derivative(e)
    pulled = pull_scalar_out_of_derivative(pushed)
    assert pulled == e
