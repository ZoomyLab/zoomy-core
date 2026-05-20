"""Unit tests for SigmaTransform — the σ-coordinate chain-rule operator.

Reference: Kowalski & Torrilhon 2019, "Moment Approximations and Model
Cascades for Shallow Flow", Commun. Comput. Phys. 25 (2019), 669-702,
eqs. (3.5)–(3.11).
"""
import sympy as sp

from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, SigmaTransform,
)


def _state_2d():
    return StateSpace(dimension=2)


def test_dz_becomes_one_over_h_dsigma():
    """K&T (3.9):  h ∂_z ψ = ∂_ζ ψ̃   ⇒   ∂_z ψ = (1/h) ∂_ζ ψ̃."""
    s = _state_2d()
    psi = sp.Function("psi", real=True)(s.t, s.x, s.z)
    leaf = sp.Derivative(psi, s.z)

    op = SigmaTransform(s)
    out = sp.simplify(op._leaf_sp(leaf))

    psi_tilde = sp.Function("psi", real=True)(s.t, s.x, s.zeta_ref)
    expected = sp.Derivative(psi_tilde, s.zeta_ref) / s.h
    assert sp.simplify(out - expected) == 0, f"got {out}, expected {expected}"


def test_dx_chain_rule():
    """∂_x ψ|_z = ∂_x ψ̃|_ζ − (∂_x(ζh+b)/h) · ∂_ζ ψ̃."""
    s = _state_2d()
    psi = sp.Function("psi", real=True)(s.t, s.x, s.z)
    leaf = sp.Derivative(psi, s.x)

    out = sp.simplify(SigmaTransform(s)._leaf_sp(leaf))

    psi_tilde = sp.Function("psi", real=True)(s.t, s.x, s.zeta_ref)
    jac = sp.Derivative(s.zeta_ref * s.h + s.b, s.x)
    expected = (sp.Derivative(psi_tilde, s.x)
                - (jac / s.h) * sp.Derivative(psi_tilde, s.zeta_ref))
    assert sp.simplify(out - expected) == 0, f"got {out}, expected {expected}"


def test_dt_chain_rule():
    """∂_t ψ|_z = ∂_t ψ̃|_ζ − (∂_t(ζh+b)/h) · ∂_ζ ψ̃."""
    s = _state_2d()
    psi = sp.Function("psi", real=True)(s.t, s.x, s.z)
    leaf = sp.Derivative(psi, s.t)

    out = sp.simplify(SigmaTransform(s)._leaf_sp(leaf))

    psi_tilde = sp.Function("psi", real=True)(s.t, s.x, s.zeta_ref)
    jac = sp.Derivative(s.zeta_ref * s.h + s.b, s.t)
    expected = (sp.Derivative(psi_tilde, s.t)
                - (jac / s.h) * sp.Derivative(psi_tilde, s.zeta_ref))
    assert sp.simplify(out - expected) == 0


def test_zeta_derivative_is_opaque_after_transform():
    """sympy must NOT auto-chain through ψ̃(t,x,ζ).

    If the substitution accidentally produced ``ψ(t, x, ζ·h + b)``,
    sympy would chain ∂_ζ through ζ·h + b and emit ``h · ∂ψ/∂z``-style
    Subs nodes — fatal for downstream Galerkin projection.
    """
    s = _state_2d()
    psi = sp.Function("psi", real=True)(s.t, s.x, s.z)
    out = SigmaTransform(s)._leaf_sp(sp.Derivative(psi, s.z))
    # Take another derivative wrt ζ and confirm it stays opaque.
    second = sp.diff(out, s.zeta_ref)
    assert not any(isinstance(a, sp.Subs) for a in sp.preorder_traversal(second)), \
        f"sympy auto-chained through the mapped function: {second}"


def test_free_z_substituted_by_sigma_h_plus_b():
    """Free atomic z (not inside a Function arg) becomes ζh + b.

    Use-case: hydrostatic pressure p_H = (h_s − z) ρ g → (h − σh) ρ g
    after the transform.
    """
    s = _state_2d()
    rho, g = sp.Symbol("rho", positive=True), sp.Symbol("g", positive=True)
    p_H = (s.b + s.h - s.z) * rho * g  # z appears as a free symbol
    out = sp.simplify(SigmaTransform(s)._leaf_sp(p_H))
    expected = sp.simplify((s.h - s.zeta_ref * s.h) * rho * g)
    assert sp.simplify(out - expected) == 0, f"got {out}, expected {expected}"


def test_continuity_matches_kt2019_3_10():
    """K&T (3.10) chain-rule form: h(∂_x u + ∂_z w) →
       h ∂_x ũ − (∂_x(ζh+b)) ∂_ζ ũ + ∂_ζ w̃.

    This is the **non-conservative** form before the product-rule
    recombination into (3.11).
    """
    s = _state_2d()
    ins = FullINS(s).apply(SigmaTransform(s))

    # Multiply the transformed continuity by h to compare with K&T (3.10).
    cont = sp.simplify(s.h * ins.continuity.expr)

    u_t = sp.Function("u", real=True)(s.t, s.x, s.zeta_ref)
    w_t = sp.Function("w", real=True)(s.t, s.x, s.zeta_ref)
    jac_x = sp.Derivative(s.zeta_ref * s.h + s.b, s.x)
    expected = (s.h * sp.Derivative(u_t, s.x)
                - jac_x * sp.Derivative(u_t, s.zeta_ref)
                + sp.Derivative(w_t, s.zeta_ref))
    assert sp.simplify(cont - expected) == 0, f"got {cont}, expected {expected}"


def test_3d_state_emits_y_chain_terms():
    """3D state must produce ∂_y chain-rule terms with ∂_y(ζh+b)."""
    s = StateSpace(dimension=3)
    psi = sp.Function("psi", real=True)(s.t, s.x, s.y, s.z)
    out = sp.simplify(SigmaTransform(s)._leaf_sp(sp.Derivative(psi, s.y)))

    psi_tilde = sp.Function("psi", real=True)(s.t, s.x, s.y, s.zeta_ref)
    jac_y = sp.Derivative(s.zeta_ref * s.h + s.b, s.y)
    expected = (sp.Derivative(psi_tilde, s.y)
                - (jac_y / s.h) * sp.Derivative(psi_tilde, s.zeta_ref))
    assert sp.simplify(out - expected) == 0
