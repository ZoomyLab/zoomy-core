"""Unit tests for LayerMeanClosure — the piecewise-constant layer
projection operator that takes a σ-resolved system into per-layer
shallow-water form.

Reference: Aguillon, Hörnschemeyer & Sainte-Marie 2026,
"Barotropic-Baroclinic Splitting for Multilayer Shallow Water Models
with Exchanges", eqs. (2)-(8).
"""
import sympy as sp

from zoomy_core.model.models.ins_generator import (
    Expression, FullINS, KinematicBC, LayerMeanClosure, SigmaTransform,
    StateSpace, Integrate,
)
from zoomy_core.model.models.derived_system import System
from zoomy_core.misc.misc import Zstruct


def _state_2d():
    return StateSpace(dimension=2)


def test_layer_count_matches_u_layer_length():
    """LayerMeanClosure should produce L sub-leaves named layer_1..layer_L."""
    s = _state_2d()
    sigma = s.zeta_ref
    u_field = s.u.xreplace({s.z: sigma})
    u_layer = [sp.Function(f"u_{k+1}", real=True)(s.t, s.x) for k in range(3)]

    # Trivial test: identity integrand over σ.  The 3-layer projection
    # should produce 3 leaves; layer α integrates u_α over a third of σ.
    test_sys = System("test_sys", s)
    test_sys.add_equation("eq", Expression(u_field, name="eq"))

    closure = LayerMeanClosure(s, sigma, u_field, u_layer)
    test_sys.eq.apply(closure)

    out_names = sorted(k for k in test_sys.eq._node._filter_dict())
    assert out_names == ["layer_1", "layer_2", "layer_3"], out_names

    # Each layer-α leaf should be u_α / 3 (the layer-mean over its third).
    for k in range(3):
        leaf = getattr(test_sys.eq, f"layer_{k+1}")
        assert sp.simplify(leaf.expr - u_layer[k] / 3) == 0, (
            f"layer_{k+1}: got {leaf.expr}, expected {u_layer[k] / 3}"
        )


def test_layer_projection_of_dx_u():
    """∂_x integrand: ∫_layer_α ∂_x u dσ = l_α · ∂_x u_α."""
    s = _state_2d()
    sigma = s.zeta_ref
    u_field = s.u.xreplace({s.z: sigma})
    u_layer = [sp.Function(f"u_{k+1}", real=True)(s.t, s.x) for k in range(2)]

    test_sys = System("test_sys", s)
    test_sys.add_equation("eq", Expression(sp.Derivative(u_field, s.x),
                                            name="eq"))

    closure = LayerMeanClosure(s, sigma, u_field, u_layer)
    test_sys.eq.apply(closure)

    expected_l1 = sp.Derivative(u_layer[0], s.x) / 2
    expected_l2 = sp.Derivative(u_layer[1], s.x) / 2
    got_l1 = test_sys.eq.layer_1.expr
    got_l2 = test_sys.eq.layer_2.expr
    assert sp.simplify(got_l1 - expected_l1) == 0, f"got {got_l1}"
    assert sp.simplify(got_l2 - expected_l2) == 0, f"got {got_l2}"


def test_layer_projection_of_dsigma_w():
    """∂_σ integrand: ∫_layer_α ∂_σ w dσ = w(σ_{α+1/2}) − w(σ_{α-1/2})
    (interface boundary differences, opaque w)."""
    s = _state_2d()
    sigma = s.zeta_ref
    u_field = s.u.xreplace({s.z: sigma})
    w_field = s.w.xreplace({s.z: sigma})
    u_layer = [sp.Function(f"u_{k+1}", real=True)(s.t, s.x) for k in range(2)]

    test_sys = System("test_sys", s)
    test_sys.add_equation("eq", Expression(sp.Derivative(w_field, sigma),
                                            name="eq"))

    closure = LayerMeanClosure(s, sigma, u_field, u_layer)
    test_sys.eq.apply(closure)

    # Layer 1: w(σ=1/2) - w(σ=0); Layer 2: w(σ=1) - w(σ=1/2).
    w_at = lambda v: w_field.subs(sigma, v)
    assert sp.simplify(test_sys.eq.layer_1.expr - (w_at(sp.S.Half) - w_at(0))) == 0
    assert sp.simplify(test_sys.eq.layer_2.expr - (w_at(1) - w_at(sp.S.Half))) == 0


def test_weights_must_sum_to_one():
    """ValueError if user-supplied weights do not sum to 1."""
    s = _state_2d()
    sigma = s.zeta_ref
    u_field = s.u.xreplace({s.z: sigma})
    u_layer = [sp.Function(f"u_{k+1}", real=True)(s.t, s.x) for k in range(2)]

    bad_weights = [sp.Rational(1, 3), sp.Rational(1, 3)]   # sum = 2/3
    try:
        LayerMeanClosure(s, sigma, u_field, u_layer, weights=bad_weights)
    except ValueError as e:
        assert "sum to 1" in str(e)
        return
    raise AssertionError("expected ValueError for weights summing to 2/3")


def test_u_dsigma_w_uses_layer_mean_in_bulk():
    """∫_layer_α u(σ) · ∂_σ w dσ: the bulk u becomes the layer-mean u_α
    (no ∂_σ u in this term, so the IBP path is not taken — direct
    layer-mean substitution), and w stays opaque at the boundaries.
    """
    s = _state_2d()
    sigma = s.zeta_ref
    u_field = s.u.xreplace({s.z: sigma})
    w_field = s.w.xreplace({s.z: sigma})
    u_layer = [sp.Function(f"u_{k+1}", real=True)(s.t, s.x) for k in range(2)]

    test_sys = System("test_sys", s)
    test_sys.add_equation("eq", Expression(u_field * sp.Derivative(w_field, sigma),
                                            name="eq"))

    closure = LayerMeanClosure(s, sigma, u_field, u_layer)
    test_sys.eq.apply(closure)

    # ∫_0^{1/2} u_1 · ∂_σ w dσ = u_1 · (w(1/2) - w(0)) since u_1 is x-only.
    expected_l1 = u_layer[0] * (w_field.subs(sigma, sp.S.Half) - w_field.subs(sigma, 0))
    assert sp.simplify(test_sys.eq.layer_1.expr - expected_l1) == 0, \
        f"got {test_sys.eq.layer_1.expr}, expected {expected_l1}"


def test_boundary_u_atoms_stay_opaque():
    """``f(σ) · ∂_σ u`` is IBP'd: the boundary atoms u(σ=a), u(σ=b)
    survive opaque (so the user can pick an interface-velocity rule).
    The bulk part reduces to ``-(∂_σ f) · u_α``."""
    s = _state_2d()
    sigma = s.zeta_ref
    u_field = s.u.xreplace({s.z: sigma})
    u_layer = [sp.Function(f"u_{k+1}", real=True)(s.t, s.x) for k in range(2)]

    # f(σ) = σ.  Integrand = σ · ∂_σ u.
    # IBP: ∫_a^b σ · ∂_σ u dσ = [σ u]_a^b - ∫_a^b u dσ.
    #                          = b u(b) - a u(a) - (b-a) u_α.
    test_sys = System("test_sys", s)
    test_sys.add_equation("eq", Expression(sigma * sp.Derivative(u_field, sigma),
                                            name="eq"))

    closure = LayerMeanClosure(s, sigma, u_field, u_layer)
    test_sys.eq.apply(closure)

    # Layer 1 (a=0, b=1/2): (1/2) u(1/2) - 0 - (1/2) u_1 = (u(1/2) - u_1)/2.
    expected_l1 = (u_field.subs(sigma, sp.S.Half) - u_layer[0]) / 2
    assert sp.simplify(test_sys.eq.layer_1.expr - expected_l1) == 0, \
        f"got {test_sys.eq.layer_1.expr}, expected {expected_l1}"
    # Layer 2 (a=1/2, b=1): u(1) - (1/2) u(1/2) - (1/2) u_2.
    expected_l2 = (u_field.subs(sigma, 1) - u_field.subs(sigma, sp.S.Half) / 2
                   - u_layer[1] / 2)
    assert sp.simplify(test_sys.eq.layer_2.expr - expected_l2) == 0, \
        f"got {test_sys.eq.layer_2.expr}, expected {expected_l2}"


def test_layer_sub_leaf_count_for_L_layers():
    """Every leaf the operator is applied to produces exactly L sub-leaves."""
    s = _state_2d()
    sigma = s.zeta_ref
    u_field = s.u.xreplace({s.z: sigma})
    for L in (1, 2, 3, 5):
        u_layer = [sp.Function(f"u_{k+1}", real=True)(s.t, s.x)
                   for k in range(L)]
        test_sys = System(f"test_L{L}", s)
        test_sys.add_equation("eq", Expression(u_field, name="eq"))
        closure = LayerMeanClosure(s, sigma, u_field, u_layer)
        test_sys.eq.apply(closure)
        n_layers = sum(1 for _ in test_sys.eq._node._filter_dict())
        assert n_layers == L, f"L={L}: got {n_layers} sub-leaves"
