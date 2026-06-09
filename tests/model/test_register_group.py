"""``Model.register_group`` → ``SystemModel`` function-group slots.

A derivation registers an explicit definition of a derived operator (the
vertical reconstruction) into the ``interpolate`` group; ``from_model`` parses
it straight into ``interpolate_to_3d``, turning every ``∂_x(state)`` the profile
needs into a runtime gradient aux in ``Qaux``.
"""
import sympy as sp

from zoomy_core import coords as C
from zoomy_core.model.derivation import Model
from zoomy_core.model.models.system_model import (
    SystemModel, register_function_slot, _FUNCTION_SLOTS,
)

t, x = C.t, C.x
zeta = sp.Symbol("zeta", real=True)


def _toy_model():
    h = sp.Function("h", positive=True)(t, x)
    q = sp.Function("q", real=True)
    m = Model(coords=(t, x), parameters={})
    m.add_equation("mass", sp.Derivative(h, t) + sp.Derivative(q(0, t, x), x))
    m.add_equation("mom0",
                   sp.Derivative(q(0, t, x), t)
                   + sp.Derivative(q(0, t, x) ** 2 / h, x))
    return m, h, q


def test_interpolate_group_fills_slot_and_emits_gradient_aux():
    m, h, q = _toy_model()
    # a w-like profile needing ∂_x q_0 and ∂_x h (CoV leaves ∂_x(q_0/h))
    w_recon = (q(0, t, x) / h) * zeta - h * sp.Derivative(q(0, t, x) / h, x) * zeta
    m.register_group("interpolate", 2, w_recon)

    sm = SystemModel.from_model(m, Q=[h, q(0, t, x)])

    assert sm.interpolate_to_3d is not None
    assert len(list(sm.interpolate_to_3d)) == 3          # indices 0,1,2 → 0,0,w
    aux = {str(s) for s in sm.aux_state}
    assert {"dq0dx", "dhdx"} <= aux                       # gradients landed in Qaux


def test_register_group_is_additive_keeps_aux_row():
    m, h, q = _toy_model()
    # an oriented aux relation  w̃ = q_0/h·ζ  (a solved Equation carries _as_relation)
    wt = sp.Function(r"\tilde{w}", real=True)(t, x, zeta)
    m.add_equation("w", sp.Eq(wt, (q(0, t, x) / h) * zeta), group="aux")
    assert "w" in m._aux_names

    m.register_group("interpolate", 2, m.w)               # additive — does NOT consume
    assert "w" in m._aux_names and "w" in m._equations     # w stays in Qaux
    assert 2 in m._function_groups["interpolate"]


def test_register_function_slot_is_open():
    register_function_slot("myslot", "interpolate_to_3d")
    assert _FUNCTION_SLOTS["myslot"] == "interpolate_to_3d"
