"""The vertical-velocity (w) reconstruction slot of ``interpolate_to_3d`` must
be PROPER: no raw spatial ``Derivative`` atoms (so every backend — incl. the
jax printer — can lower it), with the derivatives exposed as registered aux
that the solver's shared derivative-aux walk computes.

Before this fix the reconstruction gradient→aux conversion in
``_attach_function_groups`` was x-only (`dhdx`) and never registered in
``aux_registry``: in 2-D the w slot (``w = −∫(∂ₓu + ∂_y v) dz``) kept raw
``Derivative(_, y)`` atoms, which the jax printer cannot lambdify, so
``jax_runtime`` DROPPED the whole ``interpolate_to_3d`` slot.  The fix makes the
conversion dimension-complete (x AND y AND z) and registers each gradient aux
(kind='derivative', target, multi_index) so ``Solver._walk_derivative_aux``
fills it.
"""
import sympy as sp
import pytest

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel.system_model import SystemModel


def _sme(dim):
    return SystemModel.from_model(SME(level=1, dimension=dim,
               closures=[Newtonian(), NavierSlip(), StressFree()]))


def test_w_slot_has_no_raw_derivative_atoms_2d():
    """2-D w slot (index 4) carries zero raw Derivative atoms — every ∂ became
    a gradient-aux symbol — so interpolate_to_3d lowers on all backends."""
    sm = _sme(3)                       # dimension=3 → 2 horizontals (2-D map)
    rows = [sp.sympify(r) for r in sm.interpolate_to_3d]
    assert rows[4].atoms(sp.Derivative) == set()           # w slot clean
    assert sum(len(r.atoms(sp.Derivative)) for r in rows) == 0  # whole vector clean


def test_2d_reconstruction_gradients_are_registered():
    """The 2-D w slot needs ∂_y of h, b, q_y_* — they must be registered aux
    (so Solver._walk_derivative_aux computes them), not just appended to
    aux_state."""
    sm = _sme(3)
    reg = {e["name"]: e for e in (getattr(sm, "aux_registry", None) or [])
           if e["kind"] == "derivative"}
    # y-derivatives (multi_index (0,1)) must be present and target the state
    y_entries = {n: e for n, e in reg.items() if e["multi_index"] == (0, 1)}
    assert y_entries, "no ∂_y gradient aux registered for the 2-D w slot"
    for e in y_entries.values():
        assert e["target_kind"] == "state"
        assert "state_index" in e
    # specifically h and b enter the w slot via ∂_y
    names = set(reg)
    assert {"dhdy", "dbdy"} <= names, names


def test_interpolate_lowers_on_jax_2d():
    """All 6 interpolate rows lambdify on jax (the slot is NOT dropped)."""
    jax = pytest.importorskip("jax")
    sm = _sme(3)
    syms = (list(sm.state) + list(sm.aux_state) + list(sm.parameters)
            + [sp.Symbol("z")])
    for r in sm.interpolate_to_3d:
        sp.lambdify(syms, sp.sympify(r), "jax")   # raises if a Derivative survived


def test_1d_unchanged_x_only():
    """1-D models keep the x-only gradient aux (no spurious y/z), so existing
    1-D behaviour is byte-identical."""
    sm = _sme(2)                       # dimension=2 → 1 horizontal (1-D)
    reg = [e for e in (getattr(sm, "aux_registry", None) or [])
           if e["kind"] == "derivative"]
    for e in reg:
        assert e["multi_index"] == (1,)            # x-only in 1-D


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
