"""REQ-160 — GENERAL ``quadrature_order`` for all numerically-resolved integrals.

* ``quadrature_order`` is a BASE-LEVEL param inherited by every moment model
  (SME, MLSME, MLVAM, VAM) — not a one-off on SME.
* MLSME can now run a non-polynomial (Bingham) closure: the SimpleNamespace
  parameter-namespace bug is fixed (``tau_y`` resolves) and the shared order is
  routed into each layer's Galerkin reduction.
* MLSME accepts a PER-LAYER degree ``level=[2, 0]`` — bottom quadratic-sheared,
  top a stress-free deg-0 plug.
* Regression: SME(2)+Bingham+quad and MLSME(Newtonian) still build.
"""
import param
import pytest

from zoomy_core.model.basemodel import Model as BaseModel
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.ml_sme import MLSME
from zoomy_core.model.models.vam import VAM
from zoomy_core.model.models.ml_vam import MLVAM
from zoomy_core.model.models.closures import (
    Bingham, NavierSlip, StressFree, Newtonian)
from zoomy_core.systemmodel.system_model import SystemModel

_VISCOPLASTIC = dict(nu=0.1, lambda_s=1.0, tau_y=0.5, eps_reg=1e-3)


def test_quadrature_order_is_base_level():
    """A single BaseModel param, inherited by every moment model."""
    assert "quadrature_order" in BaseModel.param
    for cls in (SME, MLSME, VAM, MLVAM):
        assert "quadrature_order" in cls.param, cls.__name__
    # accepted as a ctor kwarg on a model that previously rejected it (MLSME).
    assert int(MLSME(n_layers=2, level=1, quadrature_order=12).quadrature_order) == 12


def test_mlsme_bingham_builds_with_quadrature():
    """REQ-160: MLSME + a viscoplastic (non-polynomial) closure builds — the
    SimpleNamespace ``tau_y`` bug and the missing quadrature route are fixed."""
    m = MLSME(n_layers=2, level=1, quadrature_order=12,
              closures=[Bingham(), NavierSlip(), StressFree()],
              parameters=_VISCOPLASTIC)
    sm = SystemModel.from_model(m)
    assert [str(s) for s in sm.state] == ["b", "h", "q_1_0", "q_1_1", "q_2_0", "q_2_1"]
    # the yield stress reached the compiled system (non-polynomial term survived):
    # tau_y appears somewhere in the source residuals.
    import sympy as sp
    src_syms = set()
    for e in sm.source:
        src_syms |= {str(s) for s in sp.sympify(e).free_symbols}
    assert "tau_y" in src_syms


def test_mlsme_per_layer_level_plug_top():
    """REQ-160(iii): ``level=[2, 0]`` → bottom quadratic (3 moments), top plug
    (1 moment).  The assembled state carries the per-layer moment counts."""
    m = MLSME(n_layers=2, level=[2, 0], quadrature_order=12,
              closures=[Bingham(), NavierSlip(), StressFree()],
              parameters=_VISCOPLASTIC)
    sm = SystemModel.from_model(m)
    assert [str(s) for s in sm.state] == \
        ["b", "h", "q_1_0", "q_1_1", "q_1_2", "q_2_0"]


def test_mlsme_per_layer_level_length_checked():
    with pytest.raises(ValueError):
        SystemModel.from_model(MLSME(n_layers=3, level=[2, 0]))


def test_sme_bingham_quad_regression():
    """SME(2)+Bingham+quad12 unchanged (still builds)."""
    m = SME(level=2, quadrature_order=12,
            closures=[Bingham(), NavierSlip(), StressFree()],
            parameters=_VISCOPLASTIC)
    sm = SystemModel.from_model(m)
    assert [str(s) for s in sm.state] == ["b", "h", "q_0", "q_1", "q_2"]


def test_mlsme_newtonian_regression():
    """MLSME(Newtonian) uniform-level path unchanged (byte-identical state)."""
    m = MLSME(n_layers=2, level=1,
              closures=[Newtonian(), NavierSlip(), StressFree()])
    sm = SystemModel.from_model(m)
    assert [str(s) for s in sm.state] == ["b", "h", "q_1_0", "q_1_1", "q_2_0", "q_2_1"]
