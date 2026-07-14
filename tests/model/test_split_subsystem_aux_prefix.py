"""REQ-151 — the Chorin splitter must keep the PARENT's aux layout a PREFIX of
every sub-system's, so a plain-Symbol aux the parent owns (the KP-desingularized
``hinv``) is carried, its per-cell formula travels with it, and no sub-system
references an aux it does not declare.

Before the fix ``_build_subsystem`` populated each sub-system's ``aux_state``
purely from ``expose_aux_atoms`` (which only routes Function / Derivative atoms),
so ``hinv`` — a plain Symbol — was dropped; the predictor then referenced an aux
row it did not declare (``NameError`` on jax / an out-of-range aux index on
numpy) and the parent's BC kernel indexed the wrong row.
"""

import sympy as sp

from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.model.models.vam import VAM
from zoomy_core.systemmodel.operations import desingularize_hinv
from zoomy_core.systemmodel.system_model import SystemModel


def _split_vam():
    m = VAM(level=1, dimension=3, closures=[Newtonian(), StressFree()],
            parameters={"g": 9.81, "rho": 1.0})
    sm = SystemModel.from_model(m)
    sm.apply(desingularize_hinv())          # registers hinv (a plain Symbol)
    assert any(str(a) == "hinv" for a in sm.aux_state)
    split = m.chorin_split(sp.Symbol("dt", positive=True), system_model=sm)
    return sm, split


def test_parent_aux_is_a_prefix_of_predictor():
    sm, split = _split_vam()
    parent = [str(a) for a in sm.aux_state]
    pred = [str(a) for a in split.SM_pred.aux_state]
    assert pred[: len(parent)] == parent


def test_predictor_carries_hinv_and_its_formula():
    _, split = _split_vam()
    pred = split.SM_pred
    assert any(str(a) == "hinv" for a in pred.aux_state)
    # The per-cell aux formula (hinv = kp_hinv(h)) travels with the rows so the
    # solver can actually COMPUTE hinv (defect D) — not sit it at 0.
    assert pred.update_aux_variables is not None
    assert pred.update_aux_variables.shape[0] == len(pred.aux_state)


def test_predictor_references_no_undeclared_aux():
    _, split = _split_vam()
    pred = split.SM_pred
    aux = {str(a) for a in pred.aux_state}
    st = {str(x) for x in pred.state}
    allowed = aux | st | {"g", "rho", "nu", "lambda_s", "dt", "t", "x", "y", "z"}
    need = set()
    for op in (pred.flux, pred.hydrostatic_pressure):
        for entry in sp.flatten(op):
            need |= {str(s) for s in sp.sympify(entry).free_symbols}
    assert not (need - allowed), f"predictor references undeclared: {need - allowed}"


def test_every_subsystem_passes_the_declared_symbol_guard():
    # The guard runs inside _build_subsystem; reaching here (no ValueError from
    # chorin_split) proves press + corr sub-systems also declare every symbol.
    _, split = _split_vam()
    for sub in (split.SM_pred, split.SM_press, split.SM_corr):
        assert sub.aux_state is not None
