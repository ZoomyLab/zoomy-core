"""REQ-181 — split the dry eigenvalue gate into two composable, opt-in ops.

Per the "operations, not options" decision, ``gate_eigenvalues_dry`` is split
into:

* :func:`guard_eigenvalue_powers` (A) — ONLY the always-safe ``Max(., 0)``
  power guard, no dry zeroing; and
* :func:`gate_eigenvalues_dry` (B) — the dry ``conditional(h > eps, ., 0)``
  zeroing, which carries A internally (a branchless ``conditional`` computes
  both arms, so the wet branch must stay real — REQ-74).

Neither is an NSM default any more: the only depth default is
``desingularize_hinv()``.  Cases opt into the gate via ``extra_operations``.

The split is BYTE-IDENTICAL to the pre-split combined op — both the single
``gate_eigenvalues_dry()`` and the explicit composition
``[guard_eigenvalue_powers(), gate_eigenvalues_dry()]`` reproduce it (the guard
is idempotent), so no backend's generated code changes when it opts in.
"""
import sympy as sp

from zoomy_core.model.models import SME, SWE
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.model.boundary_conditions import Extrapolation
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.systemmodel.operations import (
    desingularize_hinv,
    guard_eigenvalue_powers,
    gate_eigenvalues_dry,
    _guard_eigenvalue_expr,
    _wet_dry_eps,
)
from zoomy_core.numerics import NumericalSystemModel


def _build_swe():
    return SystemModel.from_model(SWE(
        boundary_conditions=[Extrapolation("left"), Extrapolation("right")]))


def _build_sme(level=1):
    return SystemModel.from_model(SME(
        level=level, parameters={"nu": 1e-3, "lambda_s": 0.0},
        closures=[Newtonian(), NavierSlip(), StressFree()],
        boundary_conditions=[Extrapolation("left"), Extrapolation("right")]))


def _desing(build):
    """Depth model with the KP desingularization already applied — the state
    every depth NSM reaches before the (now opt-in) gate/guard."""
    sm = build()
    sm.apply(desingularize_hinv())
    return sm


def _flat(sm):
    return list(sp.flatten(sm.eigenvalues))


def _srepr(sm):
    return [sp.srepr(sp.sympify(e)) for e in _flat(sm)]


def _h(sm):
    return next(s for s in sm.state if str(s) == "h")


# (a) guard-only: Max(.,0) present, NO conditional zeroing --------------------

def test_guard_only_has_max_no_conditional():
    for build in (_build_swe, lambda: _build_sme(1)):
        sm = _desing(build)
        raw = _flat(sm)
        sm.apply(guard_eigenvalue_powers())
        guarded = _flat(sm)
        # Max appeared somewhere (the sqrt(h**k) wet-branch got floored) ...
        assert any(sp.sympify(e).has(sp.Max) for e in guarded)
        # ... and NO dry gate was introduced.
        cond = sp.Function("conditional")
        assert not any(sp.sympify(e).has(cond) for e in guarded)
        # guard is a no-op on already-guarded input (idempotent).
        sm.apply(guard_eigenvalue_powers())
        assert _srepr(sm) == [sp.srepr(sp.sympify(e)) for e in guarded]


# (b) gate: zeros lambda at h < eps -------------------------------------------

def test_gate_zeros_eigenvalues_when_dry():
    cond = sp.Function("conditional")
    for build in (_build_swe, lambda: _build_sme(1)):
        sm = _desing(build)
        h = _h(sm)
        eps = _wet_dry_eps(sm)
        sm.apply(gate_eigenvalues_dry())
        for e in _flat(sm):
            e = sp.sympify(e)
            assert isinstance(e, cond)
            gate_cond, wet, dry = e.args
            # condition is exactly the wet test h > eps ...
            assert gate_cond == (h > eps)
            # ... and the dry branch is identically zero.
            assert dry == 0
        # numerically: a dry state (h < eps) makes the gate condition False,
        # so a branchless lowering selects the 0 branch.
        subs = {h: sp.Float(1e-12)}          # < eps 1e-8 -> dry
        if isinstance(eps, sp.Symbol):
            subs[eps] = sp.Float(1e-8)
        assert (h > eps).subs(subs) == sp.false


# (c) composition == old combined (byte identity) on SWE + SME(1) -------------

def _old_combined_reference(build):
    """Reconstruct the pre-split combined op's output independently:
    ``conditional(h > eps, guard(e), 0)`` over the desingularized eigenvalues."""
    sm = _desing(build)
    h = _h(sm)
    eps = _wet_dry_eps(sm)
    cond = sp.Function("conditional")
    return [sp.srepr(cond(h > eps, _guard_eigenvalue_expr(e, h), sp.S.Zero))
            for e in _flat(sm)]


def test_single_gate_and_composition_equal_old_combined():
    for build in (_build_swe, lambda: _build_sme(1)):
        ref = _old_combined_reference(build)

        sm = _desing(build)
        sm.apply(gate_eigenvalues_dry())
        assert _srepr(sm) == ref, "single gate() != old combined op"

        sm = _desing(build)
        sm.apply(guard_eigenvalue_powers())
        sm.apply(gate_eigenvalues_dry())
        assert _srepr(sm) == ref, "[guard, gate] != old combined op"


# (d) NO eigenvalue guard in the NSM defaults ---------------------------------

def test_nsm_default_operations_carry_no_eigenvalue_guard():
    """REQ-181: "we do not make this eigenvalues guard a default" — neither
    ``guard_eigenvalue_powers`` nor ``gate_eigenvalues_dry`` may appear
    unless the caller opts in via ``eigenvalue_guard=``.

    The depth default is still ``desingularize_hinv``; ``normalize_face_normal``
    joined the list in REQ-208 item (2) and is unrelated to the guard question
    this test pins.
    """
    for build in (_build_swe, lambda: _build_sme(1)):
        nsm = NumericalSystemModel.from_system_model(build())
        names = [getattr(op, "name", None) for op in nsm.default_operations()]
        assert "guard_eigenvalue_powers" not in names, names
        assert "gate_eigenvalues_dry" not in names, names
        assert names == ["normalize_face_normal", "desingularize_hinv"], names
