"""REQ-209 — ``fluctuations_are_zero`` is a property of the (model x Riemann)
PAIR, decided from the BUILT numerics, and it must reach the C-family headers.

The two cases below are the ones that killed the predecessor detector
(``generic_c._detect_has_nonconservative_product``, which keyed on the model's
``nonconservative_matrix``).  They fail it in OPPOSITE directions, so no
model-only test can pass both:

* 2-D SWE has a genuinely non-zero model NCP (``B[hu][b] = B[hv][b] = g*h``,
  bed slope, because ``b`` is in the state) — yet under ``PositiveRusanov``
  every emitted fluctuation entry is literal zero, because the Audusse
  reconstruction sets ``b_face = max(b_L, b_R)`` equal on both sides.  A
  model-only detector blocks the conservative fast path here.
* ``Rusanov`` / ``HLL`` / ``HLLC`` never override ``numerical_fluctuations``
  and inherit the identically-zero base — again regardless of the model.
"""

import sympy as sp

from zoomy_core.fvm.riemann_solvers import (
    HLL,
    HLLC,
    NonconservativeRusanov,
    PositiveNonconservativeHLL,
    PositiveRusanov,
    Rusanov,
)
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics.numerical_system_model import (
    NumericalSystemModel,
    ReconstructionSpec,
)
from zoomy_core.transformation.generic_c import CppModel, CppNumerics


def _nsm(model, riemann):
    return NumericalSystemModel.from_model(
        model, riemann=riemann, reconstruction=ReconstructionSpec(order=1))


def _swe():
    return SWE(dimension=2, parameters={"g": 9.81})


def _sme():
    return SME(dimension=2, level=1, parameters={"g": 9.81})


def test_swe_model_ncp_is_nonzero():
    """The premise: SWE's model NCP really is non-zero, so the two tests below
    are not vacuous — they genuinely contradict a model-only detector."""
    nsm = _nsm(_swe(), PositiveRusanov)
    entries = [sp.sympify(e) for e in sp.flatten(nsm.nonconservative_matrix)]
    assert any(e.is_zero is not True for e in entries)


def test_positive_rusanov_erases_swe_fluctuation():
    """Non-zero model NCP, zero emitted fluctuation — the Audusse case."""
    nsm = _nsm(_swe(), PositiveRusanov)
    assert nsm.fluctuations_are_zero is True


def test_conservative_solvers_never_build_a_fluctuation():
    """Rusanov/HLL/HLLC inherit the identically-zero base for ANY model."""
    for riemann in (Rusanov, HLL, HLLC):
        assert _nsm(_swe(), riemann).fluctuations_are_zero is True
        assert _nsm(_sme(), riemann).fluctuations_are_zero is True


def test_nonconservative_solvers_do_build_one():
    for riemann in (NonconservativeRusanov, PositiveNonconservativeHLL):
        assert _nsm(_swe(), riemann).fluctuations_are_zero is False


def test_audusse_cancellation_is_swe_specific():
    """PositiveRusanov zeroes SWE's fluctuation but NOT SME's — the flag must
    distinguish them, which is why it reads the built expression rather than
    keying on the Riemann class."""
    assert _nsm(_swe(), PositiveRusanov).fluctuations_are_zero is True
    assert _nsm(_sme(), PositiveRusanov).fluctuations_are_zero is False


def test_flag_is_reread_not_cached():
    """It must track ``apply()``: the numerics is built FIRST, then an
    operation changes the operator the fluctuation is built from.  A cached
    boolean goes stale in both directions."""
    def _zero_pressure(sm):
        P = sm.hydrostatic_pressure
        sm.hydrostatic_pressure = type(P)(
            [0] * len(list(sp.flatten(P)))).reshape(*P.shape)
    _zero_pressure.name = "zero_pressure"

    nsm = _nsm(_sme(), PositiveRusanov)
    nsm.build_numerics()                       # the staleness window
    assert nsm.fluctuations_are_zero is False
    nsm.apply(_zero_pressure)
    assert nsm.fluctuations_are_zero is True


def test_unknown_collapses_to_has_a_fluctuation():
    """Only a PROVEN structural zero licenses the fast path: skipping a real
    fluctuation is silently-wrong physics, a spurious one costs a branch."""
    nsm = _nsm(_swe(), PositiveRusanov)

    def _probe(definition):
        class _P:
            class functions:
                class numerical_fluctuations:
                    pass
        _P.functions.numerical_fluctuations.definition = definition
        nsm._probe_numerics = lambda: _P()
        return nsm.fluctuations_are_zero

    class _NoSlot:
        class functions:
            pass

    nsm._probe_numerics = lambda: _NoSlot()
    assert nsm.fluctuations_are_zero is False          # slot absent
    assert _probe([]) is False                         # nothing built
    assert _probe([sp.Symbol("a")]) is False           # is_zero is None
    assert _probe([sp.Integer(0), sp.Integer(0)]) is True


def test_flag_is_emitted_into_both_headers():
    """REQ-209 defect 2: the constant must actually reach the backends.
    ``Numerics.H`` is its primary home (it is the (model x riemann) artifact);
    ``Model.H`` carries it alongside ``has_diffusion`` / ``dt_max``."""
    for riemann, expected in ((PositiveRusanov, "true"),
                              (NonconservativeRusanov, "false")):
        nsm = _nsm(_swe(), riemann)
        line = f"static constexpr bool fluctuations_are_zero = {expected};"
        assert line in CppModel(nsm).create_code()
        assert line in CppNumerics(nsm.build_numerics()).create_code()


def test_old_model_only_detector_is_gone():
    """It must not come back: it answered a different question than the one
    the drivers need, and having two answer-holders is how they diverged."""
    assert not hasattr(CppModel, "_detect_has_nonconservative_product")
