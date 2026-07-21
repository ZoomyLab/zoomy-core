"""``implicit_mode`` — WHICH implicit-stage path a system needs, decided in the
NSM from the OPERATORS.

The user's ruling: ``implicit_source`` and ``implicit_diffusion`` collapse into
ONE implicit stage solve, so the scheme is a genuine IMEX-ARK rather than a Lie
split — with a FAST PATH when the implicit operator is purely a cell-local
source, which is then N independent small nonlinear solves instead of one
global system.  And: "the identification of which path we take should be
happening in the Zoomy core, hence the numerical system model stage."

So the identification is an NSM property, following the ``fluctuations_are_zero``
precedent (``test_fluctuations_are_zero.py``): derived from the built operators,
re-read rather than cached, emitted to the printers.

The case that makes it worth deriving rather than declaring is
``test_source_reading_a_gradient_aux_is_coupled``: "source" names WHERE a term
sits in the equation, not its STENCIL.  A source that reads an LSQ-gradient aux
row couples neighbouring cells exactly as a diffusion operator does, and a
hand-set ``implicit_source=True`` flag would send it down the per-cell fast path
and invert the wrong operator.
"""

import sympy as sp
import pytest

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.models.closures import (
    NavierSlip,
    Newtonian,
    NewtonianInPlane,
    StressFree,
)
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics.numerical_system_model import (
    IMPLICIT_MODE_COUPLED,
    IMPLICIT_MODE_LOCAL_SOURCE,
    IMPLICIT_MODE_NONE,
    NumericalSystemModel,
    ReconstructionSpec,
    implicit_stage_mode,
)
from zoomy_core.transformation.generic_c import CppModel

pytestmark = [pytest.mark.nsm, pytest.mark.small, pytest.mark.gate]


def _nsm(model):
    return NumericalSystemModel.from_model(
        model, reconstruction=ReconstructionSpec(order=1))


def _friction_only():
    """SWE — Manning bed friction in the implicit-source slot, no diffusion."""
    return _nsm(SWE(dimension=1, parameters={"g": 9.81}))


def _implicit_diffusion():
    """SME(1) + ``NewtonianInPlane`` — the RETAINED in-plane deviatoric stress
    is routed into ``diffusion_matrix`` (the ``implicit_diffusion`` slot) as a
    horizontal eddy viscosity.  Also carries the NavierSlip friction source, so
    this is simultaneously the BOTH-slots case below."""
    return _nsm(SME(dimension=2, level=1,
                    parameters={"g": 9.81, "nu": 1e-3},
                    closures=[Newtonian(), NavierSlip(), StressFree(),
                              NewtonianInPlane()]))


def _swe_with_implicit_diffusion():
    """The friction-only SWE with a live ``diffusion_matrix`` switched on.

    A CONTROLLED both-slots system: its implicit source is the same cell-local
    Manning friction as ``_friction_only``, so zeroing the diffusion drops it
    cleanly back to the fast path.  (``_implicit_diffusion`` cannot serve here —
    its source reads gradient aux of its own, so it stays coupled for a SECOND
    reason and the two causes would be indistinguishable.)"""
    nsm = _friction_only()

    def _add_diffusion(sm):
        D = sm.diffusion_matrix
        rows = [0] * len(list(sp.flatten(D)))
        rows[0] = sp.Symbol("nu", real=True)
        sm.diffusion_matrix = type(D)(rows).reshape(*D.shape)
    _add_diffusion.name = "add_diffusion"

    nsm.apply(_add_diffusion)
    nsm.diffusion.enabled = True
    return nsm


def _zero_source(sm):
    S = sm.source
    sm.source = type(S)([0] * len(list(sp.flatten(S)))).reshape(*S.shape)


_zero_source.name = "zero_source"


# ── the three modes ────────────────────────────────────────────────────────


def test_friction_only_reports_the_local_fast_path():
    """A cell-local source and nothing else: every entry is a function of THIS
    cell's state and pointwise aux, so the stage is N independent small solves."""
    nsm = _friction_only()
    # premise — the implicit-source slot really is live, so this is not vacuous
    assert any(sp.sympify(e).is_zero is not True
               for e in sp.flatten(nsm.source))
    assert nsm.implicit_mode == IMPLICIT_MODE_LOCAL_SOURCE


def test_implicit_diffusion_reports_coupled():
    """A live ``diffusion_matrix`` is a second-derivative operator; it reaches
    neighbours by construction, so there is no per-cell path."""
    nsm = _implicit_diffusion()
    assert any(sp.sympify(e).is_zero is not True
               for e in sp.flatten(nsm.diffusion_matrix))
    nsm.apply(_zero_source)                    # diffusion ALONE
    assert not any(sp.sympify(e).is_zero is not True
                   for e in sp.flatten(nsm.source))
    assert nsm.implicit_mode == IMPLICIT_MODE_COUPLED


def test_both_slots_report_coupled_not_two_solves():
    """THE ruling: a model carrying an implicit source AND implicit diffusion
    gets ONE coupled stage solve carrying both operators — never two separate
    solves.  The Lie split is not representable in this vocabulary: there is no
    mode meaning "do the source solve, then the diffusion solve"."""
    nsm = _swe_with_implicit_diffusion()
    live = lambda M: any(sp.sympify(e).is_zero is not True
                         for e in sp.flatten(M))
    assert live(nsm.source) and live(nsm.diffusion_matrix)   # both, genuinely
    assert nsm.implicit_mode == IMPLICIT_MODE_COUPLED
    # and the vocabulary itself admits no split mode
    assert IMPLICIT_MODE_LOCAL_SOURCE != IMPLICIT_MODE_COUPLED
    assert set((IMPLICIT_MODE_NONE, IMPLICIT_MODE_LOCAL_SOURCE,
                IMPLICIT_MODE_COUPLED)) == {"none", "local_source", "coupled"}


def test_empty_implicit_slots_report_none():
    nsm = _friction_only()
    nsm.apply(_zero_source)
    assert nsm.implicit_mode == IMPLICIT_MODE_NONE


# ── why it must be DERIVED, not declared ───────────────────────────────────


def test_source_reading_a_gradient_aux_is_coupled():
    """"Source" is a position in the equation, NOT a stencil.

    SME(1) registers LSQ-gradient aux rows (``dhdx``, ``dq0dx``, …) but its
    friction source does not read them — so it takes the fast path.  Inject one
    gradient aux into the source and the SAME slot must now report ``coupled``:
    the operator reaches neighbouring cells and inverting it per cell would
    invert the wrong thing.  A hand-declared ``implicit_source=True`` cannot see
    this difference; reading the operator can."""
    nsm = _nsm(SME(dimension=2, level=1,
                   parameters={"g": 9.81, "nu": 1e-3},
                   closures=[Newtonian(), NavierSlip(), StressFree()]))
    assert nsm.implicit_mode == IMPLICIT_MODE_LOCAL_SOURCE

    grad_aux = [e["aux_symbol"] for e in nsm.aux_registry
                if e.get("kind") == "derivative"]
    assert grad_aux, "premise: this model registers derivative aux rows"

    def _read_a_gradient(sm):
        S = sm.source
        rows = list(sp.flatten(S))
        rows[-1] = sp.sympify(rows[-1]) + grad_aux[0]
        sm.source = type(S)(rows).reshape(*S.shape)
    _read_a_gradient.name = "read_a_gradient"

    nsm.apply(_read_a_gradient)
    assert nsm.implicit_mode == IMPLICIT_MODE_COUPLED


def test_mode_is_reread_not_cached():
    """It must track ``apply()`` in BOTH directions, like
    ``fluctuations_are_zero`` — the implicit slots move long after
    construction, and a cached string goes stale."""
    nsm = _swe_with_implicit_diffusion()
    assert nsm.implicit_mode == IMPLICIT_MODE_COUPLED

    def _zero_diffusion(sm):
        D = sm.diffusion_matrix
        sm.diffusion_matrix = type(D)(
            [0] * len(list(sp.flatten(D)))).reshape(*D.shape)
    _zero_diffusion.name = "zero_diffusion"

    nsm.apply(_zero_diffusion)
    assert nsm.implicit_mode == IMPLICIT_MODE_LOCAL_SOURCE
    nsm.apply(_zero_source)
    assert nsm.implicit_mode == IMPLICIT_MODE_NONE


def test_disabled_diffusion_does_not_force_the_global_solve():
    """``DiffusionSpec.enabled`` decides whether the diffusion stage runs at
    all; a disabled one must not drag the system onto the coupled path."""
    nsm = _swe_with_implicit_diffusion()
    assert nsm.implicit_mode == IMPLICIT_MODE_COUPLED
    nsm.diffusion.enabled = False
    assert nsm.implicit_mode == IMPLICIT_MODE_LOCAL_SOURCE


# ── UNKNOWN collapses to the SAFE side ─────────────────────────────────────


def test_unknown_locality_collapses_to_coupled():
    """The global stage solve is always CORRECT; the fast path is correct only
    where locality is PROVEN.  So an underived ``aux_registry`` — no evidence
    that any aux is cell-local — must report ``coupled`` for a source that
    touches aux at all, mirroring ``fluctuations_are_zero``'s UNKNOWN -> False."""
    nsm = _friction_only()
    assert nsm.implicit_mode == IMPLICIT_MODE_LOCAL_SOURCE
    assert nsm.aux_state, "premise: the friction source reads aux (hinv)"
    nsm.aux_registry = None                       # locality no longer provable
    assert nsm.implicit_mode == IMPLICIT_MODE_COUPLED


def test_an_undecidable_operator_entry_counts_as_live():
    """``is_zero is None`` is not a zero.  Dropping a real implicit term is
    silently-wrong physics; a spurious stage costs only work."""
    class _Fake:
        diffusion_matrix = None
        aux_registry = []
        aux_state = []
        source = ZArray([[sp.Symbol("undecidable")]])
    assert implicit_stage_mode(_Fake()) == IMPLICIT_MODE_LOCAL_SOURCE
    _Fake.source = ZArray([[sp.Integer(0)]])
    assert implicit_stage_mode(_Fake()) == IMPLICIT_MODE_NONE


# ── it must reach the backends ─────────────────────────────────────────────


def test_mode_is_emitted_into_the_header():
    """Deciding it in the core is only half the ruling — a backend has to be
    able to READ the decision instead of re-deriving it from its own view of
    the slots (which is exactly how ``fluctuations_are_zero``'s predecessor
    diverged from the property)."""
    assert ('static constexpr const char* implicit_mode = "local_source";'
            in CppModel(_friction_only()).create_code())
    assert ('static constexpr const char* implicit_mode = "coupled";'
            in CppModel(_swe_with_implicit_diffusion()).create_code())
