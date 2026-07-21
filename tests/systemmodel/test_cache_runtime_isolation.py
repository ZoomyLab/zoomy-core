"""The SystemModel build cache must never serve one case's runtime state to
another model — and a WARM build must equal a COLD one.

The cache key is the model's SYMBOLIC identity; parameter VALUES and BC/IC are
deliberately excluded from it (REQ-163: the numbers stay free symbols
end-to-end, so a parameter sweep must not rebuild).  For a long time the stored
ARTIFACT did not honour that: the whole SystemModel was pickled, resolved
numbers and compiled BC kernels included.  Two models differing ONLY in
``parameters=`` / ``boundary_conditions=`` hash to the SAME key, so the first
build's runtime state was served to every later one — in-process AND, through
the user-dir tier, in every later process.  A default-constructed ``SME`` was
handed ``nu=1e-3`` and ``bc_tags ['left', 'right']`` it never declared, and
three checked-in goldens recorded the leak as if it were the truth.

The invariant these tests pin: **warm == cold**.  Everything the cache is
allowed to remember is a function of its key; everything else is re-attached
per build by ``model_builders._attach_runtime_data``.
"""
import pytest

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.model.boundary_conditions import Extrapolation
from zoomy_core.systemmodel import sm_cache
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = pytest.mark.systemmodel


def _closures():
    return [Newtonian(), NavierSlip(), StressFree()]


def _values(sm):
    return {str(k): v for k, v in sm.parameter_values.items()}


def _bc_tags(sm):
    src = getattr(sm, "_bc_source", None)
    if src is None:
        return []
    return sorted(src.boundary_conditions_list_dict.keys())


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """An EMPTY cache dir for this test only, memory tier cleared."""
    monkeypatch.setenv("ZOOMY_CACHE_DIR", str(tmp_path / "zcache"))
    monkeypatch.delenv("ZOOMY_DERIVATION_CACHE", raising=False)
    monkeypatch.delenv("ZOOMY_DERIVATION_REBUILD", raising=False)
    sm_cache._MEMORY.clear()
    yield
    sm_cache._MEMORY.clear()


@pytest.mark.small
def test_case_parameters_do_not_leak_to_the_next_model(isolated_cache):
    """A case's ``parameters=`` numbers must not reach a model that shares its
    key.  Both models below hash IDENTICALLY (values are not in the key)."""
    poison = SME(level=1, dimension=2, closures=_closures(),
                 parameters={"nu": 1e-3})
    clean = SME(level=1, dimension=2, closures=_closures())
    from zoomy_core.systemmodel.model_builders import build_sme
    assert (sm_cache.cache_key(poison, build_sme)
            == sm_cache.cache_key(clean, build_sme)), (
        "premise of this test: parameter VALUES are not part of the key")

    sm_poison = SystemModel.from_model(poison)      # writes the entry
    sm_clean = SystemModel.from_model(clean)        # reads it back
    assert _values(sm_poison)["nu"] == 1e-3         # the case keeps its number
    assert _values(sm_clean)["nu"] == 0.0           # ... and nobody else gets it


@pytest.mark.small
def test_case_boundary_conditions_do_not_leak_to_the_next_model(isolated_cache):
    """Same for BCs: a model declaring none must not inherit the entry's."""
    poison = SME(level=1, dimension=2, closures=_closures(),
                 boundary_conditions=[Extrapolation(tag="left"),
                                      Extrapolation(tag="right")])
    clean = SME(level=1, dimension=2, closures=_closures())

    sm_poison = SystemModel.from_model(poison)
    sm_clean = SystemModel.from_model(clean)
    assert _bc_tags(sm_poison) == ["left", "right"]
    assert _bc_tags(sm_clean) == []


@pytest.mark.small
def test_warm_equals_cold(isolated_cache, monkeypatch):
    """The cache owes the tree exactly this: a build served from a warm cache
    is indistinguishable from ``ZOOMY_DERIVATION_CACHE=0``.

    Poison the entry first, so a leak of EITHER kind shows up as a difference.
    """
    SystemModel.from_model(SME(level=1, dimension=2, closures=_closures(),
                               parameters={"nu": 1e-3},
                               boundary_conditions=[Extrapolation(tag="left")]))

    def build():
        return SystemModel.from_model(
            SME(level=1, dimension=2, closures=_closures()))

    warm = build()
    monkeypatch.setenv("ZOOMY_DERIVATION_CACHE", "0")
    sm_cache._MEMORY.clear()
    cold = build()

    assert _values(warm) == _values(cold)
    assert _bc_tags(warm) == _bc_tags(cold)
    assert [str(s) for s in warm.state] == [str(s) for s in cold.state]
    assert [str(s) for s in warm.aux_state] == [str(s) for s in cold.aux_state]
    assert str(warm.flux) == str(cold.flux)
    assert str(warm.source) == str(cold.source)


@pytest.mark.small
def test_store_refuses_an_entry_carrying_runtime_state(isolated_cache):
    """The guard is a HARD FAIL, not a warning: a poisoned entry must be
    impossible to write and impossible to serve."""
    sm = SystemModel.from_model(
        SME(level=1, dimension=2, closures=_closures(),
            boundary_conditions=[Extrapolation(tag="left")]))
    assert sm._bc_source is not None
    with pytest.raises(AssertionError, match="runtime state"):
        sm_cache.assert_no_runtime_state(sm, "test")
    sm_cache.strip_runtime_state(sm)
    sm_cache.assert_no_runtime_state(sm, "test")     # now clean
