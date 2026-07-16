"""REQ-164 — ``Model.system_model`` is REMOVED, not shimmed.

The property raises ``AttributeError`` carrying the migration recipe, so the
18 broken case call sites fail with an instruction instead of a bare
attribute error.  It is NOT a deprecation shim: nothing is returned,
``hasattr`` stays False, and there is no legacy code path (clean-cut rule).
"""
import pytest

from zoomy_core.model.models.swe import SWE


def test_system_model_raises_with_recipe():
    m = SWE(dimension=1)
    with pytest.raises(AttributeError, match=r"SystemModel\.from_model"):
        m.system_model


def test_hasattr_stays_false():
    assert not hasattr(SWE(dimension=1), "system_model")
