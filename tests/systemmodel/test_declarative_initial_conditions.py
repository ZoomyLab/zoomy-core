"""REQ-103: a declarative model's ``initial_conditions`` must ride onto the
built ``SystemModel`` (and thus through ``to_numerical_system_model``), so
adapters can run a model-card default without re-attaching the IC by hand.

The IC lives on the user-facing model and is threaded to
``SystemModel._from_derivation_model`` as ``canonical_source``; before the fix
that branch dropped it and ``sm.initial_conditions`` was ``None``.
"""

from __future__ import annotations

import pytest

from zoomy_core.model.models import SME, MLSME
from zoomy_core.numerics.numerical_system_model import to_numerical_system_model


@pytest.mark.parametrize("factory", [
    lambda: SME(level=2, dimension=2),
    lambda: MLSME(n_layers=2, level=1, dimension=2),
])
def test_declarative_system_model_carries_initial_conditions(factory):
    model = factory()
    sm = model.system_model
    assert sm.initial_conditions is not None, \
        "declarative .system_model dropped initial_conditions (REQ-103)"
    # identity: it is the model's own IC instance, not a fresh default
    assert sm.initial_conditions is model.initial_conditions


def test_initial_conditions_propagate_through_coercion():
    """The intended front door ``to_numerical_system_model`` (honours
    ``.system_model``) carries the IC end-to-end onto ``nsm.sm``."""
    model = SME(level=1, dimension=2)
    ic = model.initial_conditions
    assert ic is not None
    nsm = to_numerical_system_model(model)
    assert nsm.sm.initial_conditions is ic
