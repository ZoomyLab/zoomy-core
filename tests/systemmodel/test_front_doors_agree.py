"""REQ-154 — the two front doors must accept the same set of models.

``SystemModel.from_model(model)`` is documented as "the ONE entry point"
(REQ-143) and builds a SystemModel from a plain ``SWE`` by dispatching on
``_system_model_kind``.  ``to_numerical_system_model(obj)`` is "the single front
door every CORE code printer routes its entry through".  They must agree: a
model ``SystemModel.from_model`` accepts must not be rejected by
``to_numerical_system_model``.
"""

import pytest

from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics.numerical_system_model import (
    NumericalSystemModel,
    to_numerical_system_model,
)
from zoomy_core.systemmodel.system_model import SystemModel


def test_plain_swe_accepted_by_both_front_doors():
    m = SWE(dimension=2, parameters={"g": 9.81})
    # Front door 1 — SystemModel.from_model.
    sm = SystemModel.from_model(m)
    assert isinstance(sm, SystemModel)
    # Front door 2 — the printers' coercion; must NOT raise for a plain SWE
    # (it has no `.system_model` property, only `_system_model_kind`).
    nsm = to_numerical_system_model(m)
    assert isinstance(nsm, NumericalSystemModel)


def test_coercion_is_idempotent_on_systemmodel_and_nsm():
    m = SWE(dimension=2, parameters={"g": 9.81})
    sm = SystemModel.from_model(m)
    nsm = to_numerical_system_model(sm)
    assert isinstance(nsm, NumericalSystemModel)
    # An already-built NSM short-circuits (NSM is-a SystemModel).
    assert to_numerical_system_model(nsm) is nsm


def test_non_model_still_rejected():
    with pytest.raises(TypeError):
        to_numerical_system_model(object())
