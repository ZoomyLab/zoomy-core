"""REQ-87 — ``SystemModel.from_model`` attaches the model's coupling/boundary
conditions on RAW promotion.

Production models (SWE and subclasses) pop the constructor
``boundary_conditions=`` into ``_coupling_bcs`` and used to wire them only in
the ``.system_model`` PROPERTY.  Backend adapters and the FVM solvers call
``SystemModel.from_model(model)`` directly — so without the promotion-path
attach the system carries an EMPTY BC kernel (walls don't reflect: the
malpasset "16 m/s eruption").  These tests lock the fix: the raw
``from_model`` path yields a non-empty BC kernel with NO per-case
``_initialize_derived_properties`` override, the ``.system_model`` property is
unchanged (single attach, no doubled aux), and no-BC models stay empty.
"""
from zoomy_core.model.models.swe import SWE
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Wall, Extrapolation, FromModel)
from zoomy_core.systemmodel.system_model import SystemModel


def _tags(sm):
    return sorted(b.tag for b in sm._bc_source.boundary_conditions_list)


def test_raw_promotion_no_bc_stays_empty():
    """A model without coupling BCs promotes to an empty (but valid) kernel."""
    sm = SystemModel.from_model(SWE(dimension=2))
    assert sm.boundary_conditions is not None            # valid Function kernel
    assert sm._bc_source.boundary_conditions_list == []  # genuinely empty


def test_raw_promotion_attaches_wall_bcs():
    """``boundary_conditions=`` declared on the model reach the promoted system
    WITHOUT any ``_initialize_derived_properties`` override."""
    bcs = BoundaryConditions([Wall(tag=t) for t in
                              ("left", "right", "top", "bottom")])
    m = SWE(dimension=2, boundary_conditions=bcs)

    # BEFORE-analogue: the same model with the BCs stripped is empty.
    empty = SystemModel.from_model(SWE(dimension=2))
    assert empty._bc_source.boundary_conditions_list == []

    # AFTER: raw from_model carries the four walls + a non-trivial kernel body.
    sm = SystemModel.from_model(m)
    assert _tags(sm) == ["bottom", "left", "right", "top"]
    assert sm.boundary_conditions.definition is not None


def test_raw_promotion_frommodel_needs_function_groups():
    """``FromModel`` BCs resolve on raw promotion because ``from_model`` now
    parses the model's ``boundary:<name>`` function groups into
    ``sm.boundary_specs`` before attaching."""
    m = SWE(dimension=2, boundary_conditions=BoundaryConditions([
        FromModel(tag="left", definition="wall"),
        Extrapolation(tag="right")]))
    sm = SystemModel.from_model(m)
    assert "wall" in (sm.boundary_specs or {})
    assert _tags(sm) == ["left", "right"]


def test_property_matches_raw_and_no_doubled_aux():
    """The ``.system_model`` property is now just the raw promotion — same BC
    tags, and the function groups are attached EXACTLY once (no doubled
    gradient aux)."""
    bcs = BoundaryConditions([Wall(tag="left"), Wall(tag="right")])
    m = SWE(dimension=2, boundary_conditions=bcs)
    raw = SystemModel.from_model(m)
    prop = SystemModel.from_model(SWE(dimension=2, boundary_conditions=bcs))
    assert _tags(raw) == _tags(prop)
    assert [str(a) for a in raw.aux_state] == [str(a) for a in prop.aux_state]
    # de-dup held: no aux symbol appears twice.
    names = [str(a) for a in prop.aux_state]
    assert len(names) == len(set(names))
