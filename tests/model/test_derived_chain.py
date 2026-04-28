"""Smoke tests for the convenience class chain.

Verifies:
  * each level instantiates;
  * caching: two instantiations with the same args reuse the cached
    ``_system``;
  * ``describe(full_hierarchy=True)`` walks the chain back to INS.
"""
import pytest


def test_ins_instantiates():
    from zoomy_core.model.models.derived_chain import INS
    m = INS(dimension=2)
    assert m._system is not None
    assert m._system.name == "INS"


def test_depth_integrated_instantiates():
    from zoomy_core.model.models.derived_chain import DepthIntegrated
    m = DepthIntegrated(dimension=2)
    assert m._system is not None


def test_simplify_stress_inviscid_instantiates():
    from zoomy_core.model.models.derived_chain import SimplifyStress
    m = SimplifyStress(stress="inviscid", dimension=2)
    assert m._system is not None
    assert m._stress_choice == "inviscid"


def test_hydrostatic_drops_z_momentum():
    from zoomy_core.model.models.derived_chain import Hydrostatic
    m = Hydrostatic(stress="inviscid", dimension=2)
    leaves = [".".join(p) for p, _ in m._system.leaves()]
    # z-momentum should be gone after Hydrostatic
    assert not any("momentum.z" in n for n in leaves), \
        f"z-momentum still present: {leaves}"


def test_caching_reuses_system():
    from zoomy_core.model.models.derived_chain import Hydrostatic
    a = Hydrostatic(stress="inviscid", dimension=2)
    b = Hydrostatic(stress="inviscid", dimension=2)
    # Cached: deep-cloned, so different objects but same shape.
    assert id(a._system) != id(b._system)
    assert [".".join(p) for p, _ in a._system.leaves()] == \
           [".".join(p) for p, _ in b._system.leaves()]


def test_describe_full_hierarchy_walks_chain():
    from zoomy_core.model.models.derived_chain import Hydrostatic
    m = Hydrostatic(stress="inviscid", dimension=2)
    text = m.describe(full_hierarchy=True)
    # Each step has its own numbered line: "  1. INS", "  2. DepthIntegrated", …
    for name in ("INS", "DepthIntegrated", "SimplifyStress", "Hydrostatic"):
        assert name in text, f"{name} missing from full-hierarchy describe:\n{text}"
    # Verify ordering on the numbered chain lines.
    chain_lines = [line for line in text.splitlines()
                   if line.strip() and line.lstrip()[0].isdigit()]
    chain_names = [line.split(". ", 1)[1].strip() for line in chain_lines]
    assert chain_names == ["INS", "DepthIntegrated", "SimplifyStress", "Hydrostatic"], \
        f"unexpected chain order: {chain_names}\n\nfull text:\n{text}"


def test_describe_default_is_local():
    from zoomy_core.model.models.derived_chain import Hydrostatic
    m = Hydrostatic(stress="inviscid", dimension=2)
    text = m.describe()
    assert "Hydrostatic" in text
    # Default call doesn't include the full chain
    assert "Derivation chain" not in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
