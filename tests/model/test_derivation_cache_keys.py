"""Stage-1 keying foundation for the derivation cache (task 0018 / REQ-10).

These pin the cache-key contract: deterministic, cosmetic-insensitive,
content-sensitive (so an altered op parameter ⇒ MISS), and lambda transforms
keyed by symbolic effect rather than object identity.
"""
import sympy as sp

from zoomy_core.model.derivation.cache_keys import (
    cache_key, model_spec_key, op_sequence_key,
)
from zoomy_core.model.derivation.model import ResolveModes
from zoomy_core.model.derivation.projection import ExpandSums, EvaluateSums
from zoomy_core.model.derivation.operations import ChangeOfVariables
from zoomy_core.model.models.vam import VAM
from zoomy_core.model.models.sme import SME

_h = sp.Function("h")(sp.Symbol("t"), sp.Symbol("x"))


def test_cache_key_deterministic_across_instances():
    a = ResolveModes(index=sp.Symbol("k"), modes=[0, 1])
    b = ResolveModes(index=sp.Symbol("k"), modes=[0, 1])
    assert cache_key(a) == cache_key(b)
    assert len(cache_key(a)) == 64  # sha256 hex


def test_cache_key_ignores_cosmetic_name():
    a = ResolveModes(index=sp.Symbol("k"), modes=[0, 1])
    b = ResolveModes(index=sp.Symbol("k"), modes=[0, 1], name="renamed")
    assert cache_key(a) == cache_key(b)


def test_cache_key_sensitive_to_op_params():
    base = ResolveModes(index=sp.Symbol("k"), modes=[0, 1])
    assert cache_key(base) != cache_key(
        ResolveModes(index=sp.Symbol("k"), modes=[0, 1, 2]))   # modes
    assert cache_key(base) != cache_key(
        ResolveModes(index=sp.Symbol("l"), modes=[0, 1]))      # index


def test_change_of_variables_keyed_by_symbolic_effect():
    a = ChangeOfVariables("U", "q", lambda qi: qi / _h)
    b = ChangeOfVariables("U", "q", lambda qi: qi / _h)         # same effect
    c = ChangeOfVariables("U", "q", lambda qi: qi / (2 * _h))   # diff effect
    assert cache_key(a) == cache_key(b)
    assert cache_key(a) != cache_key(c)


def test_op_sequence_order_matters():
    assert op_sequence_key([ExpandSums(), EvaluateSums()]) \
        != op_sequence_key([EvaluateSums(), ExpandSums()])
    assert cache_key(ExpandSums()) != cache_key(EvaluateSums())


def test_model_spec_key_discriminates_declarative_spec():
    assert model_spec_key(VAM(level=1, dimension=3)) \
        == model_spec_key(VAM(level=1, dimension=3))
    assert model_spec_key(VAM(level=1, dimension=3)) \
        != model_spec_key(VAM(level=2, dimension=3))            # level
    assert model_spec_key(VAM(level=1, dimension=2)) \
        != model_spec_key(VAM(level=1, dimension=3))            # dimension
    assert model_spec_key(VAM(level=1, dimension=3)) \
        != model_spec_key(SME(level=1, dimension=3))            # class


def test_model_spec_key_excludes_bc_ic_by_default():
    from zoomy_core.model.boundary_conditions import Wall
    plain = VAM(level=1, dimension=3)
    walled = VAM(level=1, dimension=3,
                 boundary_conditions=[Wall("wall", on="momentum")])
    assert model_spec_key(plain) == model_spec_key(walled)
    assert model_spec_key(plain, include_bc_ic=True) \
        != model_spec_key(walled, include_bc_ic=True)
