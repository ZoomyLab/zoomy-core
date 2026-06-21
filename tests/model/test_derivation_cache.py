"""Function-level derivation cache (task 0018 / REQ-10, stage 2).

Pins the ``@derivation_cache`` contract: a 2nd call with unchanged source + args
is a HIT that does NOT re-run the body (so the heavy ``ResolveModes`` pass never
runs); changed args / version ⇒ MISS; declarative-model args key on spec;
``verify=True`` re-runs and asserts srepr-identity.
"""
import sympy as sp

from zoomy_core.model.derivation.derivation_cache import (
    derivation_cache, clear_derivation_cache, cache_size,
)
from zoomy_core.model.models.vam import VAM


def test_second_call_is_hit_body_not_rerun():
    calls = {"n": 0}

    @derivation_cache
    def stage(spec, basis, level):
        calls["n"] += 1
        return f"built:{spec}:{basis}:{level}"

    r1 = stage("vam", "legendre", 1)
    assert calls["n"] == 1 and stage.stats.misses == 1
    r2 = stage("vam", "legendre", 1)
    assert calls["n"] == 1, "2nd identical call must NOT re-run the body"
    assert stage.stats.hits == 1
    assert r1 is r2, "in-memory hit returns the same object"


def test_changed_argument_misses():
    calls = {"n": 0}

    @derivation_cache
    def stage(spec, level):
        calls["n"] += 1
        return (spec, level)

    stage("vam", 1)
    stage("vam", 2)        # changed arg
    stage("sme", 1)        # changed arg
    assert calls["n"] == 3


def test_declarative_model_arg_keys_on_spec():
    calls = {"n": 0}

    @derivation_cache
    def build(model):
        calls["n"] += 1
        return (model.level, model.dimension)

    build(VAM(level=1, dimension=3))
    build(VAM(level=1, dimension=3))      # distinct instance, same spec -> HIT
    assert calls["n"] == 1
    build(VAM(level=2, dimension=3))      # different spec -> MISS
    assert calls["n"] == 2


def test_version_bump_namespaces_entries():
    calls = {"n": 0}

    @derivation_cache(version=7)
    def stage(x):
        calls["n"] += 1
        return x * 2

    stage(3)
    stage(3)
    assert calls["n"] == 1


def test_verify_reruns_and_asserts_identity():
    calls = {"n": 0}

    @derivation_cache(verify=True)
    def stage(x):
        calls["n"] += 1
        return sp.Symbol("a") * x

    stage(sp.Integer(2))
    stage(sp.Integer(2))     # verify=True -> body re-runs and asserts equal
    assert calls["n"] == 2


def test_clear_and_size():
    @derivation_cache
    def stage(x):
        return x

    clear_derivation_cache(stage)
    stage(1)
    stage(2)
    assert cache_size() >= 2
    stage.cache_clear()
    # after clearing this fn, calling again is a miss (re-stored)
    before = stage.stats.misses
    stage(1)
    assert stage.stats.misses == before + 1


def test_real_vam_build_cached_no_recompute():
    """Acceptance: wrap a real VAM SystemModel build; 2nd call is a HIT that
    skips the heavy symbolic derivation, and returns an srepr-identical model."""
    builds = {"n": 0}

    @derivation_cache
    def derive_vam(level, dimension):
        builds["n"] += 1
        return VAM(level=level, dimension=dimension).system_model

    sm1 = derive_vam(1, 3)
    assert builds["n"] == 1
    sm2 = derive_vam(1, 3)
    assert builds["n"] == 1, "2nd build must hit cache (no ResolveModes re-run)"
    assert sm1 is sm2
    # srepr-identity of the operator content vs a fresh uncached build
    fresh = VAM(level=1, dimension=3).system_model
    assert sp.srepr(sp.Matrix(sm1.flux)) == sp.srepr(sp.Matrix(fresh.flux))
