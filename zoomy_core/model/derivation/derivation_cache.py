"""Function-level cache for the symbolic derivation (task 0018 / REQ-10).

The heavy symbolic passes (``ResolveModes`` ≈ 70 % of an SME(4) build,
``ResolveBasis``, ``simplify``) are expensive to recompute when a notebook cell
re-runs or a convergence study sweeps over ``Nu``.  Rather than checkpoint and
restore a half-built ``Model`` (fragile — the modal registry / σ-state would all
have to round-trip), we cache at **function boundaries**: decorate the expensive
*stages* of a derivation and the cache returns the prior result whenever the
function's SOURCE and ARGUMENTS are unchanged.

    @derivation_cache
    def project_sme(spec, basis, level):
        ...                      # the expensive build
        return system_model

Multi-level caching is just composition — decorate each stage and let the
orchestrator call them::

    @derivation_cache
    def stage_0(spec):  ...      # basis/level-agnostic prefix (reused across a sweep)

    @derivation_cache
    def stage_final(model0, basis, level):  ...

    def derive(spec, basis, level):          # leave the orchestrator UNcached
        return stage_final(stage_0(spec), basis, level)

**Cache the leaf stages, not the orchestrator.**  The key includes the decorated
function's own source (so editing the recipe auto-invalidates) but NOT the source
of functions it calls — caching ``derive`` would short-circuit changed stages.

The default tier is **in-memory** (session lifetime): no serialization, the
returned object is held live — ideal for notebook re-runs and sweeps.  Cross-
session disk persistence is out of scope here (it needs the return value to
round-trip; a ``SystemModel`` can, an intermediate ``Model`` cannot reliably).
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import logging

from zoomy_core.model.derivation.cache_keys import CACHE_VERSION, argument_key

logger = logging.getLogger(__name__)

# function-id → { call_key: result }
_STORE: dict[str, dict[str, object]] = {}


class CacheStats:
    """Hit / miss / call counters, per decorated function and global.

    The per-function ``calls`` counter is what the acceptance test asserts on:
    a cache HIT does NOT increment ``calls`` (the wrapped body — and therefore
    ``ResolveModes`` — never runs)."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.calls = 0          # wrapped-body executions (= misses, kept distinct for clarity)

    def __repr__(self):
        return f"CacheStats(hits={self.hits}, misses={self.misses}, calls={self.calls})"


def _fn_source(fn) -> str:
    """Source text of ``fn`` for the key — edit the body ⇒ key changes ⇒ MISS.

    Falls back to the qualified name when source is unavailable (e.g. a function
    built by ``exec``); callers then rely on args + a manual ``version=`` bump.
    """
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        logger.warning(
            "derivation_cache: no source for %s; keying on qualname only "
            "(edits to the body will NOT invalidate — pass version= to bump).",
            getattr(fn, "__qualname__", fn))
        return f"<no-source:{fn.__module__}.{getattr(fn, '__qualname__', fn)}>"


def _call_key(source: str, args, kwargs, version) -> str:
    parts = [CACHE_VERSION, f"version={version!r}", source]
    parts += [argument_key(a) for a in args]
    parts += [f"{k}={argument_key(v)}" for k, v in sorted(kwargs.items())]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def derivation_cache(fn=None, *, version=0, verify=False):
    """Memoise a derivation stage, keyed on (function source ⊕ argument content).

    Parameters
    ----------
    version : hashable, optional
        Bump to force-invalidate this function's entries without editing its
        body (or when the body's source is unavailable).
    verify : bool, optional
        On a cache HIT, RE-RUN the body and assert the fresh result is
        ``srepr``-identical to the cached one.  Defeats the purpose of caching
        (it always recomputes) — use it in tests / CI to catch a key that
        forgets a result-affecting argument.

    The wrapper exposes ``.stats`` (a :class:`CacheStats`), ``.cache_clear()``,
    and ``.__wrapped__`` (the original function).
    """
    def decorate(func):
        fid = f"{func.__module__}.{func.__qualname__}"
        source = _fn_source(func)
        _STORE.setdefault(fid, {})
        stats = CacheStats()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = _call_key(source, args, kwargs, version)
            store = _STORE[fid]
            if key in store:
                stats.hits += 1
                cached = store[key]
                if verify:
                    fresh = func(*args, **kwargs)
                    _assert_same(fresh, cached, fid)
                return cached
            stats.misses += 1
            stats.calls += 1
            result = func(*args, **kwargs)
            store[key] = result
            return result

        wrapper.stats = stats
        wrapper.cache_clear = lambda: _STORE.get(fid, {}).clear()
        wrapper._cache_id = fid
        return wrapper

    return decorate if fn is None else decorate(fn)


def _assert_same(fresh, cached, fid):
    import sympy as sp
    try:
        a, b = sp.srepr(fresh), sp.srepr(cached)
    except Exception:
        a, b = repr(fresh), repr(cached)
    if a != b:
        raise AssertionError(
            f"derivation_cache[{fid}]: verify=True found a STALE entry — fresh "
            f"result differs from cached. The cache key is missing a "
            f"result-affecting argument.")


def clear_derivation_cache(fn=None):
    """Clear all derivation-cache entries, or just one decorated function's."""
    if fn is not None:
        _STORE.get(getattr(fn, "_cache_id", None), {}).clear()
    else:
        _STORE.clear()


def cache_size() -> int:
    """Total number of cached entries across all decorated functions."""
    return sum(len(d) for d in _STORE.values())


__all__ = [
    "derivation_cache",
    "clear_derivation_cache",
    "cache_size",
    "CacheStats",
]
