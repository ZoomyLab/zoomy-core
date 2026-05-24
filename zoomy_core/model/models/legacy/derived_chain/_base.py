"""Base class + caching for the derivation chain.

Each class in the chain inherits from ``DerivationStep``.  Its
``__init__`` calls ``super().__init__(**kwargs)`` first to derive the
parent's state, then performs **its own one step** in
``_derive_step``.

The cache is class-level: ``cls._cache: dict[tuple, dict]`` keyed on
``(class.__qualname__, frozen_kwargs)``.  On hit, the cached
``_system`` is **deep-cloned** into the instance so subsequent
mutations don't pollute the cache.
"""
from __future__ import annotations

import copy
from typing import Any, Tuple


def _freeze(obj: Any) -> Any:
    """Recursively freeze a value into a hashable form."""
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, set):
        return tuple(sorted(_freeze(v) for v in obj))
    return obj


class DerivationStep:
    """Mixin for every derivation-chain class.

    Subclasses populate ``self._system`` (a ``DerivedSystem``) and
    optionally ``self._meta`` (a dict of derivation metadata used by
    ``describe``).
    """

    # Class-level cache. Each subclass gets its own dict via
    # ``__init_subclass__``.
    _cache: "dict[Tuple, dict]"

    # Optional human-readable description shown by ``describe(full_hierarchy=True)``.
    step_description: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own cache dict so different levels don't
        # collide.
        cls._cache = {}

    @classmethod
    def _cache_key(cls, kwargs: dict) -> Tuple:
        return (cls.__qualname__, _freeze(kwargs))

    @classmethod
    def _cache_get(cls, kwargs: dict):
        key = cls._cache_key(kwargs)
        return cls._cache.get(key)

    @classmethod
    def _cache_put(cls, kwargs: dict, payload: dict) -> None:
        key = cls._cache_key(kwargs)
        cls._cache[key] = payload

    def _adopt_cached(self, payload: dict) -> None:
        """Deep-clone cached state onto self so future mutations don't
        leak back into the cache."""
        for name, value in payload.items():
            setattr(self, name, copy.deepcopy(value))

    def _snapshot_for_cache(self) -> dict:
        """Subclasses override to declare what to cache.  Default: cache
        ``_system`` only (deep-cloned at lookup time)."""
        return {"_system": self._system}


# ---------------------------------------------------------------------------
# describe(full_hierarchy=True)
# ---------------------------------------------------------------------------

def walk_chain_descriptions(cls) -> list[Tuple[str, str]]:
    """Walk ``cls.__mro__`` collecting (class name, step_description) pairs
    from every class that defines ``step_description`` (skipping ``object``,
    private mixins, and ``DerivedModel`` ancestors)."""
    out = []
    for c in reversed(cls.__mro__):
        if c is object or c is DerivationStep:
            continue
        sd = c.__dict__.get("step_description")
        if sd:
            out.append((c.__name__, sd))
    return out


def describe(model, full_hierarchy: bool = False) -> str:
    """Produce a human-readable description of the model.

    ``full_hierarchy=False`` (default): only the *current* class's
    step description plus the final equation count.

    ``full_hierarchy=True``: walk every ancestor in the chain and print
    each step in order, INS first.
    """
    out_lines: list[str] = []
    if full_hierarchy:
        out_lines.append(f"Derivation chain for {type(model).__name__}:\n")
        for i, (name, desc) in enumerate(walk_chain_descriptions(type(model)), 1):
            out_lines.append(f"  {i}. {name}")
            for line in desc.split(". "):
                line = line.strip().rstrip(".")
                if line:
                    out_lines.append(f"       {line}.")
            out_lines.append("")
    else:
        out_lines.append(f"{type(model).__name__}: {type(model).step_description}")
        out_lines.append("")

    sys_ = getattr(model, "_system", None)
    if sys_ is not None:
        n_eq = sum(1 for _ in sys_.leaves())
        out_lines.append(f"  Final system: {n_eq} equations  (named: {sys_.name})")
        for path, _ in sys_.leaves():
            out_lines.append(f"    - {'.'.join(path)}")
    return "\n".join(out_lines)
