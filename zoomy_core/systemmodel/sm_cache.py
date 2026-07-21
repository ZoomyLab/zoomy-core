"""SystemModel build cache (REQ-163) — ON BY DEFAULT for every kind-tagged model.

``SystemModel.from_model(model)`` for the production models dispatches through
``model_builders.build_system_model``; this module caches THAT result — the
final lowered operator matrices ("a couple hundred small matrices", ~12 KB
per model pickled) — across three tiers:

    1. in-process memory,
    2. the user cache dir  (``$ZOOMY_CACHE_DIR`` or ``~/.cache/zoomy/systemmodels``),
    3. the PREBUILT cache shipped inside the ``zoomy_core`` package
       (``zoomy_core/systemmodel/_prebuilt/`` — regenerate with
       ``python -m zoomy_core.systemmodel.build_prebuilt_cache``).

The key is ``model_spec_key(model)`` (parameter VALUES excluded — they stay
free symbols end-to-end and are passed to the solver at runtime; see
cache_keys) combined with a source hash of EVERY class in the model's MRO
(``type(model).__mro__`` minus ``object``/``abc`` plumbing) plus a hash of its
builder (REQ-188).  Hashing the full source of every class in the chain
captures every overridden operator (``source``/``flux``/
``nonconservative_matrix``/``update_variables``/closures/…) automatically, so
editing ANY method of ANY class in the MRO invalidates the entry — no operator
enumeration, no silent-stale.  This replaces the pre-REQ-188 ``derive_model``
source hash, which missed operator overrides on subclasses that inherited
``derive_model`` (e.g. ``RainSWE(SWE)`` editing only ``source``).

If ``inspect.getsource`` fails for any class in the chain (a REPL/exec-defined
model with no on-disk source), the model is treated as UNCACHEABLE:
:func:`cache_key` returns ``None`` and the model is derived fresh every time.
Serving a possibly-stale cached entry for a source we cannot fingerprint is the
exact failure mode REQ-188 kills, so we refuse to cache rather than risk it
(warned once per model class).

Cache hits return a FRESH object (``pickle.loads`` per call) — callers may
mutate ICs/BCs on the result, exactly like a fresh build.

Environment switches (no per-case opt-in anywhere):
    ``ZOOMY_DERIVATION_CACHE=1``   enable cache reads (DEFAULT IS OFF, see below),
    ``ZOOMY_DERIVATION_REBUILD=1`` rebuild AND overwrite the cache entries.

The cache is DISABLED BY DEFAULT during model development: every build derives
from source, so a stale or mis-keyed entry cannot reach a result.  It is turned
back on once the model layer is stable and the solvers want the ~2x speed.
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import os
import pickle
from pathlib import Path

_MEMORY: dict[str, bytes] = {}

_LOG = logging.getLogger(__name__)

# Model classes we've already warned are uncacheable (REPL/exec-defined) — the
# warning fires once per class, not once per derivation.
_UNCACHEABLE_WARNED: set[str] = set()


def _enabled() -> bool:
    from zoomy_core.model.derivation.cache_keys import derivation_cache_enabled
    return derivation_cache_enabled()


def _rebuild() -> bool:
    return os.environ.get("ZOOMY_DERIVATION_REBUILD", "").strip() in {"1", "true", "yes"}


def _user_dir() -> Path:
    root = os.environ.get("ZOOMY_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "zoomy"
    return base / "systemmodels"


def _prebuilt_dir() -> Path:
    return Path(__file__).resolve().parent / "_prebuilt"


def _src_hash(obj) -> str:
    try:
        return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return "nosrc"


def _mro_classes(cls):
    """The model's MRO minus ``object`` and ``abc`` plumbing — the classes whose
    source actually defines the derivation/operators."""
    for c in cls.__mro__:
        if c is object or c.__module__ == "abc":
            continue
        yield c


def _mro_source_terms(cls) -> str | None:
    """Identity + full-source hash of every class in the MRO (REQ-188).

    Returns ``None`` if ANY class in the chain has no retrievable source
    (REPL/exec-defined) — the model is then uncacheable, because we cannot
    detect an edit to that class and must not serve a possibly-stale entry."""
    parts = []
    for c in _mro_classes(cls):
        try:
            src = inspect.getsource(c)
        except (OSError, TypeError):
            _warn_uncacheable(cls, c)
            return None
        digest = hashlib.sha256(src.encode()).hexdigest()[:16]
        parts.append(f"{c.__module__}.{c.__qualname__}:{digest}")
    return "|".join(parts)


def _warn_uncacheable(model_cls, failed_cls) -> None:
    ident = f"{model_cls.__module__}.{model_cls.__qualname__}"
    if ident in _UNCACHEABLE_WARNED:
        return
    _UNCACHEABLE_WARNED.add(ident)
    _LOG.warning(
        "SystemModel cache DISABLED for %s: cannot read the source of %s.%s "
        "(REPL/exec-defined?). Deriving fresh every call — a source we cannot "
        "fingerprint could otherwise be served stale after an edit (REQ-188).",
        ident, failed_cls.__module__, failed_cls.__qualname__,
    )


def cache_key(model, builder) -> str | None:
    """Symbolic identity of the model's built SystemModel, or ``None`` if the
    model is UNCACHEABLE (a class in its MRO has no on-disk source).

    Composition (REQ-188): ``CACHE_VERSION | model_spec_key(model) | <per-class
    identity+source hash for every class in the MRO minus object/abc> |
    builder-source hash``.  Hashing the full source of every MRO class captures
    every overridden operator automatically, so an edit to any method of any
    class in the chain invalidates the entry.

    BC/IC are NOT part of the key: they are runtime data (they embed parameter
    values) and are RE-ATTACHED to every fetched result by
    ``build_system_model`` — the cache stores only the symbolic identity."""
    from zoomy_core.model.derivation.cache_keys import CACHE_VERSION, model_spec_key
    mro = _mro_source_terms(type(model))
    if mro is None:
        return None                       # uncacheable: always derive fresh
    spec = model_spec_key(model)
    return hashlib.sha256(
        f"{CACHE_VERSION}|sm|{spec}|mro={mro}|b={_src_hash(builder)}".encode()
    ).hexdigest()


def fetch(key: str | None):
    """Return a FRESH SystemModel for ``key`` or None (miss / disabled /
    uncacheable ``key is None``)."""
    if key is None or not _enabled() or _rebuild():
        return None
    blob = _MEMORY.get(key)
    if blob is None:
        for root in (_user_dir(), _prebuilt_dir()):
            p = root / f"{key}.pkl"
            if p.is_file():
                try:
                    blob = p.read_bytes()
                except OSError:
                    continue
                _MEMORY[key] = blob
                break
    if blob is None:
        return None
    try:
        return pickle.loads(blob)
    except Exception:
        _MEMORY.pop(key, None)   # corrupt / stale-format entry: rebuild
        return None


def store(key: str | None, sm) -> None:
    if key is None or not _enabled():   # uncacheable model: never persist
        return
    try:
        blob = pickle.dumps(sm)
    except Exception:
        return                    # unpicklable exotic model: memory-only skip
    _MEMORY[key] = blob
    try:
        d = _user_dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{key}.tmp"
        tmp.write_bytes(blob)
        tmp.replace(d / f"{key}.pkl")
    except OSError:
        pass                      # read-only FS: memory tier still works
