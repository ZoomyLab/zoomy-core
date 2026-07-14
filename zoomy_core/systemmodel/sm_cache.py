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

The key is ``model_spec_key(model, include_bc_ic=True)`` (parameter VALUES
excluded — they stay free symbols end-to-end and are passed to the solver at
runtime; see cache_keys) combined with source hashes of the model's
``derive_model`` AND its builder, so editing either invalidates.

Cache hits return a FRESH object (``pickle.loads`` per call) — callers may
mutate ICs/BCs on the result, exactly like a fresh build.

Environment switches (no per-case opt-in anywhere):
    ``ZOOMY_DERIVATION_CACHE=0``   disable all cache reads (always rebuild),
    ``ZOOMY_DERIVATION_REBUILD=1`` rebuild AND overwrite the cache entries.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import pickle
from pathlib import Path

_MEMORY: dict[str, bytes] = {}


def _enabled() -> bool:
    return os.environ.get("ZOOMY_DERIVATION_CACHE", "1").strip() not in {"0", "false", "no"}


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


def cache_key(model, builder) -> str:
    """BC/IC are NOT part of the key: they are runtime data (they embed
    parameter values) and are RE-ATTACHED to every fetched result by
    ``build_system_model`` — the cache stores only the symbolic identity."""
    from zoomy_core.model.derivation.cache_keys import CACHE_VERSION, model_spec_key
    spec = model_spec_key(model)
    dm = getattr(type(model), "derive_model", None)
    return hashlib.sha256(
        f"{CACHE_VERSION}|sm|{spec}|dm={_src_hash(dm)}|b={_src_hash(builder)}".encode()
    ).hexdigest()


def fetch(key: str):
    """Return a FRESH SystemModel for ``key`` or None (miss / disabled)."""
    if not _enabled() or _rebuild():
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


def store(key: str, sm) -> None:
    if not _enabled():
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
