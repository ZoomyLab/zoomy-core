"""REQ-188 acceptance: the SystemModel cache keys on the FULL source of every
class in the model's MRO, so editing an overridden operator (``source`` on a
case-local ``SWE`` subclass that inherits ``derive_model``) invalidates the
cached entry instead of silently serving the previous, stale operators.

Before REQ-188 the key hashed only ``type(model).derive_model`` + spec, so a
subclass that overrode ``source`` but inherited ``derive_model`` kept the same
key across edits and the on-disk tier returned the old SystemModel — even in a
fresh process.
"""
import os
import subprocess
import sys
import textwrap

import pytest


# A subprocess body that derives ``M`` from a temp module on a shared cache dir
# and prints the lowered ``SystemModel.source`` (which reflects the OVERRIDDEN
# operator body).  Kept in the child so the source hash is read fresh from disk
# each run — a same-process re-import would hit Python's module/linecache.
_DERIVE = """
import os, sys
os.environ['ZOOMY_CACHE_DIR'] = {cache!r}
os.environ.pop('ZOOMY_DERIVATION_REBUILD', None)
os.environ.pop('ZOOMY_DERIVATION_CACHE', None)
sys.path.insert(0, {moddir!r})
from mod_req188 import M
from zoomy_core.systemmodel.system_model import SystemModel
sm = SystemModel.from_model(M(dimension=1))
print('SOURCE=' + str(sm.source))
"""


def _write_model(path, marker: int):
    """A minimal ``SWE`` subclass overriding ONLY ``source`` — a constant on the
    mass row whose value ``marker`` is what we watch propagate (or not)."""
    path.write_text(textwrap.dedent(f"""
        import sympy as sp
        from zoomy_core.model.models.swe import SWE
        from zoomy_core.model.basefunction import ZArray

        class M(SWE):
            def source(self):
                S = sp.Matrix.zeros(self.n_variables, 1)
                S[0, 0] = sp.Integer({marker})   # rainfall-like mass source
                return ZArray(S)
    """))


def _derive_in_subprocess(cache_dir, mod_dir):
    code = _DERIVE.format(cache=str(cache_dir), moddir=str(mod_dir))
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("SOURCE="))
    return line[len("SOURCE="):]


@pytest.mark.rederive
def test_edited_source_busts_cache_cross_process(tmp_path):
    """Acceptance (1): derive M(source=7) on a cache dir, rewrite the method
    body to source=13 in the SAME file, re-derive in a fresh process on the
    SAME cache dir → the SystemModel.source reflects the NEW body."""
    mod_dir = tmp_path / "pkg"
    mod_dir.mkdir()
    cache_dir = tmp_path / "cache"
    mod_file = mod_dir / "mod_req188.py"

    _write_model(mod_file, marker=7)
    first = _derive_in_subprocess(cache_dir, mod_dir)
    assert "7" in first and "13" not in first, first

    # edit ONLY the operator body — same class, same derive_model, same spec
    _write_model(mod_file, marker=13)
    second = _derive_in_subprocess(cache_dir, mod_dir)
    assert "13" in second, (
        "stale cache: edited source() body not reflected after re-derive "
        f"(got {second!r}, expected the new constant 13)")
    assert first != second


@pytest.mark.rederive
def test_unchanged_class_same_process_cache_hit(tmp_path, monkeypatch):
    """Acceptance (2): an UNCHANGED class re-derives to a cache HIT — the key is
    stable, so no second on-disk entry is written."""
    monkeypatch.setenv("ZOOMY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ZOOMY_DERIVATION_REBUILD", raising=False)
    monkeypatch.delenv("ZOOMY_DERIVATION_CACHE", raising=False)
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.systemmodel import sm_cache
    from zoomy_core.systemmodel.model_builders import _BUILDERS
    from zoomy_core.systemmodel.system_model import SystemModel
    sm_cache._MEMORY.clear()

    builder = _BUILDERS[SWE(dimension=1)._system_model_kind]
    k1 = sm_cache.cache_key(SWE(dimension=1), builder)
    k2 = sm_cache.cache_key(SWE(dimension=1), builder)
    assert k1 is not None and k1 == k2         # key stable for the same class

    SystemModel.from_model(SWE(dimension=1))
    n = len(list((tmp_path / "systemmodels").glob("*.pkl")))
    SystemModel.from_model(SWE(dimension=1))    # same class → HIT, no new file
    assert len(list((tmp_path / "systemmodels").glob("*.pkl"))) == n


def test_exec_defined_model_is_uncacheable():
    """getsource-failure handling: a REPL/exec-defined model (no on-disk source
    for a class in its MRO) is UNCACHEABLE — ``cache_key`` returns ``None`` so
    the model is derived fresh every call rather than risk a stale entry we
    cannot fingerprint."""
    from zoomy_core.systemmodel import sm_cache

    ns = {}
    exec("from zoomy_core.model.models.swe import SWE\n"
         "class M(SWE):\n"
         "    def source(self):\n"
         "        return super().source()\n", ns)
    inst = ns["M"](dimension=1)
    assert sm_cache.cache_key(inst, lambda m: m) is None
    # fetch/store treat a None key as uncacheable, never touching disk
    assert sm_cache.fetch(None) is None
    sm_cache.store(None, object())      # must be a silent no-op


def test_mro_source_terms_lists_expected_classes():
    """Every class in the MRO minus object/abc is fingerprinted (identity +
    source hash); a case-local subclass adds itself on top of the base chain."""
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.systemmodel import sm_cache

    class Local(SWE):
        pass

    base = [f"{c.__module__}.{c.__qualname__}" for c in sm_cache._mro_classes(SWE)]
    sub = [f"{c.__module__}.{c.__qualname__}" for c in sm_cache._mro_classes(Local)]
    assert "builtins.object" not in base
    assert base[0] == "zoomy_core.model.models.swe.SWE"
    assert "zoomy_core.model.basemodel.Model" in base
    assert sub[0].endswith("Local") and sub[1:] == base
    # source hash present + non-empty for a cacheable chain
    assert sm_cache._mro_source_terms(SWE) is not None
