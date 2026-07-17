"""REQ-163 acceptance: default-on SystemModel cache."""
import os
import pickle
import subprocess
import sys

import pytest


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOOMY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ZOOMY_DERIVATION_REBUILD", raising=False)
    monkeypatch.delenv("ZOOMY_DERIVATION_CACHE", raising=False)
    from zoomy_core.systemmodel import sm_cache
    sm_cache._MEMORY.clear()
    return tmp_path


@pytest.mark.rederive
def test_param_value_sweep_hits_cache(cache_dir):
    """Two instances differing ONLY in a parameter VALUE share one build:
    values are runtime solver inputs, not derivation identity."""
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.systemmodel import sm_cache
    from zoomy_core.systemmodel.system_model import SystemModel

    a = SystemModel.from_model(SWE(dimension=1, parameters={"g": 9.81, "n_m": 0.0}))
    n_entries = len(list((cache_dir / "systemmodels").glob("*.pkl")))
    b = SystemModel.from_model(SWE(dimension=1, parameters={"g": 1.23, "n_m": 0.7}))
    assert len(list((cache_dir / "systemmodels").glob("*.pkl"))) == n_entries
    assert str(a.flux) == str(b.flux)
    # values still reach the solver side untouched
    assert float(b.parameter_values["g"] if hasattr(b.parameter_values, "__getitem__")
                 else b.parameter_values.g) == 1.23


@pytest.mark.rederive
def test_cache_hit_returns_fresh_mutable_object(cache_dir):
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.systemmodel.system_model import SystemModel
    a = SystemModel.from_model(SWE(dimension=1))
    ref = str(a.flux)
    a.flux = None                       # mutate the returned object
    b = SystemModel.from_model(SWE(dimension=1))
    assert str(b.flux) == ref           # cache not poisoned


@pytest.mark.rederive
def test_env_disable_and_rebuild(cache_dir, monkeypatch):
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.systemmodel import sm_cache
    from zoomy_core.systemmodel.system_model import SystemModel
    # force a fresh build + user-dir write (bypassing the shipped prebuilt)
    monkeypatch.setenv("ZOOMY_DERIVATION_REBUILD", "1")
    SystemModel.from_model(SWE(dimension=1))
    monkeypatch.delenv("ZOOMY_DERIVATION_REBUILD")
    entries = [f.stem for f in (cache_dir / "systemmodels").glob("*.pkl")]
    assert entries, "rebuild must write the user tier"
    key = entries[0]
    assert sm_cache.fetch(key) is not None
    monkeypatch.setenv("ZOOMY_DERIVATION_CACHE", "0")
    sm_cache._MEMORY.clear()
    assert sm_cache.fetch(key) is None          # reads disabled
    monkeypatch.delenv("ZOOMY_DERIVATION_CACHE")
    monkeypatch.setenv("ZOOMY_DERIVATION_REBUILD", "1")
    assert sm_cache.fetch(key) is None          # rebuild bypasses reads


@pytest.mark.rederive
def test_cross_process_disk_hit(cache_dir):
    """REQ-163 (a): a SECOND python process with the same spec loads from disk."""
    from zoomy_core.model.models.sme import SME
    from zoomy_core.systemmodel.system_model import SystemModel
    SystemModel.from_model(SME(level=0, dimension=2))
    code = (
        "import os, time\n"
        f"os.environ['ZOOMY_CACHE_DIR'] = {str(cache_dir)!r}\n"
        "from zoomy_core.model.models.sme import SME\n"
        "from zoomy_core.systemmodel.system_model import SystemModel\n"
        "t0 = time.time()\n"
        "sm = SystemModel.from_model(SME(level=0, dimension=2, parameters={'g': 777.0}))\n"
        "dt = time.time() - t0\n"
        "assert sm.n_equations > 0\n"
        "assert dt < 2.0, f'expected disk hit, took {dt:.2f}s'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]


def test_prebuilt_package_cache_ships_defaults():
    """The shipped _prebuilt dir carries the default models."""
    from zoomy_core.systemmodel import sm_cache
    entries = list(sm_cache._prebuilt_dir().glob("*.pkl"))
    assert len(entries) >= 10, "prebuilt cache missing - run build_prebuilt_cache"
    sm = pickle.loads(entries[0].read_bytes())
    assert hasattr(sm, "flux") and hasattr(sm, "state")
