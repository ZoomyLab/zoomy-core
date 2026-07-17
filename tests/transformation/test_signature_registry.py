"""Operator-signature UNIFICATION: the SystemModel carries each operator as a
declared ``basefunction.Function`` (``OPERATOR_ARG_SLOTS`` → ``sm.operator(name)
.args``), and EVERY backend printer/runtime reads that single declaration and
only translates SYNTAX.  These tests pin:

1. the registry contract (declared args match ``OPERATOR_ARG_SLOTS``, incl. the
   REQ-185 time/dt/position shapes);
2. that the generic-C and OpenFOAM printers emit the argument list DERIVED from
   the registry (not a hardcoded per-printer table);
3. that numpy / to_ufl read the same declaration (source carries t/dt/x/y/z);
4. THE PROOF — re-doing the REQ-185 extension as a ONE-LINE registry edit
   propagates to every printer's emitted signature in lockstep.
"""
import re
import inspect

import sympy as sp
import pytest

from zoomy_core.model.models.swe import SWE
from zoomy_core.systemmodel.system_model import (
    SystemModel, OPERATOR_ARG_SLOTS)
from zoomy_core.transformation.generic_c import GenericCppModel, GenericCppBase
from zoomy_core.transformation.to_openfoam import FoamSystemModelPrinter, _FOAM_ARG


# ── helpers ──────────────────────────────────────────────────────────────────

def _c_argnames(code, kernel):
    """The ordered argument NAMES of a C-family ``kernel(...)`` signature in
    emitted ``code`` (last identifier of each comma-separated declaration)."""
    m = re.search(r"\b" + re.escape(kernel) + r"\s*\(([^)]*)\)", code)
    assert m, f"kernel {kernel!r} not found in emitted code"
    args = []
    for decl in m.group(1).split(","):
        ident = re.findall(r"[A-Za-z_]\w*", decl)
        if ident:
            args.append(ident[-1])
    return args


def _swe_sm():
    return SystemModel.from_model(SWE(dimension=2))


# ── 1. registry contract ─────────────────────────────────────────────────────

def test_operator_function_args_match_declaration():
    """``sm.operator(name).args`` is exactly the declared slot ordering — the
    operator is CARRIED as a Function, not a bare array."""
    sm = _swe_sm()
    for name, slots in OPERATOR_ARG_SLOTS.items():
        fn = sm.operator(name)
        assert tuple(fn.args.keys()) == tuple(slots), name
        # it is a real Function carrying both signature AND definition
        assert fn.name == name
        assert fn.definition is not None


def test_req185_declared_shapes():
    """The REQ-185 seam lives in ONE place now."""
    S = OPERATOR_ARG_SLOTS
    assert S["source"] == ("variables", "aux_variables", "parameters",
                           "time", "dt", "position")
    assert S["update_aux_variables"] == ("variables", "aux_variables",
                                         "parameters", "time", "position")
    assert S["update_aux_variables_jacobian_wrt_variables"] == \
        S["update_aux_variables"]
    assert S["update_variables"] == ("variables", "aux_variables",
                                     "parameters", "dt")
    assert S["eigenvalues"] == ("variables", "aux_variables", "parameters",
                                "normal")


# ── 2. C-family printers emit the registry-derived signature ─────────────────

@pytest.mark.parametrize("op,expected", [
    ("source", ["Q", "Qaux", "p", "time", "dt", "X"]),
    ("eigenvalues", ["Q", "Qaux", "p", "n"]),
    ("update_variables", ["Q", "Qaux", "p", "dt"]),
])
def test_generic_c_signature_from_registry(op, expected):
    sm = _swe_sm()
    code = GenericCppModel(sm).create_code()
    assert _c_argnames(code, op) == expected
    # and it equals what the shared helper derives from the Function.args
    assert GenericCppModel(sm)._operator_arg_keys(op) == expected


def test_openfoam_source_signature_from_registry():
    sm = _swe_sm()
    code = FoamSystemModelPrinter(sm).create_code()
    assert _c_argnames(code, "source") == ["Q", "Qaux", "p", "time", "dt", "X"]
    assert _c_argnames(code, "eigenvalues") == ["Q", "Qaux", "p", "n"]


def test_c_helper_lives_on_shared_base():
    """The group→C-key translation is one shared method (dmplex/amrex/openfoam),
    not a per-printer table."""
    assert hasattr(GenericCppBase, "_operator_arg_keys")


# ── 3. numpy / to_ufl read the same declaration ──────────────────────────────

def _capture_lambdify_sources(build_runtime):
    srcs = []
    orig = sp.lambdify

    def spy(a, e, **k):
        f = orig(a, e, **k)
        try:
            srcs.append(inspect.getsource(f))
        except Exception:
            pass
        return f

    sp.lambdify = spy
    try:
        build_runtime()
    finally:
        sp.lambdify = orig
    return srcs


def test_numpy_source_signature_carries_time_dt_position():
    from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
    sm = _swe_sm()
    srcs = _capture_lambdify_sources(
        lambda: NumpyRuntimeModel.from_system_model(sm))
    assert any(", t, dt, x, y, z)" in s.splitlines()[0] for s in srcs), \
        "numpy source lambdify signature missing the declared t/dt/x/y/z tail"


def test_ufl_inherits_registry_signature():
    ufl = pytest.importorskip("ufl")   # noqa: F841
    from zoomy_core.transformation.to_ufl import UFLRuntimeModel
    sm = _swe_sm()
    srcs = _capture_lambdify_sources(
        lambda: UFLRuntimeModel.from_system_model(sm))
    assert any(", t, dt, x, y, z)" in s.splitlines()[0] for s in srcs), \
        "to_ufl (inherits numpy) source signature missing t/dt/x/y/z"


# ── 4. THE PROOF: the REQ-185 extension is a ONE-LINE registry change ─────────

def test_one_line_registry_edit_propagates_to_every_printer():
    """Re-do REQ-185 as a demonstration: with ``source`` declared at its
    pre-REQ-185 baseline ``(variables, aux_variables, parameters)`` NO printer
    emits time/dt/position; flipping the SINGLE ``OPERATOR_ARG_SLOTS['source']``
    tuple to add them makes generic-C, OpenFOAM (and numpy) ALL emit them in
    lockstep — no per-printer edit.  That one tuple is the whole extension."""
    baseline = ("variables", "aux_variables", "parameters")
    extended = baseline + ("time", "dt", "position")
    saved = OPERATOR_ARG_SLOTS["source"]
    try:
        # --- baseline (simulate pre-REQ-185): the 1 line reverted ---
        OPERATOR_ARG_SLOTS["source"] = baseline
        sm = _swe_sm()
        gc = _c_argnames(GenericCppModel(sm).create_code(), "source")
        foam = _c_argnames(FoamSystemModelPrinter(sm).create_code(), "source")
        assert gc == ["Q", "Qaux", "p"]
        assert foam == ["Q", "Qaux", "p"]

        # --- the ONE-LINE change: add (time, dt, position) to the tuple ---
        OPERATOR_ARG_SLOTS["source"] = extended
        sm = _swe_sm()
        gc = _c_argnames(GenericCppModel(sm).create_code(), "source")
        foam = _c_argnames(FoamSystemModelPrinter(sm).create_code(), "source")
        assert gc == ["Q", "Qaux", "p", "time", "dt", "X"]
        assert foam == ["Q", "Qaux", "p", "time", "dt", "X"]

        # numpy reads the same declaration
        from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
        srcs = _capture_lambdify_sources(
            lambda: NumpyRuntimeModel.from_system_model(_swe_sm()))
        assert any(", t, dt, x, y, z)" in s.splitlines()[0] for s in srcs)
    finally:
        OPERATOR_ARG_SLOTS["source"] = saved
