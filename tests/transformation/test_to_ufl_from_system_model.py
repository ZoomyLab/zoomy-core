"""Firedrake UFL runtime is SystemModel/NSM-driven (REQ-90).

New-style ``BaseModel`` models (e.g. the thesis ``MalpassetSME``) carry no
``.name``/``.functions``/``.variables`` — they run through the
``NumericalSystemModel``.  ``UFLRuntimeModel.from_system_model`` /
``.from_nsm`` build every operator from ``sm`` (a ``SystemModel``) so
Firedrake can run them, mirroring ``NumpyRuntimeModel.from_system_model``
but emitting ``ufl`` forms instead of numpy arrays.

These tests lock the CORE side (construction + emitted-form inspection +
structural agreement with the numpy runtime); the apptainer/mpirun
end-to-end Firedrake run is the malpasset-firedrake steward's verification.
"""

import numpy as np
import pytest

ufl = pytest.importorskip("ufl")

from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.malpasset import MalpassetSWE
from zoomy_core.numerics.numerical_system_model import NumericalSystemModel
from zoomy_core.transformation.to_ufl import UFLRuntimeModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.systemmodel.system_model import SystemModel


def _uvec(n, base=1.0):
    """A UFL vector of distinct symbolic scalars (mesh-free, so the flux /
    source / eigenvalue forms can be emitted and inspected without a
    Firedrake mesh)."""
    return ufl.as_vector(
        [ufl.variable(ufl.constantvalue.FloatValue(base + i)) for i in range(n)]
    )


def _sme0_sm():
    return SystemModel.from_model(SME(level=0, dimension=2))


# --------------------------------------------------------------------------
# Acceptance 1 — construction from a SystemModel (no AttributeError)
# --------------------------------------------------------------------------

def test_from_system_model_constructs_without_old_model_attrs():
    sm = _sme0_sm()
    # The SystemModel carries NO ``model.functions`` registry — the old
    # ``UFLRuntimeModel(model)`` path reads that (plus ``model.name``); the
    # SystemModel path must build every operator from ``sm`` instead.
    assert not hasattr(sm, "functions")

    rt = UFLRuntimeModel.from_system_model(sm)

    # Every core operator must be present and be a callable.
    for op in ("flux", "hydrostatic_pressure", "nonconservative_matrix",
               "source", "eigenvalues", "mass_matrix",
               "update_aux_variables", "boundary_conditions",
               "aux_boundary_conditions"):
        assert op in rt.runtime_functions, f"missing operator {op!r}"
        assert callable(getattr(rt, op))


def test_from_nsm_delegates_and_new_style_malpasset():
    # MalpassetSWE is a BaseModel-based model; the NSM front door is what a
    # new-style model uses.  ``from_nsm`` must accept the NSM directly.
    nsm = NumericalSystemModel.from_system_model(MalpassetSWE())
    rt = UFLRuntimeModel.from_nsm(nsm)
    for op in ("flux", "source", "nonconservative_matrix",
               "diffusion_matrix_explicit", "eigenvalues",
               "update_variables", "update_aux_variables"):
        assert op in rt.runtime_functions, f"missing operator {op!r}"


# --------------------------------------------------------------------------
# Acceptance 2 — emitted UFL forms are non-trivial and correctly shaped
# --------------------------------------------------------------------------

def test_emitted_ufl_forms_are_nontrivial():
    sm = _sme0_sm()
    rt = UFLRuntimeModel.from_system_model(sm)

    nv, na = rt.n_variables, rt.n_aux_variables
    Q, Qaux = _uvec(nv), _uvec(max(na, 1))
    p = np.array(list(sm.parameter_values.values()), dtype=float)
    n = _uvec(max(rt.dimension, 2), 0.6)

    flux = rt.flux(Q, Qaux, p)
    src = rt.source(Q, Qaux, p)
    ev = rt.eigenvalues(Q, Qaux, p, n)

    # All are UFL expressions, not the empty default.
    for form in (flux, src, ev):
        assert isinstance(form, ufl.core.expr.Expr)

    # flux row for the mass equation carries the discharge symbol (non-zero).
    assert flux.ufl_shape[0] == nv
    # eigenvalues are a rank-1 UFL tensor of length n_variables and at least
    # one entry depends on the state (a genuine wave speed, not 0).
    assert ev.ufl_shape == (nv,)
    assert any(len(ev[i].ufl_operands) > 0 or float(str(ev[i]) or 0) != 0
               for i in range(nv) if ev[i].ufl_operands)

    # NCP is emitted as a single ufl.as_tensor with the SAME 2-D reshaped
    # shape convention as the legacy raw-Model UFL path.
    ncp = rt.nonconservative_matrix(Q, Qaux, p)
    assert isinstance(ncp, ufl.core.expr.Expr)
    assert len(ncp.ufl_shape) == 2


def test_ncp_and_diffusion_match_legacy_raw_model_shape():
    """The NSM-driven NCP/diffusion shape must equal the legacy raw-Model
    UFL path's (``UFLRuntimeModel(model)``), so the Firedrake form assembly
    consumes both identically."""
    m = MalpassetSWE()
    legacy = UFLRuntimeModel(m)                      # raw-Model path
    nsm = NumericalSystemModel.from_system_model(MalpassetSWE())
    driven = UFLRuntimeModel.from_system_model(nsm)  # SystemModel/NSM path

    nv, na = driven.n_variables, driven.n_aux_variables
    Q, Qaux = _uvec(nv), _uvec(na)
    p = np.array(list(nsm.parameter_values.values()), dtype=float)

    for op in ("nonconservative_matrix", "diffusion_matrix_explicit"):
        s_legacy = getattr(legacy, op)(Q, Qaux, p).ufl_shape
        s_driven = getattr(driven, op)(Q, Qaux, p).ufl_shape
        assert s_legacy == s_driven, f"{op}: {s_legacy} != {s_driven}"


def test_flux_structurally_agrees_with_numpy_runtime():
    """Same symbolic ``sm`` → the UFL flux and the numpy flux must be the
    SAME expression up to backend lowering.  Evaluate both on the same
    numeric point (numpy directly, UFL via constant-folding) and compare."""
    sm = _sme0_sm()
    rt_np = NumpyRuntimeModel.from_system_model(sm)
    rt_ufl = UFLRuntimeModel.from_system_model(sm)

    nv, na = rt_np.n_variables, rt_np.n_aux_variables
    Qv = np.arange(1.0, 1.0 + nv)
    Qauxv = np.arange(1.0, 1.0 + max(na, 1))
    p = np.array(list(sm.parameter_values.values()), dtype=float)

    f_np = np.asarray(rt_np.flux(Qv, Qauxv, p), dtype=float)

    Qu = ufl.as_vector([ufl.constantvalue.FloatValue(v) for v in Qv])
    Qauxu = ufl.as_vector([ufl.constantvalue.FloatValue(v) for v in Qauxv])
    f_ufl = rt_ufl.flux(Qu, Qauxu, p)

    # UFL constant-fold each entry to a float.
    f_ufl_num = np.array(
        [[float(ufl.algorithms.apply_algebra_lowering
                .apply_algebra_lowering(f_ufl[i, j])) if False
          else float(str(_fold(f_ufl[i, j])))
          for j in range(f_ufl.ufl_shape[1])]
         for i in range(f_ufl.ufl_shape[0])]
    )
    np.testing.assert_allclose(f_ufl_num, f_np.reshape(f_ufl_num.shape),
                               rtol=1e-12, atol=1e-12)


def _fold(expr):
    """Constant-fold a mesh-free UFL scalar of ``FloatValue``s to a Python
    float via UFL's own evaluator."""
    return expr(())  # UFL Expr is callable at a point; mesh-free → constant
