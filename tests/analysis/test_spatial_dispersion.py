"""Spatial dispersion branch of ``zoomy_core.analysis`` (REQ-176 items 1+2).

The temporal branch (:func:`temporal_branch` / :func:`plane_wave_dispersion`)
gives ``omega(k)`` — real wavenumbers, complex frequencies.  The SPATIAL branch
(:func:`spatial_dispersion`) is its companion: real frequencies, complex
wavenumbers ``k(omega)``, which is what the branch experiments (Mounkaila
Noma 2021) measure.  Because ``k`` enters the quasilinear pencil LINEARLY,

    det(-i omega M_t + i k A + M_0) = 0

is again a generalized eigenproblem with ``k`` as the eigenvalue — no
root-chasing.  ``spatial_dispersion`` is ONE entry point with a ``numeric=``
switch (per the user API directive):

  * ``numeric=False`` (DEFAULT, analytic): solve the determinant for
    ``k(omega)`` symbolically, then evaluate over the grid.  Feasible for
    small systems (SWE, SME(1)).
  * ``numeric=True``: evaluate the SAME pencil and take ``k`` as the
    generalized eigenvalue.  The only feasible route once the symbolic
    determinant blows up (SME(N>=2)).

Gates exercised here (all with hand-checkable classical systems, matching the
model-free convention of ``test_analysis.py``):

  (b) analytic == numeric to machine precision, and a temporal->spatial
      round-trip recovers the wavenumber exactly, on SWE (neutral) and on a
      roll-wave SWE with Chezy friction (genuine complex k, growing);
  (c) SME(2) runs via ``numeric=True`` (real model, import-guarded — symbolic
      is EXPECTED to be infeasible there, which is the whole point of the
      switch).

Sign convention: ansatz ``exp(i(k x - omega t))`` so amplitude ~
``exp(-Im(k) x)`` downstream; spatial growth ``alpha = -Im(k)`` (> 0 grows).
"""
from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from zoomy_core.misc.misc import Zstruct
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.analysis import (
    spatial_dispersion,
    spatial_cutoff,
    temporal_branch,
    NumericPencil,
)

T = sp.Symbol("t", real=True)
X = sp.Symbol("x", real=True)


def _swe():
    """Conservative SWE ``(h, u)`` (primitive velocity), non-linearised."""
    h, u = sp.symbols("h u", real=True)
    g = sp.Symbol("g", positive=True)
    return SystemModel(
        time=T, space=[X], state=[h, u], aux_state=[],
        parameters=Zstruct(g=g), parameter_values=Zstruct(g=9.81),
        flux=sp.Matrix([[h * u], [h * u**2 + g * h**2 / 2]]),
        hydrostatic_pressure=sp.zeros(2, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(2, 2, 1),
        source=sp.zeros(2, 1),
        mass_matrix=sp.Matrix([[1, 0], [u, h]]),
    ), (h, u)


def _rollwave():
    """Roll-wave SWE with Chezy friction, conservative ``(h, m=hu)``.

    Uniform normal-flow base ``u0 = sqrt(g h0 S0 / Cf)`` gives Froude
    ``Fr = u0/sqrt(g h0) = sqrt(S0/Cf)``; the Vedernikov threshold is
    ``Fr = 2``.  Above it the flow is linearly unstable — genuine complex k.
    """
    h, m = sp.symbols("h m", real=True)
    g, S0, Cf = sp.symbols("g S0 Cf", positive=True)
    return SystemModel(
        time=T, space=[X], state=[h, m], aux_state=[],
        parameters=Zstruct(g=g, S0=S0, Cf=Cf),
        parameter_values=Zstruct(g=1.0, S0=0.06, Cf=0.02),
        flux=sp.Matrix([[m], [m**2 / h + g * h**2 / 2]]),
        hydrostatic_pressure=sp.zeros(2, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(2, 2, 1),
        source=sp.Matrix([[0], [g * h * S0 - Cf * m**2 / h**2]]),
        mass_matrix=sp.eye(2),
    ), (h, m)


# ---------------------------------------------------------------------------
# (b) SWE — analytic == numeric, phase speed, exact round-trip
# ---------------------------------------------------------------------------

def test_swe_analytic_equals_numeric_and_phase_speed():
    sm, (h, u) = _swe()
    H, U, gv = 2.0, 0.5, 9.81
    base, par = {h: H, u: U}, {"g": gv}
    w = np.linspace(0.2, 2.0, 12)
    cfast = U + np.sqrt(gv * H)

    an = spatial_dispersion(sm, base, w, params=par, numeric=False, c_seed=cfast)
    nu = spatial_dispersion(sm, base, w, params=par, numeric=True, c_seed=cfast)

    assert np.max(np.abs(an.k - nu.k)) < 1e-12       # SAME pencil, two solvers
    assert np.allclose(an.c, cfast)                   # c = U + sqrt(gH)
    assert np.allclose(an.alpha, 0.0, atol=1e-12)     # SWE is non-dissipative
    assert an.k_symbolic is not None and nu.k_symbolic is None
    assert nu.residual == 0.0                          # fully pinned base


def test_swe_temporal_to_spatial_roundtrip():
    """omega(k_real) fed back through k(omega) recovers k_real exactly."""
    sm, (h, u) = _swe()
    H, U, gv = 2.0, 0.5, 9.81
    base, par = {h: H, u: U}, {"g": gv}
    cfast = U + np.sqrt(gv * H)

    k0 = np.linspace(0.3, 1.4, 12)
    mats = NumericPencil(sm).at_equilibrium(par, fixed={"h": H, "u": U})
    tmp = temporal_branch(mats, k0, c_seed=cfast)     # omega(k0), real (neutral)
    assert np.allclose(tmp.omega.imag if np.iscomplexobj(tmp.omega)
                       else 0.0, 0.0, atol=1e-12)

    for numeric in (False, True):
        sp_ = spatial_dispersion(sm, base, tmp.omega, params=par,
                                 numeric=numeric, c_seed=cfast)
        assert np.max(np.abs(sp_.k.real - k0)) < 1e-10
        assert np.max(np.abs(sp_.k.imag)) < 1e-10


# ---------------------------------------------------------------------------
# (b) roll-wave SWE with friction — genuine complex k, analytic == numeric
# ---------------------------------------------------------------------------

def _rollwave_at(Fr):
    sm, (h, m) = _rollwave()
    gv, S0v = 1.0, 0.06
    Cfv = S0v / Fr**2
    u0 = np.sqrt(gv * 1.0 * S0v / Cfv)
    return sm, {h: 1.0, m: u0}, {"g": gv, "S0": S0v, "Cf": Cfv}, u0


def test_rollwave_analytic_equals_numeric_complex_k():
    sm, base, par, u0 = _rollwave_at(3.0)             # Fr = 3 > 2: unstable
    w = np.linspace(0.05, 2.5, 80)
    an = spatial_dispersion(sm, base, w, params=par, numeric=False, c_seed=u0 + 1)
    nu = spatial_dispersion(sm, base, w, params=par, numeric=True, c_seed=u0 + 1)

    assert np.max(np.abs(an.k - nu.k)) < 1e-12        # both solve the same pencil
    assert np.any(np.abs(an.k.imag) > 1e-4)            # genuinely complex k
    assert an.alpha.max() > 0.0                        # some spatially growing band


def test_rollwave_vedernikov_threshold():
    """Temporal instability appears above Fr = 2 (physics anchor)."""
    def max_sigma(Fr):
        sm, base, par, u0 = _rollwave_at(Fr)
        mats = NumericPencil(sm).at_equilibrium(
            par, fixed={str(k): v for k, v in base.items()})
        return temporal_branch(mats, np.linspace(1e-3, 3.0, 300),
                               c_seed=u0 + 1).sigma.max()

    assert max_sigma(3.0) > 1e-4                        # unstable
    assert max_sigma(1.2) < 1e-4                        # stable


# ---------------------------------------------------------------------------
# spatial_cutoff detector
# ---------------------------------------------------------------------------

def test_spatial_cutoff_detects_crossing_and_nan():
    w = np.linspace(0.0, 1.0, 11)
    grow_then_decay = Zstruct(omega=w, alpha=0.1 - 0.2 * w)   # zero at w = 0.5
    assert abs(spatial_cutoff(grow_then_decay) - 0.5) < 0.11
    always_growing = Zstruct(omega=w, alpha=np.full_like(w, 0.3))
    assert np.isnan(spatial_cutoff(always_growing))


# ---------------------------------------------------------------------------
# Error paths — analytic needs a fully specified base state
# ---------------------------------------------------------------------------

def test_analytic_requires_all_states_pinned():
    sm, (h, u) = _swe()
    with pytest.raises(ValueError, match="every state pinned"):
        spatial_dispersion(sm, {h: 1.0}, [0.5], params={"g": 9.81})


def test_analytic_leftover_symbol_raises():
    sm, (h, u) = _swe()
    with pytest.raises(ValueError, match="still depends on"):
        spatial_dispersion(sm, {h: 1.0, u: 0.0}, [0.5], params={})  # g missing


# ---------------------------------------------------------------------------
# (c) SME(2) — the numeric switch is the only feasible route (real model,
#     import-guarded against concurrent model-package edits).
# ---------------------------------------------------------------------------

# fsolve warns "no progress" because the zero guess already sits on the trivial
# uniform equilibrium; convergence is verified below by the residual assertion.
@pytest.mark.filterwarnings("ignore:The iteration is not making good progress")
def test_sme2_numeric_switch():
    models = pytest.importorskip("zoomy_core.model.models")
    from zoomy_core.model.models.closures import (
        Newtonian, NavierSlip, StressFree)
    p0 = {"g": 1.0, "rho": 1.0, "nu": 0.05, "e_x": 0.2, "lambda_s": 50.0}
    try:
        sm = SystemModel.from_model(models.SME(
            level=2, closures=[Newtonian(), NavierSlip(), StressFree()],
            quadrature_order=8, parameters=p0))
    except Exception as exc:                            # pragma: no cover
        pytest.skip(f"SME(2) build unavailable (concurrent edit?): {exc}")

    w = np.linspace(0.05, 1.5, 20)
    S = spatial_dispersion(sm, {"b": 0.0, "h": 1.0}, w, params=p0,
                           numeric=True, drop=("b",))
    assert S.residual < 1e-8                            # equilibrium solved
    assert np.all(np.isfinite(S.k))
    assert S.alpha.shape == w.shape and S.c.shape == w.shape
