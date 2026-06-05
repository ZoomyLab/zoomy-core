"""SystemModel.from_model structural extraction for the declarative SME.

The declarative ``model._equations`` are bit-exact to production
(:class:`zoomy_core.model.models.sme.SME`); the operator decomposition
``F / P / B / S`` (flux / hydrostatic-pressure / non-conservative-matrix /
source) that ``SystemModel.from_model`` produces must therefore match
production's tag-based extractor ENTRY-BY-ENTRY.

The historic bug: the declarative extractor reported ``B = 0`` and folded the
bathymetry coupling ``g·h·∂_x b`` and the higher-moment couplings into the flux
``F``.  SME is genuinely NON-conservative — ``B ≠ 0``.  These tests pin the
correct decomposition against the production ground truth.

``_canon`` (shared with ``test_sme_kt19``) strips Function/Symbol assumptions so
the production and declarative symbol identities (e.g. ``h`` plain vs
``h`` positive) compare equal.
"""

import sympy as sp
import pytest

from zoomy_core.derivation.models import SME, SlipSME
from zoomy_core.derivation.tests.test_sme_kt19 import _canon
from zoomy_core.model.models.sme import SME as ProductionSME
from zoomy_core.model.models.system_model import SystemModel


_SLIP_PARAMS = {"g": 9.81, "rho": 1.0, "nu": 1e-3, "lambda": 1e-2}


# ── fixtures: declarative + production SystemModels ────────────────────────


@pytest.fixture(scope="module")
def sme_pair():
    """Declarative + production SME SystemModels."""
    model, ctx = SME(N=2).build()
    Q = [ctx["b"], ctx["h"]] + ctx["q_modes"]
    mine = SystemModel.from_model(model, Q=Q)

    pm = ProductionSME(N=2)
    pm.derive_model()
    prod = SystemModel.from_model(pm)
    return mine, prod


@pytest.fixture(scope="module")
def slip_pair():
    """Declarative + production SlipSME SystemModels (ν/λ friction)."""
    model, ctx = SlipSME(N=2, parameters=dict(_SLIP_PARAMS)).build()
    Q = [ctx["b"], ctx["h"]] + ctx["q_modes"]
    mine = SystemModel.from_model(model, Q=Q)

    pm = ProductionSME(N=2, parameters=dict(_SLIP_PARAMS))
    pm.derive_model()
    pm.apply_slip_newton_friction()
    prod = SystemModel.from_model(pm)
    return mine, prod


# ── helpers ────────────────────────────────────────────────────────────────


def _assert_entrywise_equal(mine, prod, accessor, shape_iter, label):
    """Assert ``accessor(mine, idx) == accessor(prod, idx)`` for every idx
    (under ``_canon`` / ``sp.cancel``)."""
    for idx in shape_iter:
        a = _canon(accessor(mine, idx))
        b = _canon(accessor(prod, idx))
        assert sp.cancel(a - b) == 0, (
            f"{label}{idx}: mine={a}  prod={b}")


def _flux(sm, idx):
    return sm.flux[idx]


def _pressure(sm, idx):
    return sm.hydrostatic_pressure[idx]


def _ncp(sm, idx):
    return sm.nonconservative_matrix[idx]


def _source(sm, idx):
    return sm.source[idx]


# ── (A.1) state layout ─────────────────────────────────────────────────────


def test_state_layout_matches_production(sme_pair):
    mine, prod = sme_pair
    assert [str(s) for s in mine.state] == [str(s) for s in prod.state]
    assert [str(s) for s in mine.state] == ["b", "h", "q_0", "q_1", "q_2"]


# ── (A.2) flux matches production entry-by-entry ───────────────────────────


def test_flux_matches_production(sme_pair):
    mine, prod = sme_pair
    n_eq, n_dim = mine.n_equations, mine.n_dim
    _assert_entrywise_equal(
        mine, prod, _flux,
        ((i, d) for i in range(n_eq) for d in range(n_dim)),
        "flux")


# ── (A.3) hydrostatic pressure matches production ──────────────────────────


def test_pressure_matches_production(sme_pair):
    mine, prod = sme_pair
    n_eq, n_dim = mine.n_equations, mine.n_dim
    _assert_entrywise_equal(
        mine, prod, _pressure,
        ((i, d) for i in range(n_eq) for d in range(n_dim)),
        "P")
    # Spot: the mean-momentum row carries g·h²/2.
    g = mine.parameters.g
    hh = next(s for s in mine.state if str(s) == "h")
    row = next(i for i, s in enumerate(mine.state) if str(s) == "q_0")
    assert sp.cancel(_canon(mine.hydrostatic_pressure[row, 0])
                     - _canon(g * hh**2 / 2)) == 0


# ── (A.4) NCP matches production AND is non-zero ───────────────────────────


def test_ncp_matches_production_and_nonzero(sme_pair):
    mine, prod = sme_pair
    n_eq, n_st, n_dim = mine.n_equations, mine.n_state, mine.n_dim
    _assert_entrywise_equal(
        mine, prod, _ncp,
        ((i, j, d) for i in range(n_eq) for j in range(n_st)
         for d in range(n_dim)),
        "B")

    # B is genuinely non-zero — the historic bug reported B = 0.
    nonzero = [(i, j) for i in range(n_eq) for j in range(n_st)
               if _canon(mine.nonconservative_matrix[i, j, 0]) != 0]
    assert nonzero, "nonconservative_matrix is identically zero — bug!"

    # The defining SME couplings: B[h,q0]=1 and B[q0,b]=g·h.
    g = mine.parameters.g
    hh = next(s for s in mine.state if str(s) == "h")
    h_row = next(i for i, s in enumerate(mine.state) if str(s) == "h")
    q0_col = next(j for j, s in enumerate(mine.state) if str(s) == "q_0")
    b_col = next(j for j, s in enumerate(mine.state) if str(s) == "b")
    q0_row = next(i for i, s in enumerate(mine.state) if str(s) == "q_0")
    assert sp.cancel(_canon(mine.nonconservative_matrix[h_row, q0_col, 0])
                     - 1) == 0
    assert sp.cancel(_canon(mine.nonconservative_matrix[q0_row, b_col, 0])
                     - _canon(g * hh)) == 0


# ── (A.5) source — open SME is empty (stress is aux) ───────────────────────


def test_source_open_sme_empty(sme_pair):
    mine, prod = sme_pair
    n_eq = mine.n_equations
    _assert_entrywise_equal(
        mine, prod, _source, ((i, 0) for i in range(n_eq)), "S")
    for i in range(n_eq):
        assert _canon(mine.source[i, 0]) == 0, (
            f"open-SME source row {i} is non-empty: {mine.source[i, 0]}")


# ── (A.6) SlipSME — flux / P / B unchanged, source carries ν/λ friction ────


def test_slip_flux_pressure_ncp_match_production(slip_pair):
    mine, prod = slip_pair
    n_eq, n_st, n_dim = mine.n_equations, mine.n_state, mine.n_dim
    _assert_entrywise_equal(
        mine, prod, _flux,
        ((i, d) for i in range(n_eq) for d in range(n_dim)), "flux")
    _assert_entrywise_equal(
        mine, prod, _pressure,
        ((i, d) for i in range(n_eq) for d in range(n_dim)), "P")
    _assert_entrywise_equal(
        mine, prod, _ncp,
        ((i, j, d) for i in range(n_eq) for j in range(n_st)
         for d in range(n_dim)), "B")


def test_slip_source_matches_production(slip_pair):
    mine, prod = slip_pair
    n_eq = mine.n_equations
    _assert_entrywise_equal(
        mine, prod, _source, ((i, 0) for i in range(n_eq)), "S")
    # The friction is genuinely present on every momentum row.
    nu = mine.parameters.nu
    lam = mine.parameters["lambda"]
    for i, s in enumerate(mine.state):
        if str(s).startswith("q_"):
            assert mine.source[i, 0].has(nu) and mine.source[i, 0].has(lam), (
                f"slip source row {i} ({s}) missing ν/λ friction: "
                f"{mine.source[i, 0]}")
