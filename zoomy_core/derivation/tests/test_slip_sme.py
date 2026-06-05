"""Milestone-5 acceptance: the declarative SlipSME inserts the slip-Newton
stress closure on top of the inherited OPEN base SME.

The composition under test (inheritance, not adaptation):

* :class:`~zoomy_core.derivation.models.SME` — the BASE model; its ``build``
  leaves the constitutive viscous stress OPEN (momentum rows carry the
  unresolved ``τ_xz(σ=0)`` / ``τ_xz(σ=1)`` boundary atoms + the
  ``∫ ẑ ∂_ẑ τ_xz dẑ`` moment integrals);
* :class:`~zoomy_core.derivation.models.SlipSME` — INHERITS ``SME`` and only
  ADDS the slip-Newton closure (``Substitution`` of the three τ laws + resolve
  of the leftover ``∫…dẑ`` integrals).

GROUND TRUTH is the production model
:class:`zoomy_core.model.models.sme.SME` with
``apply_slip_newton_friction``.  The declarative-framework and model-framework
SME are genuinely SEPARATE layers, so their field heads (``h``, ``q``, …) are
distinct sympy ``Function`` instances; ``_canon`` (shared with
``test_sme_kt19``) maps both onto assumption-free namesakes keyed on
``(name, args)`` so a ``sp.cancel`` diff sees a single symbol identity.  Once
canonicalised the slip-closed rows must match production BIT-EXACT
(``sp.cancel(mine - prod) == 0``) — INCLUDING the now-closed stress terms (the
M4 τ residual vanishes once the constitutive law is committed, which is the
whole point of the closure).
"""

import sympy as sp
import pytest

from zoomy_core.derivation.models import SME, SlipSME, build_sme
from zoomy_core.derivation.tests.test_sme_kt19 import _canon
from zoomy_core.model.models.sme import SME as ProductionSME


_PARAMS = {"g": 9.81, "rho": 1.0, "nu": 1e-3, "lambda": 1e-2}


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def slip_built():
    """The declarative SlipSME closed form."""
    model, ctx = SlipSME(N=2, parameters=dict(_PARAMS)).build()
    return model, ctx


@pytest.fixture(scope="module")
def production_slip():
    """Production ground truth: derive_model + apply_slip_newton_friction."""
    m = ProductionSME(N=2, parameters=dict(_PARAMS))
    m.derive_model()
    m.apply_slip_newton_friction()
    return {eq.name: eq for eq in m}


# ── (1) slip-closed momentum rows match production bit-exact ───────────────


@pytest.mark.parametrize("k", [0, 1, 2])
def test_slip_momentum_row_matches_production(slip_built, production_slip, k):
    """Each slip-closed ``momentum_x_k`` row equals production bit-exact,
    INCLUDING the closed stress terms — the M4 τ residual must vanish once the
    constitutive law is inserted."""
    model, _ = slip_built
    mine = _canon(model._equations[f"momentum_x_{k}"].expr)
    prod = _canon(production_slip[f"momentum_x_{k}"].expr)
    assert sp.cancel(mine - prod) == 0, (
        f"\n SlipSME momentum_x_{k}: {mine}\n production:          {prod}")


# ── (2) closure leaves NO open stress atoms ────────────────────────────────


@pytest.mark.parametrize("k", [0, 1, 2])
def test_slip_closure_leaves_no_open_stress(slip_built, k):
    """After the closure no ``tau_xz`` Function atoms and no ``Integral`` atoms
    remain on any momentum row — the stress is fully resolved into algebraic
    ``ν, λ`` friction terms."""
    model, _ = slip_built
    row = model._equations[f"momentum_x_{k}"].expr
    tau = [a for a in row.atoms(sp.Function) if "tau" in str(a.func)]
    assert not tau, f"momentum_x_{k} still carries τ atoms: {tau}"
    assert not row.atoms(sp.Integral), (
        f"momentum_x_{k} still carries Integral atoms: {row.atoms(sp.Integral)}")
    # The friction parameters ν, λ are present as model parameters.
    params = set(model.parameters.keys())
    assert "nu" in params and "lambda" in params


def test_slip_friction_terms_present(slip_built):
    """The slip closure introduces the algebraic Navier-slip friction
    ``ν·q_k/(λ·h)`` (every row) and the Newtonian bulk friction ``ν·q_k/h²``
    (higher rows) — the friction is algebraic in ``ν, λ``."""
    model, ctx = slip_built
    nu = model.parameters.nu
    lam = model.parameters["lambda"]
    t, x, h, q = ctx["t"], ctx["x"], ctx["h"], ctx["q"]
    for k in range(3):
        row = sp.expand(model._equations[f"momentum_x_{k}"].expr)
        assert row.has(nu) and row.has(lam), (
            f"momentum_x_{k} missing ν/λ friction: {row}")


# ── (3) the BASE SME still leaves stress OPEN ──────────────────────────────


def test_base_sme_leaves_stress_open():
    """``SME(N=2)`` (no ν/λ) is the base model: the momentum rows still carry
    the OPEN constitutive stress — ``τ_xz`` boundary atoms on every row, and the
    ``∫ ẑ ∂_ẑ τ_xz dẑ`` moment integrals on the higher rows."""
    model, ctx = SME(N=2).build()
    row0 = model._equations["momentum_x_0"].expr
    tau0 = [a for a in row0.atoms(sp.Function) if "tau" in str(a.func)]
    # mean row: boundary-trace stress, no moment integral yet.
    assert tau0, f"base momentum_x_0 lost its open τ atoms: {row0}"
    # higher rows: open moment integral present.
    for k in (1, 2):
        row = model._equations[f"momentum_x_{k}"].expr
        assert row.atoms(sp.Integral), (
            f"base momentum_x_{k} lost its open ∫ ẑ ∂_ẑ τ_xz dẑ integral: {row}")
    # The base model declares no constitutive parameters.
    assert "nu" not in model.parameters.keys()
    assert "lambda" not in model.parameters.keys()


def test_base_sme_dynamical_part_matches_kt19():
    """The base (open) SME still reproduces K&T's mass row and the
    ``momentum_x_0`` DYNAMICAL part (flux / gravity / time) bit-for-bit — the
    closure is the only thing that changed between base and derived."""
    model, ctx = SME(N=2).build()
    prod = ProductionSME(N=2)
    prod.derive_model()
    by = {eq.name: eq for eq in prod}

    mass = _canon(model._equations["mass"].expr)
    pmass = _canon(by["continuity_0"].expr)
    assert sp.cancel(mass - pmass) == 0, (
        f"\n mass:       {mass}\n production: {pmass}")

    # momentum_x_0 dynamical part (strip the open τ terms).
    def _strip_tau(expr):
        return sp.Add(*[
            tm for tm in sp.Add.make_args(_canon(expr))
            if not any("tau" in str(a) for a in tm.atoms(sp.Function))
        ])
    mine = _strip_tau(model._equations["momentum_x_0"].expr)
    pprod = _strip_tau(by["momentum_x_0"].expr)
    assert sp.cancel(mine - pprod) == 0, (
        f"\n momentum_x_0 dyn: {mine}\n production:      {pprod}")


# ── (4) the closure makes the M4 higher-row τ residual VANISH ──────────────


@pytest.mark.parametrize("k", [1, 2])
def test_closure_vanishes_m4_tau_residual(slip_built, production_slip, k):
    """The M4 open SME left a non-zero ``τ_xz`` residual on the higher rows
    (recorded by ``test_sme_kt19.test_higher_rows_tau_residual_recorded``).
    Once the slip closure commits the constitutive law, that residual VANISHES:
    the full row diff against production is exactly zero (not merely
    stress-only)."""
    model, _ = slip_built
    mine = _canon(model._equations[f"momentum_x_{k}"].expr)
    prod = _canon(production_slip[f"momentum_x_{k}"].expr)
    full_diff = sp.cancel(mine - prod)
    assert full_diff == 0, (
        f"momentum_x_{k}: slip closure left a residual: {full_diff}")


# ── (5) inheritance is expressed structurally ──────────────────────────────


def test_slipsme_inherits_sme():
    """``SlipSME`` IS an ``SME`` (inheritance, not composition/adaptation), and
    ``build_sme`` is a thin wrapper over the base ``SME.build``."""
    assert issubclass(SlipSME, SME)
    s = SlipSME(N=2, parameters=dict(_PARAMS))
    assert isinstance(s, SME)
    # build_sme(N) == SME(N).build() — base, open stress.
    model, ctx = build_sme(2)
    assert model._equations["momentum_x_1"].expr.atoms(sp.Integral)


def test_slipsme_requires_constitutive_parameters():
    """Building a ``SlipSME`` without ``nu`` / ``lambda`` raises naming the
    missing constitutive parameter (root-cause failure, not a silent no-op)."""
    with pytest.raises(ValueError) as exc:
        SlipSME(N=2, parameters={"g": 9.81, "rho": 1.0}).build()
    assert "nu" in str(exc.value)
