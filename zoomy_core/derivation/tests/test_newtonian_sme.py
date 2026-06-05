"""Newtonian normal-stress SME: ``τ_xx`` closed as a DIFFUSIVE flux.

The base SME drops the horizontal extensional stress ``τ_xx → 0``.  The
:class:`~zoomy_core.derivation.models.NewtonianSME` variant KEEPS it and closes
it with the Newtonian normal stress ``τ_xx = 2 μ ∂_x u = 2 ρ ν ∂_x u``
(``μ = ρ ν``), carried GENUINELY through the σ-map, the modal projection and a
CONSERVATIVE second-order Galerkin route (``Multiply → ProductRule → abstract
Integrate → ResolveIntegral``) — nothing is hand-written.

The viscous contribution splits into two physically distinct pieces:

* the genuine **second-order diffusive flux** ``∂_x(F^d)`` lands in the
  SystemModel's ``diffusion_matrix`` (``Fᵈ = A(Q) ∇Q``): a diagonal ``−2 ν`` on
  each momentum mode plus the σ-metric couplings ``A[q_k, h]`` / ``A[q_k, b]``;
* the **σ-metric corrections that are NOT of ``A ∇Q`` form** (bilinear gradients
  ``∂_x b·∂_x q``, ``q·(∂_x h)²``, …) are genuine viscous source terms that
  land in ``source`` — they vanish for a flat geometry (``∂_x b = ∂_x h = 0``)
  and were silently DROPPED by the earlier hand-coded closure.

These tests pin: (1) the diffusion tensor is non-zero and a proper
``(n_eq, n_state, n_dim, n_dim)`` rank-4 tensor, (2) only the momentum rows
carry it, (3) the HYPERBOLIC part — advective flux / hydrostatic pressure / NCP
— is UNCHANGED from the slip-closed SME, and (4) the only ``source`` change is
purely viscous (every differing term carries ``ν``).
"""

import sympy as sp
import pytest

from zoomy_core.derivation.models import NewtonianSME, SlipSME
from zoomy_core.model.models.system_model import SystemModel


_PARAMS = {"g": 9.81, "rho": 1.0, "nu": 1e-3, "lambda": 1e-2}


@pytest.fixture(scope="module")
def newtonian_sm():
    model, ctx = NewtonianSME(N=2, parameters=dict(_PARAMS)).build()
    Q = [ctx["b"], ctx["h"]] + ctx["q_modes"]
    return SystemModel.from_model(model, Q=Q), ctx


@pytest.fixture(scope="module")
def slip_sm():
    model, ctx = SlipSME(N=2, parameters=dict(_PARAMS)).build()
    Q = [ctx["b"], ctx["h"]] + ctx["q_modes"]
    return SystemModel.from_model(model, Q=Q)


# ── (B.1) diffusion_matrix is a non-zero rank-4 tensor ─────────────────────


def test_diffusion_matrix_is_nonzero_rank4(newtonian_sm):
    sm, _ = newtonian_sm
    A = sm.diffusion_matrix
    assert A is not None, "Newtonian τ_xx closure produced no diffusion_matrix"
    assert tuple(A.shape) == (sm.n_equations, sm.n_state, sm.n_dim, sm.n_dim), (
        f"diffusion_matrix shape {tuple(A.shape)} != "
        f"(n_eq, n_state, n_dim, n_dim) = "
        f"({sm.n_equations}, {sm.n_state}, {sm.n_dim}, {sm.n_dim})")
    nonzero = [tuple(idx) for idx in _iter4(A.shape) if A[idx] != 0]
    assert nonzero, "diffusion_matrix is identically zero"


# ── (B.2) only the momentum (q_k) rows carry diffusion ─────────────────────


def test_diffusion_only_on_momentum_rows(newtonian_sm):
    sm, _ = newtonian_sm
    A = sm.diffusion_matrix
    nu = sm.parameters.nu
    momentum_rows = [i for i, s in enumerate(sm.state)
                     if str(s).startswith("q_")]
    bh_rows = [i for i, s in enumerate(sm.state)
               if str(s) in ("b", "h")]
    # b / h rows carry no diffusion.
    for i in bh_rows:
        for j in range(sm.n_state):
            assert sm.diffusion_matrix[i, j, 0, 0] == 0, (
                f"non-momentum row {i} ({sm.state[i]}) carries diffusion")
    # Every momentum row carries a ν viscous coefficient.
    for i in momentum_rows:
        row_terms = [sm.diffusion_matrix[i, j, 0, 0] for j in range(sm.n_state)]
        assert any(e != 0 and sp.sympify(e).has(nu) for e in row_terms), (
            f"momentum row {i} ({sm.state[i]}) carries no ν diffusion: "
            f"{row_terms}")


# ── (B.3) hyperbolic part (F / P / B / mass) unchanged vs slip SME ─────────


def test_hyperbolic_part_unchanged(newtonian_sm, slip_sm):
    """The τ_xx normal-stress closure is PURELY viscous: the advective flux,
    hydrostatic pressure, NCP and mass matrix are EXACTLY those of the slip
    SME (only the diffusion matrix and the viscous source corrections are
    added)."""
    newt, _ = newtonian_sm
    n_eq, n_st, n_dim = newt.n_equations, newt.n_state, newt.n_dim
    for i in range(n_eq):
        for d in range(n_dim):
            assert sp.cancel(newt.flux[i, d] - slip_sm.flux[i, d]) == 0, (
                f"flux[{i},{d}] changed")
            assert sp.cancel(newt.hydrostatic_pressure[i, d]
                             - slip_sm.hydrostatic_pressure[i, d]) == 0, (
                f"pressure[{i},{d}] changed")
        for j in range(n_st):
            for d in range(n_dim):
                assert sp.cancel(newt.nonconservative_matrix[i, j, d]
                                 - slip_sm.nonconservative_matrix[i, j, d]
                                 ) == 0, f"NCP[{i},{j},{d}] changed"
        for j in range(n_st):
            assert sp.cancel(newt.mass_matrix[i, j]
                             - slip_sm.mass_matrix[i, j]) == 0, (
                f"mass_matrix[{i},{j}] changed")


# ── (B.4) the ONLY source change vs slip is the σ-metric viscous correction ─


def test_source_change_is_purely_viscous(newtonian_sm, slip_sm):
    """The Newtonian τ_xx closure changes ``source`` ONLY through the viscous
    σ-metric corrections — every differing term carries ``ν``, and the change
    vanishes when ``ν → 0`` (recovering the slip SME source exactly)."""
    newt, _ = newtonian_sm
    nu = newt.parameters.nu
    changed_any = False
    for i in range(newt.n_equations):
        diff = sp.expand(sp.sympify(newt.source[i, 0])
                         - sp.sympify(slip_sm.source[i, 0]))
        if diff == 0:
            continue
        changed_any = True
        for term in sp.Add.make_args(diff):
            assert term.has(nu), (
                f"non-viscous source change in row {i}: {term}")
        # ν → 0 must collapse the difference to zero (the change IS viscous).
        assert sp.cancel(diff.subs(nu, 0)) == 0, (
            f"source[{i}] change does not vanish at ν=0")
    assert changed_any, (
        "expected the τ_xx closure to add σ-metric viscous source terms")


def _iter4(shape):
    for i in range(shape[0]):
        for j in range(shape[1]):
            for d in range(shape[2]):
                for e in range(shape[3]):
                    yield (i, j, d, e)
