"""End-to-end test of the single canonical SME model.

``SME`` derives via the declarative ``model/derivation`` framework; this checks
the resulting :class:`SystemModel` operators (mass matrix, flux, hydrostatic
pressure, NCP, source) and the registered vertical reconstruction
(``interpolate_to_3d``) at ``level=2`` (K&T 4.17: h, q_0, q_1, q_2 + bed b).
"""
import sympy as sp

from zoomy_core.model.models import SME
from zoomy_core.systemmodel import SystemModel

# the vertical reconstruction is evaluated at position[2] == z (the runtime's
# 3-D eval coordinate), so interpolate_to_3d is written in `z`, not a free `ζ`.
zeta = sp.Symbol("z", real=True)


def _scalar(row):
    return sp.sympify(row[0] if hasattr(row, "__len__") else row)


def _sm():
    return SME(level=2).system_model


def _syms(sm):
    """The SystemModel's own state / aux symbols (carry the right assumptions)."""
    b, h, q0, q1, q2 = sm.state
    aux = {str(s): s for s in sm.aux_state}
    return b, h, q0, q1, q2, aux


def test_sme_builds_a_systemmodel():
    sm = _sm()
    assert isinstance(sm, SystemModel)
    assert [str(s) for s in sm.state] == ["b", "h", "q_0", "q_1", "q_2"]


def test_sme_mass_matrix_is_legendre_gram():
    sm = _sm()
    M = sp.Matrix(sm.mass_matrix.tolist())
    assert M == sp.diag(0, 1, 1, sp.Rational(1, 3), sp.Rational(1, 5))


def test_sme_flux_and_hydrostatic_pressure():
    sm = _sm()
    b, h, q0, q1, q2, _ = _syms(sm)
    g = sm.parameters.g
    F = [_scalar(r) for r in sm.flux]
    P = [_scalar(r) for r in sm.hydrostatic_pressure]
    # q_0 momentum row: conservative flux + hydrostatic g h^2/2
    assert sp.simplify(F[2] - (q0**2/h + q1**2/(3*h) + q2**2/(5*h))) == 0
    assert sp.simplify(P[2] - g*h**2/2) == 0
    assert P[0] == 0 and P[1] == 0 and P[3] == 0 and P[4] == 0


def test_sme_source_carries_slip_and_viscous_friction():
    sm = _sm()
    src = [_scalar(r) for r in sm.source]
    nu, lam = sp.symbols("nu lambda_s", positive=True)
    # q_1, q_2 rows carry the −(4ν q_1/h²) / −(12ν q_2/h²) viscous friction
    assert src[3].has(nu) and src[4].has(nu)
    assert src[3].has(lam) and src[4].has(lam)


def test_interpolate_is_self_contained_and_recovers_bottom_kbc():
    sm = _sm()
    b, h, q0, q1, q2, aux = _syms(sm)
    dbdx = aux["dbdx"]
    w = _scalar(list(sm.interpolate_to_3d)[4])
    # the ŵ_j are inlined — no dangling modal coefficient survives
    assert not any("hat{w}" in str(s) for s in w.free_symbols)
    # w(ζ=0) = u(0)·∂_x b = (q_0 − q_1 + q_2)/h · ∂_x b  (bottom no-penetration KBC)
    w0 = sp.simplify(w.subs(zeta, 0) - dbdx*(q0 - q1 + q2)/h)
    assert w0 == 0


def test_interpolate_u_is_modal_reconstruction():
    sm = _sm()
    b, h, q0, q1, q2, _ = _syms(sm)
    u = _scalar(list(sm.interpolate_to_3d)[2])
    # u(ζ) = Σ_i (q_i/h) φ_i(ζ) with shifted Legendre φ
    expected = sum((qi/h) * sp.legendre(i, 2*zeta - 1)
                   for i, qi in enumerate((q0, q1, q2)))
    assert sp.simplify(u - expected) == 0
