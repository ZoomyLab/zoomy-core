"""Correctness tests for the HLL / HLLC symbolic Riemann solvers.

Uses a minimal in-test Shallow Water ``Model`` (the legacy
``ShallowWaterEquations`` is deprecated) so the test is self-contained
and exercises the dimension-generic code paths (1D and 2D).

Properties checked:
  * **consistency** — ``numerical_flux(q, q) == (F + P) @ n`` (the exact
    physical normal flux when both face states coincide);
  * **upwinding** — under fully supersonic flow the numerical flux
    equals the physical flux of the upwind state;
  * HLLC reduces to the same properties and stays finite on a contact
    (pure-shear) jump.
"""

import numpy as np
import pytest
import sympy as sp
from sympy import Matrix, sqrt

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.basemodel import Model
from zoomy_core.fvm.riemann_solvers import HLL, HLLC
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel


class MiniSWE(Model):
    """Shallow Water Equations, [h, hu, hv(if 2D)], gravity ``g``."""

    def __init__(self, dimension=2, **kwargs):
        var_names = ["h", "hu", "hv"][: dimension + 1]
        super().__init__(
            dimension=dimension,
            variables=var_names,
            parameters={"g": (9.81, "positive")},
            eigenvalue_mode="symbolic",
            **kwargs,
        )

    def flux(self):
        dim = self.dimension
        h = self.variables[0]
        hu = Matrix(self.variables[1:])
        u = hu / h
        g = self._parameter_symbols.g
        F = Matrix.zeros(self.n_variables, dim)
        F[0, :] = hu.T
        F[1:, :] = h * u * u.T + 0.5 * g * h**2 * Matrix.eye(dim)
        return ZArray(F)

    def eigenvalues(self):
        dim = self.dimension
        h = self.variables[0]
        hu = Matrix(self.variables[1:])
        n = Matrix(self.normal[:dim])
        g = self._parameter_symbols.g
        un = (hu.T * n)[0] / h
        c = sqrt(g * h)
        # {un - c, un + c} plus (dim - 1) shear waves at un.
        return ZArray([un - c, un + c] + [un] * (dim - 1))


def _model_runtime(m):
    return NumpyRuntimeModel(m)


def _phys_flux_n(mrt, q, aux, p, n):
    """Exact physical normal flux ``(F + P) @ n`` for a single state."""
    F = np.asarray(mrt.flux(q, aux, p), dtype=float)
    P = np.asarray(mrt.hydrostatic_pressure(q, aux, p), dtype=float)
    return (F + P) @ n


@pytest.fixture(params=[1, 2], ids=["1d", "2d"])
def swe(request):
    dim = request.param
    m = MiniSWE(dimension=dim)
    p = np.array(list(m.parameters.values()), dtype=float)
    aux = np.zeros(m.n_aux_variables)
    return {
        "dim": dim,
        "model": m,
        "mrt": _model_runtime(m),
        "hll": HLL(m).to_runtime_numpy(),
        "hllc": HLLC(m).to_runtime_numpy(),
        "p": p,
        "aux": aux,
    }


def _normal(dim):
    return np.array([1.0] + [0.0] * (dim - 1))


def _state(dim, h, u, v=0.0):
    if dim == 1:
        return np.array([h, h * u])
    return np.array([h, h * u, h * v])


def test_hll_consistency(swe):
    """numerical_flux(q, q) == exact physical normal flux."""
    dim, aux, p = swe["dim"], swe["aux"], swe["p"]
    n = _normal(dim)
    q = _state(dim, h=2.0, u=0.7, v=-0.3)
    expect = _phys_flux_n(swe["mrt"], q, aux, p, n)
    got = np.asarray(swe["hll"].numerical_flux(q, q, aux, aux, p, n), dtype=float)
    np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-9)


def test_hllc_consistency(swe):
    """HLLC is also consistent: numerical_flux(q, q) == physical flux."""
    dim, aux, p = swe["dim"], swe["aux"], swe["p"]
    n = _normal(dim)
    q = _state(dim, h=1.3, u=-0.5, v=0.9)
    expect = _phys_flux_n(swe["mrt"], q, aux, p, n)
    got = np.asarray(swe["hllc"].numerical_flux(q, q, aux, aux, p, n), dtype=float)
    np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-9)


def test_supersonic_right_is_left_flux(swe):
    """u > c > 0 everywhere => flux equals the LEFT physical flux."""
    dim, aux, p = swe["dim"], swe["aux"], swe["p"]
    n = _normal(dim)
    qL = _state(dim, h=1.0, u=20.0, v=1.0)   # u >> sqrt(g h)
    qR = _state(dim, h=0.4, u=15.0, v=-2.0)
    expect = _phys_flux_n(swe["mrt"], qL, aux, p, n)
    for scheme in ("hll", "hllc"):
        got = np.asarray(
            swe[scheme].numerical_flux(qL, qR, aux, aux, p, n), dtype=float
        )
        np.testing.assert_allclose(got, expect, rtol=1e-7, atol=1e-7,
                                   err_msg=f"{scheme} supersonic-right")


def test_supersonic_left_is_right_flux(swe):
    """u < -c < 0 everywhere => flux equals the RIGHT physical flux."""
    dim, aux, p = swe["dim"], swe["aux"], swe["p"]
    n = _normal(dim)
    qL = _state(dim, h=1.0, u=-20.0, v=1.0)
    qR = _state(dim, h=0.4, u=-15.0, v=-2.0)
    expect = _phys_flux_n(swe["mrt"], qR, aux, p, n)
    for scheme in ("hll", "hllc"):
        got = np.asarray(
            swe[scheme].numerical_flux(qL, qR, aux, aux, p, n), dtype=float
        )
        np.testing.assert_allclose(got, expect, rtol=1e-7, atol=1e-7,
                                   err_msg=f"{scheme} supersonic-left")


def test_subsonic_jump_is_finite(swe):
    """On a genuine subsonic jump both schemes stay finite."""
    dim, aux, p = swe["dim"], swe["aux"], swe["p"]
    n = _normal(dim)
    qL = _state(dim, h=2.0, u=0.4, v=0.1)
    qR = _state(dim, h=1.0, u=0.8, v=-0.2)
    for scheme in ("hll", "hllc"):
        got = np.asarray(
            swe[scheme].numerical_flux(qL, qR, aux, aux, p, n), dtype=float
        )
        assert np.all(np.isfinite(got)), f"{scheme} produced non-finite flux"


def test_rotational_antisymmetry(swe):
    """``F(qL, qR; n) == -F(qR, qL; -n)`` — the numerical flux flips sign
    when the face is traversed the other way."""
    dim, aux, p = swe["dim"], swe["aux"], swe["p"]
    n = _normal(dim)
    qL = _state(dim, h=2.0, u=0.4, v=0.1)
    qR = _state(dim, h=1.0, u=0.8, v=-0.2)
    for scheme in ("hll", "hllc"):
        fwd = np.asarray(
            swe[scheme].numerical_flux(qL, qR, aux, aux, p, n), dtype=float
        )
        rev = np.asarray(
            swe[scheme].numerical_flux(qR, qL, aux, aux, p, -n), dtype=float
        )
        np.testing.assert_allclose(fwd, -rev, rtol=1e-7, atol=1e-7,
                                   err_msg=f"{scheme} antisymmetry")


def test_hllc_contact_preserves_normal_momentum_flux():
    """Pure shear (equal h, equal normal velocity, opposite tangential
    velocity): HLLC carries the tangential jump on the contact wave, so
    its mass flux matches the common physical mass flux exactly, whereas
    HLL adds dissipation."""
    m = MiniSWE(dimension=2)
    mrt = _model_runtime(m)
    hll = HLL(m).to_runtime_numpy()
    hllc = HLLC(m).to_runtime_numpy()
    p = np.array(list(m.parameters.values()), dtype=float)
    aux = np.zeros(m.n_aux_variables)
    n = np.array([1.0, 0.0])

    h, un = 1.5, 0.3
    qL = np.array([h, h * un, h * 1.2])
    qR = np.array([h, h * un, h * -1.2])
    fL = _phys_flux_n(mrt, qL, aux, p, n)

    got_hllc = np.asarray(hllc.numerical_flux(qL, qR, aux, aux, p, n), dtype=float)
    got_hll = np.asarray(hll.numerical_flux(qL, qR, aux, aux, p, n), dtype=float)
    # Mass flux + normal-momentum flux are shared by both physical states.
    np.testing.assert_allclose(got_hllc[:2], fL[:2], rtol=1e-7, atol=1e-7)
    # HLL smears the contact -> its tangential-momentum flux differs from HLLC.
    assert abs(got_hll[2] - got_hllc[2]) > 1e-3
