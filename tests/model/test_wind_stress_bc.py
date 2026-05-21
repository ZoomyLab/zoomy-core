"""Smoke tests for the WindStress boundary condition (9b).

Mixed Neumann + Dirichlet BC for POM-style surface wind stress
(BM87 §2.15) — some state rows get a prescribed value, others get a
prescribed face-normal gradient.  Validates the symbolic build path
that the Firedrake-DG / codegen backends consume.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.boundary_conditions import WindStress


def _make_inputs(n_vars=4, n_aux=1, n_param=4):
    """Build the (time, X, dX, Q, Qaux, parameters, normal) symbols
    that ``compute_boundary_*`` expect.  All as Zstructs of Symbols,
    matching the live BC kernel signature."""
    time = sp.Symbol("t", real=True)
    X = Zstruct(**{f"X{d}": sp.Symbol(f"X{d}", real=True) for d in range(3)})
    dX = sp.Symbol("dX", real=True)
    Q = Zstruct(**{f"Q{i}": sp.Symbol(f"Q{i}", real=True) for i in range(n_vars)})
    Qaux = Zstruct(**{f"Qaux{i}": sp.Symbol(f"Qaux{i}", real=True)
                      for i in range(n_aux)})
    parameters = Zstruct(
        u_star=sp.Symbol("u_star", positive=True),
        B1=sp.Symbol("B1", positive=True),
        K_M_eff=sp.Symbol("K_M_eff", positive=True),
        T_ref=sp.Symbol("T_ref", real=True),
    )
    normal = Zstruct(**{f"n{d}": sp.Symbol(f"n{d}", real=True) for d in range(3)})
    return time, X, dX, Q, Qaux, parameters, normal


def test_compute_boundary_condition_applies_dirichlet_rows():
    """Dirichlet rows take the prescribed value; others extrapolate
    (=interior value passed through)."""
    time, X, dX, Q, Qaux, parameters, normal = _make_inputs()

    bc = WindStress(
        tag="surface",
        prescribe_values={
            2: parameters.B1 ** sp.Rational(2, 3) * parameters.u_star ** 2,
            3: 0,
        },
        prescribe_gradients={
            0: parameters.u_star ** 2 / parameters.K_M_eff,
            1: 0,
        },
    )
    out = bc.compute_boundary_condition(time, X, dX, Q, Qaux, parameters, normal)

    # Rows 0, 1 — extrapolated (no Dirichlet spec).
    assert out[0] == Q.Q0
    assert out[1] == Q.Q1
    # Row 2 — Dirichlet.
    assert sp.simplify(
        out[2] - parameters.B1 ** sp.Rational(2, 3) * parameters.u_star ** 2
    ) == 0
    # Row 3 — Dirichlet 0.
    assert out[3] == 0


def test_compute_boundary_gradient_applies_neumann_rows():
    """Neumann rows take the prescribed gradient; others zero."""
    time, X, dX, Q, Qaux, parameters, normal = _make_inputs()

    bc = WindStress(
        tag="surface",
        prescribe_gradients={
            0: parameters.u_star ** 2 / parameters.K_M_eff,
            1: 0,
        },
        prescribe_values={2: 0, 3: 0},
    )
    out = bc.compute_boundary_gradient(time, X, dX, Q, Qaux, parameters, normal)

    # Row 0: wind stress gradient = u_*² / K_M.
    assert sp.simplify(
        out[0] - parameters.u_star ** 2 / parameters.K_M_eff
    ) == 0
    # Row 1: prescribed zero Neumann (no heat flux).
    assert out[1] == 0
    # Rows 2, 3: prescribed via values, default-zero on the gradient side.
    assert out[2] == 0
    assert out[3] == 0


def test_face_value_numeric_path():
    """Numpy face_value: numeric Dirichlet specs apply; others extrapolate."""
    bc = WindStress(
        tag="surface",
        prescribe_values={2: 4.2e-3, 3: 0.0},
        prescribe_gradients={0: 1.0e-2, 1: 0.0},
    )
    Q_inner = np.array([0.5, 18.0, 1.0e-4, 1.0e-5], dtype=float)
    out = bc.face_value(Q_inner, Qaux_inner=np.array([]),
                        normal=np.array([1.0]), d_face=0.5,
                        time=0.0, parameters=np.array([]))
    np.testing.assert_allclose(out[:2], Q_inner[:2])
    np.testing.assert_allclose(out[2], 4.2e-3)
    np.testing.assert_allclose(out[3], 0.0)


def test_face_gradient_numeric_path():
    """Numpy face_gradient: numeric Neumann specs apply; others zero."""
    bc = WindStress(
        tag="surface",
        prescribe_gradients={0: 1.0e-2, 1: 0.0},
        prescribe_values={2: 0.0, 3: 0.0},
    )
    Q_inner = np.array([0.5, 18.0, 1.0e-4, 1.0e-5], dtype=float)
    Q_face = Q_inner.copy()
    out = bc.face_gradient(Q_inner, Q_face, Qaux_inner=np.array([]),
                           normal=np.array([1.0]), d_face=0.5,
                           time=0.0, parameters=np.array([]))
    np.testing.assert_allclose(out, [1.0e-2, 0.0, 0.0, 0.0])
