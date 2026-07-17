"""LOCAL CFL dt (the real adaptive-dt ask, 2026-07-17).

``timestepping.adaptive`` pairs each face's OWN inradius with its OWN local
|λ| and takes the minimum of the local limits — it must never pair the
globally smallest cell with the globally fastest wave (the old scalar-radius
behavior, strictly over-restrictive on non-uniform meshes).  jax landed the
same semantics in 183895a; this pins the numpy/core side.
"""

import numpy as np

from zoomy_core.fvm import timestepping


def _dt(CFL, radii, lams):
    fn = timestepping.adaptive(CFL=CFL, dimension=1, degree=0)
    ev = lambda Q, Qaux, p: np.asarray(lams, float)
    return fn(None, None, None, np.asarray(radii, float), ev)


def test_local_pairing_exact():
    # Face 0: tiny cell, slow wave.  Face 1: big cell, fast wave.
    CFL, radii, lams = 0.5, [0.01, 1.0], [0.1, 10.0]
    expect = min(CFL * 2 * r / l for r, l in zip(radii, lams))
    assert np.isclose(_dt(CFL, radii, lams), expect)


def test_local_beats_global_pessimistic_pairing():
    CFL, radii, lams = 0.5, [0.01, 1.0], [0.1, 10.0]
    dt_local = _dt(CFL, radii, lams)
    dt_global = min(CFL * 2 * min(radii) / l for l in lams)  # old behavior
    assert dt_local > dt_global  # strictly better when small cell is slow


def test_uniform_mesh_unchanged():
    CFL, r, lams = 0.9, 0.25, [1.0, 3.0, 2.0]
    dt_arr = _dt(CFL, [r, r, r], lams)
    dt_scalar = _dt(CFL, r, lams)  # scalar radius still supported
    expect = CFL * 2 * r / max(lams)
    assert np.isclose(dt_arr, expect) and np.isclose(dt_scalar, expect)


def test_dry_faces_do_not_bind():
    # λ = 0 (dry-skip) → local limit inf → ignored by the min.
    dt = _dt(0.5, [0.1, 0.1], [0.0, 2.0])
    assert np.isfinite(dt) and np.isclose(dt, 0.5 * 2 * 0.1 / 2.0)
