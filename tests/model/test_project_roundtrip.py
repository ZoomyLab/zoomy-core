"""``project_from_3d`` is the exact inverse of ``interpolate_to_3d`` on a column.

``interpolate_to_3d`` reconstructs the vertical profile of a conserved-moment
state ``q`` (velocity slot ``u(ζ) = Σ_i (q_i/h)·φ_i(ζ)``); ``project_from_3d``
reduces a sampled column back to the state.  The two must be a round-trip
identity to round-off::

    project_from_3d(interpolate_to_3d(q, ...)) == q

This pins two properties that were each broken at some point:

* the **physical ``×h`` factor** — the projection returns the CONSERVED moment
  ``q_k = h·⟨φ_k, u⟩`` (so the momentum row is ``h·U``, not the bare mean ``U``);
  without it the coupling driver inflates the interface velocity by ``1/h``.
* the **exactness of the discrete projection** — the fixed-node reduction uses
  the FULL discrete-Gram inverse (``Basisfunction.projection_rows``), so it
  recovers every moment to round-off on the uniform column nodes, not just the
  modes the (diagonal-only) Galerkin formula happened to leave orthogonal.  The
  diagonal-only form left an ``O(N_z^-2)`` round-trip error on the even modes
  (``q_0``, ``q_2``, … at SME ≥ 2 and the upper layers of ML).
"""

import numpy as np
import pytest
import sympy as sp
from sympy.core.function import AppliedUndef

from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.ml_sme import MLSME
from zoomy_core.model.models.vam import VAM
from zoomy_core.model.models.ml_vam import MLVAM

_PROFILE = ("b", "h", "u", "v", "w", "p")


def _roundtrip_worst(model, hval=1.7, bval=0.3, layer_split=0.43):
    """Worst |recovered − q| over every state row of one round-trip.

    Build a column by sampling ``interpolate_to_3d`` at exactly the fixed nodes
    each ``project_from_3d`` row references, then project back.  Multi-layer
    models carry a free layer-fraction symbol (``l_1``) shared by both maps and
    a velocity profile that is discontinuous at a layer interface; sample each
    row's own node-range interior (a tiny inward nudge) so the global piecewise
    profile is read on the branch that row's layer projects.
    """
    interp = model.interpolate_to_3d()           # {slot: expr(ζ, state)}
    proj = model.project_from_3d()               # {state_field: expr(P3_*)}
    fields = list(proj.keys())
    b_sym = next(k for k in fields if str(k).startswith("b("))
    h_sym = next(k for k in fields if str(k).startswith("h("))
    q_syms = [k for k in fields if k not in (b_sym, h_sym)]

    all_free = set().union(*(e.free_symbols for e in interp.values()))
    zeta = next(a for a in all_free if str(a) in ("zeta", "xi", "z"))
    extra = {a: layer_split for a in (all_free - set(fields) - {zeta})}  # e.g. l_1
    qv = {q: 0.5 + 0.13 * i for i, q in enumerate(q_syms)}
    state = {b_sym: bval, h_sym: hval, **qv, **extra}

    # numeric velocity-profile callables for the slots the projection samples
    needed = {a.func.__name__[3:]
              for q in q_syms
              for a in sp.sympify(proj[q]).atoms(AppliedUndef)
              if a.func.__name__.startswith("P3_") and len(a.args) == 1}
    slot = {n: i for i, n in enumerate(_PROFILE)}
    # ``.doit()`` collapses any constant spatial derivative the substitution
    # leaves behind — VAM's w-profile carries ∂_x b in its (orthogonal) top
    # mode, so a flat-bed sample column makes it 0 (the projected r_k is
    # unaffected: the top mode is ⊥ to every φ_{k≤Nu}).
    fld = {n: sp.lambdify(zeta, interp[slot[n]].subs(state).doit(), "numpy")
           for n in needed}

    P3b, P3h = sp.Symbol("P3_b", real=True), sp.Symbol("P3_h", real=True)
    worst = 0.0
    for q in q_syms:
        row = sp.sympify(proj[q]).subs(extra)
        samples = [a for a in row.atoms(AppliedUndef)
                   if a.func.__name__.startswith("P3_") and len(a.args) == 1
                   and a.func.__name__[3:] in fld]
        nodes = [float(a.args[0]) for a in samples]
        lo, hi = (min(nodes), max(nodes)) if nodes else (0.0, 1.0)
        sub = {P3b: bval, P3h: hval}
        for a in samples:
            z = float(a.args[0])
            z += 1e-11 if z == lo else (-1e-11 if z == hi else 0.0)
            sub[a] = float(fld[a.func.__name__[3:]](z))
        worst = max(worst, abs(float(row.subs(sub)) - float(qv[q])))
    return worst


@pytest.mark.parametrize("model,label", [
    (SME(level=0, dimension=2), "SME(0)"),
    (SME(level=1, dimension=2), "SME(1)"),
    (SME(level=2, dimension=2), "SME(2)"),
    (SME(level=1, dimension=3), "SME(1) 3-D"),
    (MLSME(n_layers=2, level=2, dimension=2), "MLSME(2,2)"),
    (VAM(level=1, dimension=2), "VAM(1)"),
    (MLVAM(n_layers=2, level=1, dimension=2), "MLVAM(2,1)"),
])
def test_project_inverts_interpolate(model, label):
    """``project_from_3d ∘ interpolate_to_3d`` recovers every conserved moment
    to round-off — the ``×h`` factor and the exact discrete-Gram projection.

    Fails on the diagonal-only projection (SME ≥ 2 / ML upper layers carry an
    ``O(N_z^-2)`` quadrature error); passes with the full Gram inverse."""
    worst = _roundtrip_worst(model)
    assert worst <= 1e-10, f"{label}: round-trip error {worst:.2e} > 1e-10"


def test_momentum_row_carries_h_factor():
    """A constant-``u = U``, depth-``h`` column projects to the CONSERVED
    momentum ``q_0 = h·U`` — not the bare mean ``U``."""
    sm = SME(level=0, dimension=2).system_model
    P = [sp.sympify(e) for e in sp.flatten(sm.project_from_3d)]
    P3h = sp.Symbol("P3_h", real=True)
    U, hval = 1.3, 2.5
    sub = {a: U for e in P for a in e.atoms(AppliedUndef)
           if a.func.__name__ == "P3_u"}
    sub[P3h] = hval
    q0 = float(P[2].subs(sub))                       # state row [b, h, q_0]
    assert q0 == pytest.approx(hval * U, abs=1e-10)  # h·U, NOT U
    assert q0 != pytest.approx(U, abs=1e-6)


def test_vam_pressure_split_in_interpolate():
    """VAM's reconstructed pressure (slot 5) is the TOTAL pressure: the
    hydrostatic ρ g h (1−ζ) split plus the non-hydrostatic modal part, which
    vanishes at the free surface ζ=1 (the p(ζ=1)=0 closure), and the bottom-KBC
    vertical velocity (slot 4) satisfies w(0)=u(0)·∂_x b."""
    from zoomy_core import coords as C
    zeta = sp.Symbol("zeta", real=True)
    interp = VAM(level=1, dimension=2).interpolate_to_3d()
    g, rho = sp.Symbol("g"), sp.Symbol("rho")
    h = sp.Function("h", positive=True)(C.t, C.x)
    b = sp.Function("b", real=True)(C.t, C.x)
    nonhyd = sp.expand(interp[5] - g * rho * h * (1 - zeta))
    assert sp.simplify(nonhyd.subs(zeta, 1)) == 0
    w0 = sp.simplify(interp[4].subs(zeta, 0))
    u0 = sp.simplify(interp[2].subs(zeta, 0))
    assert sp.simplify(w0 - u0 * sp.Derivative(b, C.x)) == 0


def test_sme_projection_on_cell_center_exchange_nodes():
    """REQ-108: the SME projection is baked on VOF-style CELL-CENTRE nodes
    ``(j+1/2)/N_z`` (uniform weights) — NOT endpoint-inclusive ``j/(N-1)`` +
    trapezoidal — so it coincides with the coupling exchange grid (the preCICE
    map is the identity) and the ``project∘interpolate`` round-trip is exact on a
    column sampled at ``(j+1/2)/N_z``.  Endpoint nodes mismatched the exchange
    grid and tipped a dam-break transmission into a reflecting solution.  The
    round-trip exactness itself is covered by ``test_project_inverts_interpolate``
    (which samples each row at exactly the nodes it references); this pins the
    node CONVENTION so it cannot silently regress to endpoints."""
    model = SME(level=1, dimension=2)
    N_z = int(model.project_nz)
    P = [sp.sympify(e) for e in sp.flatten(model.system_model.project_from_3d)]
    nodes = sorted({float(a.args[0])
                    for e in P for a in e.atoms(AppliedUndef)
                    if a.func.__name__ == "P3_u"})
    expected = [(j + 0.5) / N_z for j in range(N_z)]
    assert nodes == pytest.approx(expected, abs=1e-12)
    # the old j/(N-1) convention sampled the endpoints ζ=0 and ζ=1
    assert 0.0 not in nodes and 1.0 not in nodes
