"""REQ-176-firedrake: the base :class:`SWE` carries the wet/dry momentum cap.

The DG(1) shoreline dt-collapse fix (a Riemann/limiter transient leaving a large
momentum on a vanishing depth, so ``u = hu/h`` blows up) is lifted from
:class:`MalpassetSWE` into the SWE base so EVERY SWE consumer inherits it via
``update_variables``.  The cap bounds each momentum component to the admissible
band ``|hu| <= h_wet * 4*sqrt(g*max(h,0))`` with ``h_wet = max(h - eps, 0)`` and
NEVER touches ``h``/``b`` (a plain h-floor merely moves the bad (h,hu) pair up a
threshold — HARD-forbidden).  ``Min``/``Max``/``sqrt`` only, resolved by field
name (dimension-agnostic).

Checks: (a) wet identity (capped == raw), (b) near-dry bound, (c) lake-at-rest
invariance (well-balanced), (d) SystemModel promotion carries it AND MalpassetSWE
keeps its own U_MAX formula byte-identical.
"""
import math

import sympy as sp

from zoomy_core.systemmodel import SystemModel
from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.malpasset import MalpassetSWE

G = 9.81
EPS = 1e-2


def _eval_update(model):
    """Return a callable ``state_dict -> [row_values]`` for the model's
    ``update_variables`` column, substituting g and wet_dry_eps."""
    rows = [sp.sympify(e) for e in sp.flatten(model.update_variables())]
    names = [str(s) for s in model.variables.get_list()]

    def run(**state):
        env = {"g": G, "wet_dry_eps": EPS, **state}
        out = []
        for e in rows:
            subs = {p: env[str(p)] for p in e.free_symbols}
            out.append(float(sp.N(e.subs(subs))))
        return dict(zip(names, out))

    return run


def _bound(h):
    return max(h - EPS, 0.0) * 4.0 * math.sqrt(G * max(h, 0.0))


# ── h/b are STRICTLY untouched — symbolic identity (all states, bit-exact) ─
def test_h_and_b_rows_are_raw_symbols():
    """USER MANDATE: no path may clamp/floor/truncate h (or b).  Prove it at the
    symbolic level — the ``h``/``b`` rows of ``update_variables`` are the RAW
    state symbols (not ``Max(h,0)`` etc.), so h_out == h_in bit-exactly on EVERY
    state.  ``Max(h,0)`` may appear only INSIDE the momentum bound, never as an
    h/b state row (mirrors MalpassetSWE: ``[b, h, cap(hu), cap(hv)]``)."""
    for dim in (1, 2):
        m = SWE(dimension=dim)
        v = m.variables
        rows = list(sp.flatten(m.update_variables()))
        names = [str(s) for s in v.get_list()]
        row = dict(zip(names, rows))
        assert row["b"] is v.b        # raw symbol, not a rewritten expression
        assert row["h"] is v.h
        # nothing in the whole column rewrites b; h appears only inside the cap
        assert not any(sp.sympify(r).has(v.b) for k, r in row.items() if k != "b")


# ── (a) wet identity — capped momentum == raw on wet flow ──────────────────
def test_wet_flow_momentum_unchanged():
    for dim in (1, 2):
        run = _eval_update(SWE(dimension=dim))
        st = {"b": 1.5, "h": 10.0, "hu": 3.0}
        if dim == 2:
            st["hv"] = -2.0
        out = run(**st)
        # h, b untouched BIT-EXACTLY; momenta pass through (well inside the band).
        assert out["b"] == st["b"]
        assert out["h"] == st["h"]
        assert math.isclose(out["hu"], st["hu"], rel_tol=1e-12)
        if dim == 2:
            assert math.isclose(out["hv"], st["hv"], rel_tol=1e-12)


# ── (b) near-dry bound — |capped hu| <= h_wet*4*sqrt(g*h) ───────────────────
def test_near_dry_momentum_bounded():
    run = _eval_update(SWE(dimension=2))
    h = 2.0 * EPS                       # just above the threshold
    out = run(b=0.0, h=h, hu=1e4, hv=-1e4)
    b = _bound(h)
    assert abs(out["hu"]) <= b + 1e-9
    assert abs(out["hv"]) <= b + 1e-9
    # the cap actually bit (raw momentum far exceeded the band)
    assert math.isclose(out["hu"], b, rel_tol=1e-9)
    assert math.isclose(out["hv"], -b, rel_tol=1e-9)
    # h and b are NEVER modified by the cap (not an h-floor)
    assert out["h"] == h
    assert out["b"] == 0.0


# ── (c) lake at rest — nothing changes (well-balanced) ─────────────────────
def test_lake_at_rest_invariant():
    for dim in (1, 2):
        run = _eval_update(SWE(dimension=dim))
        st = {"b": 3.0, "h": 5.0, "hu": 0.0}
        if dim == 2:
            st["hv"] = 0.0
        out = run(**st)
        for k, v in st.items():
            assert out[k] == v


# ── (d) promotion carries it AND MalpassetSWE stays byte-identical ─────────
def test_systemmodel_carries_update_variables():
    sm = SystemModel.from_model(SWE(dimension=2))
    assert sm.update_variables is not None
    rows = [str(e) for e in sp.flatten(sm.update_variables)]
    # b, h pass through; the momentum rows carry the celerity cap (sqrt(g*...)).
    assert rows[0] == "b" and rows[1] == "h"
    assert "sqrt(g)" in rows[2] and "Min(hu" in rows[2]
    assert "sqrt(g)" in rows[3] and "Min(hv" in rows[3]


def test_malpasset_cap_and_jacobian_unchanged():
    mal = MalpassetSWE()
    uv = sp.flatten(mal.update_variables())
    # Malpasset keeps its own U_MAX=30 cap (NOT the base celerity form): no g.
    assert all("g" not in [str(s) for s in sp.sympify(e).free_symbols] for e in uv)
    assert any("30.0" in str(e) for e in uv)
    # jacobian remains explicit zeros
    assert all(e == 0 for e in sp.flatten(mal.update_variables_jacobian_wrt_variables()))
    # KP hinv aux intact
    aux = str(sp.flatten(mal.update_aux_variables())[0])
    assert "sqrt(2)" in aux and "Max(h, wet_dry_eps)" in aux
