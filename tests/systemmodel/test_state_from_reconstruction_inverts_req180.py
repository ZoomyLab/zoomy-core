"""REQ-180 — ``state_from_reconstruction`` MUST be the left inverse of
``reconstruction_variables``.

The WB primitive map limits the free surface ``eta = b + h`` and the *velocity*
``q/h`` (SWE) / modal velocities ``alpha_k = q_k/h`` (moment models).  Its
inverse has to ``x-h`` every quotient row back to a conservative momentum
(``hu = WB_hu * (eta - b)``).  A stale/duplicate forward map left the velocity
rows as identity, so the round trip returned ``[b, h, hu/h]`` instead of
``[b, h, hu]`` — an O(h) momentum-flux error on wet meshes that detonates into
an order-2 dry-bed ``1/h`` blow-up at wet/dry fronts on every backend that
consumes this emit (numpy / jax / dmplex / foam).

The invariant: with ``WB_<name>`` fed the reconstructed primitives
(``reconstruction_variables``), ``state_from_reconstruction`` returns the exact
conservative state.  Well-balancing is preserved (at rest ``u = 0`` => ``hu = 0``;
``h`` still recovered as ``eta - b``).  No depth flooring anywhere in this path.
"""

import sympy as sp
import pytest

from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.vam import VAM
from zoomy_core.model.reconstruction_inverse import reconstruction_symbols
from zoomy_core.systemmodel.system_model import SystemModel


def _build(spec):
    kind, kw = spec
    return SystemModel.from_model({"SWE": SWE, "SME": SME, "VAM": VAM}[kind](**kw))


CASES = {
    "swe-dim1":     ("SWE", {"dimension": 1}),
    "swe-dim2":     ("SWE", {"dimension": 2}),
    "sme-level1":   ("SME", {"dimension": 2, "level": 1}),
    "vam-level1":   ("VAM", {"dimension": 2, "level": 1}),
}


@pytest.mark.parametrize("label", list(CASES))
def test_state_from_reconstruction_is_left_inverse(label):
    sm = _build(CASES[label])
    fwd = sm.reconstruction_variables
    inv = sm.state_from_reconstruction

    # A model may legitimately opt out of primitive WB reconstruction (VAM
    # well-balances via conservative bed-slope routing, REQ-80).  The invariant
    # then is CONSISTENCY: no forward map => no inverse map.
    if fwd is None:
        assert inv is None, (
            f"{label}: reconstruction_variables is None but "
            f"state_from_reconstruction is {list(inv)} — inconsistent")
        return

    state = list(sm.state)
    fwd = list(fwd)
    inv = list(inv)
    assert inv is not None and len(inv) == len(state)

    # Feed each WB_<name> with the reconstructed primitive, then simplify the
    # inverse: it must collapse to the original conservative state, slot by slot.
    wb = reconstruction_symbols(state)
    subs = {wb[k]: sp.sympify(fwd[k]) for k in range(len(state))}
    for k, s in enumerate(state):
        back = sp.simplify(sp.sympify(inv[k]).xreplace(subs) - s)
        assert back == 0, (
            f"{label}: round trip broken in slot {k} ({s}): "
            f"state_from_reconstruction[{k}] = {inv[k]} feeds back to "
            f"{sp.simplify(sp.sympify(inv[k]).xreplace(subs))}, not {s}")


@pytest.mark.parametrize("label", ["swe-dim1", "swe-dim2", "sme-level1"])
def test_lake_at_rest_maps_zero_velocity_to_zero_momentum(label):
    """Well-balancing: a reconstructed still lake (all velocities 0, flat
    surface) inverts to zero momentum — the inverse never manufactures flow."""
    sm = _build(CASES[label])
    state = list(sm.state)
    inv = list(sm.state_from_reconstruction)
    wb = reconstruction_symbols(state)
    # eta = WB_h, b = WB_b arbitrary but equal-surface; every velocity WB = 0.
    rest = {}
    for k, s in enumerate(state):
        nm = str(s)
        rest[wb[k]] = wb[k] if nm in ("b", "h") else sp.Integer(0)
    for k, s in enumerate(state):
        if str(s) in ("b", "h"):
            continue
        val = sp.simplify(sp.sympify(inv[k]).xreplace(rest))
        assert val == 0, f"{label}: slot {s} momentum nonzero at rest: {val}"
