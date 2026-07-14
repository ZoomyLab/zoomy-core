"""Per-model SystemModel builders (REQ-143).

A :class:`~zoomy_core.model.basemodel.Model` must NOT know that
:class:`~zoomy_core.systemmodel.system_model.SystemModel` exists.  The single
public entry point for turning a model instance into its runtime system is
:meth:`SystemModel.from_model`; when handed a production model it delegates
HERE, dispatching on the model's plain-string ``_system_model_kind`` class
attribute to the matching builder below.

Each builder holds the EXACT body that the model's old ``system_model``
@property used to run (``self`` → ``model``): Q selection, the manual
hydrostatic-pressure tag/un-tag, ``canonical_source=model``, the β-HSWME
spectrum registration, the positivity flags, and the boundary-condition
resolution.  The builders call the LOW-LEVEL
``SystemModel.from_model(model.derivation, Q=...)`` path (the declarative
model + Q signature), which is untouched.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.coords import t, x, y
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.model.derivation.system_extract import HydrostaticPressure
from zoomy_core.model.boundary_conditions import resolve_and_attach, Extrapolation


# ── SWE ─────────────────────────────────────────────────────────────────────
def build_swe(model) -> SystemModel:
    # ``SystemModel.from_model`` parses the model's function groups AND attaches
    # its ``_coupling_bcs`` on raw promotion (REQ-87), so every backend adapter /
    # FVM solver that builds from the model inherits it.
    return SystemModel._from_model_impl(model)


# ── SME ─────────────────────────────────────────────────────────────────────
def build_sme(model) -> SystemModel:
    m = model.derivation
    dim = int(model.dimension)
    horiz = (x,) if dim == 2 else (x, y)
    qs = list(m.explicit_state())
    # b evolves via the (trivial) bottom equation ∂_t b = 0, so it is
    # already an explicit unknown; prepend only if absent.
    bed = sp.Function("b", real=True)(t, *horiz)
    if bed not in qs:
        qs = [bed, *qs]
    # Manual hydrostatic-pressure tag (one-liner): mark g·h²/2 so the
    # structural extractor routes it to hydrostatic_pressure (well-balanced
    # reconstruction) instead of the conservative flux.  Recomputed from m.
    h = sp.Function("h", positive=True)(t, *horiz)
    pf = m.parameters.g * h ** 2 / 2
    m.apply({pf: HydrostaticPressure(pf)})
    sm = SystemModel.from_model(m, Q=qs, canonical_source=model)
    m.apply({HydrostaticPressure(pf): pf})   # un-tag: leave derivation clean
    model._register_hswme_spectrum(sm)
    resolve_and_attach(sm, model.boundary_conditions,
                       aux_bcs=model.aux_boundary_conditions)
    return sm


# ── VAM ─────────────────────────────────────────────────────────────────────
def build_vam(model) -> SystemModel:
    m = model.derivation
    Nu = int(model.level)
    # Pressure modes carry the SAME horizontal dependence as the rest of the
    # state: dim=2 → P(j,t,x); dim=3 → P(j,t,x,y).
    horiz = (x,) if int(model.dimension) == 2 else (x, y)
    P_modes = [m.functions.P.head(j, t, *horiz) for j in range(Nu + 1)]
    qs = list(m.explicit_state())
    bed = sp.Function("b", real=True)(t, *horiz)
    if bed not in qs:
        qs = [bed, *qs]
    # Manual hydrostatic-pressure tag (one-liner): mark g·h²/2 → pressure.
    h = sp.Function("h", positive=True)(t, *horiz)
    pf = m.parameters.g * h ** 2 / 2
    m.apply({pf: HydrostaticPressure(pf)})
    sm = SystemModel.from_model(m, Q=[*qs, *P_modes], canonical_source=model)
    m.apply({HydrostaticPressure(pf): pf})   # un-tag: leave derivation clean
    resolve_and_attach(sm, model.boundary_conditions,
                       aux_bcs=model.aux_boundary_conditions)
    return sm


# ── MLSWE ────────────────────────────────────────────────────────────────────
def build_mlswe(model) -> SystemModel:
    m = model.derivation
    sm = SystemModel.from_model(
        m, Q=[m.bed, m.ht, *m.q_l])
    resolve_and_attach(sm, model.boundary_conditions,
                       aux_bcs=model.aux_boundary_conditions)
    return sm


# ── MLSME ────────────────────────────────────────────────────────────────────
def build_mlsme(model) -> SystemModel:
    m = model.derivation
    sm = SystemModel.from_model(
        m, Q=[m.bed, m.ht, *m.q_flat], canonical_source=model)
    resolve_and_attach(sm, model.boundary_conditions,
                       aux_bcs=model.aux_boundary_conditions)
    return sm


# ── MLVAM ────────────────────────────────────────────────────────────────────
def build_mlvam(model) -> SystemModel:
    m = model.derivation
    sm = SystemModel.from_model(
        m, Q=[m.bed, m.ht, *m.q_flat, *m.r_flat, *m.P_flat],
        canonical_source=model)
    resolve_and_attach(sm, model.boundary_conditions,
                       aux_bcs=model.aux_boundary_conditions)
    return sm


# ── Sigma3D ──────────────────────────────────────────────────────────────────
def build_sigma3d(model) -> SystemModel:
    m = model.derivation
    qs = list(m.explicit_state())
    bed = sp.Function("b", real=True)(t, x)
    if bed not in qs:
        qs = [bed, *qs]
    # Manual hydrostatic-pressure tag (one-liner): route g·h²/2 → pressure.
    h = sp.Function("h", positive=True)(t, x)
    pf = m.parameters.g * h ** 2 / 2
    m.apply({pf: HydrostaticPressure(pf)})
    sm = SystemModel.from_model(m, Q=qs, canonical_source=model)
    m.apply({HydrostaticPressure(pf): pf})   # un-tag: leave derivation clean
    # CLOSURE → ζ-FACE-BC REDUCTION: the user states the bed/surface conditions
    # as physical closures; here the reduction emits them as ordinary solver BCs
    # on the ζ-faces (combined with the user's horizontal BCs).
    user = model.boundary_conditions
    if user is None:
        horiz_bcs = [Extrapolation(tag="left"), Extrapolation(tag="right")]
    elif isinstance(user, list):
        horiz_bcs = list(user)
    else:
        horiz_bcs = list(user.boundary_conditions_list)
    all_bcs = horiz_bcs + model._vertical_face_bcs(sm)
    resolve_and_attach(sm, all_bcs, aux_bcs=model.aux_boundary_conditions)
    return sm


# ── KESME ────────────────────────────────────────────────────────────────────
def build_kesme(model) -> SystemModel:
    # SME build + flag the k, ε moments as positivity-constrained so
    # ``NumericalSystemModel(regularization.positivity_floor>0)`` floors their
    # singular source dependence (ν_t=C_μk²/ε, wall √k).
    sm = build_sme(model)
    sm.positive_state = [s for s in sm.state
                         if str(s).startswith(("K_", "E_"))]
    return sm


# ── QRKESME ──────────────────────────────────────────────────────────────────
def build_qrkesme(model) -> SystemModel:
    # SME build + flag the sk, se moments as positivity-constrained so the floor
    # guards the 1/sk, 1/se, 1/sk² source prefactors as se (=√ε) passes near zero.
    sm = build_sme(model)
    sm.positive_state = [s for s in sm.state
                         if str(s).startswith(("SK_", "SE_"))]
    return sm


_BUILDERS = {
    "swe": build_swe,
    "sme": build_sme,
    "vam": build_vam,
    "mlswe": build_mlswe,
    "mlsme": build_mlsme,
    "mlvam": build_mlvam,
    "sigma3d": build_sigma3d,
    "kesme": build_kesme,
    "qrkesme": build_qrkesme,
}


def build_system_model(model) -> SystemModel:
    """Dispatch on ``model._system_model_kind`` to the matching builder."""
    kind = model._system_model_kind
    try:
        builder = _BUILDERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown _system_model_kind {kind!r} on {type(model).__name__}")
    return builder(model)
