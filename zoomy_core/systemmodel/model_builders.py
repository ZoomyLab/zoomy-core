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
    return sm


# ── MLSWE ────────────────────────────────────────────────────────────────────
def build_mlswe(model) -> SystemModel:
    m = model.derivation
    sm = SystemModel.from_model(
        m, Q=[m.bed, m.ht, *m.q_l])
    return sm


# ── MLSME ────────────────────────────────────────────────────────────────────
def build_mlsme(model) -> SystemModel:
    m = model.derivation
    sm = SystemModel.from_model(
        m, Q=[m.bed, m.ht, *m.q_flat], canonical_source=model)
    return sm


# ── MLVAM ────────────────────────────────────────────────────────────────────
def build_mlvam(model) -> SystemModel:
    m = model.derivation
    sm = SystemModel.from_model(
        m, Q=[m.bed, m.ht, *m.q_flat, *m.r_flat, *m.P_flat],
        canonical_source=model)
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
    # (the ζ-face BC reduction runs in ``_runtime_bcs`` — BCs are attached on
    #  BOTH the cache-hit and the fresh-build path, never inside a builder)
    return sm


def _sigma3d_runtime_bcs(model, sm):
    """CLOSURE → ζ-FACE-BC REDUCTION: the user states the bed/surface conditions
    as physical closures; here the reduction emits them as ordinary solver BCs
    on the ζ-faces (combined with the user's horizontal BCs).

    Runtime, not build-time: ``_vertical_face_bcs`` needs the built ``sm``, but
    the result embeds the case's BCs and must therefore be re-derived for the
    instance rather than served out of the cache."""
    user = model.boundary_conditions
    if user is None:
        horiz_bcs = [Extrapolation(tag="left"), Extrapolation(tag="right")]
    elif isinstance(user, list):
        horiz_bcs = list(user)
    else:
        horiz_bcs = list(user.boundary_conditions_list)
    return horiz_bcs + model._vertical_face_bcs(sm)


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


# Per-kind hook for BCs that the BUILDER can only phrase against the built
# ``sm`` (today: sigma3d's ζ-face reduction).  Everything else uses the model's
# own BC list.  Registered here so BC attachment has exactly ONE home.
_RUNTIME_BCS = {"sigma3d": _sigma3d_runtime_bcs}

# Kinds whose builder used to call ``resolve_and_attach`` UNCONDITIONALLY — an
# EMPTY BC container still produced the trivial identity BC kernel, and the
# goldens record it.  ``swe`` is the exception: its BCs ride the REQ-87
# ``_coupling_bcs`` promotion inside ``_from_model_impl``, which attaches only
# when there is something to attach.
_ALWAYS_ATTACH_BCS = frozenset(_BUILDERS) - {"swe"}


def build_system_model(model) -> SystemModel:
    """Dispatch on ``model._system_model_kind`` to the matching builder.

    Cached BY DEFAULT (REQ-163): the built SystemModel (final lowered operator
    matrices) is stored across memory / user-dir / shipped-prebuilt tiers via
    :mod:`zoomy_core.systemmodel.sm_cache`; a hit skips the derivation AND the
    build entirely and returns a fresh, safely-mutable object.  Parameter
    VALUES are not part of the identity (free symbols end-to-end).  Disable
    with ``ZOOMY_DERIVATION_CACHE=0``; force ``ZOOMY_DERIVATION_REBUILD=1``.

    Cache HIT and fresh BUILD are the SAME code path: the entry is stored
    runtime-free (``sm_cache.strip_runtime_state``) and
    :func:`_attach_runtime_data` fills the instance's parameter values and BCs
    onto the result either way.  A warm run and ``ZOOMY_DERIVATION_CACHE=0``
    must therefore agree byte-for-byte — that equality is the invariant the
    cache owes the tree, and it is gated by
    ``tests/systemmodel/test_cache_runtime_isolation.py``.
    """
    kind = model._system_model_kind
    try:
        builder = _BUILDERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown _system_model_kind {kind!r} on {type(model).__name__}")
    from zoomy_core.systemmodel import sm_cache
    key = sm_cache.cache_key(model, builder)
    sm = sm_cache.fetch(key)
    if sm is None:
        sm = builder(model)
        # The entry must be a pure function of the KEY.  The declarative
        # families already derive on their defaults; the hand-built models
        # build ``parameter_values`` from the INSTANCE, so reset those to the
        # class defaults before the entry is written.
        _reset_parameter_defaults(model, sm)
        sm_cache.store(key, sm)           # stores a BC/IC-stripped copy
        sm_cache.strip_runtime_state(sm)  # ... and the fresh object matches it
    _attach_runtime_data(model, sm)
    return sm


def _write_parameter_values(sm, vals) -> None:
    """Write ``vals`` BY NAME into ``sm``'s own parameter container.

    Never replace the container: the lambdified kernels index the parameter
    vector positionally, so its ordering and length must be preserved.  Names
    the sm does not declare are ignored (a hand-built model's table can be a
    superset once closures have been resolved out)."""
    tgt = getattr(sm, "parameter_values", None)
    if tgt is None or not hasattr(tgt, "keys"):
        return
    for k in tgt.keys():
        if k in vals:
            tgt[k] = float(vals[k])


def _reset_parameter_defaults(model, sm) -> None:
    """Reset ``sm``'s parameter values to the model's CASE-FREE defaults."""
    _write_parameter_values(sm, model.default_parameter_values())


def _attach_runtime_data(model, sm) -> None:
    """Attach the INSTANCE's runtime data to a runtime-free SystemModel.

    Parameter VALUES, BCs and ICs are deliberately NOT part of the cache
    identity — values stay free symbols through the whole derivation (REQ-163)
    and BCs/ICs are case data.  The cache therefore stores entries with those
    fields BLANK and this function is the single place that fills them, on the
    cache-hit path and the fresh-build path alike.

    It is TOTAL, not a patch-up: the entry it is handed carries the model's
    DEFAULTS and no BCs at all, so applying the instance's overrides on top is
    a full rebuild of the runtime state — and the BC slots stay CLEARED when
    the model declares no BCs, instead of inheriting someone else's.
    """
    from zoomy_core.model.boundary_conditions import resolve_and_attach
    # ── parameter values: defaults (already on the entry) | overrides ────
    pv = getattr(model, "parameter_values", None)
    if pv is not None and hasattr(pv, "items"):
        _write_parameter_values(sm, {str(k): v for k, v in pv.items()})
    # ── boundary conditions ─────────────────────────────────────────────
    # Two BC homes (REQ-87): production models POP the constructor
    # ``boundary_conditions=`` into ``_coupling_bcs`` (their param is then a
    # fresh EMPTY default — do not let it shadow the real one); declarative
    # models keep theirs on the param.  Mirror from_model's promotion check:
    # only attach a container that actually HAS entries.
    def _nonempty(bc):
        if bc is None:
            return None
        entries = (list(bc) if isinstance(bc, list)
                   else getattr(bc, "boundary_conditions_list", None))
        return bc if entries else None

    kind = getattr(model, "_system_model_kind", None)
    hook = _RUNTIME_BCS.get(kind)
    if hook is not None:
        bcs = _nonempty(hook(model, sm))
    else:
        bcs = (_nonempty(getattr(model, "_coupling_bcs", None))
               or _nonempty(getattr(model, "boundary_conditions", None)))
        if bcs is None and kind in _ALWAYS_ATTACH_BCS:
            # an EMPTY (but present) container still builds the identity kernel
            bcs = getattr(model, "boundary_conditions", None)
    if bcs is not None:
        resolve_and_attach(
            sm, bcs, aux_bcs=getattr(model, "aux_boundary_conditions", None))
    # ── initial conditions ──────────────────────────────────────────────
    for field in ("initial_conditions", "aux_initial_conditions"):
        setattr(sm, field, getattr(model, field, None))
