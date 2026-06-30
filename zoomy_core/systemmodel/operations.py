"""Reusable :meth:`SystemModel.apply` operations.

These are plain callables ``op(system_model)`` consumed through the generic
:meth:`zoomy_core.systemmodel.SystemModel.apply` hook::

    sm.apply(register_aux("hinv", kp_hinv(h, eps)))
    sm.apply(regularize_pow("h", "hinv"))

They replace the per-case "system surgery" a model wrapper used to hand-roll
(walk every operator, substitute, append to ``aux_state``, hand-build a
full-length ``update_aux_variables``, call ``refresh_derived_operators``).  Each
operation owns its own refresh, so callers chain them and never touch the
derived-operator slots by hand.
"""
from __future__ import annotations

from typing import Union

import sympy as sp

from zoomy_core.misc.misc import ZArray

__all__ = [
    "register_aux",
    "regularize_pow",
    "kp_hinv",
    "desingularize_hinv",
    "desingularize_positivity",
    "gate_eigenvalues_dry",
]


# Default wet/dry threshold when the model declares no ``wet_dry_eps``
# parameter (REQ-48).  Small enough that the desingularized ``hinv`` matches
# ``1/h`` everywhere a model without a wetting/drying threshold would actually
# run, but non-zero so the desingularized inverse depth (and the dry-cell
# eigenvalue gate) never divide by / compare against an exactly-dry cell.
_DEFAULT_WET_DRY_EPS = 1e-8


# Plain floating-point floor used for the ``1/h`` of the EIGENVALUE / wave-speed
# velocity (REQ-82).  The wave speed must NOT use the KP-desingularized ``hinv``:
# when ``h* < eps`` the ``max(h*, eps)⁴`` term in ``hinv`` suppresses the
# advective (and gravity) speed by ``(h*/eps)²``, starving the Rusanov
# dissipation at a steep wet/dry front so the centred Audusse mass flux drains a
# dry cell (``h<0``).  Instead the eigenvalue velocity uses the same classical
# floor ``max(1e-14, h)`` the flux velocity already uses, so ``λ = |u·n| +
# √(g·h*)·|n|`` with the TRUE velocity ``u = q/max(1e-14, h)``.
_WAVESPEED_H_FLOOR = 1e-14


# Transcendental nodes whose ARGUMENT must never be touched by the
# ``regularize_pow`` ``1/h -> hinv`` sweep — an ``h`` inside ``log(z/h)`` is
# not a multiplicative factor of a denominator and must stay ``h``.
_TRANSCENDENTAL = (
    sp.log, sp.exp,
    sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc,
    sp.asin, sp.acos, sp.atan, sp.acot, sp.atan2,
    sp.sinh, sp.cosh, sp.tanh, sp.coth,
    sp.asinh, sp.acosh, sp.atanh,
    sp.erf, sp.erfc,
)


# ── KP-desingularized inverse depth ────────────────────────────────────────

def kp_hinv(h, eps):
    """Kurganov–Petrova desingularized inverse depth.

        hinv = √2·h / √(h⁴ + max(h, eps)⁴)

    Equals ``1/h`` for ``h ≥ eps`` but → 0 (not ``1/eps``) as ``h → 0``, so the
    reconstructed velocity ``u = q·hinv → 0`` at a dry front instead of blowing
    up.  This is the reusable building block behind the ``hinv`` aux: feed it to
    :func:`register_aux` and then sweep the operators with
    :func:`regularize_pow`::

        sm.apply(register_aux("hinv", kp_hinv(h, eps), positive=True))
        sm.apply(regularize_pow(h, "hinv"))

    Works symbolically (``h`` a sympy expression) or numerically (``h`` a
    numpy/jax array).  The numerator uses ``Max(h, 0)`` so a transient negative-h
    cell cannot flip ``hinv`` negative (which would wrong-sign ``u = q·hinv`` and
    the wet/dry-front wavespeeds).
    """
    if isinstance(h, sp.Basic) or isinstance(eps, sp.Basic):
        Max, sqrt = sp.Max, sp.sqrt
    else:
        Max, sqrt = max, (lambda x: x ** 0.5)
    return sqrt(2) * Max(h, 0) / sqrt(h ** 4 + Max(h, eps) ** 4)


# ── PHYSICAL operator slots a 1/h→hinv substitution must sweep ─────────────
# The flux/source/diffusion/NCP/pressure/mass operators are where a bare
# ``1/h`` corrupts the wet/dry front, so those are swept.  ``update_variables``
# (the per-cell reconstruction / corrector projection map) and the
# reconstruction maps are DELIBERATELY excluded: they are not wavespeed- or
# flux-bearing, and rewriting them changes the projection update (and, for a
# split corrector, would pull ``hinv`` into the dt-bearing update kernel).
# The derived operators (``quasilinear_matrix`` / source jacobians) are NOT
# listed either — they are recomputed from these primaries by
# ``refresh_derived_operators``.
_PRIMARY_OPERATORS = (
    "flux",
    "hydrostatic_pressure",
    "source",
    "source_explicit",
    "mass_matrix",
    "nonconservative_matrix",
    "diffusion_matrix",
    "diffusion_matrix_explicit",
)


def _resolve_state_symbol(sm, field: Union[str, sp.Symbol]) -> sp.Symbol:
    """Return the state Symbol named ``field`` (accepts the Symbol itself)."""
    name = str(field)
    for s in sm.state:
        if str(s) == name:
            return s
    raise KeyError(
        f"regularize_pow: '{name}' is not a state variable of this "
        f"SystemModel (state = {[str(s) for s in sm.state]}).")


def _resolve_aux_symbol(sm, name: str) -> sp.Symbol:
    for s in sm.aux_state:
        if str(s) == str(name):
            return s
    raise KeyError(
        f"regularize_pow: aux '{name}' is not in aux_state "
        f"({[str(s) for s in sm.aux_state]}); register it first with "
        f"register_aux('{name}', …).")


# ── register_aux ───────────────────────────────────────────────────────────

def register_aux(name: str, formula, **assumptions):
    """Operation: add an algebraic auxiliary variable ``name = formula``.

    Adds the Symbol ``name`` to :attr:`SystemModel.aux_state` and AUTO-AUGMENTS
    :attr:`SystemModel.update_aux_variables` with the row ``name = formula``
    (``formula`` a sympy expression in the state / parameters / existing aux,
    e.g. a KP-desingularized ``1/h``).  The resulting ``update_aux_variables``
    is the FULL-length aux vector — identity (passthrough) on every pre-existing
    row, ``formula`` on the new row — so the per-cell solver leg (which writes
    the lowered update as a prefix of ``Qaux``) populates ``name`` at its true
    row each step without clobbering, or being clobbered by, the other aux
    (the derivative-aux LSQ walk re-fills its own rows afterwards).

    Derived operators are refreshed so the freshly sized aux vector is reflected
    in ``source_jacobian_wrt_aux_variables``.

    ``**assumptions`` are forwarded to the ``Symbol`` (default ``real=True``);
    e.g. ``register_aux("hinv", expr, positive=True)``.
    """
    if not assumptions:
        assumptions = {"real": True}

    def _op(sm):
        sym = sp.Symbol(name, **assumptions)
        if any(str(s) == name for s in sm.aux_state):
            raise ValueError(
                f"register_aux: aux '{name}' is already in aux_state.")
        expr = sp.sympify(formula)

        # Preserve any existing per-row aux formulas (rows a model already
        # declared); identity-passthrough for the rest.  ``update_aux_variables``
        # is the ``(n_aux, 1)`` column convention, so read one scalar per row.
        prev = sm.update_aux_variables
        existing = ([prev[i, 0] for i in range(prev.shape[0])]
                    if prev is not None else [])

        sm.aux_state = list(sm.aux_state) + [sym]

        rows = []
        for i, s in enumerate(sm.aux_state):
            if s is sym:
                rows.append(expr)
            elif i < len(existing):
                rows.append(existing[i])           # keep declared formula
            else:
                rows.append(s)                     # identity passthrough
        sm.update_aux_variables = ZArray([[r] for r in rows])

        # Resize / recompute the aux-dependent derived operators (the source
        # jacobian wrt aux now has one more column); flux/source unchanged here.
        sm.refresh_derived_operators(eigenvalues=False)

    _op.name = "register_aux"
    _op.description = f"add aux '{name}' = {formula}"
    return _op


# ── regularize_pow ─────────────────────────────────────────────────────────

# Derived operators that carry an exact ``∂(…/h)/∂h`` term and therefore must
# be FROZEN-THEN-SUBSTITUTED (not recomputed) by ``regularize_pow`` — see its
# docstring.
_DERIVED_OPERATORS = (
    "quasilinear_matrix",
    "eigenvalues",
    "source_jacobian_wrt_variables",
    "source_jacobian_wrt_aux_variables",
)


def regularize_pow(field: Union[str, sp.Symbol], aux_name: str):
    """Operation: replace every ``field**(-n)`` (``n > 0``) by ``aux_name**n``.

    The conservative change of variables ``u = q/h`` leaves raw ``h**(-1)`` /
    ``h**(-2)`` in the flux / source / NCP; near a dry front those blow up.
    This sweeps every (negative-integer) power of the state ``field`` in every
    PHYSICAL operator and rewrites it in terms of a desingularized auxiliary
    ``aux_name`` (typically registered via :func:`register_aux`), so e.g.
    ``q/h`` becomes ``q·hinv``.

    FREEZE-THEN-SUBSTITUTE — why the derived operators are NOT recomputed.
    ``aux_name`` (``hinv``) is an INDEPENDENT symbol, so once the flux holds
    ``q²·hinv`` a fresh ``∂/∂h`` is ``0`` — the ``−u²`` wavespeed term is lost
    and every flux-Jacobian wavespeed is corrupted (the wet 40% error).  And
    chain-ruling it back via ``∂hinv/∂h = ∂kp_hinv/∂h`` injects a
    ``Heaviside``/``DiracDelta`` mess into every Jacobian entry — the opposite
    of factored codegen.  So instead we:

    1. FORCE-MATERIALIZE the lazy derived operators from the CLEAN
       (pre-substitution) primaries — ``quasilinear_matrix`` and (when present)
       ``eigenvalues`` — so they hold the exact ``∂(q²/h)/∂h = −q²/h²``.  The
       source jacobians are already eager (channeled / refreshed by
       :func:`register_aux`).
    2. SUBSTITUTE ``field**(-n) → aux_name**n`` into the physical operators
       AND those now-frozen derived operators, so ``−q²/h² → −q²·hinv²`` (a few
       ops, no Heaviside) while ``hinv`` stays symbolic.

    No ``refresh_derived_operators`` is called (that would recompute from the
    substituted primaries and re-drop the ``−u²`` term).
    """
    def _op(sm):
        hs = _resolve_state_symbol(sm, field)
        aux = _resolve_aux_symbol(sm, aux_name)

        def _h_multiplicity(base):
            """Multiplicity of ``h`` as a genuine MULTIPLICATIVE factor of
            ``base`` — ``p`` such that ``base / h**p`` is h-free.  Returns 0
            when ``h`` is not a factor, e.g. it only appears inside a
            transcendental arg (``log(z/h)``) or in a sum with no common
            ``h`` (``h + 1``): those must be left untouched.  ``sp.factor``
            pulls the common ``h`` out of an EXPANDED denominator (the
            RoughWall ``-h·(…)²`` form) so its multiplicity is readable from
            ``as_powers_dict``."""
            base = sp.sympify(base)
            if not base.has(hs):
                return 0
            e = sp.factor(base).as_powers_dict().get(hs, sp.S.Zero)
            return int(e) if e.is_Integer and e > 0 else 0

        def _rw(e, inv):
            # Recursive rewrite of ``field**(-n) -> inv**n``, factoring the
            # maximal multiplicative power of ``h`` out of a negative-power
            # base (``h**(-n)``, ``(h**p·f)**(-n)`` AND the expanded
            # ``(Σ h·…)**(-n)`` Add).  ``inv`` is the replacement for ``1/h``:
            # the desingularized ``aux`` (``hinv``) for the flux/source/NCP
            # operators, but the plain floored inverse ``1/Max(1e-14, h)`` for
            # the EIGENVALUE / wave-speed (REQ-82 — see ``_WAVESPEED_H_FLOOR``).
            # Transcendental nodes are returned WHOLE (never descended), so an
            # ``h`` that lives only inside a transcendental arg (``log(z/h)``)
            # is never rewritten — only a genuine multiplicative ``h`` factor
            # becomes ``inv``.
            if isinstance(e, _TRANSCENDENTAL):
                return e
            if (isinstance(e, sp.Pow) and e.exp.is_number
                    and e.exp.is_negative):
                p = _h_multiplicity(e.base)
                if p > 0:
                    # base = q·h**p (q h-free) ⇒ base**e = q**e·inv**(-p·e).
                    # Clean ``h**(-n)`` ⇒ q==1 ⇒ byte-identical to the old
                    # ``inv**(-exp)`` rewrite.
                    q = sp.cancel(e.base / hs ** p)
                    return _rw(q, inv) ** e.exp * inv ** (-p * e.exp)
                return _rw(e.base, inv) ** e.exp
            if e.args:
                return e.func(*[_rw(a, inv) for a in e.args])
            return e

        # 1. Force-materialize the lazy derived operators from the CLEAN
        #    primaries (``quasilinear_matrix`` is a lazy property; touching it
        #    computes ``−q²/h²`` from the still-``1/h`` flux).  ``eigenvalues``
        #    and the source jacobians are already materialized.
        _ = sm.quasilinear_matrix

        # 2. Substitute 1/h**n -> hinv**n in the physical operators AND the
        #    frozen derived operators (so the exact h-derivative terms become
        #    hinv-powers instead of being recomputed away).  EXCEPTION: the
        #    ``eigenvalues`` (the wave-speed / Rusanov-dissipation source) get
        #    the plain ``1/Max(1e-14, h)`` floor, NOT ``hinv`` — the KP
        #    regulariser suppresses the speed by ``(h*/eps)²`` at a wet/dry
        #    front and starves the dissipation, draining the cell negative
        #    (REQ-82).  The flux velocity already uses this same FP floor.
        inv_floor = sp.S.One / sp.Max(sp.Float(_WAVESPEED_H_FLOOR), hs)
        for nm in _PRIMARY_OPERATORS + _DERIVED_OPERATORS:
            M = getattr(sm, nm, None)
            if M is not None:
                inv = inv_floor if nm == "eigenvalues" else aux
                flat = [_rw(sp.sympify(e), inv) for e in sp.flatten(M)]
                setattr(sm, nm, type(M)(flat).reshape(*M.shape))

    _op.name = "regularize_pow"
    _op.description = f"{field}**(-n) -> {aux_name}**n"
    return _op


# ── desingularize_hinv ─────────────────────────────────────────────────────

def _wet_dry_eps(sm, override=None):
    """Resolve the wet/dry threshold ``eps`` for an operation.

    ``override`` (when not ``None``) wins; otherwise the model's REQ-48
    ``wet_dry_eps`` parameter is used when present, falling back to
    :data:`_DEFAULT_WET_DRY_EPS`."""
    if override is not None:
        return sp.sympify(override)
    params = getattr(sm, "parameters", None)
    if params is not None and params.contains("wet_dry_eps"):
        return params.wet_dry_eps
    return sp.Float(_DEFAULT_WET_DRY_EPS)


def _depth_state(sm, what):
    """Return the state Symbol named ``"h"`` or raise a descriptive error."""
    h = next((s for s in sm.state if str(s) == "h"), None)
    if h is None:
        raise ValueError(
            f"{what}: no depth state 'h' found "
            f"(state = {[str(s) for s in sm.state]}).")
    return h


def desingularize_hinv(mode="kp"):
    """Operation: apply the Kurganov–Petrova ``1/h`` desingularization (REQ-67).

    Registers a named ``hinv`` aux carrying the KP inverse depth
    ``√2·h/√(h⁴ + max(h, eps)⁴)`` and rewrites every ``h**(-n)`` in the
    operators to ``hinv**n`` so the flux/source use the desingularized inverse
    depth instead of a bare ``1/h`` — the reusable core form of the per-case
    ``_register_hinv_aux`` the Malpasset SME model used to hand-roll.

    ``mode`` is ``True`` or ``"kp"`` (only the KP variant exists today);
    anything falsy makes the op a no-op.  ``h`` is found generically as the
    state named ``"h"``; ``eps`` is the model's ``wet_dry_eps`` parameter when
    present, else :data:`_DEFAULT_WET_DRY_EPS`."""
    def _op(sm):
        if not mode:
            return
        if isinstance(mode, str) and mode.lower() != "kp":
            raise ValueError(
                f"desingularize_hinv: unknown mode {mode!r}; "
                "expected True or 'kp'.")
        h = _depth_state(sm, "desingularize_hinv")
        eps = _wet_dry_eps(sm)
        if not any(str(s) == "hinv" for s in sm.aux_state):
            sm.apply(register_aux("hinv", kp_hinv(h, eps), positive=True))
        sm.apply(regularize_pow(h, "hinv"))

    _op.name = "desingularize_hinv"
    _op.description = f"KP 1/h desingularization (mode={mode!r})"
    return _op


# ── desingularize_positivity ───────────────────────────────────────────────

def desingularize_positivity(floor):
    """Operation: floor the source's singular dependence on positive state.

    For each ``base**p`` in a source row where ``base`` involves a symbol in
    ``sm.positive_state`` and ``p`` is negative (denominator) or non-integer
    (root), replace ``base`` by ``√(base² + floor²)``.  Recomputes the source
    jacobians from the regularised source.  No-op if the model declares no
    ``positive_state`` (this keeps ν_t=C_μk²/ε finite and the wall-function
    √k(0) real WITHOUT touching the symbolic derivation)."""
    def _op(sm):
        protected = set(getattr(sm, "positive_state", []) or [])
        if not protected:
            return
        f2 = sp.Float(floor) ** 2

        def safe(expr):
            e = sp.sympify(expr)
            repl = {}
            for p in e.atoms(sp.Pow):
                if (p.exp.is_number and (p.exp < 0 or not p.exp.is_integer)
                        and (p.base.free_symbols & protected)):
                    repl[p] = sp.Pow(sp.sqrt(p.base ** 2 + f2), p.exp)
            return e.xreplace(repl) if repl else e

        n = sm.n_equations
        state = list(sm.state)
        new_src = sp.zeros(n, 1)
        for i in range(n):
            new_src[i, 0] = safe(sm.source[i, 0])
        sm.source = ZArray(new_src)
        # The regularised source invalidates the channeled jacobians —
        # re-derive both from the new source (the in-place-mutation fallback,
        # mirroring ``SystemModel.refresh_derived_operators``).
        aux = list(sm.aux_state)
        J = sp.zeros(n, len(state))
        Ja = sp.zeros(n, len(aux))
        for i in range(n):
            si = sp.sympify(sm.source[i, 0])
            for j, s in enumerate(state):
                J[i, j] = sp.diff(si, s)
            for k, a in enumerate(aux):
                Ja[i, k] = sp.diff(si, a)
        sm.source_jacobian_wrt_variables = ZArray(J)
        sm.source_jacobian_wrt_aux_variables = ZArray(Ja)

    _op.name = "desingularize_positivity"
    _op.description = f"floor positive-state singularities (floor={floor})"
    return _op


# ── gate_eigenvalues_dry ───────────────────────────────────────────────────

def gate_eigenvalues_dry(eps=None):
    """Operation: gate the symbolic eigenvalues to 0 in dry cells (``h < eps``).

    Wraps every entry ``e`` of ``sm.eigenvalues`` in ``conditional(h > eps, e,
    0)`` so the wave speeds (and the Rusanov dissipation built from them)
    vanish at dry cells — the reusable core form of the Malpasset SME
    ``ev_gate``.  ``h`` is the state named ``"h"``; ``eps`` defaults to the
    model's ``wet_dry_eps`` parameter when present, else
    :data:`_DEFAULT_WET_DRY_EPS` (pass ``eps=`` to override).

    Also guards every fractional power of ``h`` (e.g. ``sqrt(h**5)`` from the
    desingularized wave speed) with ``Max(., 0)`` so the wet-branch expression
    stays finite at a transient ``h < 0`` (roundoff at the wet/dry front).  This
    matters because a backend ``conditional`` lowered branchless
    (``mask*a + (1-mask)*b``) computes BOTH arms, and ``NaN*0 = NaN`` would leak
    a ``sqrt(negative)`` past the ``h > eps`` gate (REQ-74).  ``Max(.,0)`` is a
    no-op for the physical ``h >= 0`` case, so wet wave speeds are unchanged.

    No-op when ``sm.eigenvalues is None`` (the model emits numerical wave
    speeds, gated by the solver instead)."""
    def _op(sm):
        ev = sm.eigenvalues
        if ev is None:
            return
        h = _depth_state(sm, "gate_eigenvalues_dry")
        e_eps = _wet_dry_eps(sm, eps)
        cond = sp.Function("conditional")

        def _is_frac_pow_of_h(x):
            # sqrt(h**5), h**(5/2), ... — fractional power whose base can go
            # negative because it carries the depth ``h``.
            return (isinstance(x, sp.Pow) and x.exp.is_number
                    and not x.exp.is_integer and x.base.has(h))

        def _guard(e):
            return sp.sympify(e).replace(
                _is_frac_pow_of_h,
                lambda x: sp.Pow(sp.Max(x.base, sp.S.Zero), x.exp))

        gated = [cond(h > e_eps, _guard(e), sp.S.Zero) for e in ev]
        sm.eigenvalues = ZArray(gated).reshape(*ev.shape)

    _op.name = "gate_eigenvalues_dry"
    _op.description = "gate eigenvalues to 0 for dry cells (h < eps); guard sqrt(h) args"
    return _op
