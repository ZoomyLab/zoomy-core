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

__all__ = ["register_aux", "regularize_pow"]


# ── operator slots a system-level substitution must sweep ──────────────────
# Every symbolic operator array that can carry a state/aux expression.  The
# derived operators (``quasilinear_matrix`` / source jacobians) are NOT listed:
# they are recomputed from these primaries by ``refresh_derived_operators``.
_PRIMARY_OPERATORS = (
    "flux",
    "hydrostatic_pressure",
    "source",
    "source_explicit",
    "mass_matrix",
    "nonconservative_matrix",
    "diffusion_matrix",
    "diffusion_matrix_explicit",
    "update_variables",
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

def regularize_pow(field: Union[str, sp.Symbol], aux_name: str):
    """Operation: replace every ``field**(-n)`` (``n > 0``) by ``aux_name**n``.

    The conservative change of variables ``u = q/h`` leaves raw ``h**(-1)`` /
    ``h**(-2)`` in the flux / source / NCP; near a dry front those blow up.
    This sweeps every (negative-integer) power of the state ``field`` in every
    operator and rewrites it in terms of a desingularized auxiliary ``aux_name``
    (typically registered via :func:`register_aux`), so e.g. ``q/h`` becomes
    ``q·hinv``.

    The derived operators (jacobians, quasilinear matrix) AND the eigenvalues
    are refreshed INSIDE the operation, so the caller needs no manual
    :meth:`SystemModel.refresh_derived_operators`.
    """
    def _op(sm):
        hs = _resolve_state_symbol(sm, field)
        aux = _resolve_aux_symbol(sm, aux_name)

        def _is_neg_pow(s):
            return (isinstance(s, sp.Pow) and s.base == hs
                    and s.exp.is_number and s.exp.is_negative)

        def _to_aux(s):
            return aux ** (-s.exp)

        for nm in _PRIMARY_OPERATORS:
            M = getattr(sm, nm, None)
            if M is not None:
                setattr(sm, nm, M.replace(_is_neg_pow, _to_aux))

        # Recompute quasilinear / source jacobians from the substituted
        # primaries, then push the same substitution through the eigenvalues
        # (cheap — no spectral re-derivation).
        sm.refresh_derived_operators(eigenvalues=False)
        if sm.eigenvalues is not None:
            sm.eigenvalues = sm.eigenvalues.replace(_is_neg_pow, _to_aux)

    _op.name = "regularize_pow"
    _op.description = f"{field}**(-n) -> {aux_name}**n"
    return _op
