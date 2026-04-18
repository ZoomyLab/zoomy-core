"""Collect solver-tagged terms from a DerivedSystem into operator shapes.

Single entry point: :func:`collect_solver_tag`, called from a model's
``flux()`` / ``source()`` / ``nonconservative_matrix()`` / ... bodies.

Per-canonical extraction rules:

* ``flux``, ``diffusive_flux``, ``hydrostatic_pressure``
      Conservative form. Each term must be ``coeff * Derivative(F_i, x_i)``;
      we strip the outermost first-order derivative and place
      ``coeff * F_i`` at column ``i`` of the output row.

* ``nonconservative_flux``
      Each term must be ``coeff * Derivative(q_j, x_i)`` where ``q_j`` is a
      state variable. We read ``(j, i)`` and place ``coeff`` at
      ``out[row, j, i]``.

* ``source``, ``time_derivative``
      Raw expression, placed as-is at ``out[row]``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import sympy as sp
from sympy import Add, Derivative, Mul, S

from zoomy_core.model.models.tag_catalog import canonical_solver_tag


# ---------------------------------------------------------------------------
# Per-canonical extraction helpers (private)
# ---------------------------------------------------------------------------

def _iter_terms(expr):
    return Add.make_args(sp.expand(expr))


def _split_coeff_and_derivative(term):
    """Return (coeff, Derivative) for a term that is Derivative or coeff*Derivative.

    Returns ``(None, None)`` if the term cannot be cleanly split this way.
    """
    if isinstance(term, Derivative):
        return S.One, term
    if isinstance(term, Mul):
        derivs = [a for a in term.args if isinstance(a, Derivative)]
        if len(derivs) != 1:
            return None, None
        deriv = derivs[0]
        coeff = Mul(*[a for a in term.args if a is not deriv])
        return coeff, deriv
    return None, None


def _first_order_direction(deriv, coords):
    """If ``deriv`` is first-order w.r.t. exactly one coord, return its index.

    Handles both ``Derivative(F, x)`` and ``Derivative(F, (x, 1))`` forms.
    Returns ``None`` otherwise.
    """
    variables = deriv.variables
    if len(variables) != 1:
        return None
    v = variables[0]
    if v not in coords:
        return None
    # Order check: second-order would appear as Derivative(F, x, x) with
    # variables == (x, x), length 2. We've already filtered that out above.
    return list(coords).index(v)


def _extract_conservative(expr, coords, n_directions, context,
                          state_variables=None):
    """Return ``[F_0, ..., F_{n_directions-1}]`` for a conservative-form tag.

    Each term of ``expr`` must be ``coeff * Derivative(F_i, x_i)``.
    If ``state_variables`` is provided, the coefficient must not reference
    any of them — catches terms like ``h * d hu / dx`` that are non-conservative.
    Raises ``ValueError`` otherwise with a message naming the offending term.
    """
    state_set = set(state_variables) if state_variables is not None else None
    result = [S.Zero] * n_directions
    for term in _iter_terms(expr):
        if term == S.Zero:
            continue
        coeff, deriv = _split_coeff_and_derivative(term)
        if deriv is None:
            raise ValueError(
                f"{context}: term {term} is not in conservative form "
                f"coeff * d/dx_i(F_i). Rewrite in conservative form or "
                f"use a different solver tag."
            )
        i = _first_order_direction(deriv, coords)
        if i is None:
            raise ValueError(
                f"{context}: term {term} — derivative {deriv} is not first-order "
                f"w.r.t. one of the spatial coords {list(coords)}."
            )
        if i >= n_directions:
            raise ValueError(
                f"{context}: direction index {i} out of range "
                f"(n_directions={n_directions})."
            )
        if state_set is not None:
            coeff_state_refs = state_set & coeff.free_symbols
            if coeff_state_refs:
                raise ValueError(
                    f"{context}: term {term} has state variables "
                    f"{coeff_state_refs} in the coefficient outside the "
                    f"derivative — that is a non-conservative product, not a "
                    f"conservative flux. Move them inside d/dx_i(...) or use "
                    f"the 'nonconservative_flux' tag."
                )
        F_i = coeff * deriv.args[0]
        result[i] = result[i] + F_i
    return result


def _extract_nc(expr, state_variables, coords, context):
    """Return a list of ``(j, i, coeff)`` for a ``sum_{i,j} B_ij d q_j / d x_i`` tag."""
    contributions = []
    for term in _iter_terms(expr):
        if term == S.Zero:
            continue
        coeff, deriv = _split_coeff_and_derivative(term)
        if deriv is None:
            raise ValueError(
                f"{context}: term {term} is not of form coeff * d q_j / d x_i."
            )
        i = _first_order_direction(deriv, coords)
        if i is None:
            raise ValueError(
                f"{context}: derivative {deriv} is not first-order w.r.t. one "
                f"of the spatial coords {list(coords)}."
            )
        inner = deriv.args[0]
        try:
            j = list(state_variables).index(inner)
        except ValueError:
            raise ValueError(
                f"{context}: derivative argument {inner!s} is not a state "
                f"variable. State variables: {list(state_variables)}."
            ) from None
        contributions.append((j, i, coeff))
    return contributions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def collect_solver_tag(
    system,
    tag: str,
    *,
    variable_map: dict,
    n_variables: int,
    n_directions: int = 1,
    state_variables: Optional[Sequence] = None,
    coords: Optional[Sequence] = None,
    policy: str = "strict",
):
    """Assemble a numerical operator from solver-tagged equations.

    Parameters
    ----------
    system
        A DerivedSystem or System with ``.equations`` mapping name -> Expression.
    tag
        Canonical name or alias for the solver tag to collect.
    variable_map
        ``{equation_name: [row_indices]}``. One equation may map to one or
        more output rows. For scalar equations, a single-element list.
    n_variables, n_directions
        Output shape parameters.
    state_variables
        Required for ``nonconservative_flux``. List of symbols that may
        appear as ``Derivative`` arguments in NCP terms.
    coords
        Required for flux-type tags and NCP. Spatial coords ``[x, (y)]``.
    policy
        ``"strict"`` (raise) / ``"warn"`` / ``"ignore"`` if any tagged
        equation has a non-zero ``untagged_remainder()``.

    Returns
    -------
    sp.Matrix, sp.MutableDenseNDimArray, or list, depending on the tag's rank.
    """
    canonical = canonical_solver_tag(tag)

    if canonical in ("flux", "diffusive_flux", "hydrostatic_pressure"):
        if coords is None:
            raise ValueError(f"tag {canonical!r} requires ``coords``.")
        out = sp.Matrix.zeros(n_variables, n_directions)
        kind = "conservative"
    elif canonical == "nonconservative_flux":
        if coords is None or state_variables is None:
            raise ValueError(
                "tag 'nonconservative_flux' requires ``coords`` and ``state_variables``."
            )
        out = sp.MutableDenseNDimArray.zeros(n_variables, n_variables, n_directions)
        kind = "nc"
    elif canonical in ("source", "time_derivative"):
        out = [S.Zero] * n_variables
        kind = "raw"
    else:
        raise ValueError(f"Unsupported canonical tag: {canonical!r}")

    for eq_name, eq in system.equations.items():
        if eq_name not in variable_map:
            continue
        rows = variable_map[eq_name]

        _enforce_untagged_policy(eq, eq_name, policy)

        tag_expr = eq.get_solver_tag(canonical)
        if tag_expr is None or tag_expr == 0:
            continue

        ctx = f"tag {canonical!r} on equation {eq_name!r}"

        if kind == "conservative":
            F = _extract_conservative(tag_expr, coords, n_directions, ctx,
                                      state_variables=state_variables)
            for row in rows:
                for col, F_i in enumerate(F):
                    out[row, col] = out[row, col] + F_i
        elif kind == "nc":
            contribs = _extract_nc(tag_expr, state_variables, coords, ctx)
            for row in rows:
                for (j, i, coeff) in contribs:
                    out[row, j, i] = out[row, j, i] + coeff
        else:  # raw
            for row in rows:
                out[row] = out[row] + tag_expr

    return out


def _enforce_untagged_policy(eq, eq_name, policy):
    """Complain if an equation has solver tags set but a non-zero remainder."""
    if not getattr(eq, "_solver_groups", None):
        return
    remainder = eq.untagged_remainder()
    if remainder == 0:
        return
    msg = (
        f"Equation {eq_name!r}: untagged remainder {remainder!s} "
        f"(solver_tags set: {list(eq.solver_tags.keys())}). "
        f"Either tag all terms or lower untagged_policy."
    )
    if policy == "strict":
        raise ValueError(msg)
    if policy == "warn":
        import warnings
        warnings.warn(msg, stacklevel=3)
    # policy == "ignore" → do nothing
