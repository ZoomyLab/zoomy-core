"""``DivergenceTheorem`` — symbolic integration-by-parts in 2D / 3D.

Walks the expression and, for every ``Integral`` over the supplied
:class:`Domain`, decomposes the integrand into additive components
and rewrites each component matching ``phi · ∂_i F[i]`` (or just
``∂_i F[i]`` in the bare form) via the divergence theorem:

    ``∫_D φ ∇·F dx  →  -∫_D ∇φ·F dx  +  ∮_∂D φ (F·n) ds``

Components that do not match (e.g. the source term ``phi · f`` in a
Poisson residual) are kept inside the original Integral.  Boundary
contributions from each matched component are summed into a single
``BoundaryIntegral`` atom.

The operation is **explicit**: the user names the test function ``φ``
and the flux ``F``.  No auto-tagger.
"""

from __future__ import annotations

from typing import Sequence

import sympy as sp

from zoomy_core.model.models.ins_generator import Operation
from zoomy_core.symbolic.domains import (
    BoundaryIntegral,
    Domain,
    NormalVector,
)


class DivergenceTheorem(Operation):
    """Component-wise divergence theorem on a multi-dimensional Domain.

    Parameters
    ----------
    domain
        The volume domain ``D`` whose integrals get rewritten.
    phi
        Test function (only for ``form='weighted'``).
    F
        Tuple of flux components, ``len(F) == domain.dim``.  Together
        ``F`` is the vector field whose divergence is taken.
    form
        ``'weighted'`` matches each component ``φ · ∂_i F[i]``;
        ``'bare'`` matches each component ``∂_i F[i]``.
    """

    whole_leaf_op = True

    def __init__(self, domain: Domain, *, phi: sp.Expr | None = None,
                 F: Sequence[sp.Expr] | None = None,
                 form: str = "weighted",
                 name: str | None = None,
                 description: str | None = None):
        if form not in ("weighted", "bare"):
            raise ValueError(
                f"form must be 'weighted' or 'bare', got {form!r}.")
        if form == "weighted" and phi is None:
            raise ValueError("form='weighted' requires phi=...")
        if F is None:
            raise ValueError("F=... is required (the flux components).")
        F = tuple(F)
        if len(F) != domain.dim:
            raise ValueError(
                f"len(F)={len(F)} but domain.dim={domain.dim}.")
        super().__init__(
            name=name or "divergence_theorem",
            description=(description or
                         f"Divergence theorem on {domain.name} "
                         f"({form} form, dim={domain.dim})"),
        )
        self._domain = domain
        self._phi = phi
        self._F = F
        self._form = form

    def _leaf_sp(self, expr: sp.Expr) -> sp.Expr:
        # Find every Integral whose limit-vars cover the domain coords;
        # rewrite each in place.  Each Integral whose integrand has no
        # matching component is left as-is — we don't raise on a
        # per-Integral miss, because ``Expression.apply`` invokes us
        # once per additive term, and source terms like ``φ·f`` are
        # legitimate non-matches that should pass through.
        coords = self._domain.coords
        targets = [I for I in expr.atoms(sp.Integral)
                   if frozenset(lim[0] for lim in I.args[1:])
                   == frozenset(coords)]
        if not targets:
            # No Integral over the domain at all in this leaf.  Pass
            # through; this is normal during per-term iteration.
            return expr
        replacements: dict = {}
        for I in targets:
            new_I, _matched = self._rewrite_integral(I)
            if new_I is not I:
                replacements[I] = new_I
        if not replacements:
            return expr
        return expr.xreplace(replacements)

    # --- per-Integral rewrite ----------------------------------------------

    def _rewrite_integral(self, integral: sp.Integral) -> tuple[sp.Expr, bool]:
        integrand = integral.args[0]
        limits = integral.args[1:]
        # Decompose into additive components (linearity of the integral).
        components = sp.Add.make_args(sp.expand(integrand))

        ibp_volume = []
        ibp_boundary = []
        residual = []
        any_match = False
        n_fn = NormalVector(self._domain.boundary())
        coords = self._domain.coords

        for c in components:
            matched = self._match_component(c)
            if matched is None:
                residual.append(c)
                continue
            i = matched
            any_match = True
            if self._form == "weighted":
                # ``c == phi · ∂_i F[i]`` → contributes ``∂_i phi · F[i]``
                # to the volume IBP sum (negated outside the Integral)
                # and ``phi · F[i] · n_i`` to the boundary atom.
                ibp_volume.append(
                    sp.diff(self._phi, coords[i]) * self._F[i])
                ibp_boundary.append(
                    self._phi * self._F[i] * n_fn(i))
            else:  # bare
                ibp_boundary.append(self._F[i] * n_fn(i))
                # No volume term for the bare form.

        if not any_match:
            return integral, False

        pieces = []
        if ibp_volume:
            pieces.append(-sp.Integral(sp.Add(*ibp_volume), *limits))
        if residual:
            pieces.append(sp.Integral(sp.Add(*residual), *limits))
        if ibp_boundary:
            pieces.append(BoundaryIntegral(
                sp.Add(*ibp_boundary), self._domain.boundary()))
        return sp.Add(*pieces), True

    def _match_component(self, c: sp.Expr) -> int | None:
        """Return ``i`` if ``c`` matches the ``i``-th IBP pattern, else None."""
        if self._form == "weighted":
            for i in range(self._domain.dim):
                expected = self._phi * sp.diff(
                    self._F[i], self._domain.coords[i])
                if sp.simplify(sp.expand(c - expected)) == 0:
                    return i
        else:  # bare
            for i in range(self._domain.dim):
                expected = sp.diff(self._F[i], self._domain.coords[i])
                if sp.simplify(sp.expand(c - expected)) == 0:
                    return i
        return None
