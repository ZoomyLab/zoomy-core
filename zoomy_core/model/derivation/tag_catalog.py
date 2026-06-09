"""Catalog of canonical solver tags and their aliases.

Solver tags route symbolic terms into operator slots on the
``SystemModel``.  They are the only tag system that directly impacts
SystemModel construction; physical / display tags (``Expression.tag``)
live separately and are *not* normalized through this catalog.

Canonical solver-tag names are **explicit and treatment-prefixed** —
they say *what* operator slot the term belongs in and *how* the
solver should treat it in time (explicit vs implicit).  Authoring a
SystemModel therefore looks like

.. code-block:: python

    xmom = xmom.solver_tag(
        flux                = F_x,
        hydrostatic_pressure= P_x,
        implicit_source     = manning_friction,
        explicit_source     = body_force,
        implicit_diffusion  = nu_h_grad_u,
    )

Aliases are kept narrow on purpose: only direct re-orderings of the
canonical name (``source_implicit`` ↔ ``implicit_source``) and a few
legacy synonyms (``viscous``, ``ncp``).  *Physical* names like
``friction``, ``manning``, ``gravity``, ``body_force`` deliberately
do **not** appear here — they belong on physical ``term_groups``
(display), not solver routing.  If a project wants such names to also
classify into solver tags, that mapping should be a separate,
explicit layer on top of this catalog (e.g. a per-model
``physical_to_solver_tag`` table).

Unlike physical ``term_groups`` (which drive ``\\underbrace``
rendering and survive ``apply()``), solver tags live on
``Expression._solver_groups`` and are dropped per-term whenever a
substitution changes their sub-expression (see ``Expression.apply``).

The alias table is mutable: register custom aliases via
``register_alias`` and add new canonical tags via
``register_canonical_tag``.
"""

from __future__ import annotations

from typing import Iterable


CANONICAL_SOLVER_TAGS: set[str] = {
    "flux",
    "nonconservative_flux",
    "hydrostatic_pressure",
    "time_derivative",
    "explicit_source",
    "implicit_source",
    "explicit_diffusion",
    "implicit_diffusion",
}


# Map every canonical solver tag to the corresponding SystemModel slot
# name.  This is the *only* explicit translation layer between the
# tag namespace (the model author's authoring vocabulary) and the
# SystemModel namespace (the solver's storage).
SOLVER_TAG_TO_SLOT: dict[str, str] = {
    "flux":                 "flux",
    "nonconservative_flux": "nonconservative_matrix",
    "hydrostatic_pressure": "hydrostatic_pressure",
    "time_derivative":      "mass_matrix",
    "implicit_source":      "source",
    "explicit_source":      "source_explicit",
    "implicit_diffusion":   "diffusion_matrix",
    "explicit_diffusion":   "diffusion_matrix_explicit",
}


_SOLVER_TAG_ALIASES: dict[str, str] = {
    # ── flux (convective / advective) ──────────────────────────────
    "flux":                 "flux",
    "convection":           "flux",
    "convective":           "flux",
    "convective_flux":      "flux",
    "advection":            "flux",
    "advective":            "flux",
    "advective_flux":       "flux",

    # ── non-conservative product ───────────────────────────────────
    "nonconservative_flux": "nonconservative_flux",
    "nc_flux":              "nonconservative_flux",
    "ncp":                  "nonconservative_flux",
    "nonconservative":      "nonconservative_flux",

    # ── time derivative ────────────────────────────────────────────
    "time_derivative":      "time_derivative",
    "temporal":             "time_derivative",
    "d_dt":                 "time_derivative",

    # ── hydrostatic pressure (flux-like column) ────────────────────
    "hydrostatic_pressure": "hydrostatic_pressure",
    "pressure":             "hydrostatic_pressure",
    "hydro":                "hydrostatic_pressure",

    # ── source — implicit / explicit treatment ─────────────────────
    "implicit_source":      "implicit_source",
    "source_implicit":      "implicit_source",
    "source":               "implicit_source",        # back-compat
    "explicit_source":      "explicit_source",
    "source_explicit":      "explicit_source",

    # ── diffusion (A tensor in div(A:∇Q)) — implicit / explicit ────
    "implicit_diffusion":            "implicit_diffusion",
    "diffusion_implicit":            "implicit_diffusion",
    "diffusion":                     "implicit_diffusion",        # default
    "diffusive":                     "implicit_diffusion",
    "diffusive_flux":                "implicit_diffusion",
    "viscous":                       "implicit_diffusion",
    "viscous_flux":                  "implicit_diffusion",
    "diffusion_matrix":              "implicit_diffusion",        # back-compat
    "diffusion_matrix_implicit":     "implicit_diffusion",
    "explicit_diffusion":            "explicit_diffusion",
    "diffusion_explicit":            "explicit_diffusion",
    "viscous_explicit":              "explicit_diffusion",
    "diffusion_matrix_explicit":     "explicit_diffusion",        # back-compat
}


def canonical_solver_tag(name: str) -> str:
    """Resolve an alias to its canonical solver tag.

    Raises
    ------
    ValueError
        If ``name`` is not a known alias. The error message lists all
        known aliases and their canonical targets.
    """
    try:
        return _SOLVER_TAG_ALIASES[name]
    except KeyError:
        known = ", ".join(
            f"{a}→{c}" for a, c in sorted(_SOLVER_TAG_ALIASES.items())
        )
        raise ValueError(
            f"Unknown solver tag {name!r}. Known aliases: {known}"
        ) from None


def solver_tag_slot(canonical: str) -> str:
    """Return the SystemModel slot name for a canonical solver tag.

    Raises ``KeyError`` if ``canonical`` is not a registered canonical
    solver tag — use :func:`canonical_solver_tag` to normalize an alias
    first.
    """
    return SOLVER_TAG_TO_SLOT[canonical]


def register_alias(alias: str, canonical: str) -> None:
    """Register a new alias → canonical mapping.

    ``canonical`` must be one of the registered canonical tags.
    """
    if canonical not in CANONICAL_SOLVER_TAGS:
        raise ValueError(
            f"Canonical name {canonical!r} is not registered. "
            f"Known canonicals: {sorted(CANONICAL_SOLVER_TAGS)}"
        )
    _SOLVER_TAG_ALIASES[alias] = canonical


def register_canonical_tag(name: str, aliases: Iterable[str] = (),
                           *, slot: str | None = None) -> None:
    """Register a new canonical solver tag with optional aliases.

    The canonical name is always registered as an alias for itself.
    ``slot`` declares which SystemModel field stores the extracted
    operator (defaults to ``name``).
    """
    CANONICAL_SOLVER_TAGS.add(name)
    _SOLVER_TAG_ALIASES[name] = name
    for a in aliases:
        _SOLVER_TAG_ALIASES[a] = name
    SOLVER_TAG_TO_SLOT[name] = slot if slot is not None else name


def known_aliases() -> dict[str, str]:
    """Return a copy of the current alias → canonical mapping."""
    return dict(_SOLVER_TAG_ALIASES)
