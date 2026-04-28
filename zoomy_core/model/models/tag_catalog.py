"""Catalog of canonical solver tags and their aliases.

Solver tags route symbolic terms into numerical operators:
``model.flux()`` collects terms tagged ``flux``, ``model.source()`` collects
``source``, and so on.  Aliases (``convection``, ``ncp``, ``viscous``, ...)
normalize to the canonical name.

Unlike physical ``term_groups`` (which drive ``\\underbrace`` rendering and
survive ``apply()``), solver tags live on ``Expression._solver_groups`` and
are dropped per-term whenever a substitution changes their sub-expression
(see ``Expression.apply``).

The alias table is mutable: register custom aliases via ``register_alias``
and add new canonical tags via ``register_canonical_tag``.
"""

from __future__ import annotations

from typing import Iterable


CANONICAL_SOLVER_TAGS: set[str] = {
    "flux",
    "nonconservative_flux",
    "source",
    "time_derivative",
    "diffusive_flux",
    "hydrostatic_pressure",
}


_SOLVER_TAG_ALIASES: dict[str, str] = {
    # flux (convective / advective)
    "flux":                 "flux",
    "convection":           "flux",
    "convective":           "flux",
    "convective_flux":      "flux",
    "advection":            "flux",
    "advective":            "flux",
    "advective_flux":       "flux",

    # non-conservative product
    "nonconservative_flux": "nonconservative_flux",
    "nc_flux":              "nonconservative_flux",
    "ncp":                  "nonconservative_flux",
    "nonconservative":      "nonconservative_flux",

    # source (algebraic right-hand side)
    "source":               "source",
    "algebraic":            "source",
    "reaction":             "source",
    "body_force":           "source",
    "gravity":              "source",

    # time derivative
    "time_derivative":      "time_derivative",
    "temporal":             "time_derivative",
    "d_dt":                 "time_derivative",

    # diffusive flux (needs ∇q at the face)
    "diffusive_flux":       "diffusive_flux",
    "diffusion":            "diffusive_flux",
    "diffusive":            "diffusive_flux",
    "viscous":              "diffusive_flux",
    "viscous_flux":         "diffusive_flux",

    # hydrostatic pressure contribution (flux-like column)
    "hydrostatic_pressure": "hydrostatic_pressure",
    "pressure":             "hydrostatic_pressure",
    "hydro":                "hydrostatic_pressure",
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


def register_canonical_tag(name: str, aliases: Iterable[str] = ()) -> None:
    """Register a new canonical solver tag with optional aliases.

    The canonical name is always registered as an alias for itself.
    """
    CANONICAL_SOLVER_TAGS.add(name)
    _SOLVER_TAG_ALIASES[name] = name
    for a in aliases:
        _SOLVER_TAG_ALIASES[a] = name


def known_aliases() -> dict[str, str]:
    """Return a copy of the current alias → canonical mapping."""
    return dict(_SOLVER_TAG_ALIASES)
