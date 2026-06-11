"""Material model — stress-closure injection for the depth-resolved models.

A :class:`MaterialModel` is a plain record of three callables (NO
inheritance, no pipeline knowledge), written in CORE variables — the
closure is *defined* on the (t, x, z) equations before any σ-mapping:

* ``bulk(u, dz, par)``   — constitutive relation for the shear stress
  ``τ_xz`` in the bulk, e.g. Newtonian ``par.rho * par.nu * dz(u)``.
  ``u`` is the (applied) horizontal velocity field and ``dz`` the CORE
  vertical-derivative operator (the model class hands in the σ-mapped
  realization ``∂_z = (1/h) ∂_ζ`` — the closure itself never sees σ
  coordinates).
* ``bottom(u_b, par)``   — dynamic BED boundary condition: the bed trace
  of ``τ_xz`` as a function of the slip velocity ``u_b``, e.g. Navier
  slip ``par.lambda_s * u_b``.
* ``surface(u_s, par)``  — dynamic FREE-SURFACE boundary condition,
  usually ``0`` (stress-free) or a wind stress.

It is injected into the model classes exactly like ``level`` /
``n_layers``::

    SME(level=2, material=newtonian_navier_slip())
    MLVAM(n_layers=2, level=1, material=newtonian_navier_slip())

``material=None`` (the DEFAULT) leaves the stress tensor UNCLOSED: no
substitution happens, ``τ_xz`` is expanded in the same modal basis as the
velocity, and its moments ``σ̂_j`` remain free functions in the derived
system — the Kowalski–Torrilhon / Escalante pre-closure form, the right
starting point for deriving a new material model in a notebook.
"""
from __future__ import annotations


class MaterialModel:
    """Stress closure record: ``bulk``, ``bottom``, ``surface`` callables
    (see module docstring).  Entries left ``None`` keep that part of the
    stress unclosed."""

    def __init__(self, *, bulk=None, bottom=None, surface=None,
                 name="material"):
        self.bulk = bulk
        self.bottom = bottom
        self.surface = surface
        self.name = name

    def __repr__(self):
        parts = [k for k in ("bulk", "bottom", "surface")
                 if getattr(self, k) is not None]
        return f"MaterialModel({self.name!r}: {', '.join(parts) or 'unclosed'})"


def newtonian_navier_slip():
    """The standard closure: Newtonian bulk stress ``τ = ρ ν ∂_z u``,
    Navier slip at the bed ``τ_b = λ_s·u_b``, stress-free surface.
    Reproduces the historically hard-coded model closures exactly (the
    term-by-term reference tests pin this against K&T / Escalante)."""
    return MaterialModel(
        bulk=lambda u, dz, par: par.rho * par.nu * dz(u),
        bottom=lambda u_b, par: par.lambda_s * u_b,
        surface=lambda u_s, par: 0,
        name="newtonian+navier-slip",
    )
