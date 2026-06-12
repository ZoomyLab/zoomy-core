"""Stress closures as composable, self-describing operations.

A :class:`Closure` is a single constitutive relation for one component of the
shear stress ``τ_xz`` — the *bulk* field, the *bottom* (bed) trace, or the
*surface* trace.  You compose a model's closure by listing the pieces::

    SME(level=2, closures=[Newtonian(), NavierSlip(), StressFree()])
    SME(level=2, closures=[KEpsilonViscosity(), RoughWall()])   # turbulent

Each closure is a small object that

* declares **which component** it closes — ``closes ∈ {"bulk","bottom","surface"}``;
* declares **which fields it needs** — ``requires`` (asserted against the model);
* **registers its parameters** — ``register(model)`` (register-or-query, with a
  default), so a closure is self-contained and never depends on a constant the
  model author forgot to declare;
* returns its **symbolic relation** — ``expression(s)`` over a full-access
  :class:`~zoomy_core.model.models.material.ClosureState` ``s`` (``s.u``, ``s.k``,
  ``s.dz(...)``, ``s.par.rho`` …), with NO side effects; the model performs the
  substitution.

Closures only *consume* the state a model already declares — they never mint a
new Q/Qaux field.  New physics (k, ε transport) is a new *equation* on a new
model class, not a closure (see the k–ε derivation notebook).

Class-level metadata (``closes``, ``requires``) is IMMUTABLE (str / tuple) so two
instances can never contaminate each other's defaults.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation
from zoomy_core.model.models.material import ClosureState   # the full-access state


class Closure(Operation):
    """Base for a one-component stress closure (see module docstring).

    Subclasses set the immutable class attributes ``closes`` / ``requires`` and
    implement :meth:`expression`; :meth:`register` is optional."""

    closes = None          # "bulk" | "bottom" | "surface"   (immutable)
    requires = ()          # field names the closure consumes (immutable tuple)

    def __init__(self, name=None):
        super().__init__(name=name or type(self).__name__,
                         description=f"{type(self).__name__} ({self.closes})")

    def register(self, model):
        """Register-or-query the parameters this closure needs (override)."""

    def check(self, model):
        """Assert every required field is present in the model."""
        have = set(model.functions.keys())
        for f in self.requires:
            assert f in have, (
                f"{type(self).__name__} closure needs field {f!r}, which this "
                f"model does not define (has: {sorted(have)})")

    def expression(self, s):
        """Return the symbolic relation for this stress component (override)."""
        raise NotImplementedError


# ── bulk closures ──────────────────────────────────────────────────────────


class Newtonian(Closure):
    """Newtonian bulk stress  τ = ρ ν ∂_z u."""
    closes = "bulk"; requires = ("u",)

    def register(self, m):
        m.parameter("nu", 0.0)

    def expression(self, s):
        return s.par.rho * s.par.nu * s.dz(s.u)


class KEpsilonViscosity(Closure):
    """Turbulent bulk stress with the standard k–ε eddy viscosity
    ``τ = ρ ν_t ∂_z u``,  ``ν_t = C_μ k²/ε`` — reads the transported turbulence
    fields ``k`` and ``ε`` (only available on a k–ε model class).  The Galerkin
    projection is rational in ζ → build with ``quadrature_order > 0``."""
    closes = "bulk"; requires = ("u", "k", "varepsilon")

    def register(self, m):
        m.parameter("C_mu", 0.09)

    def expression(self, s):
        return s.par.rho * s.par.C_mu * s.k ** 2 / s.varepsilon * s.dz(s.u)


# ── bottom (bed) closures ──────────────────────────────────────────────────


class NavierSlip(Closure):
    """Navier slip at the bed  τ_b = λ_s · u_b  (linear; the "easy" closure)."""
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        m.parameter("lambda_s", 0.0)

    def expression(self, s):
        return s.par.lambda_s * s.u


class RoughWall(Closure):
    """Turbulent ROUGH-WALL bed drag (OpenFOAM ``nutkRoughWallFunction`` family):

        τ_b = ρ C_f · u_b |u_b|,   C_f = (κ / ln(z_p / z_0))²,   z_0 = k_s/30,

    the physically-correct turbulent replacement for Navier slip.  ``u_b`` is the
    bed velocity trace; ``k_s`` the Nikuradse roughness, ``z_p`` the reference
    height.  The recovered friction velocity ``u_⋆ = √(C_f)|u_b|`` is what feeds
    the k–ε bed sources (Rastogi & Rodi 1978)."""
    closes = "bottom"; requires = ("u",)

    def register(self, m):
        m.parameter("kappa", 0.41); m.parameter("k_s", 1e-3); m.parameter("z_p", 0.1)

    def expression(self, s):
        Cf = (s.par.kappa / sp.log(s.par.z_p / (s.par.k_s / 30))) ** 2
        return s.par.rho * Cf * s.u * sp.Abs(s.u)


# ── surface closures ───────────────────────────────────────────────────────


class StressFree(Closure):
    """Stress-free free surface  τ(1) = 0  (the usual top BC)."""
    closes = "surface"; requires = ()

    def expression(self, s):
        return sp.S.Zero


__all__ = ["Closure", "ClosureState", "Newtonian", "KEpsilonViscosity",
           "NavierSlip", "RoughWall", "StressFree"]
