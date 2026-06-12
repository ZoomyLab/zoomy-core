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


def apply_stress_closures(closures, material, m, mx, tau, state):
    """Inject stress closures at the projected x-momentum (shared by SME / VAM /
    the multilayer models).

    ``closures`` is the composable list (closures.py); ``material`` is the legacy
    MaterialModel fallback used only when ``closures`` is empty.  ``state`` is a
    callable ``at -> ClosureState``.  Boundary traces (surface/bottom) are
    substituted BEFORE the bulk field so the trace substitutions are not
    pre-empted by the bulk rewrite.  Returns True iff a BULK closure was applied
    (else the caller leaves the bulk stress free / modally expanded)."""
    pieces = []                                       # (closes_tag, expression_fn)
    if closures:
        for c in closures:
            c.register(m)
        for c in closures:
            c.check(m)
            pieces.append((c.closes, c.expression))
    elif material is not None:
        if material.surface is not None: pieces.append(("surface", material.surface))
        if material.bottom is not None:  pieces.append(("bottom", material.bottom))
        if material.bulk is not None:    pieces.append(("bulk", material.bulk))
    order = {"surface": 0, "bottom": 1, "bulk": 2}
    pieces.sort(key=lambda p: order[p[0]])
    target = {"surface": tau.at(1), "bottom": tau.at(0), "bulk": tau.expr}
    loc = {"surface": 1, "bottom": 0, "bulk": None}
    has_bulk = False
    for closes, fn in pieces:
        mx.apply({target[closes]: fn(state(loc[closes]))})
        has_bulk = has_bulk or closes == "bulk"
    return has_bulk


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


# ── interface-transfer closures (multilayer) ───────────────────────────────
# The shared transfer velocity u* at an INTERNAL layer interface — the
# numerical scheme for the advection trace exchanged across the interface.
# A distinct closure family: ``expression(below, above, G)`` (the two one-sided
# modal interface traces + the interface mass flux G), not the stress ``(s)``.


class InterfaceFlux(Closure):
    """Base for the multilayer interface transfer-velocity scheme."""
    closes = "interface"; requires = ()

    def expression(self, below, above, G):    # noqa: D401 - distinct signature
        raise NotImplementedError


class MeanInterface(InterfaceFlux):
    """Central interface velocity  u* = (u_below + u_above) / 2."""

    def expression(self, below, above, G):
        return (below + above) / 2


class UpwindInterface(InterfaceFlux):
    """Donor interface velocity by the sign of the interface flux G
    (Hörnschemeyer Eq. 9): u* = u_below if G ≥ 0 else u_above."""

    def expression(self, below, above, G):
        return sp.Piecewise((below, G >= 0), (above, True))


def apply_layer_stress_closures(closures, material, m, mx, tau, state,
                                *, is_top, is_bottom):
    """Per-layer stress-closure injection for the multilayer models.

    Same as :func:`apply_stress_closures`, but the *surface* closure is applied
    only on the TOP layer and the *bottom* closure only on the BED layer (the
    internal interfaces carry no dynamic stress BC).  Returns True iff a bulk
    closure was applied."""
    pieces = []
    if closures:
        stress = [c for c in closures if c.closes in ("bulk", "bottom", "surface")]
        for c in stress:
            c.register(m)
        for c in stress:
            if c.closes == "surface" and not is_top:    continue
            if c.closes == "bottom" and not is_bottom:  continue
            # NB: no c.check(m) here — the multilayer state maps the generic
            # field name ("u") to the per-layer field (u_ℓ) explicitly, so the
            # model-namespace check (which only sees u_1/u_2/…) does not apply.
            pieces.append((c.closes, c.expression))
    elif material is not None:
        if is_top and material.surface is not None:
            pieces.append(("surface", material.surface))
        if is_bottom and material.bottom is not None:
            pieces.append(("bottom", material.bottom))
        if material.bulk is not None:
            pieces.append(("bulk", material.bulk))
    order = {"surface": 0, "bottom": 1, "bulk": 2}
    pieces.sort(key=lambda p: order[p[0]])
    target = {"surface": tau.at(1), "bottom": tau.at(0), "bulk": tau.expr}
    loc = {"surface": 1, "bottom": 0, "bulk": None}
    has_bulk = False
    for closes, fn in pieces:
        mx.apply({target[closes]: fn(state(loc[closes]))})
        has_bulk = has_bulk or closes == "bulk"
    return has_bulk


def interface_closure(closures):
    """Return the InterfaceFlux closure in ``closures`` (or None)."""
    return next((c for c in (closures or []) if c.closes == "interface"), None)


__all__ = ["Closure", "ClosureState", "apply_stress_closures",
           "apply_layer_stress_closures", "interface_closure",
           "Newtonian", "KEpsilonViscosity", "NavierSlip", "RoughWall",
           "StressFree", "InterfaceFlux", "MeanInterface", "UpwindInterface"]
