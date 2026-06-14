"""Material model — stress-closure injection for the depth-resolved models.

A :class:`MaterialModel` is a plain record of three callables (NO
inheritance, no pipeline knowledge), written in CORE variables — the
closure is *defined* on the (t, x, z) equations before any σ-mapping.

**All three callables share ONE signature** ``f(s)`` — a single
:class:`ClosureState` ``s`` giving the closure FULL access to the model state:

* **fields** (Q and Qaux) by name — ``s.u`` (velocity), ``s.k`` /
  ``s.varepsilon`` (turbulence), ``s.h`` (depth), … — every field the model
  carries.  For a ``bulk`` closure ``s`` exposes the *bulk* field; for the
  ``bottom`` / ``surface`` dynamic boundary closures it exposes the *trace* at
  that interface (ζ = 0 / 1).  This is what lets an eddy viscosity depend on the
  full state, e.g. ``ν_t = C_μ k²/ε``.
* **derivatives** — ``s.dz(e)`` and ``s.dx(e)`` of ANY expression.  These are
  σ-aware: post-σ-map the field is ``u(t,x,ζ)`` and physical ``∂_z = (1/h) ∂_ζ``,
  so ``s.dz`` applies that for you (the closure never needs to know whether the
  system has been σ-mapped).  A constitutive law ``τ = ρν ∂_z u`` is just
  ``s.par.rho*s.par.nu*s.dz(s.u)``.
* **parameters** — ``s.par`` (``s.par.rho``, ``s.par.nu``, …).

``bulk``  — shear stress ``τ_xz`` in the bulk (e.g. ``s.par.rho*s.par.nu*s.dz(s.u)``).
``bottom``— dynamic BED boundary condition: the bed trace of ``τ_xz``
  (Navier slip ``s.par.lambda_s * s.u``; rough-wall drag ``ρ C_f s.u|s.u|``).
``surface``— dynamic FREE-SURFACE boundary condition (usually ``0``).

Inject exactly like ``level`` / ``n_layers``::

    SME(level=2, material=newtonian_navier_slip())
    SME(level=2, material=rough_wall())                 # turbulent bed
    SME(level=2, material=kepsilon_eddy_viscosity())    # ν_t = C_μ k²/ε

``material=None`` (the DEFAULT) leaves the stress tensor UNCLOSED: no
substitution happens, ``τ_xz`` is expanded in the same modal basis as the
velocity, and its moments ``σ̂_j`` remain free functions in the derived
system — the Kowalski–Torrilhon / Escalante pre-closure form, the right
starting point for deriving a new material model in a notebook.
"""
from __future__ import annotations

import sympy as sp


class ClosureState:
    """Single full-access state object handed to a closure ``f(s)``.

    Gives the closure everything it can legitimately need:

    * **fields** — ``s.<name>`` resolves the model field ``<name>`` and returns
      its **bulk expression** (``at=None``) or its **trace** at an interface
      (``at=0`` bottom, ``at=1`` surface).  Any field the model carries (Q or
      Qaux) is reachable: ``s.u``, ``s.k``, ``s.varepsilon``, ``s.h``, ….
    * **derivatives** — ``s.dz(e)`` / ``s.dx(e)`` of any expression, σ-aware
      (``∂_z = (1/h) ∂_ζ`` post-map; ``∂_x`` in the σ-frame).
    * **parameters** — ``s.par``.

    Construct from either the model's attribute namespace ``m.functions``
    (single-layer: any field by name) or an explicit ``{name: FunctionFamily}``
    map (multi-layer: the per-layer field bound to the generic name, e.g.
    ``{"u": u_ell}``).  A
    :class:`~zoomy_core.model.derivation.model.FunctionFamily` exposes ``.expr``
    (the applied bulk field) and ``.at(value)`` (the trace); this accessor picks
    the right one for the closure's location.
    """

    def __init__(self, fields, *, params=None, h=None, x=None, zeta=None,
                 at=None, alias=None, boundary_tag=None, horiz=None):
        # ``object.__setattr__`` so __getattr__ does not recurse on these.
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_params", params)
        object.__setattr__(self, "_h", h)
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_zeta", zeta)
        object.__setattr__(self, "_at", at)
        # Boundary frame: ``boundary_tag`` ∈ {"b","eta"} identifies which
        # interface this closure sits on (bed / free surface) and ``horiz`` the
        # horizontal coords — together they pin the OPAQUE slope symbols the
        # local frame {n, t_α} is built from (see ``s.normal`` / ``s.tangents``).
        object.__setattr__(self, "_btag", boundary_tag)
        object.__setattr__(self, "_horiz", list(horiz) if horiz else [])
        # Field-NAME aliasing for dimension-agnostic closures: e.g. when
        # closing the y-momentum row, ``alias={"u": "v"}`` makes the generic
        # ``s.u`` a closure writes resolve to the y-velocity ``v`` — so a single
        # direction-agnostic closure (``ρν ∂_z u``) closes every horizontal
        # component.  1-D / x-row pass ``{"u": "u"}`` (a no-op).
        object.__setattr__(self, "_alias", alias or {})

    @property
    def par(self):
        """The model parameter namespace (``s.par.rho`` …)."""
        return self._params

    @property
    def zeta(self):
        """The σ-coordinate ζ ∈ [0, 1] — for POSITION-dependent closures
        (e.g. Elder's parabolic ν_t(ζ) = κ u_⋆ h ζ(1−ζ))."""
        return self._zeta

    @property
    def depth(self):
        """The layer depth h (the σ-map scale)."""
        return self._h

    def dz(self, e):
        """Physical vertical derivative ``∂_z`` of ``e`` — σ-aware
        (``∂_z = (1/h) ∂_ζ`` once the model has been σ-mapped)."""
        return sp.Derivative(e, self._zeta) / self._h

    def dx(self, e):
        """Horizontal derivative ``∂_x`` of ``e`` (σ-frame, at fixed ζ)."""
        return sp.Derivative(e, self._x)

    # ── opaque boundary frame {n, t_α} (projected-traction closures) ──────────
    @property
    def _slopes(self):
        """Opaque geometric slopes of THIS boundary's interface, one per
        horizontal direction (built from :func:`equations.frame_slope`)."""
        from zoomy_core.model.models.equations import frame_slope
        from zoomy_core import coords as _C
        cn = {_C.x: "x", _C.y: "y"}
        return [frame_slope(self._btag, cn[c]) for c in self._horiz]

    @property
    def normal(self):
        """Outward unit boundary normal ``n`` as a vector ``[n_x, (n_y,) n_z]``
        in the OPAQUE slopes — ``(−σ_x, (−σ_y,) 1)/√(1+Σσ²)``.  Resolves to
        ``ẑ`` under :func:`equations.small_slope_scaling`."""
        sl = self._slopes
        comps = [-s for s in sl] + [sp.S.One]
        norm = sp.sqrt(sum(c ** 2 for c in comps))
        return sp.Matrix([c / norm for c in comps])

    @property
    def tangents(self):
        """Axis-fixed unit tangent basis ``[t_x, (t_y,)]``: ``t_α`` is the
        surface tangent lying in the ``(x_α, z)`` plane — horizontal part along
        ``ê_α`` — i.e. ``(ê_α + σ_α ẑ)/√(1+σ_α²)``.  Resolves to ``ê_α`` under
        small-slope scaling."""
        sl = self._slopes
        out = []
        for a, s in enumerate(sl):
            vec = [sp.S.Zero] * len(sl) + [s]
            vec[a] = sp.S.One
            out.append(sp.Matrix(vec) / sp.sqrt(1 + s ** 2))
        return out

    @property
    def _vel3(self):
        """The full velocity vector at this boundary trace: ``[u, (v,) w]``."""
        from zoomy_core import coords as _C
        names = ([{_C.x: "u", _C.y: "v"}[c] for c in self._horiz] + ["w"])
        return [getattr(self, n) for n in names]

    def get_normal_tangential(self, vec):
        """Decompose ANY 3-D vector field ``vec = [f_x, (f_y,) f_z]`` into its
        local boundary-frame components — returns ``(f_n, [f_t1, (f_t2)])``: the
        normal projection ``vec·n̂`` and the per-axis tangential projections
        ``vec·t̂_α`` in the slope-aware frame ``{n, t_α}`` (:attr:`normal` /
        :attr:`tangents`).  Velocity → slip / penetration; ``σ·n`` → traction;
        etc.  Dimension-agnostic: one tangent in 2-D, two in 3-D."""
        v = sp.Matrix(vec)
        f_n = (v.T * self.normal)[0]
        f_t = [(v.T * T)[0] for T in self.tangents]
        return f_n, f_t

    @property
    def u_tangent(self):
        """Tangential slip velocity per axis ``[u·t_x, (u·t_y,)]`` — the
        tangential part of :meth:`get_normal_tangential` on the velocity."""
        return self.get_normal_tangential(self._vel3)[1]

    @property
    def u_normal(self):
        """Wall-normal velocity ``u·n`` (zero under no-penetration) — the normal
        part of :meth:`get_normal_tangential` on the velocity."""
        return self.get_normal_tangential(self._vel3)[0]

    def _family(self, name):
        name = self._alias.get(name, name)        # dimension-agnostic rebind
        f = self._fields
        if isinstance(f, dict):
            if name not in f:
                raise AttributeError(
                    f"closure asked for field {name!r}; this model exposes "
                    f"{sorted(f)} to the closure")
            return f[name]
        return getattr(f, name)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        fam = self._family(name)
        return fam.expr if self._at is None else fam.at(self._at)


# MaterialModel and its factories (newtonian_navier_slip, bingham_navier_slip,
# rough_wall, kepsilon_eddy_viscosity) were REMOVED in the clean cut to the
# composable closure list — there is no legacy stress-closure path.  Their
# physics now lives as composable Closure pieces in closures.py:
#   newtonian_navier_slip  →  [Newtonian(), NavierSlip(), StressFree()]
#   bingham_navier_slip    →  [Bingham(), NavierSlip(), StressFree()]  (+ quadrature_order>0)
#   rough_wall             →  [RoughWall()]
#   kepsilon_eddy_viscosity→  [KEpsilonViscosity()]   (+ quadrature_order>0)
# Only ClosureState (above) remains here — the full-access state object f(s).
