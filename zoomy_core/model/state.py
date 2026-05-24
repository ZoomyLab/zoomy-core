"""StateSpace + MassMomentum — physical-coordinate flow scaffolding.

Per the new design:

* :class:`StateSpace` declares **only coordinates** — `t, x, y (if
  dim >= 2), z`.  Parameters (`g`, `rho`, `nu`, `lambda`) live on
  the `Model`, not here.  σ-frame coordinates (`zeta`, `zeta_ref`)
  are registered later by `SigmaTransform` when it is applied; the
  pristine StateSpace knows nothing about σ-frame.
* :class:`MassMomentum` owns ALL physics state (flow fields ``u, v,
  w, p``; stress tensor ``tau``; free-surface geometry ``h, b, eta``)
  and builds the continuity / momentum.x / momentum.z balance
  equations.  Its constructor takes the Model's `parameters` Zstruct
  so the gravity / density Symbols are the *same* Symbol identity as
  on the Model — no translation maps needed downstream.

Minimal reimplementation: matches the canonical notebook usage
(`thesis/notebooks/legacy/modeling/transparent_derivations/{sme,vam,
ml_sme,ml_vam}_clean.py`), no legacy import.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.operations import Expression


__all__ = ["StateSpace", "MassMomentum"]


class StateSpace:
    """Coordinate scaffold — t, x, [y,] z.

    Carries only the spatial / temporal coordinate Symbols.  No flow
    fields, no parameters, no σ-frame.  σ-frame coordinates (``zeta``,
    ``zeta_ref``) get attached by :class:`SigmaTransform` when applied.
    """

    def __init__(self, dimension=2):
        self.dim = dimension
        self.t = sp.Symbol("t", real=True)
        self.x = sp.Symbol("x", real=True)
        # Note: ``has_y`` is True only for full-3D systems (dim == 3 in
        # the convention of the σ-mapped models — 2D = (x, z)).  The
        # canonical SME / VAM single-layer notebooks use dim=2 (no y).
        self.y = sp.Symbol("y", real=True) if dimension > 2 else None
        # Physical vertical coordinate.  Survives until `SigmaTransform`
        # substitutes it everywhere.
        self.z = sp.Symbol("z", real=True)

    @property
    def has_y(self):
        return self.dim > 2

    @property
    def coords_h(self):
        return [self.x] + ([self.y] if self.has_y else [])

    @property
    def args_h(self):
        return [self.t, self.x] + ([self.y] if self.has_y else [])

    @property
    def args_3d(self):
        return self.args_h + [self.z]

    def __repr__(self):
        fields = "(t,x,y,z)" if self.has_y else "(t,x,z)"
        return f"StateSpace(dim={self.dim}, coords={fields})"


class MassMomentum:
    """The free-surface incompressible mass + momentum balance.

    Owns the *physics* state of the system — flow fields, depth /
    bathymetry / free-surface, stress tensor — and builds the
    continuity / momentum.x / momentum.z balance equations using the
    Model's `parameters` (gravity ``g``, density ``rho``).

    Parameters
    ----------
    state : :class:`StateSpace`
        Coordinate scaffold.  Read-only on this end.
    parameters : Zstruct
        The Model's parameter Symbols Zstruct.  Must contain at least
        ``g`` (gravity) and ``rho`` (density).  These Symbols flow
        directly into the balance equations — no copies, no remap.

    Exposed attributes
    ------------------
    self.state, self.parameters
    self.u, self.v, self.w, self.p   (Function calls on (t,x,[y,]z))
    self.tau                          (Zstruct of stress-tensor components)
    self.h, self.b, self.eta          (depth, bathymetry, free surface)
    self.continuity                   (:class:`Expression`)
    self.momentum                     (Zstruct with .x [, .y] , .z Expressions)
    """

    def __init__(self, state, parameters):
        # Required parameter check — raise a clear ValueError when the
        # caller omitted g or rho.  Without this the AttributeError
        # from ``parameters.g`` deep inside ``_build_momentum`` masks
        # the real issue and hides the missing-parameter contract.
        missing = [k for k in ("g", "rho") if not hasattr(parameters, k)]
        if missing:
            raise ValueError(
                f"symbol {missing[0]!r} required by MassMomentum but "
                f"not in parameters={{...}} — supply via "
                f"`Model(..., parameters={{'g': 9.81, 'rho': 1.0, ...}})`."
            )
        self.state = state
        self.parameters = parameters

        s = state
        args_h = s.args_h
        args_3d = s.args_3d

        # Flow fields — Function calls of (t, x, [y,] z).
        self.u = sp.Function("u", real=True)(*args_3d)
        self.v = sp.Function("v", real=True)(*args_3d) if s.has_y else sp.S.Zero
        self.w = sp.Function("w", real=True)(*args_3d)
        self.p = sp.Function("p", real=True)(*args_3d)

        # Stress tensor — Zstruct of tau_ij Function calls.
        self.tau = self._build_stress(s)

        # Free-surface geometry — depth & bathymetry are functions of
        # (t, x, [y]).
        self.h = sp.Function("h", real=True)(*args_h)
        self.b = sp.Function("b", real=True)(*args_h)
        self.eta = self.b + self.h

        # Build balance equations.
        self.continuity = self._build_continuity()
        self.momentum = self._build_momentum()

    @staticmethod
    def _build_stress(s):
        labels = ["x", "y", "z"] if s.has_y else ["x", "z"]
        tau_dict = {}
        for i in labels:
            for j in labels:
                tau_dict[i + j] = sp.Function(
                    f"tau_{i}{j}", real=True)(*s.args_3d)
        return Zstruct(**tau_dict)

    def _build_continuity(self):
        """∂_x u + ∂_z w (+ ∂_y v in 3D) = 0."""
        s = self.state
        expr = sp.Derivative(self.u, s.x) + sp.Derivative(self.w, s.z)
        if s.has_y:
            expr += sp.Derivative(self.v, s.y)
        return Expression(expr, "continuity")

    def _build_momentum(self):
        """Returns a Zstruct with `.x`, [`.y`,] `.z` Expressions.

        Per equation:
            ∂_t U + ∇·(U·u) + (1/ρ) ∂_axis p − (1/ρ) ∇·τ_axis (− g · δ_axis,z) = 0

        where U is the velocity component for that axis, and the
        gravity term is added only on the z-momentum row.
        """
        s = self.state
        rho = self.parameters.rho
        g = self.parameters.g
        components = {}
        components["x"] = self._build_one_momentum(self.u, "x", rho, gravity=sp.S.Zero)
        if s.has_y:
            components["y"] = self._build_one_momentum(
                self.v, "y", rho, gravity=sp.S.Zero)
        # z-momentum carries the gravity term ``g`` (the acceleration,
        # NOT the body force ``ρg``).  The full equation is
        # ∂_t w + ∇·(w·u) + (1/ρ)∂_z p − (1/ρ)∇·τ_z + g = 0,
        # i.e. divided through by ρ — matches the legacy FullINS
        # convention and K&T 2019.  (The previous ``ρ·g`` form
        # propagated a spurious ρ factor through the hydrostatic
        # pressure substitution into the SME source term.)
        components["z"] = self._build_one_momentum(
            self.w, "z", rho, gravity=g)
        return Zstruct(**components)

    def _build_one_momentum(self, vel, axis, rho, *, gravity):
        s = self.state
        # Temporal.
        temporal = sp.Derivative(vel, s.t)
        # Convection: ∇·(vel · velocity_field) — full divergence.
        convection = (sp.Derivative(vel * self.u, s.x)
                      + sp.Derivative(vel * self.w, s.z))
        if s.has_y:
            convection += sp.Derivative(vel * self.v, s.y)
        # Pressure: (1/ρ) ∂_axis p.
        axis_coord = {"x": s.x, "y": s.y, "z": s.z}[axis]
        pressure = (1 / rho) * sp.Derivative(self.p, axis_coord)
        # Stress divergence: -(1/ρ) Σ_j ∂_j tau_axis,j.
        stress = -(1 / rho) * self._stress_divergence(axis)
        # Total.
        expr = temporal + convection + pressure + stress + gravity
        return Expression(expr, f"{axis}_momentum")

    def _stress_divergence(self, axis):
        s = self.state
        labels = ["x", "y", "z"] if s.has_y else ["x", "z"]
        coord_map = {"x": s.x, "y": s.y, "z": s.z}
        expr = sp.S.Zero
        for j in labels:
            expr += sp.Derivative(self.tau[axis + j], coord_map[j])
        return expr

    def __repr__(self):
        return (f"MassMomentum(dim={self.state.dim}, "
                f"fields=[u,{'v,' if self.state.has_y else ''}w,p], "
                f"params=[g,rho])")
