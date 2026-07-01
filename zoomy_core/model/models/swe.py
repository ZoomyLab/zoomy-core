"""SWE — shallow water with bed, dimension-agnostic (1D / 2D).

The level-0 member of the moment hierarchy as an OFFICIAL model class:
state ``[b, h, hu]`` (1D) or ``[b, h, hu, hv]`` (2D), pressure in the
flux, bed slope as a nonconservative product, optional Manning friction.
(The depth-averaged operators are stated directly — the declarative
vertical derivation machinery is (t, x, z) only today; when it grows a
transverse coordinate this class collapses onto ``SME(level=0)``.)

Couples through the same canonical contract as SME: ``interpolate``
lifts to the flat profile ``u(ζ) = hu/h`` (+ hydrostatic p), ``project``
depth-averages the sampled column — a constant profile round-trips
exactly.  Both ride the standard function-group slots
(:func:`zoomy_core.systemmodel.system_model._attach_function_groups`).

Boundary definitions (``FromModel(tag=…, definition=…)``):

- ``wall_x`` / ``wall_y``: mirror of the x- / y-momentum (a wall mirror
  flips the NORMAL momentum; the direction is encoded in the name since
  a state-component spec cannot see the face normal),
- ``wall``: alias of ``wall_x`` (the 1D convention),
- ``inflow``: prescribed depth/discharge via the model parameters
  ``h_in`` / ``q_in`` (transverse momentum zero, bed extrapolates).
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.systemmodel.system_model import SystemModel

__all__ = ["SWE"]


class SWE(StructuredDerivativeModel):
    """Shallow water + bed (+ Manning friction), 1D or 2D."""

    dimension = param.Integer(default=1, bounds=(1, 2))
    variables = ["b", "h", "hu"]          # + "hv" when dimension == 2
    parameters = {
        "g":    (9.81, "positive"),
        "n_m":  (0.0, "nonnegative"),     # Manning; 0 = frictionless
        "h_in": (0.0, "nonnegative"),     # inflow target depth
        "q_in": (0.0, "real"),            # inflow target discharge
    }

    def __init__(self, **kw):
        if int(kw.get("dimension", 1)) == 2:
            self.variables = ["b", "h", "hu", "hv"]
        # constructor parameters override the declared defaults ONE BY ONE
        # (a partial dict must not drop g)
        defaults = dict(type(self).parameters)
        defaults.update(kw.get("parameters") or {})
        kw["parameters"] = defaults
        # FromModel BCs resolve against the SystemModel — hold them aside
        # and attach in ``system_model`` (the SME semantics), instead of
        # letting the production pipeline resolve them at construction.
        self._coupling_bcs = kw.pop("boundary_conditions", None)
        super().__init__(**kw)
        self._function_groups = self._build_function_groups()

    # ── well-balanced reconstruction (model owns it) ───────────────────
    def reconstruction_variables(self):
        """Primitive well-balanced reconstruction map ``state → primitive``.

        Limit the FREE SURFACE ``eta = b + h`` instead of the conservative
        depth ``h`` so a slope/oscillation limiter is inert at lake-at-rest
        and does not corrupt the flat surface (the wet/dry "water creeping
        up the walls" defect).  Every other field reconstructs as identity;
        the inverse ``h = eta - b`` is auto-derived by
        :meth:`state_from_reconstruction`.

        Pure sympy, resolved by FIELD NAME — no index assumptions — so it
        codegens to every backend (UFL / numpy / jax / OpenFOAM) regardless
        of where ``h`` and ``b`` sit in the state vector.  Mirrors
        ``SME``'s ``_reconstruction_rows = {h: b + h}``.
        """
        v = self.variables
        eta = v.b + v.h
        return ZArray([eta if s == v.h else s for s in v.get_list()])

    # ── operators ─────────────────────────────────────────────────────
    def flux(self):
        h, hu = self.Q.h, self.Q.hu
        g = self.parameters.g
        F = sp.Matrix.zeros(self.n_variables, self.dimension)
        F[1, 0] = hu
        F[2, 0] = hu * hu / h + sp.Rational(1, 2) * g * h * h
        if self.dimension == 2:
            hv = self.Q.hv
            F[3, 0] = hu * hv / h
            F[1, 1] = hv
            F[2, 1] = hu * hv / h
            F[3, 1] = hv * hv / h + sp.Rational(1, 2) * g * h * h
        return ZArray(F)

    def nonconservative_matrix(self):
        """Bed slope: the momentum rows pick up ``g·h·∂_d b``."""
        h = self.Q.h
        g = self.parameters.g
        B = [[[0] * self.dimension for _ in range(self.n_variables)]
             for _ in range(self.n_variables)]
        B[2][0][0] = g * h
        if self.dimension == 2:
            B[3][0][1] = g * h
        return ZArray(B)

    def source(self):
        """Manning friction on the momentum rows (off for n_m = 0)."""
        h, hu = self.Q.h, self.Q.hu
        g, n_m = self.parameters.g, self.parameters.n_m
        S = sp.Matrix.zeros(self.n_variables, 1)
        if self.dimension == 1:
            S[2, 0] = -g * n_m**2 * hu * sp.Abs(hu) / h ** sp.Rational(7, 3)
        else:
            hv = self.Q.hv
            mag = sp.sqrt(hu * hu + hv * hv)
            S[2, 0] = -g * n_m**2 * hu * mag / h ** sp.Rational(7, 3)
            S[3, 0] = -g * n_m**2 * hv * mag / h ** sp.Rational(7, 3)
        return ZArray(S)

    # ── function groups: canonical operators + boundary definitions ───
    def _build_function_groups(self):
        b, h, hu = self.Q.b, self.Q.h, self.Q.hu
        g = self.parameters.g
        p = self.parameters
        zeta = sp.Symbol("zeta", real=True)
        P3b = sp.Symbol("P3_b", real=True)
        P3h = sp.Symbol("P3_h", real=True)
        P3u = sp.Function("P3_u", real=True)(zeta)

        # flat-profile lift to canonical [b, h, u, v, w, p](ζ)
        interp = {0: b, 1: h, 2: hu / h, 5: g * h * (1 - zeta)}
        # depth-average of the sampled column (exact for flat profiles)
        project = {b: P3b, h: P3h,
                   hu: P3h * sp.Integral(P3u, (zeta, 0, 1))}
        # WB primitive map: limit η = b+h and the velocities
        recon = {h: b + h, hu: hu / h}
        groups = {
            "interpolate": interp,
            "project": project,
            "reconstruction": recon,
            "boundary:wall":   {hu: -hu},
            "boundary:wall_x": {hu: -hu},
            "boundary:inflow": {h: p.h_in, hu: p.q_in},
        }
        if self.dimension == 2:
            hv = self.Q.hv
            P3v = sp.Function("P3_v", real=True)(zeta)
            interp[3] = hv / h
            project[hv] = P3h * sp.Integral(P3v, (zeta, 0, 1))
            recon[hv] = hv / h
            groups["boundary:wall_y"] = {hv: -hv}
            groups["boundary:inflow"][hv] = sp.S.Zero
        return groups

    # ── runtime form ──────────────────────────────────────────────────
    @property
    def system_model(self) -> SystemModel:
        # ``SystemModel.from_model`` now parses the model's function groups AND
        # attaches its ``_coupling_bcs`` on raw promotion (REQ-87) — the same
        # wiring this property used to do inline — so every backend adapter /
        # FVM solver that calls ``from_model`` directly inherits it.
        return SystemModel.from_model(self)
