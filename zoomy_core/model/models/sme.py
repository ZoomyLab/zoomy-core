"""Shallow Moment Equations (Kowalski & Torrilhon 2019) — the single
canonical model, derived with the declarative
:mod:`zoomy_core.model.derivation` framework (moment-projected vertical
velocity route).

``SME`` is the ONLY model class in :mod:`zoomy_core.model.models`; everything
else is archived under ``legacy/``.  It extends the empty base
:class:`zoomy_core.model.basemodel.Model` for the parameter / identity surface,
but its derivation is the declarative pipeline (the `sme_wmoments` notebook):
full 3-D system → hydrostatic → σ-map → moment-project the mass balance (h-eq +
the ŵ closure) → project & close the x-momentum → insert the shifted-Legendre
basis → conservative CoV ``û_i → q_i/h``.  The vertical reconstruction is
registered into the ``interpolate`` function group with the ŵ closure inlined
(``w(ζ) = Σ_j ŵ_j(q, ∂_x q, ∂_x h, ∂_x b) φ_j(ζ)``), so ``interpolate_to_3d`` is
self-contained.

``SME(level=2).derive_model()`` builds the declarative model; ``.system_model``
returns the runtime :class:`~zoomy_core.systemmodel.SystemModel`.
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.basemodel import Model as BaseModel
from zoomy_core.model.derivation import (
    Model as DModel, PDETransformation, Simplify, ResolveIntegral, Basis,
    Consolidate, ExpandSums, EvaluateSums, PullConstants, ExtractBrackets,
    ResolveModes, ResolveBasis, GaussQuadrature, InvertMassMatrix, SolveLinearSystem, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound,
)
from zoomy_core.model.derivation.projection import Integrate          # abstract ζ-integral
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ       # vertical (pressure) integral
from zoomy_core.systemmodel import SystemModel

t, x, z = C.t, C.x, C.z
zeta = sp.Symbol("zeta", real=True)


class SME(BaseModel):
    """Shallow Moment Equations, modal truncation order ``level`` (``N_u``).

    ``level=2`` matches K&T (4.17): 4 dynamic equations (h, q_0, q_1, q_2)
    plus the bed ``b``.
    """

    _finalize_lazy = True               # declarative path — skip the production tag pipeline
    level = param.Integer(default=2, bounds=(0, None))
    quadrature_order = param.Integer(default=0, bounds=(0, None), doc=(
        "Gauss-Legendre order for NUMERICAL integration of Galerkin "
        "integrals that survive the analytic bracket machinery "
        "(non-polynomial material closures, e.g. bingham_navier_slip). "
        "0 (default) = off: an unresolvable integral then raises at "
        "extraction."))
    closures = param.List(default=[], doc=(
        "List of composable stress Closure pieces (closures.py), e.g. "
        "closures=[Newtonian(), NavierSlip(), StressFree()] or "
        "closures=[KEpsilonViscosity(), RoughWall()].  Each closes one stress "
        "component (bulk / bottom / surface).  Takes precedence over `material`; "
        "an empty list with material=None leaves tau_xz UNCLOSED."))
    material = param.Parameter(default=None, doc=(
        "DEPRECATED — use `closures=[...]`.  Legacy MaterialModel stress "
        "closure; None (default) leaves tau_xz UNCLOSED (modal moments stay "
        "free functions)."))

    def derive_model(self):
        """Build the declarative SME model (stored as ``self.derivation``) and
        register the vertical reconstruction.  Called by the base ``__init__``."""
        Nu = int(self.level)
        # nu (kinematic viscosity) and lambda_s (Navier slip) are MODEL
        # PARAMETERS — default 0 (inviscid / free-slip); override values via
        # ``SME(level, parameters={"lambda_s": 0.5, ...})``.
        # e_x: downslope gravity component (K&T eq 4.7 "hg(e_x - ...)") -
        # the INCLINE body force; with e_x = sin(theta) and a FLAT bed the
        # model is an exact infinite incline (periodic-domain friendly)
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0,
                  "e_x": 0.0}
        # the base __init__ has already split the user's parameters= dict
        # into the Zstruct ``self.parameter_values`` — merge those numeric
        # overrides over the defaults.
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})
        m = DModel(coords=(t, x, z), parameters=values)
        g, rho = m.parameters.g, m.parameters.rho
        nu, lam = m.parameters.nu, m.parameters.lambda_s
        u = sp.Function("u", real=True)(t, x, z)
        w = sp.Function("w", real=True)(t, x, z)
        p = sp.Function("p", real=True)(t, x, z)
        h = sp.Function("h", positive=True)(t, x)
        b = sp.Function("b", real=True)(t, x)
        txz = sp.Function("tau_xz", real=True)(t, x, z)

        # 1 — full system in (t, x, z), assembled from balance blueprints
        # (equations.py).  Mass registers u, w (state); Momentum registers p
        # (state) and tau_xz (closure) and builds the (x, z) momentum with the
        # incline body force −g·e_x; h is the geometry state, b the bed.
        from zoomy_core.model.models.equations import Mass, Momentum
        m.declare_state(h)
        m.add_equation("bottom", d.t(b))
        m.add_equation(Mass(m))
        m.add_equation(Momentum(m))
        m.add_equation("kbc_top", KinematicBC(w=w, u=u, interface=b + h))
        m.add_equation("kbc_bot", KinematicBC(w=w, u=u, interface=b))

        # 2 — hydrostatic: z-momentum → eliminate p
        m.momentum.z.apply({d.t(w): 0, d.x(u * w): 0, d.z(w * w): 0})
        m.momentum.z.apply(IntegrateZ(z, z, b + h, method="analytical"))
        m.momentum.z.apply({p.subs(z, b + h): 0})
        m.momentum.x.apply(m.momentum.z.solve_for(p)); m.momentum.z.remove()
        m.momentum.x.apply(Simplify())

        # 3 — σ-map the whole model: z = b + h·ζ
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))

        basis = Basis(symbol="phi", weight="c"); c = basis.weight
        k = sp.Symbol("k", integer=True, nonnegative=True); phi_k = basis.phi(k, zeta)
        legendre = Legendre_shifted(level=Nu + 2)        # need φ_{N_u+2} for the top w-mode

        # 4 — moment-project the MASS balance + kinematic BCs (pre-SoV)
        m.mass.apply(Multiply(h))
        m.mass.apply(Multiply(c(zeta) * phi_k))
        m.mass.apply(ProductRule(variables=[zeta]))
        m.mass.apply(Integrate(zeta, bounds=(0, 1)))
        m.mass.apply(ResolveIntegral())
        m.mass.apply(m.kbc_bot); m.mass.apply(m.kbc_top)
        m.mass.apply({sp.Derivative(b, t): 0})
        m.mass.apply(Simplify())

        # 5 — project the X-MOMENTUM + close the stress (before SoV)
        mx = m.momentum.x
        mx.apply(Multiply(h))
        mx.apply(Multiply(c(zeta) * phi_k))
        mx.apply(ProductRule(variables=[zeta]))
        mx.apply(Integrate(zeta, bounds=(0, 1)))
        mx.apply(ResolveIntegral())
        mx.apply(m.kbc_bot); mx.apply(m.kbc_top); mx.apply({sp.Derivative(b, t): 0})
        tau, uu = m.functions.tau_xz, m.functions.u
        # stress BCs: free surface τ(1)=0; Navier slip τ(0)=+λ·u_b — stress
        # and slip velocity carry the SAME sign (τ = ρν/h·∂_ζu > 0 for u_b>0),
        # so the projected source DAMPS: s[q_0] = −λ·u_b/ρ.  (A minus here
        # flips the friction into anti-damping — coupling caught it as
        # exponential growth of uniform flow.)
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_stress_closures
        # full-access state handed to a closure: fields (s.u, …), σ-aware
        # derivatives (s.dz/s.dx) and parameters (s.par).
        def _state(at):
            return ClosureState(m.functions, params=m.parameters,
                                h=h, x=x, zeta=zeta, at=at)
        has_bulk = apply_stress_closures(self.closures, self.material, m, mx, tau, _state)
        mx.apply(Simplify())

        # 6 — separation of variables: u → û_i (N_u), w → ŵ_j (N_u + 1)
        uh = sp.Function(r"\hat{u}", real=True); wh = sp.Function(r"\hat{w}", real=True)
        reset_modal_indices(m)
        N_u = modal_bound("N_u")
        m.apply(separation_of_variables(u, uh(t, x), basis, N_u))
        m.apply(separation_of_variables(w, wh(t, x), basis, N_u + 1))
        if not has_bulk:
            # UNCLOSED bulk stress: expand tau in the modal basis too — its
            # moments remain free functions (K&T pre-closure form).  This also
            # covers a BOUNDARY-ONLY closure (e.g. RoughWall: bottom set, bulk
            # left free) — the dynamic BC traces are already baked in §5, the
            # bulk stays a free modal field.
            sh_ = sp.Function(r"\hat{\sigma}", real=True)
            m.apply(separation_of_variables(txz, sh_(t, x), basis, N_u + 1))

        # 7 — basis → h-equation (k=0) and the ŵ closure (k=1…N_u+2)
        m.mass.apply(ExpandSums())
        m.mass.apply(PullConstants())
        m.mass.apply(ExtractBrackets(basis, var=zeta))
        m.mass.apply({N_u: Nu})
        m.mass.apply(EvaluateSums())
        m.mass.apply(ResolveModes(index=k, modes=range(Nu + 3)))
        m.mass.apply(ResolveBasis(legendre, var=zeta))
        h_eq = m.mass[0].solve_for(d.t(h))                # ∂_t h = −∂_x(h û_0)
        for row in m.mass[1:Nu + 3]:
            row.apply(h_eq)
        w_closure = SolveLinearSystem(
            m.mass[1:Nu + 3], [wh(j, t, x) for j in range(Nu + 2)]).solve()
        for row in m.mass[1:Nu + 3]:                      # collapse the spent moment rows
            row.apply(w_closure)

        # 8 — substitute the ŵ closure into the x-momentum, resolve
        mx.apply(ExpandSums())
        mx.apply(PullConstants())
        mx.apply(ExtractBrackets(basis, var=zeta))
        mx.apply({N_u: Nu})
        mx.apply(EvaluateSums())
        mx.apply(w_closure)
        mx.apply(ResolveModes(index=k, modes=range(Nu + 1)))
        m.momentum.x.apply(ResolveBasis(legendre, var=zeta))
        if int(self.quadrature_order) > 0:
            # numerical integration of the analytically unintegrable
            # closure terms (the user-chosen escape hatch — see the
            # quadrature_order doc)
            m.momentum.x.apply(GaussQuadrature(
                var=zeta, order=int(self.quadrature_order)))

        # 9 — kill loose ∂_t h, consolidate the pressure, conservative CoV û→q/h
        for kk in range(Nu + 1):
            m.momentum.x[kk].apply(h_eq)
            m.momentum.x[kk].apply(Consolidate())
        m.apply(ChangeOfVariables(r"\hat{u}", "q", lambda q_i: q_i / h))
        # unit ∂_t coefficients — the runtime integrates ∂_t Q = RHS
        m.apply(InvertMassMatrix())

        # 10 — vertical reconstruction → interpolate (ŵ_j inlined as their closure)
        q = m.functions.q.head
        cov = {uh(i, t, x): q(i, t, x) / h for i in range(Nu + 1)}
        u_interp = sum((q(i, t, x) / h) * sp.legendre(i, 2 * zeta - 1)
                       for i in range(Nu + 1))
        w_interp = sum(sp.expand(w_closure[j].rhs.subs(cov)) * sp.legendre(j, 2 * zeta - 1)
                       for j in range(Nu + 2))
        # canonical operator: built here, returned by interpolate_to_3d()
        # (basemodel), parsed by the extraction — never copied by hand.
        self._interpolate_rows = {0: b, 1: h, 2: u_interp, 4: w_interp,
                                  5: rho * g * h * (1 - zeta)}

        # 11 — model-derived lateral wall BC: the mirror state u(ζ) → −u(ζ)
        # flips EVERY moment (odd reflection, ⟨−u φ_i⟩ = −q_i/h); h and b
        # extrapolate.  Keyed by the FIELD (no hard-coded state slots) —
        # runtime access: FromModel(tag=…, definition="wall").
        for i in range(Nu + 1):
            m.register_group("boundary:wall", q(i, t, x), -q(i, t, x))

        # 12 — WB reconstruction: the model OWNS its primitive map.  Limit the
        # free surface η = b + h and the modal velocities u_i = q_i/h instead
        # of the conservative state (bounds limited values by physical scales;
        # removes momentum overshoot at wet/dry fronts).  b reconstructs as
        # itself (identity default).
        self._reconstruction_rows = {h: b + h}
        self._reconstruction_rows.update(
            {q(i, t, x): q(i, t, x) / h for i in range(Nu + 1)})

        # 13 — project (inverse of interpolate): the EXACT Galerkin reduction
        # over the ζ-resolved column contract (z[] = water-relative ζ∈[0,1]
        # in both directions; adapters own the absolute→ζ map):
        #   q_i = (2i+1) · h · ∫₀¹ u(ζ) φ_i(ζ) dζ ,
        # registered as a sympy Integral over the sampled-profile head
        # ``P3_u(ζ)`` — the printer lowers ∫₀¹ to the normalized-trapezoid
        # column sum.  Scalar profile slots (b, h) reduce as depth averages.
        # A constant u (level-0 / flat profile) recovers q_0 = h·⟨u⟩ and
        # zero higher moments — backward compatible with the averaged
        # contract.
        P3 = {f: sp.Symbol(f"P3_{f}", real=True) for f in ("b", "h")}
        P3u = sp.Function("P3_u", real=True)(zeta)
        self._project_rows = {b: P3["b"], h: P3["h"]}
        self._project_rows.update({
            q(i, t, x): (2 * i + 1) * P3["h"]
            * sp.Integral(P3u * sp.legendre(i, 2 * zeta - 1), (zeta, 0, 1))
            for i in range(Nu + 1)})

        self.derivation = m
        self._bed = b
        return None

    @property
    def system_model(self) -> SystemModel:
        """The runtime operator-form system (conservative q-state, `b` prepended).

        Boundary conditions passed to the constructor
        (``SME(level, boundary_conditions=BoundaryConditions(...))``) are
        forwarded — the normal interface, exactly as the production models.
        ``SystemModel.attach_boundary_conditions`` remains available as the
        hook for attaching/replacing BCs on an existing SystemModel."""
        m = self.derivation
        qs = list(m.explicit_state())
        # b evolves via the (trivial) bottom equation ∂_t b = 0, so it is
        # already an explicit unknown; prepend only if absent.
        if self._bed not in qs:
            qs = [self._bed, *qs]
        sm = SystemModel.from_model(m, Q=qs, canonical_source=self)
        self._register_hswme_spectrum(sm)
        bcs = self.boundary_conditions
        if isinstance(bcs, list):
            # NEW flat per-field interface: boundary_conditions=[Wall("left",
            # on="momentum"), Extrapolation("left", on="h"), …].  Resolve `on`
            # against the model's own state; "momentum" aliases the q-family.
            from zoomy_core.model.boundary_conditions import resolve_per_field
            bcs = resolve_per_field(bcs, [str(s) for s in sm.state],
                                    aliases={"momentum": "q"})
        if bcs is not None:
            sm.attach_boundary_conditions(
                bcs, aux_bcs=self.aux_boundary_conditions)
        return sm

    def _register_hswme_spectrum(self, sm):
        """Register the closed-form β-HSWME spectrum (Koellermeier &
        Rominger 2020, Thm 3.5) as the SystemModel's symbolic eigenvalues:

            λ_{1,2} = n·(u_m ∓ √(g h + α₁²)),
            λ_{i+2} = n·(u_m + c_{i,N}·α₁),   c_{i,N} = roots of P_N,

        with u_m = q_0/h and α₁ = −q_1/h in our shifted-Legendre basis
        (their φ₁ = 1−2ζ = −ours; the spectrum is invariant under
        α₁ → −α₁ since the Legendre roots come in ± pairs).  All
        eigenvalues lie inside [u_m−√(gh+α₁²), u_m+√(gh+α₁²)], so this
        gives a SHARP Rusanov wavespeed / CFL bound for the full SME
        without per-face numerical eigensolves (the JAX/CUDA blocker and
        ~90% of the numpy step cost).  The bed row carries λ = 0."""
        import numpy as _np
        Nu = int(self.level)
        by = {str(s_): s_ for s_ in sm.state}
        h_s = by["h"]
        u_m = by["q_0"] / h_s
        g_s = sm.parameters.g
        n_x = sm.normal[0]
        a1 = (-by["q_1"] / h_s) if Nu >= 1 else sp.S.Zero
        c_wave = sp.sqrt(g_s * h_s + a1 ** 2)
        lams = [sp.S.Zero,                       # inert bed row
                n_x * (u_m - c_wave), n_x * (u_m + c_wave)]
        if Nu >= 1:
            roots = _np.polynomial.legendre.leggauss(Nu)[0]   # roots of P_N
            lams += [n_x * (u_m + sp.Float(float(c_)) * a1) for c_ in roots]
        assert len(lams) == sm.n_equations
        sm.eigenvalues = sp.Matrix(sm.n_equations, 1, lams)
