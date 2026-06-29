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
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.projection import Integrate          # abstract ζ-integral
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ       # vertical (pressure) integral
from zoomy_core.systemmodel import SystemModel

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class SME(BaseModel):
    """Shallow Moment Equations, modal truncation order ``level`` (``N_u``).

    ``level=2`` matches K&T (4.17): 4 dynamic equations (h, q_0, q_1, q_2)
    plus the bed ``b``.
    """

    _finalize_lazy = True               # declarative path — skip the production tag pipeline
    _cacheable_derivation = True        # derive_model returns m; byproducts ride on m
    level = param.Integer(default=2, bounds=(0, None))
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension INCLUDING the vertical: 2 → coords (t, x, z), "
        "one horizontal direction (the canonical 1-horizontal SME, q_0,q_1,…); "
        "3 → coords (t, x, y, z), two horizontal directions (q_x_i, q_y_i).  "
        "Closures and boundary conditions are dimension-agnostic; only the state "
        "setup and the per-direction projection loops switch on this."))
    small_slope = param.Boolean(default=True, doc=(
        "Apply small_slope_scaling — resolve the opaque boundary frame to its "
        "n→ẑ limit (drop the O(slope) traction corrections), recovering the "
        "K&T shallow-moment form.  True (default) = the standard shallow model "
        "(byte-identical to the hand-derived SME).  False = keep the "
        "geometrically-exact, slope-aware boundary tractions (the higher-order "
        "model); the projected-traction closures then carry the bed/surface "
        "slope explicitly."))
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
        "component (bulk / bottom / surface).  An empty list leaves tau_xz "
        "UNCLOSED (modal moments stay free)."))
    project_nz = param.Integer(default=33, bounds=(2, None), doc=(
        "FIXED number of vertical samples ``N_z`` for the Integral-FREE "
        "``project_from_3d`` Galerkin reduction: the resolved (e.g. VOF) "
        "column is sampled at ``N_z`` uniform nodes on the unit interval and "
        "reduced to the moments by trapezoid quadrature.  Set this to the VOF "
        "column height the coupling supplies.  Trapezoid is O(N_z^-2); raise "
        "``project_nz`` for a tighter inverse."))

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
        from zoomy_core.model.models.equations import (
            Mass, Momentum, moment_scaling, small_slope_scaling)
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_stress_closures

        # dimension setup: dim=2 → (t,x,z), ONE horizontal direction (the
        # canonical K&T SME, q_0,q_1,…); dim=3 → (t,x,y,z), TWO horizontals
        # (q_x_i, q_y_i).  Every step below loops over the horizontal directions
        # / velocity fields, so dim=2 reduces to exactly the 1-horizontal form.
        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HAXES = ("x",) if dim == 2 else ("x", "y")
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}
        # conserved-moment family per direction: 1 horizontal keeps the bare
        # `q` (byte-identical to K&T); 2 horizontals use `q_x` / `q_y`.
        QNAME = ["q"] if dim == 2 else ["q_x", "q_y"]
        SHAT = [r"\hat{u}"] if dim == 2 else [r"\hat{u}", r"\hat{v}"]

        m = DModel(coords=coords, parameters=values)
        g, rho = m.parameters.g, m.parameters.rho
        h = sp.Function("h", positive=True)(t, *horiz)
        b = sp.Function("b", real=True)(t, *horiz)
        # Free-surface hydrostatic pressure flux (g·h²/2); recomputed in
        # ``system_model`` from ``m`` and tagged there so the extractor routes
        # it to hydrostatic_pressure (no longer stashed on ``self``).

        # 1 — full system, assembled from balance blueprints (equations.py).
        # Mass registers the horizontal velocity(ies) + w; Momentum (full stress
        # tensor) registers p (state) + the shear stresses (closure) with the
        # incline body force −g·e_x; moment_scaling drops the in-plane stresses.
        m.declare_state(h)
        m.add_equation("bottom", d.t(b))
        m.add_equation(Mass(m))
        mom = Momentum(m); m.add_equation(mom)
        moment_scaling(m, mom)
        uvel, w, p = mom.uvel, mom.w, mom.p
        # hook: a turbulence subclass (KESME) declares its extra depth-averaged
        # state (k, ε) HERE so the bulk stress closure can read them in §5.
        self._declare_turbulence_fields(m, t, horiz)

        def _kbc(interface):
            kw = dict(w=w, u=uvel[0], interface=interface)
            if dim == 3:
                kw["v"] = uvel[1]
            return KinematicBC(**kw)
        m.add_equation("kbc_top", _kbc(b + h))
        m.add_equation("kbc_bot", _kbc(b))

        # 2 — hydrostatic: drop the inertial vertical-momentum terms, integrate
        # for p, broadcast the solved pressure into EVERY horizontal component,
        # then drop the vertical row.
        mz = m.momentum.z
        zdrop = {d.t(w): 0, d.z(w * w): 0}
        for i, xd in enumerate(horiz):
            zdrop[DERIV[xd](uvel[i] * w)] = 0
        mz.apply(zdrop)
        mz.apply(IntegrateZ(z, z, b + h, method="analytical"))
        mz.apply({p.subs(z, b + h): 0})
        psol = mz.solve_for(p)
        for ax in HAXES:
            getattr(m.momentum, ax).apply(psol)
        mz.remove()
        for ax in HAXES:
            getattr(m.momentum, ax).apply(Simplify())

        # 3 — σ-map the whole model: z = b + h·ζ
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))

        basis = Basis(symbol="phi", weight="c"); c = basis.weight
        # the Galerkin TEST index — "l", which _INDEX_NAMES reserves and never
        # auto-mints, so it can't collide with the trial indices of the SoV
        # families (û, v̂, ŵ); a raw Symbol("k") collided with the 3rd minted
        # trial index in 2-D and silently diagonalised the ŵ bracket.
        k = test_index(); phi_k = basis.phi(k, zeta)
        legendre = Legendre_shifted(level=Nu + 2)        # need φ_{N_u+2} for the top w-mode

        # 4 — moment-project the MASS balance + kinematic BCs (pre-SoV).  The
        # ∂_y v term rides along automatically in 2-D.
        m.mass.apply(Multiply(h)); m.mass.apply(Multiply(c(zeta) * phi_k))
        m.mass.apply(ProductRule(variables=[zeta]))
        m.mass.apply(Integrate(zeta, bounds=(0, 1))); m.mass.apply(ResolveIntegral())
        m.mass.apply(m.kbc_bot); m.mass.apply(m.kbc_top)
        m.mass.apply({sp.Derivative(b, t): 0}); m.mass.apply(Simplify())

        # 5 — project EACH horizontal momentum component, then close the stress.
        def _state(at, *, alias=None, btag=None):
            return ClosureState(m.functions, params=m.parameters, h=h, x=x,
                                zeta=zeta, at=at, alias=alias,
                                boundary_tag=btag, horiz=list(horiz))
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            mxi.apply(Multiply(h)); mxi.apply(Multiply(c(zeta) * phi_k))
            mxi.apply(ProductRule(variables=[zeta]))
            mxi.apply(Integrate(zeta, bounds=(0, 1))); mxi.apply(ResolveIntegral())
            mxi.apply(m.kbc_bot); mxi.apply(m.kbc_top); mxi.apply({sp.Derivative(b, t): 0})
        # boundary closures prescribe the traction in {n,t_α} (the frame solve
        # couples the directions for vector tractions like rough-wall friction);
        # bulk closures stay diagonal (s.u aliased to each axis velocity).
        tau_h = {"x": m.functions.tau_xz}
        if dim == 3:
            tau_h["y"] = m.functions.tau_yz
        axes = [{"mx": getattr(m.momentum, ax), "tau": tau_h[ax],
                 "velname": HNAME[xd]} for ax, xd in zip(HAXES, horiz)]
        has_bulk = apply_stress_closures(self.closures, m, axes, _state, list(horiz))
        for ax in HAXES:
            getattr(m.momentum, ax).apply(Simplify())
        # small-slope frame resolution (tracked, removable) — n→ẑ recovers the
        # K&T shallow traces; skip it (small_slope=False) to keep slope-aware.
        if bool(self.small_slope):
            small_slope_scaling(m)

        # 6 — separation of variables: each u_i → û/v̂ (N_u), w → ŵ (N_u + 1)
        coeff_heads = [sp.Function(nm, real=True) for nm in SHAT]
        wh = sp.Function(r"\hat{w}", real=True)
        reset_modal_indices(m)
        N_u = modal_bound("N_u")
        for i in range(len(horiz)):
            m.apply(separation_of_variables(uvel[i], coeff_heads[i](t, *horiz), basis, N_u))
        m.apply(separation_of_variables(w, wh(t, *horiz), basis, N_u + 1))
        if not has_bulk:
            # UNCLOSED bulk stress: expand each shear stress modally — its
            # moments stay free (K&T pre-closure / boundary-only-closure form).
            for ax, xd in zip(HAXES, horiz):
                txz_i = sp.Function(f"tau_{ax}z", real=True)(*coords)
                signame = r"\hat{\sigma}" if dim == 2 else rf"\hat{{\sigma}}_{HNAME[xd]}"
                m.apply(separation_of_variables(txz_i, sp.Function(signame, real=True)(t, *horiz),
                                                basis, N_u + 1))

        # 7 — basis → h-equation (k=0) and the ŵ closure (k=1…N_u+2).  In 2-D
        # the ŵ closure couples û and v̂; the h-equation is ∂_t h = −Σ_d ∂_d(h û_d^0).
        m.mass.apply(ExpandSums()); m.mass.apply(PullConstants())
        m.mass.apply(ExtractBrackets(basis, var=zeta)); m.mass.apply({N_u: Nu})
        m.mass.apply(EvaluateSums())
        m.mass.apply(ResolveModes(index=k, modes=range(Nu + 3)))
        m.mass.apply(ResolveBasis(legendre, var=zeta))
        h_eq = m.mass[0].solve_for(d.t(h))
        for row in m.mass[1:Nu + 3]:
            row.apply(h_eq)
        w_closure = SolveLinearSystem(
            m.mass[1:Nu + 3], [wh(j, t, *horiz) for j in range(Nu + 2)]).solve()
        for row in m.mass[1:Nu + 3]:
            row.apply(w_closure)

        # 8 — substitute the ŵ closure into each horizontal momentum, resolve.
        # (Re-fetch the component AFTER ResolveModes' in-place promotion.)
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            mxi.apply(ExpandSums()); mxi.apply(PullConstants())
            mxi.apply(ExtractBrackets(basis, var=zeta)); mxi.apply({N_u: Nu})
            mxi.apply(EvaluateSums()); mxi.apply(w_closure)
            mxi.apply(ResolveModes(index=k, modes=range(Nu + 1)))
        for ax in HAXES:
            getattr(m.momentum, ax).apply(ResolveBasis(legendre, var=zeta))
            if int(self.quadrature_order) > 0:
                getattr(m.momentum, ax).apply(
                    GaussQuadrature(var=zeta, order=int(self.quadrature_order)))

        # 9 — kill loose ∂_t h, consolidate, conservative CoV û_d → q_d/h
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            for kk in range(Nu + 1):
                mxi[kk].apply(h_eq); mxi[kk].apply(Consolidate())
        for nm, qn in zip(SHAT, QNAME):
            m.apply(ChangeOfVariables(nm, qn, lambda q_i: q_i / h))
        m.apply(InvertMassMatrix())

        # 10 — vertical reconstruction → interpolate (field order [b,h,u,v,w,p];
        # v at index 3 only in 2-D, ŵ_j inlined as their closure)
        q_heads = [getattr(m.functions, qn).head for qn in QNAME]
        cov = {}
        for i, qh in enumerate(q_heads):
            cov.update({coeff_heads[i](j, t, *horiz): qh(j, t, *horiz) / h
                        for j in range(Nu + 1)})
        interp = {0: b, 1: h}
        for vi, qh in enumerate(q_heads):
            interp[2 + vi] = sum((qh(i, t, *horiz) / h) * sp.legendre(i, 2 * zeta - 1)
                                 for i in range(Nu + 1))
        interp[4] = sum(sp.expand(w_closure[j].rhs.subs(cov)) * sp.legendre(j, 2 * zeta - 1)
                        for j in range(Nu + 2))
        interp[5] = rho * g * h * (1 - zeta)
        m.interpolate_rows = interp

        # 11 — model-derived lateral wall BC: mirror u(ζ) → −u(ζ) flips EVERY
        # moment of EVERY direction; h, b extrapolate.
        for qh in q_heads:
            for i in range(Nu + 1):
                m.register_group("boundary:wall", qh(i, t, *horiz), -qh(i, t, *horiz))

        # 12 — WB reconstruction: limit η = b+h and the modal velocities q_d/h.
        m.reconstruction_rows = {h: b + h}
        for qh in q_heads:
            m.reconstruction_rows.update(
                {qh(i, t, *horiz): qh(i, t, *horiz) / h for i in range(Nu + 1)})

        # 13 — project (inverse of interpolate): the Integral-FREE, fixed-node
        # Galerkin reduction q_{d,i} = h·α_i of the sampled column.  N_z uniform
        # nodes on [0,1] with trapezoid weights; the per-node samples are the
        # SAME P3_<vel> heads the printer maps for the column, evaluated at the
        # fixed node.  ``Basisfunction.projection_rows`` builds the plain
        # arithmetic rows — the 1/∫φ_i² normalisation is in its denominator, so
        # we supply only the physical h factor (q_i = h·α_i, matching the old
        # (2i+1)·h·∫₀¹ u_d φ_i dζ).  Integral-free ⇒ every printer lowers it.
        N_z = int(self.project_nz)
        nodes = [float(j) / (N_z - 1) for j in range(N_z)]
        weights = [1.0 / (N_z - 1)] * N_z
        weights[0] *= 0.5; weights[-1] *= 0.5
        P3 = {f: sp.Symbol(f"P3_{f}", real=True) for f in ("b", "h")}
        m.project_rows = {b: P3["b"], h: P3["h"]}
        for xd, qh in zip(horiz, q_heads):
            P3vel = sp.Function(f"P3_{HNAME[xd]}", real=True)
            samples = [P3vel(nd) for nd in nodes]
            rows = legendre.projection_rows(nodes, weights, samples,
                                            norm=lambda _k: P3["h"])
            m.project_rows.update({qh(i, t, *horiz): rows[i]
                                   for i in range(Nu + 1)})

        # hook: a turbulence subclass (KESME) adds its transported-scalar
        # balances (k, ε) now, using the conserved moments q_d (= h û_d).
        self._add_turbulence_transport(m, t, horiz, h, q_heads)

        return m

    # ── turbulence hooks (no-op on plain SME; KESME overrides) ──────────────
    def _declare_turbulence_fields(self, m, t, horiz):
        """Declare extra depth-averaged transported state (e.g. k, ε) BEFORE
        the bulk-stress closure, so the closure can read them.  No-op on SME."""

    def _add_turbulence_transport(self, m, t, horiz, h, q_heads):
        """Add the transported-scalar balances (e.g. the k–ε equations) AFTER
        the moment system is assembled.  No-op on SME."""

    @property
    def system_model(self) -> SystemModel:
        """The runtime operator-form system (conservative q-state, `b` prepended).

        Boundary conditions passed to the constructor
        (``SME(level, boundary_conditions=BoundaryConditions(...))``) are
        forwarded — the normal interface, exactly as the production models.
        ``SystemModel.attach_boundary_conditions`` remains available as the
        hook for attaching/replacing BCs on an existing SystemModel."""
        m = self.derivation
        dim = int(self.dimension)
        horiz = (x,) if dim == 2 else (x, y)
        qs = list(m.explicit_state())
        # b evolves via the (trivial) bottom equation ∂_t b = 0, so it is
        # already an explicit unknown; prepend only if absent.
        bed = sp.Function("b", real=True)(t, *horiz)
        if bed not in qs:
            qs = [bed, *qs]
        # Manual hydrostatic-pressure tag (one-liner): mark g·h²/2 so the
        # structural extractor routes it to hydrostatic_pressure (well-balanced
        # reconstruction) instead of the conservative flux.  Recomputed from m.
        from zoomy_core.model.derivation.system_extract import HydrostaticPressure
        h = sp.Function("h", positive=True)(t, *horiz)
        pf = m.parameters.g * h ** 2 / 2
        m.apply({pf: HydrostaticPressure(pf)})
        sm = SystemModel.from_model(m, Q=qs, canonical_source=self)
        m.apply({HydrostaticPressure(pf): pf})   # un-tag: leave derivation clean
        self._register_hswme_spectrum(sm)
        # legacy BoundaryConditions container OR the new flat per-field list
        from zoomy_core.model.boundary_conditions import resolve_and_attach
        resolve_and_attach(sm, self.boundary_conditions,
                           aux_bcs=self.aux_boundary_conditions)
        return sm

    def _register_hswme_spectrum(self, sm):
        """Register the β-HSWME spectrum as the SystemModel's symbolic
        eigenvalues — computed DIRECTLY from the truncated quasilinear matrix
        (dimension-agnostic, any 1-D / 2-D normal):

        Take the normal-projected quasilinear matrix ``A·n = Σ_d n_d·
        quasilinear_matrix[:,:,d]``, ZERO the higher-order moment coefficients
        (``q_i, q_{x/y,i}`` for ``i ≥ 2`` → 0 — the β-HSWME hyperbolic
        truncation that drops the α_{≥2} coupling), and take its symbolic
        eigenvalues.  This reproduces the Koellermeier-Rominger (2020, Thm 3.5)
        spectrum — gravity waves ``n·(u_m ± √(g h + α₁²))`` plus the moment
        waves ``n·(u_m + c_{i}·α₁)`` — but with the CORRECT characteristic
        roots ``c_i`` (the truncated matrix's own; the previous closed form
        wrongly used the Gauss-Legendre quadrature nodes, e.g. ±1/√3 instead of
        ±1/√5 at level 2), and it generalises to a 2-D normal with no extra
        logic (``A·n`` carries ``n_x, n_y``; the transverse moment advection
        adds ``n·u_m`` eigenvalues automatically).

        Gives a sharp Rusanov wavespeed / CFL bound without per-face numerical
        eigensolves (the JAX/CUDA blocker, ~90% of the numpy step cost); the bed
        row carries λ = 0.  ``sympy`` factors the truncated char-poly in
        radicals up to level 5; at level ≥ 6 the moment block is Abel-unsolvable
        — we then splice in the spectrum of the highest radical-solvable twin
        (:meth:`_truncated_spectrum`, level 5) and pad the remaining interior
        slots with the central advection ``n·u_m`` (see that method)."""
        import re
        n_eq = sm.n_equations
        normal = list(sm.normal.values())
        A = sp.zeros(n_eq, n_eq)
        for k in range(sm.n_dim):
            for i in range(n_eq):
                for j in range(n_eq):
                    A[i, j] += normal[k] * sm.quasilinear_matrix[i, j, k]
        # β-HSWME truncation: drop the higher-order moment coefficients
        # (moment index ≥ 2) — the regularisation that makes the spectrum
        # computable.  Matches q_2…, q_x_2…, q_y_2… (1-D and 2-D names alike).
        drop = {}
        for s_ in sm.state:
            mtc = re.match(r"q(?:_[xy])?_(\d+)$", str(s_))
            if mtc and int(mtc.group(1)) >= 2:
                drop[s_] = sp.S.Zero
        A = sp.Matrix(A).subs(drop)
        try:
            ev = A.eigenvals()
            lams = []
            for root, mult in ev.items():
                lams += [root] * int(mult)
            if len(lams) != n_eq:
                raise ValueError(f"got {len(lams)} eigenvalues, need {n_eq}")
            sm.eigenvalues = sp.Matrix(n_eq, 1, lams)
        except Exception as exc:                 # Abel wall (level ≥ 6) etc.
            self._truncated_spectrum(sm, exc)

    def _truncated_spectrum(self, sm, exc):
        """Spectrum for level ≥ 6, where the truncated moment block is
        Abel-unsolvable in radicals.

        The β-HSWME gravity waves ``n·(u_m ± √(g h + α₁²))`` are
        LEVEL-INDEPENDENT (byte-identical at every level — verified) and bound
        the entire spectrum, so the spectral radius — all that Rusanov / HLL /
        CFL need — is captured by the highest level we CAN solve in radicals.
        Build that twin (``level=5``, same dimension), take its exact spectrum,
        and pad the extra interior slots with the central advection ``n·u_m``
        (provably inside the gravity cone).  Sharp wavespeed at any level, no
        per-face eigensolve.  If even the twin has no symbolic spectrum, leave
        ``eigenvalues = None`` → the runtime's opaque numeric eigensystem."""
        n_eq = sm.n_equations
        twin = SME(level=5, dimension=int(self.dimension),
                   parameters=dict(self.parameter_values.items())).system_model
        if twin.eigenvalues is None:
            sm.eigenvalues = None
            import warnings
            warnings.warn(
                f"SME(level={self.level}): no radical-solvable HSWME spectrum "
                f"even at the level-5 twin ({type(exc).__name__}); falling back "
                f"to the opaque numeric eigensystem.")
            return
        # the twin's eigenvalues are expressions in h, q_0, q_1 (+ normal) — all
        # present in this higher-level state, so they drop straight in.
        lams = [twin.eigenvalues[i] for i in range(twin.n_equations)]
        normal = list(sm.normal.values())
        by = {str(s_): s_ for s_ in sm.state}
        h_s = by["h"]
        # central advection n·u_m — normal velocity, dimension-agnostic
        u_m = sum(normal[k] * by[nm] for k, nm in enumerate(
            (["q_0"] if self.dimension == 2 else ["q_x_0", "q_y_0"]))) / h_s
        lams += [u_m] * (n_eq - len(lams))         # pad interior, inside the cone
        sm.eigenvalues = sp.Matrix(n_eq, 1, lams)
