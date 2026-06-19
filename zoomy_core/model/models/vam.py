"""VAM — vertically-averaged moments (non-hydrostatic), the declarative
canonical model.

Same Galerkin recipe as the :class:`~zoomy_core.model.models.sme.SME`, but
NON-hydrostatic: the pressure is split hydrostatic + non-hydrostatic BEFORE
the modal ansatz (``p_total = ρ g (η − z) + p``, ``η = b + h``), and only the
non-hydrostatic part ``p`` stays modal.  The predictor therefore keeps the
SWE hyperbolic structure (gravity wave speeds in the flux); the pressure
modes ``P_0 … P_Nu`` are Lagrange multipliers of the divergence constraints
(mass projections k = 1 … Nu+1 — zero mass-matrix rows after extraction).

``VAM(level=Nu).system_model`` returns the square DAE
(state ``[b, h, q_0…q_Nu, r_0…r_Nu, P_0…P_Nu]``);
``VAM(level=Nu).chorin_split(dt)`` returns the three Chorin sub-systems
``(SM_pred, SM_press, SM_corr)`` for
:class:`~zoomy_core.fvm.solver_chorin_vam_numpy.ChorinSplitVAMSolver` via the
structural splitter (row roles read off the operators, no name conventions).
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
    ResolveModes, ResolveBasis, InvertMassMatrix, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.systemmodel import SystemModel

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class VAM(BaseModel):
    """Non-hydrostatic vertically-averaged moment equations, truncation
    ``level`` (= ``N_u``; u, w and the non-hydrostatic p share the basis)."""

    _finalize_lazy = True               # declarative path
    level = param.Integer(default=1, bounds=(0, None))
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension incl. vertical: 2 → (t,x,z), one horizontal "
        "(q_i, r_i, P_i); 3 → (t,x,y,z), two horizontal (q_x_i, q_y_i, r_i, "
        "P_i).  Every projection/closure loops over the horizontal directions."))
    closures = param.List(default=[], doc=(
        "Composable stress Closure pieces (closures.py).  Empty leaves tau_xz "
        "UNCLOSED (modal moments stay free)."))
    small_slope = param.Boolean(default=True, doc=(
        "Resolve the opaque boundary frame to its n→ẑ limit (drop O(slope) "
        "traction corrections) — the shallow form.  False keeps the slope-aware "
        "tractions."))

    def derive_model(self):
        Nu = int(self.level)
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0}
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})
        from zoomy_core.model.models.equations import (
            Mass, MomentumNonHydrostatic, small_slope_scaling)
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_stress_closures

        # dimension setup (see SME): dim=2 → 1 horizontal (q_i,r_i,P_i,
        # byte-identical Escalante); dim=3 → 2 horizontal (q_x_i,q_y_i).  The
        # vertical (r) and pressure (P) moments stay scalar; only the horizontal
        # momentum doubles and the σ-mass-flux ω̃ + flux gain the ∂_y terms.
        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}; CN = {x: "x", y: "y"}
        QNAME = ["q"] if dim == 2 else ["q_x", "q_y"]
        SHAT = [r"\hat{u}"] if dim == 2 else [r"\hat{u}", r"\hat{v}"]
        MOM = [f"momentum_{CN[xd]}" for xd in horiz]    # momentum_x[, momentum_y]

        m = DModel(coords=coords, parameters=values)
        g, rho = m.parameters.g, m.parameters.rho
        h = sp.Function("h", positive=True)(t, *horiz)
        b = sp.Function("b", real=True)(t, *horiz)
        # Free-surface hydrostatic pressure flux; tagged explicitly in
        # ``system_model`` (the extractor no longer auto-detects pressure).
        self._pressure_flux = g * h ** 2 / 2

        # 1 — full system (hydrostatic pressure pre-absorbed) from blueprints
        m.declare_state(h)
        m.add_equation("bottom", d.t(b))
        m.add_equation(Mass(m))
        momblue = MomentumNonHydrostatic(m); m.add_equation(momblue)
        uvel, w, p = momblue.uvel, momblue.w, momblue.p

        def _kbc(interface):
            kw = dict(w=w, u=uvel[0], interface=interface)
            if dim == 3:
                kw["v"] = uvel[1]
            return KinematicBC(**kw)
        m.add_equation("kbc_top", _kbc(b + h))
        m.add_equation("kbc_bot", _kbc(b))

        # 2 — σ-map:  z = b + h ζ
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))
        basis = Basis(symbol="phi", weight="c"); c = basis.weight
        k = test_index(); phi_k = basis.phi(k, zeta)
        legendre = Legendre_shifted(level=Nu + 2)

        def _project(eq):
            eq.apply(Multiply(h)); eq.apply(Multiply(c(zeta) * phi_k))
            eq.apply(ProductRule(variables=[zeta]))
            eq.apply(Integrate(zeta, bounds=(0, 1))); eq.apply(ResolveIntegral())
            eq.apply(m.kbc_bot); eq.apply(m.kbc_top)
            eq.apply({sp.Derivative(b, t): 0})

        # 3 — Galerkin-project mass + each horizontal momentum + vertical
        _project(m.mass); m.mass.apply(Simplify())
        pp = m.functions.p

        def _state(at, *, alias=None, btag=None):
            return ClosureState(m.functions, params=m.parameters, h=h, x=x,
                                zeta=zeta, at=at, alias=alias,
                                boundary_tag=btag, horiz=list(horiz))
        for mn in MOM:
            mxi = getattr(m, mn)
            _project(mxi)
            mxi.apply({pp.at(1): 0})          # dynamic surface BC p(ζ=1)=0
        tau_h = {"x": m.functions.tau_xz}
        if dim == 3:
            tau_h["y"] = m.functions.tau_yz
        axes = [{"mx": getattr(m, mn), "tau": tau_h[ax], "velname": HNAME[xd]}
                for mn, ax, xd in zip(MOM, ("x", "y"), horiz)]
        has_bulk = apply_stress_closures(self.closures, m, axes, _state, list(horiz))
        for mn in MOM:
            getattr(m, mn).apply(Simplify())
        if bool(self.small_slope):
            small_slope_scaling(m)

        mz = m.momentum_z
        _project(mz); mz.apply({pp.at(1): 0}); mz.apply(Simplify())

        # 4 — modal ansatz: each u_i ∈ P_Nu; w, p ∈ P_{Nu+1} (Escalante spaces)
        coeff_heads = [sp.Function(nm, real=True) for nm in SHAT]
        wh = sp.Function(r"\hat{w}", real=True)
        ph = sp.Function(r"\hat{p}", real=True)
        reset_modal_indices(m)
        N_u = modal_bound("N_u")
        for i in range(len(horiz)):
            m.apply(separation_of_variables(uvel[i], coeff_heads[i](t, *horiz), basis, N_u))
        m.apply(separation_of_variables(w, wh(t, *horiz), basis, N_u + 1))
        m.apply(separation_of_variables(p, ph(t, *horiz), basis, N_u + 1))
        if not has_bulk:
            for ax, xd in zip(("x", "y"), horiz):
                txz_i = sp.Function(f"tau_{ax}z", real=True)(*coords)
                signame = r"\hat{\sigma}" if dim == 2 else rf"\hat{{\sigma}}_{HNAME[xd]}"
                m.apply(separation_of_variables(txz_i, sp.Function(signame, real=True)(t, *horiz),
                                                basis, N_u + 1))

        # 5 — resolve: mass k=0…Nu+1; each momentum + vertical k=0…Nu
        resolve = ([("mass", range(Nu + 2))]
                   + [(mn, range(Nu + 1)) for mn in MOM]
                   + [("momentum_z", range(Nu + 1))])
        for nm, modes in resolve:
            getattr(m, nm).apply(ExpandSums())
            getattr(m, nm).apply(PullConstants())
            getattr(m, nm).apply(ExtractBrackets(basis, var=zeta))
            getattr(m, nm).apply({N_u: Nu})
            getattr(m, nm).apply(EvaluateSums())
            getattr(m, nm).apply(ResolveModes(index=k, modes=modes))
            getattr(m, nm).apply(ResolveBasis(legendre, var=zeta))

        # 5b — top w/p mode closures (Escalante eq 6).  2-D: the bottom
        # kinematic is w(0) = Σ_d u_d(0)·∂_d b.
        top = Nu + 1
        u_at0 = [sum((-1) ** j * coeff_heads[i](j, t, *horiz) for j in range(Nu + 1))
                 for i in range(len(horiz))]
        w_top = (-1) ** top * (sum(u_at0[i] * DERIV[xd](b) for i, xd in enumerate(horiz))
                               - sum((-1) ** j * wh(j, t, *horiz) for j in range(top)))
        p_top = -sum(ph(j, t, *horiz) for j in range(top))
        m.apply({wh(top, t, *horiz): w_top})
        m.apply({ph(top, t, *horiz): p_top})

        # 6 — conservative CoV (û_d→q_d/h, ŵ→r/h, p̂→P) + h-eq substitution
        for nm, qn in zip(SHAT, QNAME):
            m.apply(ChangeOfVariables(nm, qn, lambda qi: qi / h))
        m.apply(ChangeOfVariables(r"\hat{w}", "r", lambda ri: ri / h))
        m.apply(ChangeOfVariables(r"\hat{p}", "P", lambda pi: pi))
        h_eq = m.mass[0].solve_for(d.t(h))
        for nm in MOM + ["momentum_z"]:
            for kk in range(Nu + 1):
                getattr(m, nm)[kk].apply(h_eq)
                getattr(m, nm)[kk].apply(Consolidate())
        for kk in range(1, Nu + 2):
            m.mass[kk].apply(h_eq)
            m.mass[kk].apply(Consolidate())
        m.apply(InvertMassMatrix())

        # 6b — σ-mass-flux ω̃ correction (Escalante).  2-D: ω̃ couples ∂_x AND
        # ∂_y (the vertical-coupling operator just gains the second-horizontal
        # divergence + the v-advection term); the correction Δ_k is applied to
        # every horizontal momentum (field = u_d_m) and the vertical (w̃_m).
        phis = [sp.legendre(j, 2 * zeta - 1) for j in range(Nu + 2)]
        mus = [sp.Rational(1, 2 * j + 1) for j in range(Nu + 2)]
        s_ = sp.Symbol("_s_omega")
        qf = [[getattr(m.functions, qn).head(j, t, *horiz) for j in range(Nu + 1)]
              for qn in QNAME]                       # qf[di][j]
        rf = [m.functions.r.head(j, t, *horiz) for j in range(Nu + 1)]
        uvel_m = [sum(qf[di][j] / h * phis[j] for j in range(Nu + 1))
                  for di in range(len(horiz))]
        w_top_cons = ((-1) ** (Nu + 1)) * (
            sum(sum((-1) ** j * qf[di][j] / h for j in range(Nu + 1)) * DERIV[xd](b)
                for di, xd in enumerate(horiz))
            - sum((-1) ** j * rf[j] / h for j in range(Nu + 1)))
        wt_m = (sum(rf[j] / h * phis[j] for j in range(Nu + 1))
                + w_top_cons * phis[Nu + 1])
        # ∂_t h = −Σ_d ∂_d(q_d_0); advection Σ_d u_d_m(ζ ∂_d h + ∂_d b)
        dth = -sum(DERIV[xd](qf[di][0]) for di, xd in enumerate(horiz))
        omega_def = (wt_m - zeta * dth
                     - sum(uvel_m[di] * (zeta * DERIV[xd](h) + DERIV[xd](b))
                           for di, xd in enumerate(horiz)))
        omega_closed = -sum(DERIV[xd](qf[di][j])
                            * sp.integrate(phis[j].subs(zeta, s_), (s_, 0, zeta))
                            for di, xd in enumerate(horiz) for j in range(1, Nu + 1))
        R_om = sp.expand(omega_closed - omega_def)
        assert sp.simplify(R_om.subs(zeta, 0)) == 0   # vanishes at the bed (KBC)

        def _zint01(e):
            poly = sp.Poly(sp.expand(e.doit()), zeta)
            return sum(cc / (nn[0] + 1)
                       for nn, cc in zip(poly.monoms(), poly.coeffs()))

        for kk in range(1, Nu + 1):
            dphi = sp.diff(phis[kk], zeta)
            for di, mn in enumerate(MOM):
                getattr(m, mn)[kk].expr = sp.expand(
                    getattr(m, mn)[kk].expr - _zint01(R_om * uvel_m[di] * dphi) / mus[kk])
            m.momentum_z[kk].expr = sp.expand(
                m.momentum_z[kk].expr - _zint01(R_om * wt_m * dphi) / mus[kk])

        # 6c — conservative regroup, DERIVED from our modal flux (no hardcoding).
        # F_{row,e,k} = (2k+1) h ∫₀¹ φ_k · flux_{row,e}(ζ) dζ, the Galerkin
        # projection of OUR physical flux of momentum-component `row` in
        # direction `e`: u_row·u_e (+ p/ρ on the diagonal of a horizontal row;
        # the vertical row carries no pressure flux).  Regrouped as a compound
        # ∂_e atom so the extraction reads it as flux; bed-slope-bearing parts
        # kept in their own atom (extraction's diffusion-branch caveat).  This
        # reproduces the Escalante flux for any Nu and any dimension.
        Pf = [m.functions.P.head(j, t, *horiz) for j in range(Nu + 1)]
        p_zeta = (sum(Pf[j] * phis[j] for j in range(Nu + 1))
                  - sum(Pf[j] for j in range(Nu + 1)) * phis[Nu + 1])
        vel_of = {mn: uvel_m[di] for di, mn in enumerate(MOM)}
        vel_of["momentum_z"] = wt_m
        bxs = [sp.Derivative(b, xd) for xd in horiz]
        for fam, vel_row in vel_of.items():
            for kk in range(1, Nu + 1):
                eq = getattr(m, fam)[kk]
                # expand ONCE per row (the .doit() would otherwise destroy a
                # prior direction's compound when this row is hit again for the
                # next flux direction); the inner sp.expand below has NO .doit(),
                # so compounds added for earlier directions survive.
                ex = sp.expand(sp.sympify(eq.expr).doit())
                for e, xe in enumerate(horiz):       # each flux direction
                    press = (p_zeta / rho if (fam in MOM and MOM[e] == fam)
                             else sp.S.Zero)
                    integrand = vel_row * uvel_m[e] + press
                    Fk = sp.expand((2 * kk + 1) * h * _zint01(phis[kk] * integrand))
                    F_bx = sum((tm for tm in sp.Add.make_args(Fk)
                                if any(tm.has(bxd) for bxd in bxs)), sp.S.Zero)
                    for F in (Fk - F_bx, F_bx):      # bed-free, then bed-slope
                        if F == 0:
                            continue
                        dF = sp.Derivative(F, xe)
                        ex = sp.expand(ex - sp.expand(dF.doit())) + dF
                eq.expr = ex

        # 6d — bed CURVATURE: over a curved bed the non-hydrostatic vertical
        # momentum picks up bare second derivatives C·∂²_e b (physical: w|_bed =
        # u·∇b).  The extraction can only route a second derivative inside a
        # conservative compound, so rewrite each bare term by the exact identity
        #   C·∂²_e b = ∂_e(C·∂_e b) − ∂_e(C)·∂_e b
        # (curvature → ∂_e(D·∂_e b) compound; remainder → first-derivative NCP).
        # No-op on a flat/mild bed and in 1-D (the §6c regroup leaves no bare ∂²b).
        for fam in MOM + ["momentum_z"]:
            for kk in range(1, Nu + 1):
                eq = getattr(m, fam)[kk]
                expr = sp.sympify(eq.expr)
                for xe in horiz:
                    d2 = sp.Derivative(b, (xe, 2))
                    out, changed = [], False
                    for tm in sp.Add.make_args(expr):
                        C = tm / d2
                        if (tm.has(d2) and not isinstance(tm, sp.Derivative)
                                and not C.has(d2)
                                and not C.has(sp.Derivative(b, xe))):
                            comp = sp.Derivative(C * sp.Derivative(b, xe), xe)
                            rem = -sp.expand(sp.Derivative(C, xe).doit()) * sp.Derivative(b, xe)
                            out.append(comp + rem); changed = True
                        else:
                            out.append(tm)
                    if changed:
                        expr = sp.Add(*out)
                eq.expr = expr

        self.derivation = m
        self._bed = b
        self._P_head = m.functions.P.head
        return None

    @property
    def system_model(self) -> SystemModel:
        """The square DAE: state ``[b, h, q_k, r_k, P_k]``; the P rows are
        the divergence constraints (zero mass-matrix rows)."""
        m = self.derivation
        Nu = int(self.level)
        # Pressure modes carry the SAME horizontal dependence as the rest of the
        # state (see derive_model's ``horiz``): dim=2 → P(j,t,x); dim=3 →
        # P(j,t,x,y).  A hard-coded ``x`` here mismatches the dim=3 derivation's
        # P(j,t,x,y) atoms, so from_model fails to recognise them as state and
        # the pressure silently drops out of every momentum/vertical row.
        horiz = (x,) if int(self.dimension) == 2 else (x, y)
        P_modes = [self._P_head(j, t, *horiz) for j in range(Nu + 1)]
        qs = list(m.explicit_state())
        if self._bed not in qs:
            qs = [self._bed, *qs]
        # Manual hydrostatic-pressure tag (one-liner): mark g·h²/2 → pressure.
        from zoomy_core.model.derivation.system_extract import HydrostaticPressure
        pf = self._pressure_flux
        m.apply({pf: HydrostaticPressure(pf)})
        sm = SystemModel.from_model(m, Q=[*qs, *P_modes])
        m.apply({HydrostaticPressure(pf): pf})   # un-tag: leave derivation clean
        from zoomy_core.model.boundary_conditions import resolve_and_attach
        resolve_and_attach(sm, self.boundary_conditions,
                           aux_bcs=self.aux_boundary_conditions)
        return sm

    def chorin_split(self, dt=None, *, system_model=None):
        """Structural Chorin split ``(SM_pred, SM_press, SM_corr)``.

        ``dt`` defaults to a fresh positive Symbol (the solver renames /
        registers it).  Pass ``system_model=`` to split an sm you already
        configured (ICs/BCs attach BEFORE the split so the sub-systems
        inherit them; re-attach BCs on ``SM_pred`` after — its aux signature
        is re-derived)."""
        from zoomy_core.model.splitter import split_for_pressure_structural
        sm = system_model if system_model is not None else self.system_model
        if dt is None:
            dt = sp.Symbol("dt", positive=True)
        P_syms = [s for s in sm.state if str(s).startswith("P_")]
        return split_for_pressure_structural(sm, P_syms, dt)
