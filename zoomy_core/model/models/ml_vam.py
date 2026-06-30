"""ML-VAM — multilayer non-hydrostatic vertically-averaged moments, the
declarative canonical model.

Per layer a VAM column in the ESCALANTE spaces (u ∈ P_N; w, p ∈ P_{N+1}
with the top modes closed — the uniform truncation left the pressure space
inconsistent and drifted secularly, exactly like the single-layer VAM did
before `7103770`).  Layers are coupled by ``KinematicBC(mass_flux=G_ℓ)``
with the Hörnschemeyer fraction-multiplier closure (``h_ℓ = l_ℓ·h``).

The 2N top-mode closures cascade through the stack:

* p̂-closures run DOWNWARD: surface ``p_N(1) = 0``; every lower layer's
  top trace equals the layer-above's bottom trace (downward convention —
  the upward one makes the elliptic block singular, see the ml_vam thesis
  notebook).  The traces are FULL modal traces including the closed top
  modes, so the chain resolves from the surface to the bed.
* ŵ-closures run UPWARD: bottom KBC ``w_1(0) = u_1(0)·∂x b``; every higher
  layer's bottom trace is the kinematic interface condition
  ``w_ℓ(0) = ∂t z_{ℓ-½} + u_ℓ(0)·∂x z_{ℓ-½} + G_{ℓ-½}/ρ``.

Per layer the advective σ-mass flux ω̃ is resolved to its CLOSED form
``ω̃(ζ) = G_{ℓ-½}/ρ − ∫₀^ζ (∂t h_ℓ + ∂x(h_ℓ ũ_ℓ))`` (Escalante's
eq(3)→(4) step with the interface mass-flux offset) — the definition form
differs by {layer-mass, constraint} combinations, equivalent ON the
constraint manifold but not in the off-manifold predictor.

Interface momentum transfer: ``u*`` is the surgical mean-trace swap on the
x-momentum rows (u-traces are genuinely one-sided).  The w-rows get NO
swap — the kinematic closures make the w-traces of adjacent layers exactly
continuous, so a w* transfer correction is identically zero.

``MLVAM(...).system_model`` is the square DAE (state
``[b, h, q_ℓk, r_ℓk, P_ℓj]``; the per-layer divergence constraints are the
zero-mass-matrix rows); ``MLVAM.chorin_split(dt)`` returns the structural
predictor / pressure / corrector sub-systems for
:class:`~zoomy_core.fvm.solver_chorin_vam_numpy.ChorinSplitVAMSolver`.
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


class MLVAM(BaseModel):
    """Multilayer non-hydrostatic VAM: ``n_layers`` σ-fraction layers, each a
    moment column with u ∈ P_level and w, p ∈ P_{level+1} (closed top modes)."""

    _finalize_lazy = True
    _cacheable_derivation = True        # derive_model returns m; byproducts on m
    n_layers = param.Integer(default=2, bounds=(2, None))
    level = param.Integer(default=1, bounds=(0, None))
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension incl. vertical: 2 → (t,x,z), one horizontal "
        "(q_ℓ_i, r_ℓ_i, P_ℓ_i); 3 → (t,x,y,z), two horizontal (q_x_ℓ_i, "
        "q_y_ℓ_i; r/P stay scalar per layer)."))
    closures = param.List(default=[], doc=(
        "Composable Closure pieces (closures.py): stress AND the interface "
        "transfer scheme (MeanInterface/UpwindInterface). Default interface "
        "scheme is the mean; empty stress leaves tau UNCLOSED."))

    def derive_model(self):
        N = int(self.n_layers)
        Nu = int(self.level)
        top = Nu + 1
        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}; CN = {x: "x", y: "y"}
        def qname(xd, ell):
            return f"q_{ell}" if dim == 2 else f"q_{CN[xd]}_{ell}"
        def shat(xd, ell):
            return (rf"\hat{{u}}_{ell}" if dim == 2
                    else rf"\hat{{{HNAME[xd]}}}_{ell}")
        def sname(xd, ell):
            return f"tau_{ell}" if dim == 2 else f"tau_{CN[xd]}z_{ell}"
        MOM = [f"momentum_{CN[xd]}" for xd in horiz]
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0}
        for j in range(1, N):
            values[f"l_{j}"] = 1.0 / N
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})

        b = sp.Function("b", real=True)(t, *horiz)
        hl = [sp.Function(f"h_{ell}", positive=True)(t, *horiz)
              for ell in range(1, N + 1)]
        H = sum(hl)
        ifaces = [b]
        for ell in range(N):
            ifaces.append(ifaces[-1] + hl[ell])
        Gf = [sp.S.Zero] + [sp.Function(f"G_{ell}", real=True)(t, *horiz)
                            for ell in range(1, N)] + [sp.S.Zero]
        P_heads = [sp.Function(f"P_{ell}", real=True)
                   for ell in range(1, N + 1)]
        lam_s, nu_s = sp.symbols("lambda_s nu", positive=True)
        rho_s = sp.Symbol("rho", positive=True)

        # inner (per-layer) basis object: the per-layer modal reconstruction,
        # top-mode closures and interface traces all go through it, so the
        # bed/surface values φ_k(0)=(−1)^k / φ_k(1)=1 and the running integral
        # ∫₀^ζ φ_j are basis primitives rather than hard-coded Legendre forms.
        inner_basis = Legendre_shifted(level=Nu + 2)
        phis = [inner_basis.eval(j, zeta) for j in range(top + 1)]
        mus = [sp.Rational(1, 2 * j + 1) for j in range(top + 1)]

        def _zint01(e):
            poly = sp.Poly(sp.expand(e.doit()), zeta)
            return sum(cc / (nn[0] + 1)
                       for nn, cc in zip(poly.monoms(), poly.coeffs()))

        def derive_layer(ell, p_top_trace_full):
            """One VAM column.  Returns (cont, constraints, momx, momz,
            p_bot_trace_full) — the bottom pressure trace (FULL, incl. the
            closed top mode) feeds the layer below."""
            z_bot, z_top, h_l = ifaces[ell - 1], ifaces[ell], hl[ell - 1]
            G_bot, G_top = Gf[ell - 1], Gf[ell]
            ml = DModel(coords=coords, parameters={"g": 9.81, "rho": 1.0})
            gl, rl = ml.parameters.g, ml.parameters.rho
            p = sp.Function(f"p_{ell}", real=True)(*coords)
            from zoomy_core.model.models.equations import (
                Mass, MomentumNonHydrostatic, small_slope_scaling)
            from types import SimpleNamespace
            from zoomy_core.model.models.material import ClosureState
            from zoomy_core.model.models.closures import apply_layer_stress_closures
            ml.add_equation(Mass(ml, suffix=f"_{ell}"))
            # tau_name → tau_<ell> in 1-D (1 horizontal uses tau_name, ignores
            # suffix); ignored in 2-D where the blueprint mints per-direction
            # tau_xz_<ell>/tau_yz_<ell> (matches sname()).
            ml.add_equation(MomentumNonHydrostatic(
                ml, suffix=f"_{ell}", tau_name=f"tau_{ell}", free_surface=b + H))
            uvel = [sp.Function(HNAME[xd] + f"_{ell}", real=True)(*coords)
                    for xd in horiz]
            w = sp.Function(f"w_{ell}", real=True)(*coords)

            def _kbc(iface, G):
                kw = dict(w=w, u=uvel[0], interface=iface, rho=rl,
                          mass_flux=(G if G != 0 else None))
                if dim == 3:
                    kw["v"] = uvel[1]
                return KinematicBC(**kw)
            ml.add_equation("kbc_bot", _kbc(z_bot, G_bot))
            ml.add_equation("kbc_top", _kbc(z_top, G_top))
            ml.apply(PDETransformation({z: (zeta, sp.Eq(z, z_bot + h_l * zeta))}))

            basis = Basis(symbol="phi", weight="c"); c = basis.weight
            kk = test_index(); phi_k = basis.phi(kk, zeta)
            legendre = Legendre_shifted(level=Nu + 2)
            pp = getattr(ml.functions, f"p_{ell}")

            for nm in ["mass"] + MOM + ["momentum_z"]:
                getattr(ml, nm).apply(Multiply(h_l))
                getattr(ml, nm).apply(Multiply(c(zeta) * phi_k))
                getattr(ml, nm).apply(ProductRule(variables=[zeta]))
                getattr(ml, nm).apply(Integrate(zeta, bounds=(0, 1)))
                getattr(ml, nm).apply(ResolveIntegral())
                getattr(ml, nm).apply(ml.kbc_bot)
                getattr(ml, nm).apply(ml.kbc_top)
                getattr(ml, nm).apply({sp.Derivative(b, t): 0})
                getattr(ml, nm).apply({pp.at(1): p_top_trace_full})
            par_ns = SimpleNamespace(rho=rl, nu=nu_s, lambda_s=lam_s)

            def _state(at, *, alias=None, btag=None):
                fields = {"u": getattr(ml.functions, f"u_{ell}"),
                          "w": getattr(ml.functions, f"w_{ell}")}
                if dim == 3:
                    fields["v"] = getattr(ml.functions, f"v_{ell}")
                return ClosureState(fields, params=par_ns, h=h_l, x=C.x,
                                    zeta=zeta, at=at, alias=alias,
                                    boundary_tag=btag, horiz=list(horiz))
            axes = [{"mx": getattr(ml, f"momentum_{CN[xd]}"),
                     "tau": getattr(ml.functions, sname(xd, ell)),
                     "velname": HNAME[xd]} for xd in horiz]
            has_bulk = apply_layer_stress_closures(
                self.closures, ml, axes, _state,
                is_top=(ell == N), is_bottom=(ell == 1))
            for nm in MOM:
                getattr(ml, nm).apply(Simplify())
            small_slope_scaling(ml)          # shallow boundary frame (n→ẑ)

            # modal ansatz: each u_d ∈ P_Nu;  w, p ∈ P_{Nu+1} (Escalante)
            coeff_heads = [sp.Function(shat(xd, ell), real=True) for xd in horiz]
            wh = sp.Function(rf"\hat{{w}}_{ell}", real=True)
            reset_modal_indices(ml)
            Nb = modal_bound("N_u")
            for i, xd in enumerate(horiz):
                ml.apply(separation_of_variables(uvel[i], coeff_heads[i](t, *horiz),
                                                 basis, Nb))
            ml.apply(separation_of_variables(w, wh(t, *horiz), basis, Nb + 1))
            if not has_bulk:
                for xd in horiz:
                    tfld = sp.Function(sname(xd, ell), real=True)(*coords)
                    sig = (rf"\hat{{\sigma}}_{ell}" if dim == 2
                           else rf"\hat{{\sigma}}_{CN[xd]}_{ell}")
                    ml.apply(separation_of_variables(tfld, sp.Function(sig, real=True)(t, *horiz),
                                                     basis, Nb + 1))
            ml.apply(separation_of_variables(
                p, P_heads[ell - 1](t, *horiz), basis, Nb + 1))

            resolve = ([("mass", range(Nu + 2))]
                       + [(mn, range(Nu + 1)) for mn in MOM]
                       + [("momentum_z", range(Nu + 1))])
            for nm, modes in resolve:
                getattr(ml, nm).apply(ExpandSums())
                getattr(ml, nm).apply(PullConstants())
                getattr(ml, nm).apply(ExtractBrackets(basis, var=zeta))
                getattr(ml, nm).apply({Nb: Nu})
                getattr(ml, nm).apply(EvaluateSums())
                getattr(ml, nm).apply(ResolveModes(index=kk, modes=modes))
                getattr(ml, nm).apply(ResolveBasis(legendre, var=zeta))

            # ── MODAL closures for the top w/p modes ─────────────────────
            # bottom kinematic/interface condition (KinematicBC orientation:
            # w|_at = ∂t I + u|_at·∂x I + G/ρ;  ∂t b = 0):
            #   Σ_j (−1)^j ŵ_j = w_bot  →  closes ŵ_top
            # top-pressure trace (downward convention):
            #   Σ_j p̂_j = p_top_trace  →  closes p̂_top
            u_at0 = [sum(inner_basis.at0(j) * coeff_heads[i](j, t, *horiz)
                         for j in range(Nu + 1)) for i in range(len(horiz))]
            w_bot = (sp.Derivative(z_bot, t).doit().subs(sp.Derivative(b, t), 0)
                     + sum(u_at0[i] * DERIV[xd](z_bot)
                           for i, xd in enumerate(horiz))
                     + G_bot / rl)
            w_top_mode = (
                w_bot - sum(inner_basis.at0(j) * wh(j, t, *horiz)
                            for j in range(top))) / inner_basis.at0(top)
            p_top_mode = (p_top_trace_full
                          - sum(inner_basis.at1(j) * P_heads[ell - 1](j, t, *horiz)
                                for j in range(top))) / inner_basis.at1(top)
            ml.apply({wh(top, t, *horiz): w_top_mode})
            ml.apply({P_heads[ell - 1](top, t, *horiz): p_top_mode})
            p_bot_trace_full = sp.expand(
                sum(inner_basis.at0(j) * P_heads[ell - 1](j, t, *horiz)
                    for j in range(top))
                + inner_basis.at0(top) * p_top_mode)

            for i, xd in enumerate(horiz):
                ml.apply(ChangeOfVariables(shat(xd, ell), qname(xd, ell),
                                           lambda qi: qi / h_l))
            ml.apply(ChangeOfVariables(
                rf"\hat{{w}}_{ell}", f"r_{ell}", lambda ri: ri / h_l))
            for nm in MOM + ["momentum_z"]:
                for k in range(Nu + 1):
                    eqk = getattr(ml, nm)[k]
                    eqk.expr = sp.expand(sp.sympify(eqk.expr).doit())
            h_eq = ml.mass[0].solve_for(d.t(h_l))
            for nm in MOM + ["momentum_z"]:
                for k in range(Nu + 1):
                    getattr(ml, nm)[k].apply(h_eq)
                    getattr(ml, nm)[k].apply(Consolidate())
            for k in range(1, Nu + 2):
                ml.mass[k].apply(h_eq)
                ml.mass[k].apply(Consolidate())
            for nm in MOM + ["momentum_z"]:
                getattr(ml, nm).apply(InvertMassMatrix())

            # ── ω̃ resolution (closed σ-mass-flux form, G-offset); 2-D couples
            #    ∂_x and ∂_y ───────────────────────────────────────────────
            qfm = [[sp.Function(qname(xd, ell), real=True)(j, t, *horiz)
                    for j in range(Nu + 1)] for xd in horiz]   # qfm[di][j]
            rfm = [sp.Function(f"r_{ell}", real=True)(j, t, *horiz)
                   for j in range(Nu + 1)]
            dt_hl = sp.sympify(h_eq.rhs)
            u_at0_c = [sum(inner_basis.at0(j) * qfm[di][j] for j in range(Nu + 1)) / h_l
                       for di in range(len(horiz))]
            zb_t = sp.Derivative(z_bot, t).doit().subs(sp.Derivative(b, t), 0)
            w_bot_c = (zb_t + sum(u_at0_c[di] * DERIV[xd](z_bot)
                                  for di, xd in enumerate(horiz)) + G_bot / rl)
            w_top_c = (
                w_bot_c - sum(inner_basis.at0(j) * rfm[j] for j in range(top)) / h_l
                ) / inner_basis.at0(top)
            uvel_m = [sum(qfm[di][j] / h_l * phis[j] for j in range(Nu + 1))
                      for di in range(len(horiz))]
            wt_m = (sum(rfm[j] / h_l * phis[j] for j in range(Nu + 1))
                    + w_top_c * phis[top])
            omega_def = (wt_m - (zb_t + zeta * dt_hl)
                         - sum(uvel_m[di] * (DERIV[xd](z_bot)
                                             + zeta * DERIV[xd](h_l))
                               for di, xd in enumerate(horiz)))
            omega_closed = (
                G_bot / rl
                - zeta * (dt_hl + sum(DERIV[xd](qfm[di][0])
                                      for di, xd in enumerate(horiz)))
                - sum(DERIV[xd](qfm[di][j]) * inner_basis.eval_psi(j, zeta)
                      for di, xd in enumerate(horiz) for j in range(1, Nu + 1)))
            R_om = sp.expand(omega_closed - omega_def)
            assert sp.simplify(R_om.subs(zeta, 0)) == 0, (
                f"layer {ell}: ω̃ forms disagree at the bottom interface")
            for k in range(1, Nu + 1):
                dphi = sp.diff(phis[k], zeta)
                for di, mn in enumerate(MOM):
                    getattr(ml, mn)[k].expr = sp.expand(
                        getattr(ml, mn)[k].expr
                        - _zint01(R_om * uvel_m[di] * dphi) / mus[k])
                ml.momentum_z[k].expr = sp.expand(
                    ml.momentum_z[k].expr - _zint01(R_om * wt_m * dphi) / mus[k])

            # DERIVED conservative flux per (row, direction) for the §6c
            # regroup (no hardcoding): F = (2k+1) h_l ∫ φ_k · flux dζ on this
            # layer's modal ansatz.  Horizontal: u_d·u_e + p/ρ (diagonal).
            # Vertical: the BULK advection w_bulk·u_e — the σ-velocity top mode
            # (∂_t/G-laden, i.e. the kinematic interface part, not a material
            # spatial flux) stays nonconservative.
            p_zeta = (sum(P_heads[ell - 1](kp, t, *horiz) * phis[kp]
                          for kp in range(Nu + 1)) + p_top_mode * phis[top])
            w_bulk = sum(rfm[j] / h_l * phis[j] for j in range(Nu + 1))
            flux_F = {}
            for k2 in range(1, Nu + 1):
                for di, xd in enumerate(horiz):
                    for e, xe in enumerate(horiz):
                        integ = uvel_m[di] * uvel_m[e]
                        if di == e:
                            integ = integ + p_zeta / rl
                        flux_F[(f"momentum_{CN[xd]}", k2, xe)] = (
                            (2 * k2 + 1) * h_l * _zint01(phis[k2] * integ))
                for e, xe in enumerate(horiz):
                    flux_F[("momentum_z", k2, xe)] = (
                        (2 * k2 + 1) * h_l * _zint01(phis[k2] * (w_bulk * uvel_m[e])))

            cont = sp.expand(ml.mass[0].expr)
            constraints = [sp.expand(ml.mass[k].expr) for k in range(1, Nu + 2)]
            momd = {xd: [sp.expand(getattr(ml, f"momentum_{CN[xd]}")[k].expr)
                         for k in range(Nu + 1)] for xd in horiz}
            momz = [sp.expand(ml.momentum_z[k].expr) for k in range(Nu + 1)]
            # per-layer vertical reconstruction profiles (LOCAL ζ), in the
            # conserved layer state (qfm=q_ℓ, rfm=r_ℓ, P_ℓ) + h_l + G_bot — the
            # SAME modal columns the §6c flux uses: ``uvel_m`` horizontal
            # velocity, ``wt_m`` the full w with the bottom-KBC top mode,
            # ``p_zeta`` the non-hydrostatic pressure with the cascaded top mode.
            # Assembled into the global piecewise reconstruction below (frac /
            # ω̃ / par substitutions applied there, exactly as the flux).
            return (cont, constraints, momd, momz, flux_F,
                    list(uvel_m), wt_m, p_zeta, p_bot_trace_full)

        # layers derived TOP-DOWN so the pressure-trace cascade resolves
        layer_eqs = {}
        p_trace = sp.S.Zero                      # surface: p_N(1) = 0
        for ell in range(N, 0, -1):
            *eqs_l, p_trace = derive_layer(ell, p_trace)
            layer_eqs[ell] = tuple(eqs_l)
        _layer_eqs_debug = layer_eqs             # pre-assembly rows (attached to m below)

        # ── Hörnschemeyer closure + per-direction shared u* transfer ──
        ht = sp.Function("h", positive=True)(t, *horiz)
        l_par = [sp.Symbol(f"l_{j}", positive=True) for j in range(1, N)]
        l_all = [*l_par, 1 - sum(l_par)]
        frac = {hl[j]: l_all[j] * ht for j in range(N)}
        q_mod = {ell: {xd: [sp.Function(qname(xd, ell), real=True)(k, t, *horiz)
                            for k in range(Nu + 1)] for xd in horiz}
                 for ell in range(1, N + 1)}
        r_mod = [[sp.Function(f"r_{ell}", real=True)(k, t, *horiz)
                  for k in range(Nu + 1)] for ell in range(1, N + 1)]
        P_mod = [[P_heads[ell - 1](j, t, *horiz) for j in range(Nu + 1)]
                 for ell in range(1, N + 1)]

        glob_c = sp.expand(
            sum(layer_eqs[ell][0] for ell in range(1, N + 1)).subs(frac).doit())
        dth_glob = sp.solve(glob_c, sp.Derivative(ht, t))[0]
        G_sol = {}
        for a in range(1, N):
            part = sp.expand(
                sum(layer_eqs[ell][0] for ell in range(1, a + 1)).subs(frac).doit())
            part = sp.expand(part.subs(sp.Derivative(ht, t), dth_glob))
            G_sol[Gf[a]] = sp.solve(part, Gf[a])[0]

        def _trace(ell, side, xd):
            sgn = lambda i: inner_basis.eval(i, side)
            return (sum(sgn(i) * q_mod[ell][xd][i] for i in range(Nu + 1))
                    / (l_all[ell - 1] * ht))

        from zoomy_core.model.models.closures import interface_closure
        iface = interface_closure(self.closures)

        def _ustar(a, xd):
            below, above = _trace(a, 1, xd), _trace(a + 1, 0, xd)
            if iface is not None:
                return iface.expression(below, above, sp.S.Zero)
            return (below + above) / 2

        # vertical-z placeholder so Model.horizontal = (x[, y]) (see ml_sme)
        m = DModel(coords=(t, *horiz, z), parameters=values)
        par = {lam_s: m.parameters.lambda_s, nu_s: m.parameters.nu}
        par.update({l_par[j - 1]: getattr(m.parameters, f"l_{j}")
                    for j in range(1, N)})
        m.add_equation("bottom", d.t(b))
        m.add_equation("continuity", sp.expand(glob_c.subs(par)))
        for ell in range(1, N + 1):
            constraints, momd, momz = (layer_eqs[ell][1], layer_eqs[ell][2],
                                       layer_eqs[ell][3])
            for xd in horiz:
                for k in range(Nu + 1):
                    row = momd[xd][k].subs(frac).doit()
                    row = sp.expand(row).subs(sp.Derivative(ht, t), dth_glob)
                    for a, side, sgn in ((ell, 1, +1), (ell - 1, 0, -1)):
                        if 1 <= a <= N - 1:
                            phik = inner_basis.eval(k, side)
                            row = row + (sgn * phik
                                         * (_ustar(a, xd) - _trace(ell, side, xd))
                                         * Gf[a] / rho_s)
                    row = sp.expand(row).subs(G_sol)
                    # CN[x]="x" → momentum_x_ℓ_k in 1-D (byte-identical names)
                    m.add_equation(f"momentum_{CN[xd]}_{ell}_{k}",
                                   sp.expand(row.subs(par).doit()))
            for k in range(Nu + 1):                       # vertical (no swap)
                row = momz[k].subs(frac).doit()
                row = sp.expand(row).subs(sp.Derivative(ht, t), dth_glob).subs(G_sol)
                m.add_equation(f"momentum_z_{ell}_{k}",
                               sp.expand(row.subs(par).doit()))
            for j, cst in enumerate(constraints):
                cst = sp.expand(cst.subs(frac).doit())
                cst = sp.expand(cst.subs(sp.Derivative(ht, t), dth_glob)).subs(G_sol)
                m.add_equation(f"constraint_{ell}_{j}",
                               sp.expand(cst.subs(par).doit()))

        m.apply(InvertMassMatrix())

        # ── 6c — conservative regroup, DERIVED from the per-layer modal flux
        # (no hardcoding, any dim/Nu).  (1) Wrap the advective + pressure flux
        # F_{row,e} = (2k+1) h ∫ φ_k flux dζ as a conservative ∂_e compound
        # (bed-slope-free part); (2) a generalized second-derivative absorption
        # routes the bed-curvature / G-geometry ∂²-terms (any spatial direction,
        # incl. ∂_x∂_y) into the diffusion operator.
        space = list(horiz)
        for ell in range(1, N + 1):
            flux_F = layer_eqs[ell][4]
            by_row = {}
            for (rowbase, k2, xe), F in flux_F.items():
                by_row.setdefault((rowbase, k2), []).append((xe, F))
            for (rowbase, k2), flist in by_row.items():
                name = f"{rowbase}_{ell}_{k2}"
                if name not in m._equations:
                    continue
                eq = getattr(m, name)
                ex = sp.expand(sp.sympify(eq.expr).doit())
                for xe, F in flist:
                    Fa = sp.expand(sp.sympify(F).subs(frac).doit()
                                   .subs(sp.Derivative(ht, t), dth_glob)
                                   .subs(G_sol).subs(par))
                    # bed-slope-bearing flux stays nonconservative; the rest is
                    # the conservative advective/pressure ∂_e flux.
                    F_bed = sum((tm for tm in sp.Add.make_args(Fa) if tm.has(b)),
                                sp.S.Zero)
                    F_free = Fa - F_bed
                    if F_free != 0:
                        dF = sp.Derivative(F_free, xe)
                        ex = sp.expand(ex - sp.expand(dF.doit())) + dF
                eq.expr = ex
        bases = ([f"momentum_{CN[xd]}_{ell}"
                  for ell in range(1, N + 1) for xd in horiz]
                 + [f"momentum_z_{ell}" for ell in range(1, N + 1)])
        for base in bases:
            for k in range(Nu + 1):
                name = f"{base}_{k}"
                if name not in m._equations:
                    continue
                eq = getattr(m, name)
                e = sp.expand(sp.sympify(eq.expr).doit())
                for e_dir in space:
                    F_d = sp.S.Zero
                    for a in list(e.atoms(sp.Derivative)):
                        vs = list(a.variables)
                        if len(vs) == 2 and vs[0] == e_dir and vs[1] in space:
                            F_d = F_d + e.coeff(a) * sp.Derivative(a.expr, vs[1])
                    if F_d != 0:
                        dF = sp.Derivative(sp.expand(F_d), e_dir)
                        e = sp.expand(e - sp.expand(dF.doit())) + dF
                eq.expr = e

        # ── interpolate_to_3d + project_from_3d, PIECEWISE over the moving
        # layers (same canonical operators as VAM / ML-SME).  Layer ℓ spans the
        # global ζ ∈ [c_{ℓ-1}, c_ℓ] (c_ℓ = Σ_{j≤ℓ} l_j); within it the LOCAL
        # column ζ_loc = (ζ − c_{ℓ-1})/l_ℓ carries the per-layer modal profiles
        # returned by ``derive_layer``.  Each profile gets the SAME assembly
        # substitutions the §6c flux did (frac h_ℓ=l_ℓ·h, ∂_t h → global mass,
        # interface fluxes G, parameters).  Field order [b, h, u(, v), w, p];
        # slot 5 is the TOTAL pressure ρ g h (1−ζ) + the non-hydrostatic part
        # (Escalante split uses the GLOBAL free surface η = b + H).
        cum = [sp.S.Zero]
        for lf in l_all:
            cum.append(cum[-1] + lf)

        def _xform(prof, ell):
            """Local-ζ layer profile → global-ζ, with the assembly subs."""
            lf, c0 = l_all[ell - 1], cum[ell - 1]
            e = sp.sympify(prof).subs(zeta, (zeta - c0) / lf)
            e = sp.expand(e.subs(frac).doit())
            e = e.subs(sp.Derivative(ht, t), dth_glob).subs(G_sol)
            return e

        def _piece(prof_of_ell):
            pieces = []
            for ell in range(1, N + 1):
                val = _xform(prof_of_ell(ell), ell)
                cond = (zeta <= cum[ell]) if ell < N else True
                pieces.append((val, cond))
            return sp.Piecewise(*pieces).subs(par)

        interp = {0: b, 1: ht}
        for di, xd in enumerate(horiz):
            interp[2 + di] = _piece(lambda ell, di=di: layer_eqs[ell][5][di])
        interp[4] = _piece(lambda ell: layer_eqs[ell][6])
        interp[5] = (m.parameters.rho * m.parameters.g * ht * (1 - zeta)
                     + _piece(lambda ell: layer_eqs[ell][7]))
        m.interpolate_rows = interp

        # inverse: per-layer Integral-FREE fixed-node Galerkin reduction (see
        # ML-SME).  Layer ℓ samples its profile at N_z LOCAL nodes t∈[0,1] mapped
        # to the global ζ = c0 + l_ℓ·t.  Conserved moments q_ℓ, r_ℓ carry the
        # physical layer-height factor h·l_ℓ; the pressure modes P_ℓ are plain
        # modal coefficients of the NON-hydrostatic column (norm 1, total-pressure
        # sample with its hydrostatic ρ g h (1−ζ) removed).
        proj_legendre = Legendre_shifted(level=Nu + 2)
        N_z = 33
        loc = [float(j) / (N_z - 1) for j in range(N_z)]
        wq = [1.0 / (N_z - 1)] * N_z
        wq[0] *= 0.5; wq[-1] *= 0.5
        P3 = {f: sp.Symbol(f"P3_{f}", real=True) for f in ("b", "h")}
        rho_p, g_p = m.parameters.rho, m.parameters.g
        proj = {b: P3["b"], ht: P3["h"]}
        for xd in horiz:
            P3vel = sp.Function(f"P3_{HNAME[xd]}", real=True)
            for ell in range(1, N + 1):
                lf = l_all[ell - 1].subs(par); c0 = cum[ell - 1].subs(par)
                samples = [P3vel(c0 + lf * tt) for tt in loc]
                rows = proj_legendre.projection_rows(
                    loc, wq, samples, norm=lambda _k, _lf=lf: P3["h"] * _lf)
                for k in range(Nu + 1):
                    proj[q_mod[ell][xd][k]] = rows[k]
        P3w = sp.Function("P3_w", real=True)
        for ell in range(1, N + 1):
            lf = l_all[ell - 1].subs(par); c0 = cum[ell - 1].subs(par)
            samples = [P3w(c0 + lf * tt) for tt in loc]
            rows = proj_legendre.projection_rows(
                loc, wq, samples, norm=lambda _k, _lf=lf: P3["h"] * _lf)
            for k in range(Nu + 1):
                proj[r_mod[ell - 1][k]] = rows[k]
        P3p = sp.Function("P3_p", real=True)
        for ell in range(1, N + 1):
            lf = l_all[ell - 1].subs(par); c0 = cum[ell - 1].subs(par)
            samples = [P3p(c0 + lf * tt) - rho_p * g_p * P3["h"] * (1 - (c0 + lf * tt))
                       for tt in loc]
            rows = proj_legendre.projection_rows(loc, wq, samples, norm=None)
            for k in range(Nu + 1):
                proj[P_mod[ell - 1][k]] = rows[k]
        m.project_rows = proj

        m.bed = b
        m.ht = ht
        m.q_flat = [q_mod[ell][xd][k]
                    for ell in range(1, N + 1)
                    for xd in horiz for k in range(Nu + 1)]
        m.r_flat = [r for layer in r_mod for r in layer]
        m.P_flat = [p for layer in P_mod for p in layer]
        m.layer_eqs_debug = _layer_eqs_debug
        return m

    @property
    def system_model(self) -> SystemModel:
        m = self.derivation
        sm = SystemModel.from_model(
            m, Q=[m.bed, m.ht, *m.q_flat, *m.r_flat, *m.P_flat])
        from zoomy_core.model.boundary_conditions import resolve_and_attach
        resolve_and_attach(sm, self.boundary_conditions,
                           aux_bcs=self.aux_boundary_conditions)
        return sm

    def chorin_split(self, dt=None, *, system_model=None):
        """Structural Chorin split (predictor / pressure / corrector)."""
        from zoomy_core.model.splitter import split_for_pressure_structural
        sm = system_model if system_model is not None else self.system_model
        if dt is None:
            dt = sp.Symbol("dt", positive=True)
        P_syms = [s for s in sm.state if str(s).startswith("P_")]
        return split_for_pressure_structural(sm, P_syms, dt)
