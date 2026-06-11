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

t, x, z = C.t, C.x, C.z
zeta = sp.Symbol("zeta", real=True)


class MLVAM(BaseModel):
    """Multilayer non-hydrostatic VAM: ``n_layers`` σ-fraction layers, each a
    moment column with u ∈ P_level and w, p ∈ P_{level+1} (closed top modes)."""

    _finalize_lazy = True
    n_layers = param.Integer(default=2, bounds=(2, None))
    level = param.Integer(default=1, bounds=(0, None))

    def derive_model(self):
        N = int(self.n_layers)
        Nu = int(self.level)
        top = Nu + 1
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0}
        for j in range(1, N):
            values[f"l_{j}"] = 1.0 / N
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})

        b = sp.Function("b", real=True)(t, x)
        hl = [sp.Function(f"h_{ell}", positive=True)(t, x)
              for ell in range(1, N + 1)]
        H = sum(hl)
        ifaces = [b]
        for ell in range(N):
            ifaces.append(ifaces[-1] + hl[ell])
        Gf = [sp.S.Zero] + [sp.Function(f"G_{ell}", real=True)(t, x)
                            for ell in range(1, N)] + [sp.S.Zero]
        P_heads = [sp.Function(f"P_{ell}", real=True)
                   for ell in range(1, N + 1)]
        lam_s, nu_s = sp.symbols("lambda_s nu", positive=True)
        rho_s = sp.Symbol("rho", positive=True)

        phis = [sp.legendre(j, 2 * zeta - 1) for j in range(top + 1)]
        mus = [sp.Rational(1, 2 * j + 1) for j in range(top + 1)]
        s_om = sp.Symbol("_s_omega")

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
            ml = DModel(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
            gl, rl = ml.parameters.g, ml.parameters.rho
            u = sp.Function(f"u_{ell}", real=True)(t, x, z)
            w = sp.Function(f"w_{ell}", real=True)(t, x, z)
            p = sp.Function(f"p_{ell}", real=True)(t, x, z)
            txz = sp.Function(f"tau_{ell}", real=True)(t, x, z)
            ml.Q = [u, w, p]
            ml.add_equation("mass", d.x(u) + d.z(w))
            ml.add_equation("momentum_x",
                            d.t(u) + d.x(u * u) + d.z(u * w)
                            + gl * d.x(b + H) + d.x(p) / rl - d.z(txz) / rl)
            ml.add_equation("momentum_z",
                            d.t(w) + d.x(u * w) + d.z(w * w) + d.z(p) / rl)
            ml.add_equation("kbc_bot", KinematicBC(
                w=w, u=u, interface=z_bot, rho=rl,
                mass_flux=(G_bot if G_bot != 0 else None)))
            ml.add_equation("kbc_top", KinematicBC(
                w=w, u=u, interface=z_top, rho=rl,
                mass_flux=(G_top if G_top != 0 else None)))
            ml.apply(PDETransformation(
                {z: (zeta, sp.Eq(z, z_bot + h_l * zeta))}))

            basis = Basis(symbol="phi", weight="c"); c = basis.weight
            kk = test_index(); phi_k = basis.phi(kk, zeta)
            legendre = Legendre_shifted(level=Nu + 2)
            pp = getattr(ml.functions, f"p_{ell}")

            for nm in ("mass", "momentum_x", "momentum_z"):
                getattr(ml, nm).apply(Multiply(h_l))
                getattr(ml, nm).apply(Multiply(c(zeta) * phi_k))
                getattr(ml, nm).apply(ProductRule(variables=[zeta]))
                getattr(ml, nm).apply(Integrate(zeta, bounds=(0, 1)))
                getattr(ml, nm).apply(ResolveIntegral())
                getattr(ml, nm).apply(ml.kbc_bot)
                getattr(ml, nm).apply(ml.kbc_top)
                getattr(ml, nm).apply({sp.Derivative(b, t): 0})
                getattr(ml, nm).apply({pp.at(1): p_top_trace_full})
            tau = getattr(ml.functions, f"tau_{ell}")
            uu = getattr(ml.functions, f"u_{ell}")
            bot_stress = lam_s * uu.at(0) if ell == 1 else 0
            ml.momentum_x.apply({tau.at(1): 0, tau.at(0): bot_stress})
            ml.momentum_x.apply(
                {tau.expr: rl * nu_s / h_l * sp.Derivative(uu.expr, zeta)})
            ml.momentum_x.apply(Simplify())

            # modal ansatz: u ∈ P_Nu;  w, p ∈ P_{Nu+1} (Escalante spaces)
            uh = sp.Function(rf"\hat{{u}}_{ell}", real=True)
            wh = sp.Function(rf"\hat{{w}}_{ell}", real=True)
            reset_modal_indices(ml)
            Nb = modal_bound("N_u")
            ml.apply(separation_of_variables(u, uh(t, x), basis, Nb))
            ml.apply(separation_of_variables(w, wh(t, x), basis, Nb + 1))
            ml.apply(separation_of_variables(
                p, P_heads[ell - 1](t, x), basis, Nb + 1))

            for nm, modes in (("mass", range(Nu + 2)),
                              ("momentum_x", range(Nu + 1)),
                              ("momentum_z", range(Nu + 1))):
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
            u_at0 = sum((-1) ** j * uh(j, t, x) for j in range(Nu + 1))
            w_bot = (sp.Derivative(z_bot, t).doit().subs(
                         sp.Derivative(b, t), 0)
                     + u_at0 * sp.Derivative(z_bot, x)
                     + G_bot / rl)
            w_top_mode = (-1) ** top * (
                w_bot - sum((-1) ** j * wh(j, t, x) for j in range(top)))
            p_top_mode = p_top_trace_full - sum(
                P_heads[ell - 1](j, t, x) for j in range(top))
            ml.apply({wh(top, t, x): w_top_mode})
            ml.apply({P_heads[ell - 1](top, t, x): p_top_mode})
            # bottom pressure trace (FULL, with the closed top mode) for the
            # layer below: Σ_j (−1)^j p̂_j  with  p̂_top → its closure
            p_bot_trace_full = sp.expand(
                sum((-1) ** j * P_heads[ell - 1](j, t, x)
                    for j in range(top))
                + (-1) ** top * p_top_mode)

            ml.apply(ChangeOfVariables(
                rf"\hat{{u}}_{ell}", f"q_{ell}", lambda qi: qi / h_l))
            ml.apply(ChangeOfVariables(
                rf"\hat{{w}}_{ell}", f"r_{ell}", lambda ri: ri / h_l))
            # expand the CoV-introduced compound derivatives (∂t(r/h_ℓ) etc.)
            # to ATOMIC form — h_eq and InvertMassMatrix can't see inside a
            # compound, leaving the z-rows μ-weighted (the Δ below assumes
            # unit-∂t rows)
            for nm in ("momentum_x", "momentum_z"):
                for k in range(Nu + 1):
                    eqk = getattr(ml, nm)[k]
                    eqk.expr = sp.expand(sp.sympify(eqk.expr).doit())
            h_eq = ml.mass[0].solve_for(d.t(h_l))
            for nm in ("momentum_x", "momentum_z"):
                for k in range(Nu + 1):
                    getattr(ml, nm)[k].apply(h_eq)
                    getattr(ml, nm)[k].apply(Consolidate())
            for k in range(1, Nu + 2):
                ml.mass[k].apply(h_eq)
                ml.mass[k].apply(Consolidate())
            # AFTER the stray dt-h substitutions (op docstring).  MOMENTUM
            # families only: the mass rows are the divergence constraints
            # (no ∂t to normalize) — running IMM over them divides layer-2
            # rows by the coefficient of a CROSS-layer ∂t h₁ trace.
            ml.momentum_x.apply(InvertMassMatrix())
            ml.momentum_z.apply(InvertMassMatrix())

            # ── ω̃ resolution (closed σ-mass-flux form, G-offset) ─────────
            qfm = [sp.Function(f"q_{ell}", real=True)(j, t, x)
                   for j in range(Nu + 1)]
            rfm = [sp.Function(f"r_{ell}", real=True)(j, t, x)
                   for j in range(Nu + 1)]
            dt_hl = sp.sympify(h_eq.rhs)
            u_at0_c = sum((-1) ** j * qfm[j] for j in range(Nu + 1)) / h_l
            zb_t = sp.Derivative(z_bot, t).doit().subs(sp.Derivative(b, t), 0)
            zb_x = sp.Derivative(z_bot, x)
            w_bot_c = zb_t + u_at0_c * zb_x + G_bot / rl
            w_top_c = (-1) ** top * (
                w_bot_c - sum((-1) ** j * rfm[j] for j in range(top)) / h_l)
            ut_m = sum(qfm[j] / h_l * phis[j] for j in range(Nu + 1))
            wt_m = (sum(rfm[j] / h_l * phis[j] for j in range(Nu + 1))
                    + w_top_c * phis[top])
            omega_def = (wt_m - (zb_t + zeta * dt_hl)
                         - ut_m * (zb_x + zeta * sp.Derivative(h_l, x)))
            omega_closed = (G_bot / rl
                            - zeta * (dt_hl + sp.Derivative(qfm[0], x))
                            - sum(sp.Derivative(qfm[j], x)
                                  * sp.integrate(phis[j].subs(zeta, s_om),
                                                 (s_om, 0, zeta))
                                  for j in range(1, Nu + 1)))
            R_om = sp.expand(omega_closed - omega_def)
            assert sp.simplify(R_om.subs(zeta, 0)) == 0, (
                f"layer {ell}: ω̃ forms disagree at the bottom interface — "
                "ŵ-closure / KinematicBC orientation mismatch")
            for k in range(1, Nu + 1):
                dphi = sp.diff(phis[k], zeta)
                ml.momentum_x[k].expr = sp.expand(
                    ml.momentum_x[k].expr
                    - _zint01(R_om * ut_m * dphi) / mus[k])
                ml.momentum_z[k].expr = sp.expand(
                    ml.momentum_z[k].expr
                    - _zint01(R_om * wt_m * dphi) / mus[k])

            cont = sp.expand(ml.mass[0].expr)
            constraints = [sp.expand(ml.mass[k].expr)
                           for k in range(1, Nu + 2)]
            momx = [sp.expand(ml.momentum_x[k].expr) for k in range(Nu + 1)]
            momz = [sp.expand(ml.momentum_z[k].expr) for k in range(Nu + 1)]
            return cont, constraints, momx, momz, p_bot_trace_full

        # layers derived TOP-DOWN so the pressure-trace cascade resolves
        layer_eqs = {}
        p_trace = sp.S.Zero                      # surface: p_N(1) = 0
        for ell in range(N, 0, -1):
            *eqs_l, p_trace = derive_layer(ell, p_trace)
            layer_eqs[ell] = tuple(eqs_l)
        self._layer_eqs_debug = layer_eqs        # pre-assembly rows (tests)

        # ── Hörnschemeyer closure + shared u* transfer (x-momentum only) ──
        ht = sp.Function("h", positive=True)(t, x)
        l_par = [sp.Symbol(f"l_{j}", positive=True) for j in range(1, N)]
        l_all = [*l_par, 1 - sum(l_par)]
        frac = {hl[j]: l_all[j] * ht for j in range(N)}
        q_mod = [[sp.Function(f"q_{ell}", real=True)(k, t, x)
                  for k in range(Nu + 1)] for ell in range(1, N + 1)]
        r_mod = [[sp.Function(f"r_{ell}", real=True)(k, t, x)
                  for k in range(Nu + 1)] for ell in range(1, N + 1)]
        P_mod = [[P_heads[ell - 1](j, t, x) for j in range(Nu + 1)]
                 for ell in range(1, N + 1)]

        glob_c = sp.expand(
            sum(layer_eqs[ell][0] for ell in range(1, N + 1))
            .subs(frac).doit())
        dth_glob = sp.solve(glob_c, sp.Derivative(ht, t))[0]
        G_sol = {}
        for a in range(1, N):
            part = sp.expand(
                sum(layer_eqs[ell][0] for ell in range(1, a + 1))
                .subs(frac).doit())
            part = sp.expand(part.subs(sp.Derivative(ht, t), dth_glob))
            G_sol[Gf[a]] = sp.solve(part, Gf[a])[0]

        def _trace(mod, ell, side):
            sgn = (lambda i: 1) if side == 1 else (lambda i: (-1) ** i)
            return (sum(sgn(i) * mod[ell - 1][i] for i in range(Nu + 1))
                    / (l_all[ell - 1] * ht))

        def _ustar(a):
            return (_trace(q_mod, a, 1) + _trace(q_mod, a + 1, 0)) / 2

        m = DModel(coords=(t, x), parameters=values)
        par = {lam_s: m.parameters.lambda_s, nu_s: m.parameters.nu}
        par.update({l_par[j - 1]: getattr(m.parameters, f"l_{j}")
                    for j in range(1, N)})
        m.add_equation("bottom", d.t(b))
        m.add_equation("continuity", sp.expand(glob_c.subs(par)))
        for ell in range(1, N + 1):
            _, constraints, momx, momz = layer_eqs[ell]
            for k in range(Nu + 1):
                for rows, name, swap in ((momx, "momentum_x", True),
                                         (momz, "momentum_z", False)):
                    row = rows[k].subs(frac).doit()
                    row = sp.expand(row).subs(
                        sp.Derivative(ht, t), dth_glob)
                    if swap:
                        # u* mean-trace transfer swap (u-traces one-sided);
                        # w-rows: kinematic closures make w-traces exactly
                        # continuous → the w* swap is identically zero
                        for a, side, sgn in ((ell, 1, +1), (ell - 1, 0, -1)):
                            if 1 <= a <= N - 1:
                                phik = 1 if side == 1 else (-1) ** k
                                row = row + (sgn * phik
                                             * (_ustar(a)
                                                - _trace(q_mod, ell, side))
                                             * Gf[a] / rho_s)
                    row = sp.expand(row).subs(G_sol)
                    m.add_equation(f"{name}_{ell}_{k}",
                                   sp.expand(row.subs(par).doit()))
            for j, cst in enumerate(constraints):
                cst = sp.expand(cst.subs(frac).doit())
                cst = sp.expand(
                    cst.subs(sp.Derivative(ht, t), dth_glob)).subs(G_sol)
                m.add_equation(f"constraint_{ell}_{j}",
                               sp.expand(cst.subs(par).doit()))

        # final pass on the ASSEMBLED rows (cross-layer ∂_t h traces are
        # only eliminated at assembly via dth_glob, so the per-layer pass
        # can miss a row; idempotent for already-normalized ones).
        m.apply(InvertMassMatrix())

        # ── conservative regroup of the k=1 advective divergences ────────
        # (see VAM single-layer 6c: the closure/ω̃ substitutions leave the
        # rows fully expanded → all-NCP routing, path-dependent at shocks;
        # the b″/G-bearing flux parts go in their OWN compound atom so the
        # extraction's diffusion branch can route them without drops.)
        if Nu == 1:
            for ell in range(1, N + 1):
                h_loc = l_all[ell - 1] * ht
                q0m, q1m = q_mod[ell - 1]
                r0m, r1m = r_mod[ell - 1]
                P1m = P_mod[ell - 1][1]
                F_groups = {
                    f"momentum_x_{ell}_1": [
                        2 * q0m * q1m / h_loc + h_loc * P1m / rho_s],
                    f"momentum_z_{ell}_1": [
                        ((q0m * r1m + q1m * r0m) / h_loc
                         - sp.Rational(2, 5) * q1m * (r0m - r1m) / h_loc)],
                }
                for name, Fs in F_groups.items():
                    eq = getattr(m, name)
                    e = sp.expand(sp.sympify(eq.expr).doit())
                    # hand-derived advective flux atoms (derivative-free)
                    for F in Fs:
                        F = sp.expand(sp.sympify(F).subs(par))
                        dF = sp.Derivative(F, x)
                        e = sp.expand(e - sp.expand(dF.doit())) + dF
                    # generic absorption: every remaining BARE second-
                    # derivative term c·∂xx(s) is the expanded image of a
                    # conservative ∂x(c·∂x s) flux (the G/geometry parts of
                    # the ŵ-top closure) — regroup them into ONE compound
                    # atom (content-preserving for any c; the matching
                    # ∂x(c)·∂x(s) partners stay in NCP, which is fine)
                    F_d = sp.S.Zero
                    for a in list(e.atoms(sp.Derivative)):
                        if a.variables == (x, x):
                            F_d = F_d + e.coeff(a) * sp.Derivative(a.expr, x)
                    if F_d != 0:
                        dF = sp.Derivative(sp.expand(F_d), x)
                        e = sp.expand(e - sp.expand(dF.doit())) + dF
                    eq.expr = e

        self.derivation = m
        self._bed = b
        self._ht = ht
        self._q_flat = [q for layer in q_mod for q in layer]
        self._r_flat = [r for layer in r_mod for r in layer]
        self._P_flat = [p for layer in P_mod for p in layer]
        return None

    @property
    def system_model(self) -> SystemModel:
        m = self.derivation
        sm = SystemModel.from_model(
            m, Q=[self._bed, self._ht, *self._q_flat, *self._r_flat,
                  *self._P_flat])
        if self.boundary_conditions is not None:
            sm.attach_boundary_conditions(
                self.boundary_conditions, aux_bcs=self.aux_boundary_conditions)
        return sm

    def chorin_split(self, dt=None, *, system_model=None):
        """Structural Chorin split (predictor / pressure / corrector)."""
        from zoomy_core.model.splitter import split_for_pressure_structural
        sm = system_model if system_model is not None else self.system_model
        if dt is None:
            dt = sp.Symbol("dt", positive=True)
        P_syms = [s for s in sm.state if str(s).startswith("P_")]
        return split_for_pressure_structural(sm, P_syms, dt)
