"""ML-VAM — multilayer non-hydrostatic vertically-averaged moments, the
declarative canonical model.

Per layer a VAM column with the hydrostatic / non-hydrostatic pressure
split (only the non-hydrostatic ``p_ℓ`` stays modal); layers coupled by
``KinematicBC(mass_flux=G_ℓ)`` with the Hörnschemeyer fraction-multiplier
closure (``h_ℓ = l_ℓ·h``); shared interface transfer velocities ``u*`` AND
``w*`` (surgical trace swap, mean of the one-sided modal traces).

Pressure traces use the **downward convention** (pressure propagates from
the surface): ``p_N(ζ=1) = 0`` and every lower layer's TOP trace is the
layer-above's bottom value ``Σ_j (−1)^j P_{ℓ+1,j}``.  The upward convention
cancels the bottom layer's k=1 vertical momentum pressure exactly and makes
the elliptic block singular — see the ml_vam thesis notebook.

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
    ResolveModes, ResolveBasis, ChangeOfVariables,
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
    moment column of order ``level`` for u, w and the non-hydrostatic p."""

    _finalize_lazy = True
    n_layers = param.Integer(default=2, bounds=(2, None))
    level = param.Integer(default=1, bounds=(0, None))

    def derive_model(self):
        N = int(self.n_layers)
        Nu = int(self.level)
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

        def derive_layer(ell):
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
            # downward pressure-trace convention (see module docstring)
            if ell == N:
                p_top_trace = sp.S.Zero
            else:
                p_top_trace = sum((-1) ** j * P_heads[ell](j, t, x)
                                  for j in range(Nu + 1))

            for nm in ("mass", "momentum_x", "momentum_z"):
                getattr(ml, nm).apply(Multiply(h_l))
                getattr(ml, nm).apply(Multiply(c(zeta) * phi_k))
                getattr(ml, nm).apply(ProductRule(variables=[zeta]))
                getattr(ml, nm).apply(Integrate(zeta, bounds=(0, 1)))
                getattr(ml, nm).apply(ResolveIntegral())
                getattr(ml, nm).apply(ml.kbc_bot)
                getattr(ml, nm).apply(ml.kbc_top)
                getattr(ml, nm).apply({sp.Derivative(b, t): 0})
                getattr(ml, nm).apply({pp.at(1): p_top_trace})
            tau = getattr(ml.functions, f"tau_{ell}")
            uu = getattr(ml.functions, f"u_{ell}")
            bot_stress = lam_s * uu.at(0) if ell == 1 else 0
            ml.momentum_x.apply({tau.at(1): 0, tau.at(0): bot_stress})
            ml.momentum_x.apply(
                {tau.expr: rl * nu_s / h_l * sp.Derivative(uu.expr, zeta)})
            ml.momentum_x.apply(Simplify())

            uh = sp.Function(rf"\hat{{u}}_{ell}", real=True)
            wh = sp.Function(rf"\hat{{w}}_{ell}", real=True)
            reset_modal_indices(ml)
            Nb = modal_bound("N_u")
            ml.apply(separation_of_variables(u, uh(t, x), basis, Nb))
            ml.apply(separation_of_variables(w, wh(t, x), basis, Nb))
            ml.apply(separation_of_variables(
                p, P_heads[ell - 1](t, x), basis, Nb))

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
            ml.apply(ChangeOfVariables(
                rf"\hat{{u}}_{ell}", f"q_{ell}", lambda qi: qi / h_l))
            ml.apply(ChangeOfVariables(
                rf"\hat{{w}}_{ell}", f"r_{ell}", lambda ri: ri / h_l))
            h_eq = ml.mass[0].solve_for(d.t(h_l))
            for nm in ("momentum_x", "momentum_z"):
                for k in range(Nu + 1):
                    getattr(ml, nm)[k].apply(h_eq)
                    getattr(ml, nm)[k].apply(Consolidate())
            for k in range(1, Nu + 2):
                ml.mass[k].apply(h_eq)
                ml.mass[k].apply(Consolidate())
            cont = sp.expand(ml.mass[0].expr)
            constraints = [sp.expand(ml.mass[k].expr)
                           for k in range(1, Nu + 2)]
            momx = [sp.expand(ml.momentum_x[k].expr) for k in range(Nu + 1)]
            momz = [sp.expand(ml.momentum_z[k].expr) for k in range(Nu + 1)]
            return cont, constraints, momx, momz

        layer_eqs = {ell: derive_layer(ell) for ell in range(1, N + 1)}

        # ── Hörnschemeyer closure + shared u*/w* transfers ─────────────
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

        def _vstar(mod, a):
            return (_trace(mod, a, 1) + _trace(mod, a + 1, 0)) / 2

        m = DModel(coords=(t, x), parameters=values)
        par = {lam_s: m.parameters.lambda_s, nu_s: m.parameters.nu}
        par.update({l_par[j - 1]: getattr(m.parameters, f"l_{j}")
                    for j in range(1, N)})
        m.add_equation("bottom", d.t(b))
        m.add_equation("continuity", sp.expand(glob_c.subs(par)))
        for ell in range(1, N + 1):
            _, constraints, momx, momz = layer_eqs[ell]
            for k in range(Nu + 1):
                for rows, mod, name in ((momx, q_mod, "momentum_x"),
                                        (momz, r_mod, "momentum_z")):
                    row = rows[k].subs(frac).doit()
                    row = sp.expand(row).subs(
                        sp.Derivative(ht, t), dth_glob)
                    for a, side, sgn in ((ell, 1, +1), (ell - 1, 0, -1)):
                        if 1 <= a <= N - 1:
                            phik = 1 if side == 1 else (-1) ** k
                            row = row + (sgn * phik
                                         * (_vstar(mod, a)
                                            - _trace(mod, ell, side))
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
