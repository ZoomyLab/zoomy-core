"""ML-SME — multilayer shallow MOMENT equations, the declarative canonical
model.

Every layer is a full SME column (moments ``q_ℓ_0 … q_ℓ_level`` + the ŵ
closure), derived per layer with its own σ-map and
``KinematicBC(mass_flux=G)`` interfaces; the internal-interface mass fluxes
are closed by the **Hörnschemeyer fraction-multiplier**
(``h_ℓ = l_ℓ·h`` ⇒ ``G_ℓ = ρ·Σ_{j≤ℓ}(l_j·∂_x Q − ∂_x q_j_0)``), exactly as
:class:`~zoomy_core.model.models.ml_swe.MLSWE`.

Moment-specific care: G also enters the moment rows through the ŵ closure —
those terms are KINEMATIC (the vertical velocity carries the mass flux; no
velocity ambiguity) and stay untouched.  Only the advection-trace TRANSFER
``±φ_k(side)·u_ℓ(side)·G/ρ`` gets the shared interface velocity ``u*``
(sign-of-G upwind, Hörnschemeyer Eq. 9, or mean), via the surgical
correction ``mom_k += ±φ_k(side)·(u* − u_ℓ(side))·G/ρ`` with the modal
traces ``u_ℓ(1) = Σ_i q_ℓ_i/h_ℓ``, ``u_ℓ(0) = Σ_i (−1)^i q_ℓ_i/h_ℓ``.

State: ``[b, h, q_1_0 … q_1_level, …, q_N_0 … q_N_level]``.
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
    ResolveModes, ResolveBasis, SolveLinearSystem, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.systemmodel import SystemModel

t, x, z = C.t, C.x, C.z
zeta = sp.Symbol("zeta", real=True)


class MLSME(BaseModel):
    """Multilayer SME: ``n_layers`` moving-with-the-surface layers, each a
    moment column of order ``level``."""

    _finalize_lazy = True
    n_layers = param.Integer(default=2, bounds=(2, None))
    level = param.Integer(default=1, bounds=(0, None))
    interface_velocity = param.Selector(
        default="upwind", objects=["upwind", "mean"],
        doc="shared transfer velocity u* at internal interfaces")

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
        lam_s, nu_s = sp.symbols("lambda_s nu", positive=True)
        rho_s = sp.Symbol("rho", positive=True)

        def derive_layer(ell):
            z_bot, z_top, h_l = ifaces[ell - 1], ifaces[ell], hl[ell - 1]
            G_bot, G_top = Gf[ell - 1], Gf[ell]
            ml = DModel(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
            gl, rl = ml.parameters.g, ml.parameters.rho
            u = sp.Function(f"u_{ell}", real=True)(t, x, z)
            w = sp.Function(f"w_{ell}", real=True)(t, x, z)
            txz = sp.Function(f"tau_{ell}", real=True)(t, x, z)
            ml.Q = [u, w]
            ml.add_equation("mass", d.x(u) + d.z(w))
            ml.add_equation("momentum_x",
                            d.t(u) + d.x(u * u) + d.z(u * w)
                            + gl * d.x(b + H) - d.z(txz) / rl)
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
            for nm in ("mass", "momentum_x"):
                getattr(ml, nm).apply(Multiply(h_l))
                getattr(ml, nm).apply(Multiply(c(zeta) * phi_k))
                getattr(ml, nm).apply(ProductRule(variables=[zeta]))
                getattr(ml, nm).apply(Integrate(zeta, bounds=(0, 1)))
                getattr(ml, nm).apply(ResolveIntegral())
                getattr(ml, nm).apply(ml.kbc_bot)
                getattr(ml, nm).apply(ml.kbc_top)
                getattr(ml, nm).apply({sp.Derivative(b, t): 0})
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
            ml.apply(separation_of_variables(w, wh(t, x), basis, Nb + 1))
            # mass: k=0 h-eq + ŵ closure rows
            ml.mass.apply(ExpandSums()); ml.mass.apply(PullConstants())
            ml.mass.apply(ExtractBrackets(basis, var=zeta))
            ml.mass.apply({Nb: Nu}); ml.mass.apply(EvaluateSums())
            ml.mass.apply(ResolveModes(index=kk, modes=range(Nu + 3)))
            ml.mass.apply(ResolveBasis(legendre, var=zeta))
            h_eq = ml.mass[0].solve_for(d.t(h_l))
            for row in ml.mass[1:Nu + 3]:
                row.apply(h_eq)
            w_closure = SolveLinearSystem(
                ml.mass[1:Nu + 3],
                [wh(j, t, x) for j in range(Nu + 2)]).solve()
            for row in ml.mass[1:Nu + 3]:
                row.apply(w_closure)
            # momentum: resolve, ŵ closure, kill loose ∂_t h, CoV û→q/h
            ml.momentum_x.apply(ExpandSums())
            ml.momentum_x.apply(PullConstants())
            ml.momentum_x.apply(ExtractBrackets(basis, var=zeta))
            ml.momentum_x.apply({Nb: Nu})
            ml.momentum_x.apply(EvaluateSums())
            ml.momentum_x.apply(w_closure)
            ml.momentum_x.apply(ResolveModes(index=kk, modes=range(Nu + 1)))
            ml.momentum_x.apply(ResolveBasis(legendre, var=zeta))
            for k in range(Nu + 1):
                ml.momentum_x[k].apply(h_eq)
                ml.momentum_x[k].apply(Consolidate())
            ml.apply(ChangeOfVariables(
                rf"\hat{{u}}_{ell}", f"q_{ell}", lambda qi: qi / h_l))
            h_eq_q = ml.mass[0].solve_for(d.t(h_l))
            cont = sp.expand(h_eq_q.lhs - h_eq_q.rhs)
            mom = [sp.expand(ml.momentum_x[k].expr) for k in range(Nu + 1)]
            return cont, mom

        layer_eqs = {ell: derive_layer(ell) for ell in range(1, N + 1)}

        # ── Hörnschemeyer closure ──────────────────────────────────────
        ht = sp.Function("h", positive=True)(t, x)
        l_par = [sp.Symbol(f"l_{j}", positive=True) for j in range(1, N)]
        l_all = [*l_par, 1 - sum(l_par)]
        frac = {hl[j]: l_all[j] * ht for j in range(N)}
        q_mod = [[sp.Function(f"q_{ell}", real=True)(k, t, x)
                  for k in range(Nu + 1)] for ell in range(1, N + 1)]

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

        def _trace(ell, side):
            """Modal interface velocity of layer ℓ at ζ=side (0 or 1)."""
            sgn = (lambda i: 1) if side == 1 else (lambda i: (-1) ** i)
            return (sum(sgn(i) * q_mod[ell - 1][i] for i in range(Nu + 1))
                    / (l_all[ell - 1] * ht))

        def _ustar(a):
            """Shared transfer velocity at internal interface α (between
            layers α and α+1): donor by sign of G, or the mean of the two
            one-sided modal traces."""
            below, above = _trace(a, 1), _trace(a + 1, 0)
            if self.interface_velocity == "mean":
                return (below + above) / 2
            return sp.Piecewise((below, G_sol[Gf[a]] >= 0), (above, True))

        m = DModel(coords=(t, x), parameters=values)
        par = {lam_s: m.parameters.lambda_s, nu_s: m.parameters.nu}
        par.update({l_par[j - 1]: getattr(m.parameters, f"l_{j}")
                    for j in range(1, N)})
        m.add_equation("bottom", d.t(b))
        m.add_equation("continuity", sp.expand(glob_c.subs(par)))
        for ell in range(1, N + 1):
            for k in range(Nu + 1):
                mom = layer_eqs[ell][1][k].subs(frac).doit()
                # surgical u* swap in the TRANSFER terms only: layer ℓ sees
                # interface α=ℓ at its top (trace ζ=1, sign +) and α=ℓ−1 at
                # its bottom (trace ζ=0, sign −); φ_k(1)=1, φ_k(0)=(−1)^k.
                for a, side, sgn in ((ell, 1, +1), (ell - 1, 0, -1)):
                    if 1 <= a <= N - 1:
                        phik = 1 if side == 1 else (-1) ** k
                        mom = mom + (sgn * phik
                                     * (_ustar(a) - _trace(ell, side))
                                     * Gf[a] / rho_s)
                mom = sp.expand(mom).subs(G_sol)
                m.add_equation(f"momentum_{ell}_{k}",
                               sp.expand(mom.subs(par).doit()))

        self.derivation = m
        self._bed = b
        self._ht = ht
        self._q_flat = [q for layer in q_mod for q in layer]
        self._G_closed = {str(kk_): sp.simplify(v.subs(par))
                          for kk_, v in G_sol.items()}
        return None

    @property
    def system_model(self) -> SystemModel:
        m = self.derivation
        sm = SystemModel.from_model(
            m, Q=[self._bed, self._ht, *self._q_flat])
        if self.boundary_conditions is not None:
            sm.attach_boundary_conditions(
                self.boundary_conditions, aux_bcs=self.aux_boundary_conditions)
        return sm
