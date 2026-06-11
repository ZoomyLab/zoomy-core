"""ML-SWE — multilayer shallow water with mass exchange, the declarative
canonical model.

Per-layer derivation (each layer is its own sub-model with its own σ-map
``z = z_{ℓ-1} + h_ℓ ζ``; interfaces closed with
``KinematicBC(mass_flux=G_ℓ)``, bed/surface impermeable) + the
**Hörnschemeyer closure** (Aguillon, Hörnschemeyer & Sainte-Marie,
"Barotropic-Baroclinic Splitting for Multilayer Shallow Water Models with
Exchanges"): the internal-interface mass fluxes are the Lagrange multipliers
of the layer-fraction constraint ``h_ℓ = l_ℓ·h`` (fractions of the
INSTANTANEOUS depth — layers move with the free surface), giving the closed
form ``G_ℓ = ρ·Σ_{j≤ℓ}(l_j·∂_x Q − ∂_x q_j)`` with ``Q = Σ q_j``; the
momentum-transfer interface velocity ``u*_ℓ`` is shared by the adjacent
rows and either UPWINDED by the sign of G (their Eq. 9, the default) or the
arithmetic MEAN (admissible per Audusse et al.).

NOTE the kinematic-closure vacuity result (thesis ml_swe notebook): with
independent layer heights, NO recombination of the layers' own kinematics
determines G — the fraction constraint is the missing statement; this class
implements exactly that.

State: ``[b, h, q_1 … q_N]`` (one total depth; per-layer discharges).
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.basemodel import Model as BaseModel
from zoomy_core.model.derivation import (
    Model as DModel, PDETransformation, Simplify, ResolveIntegral, Basis,
    ExpandSums, EvaluateSums, PullConstants, ExtractBrackets,
    ResolveModes, ResolveBasis, InvertMassMatrix, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.systemmodel import SystemModel

t, x, z = C.t, C.x, C.z
zeta = sp.Symbol("zeta", real=True)


class MLSWE(BaseModel):
    """Multilayer SWE, ``n_layers`` moving-with-the-surface layers."""

    _finalize_lazy = True
    n_layers = param.Integer(default=2, bounds=(2, None))
    material = param.Parameter(default=None, doc=(
        "Stress closure (MaterialModel) injected at the CORE level; "
        "None (default) leaves tau_xz UNCLOSED - its modal moments stay "
        "free functions in the derived system. Use "
        "material=newtonian_navier_slip() for the standard closure."))
    interface_velocity = param.Selector(
        default="upwind", objects=["upwind", "mean"],
        doc="transfer velocity u* at internal interfaces: sign-of-G upwind "
            "(Hörnschemeyer Eq. 9) or arithmetic mean")

    def derive_model(self):
        N = int(self.n_layers)
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0}
        # equal fractions by default; override l_1 … l_{N-1} via parameters=
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

        def derive_layer(ell):
            """Depth-averaged continuity + momentum of layer ℓ (level 0)."""
            z_bot, z_top, h_l = ifaces[ell - 1], ifaces[ell], hl[ell - 1]
            G_bot, G_top = Gf[ell - 1], Gf[ell]
            ml = DModel(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
            gl, rl = ml.parameters.g, ml.parameters.rho
            ul = sp.Function(f"u_{ell}", real=True)(t, x, z)
            wl = sp.Function(f"w_{ell}", real=True)(t, x, z)
            tl = sp.Function(f"tau_{ell}", real=True)(t, x, z)
            ml.Q = [ul, wl]
            ml.add_equation("mass", d.x(ul) + d.z(wl))
            ml.add_equation("momentum_x",
                            d.t(ul) + d.x(ul * ul) + d.z(ul * wl)
                            + gl * d.x(b + H) - d.z(tl) / rl)
            ml.add_equation("kbc_bot", KinematicBC(
                w=wl, u=ul, interface=z_bot, rho=rl,
                mass_flux=(G_bot if G_bot != 0 else None)))
            ml.add_equation("kbc_top", KinematicBC(
                w=wl, u=ul, interface=z_top, rho=rl,
                mass_flux=(G_top if G_top != 0 else None)))
            ml.apply(PDETransformation({z: (zeta, sp.Eq(z, z_bot + h_l * zeta))}))
            basis = Basis(symbol="phi", weight="c"); c = basis.weight
            kk = test_index(); phi_k = basis.phi(kk, zeta)
            legendre = Legendre_shifted(level=1)
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
            # Navier slip at the BED only; interior interfaces inviscid here
            mat = self.material
            if mat is not None:
                from types import SimpleNamespace
                par_ns = SimpleNamespace(rho=rl, nu=nu_s, lambda_s=lam_s)
                top_tr = (mat.surface(uu.at(1), par_ns)
                          if (ell == N and mat.surface is not None) else 0)
                bot_tr = (mat.bottom(uu.at(0), par_ns)
                          if (ell == 1 and mat.bottom is not None) else 0)
                ml.momentum_x.apply({tau.at(1): top_tr, tau.at(0): bot_tr})
                if mat.bulk is not None:
                    ml.momentum_x.apply({tau.expr: mat.bulk(
                        uu.expr, lambda e: sp.Derivative(e, zeta) / h_l,
                        par_ns)})
            ml.momentum_x.apply(Simplify())
            uh = sp.Function(rf"\hat{{u}}_{ell}", real=True)
            reset_modal_indices(ml)
            Nb = modal_bound("N_u")
            ml.apply(separation_of_variables(ul, uh(t, x), basis, Nb))
            if mat is None:
                sh_ = sp.Function(rf"\hat{{\sigma}}_{ell}", real=True)
                ml.apply(separation_of_variables(tl, sh_(t, x), basis, Nb + 1))
            for nm in ("mass", "momentum_x"):
                getattr(ml, nm).apply(ExpandSums())
                getattr(ml, nm).apply(PullConstants())
                getattr(ml, nm).apply(ExtractBrackets(basis, var=zeta))
                getattr(ml, nm).apply({Nb: 0})
                getattr(ml, nm).apply(EvaluateSums())
                getattr(ml, nm).apply(ResolveModes(index=kk, modes=range(1)))
                getattr(ml, nm).apply(ResolveBasis(legendre, var=zeta))
            ml.apply(ChangeOfVariables(
                rf"\hat{{u}}_{ell}", f"q_{ell}", lambda qi: qi / h_l))
            ml.apply(InvertMassMatrix())

            def _clean(e):
                e = e.replace(
                    lambda a: isinstance(a, sp.Integral) and a.function == 0,
                    lambda a: sp.S.Zero)

                def _pull(a):
                    vs = [v for v, _ in a.variable_count]
                    co, rest = a.expr.as_independent(*vs, as_Mul=True)
                    return (co * sp.Derivative(rest, *a.args[1:])
                            if co != 1 else a)
                e = e.replace(lambda a: isinstance(a, sp.Derivative), _pull)
                return sp.expand(e)
            return _clean(ml.mass[0].expr), _clean(ml.momentum_x[0].expr)

        layer_eqs = {ell: derive_layer(ell) for ell in range(1, N + 1)}

        # ── Hörnschemeyer closure ──────────────────────────────────────
        ht = sp.Function("h", positive=True)(t, x)
        l_par = [sp.Symbol(f"l_{j}", positive=True) for j in range(1, N)]
        l_all = [*l_par, 1 - sum(l_par)]
        frac = {hl[j]: l_all[j] * ht for j in range(N)}
        q_l = [sp.Function(f"q_{ell}", real=True)(0, t, x)
               for ell in range(1, N + 1)]
        u_lc = [q_l[j] / (l_all[j] * ht) for j in range(N)]

        glob_c = sp.expand(
            sum(layer_eqs[ell][0] for ell in range(1, N + 1))
            .subs(frac).doit())
        dth_glob = sp.solve(glob_c, sp.Derivative(ht, t))[0]
        # G_α from the partial sums of the constrained continuities
        G_sol = {}
        for a in range(1, N):
            # .doit() BEFORE the ∂_t h substitution — the fraction subs
            # leaves unevaluated Derivative(l_j·h, t) compounds that the
            # bare-∂_t h rule cannot match until expanded.
            part = sp.expand(
                sum(layer_eqs[ell][0] for ell in range(1, a + 1))
                .subs(frac).doit())
            part = sp.expand(part.subs(sp.Derivative(ht, t), dth_glob))
            G_sol[Gf[a]] = sp.solve(part, Gf[a])[0]

        def _ustar(a):
            """Shared transfer velocity at internal interface α."""
            if self.interface_velocity == "mean":
                return (u_lc[a - 1] + u_lc[a]) / 2
            return sp.Piecewise((u_lc[a - 1], G_sol[Gf[a]] >= 0),
                                (u_lc[a], True))

        m = DModel(coords=(t, x), parameters=values)
        par = {lam_s: m.parameters.lambda_s, nu_s: m.parameters.nu}
        par.update({l_par[j - 1]: getattr(m.parameters, f"l_{j}")
                    for j in range(1, N)})
        m.add_equation("bottom", d.t(b))
        m.add_equation("continuity", sp.expand(glob_c.subs(par)))
        for ell in range(1, N + 1):
            mom = layer_eqs[ell][1]
            # each G_α enters this row once, as ±G·u_ℓ(trace)/ρ — swap the
            # layer's own trace for the SHARED u*: G → G·u*/u_ℓ
            for a in (ell - 1, ell):
                if 1 <= a <= N - 1:
                    mom = mom.subs(Gf[a], Gf[a] * _ustar(a) / u_lc[ell - 1])
            mom = mom.subs(frac).subs(G_sol)
            m.add_equation(f"momentum_{ell}", sp.expand(mom.subs(par).doit()))

        self.derivation = m
        self._bed = b
        self._ht = ht
        self._q_l = q_l
        self._G_closed = {str(k): sp.simplify(v.subs(par))
                          for k, v in G_sol.items()}
        return None

    @property
    def system_model(self) -> SystemModel:
        m = self.derivation
        sm = SystemModel.from_model(
            m, Q=[self._bed, self._ht, *self._q_l])
        if self.boundary_conditions is not None:
            sm.attach_boundary_conditions(
                self.boundary_conditions, aux_bcs=self.aux_boundary_conditions)
        return sm
