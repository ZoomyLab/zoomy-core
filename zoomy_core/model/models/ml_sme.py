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
    ResolveModes, ResolveBasis, InvertMassMatrix, SolveLinearSystem, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.systemmodel import SystemModel

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class MLSME(BaseModel):
    """Multilayer SME: ``n_layers`` moving-with-the-surface layers, each a
    moment column of order ``level``."""

    _finalize_lazy = True
    _cacheable_derivation = True        # derive_model returns m; byproducts on m
    n_layers = param.Integer(default=2, bounds=(2, None))
    level = param.Integer(default=1, bounds=(0, None))
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension incl. vertical: 2 → (t,x,z), one horizontal "
        "(q_ℓ_i); 3 → (t,x,y,z), two horizontal (q_x_ℓ_i, q_y_ℓ_i).  Each layer "
        "is derived dimension-agnostically; the interface mass-flux closure and "
        "the transfer velocity u* generalise per horizontal direction."))
    closures = param.List(default=[], doc=(
        "Composable Closure pieces (closures.py): stress (Newtonian/RoughWall/…) "
        "AND the interface transfer scheme (MeanInterface/UpwindInterface).  "
        "Empty leaves tau UNCLOSED (modal moments stay free)."))
    interface_velocity = param.Selector(
        default="upwind", objects=["upwind", "mean"],
        doc="DEPRECATED - use closures=[UpwindInterface()/MeanInterface()].  "
            "Shared transfer velocity u* at internal interfaces.")
    project_nz = param.Integer(default=33, bounds=(2, None), doc=(
        "FIXED per-layer vertical sample count for the Integral-FREE "
        "``project_from_3d`` Galerkin reduction (trapezoid, O(N_z^-2)); see "
        "SME.project_nz.  Each layer reduces its own sub-column to moments."))

    def derive_model(self):
        N = int(self.n_layers)
        Nu = int(self.level)
        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}; CN = {x: "x", y: "y"}
        # per-direction conserved-moment / SoV family names for a layer.  1
        # horizontal keeps q_ℓ / û_ℓ (byte-identical); 2 horizontal → q_x_ℓ,
        # q_y_ℓ / û_ℓ, v̂_ℓ.
        def qname(xd, ell):
            return f"q_{ell}" if dim == 2 else f"q_{CN[xd]}_{ell}"
        def shat(xd, ell):
            return (rf"\hat{{u}}_{ell}" if dim == 2
                    else rf"\hat{{{HNAME[xd]}}}_{ell}")
        def sname(xd, ell):
            return f"tau_{ell}" if dim == 2 else f"tau_{CN[xd]}z_{ell}"
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
        lam_s, nu_s = sp.symbols("lambda_s nu", positive=True)
        rho_s = sp.Symbol("rho", positive=True)

        from zoomy_core.model.models.equations import Mass, small_slope_scaling
        from types import SimpleNamespace
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_layer_stress_closures

        def derive_layer(ell):
            z_bot, z_top, h_l = ifaces[ell - 1], ifaces[ell], hl[ell - 1]
            G_bot, G_top = Gf[ell - 1], Gf[ell]
            ml = DModel(coords=coords, parameters={"g": 9.81, "rho": 1.0})
            gl, rl = ml.parameters.g, ml.parameters.rho
            # mass balance from the (dimension-agnostic) blueprint — not a
            # hand-written ∂_x u + ∂_z w; it gives Σ_d ∂_d u_ℓ_d + ∂_z w_ℓ.
            ml.add_equation(Mass(ml, suffix=f"_{ell}"))
            uvel = [sp.Function(HNAME[xd] + f"_{ell}", real=True)(*coords)
                    for xd in horiz]
            w = sp.Function(f"w_{ell}", real=True)(*coords)
            tau = {xd: sp.Function(sname(xd, ell), real=True)(*coords)
                   for xd in horiz}
            MOM = [f"momentum_{CN[xd]}" for xd in horiz]
            for i, xd in enumerate(horiz):
                adv = sum(DERIV[xe](uvel[i] * uvel[j]) for j, xe in enumerate(horiz))
                ml.add_equation(
                    f"momentum_{CN[xd]}",
                    d.t(uvel[i]) + adv + d.z(uvel[i] * w)
                    + gl * DERIV[xd](b + H) - d.z(tau[xd]) / rl)

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
            for nm in ["mass"] + MOM:
                getattr(ml, nm).apply(Multiply(h_l))
                getattr(ml, nm).apply(Multiply(c(zeta) * phi_k))
                getattr(ml, nm).apply(ProductRule(variables=[zeta]))
                getattr(ml, nm).apply(Integrate(zeta, bounds=(0, 1)))
                getattr(ml, nm).apply(ResolveIntegral())
                getattr(ml, nm).apply(ml.kbc_bot)
                getattr(ml, nm).apply(ml.kbc_top)
                getattr(ml, nm).apply({sp.Derivative(b, t): 0})
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
            small_slope_scaling(ml)
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
            # mass: k=0 h-eq + ŵ closure rows (2-D: couples û_ℓ, v̂_ℓ)
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
                [wh(j, t, *horiz) for j in range(Nu + 2)]).solve()
            for row in ml.mass[1:Nu + 3]:
                row.apply(w_closure)
            for xd in horiz:
                mxi = getattr(ml, f"momentum_{CN[xd]}")
                mxi.apply(ExpandSums()); mxi.apply(PullConstants())
                mxi.apply(ExtractBrackets(basis, var=zeta))
                mxi.apply({Nb: Nu}); mxi.apply(EvaluateSums())
                mxi.apply(w_closure)
                mxi.apply(ResolveModes(index=kk, modes=range(Nu + 1)))
            for xd in horiz:
                getattr(ml, f"momentum_{CN[xd]}").apply(ResolveBasis(legendre, var=zeta))
            for xd in horiz:
                mxi = getattr(ml, f"momentum_{CN[xd]}")
                for kmode in range(Nu + 1):
                    mxi[kmode].apply(h_eq)
                    mxi[kmode].apply(Consolidate())
            for i, xd in enumerate(horiz):
                ml.apply(ChangeOfVariables(shat(xd, ell), qname(xd, ell),
                                           lambda qi: qi / h_l))
            ml.apply(InvertMassMatrix())
            h_eq_q = ml.mass[0].solve_for(d.t(h_l))
            cont = sp.expand(h_eq_q.lhs - h_eq_q.rhs)
            mom = {xd: [sp.expand(getattr(ml, f"momentum_{CN[xd]}")[kmode].expr)
                        for kmode in range(Nu + 1)] for xd in horiz}
            return cont, mom

        layer_eqs = {ell: derive_layer(ell) for ell in range(1, N + 1)}

        # ── Hörnschemeyer interface closure (∇·-based, dimension-agnostic) ──
        ht = sp.Function("h", positive=True)(t, *horiz)
        l_par = [sp.Symbol(f"l_{j}", positive=True) for j in range(1, N)]
        l_all = [*l_par, 1 - sum(l_par)]
        frac = {hl[j]: l_all[j] * ht for j in range(N)}
        q_mod = {ell: {xd: [sp.Function(qname(xd, ell), real=True)(k, t, *horiz)
                            for k in range(Nu + 1)] for xd in horiz}
                 for ell in range(1, N + 1)}

        glob_c = sp.expand(
            sum(layer_eqs[ell][0] for ell in range(1, N + 1)).subs(frac).doit())
        dth_glob = sp.solve(glob_c, sp.Derivative(ht, t))[0]
        G_sol = {}
        for a in range(1, N):
            part = sp.expand(
                sum(layer_eqs[ell][0] for ell in range(1, a + 1)).subs(frac).doit())
            part = sp.expand(part.subs(sp.Derivative(ht, t), dth_glob))
            G_sol[Gf[a]] = sp.solve(part, Gf[a])[0]

        # inner (per-layer) basis object — interface traces and the piecewise
        # reconstruction evaluate it at ζ_loc∈{0,1} and at interior nodes, so the
        # bed/surface values φ_k(0)=(−1)^k / φ_k(1)=1 go through the basis object.
        inner_basis = Legendre_shifted(level=Nu + 2)

        def _trace(ell, side, xd):
            """Modal interface velocity of layer ℓ in direction xd at ζ=side."""
            sgn = lambda i: inner_basis.eval(i, side)
            return (sum(sgn(i) * q_mod[ell][xd][i] for i in range(Nu + 1))
                    / (l_all[ell - 1] * ht))

        from zoomy_core.model.models.closures import interface_closure
        iface = interface_closure(self.closures)

        def _ustar(a, xd):
            """Per-direction shared transfer velocity at internal interface α."""
            below, above = _trace(a, 1, xd), _trace(a + 1, 0, xd)
            if iface is not None:
                return iface.expression(below, above, G_sol[Gf[a]])
            if self.interface_velocity == "mean":
                return (below + above) / 2
            return sp.Piecewise((below, G_sol[Gf[a]] >= 0), (above, True))

        # include a vertical z so Model.horizontal = (x[, y]) — a coords=(t,x,y)
        # model would wrongly read y as the vertical and collapse to n_dim=1.
        # (The depth-integrated equations carry no z-derivative.)
        m = DModel(coords=(t, *horiz, z), parameters=values)
        par = {lam_s: m.parameters.lambda_s, nu_s: m.parameters.nu}
        par.update({l_par[j - 1]: getattr(m.parameters, f"l_{j}")
                    for j in range(1, N)})
        m.add_equation("bottom", d.t(b))
        m.add_equation("continuity", sp.expand(glob_c.subs(par)))
        for ell in range(1, N + 1):
            for xd in horiz:
                for k in range(Nu + 1):
                    mom = layer_eqs[ell][1][xd][k].subs(frac).doit()
                    # u* swap in the per-direction TRANSFER trace only
                    for a, side, sgn in ((ell, 1, +1), (ell - 1, 0, -1)):
                        if 1 <= a <= N - 1:
                            phik = inner_basis.eval(k, side)
                            mom = mom + (sgn * phik
                                         * (_ustar(a, xd) - _trace(ell, side, xd))
                                         * Gf[a] / rho_s)
                    mom = sp.expand(mom).subs(G_sol)
                    enm = (f"momentum_{ell}_{k}" if dim == 2
                           else f"momentum_{CN[xd]}_{ell}_{k}")
                    m.add_equation(enm, sp.expand(mom.subs(par).doit()))

        # ── basic operators: interpolate_to_3d + project_from_3d, PIECEWISE over
        # the moving layers, so ML-SME exposes the same canonical operators as
        # SME (postprocessing / coupling consume model.interpolate_to_3d).
        # Layer ℓ spans global ζ ∈ [c_{ℓ-1}, c_ℓ] (c_ℓ = Σ_{j≤ℓ} l_j); within it
        #   u_ℓ(ζ) = Σ_k (q_ℓ_k / h_ℓ)·P_k(2 ζ_loc − 1),  ζ_loc = (ζ−c_{ℓ-1})/l_ℓ,
        # h_ℓ = l_ℓ·ht.  Field order [b, h, u(, v), w, p] (SME convention).
        cum = [sp.S.Zero]
        for lf in l_all:
            cum.append(cum[-1] + lf)

        def _u_piecewise(xd):
            pieces = []
            for ell in range(1, N + 1):
                lf, c0 = l_all[ell - 1], cum[ell - 1]
                zloc = (zeta - c0) / lf
                u_l = sum((q_mod[ell][xd][k] / (lf * ht)) * inner_basis.eval(k, zloc)
                          for k in range(Nu + 1))
                pieces.append((u_l, (zeta <= cum[ell]) if ell < N else True))
            return sp.Piecewise(*pieces)

        interp = {0: b, 1: ht}
        interp[2] = _u_piecewise(x).subs(par)
        if dim == 3:
            interp[3] = _u_piecewise(y).subs(par)
        interp[4] = sp.S.Zero            # TODO: per-layer kinematic w reconstruction
        interp[5] = m.parameters.rho * m.parameters.g * ht * (1 - zeta)
        m.interpolate_rows = interp

        # inverse: q_ℓ_k = h·α_ℓ_k — the Integral-FREE, fixed-node trapezoid
        # Galerkin reduction of each layer's sub-column.  Layer ℓ spans global
        # ζ ∈ [c0, c1]; sample P3_<vel> at N_z LOCAL nodes t∈[0,1] mapped to the
        # global position ζ = c0 + l_ℓ·t.  ``projection_rows`` carries the
        # 1/∫φ_k² normalisation; the physical factor is h·l_ℓ (the dζ = l_ℓ·dt
        # Jacobian times the layer height h·l_ℓ — matching the old
        # (2k+1)·h·∫_{c0}^{c1} … dζ form).
        legendre = Legendre_shifted(level=Nu + 2)
        N_z = int(self.project_nz)
        loc = [float(j) / (N_z - 1) for j in range(N_z)]
        weights = [1.0 / (N_z - 1)] * N_z
        weights[0] *= 0.5; weights[-1] *= 0.5
        P3 = {f: sp.Symbol(f"P3_{f}", real=True) for f in ("b", "h")}
        proj = {b: P3["b"], ht: P3["h"]}
        for xd in horiz:
            P3vel = sp.Function(f"P3_{HNAME[xd]}", real=True)
            for ell in range(1, N + 1):
                lf = l_all[ell - 1].subs(par)
                c0 = cum[ell - 1].subs(par)
                samples = [P3vel(c0 + lf * t) for t in loc]
                rows = legendre.projection_rows(
                    loc, weights, samples,
                    norm=lambda _k, _lf=lf: P3["h"] * _lf)
                for k in range(Nu + 1):
                    proj[q_mod[ell][xd][k]] = rows[k]
        m.project_rows = proj

        m.bed = b
        m.ht = ht
        # state order: layer-major, then direction, then mode (matches the
        # equation registration order above)
        m.q_flat = [q_mod[ell][xd][k]
                    for ell in range(1, N + 1)
                    for xd in horiz for k in range(Nu + 1)]
        m.G_closed = {str(kk_): sp.simplify(v.subs(par))
                      for kk_, v in G_sol.items()}
        return m

    # Built via ``SystemModel.from_model(MLSME(...))`` (REQ-143); see
    # ``zoomy_core.systemmodel.model_builders.build_mlsme``.
    _system_model_kind = "mlsme"
