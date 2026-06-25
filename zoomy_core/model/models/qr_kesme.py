"""QRKESME — k–ε Shallow Moment Equations in the q–r variables (k=q², ε=r²).

Positivity-by-construction variant of :class:`KESME`.  Instead of transporting
``k`` and ``ε`` (which can — and did, in the plain KESME spin-up — go negative
and crash the run), we transport their square roots

    sk = √k ,   se = √ε                            (Fe et al. 2009, q=√k, r=√ε)

so that ``k = sk² ≥ 0`` and ``ε = se² ≥ 0`` hold IDENTICALLY, for every modal
combination.  The starting 3-D balance is the SAME k–ε system; the q–r form is
obtained by substituting ``k=sk²``, ``ε=se²`` into the 3-D k and ε transport
equations and dividing by ``2·sk`` / ``2·se`` (so the time derivative stays
linear, ``∂_t(sk²)=2 sk ∂_t sk`` → ``∂_t sk``).  The diffusion term splits
cleanly under the product rule

    (1/2 sk) ∂_z(2 sk · ν_t/σ_k · ∂_z sk)
        = ∂_z(ν_t/σ_k · ∂_z sk)              ← a clean flux divergence (IBP + 0-Neumann)
          + (ν_t/(σ_k sk))(∂_z sk)²          ← an extra cross-term source (Gauss-quad)

so the ONLY hard-coded thing is the 3-D q–r balance (the user's "3d balance as a
start"); everything else rides the SAME moment-projection pipeline as mass,
momentum and KESME.  The cross-term, the production/dissipation prefactors
``1/(2 sk)``, ``1/(2 se)`` and the eddy viscosity ``ν_t = C_μ sk⁴/se²`` are all
rational in ζ → their Galerkin brackets survive ``ExtractBrackets`` and are
resolved by ``GaussQuadrature`` (``quadrature_order>0``), exactly as for the plain
KESME ``ν_t = C_μ k²/ε``.

KESME is left completely untouched — this is a separate class so the working
model is not put at risk.

q–r transport equations (3-D residual form, the ONLY hard-coding):

    ∂_t sk + ∇·(u sk) + ∂_z(w sk)
        − ∂_z(J_k) − (ν_t/σ_k sk)(∂_z sk)²
        − (ν_t/2 sk)(∂_z u)²  +  se²/(2 sk)                       = 0
    ∂_t se + ∇·(u se) + ∂_z(w se)
        − ∂_z(J_e) − (ν_t/σ_e se)(∂_z se)²
        − (C_1 C_μ sk²/2 se)(∂_z u)²  +  C_2 se³/(2 sk²)          = 0

with  J_k = ν_t/σ_k ∂_z sk,  J_e = ν_t/σ_e ∂_z se,  ν_t = C_μ sk⁴/se².
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.derivation import (
    Model as DModel, PDETransformation, Simplify, ResolveIntegral, Basis,
    Consolidate, ExpandSums, EvaluateSums, PullConstants, ExtractBrackets,
    ResolveModes, ResolveBasis, GaussQuadrature, InvertMassMatrix,
    DeferQuadrature, ResolveNumQuad,
    SolveLinearSystem, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound, test_index,
)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.closures import (
    QRViscosity, RoughWall, StressFree)

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class QRKESME(SME):
    """k–ε Shallow Moment Equations in the q–r variables (see module docstring).

    Transports ``sk = √k`` and ``se = √ε`` at modal order ``turbulence_level``
    (N_k), independent of the velocity order ``level`` (N_u).  Default closures
    ``[QRViscosity(), RoughWall(), StressFree()]``: the eddy-viscosity bulk stress
    ``τ = ρ(ν + C_μ sk(ζ)⁴/se(ζ)²) ∂_z u`` (the k–ε feedback into the momentum,
    positive by construction), turbulent rough-wall bed, stress-free surface.
    Build with ``quadrature_order>0`` (default 6): ν_t(ζ) is rational."""

    _cacheable_derivation = True        # derive_model returns m; byproducts on m

    turbulence_level = param.Integer(default=1, bounds=(0, None), doc=(
        "Modal order N_k of the transported sk=√k, se=√ε fields, INDEPENDENT of "
        "the velocity order ``level`` (N_u)."))

    def __init__(self, **params):
        params.setdefault("closures",
                          [QRViscosity(), RoughWall(), StressFree()])
        params.setdefault("quadrature_order", 6)
        super().__init__(**params)

    def derive_model(self):
        """COPY of SME.derive_model with the sk=√k, se=√ε transport equations
        added and projected through the same pipeline (see module docstring)."""
        Nu = int(self.level)
        Nk = int(self.turbulence_level)
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0, "e_x": 0.0,
                  "C_mu": 0.09, "C_1": 1.44, "C_2": 1.92,
                  "sigma_k": 1.0, "sigma_e": 1.3,
                  # realizability floors (model-level, BEFORE GaussQuadrature):
                  # ε → ε+eps_min, k → k+k_min in every singular denominator, so
                  # quadrature integrates smooth integrands and the modal source
                  # has no near-zero denominators (the low-turbulence singularity
                  # cannot be guarded post-quadrature — the denominators are then
                  # high-degree polynomials no fixed floor can match).
                  "eps_min": 0.0, "k_min": 0.0,
                  # wall-function bed BC (weak/penalty) on the q-r variables:
                  # sk(0) → √(u_*²/√C_μ),  se(0) → √(u_*³/(κ z_p)); kappa & z_p
                  # the log-law constant & wall reference height, eps_wall_penalty
                  # the relaxation rate.
                  "kappa": 0.41, "z_p": 0.05, "eps_wall_penalty": 50.0}
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})
        from zoomy_core.model.models.equations import (
            Mass, Momentum, moment_scaling, small_slope_scaling)
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_stress_closures

        dim = int(self.dimension)
        coords = (t, x, z) if dim == 2 else (t, x, y, z)
        horiz = (x,) if dim == 2 else (x, y)
        HAXES = ("x",) if dim == 2 else ("x", "y")
        HNAME = {x: "u", y: "v"}; DERIV = {x: d.x, y: d.y}
        QNAME = ["q"] if dim == 2 else ["q_x", "q_y"]
        SHAT = [r"\hat{u}"] if dim == 2 else [r"\hat{u}", r"\hat{v}"]

        m = DModel(coords=coords, parameters=values)
        g, rho = m.parameters.g, m.parameters.rho
        h = sp.Function("h", positive=True)(t, *horiz)
        b = sp.Function("b", real=True)(t, *horiz)
        # hydrostatic pressure flux (g·h²/2): recomputed in the inherited
        # SME.system_model from m, not stashed on self.

        # 1 — full system from blueprints (mass, momentum, moment_scaling)
        m.declare_state(h)
        m.add_equation("bottom", d.t(b))
        m.add_equation(Mass(m))
        mom = Momentum(m); m.add_equation(mom)
        moment_scaling(m, mom)
        uvel, w, p = mom.uvel, mom.w, mom.p

        # 1b — ADD the q-r transport equations (the only structural addition).
        #      The 3-D k-ε system with k=sk², ε=se² substituted and divided by
        #      2 sk / 2 se.  J_k, J_e are the turbulent fluxes declared like the
        #      momentum stress; the diffusion's product-rule cross terms and the
        #      1/(2 sk), 1/(2 se) production/dissipation prefactors are sources.
        C_mu = m.parameters.C_mu
        C_1, C_2 = m.parameters.C_1, m.parameters.C_2
        sig_k, sig_e = m.parameters.sigma_k, m.parameters.sigma_e
        sk = sp.Function("sk", real=True)(*coords)       # sk = √k
        se = sp.Function("se", real=True)(*coords)       # se = √ε
        Jk = sp.Function("J_k", real=True)(*coords)
        Je = sp.Function("J_e", real=True)(*coords)
        m.declare_state(sk, se)
        m.declare_closure(Jk, Je)
        # realizability floors: ε=se²→se²+eps_min, k=sk²→sk²+k_min in EVERY
        # singular denominator (sign-preserving for odd powers, 1/se→se/(se²+ε_min)).
        # Done here, on the smooth 3-D balance, so GaussQuadrature integrates a
        # non-singular integrand — the only place the low-turbulence singularity
        # can be guarded cleanly (post-quadrature the denominators are high-degree).
        e_min, kk_min = m.parameters.eps_min, m.parameters.k_min
        inv_se2 = 1 / (se ** 2 + e_min)                  # 1/se²  (ν_t)
        inv_se = se / (se ** 2 + e_min)                  # 1/se   (sign-preserving)
        inv_sk2 = 1 / (sk ** 2 + kk_min)                 # 1/sk²
        inv_sk = sk / (sk ** 2 + kk_min)                 # 1/sk
        nu_t = C_mu * sk ** 4 * inv_se2                  # = C_μ k²/ε, rational in ζ
        shear = d.z(uvel[0]) ** 2                        # (∂_z u)²  (1-horiz shear)
        adv_k = sum(DERIV[xd](uvel[i] * sk) for i, xd in enumerate(horiz)) + d.z(w * sk)
        adv_e = sum(DERIV[xd](uvel[i] * se) for i, xd in enumerate(horiz)) + d.z(w * se)
        # sk-equation: ∂_t sk + adv − ∂_z J_k − cross_k − prod_k + se²/(2 sk)
        cross_k = nu_t * inv_sk / sig_k * d.z(sk) ** 2
        prod_k = nu_t * inv_sk / 2 * shear
        m.add_equation("sk", d.t(sk) + adv_k - d.z(Jk) - cross_k - prod_k
                       + se ** 2 * inv_sk / 2)
        # se-equation: ∂_t se + adv − ∂_z J_e − cross_e − prod_e + C_2 se³/(2 sk²)
        cross_e = nu_t * inv_se / sig_e * d.z(se) ** 2
        prod_e = C_1 * C_mu * sk ** 2 * inv_se / 2 * shear
        m.add_equation("se", d.t(se) + adv_e - d.z(Je) - cross_e - prod_e
                       + C_2 * se ** 3 * inv_sk2 / 2)

        def _kbc(interface):
            kw = dict(w=w, u=uvel[0], interface=interface)
            if dim == 3:
                kw["v"] = uvel[1]
            return KinematicBC(**kw)
        m.add_equation("kbc_top", _kbc(b + h))
        m.add_equation("kbc_bot", _kbc(b))

        # 2 — hydrostatic pressure (momentum.z only; sk, se untouched)
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

        # 3 — σ-map the whole model (sk, se ride along automatically)
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))

        basis = Basis(symbol="phi", weight="c"); c = basis.weight
        l = test_index(); phi_l = basis.phi(l, zeta)
        legendre = Legendre_shifted(level=max(Nu + 2, Nk))

        # 4 — project the MASS balance + KBCs
        m.mass.apply(Multiply(h)); m.mass.apply(Multiply(c(zeta) * phi_l))
        m.mass.apply(ProductRule(variables=[zeta]))
        m.mass.apply(Integrate(zeta, bounds=(0, 1))); m.mass.apply(ResolveIntegral())
        m.mass.apply(m.kbc_bot); m.mass.apply(m.kbc_top)
        m.mass.apply({sp.Derivative(b, t): 0}); m.mass.apply(Simplify())

        # 5 — project each horizontal momentum, then close the stress
        def _state(at, *, alias=None, btag=None):
            return ClosureState(m.functions, params=m.parameters, h=h, x=x,
                                zeta=zeta, at=at, alias=alias,
                                boundary_tag=btag, horiz=list(horiz))
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            mxi.apply(Multiply(h)); mxi.apply(Multiply(c(zeta) * phi_l))
            mxi.apply(ProductRule(variables=[zeta]))
            mxi.apply(Integrate(zeta, bounds=(0, 1))); mxi.apply(ResolveIntegral())
            mxi.apply(m.kbc_bot); mxi.apply(m.kbc_top); mxi.apply({sp.Derivative(b, t): 0})
        tau_h = {"x": m.functions.tau_xz}
        if dim == 3:
            tau_h["y"] = m.functions.tau_yz
        axes = [{"mx": getattr(m.momentum, ax), "tau": tau_h[ax],
                 "velname": HNAME[xd]} for ax, xd in zip(HAXES, horiz)]
        has_bulk = apply_stress_closures(self.closures, m, axes, _state, list(horiz))
        for ax in HAXES:
            getattr(m.momentum, ax).apply(Simplify())
        if bool(self.small_slope):
            small_slope_scaling(m)

        # 5b — project the sk, se equations exactly like the momentum, then close
        #      the turbulent fluxes J (bulk ν_t/σ ∂_z·, zero-Neumann at bed+surf)
        for nm in ("sk", "se"):
            eq = getattr(m, nm)
            eq.apply(Multiply(h)); eq.apply(Multiply(c(zeta) * phi_l))
            eq.apply(ProductRule(variables=[zeta]))
            eq.apply(Integrate(zeta, bounds=(0, 1))); eq.apply(ResolveIntegral())
            eq.apply(m.kbc_bot); eq.apply(m.kbc_top); eq.apply({sp.Derivative(b, t): 0})
        Jk_f, Je_f = m.functions.J_k, m.functions.J_e
        dz = lambda e: sp.Derivative(e, zeta) / h          # σ-aware ∂_z
        # zero-Neumann turbulent-flux traces, then the bulk diffusive flux
        m.sk.apply({Jk_f.at(0): 0, Jk_f.at(1): 0})
        m.se.apply({Je_f.at(0): 0, Je_f.at(1): 0})
        m.sk.apply({Jk_f.expr: nu_t / sig_k * dz(sk)})
        m.se.apply({Je_f.expr: nu_t / sig_e * dz(se)})
        m.sk.apply(Simplify()); m.se.apply(Simplify())

        # 6 — separation of variables: u_i → û (N_u), w → ŵ (N_u+1),
        #     sk → sk̂ (N_k), se → sê (N_k)
        coeff_heads = [sp.Function(nm, real=True) for nm in SHAT]
        wh = sp.Function(r"\hat{w}", real=True)
        skhat = sp.Function(r"\hat{sk}", real=True)
        sehat = sp.Function(r"\hat{se}", real=True)
        reset_modal_indices(m)
        N_u = modal_bound("N_u")
        N_k = modal_bound("N_k")
        for i in range(len(horiz)):
            m.apply(separation_of_variables(uvel[i], coeff_heads[i](t, *horiz), basis, N_u))
        m.apply(separation_of_variables(w, wh(t, *horiz), basis, N_u + 1))
        m.apply(separation_of_variables(sk, skhat(t, *horiz), basis, N_k))
        m.apply(separation_of_variables(se, sehat(t, *horiz), basis, N_k))
        if not has_bulk:
            for ax, xd in zip(HAXES, horiz):
                txz_i = sp.Function(f"tau_{ax}z", real=True)(*coords)
                signame = r"\hat{\sigma}" if dim == 2 else rf"\hat{{\sigma}}_{HNAME[xd]}"
                m.apply(separation_of_variables(txz_i, sp.Function(signame, real=True)(t, *horiz),
                                                basis, N_u + 1))

        # 7 — mass basis → h-equation (l=0) and the ŵ closure (l=1…N_u+2)
        m.mass.apply(ExpandSums()); m.mass.apply(PullConstants())
        m.mass.apply(ExtractBrackets(basis, var=zeta)); m.mass.apply({N_u: Nu})
        m.mass.apply(EvaluateSums())
        m.mass.apply(ResolveModes(index=l, modes=range(Nu + 3)))
        m.mass.apply(ResolveBasis(legendre, var=zeta))
        h_eq = m.mass[0].solve_for(d.t(h))
        for row in m.mass[1:Nu + 3]:
            row.apply(h_eq)
        w_closure = SolveLinearSystem(
            m.mass[1:Nu + 3], [wh(j, t, *horiz) for j in range(Nu + 2)]).solve()
        for row in m.mass[1:Nu + 3]:
            row.apply(w_closure)

        # 8 — momentum basis resolution (ŵ closure substituted, GaussQuad)
        qo = int(self.quadrature_order)
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            mxi.apply(ExpandSums()); mxi.apply(PullConstants())
            mxi.apply(ExtractBrackets(basis, var=zeta))
            mxi.apply({N_u: Nu}); mxi.apply({N_k: Nk})   # eddy-visc stress carries sk,se modes
            mxi.apply(EvaluateSums()); mxi.apply(w_closure)
            mxi.apply(ResolveModes(index=l, modes=range(Nu + 1)))
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            if qo > 0:
                # DEFER quadrature BEFORE ResolveBasis: hide the rational closure
                # ∫ as an opaque ⟨…⟩^N atom while φ is STILL OPAQUE, so the basis
                # is substituted only at the Gauss nodes (step 9c) and the
                # rational never exists in concrete-basis symbolic form.  The
                # pure-ζ polynomial brackets are NOT wrapped → ResolveBasis closes
                # them analytically right after.
                mxi.apply(DeferQuadrature(var=zeta))
            mxi.apply(ResolveBasis(legendre, var=zeta))

        # 8b — sk, se basis resolution (own bound N_k, ŵ closure).  The rational
        # ν_t and the q-r cross/source terms survive ExtractBrackets and are
        # evaluated by GaussQuadrature, exactly like a non-polynomial closure.
        # NOTE: ResolveModes promotes m.sk / m.se IN PLACE → re-fetch in a SECOND
        # loop before ResolveBasis/GaussQuadrature (else they hit a stale
        # reference and the φ/weight inside the brackets stay unresolved).
        for nm in ("sk", "se"):
            eq = getattr(m, nm)
            eq.apply(ExpandSums()); eq.apply(PullConstants())
            eq.apply(ExtractBrackets(basis, var=zeta))
            eq.apply({N_u: Nu}); eq.apply({N_k: Nk})
            eq.apply(EvaluateSums()); eq.apply(w_closure)
            eq.apply(ResolveModes(index=l, modes=range(Nk + 1)))
        for nm in ("sk", "se"):
            eq = getattr(m, nm)
            if qo > 0:
                eq.apply(DeferQuadrature(var=zeta))   # ⟨…⟩^N (see momentum note)
            eq.apply(ResolveBasis(legendre, var=zeta))

        # 9 — kill ∂_t h, consolidate, conservative CoV û_d → q_d/h, and the
        #     turbulence moments SK_i = h·sk̂_i, SE_i = h·sê_i (like q = h·û)
        for ax in HAXES:
            mxi = getattr(m.momentum, ax)
            for kk in range(Nu + 1):
                mxi[kk].apply(h_eq); mxi[kk].apply(Consolidate())
        for nm in ("sk", "se"):
            eq = getattr(m, nm)
            for kk in range(Nk + 1):
                eq[kk].apply(h_eq); eq[kk].apply(Consolidate())
        for nm, qn in zip(SHAT, QNAME):
            m.apply(ChangeOfVariables(nm, qn, lambda q_i: q_i / h))
        m.apply(ChangeOfVariables(r"\hat{sk}", "SK", lambda Si: Si / h))
        m.apply(ChangeOfVariables(r"\hat{se}", "SE", lambda Si: Si / h))
        m.apply(InvertMassMatrix())

        # 9b — wall-function bed BCs (weak/penalty) on the q-r variables, pinned
        # to the LIVE rough-wall friction velocity u_*² = C_f·U_p² (REQ-42; see
        # block below — same bed law as the momentum closure, valid for both
        # slope- and non-slope-driven flow).  φ_i(ζ=0)=P_i(−1)=(−1)^i, so the bed
        # traces are sk(0)=Σ(SK_i/h)(−1)^i, se(0)=Σ(SE_i/h)(−1)^i.  Drive
        #   sk(0) → √(u_*²/√C_μ)        (k-wall   k=u_*²/√C_μ, sk=√k)
        #   se(0) → √(u_*³/(κ z_p))     (ε-wall   ε=u_*³/(κ z_p), se=√ε)
        # via τ_p·(−1)^l·(trace − wall) on mode l.  Both wall targets are
        # POSITIVE (u_*²=C_f·U_p²≥0) so no √(negative) — and with sk,se the state,
        # k=sk²≥0 / ε=se²≥0 hold for the whole spin-up (the q-r raison d'être).
        SK_head = m.functions.SK.head
        SE_head = m.functions.SE.head
        kap, z_p = m.parameters.kappa, m.parameters.z_p
        tau_p = m.parameters.eps_wall_penalty
        # ── Live rough-wall friction velocity (REQ-42) ───────────────────────
        # The wall-function friction velocity must be the LIVE bed friction
        # velocity from the bed law (cf. REQ-12), NOT the slope value g*h*e_x.
        # g*h*e_x is ZERO for any non-slope-driven flow (dam break, sloshing,
        # tide, ...), so the wall BC then pins the bed turbulence target to zero
        # and ACTIVELY SUPPRESSES the turbulence the flow should produce — the
        # k-buildup never happens.  Replace with u_*^2 = C_f * U_p^2, the same
        # rough-wall law the momentum bed closure uses:
        #   C_f = (kappa / ln(z_p/z0))^2,  z0 = k_s/30,  U_p = u(zeta_p).
        # Recovers the old behaviour in a uniform slope-driven channel (there
        # C_f*U_p^2 == g h e_x at equilibrium) but now tracks the LOCAL flow.
        _qh_wall = [getattr(m.functions, qn).head for qn in QNAME]
        _zeta_p = z_p / h
        _Cf_wall = (kap / sp.log(z_p / (m.parameters.k_s / 30))) ** 2
        _Up2 = sum(
            (sum((qh(i, t, *horiz) / h) * sp.legendre(i, 2 * _zeta_p - 1)
                 for i in range(Nu + 1))) ** 2
            for qh in _qh_wall)
        ustar2 = _Cf_wall * _Up2
        sk_wall = sp.sqrt(ustar2 / sp.sqrt(C_mu))
        se_wall = sp.sqrt(ustar2 ** sp.Rational(3, 2) / (kap * z_p))
        sk0 = sum((SK_head(i, t, *horiz) / h) * (-1) ** i for i in range(Nk + 1))
        se0 = sum((SE_head(i, t, *horiz) / h) * (-1) ** i for i in range(Nk + 1))
        for l in range(Nk + 1):
            m.sk[l].expr = (sp.sympify(m.sk[l].expr)
                            + tau_p * (-1) ** l * (sk0 - sk_wall))
            m.se[l].expr = (sp.sympify(m.se[l].expr)
                            + tau_p * (-1) ** l * (se0 - se_wall))

        # 9c — resolve the deferred ⟨…⟩^N numerical brackets, the VERY LAST step
        # before extraction.  The rational closure integrals rode the whole
        # post-projection pipeline (Consolidate/CoV/InvertMass) as opaque atoms;
        # only now are they expanded — ONCE, numerically — into their
        # Gauss–Legendre node sums.  Their bodies still carry OPAQUE φ (the bracket
        # was formed before ResolveBasis), so ``basis=`` resolves the basis at the
        # nodes — the only place the rational ever touches the concrete polynomials.
        if qo > 0:
            for ax in HAXES:
                getattr(m.momentum, ax).apply(
                    ResolveNumQuad(var=zeta, order=qo, basis=legendre))
            for nm in ("sk", "se"):
                getattr(m, nm).apply(
                    ResolveNumQuad(var=zeta, order=qo, basis=legendre))

        # 10 — vertical reconstruction → interpolate (velocity rows, as SME)
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

        # 11 — lateral wall BC: mirror moments
        for qh in q_heads:
            for i in range(Nu + 1):
                m.register_group("boundary:wall", qh(i, t, *horiz), -qh(i, t, *horiz))

        # 12 — WB reconstruction rows
        m.reconstruction_rows = {h: b + h}
        for qh in q_heads:
            m.reconstruction_rows.update(
                {qh(i, t, *horiz): qh(i, t, *horiz) / h for i in range(Nu + 1)})

        # 13 — project rows (inverse of interpolate)
        P3 = {f: sp.Symbol(f"P3_{f}", real=True) for f in ("b", "h")}
        m.project_rows = {b: P3["b"], h: P3["h"]}
        for xd, qh in zip(horiz, q_heads):
            P3vel = sp.Function(f"P3_{HNAME[xd]}", real=True)(zeta)
            m.project_rows.update({
                qh(i, t, *horiz): (2 * i + 1) * P3["h"]
                * sp.Integral(P3vel * sp.legendre(i, 2 * zeta - 1), (zeta, 0, 1))
                for i in range(Nu + 1)})

        return m

    @property
    def system_model(self):
        """SME.system_model + flag the sk, se moments as positivity-constrained
        so ``NumericalSystemModel(regularization.positivity_floor>0)`` floors the
        remaining singular source dependence (ν_t=C_μ sk⁴/se², the 1/sk, 1/se,
        1/sk² source prefactors).  k=sk²≥0, ε=se²≥0 hold by construction; the floor
        only guards the divisions when se (=√ε) passes near zero."""
        sm = super().system_model
        sm.positive_state = [s for s in sm.state
                             if str(s).startswith(("SK_", "SE_"))]
        return sm

    def _register_hswme_spectrum(self, sm):
        """HSWME spectrum EXCLUDING sk, se (advective, in-cone — they never set
        the CFL bound): take the plain ``SME(level=N_u)`` hyperbolic spectrum for
        the b, h, q block and pad the sk, se slots with ``n·u_m`` (avoids the flaky
        symbolic eigensolve on the rational ν_t entries)."""
        n_eq = sm.n_equations
        twin = SME(level=int(self.level), dimension=int(self.dimension),
                   small_slope=bool(self.small_slope),
                   parameters=dict(self.parameter_values.items())).system_model
        if twin.eigenvalues is None:
            sm.eigenvalues = None
            return
        lams = [twin.eigenvalues[i] for i in range(twin.n_equations)]
        normal = list(sm.normal.values())
        by = {str(s_): s_ for s_ in sm.state}
        names = ["q_0"] if int(self.dimension) == 2 else ["q_x_0", "q_y_0"]
        u_m = sum(normal[k] * by[nm] for k, nm in enumerate(names)) / by["h"]
        lams += [u_m] * (n_eq - len(lams))         # sk, se slots: advective, in-cone
        sm.eigenvalues = sp.Matrix(n_eq, 1, lams)


__all__ = ["QRKESME"]
