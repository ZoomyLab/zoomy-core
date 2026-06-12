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

t, x, z = C.t, C.x, C.z
zeta = sp.Symbol("zeta", real=True)


class VAM(BaseModel):
    """Non-hydrostatic vertically-averaged moment equations, truncation
    ``level`` (= ``N_u``; u, w and the non-hydrostatic p share the basis)."""

    _finalize_lazy = True               # declarative path
    level = param.Integer(default=1, bounds=(0, None))
    closures = param.List(default=[], doc=(
        "Composable stress Closure pieces (closures.py); takes precedence over "
        "`material`.  Empty + material=None leaves tau_xz UNCLOSED."))
    material = param.Parameter(default=None, doc=(
        "DEPRECATED - use `closures=[...]`.  Legacy MaterialModel stress "
        "closure; None leaves tau_xz UNCLOSED (modal moments stay free)."))

    def derive_model(self):
        Nu = int(self.level)
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0}
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})
        m = DModel(coords=(t, x, z), parameters=values)
        g, rho = m.parameters.g, m.parameters.rho
        nu, lam = m.parameters.nu, m.parameters.lambda_s
        u = sp.Function("u", real=True)(t, x, z)
        w = sp.Function("w", real=True)(t, x, z)
        p = sp.Function("p", real=True)(t, x, z)     # NON-hydrostatic part
        h = sp.Function("h", positive=True)(t, x)
        b = sp.Function("b", real=True)(t, x)
        txz = sp.Function("tau_xz", real=True)(t, x, z)

        # 1 — full system, hydrostatic pressure already absorbed:
        # x-momentum carries g·∂_x(b+h); z-momentum keeps only ∂_z p.
        m.Q = [h, u, w, p]
        m.add_equation("bottom", d.t(b))
        m.add_equation("mass", d.x(u) + d.z(w))
        m.add_equation("momentum_x",
                       d.t(u) + d.x(u * u) + d.z(u * w) + g * d.x(b + h)
                       + d.x(p) / rho - d.z(txz) / rho)
        m.add_equation("momentum_z",
                       d.t(w) + d.x(u * w) + d.z(w * w) + d.z(p) / rho)
        m.add_equation("kbc_top", KinematicBC(w=w, u=u, interface=b + h))
        m.add_equation("kbc_bot", KinematicBC(w=w, u=u, interface=b))

        # 2 — σ-map:  z = b + h ζ
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))

        basis = Basis(symbol="phi", weight="c"); c = basis.weight
        k = test_index(); phi_k = basis.phi(k, zeta)
        legendre = Legendre_shifted(level=Nu + 2)

        # 3 — Galerkin-project all three balances onto c·φ_k (+ BCs)
        m.mass.apply(Multiply(h))
        m.mass.apply(Multiply(c(zeta) * phi_k))
        m.mass.apply(ProductRule(variables=[zeta]))
        m.mass.apply(Integrate(zeta, bounds=(0, 1)))
        m.mass.apply(ResolveIntegral())
        m.mass.apply(m.kbc_bot); m.mass.apply(m.kbc_top)
        m.mass.apply({sp.Derivative(b, t): 0})
        m.mass.apply(Simplify())

        pp = m.functions.p
        mx = m.momentum_x
        mx.apply(Multiply(h)); mx.apply(Multiply(c(zeta) * phi_k))
        mx.apply(ProductRule(variables=[zeta]))
        mx.apply(Integrate(zeta, bounds=(0, 1)))
        mx.apply(ResolveIntegral())
        mx.apply(m.kbc_bot); mx.apply(m.kbc_top)
        mx.apply({sp.Derivative(b, t): 0})
        mx.apply({pp.at(1): 0})              # dynamic surface BC p(ζ=1)=0
        tau, uu = m.functions.tau_xz, m.functions.u
        # Navier slip: τ(0) = +λ·u_b (same sign as the slip velocity)
        from zoomy_core.model.models.material import ClosureState
        from zoomy_core.model.models.closures import apply_stress_closures
        def _state(at):
            return ClosureState(m.functions, params=m.parameters,
                                h=h, x=x, zeta=zeta, at=at)
        has_bulk = apply_stress_closures(self.closures, self.material, m, mx, tau, _state)
        mx.apply(Simplify())

        mz = m.momentum_z
        mz.apply(Multiply(h)); mz.apply(Multiply(c(zeta) * phi_k))
        mz.apply(ProductRule(variables=[zeta]))
        mz.apply(Integrate(zeta, bounds=(0, 1)))
        mz.apply(ResolveIntegral())
        mz.apply(m.kbc_bot); mz.apply(m.kbc_top)
        mz.apply({sp.Derivative(b, t): 0})
        mz.apply({pp.at(1): 0})              # dynamic surface BC p(ζ=1)=0
        mz.apply(Simplify())

        # 4 — modal ansatz: u ∈ P_Nu;  w, p ∈ P_{Nu+1} (Escalante spaces —
        # a UNIFORM truncation leaves the pressure space inconsistent with
        # the surface/bottom closures and drifts secularly at runtime)
        uh = sp.Function(r"\hat{u}", real=True)
        wh = sp.Function(r"\hat{w}", real=True)
        ph = sp.Function(r"\hat{p}", real=True)
        reset_modal_indices(m)
        N_u = modal_bound("N_u")
        m.apply(separation_of_variables(u, uh(t, x), basis, N_u))
        m.apply(separation_of_variables(w, wh(t, x), basis, N_u + 1))
        m.apply(separation_of_variables(p, ph(t, x), basis, N_u + 1))
        if not has_bulk:
            # UNCLOSED bulk stress: expand tau modally (also covers a
            # boundary-only closure like RoughWall — bulk left free)
            sh_ = sp.Function(r"\hat{\sigma}", real=True)
            m.apply(separation_of_variables(txz, sh_(t, x), basis, N_u + 1))

        # 5 — resolve: mass k=0…Nu+1 (h-eq + Nu+1 divergence constraints,
        # one per pressure mode); momenta k=0…Nu.  NB: re-fetch the family
        # by name on EVERY apply — ResolveModes promotes it in place and a
        # captured handle goes stale (the later ResolveBasis would hit a
        # dead object, leaving brackets unresolved).
        for nm, modes in (("mass", range(Nu + 2)),
                          ("momentum_x", range(Nu + 1)),
                          ("momentum_z", range(Nu + 1))):
            getattr(m, nm).apply(ExpandSums())
            getattr(m, nm).apply(PullConstants())
            getattr(m, nm).apply(ExtractBrackets(basis, var=zeta))
            getattr(m, nm).apply({N_u: Nu})
            getattr(m, nm).apply(EvaluateSums())
            getattr(m, nm).apply(ResolveModes(index=k, modes=modes))
            getattr(m, nm).apply(ResolveBasis(legendre, var=zeta))

        # 5b — MODAL closures for the top w/p modes (Escalante eq 6):
        # surface dynamic p(1)=0   → p̂_{Nu+1} = −Σ_{j≤Nu} p̂_j   (φ_j(1)=1)
        # bottom kinematic w(0)=u(0)·∂_x b
        #                  → ŵ_{Nu+1} = (−1)^{Nu+1}(u(0)∂_x b − Σ_{j≤Nu}(−1)^j ŵ_j)
        top = Nu + 1
        u_at0 = sum((-1) ** j * uh(j, t, x) for j in range(Nu + 1))
        w_top = (-1) ** top * (u_at0 * sp.Derivative(b, x)
                               - sum((-1) ** j * wh(j, t, x) for j in range(top)))
        p_top = -sum(ph(j, t, x) for j in range(top))
        m.apply({wh(top, t, x): w_top})
        m.apply({ph(top, t, x): p_top})

        # 6 — conservative CoV (û→q/h, ŵ→r/h, p̂→P) + h-eq substitution
        m.apply(ChangeOfVariables(r"\hat{u}", "q", lambda qi: qi / h))
        m.apply(ChangeOfVariables(r"\hat{w}", "r", lambda ri: ri / h))
        m.apply(ChangeOfVariables(r"\hat{p}", "P", lambda pi: pi))
        h_eq = m.mass[0].solve_for(d.t(h))
        for nm in ("momentum_x", "momentum_z"):
            for kk in range(Nu + 1):
                getattr(m, nm)[kk].apply(h_eq)
                getattr(m, nm)[kk].apply(Consolidate())
        for kk in range(1, Nu + 2):
            m.mass[kk].apply(h_eq)
            m.mass[kk].apply(Consolidate())
        # AFTER the stray dt-h substitutions (op docstring): unit dt coeffs
        m.apply(InvertMassMatrix())

        # 6b — resolve the advective σ-mass flux ω̃ (Escalante eq(3)→(4)):
        # the Galerkin rows keep ω̃ in DEFINITION form; the paper substitutes
        # the σ-mass-integrated CLOSED form ω̃ = −Σ_j ∂_x(hû_j)∫₀^ζφ_j (h-eq
        # applied) BEFORE projecting.  The two differ by {h-eq, divergence-
        # constraint} combinations — equivalent ON the constraint manifold,
        # but the predictor runs OFF it and the published hyperbolicity
        # analysis holds for the closed form.  Exact correction
        #   Δ_k = −∫₀¹ (ω̃_closed − ω̃_def)·field·φ_k′ dζ / μ_k
        # per momentum row (field = ũ for x, w̃ for z); k=0 rows: Δ=0.
        phis = [sp.legendre(j, 2 * zeta - 1) for j in range(Nu + 2)]
        mus = [sp.Rational(1, 2 * j + 1) for j in range(Nu + 2)]
        s_ = sp.Symbol("_s_omega")
        qf = [m.functions.q.head(j, t, x) for j in range(Nu + 1)]
        rf = [m.functions.r.head(j, t, x) for j in range(Nu + 1)]
        ut_m = sum(qf[j] / h * phis[j] for j in range(Nu + 1))
        w_top_cons = ((-1) ** (Nu + 1)) * (
            sum((-1) ** j * qf[j] / h for j in range(Nu + 1)) * sp.Derivative(b, x)
            - sum((-1) ** j * rf[j] / h for j in range(Nu + 1)))
        wt_m = (sum(rf[j] / h * phis[j] for j in range(Nu + 1))
                + w_top_cons * phis[Nu + 1])
        omega_def = (wt_m - zeta * (-sp.Derivative(qf[0], x))
                     - ut_m * (zeta * sp.Derivative(h, x) + sp.Derivative(b, x)))
        omega_closed = -sum(sp.Derivative(qf[j], x)
                            * sp.integrate(phis[j].subs(zeta, s_), (s_, 0, zeta))
                            for j in range(1, Nu + 1))
        R_om = sp.expand(omega_closed - omega_def)
        # ω̃_closed − ω̃_def must vanish at the bottom (= the ŵ-closure/KBC)
        assert sp.simplify(R_om.subs(zeta, 0)) == 0

        def _zint01(e):
            poly = sp.Poly(sp.expand(e.doit()), zeta)
            return sum(cc / (nn[0] + 1)
                       for nn, cc in zip(poly.monoms(), poly.coeffs()))

        for kk in range(1, Nu + 1):
            dphi = sp.diff(phis[kk], zeta)
            m.momentum_x[kk].expr = sp.expand(
                m.momentum_x[kk].expr - _zint01(R_om * ut_m * dphi) / mus[kk])
            m.momentum_z[kk].expr = sp.expand(
                m.momentum_z[kk].expr - _zint01(R_om * wt_m * dphi) / mus[kk])

        # 6c — conservative regroup.  The closure/ω̃ substitutions left the
        # k≥1 rows fully EXPANDED, so the extraction routes their advective
        # divergences into NCP (path-dependent at shocks) instead of flux.
        # Regroup the conservative part F_k (the reference-pinned Escalante
        # flux, mapped to our variables) as a compound ∂_x atom:
        #   expr − expand(∂_x F.doit()) + Derivative(F, x)   (exact zero-change)
        # NB: keep ∂_x b-bearing flux parts in their OWN compound atom — the
        # extraction's diffusion branch fires on any b_x inside a compound
        # and silently skips bx-free pieces, so a MIXED atom would drop them.
        Pf = [m.functions.P.head(j, t, x) for j in range(Nu + 1)]
        if Nu == 1:
            F_groups = {
                ("momentum_x", 1): [2 * qf[0] * qf[1] / h + h * Pf[1] / rho],
                ("momentum_z", 1): [
                    (qf[0] * rf[1] / h + qf[1] * rf[0] / h
                     - sp.Rational(2, 5) * qf[1] * (rf[0] - rf[1]) / h),
                    (sp.Rational(2, 5) * qf[1] * (qf[0] - qf[1]) / h
                     * sp.Derivative(b, x)),
                ],
            }
            for (fam, kk), Fs in F_groups.items():
                row = getattr(m, fam)[kk]
                # normalize to fully-expanded atoms FIRST — the closure
                # substitutions left compound Derivative atoms behind, and
                # subtracting an expanded ∂_x F against unexpanded compounds
                # double-counts the content at extraction
                e = sp.expand(sp.sympify(row.expr).doit())
                for F in Fs:
                    dF = sp.Derivative(F, x)
                    e = sp.expand(e - sp.expand(dF.doit())) + dF
                row.expr = e

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
        P_modes = [self._P_head(j, t, x) for j in range(Nu + 1)]
        qs = list(m.explicit_state())
        if self._bed not in qs:
            qs = [self._bed, *qs]
        sm = SystemModel.from_model(m, Q=[*qs, *P_modes])
        if self.boundary_conditions is not None:
            sm.attach_boundary_conditions(
                self.boundary_conditions, aux_bcs=self.aux_boundary_conditions)
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
