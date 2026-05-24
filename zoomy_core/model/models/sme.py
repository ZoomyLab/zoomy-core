"""SME — Shallow Moment Equations (Kowalski–Torrilhon 2019).

Builds on :class:`SigmaRef` by adding:

1. Hydrostatic reduction of the z-momentum equation + algebraic
   elimination of pressure into momentum_x (done in the
   ``_pre_sigma_hook`` so the elimination happens *before*
   ``SigmaTransform``).
2. ``× h`` and per-index ``ProductRule(inverse)`` to fold conservative
   atoms.
3. Opaque-dummy modal ansatz + Galerkin projection + σ-integration.
4. KBC modal closure for ``w_0, w_1``; CoV ``u_k → q_k / h``.
5. Dummy resolution → Legendre polynomials; ``EvaluateIntegrals``.
6. Gravity self-pair fold via ``Symmetrize(ProductRule)``.
7. Higher-mode w closure (``continuity_k → w_(k+1)`` for ``k ≥ 1``).
8. Mass-matrix inversion (``× (2k+1)`` on ``momentum_x_k``).
9. Optional slip-Newton friction via :meth:`apply_slip_newton_friction`.

Output matches K&T (4.17) — see the transparent-derivation notebook
``thesis/notebooks/modeling/transparent_derivations/sme_clean.ipynb``.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.models.model import Symmetrize
from zoomy_core.model.models.operations import (
    Expression,
    Multiply,
    ProductRule,
    Integrate,
    EvaluateIntegrals,
    KinematicBC,
    Legendre_shifted,
)
from zoomy_core.model.models.sigmaref import SigmaRef


__all__ = ["SME"]


class SME(SigmaRef):
    """Shallow Moment Equations at level ``N``.

    Default ``N = 2`` matches K&T (4.17): 4 dynamic equations
    (continuity + 3 momentum modes for ``q_0, q_1, q_2``).
    """

    def __init__(self, N: int = 2, *, name: str | None = None):
        self.N = N
        # Indexed modal-coefficient Functions (declared before super()
        # so the pre-σ hook below can reference them).
        self.u_fn = sp.Function("u", real=True)
        self.w_fn = sp.Function("w", real=True)
        self.q_fn = sp.Function("q", real=True)
        # Opaque basis/test/weight dummies — resolved at the end.
        self.phi_u_fn = sp.Function("phi_u", real=True)
        self.phi_w_fn = sp.Function("phi_w", real=True)
        self.psi_u    = sp.Function("psi_u", real=True)
        self.psi_w    = sp.Function("psi_w", real=True)
        self.omega    = sp.Function("omega", real=True)

        super().__init__(name=name or f"SME-L={N}")

        # Post-σ pipeline.
        self._verify_kt_against_paper()
        self._multiply_h_and_fold_conservative()
        self._substitute_modal_ansatz()
        self._galerkin_project_continuity_and_momentum_x()
        self._sigma_integrate()
        self._kbc_modal_closure()
        self._cov_u_to_q()
        self._resolve_dummies()
        self._evaluate_integrals()
        self._gravity_self_pair_fold()
        self._higher_mode_w_closure()
        self._invert_mass_matrix()

    # ── pre-σ hook: hydrostatic + p-elimination ─────────────────────
    def _pre_sigma_hook(self):
        s, m = self.state, self.model
        m.equations["momentum_z"].apply(
            {s.w: 0, s.tau.zx: 0, s.tau.zz: 0, s.tau.xz: 0}).simplify()
        m.equations["momentum_z"].apply(
            Integrate(s.z, s.z, s.eta, method="analytical"))
        m.equations["momentum_z"].apply(
            {s.p.subs(s.z, s.eta): 0}).simplify()
        p_subst = Expression(
            m.equations["momentum_z"].expr,
            "momentum_z").solve_for(s.p)
        m.equations["momentum_x"].apply(p_subst).simplify()

    # ── pipeline steps (one method per "paragraph" of the notebook) ─
    def _verify_kt_against_paper(self):
        s = self.state
        u_t = sp.Function("u", real=True)(s.t, s.x, s.zeta_ref)
        w_t = sp.Function("w", real=True)(s.t, s.x, s.zeta_ref)
        tau_xz_t = sp.Function("tau_xz", real=True)(s.t, s.x, s.zeta_ref)
        p_tilde = s.rho * s.g * s.h * (sp.Integer(1) - s.zeta_ref)
        jac_x = sp.Derivative(s.zeta_ref * s.h + s.b, s.x)
        jac_t = sp.Derivative(s.zeta_ref * s.h + s.b, s.t)
        kt_3_11 = (sp.Derivative(s.h * u_t, s.x)
                   + sp.Derivative(w_t - jac_x * u_t, s.zeta_ref))
        kt_3_16 = (sp.Derivative(s.h * u_t, s.t)
                   + sp.Derivative(s.h * u_t**2, s.x)
                   + sp.Derivative(u_t * (w_t - jac_t - u_t * jac_x), s.zeta_ref)
                   + (1 / s.rho) * sp.Derivative(s.h * p_tilde, s.x)
                   - (1 / s.rho) * sp.Derivative(p_tilde * jac_x, s.zeta_ref)
                   - (1 / s.rho) * sp.Derivative(tau_xz_t, s.zeta_ref))
        ours_cont = sp.expand(
            (s.h * self.model.equations["continuity"].expr).doit())
        ours_xmom = sp.expand(
            (s.h * self.model.equations["momentum_x"].expr).doit())
        if sp.simplify(ours_cont - sp.expand(kt_3_11.doit())) != 0:
            raise RuntimeError("continuity ≢ K&T (3.11)")
        if sp.simplify(ours_xmom - sp.expand(kt_3_16.doit())) != 0:
            raise RuntimeError("x-momentum × h ≢ K&T (3.16)")
        self.kt_verified = True

    def _multiply_h_and_fold_conservative(self):
        s, m, t, x = self.state, self.model, self.state.t, self.state.x
        m.multiply(s.h)
        for eq in m.equations.values():
            eq.simplify()
        m.equations["momentum_x"][[0, 1]].apply(ProductRule(variables=[t, x]))
        m.equations["continuity"][[0]].apply(ProductRule(variables=[x]))
        for eq in m.equations.values():
            eq.simplify()

    def _substitute_modal_ansatz(self):
        N, s = self.N, self.state
        sigma = s.zeta_ref
        self.u_ansatz = sum(
            self.u_fn(k, s.t, s.x) * self.phi_u_fn(k, sigma)
            for k in range(N + 1))
        self.w_ansatz = sum(
            self.w_fn(k, s.t, s.x) * self.phi_w_fn(k, sigma)
            for k in range(N + 2))
        self.model.apply({s.u.xreplace({s.z: sigma}): self.u_ansatz,
                          s.w.xreplace({s.z: sigma}): self.w_ansatz})

    def _galerkin_project_continuity_and_momentum_x(self):
        s = self.state
        sigma = s.zeta_ref
        self.model.equations["continuity"].apply(
            Multiply(self.psi_w(sigma) * self.omega(sigma)))
        self.model.equations["momentum_x"].apply(
            Multiply(self.psi_u(sigma) * self.omega(sigma)))

    def _sigma_integrate(self):
        s = self.state
        sigma = s.zeta_ref
        self.model.equations["continuity"].apply(
            Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
        self.model.equations["momentum_x"].apply(
            Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
        for eq in self.model.equations.values():
            eq.simplify()

    def _kbc_modal_closure(self):
        s = self.state
        sigma = s.zeta_ref
        kbc_bot = KinematicBC(s, s.b,   at=sp.S.Zero)
        kbc_top = KinematicBC(s, s.eta, at=sp.S.One)

        def residual(kbc, at_value):
            (lhs, rhs), = kbc.subs_map.items()
            return (lhs - rhs).xreplace({
                s.w.subs(s.z, at_value): self.w_ansatz.subs(sigma, at_value),
                s.u.subs(s.z, at_value): self.u_ansatz.subs(sigma, at_value),
            })

        closure = sp.solve(
            [residual(kbc_bot, sp.S.Zero),
             residual(kbc_top, sp.S.One)],
            [self.w_fn(0, s.t, s.x), self.w_fn(1, s.t, s.x)],
            dict=True,
        )[0]
        self.model.apply(closure, level="minor",
                          description="KBC modal closure")
        self.kbc_closure = closure

    def _cov_u_to_q(self):
        N, s = self.N, self.state
        self.u_modes = [self.u_fn(k, s.t, s.x) for k in range(N + 1)]
        self.q_modes = [self.q_fn(k, s.t, s.x) for k in range(N + 1)]
        self.model.apply(
            {u: q / s.h for u, q in zip(self.u_modes, self.q_modes)},
            level="minor", description="CoV u(k,t,x) → q(k,t,x)/h",
        )
        for eq in self.model.equations.values():
            eq.simplify()

    def _resolve_dummies(self):
        N = self.N
        self.legendre_u = Legendre_shifted(level=N)
        self.legendre_w = Legendre_shifted(level=N + 1)

        def basis_value(basis, k_arg, sigma_arg):
            if k_arg.is_Integer:
                return basis.eval(int(k_arg), sigma_arg)
            return sp.Function("phi_unresolved")(k_arg, sigma_arg)

        def legendre_value(basis, k):
            return lambda arg, _b=basis, _k=k: _b.eval(_k, arg)

        m = self.model
        m.resolve_dummy(self.omega, lambda arg: sp.S.One)
        m.resolve_dummy(
            self.phi_u_fn,
            lambda k, sig, _b=self.legendre_u: basis_value(_b, k, sig))
        m.resolve_dummy(
            self.phi_w_fn,
            lambda k, sig, _b=self.legendre_w: basis_value(_b, k, sig))
        m.resolve_dummy(
            self.psi_u,
            [legendre_value(self.legendre_u, k) for k in range(N + 1)])
        m.resolve_dummy(
            self.psi_w,
            [legendre_value(self.legendre_w, k) for k in range(N + 2)])

    def _evaluate_integrals(self):
        self.model.apply(EvaluateIntegrals(self.state))
        for eq in self.model.equations.values():
            eq.simplify()

    def _gravity_self_pair_fold(self):
        s = self.state
        mom0 = self.model.equations["momentum_x_0"]
        grav = next(
            (t for t in mom0
             if t.expr.has(s.g) and t.expr.has(sp.Derivative(s.h, s.x))
             and not t.expr.has(s.b)),
            None)
        if grav is not None:
            grav.apply(Symmetrize(ProductRule(variables=[s.x])))
            mom0.simplify()

    def _higher_mode_w_closure(self):
        N, s = self.N, self.state
        w_higher = [self.w_fn(k, s.t, s.x) for k in range(2, N + 2)]
        residuals = [self.model.equations[f"continuity_{k}"].expr
                     for k in range(1, N + 1)]
        solution = sp.solve(residuals, w_higher, dict=True)[0]
        self.model.apply(solution, level="minor",
                          description="continuity_k → w_(k+1)")
        for eq in self.model.equations.values():
            eq.simplify()

    def _invert_mass_matrix(self):
        for k in range(self.N + 1):
            eq = self.model.equations[f"momentum_x_{k}"]
            eq.expr = (2 * k + 1) * eq.expr
            eq.simplify()

    # ── optional: slip-Newton friction ───────────────────────────────
    def apply_slip_newton_friction(self, *, nu=None, lam=None):
        """Substitute the raw ``τ_xz`` BC + Integral atoms with the
        slip-Newton constitutive law:

        * Newtonian: ``τ_xz(σ) = (ν/h) · ∂_σ u(σ)``
        * Free surface: ``τ_xz(σ=1) = 0``
        * Navier slip: ``τ_xz(σ=0) = (ν/λ) · u(σ=0)``

        Recipe:
        1. ``ProductRule(inverse)`` on the integrand of every
           ``∫ _ẑⁿ · ∂_ẑ τ_xz d_ẑ`` atom + sympy ``integrate`` lifts
           the boundary value out via the fundamental theorem.
        2. Override the BC atoms (slip / free surface).
        3. Newton substitution for the remaining derivative-free
           volume integrals.
        4. ``EvaluateIntegrals`` to collapse σ-polynomial integrands.

        Result matches K&T (4.17) source ``(3·u_m + s + κ +
        4λs/h)·ν/λ`` etc. modulo the ρ kinematic-viscosity convention.
        """
        s = self.state
        nu  = nu  or sp.Symbol("nu",     positive=True)
        lam = lam or sp.Symbol("lambda", positive=True)
        tau_xz = sp.Function("tau_xz", real=True)

        def u_at(sig):
            return sum((self.q_fn(k, s.t, s.x) / s.h)
                        * self.legendre_u.eval(k, sig)
                        for k in range(self.N + 1))

        def newton(t_arg, x_arg, sig):
            return (nu / s.h) * sp.diff(u_at(sig), sig)

        def integral_via_product_rule(*integral_args):
            integrand, (var, lo, hi) = integral_args[0], integral_args[1]
            rewritten = Expression(integrand, "").apply(
                ProductRule(variables=[var], direction="inverse")
            ).expr
            return sum((sp.integrate(piece, (var, lo, hi))
                        for piece in sp.Add.make_args(rewritten)),
                       sp.S.Zero)

        tau_at_0_slip = (nu / lam) * u_at(sp.S.Zero)
        tau_at_1_free = sp.S.Zero

        for eq in self.model.equations.values():
            eq.expr = eq.expr.replace(sp.Integral, integral_via_product_rule)
            eq.expr = eq.expr.xreplace({
                tau_xz(s.t, s.x, sp.S.Zero): tau_at_0_slip,
                tau_xz(s.t, s.x, sp.S.One):  tau_at_1_free,
            })
            eq.expr = eq.expr.replace(tau_xz, newton)
            eq.simplify()
        self.model.apply(EvaluateIntegrals(s))
        for eq in self.model.equations.values():
            eq.simplify()
        self.nu, self.lam = nu, lam
