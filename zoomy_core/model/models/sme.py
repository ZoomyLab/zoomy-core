"""SME — Shallow Moment Equations (Kowalski–Torrilhon 2019).

Inherits :class:`SigmaReference` (which gives the σ-mapped INS
reference + KBCs), and extends ``derive_model`` with the full SME
post-σ pipeline: modal ansatz, Galerkin projection, σ-integration,
KBC modal closure, CoV ``u → q/h``, opaque-basis resolution,
integral evaluation, gravity self-pair fold, higher-mode w closure,
mass-matrix inversion.

After ``derive_model`` finishes, the equations are in Symbol form
(``self.variables.h, self.variables.q_0, …``) and ``self._variable_map``
maps equation names to row indices in the operator-API matrices.
``SystemModel.from_model(sme)`` then extracts ``flux / source /
nonconservative_matrix`` via the basemodel default tag-extraction.

Output equation set matches K&T (4.17).
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.models.sigmaref import SigmaReference
from zoomy_core.model.operations import (
    Expression,
    Multiply,
    ResolveDummy,
    ProductRule,
    Integrate,
    EvaluateIntegrals,
    Symmetrize,
    KinematicBC,
    Legendre_shifted,
)


__all__ = ["SME"]


class SME(SigmaReference):
    """Shallow Moment Equations at level ``N``.

    Default ``N = 2`` matches K&T (4.17): 4 dynamic equations
    (``continuity_0`` for ``h``, ``momentum_x_0..2`` for ``q_0..q_2``).
    """

    # Equations leave ``derive_model`` in Function form (``h(t, x)``,
    # ``q_fn(k, t, x)``) so optional closures like
    # ``apply_slip_newton_friction`` can manipulate them with
    # ``Derivative(g·h², x)`` semantics intact.  The Function → Symbol
    # substitution + auto-tagging + ``_initialize_functions`` run later,
    # either via an explicit ``sme._finalize_for_systemmodel()`` call
    # or automatically when ``SystemModel.from_model(sme)`` is invoked.
    _finalize_lazy = True

    def __init__(self, N: int = 2, **kwargs):
        self.N = N
        # Indexed modal-coefficient Functions (declared before super()
        # so the derive_model pipeline can reference them).
        self.u_fn = sp.Function("u", real=True)
        self.w_fn = sp.Function("w", real=True)
        self.q_fn = sp.Function("q", real=True)
        # Opaque basis / test / weight dummies — resolved at the end.
        self.phi_u_fn = sp.Function("phi_u", real=True)
        self.phi_w_fn = sp.Function("phi_w", real=True)
        self.psi_u    = sp.Function("psi_u", real=True)
        self.psi_w    = sp.Function("psi_w", real=True)
        self.omega    = sp.Function("omega", real=True)

        kwargs.setdefault("name", f"SME-L={N}")
        kwargs.setdefault(
            "variables", ["h"] + [f"q_{k}" for k in range(N + 1)])
        kwargs.setdefault("parameters", {"g": 9.81, "rho": 1.0})
        kwargs.setdefault("eigenvalue_mode", "numerical")
        super().__init__(**kwargs)

    # ── Derivation hook ─────────────────────────────────────────────
    def derive_model(self):
        """Build reference equations (via SigmaReference) and run the
        SME post-σ pipeline.  Equations are left in Function form
        (``h(t, x)``, ``q_fn(k, t, x)``) so optional closures can
        manipulate them; the Function → Symbol substitution happens
        in ``_prepare_for_systemmodel`` (called automatically by
        ``SystemModel.from_model``)."""
        super().derive_model()
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

    def _prepare_for_systemmodel(self):
        """Lazy finalisation: substitute derivation Functions →
        Model Symbols and set ``_variable_map``.  Called by
        ``_finalize_for_systemmodel`` (which itself is called either
        explicitly or automatically by ``SystemModel.from_model``)."""
        self._substitute_to_model_symbols()
        self._variable_map = self._build_variable_map()

    # ── pre-σ hook: hydrostatic + p-elimination ─────────────────────
    def _pre_sigma_hook(self):
        s, src = self.state, self.src
        self.momentum_z.apply(
            {src.w: 0, src.tau.zx: 0, src.tau.zz: 0, src.tau.xz: 0}).simplify()
        self.momentum_z.apply(
            Integrate(s.z, s.z, src.eta, method="analytical"))
        self.momentum_z.apply(
            {src.p.subs(s.z, src.eta): 0}).simplify()
        p_subst = Expression(self.momentum_z.expr, "momentum_z").solve_for(src.p)
        self.momentum_x.apply(p_subst).simplify()

    # ── pipeline steps ─────────────────────────────────────────────
    def _multiply_h_and_fold_conservative(self):
        s, src, t, x = self.state, self.src, self.state.t, self.state.x
        self.apply(Multiply(src.h))
        for eq in self:
            eq.simplify()
        self.momentum_x[[0, 1]].apply(ProductRule(variables=[t, x]))
        self.continuity[[0]].apply(ProductRule(variables=[x]))
        for eq in self:
            eq.simplify()

    def _substitute_modal_ansatz(self):
        N, s, src = self.N, self.state, self.src
        sigma = s.zeta_ref
        self.u_ansatz = sum(
            self.u_fn(k, s.t, s.x) * self.phi_u_fn(k, sigma)
            for k in range(N + 1))
        self.w_ansatz = sum(
            self.w_fn(k, s.t, s.x) * self.phi_w_fn(k, sigma)
            for k in range(N + 2))
        self.apply({src.u.xreplace({s.z: sigma}): self.u_ansatz,
                    src.w.xreplace({s.z: sigma}): self.w_ansatz})

    def _galerkin_project_continuity_and_momentum_x(self):
        s = self.state
        sigma = s.zeta_ref
        self.continuity.apply(
            Multiply(self.psi_w(sigma) * self.omega(sigma)))
        self.momentum_x.apply(
            Multiply(self.psi_u(sigma) * self.omega(sigma)))

    def _sigma_integrate(self):
        s = self.state
        sigma = s.zeta_ref
        self.continuity.apply(
            Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
        self.momentum_x.apply(
            Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
        for eq in self:
            eq.simplify()

    def _kbc_modal_closure(self):
        """Use the KBC residuals to solve for w_fn(0) and w_fn(1)."""
        s, src = self.state, self.src
        sigma = s.zeta_ref

        def residual(interface, at_value):
            # KBC: w(t,x,z=interface) = ∂_t interface +
            # u(t,x,z=interface) · ∂_x interface
            lhs = src.w.subs(s.z, at_value)
            rhs = (sp.Derivative(interface, s.t)
                   + src.u.subs(s.z, at_value) * sp.Derivative(interface, s.x))
            # Plug in the ansatz at the interface σ-value:
            return (lhs - rhs).xreplace({
                src.w.subs(s.z, at_value): self.w_ansatz.subs(sigma, at_value),
                src.u.subs(s.z, at_value): self.u_ansatz.subs(sigma, at_value),
            })

        closure = sp.solve(
            [residual(src.b, sp.S.Zero), residual(src.eta, sp.S.One)],
            [self.w_fn(0, s.t, s.x), self.w_fn(1, s.t, s.x)],
            dict=True,
        )[0]
        self.apply(closure, level="minor", description="KBC modal closure")
        self.kbc_closure = closure

    def _cov_u_to_q(self):
        N, s, src = self.N, self.state, self.src
        self.u_modes = [self.u_fn(k, s.t, s.x) for k in range(N + 1)]
        self.q_modes = [self.q_fn(k, s.t, s.x) for k in range(N + 1)]
        self.apply(
            {u: q / src.h for u, q in zip(self.u_modes, self.q_modes)},
            level="minor", description="CoV u(k,t,x) → q(k,t,x)/h",
        )
        for eq in self:
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

        self.apply(ResolveDummy(self.omega, lambda arg: sp.S.One))
        self.apply(ResolveDummy(
            self.phi_u_fn,
            lambda k, sig, _b=self.legendre_u: basis_value(_b, k, sig)))
        self.apply(ResolveDummy(
            self.phi_w_fn,
            lambda k, sig, _b=self.legendre_w: basis_value(_b, k, sig)))
        self.apply(ResolveDummy(
            self.psi_u,
            [legendre_value(self.legendre_u, k) for k in range(N + 1)]))
        self.apply(ResolveDummy(
            self.psi_w,
            [legendre_value(self.legendre_w, k) for k in range(N + 2)]))

    def _evaluate_integrals(self):
        self.apply(EvaluateIntegrals(self.state))
        for eq in self:
            eq.simplify()

    def _gravity_self_pair_fold(self):
        s, src = self.state, self.src
        g = self.parameters.g
        mom0 = self.momentum_x_0
        grav = next(
            (t for t in mom0
             if t.expr.has(g) and t.expr.has(sp.Derivative(src.h, s.x))
             and not t.expr.has(src.b)),
            None)
        if grav is not None:
            grav.apply(Symmetrize(ProductRule(variables=[s.x])))
            mom0.simplify()

    def _higher_mode_w_closure(self):
        N, s = self.N, self.state
        w_higher = [self.w_fn(k, s.t, s.x) for k in range(2, N + 2)]
        residuals = [getattr(self, f"continuity_{k}").expr
                     for k in range(1, N + 1)]
        solution = sp.solve(residuals, w_higher, dict=True)[0]
        self.apply(solution, level="minor",
                   description="continuity_k → w_(k+1)")
        for eq in self:
            eq.simplify()

    def _invert_mass_matrix(self):
        for k in range(self.N + 1):
            eq = getattr(self, f"momentum_x_{k}")
            eq.expr = (2 * k + 1) * eq.expr
            eq.simplify()

    # ── final: Function → Symbol substitution ─────────────────────
    def _substitute_to_model_symbols(self):
        """Replace derivation Function calls (h(t,x), q_fn(k,t,x))
        with their Model Symbol equivalents (self.variables.h,
        self.variables.q_k) so the equations are Symbol-based ready
        for tag extraction."""
        s, src = self.state, self.src
        subs = {src.h: self.variables.h}
        for k in range(self.N + 1):
            subs[self.q_fn(k, s.t, s.x)] = self.variables[f"q_{k}"]
        for eq in self:
            eq.expr = eq.expr.xreplace(subs)

    def _build_variable_map(self):
        """``{eq_name: [row_index]}`` for tag extraction.  For
        SME(N=L): ``continuity_0 → row 0 (h)``, ``momentum_x_k → row
        k+1 (q_k)``."""
        m = {"continuity_0": [0]}
        for k in range(self.N + 1):
            m[f"momentum_x_{k}"] = [k + 1]
        return m

    # ── Optional stress closure: slip-Newton friction ─────────────
    def apply_slip_newton_friction(self):
        """Close the open ``τ_xz`` BC + Integral atoms with the
        slip-Newton constitutive law.

        Substitutions per K&T 2019 §4.3:
          * Newtonian bulk: ``τ_xz(σ) = (ν/h) · ∂_σ u(σ)``;
          * Free-surface BC: ``τ_xz(σ=1) = 0``;
          * Navier-slip bottom BC: ``τ_xz(σ=0) = (ν/λ) · u(σ=0)``.

        Reads ``ν``, ``λ`` from the Model's parameters Zstruct —
        the caller must declare them at construction:

            sme = SME(N=2, parameters={
                "g": 9.81, "rho": 1.0,
                "nu": 1e-3, "lambda": 1e-2,
            }, ...)
            sme.apply_slip_newton_friction()

        After this call, the equations carry algebraic friction
        terms in ``ν, λ, ρ`` instead of unresolved ``Integral(τ_xz,
        …)`` atoms.  Tags are re-computed; ``SystemModel.from_model``
        picks up the closed source.
        """
        # Require nu, lambda in Model.parameters so Symbol identity
        # stays consistent.
        missing = [k for k in ("nu", "lambda") if k not in self.parameters.keys()]
        if missing:
            raise ValueError(
                f"apply_slip_newton_friction: parameter {missing[0]!r} "
                f"must be declared at construction "
                f"(e.g. `parameters={{'g': 9.81, 'rho': 1.0, "
                f"'nu': 1e-3, 'lambda': 1e-2}}`)."
            )
        s, src = self.state, self.src
        nu = self.parameters.nu
        lam = self.parameters["lambda"]
        rho = self.parameters.rho
        tau_xz = sp.Function("tau_xz", real=True)

        # Reconstruct u(σ) = Σ_k (q_k/h)·φ_k(σ) in Function form
        # — uses ``src.h`` (Function call) and the derivation's
        # ``self.q_fn(k, t, x)`` Function calls so the resulting
        # substitution merges cleanly with the rest of the
        # equation (which is still in Function form at this point).
        # The Function → Symbol substitution happens later in
        # ``_prepare_for_systemmodel`` (triggered by
        # ``SystemModel.from_model``).
        def u_at(sig):
            return sum((self.q_fn(k, s.t, s.x) / src.h)
                       * self.legendre_u.eval(k, sig)
                       for k in range(self.N + 1))

        def newton(t_arg, x_arg, sig):
            return (nu / src.h) * sp.diff(u_at(sig), sig)

        from zoomy_core.model.operations import (
            Expression, ProductRule, EvaluateIntegrals,
        )

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

        for eq in self:
            eq.expr = eq.expr.replace(sp.Integral, integral_via_product_rule)
            eq.expr = eq.expr.xreplace({
                tau_xz(s.t, s.x, sp.S.Zero): tau_at_0_slip,
                tau_xz(s.t, s.x, sp.S.One):  tau_at_1_free,
            })
            eq.expr = eq.expr.replace(tau_xz, newton)
            eq.simplify()
        self.apply(EvaluateIntegrals(s))
        for eq in self:
            eq.simplify()

        # Friction mutated the equations after construction.  Reset
        # the finalize flag so the next ``SystemModel.from_model``
        # (or explicit ``_finalize_for_systemmodel`` call) re-runs
        # the substitution + tagging pass on the now-friction-laden
        # equations.
        self._finalized = False
        for eq in self:
            eq._solver_groups = None
        return self
