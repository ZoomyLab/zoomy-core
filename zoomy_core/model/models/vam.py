"""VAM — non-hydrostatic Vertically-Averaged Moment system.

Builds on :class:`SigmaRef` but does NOT eliminate pressure via
hydrostatic reduction; ``state.w`` and ``state.p`` stay as state
variables and the z-momentum equation is Galerkin-projected
alongside continuity and momentum.x.

See the transparent-derivation notebook
``thesis/notebooks/modeling/transparent_derivations/vam_clean.ipynb``.
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


__all__ = ["VAM"]


class VAM(SigmaRef):
    """Non-hydrostatic VAM at level ``N``.

    State (matrix-extraction surface):
      ``[h, q_0..q_N, r_0..r_N]``  with  ``r_k = h · w_k``;
    pressure modes ``p_0..p_N`` are *auxiliary* (closed implicitly
    by the z-momentum equation; this scaffolding leaves p as
    free fields in the source).
    """

    def __init__(self, N: int = 2, *, name: str | None = None):
        self.N = N
        # Indexed modal-coefficient Functions.
        self.u_fn = sp.Function("u", real=True)
        self.w_fn = sp.Function("w", real=True)
        self.p_fn = sp.Function("p", real=True)
        self.q_fn = sp.Function("q", real=True)
        self.r_fn = sp.Function("r", real=True)
        # Opaque basis/test/weight dummies.
        self.phi_u_fn = sp.Function("phi_u", real=True)
        self.phi_w_fn = sp.Function("phi_w", real=True)
        self.phi_p_fn = sp.Function("phi_p", real=True)
        self.psi_u    = sp.Function("psi_u", real=True)
        self.psi_w    = sp.Function("psi_w", real=True)
        self.omega    = sp.Function("omega", real=True)

        super().__init__(name=name or f"VAM-L={N}")

        self._multiply_h_and_fold_conservative()
        self._substitute_modal_ansatz()
        self._galerkin_project_all_three()
        self._sigma_integrate()
        self._kbc_modal_closure()
        self._cov_u_to_q()
        self._resolve_dummies()
        self._evaluate_integrals()
        self._cov_w_to_r()
        self._gravity_self_pair_fold()
        self._invert_mass_matrix()
        self._eliminate_dt_h_via_continuity_0()

    # No pre-σ hook needed — VAM keeps w, p as state.

    # ── pipeline steps ──────────────────────────────────────────────
    def _multiply_h_and_fold_conservative(self):
        s, m = self.state, self.model
        t, x = s.t, s.x
        m.multiply(s.h)
        for eq in m.equations.values():
            eq.simplify()
        # Per-equation conservative-form folds.
        m.equations["continuity"][[0]].apply(ProductRule(variables=[x]))
        m.equations["momentum_x"][[0, 1, 7]].apply(
            ProductRule(variables=[t, x]))
        m.equations["momentum_z"][[2, 3, 11]].apply(
            ProductRule(variables=[t, x]))
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
            for k in range(N + 1))
        self.p_ansatz = sum(
            self.p_fn(k, s.t, s.x) * self.phi_p_fn(k, sigma)
            for k in range(N + 1))
        self.model.apply({
            s.u.xreplace({s.z: sigma}): self.u_ansatz,
            s.w.xreplace({s.z: sigma}): self.w_ansatz,
            s.p.xreplace({s.z: sigma}): self.p_ansatz,
        })

    def _galerkin_project_all_three(self):
        s = self.state
        sigma = s.zeta_ref
        # Continuity and x-momentum share the u-test; z-momentum uses
        # the w-test.
        self.model.equations["continuity"].apply(
            Multiply(self.psi_u(sigma) * self.omega(sigma)))
        self.model.equations["momentum_x"].apply(
            Multiply(self.psi_u(sigma) * self.omega(sigma)))
        self.model.equations["momentum_z"].apply(
            Multiply(self.psi_w(sigma) * self.omega(sigma)))

    def _sigma_integrate(self):
        s = self.state
        sigma = s.zeta_ref
        for name in ("continuity", "momentum_x", "momentum_z"):
            self.model.equations[name].apply(
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
        for eq in self.model.equations.values():
            eq.simplify()

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
            self.phi_p_fn,
            lambda k, sig, _b=self.legendre_u: basis_value(_b, k, sig))
        m.resolve_dummy(
            self.psi_u,
            [legendre_value(self.legendre_u, k) for k in range(N + 1)])
        m.resolve_dummy(
            self.psi_w,
            [legendre_value(self.legendre_w, k) for k in range(N + 1)])

    def _evaluate_integrals(self):
        self.model.apply(EvaluateIntegrals(self.state))
        for eq in self.model.equations.values():
            eq.simplify()

    def _cov_w_to_r(self):
        """``w_k → r_k / h`` so the vertical momentum equations have
        conservative time-derivative atoms ``∂_t r_k``."""
        N, s = self.N, self.state
        self.w_modes = [self.w_fn(k, s.t, s.x) for k in range(N + 1)]
        self.r_modes = [self.r_fn(k, s.t, s.x) for k in range(N + 1)]
        self.model.apply(
            {w: r / s.h for w, r in zip(self.w_modes, self.r_modes)},
            level="minor", description="CoV w(k,t,x) → r(k,t,x)/h",
        )
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

    def _invert_mass_matrix(self):
        for k in range(self.N + 1):
            for name in (f"momentum_x_{k}", f"momentum_z_{k}"):
                if name in self.model.equations:
                    eq = self.model.equations[name]
                    eq.expr = (2 * k + 1) * eq.expr
                    eq.simplify()

    def _eliminate_dt_h_via_continuity_0(self):
        """Substitute ``∂_t h → −∂_x q_0`` (from depth-mean continuity)
        in all non-continuity equations.  Without this the chain-rule
        residual ``−w·∂_t h`` from ``ProductRule(inverse)`` on
        ``h·∂_t w`` shows up as state-dependent mass-matrix entries."""
        s = self.state
        cont0_subst = {sp.Derivative(s.h, s.t):
                        -sp.Derivative(self.q_fn(0, s.t, s.x), s.x)}
        for name, eq in self.model.equations.items():
            if name == "continuity_0":
                continue
            eq.expr = eq.expr.xreplace(cont0_subst)
            eq.simplify()
