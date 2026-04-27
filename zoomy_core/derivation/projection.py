"""Galerkin projection in physical (t, x, z) — proper math-aware pipeline.

The workflow follows the standard Galerkin derivation in
``slim_walkthrough.py``:

  1. Multiply the equation by ``φ_j(z)``.
  2. Wrap in ``∫_{b}^{b+h} dz``.
  3. For each term ``G(z) · ∂_y F(z)`` in the integrand:
       - if ``y == z``: integration-by-parts (FT-form)
         ``∫ G ∂_z F dz = [G F]_b^{b+h} − ∫ (∂_z G) F dz``
       - if ``y ∈ {t, x}``: Leibniz-IBP
         ``∫ G ∂_y F dz = ∂_y ∫ G F dz − ∫ (∂_y G) F dz
                         − G F |_{b+h} · ∂_y(b+h) + G F |_b · ∂_y b``
  4. Apply KBCs at the column ends:
       ``w|_b = u|_b · ∂_x b``,
       ``w|_{b+h} = ∂_t(b+h) + u|_{b+h} · ∂_x(b+h)``.
  5. Substitute the polynomial ansatz: replace opaque ``u, w, p``
     Functions with their expansions ``Σ q_i(t, x) φ_i((z-b)/h)``.
     **Polynomials are still in z at this point.**
  6. **Affine transformation** ``z = ξh + b``, ``dz = h dξ`` —
     applied only to the remaining ``Integral(F, (z, b, b+h))`` atoms.
  7. Polynomial-integrate in ξ on [0, 1].

The implementation uses the atomic primitives in
``zoomy_core.symbolic`` (``leibniz_general``, ``fundamental_theorem``,
``polynomial_integrate``) for the parts that can be handled by them,
and a small recursive IBP helper for products
``G · ∂_y F`` that the bare primitives don't cover.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from zoomy_core.symbolic import (
    leibniz_general,
    fundamental_theorem,
    polynomial_integrate,
)

from .ansatz import PolynomialAnsatz
from .flow import FlowSetup


@dataclass
class GalerkinProjection:
    """Project a physical-z equation against ``φ_j(z)`` and integrate
    ``dz`` from ``b`` to ``b+h``."""
    flow: FlowSetup
    ansatz: PolynomialAnsatz

    # ---- Helper: opaque field placeholders ----

    def opaque_fields(self):
        """Return ``(u_opaque, w_opaque, p_opaque)`` as sympy Functions
        of ``(t, x, z)``."""
        t, x, z = self.flow.t, self.flow.x, self.flow.z
        u = sp.Function("u", real=True)(t, x, z)
        w = sp.Function("w", real=True)(t, x, z)
        p = sp.Function("p", real=True)(t, x, z)
        return u, w, p

    # ---- Boundary-value substitution from KBCs ----

    def kbc_subs(self, u_opaque, w_opaque):
        """KBCs at z = b and z = η (with ∂_t b = 0 absorbed)."""
        z = self.flow.z
        b = self.flow.b
        eta = self.flow.eta
        x = self.flow.x
        t = self.flow.t
        return {
            w_opaque.subs(z, b):
                u_opaque.subs(z, b) * sp.Derivative(b, x),
            w_opaque.subs(z, eta):
                sp.Derivative(eta, t) + u_opaque.subs(z, eta) * sp.Derivative(eta, x),
        }

    # ---- The integration helper: handles ∫G · ∂_y F dz ----

    def _integrate_term(self, integrand, var_z, lo, hi, t_x_vars):
        """Process a single integrand over [lo, hi] in var_z.

        Detects derivative structure and applies the right rule.  Pure
        non-derivative terms are returned wrapped in a held
        ``Integral`` (later substituted via the affine transform).
        """
        if not integrand.has(var_z):
            # Constant in z: trivially integrate.
            return (hi - lo) * integrand

        # Pull constants (factors not depending on z) out front.
        if isinstance(integrand, sp.Mul):
            const_factors = []
            z_factors = []
            for a in integrand.args:
                (z_factors if a.has(var_z) else const_factors).append(a)
            if const_factors and z_factors:
                const = sp.Mul(*const_factors)
                inner = sp.Mul(*z_factors)
                return const * self._integrate_term(
                    inner, var_z, lo, hi, t_x_vars
                )

        # Pure derivative integrand whose INNER depends on z:
        # dispatch FT or Leibniz directly.  Derivatives of coefficients
        # (∂_x h, ∂_x b — inner doesn't depend on z) don't qualify.
        if isinstance(integrand, sp.Derivative):
            inner = integrand.args[0]
            if not inner.has(var_z):
                # Pure coefficient — integrate as a constant in z.
                return (hi - lo) * integrand
            vc = integrand.variable_count
            if len(vc) == 1 and vc[0][1] == 1:
                y = vc[0][0]
                if y == var_z:
                    return fundamental_theorem(integrand, var_z, lo, hi)
                if y in t_x_vars:
                    return leibniz_general(integrand, var_z, lo, hi)
            return sp.Integral(integrand, (var_z, lo, hi))

        # Product with a z-dependent Derivative factor: IBP.
        if isinstance(integrand, sp.Mul):
            deriv_factor = None
            other_factors = []
            for a in integrand.args:
                if (deriv_factor is None
                        and isinstance(a, sp.Derivative)
                        and a.args[0].has(var_z)):
                    vc = a.variable_count
                    if len(vc) == 1 and vc[0][1] == 1 and vc[0][0] in t_x_vars | {var_z}:
                        deriv_factor = a
                        continue
                other_factors.append(a)
            if deriv_factor is not None:
                G = sp.Mul(*other_factors) if other_factors else sp.S.One
                return self._ibp(G, deriv_factor, var_z, lo, hi, t_x_vars)

        # Fallback: held integral.
        return sp.Integral(integrand, (var_z, lo, hi))

    def _ibp(self, G, deriv_factor, var_z, lo, hi, t_x_vars):
        """IBP for ``∫ G · ∂_y F dvar_z``.

        ``y == var_z``: ``[G F]_lo^hi − ∫ (∂_z G) F dz``.
        ``y != var_z``: ``∂_y ∫G F dz − ∫(∂_y G) F dz
                         − G F |_hi · ∂_y hi + G F |_lo · ∂_y lo``.
        """
        y = deriv_factor.variable_count[0][0]
        F = deriv_factor.args[0]
        if y == var_z:
            boundary_hi = (G * F).xreplace({var_z: hi})
            boundary_lo = (G * F).xreplace({var_z: lo})
            new_integrand = sp.expand(sp.Derivative(G, var_z).doit() * F)
            recur = self._integrate_term(
                new_integrand, var_z, lo, hi, t_x_vars
            )
            return boundary_hi - boundary_lo - recur
        # y != var_z (Leibniz with z-dependent limits).
        bulk = sp.Derivative(sp.Integral(G * F, (var_z, lo, hi)), y)
        boundary = sp.S.Zero
        if hi.has(y):
            boundary -= (G * F).xreplace({var_z: hi}) * sp.Derivative(hi, y)
        if lo.has(y):
            boundary += (G * F).xreplace({var_z: lo}) * sp.Derivative(lo, y)
        new_integrand = sp.expand(sp.Derivative(G, y).doit() * F)
        recur = self._integrate_term(
            new_integrand, var_z, lo, hi, t_x_vars
        )
        return bulk + boundary - recur

    # ---- Top-level projection ----

    def project(self, eq_lhs, j: int,
                u_opaque, w_opaque, p_opaque):
        """Project ``eq_lhs(t, x, z) = 0`` against ``φ_j(z)`` and
        integrate ``dz``.  Apply KBCs, substitute polynomial ansatz,
        affine-transform remaining integrals, polynomial-integrate.
        """
        z = self.flow.z
        h = self.flow.h
        b = self.flow.b
        eta = self.flow.eta
        t_sym = self.flow.t
        x_sym = self.flow.x
        xi = self.ansatz.xi_ref
        t_x_vars = {t_sym, x_sym}

        # 1-3. Multiply by φ_j(z), distribute over Add, process each term.
        phi_j_z = self.ansatz.basis_at_z(j)
        full_integrand = sp.expand(phi_j_z * eq_lhs, mul=True, multinomial=False)

        # Each term goes through _integrate_term separately.
        if isinstance(full_integrand, sp.Add):
            terms = list(full_integrand.args)
        else:
            terms = [full_integrand]
        processed = sp.S.Zero
        for term in terms:
            processed += self._integrate_term(term, z, b, eta, t_x_vars)
        processed = sp.expand(processed)

        # 4. Apply KBCs and ∂_t b = 0.
        kbc = self.kbc_subs(u_opaque, w_opaque)
        processed = sp.expand(processed.xreplace(kbc))
        processed = processed.subs({sp.Derivative(b, t_sym): sp.S.Zero})

        # 5. Substitute polynomial ansatz (still in z).
        ansatz_subs = {
            u_opaque: self.ansatz.u,
            w_opaque: self.ansatz.w,
            p_opaque: self.ansatz.p,
        }
        with_ansatz = processed.xreplace(ansatz_subs)
        with_ansatz = sp.expand(with_ansatz.doit())

        # 6-7. Apply affine z → ξh + b on remaining held Integrals,
        #      polynomial-integrate.
        def _do_remaining(expr):
            if isinstance(expr, sp.Add):
                return sp.Add(*[_do_remaining(a) for a in expr.args])
            if isinstance(expr, sp.Mul):
                return sp.Mul(*[_do_remaining(a) for a in expr.args])
            if isinstance(expr, sp.Integral):
                inner = expr.args[0]
                lim = expr.args[1]
                if hasattr(lim, "__len__") and len(lim) == 3:
                    var, lo_, hi_ = lim
                    if var == z and lo_ == b and hi_ == eta:
                        inner_xi = sp.expand(inner.xreplace({z: xi * h + b}))
                        return polynomial_integrate(inner_xi * h, xi, 0, 1)
                return expr
            if isinstance(expr, sp.Derivative):
                # Could have Derivative(Integral, y); recurse on inner.
                inner_done = _do_remaining(expr.args[0])
                if inner_done is not expr.args[0]:
                    return sp.Derivative(inner_done, *expr.args[1:])
                return expr
            return expr

        return sp.expand(_do_remaining(with_ansatz))
