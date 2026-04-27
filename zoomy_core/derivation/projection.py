"""Galerkin projection in physical (t, x, z) — proper math-aware pipeline.

The workflow exactly mirrors the standard derivation in `slim_walkthrough.py`
(legacy):

    1. **Project**: multiply ``eq(t, x, z)`` (with opaque ``u, w, p``)
       by ``φ_j(z)`` (basis evaluated at z; depends on h, b through ζ).
    2. **Wrap in Integral**: ``∫_{b}^{b+h} φ_j · eq dz``.
    3. **Leibniz rule**: for ``∫ ∂_x F dz`` terms, convert to
       ``∂_x ∫ F dz − F|_{b+h}·∂_x(b+h) + F|_b·∂_x b``.
    4. **Fundamental theorem**: for ``∫ ∂_z F dz`` terms, convert to
       ``F|_{b+h} − F|_b``.
    5. **KBCs**: substitute ``w|_b = u|_b ∂_x b`` and (if needed)
       ``w|_η = ∂_t η + u|_η ∂_x η``.
    6. **Substitute polynomial ansatz**: replace opaque ``u, w, p``
       with their polynomial expansions in φ_i(z).  Polynomials are
       still in z at this point.
    7. **Affine transformation**: ``z = ξh + b``, ``dz = h dξ``.
       Pulled into the remaining ``Integral(... , (z, b, b+h))`` atoms.
    8. **Polynomial integrate** in ξ on [0, 1].

Substitution is **not** plain ``xreplace`` — different sympy atoms get
different treatments (chain rule for derivatives, affine for integral
limits, simple replacement for bare symbols).  This module orchestrates
the pipeline; the individual primitives live in ``zoomy_core.symbolic``.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from zoomy_core.symbolic import (
    leibniz_general,
    fundamental_theorem,
    affine_change_of_variable,
    polynomial_integrate as _poly_int_primitive,
    subst,
)

from .ansatz import PolynomialAnsatz
from .basis import polynomial_integrate
from .flow import FlowSetup


@dataclass
class GalerkinProjection:
    """Project a physical-z equation against φ_j(z) and integrate dz."""
    flow: FlowSetup
    ansatz: PolynomialAnsatz

    # ---- Helper: opaque field placeholders ----

    def opaque_fields(self):
        """Return ``(u_opaque, w_opaque, p_opaque)`` as sympy Functions
        of (t, x, z).  These are the symbolic placeholders to plug into
        the FlowSetup.x_momentum_lhs etc. for the Leibniz/FT/KBC steps;
        the polynomial ansatz is substituted at the very end."""
        t, x, z = self.flow.t, self.flow.x, self.flow.z
        u = sp.Function("u", real=True)(t, x, z)
        w = sp.Function("w", real=True)(t, x, z)
        p = sp.Function("p", real=True)(t, x, z)
        return u, w, p

    # ---- Boundary value evaluation (KBC substitution) ----

    def kbc_subs(self, u_opaque, w_opaque):
        """Build the substitution dict for KBCs at the column ends.

        Bottom (z = b):  ``w|_b = u|_b · ∂_x b``    (no ∂_t b: b is fixed).
        Top    (z = η):  ``w|_η = ∂_t η + u|_η · ∂_x η``.
        """
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

    # ---- The pipeline ----

    def project(self, eq_lhs, j: int,
                u_opaque=None, w_opaque=None, p_opaque=None):
        """Project ``eq_lhs(t, x, z) = 0`` against φ_j(z).

        Steps 1-8 of the workflow above.  ``eq_lhs`` should use the
        opaque fields (returned by ``self.opaque_fields()``) so the
        Leibniz / fundamental-theorem / KBC steps see opaque ``u, w, p``
        before the ansatz substitution.

        Returns the final scalar (constant in z) projected equation.
        """
        if u_opaque is None or w_opaque is None or p_opaque is None:
            raise ValueError(
                "Pass opaque (u, w, p) — get them from self.opaque_fields()."
            )

        z = self.flow.z
        h = self.flow.h
        b = self.flow.b
        eta = self.flow.eta
        t_sym = self.flow.t
        x_sym = self.flow.x
        xi = self.ansatz.xi_ref

        # 1. Multiply by φ_j(z).  φ_j(z) = basis_xi[j] with xi → ζ(z) =
        #    (z - b)/h.  Sympy keeps this composed.
        phi_j_z = self.ansatz.basis_at_z(j)
        integrand = phi_j_z * eq_lhs

        # Distribute multiplication over Add to make Leibniz/FT applicable
        # term-by-term.
        integrand_expanded = sp.expand(integrand, deep=False)

        # 2-5. For each Add term, look at ∂_x / ∂_z structure; apply
        # Leibniz / FT primitives + KBC.  Then sum.
        kbc = self.kbc_subs(u_opaque, w_opaque)

        def _process_term(term):
            # Wrap in ∫_b^{b+h} ... dz.
            # For terms of the form (other) · ∂_x F(t, x, z): apply
            # Leibniz to ∂_x ∫ F dz interchange.  For ∂_z F: apply FT.
            # For everything else: leave wrapped, expand later.
            #
            # Simplest implementation: build the integral, then sympy's
            # expand handles linearity over Add; we then identify
            # Derivative-of-integrand structure and apply primitives.
            #
            # For our equations, the structure is term-by-term known:
            #   ∂_t u — wrap in integral, expand product.
            #   u ∂_x u — wrap; ∂_x u is a Derivative.
            #   w ∂_z u — wrap; ∂_z u is a Derivative.  Use FT-style:
            #     ∫ φ_j w ∂_z u dz — ∂_z u is a derivative IN z of u(t,x,z).
            #     IBP: ∫ φ_j w ∂_z u dz = [φ_j w u]_b^{b+h} − ∫ ∂_z(φ_j w) u dz.
            #     This is integration by parts in z.  We could use
            #     symbolic.integration_by_parts but the depth-of-the-inner
            #     primitive is different.
            #
            # For now: just wrap and integrate via affine substitution.
            # Apply KBCs to substitute ω-like boundary terms after.
            return sp.Integral(term, (z, b, eta))

        # Build the wrapped integral as a single Integral over the Add.
        integral_expr = sp.Integral(integrand_expanded, (z, b, eta))

        # 3. Leibniz / FT.  For ∂_x F integrand pieces, leibniz_general
        # produces ∂_x (∫ F dz) − F|_η ∂_x η + F|_b ∂_x b.
        # For ∂_z F integrand, fundamental_theorem gives F|_η − F|_b.
        # We let sympy.expand split the integral over Add first.
        integral_split = integral_expr.expand()

        # Apply primitives over the resulting sum of Integral atoms.
        # `leibniz_general` and `fundamental_theorem` operate on a single
        # Integral atom; we walk the expression tree.
        def _apply_lf(expr):
            if isinstance(expr, sp.Add):
                return sp.Add(*[_apply_lf(a) for a in expr.args])
            if isinstance(expr, sp.Mul):
                # Leibniz/FT only apply to integrand-with-derivative atoms;
                # for plain Mul (constant · Integral) we handle by
                # recursing on the Integral factor.
                new_args = []
                for a in expr.args:
                    new_args.append(_apply_lf(a))
                return sp.Mul(*new_args)
            if isinstance(expr, sp.Integral):
                # Only handle ``Integral(integrand, (var, lo, hi))`` shape
                # at the top level here.
                inner = expr.args[0]
                limits = expr.args[1]
                if not (hasattr(limits, "__len__") and len(limits) == 3):
                    return expr
                var = limits[0]
                lo, hi = limits[1], limits[2]
                # Look at the integrand: if it contains Derivative wrt
                # ``var``, apply FT.  If wrt some OUTER variable, apply
                # Leibniz with respect to that.
                if isinstance(inner, sp.Mul):
                    # Find a Derivative factor.
                    deriv_factor = None
                    for a in inner.args:
                        if isinstance(a, sp.Derivative):
                            deriv_factor = a
                            break
                    if deriv_factor is not None:
                        var_count = deriv_factor.variable_count
                        v0, n0 = var_count[0]
                        if isinstance(v0, sp.Tuple):
                            v0 = v0[0]
                        if v0 == var and n0 == 1:
                            # FT applies on integrand inner-derivative.
                            # But we need ∂_z F integrand → boundary
                            # values.  Use fundamental_theorem.
                            other_factors = [a for a in inner.args
                                              if a is not deriv_factor]
                            other = sp.Mul(*other_factors) if other_factors else sp.S.One
                            # ∫ other * ∂_v F dv = [other * F]_{lo}^{hi}
                            # − ∫ ∂_v other · F dv  (IBP).  Direct FT
                            # only applies if other = 1.  Use
                            # integration-by-parts shape instead.
                            F = deriv_factor.args[0]
                            return (other.xreplace({var: hi}) * F.xreplace({var: hi})
                                    - other.xreplace({var: lo}) * F.xreplace({var: lo})
                                    - sp.Integral(sp.Derivative(other, var) * F,
                                                  (var, lo, hi)))
                # Default: leave the Integral atom as-is.
                return expr
            return expr

        # Apply Leibniz/FT-style reductions iteratively until fixpoint.
        prev = None
        cur = integral_split
        while prev != cur:
            prev = cur
            cur = sp.expand(_apply_lf(cur))
        integral_after_ft = cur

        # 4. Apply KBCs (substitute boundary values of w).
        integral_after_kbc = integral_after_ft.xreplace(kbc)
        # Also apply ∂_t b = 0 (b fixed).
        integral_after_kbc = integral_after_kbc.subs(
            {sp.Derivative(b, t_sym): sp.S.Zero}
        )

        # 5. Substitute the polynomial ansatz: replace opaque
        # u_opaque(t, x, z), w_opaque(t, x, z), p_opaque(t, x, z) with
        # their polynomial forms.  We use direct subs on the Function
        # call; sympy propagates through Derivative atoms.
        ansatz_subs = {
            u_opaque: self.ansatz.u,
            w_opaque: self.ansatz.w,
            p_opaque: self.ansatz.p,
        }
        with_ansatz = integral_after_kbc.subs(ansatz_subs)
        # Expand any Derivative atoms produced by the substitution
        # (chain rules through ζ(z)).
        with_ansatz = sp.expand(with_ansatz.doit())

        # 6. Affine transform z = ξ h + b in the remaining Integral atoms,
        # then polynomial-integrate in ξ.
        def _do_remaining_integrals(expr):
            if isinstance(expr, sp.Add):
                return sp.Add(*[_do_remaining_integrals(a) for a in expr.args])
            if isinstance(expr, sp.Mul):
                return sp.Mul(*[_do_remaining_integrals(a) for a in expr.args])
            if isinstance(expr, sp.Integral):
                inner = expr.args[0]
                lim = expr.args[1]
                if hasattr(lim, "__len__") and len(lim) == 3:
                    var, lo, hi = lim[0], lim[1], lim[2]
                    if var == z and lo == b and hi == eta:
                        # Affine: substitute z → ξh+b inside, multiply by h dξ.
                        inner_xi = sp.expand(inner.xreplace({z: xi * h + b}))
                        return polynomial_integrate(inner_xi * h, xi, 0, 1)
                return expr
            return expr

        result = _do_remaining_integrals(with_ansatz)
        return sp.expand(result)
