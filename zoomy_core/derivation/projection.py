"""Galerkin projection in physical (t, x, z) — strictly mechanical.

The projection does **only** what's listed in the workflow.  No
implicit product-rule / IBP / continuity substitutions.  The caller
is responsible for putting the equation in a form where every term
is one of:

  (A) **Constant in z**: a coefficient like ``g · ∂_x η`` where η = b+h
      doesn't depend on z.  Integrates trivially as ``coef · ∫ φ_j dz``.

  (B) **Pure derivative of an opaque field**: ``∂_y F`` where F is
      a Function ``u(t, x, z)``, ``w(t, x, z)`` or ``p(t, x, z)`` and
      y is one of t, x, z.  Apply :func:`leibniz_general` (y ∈ {t, x})
      or :func:`fundamental_theorem` (y = z).

  (C) **Polynomial in z** (no opaque derivative factors): integrate
      via affine map ``z = ξh + b`` + :func:`polynomial_integrate`.

To get from the raw NS equations to (A)+(B)+(C), the caller does:

  - Apply **product rule** to combine ``u · ∂_x u`` into ``(1/2) ∂_x(u²)``.
  - Apply **continuity** to convert ``w ∂_z u`` into something with no
    cross-field opaque derivatives (e.g.  ``w ∂_z u + u ∂_z w =
    ∂_z(uw)``, then use continuity ``∂_z w = -∂_x u`` to substitute).
  - In short: produce a **conservative form** before projecting.

This makes the projection 100% predictable and the user 100% in
control of the math.  No surprises from automatic primitive firing.

Workflow:

  1. Multiply ``eq`` by ``φ_j(z)``.
  2. Wrap in ``∫_{b}^{b+h} dz``.
  3. Distribute over Add (linearity of integral).
  4. For each term, identify (A)/(B)/(C); apply the matching primitive.
     - (B) terms produce ``Leibniz/FT`` results possibly with
       boundary values ``F|_b``, ``F|_{b+h}`` and held outer integrals
       ``∂_y ∫ F dz``.
     - (C) terms remain as held ``Integral`` atoms.
  5. Apply **KBCs** to substitute ``w|_b``, ``w|_{b+h}`` (∂_t b = 0).
  6. **Substitute polynomial ansatz** for opaque ``u, w, p`` in any
     remaining held expressions.
  7. **Affine map** ``z = ξh + b`` on the held ``Integral(..., (z, b, b+h))``
     atoms; integrate ``dξ`` on [0, 1] via ``polynomial_integrate``.
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
    """Mechanical Galerkin projection.  Caller pre-processes the
    equation; this module integrates."""
    flow: FlowSetup
    ansatz: PolynomialAnsatz

    # ---- Helper: opaque field placeholders ----

    def opaque_fields(self):
        """``(u, w, p)`` as sympy Functions of (t, x, z)."""
        t, x, z = self.flow.t, self.flow.x, self.flow.z
        u = sp.Function("u", real=True)(t, x, z)
        w = sp.Function("w", real=True)(t, x, z)
        p = sp.Function("p", real=True)(t, x, z)
        return u, w, p

    # ---- KBCs ----

    def kbc_subs(self, u_opaque, w_opaque):
        z, b, eta = self.flow.z, self.flow.b, self.flow.eta
        x, t = self.flow.x, self.flow.t
        return {
            w_opaque.subs(z, b):
                u_opaque.subs(z, b) * sp.Derivative(b, x),
            w_opaque.subs(z, eta):
                sp.Derivative(eta, t) + u_opaque.subs(z, eta) * sp.Derivative(eta, x),
        }

    # ---- Per-term integration: identify (A)/(B)/(C) ----

    def _integrate_term(self, term, var_z, lo, hi, opaque_fields):
        """Single term, no Add.  Identify which case applies."""
        # (A) Constant in z.
        if not term.has(var_z):
            return (hi - lo) * term

        # (B) Pure derivative integrand (any inner — FT/Leibniz are
        # general and don't require the inner to be a bare Function).
        if isinstance(term, sp.Derivative):
            vc = term.variable_count
            if len(vc) == 1 and vc[0][1] == 1:
                y = vc[0][0]
                if y == var_z:
                    return fundamental_theorem(term, var_z, lo, hi)
                if y in {self.flow.t, self.flow.x}:
                    return leibniz_general(term, var_z, lo, hi)
            # Higher-order or mixed derivatives: fall through to (C).

        # (B) factored: c(t, x) · ∂_y F  where c is z-independent.
        if isinstance(term, sp.Mul):
            const_factors = []
            z_factors = []
            for a in term.args:
                (z_factors if a.has(var_z) else const_factors).append(a)
            if const_factors and z_factors:
                const = sp.Mul(*const_factors)
                inner = sp.Mul(*z_factors)
                return const * self._integrate_term(
                    inner, var_z, lo, hi, opaque_fields
                )

        # (B) phi_j(z) · ∂_y F pattern: IBP, but only when the
        # non-derivative factors don't contain F (avoids the
        # F · ∂_y F recursion the user is meant to handle via the
        # product rule before projecting).
        if isinstance(term, sp.Mul):
            for k, c in enumerate(term.args):
                if not isinstance(c, sp.Derivative):
                    continue
                vc = c.variable_count
                if len(vc) != 1 or vc[0][1] != 1:
                    continue
                F = c.args[0]
                others = [a for j, a in enumerate(term.args) if j != k]
                G = sp.Mul(*others) if others else sp.S.One
                # If G contains F, IBP would recurse — caller should
                # have used product rule.  Skip.
                if G.has(F):
                    continue
                y = vc[0][0]
                if y == var_z:
                    boundary = (G * F).xreplace({var_z: hi}) - (G * F).xreplace({var_z: lo})
                    new_integrand = sp.expand(sp.Derivative(G, var_z).doit() * F)
                    recur = self._integrate_term(
                        new_integrand, var_z, lo, hi, opaque_fields
                    )
                    return boundary - recur
                if y in {self.flow.t, self.flow.x}:
                    bulk = sp.Derivative(sp.Integral(G * F, (var_z, lo, hi)), y)
                    boundary = sp.S.Zero
                    if hi.has(y):
                        boundary -= (G * F).xreplace({var_z: hi}) * sp.Derivative(hi, y)
                    if lo.has(y):
                        boundary += (G * F).xreplace({var_z: lo}) * sp.Derivative(lo, y)
                    new_integrand = sp.expand(sp.Derivative(G, y).doit() * F)
                    recur = self._integrate_term(
                        new_integrand, var_z, lo, hi, opaque_fields
                    )
                    return bulk + boundary - recur

        # (C) Polynomial in z (or any z-dependent expression that's not
        # a covered (B)): held Integral.  Will be evaluated after the
        # ansatz substitution + affine transform.
        return sp.Integral(term, (var_z, lo, hi))

    # ---- Top-level project ----

    def project(self, eq_lhs, j: int,
                u_opaque, w_opaque, p_opaque):
        """Project ``eq_lhs(t, x, z) = 0`` against ``φ_j(z)``.

        Caller is expected to have prepared ``eq_lhs`` so each term is
        clean (see module docstring).
        """
        z = self.flow.z
        h = self.flow.h
        b = self.flow.b
        eta = self.flow.eta
        t_sym = self.flow.t
        xi = self.ansatz.xi_ref
        opaque_fields = {u_opaque, w_opaque, p_opaque}

        # Step 1-3: multiply by φ_j(z), distribute, integrate term-by-term.
        phi_j_z = self.ansatz.basis_at_z(j)
        full = sp.expand(phi_j_z * eq_lhs, mul=True, multinomial=False)
        terms = list(full.args) if isinstance(full, sp.Add) else [full]
        processed = sp.S.Zero
        for term in terms:
            processed += self._integrate_term(
                term, z, b, eta, opaque_fields
            )

        # Step 5: KBCs + ∂_t b = 0.
        processed = sp.expand(processed.xreplace(self.kbc_subs(u_opaque, w_opaque)))
        processed = processed.subs({sp.Derivative(b, t_sym): sp.S.Zero})

        # Step 6: substitute polynomial ansatz EVERYWHERE — including
        # at boundary values like u(t, x, b+h).  We can't use
        # ``xreplace({u_op: ansatz.u})`` because boundary calls like
        # ``u_op.subs(z, eta)`` are different Function-call atoms.
        # Use ``replace`` on the Function class with a callable that
        # rebuilds the ansatz expression at the given z argument.
        u_func = u_opaque.func
        w_func = w_opaque.func
        p_func = p_opaque.func
        u_poly = self.ansatz.u
        w_poly = self.ansatz.w
        p_poly = self.ansatz.p

        def _ansatz_at(poly_expr, e):
            # ``e`` is a Function call ``f(t, x, z_val)``; substitute
            # z → z_val in the polynomial.
            z_val = e.args[2] if len(e.args) >= 3 else z
            return poly_expr.xreplace({z: z_val})

        with_ansatz = processed
        with_ansatz = with_ansatz.replace(
            lambda e: isinstance(e, sp.Function) and e.func == u_func,
            lambda e: _ansatz_at(u_poly, e),
        )
        with_ansatz = with_ansatz.replace(
            lambda e: isinstance(e, sp.Function) and e.func == w_func,
            lambda e: _ansatz_at(w_poly, e),
        )
        if p_poly != sp.S.Zero:
            with_ansatz = with_ansatz.replace(
                lambda e: isinstance(e, sp.Function) and e.func == p_func,
                lambda e: _ansatz_at(p_poly, e),
            )
        with_ansatz = sp.expand(with_ansatz.doit())

        # Step 7: affine z → ξh + b on held Integral atoms; polynomial-integrate.
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
                inner_done = _do_remaining(expr.args[0])
                if inner_done is not expr.args[0]:
                    return sp.Derivative(inner_done, *expr.args[1:])
                return expr
            return expr

        return sp.expand(_do_remaining(with_ansatz))
