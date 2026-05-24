"""``MapToReferenceElement`` — multi-dim affine change-of-variable.

Rewrites integrals and boundary integrals over a physical
:class:`Domain` to its reference element via the affine map
``x = V₀ + B·ξ``.  The integrand is transformed by:

* Coordinate substitution ``x_i → (V₀ + B·ξ)_i``.
* Chain rule on first-order ``Derivative(f, x_i)`` atoms:
  ``∂f/∂x_i  →  Σ_j (B⁻¹)_{j,i} · ∂f̃/∂ξ_j``  where ``f̃ = f.subs(x → x_ref)``.
* Volume integrals get the geometric Jacobian factor ``|det B|``.

Boundary integrals are walked too — the integrand is substituted /
chain-ruled and the domain reference is updated to the reference
boundary, but no surface-Jacobian factor is added (boundary
parameterisations are out of this layer's scope).

Limitations:

* First-order ``Derivative`` w.r.t. domain coords only.  Higher-order
  derivatives (``∂²u/∂x²``) raise ``NotImplementedError`` — IBP first
  via :class:`DivergenceTheorem`.
* The op walks every ``Integral`` whose limit-vars cover the domain
  coords — sibling integrals over different vars are left untouched.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.models.ins_generator import Operation
from zoomy_core.symbolic.domains import Domain


class MapToReferenceElement(Operation):
    """Apply ``x = V₀ + B·ξ`` to every integral / boundary-integral
    over ``domain`` in the expression.
    """

    whole_leaf_op = True

    def __init__(self, domain: Domain, *,
                 name: str | None = None,
                 description: str | None = None):
        super().__init__(
            name=name or "map_to_reference",
            description=(description or
                         f"Affine map {domain.name} → "
                         f"{domain.reference().name}"),
        )
        self._domain = domain
        self._ref = domain.reference()
        self._B, self._V0 = domain.affine_map()
        self._Binv = self._B.inv()
        self._detB = self._B.det()
        # Substitution dict: physical coord → reference-coord expression.
        ref_xi = sp.Matrix([[c] for c in self._ref.coords])
        ref_image = self._V0 + self._B * ref_xi  # column vector
        self._coord_subs = {
            self._domain.coords[i]: ref_image[i, 0]
            for i in range(domain.dim)
        }

    # --- core walker --------------------------------------------------------

    def _leaf_sp(self, expr: sp.Expr) -> sp.Expr:
        return self._walk(expr)

    def _walk(self, expr: sp.Expr) -> sp.Expr:
        # Volume integral over the domain's coords?
        if isinstance(expr, sp.Integral):
            limit_vars = tuple(lim[0] for lim in expr.args[1:])
            if frozenset(limit_vars) == frozenset(self._domain.coords):
                return self._rewrite_volume_integral(expr)
            # Other integrals (e.g. inner running integrals over a
            # different variable) — recurse into the integrand only.
            new_integrand = self._walk(expr.args[0])
            if new_integrand is expr.args[0]:
                return expr
            return sp.Integral(new_integrand, *expr.args[1:])

        # Boundary integral on this domain's boundary?
        if (hasattr(expr.func, "_domain")
                and expr.func._domain is self._domain.boundary()):
            return self._rewrite_boundary_integral(expr)

        # Derivative w.r.t. one of the domain coords — chain-rule it.
        # NOTE: only kicks in *outside* of an Integral over coords,
        # which is rare but possible (e.g. surface fluxes already
        # written outside the integral).
        if isinstance(expr, sp.Derivative):
            return self._chain_rule(expr)

        # Generic recurse.
        if expr.args:
            new_args = tuple(self._walk(a) for a in expr.args)
            if any(n is not o for n, o in zip(new_args, expr.args)):
                return expr.func(*new_args)
        return expr

    # --- volume integral rewrite -------------------------------------------

    def _rewrite_volume_integral(self, integral: sp.Integral) -> sp.Expr:
        integrand = integral.args[0]
        # Chain-rule, then substitute coords.  Order matters: chain-rule
        # acts on Derivative atoms whose .args reference the original
        # coords, so it must happen before substitution erases them.
        new_integrand = self._chain_rule_walk(integrand)
        new_integrand = new_integrand * sp.Abs(self._detB)
        # Replace integration variables with the reference coords —
        # indefinite limits, since we've kept the reference simplex
        # opaque (per design choice: reference bounds aren't materialised
        # as iterated-integral form).
        new_limits = tuple((c,) for c in self._ref.coords)
        return sp.Integral(new_integrand, *new_limits)

    # --- boundary integral rewrite -----------------------------------------

    def _rewrite_boundary_integral(self, expr: sp.Expr) -> sp.Expr:
        # Function call BoundaryIntegral_<name>(integrand) — single arg.
        old_integrand = expr.args[0]
        new_integrand = self._chain_rule_walk(old_integrand)
        # Re-build using the reference-boundary's atom factory.  Caching
        # in `Domain` ensures `self._ref.boundary()` returns the same
        # instance every call.
        ref_boundary = self._ref.boundary()
        return ref_boundary.boundary_integral_fn(new_integrand)

    # --- chain rule --------------------------------------------------------

    def _chain_rule_walk(self, expr: sp.Expr) -> sp.Expr:
        """Recursively rewrite Derivative atoms via the chain rule, then
        substitute remaining coord symbols with their reference images.
        """
        rewritten = self._chain_rule_recurse(expr)
        # After all chain-rule rewrites, any *bare* coord symbol still
        # remaining (i.e. not under a Derivative) needs the standard
        # subs.  This catches integrand factors like the explicit ``x``
        # in ``g·x·∂u/∂y``.
        return rewritten.subs(self._coord_subs)

    def _chain_rule_recurse(self, expr: sp.Expr) -> sp.Expr:
        if isinstance(expr, sp.Derivative):
            return self._chain_rule(expr)
        if expr.args:
            new_args = tuple(self._chain_rule_recurse(a) for a in expr.args)
            if any(n is not o for n, o in zip(new_args, expr.args)):
                return expr.func(*new_args)
        return expr

    def _chain_rule(self, deriv: sp.Derivative) -> sp.Expr:
        f = deriv.args[0]
        specs = deriv.args[1:]
        # We only handle a single first-order derivative w.r.t. one
        # domain coord at a time.  Multi-spec or higher-order forms
        # would compound the chain rule and need IBP first.
        for spec in specs:
            # sympy stores ``Derivative(u, x, 2)`` as args ``(u, Tuple(x, 2))``.
            # Bare-symbol specs come through as the symbol itself.
            if isinstance(spec, (tuple, sp.Tuple)):
                var, count = spec[0], spec[1]
            else:
                var, count = spec, sp.S.One
            if var in self._domain.coords:
                if count != 1:
                    raise NotImplementedError(
                        f"MapToReferenceElement: cannot chain-rule "
                        f"higher-order Derivative ({deriv}). Apply "
                        f"DivergenceTheorem first to lower the order.")
                if len(specs) > 1:
                    raise NotImplementedError(
                        f"MapToReferenceElement: cannot chain-rule a "
                        f"mixed Derivative ({deriv}); split it first.")
                return self._chain_rule_first_order(f, var)
        # Derivative is w.r.t. variables outside the domain — leave as
        # is, but recurse into f to catch nested coord-derivatives.
        new_f = self._chain_rule_recurse(f)
        if new_f is f:
            return deriv
        return sp.Derivative(new_f, *specs)

    def _chain_rule_first_order(self, f: sp.Expr,
                                var: sp.Symbol) -> sp.Expr:
        """``∂f/∂x_i  →  Σ_j (B⁻¹)_{j,i} · ∂f̃/∂ξ_j``."""
        i = self._domain.coords.index(var)
        # First chain-rule any nested coord-derivatives in f, then
        # substitute coords to obtain f̃(ξ).
        f_ref = self._chain_rule_recurse(f).subs(self._coord_subs)
        ref_coords = self._ref.coords
        terms = [
            self._Binv[j, i] * sp.Derivative(f_ref, ref_coords[j])
            for j in range(self._domain.dim)
        ]
        return sp.Add(*terms)
