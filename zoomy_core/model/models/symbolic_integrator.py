"""
SymbolicIntegrator: unified interface for basis function integration.

Dispatches to the best available strategy based on basis capabilities:
1. Default: sympy.integrate (always works, can be slow)
2. Knot spans: split at knot boundaries, integrate simple polynomials per span (splines)
3. Quadrature: use analytical quadrature nodes if basis provides them (Chebyshev)
4. Parallel: farm independent integrals to processes (for large basis matrices)

The basis declares its capabilities via methods:
- get_knot_spans() -> list of (a, b) intervals (splines)
- quadrature_nodes(n) -> (nodes, weights) (Chebyshev)
- bounds() -> [a, b] (all bases)

Usage:
    integrator = SymbolicIntegrator(basis)
    M = integrator.mass_matrix(level)
    A = integrator.triple_product(level)
    val = integrator.integrate(expr, z, (0, 1))
"""

import numpy as np
import sympy
from sympy import Symbol, integrate as sp_integrate, diff, Piecewise, Rational
from sympy.abc import z
from time import time as get_time


class SymbolicIntegrator:
    """
    Unified integration engine for basis functions.

    Automatically selects the best strategy based on what the basis provides.
    """

    def __init__(self, basis, strategy="auto", n_workers=1):
        """
        Parameters
        ----------
        basis : Basisfunction
            The basis function object.
        strategy : str
            'auto': pick best available strategy
            'sympy': always use sympy.integrate
            'knot_spans': split at knot boundaries
            'quadrature': use basis quadrature nodes
        n_workers : int
            Number of parallel workers for matrix computation (1 = serial).
        """
        self.basis = basis
        self.n_workers = n_workers
        self._strategy = self._resolve_strategy(strategy)

    def _resolve_strategy(self, strategy):
        if strategy != "auto":
            return strategy
        if hasattr(self.basis, "get_knot_spans") and self.basis.get_knot_spans():
            return "knot_spans"
        if hasattr(self.basis, "quadrature_nodes"):
            return "quadrature"
        return "sympy"

    @property
    def strategy(self):
        return self._strategy

    # --- Core integration ---

    def integrate(self, expr, var, domain=None):
        """
        Integrate expr over domain w.r.t. var.
        Uses the best available strategy.
        """
        if domain is None:
            domain = tuple(self.basis.bounds())
        a, b = domain

        if self._strategy == "knot_spans":
            return self._integrate_knot_spans(expr, var, a, b)
        elif self._strategy == "quadrature":
            return self._integrate_quadrature(expr, var, a, b)
        else:
            return sp_integrate(expr, (var, a, b))

    def _integrate_knot_spans(self, expr, var, a, b):
        """Split integration at knot boundaries."""
        spans = self.basis.get_knot_spans()
        result = sympy.Integer(0)
        for span_a, span_b in spans:
            sa = max(float(span_a), float(a))
            sb = min(float(span_b), float(b))
            if sa >= sb:
                continue
            # Within a span, Piecewise collapses to a simple polynomial
            midpoint = (sa + sb) / 2
            collapsed = self._collapse_piecewise(expr, var, midpoint)
            chunk = sp_integrate(collapsed, (var, Rational(sa).limit_denominator(10000),
                                              Rational(sb).limit_denominator(10000)))
            result += chunk
        return result

    def _collapse_piecewise(self, expr, var, point):
        """
        Evaluate Piecewise conditions at a point to collapse to a simple expression.
        Recursively handles nested Piecewise.
        """
        if not isinstance(expr, sympy.Basic):
            return expr
        if isinstance(expr, Piecewise):
            for piece_expr, piece_cond in expr.args:
                if piece_cond == True or piece_cond.subs(var, point):
                    return self._collapse_piecewise(piece_expr, var, point)
            return sympy.Integer(0)
        if expr.args:
            new_args = [self._collapse_piecewise(a, var, point) for a in expr.args]
            return expr.func(*new_args)
        return expr

    def _integrate_quadrature(self, expr, var, a, b):
        """Use basis quadrature nodes for numerical integration, return sympy Rational."""
        nodes, weights = self.basis.quadrature_nodes()
        # Filter to domain
        mask = (nodes >= float(a)) & (nodes <= float(b))
        nodes_in = nodes[mask]
        weights_in = weights[mask]

        from sympy import lambdify
        f = lambdify(var, expr, "numpy")
        try:
            vals = f(nodes_in)
            if np.isscalar(vals):
                vals = np.full_like(nodes_in, float(vals))
            result = float(np.sum(weights_in * vals))
            return sympy.nsimplify(result, tolerance=1e-12, rational=False)
        except Exception:
            # Fallback to sympy
            return sp_integrate(expr, (var, a, b))

    # --- Basis matrix computation ---

    def mass_matrix(self, level):
        """Compute M[k,i] = integral(phi_k * phi_i * weight, domain)."""
        n = level + 1
        M = np.empty((n, n), dtype=object)
        w = self.basis.weight(z)
        bounds = self.basis.bounds()
        for k in range(n):
            for i in range(n):
                integrand = w * self.basis.eval(k, z) * self.basis.eval(i, z)
                M[k, i] = self.integrate(integrand, z, tuple(bounds))
        return M

    def triple_product(self, level):
        """Compute A[k,i,j] = integral(phi_k * phi_i * phi_j * weight, domain)."""
        n = level + 1
        A = np.empty((n, n, n), dtype=object)
        w = self.basis.weight(z)
        bounds = self.basis.bounds()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    integrand = w * self.basis.eval(k, z) * self.basis.eval(i, z) * self.basis.eval(j, z)
                    A[k, i, j] = self.integrate(integrand, z, tuple(bounds))
        return A

    def derivative_product(self, level):
        """Compute D[k,i] = integral(dphi_k/dz * dphi_i/dz * weight, domain)."""
        n = level + 1
        D = np.empty((n, n), dtype=object)
        w = self.basis.weight(z)
        bounds = self.basis.bounds()
        for k in range(n):
            for i in range(n):
                integrand = w * diff(self.basis.eval(k, z), z) * diff(self.basis.eval(i, z), z)
                D[k, i] = self.integrate(integrand, z, tuple(bounds))
        return D

    def boundary_values(self, level):
        """Compute phib[k] = phi_k(bounds[0])."""
        n = level + 1
        phib = np.empty(n, dtype=object)
        z_bot = self.basis.bounds()[0]
        for k in range(n):
            phib[k] = self.basis.eval(k, z_bot)
        return phib

    def compute_all_matrices(self, level):
        """
        Compute all standard basis matrices.
        Returns a dict compatible with Basismatrices._set_matrices_from_dict().
        """
        t0 = get_time()
        n = level + 1
        w = self.basis.weight(z)
        bounds = tuple(self.basis.bounds())

        phib = self.boundary_values(level)
        M = self.mass_matrix(level)
        A = self.triple_product(level)
        D = self.derivative_product(level)

        # Additional matrices
        D1 = np.empty((n, n), dtype=object)
        Dxi = np.empty((n, n), dtype=object)
        Dxi2 = np.empty((n, n), dtype=object)
        DD = np.empty((n, n), dtype=object)
        B = np.empty((n, n, n), dtype=object)
        DT = np.empty((n, n, n), dtype=object)

        for k in range(n):
            for i in range(n):
                D1[k, i] = self.integrate(
                    w * self.basis.eval(k, z) * diff(self.basis.eval(i, z), z),
                    z, bounds)
                Dxi[k, i] = self.integrate(
                    w * diff(self.basis.eval(k, z), z) * diff(self.basis.eval(i, z), z) * z,
                    z, bounds)
                Dxi2[k, i] = self.integrate(
                    w * diff(self.basis.eval(k, z), z) * diff(self.basis.eval(i, z), z) * z * z,
                    z, bounds)
                DD[k, i] = self.integrate(
                    w * self.basis.eval(k, z) * diff(diff(self.basis.eval(i, z), z), z),
                    z, bounds)
                for j in range(n):
                    B[k, i, j] = self.integrate(
                        w * diff(self.basis.eval(k, z), z)
                        * sympy.integrate(self.basis.eval(j, z), z)
                        * self.basis.eval(i, z),
                        z, bounds)
                    DT[k, i, j] = self.integrate(
                        w * diff(self.basis.eval(k, z), z)
                        * diff(self.basis.eval(i, z), z)
                        * self.basis.eval(j, z),
                        z, bounds)

        elapsed = get_time() - t0
        return {
            "phib": phib, "M": M, "A": A, "B": B, "D": D,
            "Dxi": Dxi, "Dxi2": Dxi2, "DD": DD, "D1": D1, "DT": DT,
            "_elapsed": elapsed, "_strategy": self._strategy,
        }

    def project(self, expr, test_index, weight=None):
        """
        Galerkin projection of expr onto test function phi_{test_index}.
        Returns integral(expr * phi_k * weight, domain) / M[k,k].
        """
        if weight is None:
            weight = self.basis.weight(z)
        phi_k = self.basis.eval(test_index, z)
        bounds = tuple(self.basis.bounds())
        numerator = self.integrate(expr * phi_k * weight, z, bounds)
        denominator = self.integrate(phi_k * phi_k * weight, z, bounds)
        return numerator / denominator
