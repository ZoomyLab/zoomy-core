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

import warnings
import numpy as np
import sympy
from sympy import Symbol, integrate as sp_integrate, diff, Piecewise, Rational
from sympy.abc import z
from time import time as get_time


# Default timeout per single integral (seconds).  If an integral takes longer
# than this the integrator raises instead of hanging silently.
_DEFAULT_INTEGRAL_TIMEOUT = 30


class SymbolicIntegrator:
    """
    Unified integration engine for basis functions.

    By default, all integration is **symbolic** (sympy.integrate or knot-span
    piecewise integration).  Numerical quadrature is available but must be
    requested explicitly via ``strategy='quadrature'``.

    If a single integral exceeds ``integral_timeout`` seconds a
    ``TimeoutError`` is raised so the caller can decide how to proceed.
    """

    def __init__(self, basis, strategy="auto", n_workers=1,
                 integral_timeout=_DEFAULT_INTEGRAL_TIMEOUT):
        """
        Parameters
        ----------
        basis : Basisfunction
            The basis function object.
        strategy : str
            'auto': symbolic, with knot-span splitting for splines
            'sympy': always use sympy.integrate
            'knot_spans': split at knot boundaries (splines)
            'quadrature': numerical quadrature (must be requested explicitly)
        n_workers : int
            Number of parallel workers for matrix computation (1 = serial).
        integral_timeout : float
            Maximum wall-clock seconds for a single integral before raising.
            Set to ``None`` to disable the timeout.
        """
        self.basis = basis
        self.n_workers = n_workers
        self.integral_timeout = integral_timeout
        self._strategy = self._resolve_strategy(strategy)

    def _resolve_strategy(self, strategy):
        if strategy != "auto":
            return strategy
        # Auto: pick the best exact-symbolic strategy for the basis.
        # Quadrature is never selected automatically — the user must opt in.
        if hasattr(self.basis, "get_knot_spans") and self.basis.get_knot_spans():
            return "knot_spans"
        if hasattr(self.basis, "analytical_weighted_integral"):
            return "orthogonal"
        return "sympy"

    @property
    def strategy(self):
        return self._strategy

    # --- Core integration ---

    def integrate(self, expr, var, domain=None):
        """
        Integrate expr over domain w.r.t. var.

        Always symbolic unless the user explicitly selected 'quadrature'.
        Raises ``TimeoutError`` if a single integral exceeds the timeout.
        """
        if domain is None:
            domain = tuple(self.basis.bounds())
        a, b = domain

        if self._strategy == "knot_spans":
            return self._integrate_knot_spans(expr, var, a, b)
        elif self._strategy == "orthogonal":
            return self._integrate_orthogonal(expr, var, a, b)
        elif self._strategy == "quadrature":
            warnings.warn(
                "Using numerical quadrature for integration. "
                "Results will contain approximate floats, not exact symbolics. "
                "Pass strategy='sympy' or strategy='auto' for exact results.",
                stacklevel=2,
            )
            return self._integrate_quadrature(expr, var, a, b)
        else:
            return self._integrate_sympy_timed(expr, var, a, b)

    def _integrate_sympy_timed(self, expr, var, a, b):
        """sympy.integrate with wall-clock timeout."""
        if self.integral_timeout is None:
            return sp_integrate(expr, (var, a, b))

        import signal

        def _handler(signum, frame):
            raise TimeoutError(
                f"Symbolic integration timed out after {self.integral_timeout}s. "
                f"The integrand may be too complex for exact symbolic evaluation. "
                f"Consider using strategy='knot_spans' (for splines) or "
                f"strategy='quadrature' (for numerical fallback).\n"
                f"Integrand: {str(expr)[:200]}"
            )

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(self.integral_timeout))
        try:
            result = sp_integrate(expr, (var, a, b))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result

    def _integrate_orthogonal(self, expr, var, a, b):
        """
        Exact integration using basis orthogonality.

        The basis must provide ``analytical_weighted_integral(poly, var)``
        which computes  int poly(z) * weight(z) dz  using the orthogonal
        decomposition of the polynomial in the basis.

        All standard matrix integrands have the form weight * polynomial,
        so this method factors out the weight, passes the polynomial part
        to the basis, and gets an exact symbolic result (e.g. involving pi).
        """
        w = self.basis.weight(var)
        # Factor out the weight: expr = w * polynomial_part
        # For safety, divide and simplify
        poly_part = sympy.cancel(expr / w)

        result = self.basis.analytical_weighted_integral(poly_part, var)
        if result is not None:
            return result

        # Fallback: the integrand isn't purely polynomial × weight.
        # Use timed sympy integration.
        warnings.warn(
            f"Orthogonal decomposition failed (integrand not polynomial × weight). "
            f"Falling back to sympy.integrate.",
            stacklevel=3,
        )
        return self._integrate_sympy_timed(expr, var, a, b)

    def _integrate_knot_spans(self, expr, var, a, b):
        """Split integration at knot boundaries.

        Within each span, Piecewise collapses to a simple polynomial
        which is integrated via antiderivative evaluation (fast path).
        """
        spans = self.basis.get_knot_spans()
        result = sympy.Integer(0)
        for span_a, span_b in spans:
            sa = max(float(span_a), float(a))
            sb = min(float(span_b), float(b))
            if sa >= sb:
                continue
            midpoint = (sa + sb) / 2
            collapsed = self._collapse_piecewise(expr, var, midpoint)
            # Fast path: antiderivative + evaluate at bounds
            ra = Rational(sa).limit_denominator(10000)
            rb = Rational(sb).limit_denominator(10000)
            anti = sympy.integrate(collapsed, var)
            chunk = anti.subs(var, rb) - anti.subs(var, ra)
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

    def _precompute_basis(self, level):
        """Precompute phi_k(z) and dphi_k(z) expressions to avoid repeated eval."""
        n = level + 1
        phi = [self.basis.eval(k, z) for k in range(n)]
        dphi = [diff(p, z) for p in phi]
        psi = [sympy.integrate(p, z) for p in phi]  # antiderivatives for B
        return phi, dphi, psi

    def mass_matrix(self, level):
        """Compute M[k,i] = integral(phi_k * phi_i * weight, domain). Symmetric."""
        n = level + 1
        M = np.empty((n, n), dtype=object)
        w = self.basis.weight(z)
        bounds = tuple(self.basis.bounds())
        phi, _, _ = self._precompute_basis(level)
        for k in range(n):
            for i in range(k, n):
                val = self.integrate(w * phi[k] * phi[i], z, bounds)
                M[k, i] = val
                M[i, k] = val
        return M

    def triple_product(self, level):
        """Compute A[k,i,j] = integral(phi_k * phi_i * phi_j * weight, domain).

        Fully symmetric in (k,i,j) -- only computes sorted indices k<=i<=j
        and fills all permutations.  Saves 63% at n=3, 72% at n=5.
        """
        n = level + 1
        A = np.empty((n, n, n), dtype=object)
        w = self.basis.weight(z)
        bounds = tuple(self.basis.bounds())
        phi, _, _ = self._precompute_basis(level)
        for k in range(n):
            for i in range(k, n):
                for j in range(i, n):
                    val = self.integrate(w * phi[k] * phi[i] * phi[j], z, bounds)
                    # Fill all permutations of (k, i, j)
                    for p in {(k,i,j),(k,j,i),(i,k,j),(i,j,k),(j,k,i),(j,i,k)}:
                        A[p] = val
        return A

    def derivative_product(self, level):
        """Compute D[k,i] = integral(dphi_k/dz * dphi_i/dz * weight, domain). Symmetric."""
        n = level + 1
        D = np.empty((n, n), dtype=object)
        w = self.basis.weight(z)
        bounds = tuple(self.basis.bounds())
        _, dphi, _ = self._precompute_basis(level)
        for k in range(n):
            for i in range(k, n):
                val = self.integrate(w * dphi[k] * dphi[i], z, bounds)
                D[k, i] = val
                D[i, k] = val
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

        Exploits symmetry to minimize the number of integrals:
          A[k,i,j]: fully symmetric -> only sorted k<=i<=j  (63-72% savings)
          M, D, Dxi, Dxi2: symmetric -> upper triangle only (33-40% savings)
          DT[k,i,j]: symmetric in (k,i) -> k<=i only        (33% savings)
          B[k,i,j]: no symmetry -> all entries

        Basis evaluations phi_k(z), dphi_k(z), psi_k(z) are precomputed once.
        """
        t0 = get_time()
        n = level + 1
        w = self.basis.weight(z)
        bounds = tuple(self.basis.bounds())
        phi, dphi, psi = self._precompute_basis(level)

        phib = self.boundary_values(level)
        M = self.mass_matrix(level)
        A = self.triple_product(level)
        D = self.derivative_product(level)

        # n^2 matrices (some symmetric)
        D1 = np.empty((n, n), dtype=object)
        Dxi = np.empty((n, n), dtype=object)
        Dxi2 = np.empty((n, n), dtype=object)
        DD = np.empty((n, n), dtype=object)

        for k in range(n):
            for i in range(n):
                D1[k, i] = self.integrate(w * phi[k] * dphi[i], z, bounds)
                DD[k, i] = self.integrate(w * phi[k] * diff(dphi[i], z), z, bounds)
            # Dxi, Dxi2: symmetric in (k,i) -> upper triangle
            for i in range(k, n):
                Dxi[k, i] = self.integrate(w * dphi[k] * dphi[i] * z, z, bounds)
                Dxi[i, k] = Dxi[k, i]
                Dxi2[k, i] = self.integrate(w * dphi[k] * dphi[i] * z * z, z, bounds)
                Dxi2[i, k] = Dxi2[k, i]

        # n^3 tensors
        B = np.empty((n, n, n), dtype=object)
        DT = np.empty((n, n, n), dtype=object)

        for k in range(n):
            # DT: symmetric in (k,i) -> only compute k <= i
            for i in range(k, n):
                for j in range(n):
                    val = self.integrate(w * dphi[k] * dphi[i] * phi[j], z, bounds)
                    DT[k, i, j] = val
                    DT[i, k, j] = val
            # B: no symmetry -> all entries
            for i in range(n):
                for j in range(n):
                    B[k, i, j] = self.integrate(
                        w * dphi[k] * psi[j] * phi[i], z, bounds)

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
