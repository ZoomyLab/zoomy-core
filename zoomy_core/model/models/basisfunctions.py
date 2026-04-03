"""Module `zoomy_core.model.models.basisfunctions`."""

from copy import deepcopy

import numpy as np
import sympy
from sympy import Symbol, bspline_basis_set, diff, integrate, lambdify, legendre
from sympy.abc import z
from sympy.functions.special.polynomials import chebyshevu


class Basisfunction:
    """Basisfunction. (class)."""
    name = "Basisfunction"

    def bounds(self):
        """Bounds."""
        return [0, 1]

    def basis_definition(self):
        """Basis definition."""
        z = Symbol("z")
        b = lambda k, z: z**k
        return [b(k, z) for k in range(self.level + 1)]

    def weight(self, z):
        """Weight."""
        return 1

    def mean_coefficients(self):
        """
        Return coefficients c_k such that sum(c_k * phi_k(z)) = 1.

        Used for computing the depth-averaged velocity:
            u_mean = sum(c_k * alpha_k)

        Default: c = [1, 0, 0, ...] (assumes phi_0 = 1).
        Override for bases where the constant function is a non-trivial
        combination (e.g., B-splines: c = [1, 1, 1, ...]).
        """
        c = [sympy.Integer(0)] * (self.level + 1)
        c[0] = sympy.Integer(1)
        return c
    
    def weight_eval(self, z):
        """Weight eval."""
        z = Symbol("z")
        f = sympy.lambdify(z, self.weight(z))
        return f(z)

    def __init__(self, level=0, **kwargs):
        """Initialize the instance."""
        self.level = level
        self.basis = self.basis_definition(**kwargs)

    def get(self, k):
        """Get."""
        return self.basis[k]

    def eval(self, k, _z):
        """Eval."""
        return self.get(k).subs(z, _z)
    
    def eval_psi(self, k, _z):
        """Eval psi."""
        z = sympy.Symbol('z')
        psi = sympy.integrate(self.get(k), (z, self.bounds()[0], z))
        return psi.subs(z, _z)

    def get_lambda(self, k):
        """Get lambda."""
        f = lambdify(z, self.get(k))

        def lam(z):
            """Lam."""
            if type(z) == int or type(z) == float:
                return f(z)
            elif type(z) == list or type(z) == np.ndarray:
                return np.array([f(xi) for xi in z])
            else:
                assert False

        return lam

    def plot(self, ax):
        """Plot."""
        X = np.linspace(self.bounds()[0], self.bounds()[1], 1000)
        for i in range(len(self.basis)):
            f = lambdify(z, self.get(i))
            y = np.array([f(xi) for xi in X])
            ax.plot(X, y, label=f"basis {i}")

    def reconstruct_velocity_profile(self, alpha, N=100):
        """Reconstruct velocity profile."""
        Z = np.linspace(self.bounds()[0], self.bounds()[1], N)
        u = np.zeros_like(Z)
        for i in range(len(self.basis)):
            b = lambdify(z, self.get(i))
            u[:] += alpha[i] * b(Z)
        return u
    
    def reconstruct_velocity_profile_at(self, alpha, z):
        """Reconstruct velocity profile at."""
        u = 0
        for i in range(len(self.basis)):
            b = lambdify(z, self.eval(i, z))
            u += alpha[i] * b(z)
        return u

    def reconstruct_alpha(self, velocities, z):
        """Reconstruct alpha."""
        n_basis = len(self.basis)
        alpha = np.zeros(n_basis)
        for i in range(n_basis):
            b = lambdify(z, self.eval(i, z))
            nom = np.trapz(velocities * b(z) * self.weight(z), z)
            if type(b(z)) == int:
                den = b(z) ** 2
            else:
                den = np.trapz((b(z) * b(z)).reshape(z.shape), z)
            res = nom / den
            alpha[i] = res
        return alpha
    
    def project_onto_basis(self, Y):
        """Project onto basis."""
        Z = np.linspace(self.bounds()[0], self.bounds()[1], Y.shape[0])
        n_basis = len(self.basis)
        alpha = np.zeros(n_basis)
        z = Symbol("z")
        for i in range(n_basis):
            # b = lambdify(z, self.eval(i, Z))
            b = self.get_lambda(i)
            alpha[i] = np.trapz(Y * b(Z) * self.weight_eval(Z), Z)
        return alpha

    def get_diff_basis(self):
        """Get diff basis."""
        db = [diff(b, z) for i, b in enumerate(self.basis)]
        return db


class Monomials(Basisfunction):
    """Monomials. (class)."""
    name = "Monomials"


class Legendre_shifted(Basisfunction):
    """Legendre shifted. (class)."""
    name = "Legendre_shifted"

    def basis_definition(self):
        """Basis definition."""
        z = Symbol("z")
        b = lambda k, z: legendre(k, 2 * z - 1) * (-1) ** (k)
        return [b(k, z) for k in range(self.level + 1)]
    
class Chebyshevu(Basisfunction):
    """Chebyshevu. (class)."""
    name = "Chebyshevu"
    
    def bounds(self):
        """Bounds."""
        return [-1, 1]
    
    def weight(self, z):
        # do not forget to include the jacobian of the coordinate transformation in the weight
        """Weight."""
        return sympy.sqrt(1-z**2)

    def basis_definition(self):
        """Basis definition."""
        z = Symbol("z")
        b = lambda k, z: sympy.sqrt(2 / sympy.pi) * chebyshevu(k, z)
        return [b(k, z) for k in range(self.level + 1)]
    
class Chebyshevu_shifted(Basisfunction):
    """
    Chebyshev U polynomials shifted to [0,1] with phi_0 = 1.

    phi_k(z) = U_k(2z - 1)  (unnormalized Chebyshev of the second kind)
    weight   = sqrt(z * (1 - z))
    domain   = [0, 1]

    Properties:
    - phi_0 = 1 (recovers SWE at level 0)
    - Orthogonal: M[i,j] = (pi/8) * delta_{ij}
    - Analytical quadrature nodes: z_k = (1 + cos(k*pi/(n+1))) / 2
    """
    name = "Chebyshevu_shifted"

    def bounds(self):
        """Bounds."""
        return [0, 1]

    def weight(self, z):
        """Weight."""
        return sympy.sqrt(z * (1 - z))

    def basis_definition(self):
        """Basis definition."""
        z = Symbol("z")
        return [chebyshevu(k, 2 * z - 1) for k in range(self.level + 1)]

    def quadrature_nodes(self, n=None):
        """
        Gauss-Chebyshev U quadrature on [0,1].

        Integrates integral(f(z) * sqrt(z*(1-z)), 0, 1) exactly for
        polynomial f of degree <= 2*n + 1.

        Uses n+1 nodes. Default n = 2*(level+1) for safety with triple products.
        """
        if n is None:
            n = 2 * (self.level + 1)
        import numpy as np
        k = np.arange(1, n + 2)
        nodes_ref = np.cos(k * np.pi / (n + 2))
        weights_ref = np.pi / (n + 2) * np.sin(k * np.pi / (n + 2))**2
        nodes_01 = 0.5 * (1 + nodes_ref)
        weights_01 = weights_ref / 4
        return nodes_01, weights_01


class Legendre_DN(Basisfunction):
    """Legendre DN. (class)."""
    name = "Legendre_DN - satifying no-slip and no-stress. This is a non-SWE basis"

    def bounds(self):
        """Bounds."""
        return [-1, 1]

    def basis_definition(self):
        """Basis definition."""
        z = Symbol("z")
        def b(k, z):
            """B."""
            alpha = sympy.Rational((2*k+3), (k+2)**2)
            beta = -sympy.Rational((k+1),(k+2))**2
            return (legendre(k, z) ) + alpha * (legendre(k+1, z) ) + beta * (legendre(k+2, z))
        #normalizing makes no sence, as b(k, 0) = 0 by construction
        return [b(k, z) for k in range(self.level + 1)]


class GalerkinBasis(Basisfunction):
    """
    General BC-aware basis via Shen-type recombination.

    Constructs basis functions from a parent polynomial family where:
    - phi_0 = 1 (constant, for mass conservation / SWE limit)
    - phi_k (k >= 1) are linear combinations of parent polynomials that
      automatically satisfy user-specified linear boundary conditions

    Supported BC types at bottom (z=bounds[0]) and top (z=bounds[1]):
    - 'free':     no constraint (free surface, du/dz = 0 implied by no BC)
    - 'noslip':   u(z_bc) = 0
    - 'nostress': du/dz(z_bc) = 0
    - 'slip':     du/dz(z_bc) = u(z_bc) / slip_length

    The BC-satisfying basis is built by solving a small linear system
    for each k, combining parent polynomials P_k, P_{k+1}, P_{k+2}
    with coefficients that enforce the two BCs.

    Usage:
        basis = GalerkinBasis(level=3, parent='legendre', bc_bottom='noslip', bc_top='nostress')
        basis = GalerkinBasis(level=2, parent='chebyshev', bc_bottom='slip', bc_top='free',
                              slip_length=0.5)
    """
    name = "GalerkinBasis"

    def __init__(self, level=0, parent="legendre", bc_bottom="slip", bc_top="nostress",
                 slip_length=None, **kwargs):
        self._parent = parent
        self._bc_bottom = bc_bottom
        self._bc_top = bc_top
        self._slip_length = slip_length
        super().__init__(level=level, **kwargs)
        self.name = f"Galerkin_{parent}_{bc_bottom}_{bc_top}"

    def bounds(self):
        return [0, 1]

    def weight(self, z):
        return sympy.Integer(1)

    def _parent_poly(self, k, z):
        if self._parent == "legendre":
            return legendre(k, 2 * z - 1) * (-1) ** k
        elif self._parent == "chebyshev":
            return chebyshevu(k, 2 * z - 1)
        else:
            raise ValueError(f"Unknown parent polynomial: {self._parent}")

    def _bc_equation(self, poly, z_bc, bc_type):
        """
        Return the linear constraint for a BC at z_bc.
        Returns (value_coeff, deriv_coeff) such that:
            value_coeff * phi(z_bc) + deriv_coeff * phi'(z_bc) = 0
        """
        z = Symbol("z")
        val = poly.subs(z, z_bc)
        deriv_val = diff(poly, z).subs(z, z_bc)

        if bc_type == "noslip":
            return val  # phi(z_bc) = 0
        elif bc_type == "nostress":
            return deriv_val  # phi'(z_bc) = 0
        elif bc_type == "slip":
            lam = self._slip_length if self._slip_length is not None else sympy.Symbol("slip_length")
            return deriv_val - val / lam  # phi'(z_bc) = phi(z_bc) / lambda
        elif bc_type == "free":
            return None  # no constraint
        else:
            raise ValueError(f"Unknown BC type: {bc_type}")

    def basis_definition(self):
        z = Symbol("z")
        basis = [sympy.Integer(1)]  # phi_0 = 1 always

        z_bot = self.bounds()[0]
        z_top = self.bounds()[1]

        # Count constraints
        constraints = []
        if self._bc_bottom != "free":
            constraints.append(("bottom", z_bot, self._bc_bottom))
        if self._bc_top != "free":
            constraints.append(("top", z_top, self._bc_top))

        n_constraints = len(constraints)

        for k in range(1, self.level + 1):
            if n_constraints == 0:
                basis.append(self._parent_poly(k, z))
            elif n_constraints == 1:
                # Combine P_k and P_{k+1} to satisfy 1 BC
                P_k = self._parent_poly(k, z)
                P_k1 = self._parent_poly(k + 1, z)
                _, z_bc, bc_type = constraints[0]
                c_k = self._bc_equation(P_k, z_bc, bc_type)
                c_k1 = self._bc_equation(P_k1, z_bc, bc_type)
                if c_k1 == 0:
                    basis.append(P_k)
                else:
                    alpha = -c_k / c_k1
                    basis.append(sympy.simplify(P_k + alpha * P_k1))
            elif n_constraints == 2:
                # Combine P_k, P_{k+1}, P_{k+2} to satisfy 2 BCs
                P_k = self._parent_poly(k, z)
                P_k1 = self._parent_poly(k + 1, z)
                P_k2 = self._parent_poly(k + 2, z)

                _, z_bc0, bc0 = constraints[0]
                _, z_bc1, bc1 = constraints[1]

                c0_k = self._bc_equation(P_k, z_bc0, bc0)
                c0_k1 = self._bc_equation(P_k1, z_bc0, bc0)
                c0_k2 = self._bc_equation(P_k2, z_bc0, bc0)

                c1_k = self._bc_equation(P_k, z_bc1, bc1)
                c1_k1 = self._bc_equation(P_k1, z_bc1, bc1)
                c1_k2 = self._bc_equation(P_k2, z_bc1, bc1)

                A = sympy.Matrix([[c0_k1, c0_k2], [c1_k1, c1_k2]])
                b_vec = sympy.Matrix([-c0_k, -c1_k])

                try:
                    coeffs = A.solve(b_vec)
                    alpha = coeffs[0]
                    beta = coeffs[1]
                    phi_k = sympy.simplify(P_k + alpha * P_k1 + beta * P_k2)
                    basis.append(phi_k)
                except Exception:
                    basis.append(P_k)

        return basis


class SplineBasis(Basisfunction):
    """
    B-spline basis on [0,1] using raw B-splines (nodal DOFs).

    phi_k = B_k (k-th B-spline hat function)
    - phi_0 peaks at z=0 (bottom): phi_0(0) = 1
    - phi_{n-1} peaks at z=1 (top): phi_{n-1}(1) = 1
    - Interior phi_k peak at internal knots

    The partition of unity (sum B_k = 1) means:
    - Depth-averaged velocity = weighted sum of nodal velocities
    - Mass flux = h * sum(alpha_k * integral(B_k))
    - Bottom velocity = alpha_0 (direct nodal access)

    Provides get_knot_spans() for piecewise integration.
    """
    name = "SplineBasis"

    def __init__(self, level=0, degree=1, **kwargs):
        self._degree = degree
        super().__init__(level=level, **kwargs)

    def bounds(self):
        return [0, 1]

    def weight(self, z):
        return sympy.Integer(1)

    def _make_knots(self):
        n = self.level + 1
        n_internal = max(n - self._degree, 1)
        internal = [sympy.Rational(i, n_internal) for i in range(n_internal + 1)]
        knots = [sympy.Rational(0)] * (self._degree + 1) + internal[1:-1] + [sympy.Rational(1)] * (self._degree + 1)
        return knots

    def basis_definition(self):
        z_sym = Symbol("z")
        n = self.level + 1
        knots = self._make_knots()

        raw_basis = bspline_basis_set(self._degree, knots, z_sym)

        # Use raw B-splines directly — phi_0 = B_0 (hat at bottom)
        result = list(raw_basis[:n])

        while len(result) < n:
            result.append(sympy.Integer(0))
        return result[:n]

    def get_knot_spans(self):
        """Return list of (a, b) intervals between consecutive knot values."""
        knots = self._make_knots()
        unique_knots = sorted(set(float(k) for k in knots))
        spans = []
        for i in range(len(unique_knots) - 1):
            a, b = unique_knots[i], unique_knots[i + 1]
            if b > a:
                spans.append((sympy.Rational(a).limit_denominator(10000),
                              sympy.Rational(b).limit_denominator(10000)))
        return spans

    def mean_coefficients(self):
        """B-splines form a partition of unity: sum(B_k) = 1, so c_k = 1 for all k."""
        return [sympy.Integer(1)] * (self.level + 1)


class Spline(Basisfunction):
    """Spline. (class). Legacy — use SplineBasis instead."""
    name = "Spline"

    def basis_definition(self, degree=1, knots=[0, 0, 0.001, 1, 1]):
        """Basis definition."""
        z = Symbol("z")
        basis = bspline_basis_set(degree, knots, z)
        return basis


class OrthogonalSplineWithConstant(Basisfunction):
    """OrthogonalSplineWithConstant. (class)."""
    name = "OrthogonalSplineWithConstant"

    def basis_definition(self, degree=1, knots=[0, 0, 0.5, 1, 1]):
        """Basis definition."""
        z = Symbol("z")

        def prod(u, v):
            """Prod."""
            return integrate(u * v, (z, 0, 1))

        basis = bspline_basis_set(degree, knots, z)
        add_basis = [1]
        # add_basis = [sympy.Piecewise((0, z<0.1), (1, True))]
        basis = add_basis + basis[:-1]
        orth = deepcopy(basis)
        for i in range(1, len(orth)):
            for j in range(0, i):
                orth[i] -= prod(basis[i], orth[j]) / prod(orth[j], orth[j]) * orth[j]

        for i in range(len(orth)):
            orth[i] /= sympy.sqrt(prod(orth[i], orth[i]))

        return orth
