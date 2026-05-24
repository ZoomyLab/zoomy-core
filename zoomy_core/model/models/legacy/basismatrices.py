"""Module `zoomy_core.model.models.basismatrices`."""

import os
import numpy as np
import sympy
from sympy import integrate, diff, Matrix
from sympy.abc import z
from time import time as get_time


from scipy.optimize import least_squares as lsq

from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.misc import misc as misc
from zoomy_core.model.models.basis_cache import BasisMatrixCache, MATRIX_NAMES


class Basismatrices:
    """Basismatrices. (class)."""
    def __init__(self, basis=Legendre_shifted(), use_cache=True, cache_path=".cache",
                 use_unit_weight=False):
        self.basisfunctions = basis
        self.use_cache = use_cache
        self.use_unit_weight = use_unit_weight  # kept for backward compat, ignored
        self.cache_dir = cache_path
        self._cache = BasisMatrixCache(
            cache_root=os.path.join(cache_path, "basismatrices")
        )

    def _compute_matrices(self, level):
        """Internal helper `_compute_matrices`."""
        start = get_time()
        # object is key here, as we need to have a symbolic representation of the fractions.
        self.phib = np.empty((level + 1), dtype=object)
        self.M = np.empty((level + 1, level + 1), dtype=object)
        self.A = np.empty((level + 1, level + 1, level + 1), dtype=object)
        self.B = np.empty((level + 1, level + 1, level + 1), dtype=object)
        self.D = np.empty((level + 1, level + 1), dtype=object)
        self.Dxi = np.empty((level + 1, level + 1), dtype=object)
        self.Dxi2 = np.empty((level + 1, level + 1), dtype=object)

        self.DD = np.empty((level + 1, level + 1), dtype=object)
        self.D1 = np.empty((level + 1, level + 1), dtype=object)
        self.DT = np.empty((level + 1, level + 1, level + 1), dtype=object)

        for k in range(level + 1):
            self.phib[k] = self._phib(k)
            for i in range(level + 1):
                self.M[k, i] = self._M(k, i)
                self.D[k, i] = self._D(k, i)
                self.Dxi[k, i] = self._Dxi(k, i)
                self.Dxi2[k, i] = self._Dxi2(k, i)

                self.DD[k, i] = self._DD(k, i)
                self.D1[k, i] = self._D1(k, i)
                for j in range(level + 1):
                    self.A[k, i, j] = self._A(k, i, j)
                    self.B[k, i, j] = self._B(k, i, j)
                    self.DT[k, i, j] = self._DT(k, i, j)
            

    def _apply_mass_inverse(self, level):
        """
        Pre-multiply A, B, D, DD, D1, DT by M^-1 and set M to identity.

        For diagonal M: M^-1 is trivially 1/M[k,k] per row (fast path).
        For non-diagonal M: full symbolic inverse via SymPy.
        """
        n = level + 1
        M_sympy = Matrix([[self.M[i, j] for j in range(n)] for i in range(n)])

        is_diagonal = all(self.M[i, j] == 0 for i in range(n) for j in range(n) if i != j)
        if is_diagonal:
            Minv = Matrix([[1 / self.M[i, i] if i == j else 0 for j in range(n)] for i in range(n)])
        else:
            Minv = M_sympy.inv()

        def apply_Minv_2d(mat):
            result = np.empty_like(mat)
            for i in range(n):
                for j in range(n):
                    result[i, j] = sum(Minv[i, l] * mat[l, j] for l in range(n))
            return result

        def apply_Minv_3d(mat):
            result = np.empty_like(mat)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        result[i, j, k] = sum(Minv[i, l] * mat[l, j, k] for l in range(n))
            return result

        self.A = apply_Minv_3d(self.A)
        self.B = apply_Minv_3d(self.B)
        self.DT = apply_Minv_3d(self.DT)
        self.D = apply_Minv_2d(self.D)
        self.Dxi = apply_Minv_2d(self.Dxi)
        self.Dxi2 = apply_Minv_2d(self.Dxi2)
        self.DD = apply_Minv_2d(self.DD)
        self.D1 = apply_Minv_2d(self.D1)

        self.Minv_raw = np.array([[Minv[i, j] for j in range(n)] for i in range(n)], dtype=object)

        for i in range(n):
            for j in range(n):
                self.M[i, j] = sympy.Integer(1) if i == j else sympy.Integer(0)

    def _collect_matrices_as_dict(self):
        return {name: getattr(self, name) for name in MATRIX_NAMES if hasattr(self, name)}

    def _set_matrices_from_dict(self, matrices):
        for name, arr in matrices.items():
            setattr(self, name, arr)

    def compute_matrices(self, level, numerical=False):
        """
        Compute matrices, using hash-based cache if available.

        Strategy selection:
        1. If basis has get_knot_spans(): use SymbolicIntegrator with knot_spans strategy
        2. If numerical=True: use numerical quadrature
        3. Otherwise: use default sympy.integrate with caching
        """
        if hasattr(self.basisfunctions, "get_knot_spans") and self.basisfunctions.get_knot_spans():
            self._compute_via_integrator(level)
            return

        if numerical:
            self._compute_matrices_numerical(level)
            return

        if not self.use_cache:
            self._compute_matrices(level)
            return

        def _compute_and_collect(lvl):
            self._compute_matrices(lvl)
            return self._collect_matrices_as_dict()

        matrices = self._cache.get_or_compute(
            self.basisfunctions, level, _compute_and_collect
        )
        self._set_matrices_from_dict(matrices)

    def _compute_via_integrator(self, level):
        """Use SymbolicIntegrator for bases with special integration capabilities."""
        from zoomy_core.model.models.symbolic_integrator import SymbolicIntegrator
        integrator = SymbolicIntegrator(self.basisfunctions)
        matrices = integrator.compute_all_matrices(level)
        for name in MATRIX_NAMES:
            if name in matrices:
                setattr(self, name, matrices[name])

    def _compute_matrices_numerical(self, level):
        """Compute basis matrices via numerical quadrature."""
        from sympy import Symbol, lambdify

        nodes, weights = self.basisfunctions.quadrature_nodes()
        z_sym = Symbol("z")
        n = level + 1

        phi_fns = [lambdify(z_sym, self.basisfunctions.get(k), "numpy") for k in range(n)]
        dphi_fns = [lambdify(z_sym, diff(self.basisfunctions.get(k), z), "numpy") for k in range(n)]

        phi_at = np.array([np.broadcast_to(np.asarray(fn(nodes), dtype=float), nodes.shape) for fn in phi_fns])
        dphi_at = np.array([np.broadcast_to(np.asarray(fn(nodes), dtype=float), nodes.shape) for fn in dphi_fns])

        from sympy import Rational as R

        self.phib = np.empty(n, dtype=object)
        self.M = np.empty((n, n), dtype=object)
        self.A = np.empty((n, n, n), dtype=object)
        self.B = np.empty((n, n, n), dtype=object)
        self.D = np.empty((n, n), dtype=object)
        self.Dxi = np.empty((n, n), dtype=object)
        self.Dxi2 = np.empty((n, n), dtype=object)
        self.DD = np.empty((n, n), dtype=object)
        self.D1 = np.empty((n, n), dtype=object)
        self.DT = np.empty((n, n, n), dtype=object)

        for k in range(n):
            self.phib[k] = R(self.basisfunctions.eval(k, self.basisfunctions.bounds()[0]))

        def _to_sympy(val):
            """Convert numerical value to clean SymPy number."""
            import sympy
            r = sympy.nsimplify(val, tolerance=1e-12, rational=False)
            return r

        for k in range(n):
            for i in range(n):
                self.M[k, i] = _to_sympy(np.sum(weights * phi_at[k] * phi_at[i]))
                self.D[k, i] = _to_sympy(np.sum(weights * dphi_at[k] * dphi_at[i]))
                self.Dxi[k, i] = _to_sympy(np.sum(weights * dphi_at[k] * dphi_at[i] * nodes))
                self.Dxi2[k, i] = _to_sympy(np.sum(weights * dphi_at[k] * dphi_at[i] * nodes**2))
                self.DD[k, i] = _to_sympy(0)
                self.D1[k, i] = _to_sympy(np.sum(weights * phi_at[k] * dphi_at[i]))
                for j in range(n):
                    self.A[k, i, j] = _to_sympy(np.sum(weights * phi_at[k] * phi_at[i] * phi_at[j]))
                    self.B[k, i, j] = _to_sympy(0)
                    self.DT[k, i, j] = _to_sympy(np.sum(weights * dphi_at[k] * dphi_at[i] * phi_at[j]))

    def enforce_boundary_conditions_lsq(self, rhs=np.zeros(2), dim=1):
        """Enforce boundary conditions lsq."""
        level = len(self.basisfunctions.basis) - 1
        constraint_bottom = [self.basisfunctions.eval(i, 0.0) for i in range(level + 1)]
        constraint_top = [
            diff(self.basisfunctions.eval(i, z), z).subs(z, 1.0)
            for i in range(level + 1)
        ]
        A = Matrix([constraint_bottom, constraint_top])

        I = np.linspace(0, level, 1 + level, dtype=int)
        I_enforce = I[1:]
        rhs = np.zeros(2)
        # rhs = np.zeros(level)
        I_free = np.delete(I, I_enforce)
        A_enforce = A[:, list(I_enforce)]
        A_free = np.array(A[:, list(I_free)], dtype=float)
        AtA = A_enforce.T @ A_enforce
        reg = 10 ** (-6)
        A_enforce_inv = np.array((AtA + reg * np.eye(AtA.shape[0])).inv(), dtype=float)

        def f_1d(Q):
            """F 1d."""
            for i, q in enumerate(Q.T):
                # alpha_enforce = q[I_enforce+1]
                alpha_free = q[I_free + 1]
                b = rhs - np.dot(A_free, alpha_free)
                # b = rhs
                result = np.dot(A_enforce_inv, A_enforce.T @ b)
                alpha = 1.0
                Q[I_enforce + 1, i] = (1 - alpha) * Q[I_enforce + 1, i] + (
                    alpha
                ) * result
            return Q

        def f_2d(Q):
            """F 2d."""
            i1 = [[0] + [i + 1 for i in range(1 + level)]]
            i2 = [[0] + [i + 1 + 1 + level for i in range(1 + level)]]
            Q1 = Q[i1]
            Q2 = Q[i2]
            Q1 = f_1d(Q1)
            Q2 = f_1d(Q2)
            Q[i1] = Q1
            Q[i2] = Q2
            return Q

        if dim == 1:
            return f_1d
        elif dim == 2:
            return f_2d
        else:
            assert False

    def enforce_boundary_conditions_lsq2(self, rhs=np.zeros(2), dim=1):
        """Enforce boundary conditions lsq2."""
        level = len(self.basisfunctions.basis) - 1
        constraint_bottom = [self.basisfunctions.eval(i, 0.0) for i in range(level + 1)]
        constraint_top = [
            diff(self.basisfunctions.eval(i, z), z).subs(z, 1.0)
            for i in range(level + 1)
        ]
        A = Matrix([constraint_bottom, constraint_top])

        I = np.linspace(0, level, 1 + level, dtype=int)
        I_enforce = I[1:]
        rhs = np.zeros(2)
        # rhs = np.zeros(level)
        I_free = np.delete(I, I_enforce)
        A_enforce = A[:, list(I_enforce)]
        A_free = np.array(A[:, list(I_free)], dtype=float)

        def obj(alpha0, lam):
            """Obj."""
            def f(alpha):
                """F."""
                return np.sum((alpha - alpha0) ** 2) + lam * np.sum(
                    np.array(np.dot(A, alpha) ** 2, dtype=float)
                )

            return f

        def f_1d(Q):
            """F 1d."""
            for i, q in enumerate(Q.T):
                h = q[0]
                alpha = q[1:] / h
                f = obj(alpha, 0.1)
                result = lsq(f, alpha)
                Q[1:, i] = h * result.z
            return Q

        def f_2d(Q):
            """F 2d."""
            i1 = [[0] + [i + 1 for i in range(1 + level)]]
            i2 = [[0] + [i + 1 + 1 + level for i in range(1 + level)]]
            Q1 = Q[i1]
            Q2 = Q[i2]
            Q1 = f_1d(Q1)
            Q2 = f_1d(Q2)
            Q[i1] = Q1
            Q[i2] = Q2
            return Q

        if dim == 1:
            return f_1d
        elif dim == 2:
            return f_2d
        else:
            assert False

    def enforce_boundary_conditions(
        self, enforced_basis=[-2, -1], rhs=np.zeros(2), dim=1
    ):
        """Enforce boundary conditions."""
        level = len(self.basisfunctions.basis) - 1
        constraint_bottom = [self.basisfunctions.eval(i, 0.0) for i in range(level + 1)]
        constraint_top = [
            diff(self.basisfunctions.eval(i, z), z).subs(z, 1.0)
            for i in range(level + 1)
        ]
        A = Matrix([constraint_bottom, constraint_top][: len(enforced_basis)])

        # test to only constrain bottom
        # A = Matrix([constraint_bottom])
        # enforced_basis = [-1]
        # rhs=np.zeros(1)

        I = np.linspace(0, level, 1 + level, dtype=int)
        I_enforce = I[enforced_basis]
        I_free = np.delete(I, I_enforce)
        A_enforce = A[:, list(I_enforce)]
        A_free = np.array(A[:, list(I_free)], dtype=float)
        A_enforce_inv = np.array(A_enforce.inv(), dtype=float)

        def f_1d(Q):
            """F 1d."""
            for i, q in enumerate(Q.T):
                alpha_enforce = q[I_enforce + 1]
                alpha_free = q[I_free + 1]
                b = rhs - np.dot(A_free, alpha_free)
                result = np.dot(A_enforce_inv, b)
                alpha = 1.0
                Q[I_enforce + 1, i] = (1 - alpha) * Q[I_enforce + 1, i] + (
                    alpha
                ) * result
            return Q

        def f_2d(Q):
            """F 2d."""
            i1 = [[0] + [i + 1 for i in range(1 + level)]]
            i2 = [[0] + [i + 1 + 1 + level for i in range(1 + level)]]
            Q1 = Q[i1]
            Q2 = Q[i2]
            Q1 = f_1d(Q1)
            Q2 = f_1d(Q2)
            Q[i1] = Q1
            Q[i2] = Q2
            return Q

        if dim == 1:
            return f_1d
        elif dim == 2:
            return f_2d
        else:
            assert False
            
    """ 
    Compute phi_k(@xi=0)
    """

    def _phib(self, k):
        """Internal helper `_phib`."""
        return self.basisfunctions.eval(k, self.basisfunctions.bounds()[0])

    def _integration_weight(self, z_sym):
        """Always weight=1 for integration. The basis weight is not used."""
        return sympy.Integer(1)

    """
    Compute <phi_k, phi_i>
    """

    def _M(self, k, i):
        """Internal helper `_M`."""
        return integrate(
            self._integration_weight(z) * self.basisfunctions.eval(k, z) * self.basisfunctions.eval(i, z), (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1])
        )

    """ 
    Compute <phi_k, phi_i, phi_j>
    """

    def _A(self, k, i, j):
        """Internal helper `_A`."""
        return integrate(
            self._integration_weight(z) * self.basisfunctions.eval(k, z)
            * self.basisfunctions.eval(i, z)
            * self.basisfunctions.eval(j, z),
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )

    """ 
    Compute <(phi')_k, phi_j, int(phi)_j>
    """

    def _B(self, k, i, j):
        """Internal helper `_B`."""
        return integrate(
            self._integration_weight(z) * diff(self.basisfunctions.eval(k, z), z)
            * integrate(self.basisfunctions.eval(j, z), z)
            * self.basisfunctions.eval(i, z),
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )

    """ 
    Compute <(phi')_k, (phi')_j>
    """

    def _D(self, k, i):
        """Internal helper `_D`."""
        return integrate(
            self._integration_weight(z) * diff(self.basisfunctions.eval(k, z), z)
            * diff(self.basisfunctions.eval(i, z), z),
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )
        
    """ 
    Compute <(phi')_k, (phi')_j * xi>
    """
    def _Dxi(self, k, i):
        """Internal helper `_Dxi`."""
        return integrate(
            self._integration_weight(z) * diff(self.basisfunctions.eval(k, z), z)
            * diff(self.basisfunctions.eval(i, z), z) * z,
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )
        """ 
    Compute <(phi')_k, (phi')_j * xi**2>
    """
    def _Dxi2(self, k, i):
        """Internal helper `_Dxi2`."""
        return integrate(
            self._integration_weight(z) * diff(self.basisfunctions.eval(k, z), z)
            * diff(self.basisfunctions.eval(i, z), z) * z * z,
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )

    """ 
    Compute <(phi)_k, (phi')_j>
    """

    def _D1(self, k, i):
        """Internal helper `_D1`."""
        return integrate(
            self._integration_weight(z) * self.basisfunctions.eval(k, z) * diff(self.basisfunctions.eval(i, z), z),
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )

    """ 
    Compute <(phi)_k, (phi'')_j>
    """

    def _DD(self, k, i):
        """Internal helper `_DD`."""
        return integrate(
            self._integration_weight(z) * self.basisfunctions.eval(k, z)
            * diff(diff(self.basisfunctions.eval(i, z), z), z),
           (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )

    """ 

    Compute <(phi')_k, (phi')_j>
    """

    def _DT(self, k, i, j):
        """Internal helper `_DT`."""
        return integrate(
            self._integration_weight(z) * diff(self.basisfunctions.eval(k, z), z)
            * diff(self.basisfunctions.eval(i, z), z)
            * self.basisfunctions.eval(j, z),
            (z, self.basisfunctions.bounds()[0], self.basisfunctions.bounds()[1]),
        )


class BasisNoHOM(Basismatrices):
    """BasisNoHOM. (class)."""
    def _A(self, k, i, j):
        """Internal helper `_A`."""
        count = 0
        # count += float(k > 0)
        count += float(i > 0)
        count += float(j > 0)
        # if count > 1:
        if (i == 0 and j == k) or (j == 0 and i == k) or (k == 0 and i == j):
            return super()._A(k, i, j)
        return 0

    def _B(self, k, i, j):
        """Internal helper `_B`."""
        count = 0
        # count += float(k > 0)
        count += float(i > 0)
        count += float(j > 0)
        # if count > 1:
        # if not (i==0 or j==0):
        if (i == 0 and j == k) or (j == 0 and i == k) or (k == 0 and i == j):
            return super()._B(k, i, j)
        return 0
        # return super()._B(k, i, j)
