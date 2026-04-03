"""
Generated Shallow Water Model from automated INS Galerkin projection.

Produces a Model-compatible class with flux(), source(), eigenvalues(), etc.
derived from the 3D Incompressible Navier-Stokes equations via vertical
Galerkin projection with arbitrary basis functions and multi-layer support.

For n_layers=1, this should reproduce the hand-derived ShallowMomentsTopo.
For n_layers>1, it produces new multi-layer equations with interface terms.
"""

import sympy as sp
from sympy import Matrix, MutableDenseNDimArray, Symbol, S, Rational
import param

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.models.basismatrices import Basismatrices
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Basisfunction


class GeneratedShallowModel(Model):
    """
    Automatically generated shallow water moment equations.

    State vector layout (1D):
        Q = [b, h, h*u_{0,0}, ..., h*u_{0,L}, h*u_{1,0}, ..., h*u_{N-1,L}]
        where N = n_layers, L = level

    State vector layout (2D):
        Q = [b, h, {h*alpha per layer}, {h*beta per layer}]

    For n_layers=1, this is identical to ShallowMomentsTopo:
        Q = [b, h, h*alpha_0, ..., h*alpha_L, h*beta_0, ..., h*beta_L]
    """

    n_layers = param.Integer(default=1)
    level = param.Integer(default=0)
    basis_type = param.ClassSelector(
        class_=Basisfunction, default=Legendre_shifted, is_instance=False
    )
    # weight_mode kept for backward compatibility — always uses weight=1 now
    weight_mode = param.Selector(default="orthogonal", objects=["orthogonal", "physical"])

    def _n_moment_dofs(self):
        return self.n_layers * (self.level + 1)

    def _compute_variable_count(self):
        return 2 + self.dimension * self._n_moment_dofs()

    variables = param.Parameter(default=_compute_variable_count)
    aux_variables = param.Parameter(default=0)

    parameters = param.Parameter(
        default={
            "g": (9.81, "positive"),
            "eps": (1e-6, "positive"),
            "ez": (1.0, "positive"),
            "rho": (1000.0, "positive"),
            "lamda": (0.1, "positive"),
        }
    )

    def __init__(self, init_functions=True, **kwargs):
        super().__init__(init_functions=False, **kwargs)
        old_h = self.variables[1]
        new_h = Symbol(old_h.name, positive=True, real=True)
        self.variables[1] = new_h
        if init_functions:
            self._initialize_functions()

    def _initialize_derived_properties(self):
        self.basisfunctions = self.basis_type(level=self.level)
        self.basismatrices = Basismatrices(self.basisfunctions)
        self.basismatrices.compute_matrices(self.level)
        self._layer_interfaces = [
            Rational(k, self.n_layers) for k in range(self.n_layers + 1)
        ]
        self._compute_mass_inverse()
        super()._initialize_derived_properties()
        old_h = self.variables[1]
        new_h = Symbol(old_h.name, positive=True, real=True)
        self.variables[1] = new_h

    def get_primitives(self):
        n_mom = self.level + 1
        n_layers = self.n_layers
        b = self.variables[0]
        h = self.variables[1]
        hinv = 1 / h

        moments_u = []
        idx = 2
        for k in range(n_layers):
            layer_alpha = []
            for j in range(n_mom):
                layer_alpha.append(self.variables[idx] * hinv)
                idx += 1
            moments_u.append(layer_alpha)

        moments_v = []
        if self.dimension > 1:
            for k in range(n_layers):
                layer_beta = []
                for j in range(n_mom):
                    layer_beta.append(self.variables[idx] * hinv)
                    idx += 1
                moments_v.append(layer_beta)
        else:
            for k in range(n_layers):
                moments_v.append([S.Zero] * n_mom)

        return b, h, moments_u, moments_v, hinv

    def _compute_mass_inverse(self):
        """Compute M^-1 once. Fast path for diagonal M."""
        n = self.level + 1
        M = self.basismatrices.M
        is_diag = all(M[i, j] == 0 for i in range(n) for j in range(n) if i != j)
        if is_diag:
            self._Minv = [[Rational(1, 1) / M[i, i] if i == j else S.Zero
                           for j in range(n)] for i in range(n)]
        else:
            from sympy import Matrix as SpMatrix
            M_sp = SpMatrix([[M[i, j] for j in range(n)] for i in range(n)])
            Minv_sp = M_sp.inv()
            self._Minv = [[Minv_sp[i, j] for j in range(n)] for i in range(n)]

    def _apply_Minv(self, vec, k):
        """Apply row k of M^-1 to a vector: sum_l Minv[k,l] * vec[l]."""
        n = self.level + 1
        return sum(self._Minv[k][l] * vec[l] for l in range(n))

    def _layer_weight(self, k):
        """Fraction of total depth occupied by layer k."""
        return self._layer_interfaces[k + 1] - self._layer_interfaces[k]

    def flux(self):
        """
        Flux F(Q) with M^-1 applied.
        The raw projected flux for test function phi_k is:
          F_k^raw = sum_{i,j} h * w * alpha_i * alpha_j * A[k,i,j]
        After M^-1: F_k = sum_l Minv[k,l] * F_l^raw
        """
        dim = self.dimension
        lvl = self.level
        n_vars = self.n_variables
        n_layers = self.n_layers
        n_mom = lvl + 1
        A = self.basismatrices.A
        c_mean = self.basisfunctions.mean_coefficients()

        b, h, moments_u, moments_v, hinv = self.get_primitives()

        F = Matrix.zeros(n_vars, dim)

        for lk in range(n_layers):
            w_k = self._layer_weight(lk)
            alpha = moments_u[lk]
            beta = moments_v[lk]

            F[1, 0] += h * sum(c_mean[k] * alpha[k] for k in range(n_mom)) * w_k

            row_base_u = 2 + lk * n_mom

            # Raw projected flux per test function l
            F_raw_u = [S.Zero] * n_mom
            for l in range(n_mom):
                for i in range(n_mom):
                    for j in range(n_mom):
                        F_raw_u[l] += h * w_k * alpha[i] * alpha[j] * A[l, i, j]

            # Apply M^-1
            for k in range(n_mom):
                F[row_base_u + k, 0] += self._apply_Minv(F_raw_u, k)

            if dim == 2:
                F[1, 1] += h * sum(c_mean[k] * beta[k] for k in range(n_mom)) * w_k

                row_base_v = 2 + n_layers * n_mom + lk * n_mom

                F_raw_vv = [S.Zero] * n_mom
                F_raw_vu = [S.Zero] * n_mom
                F_raw_uv = [S.Zero] * n_mom
                for l in range(n_mom):
                    for i in range(n_mom):
                        for j in range(n_mom):
                            F_raw_vv[l] += h * w_k * beta[i] * beta[j] * A[l, i, j]
                            F_raw_vu[l] += h * w_k * beta[i] * alpha[j] * A[l, i, j]
                            F_raw_uv[l] += h * w_k * alpha[i] * beta[j] * A[l, i, j]

                for k in range(n_mom):
                    F[row_base_v + k, 1] += self._apply_Minv(F_raw_vv, k)
                    F[row_base_v + k, 0] += self._apply_Minv(F_raw_vu, k)
                    F[row_base_u + k, 1] += self._apply_Minv(F_raw_uv, k)

        return ZArray(F)

    def hydrostatic_pressure(self):
        """Hydrostatic pressure with M^-1 applied."""
        dim = self.dimension
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        p = self.parameters
        b, h, moments_u, moments_v, hinv = self.get_primitives()
        c_mean = self.basisfunctions.mean_coefficients()

        F = Matrix.zeros(n_vars, dim)

        # Raw: P_l = g*ez*h^2/2 * c_mean[l] (pressure projects onto same space as mass)
        raw_p = [p.g * p.ez * h**2 / 2 * c_mean[l] for l in range(n_mom)]
        for k in range(n_mom):
            F[2 + k, 0] = self._apply_Minv(raw_p, k)

        if dim == 2:
            row_v0 = 2 + n_layers * n_mom
            for k in range(n_mom):
                F[row_v0 + k, 1] = self._apply_Minv(raw_p, k)

        return ZArray(F)

    def nonconservative_matrix(self):
        """Non-conservative matrix B with M^-1 applied row-wise."""
        dim = self.dimension
        lvl = self.level
        n_vars = self.n_variables
        n_layers = self.n_layers
        n_mom = lvl + 1
        p = self.parameters
        B_mat = self.basismatrices.B

        b, h, moments_u, moments_v, hinv = self.get_primitives()

        nc_x = Matrix.zeros(n_vars, n_vars)
        nc_y = Matrix.zeros(n_vars, n_vars)

        for lk in range(n_layers):
            w_k = self._layer_weight(lk)
            alpha = moments_u[lk]
            beta = moments_v[lk]
            um = alpha[0]
            row_base_u = 2 + lk * n_mom

            # Raw NC entries per test function l, then apply M^-1
            for col in range(n_vars):
                raw_x = [S.Zero] * n_mom
                for l in range(n_mom):
                    # um * delta_{l, col-row_base_u} for l>=1
                    ci = col - row_base_u
                    if 1 <= l == ci:
                        raw_x[l] -= um

                    for i in range(1, n_mom):
                        for j in range(1, n_mom):
                            if col == row_base_u + i:
                                raw_x[l] += alpha[j] * B_mat[l, j, i]

                for k in range(n_mom):
                    nc_x[row_base_u + k, col] += self._apply_Minv(raw_x, k)

            if dim == 2:
                vm = beta[0]
                row_base_v = 2 + n_layers * n_mom + lk * n_mom

                # Build raw NC for y-direction and cross terms
                for col in range(n_vars):
                    raw_ux = [S.Zero] * n_mom
                    raw_uy = [S.Zero] * n_mom
                    raw_vx = [S.Zero] * n_mom
                    raw_vy = [S.Zero] * n_mom

                    for l in range(n_mom):
                        ci_u = col - row_base_u
                        ci_v = col - row_base_v

                        if 1 <= l and l == ci_v:
                            raw_uy[l] -= um
                        if 1 <= l and l == ci_u:
                            raw_vx[l] -= vm
                        if 1 <= l and l == ci_v:
                            raw_vy[l] -= vm

                        for i in range(1, n_mom):
                            for j in range(1, n_mom):
                                if ci_u == i:
                                    term = alpha[j] * B_mat[l, i, j]
                                    raw_ux[l] -= term
                                if ci_v == i:
                                    raw_uy[l] -= alpha[j] * B_mat[l, i, j]
                                if ci_u == i:
                                    raw_vx[l] -= beta[j] * B_mat[l, i, j]
                                if ci_v == i:
                                    raw_vy[l] -= beta[j] * B_mat[l, i, j]

                    for k in range(n_mom):
                        nc_x[row_base_u + k, col] += self._apply_Minv(raw_ux, k)
                        nc_y[row_base_u + k, col] += self._apply_Minv(raw_uy, k)
                        nc_x[row_base_v + k, col] += self._apply_Minv(raw_vx, k)
                        nc_y[row_base_v + k, col] += self._apply_Minv(raw_vy, k)

        A_tensor = MutableDenseNDimArray.zeros(n_vars, n_vars, dim)
        for r in range(n_vars):
            for c in range(n_vars):
                A_tensor[r, c, 0] = nc_x[r, c]
                if dim > 1:
                    A_tensor[r, c, 1] = nc_y[r, c]

        return ZArray(A_tensor)

    def source(self):
        return ZArray.zeros(self.n_variables)

    def newtonian(self):
        """Newtonian viscous source with M^-1 applied."""
        p = self.parameters
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        out = ZArray.zeros(n_vars)
        b, h, moments_u, moments_v, hinv = self.get_primitives()
        D = self.basismatrices.D

        for lk in range(n_layers):
            w_k = self._layer_weight(lk)
            alpha = moments_u[lk]
            beta = moments_v[lk]
            row_base_u = 2 + lk * n_mom

            raw_u = [sum(-p.nu * alpha[i] * hinv * D[i, l] / w_k
                         for i in range(n_mom)) for l in range(n_mom)]
            for k in range(n_mom):
                out[row_base_u + k] += self._apply_Minv(raw_u, k)

            if self.dimension == 2:
                row_base_v = 2 + n_layers * n_mom + lk * n_mom
                raw_v = [sum(-p.nu * beta[i] * hinv * D[i, l] / w_k
                             for i in range(n_mom)) for l in range(n_mom)]
                for k in range(n_mom):
                    out[row_base_v + k] += self._apply_Minv(raw_v, k)
        return out

    def slip(self):
        """
        Navier-slip bottom friction with M^-1 applied.
        Raw projected: S_l^raw = -(1/(lambda*rho)) * u_b * phi_l(0)
        Resolved: S_k = sum_l Minv[k,l] * S_l^raw
        """
        p = self.parameters
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        out = ZArray.zeros(n_vars)
        b, h, moments_u, moments_v, hinv = self.get_primitives()
        phib = self.basismatrices.phib

        for lk in range(n_layers):
            alpha = moments_u[lk]
            beta = moments_v[lk]
            row_base_u = 2 + lk * n_mom

            u_bottom = sum(alpha[i] * phib[i] for i in range(n_mom))
            raw_u = [-Rational(1, 1) / p.lamda / p.rho * u_bottom * phib[l]
                     for l in range(n_mom)]
            for k in range(n_mom):
                out[row_base_u + k] += self._apply_Minv(raw_u, k)

            if self.dimension > 1:
                row_base_v = 2 + n_layers * n_mom + lk * n_mom
                v_bottom = sum(beta[i] * phib[i] for i in range(n_mom))
                raw_v = [-Rational(1, 1) / p.lamda / p.rho * v_bottom * phib[l]
                         for l in range(n_mom)]
                for k in range(n_mom):
                    out[row_base_v + k] += self._apply_Minv(raw_v, k)

        return out

    def _regularize(self, expr):
        if self.level > 1:
            b, h, moments_u, moments_v, hinv = self.get_primitives()
            for lk in range(self.n_layers):
                for d_comp in [moments_u, moments_v]:
                    for i in range(2, self.level + 1):
                        if d_comp[lk][i] != S.Zero:
                            expr = expr.subs(d_comp[lk][i], 0)
        return expr

    def eigenvalues(self):
        if self.eigenvalue_mode == "numerical":
            return ZArray([sp.Integer(0)] * self.n_variables)
        b, h, moments_u, moments_v, hinv = self.get_primitives()
        A = ZArray(self.quasilinear_matrix())
        n = self.normal
        An = ZArray.zeros(self.n_variables, self.n_variables)
        for d in range(self.dimension):
            An[:, :] += A[:, :, d] * n[d]
        An = An.tomatrix()
        An = self._regularize(An)

        lam = sp.Symbol("lam")
        char_poly = An.charpoly(lam)
        evs = sp.solve(char_poly.as_expr(), lam)
        return ZArray([self._simplify(ev) for ev in evs])

    def print_equations(self):
        """Pretty-print the generated equations for inspection."""
        print("=" * 70)
        print(f"Generated Shallow Model")
        print(f"  n_layers={self.n_layers}, level={self.level}, "
              f"dim={self.dimension}, basis={self.basisfunctions.name}")
        print(f"  n_variables={self.n_variables}")
        print(f"  variables: {self.variables.keys()}")
        print("=" * 70)

        b, h, moments_u, moments_v, hinv = self.get_primitives()
        print("\nPrimitive variables:")
        for lk in range(self.n_layers):
            alpha = moments_u[lk]
            print(f"  Layer {lk} u-moments: {alpha}")
            if self.dimension > 1:
                beta = moments_v[lk]
                print(f"  Layer {lk} v-moments: {beta}")

        self.print_model_functions(
            ["flux", "hydrostatic_pressure", "nonconservative_matrix", "source"]
        )
