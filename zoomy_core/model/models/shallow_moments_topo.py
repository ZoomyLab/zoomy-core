import sympy
from sympy import Matrix, MutableDenseNDimArray, Symbol, S, sqrt, Piecewise
import param
import sympy as sp

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.custom_sympy_functions import conditional
from zoomy_core.model.models.basismatrices import Basismatrices
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Basisfunction


class ShallowMomentsTopo(Model):
    """
    Shallow Water Moments with Topography (SMT).
    State vector: Q = [b, h, alpha_0...k, beta_0...k (if 2D)].
    """

    # --- 1. System Configuration (Params) ---
    level = param.Integer(default=0)

    # FIX: Use class_=Basisfunction with is_instance=False.
    # This means: "The value must be a class that inherits from Basisfunction"
    basis_type = param.ClassSelector(
        class_=Basisfunction, default=Legendre_shifted, is_instance=False
    )
    
    def __init__(self, init_functions=True, **kwargs):
        super().__init__(init_functions=False, **kwargs)

        # Enforce positivity for Depth 'h' (Index 1)
        old_h = self.variables[1]
        new_h = Symbol(old_h.name, positive=True, real=True)
        self.variables[1] = new_h
        
        if init_functions:
            self._initialize_functions()

    # --- 2. Dynamic Properties ---

    def _compute_variable_count(self):
        # b(1) + h(1) + dim * (level + 1)
        return 2 + self.dimension * (self.level + 1)

    def _compute_aux_variable_count(self):
        # Original logic: 2 * (2 + dim*(level+1))
        return 2 * (2 + self.dimension * (self.level + 1))

    # Bind the methods so Model._resolve_input can call them
    variables = param.Parameter(default=_compute_variable_count)
    aux_variables = param.Parameter(default=_compute_aux_variable_count)

    # --- 3. Physical Constants ---
    parameters = param.Parameter(
        default={
            "g": (9.81, "positive"),
            "chezy_C": (50.0, "positive"),
            "eps": (1e-6, "positive"),
            "ex": (0.0, "positive"),
            "ey": (0.0, "positive"),
            "ez": (1.0, "positive"),
            "rho": (1000.0, "positive"),
            "nu": (1e-6, "positive"),
            "lamda": (0.1, "positive"),
            "c_slipmod": (0.0, "positive"),
            "l_bl": (0.1, "positive"),
            "l_turb": (0.1, "positive"),
            "kappa": (0.41, "positive"),
        }
    )

    def _initialize_derived_properties(self):
        """
        Orchestrates the setup order:
        1. Initialize Basis (needs level)
        2. Create Symbols (needs variable count, which needs level)
        3. Customize Symbols (positivity)
        4. Register Functions
        """

        # A. Setup Basis Functions & Matrices
        # FIX: Use 'self.basisfunctions' as the name, as expected by project_2d_to_3d
        self.basisfunctions = self.basis_type(level=self.level)
        self.basismatrices = Basismatrices(self.basisfunctions)
        self.basismatrices.compute_matrices(self.level)

        # B. Call Parent to generate self.variables (SymPy symbols)
        # This calls _resolve_input, which calls _compute_variable_count
        super()._initialize_derived_properties()

        # C. Enforce positivity for Depth 'h' (Index 1)
        old_h = self.variables[1]
        new_h = Symbol(old_h.name, positive=True, real=True)
        self.variables[1] = new_h
        
        
    def get_primitives(self):
        """
        Extracts primitive variables.
        Returns:
            b: Bathymetry (Scalar)
            h: Depth (Scalar)
            moments: List of lists [[alpha_0..k], [beta_0..k]]
            hinv: Inverse Depth (Scalar)
        """
        n_moments = self.level + 1

        b = self.variables[0]
        h = self.variables[1]
        hinv = 1 / h

        moments = []

        # Extract Alpha (X-direction moments)
        start_alpha = 2
        end_alpha = 2 + n_moments
        ha = self.variables[start_alpha:end_alpha]
        alpha = [val * hinv for val in ha]
        moments.append(alpha)

        # Extract Beta (Y-direction moments) if 2D
        if self.dimension > 1:
            start_beta = end_alpha
            end_beta = start_beta + n_moments
            hb = self.variables[start_beta:end_beta]
            beta = [val * hinv for val in hb]
            moments.append(beta)
        else:
            moments.append([S.Zero] * n_moments)

        return b, h, moments, hinv

    # --- 4. Conservation Laws ---

    def flux(self):
        dim = self.dimension
        p = self.parameters
        lvl = self.level
        n_vars = self.n_variables
        A_basis = self.basismatrices.A
        M_basis = self.basismatrices.M

        b, h, moments, hinv = self.get_primitives()
        alpha = moments[0]
        beta = moments[1]

        F = Matrix.zeros(n_vars, dim)

        # --- X-Flux (Column 0) ---
        F[1, 0] = h * alpha[0]
        F[2, 0] = p.g * p.ez * h**2 / 2
        for k in range(lvl + 1):
            for i in range(lvl + 1):
                for j in range(lvl + 1):
                    F[2 + k, 0] += (
                        h * alpha[i] * alpha[j] * A_basis[k, i, j] / M_basis[k, k]
                    )

        if dim == 2:
            offset = lvl + 1
            # Coupled terms in X-flux (transport of v-momentum by u-velocity)
            # F_x for Y-momentum equations
            for k in range(lvl + 1):
                for i in range(lvl + 1):
                    for j in range(lvl + 1):
                        # Index: 2 (start) + offset (skip alpha) + k
                        F[2 + offset + k, 0] += (
                            h * beta[i] * alpha[j] * A_basis[k, i, j] / M_basis[k, k]
                        )

            # --- Y-Flux (Column 1) ---
            F[1, 1] = h * beta[0]
            # Pressure part for Y-momentum
            F[2 + offset, 1] = p.g * p.ez * h**2 / 2

            # Y-momentum flux (transport of v by v)
            for k in range(lvl + 1):
                for i in range(lvl + 1):
                    for j in range(lvl + 1):
                        F[2 + offset + k, 1] += (
                            h * beta[i] * beta[j] * A_basis[k, i, j] / M_basis[k, k]
                        )

            # X-momentum flux Y-component (transport of u by v)
            for k in range(lvl + 1):
                for i in range(lvl + 1):
                    for j in range(lvl + 1):
                        F[2 + k, 1] += (
                            h * alpha[i] * beta[j] * A_basis[k, i, j] / M_basis[k, k]
                        )

        return ZArray(F)

    def nonconservative_matrix(self):
        """
        Logic ported directly from reference.
        """
        dim = self.dimension
        lvl = self.level
        n_vars = self.n_variables
        p = self.parameters

        b, h, moments, hinv = self.get_primitives()
        alpha = moments[0]
        beta = moments[1]

        B_basis = self.basismatrices.B
        M_basis = self.basismatrices.M

        # We construct 2D matrices nc_x and nc_y first, then combine into Tensor
        nc_x = Matrix.zeros(n_vars, n_vars)
        nc_y = Matrix.zeros(n_vars, n_vars)

        # --- Logic for Dimension 1 (and part of 2) ---
        um = alpha[0]

        for k in range(1, lvl + 1):
            nc_x[2 + k, 2 + k] -= um

        for k in range(lvl + 1):
            for i in range(1, lvl + 1):
                for j in range(1, lvl + 1):
                    nc_x[2 + k, 2 + i] += alpha[j] * B_basis[k, j, i] / M_basis[k, k]

        if dim == 2:
            offset = lvl + 1
            vm = beta[0]

            for k in range(1, lvl + 1):
                nc_y[2 + k, 2 + k + offset] -= (
                    um
                )

            for k in range(lvl + 1):
                for i in range(1, lvl + 1):
                    for j in range(1, lvl + 1):
                        term = alpha[j] * B_basis[k, i, j] / M_basis[k, k]
                        nc_x[2 + k, 2 + i] -= term 
                        nc_y[2 + k, 2 + i + offset] -= term

            for k in range(1, lvl + 1):
                nc_x[2 + k + offset, 2 + k] -= vm
                nc_y[2 + k + offset, 2 + k + offset] -= vm

            for k in range(lvl + 1):
                for i in range(1, lvl + 1):
                    for j in range(1, lvl + 1):
                        term = beta[j] * B_basis[k, i, j] / M_basis[k, k]
                        nc_x[2 + k + offset, 2 + i] -= term
                        nc_y[2 + k + offset, 2 + i + offset] -= term

        

        nc_x[2, 0] += p.ez * p.g * h 
        if dim == 2:
            offset = lvl + 1
            nc_y[2 + offset, 0] += p.ez * p.g * h 

        # Pack into Tensor
        A_tensor = MutableDenseNDimArray.zeros(n_vars, n_vars, dim)

        for r in range(n_vars):
            for c in range(n_vars):
                A_tensor[r, c, 0] = nc_x[r, c]
                if dim > 1:
                    A_tensor[r, c, 1] = nc_y[r, c]

        return ZArray(A_tensor)

    def source(self):
        return Matrix.zeros(self.n_variables, 1)

    # --- 5. Additional Source Terms ---

    def gravity(self):
        out = Matrix.zeros(self.n_variables, 1)
        out[2] = -self.parameters.g * self.parameters.ex * self.variables[0]
        if self.dimension == 2:
            offset = self.level + 1
            out[2 + offset] = (
                -self.parameters.g * self.parameters.ey * self.variables[0]
            )
        return out

    def newtonian(self):
        # Viscous terms
        p = self.parameters
        out = Matrix.zeros(self.n_variables, 1)
        b, h, moments, hinv = self.get_primitives()
        alpha = moments[0]
        beta = moments[1]
        offset = self.level + 1

        for k in range(1 + self.level):
            for i in range(1 + self.level):
                term_x = (
                    -p.nu
                    * alpha[i]
                    * hinv
                    * self.basismatrices.D[i, k]
                    / self.basismatrices.M[k, k]
                )
                out[2 + k] += term_x

                if self.dimension == 2:
                    term_y = (
                        -p.nu
                        * beta[i]
                        * hinv
                        * self.basismatrices.D[i, k]
                        / self.basismatrices.M[k, k]
                    )
                    out[2 + k + offset] += term_y
        return out

    def slip(self):
        # Navier-slip boundary condition
        p = self.parameters
        out = Matrix.zeros(self.n_variables, 1)
        b, h, moments, hinv = self.get_primitives()
        alpha = moments[0]
        beta = moments[1]
        offset = self.level + 1

        for k in range(1 + self.level):
            for i in range(1 + self.level):
                term_x = -1.0 / p.lamda / p.rho * alpha[i] / self.basismatrices.M[k, k]
                out[2 + k] += term_x

                if self.dimension == 2:
                    term_y = (
                        -1.0 / p.lamda / p.rho * beta[i] / self.basismatrices.M[k, k]
                    )
                    out[2 + k + offset] += term_y
        return out

    def slip_mod(self):
        # Modified slip
        p = self.parameters
        out = Matrix.zeros(self.n_variables, 1)
        b, h, moments, hinv = self.get_primitives()
        alpha = moments[0]
        beta = moments[1]
        offset = self.level + 1

        ub = sum(alpha)  # Sum of alpha_i
        vb = sum(beta) if self.dimension == 2 else 0

        for k in range(1, 1 + self.level):
            term_x = (
                -1.0 * p.c_slipmod / p.lamda / p.rho * ub / self.basismatrices.M[k, k]
            )
            out[2 + k] += term_x

            if self.dimension == 2:
                term_y = (
                    -1.0
                    * p.c_slipmod
                    / p.lamda
                    / p.rho
                    * vb
                    / self.basismatrices.M[k, k]
                )
                out[2 + k + offset] += term_y
        return out

    def chezy(self):
        # Chezy friction
        p = self.parameters
        out = Matrix.zeros(self.n_variables, 1)
        b, h, moments, hinv = self.get_primitives()
        alpha = moments[0]
        beta = moments[1]
        offset = self.level + 1

        # Approximation of |U|
        tmp = 0
        for i in range(1 + self.level):
            for j in range(1 + self.level):
                tmp += alpha[i] * alpha[j]
                if self.dimension == 2:
                    tmp += beta[i] * beta[j]
        u_mag = sympy.sqrt(tmp)

        for k in range(1 + self.level):
            for l in range(1 + self.level):
                term_x = (
                    -1.0
                    / (p.chezy_C**2 * self.basismatrices.M[k, k])
                    * alpha[l]
                    * u_mag
                )
                out[2 + k] += term_x

                if self.dimension == 2:
                    term_y = (
                        -1.0
                        / (p.chezy_C**2 * self.basismatrices.M[k, k])
                        * beta[l]
                        * u_mag
                    )
                    out[2 + k + offset] += term_y
        return out

    def newtonian_turbulent_algebraic(self):
        # Algebraic turbulence model
        p = self.parameters
        out = Matrix.zeros(self.n_variables, 1)
        b, h, moments, hinv = self.get_primitives()
        alpha = moments[0]
        beta = moments[1]
        offset = self.level + 1

        # X-Direction
        dU_dx = alpha[0] / (p.l_turb * h)
        abs_dU_dx = Piecewise((dU_dx, dU_dx >= 0), (-dU_dx, True))

        # Viscosity coefficient
        nu_eff = p.nu + p.kappa * sympy.sqrt(p.nu * abs_dU_dx) * p.l_bl * (1 - p.l_bl)

        for k in range(1 + self.level):
            # Term 1
            out[2 + k] += -nu_eff * dU_dx * self.basismatrices.phib[k] * hinv

            # Term 2 & 3
            for i in range(1 + self.level):
                out[2 + k] += -p.nu * hinv * alpha[i] * self.basismatrices.D[i, k]

                term_turb = (
                    -p.kappa
                    * sympy.sqrt(p.nu * abs_dU_dx)
                    * hinv
                    * alpha[i]
                    * (self.basismatrices.Dxi[i, k] - self.basismatrices.Dxi2[i, k])
                )
                out[2 + k] += term_turb

        # Y-Direction
        if self.dimension == 2:
            dV_dy = beta[0] / (p.l_turb * h)
            abs_dV_dy = Piecewise((dV_dy, dV_dy >= 0), (-dV_dy, True))

            nu_eff_y = p.nu + p.kappa * sympy.sqrt(p.nu * abs_dV_dy) * p.l_bl * (
                1 - p.l_bl
            )

            for k in range(1 + self.level):
                out[2 + k + offset] += (
                    -nu_eff_y * dV_dy * self.basismatrices.phib[k] * hinv
                )

                for i in range(1 + self.level):
                    out[2 + k + offset] += (
                        -p.nu * hinv * beta[i] * self.basismatrices.D[i, k]
                    )

                    term_turb_y = (
                        -p.kappa
                        * sympy.sqrt(p.nu * abs_dV_dy)
                        * hinv
                        * beta[i]
                        * (self.basismatrices.Dxi[i, k] - self.basismatrices.Dxi2[i, k])
                    )
                    out[2 + k + offset] += term_turb_y

        return out

    # --- 6. Visualization ---

    def project_2d_to_3d(self):
        """
        Maps state vector to 3D visualization.
        """
        out = ZArray.zeros(6)
        z = self.position[2]

        # Needs gradient variables (dbdx, dhdx...) typically found in aux_vars
        # Note: In the base Model, self.aux_variables is just an int definition.
        # This function assumes 'self.aux_variables' contains actual symbolic variables
        # which happens during Numerics instantiation or if registered manually.
        # This implementation follows the reference logic assuming symbols exist.

        # Note: self.aux_variables is typically a list of Symbols in the context
        # where this is called (e.g. inside Numerics).
        # If called on the raw Model class, this might fail unless aux vars are registered.

        # We assume self.aux_variables is accessible as a list/struct here.

        b, h, moments, hinv = self.get_primitives()
        alpha = moments[0]
        beta = moments[1]

        # Aux layout assumed from reference:
        # [dbdx, dhdx, alpha_x_0..k, beta_y_0..k (if 2D), dbdy, dhdy...]
        # The reference used a specific layout.

        # Placeholder for 3D reconstruction using basis functions
        psi = [self.basisfunctions.eval_psi(k, z) for k in range(self.level + 1)]
        phi = [self.basisfunctions.eval(k, z) for k in range(self.level + 1)]

        u_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(
            alpha, z
        )
        v_3d = S.Zero
        w_3d = S.Zero

        # TODO: Implement w_3d logic cleanly once aux variable layout is unified.
        # The reference logic relies on explicit aux indices which might shift.
        # For now, we return valid U, V, and hydrostatic P.

        if self.dimension > 1:
            v_3d = self.basismatrices.basisfunctions.reconstruct_velocity_profile_at(
                beta, z
            )

        rho_w = 1000.0
        g = 9.81

        out[0] = b
        out[1] = h
        out[2] = u_3d
        out[3] = v_3d
        out[4] = w_3d
        out[5] = rho_w * g * h * (1 - z)

        return out
    
    def _eigenvalues(self, A):

        b, h, moments, hinv = self.get_primitives()
        n = self.normal
        An = ZArray.zeros(self.n_variables, self.n_variables)
        for d in range(self.dimension):
            An[:, :] += A[:, :, d] * n[d]
        An = An.tomatrix()
        if self.level > 1:
            for d in range(self.dimension):
                for i in range(2, self.level + 1):
                    An = An.subs(moments[d][i], 0)

        lam = sp.Symbol("lam")
        char_poly = An.charpoly(lam)

        all_evs = sp.solve(char_poly.as_expr(), lam)

        evs_zarray = ZArray([self._simplify(ev) for ev in all_evs])

        return evs_zarray
    
    def eigenvalues(self):
        """
        Regularized eigenvalues.
        """
        # We must calculate eigenvalues analytically from the base logic,
        # then apply regularization.
        
        b, h, moments, hinv = self.get_primitives()

        # Using base implementation (symbolic):
        A = ZArray(self.quasilinear_matrix())
        evs = self._eigenvalues(A)
        return evs

    
    def _primitive_eigenvalues(self):
        """
        Regularized eigenvalues.
        """
        # We must calculate eigenvalues analytically from the base logic,
        # then apply regularization.
        
        b, h, moments, hinv = self.get_primitives()
        cons_var = [b] + [h]
        primi_var = [b] + [h]
        for d in range(self.dimension):
            for i in range(self.level+1):
                cons_var += [moments[d][i] * h]
                primi_var += [moments[d][i]]
        
        M = sp.Matrix(sp.derive_by_array(cons_var, primi_var))


        # Using base implementation (symbolic):
        A = ZArray(self.quasilinear_matrix())
        n = self.normal
        An = ZArray.zeros(self.n_variables, self.n_variables)
        for d in range(self.dimension):
            An[:, :] += A[:, :, d] * n[d]
        An = An.tomatrix()
        Apn = M.inv() @ An @ M
        offset = self.level+1
        if self.level > 1:
            for d in range(self.dimension):
                for i in range(2, self.dimension * (self.level+1)):
                    Apn = Apn.subs(primi_var[2 + d * offset + i], 0)
            
        ev_dict = Apn.eigenvals()
        evs = ZArray(
            [self._simplify(ev) for ev, mult in ev_dict.items() for _ in range(mult)]
        )
        return evs
    
class NumericalShallowMomentsTopo(ShallowMomentsTopo):
    """
    Numerical implementation. 
    Overrides aux_variables to be just 1 (hinv).
    """
    
    def __init__(self, init_functions=True, **kwargs):
        super().__init__(init_functions=False, **kwargs)

        if init_functions:
            self._initialize_functions()
    
    # Simple override of the default parameter
    aux_variables = param.Parameter(default=1)      

    def get_primitives(self):
        n_moments = self.level + 1
        b = self.variables[0]
        h = self.variables[1]

        # Use precomputed hinv from Aux (Index 0)
        # Note: Ensure the Solver fills this!
        hinv = self.aux_variables[0]

        moments = []
        start_alpha = 2
        end_alpha = 2 + n_moments
        ha = self.variables[start_alpha:end_alpha]
        alpha = [val * hinv for val in ha]
        moments.append(alpha)

        if self.dimension > 1:
            start_beta = end_alpha
            end_beta = start_beta + n_moments
            hb = self.variables[start_beta:end_beta]
            beta = [val * hinv for val in hb]
            moments.append(beta)
        else:
            moments.append([S.Zero] * n_moments)

        return b, h, moments, hinv

    def eigenvalues(self):
        b, h, moments, hinv = self.get_primitives()
        analytical_model = ShallowMomentsTopo(level=self.level, dimension=self.dimension, basis_type=self.basis_type)
        evs = analytical_model.eigenvalues()
        eps = self.parameters.eps
        return conditional(h > eps, evs, ZArray.zeros(self.n_variables))
        # evs = sp.Matrix(evs).subs(h, (h+eps))
        # return ZArray([ev for ev in evs])
