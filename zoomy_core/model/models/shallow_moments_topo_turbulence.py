import sympy as sp
from sympy import Matrix, Symbol, S, Piecewise
import param
import numpy as np

# Adjust imports based on your actual project structure
from zoomy_core.misc.misc import ZArray
from .shallow_moments_topo import ShallowMomentsTopo


class ShallowMomentsTopoTurbulence(ShallowMomentsTopo):
    """
    Shallow Water Moments augmented with k-omega SST Turbulence (ML-Discovered).
    State vector: Q = [b, h, alpha_0...L, beta_0...L (if 2D), k_0...L, omega_0...L].
    """

    # --- 1. Augmented Constants ---
    parameters = param.Parameter(
        default={
            # ... (inherits base parameters automatically in Zoomy) ...
            "g": (9.81, "positive"),
            "ez": (1, "positive"),
            "ex": (0, "positive"),
            "nu": (1e-6, "positive"),
            "a1": (0.31, "positive"),  # Bradshaw's constant for SST
            "delta_z": (0.01, "positive"),  # Distance for finite difference wall BC
            # ML Discovered Coefficients (using the positive=True run as default)
            "c_k_diff": (0.61, "positive"),
            "c_k_cross": (1.20, "positive"),
            "c_o_cross_eta": (7.46, "positive"),
            "c_o_cross_base": (5.20, "positive"),
            "c_o_dest_base": (0.0033, "positive"),
            "c_o_dest_eta": (0.147, "positive"),
        }
    )
    
    def _compute_variable_count(self):
        # b(1) + h(1) + X-moments + Y-moments(if 2D) + k-moments + omega-moments
        base_vars = 2 + self.dimension * (self.level + 1)
        turb_vars = 2 * (self.level + 1)  # k and omega
        return base_vars + turb_vars

    variables = param.Parameter(default=_compute_variable_count)

    def _compute_variable_count(self):
        # b(1) + h(1) + X-moments + Y-moments(if 2D) + k-moments + omega-moments
        base_vars = 2 + self.dimension * (self.level + 1)
        turb_vars = 2 * (self.level + 1)  # k and omega
        return base_vars + turb_vars

    def get_primitives(self):
        """
        Extracts primitive variables including turbulence moments.
        Returns: b, h, [alpha, beta, tke, omega], hinv
        """
        n_moments = self.level + 1
        b = self.variables[0]
        h = self.variables[1]
        hinv = 1 / h

        moments = []

        # 1. Alpha (X-velocity)
        idx = 2
        ha = self.variables[idx : idx + n_moments]
        moments.append([val * hinv for val in ha])
        idx += n_moments

        # 2. Beta (Y-velocity)
        if self.dimension > 1:
            hb = self.variables[idx : idx + n_moments]
            moments.append([val * hinv for val in hb])
            idx += n_moments
        else:
            moments.append([S.Zero] * n_moments)

        # 3. k (Turbulent Kinetic Energy)
        hk = self.variables[idx : idx + n_moments]
        moments.append([val * hinv for val in hk])
        idx += n_moments

        # 4. omega (Specific Dissipation)
        ho = self.variables[idx : idx + n_moments]
        moments.append([val * hinv for val in ho])

        return b, h, moments, hinv

    # --- 2. Numerical Integration & Limiters ---

    def _smooth_max(self, a, b, eps=1e-6):
        """
        A differentiable, regularized maximum function to replace Piecewise/max.
        max(a, b) approx = 0.5 * (a + b + sqrt((a - b)^2 + eps))
        """
        return 0.5 * (a + b + sp.sqrt((a - b) ** 2 + eps))

    def _numerical_projection(self, integrand_expr, eta_sym, k_index, n_points=5):
        """
        Computes the Galerkin projection integral over depth [0, 1] using symbolic
        Gauss-Legendre quadrature.

        integral_0^1 (integrand * phi_k) d_eta
        """
        # Get standard Gauss-Legendre points and weights for [-1, 1]
        pts, weights = np.polynomial.legendre.leggauss(n_points)

        # Map from [-1, 1] to [0, 1]
        pts_mapped = 0.5 * pts + 0.5
        weights_mapped = 0.5 * weights

        integral_approx = S.Zero
        for pt, w in zip(pts_mapped, weights_mapped):
            # Convert float to SymPy Float to maintain symbolic purity
            sym_pt = sp.Float(pt, 15)
            sym_w = sp.Float(w, 15)

            # Evaluate integrand and test function at the Gauss point
            phi_val = self.basisfunctions.eval(k_index, sym_pt)
            val = integrand_expr.subs(eta_sym, sym_pt)

            integral_approx += sym_w * val * phi_val

        return integral_approx

    # --- 3. Physics & Profiles ---

    def get_profiles(self, moments, eta_sym):
        """Reconstructs 1D symbolic profiles for the current spatial column."""
        alpha, beta, tke, omega = moments

        U = sum(
            alpha[i] * self.basisfunctions.eval(i, eta_sym)
            for i in range(self.level + 1)
        )
        K = sum(
            tke[i] * self.basisfunctions.eval(i, eta_sym) for i in range(self.level + 1)
        )
        O = sum(
            omega[i] * self.basisfunctions.eval(i, eta_sym)
            for i in range(self.level + 1)
        )

        # Derivatives with respect to eta
        dU_deta = sum(
            alpha[i] * self.basisfunctions.eval_d1(i, eta_sym)
            for i in range(self.level + 1)
        )
        dK_deta = sum(
            tke[i] * self.basisfunctions.eval_d1(i, eta_sym)
            for i in range(self.level + 1)
        )
        dO_deta = sum(
            omega[i] * self.basisfunctions.eval_d1(i, eta_sym)
            for i in range(self.level + 1)
        )

        return U, K, O, dU_deta, dK_deta, dO_deta

    def compute_tau_wall(self, U_profile, eta_sym):
        """
        Customizable rough-wall boundary condition.
        Currently uses a finite difference approximation over a small delta_z.
        """
        p = self.parameters
        # Approximate dU/dy at the wall (eta = 0)
        # U(delta_z) - U(0) / delta_z. For simplicity, we just use the gradient at eta=0.
        # Note: Physical derivative dU/dy = dU_deta / h
        dU_deta_wall = sp.diff(U_profile, eta_sym).subs(eta_sym, 0)

        # We can substitute this later with a specific rough wall function
        tau_wall = p.nu * (dU_deta_wall)  # Divided by h in the actual source term
        return tau_wall

    # --- 4. Flux and Sources ---

    def flux(self):
        # Get base momentum fluxes
        F_base = super().flux()

        # We need to append the advection of k and omega
        # Advection of tracer 'c' by velocity 'u': F = h * u * c
        dim = self.dimension
        lvl = self.level
        A_basis = self.basismatrices.A
        M_basis = self.basismatrices.M

        b, h, moments, hinv = self.get_primitives()
        alpha, beta, tke, omega = moments

        F_turb = Matrix.zeros(2 * (lvl + 1), dim)

        # X-Flux for k and omega
        for k in range(lvl + 1):
            for i in range(lvl + 1):
                for j in range(lvl + 1):
                    # k flux (alpha * tke)
                    F_turb[k, 0] += (
                        h * alpha[i] * tke[j] * A_basis[k, i, j] / M_basis[k, k]
                    )
                    # omega flux (alpha * omega)
                    F_turb[(lvl + 1) + k, 0] += (
                        h * alpha[i] * omega[j] * A_basis[k, i, j] / M_basis[k, k]
                    )

        # Y-Flux for k and omega
        if dim == 2:
            for k in range(lvl + 1):
                for i in range(lvl + 1):
                    for j in range(lvl + 1):
                        F_turb[k, 1] += (
                            h * beta[i] * tke[j] * A_basis[k, i, j] / M_basis[k, k]
                        )
                        F_turb[(lvl + 1) + k, 1] += (
                            h * beta[i] * omega[j] * A_basis[k, i, j] / M_basis[k, k]
                        )

        # Combine vertically
        return ZArray(F_base.tomatrix().col_join(F_turb))

    def nonconservative_matrix(self):
        """
        The nonconservative matrix for the turbulence model.
        The base model's nonconservative terms are handled by the superclass.
        The turbulence equations for k and omega are assumed to have no
        nonconservative terms, so we just need to ensure the matrix
        has the correct shape with zeros for the turbulence variables.
        A simple way is to return a zero matrix of the correct size.
        """
        return ZArray.zeros(self.n_variables, self.n_variables, self.dimension)

    def source(self):
        """
        Builds the complete source vector including gravity, bed friction,
        and the ML-discovered turbulence PDEs.
        """
        out = Matrix.zeros(self.n_variables, 1)
        p = self.parameters
        b, h, moments, hinv = self.get_primitives()
        lvl = self.level
        eta = sp.Symbol("eta", real=True)

        # 1. Base Gravity
        base_grav = self.gravity()
        for i in range(base_grav.shape[0]):
            out[i] += base_grav[i]

        # 2. Reconstruct Profiles
        U, K, O, dU_deta, dK_deta, dO_deta = self.get_profiles(moments, eta)

        # Physical gradients
        dU_dy = dU_deta * hinv
        dK_dy = dK_deta * hinv
        dO_dy = dO_deta * hinv

        # 3. Regularized SST Limiter for nu_t
        S_shear = sp.Abs(dU_dy)  # Or sqrt((dU/dy)^2 + (dV/dy)^2) in 2D
        sst_denom = self._smooth_max(O, S_shear / p.a1)
        nu_t = K / sst_denom

        dnut_deta = sp.diff(nu_t, eta)
        dnut_dy = dnut_deta * hinv

        # 4. Momentum Diffusion (Numerical Projection)
        # Integral of: d/dy ( (nu + nu_t) dU/dy )
        # Using Integration by parts: - (nu + nu_t) * dU/dy * dPhi/dy
        for k_idx in range(lvl + 1):
            # Viscous + Turbulent momentum diffusion integrand
            mom_diff_integrand = (
                -(p.nu + nu_t)
                * dU_dy
                * (self.basisfunctions.eval_d1(k_idx, eta) * hinv)
            )
            proj_val = self._numerical_projection(
                mom_diff_integrand, eta, k_idx, n_points=5
            )

            # Add to X-Momentum equation
            out[2 + k_idx] += proj_val

            # Rough Wall Boundary Condition penalty (subtracting tau_wall at eta=0)
            tau_w = self.compute_tau_wall(U, eta) * hinv
            out[2 + k_idx] -= tau_w * self.basisfunctions.eval(k_idx, S.Zero)

        # 5. ML-Discovered Turbulence Source Terms
        offset_k = 2 + self.dimension * (lvl + 1)
        offset_o = offset_k + (lvl + 1)

        # Build RHS integrands based on SINDy discovery
        # Note: 2nd derivatives are converted via integration by parts automatically
        # below by projecting the gradient against the gradient of the basis function.

        for k_idx in range(lvl + 1):
            dPhi_dy = self.basisfunctions.eval_d1(k_idx, eta) * hinv

            # --- K-Equation ---
            # Term: + 0.61 * (1-eta) * nu_t * d2k_dy2
            # IBP -> - 0.61 * d/dy[(1-eta)*nu_t] * dk/dy * phi  - 0.61 * (1-eta)*nu_t * dk/dy * dphi/dy
            # Note: For simplicity in the symbolic generator, we implement the cross-gradient and let IBP handle the diffusion natively

            k_diff_integrand = -p.c_k_diff * (1 - eta) * nu_t * dK_dy * dPhi_dy
            k_cross_integrand = -p.c_k_cross * (1 - eta) * dnut_dy * dK_dy

            rhs_k = k_diff_integrand + (
                k_cross_integrand * self.basisfunctions.eval(k_idx, eta)
            )
            out[offset_k + k_idx] += h * self._numerical_projection(
                rhs_k, eta, k_idx, n_points=5
            )

            # --- Omega-Equation ---
            o_cross_term = (dK_dy * dO_dy) / self._smooth_max(O, S(1e-8))
            o_cross_integrand = (p.c_o_cross_eta * eta * o_cross_term) - (
                p.c_o_cross_base * eta * dnut_dy * dO_dy
            )
            o_dest_integrand = -(p.c_o_dest_base * O**2) - (p.c_o_dest_eta * eta * O**2)

            rhs_o = (o_cross_integrand + o_dest_integrand) * self.basisfunctions.eval(
                k_idx, eta
            )
            out[offset_o + k_idx] += h * self._numerical_projection(
                rhs_o, eta, k_idx, n_points=5
            )

        return out

    def eigenvalues(self):
        """
        Calculates eigenvalues of the base shallow water system and appends
        zeros for the turbulence advection (since they are passively advected).
        """
        # Call the parent class to get physical eigenvalues (waves + momentum advection)
        base_evs = super().eigenvalues()

        # TKE and Omega are advected at velocity U, meaning their eigenvalues are just 'u'.
        # For simplicity in Riemann solvers, padding with zeros or using the max fluid velocity is standard.
        turb_vars_count = 2 * (self.level + 1)
        pad = ZArray.zeros(turb_vars_count)

        # Append padding
        return ZArray(list(base_evs) + list(pad))
