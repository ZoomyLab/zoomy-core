"""
VAM Derivation: Phase 3 -- Concrete model assembly.

Produces two Model objects from the VAM projection:

1. **VAMProjectedHyperbolic**: advective flux + gravity (explicit),
   pressure source (implicit via aux variables from Poisson solve).
2. **VAMProjectedPoisson**: constraint equations I1, I2 for pressure,
   derived from the splitting approach (predictor U* → pressure correction).

Each velocity component (u, w) and pressure (p) can use a **different basis**
and level.  By default all share the same Legendre basis.

At L1 with Legendre, these reproduce the hardcoded VAMHyperbolic/VAMPoissonFull.

Usage:
    state = StateSpace(dimension=2)
    vam = derive_vam_moments(state)
    hyp = VAMProjectedHyperbolic(vam, level=1)
    poi = VAMProjectedPoisson(hyp)
"""

import sympy as sp
from sympy import Symbol, Function, Matrix, Rational, S, sqrt, Derivative
import param
import numpy as np

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Basisfunction
from zoomy_core.model.models.symbolic_integrator import SymbolicIntegrator
from zoomy_core.model.models.projected_model import get_cached_matrices
from zoomy_core.model.models.vam_derivation import VAMPreProjectedEquations
from zoomy_core.model.models.vam_zeta_projection import (
    project_vam_to_zeta, VAMZetaProjectedEquations,
)


# ======================================================================
# VAMProjectedHyperbolic
# ======================================================================

class VAMProjectedHyperbolic(Model):
    """
    Hyperbolic part of the VAM.

    State vector Q = [h, hu0..huL, hw0..hwL, b]
    Auxiliary:   hw_{L+1} (closure), hp0..hpL, dbdx, dhdx, dhp_k_dx

    u, w, p can each use a different basis (default: all Legendre).
    w automatically gets one extra closure mode from continuity.
    """

    level = param.Integer(default=1)
    basis_type = param.ClassSelector(
        class_=Basisfunction, default=Legendre_shifted, is_instance=False,
    )

    def __init__(self, projected, basis_type=Legendre_shifted,
                 w_basis_type=None, p_basis_type=None,
                 level=1, eigenvalue_mode="symbolic", **kwargs):

        if isinstance(projected, VAMPreProjectedEquations):
            self._vam_zeta = project_vam_to_zeta(projected)
        elif isinstance(projected, VAMZetaProjectedEquations):
            self._vam_zeta = projected
        else:
            raise TypeError(f"Expected VAM equations, got {type(projected).__name__}")

        self._state = self._vam_zeta.state
        self._w_basis_type = w_basis_type or basis_type
        self._p_basis_type = p_basis_type or basis_type

        n_u = level + 1     # u-modes
        n_w = level + 1     # tracked w-modes (+ 1 closure = auxiliary)
        n_p = level + 1     # p-modes
        n_vars = 2 + n_u + n_w   # h + u-moments + w-moments + b

        var_names = (["h"]
                     + [f"hu{k}" for k in range(n_u)]
                     + [f"hw{k}" for k in range(n_w)]
                     + ["b"])

        param_dict = {
            "g": (9.81, "positive"),
            "eps": (1e-6, "positive"),
        }

        self._n_u = n_u
        self._n_w = n_w
        self._n_p = n_p

        super().__init__(
            init_functions=False,
            dimension=1,
            variables=var_names,
            parameters=param_dict,
            eigenvalue_mode=eigenvalue_mode,
            level=level,
            basis_type=basis_type,
            **kwargs,
        )
        self._initialize_functions()

    # ----- index helpers -----

    @property
    def _u_slice(self):
        """Indices of hu0..huL in the state vector."""
        return slice(1, 1 + self._n_u)

    @property
    def _w_slice(self):
        """Indices of hw0..hwL in the state vector."""
        return slice(1 + self._n_u, 1 + self._n_u + self._n_w)

    @property
    def _b_idx(self):
        return self.n_variables - 1

    # ----- initialisation -----

    def _initialize_derived_properties(self):
        super()._initialize_derived_properties()

        h_sym = self.variables[0]
        new_h = Symbol(h_sym.name, positive=True, real=True)
        self.variables[0] = new_h

        n_u, n_w, n_p = self._n_u, self._n_w, self._n_p

        # --- u basis ---
        self._u_basis = self.basis_type(level=self.level)
        self._u_integrator = SymbolicIntegrator(self._u_basis)
        u_mats = get_cached_matrices(self.basis_type, self.level, self._u_integrator)
        self._M_u = u_mats["M"]
        self._A_u = u_mats["A"]
        self._B_u = u_mats.get("B", np.zeros((n_u, n_u, n_u), dtype=object))
        self._D_u = u_mats["D"]
        self._D1_u = u_mats["D1"]
        self._phib_u = u_mats["phib"]
        self._c_mean_u = self._u_basis.mean_coefficients()
        self._phi_int_u = [
            sum(self._M_u[l, j] * self._c_mean_u[j] for j in range(n_u))
            for l in range(n_u)
        ]
        self._compute_Minv(self._M_u, n_u, "_Minv_u")

        # --- w basis (same family, one extra closure mode) ---
        self._w_basis = self._w_basis_type(level=self.level)
        # Extended basis for closure mode phi_{L+1}
        self._w_basis_ext = self._w_basis_type(level=self.level + 1)
        self._phib_w = np.array([self._w_basis.eval(k, self._w_basis.bounds()[0])
                                  for k in range(n_w)], dtype=object)
        # Reuse same M for w if same basis type
        if self._w_basis_type == self.basis_type:
            self._M_w = self._M_u
            self._Minv_w = self._Minv_u
        else:
            w_mats = get_cached_matrices(self._w_basis_type, self.level,
                                          SymbolicIntegrator(self._w_basis))
            self._M_w = w_mats["M"]
            self._compute_Minv(self._M_w, n_w, "_Minv_w")

        # --- p basis ---
        self._p_basis = self._p_basis_type(level=self.level)
        self._phib_p = np.array([self._p_basis.eval(k, self._p_basis.bounds()[0])
                                  for k in range(n_p)], dtype=object)
        self._phit_p = np.array([self._p_basis.eval(k, self._p_basis.bounds()[1])
                                  for k in range(n_p)], dtype=object)
        if self._p_basis_type == self.basis_type:
            self._D1_p = self._D1_u
        else:
            p_mats = get_cached_matrices(self._p_basis_type, self.level,
                                          SymbolicIntegrator(self._p_basis))
            self._D1_p = p_mats["D1"]

        # --- cross-product A tensors ---
        self._compute_cross_A()

    def _compute_Minv(self, M, n, attr_name):
        is_diag = all(M[i, j] == 0 for i in range(n) for j in range(n) if i != j)
        if is_diag:
            Minv = [[Rational(1) / M[i, i] if i == j else S.Zero
                      for j in range(n)] for i in range(n)]
        else:
            M_sp = sp.Matrix([[M[i, j] for j in range(n)] for i in range(n)])
            Minv_sp = M_sp.inv()
            Minv = [[Minv_sp[i, j] for j in range(n)] for i in range(n)]
        setattr(self, attr_name, Minv)

    def _apply_Minv_u(self, raw, k):
        return sum(self._Minv_u[k][l] * raw[l] for l in range(self._n_u))

    def _apply_Minv_w(self, raw, k):
        return sum(self._Minv_w[k][l] * raw[l] for l in range(self._n_w))

    def _compute_cross_A(self):
        """Compute cross-product tensors for u*w flux.

        A_uw[l, i, j]: int phi^u_l * phi^u_i * phi^w_j * w dz
            l, i ∈ [0..n_u-1], j ∈ [0..n_w-1]

        A_uw_ext[l, i]: int phi^u_l * phi^u_i * phi^w_{L+1} * w dz
            contribution from the w-closure mode
        """
        from sympy.abc import z
        n_u, n_w = self._n_u, self._n_w
        w = self._u_basis.weight(z)
        bounds = tuple(self._u_basis.bounds())

        # If u and w use the same basis, A_uw == A_u (standard triple product)
        if self._w_basis_type == self.basis_type:
            self._A_uw = self._A_u
        else:
            self._A_uw = np.empty((n_u, n_u, n_w), dtype=object)
            for l in range(n_u):
                for i in range(n_u):
                    for j in range(n_w):
                        integrand = (w * self._u_basis.eval(l, z)
                                     * self._u_basis.eval(i, z)
                                     * self._w_basis.eval(j, z))
                        self._A_uw[l, i, j] = self._u_integrator.integrate(
                            integrand, z, bounds)

        # Extended: closure mode phi_{L+1}
        self._A_uw_ext = np.empty((n_u, n_u), dtype=object)
        for l in range(n_u):
            for i in range(n_u):
                integrand = (w * self._u_basis.eval(l, z)
                             * self._u_basis.eval(i, z)
                             * self._w_basis_ext.eval(self.level + 1, z))
                self._A_uw_ext[l, i] = self._u_integrator.integrate(
                    integrand, z, bounds)

    # ----- primitives -----

    def get_primitives(self):
        n_u, n_w = self._n_u, self._n_w
        h = self.variables[0]
        hinv = 1 / h
        b = self.variables[self._b_idx]
        alpha = [self.variables[1 + k] * hinv for k in range(n_u)]
        gamma = [self.variables[1 + n_u + k] * hinv for k in range(n_w)]
        return b, h, alpha, gamma, hinv

    # ----- flux -----

    def flux(self):
        """
        F[0]          = h * sum c_k alpha_k           (mass)
        F[1..n_u]     = M_u^{-1} h A_u alpha alpha   (u-advection)
        F[n_u+1..end] = M_w^{-1} h A_uw alpha gamma  (cross u*w, incl. closure)
        """
        n_u, n_w = self._n_u, self._n_w
        nv = self.n_variables
        b, h, alpha, gamma, hinv = self.get_primitives()

        F = Matrix.zeros(nv, 1)

        # --- mass flux ---
        F[0] = h * sum(self._c_mean_u[k] * alpha[k] for k in range(n_u))

        # --- u-momentum advection: h A[l,i,j] alpha_i alpha_j → M_u^{-1} ---
        for k_out in range(n_u):
            raw = [S.Zero] * n_u
            for l in range(n_u):
                for i in range(n_u):
                    for j in range(n_u):
                        raw[l] += h * alpha[i] * alpha[j] * self._A_u[l, i, j]
            F[1 + k_out] = self._apply_Minv_u(raw, k_out)

        # --- cross-momentum flux: h A_uw[l,i,j] alpha_i gamma_j → M_w^{-1}
        #     PLUS closure: h A_uw_ext[l,i] alpha_i gamma_{L+1} ---
        # gamma_{L+1} is the closure mode, stored as auxiliary hw_{n_w}
        # We represent it as a Symbol that will be resolved via aux_variables
        gamma_closure = Symbol(f"hw{n_w}", real=True) * hinv

        for k_out in range(n_w):
            raw = [S.Zero] * n_w
            for l in range(n_w):
                for i in range(n_u):
                    # tracked w-modes
                    for j in range(n_w):
                        raw[l] += h * alpha[i] * gamma[j] * self._A_uw[l, i, j]
                    # closure w-mode
                    raw[l] += h * alpha[i] * gamma_closure * self._A_uw_ext[l, i]
            F[1 + n_u + k_out] = self._apply_Minv_w(raw, k_out)

        return ZArray(F)

    # ----- nonconservative matrix -----

    def nonconservative_matrix(self):
        """
        B(Q) dQ/dx coupling:
        - u-momentum: gravity (g*h on dh/dx and db/dx) + vertical advection NC
        - w-momentum: vertical advection NC from d(uw)/dz IBP
        """
        n_u, n_w = self._n_u, self._n_w
        nv = self.n_variables
        b, h, alpha, gamma, hinv = self.get_primitives()
        p = self.parameters
        B_mat = self._B_u

        nc = Matrix.zeros(nv, nv)

        # --- u-momentum: gravity/topography ---
        for k_out in range(n_u):
            raw = [p.g * h * self._phi_int_u[l] for l in range(n_u)]
            nc[1 + k_out, 0] = self._apply_Minv_u(raw, k_out)              # dh/dx
            nc[1 + k_out, self._b_idx] = self._apply_Minv_u(raw, k_out)    # db/dx

        # --- u-momentum: vertical advection NC (B matrix) ---
        for k_out in range(n_u):
            for col_idx in range(1, n_u):
                col = 1 + col_idx  # state variable index
                raw = [S.Zero] * n_u
                for l in range(n_u):
                    for j in range(1, n_u):
                        raw[l] += alpha[j] * B_mat[l, j, col_idx]
                nc[1 + k_out, col] += self._apply_Minv_u(raw, k_out)

        # --- u-momentum: mean velocity NC coupling ---
        um = alpha[0]
        for k in range(1, n_u):
            raw = [S.Zero] * n_u
            raw[k] = -um
            nc[1 + k, 1 + k] += self._apply_Minv_u(raw, k)

        # --- w-momentum: vertical advection NC from d(uw)/dz IBP ---
        # After IBP on d(uw)/dz in z-momentum, projected onto phi^w_l:
        # The NC coupling acts on d(hu_j)/dx through the B matrix
        # B_ww[l, i, j] = int dphi^w_l/dz * psi^w_j * phi^u_i dz
        # For same basis, this is the same B matrix but with w-indices
        for k_out in range(n_w):
            # coupling: u0 * d(hu1)/dx type terms
            for col_idx in range(n_u):
                col = 1 + col_idx
                raw = [S.Zero] * n_w
                for l in range(n_w):
                    for j in range(n_u):
                        raw[l] += alpha[j] * B_mat[l, j, col_idx] if col_idx < n_u else S.Zero
                val = self._apply_Minv_w(raw, k_out)
                if val != 0:
                    nc[1 + n_u + k_out, col] += val

        from sympy import MutableDenseNDimArray
        A_tensor = MutableDenseNDimArray.zeros(nv, nv, 1)
        for r in range(nv):
            for c in range(nv):
                A_tensor[r, c, 0] = nc[r, c]
        return ZArray(A_tensor)

    # ----- pressure source (implicit) -----

    def source_implicit(self):
        """
        Pressure source from aux variables hp0..hpL.

        x-momentum: projection of (1/rho) dp/dx
          mode l: d(h*pi_k*M_lk)/dx + boundary terms from p*db/dx
          After Leibniz + simplification:
            raw_l = sum_k dhp_k_dx * M[l,k] / h + pressure * boundary terms

        z-momentum: projection of (1/rho) dp/dz via IBP
          int phi_l dp/dz dz = [p phi_l]_b^eta - int p dphi_l/dz dz
          volume: -sum_k pi_k * D1[l,k]   (D1[l,k] = int phi_l dphi_k/dz)
          boundary at zeta=1: p(eta)*phi_l(1) = 0 (gauge)
          boundary at zeta=0: -p(b)*phi_l(0)
        """
        n_u, n_w, n_p = self._n_u, self._n_w, self._n_p
        nv = self.n_variables
        b, h, alpha, gamma, hinv = self.get_primitives()

        R = ZArray.zeros(nv)

        # Symbolic aux: pressure moments pi_k = hp_k / h
        hp = [Symbol(f"hp{k}") for k in range(n_p)]
        pi_ = [hp[k] * hinv for k in range(n_p)]      # pi_k = hp_k / h
        dhpdx = [Symbol(f"dhp{k}dx") for k in range(n_p)]
        dbdx = Symbol("dbdx")
        dhdx = Symbol("dhdx")

        D1 = self._D1_p
        phib_p = self._phib_p
        phit_p = self._phit_p

        # --- x-momentum pressure source (Leibniz) ---
        # d/dx [H int phi_l p dzeta] = sum M[l,k] dhp_k/dx
        # Boundary from Leibniz lower limit: -p(b)*phib_u[l]*dbdx
        # p(eta)=0 gauge, so upper boundary vanishes.
        p_bottom = sum(pi_[k] * phib_p[k] for k in range(n_p))
        for k_out in range(n_u):
            raw = [S.Zero] * n_u
            for l in range(n_u):
                for k in range(n_p):
                    raw[l] += dhpdx[k] * self._M_u[l, k]
                raw[l] -= p_bottom * self._phib_u[l] * dbdx * h
            R[1 + k_out] = self._apply_Minv_u(raw, k_out)

        # --- z-momentum pressure source ---
        # From IBP of (1/rho) dp/dz against phi^w_l in [0,1]:
        #   (1/h) * { -sum_k pi_k * D1[l,k] * h  (volume)
        #             + [p*phi_l]_0^1                (boundary) }
        #
        #   boundary at zeta=1: p(eta)*phi^w_l(1) = 0 (gauge, p_atm=0)
        #   boundary at zeta=0: -p(b)*phi^w_l(0)
        #     p(b) = sum_k pi_k * phi^p_k(0)
        #
        # After dividing by h and M_w^{-1}:
        for k_out in range(n_w):
            raw = [S.Zero] * n_w
            for l in range(n_w):
                # Volume term: -sum_k pi_k * D1[l,k]
                for k in range(n_p):
                    raw[l] += -pi_[k] * D1[l, k]
                # Boundary at zeta=0: -p(b) * phi^w_l(0) / h
                p_bottom = sum(pi_[k] * phib_p[k] for k in range(n_p))
                raw[l] += -p_bottom * self._phib_w[l]
                # Boundary at zeta=1: p(eta)*phi_l(1) = 0
            R[1 + n_u + k_out] = self._apply_Minv_w(raw, k_out)

        return R

    # ----- eigenvalues -----

    def eigenvalues(self):
        nv = self.n_variables
        n_u = self._n_u
        b, h, alpha, gamma, hinv = self.get_primitives()
        p = self.parameters
        ev = ZArray.zeros(nv)

        if n_u >= 1:
            ev[0] = alpha[0]
        if n_u >= 2:
            u0, u1 = alpha[0], alpha[1]
            ev[1] = u0 + u1 / sqrt(3)
            ev[2] = u0 - u1 / sqrt(3)
            ev[1 + n_u] = u0 + sqrt(p.g * h + u1**2)
            ev[2 + n_u] = u0 - sqrt(p.g * h + u1**2)
        ev[nv - 1] = S.Zero
        return ev

    # ----- inspection -----

    def mass_matrix(self):
        n = self._n_u
        return sp.Matrix([[self._M_u[i, j] for j in range(n)] for i in range(n)])

    def mass_matrix_inverse(self):
        n = self._n_u
        return sp.Matrix([[self._Minv_u[i][j] for j in range(n)] for i in range(n)])


# ======================================================================
# VAMProjectedPoisson
# ======================================================================

class VAMProjectedPoisson:
    """
    Pressure Poisson model for VAM.

    Derives the constraint equations I1, I2 from:
    1. Depth-integrated continuity (I1)
    2. Depth-integrated vorticity constraint (I2)

    After the splitting: U* = predictor (hyperbolic step),
    U^{n+1} = U* - dt * tau(hp), substitute into constraints
    and solve for hp0, hp1.

    This is NOT a Model subclass — it provides the symbolic
    constraint equations that the solver uses for the implicit step.
    """

    def __init__(self, hyp_model: VAMProjectedHyperbolic):
        self._hyp = hyp_model
        self.level = hyp_model.level
        self._n_p = hyp_model._n_p
        self._derive_constraints()

    def _derive_constraints(self):
        """
        Derive the Poisson constraints I1, I2 symbolically.

        Uses the splitting approach from tutorials/vam_poisson_hp.ipynb:
        1. Define tau vector (pressure source on state)
        2. U = U* - dt * tau
        3. Substitute into continuity + vorticity constraints
        4. Expand to get equations in hp0, hp1 and their derivatives
        """
        n_u = self._hyp._n_u
        n_p = self._n_p

        # At L1: I1 and I2 are given directly.
        # For arbitrary level, we project the continuity equation
        # onto each test function to get n_p constraint equations.

        # Symbols for the predictor state (known after hyperbolic step)
        x = Symbol("x")
        h = Function("h")(x)
        b = Function("b")(x)
        dt = Symbol("dt", positive=True)

        u = [Function(f"u{k}")(x) for k in range(n_u)]
        w = [Function(f"w{k}")(x) for k in range(n_u)]
        hp = [Function(f"hp{k}")(x) for k in range(n_p)]

        # --- Build tau vector (pressure source on state) ---
        # tau comes from source_implicit evaluated symbolically
        # At L1: tau = [0, dhp0dx + 2*p1*dbdx, -2*p1,
        #               dhp1dx - (3*p0-p1)*dhdx - 6*(p0-p1)*dbdx, 6*(p0-p1), 0]
        #
        # We compute this from the basis matrices.
        tau = self._compute_tau_vector(h, b, hp, x)

        # --- Implicit update: U = U* - dt * tau ---
        # u_k_new = u_k_star - dt * tau_{u_k} / h
        # w_k_new = w_k_star - dt * tau_{w_k} / h
        u_new = [u[k] - dt * tau[1 + k] / h for k in range(n_u)]
        w_new = [w[k] - dt * tau[1 + n_u + k] / h for k in range(n_u)]

        # --- Continuity constraints ---
        # At L1: I1 = h*du0/dx + (1/3)*d(h*u1)/dx + (1/3)*u1*dh/dx
        #             + 2*(w0 - u0*db/dx)
        #         I2 = h*du0/dx + u1*dh/dx + 2*(u1*db/dx - w1)
        #
        # General: project du/dx + dw/dz = 0 onto test functions
        # after depth integration.
        self._I1, self._I2 = self._build_constraints(
            h, b, u_new, w_new, x,
        )

        self._symbols = {
            "x": x, "h": h, "b": b, "dt": dt,
            "u": u, "w": w, "hp": hp,
        }

    def _compute_tau_vector(self, h, b, hp, x):
        """Compute the pressure source vector tau symbolically."""
        n_u = self._hyp._n_u
        n_p = self._n_p
        D1 = self._hyp._D1_p
        phib_p = self._hyp._phib_p
        phib_u = self._hyp._phib_u
        phib_w = self._hyp._phib_w
        M_u = self._hyp._M_u

        pi_ = [hp[k] / h for k in range(n_p)]
        dbdx = Derivative(b, x)
        dhdx = Derivative(h, x)
        dhpdx = [Derivative(hp[k], x) for k in range(n_p)]

        nv = self._hyp.n_variables
        tau = [S.Zero] * nv

        # x-momentum pressure source (before M^{-1})
        for l in range(n_u):
            raw_l = S.Zero
            for k in range(n_p):
                raw_l += dhpdx[k] * M_u[l, k]
            p_bottom = sum(pi_[k] * phib_p[k] for k in range(n_p))
            raw_l += p_bottom * phib_u[l] * dbdx * h
            tau[1 + l] = raw_l

        # Apply M_u^{-1}
        raw_u = [tau[1 + l] for l in range(n_u)]
        for k_out in range(n_u):
            tau[1 + k_out] = self._hyp._apply_Minv_u(raw_u, k_out)

        # z-momentum pressure source (before M^{-1})
        raw_w = [S.Zero] * n_u
        for l in range(n_u):
            for k in range(n_p):
                raw_w[l] += -pi_[k] * D1[l, k]
            p_bottom = sum(pi_[k] * phib_p[k] for k in range(n_p))
            raw_w[l] += -p_bottom * phib_w[l]

        for k_out in range(n_u):
            tau[1 + n_u + k_out] = self._hyp._apply_Minv_w(raw_w, k_out)

        return tau

    def _build_constraints(self, h, b, u, w, x):
        """
        Build the depth-integrated constraints I1, I2.

        I1 (continuity): sum over modes of d(h*u_k*c_k)/dx + w boundary terms
        I2 (vorticity): additional constraint from curl of velocity field
        """
        n_u = self._hyp._n_u
        c_mean = self._hyp._c_mean_u
        dbdx = Derivative(b, x)
        dhdx = Derivative(h, x)

        # I1: depth-integrated continuity
        # h * du_mean/dx + ... + 2*(w0 - u_mean*db/dx)
        # At L1: h*du0/dx + (1/3)*d(h*u1)/dx + (1/3)*u1*dh/dx + 2*(w0-u0*db/dx)
        u_mean_dx = sum(c_mean[k] * Derivative(u[k], x) for k in range(n_u))
        I1 = h * u_mean_dx

        # Mode 1 contribution: (1/3)*d(h*u1)/dx + (1/3)*u1*dh/dx
        if n_u >= 2:
            M01 = self._hyp._M_u[0, 1] if n_u > 1 else 0
            # From the Galerkin projection of du/dx against phi_0:
            # Additional terms from non-constant basis functions
            for k in range(1, n_u):
                c_k = Rational(1, 2*k + 1)  # Legendre M[0,k] for shifted
                I1 += c_k * Derivative(h * u[k], x) + c_k * u[k] * dhdx

        # Vertical velocity contribution: 2*(w0 - u0*db/dx)
        I1 += 2 * (w[0] - u[0] * dbdx)

        # I2: vorticity constraint
        # h*du0/dx + u1*dh/dx + 2*(u1*db/dx - w1)
        I2 = h * Derivative(u[0], x)
        if n_u >= 2:
            I2 += u[1] * dhdx + 2 * (u[1] * dbdx - w[1])

        return sp.expand(I1), sp.expand(I2)

    def get_constraints(self):
        """Return (I1, I2) as sympy expressions."""
        return self._I1, self._I2

    def get_residual_equations(self):
        """
        Return the Poisson residual: R[0] = I1+I2, R[1] = I1-I2.

        These are the equations that must be solved for hp0, hp1.
        """
        return self._I1 + self._I2, self._I1 - self._I2

    def make_derivative_symbols(self, expr):
        """
        Replace Derivative objects with Symbol objects for code generation.

        E.g. Derivative(hp0(x), x) → dhp0dx
             Derivative(hp0(x), (x, 2)) → ddhp0dxx
        """
        subs_dict = {}
        for d in sorted(expr.atoms(Derivative), key=str):
            f_name = d.expr.func.__name__
            var_counts = {}
            for v in d.variables:
                var_counts[v.name] = var_counts.get(v.name, 0) + 1
            name = "d" * sum(var_counts.values()) + f_name
            for vname, count in sorted(var_counts.items()):
                name += "d" * count + vname
            subs_dict[d] = Symbol(name)
        # Also replace Function applications: hp0(x) → hp0
        for f in sorted(expr.atoms(Function("hp0"), Function("hp1"),
                                    Function("u0"), Function("u1"),
                                    Function("w0"), Function("w1"),
                                    Function("h"), Function("b")),
                         key=str):
            subs_dict[f] = Symbol(f.func.__name__)
        return expr.subs(subs_dict), subs_dict

    def print_equations(self):
        """Print the Poisson constraints in readable form."""
        I1, I2 = self.get_constraints()
        R0, R1 = self.get_residual_equations()

        print("=== VAM Poisson Constraints ===")
        print(f"\nI1 (continuity):")
        I1_clean, _ = self.make_derivative_symbols(I1)
        print(f"  {I1_clean}")
        print(f"\nI2 (vorticity):")
        I2_clean, _ = self.make_derivative_symbols(I2)
        print(f"  {I2_clean}")
        print(f"\nResidual:")
        R0_clean, _ = self.make_derivative_symbols(R0)
        R1_clean, _ = self.make_derivative_symbols(R1)
        print(f"  R[hp0] = I1+I2 = {R0_clean}")
        print(f"  R[hp1] = I1-I2 = {R1_clean}")
