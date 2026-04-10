"""VAM -- Viscous Alignment Model (non-hydrostatic shallow moments).

Derivation chain (class hierarchy)::

    INSModel(DerivedModel)
      -> derives FullINS (continuity, x_momentum, z_momentum)

    VAMModel(INSModel)
      -> keeps z_momentum (non-hydrostatic, unlike SME which drops it)
      -> applies material model (Inviscid by default)
      -> depth-integrates all three equations
      -> applies kinematic BCs + atmospheric pressure gauge

The VAM extends SME with:
    - w-moments (vertical velocity basis expansion)
    - Pressure treated as unknown (not substituted hydrostatically)
    - Cross-momentum flux (u*w coupling)
    - Poisson pressure constraint for implicit step

State vector layout::

    Q = [b, h, hu0..huL, hw0..hwL]
         0   1   2..(2+n_u-1)  (2+n_u)..(2+n_u+n_w-1)

    where n_u = n_w = level + 1

Auxiliary variables (for pressure splitting)::

    Qaux = [hw_closure, hp0..hpL, dbdx, dhdx, dhp0dx..dhpLdx]

Usage::

    model = VAMModel(level=1)
    model.describe()

    # Compile for evaluation
    from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
    rt = NumpyRuntimeModel(model)

    # Generate C code
    from zoomy_core.transformation.to_c import CppModel
    CppModel(model).create_code()
"""

from __future__ import annotations

import sympy as sp
from sympy import Matrix, MutableDenseNDimArray, Rational, S, Symbol, sqrt
import param
import numpy as np

from zoomy_core.model.models.derived_model import (
    DerivedModel, get_cached_matrices,
)
from zoomy_core.model.models.sme_model import INSModel
from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Basisfunction
from zoomy_core.model.models.symbolic_integrator import SymbolicIntegrator


class VAMModel(INSModel):
    """Non-hydrostatic shallow moment model with vertical velocity moments.

    Derivation: INS -> material model -> depth integrate -> kinematic BCs.
    Unlike SMEModel, z-momentum is kept and pressure is NOT substituted.

    The state vector includes both horizontal velocity moments (alpha_k)
    and vertical velocity moments (gamma_k), giving the cross-momentum
    coupling that characterises the VAM.
    """

    projectable = True

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            Inviscid, DepthIntegrate,
            ApplyKinematicBCs, StressFreeSurface,
            ZeroAtmosphericPressure, SimplifyIntegrals,
        )
        super().derive_model()
        s = self.state

        # Non-hydrostatic: apply inviscid material to all equations (keep z-mom)
        self.apply(Inviscid(s))

        # Depth integrate all three equations (continuity, x-mom, z-mom)
        self.apply(DepthIntegrate(s))
        self.apply(ApplyKinematicBCs(s))
        self.apply(StressFreeSurface(s))
        self.apply(ZeroAtmosphericPressure(s))
        self.apply(SimplifyIntegrals(s))

    # -- Custom initialisation for extended state vector -------------------

    def __init__(self, level=0, n_layers=1, basis_type=Legendre_shifted,
                 eigenvalue_mode="symbolic", dimension=None, **kwargs):

        # We need to intercept the DerivedModel.__init__ logic because
        # VAM has a different variable structure (u + w moments).

        self._system = None
        self._applied = []

        # Run the derivation graph
        self.derive_model()

        if self._system is not None:
            self._system.name = type(self).__name__

        # Infer dimension from system state
        if dimension is None and self._system is not None:
            state = self._system.state
            state_dim = getattr(state, "dim", getattr(state, "dimension", 2))
            dimension = state_dim - 1
            if dimension < 1:
                dimension = 1

        # VAM variable structure: [b, h, hu0..huL, hw0..hwL]
        n_mom = level + 1
        hdim = dimension or 1
        n_u = n_mom          # horizontal velocity moments
        n_w = n_mom          # vertical velocity moments
        n_p = n_mom          # pressure modes (auxiliary)
        n_vars = 2 + hdim * n_u + n_w  # b + h + u-moments + w-moments

        var_names = ["b", "h"]
        var_names += [f"hu{k}" for k in range(n_u)]
        if hdim == 2:
            var_names += [f"hv{k}" for k in range(n_u)]
        var_names += [f"hw{k}" for k in range(n_w)]

        # Auxiliary variables for pressure splitting
        aux_names = [f"hw{n_w}"]  # w-closure mode (from continuity)
        aux_names += [f"hp{k}" for k in range(n_p)]
        aux_names += ["dbdx", "dhdx"]
        aux_names += [f"dhp{k}dx" for k in range(n_p)]

        param_dict = {
            "g": (9.81, "positive"),
            "eps": (1e-6, "positive"),
            "ez": (1.0, "positive"),
            "rho": (1000.0, "positive"),
            "lamda": (0.1, "positive"),
            "nu": (1e-6, "positive"),
        }

        # Store counts for projection methods
        self._n_u = n_u
        self._n_w = n_w
        self._n_p = n_p
        self._hdim = hdim

        # Call Model.__init__ (skip DerivedModel's, which we've replicated)
        Model.__init__(
            self,
            init_functions=False,
            dimension=hdim,
            variables=var_names,
            aux_variables=aux_names,
            parameters=param_dict,
            eigenvalue_mode=eigenvalue_mode,
            level=level,
            n_layers=n_layers,
            basis_type=basis_type,
            **kwargs,
        )
        self._initialize_functions()

    # -- Derived properties (basis matrices) --------------------------------

    def _initialize_derived_properties(self):
        super()._initialize_derived_properties()

        if self._system is None:
            return

        # Enforce positive h
        old_h = self.variables[1]
        new_h = Symbol(old_h.name, positive=True, real=True)
        self.variables[1] = new_h

        n_u, n_w, n_p = self._n_u, self._n_w, self._n_p

        # Build basis and compute matrices (u and w share same basis)
        self.basisfunctions = self.basis_type(level=self.level)
        self._integrator = SymbolicIntegrator(self.basisfunctions)
        matrices = get_cached_matrices(self.basis_type, self.level,
                                       self._integrator)
        self._M = matrices["M"]
        self._A = matrices["A"]
        self._D = matrices["D"]
        self._D1 = matrices["D1"]
        self._B = matrices.get("B", np.zeros_like(self._A))
        self._phib = matrices["phib"]
        self._c_mean = self.basisfunctions.mean_coefficients()

        n = self.level + 1
        self._phi_int = [
            sum(self._M[l, j] * self._c_mean[j] for j in range(n))
            for l in range(n)
        ]

        # Compute M^{-1}
        M = self._M
        is_diag = all(M[i, j] == 0 for i in range(n) for j in range(n)
                      if i != j)
        if is_diag:
            self._Minv = [[Rational(1) / M[i, i] if i == j else S.Zero
                           for j in range(n)] for i in range(n)]
        else:
            M_sp = sp.Matrix([[M[i, j] for j in range(n)] for i in range(n)])
            Minv_sp = M_sp.inv()
            self._Minv = [[Minv_sp[i, j] for j in range(n)]
                          for i in range(n)]

        # Extended basis for w-closure mode phi_{L+1}
        self._w_basis_ext = self.basis_type(level=self.level + 1)
        self._compute_cross_A()

    def _apply_Minv(self, raw_vec, k):
        n = self.level + 1
        return sum(self._Minv[k][l] * raw_vec[l] for l in range(n))

    def _compute_cross_A(self):
        """Compute A_uw_ext tensor for closure mode contribution.

        A_uw_ext[l, i] = int phi_l * phi_i * phi_{L+1} dz
        (contribution from the w-closure mode phi_{L+1})

        When u and w use the same basis, the standard A tensor covers
        all tracked modes. Only the closure mode needs extra computation.
        """
        from sympy.abc import z
        n_u = self._n_u
        w = self.basisfunctions.weight(z)
        bounds = tuple(self.basisfunctions.bounds())

        self._A_uw_ext = np.empty((n_u, n_u), dtype=object)
        for l in range(n_u):
            for i in range(n_u):
                integrand = (w
                             * self.basisfunctions.eval(l, z)
                             * self.basisfunctions.eval(i, z)
                             * self._w_basis_ext.eval(self.level + 1, z))
                self._A_uw_ext[l, i] = self._integrator.integrate(
                    integrand, z, bounds)

    # -- Primitives extraction -----------------------------------------------

    def get_primitives(self):
        """Extract primitive variables from the state vector.

        Returns (b, h, alpha, gamma, hinv) where:
            alpha[k] = hu_k / h  (horizontal velocity moments)
            gamma[k] = hw_k / h  (vertical velocity moments)
        """
        n_u, n_w = self._n_u, self._n_w
        hdim = self._hdim
        b = self.variables[0]
        h = self.variables[1]
        hinv = 1 / h

        # u-moments: indices 2 .. 2+n_u-1
        alpha = [self.variables[2 + k] * hinv for k in range(n_u)]

        # v-moments (2D only): indices 2+n_u .. 2+2*n_u-1
        if hdim == 2:
            beta = [self.variables[2 + n_u + k] * hinv for k in range(n_u)]
        else:
            beta = [S.Zero] * n_u

        # w-moments: indices after all horizontal moments
        w_start = 2 + hdim * n_u
        gamma = [self.variables[w_start + k] * hinv for k in range(n_w)]

        return b, h, alpha, beta, gamma, hinv

    # -- Flux ----------------------------------------------------------------

    def flux(self):
        """Flux F(Q) with M^{-1} applied.

        Layout:
            F[0, :]      = 0              (b: bathymetry, no flux)
            F[1, 0]      = h * u_mean     (mass flux in x)
            F[2..n_u+1, 0] = h A alpha alpha  (u-momentum advection)
            F[w_start.., 0] = h A alpha gamma  (cross u*w advection)

        For 2D (hdim=2), additional y-fluxes and v-momentum are included.
        """
        dim = self.dimension
        n_vars = self.n_variables
        n_u, n_w = self._n_u, self._n_w
        b, h, alpha, beta, gamma, hinv = self.get_primitives()
        A = self._A
        c_mean = self._c_mean

        F = Matrix.zeros(n_vars, dim)

        # --- Mass flux: h * u_mean ---
        F[1, 0] = h * sum(c_mean[k] * alpha[k] for k in range(n_u))
        if dim == 2:
            F[1, 1] = h * sum(c_mean[k] * beta[k] for k in range(n_u))

        # --- u-momentum advection: h A[l,i,j] alpha_i alpha_j ---
        row_base_u = 2
        for k_out in range(n_u):
            raw = [S.Zero] * n_u
            for l in range(n_u):
                for i in range(n_u):
                    for j in range(n_u):
                        raw[l] += h * alpha[i] * alpha[j] * A[l, i, j]
            F[row_base_u + k_out, 0] = self._apply_Minv(raw, k_out)

        # --- Cross-momentum flux: h A[l,i,j] alpha_i gamma_j ---
        # Note: closure mode (hw_{L+1}) contribution is handled separately
        # in flux_closure() since it depends on aux variables.
        w_start = 2 + self._hdim * n_u

        for k_out in range(n_w):
            raw = [S.Zero] * n_w
            for l in range(n_w):
                for i in range(n_u):
                    for j in range(n_w):
                        raw[l] += h * alpha[i] * gamma[j] * A[l, i, j]
            F[w_start + k_out, 0] = self._apply_Minv(raw, k_out)

        # --- 2D: v-momentum and cross fluxes ---
        if dim == 2:
            row_base_v = 2 + n_u
            for k_out in range(n_u):
                raw_vv = [S.Zero] * n_u
                raw_vu = [S.Zero] * n_u
                raw_uv = [S.Zero] * n_u
                for l in range(n_u):
                    for i in range(n_u):
                        for j in range(n_u):
                            raw_vv[l] += h * beta[i] * beta[j] * A[l, i, j]
                            raw_vu[l] += h * beta[i] * alpha[j] * A[l, i, j]
                            raw_uv[l] += h * alpha[i] * beta[j] * A[l, i, j]
                for k in range(n_u):
                    F[row_base_v + k, 1] += self._apply_Minv(raw_vv, k)
                    F[row_base_v + k, 0] += self._apply_Minv(raw_vu, k)
                    F[row_base_u + k, 1] += self._apply_Minv(raw_uv, k)

        return ZArray(F)

    # -- Hydrostatic pressure ------------------------------------------------

    def hydrostatic_pressure(self):
        """Hydrostatic pressure contribution (same as SME).

        In VAM the full pressure is solved via Poisson, but the hydrostatic
        part still enters the advective flux splitting.
        """
        dim = self.dimension
        n_vars = self.n_variables
        n_u = self._n_u
        p = self.parameters
        b, h, alpha, beta, gamma, hinv = self.get_primitives()

        F = Matrix.zeros(n_vars, dim)
        phi_int = self._phi_int
        raw_p = [p.g * p.ez * h**2 / 2 * phi_int[l] for l in range(n_u)]
        for k in range(n_u):
            F[2 + k, 0] = self._apply_Minv(raw_p, k)

        if dim == 2:
            row_v0 = 2 + n_u
            for k in range(n_u):
                F[row_v0 + k, 1] = self._apply_Minv(raw_p, k)

        return ZArray(F)

    # -- Non-conservative matrix ---------------------------------------------

    def nonconservative_matrix(self):
        """Non-conservative matrix B(Q) dQ/dx.

        Includes:
            - Topography: g*h on db/dx (u-momentum)
            - Vertical advection NC coupling (B matrix, u-momentum)
            - Mean velocity NC (u-momentum)
            - Vertical advection NC (w-momentum, from d(uw)/dz IBP)
        """
        dim = self.dimension
        n_vars = self.n_variables
        n_u, n_w = self._n_u, self._n_w
        p = self.parameters
        B_mat = self._B

        b, h, alpha, beta, gamma, hinv = self.get_primitives()
        phi_int = self._phi_int

        nc_x = Matrix.zeros(n_vars, n_vars)
        nc_y = Matrix.zeros(n_vars, n_vars)

        row_base_u = 2
        w_start = 2 + self._hdim * n_u

        # --- u-momentum: topography (g*h * db/dx) ---
        raw_topo = [p.g * p.ez * h * phi_int[l] for l in range(n_u)]
        for k in range(n_u):
            nc_x[row_base_u + k, 0] = self._apply_Minv(raw_topo, k)

        # --- u-momentum: vertical advection NC (B matrix) ---
        for col_idx in range(1, n_u):
            col = row_base_u + col_idx  # state variable index
            raw = [S.Zero] * n_u
            for l in range(n_u):
                for j in range(1, n_u):
                    raw[l] += alpha[j] * B_mat[l, j, col_idx]
            for k in range(n_u):
                nc_x[row_base_u + k, col] += self._apply_Minv(raw, k)

        # --- u-momentum: mean velocity NC coupling ---
        um = alpha[0]
        for k in range(1, n_u):
            raw_um = [S.Zero] * n_u
            raw_um[k] = -um
            nc_x[row_base_u + k, row_base_u + k] += self._apply_Minv(raw_um, k)

        # --- w-momentum: vertical advection NC from u*w coupling ---
        for col_idx in range(n_u):
            col = row_base_u + col_idx
            raw = [S.Zero] * n_w
            for l in range(n_w):
                for j in range(n_u):
                    if col_idx < n_u:
                        raw[l] += alpha[j] * B_mat[l, j, col_idx]
            for k in range(n_w):
                val = self._apply_Minv(raw, k)
                if val != 0:
                    nc_x[w_start + k, col] += val

        # --- 2D: y-direction topography ---
        if dim == 2:
            row_base_v = 2 + n_u
            for k in range(n_u):
                nc_y[row_base_v + k, 0] = self._apply_Minv(raw_topo, k)

        A_tensor = MutableDenseNDimArray.zeros(n_vars, n_vars, dim)
        for r in range(n_vars):
            for c in range(n_vars):
                A_tensor[r, c, 0] = nc_x[r, c]
                if dim > 1:
                    A_tensor[r, c, 1] = nc_y[r, c]

        return ZArray(A_tensor)

    # -- Source (gravity on w-momentum) --------------------------------------

    def source(self):
        """Source terms.

        w-momentum gets gravity: g * phi_int[l] (projected)
        u-momentum: no source in inviscid case (viscosity added by subclass)
        """
        n_vars = self.n_variables
        n_w = self._n_w
        p = self.parameters
        phi_int = self._phi_int
        w_start = 2 + self._hdim * self._n_u

        out = ZArray.zeros(n_vars)

        # Gravity source on w-momentum: g * Phi_l (after M^{-1})
        raw_grav = [p.g * p.ez * phi_int[l] for l in range(n_w)]
        for k in range(n_w):
            out[w_start + k] = self._apply_Minv(raw_grav, k)

        return out

    # -- Pressure source (implicit, from auxiliary variables) ----------------

    def source_implicit(self):
        """Pressure source from auxiliary variables hp0..hpL.

        x-momentum: d(h*pi_k)/dx terms (Leibniz of depth-integrated pressure)
        z-momentum: IBP of dp/dz against test function (volume + boundary)

        These enter the implicit pressure-correction step.
        """
        n_u, n_w, n_p = self._n_u, self._n_w, self._n_p
        n_vars = self.n_variables
        b, h, alpha, beta, gamma, hinv = self.get_primitives()

        R = ZArray.zeros(n_vars)

        # Auxiliary variable layout:
        # [hw_closure, hp0..hpN, dbdx, dhdx, dhp0dx..dhpNdx]
        hp = [self.aux_variables[1 + k] for k in range(n_p)]
        pi_ = [hp[k] * hinv for k in range(n_p)]
        dbdx = self.aux_variables[1 + n_p]
        # dhdx = self.aux_variables[1 + n_p + 1]  # available if needed
        dhpdx = [self.aux_variables[1 + n_p + 2 + k] for k in range(n_p)]

        D1 = self._D1
        phib = self._phib
        M = self._M

        row_base_u = 2
        w_start = 2 + self._hdim * n_u

        # --- x-momentum: Leibniz of depth-integrated pressure ---
        p_bottom = sum(pi_[k] * phib[k] for k in range(n_p))
        for k_out in range(n_u):
            raw = [S.Zero] * n_u
            for l in range(n_u):
                for k in range(n_p):
                    raw[l] += dhpdx[k] * M[l, k]
                raw[l] -= p_bottom * phib[l] * dbdx * h
            R[row_base_u + k_out] = self._apply_Minv(raw, k_out)

        # --- z-momentum: IBP of dp/dz ---
        for k_out in range(n_w):
            raw = [S.Zero] * n_w
            for l in range(n_w):
                # Volume term: -sum_k pi_k * D1[l,k]
                for k in range(n_p):
                    raw[l] += -pi_[k] * D1[l, k]
                # Boundary at zeta=0: -p(b) * phi_l(0) / h
                p_bot = sum(pi_[k] * phib[k] for k in range(n_p))
                raw[l] += -p_bot * phib[l]
            R[w_start + k_out] = self._apply_Minv(raw, k_out)

        return R

    # -- Eigenvalues ---------------------------------------------------------

    def eigenvalues(self):
        """Approximate eigenvalues for the VAM system.

        Uses analytical eigenvalues when possible (L0, L1),
        falls back to numerical mode for higher levels.
        """
        n_vars = self.n_variables
        n_u = self._n_u
        p = self.parameters
        b, h, alpha, beta, gamma, hinv = self.get_primitives()

        ev = ZArray.zeros(n_vars)

        if self.eigenvalue_mode == "numerical":
            return ev

        # Bathymetry wave speed = 0
        ev[0] = S.Zero

        # Simple approximation: u_mean +/- sqrt(g*h)
        if n_u >= 1:
            u_mean = alpha[0]
            c = sqrt(p.g * p.ez * h)
            ev[1] = u_mean + c
            if n_vars > 2:
                ev[2] = u_mean - c

            # Higher u-modes
            for k in range(1, n_u):
                if 2 + k < n_vars:
                    ev[2 + k] = u_mean

        # w-modes: wave speed ~ u_mean (advected vertically)
        w_start = 2 + self._hdim * n_u
        for k in range(self._n_w):
            if w_start + k < n_vars:
                ev[w_start + k] = alpha[0] if n_u >= 1 else S.Zero

        return ev

    # -- Inspection -----------------------------------------------------------

    def mass_matrix(self):
        """Return the basis mass matrix M."""
        n = self.level + 1
        return sp.Matrix([[self._M[i, j] for j in range(n)] for i in range(n)])

    def mass_matrix_inverse(self):
        """Return M^{-1}."""
        n = self.level + 1
        return sp.Matrix([[self._Minv[i][j] for j in range(n)]
                          for i in range(n)])

    def print_equations(self):
        print("=" * 70)
        print(f"VAMModel (basis={self.basisfunctions.name}, level={self.level})")
        print(f"  n_variables={self.n_variables}, dimension={self.dimension}")
        print(f"  n_u={self._n_u}, n_w={self._n_w}, n_p={self._n_p}")
        if self._system:
            print(f"  equations: {list(self._system.equations.keys())}")
            print(f"  assumptions: {self._system.assumptions}")
        print(f"  Mass matrix diagonal? "
              f"{self.mass_matrix() == sp.eye(self.level + 1)}")
        print("=" * 70)
        self.print_model_functions(
            ["flux", "hydrostatic_pressure", "nonconservative_matrix", "source"]
        )


class VAMNewtonian(VAMModel):
    """VAM with Newtonian viscosity (adds viscous + slip source terms).

    Same non-hydrostatic derivation as VAMModel, but applies Newtonian
    constitutive model instead of Inviscid. The source term includes
    viscous diffusion (D matrix) and Navier-slip friction (phib).
    """

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            Newtonian, DepthIntegrate,
            ApplyKinematicBCs, StressFreeSurface,
            ZeroAtmosphericPressure, SimplifyIntegrals,
        )
        # Call INSModel.derive_model() to create FullINS
        INSModel.derive_model(self)
        s = self.state

        # Newtonian instead of Inviscid
        self.apply(Newtonian(s))
        self.apply(DepthIntegrate(s))
        self.apply(ApplyKinematicBCs(s))
        self.apply(StressFreeSurface(s))
        self.apply(ZeroAtmosphericPressure(s))
        self.apply(SimplifyIntegrals(s))

    def source(self):
        """Source = gravity + Newtonian viscosity + Navier slip."""
        return super().source() + self._newtonian_viscosity() + self._navier_slip()

    def _newtonian_viscosity(self):
        """Newtonian viscous source for u and w moments."""
        p = self.parameters
        n_vars = self.n_variables
        n_u, n_w = self._n_u, self._n_w
        D = self._D
        out = ZArray.zeros(n_vars)
        b, h, alpha, beta, gamma, hinv = self.get_primitives()

        row_base_u = 2
        w_start = 2 + self._hdim * n_u

        # u-momentum viscous term
        raw_u = [sum(-p.nu * alpha[i] * hinv * D[i, l]
                     for i in range(n_u)) for l in range(n_u)]
        for k in range(n_u):
            out[row_base_u + k] = self._apply_Minv(raw_u, k)

        # w-momentum viscous term
        raw_w = [sum(-p.nu * gamma[i] * hinv * D[i, l]
                     for i in range(n_w)) for l in range(n_w)]
        for k in range(n_w):
            out[w_start + k] += self._apply_Minv(raw_w, k)

        return out

    def _navier_slip(self):
        """Navier-slip friction for u moments at the bottom."""
        p = self.parameters
        n_vars = self.n_variables
        n_u = self._n_u
        phib = self._phib
        out = ZArray.zeros(n_vars)
        b, h, alpha, beta, gamma, hinv = self.get_primitives()

        row_base_u = 2

        u_bottom = sum(alpha[i] * phib[i] for i in range(n_u))
        raw_slip = [-u_bottom * phib[l] / p.lamda for l in range(n_u)]
        for k in range(n_u):
            out[row_base_u + k] = self._apply_Minv(raw_slip, k)

        return out
