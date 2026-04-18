"""
Two-Pass PDE Model Derivation: Pass 2.

Takes PreProjectedEquations (basis-independent) + a basis + level,
projects onto the basis, applies M^-1, and produces a Model-compatible class.

Usage:
    from zoomy_core.model.models.model_derivation import derive_shallow_moments
    from zoomy_core.model.models.projected_model import ProjectedModel

    state = StateSpace(dimension=2)
    pre = derive_shallow_moments(state)
    model = ProjectedModel(pre, basis_type=Legendre_shifted, level=2)
    # model has flux(), source(), eigenvalues(), etc.
"""

import sympy as sp
from sympy import Symbol, Matrix, MutableDenseNDimArray, Rational, S
import param
import numpy as np

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted, Basisfunction
from zoomy_core.model.models.symbolic_integrator import SymbolicIntegrator


# ---------------------------------------------------------------------------
# Basis matrix cache (model-agnostic, keyed by basis identity + level)
# ---------------------------------------------------------------------------

_basis_matrix_cache: dict = {}


def _cache_key(basis_type, level):
    if isinstance(basis_type, type):
        name = getattr(basis_type, "name", basis_type.__name__)
    else:
        name = getattr(basis_type, "name", str(basis_type))
    return (name, level)


def get_cached_matrices(basis, level, integrator):
    key = _cache_key(basis, level)
    if key not in _basis_matrix_cache:
        _basis_matrix_cache[key] = integrator.compute_all_matrices(level)
    return _basis_matrix_cache[key]


def clear_matrix_cache():
    _basis_matrix_cache.clear()
from zoomy_core.model.models.model_derivation import PreProjectedEquations, TaggedTerm


class ProjectedModel(Model):
    """
    Model produced by projecting PreProjectedEquations onto a specific basis.

    Construction modes:

    1. From pre-computed equations (notebook / advanced):
        ``ProjectedModel(pre_projected, level=2)``

    2. From simple config (GUI / server / CLI):
        ``ProjectedModel(dimension=2, level=2, n_layers=1)``
       Auto-runs Phase 1 (derive_shallow_moments).

    The projection uses the SymbolicIntegrator to compute basis matrices (M, A, D, phib)
    and assembles the Model functions (flux, source, NC matrix) from the tagged terms.

    M^{-1} is applied once to all non-temporal terms.
    The user can inspect both raw (with M) and resolved (with M^{-1}) forms.
    """

    level = param.Integer(default=0, doc="Vertical basis function order")
    n_layers = param.Integer(default=1, doc="Number of vertical layers")
    basis_type = param.ClassSelector(
        class_=Basisfunction, default=Legendre_shifted, is_instance=False,
        doc="Vertical basis function family"
    )
    material = param.Selector(
        objects=["newtonian", "inviscid"], default="newtonian",
        doc="Material model for viscous stress"
    )

    def __init__(self, pre_projected=None, basis_type=Legendre_shifted,
                 level=0, n_layers=1, eigenvalue_mode="symbolic",
                 dimension=2, material="newtonian", slip_length=None,
                 **kwargs):
        if pre_projected is None:
            # Auto-derive: run Phase 1 from config
            from zoomy_core.model.models.ins_generator import (
                StateSpace, Newtonian, Inviscid,
            )
            from zoomy_core.model.models.model_derivation import derive_shallow_moments

            state = StateSpace(dimension=dimension)
            material_map = {"newtonian": Newtonian, "inviscid": Inviscid}
            mat_cls = material_map.get(material)
            mat_obj = mat_cls(state) if mat_cls else None
            pre_projected = derive_shallow_moments(state, material=mat_obj,
                                                    slip_length=slip_length)

        self._pre = pre_projected
        self._state = pre_projected.state

        hdim = pre_projected.horizontal_dim
        n_mom = level + 1
        n_vars = 2 + hdim * n_layers * n_mom

        var_names = ["b", "h"] + [f"q{i}" for i in range(2, n_vars)]
        param_dict = {
            "g": (9.81, "positive"),
            "eps": (1e-6, "positive"),
            "ez": (1.0, "positive"),
            "rho": (1000.0, "positive"),
            "lamda": (0.1, "positive"),
            "nu": (1e-6, "positive"),
        }

        super().__init__(
            init_functions=False,
            dimension=hdim,
            variables=var_names,
            parameters=param_dict,
            eigenvalue_mode=eigenvalue_mode,
            level=level,
            n_layers=n_layers,
            basis_type=basis_type,
            **kwargs,
        )

        self._initialize_functions()

    def _initialize_derived_properties(self):
        super()._initialize_derived_properties()

        # Enforce positive h
        old_h = self.variables[1]
        new_h = Symbol(old_h.name, positive=True, real=True)
        self.variables[1] = new_h

        # Build basis and compute matrices
        self.basisfunctions = self.basis_type(level=self.level)
        self._integrator = SymbolicIntegrator(self.basisfunctions)
        self._compute_basis_matrices()
        self._compute_mass_inverse()

    def _compute_basis_matrices(self):
        """Compute all needed basis matrices via SymbolicIntegrator."""
        matrices = get_cached_matrices(self.basis_type, self.level, self._integrator)
        self._M = matrices["M"]
        self._A = matrices["A"]
        self._D = matrices["D"]
        self._D1 = matrices["D1"]
        self._B = matrices.get("B", np.zeros_like(self._A))
        self._phib = matrices["phib"]
        self._c_mean = self.basisfunctions.mean_coefficients()
        # Precompute raw Galerkin projection weights for a constant:
        # _phi_int[l] = integral(phi_l, domain) = (M @ c_mean)_l
        # This is what enters raw vectors before M^{-1}, NOT c_mean itself.
        n = self.level + 1
        self._phi_int = [sum(self._M[l, j] * self._c_mean[j] for j in range(n))
                         for l in range(n)]

    def _compute_mass_inverse(self):
        """Compute M^{-1}. Fast path for diagonal M."""
        n = self.level + 1
        M = self._M
        is_diag = all(M[i, j] == 0 for i in range(n) for j in range(n) if i != j)
        if is_diag:
            self._Minv = [[Rational(1) / M[i, i] if i == j else S.Zero
                           for j in range(n)] for i in range(n)]
        else:
            M_sp = sp.Matrix([[M[i, j] for j in range(n)] for i in range(n)])
            Minv_sp = M_sp.inv()
            self._Minv = [[Minv_sp[i, j] for j in range(n)] for i in range(n)]

    def _apply_Minv(self, raw_vec, k):
        """Apply row k of M^{-1} to a raw projected vector."""
        n = self.level + 1
        return sum(self._Minv[k][l] * raw_vec[l] for l in range(n))

    def get_primitives(self):
        """Extract primitive variables from the state vector."""
        n_mom = self.level + 1
        n_layers = self.n_layers
        b = self.variables[0]
        h = self.variables[1]
        hinv = 1 / h

        moments_u = []
        idx = 2
        for lk in range(n_layers):
            layer = []
            for j in range(n_mom):
                layer.append(self.variables[idx] * hinv)
                idx += 1
            moments_u.append(layer)

        moments_v = []
        if self.dimension > 1:
            for lk in range(n_layers):
                layer = []
                for j in range(n_mom):
                    layer.append(self.variables[idx] * hinv)
                    idx += 1
                moments_v.append(layer)
        else:
            for lk in range(n_layers):
                moments_v.append([S.Zero] * n_mom)

        return b, h, moments_u, moments_v, hinv

    # -----------------------------------------------------------------------
    # Model interface: flux, pressure, NC, source — all derived from projection
    # -----------------------------------------------------------------------

    def flux(self):
        """
        Flux F(Q) with M^{-1} applied.

        Built from the tagged 'flux' terms in PreProjectedEquations:
        - Mass flux: h * Σ c_k α_k (from continuity)
        - Momentum advection: Σ α_i α_j A[k,i,j] (from u² term)
        - Hydrostatic pressure: g*h²/2 * c_mean (from pressure term)
        """
        dim = self.dimension
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers

        b, h, moments_u, moments_v, hinv = self.get_primitives()
        p = self._parameter_symbols
        A = self._A
        c_mean = self._c_mean

        F = Matrix.zeros(n_vars, dim)

        for lk in range(n_layers):
            w_k = Rational(1, n_layers)
            alpha = moments_u[lk]
            beta = moments_v[lk]

            # Mass flux (continuity): h * u_mean
            F[1, 0] += h * sum(c_mean[k] * alpha[k] for k in range(n_mom)) * w_k

            # Momentum advection + pressure: raw then M^{-1}
            row_base_u = 2 + lk * n_mom

            raw_adv = [S.Zero] * n_mom
            for l in range(n_mom):
                for i in range(n_mom):
                    for j in range(n_mom):
                        raw_adv[l] += h * w_k * alpha[i] * alpha[j] * A[l, i, j]

            for k in range(n_mom):
                F[row_base_u + k, 0] += self._apply_Minv(raw_adv, k)

            if dim == 2:
                F[1, 1] += h * sum(c_mean[k] * beta[k] for k in range(n_mom)) * w_k

                row_base_v = 2 + n_layers * n_mom + lk * n_mom

                raw_vv = [S.Zero] * n_mom
                raw_vu = [S.Zero] * n_mom
                raw_uv = [S.Zero] * n_mom
                for l in range(n_mom):
                    for i in range(n_mom):
                        for j in range(n_mom):
                            raw_vv[l] += h * w_k * beta[i] * beta[j] * A[l, i, j]
                            raw_vu[l] += h * w_k * beta[i] * alpha[j] * A[l, i, j]
                            raw_uv[l] += h * w_k * alpha[i] * beta[j] * A[l, i, j]

                for k in range(n_mom):
                    F[row_base_v + k, 1] += self._apply_Minv(raw_vv, k)
                    F[row_base_v + k, 0] += self._apply_Minv(raw_vu, k)
                    F[row_base_u + k, 1] += self._apply_Minv(raw_uv, k)

        return ZArray(F)

    def hydrostatic_pressure(self):
        """
        Hydrostatic pressure with M^{-1} applied.
        Separated from advective flux for well-balanced numerical schemes.
        Raw: P_l = g*ez*h²/2 * c_mean[l]
        """
        dim = self.dimension
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        p = self._parameter_symbols
        c_mean = self._c_mean

        b, h, moments_u, moments_v, hinv = self.get_primitives()

        F = Matrix.zeros(n_vars, dim)
        phi_int = self._phi_int
        raw_p = [p.g * p.ez * h**2 / 2 * phi_int[l] for l in range(n_mom)]
        for k in range(n_mom):
            F[2 + k, 0] = self._apply_Minv(raw_p, k)

        if dim == 2:
            row_v0 = 2 + n_layers * n_mom
            for k in range(n_mom):
                F[row_v0 + k, 1] = self._apply_Minv(raw_p, k)

        return ZArray(F)

    def nonconservative_matrix(self):
        """
        Non-conservative matrix B_x with M^{-1} applied.

        Built from tagged 'nonconservative' terms:
        - Topography: g*h * db/dx
        - Vertical advection coupling (B matrix terms)
        """
        dim = self.dimension
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        p = self._parameter_symbols
        B_mat = self._B

        b, h, moments_u, moments_v, hinv = self.get_primitives()
        c_mean = self._c_mean

        nc_x = Matrix.zeros(n_vars, n_vars)
        nc_y = Matrix.zeros(n_vars, n_vars)

        for lk in range(n_layers):
            w_k = Rational(1, n_layers)
            alpha = moments_u[lk]
            beta = moments_v[lk]
            row_base_u = 2 + lk * n_mom

            # Topography: g*h acts on column 0 (bathymetry b)
            phi_int = self._phi_int
            raw_topo = [p.g * p.ez * h * phi_int[l] for l in range(n_mom)]
            for k in range(n_mom):
                nc_x[row_base_u + k, 0] += self._apply_Minv(raw_topo, k)

            # Vertical advection NC coupling (from B matrix)
            for col in range(n_vars):
                ci = col - row_base_u
                if ci < 1 or ci >= n_mom:
                    continue
                raw_nc = [S.Zero] * n_mom
                for l in range(n_mom):
                    for j in range(1, n_mom):
                        raw_nc[l] += alpha[j] * B_mat[l, j, ci]
                for k in range(n_mom):
                    nc_x[row_base_u + k, col] += self._apply_Minv(raw_nc, k)

            # Mean velocity non-conservative coupling
            um = alpha[0]
            for k in range(1, n_mom):
                raw_um = [S.Zero] * n_mom
                raw_um[k] = -um
                nc_x[row_base_u + k, row_base_u + k] += self._apply_Minv(raw_um, k)

            if dim == 2:
                row_base_v = 2 + n_layers * n_mom + lk * n_mom
                raw_topo_y = [p.g * p.ez * h * phi_int[l] for l in range(n_mom)]
                for k in range(n_mom):
                    nc_y[row_base_v + k, 0] += self._apply_Minv(raw_topo_y, k)

        A_tensor = MutableDenseNDimArray.zeros(n_vars, n_vars, dim)
        for r in range(n_vars):
            for c in range(n_vars):
                A_tensor[r, c, 0] = nc_x[r, c]
                if dim > 1:
                    A_tensor[r, c, 1] = nc_y[r, c]

        return ZArray(A_tensor)

    def source(self):
        """Source = 0 by default. Override to add gravity, friction, viscosity."""
        return ZArray.zeros(self.n_variables)

    def newtonian(self):
        """
        Newtonian viscous source with M^{-1} applied.

        From tagged 'newtonian_viscosity' term:
        Raw: S_l = -nu/h * Σ_i α_i * D[i,l]
        Resolved: S_k = Σ_l Minv[k,l] * S_l
        """
        p = self._parameter_symbols
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        D = self._D

        out = ZArray.zeros(n_vars)
        b, h, moments_u, moments_v, hinv = self.get_primitives()

        for lk in range(n_layers):
            w_k = Rational(1, n_layers)
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
        Navier-slip friction with M^{-1} applied.

        From tagged 'navier_slip' term:
        Raw: S_l = -(1/(λρ)) * u_b * φ_l(0)
        where u_b = Σ_i α_i * φ_i(0)
        Resolved: S_k = Σ_l Minv[k,l] * S_l
        """
        p = self._parameter_symbols
        n_vars = self.n_variables
        n_mom = self.level + 1
        n_layers = self.n_layers
        phib = self._phib

        out = ZArray.zeros(n_vars)
        b, h, moments_u, moments_v, hinv = self.get_primitives()

        for lk in range(n_layers):
            alpha = moments_u[lk]
            beta = moments_v[lk]
            row_base_u = 2 + lk * n_mom

            u_bottom = sum(alpha[i] * phib[i] for i in range(n_mom))
            raw_u = [-Rational(1) / p.lamda / p.rho * u_bottom * phib[l]
                     for l in range(n_mom)]
            for k in range(n_mom):
                out[row_base_u + k] += self._apply_Minv(raw_u, k)

            if self.dimension > 1:
                row_base_v = 2 + n_layers * n_mom + lk * n_mom
                v_bottom = sum(beta[i] * phib[i] for i in range(n_mom))
                raw_v = [-Rational(1) / p.lamda / p.rho * v_bottom * phib[l]
                         for l in range(n_mom)]
                for k in range(n_mom):
                    out[row_base_v + k] += self._apply_Minv(raw_v, k)

        return out

    def eigenvalues(self):
        if self.eigenvalue_mode == "numerical":
            return ZArray([sp.Integer(0)] * self.n_variables)
        q_mat = self.quasilinear_matrix()
        A = self.normal[0] * q_mat[:, :, 0]
        for i in range(1, self.dimension):
            A += self.normal[i] * q_mat[:, :, i]
        A = A.tomatrix()
        lam = sp.symbols("lam")
        char_poly = A.charpoly(lam)
        evs = sp.solve(char_poly, lam)
        return ZArray([self._simplify(ev) for ev in evs])

    # -----------------------------------------------------------------------
    # Inspection methods
    # -----------------------------------------------------------------------

    def mass_matrix(self):
        """Return the raw mass matrix M."""
        n = self.level + 1
        return sp.Matrix([[self._M[i, j] for j in range(n)] for i in range(n)])

    def mass_matrix_inverse(self):
        """Return M^{-1}."""
        n = self.level + 1
        return sp.Matrix([[self._Minv[i][j] for j in range(n)] for i in range(n)])

    def mean_coefficients(self):
        """Return c_k such that Σ c_k φ_k = 1."""
        return self._c_mean

    def print_equations(self):
        print("=" * 70)
        print(f"ProjectedModel (basis={self.basisfunctions.name}, level={self.level})")
        print(f"  n_variables={self.n_variables}, dimension={self.dimension}")
        print(f"  Pre-projected assumptions: {self._pre.assumptions_applied}")
        print(f"  Mass matrix diagonal? {self.mass_matrix() == sp.eye(self.level + 1)}")
        print("=" * 70)
        self.print_model_functions(
            ["flux", "hydrostatic_pressure", "nonconservative_matrix", "source"]
        )
