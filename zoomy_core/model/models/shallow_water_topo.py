import sympy
from sympy import Matrix, sqrt, MutableDenseNDimArray, Symbol, S

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.custom_sympy_functions import conditional

class ShallowWaterEquationsWithTopo(Model):
    """
    Shallow Water Equations with Topography (SWEB).
    State vector includes bathymetry: Q = [b, h, hu, hv].
    Automatically handles 1D and 2D based on 'dimension'.
    """

    # --- 1. System Configuration ---
    dimension = 2

    # Variables: [b, h, hu, hv (if 2D)]
    variables = lambda self: self.dimension + 2

    # Aux Variables: 0 (Base model has none, subclasses might add hinv)
    aux_variables = 0

    # --- 2. Physical Constants ---
    parameters = {
        "g": (9.81, "positive"),
        "chezy_C": (50.0, "real"),
        "eps": (1e-6, "positive"),  # Added eps for numerical compatibility
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Enforce positivity for Depth 'h' (Index 1)
        old_h = self.variables[1]
        new_h = Symbol(old_h.name, positive=True, real=True)
        self.variables[1] = new_h

        self._initialize_functions()

    # --- 3. Helper: Primitive Variables ---

    def get_primitives(self):
        """
        Extracts primitive variables using a velocity vector U.

        Returns:
            b: Bathymetry (Scalar)
            h: Depth (Scalar)
            U: Velocity Vector (Matrix of shape [dim, 1])
            hinv: Inverse Depth (Scalar)
        """
        dim = self.dimension

        # 1. Scalars
        b = self.variables[0]
        h = self.variables[1]

        # 2. Regularization (Base model uses exact 1/h)
        # Subclasses can override this to read from aux_variables or apply desingularization
        hinv = 1.0 / h

        # 3. Momentum Vector (HU)
        # We slice dynamically based on dimension: variables[2 : 2+dim]
        HU = Matrix(self.variables[2 : 2 + dim])

        # 4. Velocity Vector (U)
        U = HU * hinv

        return b, h, U, hinv

    # --- 4. Conservation Laws ---

    def flux(self):
        """
        Returns the Flux Tensor F of shape (n_vars, dimension).
        """
        dim = self.dimension
        p = self.parameters

        # Get Primitives
        b, h, U, hinv = self.get_primitives()

        # Identity Matrix
        I = Matrix.eye(dim)

        # --- Construct Flux Tensor ---
        F = Matrix.zeros(self.n_variables, dim)

        # Row 0: Bathymetry (db/dt = 0) -> Flux is 0

        # Row 1: Mass Flux (Continuity) -> h * U^T
        # U is col vector (dim x 1), so U.T is row vector (1 x dim)
        F[1, :] = (h * U).T

        # Rows 2..N: Momentum Flux -> h * U * U^T + 0.5 * g * h^2 * I
        # Outer product U*U.T creates the advection matrix (dim x dim)
        momentum_flux = (h * (U * U.T)) + (0.5 * p.g * h**2 * I)
        F[2:, :] = momentum_flux

        return ZArray(F)

    def nonconservative_matrix(self):
        """
        Returns the Non-Conservative Tensor A.
        Term: g * h * grad(b)
        """
        dim = self.dimension
        p = self.parameters
        n_vars = self.n_variables

        # Get Primitives (only h needed here)
        b, h, U, hinv = self.get_primitives()

        A_tensor = MutableDenseNDimArray.zeros(n_vars, n_vars, dim)

        # Term: g * h * (db/dx_d) in momentum equations
        for d in range(dim):
            row_idx = 2 + d  # Momentum indices start at 2
            col_idx = 0  # Derivative of b (index 0)

            A_tensor[row_idx, col_idx, d] = p.g * h

        return ZArray(A_tensor)

    def source(self):
        return Matrix.zeros(self.n_variables, 1)

    # --- 5. Physics Terms ---

    def chezy_friction_term(self):
        """Chezy friction: S = -1/C^2 * U * |U|"""
        p = self.parameters

        # Get Primitives
        b, h, U, hinv = self.get_primitives()

        # Magnitude |U| = sqrt(U dot U)
        u_mag = sqrt(U.dot(U))

        factor = -1.0 / (p.chezy_C**2) * u_mag

        S = Matrix.zeros(self.n_variables, 1)
        # Apply to momentum rows (2 onwards)
        S[2:, 0] = factor * U
        return S

    # --- 6. Visualization ---

    def project_2d_to_3d(self):
        """
        Maps state vector to 3D visualization vector.
        Out: [b, h, u, v, w, pressure]
        """
        out = ZArray.zeros(6)

        # Physical z-coordinate
        z = self.position[2]

        # Get Primitives
        b, h, U, hinv = self.get_primitives()

        # Unpack U for explicit assignment
        u = U[0]
        v = U[1] if self.dimension > 1 else S.Zero

        rho_w = 1000.0
        g = 9.81

        out[0] = b
        out[1] = h
        out[2] = u
        out[3] = v
        out[4] = 0.0  # w
        out[5] = rho_w * g * h * (1 - z)  # Hydrostatic Pressure

        return out
    
    
class NumericalShallowWaterEquationsWithTopo(ShallowWaterEquationsWithTopo):
    
    def get_primitives(self):
        dim = self.dimension
        b = self.variables[0]
        h = self.variables[1]
        hinv = self.aux_variables[0]
        U = Matrix([hu * hinv for hu in self.variables[2 : 2 + dim]])

        return b, h, U, hinv

    def eigenvalues(self):
        ev = super().eigenvalues()
        h = self.variables[1]
        return conditional(
            h > self.parameters.eps, ev, ZArray.zeros(*ev.shape)
        )
    

