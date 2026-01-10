import sympy
from sympy import Matrix, sqrt
import param

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray


class ShallowWaterEquations(Model):
    """
    Shallow Water Equations (SWE) Model.
    Automatically handles 1D and 2D based on 'dimension'.
    """

    # --- 1. System Configuration ---
    dimension = param.Integer(default=1)

    # Variables: [h, hu, hv (if 2D)]
    variables = lambda self: self.dimension + 1

    # Aux Variables: [dh/dx, dh/dy (if 2D)]
    aux_variables = lambda self: self.dimension

    # --- 2. Physical Constants (Auto-detected by Model) ---
    g = param.Number(default=9.81, doc="Gravitational acceleration")

    # Gravity Vector Components
    ex = param.Number(default=0.0)
    ey = param.Number(default=0.0)
    ez = param.Number(default=1.0)

    # Friction Coefficients
    manning_n = param.Number(default=0.03, doc="Manning roughness")
    chezy_C = param.Number(default=50.0, doc="Chezy coefficient")
    nu = param.Number(default=0.0, doc="Kinematic viscosity")

    # --- 3. Conservation Laws ---

    def flux(self):
        """
        Returns the Flux Tensor F of shape (n_vars, dimension).
        """
        # Unpack State
        h = self.variables[0]
        hu_vec = Matrix(self.variables[1:])  # Momentum vector [hu, hv]

        # Helper Variables
        dim = self.dimension
        p = self.parameters  # Symbolic parameters

        # Velocity vector: u = hu / h
        u_vec = hu_vec / h

        # Identity Matrix for Pressure Term
        I = Matrix.eye(dim)

        # --- Construct Flux Tensor ---
        # F = [ Mass_Flux_X   Mass_Flux_Y ]
        #     [ Mom_Flux_XX   Mom_Flux_XY ]
        #     [ Mom_Flux_YX   Mom_Flux_YY ]

        F = Matrix.zeros(self.n_variables, dim)

        # 1. Mass Flux (Row 0): hu^T
        F[0, :] = hu_vec.T

        # 2. Momentum Flux (Rows 1..N): h * u * u^T + 0.5 * g * h^2 * I
        # Note: We use p.g and p.ez for gravity scaling
        momentum_flux = (h * u_vec * u_vec.T) + (0.5 * p.g * p.ez * h**2 * I)
        F[1:, :] = momentum_flux

        return ZArray(F)

    def source(self):
        """
        Default source is zero.
        Subclass this or modify to return specific terms like:
        return self.topography_term() + self.manning_friction_term()
        """
        return Matrix.zeros(self.n_variables, 1)

    # --- 4. Physics Terms (Building Blocks) ---

    def topography_term(self):
        """
        Bathymetry source: S = [0, -gh * dz/dx, -gh * dz/dy]^T
        Assumes aux_variables contains [dh/dx, dh/dy].
        """
        dim = self.dimension
        h = self.variables[0]
        p = self.parameters

        # Gradient of bathymetry/elevation assumed in aux variables
        # dh_vec = [dh/dx, dh/dy]
        dh_vec = Matrix(self.aux_variables[:dim])

        # Gravity vector components in horizontal plane (usually 0)
        e_horizontal = Matrix([p.ex])
        if dim > 1:
            e_horizontal = e_horizontal.col_join(Matrix([p.ey]))

        S = Matrix.zeros(self.n_variables, 1)

        # Momentum source: h * g * (e_horiz - ez * grad(h))
        S[1:, 0] = h * p.g * (e_horizontal - p.ez * dh_vec)

        return S

    def newtonian_friction_term(self):
        """Viscous friction: S = -nu * u / h"""
        h = self.variables[0]
        hu_vec = Matrix(self.variables[1:])
        p = self.parameters

        S = Matrix.zeros(self.n_variables, 1)
        S[1:, 0] = -p.nu * hu_vec / h  # Apply to momentum rows
        return S

    def manning_friction_term(self):
        """Manning friction: S = -g * n^2 * u * |u| / h^(7/3)"""
        h = self.variables[0]
        hu_vec = Matrix(self.variables[1:])
        p = self.parameters

        # Velocity u = hu / h
        u_vec = hu_vec / h
        u_mag = sqrt(u_vec.dot(u_vec))

        # Term: -g * n^2 * |u| / h^(7/3)
        # We use a small epsilon for h to avoid division by zero if needed,
        # but pure symbolic usually handles it until code gen.
        factor = -p.g * (p.manning_n**2) * u_mag / (h ** (sympy.Rational(7, 3)))

        S = Matrix.zeros(self.n_variables, 1)
        S[1:, 0] = factor * hu_vec
        return S

    def chezy_friction_term(self):
        """Chezy friction: S = -1/C^2 * u * |u|"""
        h = self.variables[0]
        hu_vec = Matrix(self.variables[1:])
        p = self.parameters

        u_vec = hu_vec / h
        u_mag = sqrt(u_vec.dot(u_vec))

        factor = -1.0 / (p.chezy_C**2) * u_mag

        S = Matrix.zeros(self.n_variables, 1)
        S[1:, 0] = factor * u_vec  # Apply to velocity directly? Or momentum?
        # Usually friction opposes momentum: S = - (1/C^2) * u * |u|
        # If the state is momentum (hu), S should equate to d(hu)/dt.
        # d(hu)/dt ~ force.
        # So S should be factor * u_vec? No, factor * u_vec is acceleration.
        # It should be factor * u_vec * h?
        # Standard Chezy for momentum eq: -g * u * |u| / (C^2 * h)?
        # Or - u * |u| / (C^2)?
        # Let's stick to the implementation provided: -1/C^2 * u * |u|

        S[1:, 0] = factor * u_vec
        return S

    # --- 5. Visualization ---

    def project_2d_to_3d(self):
        """
        Maps state vector to 3D visualization vector.
        Out: [b, h, u, v, w, pressure]
        """
        out = ZArray.zeros(6)
        dim = self.dimension
        z = self.position[2]  # Vertical coordinate if 3D mesh

        h = self.variables[0]
        hu_vec = self.variables[1:]

        # Calculate Velocities
        u = hu_vec[0] / h
        v = hu_vec[1] / h if dim > 1 else 0.0

        # Constants for visualization
        rho_w = 1000.0
        g = 9.81
        b = 0.0  # Placeholder for bathymetry if not in state

        out[0] = b
        out[1] = h
        out[2] = u
        out[3] = v
        out[4] = 0.0  # w
        out[5] = rho_w * g * h * (1 - z)  # Hydrostatic Pressure

        return out
