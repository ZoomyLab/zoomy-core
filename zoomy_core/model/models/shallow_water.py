import sympy
from sympy import Matrix, sqrt, Abs
import param

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import Zstruct, ZArray


class ShallowWaterEquations(Model):
    """
    Shallow Water Equations model (1D and 2D).

    Usage:
        To add physics, subclass and override source():

        class MyRiver(ShallowWaterEquations):
            def source(self):
                return self.topography_term() + self.manning_term()
    """

    # --- Physical Parameters ---
    g = param.Number(default=9.81, doc="Gravitational acceleration")

    # Gravity direction vectors (usually ez=1 for standard shallow water)
    ex = param.Number(default=0.0)
    ey = param.Number(default=0.0)
    ez = param.Number(default=1.0)

    # --- Coefficients (Used only if their respective term is called) ---
    manning_n = param.Number(default=0.03, doc="Manning roughness coefficient")
    chezy_C = param.Number(default=50.0, doc="Chezy friction coefficient")
    nu = param.Number(
        default=0.0, doc="Kinematic viscosity for Newtonian friction")

    def __init__(self, **params):
        # 1. Defaults
        dim = params.get("dimension", 1)

        if "variables" not in params:
            params["variables"] = dim + 1

        # Default aux_variables to 0 unless user specifies (e.g. for topography)
        if "aux_variables" not in params:
            params["aux_variables"] = 0

        # 2. Init Base
        super().__init__(**params)

        # 3. Register Symbols
        self._update_symbolic_parameters()

    def _update_symbolic_parameters(self):
        """Inject class parameters into SymPy symbols."""
        extra_params = {
            "g": self.g,
            "ex": self.ex,
            "ey": self.ey,
            "ez": self.ez,
            "nm": self.manning_n,
            "C": self.chezy_C,
            "nu": self.nu,
        }
        self._default_parameters.update(extra_params)

        from zoomy_core.model.basemodel import (
            register_sympy_attribute,
            register_parameter_values,
        )

        self.parameters = register_sympy_attribute(
            self._default_parameters, "p")
        self.parameter_values = register_parameter_values(
            self._default_parameters)

        # Refresh derived functions
        self._initialize_derived_properties()

    # --- Core Conservation Laws ---

    def flux(self):
        h = self.variables[0]
        hu_vec = self.variables[1:]  # vector of momentums
        p = self.parameters
        fluxes = []

        # -- X-Direction Flux --
        fx = Matrix.zeros(self.n_variables, 1)
        fx[0] = hu_vec[0]
        fx[1] = (hu_vec[0] ** 2 / h) + (0.5 * p.g * p.ez * h**2)

        if self.dimension > 1:
            fx[2] = hu_vec[0] * hu_vec[1] / h

        fluxes.append(fx)

        # -- Y-Direction Flux --
        if self.dimension > 1:
            fy = Matrix.zeros(self.n_variables, 1)
            fy[0] = hu_vec[1]
            fy[1] = hu_vec[0] * hu_vec[1] / h
            fy[2] = (hu_vec[1] ** 2 / h) + (0.5 * p.g * p.ez * h**2)
            fluxes.append(fy)

        return fluxes

    def source(self):
        """
        Default source term is zero.
        Overwrite this method to add physics terms (e.g. return self.topography_term())
        """
        return Matrix.zeros(self.n_variables, 1)

    # --- Building Blocks (Physics Terms) ---

    def topography_term(self):
        """
        Bathymetry source term: S = [0, -gh * dz/dx, -gh * dz/dy]
        Requires: aux_variables >= dimension (for gradients dhdx, dhdy)
        """
        out = Matrix.zeros(self.n_variables, 1)
        h = self.variables[0]
        p = self.parameters

        # We assume aux_variables[0] is dhdx, aux_variables[1] is dhdy
        # (User must ensure aux_variables are set up correctly when using this)
        dhdx = self.aux_variables[0]
        out[1] = h * p.g * (p.ex - p.ez * dhdx)

        if self.dimension > 1:
            dhdy = self.aux_variables[1]
            out[2] = h * p.g * (p.ey - p.ez * dhdy)

        return out

    def newtonian_friction_term(self):
        """Viscous friction: S = -nu * u / h"""
        out = Matrix.zeros(self.n_variables, 1)
        h = self.variables[0]
        p = self.parameters

        for i in range(1, self.n_variables):
            qi = self.variables[i]  # momentum
            out[i] = -p.nu * qi / h
        return out

    def manning_friction_term(self):
        """Manning friction: S = -g * n^2 * u * |u| / h^(7/3)"""
        out = Matrix.zeros(self.n_variables, 1)
        h = self.variables[0]
        p = self.parameters

        # Calculate velocity magnitude
        hu_vec = self.variables[1:]
        u_vec = Matrix([q / h for q in hu_vec])
        u_mag = sqrt(u_vec.dot(u_vec))

        # Using the formulation provided in original code
        factor = -p.g * (p.nm**2) * u_mag / (h ** (7 / 3))

        for i in range(1, self.n_variables):
            qi = self.variables[i]
            out[i] = factor * qi
        return out

    def chezy_friction_term(self):
        """Chezy friction: S = -1/C^2 * u * |u|"""
        out = Matrix.zeros(self.n_variables, 1)
        h = self.variables[0]
        p = self.parameters

        hu_vec = self.variables[1:]
        u_vec = Matrix([q / h for q in hu_vec])
        u_mag = sqrt(u_vec.dot(u_vec))

        factor = -1.0 / (p.C**2) * u_mag

        for i in range(1, self.n_variables):
            u = self.variables[i] / h
            out[i] = factor * u
        return out

    # --- Visualization ---

    def project_2d_to_3d(self):
        out = Matrix([0 for i in range(5)])
        dim = self.dimension
        z = self.position[2]

        h = self.variables[0]
        hu_vec = self.variables[1:]

        u = hu_vec[0] / h
        v = 0
        if dim > 1:
            v = hu_vec[1] / h

        rho_w = 1000.0
        p = self.parameters

        out[0] = h
        out[1] = u
        out[2] = v
        out[3] = 0
        out[4] = rho_w * p.g * h * (1 - z)
        return out
