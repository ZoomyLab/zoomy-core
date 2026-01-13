import sympy
import numpy as np
import param
from typing import Callable, Union

from sympy import init_printing, powsimp

from zoomy_core.model.boundary_conditions import BoundaryConditions
from zoomy_core.model.initial_conditions import Constant, InitialConditions
from zoomy_core.misc.custom_types import FArray
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basefunction import Function

init_printing()

# --- Helper Functions ---


def default_simplify(expr):
    return powsimp(expr, combine="all", force=False, deep=True)


def transform_positive_variable_intput_to_list(argument, positive, n_variables):
    out = [False for _ in range(n_variables)]
    if positive is None:
        return out
    if isinstance(positive, dict):
        assert isinstance(argument, dict)
        for i, a in enumerate(argument.keys()):
            if a in positive.keys():
                out[i] = positive[a]
    if isinstance(positive, list):
        for i in positive:
            out[i] = True
    return out


def register_sympy_attribute(argument, string_identifier="q_", positives=None):
    if isinstance(argument, int):
        positive = transform_positive_variable_intput_to_list(
            argument, positives, argument
        )
        attributes = {
            string_identifier + str(i): sympy.symbols(
                string_identifier + str(i), real=True, positive=positive[i]
            )
            for i in range(argument)
        }
    elif isinstance(argument, dict):
        positive = transform_positive_variable_intput_to_list(
            argument, positives, len(argument)
        )
        attributes = {
            name: sympy.symbols(str(name), real=True, positive=pos)
            for name, pos in zip(argument.keys(), positive)
        }
    elif isinstance(argument, list):
        positive = transform_positive_variable_intput_to_list(
            argument, positives, len(argument)
        )
        attributes = {
            name: sympy.symbols(str(name), real=True, positive=pos)
            for name, pos in zip(argument, positive)
        }
    else:
        attributes = {}

    return Zstruct(**attributes)


def register_parameter_values(parameters):
    if isinstance(parameters, int):
        default_values = np.zeros(parameters, dtype=float)
    elif isinstance(parameters, dict):
        default_values = np.array([value for value in parameters.values()])
    else:
        default_values = np.array([])
    return default_values


def eigenvalue_dict_to_matrix(eigenvalues, simplify=default_simplify):
    evs = []
    for ev, mult in eigenvalues.items():
        for i in range(mult):
            evs.append(simplify(ev))
    return ZArray(evs)


# --- The Base Model ---


class Model(param.Parameterized):
    """
    Generic (virtual) model implementation.
    """

    # --- System Configuration ---
    name = param.String(default="Model")
    dimension = param.Integer(default=1)
    disable_differentiation = param.Boolean(default=False)
    number_of_points_3d = param.Integer(default=10)

    # --- Inputs ---
    variables = param.Parameter(default=1)
    aux_variables = param.Parameter(default=0)
    positive_variables = param.Parameter(default=None)

    boundary_conditions = param.ClassSelector(class_=BoundaryConditions, default=None)
    initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)
    aux_initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)

    parameters = param.ClassSelector(class_=Zstruct, default=None)
    parameter_values = param.Array(default=None)

    def __init__(self, **params):
        super().__init__(**params)

        # Resolve Variable Definitions
        self.variables = self._resolve_variable(self.variables)
        self.aux_variables = self._resolve_variable(self.aux_variables)

        # Handle Defaults
        if self.boundary_conditions is None:
            self.boundary_conditions = BoundaryConditions(boundary_conditions=[])

        if self.initial_conditions is None:
            self.initial_conditions = Constant()

        if self.aux_initial_conditions is None:
            self.aux_initial_conditions = Constant()

        # Internal System Constants
        self._simplify = default_simplify
        self.time = sympy.symbols("t", real=True)
        self.distance = sympy.symbols("dX", real=True)
        self.position = register_sympy_attribute(3, "X")

        # Trigger Setup
        self._initialize_derived_properties()

    def _resolve_variable(self, val):
        if isinstance(val, param.Parameter):
            val = val.default

        if callable(val):
            if hasattr(val, "__self__"):
                return val()
            return val(self)

        return val

    def _initialize_derived_properties(self):
        # --- A. Auto-Discover Physical Parameters ---
        system_params = {
            "name",
            "dimension",
            "disable_differentiation",
            "number_of_points_3d",
            "variables",
            "aux_variables",
            "positive_variables",
            "boundary_conditions",
            "initial_conditions",
            "aux_initial_conditions",
            "parameters",
            "parameter_values",
        }

        math_params = {}
        for k, v in self.param.values().items():
            if k in system_params:
                continue
            if isinstance(v, (int, float, bool, np.number)):
                math_params[k] = v

        # Register Symbols
        self.parameters = register_sympy_attribute(math_params, "p")
        self.parameter_values = register_parameter_values(math_params)
        self.n_parameters = self.parameters.length()

        # --- B. Register State Variables ---
        self.variables = register_sympy_attribute(
            self.variables, "q", self.positive_variables
        )
        self.aux_variables = register_sympy_attribute(self.aux_variables, "qaux")

        self.n_variables = self.variables.length()
        self.n_aux_variables = self.aux_variables.length()

        # --- C. Register Geometry ---
        self.normal = register_sympy_attribute(
            ["n" + str(i) for i in range(self.dimension)], "n"
        )

        self.z_3d = register_sympy_attribute(self.number_of_points_3d, "z")
        self.u_3d = register_sympy_attribute(self.number_of_points_3d, "u")
        self.p_3d = register_sympy_attribute(self.number_of_points_3d, "p")
        self.alpha_3d = register_sympy_attribute(self.number_of_points_3d, "alpha")

        # --- D. Build Function Wrappers ---
        def make_func(name, definition, extra_args=None):
            # (Using .as_dict() recursively destroys them into pure dicts)
            args_data = {
                "variables": self.variables,
                "aux_variables": self.aux_variables,
                "parameters": self.parameters,
            }

            if extra_args:
                args_data.update(extra_args)

            args = Zstruct(**args_data)
            return Function(name=name, definition=definition, args=args)

        self._flux = make_func("flux", self.flux())
        self._dflux = make_func("dflux", self.dflux())
        self._nonconservative_matrix = make_func(
            "nonconservative_matrix", self.nonconservative_matrix()
        )

        self._quasilinear_matrix = make_func(
            "quasilinear_matrix", self.quasilinear_matrix()
        )

        self._source = make_func("source", self.source())
        self._source_jacobian_wrt_variables = make_func(
            "source_jacobian_wrt_variables", self.source_jacobian_wrt_variables()
        )
        self._source_jacobian_wrt_aux_variables = make_func(
            "source_jacobian_wrt_aux_variables",
            self.source_jacobian_wrt_aux_variables(),
        )

        eig_args = {"normal": self.normal}
        self._eigenvalues = make_func("eigenvalues", self.eigenvalues(), eig_args)
        self._left_eigenvectors = make_func(
            "left_eigenvectors", self.left_eigenvectors(), eig_args
        )
        self._right_eigenvectors = make_func(
            "right_eigenvectors", self.right_eigenvectors(), eig_args
        )

        res_args = {
            "time": self.time,
            "position": self.position,
            "distance": self.distance,
        }
        self._residual = make_func("residual", self.residual(), res_args)

        proj_args = {"Z": self.position[2]}
        self._project_2d_to_3d = make_func(
            "project_2d_to_3d", self.project_2d_to_3d(), proj_args
        )
        self._project_3d_to_2d = make_func("project_3d_to_2d", self.project_3d_to_2d())

        self._boundary_conditions = (
            self.boundary_conditions.get_boundary_condition_function(
                self.time,
                self.position,
                self.distance,
                self.variables,
                self.aux_variables,
                self.parameters,
                self.normal,
            )
        )
    
    def print_boundary_conditions(self):
        bc_expr = self.boundary_conditions.get_boundary_condition_function(
            self.time,
            self.position,
            self.distance,
            self.variables,
            self.aux_variables,
            self.parameters,
            self.normal,
        )
        return bc_expr.definition

    # --- Default Implementations ---

    def flux(self):
        return ZArray.zeros(self.n_variables, self.dimension)

    def dflux(self):
        return ZArray.zeros(self.n_variables, self.dimension)

    def nonconservative_matrix(self):
        return ZArray.zeros(self.n_variables, self.n_variables, self.dimension)

    def source(self):
        return ZArray.zeros(self.n_variables)

    def quasilinear_matrix(self):
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables, self.dimension)

        JacF = ZArray(sympy.derive_by_array(self.flux(), self.variables.get_list()))

        for d in range(self.dimension):
            JacF_d = JacF[:, :, d]
            JacF_d = ZArray(JacF_d.tomatrix().T)
            JacF[:, :, d] = JacF_d

        return self._simplify(JacF + self.nonconservative_matrix())

    def source_jacobian_wrt_variables(self):
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables)
        return self._simplify(
            sympy.derive_by_array(self.source(), self.variables.get_list())
        )

    def source_jacobian_wrt_aux_variables(self):
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_aux_variables)
        return self._simplify(
            sympy.derive_by_array(self.source(), self.aux_variables.get_list())
        )

    def residual(self):
        return ZArray.zeros(self.n_variables)

    def project_2d_to_3d(self):
        return ZArray.zeros(6)

    def project_3d_to_2d(self):
        return ZArray.zeros(self.n_variables)

    def eigenvalues(self):
        A = self.normal[0] * self.quasilinear_matrix()[:, :, 0]
        for d in range(1, self.dimension):
            A += self.normal[d] * self.quasilinear_matrix()[:, :, d]
        return ZArray(self._simplify(eigenvalue_dict_to_matrix(sympy.Matrix(A).eigenvals())))

    def left_eigenvectors(self):
        return ZArray.zeros(self.n_variables, self.n_variables)

    def right_eigenvectors(self):
        return ZArray.zeros(self.n_variables, self.n_variables)
