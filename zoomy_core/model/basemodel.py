import sympy
import numpy as np
import param
from typing import Union, Dict, List, Any, Tuple, Optional

from sympy import init_printing, powsimp

from zoomy_core.model.boundary_conditions import BoundaryConditions
from zoomy_core.model.initial_conditions import Constant, InitialConditions
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basefunction import Function

init_printing()

# --- Helper Functions ---


def default_simplify(expr):
    return powsimp(expr, combine="all", force=False, deep=True)


def create_symbol(name, assumptions=None):
    """Creates a SymPy symbol, defaulting to real=True."""
    kwargs = {"real": True}
    if assumptions:
        if isinstance(assumptions, str):
            kwargs[assumptions] = True
        elif isinstance(assumptions, (list, tuple, set)):
            for attr in assumptions:
                kwargs[str(attr)] = True
        elif isinstance(assumptions, dict):
            kwargs.update(assumptions)
    return sympy.Symbol(str(name), **kwargs)


def parse_definition_to_zstruct(definition, prefix="q_"):
    """
    Parses inputs (int, list, dict) into a Zstruct of SymPy symbols.
    """
    attributes = {}
    if isinstance(definition, int):
        for i in range(definition):
            name = f"{prefix}{i}"
            attributes[name] = create_symbol(name)
    elif isinstance(definition, (list, tuple)):
        for name in definition:
            attributes[str(name)] = create_symbol(name)
    elif isinstance(definition, dict):
        for name, data in definition.items():
            # Handle {'g': (9.81, 'positive')} case -> extract 'positive'
            assumption = data
            if (
                isinstance(data, (tuple, list))
                and len(data) == 2
                and isinstance(data[0], (int, float))
            ):
                assumption = data[1]
            elif isinstance(data, (int, float)):
                assumption = None
            attributes[str(name)] = create_symbol(name, assumption)

    return Zstruct(**attributes)


def extract_parameter_defaults(definition):
    """
    Extracts default numerical values from the parameters dict.
    Format: {'g': 9.81} OR {'g': (9.81, 'positive')}
    """
    defaults = {}
    if isinstance(definition, dict):
        for name, val in definition.items():
            if isinstance(val, (int, float)):
                defaults[name] = val
            elif isinstance(val, (list, tuple)) and len(val) == 2:
                if isinstance(val[0], (int, float)):
                    defaults[name] = val[0]
            else:
                defaults[name] = 0.0
    return defaults


def eigenvalue_dict_to_matrix(eigenvalues, simplify=default_simplify):
    evs = []
    for ev, mult in eigenvalues.items():
        for i in range(mult):
            evs.append(simplify(ev))
    return ZArray(evs)


# --- The Model ---


class Model(param.Parameterized):
    """
    Single Model class based on param.
    """

    # --- Configuration (Input Phase) ---
    name = param.String(default="Model")
    dimension = param.Integer(default=1)
    disable_differentiation = param.Boolean(default=False)
    number_of_points_3d = param.Integer(default=10)

    # Inputs: Int, List, or Dict (can also be Callables/Lambdas)
    variables = param.Parameter(default=1)
    aux_variables = param.Parameter(default=0)
    parameters = param.Parameter(default={})

    # Conditions
    boundary_conditions = param.ClassSelector(class_=BoundaryConditions, default=None)
    initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)
    aux_initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)

    def __init__(self, **params):
        # 1. Param Initialization
        super().__init__(**params)

        # 2. Logic Setup
        self._initialize_derived_properties()

    def _resolve_input(self, val):
        """
        Helper to resolve configuration that might be a callable (lambda),
        a param.Parameter object, or a direct value.
        """
        # If accessing class attribute directly on instance might return Parameter
        if isinstance(val, param.Parameter):
            val = val.default

        # If it is a function/lambda (e.g. lambda self: self.dimension + 2)
        if callable(val):
            # If bound method (has __self__), call it.
            if hasattr(val, "__self__"):
                return val()
            # If plain function/lambda, pass self
            return val(self)

        return val

    def _initialize_derived_properties(self):
        # --- 0. Resolve Inputs (Handle Lambdas/Callables) ---
        # This fixes the issue: we evaluate the lambda BEFORE parsing symbols
        p_def = self._resolve_input(self.parameters)
        var_def = self._resolve_input(self.variables)
        aux_def = self._resolve_input(self.aux_variables)

        # --- A. Parameters ---
        # 1. Extract Values (Backend)
        self.parameter_defaults_map = extract_parameter_defaults(p_def)

        # 2. Parse Symbols (Physics)
        self.parameters = parse_definition_to_zstruct(p_def, "p")

        # 3. Create Ordered Value Array
        vals = []
        for name in self.parameters.keys():
            val = self.parameter_defaults_map.get(name, 0.0)
            vals.append(val)
        self.parameter_values = np.array(vals)
        self.n_parameters = self.parameters.length()

        # --- B. Variables ---
        self.variables = parse_definition_to_zstruct(var_def, "q")
        self.aux_variables = parse_definition_to_zstruct(aux_def, "qaux")
        self.n_variables = self.variables.length()
        self.n_aux_variables = self.aux_variables.length()

        # --- C. Geometry & Utils ---
        self._simplify = default_simplify
        self.time = sympy.symbols("t", real=True)
        self.distance = sympy.symbols("dX", real=True)
        self.position = parse_definition_to_zstruct(3, "X")
        self.normal = parse_definition_to_zstruct(
            ["n" + str(i) for i in range(self.dimension)]
        )

        self.z_3d = parse_definition_to_zstruct(self.number_of_points_3d, "z")
        self.u_3d = parse_definition_to_zstruct(self.number_of_points_3d, "u")
        self.p_3d = parse_definition_to_zstruct(self.number_of_points_3d, "p")
        self.alpha_3d = parse_definition_to_zstruct(self.number_of_points_3d, "alpha")

        # --- D. Conditions ---
        if self.boundary_conditions is None:
            self.boundary_conditions = BoundaryConditions([])
        if self.initial_conditions is None:
            self.initial_conditions = Constant()
        if self.aux_initial_conditions is None:
            self.aux_initial_conditions = Constant()

        # --- E. Functions ---
        self._initialize_functions()

    def _initialize_functions(self):
        def make_func(name, definition, extra_args=None):
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

    # Default Methods
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
        return ZArray(
            self._simplify(eigenvalue_dict_to_matrix(sympy.Matrix(A).eigenvals()))
        )

    def left_eigenvectors(self):
        return ZArray.zeros(self.n_variables, self.n_variables)

    def right_eigenvectors(self):
        return ZArray.zeros(self.n_variables, self.n_variables)
