import sympy as sp
import numpy as np
import param
from zoomy_core.model.boundary_conditions import BoundaryConditions
from zoomy_core.model.initial_conditions import Constant, InitialConditions
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basefunction import Function, SymbolicRegistrar

sp.init_printing()


def default_simplify(expr):
    return sp.powsimp(expr, combine="all", force=False, deep=True)


def parse_definition_to_zstruct(definition, prefix="q_"):
    attributes = {}
    if isinstance(definition, int):
        for i in range(definition):
            name = f"{prefix}{i}"
            attributes[name] = sp.Symbol(name, real=True)
    elif isinstance(definition, (list, tuple)):
        for name in definition:
            attributes[str(name)] = sp.Symbol(str(name), real=True)
    elif isinstance(definition, dict):
        for name, data in definition.items():
            # Default assumptions
            assumptions = {"real": True}

            # Check if data is provided as (value, constraint)
            if isinstance(data, (list, tuple)) and len(data) > 1:
                constraint = data[1]
                if constraint == "positive":
                    assumptions["positive"] = True
                # Add other constraints here if needed (e.g. "integer")

            attributes[str(name)] = sp.Symbol(str(name), **assumptions)

    return Zstruct(**attributes)


def extract_parameter_defaults(definition):
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


class Model(param.Parameterized, SymbolicRegistrar):
    name = param.String(default="Model")
    dimension = param.Integer(default=1)
    disable_differentiation = param.Boolean(default=False)
    variables = param.Parameter(default=1)
    aux_variables = param.Parameter(default=0)
    parameters = param.Parameter(default={})
    boundary_conditions = param.ClassSelector(class_=BoundaryConditions, default=None)

    initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)
    aux_initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)

    def __init__(self, init_functions = True, **params):
        super().__init__(**params)
        self.functions, self.call = Zstruct(), Zstruct()
        self._initialize_derived_properties()
        if init_functions:
            self._initialize_functions()


    def _resolve_input(self, val):
        if isinstance(val, param.Parameter):
            val = val.default
        if callable(val):
            try:
                return val(self)
            except TypeError:
                return val()
        return val

    def _initialize_derived_properties(self):
        p_def = self._resolve_input(self.parameters)
        var_def = self._resolve_input(self.variables)
        aux_def = self._resolve_input(self.aux_variables)

        # 1. Parse Parameters & Tag as 'p'
        self.parameter_defaults_map = extract_parameter_defaults(p_def)
        self.parameters = parse_definition_to_zstruct(p_def, "p")
        self.parameters._symbolic_name = "p"
        self.parameter_values = np.array(
            [self.parameter_defaults_map.get(k, 0.0) for k in self.parameters.keys()]
        )

        # 2. Parse Variables & Tag as 'Q'
        self.variables = parse_definition_to_zstruct(var_def, "q")
        self.variables._symbolic_name = "Q"

        # 3. Parse Aux & Tag as 'Qaux'
        self.aux_variables = parse_definition_to_zstruct(aux_def, "qaux")
        self.aux_variables._symbolic_name = "Qaux"

        self.n_variables = self.variables.length()
        self.n_aux_variables = self.aux_variables.length()

        self.time, self.distance = sp.symbols("t dX", real=True)
        self.position = parse_definition_to_zstruct(3, "X")
        self.position._symbolic_name = "X"
        self.normal = parse_definition_to_zstruct(
            ["n" + str(i) for i in range(self.dimension)]
        )
        self.normal._symbolic_name = "n"

        self.boundary_conditions = self.boundary_conditions or BoundaryConditions([])
        self.initial_conditions = self.initial_conditions or Constant()
        self.aux_initial_conditions = self.aux_initial_conditions or Constant()

        self._simplify = default_simplify

    def _initialize_functions(self):
        std_sig = Zstruct(
            variables=self.variables,
            aux_variables=self.aux_variables,
            p=self.parameters,
        )
        eig_sig = Zstruct(**std_sig.as_dict(), normal=self.normal)
        res_sig = Zstruct(
            time=self.time,
            position=self.position,
            distance=self.distance,
            **std_sig.as_dict(),
        )
        base_proj = (
            Zstruct(Z=self.position[2]) if self.position.length() > 2 else Zstruct()
        )
        proj_sig = Zstruct(position=self.position, **std_sig.as_dict())

        # IC signature (X, p) -> Q
        ic_sig = Zstruct(position=self.position, p=self.parameters)

        regs = [
            ("flux", self.flux, std_sig),
            ("dflux", self.dflux, std_sig),
            ("nonconservative_matrix", self.nonconservative_matrix, std_sig),
            ("quasilinear_matrix", self.quasilinear_matrix, std_sig),
            ("eigenvalues", self.eigenvalues, eig_sig),
            ("left_eigenvectors", self.left_eigenvectors, eig_sig),
            ("right_eigenvectors", self.right_eigenvectors, eig_sig),
            ("source", self.source, std_sig),
            (
                "source_jacobian_wrt_variables",
                self.source_jacobian_wrt_variables,
                std_sig,
            ),
            (
                "source_jacobian_wrt_aux_variables",
                self.source_jacobian_wrt_aux_variables,
                std_sig,
            ),
            ("project_2d_to_3d", self.project_2d_to_3d, proj_sig),
            ("project_3d_to_2d", self.project_3d_to_2d, std_sig),
            ("residual", self.residual, res_sig),
            ("interpolate", self.interpolate, std_sig),
            ("initial_condition", self.initial_condition, ic_sig),
            ("initial_aux_condition", self.initial_aux_condition, ic_sig),
            # --- NEW: Variable Update Functions ---
            ("update_variables", self.update_variables, std_sig),
            ("update_aux_variables", self.update_aux_variables, std_sig),
            (
                "update_variables_jacobian_wrt_variables",
                self.update_variables_jacobian_wrt_variables,
                std_sig,
            ),
            (
                "update_aux_variables_jacobian_wrt_variables",
                self.update_aux_variables_jacobian_wrt_variables,
                std_sig,
            ),
        ]
        for name, method, sig in regs:
            self.register_symbolic_function(name, method, sig)

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
        return self._boundary_conditions.definition

    # --- Physics Methods ---
    def flux(self):
        return ZArray.zeros(self.n_variables, self.dimension)

    def dflux(self):
        return ZArray.zeros(self.n_variables, self.dimension)

    def nonconservative_matrix(self):
        return ZArray.zeros(self.n_variables, self.n_variables, self.dimension)

    def source(self):
        return ZArray.zeros(self.n_variables)

    def residual(self):
        return ZArray.zeros(self.n_variables)

    def interpolate(self):
        return ZArray(self.variables)

    def project_2d_to_3d(self):
        return ZArray.zeros(6)

    def project_3d_to_2d(self):
        return ZArray.zeros(self.n_variables)

    def initial_condition(self):
        return ZArray.zeros(self.n_variables)

    def initial_aux_condition(self):
        return ZArray.zeros(self.n_aux_variables)

    # --- NEW: Implicit Update Defaults ---
    def update_variables(self):
        # Default: Identity
        return ZArray(self.variables)

    def update_aux_variables(self):
        # Default: Identity
        return ZArray(self.aux_variables)

    def update_variables_jacobian_wrt_variables(self):
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables)
        return self._simplify(
            sp.derive_by_array(self.update_variables(), self.variables.get_list())
        )

    def update_aux_variables_jacobian_wrt_variables(self):
        if self.disable_differentiation:
            return ZArray.zeros(self.n_aux_variables, self.n_variables)
        return self._simplify(
            sp.derive_by_array(self.update_aux_variables(), self.variables.get_list())
        )

    def quasilinear_matrix(self):
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables, self.dimension)
        # Fix: ZArray robust handling ensures matrix is preserved
        JacF = ZArray(sp.derive_by_array(self.flux(), self.variables.get_list()))
        for d in range(self.dimension):
            JacF[:, :, d] = ZArray(JacF[:, :, d].tomatrix().T)
        return self._simplify(JacF + self.nonconservative_matrix())

    def source_jacobian_wrt_variables(self):
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables)
        return self._simplify(
            sp.derive_by_array(self.source(), self.variables.get_list())
        )

    def source_jacobian_wrt_aux_variables(self):
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_aux_variables)
        return self._simplify(
            sp.derive_by_array(self.source(), self.aux_variables.get_list())
        )

    def eigenvalues(self):
        q_mat = self.quasilinear_matrix()
        A = self.normal[0] * q_mat[:, :, 0]
        for i in range(1, self.dimension):
            A += self.normal[i] * q_mat[:, :, i]
        # ev_dict = sp.Matrix(A.tolist()).eigenvals()
        # return ZArray(
        #     [self._simplify(ev) for ev, mult in ev_dict.items() for _ in range(mult)]
        # )
        lam = sp.symbols("lam")
        char_poly = A.charpoly(lam)
        evs = sp.solve(char_poly, lam)
        return ZArray([self._simplify(ev) for ev in evs])


    def left_eigenvectors(self):
        return ZArray.zeros(self.n_variables, self.n_variables)

    def right_eigenvectors(self):
        return ZArray.zeros(self.n_variables, self.n_variables)
