"""Symbolic PDE model base: variables, parameters, registered flux/source callbacks, and BC wiring."""

import sympy as sp
import numpy as np
import param

from zoomy_core.model.boundary_conditions import BoundaryConditions

# Import the new Aux specific classes
import zoomy_core.model.aux_boundary_conditions as AuxBC
from zoomy_core.model.initial_conditions import Constant, InitialConditions
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basefunction import Function, SymbolicRegistrar

sp.init_printing()


def register_sympy_attribute(definition, prefix="q"):
    """Turn int or list field specs into a Zstruct of real sympy Symbols."""
    if isinstance(definition, int):
        names = [f"{prefix}_{i}" for i in range(definition)]
    elif isinstance(definition, (list, tuple)):
        names = [str(n) for n in definition]
    elif isinstance(definition, Zstruct):
        return definition
    else:
        raise TypeError(f"Unsupported definition type: {type(definition)}")
    attrs = {n: sp.Symbol(n, real=True) for n in names}
    z = Zstruct(**attrs)
    z._symbolic_name = prefix
    return z


def eigenvalue_dict_to_matrix(eigenvals_dict):
    """Flatten a sympy eigenvals() dict {eigenvalue: multiplicity} into a ZArray."""
    result = []
    for ev, mult in eigenvals_dict.items():
        result.extend([ev] * int(mult))
    return ZArray(result)


def default_simplify(expr):
    return sp.powsimp(expr, combine="all", force=False, deep=True)


def parse_definition_to_zstruct(definition, prefix="q_"):
    """Turn int/list/dict/:class:`~zoomy_core.misc.misc.Zstruct` specs into symbolic :class:`~zoomy_core.misc.misc.Zstruct` fields."""
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
            assumptions = {"real": True}
            if isinstance(data, (list, tuple)) and len(data) > 1:
                constraint = data[1]
                if constraint == "positive":
                    assumptions["positive"] = True
            attributes[str(name)] = sp.Symbol(str(name), **assumptions)
    elif isinstance(definition, Zstruct):
        return parse_definition_to_zstruct(definition.as_dict(), prefix=prefix)

    return Zstruct(**attributes)


def extract_parameter_defaults(definition):
    """Numeric defaults for parameters (feeds ``parameter_values`` arrays in solvers)."""
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
    elif isinstance(definition, Zstruct):
        return extract_parameter_defaults(definition.as_dict())
    else:
        assert False
    return defaults


class Model(param.Parameterized, SymbolicRegistrar):
    """Model. (class)."""
    name = param.String(default="Model")
    dimension = param.Integer(default=1)
    disable_differentiation = param.Boolean(default=False)

    eigenvalue_mode = param.Selector(
        default="symbolic",
        objects=["symbolic", "numerical"],
        doc="'symbolic': solve characteristic polynomial in SymPy (exact, can be slow). "
            "'numerical': defer to np.linalg.eigvals at runtime (fast, required for large systems)."
    )

    variables = param.Parameter(default=1)
    aux_variables = param.Parameter(default=0)
    parameters = param.Parameter(default={})

    boundary_conditions = param.ClassSelector(class_=BoundaryConditions, default=None)
    aux_boundary_conditions = param.ClassSelector(
        class_=BoundaryConditions, default=None
    )

    initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)
    aux_initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)

    def __init__(self, init_functions=True, **params):
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
        """Internal helper `_initialize_derived_properties`."""
        p_def = self._resolve_input(self.parameters)
        var_def = self._resolve_input(self.variables)
        aux_def = self._resolve_input(self.aux_variables)

        # 1. Parameters
        # Symbols (used in symbolic derivation) live in ``_parameter_symbols``.
        # Values (the user-facing interface) live in ``parameters``.
        self._parameter_symbols = parse_definition_to_zstruct(p_def, "p")
        self._parameter_symbols._symbolic_name = "p"
        defaults = extract_parameter_defaults(p_def)
        # ``self.parameters`` is a plain Zstruct holding numeric values.
        # Users can do ``model.parameters.nu = 0.01`` or
        # ``model.parameters.update({"nu": 0.01, "lamda": 1e-2})``.
        self.parameters = Zstruct(
            **{k: float(defaults.get(k, 0.0)) for k in self._parameter_symbols.keys()}
        )
        self.parameters._symbolic_name = "p"

        # 2. Parse Variables
        self.variables = parse_definition_to_zstruct(var_def, "q")
        self.variables._symbolic_name = "Q"

        # 3. Parse Aux
        self.aux_variables = parse_definition_to_zstruct(aux_def, "qaux")
        self.aux_variables._symbolic_name = "Qaux"

        self.n_variables = self.variables.length()
        self.n_aux_variables = self.aux_variables.length()
        self.n_parameters = self.parameters.length()

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
            p=self._parameter_symbols,
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

        ic_sig = Zstruct(position=self.position, p=self._parameter_symbols)

        # Gradient symbols for diffusive_flux: gradQ[var, dim]
        grad_syms = []
        for v in self.variables.keys():
            for d in range(self.dimension):
                grad_syms.append(sp.Symbol(f"dQ_{v}_d{d}", real=True))
        self.gradient_variables = parse_definition_to_zstruct(
            [str(s) for s in grad_syms], "gradQ"
        )
        self.gradient_variables._symbolic_name = "gradQ"

        diff_sig = Zstruct(
            variables=self.variables,
            aux_variables=self.aux_variables,
            gradient_variables=self.gradient_variables,
            p=self._parameter_symbols,
        )

        regs = [
            ("flux", self.flux, std_sig),
            ("diffusive_flux", self.diffusive_flux, diff_sig),
            ("dflux", self.dflux, std_sig),
            ("hydrostatic_pressure", self.hydrostatic_pressure, std_sig),
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

        # --- Boundary Conditions Setup ---

        # 1. Main Boundary Conditions
        self._boundary_conditions = (
            self.boundary_conditions.get_boundary_condition_function(
                self.time,
                self.position,
                self.distance,
                self.variables,
                self.aux_variables,
                self._parameter_symbols,
                self.normal,
                function_name="boundary_conditions",
            )
        )

        # 2. Aux Boundary Conditions Synchronization
        #    Collect all tags used in the main BCs
        main_tags = set(
            bc.tag for bc in self.boundary_conditions.boundary_conditions_list
        )

        # Initialize aux_boundary_conditions if None
        if self.aux_boundary_conditions is None:
            # Default: Create AuxBC.Extrapolation for every tag found in main BCs
            default_aux = [
                AuxBC.Extrapolation(tag=bc.tag)
                for bc in self.boundary_conditions.boundary_conditions_list
            ]
            self.aux_boundary_conditions = BoundaryConditions(default_aux)
        else:
            # If user provided some, ensure we cover any missing tags with AuxBC.Extrapolation
            aux_tags = set(
                bc.tag for bc in self.aux_boundary_conditions.boundary_conditions_list
            )
            missing = main_tags - aux_tags
            if missing:
                current_list = list(
                    self.aux_boundary_conditions.boundary_conditions_list
                )
                for t in missing:
                    current_list.append(AuxBC.Extrapolation(tag=t))
                # Re-create to ensure sorting by tag happens in __init__
                self.aux_boundary_conditions = BoundaryConditions(current_list)

        # 3. Aux Boundary Conditions Generation
        #    We use standard argument order (variables, aux_variables).
        #    The objects in self.aux_boundary_conditions are instances of AuxBC.Extrapolation/Lambda,
        #    so they will correctly return Qaux.
        self._aux_boundary_conditions = (
            self.aux_boundary_conditions.get_boundary_condition_function(
                self.time,
                self.position,
                self.distance,
                self.variables,
                self.aux_variables,
                self._parameter_symbols,
                self.normal,
                function_name="aux_boundary_conditions",  # [FIX] Pass the name here!
            )
        )
        self._aux_boundary_conditions.name = "aux_boundary_conditions"

    def print_boundary_conditions(self):
        """Print boundary conditions."""
        return self._boundary_conditions.definition

    # --- Physics Methods (Unchanged) ---
    def flux(self):
        """Flux."""
        return ZArray.zeros(self.n_variables, self.dimension)

    def diffusive_flux(self):
        """Diffusive flux F_diff(Q, ∇Q). Shape (n_variables, dimension).

        Evaluated at faces using reconstructed gradients.
        Default: zero (no diffusion).
        """
        return ZArray.zeros(self.n_variables, self.dimension)

    def dflux(self):
        """Dflux (legacy)."""
        return ZArray.zeros(self.n_variables, self.dimension)
    
    def hydrostatic_pressure(self):
        """Hydrostatic pressure."""
        return ZArray.zeros(self.n_variables, self.dimension)

    def nonconservative_matrix(self):
        """Nonconservative matrix."""
        return ZArray.zeros(self.n_variables, self.n_variables, self.dimension)

    def source(self):
        """Source."""
        return ZArray.zeros(self.n_variables)

    def residual(self):
        """Residual."""
        return ZArray.zeros(self.n_variables)

    def interpolate(self):
        """Interpolate."""
        return ZArray(self.variables)

    def project_2d_to_3d(self):
        """Project 2d to 3d."""
        return ZArray.zeros(6)

    def project_3d_to_2d(self):
        """Project 3d to 2d."""
        return ZArray.zeros(self.n_variables)

    def initial_condition(self):
        """Initial condition."""
        return ZArray.zeros(self.n_variables)

    def initial_aux_condition(self):
        """Initial aux condition."""
        return ZArray.zeros(self.n_aux_variables)

    def update_variables(self):
        """Update variables."""
        return ZArray(self.variables)

    def update_aux_variables(self):
        """Update aux variables."""
        return ZArray(self.aux_variables)

    def update_variables_jacobian_wrt_variables(self):
        """Update variables jacobian wrt variables."""
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables)
        return self._simplify(
            sp.derive_by_array(self.update_variables(), self.variables.get_list())
        )

    def update_aux_variables_jacobian_wrt_variables(self):
        """Update aux variables jacobian wrt variables."""
        if self.disable_differentiation:
            return ZArray.zeros(self.n_aux_variables, self.n_variables)
        return self._simplify(
            sp.derive_by_array(self.update_aux_variables(), self.variables.get_list())
        )

    def quasilinear_matrix(self):
        """Quasilinear matrix."""
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables, self.dimension)
        JacF = ZArray(sp.derive_by_array(self.flux(), self.variables.get_list()))
        for d in range(self.dimension):
            JacF[:, :, d] = ZArray(JacF[:, :, d].tomatrix().T)
        JacP = ZArray(sp.derive_by_array(self.hydrostatic_pressure(), self.variables.get_list()))
        for d in range(self.dimension):
            JacP[:, :, d] = ZArray(JacP[:, :, d].tomatrix().T)
        return self._simplify(JacF + JacP + self.nonconservative_matrix())

    def source_jacobian_wrt_variables(self):
        """Source jacobian wrt variables."""
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables)
        return self._simplify(
            sp.derive_by_array(self.source(), self.variables.get_list())
        )

    def source_jacobian_wrt_aux_variables(self):
        """Source jacobian wrt aux variables."""
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_aux_variables)
        return self._simplify(
            sp.derive_by_array(self.source(), self.aux_variables.get_list())
        )

    def eigenvalues(self):
        """
        Eigenvalues of the normal-projected quasilinear matrix.

        In 'symbolic' mode: solves the characteristic polynomial in SymPy.
        In 'numerical' mode: returns an empty ZArray (eigenvalues computed
        at runtime by the Numerics class via np.linalg.eigvals).
        """
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

    def left_eigenvectors(self):
        """Left eigenvectors."""
        return ZArray.zeros(self.n_variables, self.n_variables)

    def right_eigenvectors(self):
        """Right eigenvectors."""
        return ZArray.zeros(self.n_variables, self.n_variables)

    def print_model_functions(self, function_names=None):
        """Print model functions."""
        if function_names is None:
            function_names = ["flux", "nonconservative_matrix", "source", "residual"]

        print("=" * 80, flush=True)
        print(f"Model: {self.__class__.__name__}", flush=True)
        print("variables:", self.variables.keys(), flush=True)
        print("aux_variables:", self.aux_variables.keys(), flush=True)
        if hasattr(self, "derivative_specs"):
            specs = [f"{s.field}:{''.join(s.axes)}" for s in self.derivative_specs]
            print("derivative_specs:", specs, flush=True)
        print("-" * 80, flush=True)

        available = set(self.functions.keys()) if hasattr(self, "functions") else set()
        for name in function_names:
            if name not in available:
                continue
            print(f"{name}:", flush=True)
            print(self.functions[name].definition, flush=True)
            print("-" * 80, flush=True)

    def summarize_model(self, tex=False):
        """
        Complete model description as a dict (for display) or LaTeX string.

        Returns a dict with keys:
          - 'pde_form': the general PDE structure
          - 'Q': state vector
          - 'Qaux': auxiliary variables
          - 'F': flux matrix
          - 'P': hydrostatic pressure
          - 'B': nonconservative matrix (per dimension)
          - 'S': source vector
          - 'eigenvalues': list of eigenvalue expressions (or 'numerical')
          - 'parameters': {name: default_value}
          - 'config': model configuration summary
        If tex=True, all SymPy objects are returned as LaTeX strings.
        """
        fmt = (lambda e: sp.latex(e)) if tex else (lambda e: e)

        Q_vec = sp.Matrix(list(self.variables.values()))

        F_def = self.flux() if hasattr(self, "functions") and "flux" in self.functions.keys() else self.flux()
        P_def = self.hydrostatic_pressure()
        S_def = self.source()

        B_defs = {}
        nc = self.nonconservative_matrix()
        dim_labels = ["x", "y", "z"][:self.dimension]
        for d in range(self.dimension):
            B_d = sp.Matrix([[nc[r, c, d] for c in range(self.n_variables)]
                             for r in range(self.n_variables)])
            B_defs[dim_labels[d]] = B_d

        has_nc = any(B_defs[k] != sp.zeros(self.n_variables) for k in B_defs)
        has_source = not all(S_def[i] == 0 for i in range(len(S_def)))
        has_pressure = not all(P_def[i] == 0 for i in range(len(P_def._array)))

        pde_parts = [r"\partial_t \mathbf{Q}"]
        for d, label in enumerate(dim_labels):
            pde_parts.append(rf"+ \partial_{label} \mathbf{{F}}_{label}(\mathbf{{Q}})")
        if has_pressure:
            for d, label in enumerate(dim_labels):
                pde_parts.append(rf"+ \partial_{label} \mathbf{{P}}_{label}(\mathbf{{Q}})")
        if has_nc:
            for d, label in enumerate(dim_labels):
                pde_parts.append(rf"+ \mathbf{{B}}_{label}(\mathbf{{Q}}) \partial_{label} \mathbf{{Q}}")
        pde_parts.append(r"= \mathbf{S}(\mathbf{Q})" if has_source else r"= \mathbf{0}")
        pde_form = " ".join(pde_parts)

        eig_mode = getattr(self, "eigenvalue_mode", "symbolic")
        if eig_mode == "numerical":
            eig_summary = "numerical (np.linalg.eigvals at runtime)"
            eig_list = None
        else:
            try:
                evs = self.functions.eigenvalues.definition if hasattr(self, "functions") and "eigenvalues" in self.functions.keys() else self.eigenvalues()
                eig_list = list(evs)
                eig_summary = "symbolic"
            except Exception:
                eig_list = None
                eig_summary = "symbolic (failed to compute)"

        params = dict(self.parameters.as_dict(recursive=False))

        config = {
            "class": self.__class__.__name__,
            "dimension": self.dimension,
            "n_variables": self.n_variables,
            "n_aux_variables": self.n_aux_variables,
            "n_parameters": self.n_parameters,
            "eigenvalue_mode": eig_mode,
        }
        for attr in ["n_layers", "level", "basis_type"]:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if hasattr(val, "name"):
                    config[attr] = val.name
                elif hasattr(val, "__name__"):
                    config[attr] = val.__name__
                else:
                    config[attr] = val
        if hasattr(self, "basisfunctions"):
            config["basis_name"] = self.basisfunctions.name

        F_mat = sp.Matrix([[F_def[r, d] for d in range(self.dimension)]
                           for r in range(self.n_variables)]) if F_def.rank() == 2 else sp.Matrix(list(F_def))
        P_mat = sp.Matrix([[P_def[r, d] for d in range(self.dimension)]
                           for r in range(self.n_variables)]) if P_def.rank() == 2 else sp.Matrix(list(P_def))

        result = {
            "pde_form": pde_form,
            "Q": fmt(Q_vec),
            "Qaux": fmt(sp.Matrix(list(self.aux_variables.values()))) if self.n_aux_variables > 0 else None,
            "F": fmt(F_mat),
            "P": fmt(P_mat) if has_pressure else None,
            "B": {k: fmt(v) for k, v in B_defs.items()} if has_nc else None,
            "S": fmt(sp.Matrix(list(S_def))) if has_source else None,
            "eigenvalues": [fmt(e) for e in eig_list] if eig_list else eig_summary,
            "parameters": params,
            "config": config,
        }
        return result
    
