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
    """Numeric defaults for parameters (feeds ``model.parameters`` Zstruct values)."""
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

    # Accept EITHER the legacy BoundaryConditions container OR the new flat
    # per-field list (boundary_conditions=[Wall("left", on="momentum"), …]),
    # resolved against the model's state in `system_model`.
    boundary_conditions = param.ClassSelector(
        class_=(BoundaryConditions, list), default=None)
    aux_boundary_conditions = param.ClassSelector(
        class_=BoundaryConditions, default=None
    )

    initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)
    aux_initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)

    # ── Lazy-finalization flag ─────────────────────────────────────
    # Subclasses whose ``derive_model`` leaves equations in Function
    # form (e.g. SME, VAM) set ``_finalize_lazy = True``.  For those:
    # ``__init__`` stops after the pipeline; the Function → Symbol
    # substitution + auto-tagging + ``_initialize_functions`` happen
    # later — either via an explicit ``model._finalize_for_systemmodel()``
    # call or automatically when ``SystemModel.from_model(model)``
    # runs.  This lets the user apply optional closures
    # (e.g. ``apply_slip_newton_friction``) on Function-form equations
    # without breaking ``Derivative(g·h², x)`` (which would evaluate
    # to 0 on bare Symbols).  Default ``False`` preserves the
    # existing flow for hand-written subclasses (SWE etc.).
    _finalize_lazy = False

    def __init__(self, init_functions=True, **params):
        super().__init__(**params)
        self.functions, self.call = Zstruct(), Zstruct()
        self._equations: dict = {}
        self.history: list = []
        # ``_variable_map`` maps equation names to row indices in the
        # operator-API matrices (e.g. {"continuity_0": [0],
        # "momentum_x_0": [1], ...}).  Subclasses set this in
        # ``_finalize_for_systemmodel`` *after* the pipeline produces
        # the final equation set.  Default: empty (no extraction).
        self._variable_map: dict = {}
        # Set by ``_finalize_for_systemmodel`` so subsequent calls
        # short-circuit; reset by mutating methods like
        # ``apply_slip_newton_friction``.
        self._finalized = False
        self._initialize_derived_properties()
        # Subclass derivation hook — populates ``self._equations``
        # via ``self.add_equation`` + ``self.apply(Op(...))`` and may
        # extend ``self.variables`` / ``self.parameters``.
        self.derive_model()
        # After the derivation, every Symbol used in any equation must
        # have a numeric value declared in ``self.parameter_values``.
        self._assert_parameter_values_supplied()
        if self._finalize_lazy:
            # Stop here — equations stay in Function form; user can
            # apply closures, then ``SystemModel.from_model`` will
            # auto-call ``_finalize_for_systemmodel``.
            return
        # Non-lazy (default) path: tag + register functions now.
        self._finalize_for_systemmodel(init_functions=init_functions)

    def _finalize_for_systemmodel(self, *, init_functions=True):
        """Run the steps that prepare equations for SystemModel
        hand-off: subclass-specific Function → Symbol substitution,
        ``_variable_map`` setup, auto-tagging, and (optionally)
        ``_initialize_functions``.

        Idempotent — subsequent calls short-circuit unless
        ``self._finalized`` is reset (e.g. by a mutating method like
        ``apply_slip_newton_friction``).  Subclasses override
        ``_prepare_for_systemmodel`` to do the substitution; this
        outer method orchestrates the universal steps.
        """
        if self._finalized:
            return self
        self._prepare_for_systemmodel()
        self._auto_tag_equations()
        if init_functions:
            self._initialize_functions()
        self._finalized = True
        return self

    def _prepare_for_systemmodel(self):
        """Subclass hook called by ``_finalize_for_systemmodel`` before
        auto-tagging.  Default: no-op.  Subclasses (SME, VAM) override
        to substitute derivation Function calls with Model Symbols
        and to set ``self._variable_map``."""
        return None

    # ── derivation hook + minimal Model surface ───────────────────
    #
    # The Model class exposes ONLY four public methods:
    # `add_equation / remove_equation / apply / describe`.  All
    # transformations are `Operation` subclasses (see
    # `zoomy_core.model.operations`) and are applied via
    # `model.apply(SomeOp(args))`.  In particular: there is NO
    # `multiply` or `resolve_dummy` convenience method — those become
    # `Multiply(...)` and `ResolveDummy(...)` Operations.
    #
    # Equations live in the private `self._equations` dict and are
    # accessed externally via `__getattr__` (named-attribute style):
    # `self.momentum_x.apply(...)`, `self.continuity.apply(...)`.
    # Iteration is via `for eq in self: ...` (`__iter__`).  There is
    # NO public `.equations` collection on Model.

    def derive_model(self):
        """Subclass hook — populate equations via
        `self.add_equation(name, expr)` and `self.apply(Op(...))`.
        No-op on the base class."""
        return None

    def add_equation(self, name, expression=None, shape=None):
        """Insert an equation into the model under ``name``.
        Accessible afterwards as ``self.<name>``.

        With ``shape=None``: scalar equation; ``self.<name>`` is the
        single :class:`Equation`.

        With ``shape=(s1, s2, ...)`` or a list of component labels:
        creates one sub-equation per index combo, named
        ``"{name}_{i}_{j}_..."``.
        """
        from zoomy_core.model.equation import Equation
        if shape is None:
            expr = expression if expression is not None else sp.S.Zero
            self._equations[name] = Equation(expr, name=name, model=self)
            self._history("add_equation", name)
            return self
        # Branched: vector / tensor add_equation.
        import itertools as _it
        if all(isinstance(c, str) for c in shape):
            combos = [(c,) for c in shape]
        else:
            combos = list(_it.product(*shape))
        for combo in combos:
            sub = "_".join([name] + list(combo))
            self._equations[sub] = Equation(sp.S.Zero, name=sub, model=self)
            self._history("add_equation", sub)
        return self

    def remove_equation(self, name):
        """Remove an equation by name.  After this, ``self.<name>``
        raises ``AttributeError``."""
        del self._equations[name]
        self._history("remove_equation", name)
        return self

    def apply(self, op, *, level="major", description=None):
        """Broadcast an Operation across every equation in the model.

        ``op`` may be:

        * an :class:`Operation` whose ``whole_model_op == True`` — in
          which case ``op.apply_to_model(self)`` is called once;
        * an :class:`Operation` whose ``whole_model_op == False`` —
          applied per-equation via ``Equation.apply(op)``;
        * a substitution :class:`dict` — broadcast as ``xreplace``;
        * a :class:`Relation` — broadcast via its substitution map;
        * a callable — broadcast via ``Equation.apply``.
        """
        # Whole-model dispatch: e.g. ResolveDummy([v1, v2]) needs
        # access to the whole equation dict to branch.
        whole_model = getattr(op, "whole_model_op", False)
        if whole_model and hasattr(op, "apply_to_model"):
            op.apply_to_model(self)
        else:
            for eq in self._equations.values():
                eq.apply(op, _no_history=True)
        self._history(
            getattr(op, "name", None) or type(op).__name__,
            "*", level=level, description=description,
        )
        return self

    def describe(self, *, show_history=False, include_minor=False):
        """Return a :class:`ModelDescription` rendering the equation set.

        In a Jupyter cell, returning the description as the last
        expression triggers ``_repr_markdown_`` and the equations
        render as LaTeX.  In a plain Python REPL or via ``print(...)``
        / ``str(...)``, the plain-text form is shown."""
        return ModelDescription(
            self,
            show_history=show_history,
            include_minor=include_minor,
        )

    def _history(self, op_label, target, *, level="major", description=None,
                 log_level=1):
        self.history.append({
            "op": op_label, "target": target,
            "level": level, "log_level": log_level,
            "description": description or "",
        })

    # ── Dunders ──────────────────────────────────────────────────
    def __getattr__(self, name):
        """Named-attribute access to equations.  Only called when
        normal attribute lookup fails — so ``self.parameters``,
        ``self.variables``, etc. (real attributes) are not affected.
        ``self.momentum_x`` returns the Equation registered under
        ``"momentum_x"`` if any, else raises AttributeError."""
        # Avoid infinite recursion during `__init__` when
        # `self._equations` itself doesn't yet exist.
        eqs = self.__dict__.get("_equations", None)
        if eqs is not None and name in eqs:
            return eqs[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute "
            f"or equation {name!r}"
        )

    def __iter__(self):
        """Iterate over equations in insertion order."""
        return iter(self._equations.values())

    def __len__(self):
        return len(self._equations)

    def _assert_parameter_values_supplied(self):
        """Walk all equation expressions and collect their free
        Symbols.  Every Symbol that names a declared parameter
        (matched by ``str(sym)`` against ``self.parameter_values``
        keys) must have a numeric value supplied; otherwise raise.

        Coordinate Symbols (``t``, ``x``, ``y``, ``z``, ``dX``,
        ``zeta``, normal components) and state Symbols are skipped.

        Symbols *not* in ``self.parameter_values`` and *not* in the
        skip list are unresolved free parameters — raise so the user
        learns at construction time."""
        if not self._equations:
            return
        # Symbols that legitimately appear in equations but are not
        # parameters (coordinates, distance, normal axes, state).
        skip = set()
        for attr in ("time", "distance"):
            v = getattr(self, attr, None)
            if v is not None:
                skip.add(v)
        for zstruct_attr in ("position", "normal", "variables", "aux_variables"):
            z = getattr(self, zstruct_attr, None)
            if z is None:
                continue
            for k in z.keys():
                v = z[k]
                if isinstance(v, sp.Symbol):
                    skip.add(v)
        # StateSpace exposes reference coordinates that propagate
        # through σ-mapped equations (zeta, zeta_ref, optionally y, z).
        # Parameters (state.g, state.rho) are *not* skipped — they're
        # exactly what we want to flag when the user forgot to supply
        # a numeric default.  (In the new design, state.g / state.rho
        # don't exist on StateSpace anyway — parameters live on
        # `self.parameters`.)
        state = getattr(self, "state", None)
        if state is not None:
            for nm in ("t", "x", "y", "z", "zeta", "zeta_ref"):
                v = getattr(state, nm, None)
                if isinstance(v, sp.Symbol):
                    skip.add(v)
        param_value_names = set(self.parameter_values.keys()) if hasattr(self, "parameter_values") else set()
        param_symbol_names = set(self.parameters.keys()) if hasattr(self, "parameters") else set()
        missing: dict = {}
        for name, eq in self._equations.items():
            try:
                fsyms = eq.expr.free_symbols
            except AttributeError:
                continue
            for sym in fsyms:
                if sym in skip:
                    continue
                sym_name = str(sym)
                if sym_name in param_value_names:
                    continue
                if sym_name in param_symbol_names:
                    # Declared as a Symbol on the Model but no value —
                    # still record as missing (user must declare value).
                    if sym_name not in param_value_names:
                        missing.setdefault(sym_name, sym)
                    continue
                # Unknown free Symbol (neither declared nor in skip).
                missing.setdefault(sym_name, sym)
        if missing:
            names = ", ".join(sorted(missing.keys()))
            raise ValueError(
                f"symbol {sorted(missing.keys())[0]!r} found in {type(self).__name__} "
                f"equations but no numeric default supplied in "
                f"parameters={{...}}; missing: [{names}]. Supply via "
                f"`{type(self).__name__}(..., parameters={{...}})`."
            )

    def _auto_tag_equations(self):
        """For every equation lacking solver tags, run
        :func:`zoomy_core.model.derivation.tag_extraction.auto_solver_tag`
        on its expression so the default operator extractors can
        pull tagged sub-expressions.

        State variables (``self.variables``) are passed as the "state
        atoms" the classifier uses.  In the new design, after
        ``derive_model()`` finishes its Function→Symbol substitution
        pass, equation atoms ARE the Symbols on ``self.variables`` —
        no separate "state Function calls" list is needed.

        Equations that have no state atoms (e.g. ``bottom: ∂_t b = 0``)
        produce no tags and are skipped naturally."""
        if not self._equations:
            return
        from zoomy_core.model.derivation.tag_extraction import auto_solver_tag
        # State atoms = self.variables (Symbols after derive_model
        # finishes its substitution pass).
        state_atoms = [v for v in self.variables.values()
                       if isinstance(v, sp.Symbol)]
        if not state_atoms:
            return
        t_sym = self.time
        coords = [self.position[d] for d in range(self.dimension)]
        x_sym = coords[0] if coords else None
        if x_sym is None:
            return
        # Gravity parameter Symbol for the hydrostatic_pressure
        # classifier.  In the new design, equation atoms reference
        # ``self.parameters.g`` directly (parameters live on the
        # Model and are propagated into MassMomentum at construction
        # time — no Symbol-identity mismatch).
        gravity_param = None
        if hasattr(self, "parameters") and "g" in self.parameters.keys():
            gravity_param = self.parameters.g
        for name, eq in self._equations.items():
            sg = getattr(eq, "_solver_groups", None)
            if sg:  # already tagged
                continue
            try:
                tagged = auto_solver_tag(
                    eq.expr, state_funcs=state_atoms,
                    t=t_sym, x=x_sym, gravity_param=gravity_param,
                )
            except Exception:
                # Heuristic can fail on degenerate shapes; leave
                # untagged so the extractor sees zero.
                continue
            # Stash solver groups on the equation.  The Expression
            # returned by auto_solver_tag carries ``_solver_groups``;
            # the Equation interface expects ``get_solver_tag`` /
            # ``solver_tags`` / ``untagged_remainder``.
            eq._solver_groups = getattr(tagged, "_solver_groups", {})

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

        # 1. Parameters — symbolic / numeric split (canonical naming).
        # ``self.parameters`` — Zstruct of sympy Symbols used in
        #   equations and in the operator API (``flux``, ``source``,
        #   …).  This is the *symbolic identity* of each parameter.
        # ``self.parameter_values`` — Zstruct of numeric floats,
        #   user-mutable (e.g. ``model.parameter_values.g = 12.0``).
        #   Carried into ``SystemModel.parameter_values`` and lifted
        #   into the runtime callable argument named ``parameter`` at
        #   the printer / ``lambdify`` boundary.
        self.parameters = parse_definition_to_zstruct(p_def, "p")
        self.parameters._symbolic_name = "p"
        defaults = extract_parameter_defaults(p_def)
        self.parameter_values = Zstruct(
            **{k: float(defaults.get(k, 0.0)) for k in self.parameters.keys()}
        )
        self.parameter_values._symbolic_name = "p"

        # 2. Parse Variables
        self.variables = parse_definition_to_zstruct(var_def, "q")
        self.variables._symbolic_name = "Q"

        # 3. Parse Aux
        self.aux_variables = parse_definition_to_zstruct(aux_def, "qaux")
        self.aux_variables._symbolic_name = "Qaux"

        self.n_variables = self.variables.length()
        self.n_aux_variables = self.aux_variables.length()
        # Equal cardinality on the symbol Zstruct and the values Zstruct
        # — same keys, just different value types.
        self.n_parameters = self.parameters.length()

        self.time, self.distance = sp.symbols("t dX", real=True)
        # Position carries x / y / z (matches StateSpace + SystemModel
        # + sympy convention).  The `_symbolic_name = "X"` is preserved
        # for the codegen path which prefixes positional args as `X[d]`.
        self.position = parse_definition_to_zstruct(
            ["x", "y", "z"], "X")
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

        ic_sig = Zstruct(position=self.position, p=self.parameters)

        # Gradient symbols — still produced for code-generation backends
        # (generic_c / amrex) that carry ``∇Q`` as an LDG-style explicit
        # auxiliary field.  The standard operator API
        # (``diffusion_matrix``, ``flux``, ``source``, …) is purely a
        # function of ``(Q, Qaux, p)``; derivatives that diffusion
        # depends on enter via ``Qaux`` (auto-exposed by
        # :meth:`SystemModel.expose_aux_atoms`).
        grad_syms = []
        for v in self.variables.keys():
            for d in range(self.dimension):
                grad_syms.append(sp.Symbol(f"dQ_{v}_d{d}", real=True))
        self.gradient_variables = parse_definition_to_zstruct(
            [str(s) for s in grad_syms], "gradQ"
        )
        self.gradient_variables._symbolic_name = "gradQ"

        regs = [
            ("flux", self.flux, std_sig),
            ("diffusion_matrix", self.diffusion_matrix, std_sig),
            ("diffusion_matrix_explicit",
             self.diffusion_matrix_explicit, std_sig),
            ("source_explicit", self.source_explicit, std_sig),
            ("dflux", self.dflux, std_sig),
            ("hydrostatic_pressure", self.hydrostatic_pressure, std_sig),
            ("nonconservative_matrix", self.nonconservative_matrix, std_sig),
            ("quasilinear_matrix", self.quasilinear_matrix, std_sig),
            ("eigenvalues", self.eigenvalues, eig_sig),
            ("left_eigenvectors", self.left_eigenvectors, eig_sig),
            ("right_eigenvectors", self.right_eigenvectors, eig_sig),
            ("eigensystem", self.eigensystem, eig_sig),
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
            ("interpolate_to_3d", self.interpolate_to_3d, proj_sig),
            ("project_from_3d", self.project_from_3d, std_sig),
            ("residual", self.residual, res_sig),
            ("interpolate", self.interpolate, std_sig),
            ("initial_condition", self.initial_condition, ic_sig),
            ("initial_aux_condition", self.initial_aux_condition, ic_sig),
            ("update_variables", self.update_variables, std_sig),
            ("update_aux_variables", self.update_aux_variables, std_sig),
            ("reconstruction_variables",
             self.reconstruction_variables, std_sig),
            ("state_from_reconstruction",
             self.state_from_reconstruction, std_sig),
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

        # 1. Main Boundary Conditions — value + face-normal gradient.
        # Both are built as indexed ``Piecewise`` ``Function`` kernels
        # so the runtime can call them as
        # ``rt.boundary_conditions(bc_idx, …)`` /
        # ``rt.boundary_gradients(bc_idx, …)`` per boundary face.
        self._boundary_conditions = (
            self.boundary_conditions.get_boundary_condition_function(
                self.time,
                self.position,
                self.distance,
                self.variables,
                self.aux_variables,
                self.parameters,
                self.normal,
                function_name="boundary_conditions",
            )
        )
        self._boundary_gradients = (
            self.boundary_conditions.get_boundary_gradient_function(
                self.time,
                self.position,
                self.distance,
                self.variables,
                self.aux_variables,
                self.parameters,
                self.normal,
                function_name="boundary_gradients",
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
                self.parameters,
                self.normal,
                function_name="aux_boundary_conditions",  # [FIX] Pass the name here!
            )
        )
        self._aux_boundary_conditions.name = "aux_boundary_conditions"

    def print_boundary_conditions(self):
        """Print boundary conditions."""
        return self._boundary_conditions.definition

    # --- Physics Methods ---
    # Default implementations route through ``tag_extraction``: every
    # equation in ``self.equations`` is walked, terms are pulled by
    # canonical solver tag, and the resulting matrix is translated
    # from derivation Symbols / Function calls (e.g. ``h(t, x)``,
    # ``q(k, t, x)``, ``Symbol("g", positive=True)``) to Model Symbols
    # (``self.variables.h``, ``self.variables.q_k``,
    # ``self.parameters.g``) via ``self._symbol_map()``.
    #
    # Subclasses that prefer hand-written operators (legacy SWE-style)
    # may still override these methods directly — the override wins
    # and ``self.equations`` need not be populated.

    def _extract_via_tag(self, canonical_tag):
        """Walk ``self._equations.values()`` and collect every term
        carrying the canonical solver tag into an operator matrix.

        Returns ``None`` (signal to fall back to a zero default) when
        the model has no equations or no variable map (i.e. the
        subclass has not declared which equation goes into which row
        of the operator matrix).

        In the new design, equation atoms are already Model Symbols
        (after derive_model's Function→Symbol substitution).  No
        post-extraction substitution / xreplace is needed.
        """
        if not self._equations or not self._variable_map:
            return None
        from zoomy_core.model.derivation.tag_extraction import (
            collect_solver_tag,
            canonical_solver_tag,
        )
        canonical = canonical_solver_tag(canonical_tag)
        state_atoms = [v for v in self.variables.values()
                       if isinstance(v, sp.Symbol)]
        coords = [self.position[d] for d in range(self.dimension)]
        n_dir = self.dimension if canonical in (
            "flux", "hydrostatic_pressure", "nonconservative_flux") else 1
        # `collect_solver_tag` walks `system._equations` natively (we
        # wrap self in an adapter only because the legacy signature
        # expected `.equations` dict-attr — Model exposes `_equations`
        # now).
        class _EqAdapter:
            equations = self._equations
        return collect_solver_tag(
            _EqAdapter(), canonical,
            variable_map=self._variable_map,
            n_variables=self.n_variables,
            n_directions=n_dir,
            state_variables=state_atoms,
            coords=coords,
            policy="warn",
        )

    def flux(self):
        """Flux ``F(Q, Qaux, p)`` — rank-2 ZArray ``(n_eq, n_dim)``.

        Default: walks ``self.equations`` via tag extraction, pulls
        every ``flux``-tagged term, substitutes derivation Symbols
        for Model Symbols, returns as ``ZArray``.  Falls back to
        zeros when the derivation tree is empty.
        """
        raw = self._extract_via_tag("flux")
        if raw is None:
            return ZArray.zeros(self.n_variables, self.dimension)
        return ZArray(raw)

    def diffusion_matrix(self):
        """Diffusion matrix A(Q, Qaux, p) — **implicit treatment**.

        Shape ``(n_variables, n_variables, dimension, dimension)``.

        Defines the diffusive flux structurally via
        ``F_diff[i, d] = Σ_{j, e} A[i, j, d, e] · ∂_e Q[j]``; the PDE
        residual contributes ``-∇·(A:∇Q)``.  ``A`` is a pure ``(Q,
        Qaux, p)`` expression — derivatives enter only via ``Qaux``,
        exposed automatically by :meth:`SystemModel.expose_aux_atoms`.

        IMEX-capable backends evaluate this slot at ``Qnp1`` inside the
        source step (no parabolic CFL).  Override
        :meth:`diffusion_matrix_explicit` instead for explicit
        treatment.  Default: zero tensor (no diffusion).
        """
        return ZArray.zeros(
            self.n_variables, self.n_variables,
            self.dimension, self.dimension,
        )

    def diffusion_matrix_explicit(self):
        """Diffusion matrix A(Q, Qaux, p) — **explicit treatment**.

        Same shape and contraction contract as :meth:`diffusion_matrix`.
        IMEX-capable backends evaluate this slot at ``Qn`` inside the
        convective step (Forward-Euler-equivalent; subject to the
        parabolic CFL ``dt ≤ h²/(2ν)``).  Default: zero tensor.

        A SystemModel may declare *both* implicit and explicit
        diffusion contributions; the solver adds each at the
        appropriate stage.  Explicit-only backends compound:
        ``A_total = A_implicit + A_explicit``.
        """
        return ZArray.zeros(
            self.n_variables, self.n_variables,
            self.dimension, self.dimension,
        )

    def dflux(self):
        """Dflux (legacy)."""
        return ZArray.zeros(self.n_variables, self.dimension)
    
    def hydrostatic_pressure(self):
        """Hydrostatic pressure ``P(Q, Qaux, p)`` — rank-2 ZArray
        ``(n_eq, n_dim)``.

        Default: tag-extracted from ``self.equations`` (canonical tag
        ``hydrostatic_pressure``).  Falls back to zeros."""
        raw = self._extract_via_tag("hydrostatic_pressure")
        if raw is None:
            return ZArray.zeros(self.n_variables, self.dimension)
        return ZArray(raw)

    def nonconservative_matrix(self):
        """Nonconservative matrix ``B(Q, Qaux, p)`` — rank-3 ZArray
        ``(n_eq, n_state, n_dim)``.

        Default: tag-extracted from ``self.equations`` (canonical tag
        ``nonconservative_flux``).  Falls back to zeros."""
        raw = self._extract_via_tag("nonconservative_flux")
        if raw is None:
            return ZArray.zeros(self.n_variables, self.n_variables, self.dimension)
        return ZArray(raw)

    def source(self):
        """Source — **implicit treatment** (Manning friction,
        reactions, stiff body forces).  Rank-1 ZArray of length
        ``n_variables``.

        Default: tag-extracted from ``self.equations`` (canonical tag
        ``implicit_source``).  Falls back to zeros."""
        raw = self._extract_via_tag("implicit_source")
        if raw is None:
            return ZArray.zeros(self.n_variables)
        return ZArray(raw)

    def source_explicit(self):
        """Source — **explicit treatment** (non-stiff body forces,
        gravity, prescribed momentum sources).  Rank-1 ZArray of
        length ``n_variables``.

        Default: tag-extracted from ``self.equations`` (canonical tag
        ``explicit_source``).  Falls back to zeros."""
        raw = self._extract_via_tag("explicit_source")
        if raw is None:
            return ZArray.zeros(self.n_variables)
        return ZArray(raw)

    def mass_matrix(self):
        """Mass matrix ``M(Q, Qaux, p)`` — rank-2 ZArray
        ``(n_eq, n_state)``.

        Extracted from ``time_derivative``-tagged terms.  Each equation
        row ``i`` contributes ``M[i, j] = coefficient of ∂_t Q[j]``
        for ``j = 0..n_state-1``.  Rows with no ``time_derivative``
        term (e.g. elliptic constraint equations for pressure modes
        in VAM, or trivial conservation that lost its ∂_t coefficient)
        come out as all-zero rows — a singular M flags the constraint
        to the solver, which must use a DAE-aware time-stepper or a
        split formulation.

        The mass matrix arises naturally from the symbolic derivation;
        every ``∂_t Q[j]`` atom on the LHS of an equation contributes
        to its row.  There is NO fallback — if the derivation didn't
        produce time-derivative terms, the model genuinely has no
        time-evolution slot, and the solver gets a singular M as the
        honest answer.
        """
        raw = self._extract_via_tag("time_derivative")
        if raw is None:
            return ZArray.zeros(self.n_variables, self.n_variables)
        state_atoms = [v for v in self.variables.values()
                       if isinstance(v, sp.Symbol)]
        n_eq = self.n_variables
        n_state = len(state_atoms)
        M = sp.zeros(n_eq, n_state)
        t = self.time
        for i, row_expr in enumerate(raw):
            if row_expr == 0:
                continue
            expanded = sp.expand(row_expr)
            for j, q_j in enumerate(state_atoms):
                coef = expanded.coeff(sp.Derivative(q_j, t))
                if coef != 0:
                    M[i, j] = coef
        return ZArray(M)

    def residual(self):
        """Residual."""
        return ZArray.zeros(self.n_variables)

    def interpolate(self):
        """Interpolate."""
        return ZArray(self.variables)

    def interpolate_to_3d(self):
        """Vertical reconstruction (canonical operator).

        Declarative models build the rows in ``derive_model`` and stash them
        as ``self._interpolate_rows`` ({profile_slot: expr}); the extraction
        picks the dict up through this method and parses it with the same
        machinery as ``register_group`` — one definition, one parser.
        Production overrides returning a ZArray keep their old meaning."""
        rows = getattr(self, "_interpolate_rows", None)
        if rows is not None:
            return rows
        return ZArray.zeros(6)

    def project_from_3d(self):
        """Profile → state projection (canonical operator); see
        :meth:`interpolate_to_3d` for the stash/dict contract
        ({state_field_or_slot: expr})."""
        rows = getattr(self, "_project_rows", None)
        if rows is not None:
            return rows
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

    def reconstruction_variables(self):
        """Symbolic map ``state → primitive well-balanced variables``
        used by MUSCL-style reconstruction.

        Override in subclasses to limit *primitive* quantities (η = h+b,
        u = q_U/h, …) instead of *conservative* ones (h, q_U), which
        bounds the limited values by physical scales and removes
        momentum overshoot at wet/dry fronts (Audusse-Bouchut-Bristeau
        et al.).  Default: identity — the reconstruction layer then
        limits the conservative state directly.

        Returns a ZArray of length ``n_variables``; entry ``k`` is the
        symbolic expression for the reconstruction-side variable in
        the same slot.  The inverse is auto-derived from this expression
        (see :meth:`state_from_reconstruction`); subclasses with a
        non-invertible or hand-tuned inverse may override
        ``state_from_reconstruction`` directly.

        See ``thesis/chapters/30_numerics.md`` "Primitive-variable
        MUSCL reconstruction".

        Declarative models stash ``self._reconstruction_rows``
        ({state_field_or_slot: expr}) in ``derive_model``; unregistered
        slots default to the identity at extraction time.
        """
        rows = getattr(self, "_reconstruction_rows", None)
        if rows is not None:
            return rows
        return ZArray(self.variables)

    def state_from_reconstruction(self):
        """Symbolic inverse of :meth:`reconstruction_variables`:
        primitive reconstruction variables → conservative state.

        Default: auto-derived from :meth:`reconstruction_variables`
        via :func:`zoomy_core.model.reconstruction_inverse.invert_reconstruction`.
        The result is expressed in terms of fresh ``WB_<state_name>``
        symbols (one per state slot) which the runtime feeds with the
        reconstructed primitive face values; the callable then returns
        the conservative face state.

        Override only if sympy's solver cannot close the inverse
        symbolically (rare — the SWE / SME / VAM cases all close in
        closed form).
        """
        from zoomy_core.model.reconstruction_inverse import invert_reconstruction
        return invert_reconstruction(
            self.reconstruction_variables(),
            list(self.variables.get_list()),
        )

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

    def eigensystem(self):
        """Eigendecomposition of the normal-projected quasilinear matrix A_n,
        stacked as ``[Lambda(n), R(n*n), L(n*n)]`` (row-major), each entry an
        opaque ``eigensystem`` kernel call — numerical in the backends.  The Roe
        scheme builds ``|A| = R|Lambda|L`` from this."""
        from zoomy_core.model.kernel_functions import eigensystem as _es
        n = self.n_variables
        qm = self.quasilinear_matrix()
        A_flat = [
            sum(qm[i, j, d] * self.normal[d] for d in range(self.dimension))
            for i in range(n) for j in range(n)
        ]
        return ZArray([_es(sp.Integer(idx), *A_flat) for idx in range(n + 2 * n * n)])

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

        params = dict(self.parameter_values.as_dict(recursive=False))

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


class ModelDescription:
    """Markdown / plain-text description of a :class:`Model`.

    Returned by :meth:`Model.describe`.  Renders LaTeX equations via
    ``_repr_markdown_`` in Jupyter; ``str(...)``  /  ``print(...)`` give
    the plain-text fallback.
    """

    def __init__(self, model, *, show_history=False, include_minor=False):
        self._model = model
        self._show_history = show_history
        self._include_minor = include_minor

    # ── Jupyter LaTeX path ──────────────────────────────────────────
    def _repr_markdown_(self):
        m = self._model
        n_eq = len(m._equations)
        n_ops = len(m.history)
        parts = [
            f"**{type(m).__name__}** `{m.name}` — "
            f"{n_eq} equation{'s' if n_eq != 1 else ''}, "
            f"{n_ops} op{'s' if n_ops != 1 else ''}"
        ]
        # State + parameters one-line summaries (only if non-empty).
        if hasattr(m, "variables") and m.variables.length():
            state_syms = ", ".join(
                f"${sp.latex(v)}$" for v in m.variables.values()
            )
            parts.append(f"**State $Q$:** {state_syms}")
        if hasattr(m, "parameters") and m.parameters.length():
            param_kv = ", ".join(
                f"${sp.latex(m.parameters[k])} = "
                f"{getattr(m.parameter_values, k, '?')}$"
                for k in m.parameters.keys()
            )
            parts.append(f"**Parameters:** {param_kv}")
        # Equations as LaTeX, one per line.
        if n_eq:
            parts.append("**Equations:**")
            for name, eq in m._equations.items():
                latex = sp.latex(eq.expr)
                parts.append(
                    f"- `{name}`: $\\displaystyle {latex} \\;=\\; 0$"
                )
        # Optional history.
        if self._show_history and m.history:
            parts.append("**History:**")
            for h in m.history:
                if not self._include_minor and h.get("level") == "minor":
                    continue
                desc = (f" — {h['description']}"
                        if h.get("description") else "")
                parts.append(
                    f"- `[{h['op']}]` target=`{h['target']}`{desc}"
                )
        return "\n\n".join(parts)

    # ── Plain-text fallback ─────────────────────────────────────────
    def __str__(self):
        m = self._model
        lines = [f"{type(m).__name__}({m.name!r}) — "
                 f"{len(m._equations)} equations, "
                 f"{len(m.history)} ops"]
        for name, eq in m._equations.items():
            lines.append(f"  {name}  :  {eq.expr}  =  0")
        if self._show_history and m.history:
            lines.append("")
            lines.append("history:")
            for h in m.history:
                if not self._include_minor and h.get("level") == "minor":
                    continue
                desc = (f" — {h['description']}"
                        if h.get("description") else "")
                lines.append(f"  [{h['op']}] target={h['target']}{desc}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

