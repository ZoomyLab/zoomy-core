"""Module `zoomy_core.transformation.to_numpy`."""

from typing import Callable, Dict, Optional
import sympy as sp
import numpy as np

from zoomy_core.misc.custom_types import FArray
from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.basefunction import Function
from zoomy_core.model.basemodel import Model


class NumpyRuntimeModel:
    """Runtime model generated from a symbolic Model.

    Instead of assuming hardcoded attributes (e.g. ``_flux``), this class
    compiles all functions registered in ``model.functions`` and exposes them
    as callable runtime attributes.
    """

    # --- Constants ---
    module = {
        "ones_like": np.ones_like,
        "zeros_like": np.zeros_like,
        "array": np.array,
        "squeeze": np.squeeze,
        "conditional": lambda c, t, f: np.where(c, t, f),
        "clamp_positive": lambda x: np.maximum(x, 0.0),
        "clamp_momentum": lambda hu, h, u_max: np.clip(hu, -h * u_max, h * u_max),
        "max_wavespeed": None,  # must be provided by the solver before compilation
    }
    printer = "numpy"
    
    def _flatten_signature_args(self, arg_struct):
        """Internal helper `_flatten_signature_args`."""
        flat_args = []

        def _flatten(value):
            """Internal helper `_flatten`."""
            if hasattr(value, "values") and callable(value.values):
                for item in value.values():
                    _flatten(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    _flatten(item)
            else:
                flat_args.append(value)

        _flatten(arg_struct)
        return flat_args

    def _lambdify_function(self, function_obj, modules):
        """Internal helper `_lambdify_function`."""
        args = self._flatten_signature_args(function_obj.args)
        expr = self._vectorize_expression(function_obj.definition, function_obj.args)

        use_cse = getattr(self, 'use_cse', True)
        try:
            compiled = sp.lambdify(args, expr, modules=modules, cse=use_cse)
        except (TypeError, Exception):
            compiled = sp.lambdify(args, expr, modules=modules)

        fast_flatten = self._compile_flattener(function_obj.args)

        def runtime_callable(*runtime_args):
            """Runtime callable."""
            return compiled(*fast_flatten(runtime_args))

        return runtime_callable

    @staticmethod
    def _compile_flattener(signature):
        """
        Pre-compute the arg extraction plan at compile time.

        Returns a fast closure that converts runtime args to a flat list
        using only integer indexing — no Zstruct iteration at runtime.
        """
        plan = []

        def _build_plan(expected, arg_idx, path):
            if hasattr(expected, "values") and callable(expected.values):
                keys = list(expected.keys()) if hasattr(expected, "keys") else None
                for i, child in enumerate(expected.values()):
                    child_path = path + (("key", keys[i], i),)
                    _build_plan(child, arg_idx, child_path)
            elif isinstance(expected, (list, tuple)):
                for i, child in enumerate(expected):
                    _build_plan(child, arg_idx, path + (("idx", None, i),))
            else:
                plan.append((arg_idx, path))

        sig_values = list(signature.values()) if hasattr(signature, "values") else list(signature)
        for arg_idx, expected in enumerate(sig_values):
            _build_plan(expected, arg_idx, ())

        def _safe_hasattr(obj, name):
            """``hasattr`` shadowed by ``getattr`` with broad exception
            handling — UFL exposes ``.T`` as a property that raises
            ``ValueError`` on rank-1 tensors, and Python's ``hasattr``
            returns ``False`` only on AttributeError, propagating
            everything else.  Treat any exception as 'attribute not
            usable here'."""
            try:
                getattr(obj, name)
                return True
            except Exception:
                return False

        def fast_flatten(runtime_args):
            result = []
            for arg_idx, path in plan:
                val = runtime_args[arg_idx]
                for step_type, step_key, step_idx in path:
                    if isinstance(val, np.ndarray):
                        val = val[step_idx]
                    elif step_type == "key" and _safe_hasattr(val, step_key):
                        val = getattr(val, step_key)
                    else:
                        try:
                            val = val[step_idx]
                        except (TypeError, IndexError, KeyError):
                            pass
                result.append(val)
            return result

        return fast_flatten

    def _flatten_runtime_args(self, signature, runtime_args):
        """Internal helper `_flatten_runtime_args`."""
        expected_args = signature.values() if hasattr(signature, "values") else signature
        expected_args = list(expected_args)
        if len(runtime_args) != len(expected_args):
            raise TypeError(
                f"Expected {len(expected_args)} runtime args, got {len(runtime_args)}"
            )

        flat = []
        for expected, value in zip(expected_args, runtime_args):
            flat.extend(self._flatten_runtime_value(expected, value))
        return flat

    def _flatten_runtime_value(self, expected, value):
        """Internal helper `_flatten_runtime_value`."""
        if hasattr(expected, "values") and callable(expected.values):
            out = []
            keys = list(expected.keys()) if hasattr(expected, "keys") else None
            for i, child in enumerate(expected.values()):
                child_value = self._extract_component(value, i, keys[i] if keys else None)
                out.extend(self._flatten_runtime_value(child, child_value))
            return out

        if isinstance(expected, (list, tuple)):
            out = []
            for i, child in enumerate(expected):
                child_value = self._extract_component(value, i, None)
                out.extend(self._flatten_runtime_value(child, child_value))
            return out

        return [value]

    def _collect_vector_symbols(self, signature):
        """Internal helper `_collect_vector_symbols`."""
        vector_symbols = []
        vector_keys = {"variables", "aux_variables", "normal", "position"}

        def _collect(node, key=None, in_vector_context=False):
            """Internal helper `_collect`."""
            context = in_vector_context or (key in vector_keys)
            if hasattr(node, "values") and callable(node.values):
                keys = list(node.keys()) if hasattr(node, "keys") else [None] * len(node.values())
                for child_key, child_value in zip(keys, node.values()):
                    _collect(child_value, key=child_key, in_vector_context=context)
            elif isinstance(node, (list, tuple)):
                for child_value in node:
                    _collect(child_value, key=None, in_vector_context=context)
            else:
                if context and isinstance(node, sp.Basic):
                    vector_symbols.append(node)

        _collect(signature)
        return tuple(vector_symbols)

    def _get_anchor_symbol(self, signature):
        """Internal helper `_get_anchor_symbol`."""
        if hasattr(signature, "contains") and signature.contains("variables"):
            variables = signature["variables"]
            if hasattr(variables, "values") and len(variables.values()) > 0:
                return list(variables.values())[0]

        vector_symbols = self._collect_vector_symbols(signature)
        return vector_symbols[0] if vector_symbols else None

    def _vectorize_expression(self, expr, signature):
        """Internal helper `_vectorize_expression`."""
        if not (hasattr(expr, "tolist") and callable(expr.tolist)):
            return expr
        shape = getattr(expr, "shape", ())
        if shape and any(int(s) == 0 for s in shape):
            # SymPy currently fails tolist() for empty dimensions.
            # Return an explicit SymPy empty array so lambdify can handle it.
            return sp.Array([], tuple(int(s) for s in shape))

        arr = sp.Array(expr.tolist())
        vector_symbols = self._collect_vector_symbols(signature)
        anchor = self._get_anchor_symbol(signature)
        if not vector_symbols or anchor is None:
            return arr

        ones_like = sp.Function("ones_like")
        zeros_like = sp.Function("zeros_like")
        flat = []
        for item in list(arr._array):
            if isinstance(item, sp.Basic) and not item.has(*vector_symbols):
                if item.is_zero:
                    flat.append(zeros_like(anchor))
                else:
                    flat.append(item * ones_like(anchor))
            else:
                flat.append(item)
        return sp.Array(flat, arr.shape)

    @staticmethod
    def _extract_component(value, index, key):
        """Internal helper `_extract_component`."""
        if key is not None and hasattr(value, key):
            return getattr(value, key)
        try:
            return value[index]
        except Exception:
            return value

    def __init__(
        self,
        model: Model,
        module: Optional[Dict[str, Callable]] = None,
        printer: Optional[str] = None,
        kernel=None,
    ):
        """Initialize the instance."""
        self.model = model
        self.name = model.name
        self.dimension = model.dimension
        self.n_variables = model.n_variables
        self.n_aux_variables = model.n_aux_variables
        self.n_parameters = model.n_parameters
        # Extract numeric parameter values from the symbolic model's Zstruct
        # at compile time. Values live in ``model.parameters`` (a plain Zstruct).
        self.parameters: FArray = np.array(list(model.parameters.values()), dtype=float)

        # Copy the class-level mapping to avoid shared mutable state.
        self.module = dict(type(self).module) if module is None else dict(module)
        self.printer = type(self).printer if printer is None else printer

        # If a Kernel is provided, compile its functions and merge into module
        if kernel is not None:
            kernel_rt = NumpyRuntimeSymbolic(kernel, module=self.module, printer=self.printer)
            for name, fn in kernel_rt.runtime_functions.items():
                self.module[name] = fn

        modules = [self.module]
        if self.printer:
            modules.append(self.printer)

        self.runtime_functions: Dict[str, Callable] = {}
        for name, function_obj in model.functions.items():
            self.runtime_functions[name] = self._lambdify_function(function_obj, modules)

        # Boundary condition wrappers are kept as separate members on the
        # symbolic model and are not part of model.functions.  They are
        # *required* — if the model has no ``_boundary_conditions`` /
        # ``_aux_boundary_conditions`` / ``_boundary_gradients``, the
        # access fails loudly (per the zoomy "prefer breaking over silent
        # skip" rule).
        self.runtime_functions["boundary_conditions"] = self._lambdify_function(
            model._boundary_conditions, modules
        )
        self.runtime_functions["aux_boundary_conditions"] = self._lambdify_function(
            model._aux_boundary_conditions, modules
        )
        self.runtime_functions["boundary_gradients"] = self._lambdify_function(
            model._boundary_gradients, modules
        )

        # Keep attribute-style access for existing solver code paths.
        for name, function in self.runtime_functions.items():
            setattr(self, name, function)

    @classmethod
    def from_system_model(cls, sm, *, module=None, printer=None):
        """Build a runtime by lambdifying a :class:`SystemModel`'s
        stored matrices.

        Lightweight adapter that mirrors the operator-API surface of
        the runtime built from a ``Model``: the resulting object has
        ``flux``, ``nonconservative_matrix``, ``source``,
        ``hydrostatic_pressure``, ``mass_matrix`` callable attributes,
        each accepting ``(Q, Qaux, p)`` and returning the corresponding
        numpy array.

        For most solver code paths the existing ``NumpyRuntimeModel(model)``
        flow is still preferred because it carries the model's full
        function registry (``boundary_conditions`` kernel etc.) — this
        factory is the right entry point when the analysis or
        transformation pipeline starts from a SystemModel that may not
        have a backing Model (e.g. SystemModel constructed from
        scratch).
        """
        rt = cls.__new__(cls)
        # The SystemModel *is* the symbolic model for this runtime —
        # solvers unwrap it via ``_get_symbolic_model`` to build their
        # operators.
        rt.model = sm
        rt.name = "SystemModelRuntime"
        rt.dimension = sm.n_dim
        rt.n_variables = sm.n_equations
        rt.n_aux_variables = len(sm.aux_state)
        rt.n_parameters = sm.parameters.length()
        rt.parameters = np.array(list(sm.parameter_values.values()),
                                 dtype=float)
        rt.module = dict(cls.module) if module is None else dict(module)
        rt.printer = cls.printer if printer is None else printer

        modules = [rt.module]
        if rt.printer:
            modules.append(rt.printer)

        # Every operator is lambdified through ``_lambdify_function`` so
        # ``_vectorize_expression`` is applied first — that wraps every
        # constant entry in ``zeros_like(anchor)`` / ``c·ones_like(anchor)``
        # so the runtime broadcasts cleanly when called with full-grid
        # arrays (Q shape ``(n_vars, n_cells)`` → matrix output
        # ``(n_eq, n_state, n_cells)``).  Without this, IMEX
        # implicit-source / source-jacobian calls (and any vectorised
        # full-grid pattern) collapse to 2-D and break.
        #
        # Operator signatures (kwarg keys feed
        # ``_collect_vector_symbols``: ``variables`` / ``aux_variables``
        # / ``normal`` / ``position`` are the *vector* groups whose
        # symbols anchor the broadcast).
        Q_struct = sm.variables          # Zstruct of state Symbols
        Qaux_struct = sm.aux_variables   # Zstruct of aux Symbols
        p_struct = sm.parameters         # Zstruct of parameter Symbols
        n_struct = sm.normal             # Zstruct of normal Symbols

        std_sig = Zstruct(variables=Q_struct, aux_variables=Qaux_struct,
                          parameters=p_struct)
        eig_sig = Zstruct(variables=Q_struct, aux_variables=Qaux_struct,
                          parameters=p_struct, normal=n_struct)

        rt.runtime_functions = {}

        def _register(name, definition, signature):
            """Synthesize a ``Function`` and route through
            ``_lambdify_function`` so vectorisation is applied."""
            if definition is None:
                return
            fn = Function(name=name, args=signature, definition=definition)
            rt.runtime_functions[name] = rt._lambdify_function(fn, modules)
            setattr(rt, name, rt.runtime_functions[name])

        def _column_to_rank1(mat):
            """``(n, 1)`` Matrix → rank-1 ``sp.Array(n)``.  ``_to_matrix``
            in ``from_model`` coerces column-like ZArrays to ``(n, 1)``;
            consumers (IMEX ``S[:, c]``, the source operator broadcast
            pattern) expect rank-1 → ``(n_eq, n_cells)`` after
            vectorisation, not 3-D ``(n_eq, 1, n_cells)``."""
            if mat is None:
                return None
            return sp.Array([mat[i, 0] for i in range(mat.shape[0])])

        _register("flux", sm.flux, std_sig)
        _register("hydrostatic_pressure", sm.hydrostatic_pressure, std_sig)
        _register("source", _column_to_rank1(sm.source), std_sig)
        _register("source_explicit",
                  _column_to_rank1(sm.source_explicit), std_sig)
        _register("mass_matrix", sm.mass_matrix, std_sig)
        _register("source_jacobian", sm.source_jacobian, std_sig)
        _register("update_variables",
                  _column_to_rank1(sm.update_variables), std_sig)
        _register("eigenvalues", _column_to_rank1(sm.eigenvalues), eig_sig)
        # ``state_update``: explicit-update operator for split substeps
        # (e.g. Chorin corrector).  Returns a rank-1 array of length
        # ``len(equation_to_state_index)``: the new values for those
        # state slots.  Same broadcast pattern as ``source``.
        _register("state_update", sm.state_update, std_sig)

        # ``∂S/∂Q`` is exposed under both names — the Model-based
        # runtime calls it ``source_jacobian_wrt_variables``.
        if "source_jacobian" in rt.runtime_functions:
            rt.source_jacobian_wrt_variables = (
                rt.runtime_functions["source_jacobian"])

        # NDimArray operators (NCP, quasilinear) — per-axis slab as a
        # Matrix, each routed through ``_lambdify_function`` (so each
        # slab is vectorised), then ``np.stack`` along the last axis.
        n_dim = sm.n_dim
        n_eq = sm.n_equations
        n_st = sm.n_state

        def _register_ndarray(name, arr, n_cols):
            slab_fns = []
            for d in range(n_dim):
                slab = sp.Matrix(
                    n_eq, n_cols,
                    lambda i, j, _d=d: arr[i, j, _d],
                )
                fn = Function(name=f"{name}__d{d}", args=std_sig,
                              definition=slab)
                slab_fns.append(rt._lambdify_function(fn, modules))

            def _runtime(Q, Qaux, p, _slab_fns=slab_fns):
                slabs = [np.asarray(f(Q, Qaux, p), dtype=float)
                         for f in _slab_fns]
                return np.stack(slabs, axis=-1)
            rt.runtime_functions[name] = _runtime
            setattr(rt, name, _runtime)

        _register_ndarray("nonconservative_matrix",
                          sm.nonconservative_matrix, n_eq)
        _register_ndarray("quasilinear_matrix",
                          sm.quasilinear_matrix, n_st)

        # ``diffusion_matrix``: rank-4 ``A(Q, Qaux, p)`` of shape
        # ``(n_eq, n_state, n_dim, n_dim)`` — the constitutive tensor in
        # ``div(A : grad Q)``.  Lambdify per ``(d_flux, d_grad)`` slab as
        # an ``(n_eq, n_state)`` matrix and stack along the trailing two
        # axes.  Skipped when the SystemModel does not carry diffusion.
        def _register_rank4(name, A_arr):
            if A_arr is None:
                return
            slab_fns_4d = []
            for d in range(n_dim):
                for e in range(n_dim):
                    slab = sp.Matrix(
                        n_eq, n_st,
                        lambda i, j, _d=d, _e=e: A_arr[i, j, _d, _e],
                    )
                    fn = Function(name=f"{name}__d{d}_e{e}",
                                  args=std_sig, definition=slab)
                    slab_fns_4d.append(rt._lambdify_function(fn, modules))

            def _runtime_A(Q, Qaux, p, _fns=slab_fns_4d,
                           _n_dim=n_dim, _n_eq=n_eq, _n_st=n_st):
                slabs = [np.asarray(f(Q, Qaux, p), dtype=float)
                         for f in _fns]
                stacked = np.stack(slabs, axis=-1)
                new_shape = stacked.shape[:-1] + (_n_dim, _n_dim)
                return stacked.reshape(new_shape)
            rt.runtime_functions[name] = _runtime_A
            setattr(rt, name, _runtime_A)

        _register_rank4("diffusion_matrix", sm.diffusion_matrix)
        _register_rank4("diffusion_matrix_explicit", sm.diffusion_matrix_explicit)

        # Indexed boundary-condition kernels — lambdified via
        # ``_lambdify_function`` so the per-face call is
        # ``rt.boundary_conditions(bc_idx, time, position, distance,
        # Q_cell, Qaux_cell, parameters, normal) → q_face``.  The
        # SystemModel must carry both (per the zoomy rule: prefer
        # breaking over silent skip).
        rt.runtime_functions["boundary_conditions"] = rt._lambdify_function(
            sm.boundary_conditions, modules)
        rt.boundary_conditions = rt.runtime_functions["boundary_conditions"]
        rt.runtime_functions["aux_boundary_conditions"] = rt._lambdify_function(
            sm.aux_boundary_conditions, modules)
        rt.aux_boundary_conditions = rt.runtime_functions[
            "aux_boundary_conditions"]
        rt.runtime_functions["boundary_gradients"] = rt._lambdify_function(
            sm.boundary_gradients, modules)
        rt.boundary_gradients = rt.runtime_functions["boundary_gradients"]

        return rt


class NumpyRuntimeSymbolic(NumpyRuntimeModel):
    """
    Runtime wrapper for generic symbolic registrars (e.g. Numerics).

    Compiles all entries from ``symbolic_obj.functions`` using the same
    lambdify/argument-flattening machinery as ``NumpyRuntimeModel``.
    """

    def __init__(
        self,
        symbolic_obj,
        module: Optional[Dict[str, Callable]] = None,
        printer: Optional[str] = None,
    ):
        """Initialize the instance."""
        self.symbolic_obj = symbolic_obj
        self.module = dict(type(self).module) if module is None else dict(module)
        self.printer = type(self).printer if printer is None else printer

        modules = [self.module]
        if self.printer:
            modules.append(self.printer)

        self.runtime_functions: Dict[str, Callable] = {}
        for name, function_obj in symbolic_obj.functions.items():
            self.runtime_functions[name] = self._lambdify_function(function_obj, modules)

        for name, function in self.runtime_functions.items():
            setattr(self, name, function)
