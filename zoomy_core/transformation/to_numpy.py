"""Module `zoomy_core.transformation.to_numpy`."""

from typing import Callable, Dict, Optional
import sympy as sp
import numpy as np

from zoomy_core.misc.custom_types import FArray
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
            # Fallback: try without CSE, then try converting NDimArray to list
            try:
                compiled = sp.lambdify(args, expr, modules=modules)
            except Exception:
                if hasattr(expr, 'tolist'):
                    compiled = sp.lambdify(args, expr.tolist(), modules=modules)
                else:
                    raise

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

        def fast_flatten(runtime_args):
            result = []
            for arg_idx, path in plan:
                val = runtime_args[arg_idx]
                for step_type, step_key, step_idx in path:
                    if isinstance(val, np.ndarray):
                        val = val[step_idx]
                    elif step_type == "key" and hasattr(val, step_key):
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
        self.parameters: FArray = model.parameter_values

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

        # Boundary condition wrappers are currently kept as separate members
        # on the symbolic model and are not part of model.functions.
        if hasattr(model, "_boundary_conditions"):
            self.runtime_functions["boundary_conditions"] = self._lambdify_function(
                model._boundary_conditions, modules
            )
        if hasattr(model, "_aux_boundary_conditions"):
            self.runtime_functions["aux_boundary_conditions"] = self._lambdify_function(
                model._aux_boundary_conditions, modules
            )

        # Keep attribute-style access for existing solver code paths.
        for name, function in self.runtime_functions.items():
            setattr(self, name, function)


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
