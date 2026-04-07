"""
Symbolic numerics: regularization and numerical safety for PDE models.

This module provides tools to make a symbolic Model numerically safe
WITHOUT creating a separate NumericalModel with duplicate symbols.

The approach:
1. Take the analytical model as-is (keep its symbols)
2. Walk its expressions and replace dangerous patterns:
   - h^(-n) → (h + eps)^(-n)  (denominator regularization)
   - sqrt(h) → sqrt(h + eps)  (fractional power regularization)
3. Add an update_variables() function using opaque sympy functions:
   - clamp_positive(h): max(h, 0)
   - clamp_momentum(hu, h, u_max): clip velocity
   - conditional(c, t, f): vectorized if-then-else
4. These opaque functions are defined in custom_sympy_functions.py
   and have backend implementations in to_numpy.py (numpy),
   to_jax.py (JAX), etc.

Usage:
    from zoomy_core.model.symbolic_numerics import regularize_model

    model = ProjectedModel(...)         # analytical model
    reg_model = regularize_model(model)  # same symbols, regularized expressions
    # reg_model can be passed directly to NumpyRuntimeModel
"""

import sympy as sp
from sympy import Symbol, Pow, Rational, S

from zoomy_core.model.custom_sympy_functions import (
    clamp_positive, clamp_momentum, conditional,
)
from zoomy_core.misc.misc import ZArray


def regularize_denominators(expr, h_sym, eps_sym):
    """Replace h^(-n) with (h + eps)^(-n) in the expression.

    Finds all negative powers of ``h_sym`` and replaces the base
    with ``h_sym + eps_sym``.

    Parameters
    ----------
    expr : sympy expression
    h_sym : Symbol
        The water depth variable
    eps_sym : Symbol
        The regularization parameter (small positive number)

    Returns
    -------
    sympy expression with regularized denominators
    """
    if not isinstance(expr, sp.Basic):
        return expr

    def _replace(e):
        if isinstance(e, Pow):
            base, exp = e.args
            if base == h_sym and exp.is_negative:
                return Pow(h_sym + eps_sym, exp)
            if base == h_sym and isinstance(exp, Rational) and exp < 0:
                return Pow(h_sym + eps_sym, exp)
        if e.args:
            new_args = [_replace(a) for a in e.args]
            return e.func(*new_args)
        return e

    return _replace(expr)


def regularize_sqrt_arguments(expr, h_sym, eps_sym):
    """Replace h inside fractional powers with (h + eps).

    Prevents sqrt(negative) in eigenvalue expressions.

    Handles patterns like ``h**(1/2)``, ``(g*h)**(1/2)``, etc.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    def _replace(e):
        if isinstance(e, Pow):
            base, exp = e.args
            if (isinstance(exp, Rational) and not exp.is_integer
                    and base.has(h_sym)):
                new_base = base.subs(h_sym, h_sym + eps_sym)
                return Pow(new_base, exp)
        if e.args:
            new_args = [_replace(a) for a in e.args]
            return e.func(*new_args)
        return e

    return _replace(expr)


def regularize_expr(expr, h_sym, eps_sym):
    """Apply all regularizations to an expression."""
    result = regularize_denominators(expr, h_sym, eps_sym)
    return result


def regularize_eigenvalue_expr(expr, h_sym, eps_sym):
    """Apply regularizations suitable for eigenvalue expressions."""
    result = regularize_denominators(expr, h_sym, eps_sym)
    result = regularize_sqrt_arguments(result, h_sym, eps_sym)
    return result


def build_update_variables(variables, h_index, scaled_indices, g_sym, eps_sym,
                            eps_wet_sym):
    """Build the update_variables expression using opaque functions.

    This creates the wet/dry treatment:
    1. clamp_positive(h) — prevent negative depth
    2. clamp_momentum(hu, h_safe, u_max) — cap velocity
    3. Linear ramp — zero momentum when h < eps_wet

    Parameters
    ----------
    variables : Zstruct
        The model's variable symbols
    h_index : int
        Index of h in the variable array
    scaled_indices : list of int
        Indices of momentum variables (hu, hv, etc.)
    g_sym : Symbol
        Gravity parameter
    eps_sym : Symbol
        Denominator regularization
    eps_wet_sym : Symbol
        Wet/dry threshold

    Returns
    -------
    ZArray
        Expression for updated variables
    """
    n = len(variables)
    result = ZArray([variables[i] for i in range(n)])

    h = variables[h_index]
    h_safe = clamp_positive(h)
    result[h_index] = h_safe

    u_max = sp.sqrt(g_sym * (h_safe + eps_sym)) + eps_wet_sym
    ramp = sp.Min(h_safe / eps_wet_sym, S.One)

    for i in scaled_indices:
        result[i] = clamp_momentum(result[i], h_safe, u_max) * ramp

    return result


def regularize_model(model, h_index=None, eps=1e-8, eps_wet=1e-3):
    """Regularize a model's expressions for numerical safety.

    This does NOT create a new model or duplicate symbols.
    It modifies the model's registered functions IN PLACE by:
    1. Adding eps, eps_wet parameters (if not present)
    2. Regularizing flux/source/eigenvalue denominators
    3. Adding update_variables with wet/dry treatment

    Parameters
    ----------
    model : Model
        The analytical model (ProjectedModel, VAMProjectedHyperbolic, etc.)
    h_index : int, optional
        Index of h in the variable array. Auto-detected if possible.
    eps : float
        Denominator regularization value
    eps_wet : float
        Wet/dry threshold

    Returns
    -------
    Model
        The same model object, with regularized expressions and
        added parameters (eps, eps_wet).
    """
    # Auto-detect h index
    if h_index is None:
        var_keys = list(model.variables.keys())
        if "h" in var_keys:
            h_index = var_keys.index("h")
        elif len(var_keys) > 1:
            h_index = 1  # default: second variable (after b)
        else:
            h_index = 0

    h_sym = model.variables[h_index]

    # Get or add eps parameters
    param_keys = list(model.parameters.keys())
    if "eps" in param_keys:
        eps_sym = model.parameters["eps"]
    else:
        eps_sym = Symbol("eps", positive=True)
        model.parameters["eps"] = eps_sym
        if hasattr(model, "parameter_defaults_map"):
            model.parameter_defaults_map["eps"] = eps
        if hasattr(model, "parameter_values"):
            model.parameter_values = list(model.parameter_values) + [eps]

    if "eps_wet" in param_keys:
        eps_wet_sym = model.parameters["eps_wet"]
    else:
        eps_wet_sym = eps_sym  # reuse eps for now (don't add new parameter)

    # Identify scaled (momentum) variables
    scaled_indices = [i for i in range(model.n_variables)
                      if i != h_index and i != 0]  # skip b and h

    # Store regularization info on the model
    model._h_index = h_index
    model._h_sym = h_sym
    model._eps_sym = eps_sym
    model._eps_wet_sym = eps_wet_sym
    model._scaled_indices = scaled_indices

    # Re-initialize compiled functions with regularized expressions
    if hasattr(model, '_initialize_functions'):
        model._initialize_functions()

    return model
