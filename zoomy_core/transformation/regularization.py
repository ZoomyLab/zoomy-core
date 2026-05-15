"""Numerical regularization passes for :class:`SystemModel`.

A :class:`~zoomy_core.model.models.system_model.SystemModel` is an
*analytical* PDE — it deliberately carries no numerical safety hacks.
This module provides symbolic ``SystemModel → SystemModel`` passes that
produce a *derived*, numerically-regularized SystemModel for the
code-transformation pipeline, leaving the analytical SystemModel
pristine.

The regularization is a floating-point-safety perturbation that
vanishes as ``eps → 0`` and does **not** change the analytical PDE —
hence it lives here, in the numerical layer, not in the Model /
SystemModel.  The result is "a SystemModel that has been regularized":
same type, no new class, the pass recorded in ``history``.  It shares
the analytical structure (``state`` / ``space`` / ``parameters`` / …)
with the input by reference — only the operator matrices are new.
"""
from __future__ import annotations

import itertools

import sympy as sp


def _reg_scalar(expr, positive_vars, eps):
    """Targeted denominator rewrite on a single scalar sympy expression.

    Replaces ``Pow(v, -n) → Pow(v + eps, -n)`` for each positive
    variable ``v`` — *only* inside negative powers.  Other occurrences
    of ``v`` (e.g. ``g*h``) are untouched.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    for v in positive_vars:
        expr = expr.replace(
            lambda sub: (isinstance(sub, sp.Pow)
                         and sub.base == v
                         and sub.exp.is_negative),
            lambda sub: sp.Pow(v + eps, sub.exp),
        )
    return expr


def _reg_ndarray(arr, reg):
    """Element-wise regularised copy of an ``NDimArray``."""
    out = sp.MutableDenseNDimArray.zeros(*arr.shape)
    for idx in itertools.product(*[range(s) for s in arr.shape]):
        out[idx] = reg(arr[idx])
    return out


def regularize_denominators(sm, eps=1e-8):
    """Return a numerically-regularized *copy* of ``sm``.

    Rewrites every ``Pow(v, -n)`` (division by a positive variable
    ``v`` — identified by its sympy ``positive`` assumption) in the
    operator matrices to ``Pow(v + eps, -n)``.  This is a
    floating-point-safety perturbation: it vanishes as ``eps → 0`` and
    does not change the analytical PDE.

    The input ``sm`` is left untouched; the returned SystemModel is a
    derived artifact — same ``SystemModel`` type, the analytical
    structure shared by reference, only the operators new and
    regularized, the pass recorded in ``history``.

    Parameters
    ----------
    sm : SystemModel
        The analytical SystemModel.
    eps : float | sympy.Expr
        Regularization constant (default ``1e-8``).  A sympy ``Symbol``
        may be passed to keep it symbolic.

    Returns
    -------
    SystemModel
        A derived, regularized copy.  If no positive variable is
        present (nothing to regularize) the input is returned unchanged.
    """
    from zoomy_core.model.models.system_model import SystemModel

    positive_vars = [s for s in list(sm.state) + list(sm.aux_state)
                     if getattr(s, "is_positive", False)]
    if not positive_vars:
        return sm
    eps_expr = sp.sympify(eps)

    def reg(e):
        return _reg_scalar(e, positive_vars, eps_expr)

    sj = (sm.source_jacobian.applyfunc(reg)
          if sm.source_jacobian is not None else None)
    ev = (sm.eigenvalues.applyfunc(reg)
          if sm.eigenvalues is not None else None)

    out = SystemModel(
        time=sm.time,
        space=list(sm.space),
        state=list(sm.state),
        aux_state=list(sm.aux_state),
        parameters=sm.parameters,
        flux=sm.flux.applyfunc(reg),
        hydrostatic_pressure=sm.hydrostatic_pressure.applyfunc(reg),
        nonconservative_matrix=_reg_ndarray(sm.nonconservative_matrix, reg),
        source=sm.source.applyfunc(reg),
        mass_matrix=sm.mass_matrix.applyfunc(reg),
        quasilinear_matrix=_reg_ndarray(sm.quasilinear_matrix, reg),
        source_jacobian=sj,
        eigenvalues=ev,
        normal=sm.normal,
        parameter_values=sm.parameter_values,
        equation_to_state_index=(list(sm.equation_to_state_index)
                                 if sm.equation_to_state_index is not None
                                 else None),
        boundary_conditions=sm.boundary_conditions,
        initial_conditions=sm.initial_conditions,
        aux_initial_conditions=sm.aux_initial_conditions,
        history=list(sm.history),
    )
    # Carry the dynamically-set attributes — regularization touches
    # only Pow bases (h → h+eps), so it introduces no new Function /
    # Derivative atoms: the aux scan and equation names are unchanged.
    if getattr(sm, "aux_registry", None) is not None:
        out.aux_registry = sm.aux_registry
    if hasattr(sm, "equation_names"):
        out.equation_names = list(sm.equation_names)

    out.history.append({
        "name": "regularize_denominators",
        "description": (
            f"numerical regularization: Pow(v,-n) → Pow(v+{eps}, -n) "
            f"for positive vars {[str(v) for v in positive_vars]}"
        ),
    })
    return out


__all__ = ["regularize_denominators"]
