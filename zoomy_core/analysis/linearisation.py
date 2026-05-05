"""Generic linearisation around a base state.

Replaces every state field ``q(t, x[, y, ...])`` in a ``PDESystem``'s
equations with ``q_0 + ε δq(t, x[, y, ...])``, expands to first order
in ε, and returns a new ``PDESystem`` whose fields are the
perturbations ``δq``.

Works for arbitrary equations — differential, algebraic, or mixed.
The base values ``q_0`` may themselves be expressions in coordinates
(useful when the steady state is non-uniform, e.g. a varying bottom
profile ``b(x)``).
"""
from __future__ import annotations

from typing import Dict

import sympy as sp

from .pde_system import PDESystem


def _delta_function(field, prefix=r"\delta "):
    """Construct ``δq(*args)`` for a state field ``q(*args)``."""
    head = field.func
    name = head.__name__ if hasattr(head, "__name__") else str(head)
    delta_head = sp.Function(prefix + name, real=True)
    return delta_head(*field.args)


def linearise(system: PDESystem, base_state: Dict, *, eps=None,
              simplify=True) -> PDESystem:
    """Insert ``q = q_0 + ε δq``, expand, return O(ε) system.

    Args:
        system:      the input PDE system.
        base_state:  dict mapping each field in ``system.fields`` to its
                     base value (can be a constant or an expression in
                     coordinates).
        eps:         small parameter symbol; created internally if None.
        simplify:    apply ``sp.expand`` to each linearised equation.

    Returns:
        a new ``PDESystem`` whose ``equations`` are the O(ε)
        coefficients and whose ``fields`` are the perturbation
        ``δq`` Function-calls (in the same order as the input fields).
    """
    if eps is None:
        eps = sp.Symbol("epsilon", positive=True)

    if set(base_state.keys()) != set(system.fields):
        missing = set(system.fields) - set(base_state.keys())
        extra = set(base_state.keys()) - set(system.fields)
        raise ValueError(
            f"base_state must list every field; missing={missing!r}, "
            f"extra={extra!r}."
        )

    # Build replacement map: q → q_0 + ε δq.
    delta_fields = []
    repl = {}
    for f in system.fields:
        df = _delta_function(f)
        delta_fields.append(df)
        repl[f] = base_state[f] + eps * df

    # Substitute and extract O(ε) coefficient.  ``xreplace`` propagates
    # through ``Derivative`` atoms — we then call ``.doit()`` so the
    # outer derivative distributes onto the new linear sum.  For
    # equations with rational nonlinearities (e.g. ``q²/h`` in
    # conservative-variable strong form), ``expand().coeff(eps, 1)``
    # alone fails because ``1/(h₀ + ε δh)`` is not a polynomial in ε —
    # so we fall back to ``sp.series`` to Taylor-expand to first order.
    lin_eqs = []
    for eq in system.equations:
        eq_sub = eq.xreplace(repl)
        eq_sub = eq_sub.doit()
        try:
            lin = sp.expand(eq_sub).coeff(eps, 1)
            # ``coeff`` silently drops ε-dependent rationals; cross-check
            # that no ε remains in the result and fall back if it does.
            if lin.has(eps):
                raise ValueError("ε-dependent residue after coeff")
        except (sp.PolynomialError, ValueError, AttributeError):
            lin = sp.series(eq_sub, eps, 0, 2).removeO().coeff(eps, 1)
        if simplify:
            lin = sp.expand(lin)
        lin_eqs.append(lin)

    return PDESystem(
        equations=lin_eqs,
        fields=delta_fields,
        time=system.time,
        space=list(system.space),
        parameters=dict(system.parameters),
        aux_fields=list(system.aux_fields),
    )
