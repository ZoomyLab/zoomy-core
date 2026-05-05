"""SystemModel-based analysis: linearise + dispersion + eigenvalues.

Operates directly on :class:`zoomy_core.model.models.system_model.SystemModel`
— no ``PDESystem`` wrapper, no ``linearise``-on-equation-tree machinery.
The model's stored sympy matrices already encode the operator surface;
analysis is straightforward symbolic substitution + eigenvalue solve.

Three entry points:

* :func:`linearise_system_model` — substitute a base state into the
  flux / NCP / source / hydrostatic-pressure / mass matrices and
  return a new ``SystemModel`` whose matrices are constant in the
  state Symbols (only depend on coordinates and parameters).

* :func:`plane_wave_dispersion` — given a constant-coefficient
  linearised SystemModel, build the principal-symbol pencil
  ``(M_t, M_x[axis], M_0)`` and return the symbolic eigenvalues
  ``ω(k)`` (or equivalently ``λ = ω/k`` solving
  ``det(M_x − λ M_t) = 0``).

* :func:`hyperbolic_eigenvalues` — same pencil but returning the
  generalized eigenvalues of ``(M_x[axis], M_t)`` directly via
  :func:`zoomy_core.analysis.pencil.generalised_eigenvalues`.

For shapes where the analyst wants ``Sum``/``Indexed`` form (linear
stability with symbolic ``k``, parameter-sweep hyperbolicity, etc.)
the substitution dict accepts any sympy values — ``base_state[h] = h0``
keeps ``h0`` symbolic.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import sympy as sp

from zoomy_core.model.models.system_model import SystemModel


def linearise_system_model(
    sm: SystemModel,
    base_state: Dict[Any, Any],
    *,
    parameters: Optional[Dict[Any, Any]] = None,
) -> SystemModel:
    """Substitute ``base_state`` into every operator matrix; return a
    new ``SystemModel`` whose matrices are constant in the state.

    Parameters
    ----------
    sm : SystemModel
        The source operator-form system.
    base_state : dict
        Map of state Symbol → base value.  Missing entries are left
        symbolic.  Aux-state Symbols may also be substituted.
    parameters : dict, optional
        Additional parameter substitutions (e.g. ``{ez: 1, g: 9.81}``).
        Combined with ``base_state`` for the substitution.
    """
    sub = dict(base_state)
    if parameters:
        sub.update(parameters)

    def _sub(M):
        if isinstance(M, sp.Matrix):
            return sp.Matrix(
                M.shape[0], M.shape[1],
                lambda i, j: sp.simplify(M[i, j].xreplace(sub)),
            )
        # NDimArray (NCP)
        out = sp.MutableDenseNDimArray.zeros(*M.shape)
        for idx in _iter_indices(M.shape):
            out[idx] = sp.simplify(M[idx].xreplace(sub))
        return out

    return SystemModel(
        time=sm.time,
        space=list(sm.space),
        state=list(sm.state),
        aux_state=list(sm.aux_state),
        parameters=dict(sm.parameters),
        flux=_sub(sm.flux),
        hydrostatic_pressure=_sub(sm.hydrostatic_pressure),
        nonconservative_matrix=_sub(sm.nonconservative_matrix),
        source=_sub(sm.source),
        mass_matrix=_sub(sm.mass_matrix),
        boundary_conditions=sm.boundary_conditions,
        history=list(sm.history) + [
            {"name": "linearise", "description": f"base state {dict(base_state)}"}
        ],
    )


def _iter_indices(shape):
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for rest in _iter_indices(shape[1:]):
            yield (i,) + rest


def plane_wave_dispersion(
    sm: SystemModel,
    base_state: Dict[Any, Any],
    *,
    axis: int = 0,
    parameters: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """Compute the principal-symbol dispersion of ``sm`` linearised at
    ``base_state``.

    Linearisation is done on the **quasilinear matrix** (the
    state-symbolic Jacobian ``∂F/∂Q + ∂P/∂Q + B``) rather than on
    the flux matrix.  The Jacobian must be taken *before* substituting
    the base state — substituting first would collapse the flux to a
    constant, whose Jacobian is zero.

    Returns a dict with:

    * ``M_t`` — mass matrix at base state (``I`` for canonical models).
    * ``M_x`` — quasilinear matrix at base state in direction ``axis``.
    * ``M_0`` — ``-∂S/∂Q`` at base state.
    * ``eigenvalues`` — symbolic generalised eigenvalues of
      ``(M_x, M_t)`` (wave speeds; real for hyperbolic systems).
    """
    sub = dict(base_state)
    if parameters:
        sub.update(parameters)

    # Quasilinear / source-Jacobian on the FULL state-symbolic matrices,
    # then substitute the base state into the result.
    qm_sym = sm.quasilinear_matrix()
    M_x = sp.Matrix(
        sm.n_equations,
        sm.n_equations,
        lambda i, j: sp.simplify(qm_sym[i, j, axis].xreplace(sub)),
    )

    src_jac_sym = sm.source_jacobian_wrt_state()
    M_0 = sp.Matrix(
        sm.n_equations,
        sm.n_equations,
        lambda i, j: -sp.simplify(src_jac_sym[i, j].xreplace(sub)),
    )

    M_t = sp.Matrix(
        sm.n_equations,
        sm.n_equations,
        lambda i, j: sp.simplify(sm.mass_matrix[i, j].xreplace(sub)),
    )

    # Generalised eigenvalues of (M_x, M_t) — solves det(M_x − λ M_t) = 0.
    lam = sp.Symbol("lambda")
    char_poly = (M_x - lam * M_t).det(method="berkowitz")
    eigenvalues = sp.solve(sp.Eq(sp.simplify(char_poly), 0), lam)

    return {
        "M_t": M_t,
        "M_x": M_x,
        "M_0": M_0,
        "eigenvalues": eigenvalues,
    }


def hyperbolic_eigenvalues(
    sm: SystemModel,
    base_state: Dict[Any, Any],
    *,
    axis: int = 0,
    parameters: Optional[Dict[Any, Any]] = None,
) -> List[sp.Expr]:
    """Convenience: return only the eigenvalues from
    :func:`plane_wave_dispersion`.

    Useful for hyperbolicity sampling: real eigenvalues = hyperbolic;
    complex conjugate pairs = elliptic mode = ill-posed initial-value
    problem.
    """
    return plane_wave_dispersion(
        sm, base_state, axis=axis, parameters=parameters
    )["eigenvalues"]


__all__ = [
    "linearise_system_model",
    "plane_wave_dispersion",
    "hyperbolic_eigenvalues",
]
