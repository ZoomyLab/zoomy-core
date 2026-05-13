"""Predictor / pressure / corrector splitter for chain-DAE SystemModels.

Mechanises the substitution rule from
``tutorials/vam/escalante2024_poisson_generic.py`` (the verified
hand-coded reference).

The splitter consumes a :class:`SystemModel` whose ``equation_names``
follow the chain-DAE convention:

  * ``mass``                              — mass evolution row
  * ``xmom_j0``, ``xmom_j1``, …           — x-momentum projections
  * ``zmom_j0``, ``zmom_j1``, …           — z-momentum projections
  * ``cont_j1``, ``cont_j2``, …           — pressure-projection
                                            (algebraic) constraint rows

The splitter exposes two entry points:

  * :func:`build_pressure_elliptic_block` — the irreducible algebraic
    core (extracts ``T_u[k]``, ``T_w[k]``, builds the elliptic rows).
  * :func:`split_for_pressure` — wraps the core into three rectangular
    sub-SystemModels (predictor / pressure / corrector) sharing the
    same state vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp


# ---------------------------------------------------------------------------
# Internal: pull row residuals from a SystemModel via reconstruct_residuals.
# ---------------------------------------------------------------------------


def _residuals_by_name(sm):
    """Return ``{equation_name: residual_expr}`` for the SystemModel."""
    if not hasattr(sm, "equation_names"):
        raise ValueError(
            "SystemModel must carry equation_names for the splitter to "
            "dispatch rows by name."
        )
    residuals = sm.reconstruct_residuals()
    return dict(zip(sm.equation_names, residuals))


def _state_symbol_by_name(sm):
    """Map state-entry name (str) → Symbol."""
    return {str(s): s for s in sm.state}


# ---------------------------------------------------------------------------
# Build pressure elliptic block.
# ---------------------------------------------------------------------------


def build_pressure_elliptic_block(
    sm,
    pressure_vars: Sequence,
    dt: sp.Symbol,
    *,
    bottom=None,
):
    """Build the pressure-stage elliptic block for ``SM_press``.

    Parameters
    ----------
    sm : SystemModel
        Chain-DAE SystemModel (e.g. ``SystemModel.from_model(
        VAMModelGalerkin(level=1))``).  Must carry ``equation_names``
        with the canonical mass / xmom_j / zmom_j / cont_j layout.
    pressure_vars : Sequence
        Pressure mode state Symbols (e.g. ``[P_0, P_1]`` for
        VAM(1,2,2)).  Internally converted to Function form to match
        the residual representation produced by
        :meth:`SystemModel.reconstruct_residuals`.
    dt : sp.Symbol
        Symbolic time-step.
    bottom : optional
        Bottom-topography Function ``b(x)``.  If None, located by
        scanning equation residuals for a Function named ``b``.

    Returns
    -------
    dict with keys ``rows``, ``U_tilde``, ``W_tilde``, ``T_u``, ``T_w``,
    ``U_corr``, ``W_corr``, ``pressure_vars``, ``h``, ``bottom``, ``t``,
    ``x``.  All entries are in Function form (coordinate-dependent).
    """
    residuals = _residuals_by_name(sm)
    name_to_sym = _state_symbol_by_name(sm)

    if "h" not in name_to_sym:
        raise ValueError("SystemModel must include the ``h`` state entry.")
    t = sm.time
    coords = list(sm.space)
    x = coords[0]

    # Convert state Symbols to coordinate-dependent Functions to match
    # the reconstruct_residuals representation.
    def _to_fn(sym):
        return sp.Function(str(sym), real=True)(t, *coords)

    h_fn = _to_fn(name_to_sym["h"])

    # Discover U_0..U_M and W_0..W_{N_w-1} by name (Symbol → Function).
    coeffs_u = []
    k = 0
    while f"U_{k}" in name_to_sym:
        coeffs_u.append(_to_fn(name_to_sym[f"U_{k}"]))
        k += 1
    M_M = len(coeffs_u) - 1
    if M_M < 0:
        raise ValueError("SystemModel has no U_k state entries.")

    coeffs_w = []
    k = 0
    while f"W_{k}" in name_to_sym:
        coeffs_w.append(_to_fn(name_to_sym[f"W_{k}"]))
        k += 1
    N_w_active = len(coeffs_w)
    if N_w_active < 1:
        raise ValueError("SystemModel has no W_k state entries.")

    # Pressure Functions in residual form.
    pressure_funcs = [_to_fn(p) for p in pressure_vars]

    # The bottom topography is a coordinate-dependent Function ``b``;
    # locate it by scanning residual atoms (it isn't a state entry).
    if bottom is None:
        for eq in residuals.values():
            for atom in eq.atoms(sp.Function):
                if atom.func.__name__ == "b":
                    bottom = atom
                    break
            if bottom is not None:
                break
    if bottom is None:
        raise ValueError(
            "Could not locate bottom topography 'b' in residuals."
        )

    # Predictor-stage placeholders.
    U_tilde = [sp.Function(f"U_{k}_tilde", real=True)(t, x)
               for k in range(M_M + 1)]
    W_tilde = [sp.Function(f"W_{k}_tilde", real=True)(t, x)
               for k in range(N_w_active)]

    def mu(j):
        return sp.Rational(1, 2 * j + 1)

    # Per-conserved-variable pressure sources.
    T_u = [
        _extract_pressure_T(residuals, f"xmom_j{k}", pressure_funcs, mu(k))
        for k in range(M_M + 1)
    ]
    T_w = [
        _extract_pressure_T(residuals, f"zmom_j{k}", pressure_funcs, mu(k))
        for k in range(N_w_active)
    ]

    # Corrector update in primitive form:
    #   U_k^(corr) = U_k_tilde - (dt / h) * T_uk(P)
    #   W_k^(corr) = W_k_tilde - (dt / h) * T_wk(P)
    U_corr = [U_tilde[k] - (dt / h_fn) * T_u[k] for k in range(M_M + 1)]
    W_corr = [W_tilde[k] - (dt / h_fn) * T_w[k] for k in range(N_w_active)]

    repl = {coeffs_u[k]: U_corr[k] for k in range(M_M + 1)}
    repl.update({coeffs_w[k]: W_corr[k] for k in range(N_w_active)})

    elliptic_rows = {}
    for name, eq in residuals.items():
        if name.startswith("cont_j"):
            j = int(name[len("cont_j"):])
            substituted = sp.expand(eq.subs(repl))
            elliptic_rows[j] = substituted

    return {
        "rows": elliptic_rows,
        "U_tilde": U_tilde,
        "W_tilde": W_tilde,
        "T_u": T_u,
        "T_w": T_w,
        "U_corr": U_corr,
        "W_corr": W_corr,
        "pressure_vars": list(pressure_funcs),
        "h": h_fn,
        "bottom": bottom,
        "t": t,
        "x": x,
    }


def verify_p_linearity(rows, pressure_vars, x):
    """Verify each row is linear in ``(P_l, ∂_x P_l, ∂_xx P_l)``."""
    P_VARS = []
    for p in pressure_vars:
        name = str(p)
        P_VARS.append((name, p))
        P_VARS.append((f"d_x {name}", sp.Derivative(p, x)))
        P_VARS.append((f"d_xx {name}", sp.Derivative(p, x, x)))

    dummies = [sp.Dummy(n) for n, _ in P_VARS]
    repl = {atom: d for d, (_, atom) in zip(dummies, P_VARS)}

    coeffs_out = {}
    constants_out = {}
    for j, row in rows.items():
        e = sp.expand(row).doit()
        e = sp.expand(e)
        e_poly = e.xreplace(repl)
        leftover_p = [p for p in pressure_vars if e_poly.has(p)]
        leftover_d = [
            d for d in e_poly.atoms(sp.Derivative)
            if d.args[0] in pressure_vars
        ]
        if leftover_p or leftover_d:
            raise AssertionError(
                f"row {j}: not in span of (P, ∂_x P, ∂_xx P); "
                f"unexpected atoms: {leftover_p + leftover_d}"
            )
        try:
            poly = sp.Poly(e_poly, *dummies)
        except sp.PolynomialError as exc:
            raise AssertionError(
                f"row {j}: pressure dependence is not polynomial: {exc}"
            )
        if poly.total_degree() > 1:
            raise AssertionError(
                f"row {j}: pressure dependence is degree "
                f"{poly.total_degree()}, expected ≤ 1."
            )
        constants_out[j] = poly.nth(*([0] * len(dummies)))
        coeffs_out[j] = {}
        for i, (name, _atom) in enumerate(P_VARS):
            idx = [0] * len(dummies)
            idx[i] = 1
            coeffs_out[j][name] = sp.simplify(poly.nth(*idx))

    return {"coefficients": coeffs_out, "constants": constants_out}


# ---------------------------------------------------------------------------
# Internal: pressure-source extraction.
# ---------------------------------------------------------------------------


def _extract_pressure_T(residuals, equation_name, pressure_vars, mu_k):
    """Extract ``T_*[k]`` from a chain-DAE residual.

    ``residual = μ_k ∂_t(h·Q_k) + … + pressure_terms = 0``.  The
    pressure-linear part divided by ``μ_k`` is the source per
    conserved variable ``h·Q_k``.

    ``.doit()`` is called first so compound ``Derivative(A+P*B, x)``
    atoms distribute via the product / sum rule; otherwise the
    eq-vs-no_p subtraction leaves spurious cancelling pairs.
    """
    eq = sp.expand(residuals[equation_name].doit())
    no_p = sp.expand(eq.subs({p: sp.S.Zero for p in pressure_vars}).doit())
    pressure_part = sp.expand(eq - no_p)
    return sp.expand(pressure_part / mu_k)


# ---------------------------------------------------------------------------
# Three-sub-system splitter.
# ---------------------------------------------------------------------------


@dataclass
class SplitForPressureResult:
    """Three sub-SystemModels produced by :func:`split_for_pressure`.

    For VAM(1, 2, 2) on ``Q = [h, U_0, U_1, W_0, W_1, P_0, P_1]``:

    * ``SM_pred`` — predictor: 5 evolution rows
      (``mass`` + 2 ``xmom`` + 2 ``zmom``) updating
      ``Q[0..4] = h, U_k, W_k`` with ``P_k`` frozen at ``P_k^n``.
    * ``SM_press`` — pressure stage: 2 algebraic rows (the elliptic
      block in ``(P_0, P_1)``) updating ``Q[5..6]``.  Mass matrix
      all-zero.
    * ``SM_corr`` — corrector: 4 algebraic update rows.
    """
    SM_pred: object
    SM_press: object
    SM_corr: object


def _build_subsystem(*, eq_names, eq_residuals, sm_parent, state,
                     equation_to_state_index, history_entry):
    """Build a rectangular SystemModel from a list of (name, residual)
    pairs.  Each residual is auto-tagged then converted to an
    operator-form SystemModel by walking the tags.

    Returns a fresh :class:`SystemModel`.
    """
    from zoomy_core.model.models.system_model import SystemModel
    from zoomy_core.model.models.tag_extraction import (
        auto_solver_tag, collect_solver_tag,
    )

    t = sm_parent.time
    coords = list(sm_parent.space)
    x = coords[0]
    params = dict(sm_parent.parameters)
    g_param = next((s for s in params if str(s) == "g"), None)

    # State Symbols and coordinate-dependent Functions for tag classifier.
    state_syms = list(state)
    name_to_sym = {str(s): s for s in state_syms}
    state_funcs = [sp.Function(str(s), real=True)(t, *coords)
                   for s in state_syms]
    sym_to_func = dict(zip(state_syms, state_funcs))
    func_to_sym = dict(zip(state_funcs, state_syms))

    # Auto-tag every residual (converting Symbol state to Function form
    # for the tag classifier).
    from zoomy_core.model.models.ins_generator import Expression
    tagged: dict = {}
    for name, res in zip(eq_names, eq_residuals):
        res_func = sp.sympify(res).xreplace(sym_to_func)
        tagged[name] = auto_solver_tag(
            Expression(res_func, name=name),
            state_funcs=state_funcs,
            gravity_param=g_param,
            t=t, x=x,
        )

    class _Holder:
        pass

    holder = _Holder()
    holder.equations = tagged

    n_eq = len(eq_names)
    n_state = len(state_syms)
    n_dim = len(coords)
    variable_map = {n: [i] for i, n in enumerate(eq_names)}

    F = collect_solver_tag(
        holder, "flux", variable_map=variable_map,
        n_variables=n_eq, n_directions=n_dim, coords=coords,
        state_variables=state_funcs, policy="strict",
    )
    P = collect_solver_tag(
        holder, "hydrostatic_pressure", variable_map=variable_map,
        n_variables=n_eq, n_directions=n_dim, coords=coords,
        state_variables=state_funcs, policy="strict",
    )
    # ``collect_solver_tag`` allocates a square (N × N × n_dim) NCP
    # array.  Our sub-systems are rectangular (n_eq < n_state); pass
    # ``n_variables=max(n_eq, n_state)`` and then take the first n_eq
    # rows / n_state columns.
    n_max = max(n_eq, n_state)
    B_raw = collect_solver_tag(
        holder, "nonconservative_flux", variable_map=variable_map,
        n_variables=n_max, n_directions=n_dim, coords=coords,
        state_variables=state_funcs, policy="strict",
    )
    S_list = collect_solver_tag(
        holder, "source", variable_map=variable_map,
        n_variables=n_eq, policy="strict",
    )
    # SystemModel residual form is ``... − S(Q) = 0``; auto-tagger
    # stores source terms with their original LHS sign — negate.
    S_mat = sp.Matrix(n_eq, 1, lambda i, _j: -S_list[i])

    B = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim)
    for i in range(n_eq):
        for j in range(n_state):
            for d in range(n_dim):
                B[i, j, d] = B_raw[i, j, d]

    # Mass matrix: rebuild from the time-derivative tag on each row.
    # ``.doit()`` expands compound ``Derivative(c·F, t)`` atoms.
    M_mat = sp.zeros(n_eq, n_state)
    for i, name in enumerate(eq_names):
        eq = tagged[name]
        td_part = eq.get_solver_tag("time_derivative")
        if td_part is None or td_part == 0:
            continue
        td = sp.expand(td_part.doit())
        for j, fj in enumerate(state_funcs):
            dot_sym = sp.Symbol(f"_dot_{str(state_syms[j])}", real=True)
            dt_atom = sp.Derivative(fj, t)
            replaced = td.subs(dt_atom, dot_sym)
            coeff = sp.diff(replaced, dot_sym)
            if coeff != 0:
                M_mat[i, j] = sp.expand(coeff)

    # Map Function-form atoms back to Symbol state.
    def _to_sym(matrix):
        if isinstance(matrix, sp.Matrix):
            return matrix.xreplace(func_to_sym)
        out = sp.MutableDenseNDimArray.zeros(*matrix.shape)
        shape = matrix.shape
        ndim = len(shape)

        def _iter(shape):
            if not shape:
                yield ()
                return
            for i in range(shape[0]):
                for rest in _iter(shape[1:]):
                    yield (i,) + rest

        for idx in _iter(shape):
            entry = sp.sympify(matrix[idx])
            out[idx] = entry.xreplace(func_to_sym)
        return out

    F = _to_sym(F)
    P = _to_sym(P)
    B = _to_sym(B)
    S_mat = _to_sym(S_mat)
    M_mat = _to_sym(M_mat)

    sm = SystemModel(
        time=t,
        space=coords,
        state=state_syms,
        aux_state=[],
        parameters=params,
        flux=F,
        hydrostatic_pressure=P,
        nonconservative_matrix=B,
        source=S_mat,
        mass_matrix=M_mat,
        equation_to_state_index=list(equation_to_state_index),
    )
    sm.equation_names = list(eq_names)
    sm.history.append(history_entry)
    return sm


def split_for_pressure(sm, pressure_vars, dt, *, bottom=None):
    """Split a chain-DAE SystemModel into three :class:`SystemModel`
    sub-systems ``(SM_pred, SM_press, SM_corr)`` sharing the same
    state vector ``Q``.
    """
    block = build_pressure_elliptic_block(sm, pressure_vars, dt,
                                          bottom=bottom)
    residuals = _residuals_by_name(sm)
    name_to_sym = _state_symbol_by_name(sm)
    state = list(sm.state)
    state_names = [str(s) for s in state]

    pressure_indices = [state.index(p) for p in pressure_vars]
    t = sm.time
    coords = list(sm.space)
    x = coords[0]

    # Use the Function-form pressure variants from the block to match
    # the Function-form residuals.
    pressure_funcs = block["pressure_vars"]

    # Placeholder Functions for stage-aux (previous-stage outputs).
    Q_n = [sp.Function(f"Q_n_{n}", real=True)(t, x) for n in state_names]
    Q_star = [sp.Function(f"Q_star_{n}", real=True)(t, x) for n in state_names]
    P_new = [sp.Function(f"P_new_{p.func.__name__}", real=True)(t, x)
             for p in pressure_funcs]

    # ── SM_pred ──────────────────────────────────────────────────────────
    pred_p_freeze = {p: Q_n[pressure_indices[i]]
                     for i, p in enumerate(pressure_funcs)}
    pred_eq_names = []
    pred_eq_residuals = []
    pred_e2s_index = []
    for name in sm.equation_names:
        if name.startswith("cont_j"):
            continue
        if name == "mass":
            target = "h"
        elif name.startswith("xmom_j"):
            target = f"U_{name[len('xmom_j'):]}"
        elif name.startswith("zmom_j"):
            target = f"W_{name[len('zmom_j'):]}"
        else:
            continue
        if target not in state_names:
            continue
        eq = residuals[name]
        eq_pred = sp.expand(eq.subs(pred_p_freeze))
        pred_eq_names.append(name)
        pred_eq_residuals.append(eq_pred)
        pred_e2s_index.append(state_names.index(target))

    SM_pred = _build_subsystem(
        eq_names=pred_eq_names,
        eq_residuals=pred_eq_residuals,
        sm_parent=sm,
        state=state,
        equation_to_state_index=pred_e2s_index,
        history_entry={
            "name": "split_for_pressure[pred]",
            "description": f"predictor: {len(pred_eq_names)} rows updating "
                           f"{[state_names[i] for i in pred_e2s_index]}",
        },
    )

    # ── SM_press ─────────────────────────────────────────────────────────
    press_eq_names = [f"elliptic_j{j}" for j in sorted(block["rows"])]
    press_eq_residuals = [
        sp.expand(block["rows"][j]) for j in sorted(block["rows"])
    ]
    SM_press = _build_subsystem(
        eq_names=press_eq_names,
        eq_residuals=press_eq_residuals,
        sm_parent=sm,
        state=state,
        equation_to_state_index=list(pressure_indices),
        history_entry={
            "name": "split_for_pressure[press]",
            "description": f"pressure: {len(press_eq_names)} algebraic rows "
                           f"determining {[str(p) for p in pressure_vars]}",
        },
    )

    # ── SM_corr ──────────────────────────────────────────────────────────
    corr_eq_names = []
    corr_eq_residuals = []
    corr_e2s_index = []
    M_M = len(block["U_corr"]) - 1
    N_w_active = len(block["W_corr"])

    for k in range(M_M + 1):
        target = f"U_{k}"
        if target not in state_names:
            continue
        U_k_fn = sp.Function(target, real=True)(t, x)
        U_corr_expr = block["U_corr"][k]
        for i_p, p in enumerate(pressure_funcs):
            U_corr_expr = U_corr_expr.subs(p, P_new[i_p])
        residual = sp.expand(U_k_fn - U_corr_expr)
        corr_eq_names.append(f"corr_U_{k}")
        corr_eq_residuals.append(residual)
        corr_e2s_index.append(state_names.index(target))
    for k in range(N_w_active):
        target = f"W_{k}"
        if target not in state_names:
            continue
        W_k_fn = sp.Function(target, real=True)(t, x)
        W_corr_expr = block["W_corr"][k]
        for i_p, p in enumerate(pressure_funcs):
            W_corr_expr = W_corr_expr.subs(p, P_new[i_p])
        residual = sp.expand(W_k_fn - W_corr_expr)
        corr_eq_names.append(f"corr_W_{k}")
        corr_eq_residuals.append(residual)
        corr_e2s_index.append(state_names.index(target))

    SM_corr = _build_subsystem(
        eq_names=corr_eq_names,
        eq_residuals=corr_eq_residuals,
        sm_parent=sm,
        state=state,
        equation_to_state_index=corr_e2s_index,
        history_entry={
            "name": "split_for_pressure[corr]",
            "description": f"corrector: {len(corr_eq_names)} algebraic rows "
                           f"updating {[state_names[i] for i in corr_e2s_index]}",
        },
    )

    return SplitForPressureResult(
        SM_pred=SM_pred,
        SM_press=SM_press,
        SM_corr=SM_corr,
    )


__all__ = [
    "build_pressure_elliptic_block",
    "verify_p_linearity",
    "split_for_pressure",
    "SplitForPressureResult",
]
