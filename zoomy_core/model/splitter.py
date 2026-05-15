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

from zoomy_core.model.models.system_model import SystemModel


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

    # Discover momentum-mode state entries by name.  Supports both
    # primitive form (``U_k``, ``W_k``) and conservative form
    # (``q_Uk``, ``q_Wk``) — the splitter is agnostic, picks whichever
    # naming the SystemModel uses.  Pressure modes are excluded
    # explicitly (``pressure_vars`` is passed by the caller).
    def _collect(prefixes):
        out = []
        k = 0
        while True:
            for pref in prefixes:
                key = f"{pref}{k}"
                if key in name_to_sym:
                    out.append(_to_fn(name_to_sym[key]))
                    break
            else:
                break
            k += 1
        return out

    coeffs_u = _collect(("U_", "q_U"))
    coeffs_w = _collect(("W_", "q_W"))
    M_M = len(coeffs_u) - 1
    N_w_active = len(coeffs_w)
    if M_M < 0:
        raise ValueError(
            "SystemModel has no U_k / q_Uk state entries."
        )
    if N_w_active < 1:
        raise ValueError(
            "SystemModel has no W_k / q_Wk state entries."
        )

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

    def mu(j):
        return sp.Rational(1, 2 * j + 1)

    # Per-momentum-row pressure sources T_k(P).  ``_extract_pressure_T``
    # walks the row residual for pressure-dependent atoms.  Unified
    # treatment — U and W rows go through the same machinery.
    T_u = [
        _extract_pressure_T(residuals, f"xmom_j{k}", pressure_funcs, mu(k))
        for k in range(M_M + 1)
    ]
    T_w = [
        _extract_pressure_T(residuals, f"zmom_j{k}", pressure_funcs, mu(k))
        for k in range(N_w_active)
    ]

    # Corrector update — written in state Functions, no tilde rename.
    # The runtime semantics (``U_k`` here refers to the predictor's
    # tilde value at substep time, since the pressure stage runs after
    # the predictor has written to ``Q[U_k]``) lives in execution order,
    # not symbol identity.
    U_corr = [coeffs_u[k] - (dt / h_fn) * T_u[k] for k in range(M_M + 1)]
    W_corr = [coeffs_w[k] - (dt / h_fn) * T_w[k] for k in range(N_w_active)]

    # Elliptic block: substitute U_k → U_k - (dt/h)·T_u_k(P) (and
    # similarly W_k) into every algebraic continuity row.  Sympy's
    # ``.subs`` is single-pass: the substituted-in expression's U_k
    # remains the state Function, not a tilde alias.
    repl = {coeffs_u[k]: U_corr[k] for k in range(M_M + 1)}
    repl.update({coeffs_w[k]: W_corr[k] for k in range(N_w_active)})

    elliptic_rows = {}
    for name, eq in residuals.items():
        if name.startswith("cont_j"):
            j = int(name[len("cont_j"):])
            # ``.doit()`` distributes the compound ``Derivative`` atoms
            # introduced by the substitution down to atomic derivatives.
            substituted = sp.expand(eq.subs(repl).doit())
            elliptic_rows[j] = substituted

    return {
        "rows": elliptic_rows,
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
    g_param = (sm_parent.parameters.g
               if sm_parent.parameters.contains("g") else None)

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
        parameters=sm_parent.parameters,
        parameter_values=sm_parent.parameter_values,
        flux=F,
        hydrostatic_pressure=P,
        nonconservative_matrix=B,
        source=S_mat,
        mass_matrix=M_mat,
        equation_to_state_index=list(equation_to_state_index),
        # Indexed BC kernels carry across unchanged — the sub-system
        # shares the parent's state Symbols, so the parent's
        # ``boundary_conditions(idx, ..., Q, Qaux, p, normal) → Q_face``
        # Function still resolves against the same Q layout.  Without
        # this propagation the runtime build (which lambdifies these
        # Functions unconditionally — "prefer breaking over silent
        # skip") would fail.
        boundary_conditions=sm_parent.boundary_conditions,
        aux_boundary_conditions=sm_parent.aux_boundary_conditions,
        boundary_gradients=sm_parent.boundary_gradients,
        initial_conditions=sm_parent.initial_conditions,
        aux_initial_conditions=sm_parent.aux_initial_conditions,
        update_variables=sm_parent.update_variables,
    )
    # Auto-scan: route every non-state Function / Derivative atom in
    # this sub-system's operators into ``aux_state`` + ``aux_registry``
    # — the same machinery ``SystemModel.from_model`` runs on the
    # monolithic chain.  So the predictor carries ``b`` / ``b_x`` /
    # ``h_x``, the pressure stage carries the pressure derivatives
    # ``P_l_x`` / ``P_l_x_x``, etc.
    sm.expose_aux_atoms()
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

    # ── SM_pred ──────────────────────────────────────────────────────────
    # No pred_p_freeze rename — the predictor reads pressure straight
    # from the state Functions, and at runtime ``Q[pressure_indices]``
    # holds the start-of-step pressure (the predictor's substep does
    # not write there, so it stays at P^n implicitly).
    pred_eq_names = []
    pred_eq_residuals = []
    pred_e2s_index = []
    def _state_for_equation(eq_name):
        """Resolve the state slot updated by an equation row.  Tries
        primitive-form names first (``U_k`` / ``W_k`` / ``h``), then
        conservative-form (``q_Uk`` / ``q_Wk`` / ``h``).  Returns the
        first match in ``state_names`` or ``None``."""
        if eq_name == "mass":
            return "h"
        if eq_name.startswith("xmom_j"):
            k = eq_name[len("xmom_j"):]
            for cand in (f"U_{k}", f"q_U{k}"):
                if cand in state_names:
                    return cand
        if eq_name.startswith("zmom_j"):
            k = eq_name[len("zmom_j"):]
            for cand in (f"W_{k}", f"q_W{k}"):
                if cand in state_names:
                    return cand
        return None

    for name in sm.equation_names:
        if name.startswith("cont_j"):
            continue
        target = _state_for_equation(name)
        if target is None:
            continue
        eq_pred = sp.expand(residuals[name])
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
    # Explicit-update operator: ``Q[corr_idx] ← state_update(Q, Qaux, p, dt)``.
    # The expression for each updated slot is the closed-form
    # ``coeffs_u[k] - (dt/h)·T_u_k(P)`` (and W analogue) in pure state
    # Functions — no P_new rename, no tilde rename.  The "tilde" /
    # "new-pressure" semantics live in execution order: at SM_corr
    # apply time ``Q[1:5]`` holds the predictor output and ``Q[5:6]``
    # holds the new pressure.
    M_M = len(block["U_corr"]) - 1
    N_w_active = len(block["W_corr"])
    corr_e2s_index = []
    update_exprs = []
    update_names = []  # for SM_corr.equation_names — diagnostic only
    def _state_index_for(prefixes_and_k):
        for pref, k in prefixes_and_k:
            cand = f"{pref}{k}"
            if cand in state_names:
                return state_names.index(cand), cand
        return None, None
    for k in range(M_M + 1):
        idx, _ = _state_index_for([("U_", k), ("q_U", k)])
        if idx is None:
            continue
        update_exprs.append(sp.expand(block["U_corr"][k]))
        corr_e2s_index.append(idx)
        update_names.append(f"corr_U_{k}")
    for k in range(N_w_active):
        idx, _ = _state_index_for([("W_", k), ("q_W", k)])
        if idx is None:
            continue
        update_exprs.append(sp.expand(block["W_corr"][k]))
        corr_e2s_index.append(idx)
        update_names.append(f"corr_W_{k}")

    # Map state Functions back to state Symbols — the SystemModel
    # convention: operators are in Symbol form.
    state_funcs = [sp.Function(str(s), real=True)(t, *coords) for s in state]
    func_to_sym = dict(zip(state_funcs, state))
    update_exprs_sym = [e.xreplace(func_to_sym) for e in update_exprs]

    # Build SM_corr with zero residual fields + ``state_update`` set.
    # ``_build_subsystem`` autotags an empty residual list to all-zero
    # operators; the state_update field then carries the explicit
    # update.
    n_eq = len(corr_e2s_index)
    n_dim = sm.n_dim
    n_st = sm.n_state
    SM_corr = SystemModel(
        time=sm.time,
        space=list(sm.space),
        state=list(state),
        aux_state=[],
        parameters=sm.parameters,
        parameter_values=sm.parameter_values,
        flux=sp.zeros(n_eq, n_dim),
        hydrostatic_pressure=sp.zeros(n_eq, n_dim),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(n_eq, n_st, n_dim),
        source=sp.zeros(n_eq, 1),
        mass_matrix=sp.zeros(n_eq, n_st),
        equation_to_state_index=list(corr_e2s_index),
        state_update=sp.Array(update_exprs_sym),
        boundary_conditions=sm.boundary_conditions,
        aux_boundary_conditions=sm.aux_boundary_conditions,
        boundary_gradients=sm.boundary_gradients,
        initial_conditions=sm.initial_conditions,
        aux_initial_conditions=sm.aux_initial_conditions,
        update_variables=sm.update_variables,
    )
    SM_corr.equation_names = list(update_names)
    SM_corr.expose_aux_atoms()
    SM_corr.history.append({
        "name": "split_for_pressure[corr]",
        "description": (
            f"corrector: explicit update on "
            f"{[state_names[i] for i in corr_e2s_index]} "
            f"via state_update field"
        ),
    })

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
