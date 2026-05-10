"""Predictor / pressure / corrector splitter for chain-DAE PDESystems.

Mechanises the substitution rule from
``tutorials/vam/escalante2024_poisson_generic.py`` (the verified
hand-coded reference).  See ``thesis/chapters/derivation_vam.md``
§7.5–§7.6 for the contract, §5.5–§5.6 for the worked derivation.

This module exposes the **elliptic-block extractor**
``build_pressure_elliptic_block`` — the irreducible algebraic core of
the splitter.  The full ``split_for_pressure`` (returning three
``SystemModel`` objects per the §7 contract) wraps this primitive
with stage assembly; that wrapper is implemented separately.
"""
from __future__ import annotations

from typing import Sequence

import sympy as sp


def build_pressure_elliptic_block(
    pdesys,
    pressure_vars: Sequence,
    dt: sp.Symbol,
    *,
    bottom=None,
):
    """Build the pressure-stage elliptic block for ``SM_press``.

    Given a chain-DAE PDESystem in System B form (cont_j algebraic
    rows that don't reference ``pressure_vars`` directly, plus
    xmom_jk and zmom_jk evolution rows whose pressure-dependent
    parts define the corrector source), substitute the corrector
    update for ``(U_k, W_k)`` into the algebraic rows and return the
    resulting equations — which are linear in
    ``(P_l, ∂_x P_l, ∂_{xx} P_l)``.

    Parameters
    ----------
    pdesys : PDESystem
        Chain-DAE built by ``VAMModelGalerkin._chain_dae`` (or any
        PDESystem with the same row-naming convention: ``mass``,
        ``xmom_j0..M``, ``zmom_j0..N_w-1``, ``cont_j1..N_p``).
    pressure_vars : Sequence
        Pressure mode functions, e.g. ``[P_0, P_1]`` for VAM(1, 2, 2).
        Must be entries in ``pdesys.fields``.
    dt : sp.Symbol
        Symbolic time-step.
    bottom : Optional
        Bottom-topography function ``b(t, x)``.  If None, located by
        scanning equation atoms for the function named ``b``.

    Returns
    -------
    dict
        ``rows``: dict ``{j: expr}`` of the elliptic-block equations,
        one per ``cont_jk`` algebraic row.  Each expr is linear in
        ``(P_l, ∂_x P_l, ∂_{xx} P_l)`` — verifiable by
        ``verify_p_linearity``.

        ``U_tilde``, ``W_tilde``: lists of predictor-stage symbols.

        ``T_u``, ``T_w``: per-conserved-variable pressure sources.

        ``U_corr``, ``W_corr``: corrector update expressions.

        ``pressure_vars``: the list of pressure variables (echoed).
    """
    if not hasattr(pdesys, "equation_names"):
        raise ValueError(
            "PDESystem must have equation_names to identify mass / xmom / "
            "zmom / cont rows."
        )

    names = list(pdesys.equation_names)
    eqs = list(pdesys.equations)
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}

    if "h" not in fields_by_name:
        raise ValueError("PDESystem must include the ``h`` field.")
    h = fields_by_name["h"]
    t = pdesys.time
    x = pdesys.space[0]

    # Discover U_0..U_M and W_0..W_{N_w-1} by name convention.
    coeffs_u = []
    k = 0
    while f"U_{k}" in fields_by_name:
        coeffs_u.append(fields_by_name[f"U_{k}"])
        k += 1
    M = len(coeffs_u) - 1
    if M < 0:
        raise ValueError("PDESystem has no U_k fields.")

    coeffs_w = []
    k = 0
    while f"W_{k}" in fields_by_name:
        coeffs_w.append(fields_by_name[f"W_{k}"])
        k += 1
    N_w_active = len(coeffs_w)
    if N_w_active < 1:
        raise ValueError("PDESystem has no W_k fields.")

    # Locate the bottom topography in the equations.
    if bottom is None:
        for eq in eqs:
            for atom in eq.atoms(sp.Function):
                if atom.func.__name__ == "b":
                    bottom = atom
                    break
            if bottom is not None:
                break
    if bottom is None:
        raise ValueError(
            "Could not locate bottom topography 'b' in pdesys equations."
        )

    # Predictor-stage symbols for the velocities.  Distinct from the
    # state coefficients so substitution is unambiguous; named with a
    # ``_tilde`` suffix.
    U_tilde = [
        sp.Function(f"U_{k}_tilde", real=True)(t, x)
        for k in range(M + 1)
    ]
    W_tilde = [
        sp.Function(f"W_{k}_tilde", real=True)(t, x)
        for k in range(N_w_active)
    ]

    def mu(j):
        return sp.Rational(1, 2 * j + 1)

    # Per-conservative-variable pressure sources extracted directly
    # from the chain DAE's ``xmom_jk`` / ``zmom_jk`` rows.  Each row
    # has the residual form ``mu_k ∂_t(h Q_k) + ... + (pressure
    # terms) = 0``; the pressure-linear part divided by ``mu_k``
    # gives the source per conserved variable ``h Q_k`` matching the
    # ``Q_k^(corr) = Q_k_tilde - (dt/h) T_*[k]`` corrector convention.
    T_u = [
        _extract_pressure_T_from_chain(
            pdesys, f"xmom_j{k}", pressure_vars, mu(k))
        for k in range(M + 1)
    ]
    T_w = [
        _extract_pressure_T_from_chain(
            pdesys, f"zmom_j{k}", pressure_vars, mu(k))
        for k in range(N_w_active)
    ]

    # Corrector update in primitive form:
    #     U_k^(corr) = U_k_tilde - (dt / h) * T_uk(P)
    #     W_k^(corr) = W_k_tilde - (dt / h) * T_wk(P)
    # (Coming from the conservative form q_k^(corr) = q_k_tilde - dt T_uk
    # and dividing by h.)
    U_corr = [U_tilde[k] - (dt / h) * T_u[k] for k in range(M + 1)]
    W_corr = [W_tilde[k] - (dt / h) * T_w[k] for k in range(N_w_active)]

    # Substitute corrector for U_k, W_k in the cont-projection
    # algebraic rows.  The result is a function of (h, U_tilde,
    # W_tilde, b, P, ∂P, ∂²P) — which is the elliptic block.
    repl = {coeffs_u[k]: U_corr[k] for k in range(M + 1)}
    repl.update({coeffs_w[k]: W_corr[k] for k in range(N_w_active)})

    elliptic_rows = {}
    for name, eq in zip(names, eqs):
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
        "pressure_vars": list(pressure_vars),
        "h": h,
        "bottom": bottom,
        "t": t,
        "x": x,
    }


def verify_p_linearity(rows, pressure_vars, x):
    """Verify each row is linear in ``(P_l, ∂_x P_l, ∂_{xx} P_l)`` and
    return the coefficient matrix.

    Mirrors ``escalante2024_poisson{,_generic}.py``'s
    ``collect_pressure_coeffs`` strict-linearity assertion.

    Returns
    -------
    dict
        ``coefficients``: dict ``{row_idx: {atom_name: coefficient}}``.

        ``constants``: dict ``{row_idx: constant_part}`` (the part
        independent of any pressure atom — i.e. the negative RHS).
    """
    P_VARS = []
    for p in pressure_vars:
        name = p.func.__name__
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
# Chain-row extraction (post opaque-ζ migration).
# ---------------------------------------------------------------------------


def _extract_pressure_T_from_chain(pdesys, equation_name, pressure_vars, mu_k):
    """Extract the pressure source ``T_*[k]`` from a chain DAE row.

    The chain DAE row ``xmom_jk`` / ``zmom_jk`` is the residual

        ``mu_k ∂_t(h Q_k) + (flux + boundary terms) + (pressure terms) = 0``

    where ``mu_k = 1/(2k+1)`` is the shifted-Legendre normalisation.
    The pressure-linear part divided by ``mu_k`` is the source per
    conserved variable ``h Q_k`` — i.e. the ``T_*[k]`` used in the
    corrector update ``Q_k^(corr) = Q_k_tilde - (dt/h) T_*[k]``.
    """
    idx = pdesys.equation_names.index(equation_name)
    eq = pdesys.equations[idx]
    no_p = eq.subs({p: sp.S.Zero for p in pressure_vars})
    pressure_part = sp.expand(eq - no_p)
    return sp.expand(pressure_part / mu_k)


# ---------------------------------------------------------------------------
# Three-sub-system splitter (§7 contract).
# ---------------------------------------------------------------------------


from dataclasses import dataclass, field as _dc_field


@dataclass
class SplitForPressureResult:
    """Three sub-systems produced by :func:`split_for_pressure`.

    Per ``thesis/chapters/derivation_vam.md`` §7.3, each sub-system
    shares the same state vector ``Q`` and parameters as the input
    chain DAE; the sub-systems differ in which entries of ``Q`` they
    update and in their equation count.

    For VAM(1, 2, 2) on ``Q = [h, U_0, U_1, W_0, W_1, P_0, P_1]``:

    * ``SM_pred`` — predictor: 5 evolution equations
      (``mass`` + 2 ``xmom`` + 2 ``zmom``) updating
      ``Q[0..4] = h, U_k, W_k`` with ``P_k`` frozen at the previous
      time step ``P_k^n``.  Pressure entries pass through unchanged.
    * ``SM_press`` — pressure stage: 2 algebraic equations (the
      elliptic block in ``(P_0, P_1)``) updating ``Q[5..6]``.  All
      other entries pass through.  Mass matrix is all-zero.
    * ``SM_corr`` — corrector: 4 algebraic update equations
      ``Q_k = Q_k^* − (Δt/h) T_*[k](P^{n+1})`` updating
      ``Q[1..4] = U_k, W_k``.  ``h`` and ``P_k`` pass through.  Mass
      matrix is all-zero.

    Pipeline ``Q → Q_1 → Q_2 → Q_3 = Q^{n+1}``: each stage overwrites
    only its indexed entries; others are passed forward verbatim.
    """
    SM_pred: "SystemModel"
    SM_press: "SystemModel"
    SM_corr: "SystemModel"


def split_for_pressure(pdesys, pressure_vars, dt, *, bottom=None):
    """Split a chain-DAE PDESystem into three :class:`SystemModel`
    sub-systems ``(SM_pred, SM_press, SM_corr)`` sharing the same
    state vector ``Q``.

    Each sub-system is rectangular: ``n_eq < n_state`` and
    ``equation_to_state_index`` records which state entry each
    equation updates.  All other state entries are implicitly held
    constant by the stage (the runtime harness passes them through).

    See :class:`SplitForPressureResult` for the per-stage
    description of what each sub-system updates.
    """
    from zoomy_core.model.models.system_model import SystemModel

    block = build_pressure_elliptic_block(pdesys, pressure_vars, dt,
                                          bottom=bottom)

    state_funcs = list(pdesys.fields)
    state_names = [f.func.__name__ for f in state_funcs]
    state_syms = [sp.Symbol(n, real=True) for n in state_names]
    func_to_sym = dict(zip(state_funcs, state_syms))

    pressure_name_set = {p.func.__name__ for p in pressure_vars}
    pressure_indices = [
        i for i, n in enumerate(state_names) if n in pressure_name_set
    ]
    non_pressure_indices = [
        i for i in range(len(state_funcs)) if i not in pressure_indices
    ]

    t = pdesys.time
    x = pdesys.space[0]
    space = list(pdesys.space)
    n_state = len(state_funcs)
    n_dim = len(space)
    params = (dict(pdesys.parameters)
              if hasattr(pdesys, "parameters") else {})

    # Stage-aux placeholders (Q_n, Q_star, P_new) — the runtime carries
    # the previous-stage output into the next stage via these.
    Q_n = [sp.Function(f"Q_n_{n}", real=True)(t, x) for n in state_names]
    Q_star = [sp.Function(f"Q_star_{n}", real=True)(t, x)
              for n in state_names]
    P_new = [sp.Function(f"P_new_{p.func.__name__}", real=True)(t, x)
             for p in pressure_vars]

    # ── SM_pred: 5 evolution rows on (h, U_0..M, W_0..N_w-1) ─────────────
    pred_p_freeze = {pressure_vars[i]: Q_n[pressure_indices[i]]
                     for i in range(len(pressure_vars))}
    pred_eq_names = []
    pred_eq_residuals = []
    pred_e2s_index = []
    for name, eq in zip(pdesys.equation_names, pdesys.equations):
        if name.startswith("cont_j"):
            continue  # algebraic constraint rows belong to SM_press
        # Determine which state entry this row updates.  By chain DAE
        # naming convention: mass→h, xmom_jk→U_k, zmom_jk→W_k.
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
        # Substitute P → Q_n constants and convert state Functions to
        # Symbols so the operator surface lines up with state_syms.
        eq_raw = eq.expr if hasattr(eq, "expr") else eq
        eq_pred = sp.expand(eq_raw.subs(pred_p_freeze))
        pred_eq_names.append(name)
        pred_eq_residuals.append(eq_pred.xreplace(func_to_sym))
        pred_e2s_index.append(state_names.index(target))

    # Build mass matrix from the chain DAE's M_t (rows aligned with
    # pred_eq_names).  Linearise the original chain rows around the
    # symbolic state to get the mass-matrix coefficients per row.
    from zoomy_core.analysis.linearisation import linearise
    from zoomy_core.analysis.pencil import extract_quasilinear_pencil
    base_state = {f: f for f in state_funcs}
    sys_lin = linearise(pdesys, base_state)
    M_t_full, _, _ = extract_quasilinear_pencil(sys_lin)
    M_t_full = M_t_full.xreplace(func_to_sym)

    pred_n_eq = len(pred_eq_names)
    M_pred = sp.Matrix(pred_n_eq, n_state, lambda i, j:
                       M_t_full[pdesys.equation_names.index(pred_eq_names[i]), j])
    SM_pred = SystemModel(
        time=t, space=space,
        state=state_syms, aux_state=list(Q_n),
        parameters=params,
        flux=sp.zeros(pred_n_eq, n_dim),
        hydrostatic_pressure=sp.zeros(pred_n_eq, n_dim),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(
            pred_n_eq, n_state, n_dim),
        source=sp.Matrix(pred_n_eq, 1,
                         lambda i, _j: -pred_eq_residuals[i]),
        mass_matrix=M_pred,
        equation_to_state_index=pred_e2s_index,
    )
    SM_pred.history.append({
        "name": "split_for_pressure[pred]",
        "description": (f"predictor stage, {pred_n_eq} evolution rows "
                        f"updating {[state_names[i] for i in pred_e2s_index]}"),
    })

    # ── SM_press: elliptic block on pressure entries ─────────────────────
    press_eq_names = [f"elliptic_j{j}" for j in sorted(block["rows"])]
    press_eq_residuals = [
        sp.expand(block["rows"][j]).xreplace(func_to_sym)
        for j in sorted(block["rows"])
    ]
    press_n_eq = len(press_eq_names)
    SM_press = SystemModel(
        time=t, space=space,
        state=state_syms, aux_state=list(Q_star),
        parameters=params,
        flux=sp.zeros(press_n_eq, n_dim),
        hydrostatic_pressure=sp.zeros(press_n_eq, n_dim),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(
            press_n_eq, n_state, n_dim),
        source=sp.Matrix(press_n_eq, 1,
                         lambda i, _j: -press_eq_residuals[i]),
        mass_matrix=sp.zeros(press_n_eq, n_state),
        equation_to_state_index=list(pressure_indices),
    )
    SM_press.history.append({
        "name": "split_for_pressure[press]",
        "description": (f"pressure stage, {press_n_eq} algebraic rows "
                        f"determining {[p.func.__name__ for p in pressure_vars]}"),
    })

    # ── SM_corr: corrector updates on (U_k, W_k) ─────────────────────────
    corr_eq_names = []
    corr_eq_residuals = []
    corr_e2s_index = []
    M_M = len(block["U_corr"]) - 1
    N_w_active = len(block["W_corr"])
    for k in range(M_M + 1):
        target = f"U_{k}"
        if target not in state_names:
            continue
        U_k_func = state_funcs[state_names.index(target)]
        U_corr_expr = block["U_corr"][k]
        for i_p, p in enumerate(pressure_vars):
            U_corr_expr = U_corr_expr.subs(p, P_new[i_p])
        residual = sp.expand(U_k_func - U_corr_expr).xreplace(func_to_sym)
        corr_eq_names.append(f"corr_U_{k}")
        corr_eq_residuals.append(residual)
        corr_e2s_index.append(state_names.index(target))
    for k in range(N_w_active):
        target = f"W_{k}"
        if target not in state_names:
            continue
        W_k_func = state_funcs[state_names.index(target)]
        W_corr_expr = block["W_corr"][k]
        for i_p, p in enumerate(pressure_vars):
            W_corr_expr = W_corr_expr.subs(p, P_new[i_p])
        residual = sp.expand(W_k_func - W_corr_expr).xreplace(func_to_sym)
        corr_eq_names.append(f"corr_W_{k}")
        corr_eq_residuals.append(residual)
        corr_e2s_index.append(state_names.index(target))
    corr_n_eq = len(corr_eq_names)
    SM_corr = SystemModel(
        time=t, space=space,
        state=state_syms, aux_state=list(Q_star) + list(P_new),
        parameters=params,
        flux=sp.zeros(corr_n_eq, n_dim),
        hydrostatic_pressure=sp.zeros(corr_n_eq, n_dim),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(
            corr_n_eq, n_state, n_dim),
        source=sp.Matrix(corr_n_eq, 1,
                         lambda i, _j: -corr_eq_residuals[i]),
        mass_matrix=sp.zeros(corr_n_eq, n_state),
        equation_to_state_index=corr_e2s_index,
    )
    SM_corr.history.append({
        "name": "split_for_pressure[corr]",
        "description": (f"corrector stage, {corr_n_eq} algebraic rows "
                        f"updating {[state_names[i] for i in corr_e2s_index]}"),
    })

    # Attach descriptive equation_names so describe() can label the rows.
    SM_pred.equation_names = pred_eq_names
    SM_press.equation_names = press_eq_names
    SM_corr.equation_names = corr_eq_names

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
