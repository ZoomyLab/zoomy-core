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

    Per derivation_vam.md §7.3, each sub-system is a DAE on the SAME
    state, aux_state, and parameters as the input ``SystemModel``;
    only the equations and mass-matrix partition differ between
    stages.

    Attributes
    ----------
    pdesys : PDESystem
        The original chain-DAE PDESystem (echoed for convenience).
    pressure_vars : list
        The pressure-mode functions (echoed).
    SM_pred : dict
        Predictor sub-system spec.  Evolution rows for non-pressure
        state with frozen ``P_k → P_k^n``; algebraic rows
        ``P_k − P_k^n = 0``.  Stage context: ``Q^n``.
    SM_press : dict
        Pressure sub-system spec.  Evolution rows freeze
        ``Q ∖ P`` at ``Q^*``; algebraic rows = the
        ``N_p × N_p`` elliptic block in
        ``(P_0, …, P_{N_p−1})`` (the corrector‑substituted
        cont‑projections).  Stage context: ``Q^*``.
    SM_corr : dict
        Corrector sub-system spec.  Evolution rows update ``(U_k, W_k)``
        via the corrector source ``T``; ``h``, ``P_k`` frozen.
        Stage context: ``Q^*``, ``P^{n+1}``.
    elliptic_block : dict
        The output of :func:`build_pressure_elliptic_block` — the
        substantive symbolic content shared across the three
        sub-systems (``T_u``, ``T_w``, ``U_corr``, ``W_corr``,
        elliptic ``rows``).
    """
    pdesys: object
    pressure_vars: list
    SM_pred: dict
    SM_press: dict
    SM_corr: dict
    elliptic_block: dict


def split_for_pressure(pdesys, pressure_vars, dt, *, bottom=None):
    """Split a chain-DAE PDESystem into ``(SM_pred, SM_press, SM_corr)``.

    Implements the §7 contract from ``thesis/chapters/derivation_vam.md``:
    three DAE sub-systems sharing state, aux_state, and parameters,
    differing only in their equations and mass-matrix partitions.

    The substantive content is the elliptic block in ``SM_press``,
    obtained from :func:`build_pressure_elliptic_block`.  Predictor
    and corrector sub-systems are constructed around the same shared
    state by tagging which rows are evolution / algebraic / frozen
    at each stage.

    Each ``SM_*`` is returned as a dict (not a full ``SystemModel``
    object) carrying:

    * ``state``: list of state functions (same across stages).
    * ``aux_state_extension``: stage-specific aux entries
      (``Q^n``, ``Q^*``, ``P^{n+1}`` placeholders).
    * ``equation_names``: ordered names of rows.
    * ``equations``: list of symbolic expressions per row.
    * ``mass_matrix_partition``: dict with ``evolution_indices`` and
      ``algebraic_indices``.

    The runtime harness consumes these in sequence
    (``Q^n → SM_pred → Q^* → SM_press → Q+P^{n+1} → SM_corr → Q^{n+1}``)
    against one shared ``Q`` vector per derivation_vam.md §7.2.

    Parameters
    ----------
    pdesys : PDESystem
        Chain-DAE built by ``VAMModelGalerkin._chain_dae``.
    pressure_vars : Sequence
        Pressure mode functions, e.g. ``[P_0, P_1]`` for VAM(1, 2, 2).
    dt : sp.Symbol
        Symbolic time-step.
    bottom : Optional
        Bottom topography (auto-detected if None).

    Returns
    -------
    SplitForPressureResult
    """
    block = build_pressure_elliptic_block(pdesys, pressure_vars, dt,
                                          bottom=bottom)

    state = list(pdesys.fields)
    state_names = [f.func.__name__ for f in state]
    pressure_name_set = {p.func.__name__ for p in pressure_vars}
    pressure_indices = [
        i for i, n in enumerate(state_names) if n in pressure_name_set
    ]
    non_pressure_indices = [
        i for i in range(len(state)) if i not in pressure_indices
    ]
    pressure_names_ordered = [p.func.__name__ for p in pressure_vars]

    # Symbolic placeholders for stage context.  These would be real
    # functions in the runtime harness; here we just declare them as
    # named ``Q_n_<name>``, ``Q_star_<name>``, ``P_new_<name>``.
    t = pdesys.time
    x = pdesys.space[0]
    Q_n = [sp.Function(f"Q_n_{n}", real=True)(t, x) for n in state_names]
    Q_star = [sp.Function(f"Q_star_{n}", real=True)(t, x)
              for n in state_names]
    P_new = [sp.Function(f"P_new_{n}", real=True)(t, x)
             for n in pressure_names_ordered]

    # --- SM_pred: predictor with frozen P_k.
    # Evolution rows: original mass + xmom_jk + zmom_jk with P_k → Q_n[P_k].
    # Algebraic rows: P_k = P_k^n (i.e. Q_pressure_idx − Q_n_pressure_idx = 0).
    pred_p_freeze = {pressure_vars[i]: Q_n[pressure_indices[i]]
                     for i in range(len(pressure_vars))}
    pred_evolution = []
    pred_evolution_names = []
    for name, eq in zip(pdesys.equation_names, pdesys.equations):
        if name.startswith("cont_j") and name != "cont_j0":
            continue  # algebraic constraint rows excluded from evolution
        eq_pred = sp.expand(eq.subs(pred_p_freeze))
        pred_evolution.append(eq_pred)
        pred_evolution_names.append(name)
    pred_algebraic = []
    pred_algebraic_names = []
    for i, p in enumerate(pressure_vars):
        pred_algebraic.append(p - Q_n[pressure_indices[i]])
        pred_algebraic_names.append(f"freeze_{p.func.__name__}")
    SM_pred = {
        "state": state,
        "aux_state_extension": {"Q_n": Q_n},
        "equation_names": pred_evolution_names + pred_algebraic_names,
        "equations": pred_evolution + pred_algebraic,
        "mass_matrix_partition": {
            "evolution_indices": list(range(len(pred_evolution))),
            "algebraic_indices": list(
                range(len(pred_evolution),
                      len(pred_evolution) + len(pred_algebraic))),
        },
    }

    # --- SM_press: pressure stage.
    # Evolution rows: Q ∖ P frozen at Q^* (i.e. q_i − q_i^* = 0 for
    # non-pressure state); but we represent these as algebraic
    # identities (M_t = 0).  The actual pressure determination comes
    # from the elliptic block.
    press_freeze = []
    press_freeze_names = []
    for i in non_pressure_indices:
        press_freeze.append(state[i] - Q_star[i])
        press_freeze_names.append(f"freeze_{state_names[i]}")
    # Algebraic block: corrector-substituted cont_j rows.
    press_elliptic = []
    press_elliptic_names = []
    for j in sorted(block["rows"]):
        press_elliptic.append(block["rows"][j])
        press_elliptic_names.append(f"elliptic_j{j}")
    SM_press = {
        "state": state,
        "aux_state_extension": {"Q_star": Q_star},
        "equation_names": press_freeze_names + press_elliptic_names,
        "equations": press_freeze + press_elliptic,
        "mass_matrix_partition": {
            "evolution_indices": [],
            "algebraic_indices": list(
                range(len(press_freeze) + len(press_elliptic))),
        },
        "U_corr": block["U_corr"],
        "W_corr": block["W_corr"],
        "T_u": block["T_u"],
        "T_w": block["T_w"],
    }

    # --- SM_corr: corrector update.
    # Evolution rows for (U_k, W_k): apply U_k = U_k^* − (Δt/h) T_u_k(P^{n+1}).
    # h is unchanged (T_h = 0 in mass row).
    # P_k held at P_k^{n+1}.
    corr_evolution = []
    corr_evolution_names = []
    fields_by_name = {n: state[i] for i, n in enumerate(state_names)}
    M = len(block["U_corr"]) - 1
    N_w_active = len(block["W_corr"])
    for k in range(M + 1):
        U_k = fields_by_name.get(f"U_{k}")
        if U_k is None:
            continue
        # Substitute P_k → P_new in U_corr expression.
        U_corr_expr = block["U_corr"][k]
        for i_p, p in enumerate(pressure_vars):
            U_corr_expr = U_corr_expr.subs(
                p, P_new[i_p] if i_p < len(P_new) else p)
        corr_evolution.append(U_k - U_corr_expr)
        corr_evolution_names.append(f"corr_U_{k}")
    for k in range(N_w_active):
        W_k = fields_by_name.get(f"W_{k}")
        if W_k is None:
            continue
        W_corr_expr = block["W_corr"][k]
        for i_p, p in enumerate(pressure_vars):
            W_corr_expr = W_corr_expr.subs(
                p, P_new[i_p] if i_p < len(P_new) else p)
        corr_evolution.append(W_k - W_corr_expr)
        corr_evolution_names.append(f"corr_W_{k}")
    # h frozen at h^*.
    h_idx = state_names.index("h")
    corr_evolution.append(state[h_idx] - Q_star[h_idx])
    corr_evolution_names.append("freeze_h")
    # P_k held at P_k^{n+1}.
    corr_p_freeze = []
    corr_p_freeze_names = []
    for i, p in enumerate(pressure_vars):
        corr_p_freeze.append(p - (P_new[i] if i < len(P_new) else p))
        corr_p_freeze_names.append(f"hold_{p.func.__name__}")
    SM_corr = {
        "state": state,
        "aux_state_extension": {"Q_star": Q_star, "P_new": P_new},
        "equation_names": corr_evolution_names + corr_p_freeze_names,
        "equations": corr_evolution + corr_p_freeze,
        "mass_matrix_partition": {
            "evolution_indices": [],
            "algebraic_indices": list(
                range(len(corr_evolution) + len(corr_p_freeze))),
        },
    }

    return SplitForPressureResult(
        pdesys=pdesys,
        pressure_vars=list(pressure_vars),
        SM_pred=SM_pred,
        SM_press=SM_press,
        SM_corr=SM_corr,
        elliptic_block=block,
    )


__all__ = [
    "build_pressure_elliptic_block",
    "verify_p_linearity",
    "split_for_pressure",
    "SplitForPressureResult",
]
