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

    def pressure_part(expr):
        """Part of expr that depends on any pressure variable.

        Strategy: zero out all P_l (sympy propagates through
        ``Derivative``) and subtract from the full expression.
        """
        no_p = expr.subs({p: sp.S.Zero for p in pressure_vars})
        return sp.expand(expr - no_p)

    # Per-conservative-variable pressure sources, computed directly
    # via the Escalante master formula (mirrors
    # ``derive_xmom_j`` / ``derive_zmom_j`` in
    # ``escalante2024_poisson_generic.py``).  We do NOT extract from
    # the input pdesys's xmom/zmom rows because Zoomy's
    # ``Multiply + Integrate`` chain misses the chain-rule contribution
    # from ``∂_x φ_j|_z`` for non-constant test functions (j ≥ 1),
    # producing rows that are correct at j=0 but missing terms at
    # j ≥ 1.  Direct master-formula evaluation sidesteps that bug.
    #
    # Convention: the chain DAE works with dynamic pressure P_k and
    # the momentum equations carry a ``(1/rho) ∂_x p_NH`` term, so the
    # master-formula pressure sources need a ``1/rho`` factor.
    rho = next((s for s in pdesys.parameters if str(s) == "rho"), None)
    if rho is None:
        # No density parameter — treat as unit (kinematic convention).
        rho = sp.S.One
    T_u = _master_formula_T_u(
        coeffs_u=coeffs_u, coeffs_w=coeffs_w,
        pressure_vars=pressure_vars,
        h=h, bottom=bottom, t=t, x=x,
        M=M, N_w_active=N_w_active, rho=rho,
    )
    T_w = _master_formula_T_w(
        coeffs_u=coeffs_u, coeffs_w=coeffs_w,
        pressure_vars=pressure_vars,
        h=h, bottom=bottom, t=t, x=x,
        M=M, N_w_active=N_w_active, rho=rho,
    )

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
# Master-formula projections.  Adapted from
# ``tutorials/vam/escalante2024_poisson_generic.py:103-136``.
# ---------------------------------------------------------------------------


def _master_formula_T_u(
    *, coeffs_u, coeffs_w, pressure_vars, h, bottom, t, x, M, N_w_active, rho,
):
    """Return T_u[k] = pressure source per conserved variable h*U_k,
    for k = 0, …, M.  Computed via the Escalante master formula for
    the j-th x-momentum projection (with the time-derivative weight
    ``mu_j`` divided out).
    """
    return [
        _master_pressure_source_xmom(
            j=j,
            coeffs_u=coeffs_u, coeffs_w=coeffs_w,
            pressure_vars=pressure_vars,
            h=h, bottom=bottom, t=t, x=x,
            M=M, N_w_active=N_w_active, rho=rho,
        )
        for j in range(M + 1)
    ]


def _master_formula_T_w(
    *, coeffs_u, coeffs_w, pressure_vars, h, bottom, t, x, M, N_w_active, rho,
):
    """Return T_w[k] = pressure source per conserved variable h*W_k,
    for k = 0, …, N_w_active − 1.  Computed via the Escalante master
    formula for the j-th z-momentum projection.
    """
    return [
        _master_pressure_source_zmom(
            j=j,
            coeffs_u=coeffs_u, coeffs_w=coeffs_w,
            pressure_vars=pressure_vars,
            h=h, bottom=bottom, t=t, x=x,
            M=M, N_w_active=N_w_active, rho=rho,
        )
        for j in range(N_w_active)
    ]


def _master_pressure_source_xmom(
    *, j, coeffs_u, coeffs_w, pressure_vars,
    h, bottom, t, x, M, N_w_active, rho,
):
    """Master-formula pressure source for ``∂_t(h U_j)`` (j-th x-mom).

    Mirrors ``derive_xmom_j`` in escalante2024_poisson_generic.py:
    pressure-dependent part of (boundary term − ∫ φ_j' (omega·u −
    p (ζ ∂_x h + ∂_x b)) dζ + flux pressure-part), divided by
    ``mu(j) = 1/(2j+1)``.
    """
    return _master_pressure_source(
        component="xmom", j=j,
        coeffs_u=coeffs_u, coeffs_w=coeffs_w,
        pressure_vars=pressure_vars,
        h=h, bottom=bottom, t=t, x=x,
        M=M, N_w_active=N_w_active, rho=rho,
    )


def _master_pressure_source_zmom(
    *, j, coeffs_u, coeffs_w, pressure_vars,
    h, bottom, t, x, M, N_w_active, rho,
):
    """Master-formula pressure source for ``∂_t(h W_j)`` (j-th z-mom).

    Mirrors ``derive_zmom_j`` in escalante2024_poisson_generic.py.
    """
    return _master_pressure_source(
        component="zmom", j=j,
        coeffs_u=coeffs_u, coeffs_w=coeffs_w,
        pressure_vars=pressure_vars,
        h=h, bottom=bottom, t=t, x=x,
        M=M, N_w_active=N_w_active, rho=rho,
    )


def _master_pressure_source(
    *, component, j, coeffs_u, coeffs_w, pressure_vars,
    h, bottom, t, x, M, N_w_active, rho,
):
    """Common implementation for x-mom / z-mom master pressure source.

    Steps:
      1. Build polynomial ansatz u(ζ), w(ζ), p_NH(ζ) using shifted
         Legendre on [0, 1].  The pressure ansatz uses the active
         ``pressure_vars`` plus the surface-BC closure
         ``P_{N_p} = (eliminated)``.
      2. Apply Escalante's ``derive_xmom_j`` / ``derive_zmom_j``
         formulae.
      3. Extract the pressure-dependent part (zero out P, subtract).
      4. Divide by ``mu(j) = 1/(2j+1)``.
    """
    from sympy.functions.special.polynomials import legendre

    zeta = sp.Symbol("_zeta_master", positive=True)
    eta = h + bottom

    # The active pressure modes are pressure_vars (P_0, …, P_{N_p-1}).
    # The full ansatz needs N_p+1 modes; the highest is closed via
    # the surface BC.  Reconstruct N_p from len(pressure_vars).
    N_p_active = len(pressure_vars)

    def shifted_legendre(k):
        return sp.expand(legendre(k, 1 - 2 * zeta))

    max_order = max(M, N_w_active, N_p_active) + 1
    phi = [shifted_legendre(k) for k in range(max_order + 2)]

    # Closure for the surface pressure mode P_{N_p}: solve
    # Σ_k φ_k(1) · P_k = 0 for the highest mode.
    # φ_k(1) = (−1)^k for shifted Legendre, so
    #     Σ_{k=0}^{N_p} (−1)^k P_k = 0
    #     ⇒ P_{N_p} = − (−1)^{N_p} · Σ_{k<N_p} (−1)^k P_k
    #              = − Σ_{k<N_p} (−1)^{N_p − k − 1} … (just solve via sympy)
    # We construct a placeholder P_{N_p} symbol and use sp.solve.
    P_top_sym = sp.Function(f"P_{N_p_active}_top", real=True)(t, x)
    p_NH_full = sum(
        pressure_vars[k] * phi[k] for k in range(N_p_active)
    ) + P_top_sym * phi[N_p_active]
    p_at_eta = sp.expand(p_NH_full.subs(zeta, sp.S.One))
    P_top_sol = sp.solve(p_at_eta, P_top_sym)[0]
    p_NH = sp.expand(p_NH_full.subs(P_top_sym, P_top_sol))

    # Velocity ansatz (full, including W_{N_w} which will be closed by
    # the bottom KBC after the pressure source is extracted).  The
    # input ``coeffs_w`` carries only the active modes
    # (W_0..W_{N_w_active-1}); we add a placeholder for the closed
    # mode here, then resolve it after pressure-part extraction.
    W_top_placeholder = sp.Function(
        f"_W_top_placeholder_master", real=True)(t, x)
    coeffs_w_full = list(coeffs_w) + [W_top_placeholder]
    u_poly = sum(coeffs_u[k] * phi[k] for k in range(M + 1))
    w_poly = sum(coeffs_w_full[k] * phi[k] for k in range(N_w_active + 1))

    # Sigma-coord vertical velocity ω(ζ).
    omega = (w_poly
             - zeta * sp.Derivative(h, t)
             - u_poly * (zeta * sp.Derivative(h, x).doit()
                         + sp.Derivative(bottom, x).doit()))

    phi_j = phi[j]
    phi_j_prime = sp.diff(phi_j, zeta)

    def _ipoly(integrand):
        """Polynomial integral over ζ ∈ [0, 1]."""
        e = sp.expand(integrand)
        if not e.has(zeta):
            return e
        try:
            return sp.expand(sp.integrate(e, (zeta, 0, 1)))
        except (NotImplementedError, sp.PolynomialError):
            return sp.expand(sp.integrate(e, (zeta, 0, 1)))

    int_phi_j = _ipoly(phi_j)
    phi_j_at_0 = sp.expand(phi_j.subs(zeta, sp.S.Zero))
    phi_j_at_1 = sp.expand(phi_j.subs(zeta, sp.S.One))
    p_at_0 = sp.expand(p_NH.subs(zeta, sp.S.Zero))
    p_at_1 = sp.expand(p_NH.subs(zeta, sp.S.One))

    if component == "xmom":
        int_phi_j_p = _ipoly(phi_j * p_NH)
        boundary = (-phi_j_at_1 * p_at_1 * sp.Derivative(eta, x).doit()
                    + phi_j_at_0 * p_at_0 * sp.Derivative(bottom, x).doit())
        # The pressure-dependent piece of ``-∫ φ_j' (ω·u - p(ζ∂_x h + ∂_x b)) dζ``
        # is ``+∫ φ_j' p (ζ∂_x h + ∂_x b) dζ`` (the ω·u part has no P).
        int_dphi_inner_p = _ipoly(
            phi_j_prime * p_NH
            * (zeta * sp.Derivative(h, x).doit()
               + sp.Derivative(bottom, x).doit())
        )
        # Pressure-dependent part of:
        #   ∂_x(h ∫ φ_j u² dζ + h ∫ φ_j p dζ)  [only the p-piece]
        # + g h ∂_x η ∫ φ_j dζ                  [no P]
        # + boundary
        # − ∫ φ_j' inner                        [pressure piece is +∫φ_j'p(...)]
        full_pressure_part = (
            sp.Derivative(h * int_phi_j_p, x).doit()
            + boundary
            + int_dphi_inner_p
        )
    elif component == "zmom":
        # z-mom Escalante eq:
        #   ∂_t(h ∫ φ_j w dζ) + ∂_x(h ∫ φ_j u w dζ) + boundary
        #     - ∫ φ_j' (ω·w + p) dζ = 0
        boundary = phi_j_at_1 * p_at_1 - phi_j_at_0 * p_at_0
        # Pressure-dependent piece of ``-∫ φ_j' (ω·w + p) dζ`` is
        # ``-∫ φ_j' p dζ`` (ω·w has no P).
        int_dphi_p = _ipoly(phi_j_prime * p_NH)
        full_pressure_part = boundary - int_dphi_p
    else:
        raise ValueError(f"Unknown component {component!r}")

    # Now extract the part that depends on the active pressure
    # variables (pressure_vars).  After the surface-BC substitution
    # above, ``p_NH`` is already linear in pressure_vars and excludes
    # P_{N_p_active}; pressure_part(...) just normalises.
    no_p = full_pressure_part.subs(
        {p: sp.S.Zero for p in pressure_vars})
    pressure_only = sp.expand(full_pressure_part - no_p)

    # Resolve the W_top placeholder via the bottom KBC closure.
    # Bot KBC: w(0) − u(0) ∂_x b = 0 → solve for W_top.
    sum_u_at_0 = sum(coeffs_u[k] for k in range(M + 1))
    # φ_k(0) = 1 for all k (shifted Legendre), so the closure is
    # just the unweighted sum.
    sum_w_active_at_0 = sum(coeffs_w[k] for k in range(N_w_active))
    w_top_closure = (sum_u_at_0 * sp.Derivative(bottom, x).doit()
                     - sum_w_active_at_0)
    pressure_only = sp.expand(
        pressure_only.subs(W_top_placeholder, w_top_closure))

    # Divide by mu(j) = 1/(2j+1) to get the source per conserved
    # variable (h Q_j) instead of per μ_j (h Q_j), and by ``rho`` to
    # match the chain DAE's ``(1/rho) ∂_x p_NH`` convention.  The
    # master-formula derivation (Escalante) writes everything in
    # kinematic form; the chain DAE uses dynamic pressure with an
    # explicit 1/rho factor, so T must absorb that factor.
    mu_j = sp.Rational(1, 2 * j + 1)
    return sp.expand(pressure_only / (mu_j * rho))


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
