"""Residual → operator structural extraction for the declarative
:class:`~zoomy_core.model.derivation.model.Model`.

The clean-redesign :class:`Model` carries plain sympy residuals in Function
form (``h(t, x)``, ``q(k, t, x)``) with NO solver tags.  Its equations are
nonetheless bit-exact to production (:class:`zoomy_core.model.models.sme.SME`)
and — after step 8b of the SME pipeline — bundled in production's K&T (4.17)
form: the conservative flux + hydrostatic-pressure divergences fold into
``∂_x(F)`` / ``∂_x(g h²/2)`` units, while the genuinely non-conservative
couplings (the bed ``g·h·∂_x b`` and the cross-mode ``q_i/h·∂_x q_j`` terms)
stay UNFOLDED.

``SystemModel.from_model(model, Q, Qaux)`` therefore extracts the operators by
feeding each residual — rewritten from Function form into the state Symbols —
through the SAME term classifier production uses
(:func:`zoomy_core.model.derivation.tag_extraction._classify_term`).  Per additive
term:

* ``Derivative(*, t)``                          → ``mass_matrix`` (coeff of
  the ``∂_t Q_j`` it carries);
* ``coeff · Derivative(F(Q), x)``, state-free
  coeff, gravity inside                         → ``hydrostatic_pressure``;
* ``coeff · Derivative(F(Q), x)``, state-free
  coeff, no gravity                             → ``flux``;
* ``coeff · Derivative(Q_j, x)``                → ``nonconservative_matrix``
  (``B[row, j]`` += coeff);
* ``coeff(Q) · Derivative(F(Q), x)`` with a
  state-dependent coeff                         → ``nonconservative_matrix``
  (recovered via the ``∂_x q_j`` couplings it expands into);
* everything else (no spatial derivative)       → ``source`` (sign-flipped).

Second-order ``∂_x(coeff · ∂_x Q_j)`` terms are routed to the rank-4
``diffusion_matrix`` (the diffusive flux ``Fᵈ = A(Q)∇Q``) — see the Newtonian
normal-stress variant.

``Q`` ∪ ``Qaux`` must cover every field atom; otherwise a ``ValueError`` names
the uncovered fields.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.misc.misc import Zstruct


__all__ = ["extract_system_operators"]


def _field_name(field):
    head = field.func if hasattr(field, "func") else field
    return getattr(head, "__name__", str(head))


def _state_symbol(field):
    """The state/aux Symbol for a modal field — ``q(0, t, x) → q_0``,
    ``h(t, x) → h``, ``aw(2, t, x) → aw_2``.  The leading integer mode index
    (if any) becomes a ``_k`` suffix; bare scalar fields keep their name."""
    name = _field_name(field)
    args = getattr(field, "args", ())
    if args and getattr(args[0], "is_Integer", False):
        return sp.Symbol(f"{name}_{int(args[0])}", real=True)
    return sp.Symbol(name, real=True)


def _collect_fields(expr, param_names):
    """Every undefined-like field application in ``expr`` (excludes parameters,
    basis machinery, sympy builtins)."""
    from sympy.core.function import AppliedUndef
    out = set()
    for atom in expr.atoms(sp.Function):
        if not isinstance(atom, AppliedUndef):
            func = getattr(atom, "func", None)
            if (func is None
                    or sp.Function not in getattr(func, "__bases__", ())
                    or func.__dict__.get("eval") is not None):
                continue
        if getattr(atom.func, "_is_basis_head", False):
            continue
        if atom.func.__name__ in param_names:
            continue
        out.add(atom)
    return out


def extract_system_operators(model, Q, Qaux=None):
    """Structurally extract operator matrices from a declarative ``Model``.

    Parameters
    ----------
    model : zoomy_core.model.derivation.model.Model
        The threaded model after the full closure pipeline.
    Q : list of applied field Functions
        The state fields, one per evolution row (``[b, h, q(0,…), q(1,…), …]``).
    Qaux : list of applied field Functions, optional
        The auxiliary fields.  ``None`` ⇒ auto-populate with every field not in
        ``Q``.

    Returns
    -------
    dict
        ``time, space, state, aux_state, parameters, parameter_values, normal,
        flux, hydrostatic_pressure, nonconservative_matrix, source,
        mass_matrix, diffusion_matrix`` — ready to splat into
        ``SystemModel(...)``.
    """
    t = model.coords[0]
    x = model.horizontal[0] if model.horizontal else model.coords[1]
    space = [sp.Symbol("x", real=True)]
    x_sym = space[0]

    param_names = set(model.parameters.keys())
    gravity_param = (model.parameters.g
                     if "g" in model.parameters.keys() else None)

    # ── field coverage validation ────────────────────────────────────────
    all_fields = set()
    for eq in model._equations.values():
        all_fields |= _collect_fields(eq.expr, param_names)

    q_symbols = {_state_symbol(f) for f in Q}

    if Qaux is None:
        Qaux = []
        seen = set(q_symbols)
        for f in sorted(all_fields, key=str):
            s = _state_symbol(f)
            if s in seen:
                continue
            seen.add(s)
            Qaux.append(f)
    aux_symbols = {_state_symbol(f) for f in Qaux}

    covered = q_symbols | aux_symbols
    uncovered = sorted(
        {f for f in all_fields if _state_symbol(f) not in covered}, key=str)
    if uncovered:
        names = ", ".join(sp.sstr(f) for f in uncovered)
        raise ValueError(
            "SystemModel.from_model: the following field(s) appear in the "
            f"equations but are in neither Q nor Qaux: {names}.  Add each to "
            "Q (state) or Qaux (auxiliary).")

    # ── state / aux Symbol lists (ordered as given) ──────────────────────
    state = [_state_symbol(f) for f in Q]
    aux_state = [_state_symbol(f) for f in Qaux]
    n_eq = len(state)
    n_state = len(state)
    n_dim = 1

    # Map every field application → its state/aux Symbol-as-(t, x)-function so
    # structural ``∂_t`` / ``∂_x`` collection is well-defined.  ``x_sym`` is the
    # SystemModel spatial Symbol; the field's own ``x`` is rewritten onto it.
    state_fn = {s: sp.Function(str(s), real=True)(t, x_sym) for s in state}
    aux_fn = {s: sp.Function(str(s), real=True)(t, x_sym) for s in aux_state}
    field_to_fn = {}
    for f in Q:
        field_to_fn[f] = state_fn[_state_symbol(f)]
    for f in Qaux:
        field_to_fn[f] = aux_fn[_state_symbol(f)]

    state_funcs = [state_fn[s] for s in state]

    # ── operator tensors ─────────────────────────────────────────────────
    F = sp.zeros(n_eq, n_dim)
    P = sp.zeros(n_eq, n_dim)
    B = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim)
    S = sp.zeros(n_eq, 1)
    M = sp.zeros(n_eq, n_state)
    A = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim, n_dim)
    A_nonzero = False

    rows = _assign_rows(model, Q, field_to_fn, state, state_fn, t)

    for i in range(n_eq):
        residual = sp.expand(rows[i].xreplace(field_to_fn))
        _classify_row(
            residual, i, state, state_funcs, t, x_sym, gravity_param,
            F, P, B, S, M, A,
        )
        for d in range(n_dim):
            for j in range(n_state):
                for e in range(n_dim):
                    if A[i, j, d, e] != 0:
                        A_nonzero = True

    # Back-substitute Symbol-functions → bare Symbols for storage.
    rev = {v: k for k, v in {**state_fn, **aux_fn}.items()}

    def _to_sym(m):
        return m.xreplace(rev) if hasattr(m, "xreplace") else m

    F = _to_sym(F)
    P = _to_sym(P)
    S = _to_sym(S)
    M = _to_sym(M)
    B = sp.MutableDenseNDimArray(
        [[[_to_sym(sp.sympify(B[r, c, 0]))] for c in range(n_state)]
         for r in range(n_eq)])
    A_out = None
    if A_nonzero:
        A_out = sp.MutableDenseNDimArray(
            [[[[_to_sym(sp.sympify(A[r, c, dd, ee]))
                for ee in range(n_dim)] for dd in range(n_dim)]
              for c in range(n_state)] for r in range(n_eq)])

    # ── mass-matrix normalization ──────────────────────────────────────
    # The runtime solvers integrate ``∂_t Q = RHS`` — they do NOT consume a
    # mass matrix.  Rows with a CONSTANT diagonal mass entry (the Galerkin
    # Gram, e.g. shifted-Legendre 1/(2k+1)) are normalized here:
    # ``M_ii·∂_t q_i + ∂_x F_i + B_i·∂_x Q − S_i = 0  →  ÷ M_ii``.
    # Without this every higher-moment row evolves 1/M_ii times too fast
    # (the legacy chains applied InvertMassMatrix for the same reason).
    # Constraint rows (M_i· ≡ 0) and rows with non-constant or off-diagonal
    # mass entries are left untouched (the latter raise — no silent wrong
    # dynamics).
    for i in range(n_eq):
        offdiag = [sp.simplify(M[i, j]) for j in range(n_state)
                   if j != i and sp.simplify(M[i, j]) != 0]
        m_ii = (sp.cancel(sp.sympify(M[i, i])) if i < n_state
                else sp.S.Zero)
        if not offdiag and m_ii != 0 and m_ii != 1:
            if not m_ii.is_number:
                raise ValueError(
                    f"row {i}: non-constant mass-matrix diagonal {m_ii} — "
                    "cannot normalize; apply an explicit mass-matrix "
                    "inversion in the derivation.")
            F[i, 0] = sp.expand(F[i, 0] / m_ii)
            P[i, 0] = sp.expand(P[i, 0] / m_ii)
            S[i, 0] = sp.expand(S[i, 0] / m_ii)
            for j in range(n_state):
                B[i, j, 0] = sp.expand(sp.sympify(B[i, j, 0]) / m_ii)
            if A_out is not None:
                for j in range(n_state):
                    for dd in range(n_dim):
                        for ee in range(n_dim):
                            A_out[i, j, dd, ee] = sp.expand(
                                sp.sympify(A_out[i, j, dd, ee]) / m_ii)
            M[i, i] = sp.S.One
        elif offdiag:
            raise ValueError(
                f"row {i}: off-diagonal mass-matrix entries {offdiag} — "
                "the runtime cannot integrate this; apply an explicit "
                "mass-matrix inversion in the derivation.")

    parameters = Zstruct(**{k: model.parameters[k]
                            for k in model.parameters.keys()})
    parameters._symbolic_name = "p"
    parameter_values = Zstruct(**{k: getattr(model.parameter_values, k, 0.0)
                                  for k in model.parameters.keys()})

    return dict(
        time=t, space=space, state=state, aux_state=aux_state,
        parameters=parameters, parameter_values=parameter_values,
        flux=F, hydrostatic_pressure=P,
        nonconservative_matrix=B, source=S, mass_matrix=M,
        diffusion_matrix=A_out,
    )


def _classify_row(residual, i, state, state_funcs, t, x, gravity_param,
                  F, P, B, S, M, A):
    """Split one row residual into ``M / F / P / B / S / A`` slots using the
    production term classifier.  ``residual`` is in state-Symbol-as-(t, x)
    Function form."""
    from zoomy_core.model.derivation.tag_extraction import (
        _split_coeff_and_derivative,
    )

    n_state = len(state)
    state_set = set(state_funcs)
    sym_of_func = {sf: state[j] for j, sf in enumerate(state_funcs)}

    for term in sp.Add.make_args(residual):
        if term == 0:
            continue

        # 1. Time derivative → mass matrix (no sign flip — LHS).
        if any(isinstance(d, sp.Derivative) and t in d.variables
               for d in term.atoms(sp.Derivative)):
            coeff, deriv = _split_coeff_and_derivative(term)
            if deriv is not None and deriv.variables == (t,):
                inner = deriv.args[0]
                if inner in sym_of_func:
                    j = state_funcs.index(inner)
                    M[i, j] = M[i, j] + coeff
                    continue
                # ``∂_t(c · Q_j)`` — the conservative time term carries the Gram
                # factor INSIDE the derivative (``∂_t(q_k/(2k+1))``).  Pull the
                # state-free constant ``c`` out → mass-matrix entry ``coeff·c``.
                c_in, base = inner.as_independent(*state_funcs, as_Mul=True)
                if base in sym_of_func:
                    j = state_funcs.index(base)
                    M[i, j] = M[i, j] + coeff * c_in
                    continue
            # Fallback: ∂_t of a non-state function (shouldn't happen for SME).
            raise ValueError(
                f"row {i}: unhandled time-derivative term {term}")

        coeff, deriv = _split_coeff_and_derivative(term)

        # 2. Second-order diffusion ``∂_x(coeff · ∂_x Q_j)`` → diffusion_matrix.
        # Checked BEFORE the flux / NCP branches: the outer ``∂_x`` is
        # first-order in x but its argument carries an inner ``∂_x`` (the
        # diffusive flux ``Fᵈ = A ∇Q``), so it must NOT be mistyped as an
        # advective flux.
        if deriv is not None and _is_second_order_x(deriv, x):
            _route_diffusion(term, i, deriv, coeff, state_funcs, x, A)
            continue

        # 3. No spatial derivative → source (sign-flipped: S = −LHS).
        if deriv is None or deriv.variables != (x,):
            S[i, 0] = S[i, 0] - term
            continue

        inner = deriv.args[0]
        coeff_has_state = any(coeff.has(f) for f in state_set)

        # 4. coeff·∂_x(state).  A state-free, (t,x)-free coeff is an EXACT
        # divergence ``∂_x(coeff·Q_j)`` — a conservative FLUX, never a
        # ``B·∂_x Q`` coupling (the mass row ``∂_x q_0`` must reach the
        # generated flux kernels: interface/open-boundary mass exchange uses
        # flux + fluctuations, and mass-in-NCP transmits only the jump).
        # Genuinely non-conservative couplings (state-dependent coeff: the
        # bed slope ``g h ∂_x b``, the SME cross-mode ``q_i/h·∂_x q_j``)
        # stay in B.
        if inner in sym_of_func:
            if not coeff_has_state and not coeff.has(x) and not coeff.has(t):
                flux_i = coeff * inner
                if gravity_param is not None and flux_i.has(gravity_param):
                    P[i, 0] = P[i, 0] + flux_i
                else:
                    F[i, 0] = F[i, 0] + flux_i
                continue
            j = state_funcs.index(inner)
            B[i, j, 0] = B[i, j, 0] + coeff
            continue

        inner_has_state = any(inner.has(f) for f in state_set)

        # 5. ∂_x(F(state)) with state-free coeff → flux / pressure.
        if inner_has_state and not coeff_has_state:
            flux_i = coeff * inner
            if gravity_param is not None and inner.has(gravity_param):
                P[i, 0] = P[i, 0] + flux_i
            else:
                F[i, 0] = F[i, 0] + flux_i
            continue

        # 6. state-dependent coeff on a ∂_x argument — the non-conservative
        # product that production keeps unfolded (the SME cross-mode couplings
        # ``q_i/h · ∂_x q_j``).  Route to ``B`` (or source if not a clean
        # coupling).
        _route_nonconservative_product(
            term, i, state, state_funcs, x, B, S)


def _is_second_order_x(deriv, x):
    """True if ``deriv`` is a pure second-order ``∂_x ∂_x`` (or contains a
    nested ``∂_x`` of a ``∂_x``)."""
    if deriv.variables == (x, x):
        return True
    # ∂_x(coeff·∂_x Q): the outer is first-order in x but its argument carries
    # an inner ∂_x.
    if deriv.variables == (x,):
        inner = deriv.args[0]
        return any(d.variables == (x,) for d in inner.atoms(sp.Derivative))
    return False


def _route_diffusion(term, i, deriv, coeff, state_funcs, x, A):
    """Route a second-order ``coeff · ∂_x(D · ∂_x Q_j)`` term into the rank-4
    diffusion tensor ``A[i, j, 0, 0]`` (the diffusive flux ``Fᵈ = A ∇Q``).

    The residual carries ``+∂_x(Fᵈ)`` with ``Fᵈ[i] = Σ_j A[i, j, 0, 0]·∂_x Q_j``;
    we read the inner ``D·∂_x Q_j`` and place ``coeff · D`` at ``A[i, j, 0, 0]``.
    """
    inner = deriv.args[0]            # D · ∂_x Q_j  (the diffusive flux)
    # ``.doit()`` resolves an inner ``∂_x(q_j/h)`` (the CoV-introduced
    # conserved-variable derivative) into ``∂_x q_j/h − q_j·∂_x h/h²`` so the
    # BARE state derivatives ``∂_x q_j`` / ``∂_x h`` surface and are routed; a
    # raw ``∂_x(q_j/h)`` would otherwise carry a non-state ``q_j/h`` argument
    # and the dominant diagonal diffusion would be silently dropped.
    for sub in sp.Add.make_args(sp.expand(inner.doit())):
        c2, d2 = _split_inner_x_derivative(sub, x)
        if d2 is None:
            continue
        q = d2.args[0]
        if q in state_funcs:
            j = state_funcs.index(q)
            A[i, j, 0, 0] = A[i, j, 0, 0] + coeff * c2


def _split_inner_x_derivative(term, x):
    """Return ``(coeff, ∂_x Q)`` for a ``coeff · ∂_x Q`` term, else
    ``(None, None)``."""
    factors = term.args if isinstance(term, sp.Mul) else [term]
    deriv = None
    for f in factors:
        if isinstance(f, sp.Derivative) and f.variables == (x,):
            deriv = f
            break
    if deriv is None:
        return None, None
    coeff = sp.Mul(*[f for f in factors if f is not deriv])
    return coeff, deriv


def _route_nonconservative_product(term, i, state, state_funcs, x, B, S):
    """A ``coeff(state) · ∂_x(F(state))`` term that production keeps unfolded —
    it is already a ``coeff · ∂_x Q_j`` non-conservative coupling (the SME
    cross-mode terms ``q_i/h · ∂_x q_j``).  Read ``(j, coeff)`` and add to
    ``B[i, j]``.  If the derivative argument is not a bare state Symbol the
    term cannot be a clean NCP coupling — route it to source as a fallback."""
    from zoomy_core.model.derivation.tag_extraction import (
        _split_coeff_and_derivative,
    )
    coeff, deriv = _split_coeff_and_derivative(term)
    if deriv is not None and deriv.variables == (x,):
        inner = deriv.args[0]
        if inner in state_funcs:
            j = state_funcs.index(inner)
            B[i, j, 0] = B[i, j, 0] + coeff
            return
    # Not a clean coupling — keep as source (sign-flipped).
    S[i, 0] = S[i, 0] - term


def _assign_rows(model, Q, field_to_fn, state, state_fn, t):
    """Order the equation residuals so row ``i`` is the equation whose ``∂_t``
    carries ``state[i]`` (the evolution row for that state).  Falls back to the
    declared equation order when no ``∂_t`` match is found (algebraic rows)."""
    eqs = list(model._equations.values())

    def _has_dt_of(expr, sf):
        """True iff ``expr`` carries a ``∂_t(c · sf)`` term — matching the
        conservative time term whose Gram factor sits inside the derivative
        (``∂_t(q_k/(2k+1))``), not only the bare ``∂_t(sf)``."""
        for der in expr.atoms(sp.Derivative):
            if der.variables == (t,):
                _, base = der.args[0].as_independent(sf, as_Mul=True)
                if base == sf:
                    return True
        return False

    rows = [None] * len(state)
    used = set()
    for i, s in enumerate(state):
        sf = state_fn[s]
        for k, eq in enumerate(eqs):
            if k in used:
                continue
            if _has_dt_of(sp.expand(eq.expr.xreplace(field_to_fn)), sf):
                rows[i] = eq.expr
                used.add(k)
                break
    remaining = [eq for k, eq in enumerate(eqs) if k not in used]
    ri = 0
    for i in range(len(state)):
        if rows[i] is None:
            rows[i] = remaining[ri].expr if ri < len(remaining) else sp.S.Zero
            ri += 1
    return rows
