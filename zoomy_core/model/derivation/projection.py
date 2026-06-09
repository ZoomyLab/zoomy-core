"""Modal projection operations for the clean-redesign framework.

After :func:`~zoomy_core.model.derivation.modal.separation_of_variables` has put the
unexpanded ansatz ``Σ_i a(i, …)·φ(i, ζ)`` into every equation, these four ops
carry the derivation to the Galerkin-projected, basis-resolved form:

:class:`ExpandSums`
    Products / integer powers of unexpanded ``sp.Sum`` factors → ONE
    multi-index ``Sum`` with DISTINCT dummies (cross terms kept):
    ``(Σ_i a_iφ_i)² → Σ_iΣ_j a_i a_j φ_iφ_j``.

:class:`Project`
    Galerkin PROJECT — sugar for ``Multiply(test_function)`` then integrate
    over ``(ζ, 0, 1)``, pushing the ζ-integral *through* the unexpanded Sum
    so the opaque ``∫ φ·c·φ dζ`` brackets stay unevaluated:
    ``Σ_i a_i ⟨φ_i, c φ_l⟩``.

:class:`ExtractBrackets`
    The GENERIC bracket split: a ζ-INDEPENDENT factor comes OUT as ``const``;
    the ζ-DEPENDENT body stays as ``Integral(body, (ζ, 0, 1))`` (rendered
    ``⟨body⟩``).  ONLY ``Gram(i, l)`` and ``Weight(l)`` keep closed forms.
    ``∂_x`` commutes out of the ζ-integral (conservative flux).

:class:`ResolveBasis`
    Per-field bracket resolution: substitute the concrete basis's
    ``closed_form_bracket`` (Gram / Weight → δ-form, ``l`` kept symbolic) and
    integrate every leftover ``∫_0^1 … dζ`` whose integrand carries this
    field's index, then ``.doit()`` collapses the δ-Sums.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import (
    Operation, Multiply, is_bracket, is_bracket_body)


__all__ = [
    "Gram",
    "Weight",
    "is_bracket",
    "is_bracket_body",
    "bracket_atoms",
    "ExpandSums",
    "EvaluateSums",
    "Integrate",
    "Project",
    "PullConstants",
    "pull_out",
    "ExtractBrackets",
    "ResolveBasis",
]


# ── named bracket Function classes (closed-form-able) ──────────────────────


def _bracket_tex(name):
    return {
        "Gram":   lambda A: rf"\langle \phi_{{{A[0]}}}, c\phi_{{{A[1]}}}\rangle",
        "Weight": lambda A: rf"\langle c, \phi_{{{A[0]}}}\rangle",
    }[name]


def _mk_bracket(name):
    body = _bracket_tex(name)

    def _latex(expr, printer=None, exp=None):
        render = printer._print if printer is not None else sp.latex
        A = [str(render(a)) for a in expr.args]
        s = body(A)
        return s if exp is None else f"{s}^{{{exp}}}"

    return type(name, (sp.Function,), {"_latex": _latex, "_is_bracket": True})


# ⟨φ_i, c φ_l⟩ and ⟨c, φ_l⟩ — the only shapes with a closed form for an
# orthogonal basis (everything else stays an explicit ⟨…⟩ Integral).
Gram = _mk_bracket("Gram")
Weight = _mk_bracket("Weight")


def bracket_atoms(expr):
    """All named-bracket Function atoms present in ``expr``."""
    return {a for a in expr.atoms(sp.Function)
            if getattr(a.func, "_is_bracket", False)}


# ── ExpandSums ─────────────────────────────────────────────────────────────


# Mode-index letters, in order (``l`` is the Galerkin test index — skipped).
_MODE_LETTERS = ["i", "j", "k", "m", "n", "o", "p", "q", "r", "s"]


def _fresh_index(idx, used):
    """A fresh integer, non-negative summation index NOT in ``used``, drawn from
    the canonical mode-letter pool ``i, j, k, m, …`` so every expanded sum reads
    in the SAME convention as the ansatz / closure indices (``Σ_{i,j}``, not the
    suffixed ``Σ_{i_1,i_2}``).  Falls back to ``idx_n`` only if the pool is
    exhausted."""
    for nm in _MODE_LETTERS:
        cand = sp.Symbol(nm, integer=True, nonnegative=True)
        if cand not in used:
            return cand
    base = idx.name if isinstance(idx, sp.Symbol) else "i"
    n = 1
    while True:
        cand = sp.Symbol(f"{base}_{n}", integer=True, nonnegative=True)
        if cand not in used:
            return cand
        n += 1


def _merge_sums(sums):
    """Merge ``sp.Sum`` factors into ONE multi-index Sum with DISTINCT indices
    (so cross terms ``i ≠ j`` survive).  The FIRST occurrence of an index keeps
    its name; later duplicates are relabelled to fresh mode letters — so
    ``(Σ_i a_i φ_i)² → Σ_{i,j} a_i a_j φ_i φ_j`` rather than ``Σ_{i_1,i_2}``."""
    used = set()
    for s in sums:
        used |= set(s.free_symbols)
    claimed, summands, limits = set(), [], []
    for s in sums:
        repl, new_limits = {}, []
        for lim in s.limits:
            idx = lim[0]
            new = idx if idx not in claimed else _fresh_index(idx, used | claimed)
            claimed.add(new)
            used.add(new)
            repl[idx] = new
            new_limits.append((new,) + tuple(lim[1:]))
        summands.append(s.function.xreplace(repl))
        limits.extend(new_limits)
    return sp.Sum(sp.Mul(*summands), *limits)


def _expand_sum_products(e):
    """Recursively rewrite products / positive-integer powers of ``sp.Sum``
    into a single multi-index Sum."""
    if not getattr(e, "args", None):
        return e
    e = e.func(*[_expand_sum_products(a) for a in e.args])
    if (isinstance(e, sp.Pow) and isinstance(e.base, sp.Sum)
            and e.exp.is_Integer and e.exp > 0):
        return _merge_sums([e.base] * int(e.exp))
    if isinstance(e, sp.Mul):
        sums, rest = [], []
        for f in e.args:
            if isinstance(f, sp.Sum):
                sums.append(f)
            elif (isinstance(f, sp.Pow) and isinstance(f.base, sp.Sum)
                  and f.exp.is_Integer and f.exp > 0):
                sums.extend([f.base] * int(f.exp))
            else:
                rest.append(f)
        if len(sums) >= 2:
            return sp.Mul(*rest) * _merge_sums(sums)
    return e


class ExpandSums(Operation):
    """Expand products / integer powers of unexpanded ``sp.Sum`` factors into a
    SINGLE multi-index ``Sum`` with DISTINCT dummies — cross terms kept.

    A quadratic ``u·u`` with the modal ansatz ``u = Σ_i a_i φ_i`` becomes
    ``(Σ_i a_i φ_i)²`` after substitution, and ``sympy`` keeps the two factors
    SHARING ``i`` (silently dropping the ``i ≠ j`` cross terms).  ``ExpandSums``
    relabels and merges:

    .. math::

        \\Bigl(\\sum_i a_i\\phi_i\\Bigr)^2
            \\;\\longrightarrow\\;
        \\sum_i\\sum_j a_i\\,a_j\\,\\phi_i\\,\\phi_j .

    Apply it AFTER the ansatz substitution and BEFORE :class:`Project`.
    """

    whole_leaf_op = True

    def __init__(self, name="expand_sums"):
        super().__init__(
            name=name,
            description="products/powers of Sum → one multi-index Sum",
        )

    def _leaf_sp(self, sp_expr):
        return _expand_sum_products(sp_expr)


class EvaluateSums(Operation):
    """Unroll every FINITE ``sp.Sum`` (a concrete integer bound) to its explicit
    modes: ``Σ_{i=0}^{2} a_i φ_i → a_0 φ_0 + a_1 φ_1 + a_2 φ_2``.

    The modal ansatz from :func:`~zoomy_core.model.derivation.modal.separation_of_variables`
    is an UNEXPANDED ``sp.Sum`` with an abstract bound ``N_u``; after
    ``Substitution({N_u: N})`` binds the bound, ``sympy`` still keeps the ``Sum``
    node — ``.doit()`` is what expands it.  This op is the operation form of the
    bind-then-expand step (replacing the raw ``for eq: eq.expr.replace(Sum,
    doit)`` loop), and is deliberately NARROWER than a bare ``expr.doit()``: it
    only fires on ``Sum`` atoms, leaving any deferred ``Integral`` / ``Derivative``
    untouched (the pipeline defers those on purpose)."""

    whole_leaf_op = True

    def __init__(self, name="evaluate_sums"):
        super().__init__(
            name=name,
            description="finite Sum.doit() → explicit modes",
        )

    def _leaf_sp(self, sp_expr):
        # ``deep=False`` is essential: a bare ``Sum.doit()`` first calls
        # ``self.function.doit()``, which descends into the opaque Galerkin
        # ``Integral(c φ_i φ_j φ_k …)`` and hands it to sympy's ``heurisch``
        # symbolic integrator — that recurses without bound on the unevaluated
        # ``φ`` heads and hangs.  We only want the SUMMATION unrolled to explicit
        # modes; the inner ``Integral`` / ``Derivative`` stay deferred (ResolveBasis
        # evaluates the brackets later, by antiderivative, not ``heurisch``).
        return sp_expr.replace(lambda e: isinstance(e, sp.Sum),
                               lambda e: e.doit(deep=False))


# ── Project (Multiply(test) → ∫ dζ through the Sum) ────────────────────────


def _integrate_one_term(term, var, lo, hi):
    """Integrate a single additive term over ``(var, lo, hi)``.

    The ``var``-INDEPENDENT factor comes out front; the ``var``-dependent body
    ``rest`` is then resolved by shape:

    * a BARE outer ``Derivative(g, (var,))`` (``∂_ζ`` of the whole body) →
      **fundamental theorem of calculus**: ``g|_{var=hi} − g|_{var=lo}`` (this is
      how the vertical-velocity boundary traces ``w̃(·,0/1)`` surface, ready for a
      :class:`~zoomy_core.model.operations.KinematicBC`);
    * a BARE outer ``Derivative(g, (w,))`` in another variable ``w ≠ var`` (a
      CONSERVATIVE flux ``∂_x(F)`` whose ``φ·h`` was moved INSIDE by a prior
      ``ProductRule``) → the derivative **commutes OUT** of the ζ-integral:
      ``∂_x(∫ g dζ)``;
    * a single unexpanded ``Sum`` factor → push the integral inside, leaving the
      opaque ``φ`` bracket unevaluated;
    * everything else (incl. σ-metric coefficient-derivative factors like
      ``∂_x(ζh)`` that must NOT commute, and ``coeff·∂_x F`` terms whose ``φ/h``
      were NOT folded in) → left as an opaque ``sp.Integral`` for
      :class:`~zoomy_core.model.derivation.closure.ResolveIntegral` to close by basis.

    The conservative commute fires ONLY on a *bare* outer derivative: a generic
    ``coeff·∂_x F`` is left abstract, because pulling ``∂_x`` past an
    ``x``-dependent ``coeff`` (e.g. ``h``) would give the NON-conservative
    ``h·∂_x(…)`` instead of ``∂_x(h·…)``.  Moving ``φ·h`` inside the derivative
    (so the term is bare) is the job of the ``ProductRule`` that precedes
    ``Integrate`` in the projection.
    """
    coeff, rest = term.as_independent(var, as_Add=False)
    if rest == 1:
        return coeff * (hi - lo)
    # A bare outer Derivative as the whole var-dependent body.
    if isinstance(rest, sp.Derivative) and len(rest.variables) == 1:
        dvar = rest.variables[0]
        g = rest.args[0]
        if dvar == var:                       # FTC over the integration var
            return coeff * (g.subs(var, hi) - g.subs(var, lo))
        return coeff * sp.Derivative(         # conservative flux: ∂_x commutes out
            _integrate_through_sum(g, var, lo, hi), dvar)
    factors = list(rest.args) if isinstance(rest, sp.Mul) else [rest]
    # A conservative ``∂_w`` (non-``var``) derivative FACTOR whose remaining
    # factors + coefficient are ``w``-INDEPENDENT commutes OUT of the
    # ζ-integral: ``∫ c φ_l ∂_x F dζ = ∂_x ∫ c φ_l F dζ``.  When the coefficient
    # is ``w``-DEPENDENT (an ``h`` pulled out by ``Multiply(h)``) the pull-out
    # is INVALID (``∂_x(h·…) ≠ h·∂_x(…)``) — leave the term abstract so a prior
    # ``ProductRule`` folds ``h`` inside the derivative first.
    for f in factors:
        if (isinstance(f, sp.Derivative) and len(f.variables) == 1
                and f.variables[0] != var):
            dw = f.variables[0]
            others = sp.Mul(*[g for g in factors if g is not f])
            if not (coeff * others).has(dw):
                inner = _integrate_through_sum(others * f.expr, var, lo, hi)
                return coeff * sp.Derivative(inner, dw)
            break
    # A single unexpanded Sum factor: push the integral inside.
    sums = [f for f in factors if isinstance(f, sp.Sum)]
    if len(sums) == 1:
        the_sum = sums[0]
        others = sp.Mul(*[f for f in factors if f is not the_sum])
        cc, dep = the_sum.function.as_independent(var, as_Add=False)
        oc, od = others.as_independent(var, as_Add=False)
        return coeff * oc * sp.Sum(
            cc * sp.Integral(od * dep, (var, lo, hi)), *the_sum.limits)
    # Otherwise leave the var-dependent body as an opaque Integral.
    return coeff * sp.Integral(rest, (var, lo, hi))


def _integrate_through_sum(expr, var, lo, hi):
    """Integrate ``expr`` over ``(var, lo, hi)``, pushing the integral *through*
    any unexpanded ``sp.Sum``.  ``sympy``'s ``Integral.doit`` refuses to
    interchange a Sum and an Integral with an abstract bound, so we do it by
    hand — leaving the opaque ``φ`` integrals unevaluated (they become the
    brackets).  Result: ``Σ_i coeff(i)·∫(var-dep) dvar``.  Zero ``Subs``."""
    expr = sp.expand_mul(expr)
    out = sp.S.Zero
    for term in sp.Add.make_args(expr):
        out += _integrate_one_term(term, var, lo, hi)
    return out


class Integrate(Operation):
    """PURELY ABSTRACT definite integral: wrap EACH additive term of the leaf in
    an unevaluated ``sp.Integral(term, (var, lo, hi))`` and stop.  It performs NO
    evaluation — no FTC, no ``∂_x``-commute, no Sum-push, no basis substitution.
    All of that "smart" resolution is the job of
    :class:`~zoomy_core.model.derivation.closure.ResolveIntegral` (the ``auto`` method
    classifies each ``∫`` atom: ``∂_ζ`` → FTC, ``∂_x`` → commute, ``Sum`` → push,
    ``φ``-bracket → basis matrix) or of
    :class:`ExtractBrackets` (``∫c·φ_i φ_l → Gram(i,l)``).

    Wrapping per additive term keeps the result readable — one ``∫`` per term —
    and lets the resolution step decide each term's fate independently.

    When ``var`` also appears in the bounds (a RUNNING integral, e.g.
    ``bounds=(0, ζ)``) the integration variable would collide with the bound and
    later ops would drag the σ-coordinate ζ into the rename.  In that case the
    integrand is bound to a fresh ``\\hat{<var>}`` Dummy, leaving the bound ζ
    free: ``∫_0^ζ ũ(\\hat ζ) d\\hat ζ``.  Pass ``dummy=`` to choose the symbol.
    """

    whole_leaf_op = True

    def __init__(self, var=None, bounds=(0, 1), dummy=None, name="integrate"):
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._bounds = tuple(bounds)
        self._dummy = dummy
        super().__init__(
            name=name,
            description=f"∫ d{self._var} over {self._bounds} (abstract)")

    def _leaf_sp(self, sp_expr):
        var, (lo, hi) = self._var, self._bounds
        # bind to a fresh dummy ONLY when var appears in the bounds (collision).
        bound_syms = sp.sympify(lo).free_symbols | sp.sympify(hi).free_symbols
        if self._dummy is not None:
            bvar = self._dummy
        elif var in bound_syms:
            bvar = sp.Dummy(rf"\hat{{{sp.latex(var)}}}", real=True)
        else:
            bvar = var
        return sp.Add(*[sp.Integral(term.subs(var, bvar) if bvar is not var else term,
                                    (bvar, lo, hi))
                        for term in sp.Add.make_args(sp.expand(sp_expr))])


class Project(Operation):
    """Galerkin PROJECT — sugar for ``Multiply(test_function)`` then integrate
    over ``(ζ, 0, 1)``.

    The test function is passed VERBATIM (typically ``c(ζ)·φ(l, ζ)``); the
    multiply is the production
    :class:`~zoomy_core.model.operations.Multiply`, and the integral pushes
    *through* the unexpanded Sum so the opaque ``∫ φ·c·φ dζ`` brackets stay
    unevaluated:

    .. math::

        \\sum_i a_i(t,x)\\,\\langle\\phi_i, c\\phi_l\\rangle .

    Parameters
    ----------
    test_function : sympy.Expr
        The full test weight to multiply by — e.g. ``c(ζ)·φ(l, ζ)``.
    var : sympy.Symbol
        ζ integration variable (default ``Symbol("zeta")``).
    bounds : tuple[Expr, Expr]
        Integration interval (default ``(0, 1)``).
    """

    whole_leaf_op = True

    def __init__(self, test_function, var=None, bounds=(0, 1), name="project"):
        self._test = sp.sympify(test_function)
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._bounds = tuple(bounds)
        self._multiply = Multiply(self._test)
        self._integrate = Integrate(var=self._var, bounds=self._bounds)
        super().__init__(
            name=name,
            description=f"PROJECT onto {self._test} over {self._var}",
        )

    def _leaf_sp(self, sp_expr):
        scaled = self._multiply._leaf_sp(sp_expr)
        return self._integrate._leaf_sp(scaled)


# ── PullConstants (linearity normalisation, shared by ∫ and ∂) ─────────────


def _dist_deriv_over_sum(expr):
    """``∂_v(Σ_j body) → Σ_j ∂_v body`` with the ``v``-independent part of the
    summand pulled out of the derivative — the derivative-side of "push the
    operator-independent factor out"."""
    def _f(e):
        body, wrt = e.args[0].function, e.variables
        const, dep = (body.as_independent(*wrt)
                      if isinstance(body, sp.Mul) else (sp.S.One, body))
        return sp.Sum(const * sp.Derivative(dep, *wrt), *e.args[0].limits)
    return expr.replace(
        lambda e: isinstance(e, sp.Derivative) and isinstance(e.args[0], sp.Sum),
        _f)


def _pull_piece(piece, v, lo, hi):
    """One additive term of an ``∫_v`` integrand: push an independent ``Σ`` out,
    commute an independent ``∂_w`` (``w⊥v``, ``v``-dependent argument) out, or
    hoist the ``v``-independent constant factor out — leaving a ``v``-dependent
    body under ``∫_v``.  Returns ``(rewritten, changed)``."""
    facs = list(piece.args) if isinstance(piece, sp.Mul) else [piece]
    # An independent ``Σ_j`` (``j ⊥ v``) commutes OUT of the integral.
    the_sum = next((f for f in facs if isinstance(f, sp.Sum)), None)
    if the_sum is not None:
        others = sp.Mul(*[f for f in facs if f is not the_sum])
        inner = _normalize_one_integral(
            sp.Integral(others * the_sum.function, (v, lo, hi)))
        return sp.Sum(inner, *the_sum.limits), True
    # A commuting ``∂_w`` (non-``v`` var, ``v``-DEPENDENT argument, the rest
    # ``w``-independent) commutes OUT: ``∫_v c·∂_w F dv = ∂_w ∫_v c·F dv``.
    # A derivative of a ``v``-INDEPENDENT factor (``∂_x a_j``) is not a flux —
    # it is hoisted as a plain constant below, never commuted.
    deriv = next((f for f in facs
                  if isinstance(f, sp.Derivative)
                  and v not in f.variables and f.args[0].has(v)), None)
    if deriv is not None:
        rest = sp.Mul(*[f for f in facs if f is not deriv])
        if not rest.has(*deriv.variables):
            inner = _normalize_one_integral(
                sp.Integral(rest * deriv.expr, (v, lo, hi)))
            return sp.Derivative(inner, *deriv.variables), True
    # Hoist the ``v``-independent constant factor out.
    const, dep = (piece.as_independent(v, as_Add=False)
                  if isinstance(piece, sp.Mul) else (sp.S.One, piece))
    return const * sp.Integral(dep, (v, lo, hi)), const != sp.S.One


def _normalize_one_integral(integ):
    """Pull every bound-variable-INDEPENDENT factor / independent ``Σ`` /
    commuting ``∂_w`` OUT of one ``∫_v`` (var-agnostic: reads the integral's own
    bound variable), leaving a single ``v``-dependent body under each ``∫_v``.
    No basis naming."""
    limits = integ.args[1]
    if len(limits) != 3:
        return integ
    v, lo, hi = limits
    integrand = sp.expand(_dist_deriv_over_sum(integ.args[0]))
    out, changed = sp.S.Zero, False
    for piece in sp.Add.make_args(integrand):
        new, did = _pull_piece(piece, v, lo, hi)
        out += new
        changed = changed or did
    return out if changed else integ


def pull_out(expr):
    """Pull every factor INDEPENDENT of a binding operator's variable OUT of that
    operator — the linearity normalisation shared by ``∂`` and ``∫``:

        ∂_v(c·f) → c·∂_v f          ∫_v c·f dv  → c·∫_v f dv        (c free of v)
        ∂_v(Σ_j) → Σ_j ∂_v          ∫_v Σ_j  dv → Σ_j ∫_v dv        (j ⊥ v)
                                    ∫_v ∂_w g dv → ∂_w ∫_v g dv     (w ⊥ v)

    "Constant" means INDEPENDENT OF THE OPERATOR'S VARIABLE.  It is the GENTLE
    normalisation: it NEVER applies the product rule to a ``v``-dependent product
    (so it can neither split ``∂_v(a·b)`` nor undo a Leibniz fold), which is what
    keeps it from competing with :func:`~zoomy_core.model.derivation.closure.consolidate`.
    Applied bottom-up to a fixpoint."""
    from .closure import pull_consts

    def _walk(e):
        if hasattr(e, "args") and e.args:
            try:
                e = e.func(*(_walk(a) for a in e.args))
            except (TypeError, ValueError):
                return e
        if isinstance(e, sp.Integral):
            return _normalize_one_integral(e)
        return e

    prev = None
    while expr != prev:
        prev = expr
        expr = _dist_deriv_over_sum(expr)   # ∂_v Σ → Σ ∂_v  (coeff pulled out)
        expr = pull_consts(expr)            # ∂_v(c·f) → c·∂_v f
        expr = _walk(expr)                  # ∫ normalisation, bottom-up
    return expr


class PullConstants(Operation):
    """Pull every factor INDEPENDENT of a binding operator's variable OUT of that
    operator — for ``∫`` AND ``∂`` alike (see :func:`pull_out`).

    This is the dedicated "factor-out" step the bracket pipeline runs BEFORE
    :class:`ExtractBrackets`: coefficients ``a_j(t,x)``, mode-sums ``Σ_j`` and
    coefficient derivatives ``∂_x a_j`` are lifted out of every ζ-integral, so
    that what is left under an ``∫_ζ`` is purely ζ-dependent — only then is it
    legitimate to NAME it a bracket (a pure number).  It is basis-agnostic
    (knows nothing of ``φ``/``c``) and gentle (never splits a ζ-dependent
    product), so it neither competes with the Leibniz folder
    (:class:`~zoomy_core.model.derivation.closure.Consolidate`) nor hard-codes ζ."""

    whole_leaf_op = True

    def __init__(self, name="pull_constants"):
        super().__init__(
            name=name,
            description="pull operator-independent factors out of ∫ / ∂")

    def _leaf_sp(self, sp_expr):
        return pull_out(sp_expr)


# ── ExtractBrackets (naming only — assumes a PullConstants-normalised input) ─


class ExtractBrackets(Operation):
    """Replace ``Integral(φ-product, (ζ, 0, 1))`` with a named bracket atom or
    an explicit ⟨…⟩ Integral, pulling ζ-independent factors out first.

    The UNIVERSAL rule (no Mixed/Triple pattern dance):

    * a ζ-INDEPENDENT factor comes OUT as ``const``;
    * the ζ-DEPENDENT body stays inside;
    * ONLY two shapes keep a closed form for symbolic ``l`` (orthogonal
      bases): ``∫ φ_i·c·φ_l dζ → Gram(i, l)`` and ``∫ c·φ_l dζ → Weight(l)``;
    * every other ζ-dependent shape stays as ``Integral(body, (ζ, 0, 1))``,
      rendered ``⟨body⟩`` by the strip-args printer, for ``ResolveBasis`` to
      substitute the concrete basis and integrate;
    * a non-ζ ``∂_x`` derivative wrapping the φ-product commutes OUT of the
      integral (conservative flux): ``∫ c·φ_l·∂_x F dζ = ∂_x ∫ c·φ_l·F dζ``.

    Parameters
    ----------
    basis : Basis
        The opaque basis whose ``phi`` head is recognised.
    var : sympy.Symbol
        ζ integration variable (default ``Symbol("zeta")``).
    weight_name : str, optional
        Name of the weight Function ``c`` (default read off the basis).
    """

    whole_leaf_op = True

    def __init__(self, basis, var=None, weight_name=None,
                 name="extract_brackets"):
        super().__init__(name=name, description="extract basis brackets")
        self._basis = basis
        self._phi = getattr(basis, "phi", None) or getattr(basis, "phi_fn", basis)
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        if weight_name is None:
            weight_name = getattr(basis, "weight_name", "c")
        self._weight_name = weight_name

    def _phi_indices(self, factors):
        """φ-indices among ``factors`` plus weight / ∂-weight flags."""
        idx, rest = [], []
        has_weight = has_dweight = False
        for a in factors:
            if isinstance(a, sp.Function) and a.func is self._phi:
                idx.append(a.args[0])
            elif (isinstance(a, sp.Pow) and isinstance(a.base, sp.Function)
                  and a.base.func is self._phi and a.exp.is_Integer
                  and a.exp > 0):
                idx.extend([a.base.args[0]] * int(a.exp))
            elif (isinstance(a, sp.Function)
                  and a.func.__name__ == self._weight_name):
                has_weight = True
            elif isinstance(a, sp.Derivative):
                has_dweight = True
                rest.append(a)
            else:
                rest.append(a)
        return idx, rest, has_weight, has_dweight

    def _to_bracket(self, integrand, v=None, lo=sp.S.Zero, hi=sp.S.One):
        """Generic split: const out, ⟨body⟩ stays, Gram/Weight closed form.

        Returns the rewritten piece, or ``None`` when there is nothing for
        EXTRACT to claim (a pure non-φ term) so the caller keeps the original
        integral as-is.  The opaque bracket keeps the integral's OWN ``(lo, hi)``
        bounds; the closed-form ``Gram``/``Weight`` names are emitted only over
        the canonical reference interval ``(0, 1)`` (their δ-form assumes it).
        """
        if v is None:
            v = self._var
        const, dep = (sp.S.One, integrand)
        if isinstance(integrand, sp.Mul):
            const, dep = integrand.as_independent(v, as_Add=False)
        factors = list(dep.args) if isinstance(dep, sp.Mul) else [dep]
        idx, rest, has_w, has_dw = self._phi_indices(factors)
        canonical = (lo == 0 and hi == 1)
        if canonical and not has_dw:
            n = len(idx)
            # ⟨c, φ_l⟩ — closed form ``δ_{l,0}`` (orthogonal basis).
            if n == 1 and has_w and len(rest) == 0:
                return const * Weight(idx[0])
            # ⟨φ_i, c φ_l⟩ — closed form ``δ_{il}/(2l+1)``.
            if n == 2 and has_w and len(rest) == 0:
                return const * Gram(idx[0], idx[1])
        # A bracket is a PURE NUMBER: claim the integral as an opaque ⟨…⟩ ONLY
        # when its body depends on nothing but ``v`` (a ``t``/``x``-bearing body
        # is NOT a bracket — leave it a plain ``∫`` for a prior PullConstants to
        # finish hoisting).  The ``const`` prefactor stays outside.
        if not is_bracket_body(dep, v):
            return None
        return const * sp.Integral(dep, (v, lo, hi))

    def _name_one_integral(self, integ):
        """NAME one ``∫_v`` — assuming a :class:`PullConstants`-normalised input
        (no ``v``-independent factors / sums left inside).  Each additive term's
        ζ-body is matched by HEAD to ``Gram`` / ``Weight`` (over the canonical
        ``(0,1)``) or kept as an explicit ⟨…⟩ Integral over its own bounds.  A
        non-φ term is left as a plain Integral.  Var-agnostic (reads the
        integral's OWN bound variable, so the canonicalised ``\\hat ζ`` dummy and
        a running ``∫_0^ζ`` are handled the same way)."""
        limits = integ.args[1]
        if len(limits) != 3:
            return integ
        v, lo, hi = limits
        out, changed = sp.S.Zero, False
        for piece in sp.Add.make_args(sp.expand(integ.args[0])):
            b = self._to_bracket(piece, v, lo, hi)
            if b is None:
                out += sp.Integral(piece, limits)
            else:
                out += b
                changed = True
        return out if changed else integ

    def _leaf_sp(self, expr):
        # SHARP gate: first run the general :func:`pull_out` (PullConstants) so
        # every ζ-independent factor / sum is hoisted out — only THEN is a
        # ``∫_0^1`` over a purely ζ-dependent body, which :meth:`_to_bracket`
        # may name.  Composing the normaliser keeps a lone ``ExtractBrackets``
        # correct on its own, while the push-out logic still lives in one place.
        expr = pull_out(expr)

        def _walk(e):
            if hasattr(e, "args") and e.args:
                try:
                    e = e.func(*(_walk(a) for a in e.args))
                except (TypeError, ValueError):
                    return e
            if isinstance(e, sp.Integral):
                return self._name_one_integral(e)
            return e
        return _walk(expr)


# ── ResolveBasis (resolve every Galerkin bracket to a number) ──────────────


class ResolveBasis(Operation):
    """Resolve EVERY Galerkin bracket in the equation to a NUMBER against a
    CONCRETE basis — a thin op over
    :meth:`~zoomy_core.model.models.basisfunctions.Basisfunction.resolve` (fast
    antiderivative + per-instance cache; named ``Gram``/``Weight`` close by their
    orthogonality forms, opaque ``⟨…⟩`` and the nested ω-coupling integrals by
    polynomial evaluation, loose ``φ_i(0)``/``c(0)`` boundary terms concretised).

    Apply it AFTER :class:`~zoomy_core.model.derivation.model.ResolveModes` has
    specialised the abstract test index to a concrete moment (and the modal sums
    are unrolled), so every bracket index is an integer::

        legendre = Legendre_shifted(level=N)
        m.momentum.x.apply(ResolveModes(index=k, modes=range(N + 1)))
        m.momentum.apply(ResolveBasis(legendre))
    """

    whole_leaf_op = True

    def __init__(self, basis, var=None, name="resolve_basis"):
        self._basis = basis
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        super().__init__(
            name=name,
            description=f"resolve brackets via {getattr(basis, 'name', basis)}")

    def _leaf_sp(self, sp_expr):
        return self._basis.resolve(sp_expr, self._var)
