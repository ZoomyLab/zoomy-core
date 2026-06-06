"""Modal projection operations for the clean-redesign framework.

After :func:`~zoomy_core.derivation.modal.separation_of_variables` has put the
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

from zoomy_core.model.operations import Operation, Multiply


__all__ = [
    "Gram",
    "Weight",
    "bracket_atoms",
    "ExpandSums",
    "EvaluateSums",
    "Integrate",
    "Project",
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


def _fresh_index(idx, used):
    """A fresh integer, non-negative dummy named after ``idx`` not in ``used``."""
    base = idx.name if isinstance(idx, sp.Symbol) else "i"
    n = 1
    while True:
        cand = sp.Symbol(f"{base}_{n}", integer=True, nonnegative=True)
        if cand not in used:
            return cand
        n += 1


def _relabel_sum(the_sum, used):
    repl, new_limits = {}, []
    for lim in the_sum.limits:
        idx = lim[0]
        new = _fresh_index(idx, used)
        used.add(new)
        repl[idx] = new
        new_limits.append((new,) + tuple(lim[1:]))
    return the_sum.function.xreplace(repl), new_limits


def _merge_sums(sums):
    """Merge ``sp.Sum`` factors into ONE multi-index Sum with DISTINCT dummies
    (so cross terms ``i ≠ j`` survive)."""
    used = set()
    for s in sums:
        used |= set(s.free_symbols)
    summands, limits = [], []
    for s in sums:
        fn, lims = _relabel_sum(s, used)
        summands.append(fn)
        limits.extend(lims)
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

    The modal ansatz from :func:`~zoomy_core.derivation.modal.separation_of_variables`
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
        return sp_expr.replace(lambda e: isinstance(e, sp.Sum),
                               lambda e: e.doit())


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
      :class:`~zoomy_core.derivation.closure.ResolveIntegral` to close by basis.

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
    """Definite integral over ``(var, lo, hi)`` that pushes the integral THROUGH
    any unexpanded ``sp.Sum`` and leaves the opaque ``φ``-integrals (and any
    other ``var``-dependent body) as **abstract** ``sp.Integral`` atoms — for a
    later :class:`~zoomy_core.derivation.closure.ResolveIntegral` to evaluate by
    a chosen method (``basis`` / ``ftc`` / ``numerical``).

    A ``var``-independent ``∂_x`` derivative commutes OUT of the integral
    (conservative flux).  This is the "slim" projection integral — it does NOT
    evaluate the bracket; resolution is a separate, method-selectable step.
    :class:`Project` is just ``Multiply(test)`` then this ``Integrate``.
    """

    whole_leaf_op = True

    def __init__(self, var=None, bounds=(0, 1), name="integrate"):
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._bounds = tuple(bounds)
        super().__init__(
            name=name,
            description=f"∫ d{self._var} over {self._bounds} (abstract)")

    def _leaf_sp(self, sp_expr):
        return _integrate_through_sum(sp_expr, self._var,
                                      self._bounds[0], self._bounds[1])


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


# ── ExtractBrackets (generic split) ────────────────────────────────────────


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

    def _to_bracket(self, integrand):
        """Generic split: const out, ⟨body⟩ stays, Gram/Weight closed form.

        Returns the rewritten piece, or ``None`` when there is nothing for
        EXTRACT to claim (a pure non-φ term) so the caller keeps the original
        integral as-is.
        """
        v = self._var
        const, dep = (sp.S.One, integrand)
        if isinstance(integrand, sp.Mul):
            const, dep = integrand.as_independent(v, as_Add=False)
        factors = list(dep.args) if isinstance(dep, sp.Mul) else [dep]
        idx, rest, has_w, has_dw = self._phi_indices(factors)
        if not has_dw:
            n = len(idx)
            # ⟨c, φ_l⟩ — closed form ``δ_{l,0}`` (orthogonal basis).
            if n == 1 and has_w and len(rest) == 0:
                return const * Weight(idx[0])
            # ⟨φ_i, c φ_l⟩ — closed form ``δ_{il}/(2l+1)``.
            if n == 2 and has_w and len(rest) == 0:
                return const * Gram(idx[0], idx[1])
        # A pure non-φ piece — nothing to claim.
        if not idx and not has_w and not has_dw:
            return None
        # Everything else: leave the ζ-DEPENDENT body as an explicit Integral
        # but HOIST the ζ-independent ``const`` prefactor back outside it.
        return const * sp.Integral(dep, (v, sp.S.Zero, sp.S.One))

    def _pull_outer_derivative(self, piece, limits):
        """``c·φ_l·∂_w(F)`` (∂_w in a non-ζ var) → ``∂_w(∫ c·φ_l·F dζ)`` — the
        derivative commutes out of the ζ-integral.  Returns the rewritten
        expression, or ``None`` when there is no such outer derivative."""
        v = self._var
        factors = list(piece.args) if isinstance(piece, sp.Mul) else [piece]
        deriv = next((f for f in factors
                      if isinstance(f, sp.Derivative)
                      and v not in f.variables), None)
        if deriv is None:
            return None
        rest = [f for f in factors if f is not deriv]
        rest_prod = sp.Mul(*rest) if rest else sp.S.One
        inner = deriv.args[0]
        wrt = deriv.variables
        new_integrand = rest_prod * inner
        inner_extracted = self._extract_one_integral(
            sp.Integral(new_integrand, limits), _allow_pull=False)
        return sp.Derivative(inner_extracted, *wrt)

    def _extract_one_integral(self, integ, _allow_pull=True):
        v = self._var
        limits = integ.args[1]
        if len(limits) != 3 or limits[0] != v:
            return integ
        _, lo, hi = limits
        if not (lo == 0 and hi == 1):
            return integ
        integrand = sp.expand(integ.args[0])
        out = sp.S.Zero
        changed = False
        for piece in sp.Add.make_args(integrand):
            if _allow_pull:
                pulled = self._pull_outer_derivative(piece, limits)
                if pulled is not None:
                    out += pulled
                    changed = True
                    continue
            b = self._to_bracket(piece)
            if b is None:
                out += sp.Integral(piece, limits)
            else:
                out += b
                changed = True
        return out if changed else integ

    def _leaf_sp(self, expr):
        def _walk(e):
            if isinstance(e, sp.Integral):
                return self._extract_one_integral(e)
            if hasattr(e, "args") and e.args:
                try:
                    return e.func(*(_walk(a) for a in e.args))
                except (TypeError, ValueError):
                    return e
            return e
        return _walk(expr)


# ── ResolveBasis (per-field bracket resolution) ────────────────────────────


def _resolve_weight(basis, weight):
    """Pick the test weight ``c(ζ)`` — explicit ``weight`` wins, else the
    basis's own ``weight`` (1 for Legendre, …)."""
    if weight is None:
        return getattr(basis, "weight")
    if callable(weight):
        return weight
    return lambda _z: weight


class ResolveBasis(Operation):
    """Per-field bracket resolution: substitute the concrete basis and
    integrate.

    Looks up the field's trial index (preferred: an index Symbol from
    ``model.modal_index(u)``) and, for every named bracket OR leftover
    ``∫_0^1 … dζ`` carrying THIS index:

    1. force-expand bounded Sums so concrete-index integrands surface;
    2. substitute each ``Gram`` / ``Weight`` atom by the basis's
       ``closed_form_bracket`` (δ-form, ``l`` kept symbolic);
    3. concretize the basis polynomial + integrate every leftover ζ-integral;
    4. ``.doit()`` collapses the δ-Sums analytically (``Σ_i a_i δ_{il}/(2l+1)
       → a_l/(2l+1)`` for ``N ≥ l``).

    Different fields can use different bases by chaining one ``ResolveBasis``
    per field.

    Parameters
    ----------
    target : sympy.Symbol | Function | applied
        The trial index Symbol (preferred — ``model.modal_index(u)``), or a
        field / coefficient that resolves via ``model.modal_index`` when
        ``model`` is given.
    basis_cls : type
        The concrete basis class (e.g. ``Legendre_shifted``).
    level : int, optional
        Basis level, passed as ``basis_cls(level=level)``.
    model : Model, optional
        Required if ``target`` is not already an index Symbol.
    var : sympy.Symbol
        ζ integration variable (default ``Symbol("zeta")``).
    weight, weight_name : passed through to the basis weight resolution.
    """

    whole_leaf_op = True

    def __init__(self, target, basis_cls, level=None, *, model=None,
                 var=None, weight=None, weight_name=None,
                 name="resolve_basis"):
        if isinstance(target, sp.Symbol):
            self._index = target
        elif model is not None:
            self._index = model.modal_index(target)
        else:
            raise TypeError(
                "ResolveBasis: target must be an index Symbol, or a field / "
                "coefficient with `model=` so the index can be looked up via "
                "`model.modal_index(target)`."
            )
        # ``level`` may be a SYMBOLIC bound (``N_u``) when only the
        # closed-form bracket path (Gram/Weight → δ-form, ``l`` symbolic) is
        # needed — that resolution is level-independent.  Instantiate the
        # basis at a concrete level for ``closed_form_bracket`` / ``weight`` /
        # ``resolve_atoms`` access: use the integer level when given, else 0
        # (the closed-form table and weight do not depend on the truncation).
        concrete_level = level if (isinstance(level, int)) else 0
        self._basis = basis_cls(level=concrete_level)
        self._level = level
        self._symbolic_level = not isinstance(level, int) and level is not None
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._weight = weight
        self._weight_name = (weight_name if weight_name is not None
                             else getattr(self._basis, "weight_name", "c"))
        bname = getattr(basis_cls, "__name__", str(basis_cls))
        lvl_s = f" level {level}" if level is not None else ""
        super().__init__(
            name=name,
            description=(f"resolve trial index {self._index} via "
                         f"{bname}{lvl_s}"),
        )

    def _trial_indices(self, atom):
        # Trial slots for the named brackets ExtractBrackets emits.
        slots = {"Gram": (0,), "Weight": ()}.get(atom.func.__name__)
        if slots is not None:
            return {atom.args[s] for s in slots}
        l_sym = sp.Symbol("l", integer=True, nonnegative=True)
        return {a for a in atom.args if a != l_sym}

    def _resolve_named(self, expr):
        subs = {}
        for atom in bracket_atoms(expr):
            trial = self._trial_indices(atom)
            if not trial or self._index not in trial:
                continue
            cf = self._basis.closed_form_bracket(
                atom.func.__name__, tuple(atom.args))
            if cf is not None:
                subs[atom] = cf
        return expr.xreplace(subs)

    def _resolve_integrals(self, expr):
        """Substitute the basis polynomial + integrate every leftover
        ``∫_0^1 … dζ`` whose integrand involves ``self._index``."""
        z = sp.Symbol("z")
        wfn = _resolve_weight(self._basis, self._weight)
        cname = self._weight_name
        idx = self._index

        def _resolve_one(integ):
            limits = integ.limits[0]
            if (len(limits) != 3
                    or limits[0] != self._var
                    or limits[1] != sp.S.Zero or limits[2] != sp.S.One):
                return integ
            body = sp.sympify(integ.function)
            if idx not in body.free_symbols:
                return integ
            body = body.xreplace({self._var: z})
            body = sp.sympify(body.replace(
                lambda e: (isinstance(e, sp.Function)
                           and e.func.__name__ == cname
                           and len(e.args) == 1),
                lambda e: wfn(e.args[0]),
            ))
            poly = sp.sympify(self._basis.resolve_atoms(body)).doit()
            return sp.integrate(sp.expand(poly), (z, 0, 1))

        return expr.replace(lambda e: isinstance(e, sp.Integral), _resolve_one)

    def _leaf_sp(self, sp_expr):
        # 1. Closed-form-resolve named brackets (Gram / Weight → δ-form) FIRST,
        #    so the δ inside an abstract-N Sum collapses under .doit() with the
        #    test index ``l`` kept symbolic.  This path is level-independent.
        expr = self._resolve_named(sp_expr)
        # 2. With a concrete level, force-expand bounded Sums so concrete-index
        #    integrands surface, then concretize the basis polynomial +
        #    integrate every leftover ζ-integral.  With a SYMBOLIC bound the
        #    Sum cannot be expanded and the leftover ⟨…⟩ integrals stay opaque
        #    (only the orthogonality brackets close).
        if not self._symbolic_level:
            expr = expr.replace(lambda e: isinstance(e, sp.Sum),
                                lambda e: e.doit())
            expr = self._resolve_integrals(expr)
        # 3. .doit() collapses any δ-Sums produced by step 1.
        return expr.doit(deep=True)
