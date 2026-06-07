"""Milestone-4 closure operations for the clean-redesign framework.

After :class:`~zoomy_core.derivation.modal.separation_of_variables` has put
every field into its unexpanded modal ansatz and the abstract truncation bound
``N_u`` has been bound to a concrete integer (so the finite ``sp.Sum`` expands),
these ops carry the SME derivation to the Galerkin-projected, basis-resolved,
KBC-closed K&T (4.17) form.

Three pieces live here:

:class:`Resolve`
    The concrete-level Galerkin projection-and-resolve.  Per row it multiplies
    by the test weight ``c·φ_l`` (with ``l`` already bound to the row's mode),
    substitutes every opaque ``φ(k, ζ)`` of each field's basis by the concrete
    polynomial ``basis.eval(k, ζ)`` (and the weight ``c`` → the basis weight),
    then integrates over ``(ζ, 0, 1)`` **in place** — the ``∂_x`` derivative
    stays INSIDE the integral.

    This is the crucial distinction from the symbolic-level
    :class:`~zoomy_core.derivation.projection.ExtractBrackets` /
    :class:`~zoomy_core.derivation.projection.ResolveBasis` pair, which pull a
    conservative ``∂_x`` OUT of the ζ-integral before resolving.  That pull-out
    is correct for the orthogonal flux brackets but corrupts the σ-metric mass
    closure: the ζ-dependent metric factor ``∂_x(ζh+b)`` left inside the
    pulled-out ``∂_x(…)`` leaves a spurious ``b``-coefficient residue that no
    kinematic-BC substitution can cancel.  Integrating in place keeps the
    metric terms ζ-local, so the fundamental-theorem boundary trace and the
    advection metric terms combine exactly — the mass row closes bit-for-bit to
    ``∂_t h + ∂_x(h·a_0)`` after the KBC absorption.

:func:`kinematic_modal_closure`
    Build the ``Substitution`` that closes the lower w-modes (``aw_0, aw_1``)
    from the two kinematic boundary conditions, via ``sp.solve`` on the modal
    evaluations of the surface (ζ=1) and bed (ζ=0) relations — the proven
    ``mlme.py`` / production ``_kbc_modal_closure`` pattern, lifted to the
    declarative :class:`Model`.

:func:`mass_relation`
    The ``Substitution`` ``∂_t h → −∂_x(h·a_0)`` built from the depth-averaged
    mass row, used to cancel the stray ``∂_t h`` the σ-metric chain-rule injects
    into the momentum rows.

:class:`Simplify`
    A thin whole-equation op wrapping ``Equation.simplify`` so a derivation can
    record a canonicalisation step in the graph.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core import coords as _coords
from zoomy_core.model.operations import Operation, Multiply


__all__ = [
    "Resolve",
    "ResolveIntegral",
    "InvertMassMatrix",
    "FoldConservative",
    "Split",
    "Simplify",
    "AutoTag",
    "SortByTag",
    "Sort",
    "TAG_ORDER",
    "fold_to_conservative_form",
    "is_conservative_diffusion",
    "project_conservative_diffusion",
]


class Split(Operation):
    """Fold a SELF-PRODUCT flux term ``c·f·∂_x f`` into the conservative form
    ``∂_x(c·f²/2)`` by the HALF product rule.

    A plain inverse :class:`~zoomy_core.model.operations.ProductRule` on
    ``g·h·∂_x h`` only *recombines* it (``∂_x(g h²) − g·h·∂_x h = g·h·∂_x h``,
    value-preserving) because the field ``h`` appears both as coefficient and
    under the derivative.  The trick is to SPLIT the term in two and apply the
    product rule to ONE half — whose residual exactly cancels the other half:

    .. math::

        g h\\,∂_x h
          = \\tfrac12 g h\\,∂_x h + \\tfrac12 g h\\,∂_x h
          \\;\\xrightarrow{\\text{PR on 1st half}}\\;
          \\tfrac12\\big(∂_x(g h^2) - g h\\,∂_x h\\big) + \\tfrac12 g h\\,∂_x h
          = ∂_x\\!\\big(\\tfrac{g h^2}{2}\\big).

    So ``Split = ½·I + ½·ProductRule⁻¹`` — value-preserving, and the natural way
    to recover the hydrostatic pressure ``∂_x(g h²/2)`` from ``g·h·∂_x h``
    WITHOUT a post-hoc fold.  Apply it (via ``term[[i]].apply(Split())``) to the
    self-product terms only."""

    whole_leaf_op = True

    def __init__(self, variables=None, name="split"):
        self._vars = variables
        desc = ("half product rule  c·f·∂f → ∂(c·f²/2)"
                + (f" on {{{', '.join(str(v) for v in variables)}}}"
                   if variables is not None else ""))
        super().__init__(name=name, description=desc)

    def _leaf_sp(self, sp_expr):
        from zoomy_core.model.operations import ProductRule
        pr = ProductRule(variables=self._vars, direction="inverse")
        return sp.Rational(1, 2) * sp_expr \
            + sp.Rational(1, 2) * pr._leaf_sp(sp_expr)


class InvertMassMatrix(Operation):
    """Invert the Galerkin mass matrix by reading it off the equation.

    After a moment row is projected and resolved it has the shape
    ``M_kk·∂_t Q_k + (flux/source) = 0``, where ``M_kk`` — the (diagonal)
    Galerkin mass-matrix entry ``⟨φ_k, c φ_k⟩`` (``1/(2k+1)`` for shifted
    Legendre) — is EXACTLY the coefficient of the row's ``∂_t`` term.  This op
    finds that ``∂_t`` term, divides the whole row by its coefficient, and so
    normalises ``∂_t Q_k`` to unit coefficient — the "invert the mass matrix"
    step, with the matrix read from the model rather than a passed basis.  For a
    non-orthogonal basis the same idea generalises to a per-row linear solve;
    for the diagonal SME it is a single division (``×(2k+1)``).

    Apply it AFTER the stray ``∂_t h`` has been removed (``mass_relation``) so
    the row carries the single diagonal ``∂_t`` term."""

    whole_leaf_op = True

    def __init__(self, time=None, name="invert_mass_matrix"):
        self._t = time if time is not None else _coords.t
        super().__init__(
            name=name,
            description="divide each row by its ∂_t coefficient (mass matrix)")

    def _leaf_sp(self, sp_expr):
        t = self._t
        expr = sp.expand(sp_expr)
        mass = None
        for term in sp.Add.make_args(expr):
            dts = [D for D in term.atoms(sp.Derivative)
                   if D.variables == (t,)]
            if len(dts) == 1:
                cand = sp.simplify(term / dts[0])
                # The mass-matrix diagonal is the PURELY NUMERIC Legendre factor
                # ``1/(2k+1)`` multiplying the conserved ``∂_t Q_k``.  Apply this
                # op AFTER the change of variables ``a→q/h``: the conservative
                # ``(1/(2k+1))∂_t(h a_k)`` then reads ``(1/(2k+1))∂_t q_k`` with a
                # t-FREE coefficient, while any leftover surface-trace ``∂_t h``
                # keeps a t-bearing coefficient ``(Σq_k)/h`` — so the t-free guard
                # selects exactly the mass-matrix term and ignores the traces.
                if cand != 0 and not cand.has(t):
                    mass = cand
                    break
        if mass in (None, sp.S.Zero, sp.S.One):
            return sp_expr
        return sp.expand(expr / mass)


# ── conservative fold (flux/pressure bundling, NCP-preserving) ─────────────


def _antiderivative_in(coeff, var, field_to_sym):
    """``∫ coeff d(var)`` treating every applied field in ``field_to_sym`` as
    an independent symbol — used to recover a flux/pressure potential ``F``
    with ``∂F/∂var = coeff``.

    ``var`` is the applied Function being integrated against (``h(t, x)``);
    ``field_to_sym`` maps every applied field (incl. ``var``) to a fresh bare
    Symbol so ``sp.integrate`` sees a clean rational/polynomial integrand.
    """
    sym_of_var = field_to_sym[var]
    inv = {s: f for f, s in field_to_sym.items()}
    integrand = coeff.xreplace(field_to_sym)
    F = sp.integrate(integrand, sym_of_var)
    return sp.expand(F.xreplace(inv))


def fold_to_conservative_form(expr, flux_fields, *, h, b, x, gravity_param):
    """Fold the spatial part of a momentum/continuity residual into the
    production K&T (4.17) decomposition: conservative flux + hydrostatic
    pressure bundles, with the genuinely non-conservative couplings LEFT
    UNFOLDED.

    The Resolve / CoV chain leaves the spatial part fully expanded
    (``2 q_k/h·∂_x q_k − q_k²/h²·∂_x h + …``).  This op re-bundles it EXACTLY
    as production does:

    * the bed coupling ``g·h·∂_x b`` stays an explicit ``coeff·∂_x b`` term
      (→ ``nonconservative_matrix[row, b]``);
    * the ``∂_x h`` coefficient is integrated in ``h`` to recover the
      conservative bundle; its gravity part (the ``∂_x(g·h²/2)`` pressure)
      is split into a SEPARATE ``∂_x(P)`` term so the downstream classifier
      routes it to ``hydrostatic_pressure``, and the advective part becomes
      the flux bundle ``∂_x(F)``;
    * whatever ``∂_x q_j`` coupling remains after subtracting ``∂_x(F + P)``
      is left as explicit ``coeff·∂_x q_j`` NCP terms
      (→ ``nonconservative_matrix[row, q_j]``).

    This reproduces production bit-exact, e.g.::

        g·h·∂_x b + ∂_x(g·h²/2) + ∂_x(q_0²/h + q_1²/3h + q_2²/5h) + ∂_t q_0

    Parameters
    ----------
    expr : sympy.Expr
        The residual (one momentum / continuity row), Function form.
    flux_fields : list of applied Functions
        The conserved flux variables (the modal ``q(k, t, x)``).
    h, b : applied Functions
        Depth and bed.
    x : sympy.Symbol
        The spatial coordinate.
    gravity_param : sympy.Symbol or None
        The gravity parameter ``g`` — its ``∂_x h`` contribution is the
        hydrostatic pressure; ``None`` ⇒ no pressure split.
    """
    expr = sp.expand(expr.doit())
    flux_fields = list(flux_fields)

    def dx(f):
        return sp.Derivative(f, x)

    # Fresh bare symbols for the antiderivative-in-h integration.
    field_to_sym = {h: sp.Symbol("_H_fold", positive=True)}
    for i, f in enumerate(flux_fields):
        field_to_sym[f] = sp.Symbol(f"_Qf{i}", real=True)

    # Split the residual into non-spatial (time / source) and spatial terms.
    nonspatial, spatial = [], []
    for term in sp.Add.make_args(expr):
        if any(d.variables == (x,) for d in term.atoms(sp.Derivative)):
            spatial.append(term)
        else:
            nonspatial.append(term)
    spatial = sp.expand(sp.Add(*spatial))

    # 1. Bed coupling g·h·∂_x b stays an explicit NCP term.
    cb = sp.expand(spatial.coeff(dx(b)))
    bed = cb * dx(b)
    spatial = sp.expand(spatial - bed)

    # 2. The ∂_x h coefficient integrates in h to the conservative bundle.
    ch = sp.expand(spatial.coeff(dx(h)))
    if gravity_param is not None:
        ch_grav = sp.expand(ch.coeff(gravity_param)) * gravity_param
    else:
        ch_grav = sp.S.Zero
    ch_flux = sp.expand(ch - ch_grav)
    P = _antiderivative_in(ch_grav, h, field_to_sym) if ch_grav != 0 \
        else sp.S.Zero
    F = _antiderivative_in(ch_flux, h, field_to_sym) if ch_flux != 0 \
        else sp.S.Zero

    # 3. The ∂_x q_j couplings left after subtracting ∂_x(F + P) are NCP.
    dFP = sp.expand(sp.Derivative(P + F, x).doit())
    ncp = sp.expand(spatial - dFP)

    out = sp.Add(*nonspatial) + bed + ncp
    if P != 0:
        out += sp.Derivative(P, x)
    if F != 0:
        out += sp.Derivative(F, x)
    return out


class FoldConservative(Operation):
    """Re-bundle a momentum row's spatial part into the K&T (4.17) conservative
    decomposition, as a tracked ``.apply`` op (the loop-free form of pipeline
    step 8b).

    Per row: peel off the conservative DIFFUSION atoms ``∂_x(D·∂_x q)``
    (:func:`is_conservative_diffusion`) and leave them untouched (the
    SystemModel types them as ``diffusion_matrix``); ``.doit()`` + fold the
    hyperbolic remainder via :func:`fold_to_conservative_form` (flux + pressure
    → ``∂_x(F)`` / ``∂_x(g h²/2)``, bed/cross-mode couplings stay UNFOLDED as
    NCP); reattach the diffusion.

    Parameters mirror :func:`fold_to_conservative_form`: ``flux_fields`` (the
    conserved modal ``q(k,t,x)``), ``h``, ``b``, ``x``, and ``gravity``."""

    whole_leaf_op = True

    def __init__(self, flux_fields, *, h, b, x, gravity=None,
                 name="fold_conservative"):
        self._flux = list(flux_fields)
        self._h, self._b, self._x, self._g = h, b, x, gravity
        super().__init__(
            name=name, description="conservative flux/pressure fold")

    def _leaf_sp(self, sp_expr):
        x = self._x
        diff_terms, base_terms = [], []
        for term in sp.Add.make_args(sp_expr):
            (diff_terms if is_conservative_diffusion(term, x)
             else base_terms).append(term)
        base = sp.expand(sp.Add(*base_terms).doit())
        folded = fold_to_conservative_form(
            base, self._flux, h=self._h, b=self._b, x=x, gravity_param=self._g)
        return folded + sp.Add(*diff_terms)


# ── Resolve (concrete-level project + in-place ζ-integration) ──────────────


class Resolve(Operation):
    """Concrete-level Galerkin project-and-resolve, integrating in place.

    Apply AFTER the abstract bound ``N_u`` has been bound to a concrete integer
    and the finite ``sp.Sum`` ansatz expanded to explicit modes ``φ(0,ζ),
    φ(1,ζ), …``.  For the row's test mode ``l`` (already bound — e.g. via
    ``Substitution({l: 0})`` for the depth-mean row), this op:

    1. multiplies the residual by the test weight ``c(ζ)·φ(l, ζ)``;
    2. substitutes every opaque ``φ(k, ζ)`` of every registered field-basis by
       the concrete polynomial ``basis_cls.eval(int(k), ζ)`` and the opaque
       weight ``c(ζ)`` by the basis weight (``1`` for Legendre);
    3. integrates the whole product over ``(ζ, 0, 1)`` **in place** (the
       ``∂_x`` / ``∂_t`` derivatives stay inside).

    A ``Multiply(h)`` should already have cleared the ``1/h`` σ-Jacobian on the
    row; this op leaves the modal coefficients ``a(k, t, x)`` and any ``h`` /
    ``b`` factors untouched.

    Parameters
    ----------
    test_function : sympy.Expr
        The Galerkin test weight to multiply by — typically ``c(ζ)·φ(l, ζ)``
        with the opaque heads, ``l`` bound to the row's mode.
    basis_cls : type
        The concrete basis class providing ``.eval(k, ζ)`` and ``.weight``
        (e.g. ``Legendre_shifted``).
    level : int
        Concrete basis level for the polynomial evaluation.
    var : sympy.Symbol
        ζ integration variable (default ``Symbol("zeta")``).
    bounds : tuple[Expr, Expr]
        Integration interval (default ``(0, 1)``).
    """

    whole_leaf_op = True

    def __init__(self, test_function, basis_cls, level, *, var=None,
                 bounds=(0, 1), name="resolve"):
        self._test = sp.sympify(test_function)
        self._basis = basis_cls(level=level)
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._bounds = tuple(bounds)
        self._multiply = Multiply(self._test)
        # Concrete weight callable (1 for Legendre): the basis exposes
        # ``weight`` as a callable / value; resolve it once.
        w = getattr(self._basis, "weight", None)
        self._weight_value = w
        super().__init__(
            name=name,
            description=(f"resolve+integrate over {self._var} via "
                         f"{getattr(basis_cls, '__name__', basis_cls)}"),
        )

    def _resolve_opaque_phi(self, expr):
        """Substitute every opaque ``φ(k, ζ)`` (basis-machinery head, any
        basis) by the concrete polynomial, and the opaque weight ``c(ζ)`` by
        the basis weight.  Opaque heads carry ``_is_basis_head``."""
        z = self._var

        def _is_phi(a):
            return (isinstance(a, sp.Function)
                    and getattr(a.func, "_is_basis_head", False)
                    and len(a.args) == 2)

        def _is_weight(a):
            return (isinstance(a, sp.Function)
                    and getattr(a.func, "_is_basis_head", False)
                    and len(a.args) == 1)

        # weight c(ζ) → basis weight (callable → value at its arg, else value).
        wv = self._weight_value

        def _weight_at(arg):
            if callable(wv):
                return wv(arg)
            return sp.S.One if wv is None else wv

        expr = expr.replace(_is_weight, lambda a: _weight_at(a.args[0]))
        # φ(k, ζ) → eval(k, ζ) for concrete integer k.
        expr = expr.replace(
            _is_phi,
            lambda a: (self._basis.eval(int(a.args[0]), a.args[1])
                       if a.args[0].is_Integer else a),
        )
        return expr

    def _integrate_poly(self, expr):
        """``∫_lo^hi expr dζ`` for an integrand that is polynomial in ζ — split
        each additive term into the ζ-independent coefficient and the ζ-power
        and integrate ``ζ^n`` analytically.  Far cheaper than ``sp.integrate``
        on the rational (``1/h``) coefficients the modal flux carries; falls
        back to ``sp.integrate`` for any non-polynomial ζ-dependence."""
        z = self._var
        lo, hi = self._bounds
        out = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(expr)):
            coeff, dep = term.as_independent(z, as_Add=False)
            if dep == 1:
                out += coeff * (hi - lo)
            elif dep == z:
                out += coeff * (hi**2 - lo**2) / 2
            elif (isinstance(dep, sp.Pow) and dep.base == z
                  and dep.exp.is_Integer and dep.exp >= 0):
                n = int(dep.exp)
                out += coeff * (hi**(n + 1) - lo**(n + 1)) / (n + 1)
            else:
                out += sp.integrate(term, (z, lo, hi))
        return out

    def _leaf_sp(self, sp_expr):
        scaled = self._multiply._leaf_sp(sp_expr)
        resolved = self._resolve_opaque_phi(sp.expand(scaled))
        # ``.doit()`` collapses any concrete-index basis derivatives.
        resolved = resolved.doit()
        return self._integrate_poly(resolved)


# ── ResolveIntegral (decoupled, method-per-integral resolution) ────────────


class ResolveIntegral(Operation):
    """Resolve the unresolved ``∫…dζ`` integrals left by an ABSTRACT projection,
    each by a selectable METHOD — resolution decoupled from projection so it can
    differ per integral (and per term).

    Unlike :class:`Resolve` (which projects AND integrates in one welded,
    basis-only step), this op acts on an expression that already carries opaque
    ``sp.Integral(integrand, (ζ, 0, 1))`` atoms and rewrites EACH by:

      * ``"basis"``     — substitute the concrete basis polynomial for every
                          opaque ``φ(k, ζ)`` (and the weight ``c``) and integrate
                          the resulting ζ-polynomial analytically (δ-form);
      * ``"ftc"``       — fundamental theorem of calculus
                          ``∫_0^1 ∂_ζ g dζ → g|_{ζ=1} − g|_{ζ=0}`` (term-by-term
                          on the ζ-independent-coefficient parts);
      * ``"numerical"`` — ``sp.integrate`` quadrature fallback.

    ``method`` is the default; pass ``classify`` (a callable
    ``Integral → method | None``) to override per integral — e.g. resolve a
    ``∂_ζ`` flux by ``"ftc"`` and a φ-bracket by ``"basis"`` in the SAME row.
    """

    whole_leaf_op = True

    def __init__(self, basis_cls=None, *, var=None, method="auto", level=None,
                 weight=None, classify=None, bounds=(0, 1),
                 name="resolve_integral"):
        self._var = var if var is not None else sp.Symbol("zeta", real=True)
        self._method = method
        self._classify = classify
        self._basis = basis_cls(level=level) if basis_cls is not None else None
        self._weight_value = (getattr(self._basis, "weight", None)
                              if self._basis is not None else weight)
        self._bounds = tuple(bounds)
        super().__init__(
            name=name,
            description=f"resolve ∫d{self._var} (default method={method})")

    # ── per-integral method dispatch ─────────────────────────────────────
    def _method_for(self, integral):
        if self._classify is not None:
            chosen = self._classify(integral)
            if chosen:
                return chosen
        return self._method

    def _resolve_one(self, integral):
        integrand = integral.function
        # Resolve using the integral's OWN bound variable (var-agnostic): the
        # field-level KinematicBC alpha-renames bound ζ → fresh ``\hat{z}``
        # Dummies, so the integration variable that actually appears is read off
        # the integral itself rather than assumed to be ``self._var``.
        lim = integral.limits[0]
        var = lim[0]
        lo, hi = (lim[1], lim[2]) if len(lim) == 3 else self._bounds
        method = self._method_for(integral)
        if method == "auto":
            return self._auto_resolve(integrand, var, lo, hi)
        if method == "ftc":
            return self._ftc(integrand, var, lo, hi)
        if method == "basis":
            return self._resolve_basis(integrand, var, lo, hi)
        if method == "numerical":
            return sp.integrate(integrand, (var, lo, hi))
        raise ValueError(f"ResolveIntegral: unknown method {method!r}")

    # ── auto: classify each additive term by SHAPE (the "smart" resolution) ──
    def _auto_resolve(self, integrand, var, lo, hi):
        """Resolve ``∫_lo^hi integrand d(var)`` by classifying each additive term:
        a bare ``∂_var`` body → FTC; a bare ``∂_w`` (``w≠var``) with a
        ``w``-independent rest → the derivative COMMUTES OUT (conservative flux);
        a single ``Sum`` factor → push the integral inside; a φ-basis bracket →
        substitute the concrete basis and integrate the polynomial; anything
        else stays an opaque ``∫`` atom.  This is the only smart integrator —
        :class:`~zoomy_core.derivation.projection.Integrate` merely builds the
        abstract ``∫`` for it to act on."""
        out = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(integrand)):
            out += self._auto_term(term, var, lo, hi)
        return out

    def _auto_term(self, term, var, lo, hi):
        coeff, rest = term.as_independent(var, as_Add=False)
        if rest == 1:                                       # constant in var
            return coeff * (hi - lo)
        # a bare outer derivative as the whole var-dependent body
        if isinstance(rest, sp.Derivative) and len(rest.variables) == 1:
            dvar = rest.variables[0]
            g = rest.args[0]
            if dvar == var:                                 # FTC over var
                return coeff * (g.subs(var, hi) - g.subs(var, lo))
            return coeff * sp.Derivative(                   # conservative commute
                self._auto_resolve(g, var, lo, hi), dvar)
        factors = list(rest.args) if isinstance(rest, sp.Mul) else [rest]
        # a bare ``∂_w`` factor (w≠var) whose remaining factors+coeff are
        # w-independent commutes OUT of the integral (conservative flux).
        for f in factors:
            if (isinstance(f, sp.Derivative) and len(f.variables) == 1
                    and f.variables[0] != var):
                dw = f.variables[0]
                others = sp.Mul(*[g for g in factors if g is not f])
                if not (coeff * others).has(dw):
                    inner = self._auto_resolve(others * f.expr, var, lo, hi)
                    return coeff * sp.Derivative(inner, dw)
                break
        # a single unexpanded Sum factor: push the integral inside.
        sums = [f for f in factors if isinstance(f, sp.Sum)]
        if len(sums) == 1:
            the_sum = sums[0]
            others = sp.Mul(*[f for f in factors if f is not the_sum])
            cc, dep = the_sum.function.as_independent(var, as_Add=False)
            oc, od = others.as_independent(var, as_Add=False)
            return coeff * oc * sp.Sum(
                cc * sp.Integral(od * dep, (var, lo, hi)), *the_sum.limits)
        # a φ-basis bracket: substitute the concrete basis and integrate.
        if self._basis is not None and any(
                getattr(a.func, "_is_basis_head", False)
                for a in rest.atoms(sp.Function)):
            return coeff * self._resolve_basis(rest, var, lo, hi)
        # otherwise leave the var-dependent body as an opaque Integral.
        return coeff * sp.Integral(rest, (var, lo, hi))

    # ── methods ──────────────────────────────────────────────────────────
    def _ftc(self, integrand, var, lo, hi):
        """``∫_lo^hi ∂_ζ g dζ → g|_{ζ=hi} − g|_{ζ=lo}`` on the
        ζ-independent-coefficient parts; ``sp.integrate`` fallback otherwise."""
        z = var
        out = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(integrand)):
            coeff, dep = term.as_independent(z, as_Add=False)
            if (isinstance(dep, sp.Derivative)
                    and len(dep.variables) == 1 and dep.variables[0] == z):
                g = dep.expr
                out += coeff * (g.subs(z, hi) - g.subs(z, lo))
            else:
                out += sp.integrate(term, (z, lo, hi))
        return out

    def _resolve_basis(self, integrand, var, lo, hi):
        """Substitute opaque ``φ(k,ζ)`` → concrete polynomial (and weight ``c``)
        and integrate the ζ-polynomial analytically."""
        z = var
        if self._basis is None:
            return sp.integrate(integrand, (z, lo, hi))
        wv = self._weight_value

        def _is_phi(a):
            return (isinstance(a, sp.Function)
                    and getattr(a.func, "_is_basis_head", False)
                    and len(a.args) == 2)

        def _is_weight(a):
            return (isinstance(a, sp.Function)
                    and getattr(a.func, "_is_basis_head", False)
                    and len(a.args) == 1)

        def _weight_at(arg):
            return (wv(arg) if callable(wv)
                    else (sp.S.One if wv is None else wv))

        e = sp.expand(integrand)
        e = e.replace(_is_weight, lambda a: _weight_at(a.args[0]))
        e = e.replace(
            _is_phi,
            lambda a: (self._basis.eval(int(a.args[0]), a.args[1])
                       if a.args[0].is_Integer else a))
        return self._integrate_poly(e.doit(), z, lo, hi)

    def _integrate_poly(self, expr, var, lo, hi):
        z = var
        out = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(expr)):
            coeff, dep = term.as_independent(z, as_Add=False)
            if dep == 1:
                out += coeff * (hi - lo)
            elif dep == z:
                out += coeff * (hi**2 - lo**2) / 2
            elif (isinstance(dep, sp.Pow) and dep.base == z
                  and dep.exp.is_Integer and dep.exp >= 0):
                n = int(dep.exp)
                out += coeff * (hi**(n + 1) - lo**(n + 1)) / (n + 1)
            else:
                out += sp.integrate(term, (z, lo, hi))
        return out

    def _resolve_basis_atoms(self, expr):
        """Substitute opaque ``φ(k, arg)`` → ``basis.eval(k, arg)`` for concrete
        mode ``k`` and the weight ``c(arg)`` → weight value, EVERYWHERE the
        atom survives outside an integral — i.e. the FTC boundary traces
        ``φ(k, 0)`` / ``φ(k, 1)``."""
        if self._basis is None:
            return expr
        bas = self._basis
        wv = self._weight_value
        expr = expr.replace(
            lambda a: (isinstance(a, sp.Function)
                       and getattr(a.func, "_is_basis_head", False)
                       and len(a.args) == 2 and a.args[0].is_Integer),
            lambda a: bas.eval(int(a.args[0]), a.args[1]))
        expr = expr.replace(
            lambda a: (isinstance(a, sp.Function)
                       and getattr(a.func, "_is_basis_head", False)
                       and len(a.args) == 1),
            lambda a: (wv(a.args[0]) if callable(wv)
                       else (sp.S.One if wv is None else wv)))
        return expr

    def _leaf_sp(self, sp_expr):
        z = self._var

        def _has_basis_atom(a):
            return any(getattr(f.func, "_is_basis_head", False)
                       for f in a.atoms(sp.Function))

        def _mine(a):
            # Resolve any single-variable definite integral over the configured
            # var, over a fresh ``\hat{<var>}`` integration Dummy (running
            # integrals ``∫_0^ζ … d\hat ζ``, whose var is the Dummy not ζ), over
            # the configured var as a BOUND (so the running integral is caught by
            # its ζ upper limit), OR carrying an opaque basis φ/weight.
            if not (isinstance(a, sp.Integral) and len(a.limits) == 1
                    and len(a.limits[0]) == 3):
                return False
            var, lo, hi = a.limits[0]
            return (var == z or isinstance(var, sp.Dummy)
                    or z in (sp.sympify(lo).free_symbols | sp.sympify(hi).free_symbols)
                    or _has_basis_atom(a))

        out = sp_expr.replace(_mine, self._resolve_one)
        # close the remaining opaque φ-evals (FTC boundary traces) + weight.
        return self._resolve_basis_atoms(out)


# ── conservative second-order (diffusion) projection ───────────────────────


def is_conservative_diffusion(term, x):
    """True iff ``term`` is a conservative diffusive-flux atom
    ``coeff · ∂_x(F)`` whose flux ``F`` itself carries an inner ``∂_x`` — the
    ``∂_x(D·∂_x q)`` shape the :func:`~zoomy_core.derivation.system_extract`
    classifier routes to the rank-4 ``diffusion_matrix``.

    Mirrors the extractor's ``_is_second_order_x`` test so the SME pipeline and
    the SystemModel agree on what "a diffusion term" is: a single first-order
    ``∂_x`` factor wrapping an argument that still contains a ``∂_x``.  The
    advective flux ``∂_x(q²/h)`` (algebraic inner) and the first-order NCP
    couplings ``coeff·∂_x q`` are NOT matched.
    """
    factors = term.args if isinstance(term, sp.Mul) else [term]
    dxs = [f for f in factors
           if isinstance(f, sp.Derivative) and f.variables == (x,)]
    if len(dxs) != 1:
        return False
    inner = dxs[0].args[0]
    return any(dd.variables == (x,) for dd in inner.atoms(sp.Derivative))


def project_conservative_diffusion(visc_expr, test_weight, *, basis_cls,
                                   level, var, x):
    """Galerkin-project a viscous (second-order) momentum block CONSERVATIVELY.

    The base SME resolution (:class:`Resolve`) integrates the ``∂_x`` in place,
    which scatters a second-order viscous term ``−2ν h Dₓ²ũ`` into the expanded
    ``−2ν h ∂_xx q + (σ-metric)`` form that the SystemModel can only type as a
    SOURCE.  This routine instead keeps the genuine diffusive flux in the
    conservative ``∂_x(F^d)`` shape the extractor types as ``diffusion_matrix``,
    using ONLY operations — nothing about the answer is hand-written:

    1. ``Multiply(test_weight)`` — the Galerkin test weight ``c·φ_k``;
    2. ``ProductRule()`` — inverse product rule (default ``direction="inverse"``,
       acting on ``∂_x`` by coord-name; ``∂_ζ`` is skipped) moves the outer
       ``h·φ_k`` INTO the ``∂_x``, exposing the bare conservative flux
       ``∂_x(−2ν h φ_k ∂_x ũ)``;
    3. SPLIT — terms that are now a bare conservative diffusion atom
       (:func:`is_conservative_diffusion`) go through the abstract
       :class:`~zoomy_core.derivation.projection.Integrate` (which commutes the
       single OUTERMOST ``∂_x`` out, leaving the inner ``∂_x q`` intact) then
       :class:`ResolveIntegral` (basis); the σ-metric residual integrates in
       place via :class:`ResolveIntegral` (basis) on an explicit ``∫dζ``;
    4. their SUM is bit-identical to the welded in-place :class:`Resolve`, but
       the diffusive part stays ``∂_x(F^d)``.

    Returns the resolved viscous moment (conservative flux + metric residual).
    """
    from zoomy_core.derivation.projection import Integrate as _AbstractIntegrate
    from zoomy_core.model.operations import Multiply, ProductRule, Expression

    scaled = Expression(visc_expr, "").apply(Multiply(test_weight)).expr
    pr = Expression(scaled, "").apply(ProductRule()).expr

    cons_terms, rest_terms = [], []
    for term in sp.Add.make_args(sp.expand(pr)):
        (cons_terms if is_conservative_diffusion(term, x)
         else rest_terms).append(term)

    ri = ResolveIntegral(var=var, method="basis", basis_cls=basis_cls,
                         level=level)
    # Conservative flux: abstract ∫ commutes the outer ∂_x out → resolve basis.
    cons_int = Expression(sp.Add(*cons_terms), "").apply(
        _AbstractIntegrate(var=var, bounds=(0, 1))).expr
    cons_res = Expression(cons_int, "").apply(ri).expr
    # σ-metric residual: resolve in place (explicit ∫dζ wrapper, basis).
    rest_res = Expression(
        sp.Integral(sp.Add(*rest_terms), (var, 0, 1)), "").apply(ri).expr
    return sp.expand(cons_res) + sp.expand(rest_res)


# ── Simplify (graph-recorded canonicalisation) ─────────────────────────────


def pull_consts(expr):
    """``∂_v(c·f) → c·∂_v f`` for every factor ``c`` of the integrand that is
    INDEPENDENT of the derivative variables — applied to fixpoint, recursively
    through nested derivatives.

    This is the GENTLE normalisation that makes cancellations surface WITHOUT
    blowing fluxes apart: it never applies the product rule to a ``v``-dependent
    product, so ``∂_x(g h²/2) → (g/2)∂_x(h²)`` (``h²`` stays grouped) while
    ``∂_x(ζ h) → ζ ∂_x h`` (so ``∂_ζ(−ũ²∂_x(ζh))`` and ``∂_ζ(ζũ²∂_x h)`` become
    syntactically equal-and-opposite and cancel)."""
    def _pull(a):
        if not a.variables:
            return a
        const, dep = a.expr.as_independent(*a.variables)
        return a if const == sp.S.One else const * sp.Derivative(dep, *a.variables)
    prev = None
    while expr != prev:
        prev = expr
        expr = expr.replace(lambda a: isinstance(a, sp.Derivative), _pull)
    return expr


def _single_deriv_factor(term):
    """``(coeff, deriv)`` when ``term`` has exactly ONE Derivative factor."""
    factors = sp.Mul.make_args(term)
    ds = [f for f in factors if isinstance(f, sp.Derivative)]
    if len(ds) != 1:
        return None
    coeff = sp.Mul(*[f for f in factors if f is not ds[0]])
    return coeff, ds[0]


def fold_conservatives(expr):
    """Recover conservative groupings ``g·∂_v f + f·∂_v g → ∂_v(f·g)`` by a
    count-guarded pairwise fold: a pair is replaced by its single grouped form
    ONLY when the fold is EXACT (value-preserving) — so it always reduces the
    term count by one and never expands a flux.  Genuine non-conservative
    couplings (no exact partner) are left untouched."""
    terms = list(sp.Add.make_args(expr))
    changed = True
    while changed:
        changed = False
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                si, sj = _single_deriv_factor(terms[i]), _single_deriv_factor(terms[j])
                if not (si and sj):
                    continue
                (ci, di), (cj, dj) = si, sj
                if di.variables != dj.variables:
                    continue
                cand = sp.Derivative(ci * di.expr, *di.variables)     # ∂_v(coeff·field)
                if pull_consts(sp.expand((cand - terms[i] - terms[j]).doit())) == 0:
                    terms[i] = cand
                    del terms[j]
                    changed = True
                    break
            if changed:
                break
    return sp.Add(*terms)


class Simplify(Operation):
    """Canonicalisation that keeps fluxes as fluxes while exposing cancellations.

    Two competing objectives — (1) keep conservative divergences ``∂_v(F)``
    grouped (do NOT product-rule them) and (2) extract constants so cancelling
    terms become visible — are reconciled in three passes:

    1. :func:`pull_consts` — pull derivative-independent constants OUT of every
       ``∂_v`` (gentle: never product-rules a ``v``-dependent product).  Equal-
       and-opposite terms then cancel automatically in the ``Add``.
    2. (cancellation is implicit in the ``Add`` once normalised.)
    3. :func:`fold_conservatives` — count-guarded pairwise fold
       ``g∂_v f + f∂_v g → ∂_v(fg)``: applied ONLY when exact, so it never
       increases the term count and never breaks a genuine non-conservative
       coupling apart.
    """

    whole_leaf_op = True
    log_level = 5            # minor canonicalisation — hidden at low verbosity

    def __init__(self, name="simplify"):
        super().__init__(name=name, description="canonical simplify")

    def _leaf_sp(self, sp_expr):
        return fold_conservatives(pull_consts(sp_expr))


# ── term tagging + sorting ─────────────────────────────────────────────────

#: Canonical physics order — the order terms are written by hand.
TAG_ORDER = [
    "time_derivative",        # ∂_t(…)            — the local time derivative
    "flux",                   # ∂_x(…) / ∂_ζ(…)   — conservative flux
    "diffusion",              # ∂_x(… ∂_x …)      — diffusive (2nd-order) flux
    "nonconservative_flux",   # c(Q)·∂(Q)         — NCP
    "source",                 # algebraic / ∂(known field) — source
    "untagged",
]


def _classify_term(term, q_heads, t, spatial):
    """Heuristic physics category for one additive ``term``.

    ``q_heads`` = the unknown family heads (so a field coefficient times a
    derivative of an UNKNOWN is an NCP, but of a KNOWN field, e.g. the bed
    ``b``, is a source).  ``t`` is the time coord, ``spatial`` the horizontal
    coords.  Best-effort — the user pre-tags the cases the heuristic misses
    (``AutoTag`` never overwrites an existing tag)."""
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    derivs = [f for f in factors
              if isinstance(f, sp.Derivative) and len(f.variables) == 1]
    if len(derivs) != 1:
        # No single outer derivative: a plain algebraic source, or — if it is
        # a tangle of derivatives — left untagged for the user to resolve.
        if any(isinstance(f, sp.Derivative) for f in factors):
            return "untagged"
        return "source"
    dv = derivs[0]
    var = dv.variables[0]
    arg = dv.expr
    coeff = sp.Mul(*[f for f in factors if f is not dv])
    coeff_has_field = bool(coeff.atoms(sp.Function))
    if var == t:
        return "time_derivative"
    if var in spatial and any(var in a.variables
                              for a in arg.atoms(sp.Derivative)):
        return "diffusion"
    if coeff_has_field:
        arg_has_unknown = any(getattr(a, "func", None) in q_heads
                              for a in arg.atoms(sp.Function))
        return "nonconservative_flux" if arg_has_unknown else "source"
    return "flux"


def _tag_context(eq):
    """``(q_heads, t, spatial)`` read off the equation's parent model."""
    m = getattr(eq, "_model", None)
    if m is None:
        return set(), None, set()
    q_heads = {m._head(f) for v in m._Q.values()
               for f in m._as_field_list(v)}
    coords = m._coords
    return q_heads, coords[0], set(coords[1:-1])


class AutoTag(Operation):
    """Tag each UNTAGGED additive term with its physics category
    (``time_derivative`` / ``flux`` / ``diffusion`` / ``nonconservative_flux``
    / ``source``), leaving already-tagged terms alone.

    Tags live on the terms and are hidden in ``describe`` unless
    ``describe(show_tags=True)`` is asked for.  Pairs with :class:`SortByTag`;
    :class:`Sort` runs both."""

    log_level = 5

    def __init__(self, name="auto_tag"):
        super().__init__(name=name, description="tag untagged terms by category")

    def apply_to_equation(self, eq):
        q_heads, t, spatial = _tag_context(eq)
        for tm in eq._terms:
            if tm.tag is None:
                tm.tag = _classify_term(tm.expr, q_heads, t, spatial)
        return eq


class SortByTag(Operation):
    """Reorder the equation's terms into the canonical :data:`TAG_ORDER`
    (time derivative, flux, diffusion, NCP, source, untagged), stable within
    each group.  Display order is honoured by ``describe`` (which renders the
    terms in their current order, bypassing sympy's ``Add`` canonicalisation)."""

    log_level = 5

    def __init__(self, name="sort_by_tag"):
        super().__init__(name=name, description="order terms by physics tag")

    def apply_to_equation(self, eq):
        rank = {tag: i for i, tag in enumerate(TAG_ORDER)}
        eq._terms.sort(
            key=lambda tm: rank.get(tm.tag or "untagged", len(TAG_ORDER)))
        return eq


class Sort(Operation):
    """:class:`AutoTag` then :class:`SortByTag` — order the terms the way you
    would write them by hand.  Apply it LAST (after the algebra is settled): a
    later expression-level op rebuilds the term list and drops the ordering."""

    log_level = 5

    def __init__(self, name="sort"):
        super().__init__(name=name, description="auto-tag + sort by physics tag")

    def apply_to_equation(self, eq):
        AutoTag().apply_to_equation(eq)
        SortByTag().apply_to_equation(eq)
        return eq
