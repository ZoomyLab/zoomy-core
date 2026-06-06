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
    "Simplify",
    "kinematic_modal_closure",
    "mass_relation",
    "fold_to_conservative_form",
    "is_conservative_diffusion",
    "project_conservative_diffusion",
]


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
                if cand != 0 and not cand.has(t):     # a clean coefficient
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

    def __init__(self, basis_cls=None, *, var=None, method="basis", level=None,
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
        if method == "ftc":
            return self._ftc(integrand, var, lo, hi)
        if method == "basis":
            return self._resolve_basis(integrand, var, lo, hi)
        if method == "numerical":
            return sp.integrate(integrand, (var, lo, hi))
        raise ValueError(f"ResolveIntegral: unknown method {method!r}")

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
            # Resolve any single-variable definite integral that is either over
            # the configured var OR carries an opaque basis φ/weight (so the
            # ``\hat{z}``-renamed φ-brackets from the KinematicBC are caught).
            if not (isinstance(a, sp.Integral) and len(a.limits) == 1
                    and len(a.limits[0]) == 3):
                return False
            return a.limits[0][0] == z or _has_basis_atom(a)

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


class Simplify(Operation):
    """A whole-equation canonicalisation that records a step in the graph.

    Routes through :meth:`zoomy_core.model.equation.Equation.simplify` (the
    project's ``expand → pull-constants-out → recover-fluxes`` pipeline) when
    applied to a full equation, and falls back to ``sp.expand`` on a bare leaf.
    """

    whole_leaf_op = True
    log_level = 5            # minor canonicalisation — hidden at low verbosity

    def __init__(self, name="simplify"):
        super().__init__(name=name, description="canonical simplify")

    def _leaf_sp(self, sp_expr):
        return sp.expand(sp_expr)


# ── kinematic modal closure (sp.solve on the two KBCs) ─────────────────────


def kinematic_modal_closure(model, *, u_field, w_field, h, b,
                            basis_cls, n_u, name="kinematic_modal_closure"):
    """Build the ``Substitution`` closing the lower w-modes from the two
    kinematic BCs (surface ζ=1, bed ζ=0).

    Mirrors the production ``SME._kbc_modal_closure`` (and ``mlme.py``): the
    modal ansatzes for ``u`` (modes ``0..n_u``) and ``w`` (modes ``0..n_u+1``)
    are evaluated at the boundaries ``ζ ∈ {0, 1}`` via ``basis_cls.eval``, the
    surface / bed kinematic relations

    .. math::

        w|_{ζ=1} &= ∂_t(b+h) + u|_{ζ=1}\\,∂_x(b+h), \\\\
        w|_{ζ=0} &= ∂_t b      + u|_{ζ=0}\\,∂_x b,

    are formed, and ``sp.solve`` closes for the two lowest free w-modes
    ``aw_0(t,x), aw_1(t,x)``.  Returns a
    :class:`~zoomy_core.derivation.operations.Substitution` ready for
    ``model.apply(...)``.

    Parameters
    ----------
    model : Model
        The threaded model (after PDE-transform + SoV) — provides the modal
        coefficient heads via its registry; the heads are passed explicitly to
        keep Symbol identity unambiguous.
    u_field, w_field : sympy Function head
        The coefficient family heads ``a`` (u-modes) and ``aw`` (w-modes).
    h, b : sympy applied Function
        Depth ``h(t, x)`` and bed ``b(t, x)``.
    basis_cls : type
        The concrete basis providing ``.eval(k, point)``.
    n_u : int
        u truncation level (u modes ``0..n_u``; w modes ``0..n_u+1``).
    """
    from zoomy_core.derivation.operations import Substitution

    t = h.args[0]
    x = h.args[1]
    basis = basis_cls(level=n_u + 1)
    a_head = u_field
    aw_head = w_field

    def u_at(point):
        return sum(a_head(k, t, x) * basis.eval(k, point)
                   for k in range(n_u + 1))

    def w_at(point):
        return sum(aw_head(k, t, x) * basis.eval(k, point)
                   for k in range(n_u + 2))

    surface = w_at(sp.S.One) - (sp.Derivative(b + h, t)
                                + u_at(sp.S.One) * sp.Derivative(b + h, x))
    bed = w_at(sp.S.Zero) - (sp.Derivative(b, t)
                             + u_at(sp.S.Zero) * sp.Derivative(b, x))
    solution = sp.solve(
        [surface, bed], [aw_head(0, t, x), aw_head(1, t, x)], dict=True,
    )[0]
    return Substitution(solution, name=name)


def mass_relation(h, a_head, *, name="mass_relation"):
    """Build the ``Substitution`` ``∂_t h → −∂_x(h·a_0)`` from the depth-
    averaged mass row, used to cancel the stray ``∂_t h`` the σ-metric
    chain-rule injects into the momentum rows."""
    from zoomy_core.derivation.operations import Substitution

    t = h.args[0]
    x = h.args[1]
    return Substitution(
        {sp.Derivative(h, t): -sp.Derivative(h * a_head(0, t, x), x)},
        name=name,
    )
