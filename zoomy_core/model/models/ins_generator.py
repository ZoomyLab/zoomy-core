"""
General-purpose INS equation framework with composable projection operations.

Class hierarchy:
    StateSpace      — shared symbols (coordinates, fields, stress tensor, parameters)
    SymbolicBase    — name + display
    ├── Expression  — PDE terms with .project(), .ibp(), .apply(), .terms, [i]
    └── Relation    — lhs = rhs substitution rules with .apply_to()
        ├── Assumption  — physical conditions (kinematic BCs, hydrostatic)
        └── Material    — constitutive models (Newtonian, inviscid)

    FullINS(state)  — builds INS equations from a StateSpace

Design: the user drives every step. No hidden assumptions.
"""

import sympy as sp
from sympy import (
    Function, Symbol, Derivative, Integral, Add, Mul, S, Rational,
    Heaviside, DiracDelta,
)
import numpy as np


# Sentinel for optional kwargs that must distinguish "not passed" from
# "passed a falsy value" (most notably ``rhs=0``).
_UNSET = object()


# ---------------------------------------------------------------------------
# StateSpace: shared symbolic state
# ---------------------------------------------------------------------------

class StateSpace:
    """
    Shared symbolic state: coordinates, velocity fields, pressure,
    stress tensor, bathymetry, and physical parameters.

    dimension is the physical space dimension:
        2 = xz plane (1D horizontal shallow water)
        3 = xyz space (2D horizontal shallow water)

    The vertical coordinate is always z. Horizontal coordinates are
    x (and y if dimension=3). The horizontal_dim property gives the
    number of horizontal directions (1 or 2).
    """

    def __init__(self, dimension=2):
        if dimension < 2 or dimension > 3:
            raise ValueError(f"dimension must be 2 (xz) or 3 (xyz), got {dimension}")
        self.dim = dimension

        self.t = Symbol("t", real=True)
        self.x = Symbol("x", real=True)
        self.y = Symbol("y", real=True)
        self.z = Symbol("z", real=True)
        self.zeta = Symbol("zeta", real=True)

        has_y = dimension > 2

        args_h = [self.t, self.x]
        if has_y:
            args_h.append(self.y)
        args_3d = args_h + [self.z]
        self._args_h = args_h
        self._args_3d = args_3d

        self.u = Function("u", real=True)(*args_3d)
        self.v = Function("v", real=True)(*args_3d) if has_y else S.Zero
        self.w = Function("w", real=True)(*args_3d)
        self.p = Function("p", real=True)(*args_3d)

        self.rho = Symbol("rho", positive=True)
        self.g = Symbol("g", positive=True)

        self._build_stress_tensor(has_y, args_3d)

        self.b = Function("b", real=True)(*args_h)
        self.H = Function("h", real=True)(*args_h)
        self.eta = self.b + self.H

        self.coords_h = [self.x] + ([self.y] if has_y else [])
        self.velocities_h = [self.u] + ([self.v] if has_y else [])

    def _build_stress_tensor(self, has_y, args_3d):
        from zoomy_core.misc.misc import Zstruct
        labels = ["x", "y", "z"] if has_y else ["x", "z"]
        tau_dict = {}
        for i in labels:
            for j in labels:
                tau_dict[i + j] = Function(f"tau_{i}{j}", real=True)(*args_3d)
        self.tau = Zstruct(**tau_dict)

    @property
    def horizontal_dim(self):
        return self.dim - 1

    @property
    def has_y(self):
        return self.dim > 2

    def __repr__(self):
        n_tau = len(self.tau)
        fields = "[u,v,w,p]" if self.has_y else "[u,w,p]"
        return f"StateSpace(dim={self.dim}, fields={fields}, tau={n_tau} components)"


# ---------------------------------------------------------------------------
# SymbolicBase: shared name + display
# ---------------------------------------------------------------------------

class SymbolicBase:
    """Base for all symbolic objects. Provides name and notebook display."""

    def __init__(self, name=""):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Expression: composable symbolic PDE term
# ---------------------------------------------------------------------------

class Expression(SymbolicBase):
    """
    Symbolic expression for PDE terms.

    Supports:
    - Term access: .terms, [i], len(), iteration
    - Projection: .project(test, var, domain, ...)
    - Integration by parts: .ibp(var, test_weight, domain) -> IBPResult
    - Apply conditions: .apply({old: new}, ...) or .apply(relation)
    - SymPy: .subs(), .simplify(), .expand(), .doit()
    - Notebook: _repr_latex_
    """

    def __init__(self, expr, name="", term_groups=None, solver_groups=None,
                 term_tags=None, tag_order=None):
        super().__init__(name)
        if isinstance(expr, Expression):
            expr = expr.expr
        self.expr = sp.sympify(expr) if not isinstance(expr, sp.Basic) else expr
        # Physical tags: per-additive-term labels for the display view.
        # ``_term_tags`` maps each tagged term (sp.Expr) to the tag name it
        # belongs to.  Tags are propagated *follow-the-term* by ``.apply``,
        # ``.simplify``, ``.subs``, ``.expand``: when a transformation
        # replaces ``old_term`` with ``new_term(s)``, every new term
        # inherits the old term's tag.  Cross-tag cancellation in the main
        # ``self.expr`` is reflected by terms vanishing from both
        # ``Add.make_args(self.expr)`` and ``_term_tags``.
        self._term_tags = dict(term_tags) if term_tags else {}
        # Display order for tags (first-seen wins).
        self._tag_order = list(tag_order) if tag_order else []
        # Back-compat: legacy ``term_groups`` kwarg (parallel-sum dict).
        # Convert to per-term entries.  Summing is associative, so splitting
        # a tag's group into Add.make_args and recording each piece gives
        # an equivalent view.
        if term_groups:
            for name_, g in term_groups.items():
                g_sp = g if isinstance(g, sp.Basic) else sp.sympify(g)
                # ``sp.expand`` distributes scalar factors across Adds so
                # something like ``-(∂_x τ_xx + ∂_z τ_xz)/ρ`` gets split
                # into its atomic additive terms before we record each
                # under the tag.
                for term in Add.make_args(sp.expand(g_sp)):
                    if term != S.Zero:
                        self._term_tags[term] = name_
                if name_ not in self._tag_order:
                    self._tag_order.append(name_)
        # Solver tag groups for operator routing (flux, source, NC, ...).
        # {canonical_tag: sp.Expr}. Dropped per-tag whenever apply/simplify
        # changes the tagged sub-expression (structural equality check).
        self._solver_groups = dict(solver_groups) if solver_groups else None

    # ``_term_groups`` is a lazy view over ``_term_tags`` filtered by the
    # current ``self.expr`` terms.  It's always in sync with the main
    # expression (terms that cancelled out of ``self.expr`` disappear
    # from the groups automatically).  Kept as a property for any code
    # that still reads ``_term_groups``.
    @property
    def _term_groups(self):
        if not self._term_tags:
            return None
        current = set(Add.make_args(sp.expand(self.expr)))
        groups = {name: S.Zero for name in self._tag_order}
        for term in current:
            tag_name = self._term_tags.get(term)
            if tag_name is None:
                continue
            if tag_name not in groups:
                groups[tag_name] = S.Zero
            groups[tag_name] = groups[tag_name] + term
        non_empty = {k: v for k, v in groups.items() if v != S.Zero}
        return non_empty if non_empty else None

    @_term_groups.setter
    def _term_groups(self, value):
        # Setter exists only so ``__init__`` could still do the legacy
        # assignment during early-init before the dict is built.  The
        # attribute has been fully migrated to ``_term_tags`` / ``_tag_order``.
        pass

    @property
    def terms(self):
        expanded = sp.expand(self.expr)
        raw = Add.make_args(expanded)
        return [Expression(t, f"{self.name}[{i}]") for i, t in enumerate(raw)]

    def __getitem__(self, i):
        if isinstance(i, slice):
            ts = self.terms[i]
            combined = sum((t.expr for t in ts), S.Zero)
            return Expression(combined, self.name)
        return self.terms[i]

    def apply_to_term(self, index, *operations):
        """Apply operations to a specific term and return the full expression.

        Usage::

            xmom.apply_to_term(5, ProductRule())
            xmom.apply_to_term(5, {old: new})
        """
        terms = self.terms
        modified = terms[index].apply(*operations)
        new_terms = [t.expr if i != index else modified.expr
                     for i, t in enumerate(terms)]
        return Expression(sum(new_terms, S.Zero), self.name)

    def keep_groups(self, *group_names):
        """Return a new Expression keeping only the specified tags' terms."""
        if not self._term_tags:
            raise ValueError("Expression has no tags.")
        kept_terms = {t: name for t, name in self._term_tags.items()
                      if name in group_names}
        if not kept_terms:
            return Expression(S.Zero, self.name)
        kept_order = [n for n in self._tag_order if n in group_names]
        new_expr = sum(kept_terms.keys(), S.Zero)
        return Expression(new_expr, self.name,
                          term_tags=kept_terms, tag_order=kept_order)

    def drop_groups(self, *group_names):
        """Return a new Expression dropping the specified tags' terms."""
        if not self._term_tags:
            raise ValueError("Expression has no tags.")
        kept_terms = {t: name for t, name in self._term_tags.items()
                      if name not in group_names}
        if not kept_terms:
            return Expression(S.Zero, self.name)
        kept_order = [n for n in self._tag_order if n not in group_names]
        new_expr = sum(kept_terms.keys(), S.Zero)
        return Expression(new_expr, self.name,
                          term_tags=kept_terms, tag_order=kept_order)

    # ------------------------------------------------------------------
    # Tagging: physical (display) and solver (operator routing)
    # ------------------------------------------------------------------

    def tag(self, **named_groups):
        """Attach physical tags to terms for display.

        Each ``name=value`` kwarg is split into additive terms and each
        term is recorded as belonging to ``name`` in the per-term tag map.

        Tags follow the terms through ``.apply`` / ``.subs`` / ``.simplify``:
        when a transformation replaces a tagged term, the new term(s)
        inherit the tag.  Terms that cancel (via ``simplify``) silently
        disappear from the view — no parallel-sum math, no cross-tag
        contamination.

        Usage::

            xmom = xmom.tag(pressure=grad_p, stress=div_tau)
        """
        new_tags = dict(self._term_tags)
        new_order = list(self._tag_order)
        for name_, v in named_groups.items():
            v_sp = v.expr if isinstance(v, Expression) else sp.sympify(v)
            for term in Add.make_args(sp.expand(v_sp)):
                if term != S.Zero:
                    new_tags[term] = name_
            if name_ not in new_order:
                new_order.append(name_)
        return Expression(self.expr, self.name,
                          term_tags=new_tags, tag_order=new_order,
                          solver_groups=self._solver_groups)

    @property
    def tags(self):
        """Dict of current-expr terms grouped by physical tag.

        Computed on demand by walking ``Add.make_args(self.expr)`` and
        assigning each term to the tag recorded in ``_term_tags``.  Terms
        not in the main expression (cancelled by simplify, etc.) do not
        appear — the view stays in sync with the single source of truth.
        """
        if not self._term_tags:
            return {}
        # Group current expression terms by tag, preserving tag order.
        groups = {name: S.Zero for name in self._tag_order}
        for term in Add.make_args(sp.expand(self.expr)):
            tag_name = self._term_tags.get(term)
            if tag_name is None:
                continue
            if tag_name not in groups:
                groups[tag_name] = S.Zero
            groups[tag_name] = groups[tag_name] + term
        # Drop empty tags (their terms all cancelled).
        return {k: v for k, v in groups.items() if v != S.Zero}

    @property
    def untagged(self):
        """Sum of current-expr terms that have no physical tag."""
        tagged_terms = set(self._term_tags)
        return sum(
            (t for t in Add.make_args(sp.expand(self.expr)) if t not in tagged_terms),
            S.Zero,
        )

    def solver_tag(self, **named_groups):
        """Attach/replace solver tag groups for operator routing.

        Tag names are normalized via the solver-tag alias catalog, so
        ``.solver_tag(convection=...)`` and ``.solver_tag(flux=...)`` both
        write to the canonical ``flux`` slot.

        Solver tags are dropped per-tag whenever ``apply()`` / ``simplify()``
        changes the tagged sub-expression (structural equality check).

        Usage::

            xmom = xmom.solver_tag(flux=F_x, source=g * h * dbdx)
        """
        from zoomy_core.model.models.tag_catalog import canonical_solver_tag

        merged = dict(self._solver_groups or {})
        for k, v in named_groups.items():
            canonical = canonical_solver_tag(k)
            merged[canonical] = v.expr if isinstance(v, Expression) else sp.sympify(v)
        return Expression(self.expr, self.name,
                          term_groups=self._term_groups, solver_groups=merged)

    @property
    def solver_tags(self):
        """Dict of solver tags (canonical names), or empty dict if none set."""
        return dict(self._solver_groups) if self._solver_groups else {}

    def get_solver_tag(self, name):
        """Return the sp.Expr for a solver tag, or None if not set.

        ``name`` can be any alias; it is normalized to canonical form.
        """
        from zoomy_core.model.models.tag_catalog import canonical_solver_tag

        if not self._solver_groups:
            return None
        return self._solver_groups.get(canonical_solver_tag(name))

    def untagged_remainder(self):
        """Return ``self.expr - sum(solver_tags.values())``, simplified.

        Zero means every term has been routed to a solver tag; a non-zero
        remainder is what drives the model's ``untagged_policy`` (warn/error).
        """
        if not self._solver_groups:
            return self.expr
        tagged_sum = sum(self._solver_groups.values(), S.Zero)
        return sp.expand(self.expr - tagged_sum)

    def auto_solver_tag(self, *, state_vars, time_var, coords,
                        parameters=()):
        """Algorithmically assign solver tags by structural matching.

        Each additive term of ``self.expr`` is classified:

        * ``coeff * Derivative(q, t)`` with ``q`` a state variable →
          ``time_derivative``.
        * ``coeff * Derivative(F, x_i)`` where ``coeff`` is state-variable-
          free → ``flux`` (conservative form).
        * ``coeff * Derivative(q_j, x_i)`` with ``q_j`` a state variable
          and ``coeff`` containing state vars → ``nonconservative_flux``.
        * ``coeff * Derivative(p, x_i)`` where ``p`` is a declared
          parameter and ``coeff`` contains state vars → also
          ``nonconservative_flux`` (the term couples a state coefficient
          to a parameter gradient — e.g. bathymetry-gradient forcing).
        * No derivative and purely algebraic in ``state_vars ∪
          parameters`` space → ``source``.
        * Anything else (unclosed fields like ``u(t, x, z)``, unmatched
          Derivative shapes) stays in the untagged remainder.

        Parameters
        ----------
        state_vars : iterable of sympy Functions / Symbols
            The evolution variables (``α_k(t, x)``, interface heights,
            etc.).
        time_var : sympy Symbol
            The time coordinate (for ``time_derivative`` detection).
        coords : iterable of sympy Symbols
            Spatial coordinates (``x``, ``y``, …) for flux / NCP.
        parameters : iterable, optional
            Known external Function/Symbol atoms (bathymetry ``b(t, x)``,
            prescribed fluxes, etc.).  These may freely appear in any
            term without the term being marked "unclosed", and they may
            appear as the inner of a spatial Derivative in the NCP
            tagging.

        Returns a new Expression with ``solver_groups`` populated.
        Physical tags are preserved.
        """
        state_set = set(state_vars)
        parameters = set(parameters)
        allowed_fn_set = state_set | parameters
        coords_list = list(coords)
        tags = {
            "time_derivative": S.Zero,
            "flux": S.Zero,
            "nonconservative_flux": S.Zero,
            "source": S.Zero,
        }

        def _has_unclosed_function(term):
            """``True`` iff the term contains a Function call that is
            neither a declared state variable nor a declared parameter.
            Those are the terms the auto-tagger can't classify safely;
            everything else (including Function atoms of state_vars /
            parameters — they're allowed to carry ``(t, x)`` args) is
            fair game.
            """
            return any(a not in allowed_fn_set and a.args
                       for a in term.atoms(Function))

        for term in Add.make_args(sp.expand(self.expr)):
            if term == S.Zero:
                continue

            if _has_unclosed_function(term):
                continue

            # Try to split as `coeff * Derivative(...)`.
            coeff = S.One
            deriv = None
            if isinstance(term, Derivative):
                deriv = term
            elif isinstance(term, Mul):
                ds = [a for a in term.args if isinstance(a, Derivative)]
                if len(ds) == 1:
                    deriv = ds[0]
                    coeff = Mul(*[a for a in term.args if a is not deriv])

            if deriv is not None and len(deriv.variables) == 1:
                v = deriv.variables[0]
                inner = deriv.args[0]

                # time_derivative: coeff * ∂_t(state_var_or_parameter).
                # Time-derivatives of parameters (e.g. moving bathymetry)
                # get lumped here too — solvers typically treat them as
                # time-varying forcing alongside the genuine ∂_t q terms.
                if v == time_var and inner in allowed_fn_set:
                    tags["time_derivative"] = tags["time_derivative"] + term
                    continue

                if v in coords_list:
                    # A coeff counts as "state-bearing" iff it contains a
                    # state-variable atom.  Function state variables
                    # (``α_k(t, x)``) don't show up in ``free_symbols``,
                    # so test via ``.atoms(Function) | .free_symbols``.
                    coeff_symbols = coeff.free_symbols | coeff.atoms(Function)
                    coeff_state_refs = state_set & coeff_symbols
                    if not coeff_state_refs:
                        tags["flux"] = tags["flux"] + term
                        continue
                    # coeff has state; treat any ∂_xᵢ(state_or_parameter) as NCP.
                    if inner in allowed_fn_set:
                        tags["nonconservative_flux"] = (
                            tags["nonconservative_flux"] + term)
                        continue

            has_derivative = any(
                isinstance(a, Derivative) for a in term.atoms(Derivative)
            )
            if not has_derivative:
                tags["source"] = tags["source"] + term
                continue

        # Drop empty tag groups.
        nonempty = {t: v for t, v in tags.items() if v != 0}
        return Expression(self.expr, self.name,
                          term_groups=self._term_groups,
                          solver_groups=nonempty)

    def __len__(self):
        return len(Add.make_args(sp.expand(self.expr)))

    def __iter__(self):
        return iter(self.terms)

    def __add__(self, other):
        other_expr = other.expr if isinstance(other, Expression) else other
        return Expression(self.expr + other_expr, self.name)

    def __radd__(self, other):
        if other == 0:
            return self
        return Expression(other + self.expr, self.name)

    def __sub__(self, other):
        other_expr = other.expr if isinstance(other, Expression) else other
        return Expression(self.expr - other_expr, self.name)

    def __neg__(self):
        return Expression(-self.expr, self.name)

    def __mul__(self, other):
        other_expr = other.expr if isinstance(other, Expression) else other
        return Expression(self.expr * other_expr, self.name)

    def __rmul__(self, other):
        return Expression(other * self.expr, self.name)

    def project(self, test_func, var, domain=(0, 1), weight=S.One,
                scale=S.One, numerical=False, order=4):
        """Galerkin projection: integral(expr * test * weight * scale, (var, a, b))."""
        integrand = self.expr * test_func * weight * scale
        if numerical:
            result = gauss_legendre_integrate(integrand, var, domain[0], domain[1], order)
            return Expression(result, f"project_num({self.name})")
        return Expression(
            Integral(integrand, (var, domain[0], domain[1])),
            f"project({self.name})"
        )

    def ibp(self, var, test_weight, domain=(0, 1), scale=S.One):
        """
        Integration by parts on the outermost Derivative w.r.t. var.
        Returns IBPResult(integrate, boundary_upper, boundary_lower).
        """
        inner, coeff = _extract_derivative(self.expr, var)
        if inner is None:
            raise ValueError(
                f"No Derivative w.r.t. {var} found in: {self.expr}\n"
                f"Use .project() for terms without derivatives."
            )
        a, b = domain
        tw = test_weight * scale
        return IBPResult(
            integrate=Expression(
                -Integral(coeff * inner * Derivative(test_weight, var) * scale, (var, a, b)),
                f"ibp_integrate({self.name})"
            ),
            boundary_upper=Expression(
                (coeff * inner * tw).subs(var, b),
                f"ibp_upper({self.name})"
            ),
            boundary_lower=Expression(
                (coeff * inner * tw).subs(var, a),
                f"ibp_lower({self.name})"
            ),
        )

    def apply(self, *conditions, rhs=_UNSET):
        """Apply substitutions / Relations / Operations — per term.

        The transformation is applied to every additive term of
        ``self.expr`` individually.  Each output term inherits the tag of
        its parent term (follow-the-term), so the per-term tag dict stays
        in sync with the evolving main expression automatically.

        Shorthand: ``expr.apply(lhs, rhs=0)`` is equivalent to
        ``expr.apply({lhs: 0})``.  Valid only with a single positional
        argument which is a sympy expression (or :class:`Expression`).

        After Relation/dict-style applies the result is lightly simplified
        via ``_simplify_preserve_integrals`` (linearity-only derivatives,
        cancellations across the full expr).  Terms that cancel vanish
        from both ``self.expr`` and the tag dict.

        Operations are invoked via their unified ``__call__`` interface;
        when the op produces multiple output pieces per input term
        (``DepthIntegrate`` etc.) all pieces inherit the parent tag.
        """
        if rhs is not _UNSET:
            if len(conditions) != 1:
                raise TypeError(
                    "apply(lhs, rhs=...) shorthand requires exactly one "
                    f"positional argument (got {len(conditions)})."
                )
            lhs = conditions[0]
            lhs_sp = lhs.expr if isinstance(lhs, Expression) else lhs
            conditions = ({lhs_sp: rhs},)
        _resolve_subs = _resolve_subs_safe  # module-level helper

        def _apply_one(expr, cond):
            if isinstance(cond, Operation):
                result = cond(Expression(expr, self.name))
                if isinstance(result, Expression):
                    return result.expr
                raise TypeError(
                    f"{type(cond).__name__} returned {type(result).__name__} "
                    f"from per-term apply; rank-changing Operations must be "
                    f"invoked at the tree level, not through Expression.apply."
                )
            rel = getattr(cond, "_as_relation", None)
            if isinstance(rel, dict) and rel:
                expr = _resolve_subs(expr)
                return expr.subs(rel)
            if isinstance(cond, Relation) or hasattr(cond, 'apply_to'):
                expr = _resolve_subs(expr)
                return cond.apply_to(expr)
            elif isinstance(cond, dict):
                expr = _resolve_subs(expr)
                # ``xreplace`` is purely structural — no dummy-dependency
                # safety check — so it handles the basis-expansion case
                # where the RHS contains the integration variable (``ζ``
                # inside a zeta-integral).  All our dict keys are
                # concrete Function applications or Subs-derived forms,
                # both of which match exactly under structural equality.
                expr = expr.xreplace(cond)
                # Second ``_resolve_subs`` pass: the substitution may
                # have closed running integrals / nested Subs that
                # previously blocked resolution (``ContinuityClosure``'s
                # ``∫_b^z ∂_x u dz'`` becomes a polynomial in ``z``
                # once ``u`` is basis-expanded), so Subs around them
                # now resolve cleanly.
                return _resolve_subs(expr)
            elif isinstance(cond, (list, tuple)):
                if len(cond) == 2 and isinstance(cond[0], sp.Basic):
                    return expr.subs(cond[0], cond[1])
                for pair in cond:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        expr = expr.subs(pair[0], pair[1])
                return expr
            return expr

        # Per-term iteration with tag inheritance.
        #
        # For non-Operation conditions we run the lightweight simplify
        # (``_simplify_preserve_integrals``, linearity-only on
        # Derivatives) **per term** before splitting into sub-pieces.
        # This fragments things like ``Derivative(g*ρ*(η − z), x) / ρ``
        # into ``g·∂_x b + g·∂_x h`` before we record each piece under
        # the parent tag — otherwise the unfragmented form becomes an
        # orphan key when a later global simplify fragments it.
        simplifying = not any(isinstance(c, Operation) for c in conditions)
        new_tags = {}
        pieces = []
        for term in Add.make_args(sp.expand(self.expr)):
            if term == S.Zero:
                continue
            parent_tag = self._term_tags.get(term)
            result = term
            for cond in conditions:
                result = _apply_one(result, cond)
            if simplifying:
                result = _simplify_preserve_integrals(result)
            for sub in Add.make_args(sp.expand(result)):
                if sub == S.Zero:
                    continue
                pieces.append(sub)
                if parent_tag is not None:
                    new_tags[sub] = parent_tag

        new_expr = sum(pieces, S.Zero)

        # Global simplify catches cross-term cancellations (e.g. the
        # ``u|_η · ∂_t b`` pieces from two different tags that sum to
        # zero after kinematic BCs).  Surviving terms keep their tags;
        # cancelled ones silently disappear.
        if simplifying:
            new_expr = _simplify_preserve_integrals(new_expr)
            current = set(Add.make_args(sp.expand(new_expr)))
            new_tags = {t: name for t, name in new_tags.items() if t in current}

        # Solver tags: drop any whose stored expression was touched by
        # this transformation.
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {}
            for tag, group_expr in self._solver_groups.items():
                g = group_expr
                for cond in conditions:
                    g = _apply_one(g, cond)
                if g == group_expr:
                    new_solver_groups[tag] = group_expr

        return Expression(new_expr, self.name,
                          term_tags=new_tags, tag_order=self._tag_order,
                          solver_groups=new_solver_groups)

    def simplify(self):
        """Return a new Expression with sympy simplification applied."""
        simplified = sp.simplify(self.expr)
        new_groups = None
        # Cleanup: keep only tags whose term is still present post-simplify.
        current = set(Add.make_args(sp.expand(simplified)))
        new_tags = {t: name for t, name in self._term_tags.items() if t in current}
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {t: g for t, g in self._solver_groups.items()
                                 if sp.simplify(g) == g}
        return Expression(simplified, self.name,
                          term_tags=new_tags, tag_order=self._tag_order,
                          solver_groups=new_solver_groups)

    def expand(self):
        """Return a new Expression with sympy expand applied."""
        expanded = sp.expand(self.expr)
        current = set(Add.make_args(expanded))
        new_tags = {t: name for t, name in self._term_tags.items() if t in current}
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {t: g for t, g in self._solver_groups.items()
                                 if sp.expand(g) == g}
        return Expression(expanded, self.name,
                          term_tags=new_tags, tag_order=self._tag_order,
                          solver_groups=new_solver_groups)

    def depth_integrate(self, lower, upper, var, method="auto"):
        """
        Depth-integrate this expression over [lower, upper] w.r.t. var.

        Parameters
        ----------
        lower, upper : sympy expressions for the integration bounds
            (typically b and b+H, both functions of t and x)
        var : Symbol
            The vertical coordinate (z)
        method : str
            'auto'    : detect derivative direction and choose method
            'leibniz' : pull horizontal derivative outside:
                        int df/dx dz = d/dx[int f dz] - f(upper)*d(upper)/dx + f(lower)*d(lower)/dx
            'fundamental_theorem' : for vertical derivatives:
                        int df/dz dz = f(upper) - f(lower)
            'direct'  : keep as Integral(expr, (var, lower, upper))

        Returns
        -------
        DepthIntegralResult with .volume, .boundary_upper, .boundary_lower
        or a plain Expression for 'direct' and 'fundamental_theorem'.
        """
        expr = self.expr

        if method == "auto":
            # Detect: does the expression contain d/dz?
            inner_z, coeff_z = _extract_derivative(expr, var)
            if inner_z is not None:
                method = "fundamental_theorem"
            else:
                # Check for d/dx (or any horizontal derivative)
                method = "direct"
                for s in expr.free_symbols:
                    if s != var:
                        inner_h, coeff_h = _extract_derivative(expr, s)
                        if inner_h is not None:
                            method = "leibniz"
                            break

        if method == "fundamental_theorem":
            # int df/dz dz = f(upper) - f(lower)
            inner, coeff = _extract_derivative(expr, var)
            if inner is None:
                raise ValueError(
                    f"No Derivative w.r.t. {var} found for fundamental theorem: {expr}"
                )
            f = coeff * inner
            f_upper = sp.Subs(f, var, upper)
            f_lower = sp.Subs(f, var, lower)
            return DepthIntegralResult(
                volume=Expression(S.Zero, f"ft_volume({self.name})"),
                boundary_upper=Expression(f_upper, f"ft_upper({self.name})"),
                boundary_lower=Expression(-f_lower, f"ft_lower({self.name})"),
            )

        elif method == "leibniz":
            # int df/dx dz = d/dx[int f dz] - f(upper)*d(upper)/dx + f(lower)*d(lower)/dx
            for s in list(expr.free_symbols) + [Symbol("x"), Symbol("t")]:
                if s == var:
                    continue
                inner, coeff = _extract_derivative(expr, s)
                if inner is not None:
                    break
            else:
                raise ValueError(f"No horizontal Derivative found for Leibniz: {expr}")

            # Volume: d/dx[int f dz]
            int_f = Integral(coeff * inner, (var, lower, upper))
            volume = Derivative(int_f, s)

            # Boundary terms: use Subs to keep evaluation explicit
            f = coeff * inner
            f_at_upper = sp.Subs(f, var, upper)
            f_at_lower = sp.Subs(f, var, lower)
            bnd_upper = -f_at_upper * Derivative(upper, s)
            bnd_lower = f_at_lower * Derivative(lower, s)

            return DepthIntegralResult(
                volume=Expression(volume, f"leibniz_volume({self.name})"),
                boundary_upper=Expression(bnd_upper, f"leibniz_upper({self.name})"),
                boundary_lower=Expression(bnd_lower, f"leibniz_lower({self.name})"),
            )

        else:  # direct
            return Expression(
                Integral(expr, (var, lower, upper)),
                f"integral({self.name})",
            )

    def subs(self, *args, **kwargs):
        # Per-term substitution with tag inheritance, then cleanup.
        new_tags = {}
        pieces = []
        for term in Add.make_args(sp.expand(self.expr)):
            if term == S.Zero:
                continue
            parent_tag = self._term_tags.get(term)
            sub = term.subs(*args, **kwargs)
            for piece in Add.make_args(sp.expand(sub)):
                if piece == S.Zero:
                    continue
                pieces.append(piece)
                if parent_tag is not None:
                    new_tags[piece] = parent_tag
        new_expr = sum(pieces, S.Zero)
        current = set(Add.make_args(sp.expand(new_expr)))
        new_tags = {t: name for t, name in new_tags.items() if t in current}
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {t: g for t, g in self._solver_groups.items()
                                 if g.subs(*args, **kwargs) == g}
        return Expression(new_expr, self.name,
                          term_tags=new_tags, tag_order=self._tag_order,
                          solver_groups=new_solver_groups)

    def simplify(self):
        """Simplify: expand + cancel, preserving Integral and Derivative(Integral) terms."""
        simplified = _simplify_preserve_integrals(self.expr)
        current = set(Add.make_args(sp.expand(simplified)))
        new_tags = {t: name for t, name in self._term_tags.items() if t in current}
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {t: g for t, g in self._solver_groups.items()
                                 if _simplify_preserve_integrals(g) == g}
        out = Expression(simplified, self.name,
                         term_tags=new_tags, tag_order=self._tag_order,
                         solver_groups=new_solver_groups)
        rel = getattr(self, "_as_relation", None)
        if rel is not None:
            out._as_relation = rel
        return out

    def expand(self):
        expanded = sp.expand(self.expr)
        current = set(Add.make_args(expanded))
        new_tags = {t: name for t, name in self._term_tags.items() if t in current}
        new_solver_groups = None
        if self._solver_groups:
            new_solver_groups = {t: g for t, g in self._solver_groups.items()
                                 if sp.expand(g) == g}
        out = Expression(expanded, self.name,
                         term_tags=new_tags, tag_order=self._tag_order,
                         solver_groups=new_solver_groups)
        rel = getattr(self, "_as_relation", None)
        if rel is not None:
            out._as_relation = rel
        return out

    def doit(self):
        return Expression(self.expr.doit(), self.name)

    def has(self, *args):
        return self.expr.has(*args)

    @property
    def free_symbols(self):
        return self.expr.free_symbols

    def __repr__(self):
        short = str(self.expr)
        if len(short) > 80:
            short = short[:77] + "..."
        label = f" ({self.name})" if self.name else ""
        return f"Expression{label}: {short}"

    def _repr_latex_(self):
        return f"${sp.latex(self.expr)}$"

    def _sympy_(self):
        return self.expr

    def __eq__(self, other):
        if isinstance(other, Expression):
            return sp.simplify(self.expr - other.expr) == 0
        return sp.simplify(self.expr - other) == 0

    # ------------------------------------------------------------------
    # Per-term operations
    # ------------------------------------------------------------------

    def map(self, fn):
        """Apply fn to each term, reassemble into a single Expression.

        fn receives an Expression (single term) and must return either
        an Expression or a DepthIntegralResult.  DepthIntegralResults are
        assembled (volume + boundaries) before summing.

        Example:
            integrated = expr.map(lambda t: t.depth_integrate(b, eta, z))
        """
        results = []
        for term in self.terms:
            r = fn(term)
            if isinstance(r, DepthIntegralResult):
                results.append(r.assemble())
            elif isinstance(r, Expression):
                results.append(r)
            else:
                results.append(Expression(r, term.name))
        return sum(results, Expression(S.Zero))

    def map_with_bcs(self, fn, bcs):
        """Like map(), but collects boundary terms and applies BCs globally.

        This is the correct way to depth-integrate a full equation:
        boundary terms from ALL terms are combined first, then BCs are
        applied once (so cross-term cancellations happen properly).

        Parameters
        ----------
        fn : callable
            Applied to each term.  Must return DepthIntegralResult or Expression.
        bcs : list of Relation
            Kinematic BCs etc. applied to the combined boundary expression.

        Returns
        -------
        Expression
            The fully depth-integrated equation with BCs applied.
        """
        total_volume = Expression(S.Zero)
        total_boundary = Expression(S.Zero)

        for term in self.terms:
            r = fn(term)
            if isinstance(r, DepthIntegralResult):
                total_volume = total_volume + r.volume
                total_boundary = total_boundary + r.boundary_upper + r.boundary_lower
            elif isinstance(r, Expression):
                total_volume = total_volume + r
            else:
                total_volume = total_volume + Expression(r)

        # Apply all BCs to the combined boundary
        bnd = total_boundary
        for bc in bcs:
            bnd = bnd.apply(bc)

        # Simplify: evaluate derivatives (to combine d(H+b)/dt - db/dt → dH/dt)
        # but preserve Integrals (don't re-apply Leibniz)
        result_expr = (total_volume + bnd).expr
        result_expr = _simplify_derivatives_only(result_expr)

        return Expression(result_expr, self.name)

    # ------------------------------------------------------------------
    # Term classification
    # ------------------------------------------------------------------

    def classify(self, t=None, x=None, z=None):
        """Classify each term by its role in the PDE.

        Returns a dict: {role: Expression} where role is one of:
        'temporal', 'convective', 'diffusive', 'source'.

        Detection rules:
        - Has d/dt → temporal
        - Has d/dx (first-order) of a product → convective flux
        - Has d²/dz² or d/dz of d/dz → diffusive
        - Otherwise → source (algebraic)
        """
        roles = {
            "temporal": [],
            "convective": [],
            "diffusive": [],
            "source": [],
        }

        for term in self.terms:
            e = term.expr
            classified = False

            # Check temporal
            if t is not None and e.has(Derivative) and any(
                t in d.variables for d in e.atoms(Derivative)
            ):
                roles["temporal"].append(term)
                classified = True

            # Check diffusive (second derivatives in z)
            if not classified and z is not None:
                for d in e.atoms(Derivative):
                    if d.variables.count(z) >= 2:
                        roles["diffusive"].append(term)
                        classified = True
                        break

            # Check convective (first derivative in x)
            if not classified and x is not None:
                for d in e.atoms(Derivative):
                    if x in d.variables and d.variables.count(x) == 1:
                        roles["convective"].append(term)
                        classified = True
                        break

            if not classified:
                roles["source"].append(term)

        return {k: Expression(sum((t.expr for t in v), S.Zero), k)
                for k, v in roles.items() if v}

    @property
    def temporal(self):
        """View: only temporal (d/dt) terms."""
        c = self.classify(t=Symbol("t"))
        return c.get("temporal", Expression(S.Zero))

    @property
    def convective(self):
        """View: only convective flux (d/dx) terms."""
        c = self.classify(x=Symbol("x"))
        return c.get("convective", Expression(S.Zero))

    # ------------------------------------------------------------------
    # Basis projection
    # ------------------------------------------------------------------

    def project_onto_basis(self, basis, level, field_map, z_var,
                           lower=None, upper=None, test_mode=None):
        """
        Project a depth-integrated equation onto a polynomial basis.

        Replaces every ``Integral(f(u,...), (z, b, eta))`` by substituting
        the basis expansion ``u(z) = sum alpha_k phi_k(zeta)`` and evaluating
        the resulting integrals using the ``SymbolicIntegrator``.

        If ``test_mode`` is an integer, multiplies each integral by the test
        function phi_{test_mode}(zeta) before evaluating (Galerkin projection
        for a specific mode).  If ``test_mode=None``, returns the scalar
        integral (e.g. for the mass equation where no test function is needed).

        Parameters
        ----------
        basis : Basisfunction class (e.g. Legendre_shifted)
        level : int
        field_map : dict
            Maps the original Function name to a list of SymPy Symbols
            for the basis coefficients.
            Example: {'u': [alpha_0, alpha_1, alpha_2]}
        z_var : Symbol
            The vertical coordinate (z) that appears in the integrals.
        lower, upper : sympy expressions (optional)
            The integration bounds (b, eta).  If None, detected from
            the first Integral found.
        test_mode : int or None
            If int, project onto test function phi_{test_mode}.

        Returns
        -------
        Expression
            With all depth integrals replaced by basis matrix products.
        """
        from zoomy_core.model.models.symbolic_integrator import SymbolicIntegrator
        from zoomy_core.model.models.projected_model import get_cached_matrices

        basis_obj = basis(level=level)
        integrator = SymbolicIntegrator(basis_obj)
        matrices = get_cached_matrices(basis, level, integrator)

        M = matrices["M"]
        A = matrices["A"]
        n = level + 1
        zeta = Symbol("zeta")
        c_mean = basis_obj.mean_coefficients()

        def _replace_integral(expr):
            """Walk the expression tree, replacing Integral nodes."""
            if not isinstance(expr, sp.Basic):
                return expr

            if isinstance(expr, Integral):
                integrand = expr.args[0]
                limits = expr.args[1]
                int_var = limits[0]

                if int_var != z_var:
                    return expr

                lo, hi = limits[1], limits[2]

                # Transform to zeta-space:
                # z = lo + (hi - lo)*zeta, dz = (hi-lo)*dzeta
                h_expr = hi - lo  # water depth H
                integrand_zeta = integrand.subs(int_var, lo + h_expr * zeta)

                # Substitute basis expansion for each field
                for fname, coeffs in field_map.items():
                    expansion = sum(
                        coeffs[k] * basis_obj.eval(k, zeta)
                        for k in range(min(len(coeffs), n))
                    )
                    # Find all applications of this function and replace
                    for atom in integrand_zeta.atoms(sp.Function):
                        if atom.func.__name__ == fname:
                            integrand_zeta = integrand_zeta.subs(atom, expansion)

                # Multiply by Jacobian H and test function
                integrand_final = h_expr * integrand_zeta
                if test_mode is not None:
                    integrand_final *= basis_obj.eval(test_mode, zeta)

                # Evaluate the integral using the integrator
                result = integrator.integrate(
                    sp.expand(integrand_final) * basis_obj.weight(zeta),
                    zeta,
                    tuple(basis_obj.bounds()),
                )
                return result

            # Recurse into Derivative, Mul, Add, etc.
            if isinstance(expr, Derivative):
                new_expr = _replace_integral(expr.args[0])
                return Derivative(new_expr, *expr.args[1:])

            if expr.args:
                new_args = [_replace_integral(a) for a in expr.args]
                return expr.func(*new_args)

            return expr

        result = _replace_integral(self.expr)
        return Expression(result, f"projected({self.name})")

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def latex(self, strip_args=False, multiline=False):
        """LaTeX representation.

        Parameters
        ----------
        strip_args : bool
            ``u(t,x,z)`` → ``u``. Partial derivatives preserved.
        multiline : bool
            Render as ``\\begin{aligned}`` with one group per line
            (requires ``term_groups``).
        """
        printer = _StripArgsLatexPrinter() if strip_args else None

        def _tex(expr):
            return printer.doprint(expr) if printer else sp.latex(expr)

        if multiline and self._term_tags:
            # Group current-expression terms by their tag.
            current_terms = list(Add.make_args(sp.expand(self.expr)))
            by_tag = {name: [] for name in self._tag_order}
            untagged = []
            for term in current_terms:
                name = self._term_tags.get(term)
                if name is None:
                    untagged.append(term)
                else:
                    by_tag.setdefault(name, []).append(term)
            lines = []
            first = True
            for role in list(by_tag.keys()):
                terms = by_tag.get(role, [])
                if not terms:
                    continue
                g = sum(terms, S.Zero)
                if g == S.Zero:
                    continue
                tex = _tex(g)
                if first:
                    lines.append(f"  & \\underbrace{{{tex}}}_{{{role}}}")
                    first = False
                elif tex.startswith("-"):
                    lines.append(f"  & \\underbrace{{{tex}}}_{{{role}}}")
                else:
                    lines.append(f"  & + \\underbrace{{{tex}}}_{{{role}}}")
            if untagged:
                g = sum(untagged, S.Zero)
                if g != S.Zero:
                    tex = _tex(g)
                    prefix = "" if first or tex.startswith("-") else "+ "
                    lines.append(f"  & {prefix}\\underbrace{{{tex}}}_{{untagged}}")
            return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n  &= 0\n\\end{aligned}"

        if multiline and not self._term_groups:
            # No term groups — split by additive terms for multiline rendering
            terms = Add.make_args(self.expr)
            if len(terms) > 1:
                lines = []
                for i, term in enumerate(terms):
                    tex = _tex(term)
                    if i == 0:
                        lines.append(f"  & {tex}")
                    elif tex.startswith("-"):
                        lines.append(f"  & {tex}")
                    else:
                        lines.append(f"  & + {tex}")
                return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n  &= 0\n\\end{aligned}"

        if self._term_groups:
            # Render in group order (single line) — preserves physical ordering
            parts = []
            for role, g in self._term_groups.items():
                if g == S.Zero:
                    continue
                tex = _tex(g)
                if parts and not tex.startswith("-"):
                    tex = "+ " + tex
                parts.append(tex)
            return " ".join(parts)

        return _tex(self.expr)

    def describe(self, header=True, final_equation=True, parameters=False,
                 strip_args=True):
        """Composable description of this expression.

        Returns a ``Description`` that renders as markdown in Jupyter.

        Parameters
        ----------
        header : bool
            Show expression name + term count.
        final_equation : bool
            Show the symbolic equation.
        parameters : bool
            List free symbols.
        strip_args : bool
            Display ``u`` instead of ``u(t, x, z)``.
        """
        from zoomy_core.misc.description import Description

        parts = []

        if header:
            parts.append(f"**{self.name}** ({len(self)} terms)")

        if final_equation:
            # Match System.describe: use multiline underbrace rendering when
            # term_groups are populated (otherwise model.describe and
            # model.<eq>.describe show the same equation differently).
            tex = self.latex(strip_args=strip_args,
                             multiline=bool(self._term_groups))
            # latex(multiline=True) already emits a trailing "&= 0" inside the
            # aligned block, so in that case we don't append another "= 0".
            if self._term_groups:
                parts.append(f"\n$$\n{tex}\n$$")
            else:
                parts.append(f"\n$$\n{tex} = 0\n$$")

        if parameters:
            from sympy import Symbol
            syms = sorted([s for s in self.expr.free_symbols
                          if isinstance(s, Symbol) and not s.is_Function],
                         key=str)
            if syms:
                sym_str = ", ".join(f"${sp.latex(s)}$" for s in syms)
                parts.append(f"\n**Parameters:** {sym_str}")

        return Description("\n".join(parts))


from sympy.printing.latex import LatexPrinter as _LatexPrinter


class _StripArgsLatexPrinter(_LatexPrinter):
    """LaTeX printer that renders function calls cleanly:

    - ``u(t,x,z)`` → ``u``  (standard args stripped)
    - ``u(t,x,b+h)`` → ``u|_{z=b+h}``  (boundary evaluation shown)
    - ``Subs(u(t,x,z), z, b)`` → ``u|_{z=b}``  (via sympy Subs printing)

    Horizontal functions (b, h, p_atm with fewer args) always stripped.
    """

    # The vertical coordinate symbol — functions with this as last arg are "standard"
    _z = sp.Symbol("z", real=True)

    def _print_Function(self, expr, exp=None):
        name = expr.func.__name__
        tex = self._deal_with_super_sub(name)
        args = expr.args

        # Check if this is a 3D function (u, w, tau_xx, ...) evaluated at a boundary
        # Heuristic: if last arg is NOT z and the function has 3+ args, it's a boundary eval
        if len(args) >= 3 and args[-1] != self._z:
            z_val = args[-1]
            z_tex = self.doprint(z_val)
            tex = r"\left. %s \right|_{\substack{ z=%s }}" % (tex, z_tex)

        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex


class DepthIntegralResult:
    """
    Result of depth-integrating a term over [lower, upper].

    Attributes:
        volume:          the volume integral (Expression)
        boundary_upper:  boundary term at z=upper (Expression)
        boundary_lower:  boundary term at z=lower (Expression)

    The full integral = volume + boundary_upper - boundary_lower.
    Kinematic BCs can be applied to the boundary terms via .apply_bcs().
    """

    def __init__(self, volume, boundary_upper, boundary_lower):
        self.volume = volume
        self.boundary_upper = boundary_upper
        self.boundary_lower = boundary_lower

    def apply_bcs(self, bc_lower=None, bc_upper=None):
        upper = self.boundary_upper
        lower = self.boundary_lower
        if bc_upper is not None:
            upper = upper.apply(bc_upper)
        if bc_lower is not None:
            lower = lower.apply(bc_lower)
        return DepthIntegralResult(self.volume, upper, lower)

    def assemble(self):
        """Combine all terms: volume + boundary_upper + boundary_lower.

        The sign convention is that each component already carries its
        correct sign.  For Leibniz: upper = -f(eta)*d(eta)/dx,
        lower = +f(b)*db/dx.  For fundamental theorem: upper = +f(eta),
        lower = -f(b).
        """
        return self.volume + self.boundary_upper + self.boundary_lower

    def __repr__(self):
        return (f"DepthIntegralResult(\n"
                f"  volume={self.volume},\n"
                f"  upper={self.boundary_upper},\n"
                f"  lower={self.boundary_lower}\n)")


class IBPResult:
    """
    Structured result from integration by parts.

    Attributes:
        integrate: the volume integral (Expression)
        boundary_upper: boundary term at upper limit (Expression)
        boundary_lower: boundary term at lower limit (Expression)
    """

    def __init__(self, integrate, boundary_upper, boundary_lower):
        self.integrate = integrate
        self.boundary_upper = boundary_upper
        self.boundary_lower = boundary_lower

    def apply_bcs(self, bc_lower=None, bc_upper=None):
        upper = self.boundary_upper
        lower = self.boundary_lower
        if bc_upper is not None:
            upper = upper.apply(bc_upper)
        if bc_lower is not None:
            lower = lower.apply(bc_lower)
        return IBPResult(self.integrate, upper, lower)

    def assemble(self):
        return self.integrate + self.boundary_upper - self.boundary_lower

    def __repr__(self):
        return (f"IBPResult(\n"
                f"  integrate={self.integrate},\n"
                f"  upper={self.boundary_upper},\n"
                f"  lower={self.boundary_lower}\n)")


# ---------------------------------------------------------------------------
# Relation: lhs = rhs substitution rules
# ---------------------------------------------------------------------------

class Relation(SymbolicBase):
    """
    A symbolic relation: one or more substitution rules lhs_i = rhs_i.

    Used as base for Assumption and Material. Can be applied to Expressions
    via expr.apply(relation), which calls relation.apply_to(expr).

    Displays as a system of equations in notebooks.
    """

    def __init__(self, substitutions, name=""):
        """
        Parameters
        ----------
        substitutions : dict {lhs_expr: rhs_expr} or list of (lhs, rhs) tuples
        name : str
        """
        super().__init__(name)
        if isinstance(substitutions, dict):
            self.subs_map = dict(substitutions)
        elif isinstance(substitutions, (list, tuple)):
            self.subs_map = dict(substitutions)
        else:
            raise TypeError("substitutions must be a dict or list of (lhs, rhs) tuples")

    def apply_to(self, expr):
        """Substitute all lhs -> rhs in the given SymPy expression."""
        result = expr
        for lhs, rhs in self.subs_map.items():
            result = result.subs(lhs, rhs)
        return result

    def __len__(self):
        return len(self.subs_map)

    def __repr__(self):
        lines = [f"{self.__class__.__name__}(name={self.name!r}, {len(self)} rules):"]
        for lhs, rhs in self.subs_map.items():
            lines.append(f"  {lhs} = {rhs}")
        return "\n".join(lines)

    def _repr_latex_(self):
        lines = []
        for lhs, rhs in self.subs_map.items():
            lines.append(f"{sp.latex(lhs)} = {sp.latex(rhs)}")
        body = " \\\\ ".join(lines)
        return f"$\\begin{{aligned}} {body} \\end{{aligned}}$"


class Assumption(Relation):
    """Physical assumption (kinematic BC, hydrostatic, etc.)."""
    pass


class Material(Relation):
    """Constitutive model (Newtonian, inviscid, etc.)."""
    pass


# ---------------------------------------------------------------------------
# Operation: callable transformation applied to all equations
# ---------------------------------------------------------------------------

class Operation(SymbolicBase):
    """A structural transformation on a tree node.

    A node is either an :class:`Expression` (leaf) or a ``Zstruct`` of
    nodes (intermediate).  The unified entry point is ``__call__(node)``:

    - Subclasses override :meth:`_apply_leaf` to transform a leaf
      Expression.  The default ``__call__`` dispatches over the tree,
      recursing into intermediate Zstructs.
    - Rank-changing Operations (``Multiply`` with ``outer=True`` on a
      leaf, or contraction on a Zstruct) override ``__call__`` directly
      and may return a node of different shape than they received.

    Every Operation carries ``name`` + ``description`` so
    ``System.apply`` can record what was done in the derivation history.
    """

    def __init__(self, name="", description=None):
        super().__init__(name)
        self.description = description or name

    def __call__(self, node):
        """Dispatch: leaf Expression → ``_apply_leaf``; Zstruct → recurse."""
        from zoomy_core.misc.misc import Zstruct
        if isinstance(node, Expression):
            return self._apply_leaf(node)
        if isinstance(node, Zstruct):
            out = Zstruct()
            for key in node._filter_dict():
                setattr(out, key, self(getattr(node, key)))
            return out
        raise TypeError(
            f"{type(self).__name__} got an unsupported node type: {type(node).__name__}"
        )

    def _apply_leaf(self, expression):
        """Transform a leaf Expression → Expression (or Zstruct for rank-changers).

        Default: delegate to :meth:`_leaf_sp` and re-wrap the result,
        inheriting the leaf's name, tags, and ``_as_relation``.
        """
        new_sp = self._leaf_sp(expression.expr)
        if new_sp is expression.expr:
            return expression
        out = Expression(
            new_sp, expression.name,
            term_tags=dict(expression._term_tags),
            tag_order=list(expression._tag_order),
            solver_groups=expression._solver_groups,
        )
        rel = getattr(expression, "_as_relation", None)
        if rel is not None:
            out._as_relation = rel
        return out

    def _leaf_sp(self, sp_expr):
        """Transform a raw sympy expression. Override for simple ops.

        Subclasses override *either* this (for per-term sympy transforms)
        *or* :meth:`_apply_leaf` (for ops that need the Expression's
        ``.terms`` / tags), not both.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _leaf_sp or override _apply_leaf"
        )

    # Ops that turn a leaf Expression into an intermediate Zstruct
    # (``Multiply(outer=True)`` etc.) set this to ``True`` so tree-level
    # dispatchers (``_NodeProxy.apply`` / ``System.apply``) know to call
    # ``op(node)`` directly and replace the node, instead of routing
    # through ``Expression.apply``'s per-term loop.
    rank_changes_leaf = False

    # Ops that need to see the whole leaf at once — i.e. they look at
    # cross-term relationships rather than acting per-additive-term —
    # set this to ``True``.  ``Recombine`` is the canonical example: to
    # fold ``α·∂_v f + f·∂_v α → ∂_v(α·f)`` we need to see both terms
    # simultaneously, which the default per-term dispatch hides.
    # Dispatchers call ``op(node)`` directly when this is set, just
    # like ``rank_changes_leaf``, but the result must remain an
    # Expression (not a rank-changing Zstruct).
    whole_leaf_op = False

    def _repr_latex_(self):
        return ""


class Multiply(Operation):
    """Multiply an Expression (or tree of Expressions) by a factor.

    Default behaviour (``outer=False``) — *rank-preserving*:
      Scales every term of a leaf Expression by ``factor``.  Tags follow
      the terms through the per-term dispatch in
      :meth:`Expression.apply`, so ``model.apply(Multiply(1/rho))``
      scales the whole system and keeps physical tags intact.

    ``outer=True`` — *rank-changing* (leaf → Zstruct):
      ``factor`` must be a ``Zstruct`` of scalar factors
      (e.g. test functions ``phi_0, phi_1, ...``).  The leaf Expression
      is promoted to a ``Zstruct`` with one child Expression per factor,
      keyed ``test_<index>``.  Each child is ``leaf.expr * factor_k``.

      Per-term tag inheritance: each parent tag is re-keyed through the
      multiplication (same rule as :meth:`Expression.apply`), so physical
      tags follow into every ``test_<k>`` child.

    Usage::

        model.apply(Multiply(1 / state.rho))                 # scalar scaling
        phis = Zstruct(**{f"phi_{k}": ... for k in range(N)})
        model.momentum.x.apply(Multiply(phis, outer=True))   # Galerkin test
    """

    def __init__(self, factor, outer=False, name=None, description=None):
        from zoomy_core.misc.misc import Zstruct
        self._outer = outer
        if outer:
            if not isinstance(factor, Zstruct):
                raise TypeError(
                    "Multiply(outer=True) requires a Zstruct of factors "
                    f"(got {type(factor).__name__})."
                )
            self._factor = factor
            default_desc = (
                f"outer-multiply by {len(factor._filter_dict())} test function(s)"
            )
        else:
            f_sp = factor.expr if isinstance(factor, Expression) else sp.sympify(factor)
            self._factor = f_sp
            default_desc = f"multiply by {f_sp}"
        super().__init__(
            name=name or "multiply",
            description=description or default_desc,
        )

    @property
    def rank_changes_leaf(self):
        return self._outer

    def _leaf_sp(self, sp_expr):
        if self._outer:
            raise NotImplementedError(
                "Multiply(outer=True) is rank-changing; invoke via "
                "node.apply() so the leaf can be replaced with a Zstruct."
            )
        return sp_expr * self._factor

    def _apply_leaf(self, expression):
        if not self._outer:
            return super()._apply_leaf(expression)
        from zoomy_core.misc.misc import Zstruct
        out = Zstruct()
        for i, key in enumerate(self._factor._filter_dict()):
            f = getattr(self._factor, key)
            f_sp = f.expr if isinstance(f, Expression) else sp.sympify(f)
            # Deep-clone the parent (expr + tags + solver_groups + relation),
            # rename to ``<parent>_<index>``, then pointwise scalar-multiply
            # by ``f_sp``.  The scalar ``Multiply`` routes through
            # ``Expression.apply``'s per-term loop, so tag re-keying against
            # the new terms is handled by the existing single-source-of-truth
            # path — no parallel logic here.
            child_name = (f"{expression.name}_{i}" if expression.name
                          else f"test_{i}")
            clone = Expression(
                expression.expr, child_name,
                term_tags=dict(expression._term_tags),
                tag_order=list(expression._tag_order),
                solver_groups=expression._solver_groups,
            )
            rel = getattr(expression, "_as_relation", None)
            if rel is not None:
                clone._as_relation = dict(rel)
            child = clone.apply(Multiply(f_sp))
            setattr(out, f"test_{i}", child)
        return out


class DepthIntegrate(Operation):
    """Depth-integrate all equations over [b, b+h] w.r.t. z.

    Applies Leibniz rule and fundamental theorem term-by-term.
    Boundary values (w at z=b, u at z=b+h, etc.) remain as ``Subs``
    objects.  Apply ``ApplyKinematicBCs`` to see the cancellations.
    """

    def __init__(self, state):
        super().__init__(
            name="depth_integrate",
            description="Depth integration over [b, b+h] (Leibniz rule)",
        )
        self._state = state

    def _apply_leaf(self, expression):
        s = self._state
        return expression.map(lambda t: t.depth_integrate(s.b, s.eta, s.z))

    def _repr_latex_(self):
        s = self._state
        return (
            f"$\\int_{{{sp.latex(s.b)}}}^{{{sp.latex(s.eta)}}} "
            f"(\\cdot)\\, d{sp.latex(s.z)}$"
        )


class ApplyKinematicBCs(Operation):
    """Apply kinematic BCs globally to combined boundary terms.

    Evaluates ``Subs`` boundary values, applies kinematic BCs at
    surface and bottom, and simplifies. The Leibniz boundary u-terms
    cancel with the fundamental theorem w-terms.

    Must be applied immediately after ``DepthIntegrate``.
    """

    def __init__(self, state):
        super().__init__(
            name="kinematic_bcs",
            description="Kinematic BCs (surface + bottom): w = u·∂b/∂x + ∂b/∂t",
        )
        self._kbc_s = KinematicBCSurface(state)
        self._kbc_b = KinematicBCBottom(state)

    def _leaf_sp(self, expr):
        # Evaluate only Subs objects (NOT Derivative(Integral))
        if expr.has(sp.Subs):
            subs_map = {s: s.doit() for s in expr.atoms(sp.Subs)}
            expr = expr.subs(subs_map)
        # Apply both BCs
        for bc in [self._kbc_s, self._kbc_b]:
            expr = bc.apply_to(expr)
        # Simplify (cancels d(b+h)/dt - db/dt → dh/dt)
        return _simplify_preserve_integrals(expr)

    def _repr_latex_(self):
        parts = [self._kbc_s._repr_latex_(), self._kbc_b._repr_latex_()]
        return " \\\\ ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# FullINS: 3D INS equations built from a StateSpace
# ---------------------------------------------------------------------------

_integrate_cache: dict = {}
_integrate_cache_stats = {"hits": 0, "misses": 0}


def _cached_integrate(integrand, limits):
    """Cached wrapper around ``sympy.integrate(integrand, limits)``.

    Keyed on ``(integrand, limits)``; both are hashable sympy objects.
    Across a full SME derivation — many leaves, many ``test_k`` clones
    — most ``∫_0^1 polynomial(ζ) dζ`` integrands appear repeatedly, so a
    hash cache cuts ``sympy.integrate`` work by an order of magnitude.

    Returns the original (possibly-unevaluated) ``Integral`` on failure;
    caller can detect "not evaluated" via ``isinstance(..., Integral)``.
    """
    key = (integrand, limits)
    hit = _integrate_cache.get(key)
    if hit is not None:
        _integrate_cache_stats["hits"] += 1
        return hit
    _integrate_cache_stats["misses"] += 1
    try:
        result = sp.integrate(integrand, limits)
    except Exception:
        result = Integral(integrand, limits)
    _integrate_cache[key] = result
    return result


def integrate_cache_stats():
    """Return ``(hits, misses)`` for the cached-integrate helper."""
    return _integrate_cache_stats["hits"], _integrate_cache_stats["misses"]


def integrate_cache_clear():
    """Drop all cached integrals and reset stats (useful between runs)."""
    _integrate_cache.clear()
    _integrate_cache_stats["hits"] = 0
    _integrate_cache_stats["misses"] = 0


class EvaluateIntegrals(Operation):
    """Run ``sympy.integrate`` on every ``Integral`` node in the expression.

    After :class:`ZetaTransform` + basis expansion the volume integrals
    look like ``Integral(polynomial_in_zeta, (zeta, 0, 1))``.  This op
    collapses each such integral to a scalar.

    Walks ``Derivative(Integral(...), x)`` too: the inner Integral is
    evaluated first, then the outer Derivative survives symbolically.

    Uses :func:`_cached_integrate` so repeated integrands across leaves
    (every ``test_k`` clone after ``Multiply(basis.phi, outer=True)``)
    cost one ``sympy.integrate`` call each, not one-per-leaf.

    Leaves unevaluatable Integrals untouched (e.g. non-polynomial ones
    or ones whose result would need a closed form sympy can't find).
    """

    def __init__(self, state=None, name="evaluate_integrals",
                 description=None):
        super().__init__(
            name=name,
            description=description or "Evaluate all Integral nodes via sympy.integrate",
        )

    def _leaf_sp(self, expr):
        from sympy import Integral, Derivative

        def _walk(e):
            if isinstance(e, Integral):
                # Recurse into nested integrands first.
                integrand = _walk(e.args[0])
                limits = e.args[1]
                evaluated = _cached_integrate(integrand, limits)
                if not isinstance(evaluated, Integral):
                    return evaluated
                return Integral(integrand, limits, *e.args[2:])
            if isinstance(e, Derivative):
                inner = _walk(e.args[0])
                if inner != e.args[0]:
                    return Derivative(inner, *e.args[1:])
                return e
            if hasattr(e, "args") and e.args:
                new_args = [_walk(a) for a in e.args]
                if any(n != o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        # Alternate integrate / resolve-Subs until fixpoint: each
        # integration may uncover ``Subs`` nodes whose inner just got
        # cleaner, and each resolve may expose new ``Integral`` nodes
        # (outer integrals whose integrand was previously a ``Subs``).
        # The loop usually converges in 2 iterations but the bound is
        # generous to handle deeply nested cases.
        result = expr
        for _ in range(6):
            prev = result
            result = _walk(result)
            result = _resolve_subs_safe(result)
            if result == prev:
                break
        return result


class SimplifyIntegrals(Operation):
    """Evaluate integrals with constant integrand and remove zero integrals.

    - ``∫ 0 dz → 0``
    - ``∫ c dz → c·h`` (if integrand has no z-dependence)
    - ``∂/∂x ∫ 0 dz → 0``
    """

    def __init__(self, state):
        super().__init__(
            name="simplify_integrals",
            description="Evaluate constant/zero integrals",
        )

    def _leaf_sp(self, expr):
        from sympy import Integral, Derivative

        def _simplify_int(e):
            if isinstance(e, Integral):
                integrand = e.args[0]
                limits = e.args[1]
                var = limits[0]
                lower, upper = limits[1], limits[2]
                if integrand == S.Zero:
                    return S.Zero
                if not integrand.has(var):
                    return integrand * (upper - lower)
                return e
            if isinstance(e, Derivative):
                inner = _simplify_int(e.args[0])
                if inner == S.Zero:
                    return S.Zero
                if inner != e.args[0]:
                    return Derivative(inner, *e.args[1:])
                return e
            if e.args:
                new_args = [_simplify_int(a) for a in e.args]
                if any(n != o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        return _simplify_int(expr)


class Integrate(Operation):
    """Integrate each additive term of an equation w.r.t. ``var`` over [lower, upper].

    Per-term strategy is driven by ``method``:

    * ``"auto"`` (default) — detect the outermost derivative direction
      per term and pick Leibniz / fundamental theorem / direct accordingly.
    * ``"leibniz"`` — force the Leibniz rule (for terms containing
      ``∂/∂x_i`` where ``x_i != var``):
      ``∫ ∂_x f dvar = ∂_x [∫ f dvar]  − f|_upper · ∂_x upper  + f|_lower · ∂_x lower``.
    * ``"fundamental_theorem"`` — for terms containing ``∂/∂var`` as the
      outermost derivative: ``∫ ∂_var f dvar = f|_upper − f|_lower``.
    * ``"direct"`` — keep as an unevaluated ``Integral(expr, (var, lower, upper))``.
    * ``"analytical"`` — run ``sympy.integrate`` on the whole expression
      (what the original small ``Integrate`` did; useful for z-momentum
      analytic integration when the integrand has no horizontal derivatives).

    Boundary terms produced by Leibniz / fundamental-theorem are kept as
    ``Subs(f, var, bound)`` expressions — no kinematic BCs are applied here.
    Resolve them later via an explicit ``.apply({Subs-pattern: value})`` or
    a convenience Relation (``ApplyKinematicBCs``, ``StressFreeSurface``, ...).

    Partial integration (``upper == var``) is supported by the ``"direct"``
    and ``"analytical"`` methods — sympy will produce a running integral.

    Usage::

        zmom.apply(Integrate(state.z, state.b, state.z))             # partial
        zmom.apply(Integrate(state.z, state.b, state.eta))           # full depth
        zmom.apply(Integrate(state.z, state.b, state.b + state.H,
                             method="analytical"))                    # sympy.integrate
    """

    def __init__(self, var, lower, upper, method="auto"):
        super().__init__(
            name="integrate",
            description=(f"Integrate w.r.t. {var} from {lower} to {upper} "
                         f"(method={method})"),
        )
        self._var = var
        self._lower = lower
        self._upper = upper
        self._method = method

    def _apply_leaf(self, expression):
        if self._method == "analytical":
            # Whole-expression analytic integration, with Dummy substitution
            # to avoid clashes when upper == var (partial / running integral).
            dummy = sp.Dummy("_int_dummy")
            integrand = expression.expr.subs(self._var, dummy)
            result = sp.integrate(integrand, (dummy, self._lower, self._upper))
            return Expression(result, expression.name)

        # Per-term dispatch through Expression.depth_integrate.
        total = Expression(S.Zero)
        for term in expression.terms:
            r = term.depth_integrate(self._lower, self._upper, self._var,
                                     method=self._method)
            if isinstance(r, DepthIntegralResult):
                # DepthIntegralResult's components already carry their signs;
                # assemble = volume + boundary_upper + boundary_lower.
                total = total + r.assemble()
            elif isinstance(r, Expression):
                total = total + r
            else:
                total = total + Expression(r)
        return total

    def _repr_latex_(self):
        return (
            f"$\\int_{{{sp.latex(self._lower)}}}^{{{sp.latex(self._upper)}}} "
            f"(\\cdot)\\, d{sp.latex(self._var)}$"
        )


class PartialIntegrate(Operation):
    """Rewrite a matching ``Integral(f, (var, lo, hi))`` to a running integral
    with variable upper bound: ``Integral(f, (var, lo, upper))``.

    The result is left symbolic — no evaluation. Downstream, the expression
    can be promoted to a ``Qaux`` variable via ``DerivedModel.promote_to_qaux``
    and rewritten into a runtime primitive (e.g. ``column_partial_integrate``).

    Matching
    --------
    Every ``Integral`` whose integration variable equals ``var`` is inspected.
    Additional matchers ``lo_match`` / ``hi_match`` restrict the rewrite to
    Integrals whose lower/upper bounds also match (structural equality); pass
    ``None`` (default) to leave them unrestricted.

    The new upper bound is ``upper_symbol`` (default: ``var`` itself, producing
    the classic running integral ``∫_lo^{var} f dvar``).

    Usage::

        # depth integral [b, b+h] → running integral [b, z]
        eq.apply(PartialIntegrate(state.z, lo_match=state.b, hi_match=state.eta))

        # unrestricted: rewrite every ∫ dz to ∫_·^z dz
        eq.apply(PartialIntegrate(z))
    """

    def __init__(self, var, upper_symbol=None, lo_match=None, hi_match=None,
                 name="partial_integrate"):
        super().__init__(
            name=name,
            description=(f"partial integrate w.r.t. {var} "
                         f"(upper → {upper_symbol if upper_symbol is not None else var})"),
        )
        self._var = var
        self._upper = upper_symbol if upper_symbol is not None else var
        self._lo_match = lo_match
        self._hi_match = hi_match

    def _leaf_sp(self, expr):
        v = self._var
        new_upper = self._upper
        lo_m = self._lo_match
        hi_m = self._hi_match

        def _walk(e):
            if isinstance(e, Integral):
                # sympy stores limits as sp.Tuple, not Python tuple.
                limits = e.args[1] if len(e.args) >= 2 else None
                if limits is not None and len(limits) == 3:
                    int_var, lo, hi = limits[0], limits[1], limits[2]
                    if int_var == v:
                        if (lo_m is None or lo_m == lo) and (hi_m is None or hi_m == hi):
                            new_integrand = _walk(e.args[0])
                            tail = e.args[2:]
                            return Integral(new_integrand, (int_var, lo, new_upper), *tail)
            if hasattr(e, "args") and e.args:
                try:
                    return e.func(*(_walk(a) for a in e.args))
                except (TypeError, ValueError):
                    return e
            return e

        return _walk(expr)

    def _repr_latex_(self):
        return (f"$\\int_{{\\cdot}}^{{{sp.latex(self._upper)}}} "
                f"(\\cdot)\\, d{sp.latex(self._var)}$")


class ZetaTransform(Operation):
    """Transform vertical coordinate: z = ζ·(upper−lower) + lower, dz = (upper−lower)·dζ.

    Transforms z-integrals whose bounds match the configured
    ``(lower, upper)``::

        ∫_{lower}^{upper} f(z) dz → (upper − lower) · ∫_0^1 f(ζ·(upper−lower)+lower) dζ

    With no ``lower`` / ``upper`` given, defaults to ``state.b`` /
    ``state.eta`` — the full free-surface transform used by the
    single-layer SME walkthrough.  Pass layer-specific bounds
    (e.g. ``lower=state.b, upper=z_1``) in the multi-layer case so
    each layer's integrals get mapped to their own ζ-interval.

    Only ``Integral`` nodes whose bounds structurally equal the
    configured pair are rewritten; other integrals (including
    already-transformed ``(zeta, 0, 1)`` ones) are left untouched.
    That makes it safe to apply multiple ZetaTransforms back-to-back
    on a branch with several integration ranges.
    """

    def __init__(self, state, lower=None, upper=None, zeta=None,
                 name=None, description=None):
        self._z = state.z
        self._lower = lower if lower is not None else state.b
        self._upper = upper if upper is not None else state.eta
        self._h = self._upper - self._lower
        self._zeta = zeta if zeta is not None else state.zeta
        super().__init__(
            name=name or "zeta_transform",
            description=(description or
                         f"z = ζ·({self._upper} − {self._lower}) + {self._lower}"),
        )

    def _leaf_sp(self, expr):
        from sympy import Integral, Derivative
        z, lower, upper, h, zeta = (
            self._z, self._lower, self._upper, self._h, self._zeta,
        )

        def _transform(e):
            if isinstance(e, Integral):
                integrand = e.args[0]
                limits = e.args[1]
                if (len(limits) == 3
                        and limits[0] == z
                        and limits[1] == lower
                        and limits[2] == upper):
                    new_integrand = integrand.subs(z, zeta * h + lower) * h
                    return Integral(new_integrand, (zeta, S.Zero, S.One))
                return e
            if isinstance(e, Derivative):
                inner = _transform(e.args[0])
                if inner != e.args[0]:
                    return Derivative(inner, *e.args[1:])
                return e
            if e.args:
                new_args = [_transform(a) for a in e.args]
                if any(n != o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        return _transform(expr)

    def _repr_latex_(self):
        return (f"$z = \\zeta \\cdot ({sp.latex(self._upper)} - "
                f"{sp.latex(self._lower)}) + {sp.latex(self._lower)}, "
                f"\\quad dz = ({sp.latex(self._h)}) \\, d\\zeta$")


def _decompose_deriv_factor(term, var):
    """Decompose ``term`` as ``(coeff, inner)`` so that ``term == coeff * Derivative(inner, var)``.

    Returns ``None`` if no such decomposition exists (term is not of the
    form ``coeff * ∂_var(inner)`` with a first-order ``var``-derivative).
    Used by :func:`_rule_anti_product_rule` to match conjugate pairs.
    """
    if isinstance(term, Derivative):
        if term.variables == (var,):
            return S.One, term.args[0]
        return None
    if isinstance(term, Mul):
        factors = list(term.args)
        ds = [f for f in factors
              if isinstance(f, Derivative) and f.variables == (var,)]
        if len(ds) != 1:
            return None
        d = ds[0]
        inner = d.args[0]
        coeff = Mul(*[f for f in factors if f is not d]) if len(factors) > 1 else S.One
        return coeff, inner
    return None


def _rule_anti_product_rule(expr, *, vars=(), **kwargs):
    """Reverse the product rule: ``α·∂_v f + f·∂_v α → ∂_v(α·f)``.

    Walks the additive decomposition of ``expr`` once per ``v`` in
    ``vars`` and greedily pairs up conjugate terms (``α·∂_v f`` with
    ``f·∂_v α``), folding each matched pair into a single conservative
    ``Derivative(α·f, v)`` term.  Unmatched Derivative terms and
    Derivative-free terms pass through unchanged.

    The match is signed: two terms must have the same sign to combine
    (``+α·∂_v f`` with ``+f·∂_v α`` → ``+∂_v(α f)``;
    ``−α·∂_v f`` with ``−f·∂_v α`` → ``−∂_v(α f)``).  Numerical
    coefficients on each side are absorbed into the coeff / inner parts
    before matching, so ``2α·∂_v f + 2f·∂_v α`` still pairs correctly.
    """
    for v in vars:
        terms = list(Add.make_args(sp.expand(expr)))
        decomp = {}
        for i, term in enumerate(terms):
            d = _decompose_deriv_factor(term, v)
            if d is not None:
                decomp[i] = d
        used = set()
        for i in list(decomp):
            if i in used:
                continue
            ci, ii = decomp[i]
            for j in list(decomp):
                if j <= i or j in used:
                    continue
                cj, ij = decomp[j]
                # α·∂_v f + f·∂_v α  ⇔  ci == ij AND cj == ii
                if ci == ij and cj == ii:
                    terms[i] = sp.Derivative(ci * ii, v)
                    terms[j] = S.Zero
                    used.update([i, j])
                    break
        if used:
            expr = Add(*terms)
    return expr


def _rule_apply_aliases(expr, *, aliases=None, **kwargs):
    """Rewrite recognised sub-expressions with human-readable names.

    ``aliases`` is a dict ``{name: value}`` — for each pair we
    replace ``value`` with ``name`` in the expression.  A direct
    ``.subs`` matches only if the expanded form structurally contains
    ``value``; we first try that, and if nothing changes we run
    ``sp.factor`` on the current expression before re-trying so that
    multi-term values like ``z_1 - b`` fire on the factored form
    ``α·(z_1 - b)`` rather than the expanded ``α·z_1 - α·b``.

    Recurses into Derivative / Integral / Add / Mul children so that
    aliases inside a ``Derivative(..., t)`` also get rewritten.
    """
    aliases = aliases or {}
    if not aliases:
        return expr

    def _try_alias_here(e):
        for name, value in aliases.items():
            e2 = e.subs(value, name)
            if e2 != e:
                e = e2
                continue
            # Fall back to factoring and retrying.  ``factor`` can be
            # slow on big expressions but fires at structural leaves
            # like ``α·z_1 - α·b`` → ``α·(z_1 - b)`` where .subs matches.
            try:
                factored = sp.factor(e)
            except Exception:
                factored = e
            e2 = factored.subs(value, name)
            if e2 != factored:
                e = e2
        return e

    def _walk(e):
        # Apply at the current node first, then **also** recurse into
        # sub-expressions — a partial match at the current level (e.g.
        # ``z_1 - b`` in one Add child) shouldn't block alias
        # substitution inside a sibling child (e.g. ``α·z_1 - α·b`` inside
        # a Derivative), which we only see after factoring the sibling.
        candidate = _try_alias_here(e)
        if candidate != e:
            e = candidate
        if hasattr(e, "args") and e.args:
            new_args = [_walk(a) for a in e.args]
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _walk(expr)


def _rule_combine_derivatives(expr, **kwargs):
    """Combine additive ``Derivative`` terms sharing the same variables.

    ``Derivative(f, v) + Derivative(g, v) → Derivative(f + g, v)`` (linearity).
    Runs across all additive siblings; terms with other factors (e.g.
    ``α(t,x)·Derivative(β, v)`` where the coefficient is not a pure
    scalar) are **not** folded — that would need the product rule,
    which is ``anti_product_rule``'s job.

    In practice this picks up things the pipeline emits as two
    adjacent conservative terms, e.g.
    ``Derivative(α·z_1, t) + Derivative(−α·b, t)`` folds to
    ``Derivative(α·(z_1 − b), t)`` which an ``apply_aliases`` rule can
    then rename to ``Derivative(α·h_0, t)``.
    """
    terms = list(Add.make_args(sp.expand(expr)))
    grouped: dict = {}
    leftovers = []
    for term in terms:
        if isinstance(term, Derivative):
            key = term.args[1:]
            grouped[key] = grouped.get(key, S.Zero) + term.args[0]
            continue
        if isinstance(term, Mul):
            deriv_factors = [f for f in term.args if isinstance(f, Derivative)]
            if len(deriv_factors) == 1:
                d = deriv_factors[0]
                coeff = Mul(*[f for f in term.args if f is not d])
                # Only fold when the remaining factor is a pure scalar
                # (no free_symbols overlap with the Derivative vars).
                deriv_vars_set = set(d.variables)
                coeff_depends = bool(coeff.free_symbols & deriv_vars_set)
                if not coeff_depends and not any(
                        isinstance(a, sp.Function)
                        and any(v in a.free_symbols for v in deriv_vars_set)
                        for a in coeff.atoms(sp.Function)):
                    key = d.args[1:]
                    grouped[key] = grouped.get(key, S.Zero) + coeff * d.args[0]
                    continue
        leftovers.append(term)
    new_parts = list(leftovers)
    for key, inner in grouped.items():
        if inner == 0:
            continue
        new_parts.append(sp.Derivative(inner, *key))
    return Add(*new_parts)


def _rule_collapse_trivial_derivative(expr, **kwargs):
    """``Derivative(c, v) → 0`` when ``c`` doesn't depend on ``v``.

    A safety-net rule — :func:`_evaluate_linear_derivatives` already
    handles this during the lightweight simplify, so in practice
    Recombine only hits it when a downstream op re-introduces a
    trivial Derivative (e.g. a manual ``sp.Derivative(f, v)`` with a
    ``v``-free ``f``).
    """
    def _walk(e):
        if isinstance(e, Derivative):
            inner = e.args[0]
            if not any(inner.has(v) for v in e.variables):
                return S.Zero
            inner_new = _walk(inner)
            if inner_new is not inner:
                return sp.Derivative(inner_new, *e.args[1:])
            return e
        if hasattr(e, "args") and e.args:
            new_args = [_walk(a) for a in e.args]
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
        return e
    return _walk(expr)


class Recombine(Operation):
    """Fold expanded expressions back into compact / conservative shapes.

    Typically run at the end of a derivation to pretty up ``describe()``
    output: ``h·∂_t α + α·∂_t h`` becomes ``∂_t(h α)``, cancels like
    ``(z_1 − b)`` can be aliased to ``h_0``, and any trivially-zero
    Derivative re-introduced downstream gets collapsed.

    **Rule catalog.**  ``Recombine.RULES`` is a module-level dict
    mapping a rule's name to a ``callable(expr, **kwargs) → expr``.
    The default implementation ships three rules:

    * ``anti_product_rule`` — the classic
      ``α·∂_v f + f·∂_v α → ∂_v(α·f)`` fold (needs ``vars=[...]``).
    * ``apply_aliases`` — replaces user-declared sub-expressions with
      shorter names (needs ``aliases={...}``).
    * ``collapse_trivial_derivative`` — ``Derivative(c, v) → 0`` if
      ``c`` has no ``v``.

    Select / order rules via the ``rules`` argument:

        # Default: all rules, in catalog order.
        eq.apply(Recombine(vars=[t, x], aliases={h_0: z_1 - b}))

        # Only the anti-product-rule, for display cleanup:
        eq.apply(Recombine(rules=["anti_product_rule"], vars=[t, x]))

        # Custom rule: pass a callable directly alongside / instead
        # of a named rule.
        eq.apply(Recombine(rules=["anti_product_rule", my_rule],
                           vars=[t, x]))

    **Extending the catalog.**  Add a rule once — available to every
    subsequent ``Recombine``:

        Recombine.RULES["my_rule"] = my_rule_fn

    A rule is any callable with signature ``fn(expr, **kwargs) → expr``.
    It receives the full ``vars`` / ``aliases`` kwargs; unknown kwargs
    are harmless (they get swallowed by ``**kwargs``).
    """

    RULES = {
        "anti_product_rule": _rule_anti_product_rule,
        "combine_derivatives": _rule_combine_derivatives,
        "apply_aliases": _rule_apply_aliases,
        "collapse_trivial_derivative": _rule_collapse_trivial_derivative,
    }

    # Recombine needs to see the whole leaf — pair-matching rules
    # (``anti_product_rule``, ``combine_derivatives``) only work when
    # multiple additive terms are visible at once.
    whole_leaf_op = True

    def __init__(self, rules=None, vars=(), aliases=None,
                 name=None, description=None):
        if rules is None:
            rules = list(self.RULES)
        # Normalise each entry to a callable.
        self._rule_fns = []
        rule_labels = []
        for r in rules:
            if isinstance(r, str):
                if r not in self.RULES:
                    raise KeyError(
                        f"Unknown Recombine rule {r!r}. "
                        f"Available: {sorted(self.RULES)}"
                    )
                self._rule_fns.append(self.RULES[r])
                rule_labels.append(r)
            elif callable(r):
                self._rule_fns.append(r)
                rule_labels.append(getattr(r, "__name__", repr(r)))
            else:
                raise TypeError(
                    f"Recombine rules must be strings or callables "
                    f"(got {type(r).__name__})."
                )
        self._vars = tuple(vars)
        self._aliases = dict(aliases) if aliases else {}
        super().__init__(
            name=name or "recombine",
            description=(description
                         or f"recombine via {', '.join(rule_labels)}"),
        )

    def _leaf_sp(self, expr):
        for rule in self._rule_fns:
            expr = rule(expr, vars=self._vars, aliases=self._aliases)
        return expr


class ExpandProductRule(Operation):
    """Split ``φ(…) · ∂_var(f)`` into ``∂_var(φ·f) - ∂_var(φ)·f``.

    Required when a test function factor carries the same coordinate
    that the derivative is w.r.t.  For instance, after
    ``Multiply(basis.phi_of_z, outer=True)``, terms look like
    ``φ_l((z-b)/h) · ∂_x(u²)``.  The coefficient depends on ``x`` via
    ``b(t,x)``, ``h(t,x)``, so it's **not** in conservative form and
    ``Integrate(method='auto')`` would fall back to ``direct`` —
    leaving an un-Leibniz'd ``Integral(φ · ∂_x f, (z, b, eta))``.

    Running ``ExpandProductRule([t, x, z])`` *before* ``Integrate``
    rewrites each such term into:

    * ``∂_var(φ·f)`` — now genuinely conservative in ``var``.
      ``Integrate`` applies Leibniz or the fundamental theorem to it.
    * ``-∂_var(φ)·f`` — the non-conservative coupling.  ``∂_var φ`` is
      evaluated via ``.doit()`` so the explicit dependence on
      ``∂_var b``, ``∂_var h`` shows up (this is what generates the
      ``W_sigma``-like terms of the sigma-SME derivation).  The
      resulting integrand is not in derivative form; ``Integrate``
      keeps it inside a direct integral, which collapses under
      ``EvaluateIntegrals`` after basis expansion and
      ``ZetaTransform``.

    Each term that's already conservative (``var ∉ coeff.free_symbols``)
    is left untouched.
    """

    def __init__(self, variables, name="expand_product_rule",
                 description=None):
        self._vars = tuple(variables)
        super().__init__(
            name=name,
            description=(description or
                         f"Expand φ·∂_v(f) → ∂_v(φ·f) − ∂_v(φ)·f for v∈"
                         f"{{{', '.join(str(v) for v in variables)}}}"),
        )

    def _leaf_sp(self, expr):
        def _expand_term(term):
            if isinstance(term, Derivative):
                return term  # bare Derivative → already conservative
            if not isinstance(term, Mul):
                return term
            factors = list(term.args)
            derivs = [f for f in factors if isinstance(f, Derivative)]
            if len(derivs) != 1:
                return term
            d = derivs[0]
            if len(d.variables) != 1 or d.variables[0] not in self._vars:
                return term
            var = d.variables[0]
            inner = d.args[0]
            coeff = Mul(*(f for f in factors if f is not d)) if len(factors) > 1 else S.One
            if var not in coeff.free_symbols:
                return term  # already conservative
            return (sp.Derivative(coeff * inner, var)
                    - sp.Derivative(coeff, var).doit() * inner)

        return sum((_expand_term(t) for t in Add.make_args(sp.expand(expr))), S.Zero)


class ProductRule(Operation):
    """Inverse product rule: combine ``coeff · d(f)/dx`` into ``d(F)/dx``.

    Applied to a single term (use ``expr.terms[i].apply(ProductRule())``).
    Detects the derivative in the term, checks if the remaining coefficient
    can be integrated w.r.t. the differentiated variable, and combines.

    Example: ``g·h·∂h/∂x → ∂/∂x(g·h²/2)``
    """

    def __init__(self):
        super().__init__(
            name="product_rule",
            description="Inverse product rule: coeff·∂f/∂x → ∂(F)/∂x",
        )

    def _leaf_sp(self, expr):
        from sympy import Derivative, Mul

        # Find the derivative factor in the term
        factors = Mul.make_args(expr)
        deriv_factor = None
        other_factors = []
        for f in factors:
            if isinstance(f, Derivative) and deriv_factor is None:
                deriv_factor = f
            else:
                other_factors.append(f)

        if deriv_factor is None:
            return expr  # no derivative found, nothing to do

        inner = deriv_factor.args[0]
        var = deriv_factor.variables[0]
        coeff = sp.Mul(*other_factors) if other_factors else S.One

        # Check if coeff is proportional to inner^n
        # Common case: coeff = c * inner → term = c * inner * d(inner)/dx = c * d(inner²/2)/dx
        if coeff.has(inner):
            ratio = sp.simplify(coeff / inner)
            if not ratio.has(inner):
                flux = ratio * inner**2 / 2
                return Derivative(flux, var)

        # General case: coeff doesn't contain inner
        # term = coeff * d(inner)/dx = d(coeff * inner)/dx - d(coeff)/dx * inner
        # Only useful if d(coeff)/dx = 0 (coeff is constant w.r.t. var)
        if not coeff.has(var):
            flux = coeff * inner
            return Derivative(flux, var)

        return expr  # can't simplify

    def _repr_latex_(self):
        return ""


class _INSBuilder:
    """Internal: builds INS equations from a StateSpace."""

    def __init__(self, state: StateSpace):
        self.state = state
        self.dim = state.dim

    def _stress_divergence(self, row):
        s = self.state
        labels = ["x", "y", "z"] if s.has_y else ["x", "z"]
        coord_map = {"x": s.x, "y": s.y, "z": s.z}
        expr = S.Zero
        for j in labels:
            expr += Derivative(s.tau[row + j], coord_map[j])
        return expr

    @property
    def continuity(self):
        s = self.state
        expr = Derivative(s.u, s.x) + Derivative(s.w, s.z)
        if s.has_y:
            expr += Derivative(s.v, s.y)
        return Expression(expr, "continuity")

    def _momentum(self, vel, name, gravity=S.Zero):
        """Build a momentum equation with canonical term ordering.

        Order: temporal → convection → pressure → stress → source
        """
        s = self.state
        row = name.split("_")[0]  # "x", "y", "z"

        temporal = Derivative(vel, s.t)

        convection = Derivative(vel * s.u, s.x) + Derivative(vel * s.w, s.z)
        if s.has_y:
            convection += Derivative(vel * s.v, s.y)

        pressure = Rational(1, 1) / s.rho * Derivative(s.p, {"x": s.x, "y": s.y, "z": s.z}[row])
        stress = -Rational(1, 1) / s.rho * self._stress_divergence(row)

        groups = {"temporal": temporal, "convection": convection,
                  "pressure": pressure, "stress": stress}
        if gravity != S.Zero:
            groups["source"] = gravity

        full_expr = sum(groups.values())
        return Expression(full_expr, name, term_groups=groups)

    @property
    def x_momentum(self):
        return self._momentum(self.state.u, "x_momentum")

    @property
    def y_momentum(self):
        if not self.state.has_y:
            return None
        return self._momentum(self.state.v, "y_momentum")

    @property
    def z_momentum(self):
        return self._momentum(self.state.w, "z_momentum", gravity=self.state.g)

    @property
    def equations(self):
        eqs = [self.continuity, self.x_momentum]
        if self.state.has_y:
            eqs.append(self.y_momentum)
        eqs.append(self.z_momentum)
        return eqs

    def describe(self, header=True, final_equation=True, strip_args=True,
                 **kwargs):
        """Composable description of the full INS system.

        Returns a ``Description`` that renders as markdown in Jupyter.
        """
        from zoomy_core.misc.description import Description

        parts = []
        if header:
            dim_label = "3D (x,y,z)" if self.state.has_y else "2D (x,z)"
            parts.append(f"**Incompressible Navier-Stokes** ({dim_label})")

        if final_equation:
            for eq in self.equations:
                tex = eq.latex(strip_args=strip_args)
                parts.append(f"\n**{eq.name}:**\n$$\n{tex} = 0\n$$")

        return Description("\n".join(parts))

    def _repr_markdown_(self):
        return self.describe()._repr_markdown_()


def FullINS(state, equations=None):
    """Build the incompressible Navier-Stokes as a ``System`` tree.

    Tree shape::

        system
        ├── continuity   (scalar leaf)
        └── momentum     (intermediate Zstruct)
            ├── x        (leaf)
            ├── y        (leaf, only in 3D)
            └── z        (leaf)

    Access::

        ins = FullINS(state)
        ins.continuity                  # scalar proxy
        ins.momentum                    # intermediate proxy
        ins.momentum.x.apply(op)        # mutate one component leaf
        ins.momentum.apply(op)          # mutate every component leaf

    Parameters
    ----------
    state : StateSpace
    equations : list of str, optional
        Components to include.  Options: ``"continuity"``, ``"momentum.x"``,
        ``"momentum.y"``, ``"momentum.z"``.  Top-level ``"momentum"`` means
        all momentum components.  Default: everything.
    """
    from zoomy_core.model.models.derived_system import System

    builder = _INSBuilder(state)
    system = System("INS", state)

    want_continuity = (equations is None) or ("continuity" in equations)
    want_all_momentum = (equations is None) or ("momentum" in equations)
    momentum_components = []
    for axis in ("x", "y", "z"):
        if axis == "y" and not state.has_y:
            continue
        if want_all_momentum or f"momentum.{axis}" in (equations or ()):
            momentum_components.append(axis)

    if want_continuity:
        system.add_equation("continuity", builder.continuity)

    for axis in momentum_components:
        expr = getattr(builder, f"{axis}_momentum")
        system.add_equation(("momentum", axis), expr)

    return system


# ---------------------------------------------------------------------------
# Material models library
# ---------------------------------------------------------------------------

class Newtonian(Material):
    """Newtonian fluid: tau_ij = mu * (du_i/dx_j + du_j/dx_i), mu = rho * nu."""

    def __init__(self, state: StateSpace, nu=None):
        self.nu = nu if nu is not None else Symbol("nu", positive=True)
        subs = self._build(state, self.nu)
        super().__init__(subs, name="Newtonian")

    @staticmethod
    def _build(s, nu):
        mu = nu * s.rho
        u, w = s.u, s.w
        x, z = s.x, s.z
        subs = {
            s.tau["xx"]: 2 * mu * Derivative(u, x),
            s.tau["xz"]: mu * (Derivative(u, z) + Derivative(w, x)),
            s.tau["zx"]: mu * (Derivative(w, x) + Derivative(u, z)),
            s.tau["zz"]: 2 * mu * Derivative(w, z),
        }
        if s.has_y:
            v, y = s.v, s.y
            subs.update({
                s.tau["xy"]: mu * (Derivative(u, y) + Derivative(v, x)),
                s.tau["yx"]: mu * (Derivative(v, x) + Derivative(u, y)),
                s.tau["yy"]: 2 * mu * Derivative(v, y),
                s.tau["yz"]: mu * (Derivative(v, z) + Derivative(w, y)),
                s.tau["zy"]: mu * (Derivative(w, y) + Derivative(v, z)),
            })
        return subs


class Inviscid(Material):
    """Inviscid fluid: all tau_ij = 0."""

    def __init__(self, state: StateSpace):
        subs = {v: S.Zero for v in state.tau.values()}
        super().__init__(subs, name="Inviscid")


class materials:
    """Material model library. Usage: materials.newtonian(state)"""
    newtonian = Newtonian
    inviscid = Inviscid


# ---------------------------------------------------------------------------
# Assumptions library
# ---------------------------------------------------------------------------

class InterfaceKBC(Assumption):
    """Kinematic BC at an arbitrary interface ``z = interface(t, x[, y])``.

    Generalises :class:`KinematicBCBottom` / :class:`KinematicBCSurface`:
    parameterised on the interface height and an optional mass flux
    ``m`` (mass per unit area per unit time crossing the interface),
    so a single class covers solid bed, free surface, and every
    internal layer interface in a multi-layer derivation.

    ``w|_{z=interface}`` is substituted with::

        ∂_t interface + u|·∂_x interface [+ v|·∂_y interface] + mass_flux / rho

    Pass ``mass_flux=None`` for impermeable interfaces (bottom / top);
    pass a sympy expression (typically a freshly-minted ``Function``)
    for internal interfaces whose flux is an unknown of the resulting
    system.

    Usage::

        # Bottom (identical to KinematicBCBottom):
        model.apply(InterfaceKBC(state, state.b))

        # Free surface (identical to KinematicBCSurface):
        model.apply(InterfaceKBC(state, state.eta))

        # Internal layer interface with mass flux m_1:
        model.apply(InterfaceKBC(state, z_1, mass_flux=m_1))
    """

    def __init__(self, state: StateSpace, interface, mass_flux=None,
                 name=None, description=None):
        s = state
        w_at = s.w.subs(s.z, interface)
        u_at = s.u.subs(s.z, interface)
        rhs = Derivative(interface, s.t) + u_at * Derivative(interface, s.x)
        if s.has_y:
            v_at = s.v.subs(s.z, interface)
            rhs += v_at * Derivative(interface, s.y)
        if mass_flux is not None:
            rhs = rhs + mass_flux / s.rho
        default_name = f"kinematic_bc@{interface}"
        super().__init__({w_at: rhs}, name=name or default_name)
        # ``Relation.__init__`` doesn't carry a description, so attach
        # one directly so history_mermaid labels stay readable.
        self.description = description or (
            f"w|_{{z={interface}}} = ∂_t + u·∂_x (+ v·∂_y) (+ m/ρ)"
        )


class KinematicBCBottom(Assumption):
    """w|_{z=b} = db/dt + u_b * db/dx [+ v_b * db/dy]"""

    def __init__(self, state: StateSpace):
        s = state
        w_at_b = s.w.subs(s.z, s.b)
        u_at_b = s.u.subs(s.z, s.b)
        rhs = Derivative(s.b, s.t) + u_at_b * Derivative(s.b, s.x)
        if s.has_y:
            v_at_b = s.v.subs(s.z, s.b)
            rhs += v_at_b * Derivative(s.b, s.y)
        super().__init__({w_at_b: rhs}, name="kinematic_bc_bottom")


class KinematicBCSurface(Assumption):
    """w|_{z=eta} = d(eta)/dt + u_s * d(eta)/dx [+ v_s * d(eta)/dy]"""

    def __init__(self, state: StateSpace):
        s = state
        w_at_s = s.w.subs(s.z, s.eta)
        u_at_s = s.u.subs(s.z, s.eta)
        rhs = Derivative(s.eta, s.t) + u_at_s * Derivative(s.eta, s.x)
        if s.has_y:
            v_at_s = s.v.subs(s.z, s.eta)
            rhs += v_at_s * Derivative(s.eta, s.y)
        super().__init__({w_at_s: rhs}, name="kinematic_bc_surface")


class ContinuityClosure(Assumption):
    """Close the bulk ``w(t, x, z)`` via continuity + lower-interface KBC.

    The pointwise continuity equation ``∂_x u + ∂_z w = 0`` integrated
    from the lower interface ``lower(t, x)`` upward gives::

        w(t, x, z) = w|_{z = lower}
                     − ∫_{lower}^{z} ∂_x u(t, x, z') dz'

    The ``w|_{z = lower}`` value is supplied by the lower-interface KBC
    (bottom / free-surface / internal layer interface depending on the
    context), with an optional mass-flux contribution.

    The resulting substitution ``{w(t, x, z): w|_lower − ∫_{lower}^z ∂_x u dz'}``
    closes every bulk appearance of ``w(t, x, z)`` in the expression
    tree.  Boundary evaluations ``w(t, x, b)``, ``w(t, x, η)`` etc. are
    handled by the corresponding :class:`InterfaceKBC` and are **not**
    touched here (those are distinct structural forms).

    Parameters
    ----------
    state : StateSpace
    lower : sympy expression, optional
        Lower integration bound — the interface that fixes ``w``.
        Default: ``state.b`` (bottom).  For multi-layer use, pass the
        relevant layer interface height.
    mass_flux : sympy expression, optional
        Mass flux through ``lower`` (mass per unit area per unit time).
        Default: ``None`` (impermeable lower boundary).  If supplied,
        added to the KBC value as ``mass_flux / rho``.

    Typical placement
    -----------------
    Apply **after** :class:`Integrate` (so the depth-integration has
    already introduced the ``Integral(f, (z, lower, upper))`` volume
    terms) and **after** :class:`InterfaceKBC` (so the boundary
    evaluations of ``w`` are already closed), but **before**
    :class:`ZetaTransform` (so the running integral's ``z``-bounded
    form is converted cleanly in the same pass).
    """

    def __init__(self, state: StateSpace, lower=None, mass_flux=None,
                 name=None, description=None):
        s = state
        if lower is None:
            lower = s.b
        u_at_lower = s.u.subs(s.z, lower)
        w_at_lower = Derivative(lower, s.t) + u_at_lower * Derivative(lower, s.x)
        if s.has_y:
            v_at_lower = s.v.subs(s.z, lower)
            w_at_lower = w_at_lower + v_at_lower * Derivative(lower, s.y)
        if mass_flux is not None:
            w_at_lower = w_at_lower + mass_flux / s.rho
        # Running integral in z — ``Integral(f, (z, lower, z))`` is sympy's
        # standard form for an indefinite / running integral with a
        # symbolic upper bound that happens to match the integration
        # variable's name.  Downstream ops (``ZetaTransform``,
        # ``EvaluateIntegrals``) treat the inner integral as an opaque
        # sub-expression until the basis substitution closes ``u``.
        w_of_z = w_at_lower - Integral(Derivative(s.u, s.x),
                                       (s.z, lower, s.z))
        if s.has_y:
            w_of_z = w_of_z - Integral(Derivative(s.v, s.y),
                                       (s.z, lower, s.z))
        super().__init__({s.w: w_of_z},
                         name=name or f"continuity_closure@{lower}")
        self.description = description or (
            f"w(z) = w|_{{z={lower}}} − ∫_{{{lower}}}^z ∂_x u dz' "
            f"(continuity + lower-interface KBC)"
        )


class NoTangentialBoundaryStress(Assumption):
    """Zero tangential normal stress at both surface and bottom.

    Closes the Leibniz boundary terms ``tau_xx(z=b)``, ``tau_xx(z=eta)``
    (and y-components in 3D) that remain after depth-integrating
    ``∂_x tau_xx``.  Uses a Newtonian-style "no flow variation" closure
    at the vertical boundaries.
    """

    def __init__(self, state: StateSpace):
        s = state
        subs = {}
        if "xx" in s.tau:
            subs[s.tau["xx"].subs(s.z, s.b)] = S.Zero
            subs[s.tau["xx"].subs(s.z, s.eta)] = S.Zero
        if s.has_y and "yy" in s.tau:
            subs[s.tau["yy"].subs(s.z, s.b)] = S.Zero
            subs[s.tau["yy"].subs(s.z, s.eta)] = S.Zero
            subs[s.tau["xy"].subs(s.z, s.b)] = S.Zero
            subs[s.tau["xy"].subs(s.z, s.eta)] = S.Zero
            subs[s.tau["yx"].subs(s.z, s.b)] = S.Zero
            subs[s.tau["yx"].subs(s.z, s.eta)] = S.Zero
        super().__init__(subs, name="no_tangential_boundary_stress")


class HydrostaticPressure(Assumption):
    """p = p_atm + rho * g * (eta - z)"""

    def __init__(self, state: StateSpace):
        s = state
        p_atm = Function("p_atm", real=True)(s.t, *s.coords_h)
        p_hydro = p_atm + s.rho * s.g * (s.eta - s.z)
        super().__init__({s.p: p_hydro}, name="hydrostatic_pressure")


class StressFreeSurface(Assumption):
    """Stress-free surface: τ·n|_{z=η} = 0.

    For a free surface with normal n ≈ (0,0,1), this gives:
      τ_xz|_{z=η} = 0,  τ_zz|_{z=η} = 0  (and τ_yz|_{z=η} = 0 in 3D)
    """

    def __init__(self, state: StateSpace):
        s = state
        subs = {}
        # τ_xz at surface = 0
        subs[s.tau["xz"].subs(s.z, s.eta)] = S.Zero
        # τ_zx at surface = 0 (symmetric)
        if "zx" in s.tau:
            subs[s.tau["zx"].subs(s.z, s.eta)] = S.Zero
        # τ_zz at surface = 0
        subs[s.tau["zz"].subs(s.z, s.eta)] = S.Zero
        if s.has_y:
            subs[s.tau["yz"].subs(s.z, s.eta)] = S.Zero
            if "zy" in s.tau:
                subs[s.tau["zy"].subs(s.z, s.eta)] = S.Zero
        super().__init__(subs, name="stress_free_surface")


class ZeroAtmosphericPressure(Assumption):
    """p_atm = 0 (no atmospheric pressure)."""

    def __init__(self, state: StateSpace):
        p_atm = Function("p_atm", real=True)(state.t, *state.coords_h)
        super().__init__({p_atm: S.Zero}, name="p_atm=0")


class assumptions:
    """Assumptions library."""
    kinematic_bc_bottom = KinematicBCBottom
    kinematic_bc_surface = KinematicBCSurface
    hydrostatic_pressure = HydrostaticPressure
    stress_free_surface = StressFreeSurface
    zero_atmospheric_pressure = ZeroAtmosphericPressure


# ---------------------------------------------------------------------------
# Basis: vertical polynomial basis + coefficient functions
# ---------------------------------------------------------------------------

class Basis:
    """Vertical polynomial basis for the SME expansion ``u = Σ α_k φ_k(ζ)``.

    Wraps a :class:`Basisfunction` class (``Legendre_shifted`` etc.) and
    exposes symbolic pieces for the apply-based walkthrough:

    * ``basis.phi`` — ``Zstruct(phi_0, …, phi_N)`` of sympy expressions in
      ``state.zeta``.  Drop-in factor for ``Multiply(basis.phi, outer=True)``.
    * ``basis.alpha`` — ``Zstruct(alpha_0, …, alpha_N)`` of
      :class:`Function` coefficients ``α_k(t, x, [y])``.  Using Functions
      (not bare Symbols) keeps downstream partial derivatives honest and
      makes linear-stability analysis straightforward.
    * ``basis.expand(field)`` — single substitution dict covering the
      volume evaluation (in ζ-transformed form) and both boundary
      evaluations at ``z=b`` / ``z=η``.  Feed it to ``model.apply(...)``.

    Usage::

        from zoomy_core.model.models.basisfunctions import Legendre_shifted
        basis = Basis(state, Legendre_shifted, level=2)
        model.momentum.x.apply(Multiply(basis.phi, outer=True))
        model.apply(ZetaTransform(state))
        model.apply(basis.expand(state.u))

    The volume substitution uses the ζ-transformed key
    ``field.subs(z, ζ·h + b)`` — apply it **after** :class:`ZetaTransform`
    so it matches what's under the integrals.
    """

    def __init__(self, state: StateSpace, basisfunction_cls, level=0,
                 alpha_name="alpha", **basis_kwargs):
        """Wrap a :class:`Basisfunction` subclass for the apply-based pipeline.

        ``basis_kwargs`` are forwarded to the ``basisfunction_cls`` constructor
        — use them for bases that need extra structure beyond the level
        index (e.g. ``PiecewiseConstant(interfaces=[state.b, z_1, state.eta])``).
        When ``basis_kwargs`` is non-empty, ``level`` is ignored and
        re-read from the instantiated basisfunction.
        """
        from zoomy_core.misc.misc import Zstruct

        self._state = state
        if basis_kwargs:
            self._bf = basisfunction_cls(**basis_kwargs)
            self.level = self._bf.level
        else:
            self._bf = basisfunction_cls(level=level)
            self.level = level
        self._alpha_name = alpha_name

        zeta = state.zeta
        self.phi = Zstruct(**{
            f"phi_{k}": self._bf.eval(k, zeta)
            for k in range(self.level + 1)
        })
        args_h = state._args_h
        self.alpha = Zstruct(**{
            f"{alpha_name}_{k}": Function(f"{alpha_name}_{k}", real=True)(*args_h)
            for k in range(self.level + 1)
        })
        # Test functions expressed as functions of z: phi_l((z-b)/h).
        # Use this BEFORE depth-integration so the test function
        # participates in the Leibniz rule and produces the W_sigma-like
        # coupling terms on ∂_t phi, ∂_x phi.  Use ``self.phi`` (in zeta)
        # AFTER ZetaTransform, inside integrands.
        zeta_of_z = (state.z - state.b) / state.H
        self.phi_of_z = Zstruct(**{
            f"phi_{k}": self._bf.eval(k, zeta_of_z)
            for k in range(self.level + 1)
        })
        # Piecewise-constant bases that live in physical z-space expose
        # their breakpoints directly — this is what the layer-wise
        # walkthrough reads to pull z_i and u_i.
        self.interfaces = getattr(self._bf, "interfaces", None)

    def _sum(self, zeta_value):
        """Return ``Σ α_k · φ_k(zeta_value)``."""
        return sum(
            getattr(self.alpha, f"{self._alpha_name}_{k}")
            * self._bf.eval(k, zeta_value)
            for k in range(self.level + 1)
        )

    def expand(self, field):
        """Substitution dict for every bulk / boundary evaluation of ``field``.

        Four keys:

        * ``field``                    → ``Σ α_k(t,x) · φ_k((z−b)/h)``
          (pointwise form — matches ``field(t,x,z)`` in inner running
          integrals produced by e.g. :class:`ContinuityClosure`).
        * ``field.subs(z, ζ·h + b)`` → ``Σ α_k(t,x) · φ_k(ζ)``
          (ζ-transformed form occurring inside outer ζ-integrals).
        * ``field.subs(z, b)``         → ``Σ α_k · φ_k(0)`` (bottom eval).
        * ``field.subs(z, b + h)``     → ``Σ α_k · φ_k(1)`` (surface eval).

        Including the pointwise key is free — it only fires where the
        expression structurally carries ``field(t, x, z)`` — and makes
        ``ContinuityClosure``'s inner ``∫_b^z ∂_x u dz'`` close
        cleanly once ``u`` has been expanded.
        """
        state = self._state
        z, zeta, b, h = state.z, state.zeta, state.b, state.H
        volume_key = field.subs(z, zeta * h + b)
        bottom_key = field.subs(z, b)
        surface_key = field.subs(z, state.eta)
        # Pointwise form: ζ is a *function* of z here, so we evaluate
        # the inner basis at (z−b)/h and let sympy carry the resulting
        # polynomial-in-z through any outstanding z-integrals.
        pointwise_rhs = self._sum((z - b) / h)
        return {
            field: pointwise_rhs,
            volume_key: self._sum(zeta),
            bottom_key: self._sum(S.Zero),
            surface_key: self._sum(S.One),
        }

    def layer_expand(self, field, layer_idx, zeta_transformed=False):
        """Per-layer closure substitution for a :class:`LayeredBasis`.

        Inside layer ``i`` the field is closed against that layer's
        coefficients:

        .. math::

            \\text{field}(t, x, z)
            \\;\\longrightarrow\\;
            \\sum_{k=0}^{m-1} \\alpha_{i\\,m + k}(t, x)\\,
                              \\phi^{\\text{inner}}_{k}(\\zeta_i(z))

        where ``m = inner_level + 1`` is the number of moments per
        layer and ``ζ_i(z) = (z - z_i) / h_i``.  The returned
        substitution dict has three entries (volume + lower interface
        + upper interface) — the side-local convention means the two
        interface evaluations plug in ``ζ_i = 0`` and ``ζ_i = 1`` of
        the **current** layer's inner basis; a neighbouring layer
        applying its own ``layer_expand`` produces its own (generally
        different) interface value, which is exactly the jump that
        drives the multi-layer mass-flux terms.

        Parameters
        ----------
        field : sympy Function call
            The variable being closed (e.g. ``state.u``).
        layer_idx : int
            Index into ``self._bf.interfaces`` specifying the layer.
        zeta_transformed : bool, default ``False``
            If ``False`` (the SWE case), the bulk key is ``field``
            itself — substitutes ``u(t,x,z) → Σ α·φ_inner(ζ_i(z))``.
            If ``True`` (the SME case, applied *after*
            :class:`ZetaTransform`), the bulk key becomes
            ``field.subs(z, ζ·h_i + z_i)`` and the substituted form
            reduces to ``Σ α·φ_inner(ζ)`` in the shared zeta symbol,
            ready for ``EvaluateIntegrals`` to collapse the
            orthogonality integrals.

        For the piecewise-constant case (``Monomials, inner_level=0``)
        the sum collapses to a single coefficient per layer and you
        recover the simple SWE closure ``u → α_i``.

        Raises ``ValueError`` if the wrapped basisfunction isn't a
        ``LayeredBasis``.
        """
        from zoomy_core.model.models.basisfunctions import LayeredBasis
        if not isinstance(self._bf, LayeredBasis):
            raise ValueError(
                "Basis.layer_expand requires a LayeredBasis "
                "(e.g. LayeredBasis(Monomials, interfaces=[b, z_1, eta]))."
            )
        if not (0 <= layer_idx < self._bf.n_layers):
            raise IndexError(
                f"layer_idx={layer_idx} out of range for "
                f"{self._bf.n_layers} layer(s)."
            )
        state = self._state
        lower = self._bf.interfaces[layer_idx]
        upper = self._bf.interfaces[layer_idx + 1]

        def _closure(z_value):
            """Σ α_{i*m+k} · φ_inner_k(ζ_i(z_value))."""
            zeta_local = self._bf.layer_zeta(layer_idx, z_value)
            return sum(
                getattr(self.alpha,
                        f"{self._alpha_name}_"
                        f"{self._bf.flat_index(layer_idx, k)}")
                * self._bf.inner.eval(k, zeta_local)
                for k in range(self._bf.inner_n_basis)
            )

        if zeta_transformed:
            # After ZetaTransform the bulk form is ``field.subs(z, ζ·h_i + lower)``.
            # ``ZetaTransform`` + the lightweight simplify path store the
            # subbed argument in **expanded** form (``ζ·upper − ζ·lower + lower``),
            # while a naive ``ζ·(upper−lower)+lower`` stays factored.
            # We need the bulk key to match the expanded form produced by
            # ZetaTransform, because ``xreplace`` is structural and
            # ``a·(b−c)+c`` isn't the same subtree as ``a·b − a·c + c``.
            zeta_of_z = sp.expand(state.zeta * (upper - lower) + lower)
            bulk_key = field.subs(state.z, zeta_of_z)
            # RHS shortcut: after ZetaTransform the bulk argument is
            # identically ``ζ``, so we evaluate the inner basis at
            # ``state.zeta`` directly.  Going through ``_closure`` would
            # produce ``ζ·(upper-lower)/(upper-lower)`` which sympy
            # doesn't cancel (it can't rule out ``upper == lower``),
            # leaving unsimplified ratios inside integrands and
            # blocking ``EvaluateIntegrals`` from collapsing them.
            bulk_val = sum(
                getattr(self.alpha,
                        f"{self._alpha_name}_"
                        f"{self._bf.flat_index(layer_idx, k)}")
                * self._bf.inner.eval(k, state.zeta)
                for k in range(self._bf.inner_n_basis)
            )
        else:
            bulk_key = field
            bulk_val = _closure(state.z)

        return {
            bulk_key: bulk_val,
            field.subs(state.z, lower): _closure(lower),
            field.subs(state.z, upper): _closure(upper),
        }

    def layer_phi(self, layer_idx):
        """Zstruct of inner test functions in the shared ``ζ`` symbol.

        Equivalent to ``basis.phi`` but for one layer of a
        :class:`LayeredBasis`.  Use after ZetaTransform, inside
        integrands, when you need ``φ_inner_k(ζ)`` multiplied into an
        ``EvaluateIntegrals`` target.

        Raises ``ValueError`` on a non-layered basis.
        """
        from zoomy_core.misc.misc import Zstruct
        from zoomy_core.model.models.basisfunctions import LayeredBasis
        if not isinstance(self._bf, LayeredBasis):
            raise ValueError("layer_phi requires a LayeredBasis.")
        if not (0 <= layer_idx < self._bf.n_layers):
            raise IndexError(f"layer_idx={layer_idx} out of range.")
        zeta = self._state.zeta
        return Zstruct(**{
            f"phi_{k}": self._bf.inner.eval(k, zeta)
            for k in range(self._bf.inner_n_basis)
        })

    def layer_phi_of_z(self, layer_idx):
        """Zstruct of inner test functions at ``ζ_i(z) = (z − z_i)/h_i``.

        Equivalent to ``basis.phi_of_z`` but for one layer of a
        :class:`LayeredBasis`.  Use **before** depth-integration —
        ``Multiply(basis.layer_phi_of_z(i), outer=True)`` on a layer's
        branch produces its Galerkin test equations, one per inner
        moment, each carrying the rescaled ``φ_inner_k((z − z_i)/h_i)``
        factor that the Leibniz rule will dismantle.

        Raises ``ValueError`` on a non-layered basis.
        """
        from zoomy_core.misc.misc import Zstruct
        from zoomy_core.model.models.basisfunctions import LayeredBasis
        if not isinstance(self._bf, LayeredBasis):
            raise ValueError("layer_phi_of_z requires a LayeredBasis.")
        if not (0 <= layer_idx < self._bf.n_layers):
            raise IndexError(f"layer_idx={layer_idx} out of range.")
        zeta_i_of_z = self._bf.layer_zeta(layer_idx, self._state.z)
        return Zstruct(**{
            f"phi_{k}": self._bf.inner.eval(k, zeta_i_of_z)
            for k in range(self._bf.inner_n_basis)
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simplify_derivatives_only(expr):
    """Simplify Derivative sums without touching Integrals.

    Turns ``d(H+b)/dt - db/dt`` into ``dH/dt`` while preserving
    ``Derivative(Integral(...), x)`` and bare ``Integral(...)`` nodes.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    # Protect anything involving Integrals
    integral_map = {}
    counter = [0]

    def _protect(e):
        # Protect Derivative(Integral(...), ...) as a unit
        if isinstance(e, Derivative) and e.args[0].has(Integral):
            key = sp.Dummy(f"_DINT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        # Protect bare Integrals
        if isinstance(e, Integral):
            key = sp.Dummy(f"_INT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if e.args:
            new_args = [_protect(a) for a in e.args]
            return e.func(*new_args)
        return e

    protected = _protect(expr)
    # Now safe to simplify — only pure Derivative(Function, var) remain
    simplified = protected.doit() if protected.has(Derivative) else protected
    # Restore integral-containing nodes
    return simplified.subs(integral_map)


def _simplify_preserve_integrals(expr):
    """Expand + cancel while protecting Integral and Derivative(Integral) terms.

    Cancels terms like u²·d(b+h)/dx - u²·db/dx → u²·dh/dx and
    (-gρ(b+h) + gρb + gρh)·... → 0, while leaving ∫...dz terms intact.

    Uses expand + powsimp (not simplify) to avoid common denominator wrapping.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    integral_map = {}
    counter = [0]

    def _protect(e):
        if isinstance(e, Derivative) and e.args[0].has(Integral):
            key = sp.Dummy(f"_DINT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if isinstance(e, Integral):
            key = sp.Dummy(f"_INT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if e.args:
            new_args = [_protect(a) for a in e.args]
            return e.func(*new_args)
        return e

    protected = _protect(expr)
    # Evaluate derivatives by *linearity only* — ``d(b+h)/dt → db/dt + dh/dt``
    # — but do NOT chain-rule through ``Derivative(u², x)`` or ``Derivative(u·w, z)``.
    # Conservative-form derivatives must survive simplification so that a
    # later ``Integrate(..., method="auto")`` can apply the Leibniz rule /
    # fundamental theorem cleanly.  ``Derivative(Integral(...))`` is
    # already protected above.
    evaled = _evaluate_linear_derivatives(protected)
    # Expand to reveal cancellations, then collect like terms.  We pass
    # ``mul=True, multinomial=True`` explicitly and keep Derivatives unexpanded.
    expanded = sp.expand(evaled)
    simplified = sp.powsimp(expanded, combine="all")
    return simplified.subs(integral_map)


def _resolve_subs_safe(expr):
    """Unwrap every ``Subs(f, var, val)`` in ``expr`` whose inner is safe
    to ``xreplace(var → val)``.

    "Safe" means **none** of the following:

    * ``var`` appears as a ``Derivative`` differentiation variable in
      ``f`` — would commit to chain-rule-through-val ambiguity sympy is
      deliberately conservative about.
    * ``var`` is the integration variable of an ``Integral`` nested in
      ``f`` — a naive ``xreplace`` would overwrite the integration
      binder's tuple and produce nonsense limits (``Integral(_,
      (val, lo, val))``).  This protects running integrals introduced
      by :class:`ContinuityClosure`.
    * ``var`` is the binding variable of a nested ``Subs`` in ``f``
      — same shadowing concern.

    When all guards pass, the Subs is replaced structurally via
    ``f.xreplace({var: val})`` — using ``xreplace`` on purpose, so
    that running integrals / nested Subs / ``ζ`` appearances the
    caller *did* want subbed (top-level free occurrences) are resolved
    rather than refused by sympy's ``.subs`` "dummy dependency"
    guard.
    """
    if not expr.has(sp.Subs):
        return expr
    mapping = {}
    for s in expr.atoms(sp.Subs):
        inner = s.args[0]
        vars_tup = s.args[1]
        vals_tup = s.args[2]
        deriv_vars = set()
        for d in inner.atoms(sp.Derivative):
            deriv_vars.update(d.variables)
        if any(v in deriv_vars for v in vars_tup):
            continue
        bound_vars = set()
        for I in inner.atoms(sp.Integral):
            for lim in I.args[1:]:
                if hasattr(lim, "__getitem__") and len(lim) >= 1:
                    bound_vars.add(lim[0])
        for nested in inner.atoms(sp.Subs):
            bound_vars.update(nested.args[1])
        if any(v in bound_vars for v in vars_tup):
            continue
        mapping[s] = inner.xreplace(dict(zip(vars_tup, vals_tup)))
    return expr.xreplace(mapping) if mapping else expr


def _evaluate_linear_derivatives(expr):
    """``Derivative(Add(...) | expandable-to-Add, var)`` → sum of per-term derivatives.

    Evaluates the Derivative only when it can be resolved by linearity.
    Concretely: if ``sp.expand(inner)`` is an ``Add``, distribute the
    Derivative across the summands (``.doit()`` on the Add-inner form).
    Otherwise leave the Derivative intact so conservative forms like
    ``Derivative(u**2, x)`` and ``Derivative(u*w, z)`` survive.

    Trivial zero collapses are handled here too: ``Derivative(0, var)``
    → ``0`` and ``Subs(0, var, val)`` → ``0``.  Without this,
    post-``hydrostatic_scaling`` z-momentum carries leftover
    ``∂/∂t 0``, ``∂/∂x 0``, and ``Subs(∂/∂Dummy 0, Dummy, b+h)``
    residue from the analytical integrate (Dummies come from
    ``sp.integrate``).  Sympy refuses to reduce these on its own —
    ``Derivative.doit()`` only evaluates when the inner depends on the
    diff variable, and ``Subs`` stays lazy for the same chain-rule
    reason that keeps ``Subs(Derivative(u, x), z, b)`` frozen.

    We expand the inner first so that factored forms like
    ``Derivative(g*ρ*(η − z), x)`` — which are algebraically
    ``Derivative(−g*ρ*z + g*ρ*b + g*ρ*h, x)`` — also get linearly
    distributed, producing ``g*ρ*∂_x b + g*ρ*∂_x h`` (and 0 for the
    z-independent-of-x part).  Without the expand-first step, the
    factored form stays as a single opaque Derivative and its tag
    cannot be follow-the-term propagated during ``.apply``.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    if isinstance(expr, sp.Subs) and expr.args[0] == S.Zero:
        return S.Zero
    if isinstance(expr, Derivative):
        inner = expr.args[0]
        if inner == S.Zero:
            return S.Zero
        # Trivially zero: inner doesn't mention any of the diff vars.
        # Without this shortcut, expressions like Derivative(-g·ρ·z, x)
        # (coming out of the hydrostatic pressure solve) would survive
        # our post-distribute pass.  ``sympy.Derivative.doit()`` would
        # handle it, but we can't ``.doit()`` the Add-distributed form
        # in general (it also applies the product rule, flattening
        # conservative ``Derivative(ν·∂_z u, z)`` shapes that the
        # fundamental theorem of calculus needs).  The targeted
        # "no diff-var inside" check keeps both cases correct.
        if not any(inner.has(v) for v in expr.variables):
            return S.Zero
        inner_expanded = sp.expand(inner) if isinstance(inner, sp.Basic) else inner
        if isinstance(inner_expanded, sp.Add):
            inner_walked = _evaluate_linear_derivatives(inner_expanded)
            # Distribute the outer Derivative over the Add via linearity
            # **without** ``.doit()``.  ``.doit()`` at this stage applies
            # the product rule inside each summand, turning conservative
            # shapes like ``Derivative(ν·∂_z u, z)`` into the expanded
            # ``∂_z ν · ∂_z u + ν · ∂²u/∂z²`` form.  The second piece
            # then has a two-variable Derivative tuple that
            # ``_extract_derivative`` refuses, and the fundamental
            # theorem of calculus can no longer integrate the term.
            # By building the distributed sum manually we keep each
            # summand in conservative ``Derivative(coeff·f, var)`` form,
            # where Leibniz / fund-thm apply cleanly.
            args_rest = expr.args[1:]
            return sp.Add(*(expr.func(t, *args_rest)
                            for t in sp.Add.make_args(inner_walked)))
        inner_walked = _evaluate_linear_derivatives(inner)
        return expr.func(inner_walked, *expr.args[1:])
    if expr.args:
        return expr.func(*(_evaluate_linear_derivatives(a) for a in expr.args))
    return expr


def _extract_derivative(expr, var):
    """Decompose ``expr`` as ``coeff * Derivative(inner, var)``.

    Returns ``(inner, coeff)`` when the decomposition is valid,
    ``(None, None)`` otherwise.  Validity requires that ``coeff`` is
    independent of ``var``.

    Without that check, a product-rule-expanded term like
    ``2u · ∂_x u`` would be split as ``coeff = 2u, inner = u`` and the
    downstream Leibniz rule ``∫ coeff · ∂_x inner dz = ∂_x[∫ coeff ·
    inner dz] + …`` would compute ``∂_x[∫ 2u² dz]`` — **double** the
    correct value, since the true conservative form ``∂_x(u²)`` gives
    ``∂_x[∫ u² dz]``.

    So we only succeed when the term is in genuine conservative form:
    either a bare ``Derivative(f, var)``, or a product whose only
    var-dependent factor is the ``Derivative``.  Expanded product-rule
    forms fall through to "direct" integration and stay wrapped in an
    unevaluated ``Integral``.
    """
    expr = sp.sympify(expr)
    if isinstance(expr, Derivative):
        if expr.variables == (var,):
            return expr.args[0], S.One
    factors = Mul.make_args(expr)
    coeff_factors = []
    inner = None
    for f in factors:
        if isinstance(f, Derivative) and f.variables == (var,) and inner is None:
            inner = f.args[0]
        else:
            coeff_factors.append(f)
    if inner is None:
        return None, None
    coeff = Mul(*coeff_factors) if coeff_factors else S.One
    # Refuse when ``coeff`` still depends on the derivative variable —
    # the term is product-rule expanded, not conservative.
    if var in coeff.free_symbols:
        return None, None
    return inner, coeff


def integrate_by_parts(f, g, var, domain=(0, 1)):
    """
    Standalone IBP: integral d(f)/dvar * g dvar = [f*g]_a^b - integral f * dg/dvar dvar
    Returns IBPResult(integrate, boundary_upper, boundary_lower).
    """
    a, b = domain
    return IBPResult(
        integrate=Expression(-Integral(f * Derivative(g, var), (var, a, b)), "ibp_integrate"),
        boundary_upper=Expression((f * g).subs(var, b), "ibp_upper"),
        boundary_lower=Expression((f * g).subs(var, a), "ibp_lower"),
    )


def gauss_legendre_integrate(expr, var, a, b, order=4):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes_shifted = [(b - a) / 2 * xi + (b + a) / 2 for xi in nodes]
    weights_scaled = [(b - a) / 2 * wi for wi in weights]
    result = S.Zero
    for xi, wi in zip(nodes_shifted, weights_scaled):
        result += sp.Rational(wi).limit_denominator(10**8) * expr.subs(var, sp.nsimplify(xi))
    return result
