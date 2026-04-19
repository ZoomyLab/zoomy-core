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

    def auto_solver_tag(self, *, state_vars, time_var, coords):
        """Algorithmically assign solver tags by structural matching.

        Each additive term of ``self.expr`` is classified:

        * ``coeff * Derivative(q, t)`` with ``q`` a state variable →
          ``time_derivative``.
        * ``coeff * Derivative(F, x_i)`` where ``coeff`` is state-variable-
          free → ``flux`` (conservative form, regardless of whether ``F``
          is pure advection or hydrostatic pressure — they're lumped here;
          well-balanced separation is a follow-up).
        * ``coeff * Derivative(q_j, x_i)`` where ``q_j`` is a state variable
          → ``nonconservative_flux``.
        * No derivatives and no un-substituted Function-call atoms (purely
          algebraic in state-symbol space) → ``source``.
        * Anything else (un-substituted ``f(t,x,z)`` evaluations, unmatched
          Derivatives) stays in the untagged remainder.

        Returns a new Expression with ``solver_groups`` populated. Physical
        tags are preserved.
        """
        state_set = set(state_vars)
        coords_list = list(coords)
        tags = {
            "time_derivative": S.Zero,
            "flux": S.Zero,
            "nonconservative_flux": S.Zero,
            "source": S.Zero,
        }

        def _is_un_subbed_function(term):
            """True iff term contains a Function call (e.g. u(t, x, b))."""
            return any(a.args for a in term.atoms(Function))

        for term in Add.make_args(sp.expand(self.expr)):
            if term == S.Zero:
                continue

            # Un-substituted Function calls cannot be evaluated in state-
            # symbol space → leave untagged.
            if _is_un_subbed_function(term):
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

                # time_derivative: coeff * ∂_t(state_var)
                if v == time_var and inner in state_set:
                    tags["time_derivative"] = tags["time_derivative"] + term
                    continue

                # spatial derivative
                if v in coords_list:
                    coeff_state_refs = state_set & coeff.free_symbols
                    if not coeff_state_refs:
                        # Conservative form (no state vars in coefficient).
                        tags["flux"] = tags["flux"] + term
                        continue
                    if inner in state_set:
                        # NCP: coeff * ∂(state)/∂x_i
                        tags["nonconservative_flux"] = tags["nonconservative_flux"] + term
                        continue

            # Anything with no Derivative at all AND no Function call
            # (filtered above) is pure algebraic in state space → source.
            has_derivative = any(
                isinstance(a, Derivative) for a in term.atoms(Derivative)
            )
            if not has_derivative:
                tags["source"] = tags["source"] + term
                continue

            # Unmatched Derivative shape → untagged (caller sees via remainder).

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

    def apply(self, *conditions):
        """Apply substitutions / Relations / Operations — per term.

        The transformation is applied to every additive term of
        ``self.expr`` individually.  Each output term inherits the tag of
        its parent term (follow-the-term), so the per-term tag dict stays
        in sync with the evolving main expression automatically.

        After Relation/dict-style applies the result is lightly simplified
        via ``_simplify_preserve_integrals`` (linearity-only derivatives,
        cancellations across the full expr).  Terms that cancel vanish
        from both ``self.expr`` and the tag dict.

        Operations that carry their own ``apply_to_expression`` are
        dispatched per term too; when the operation produces multiple
        output pieces per input term (``DepthIntegrate`` etc.) all pieces
        inherit the parent tag.
        """
        def _apply_one(expr, cond):
            if isinstance(cond, Operation):
                if hasattr(cond, 'apply_to_expression'):
                    try:
                        eq = Expression(expr, self.name)
                        return cond.apply_to_expression(eq).expr
                    except NotImplementedError:
                        pass
                return cond.apply_to(expr)
            rel = getattr(cond, "_as_relation", None)
            if isinstance(rel, dict) and rel:
                if expr.has(sp.Subs):
                    subs_map = {s: s.doit() for s in expr.atoms(sp.Subs)}
                    expr = expr.subs(subs_map)
                return expr.subs(rel)
            if isinstance(cond, Relation) or hasattr(cond, 'apply_to'):
                if expr.has(sp.Subs):
                    subs_map = {s: s.doit() for s in expr.atoms(sp.Subs)}
                    expr = expr.subs(subs_map)
                return cond.apply_to(expr)
            elif isinstance(cond, dict):
                if expr.has(sp.Subs):
                    subs_map = {s: s.doit() for s in expr.atoms(sp.Subs)}
                    expr = expr.subs(subs_map)
                return expr.subs(cond)
            elif isinstance(cond, (list, tuple)):
                if len(cond) == 2 and isinstance(cond[0], sp.Basic):
                    return expr.subs(cond[0], cond[1])
                for pair in cond:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        expr = expr.subs(pair[0], pair[1])
                return expr
            elif hasattr(cond, 'apply_to'):
                return cond.apply_to(expr)
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
                 strip_args=False):
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
    """An operation that transforms an Expression (e.g. depth integration).

    Unlike a ``Relation`` (which substitutes symbols), an ``Operation``
    applies a structural transformation.  Works with both
    ``Expression.apply()`` and ``DerivedModel.apply()``.

    Subclasses override ``apply_to(expr)`` which receives and returns
    a sympy expression, or ``apply_to_expression(expression)`` which
    receives and returns an ``Expression`` object.
    """

    def __init__(self, name="", description=None):
        super().__init__(name)
        self.description = description or name

    def apply_to(self, expr):
        """Transform a sympy expression. Override for simple operations."""
        # Default: wrap in Expression, call apply_to_expression, unwrap
        eq = Expression(expr, "")
        result = self.apply_to_expression(eq)
        return result.expr

    def apply_to_expression(self, expression):
        """Transform an Expression object. Override for operations that
        need access to .terms, .map(), etc."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement apply_to or apply_to_expression"
        )

    def _repr_latex_(self):
        return ""


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

    def apply_to_expression(self, eq):
        s = self._state
        return eq.map(lambda t: t.depth_integrate(s.b, s.eta, s.z))

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

    def apply_to(self, expr):
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

    def apply_to(self, expr):
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

    def apply_to_expression(self, expression):
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

    def apply_to(self, expr):
        # Fall-back for older direct-subs call sites that skip apply_to_expression.
        return self.apply_to_expression(Expression(expr, "")).expr

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

    def apply_to(self, expr):
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
    """Transform vertical coordinate: z = ζ·h + b, dz = h·dζ.

    Transforms integrals: ∫_b^{b+h} f(z) dz → h·∫_0^1 f(ζ·h+b) dζ.
    """

    def __init__(self, state):
        super().__init__(
            name="zeta_transform",
            description="Coordinate transform z = ζ·h + b",
        )
        self._z = state.z
        self._b = state.b
        self._h = state.H
        self._zeta = state.zeta

    def apply_to(self, expr):
        from sympy import Integral, Derivative
        z, b, h, zeta = self._z, self._b, self._h, self._zeta

        def _transform(e):
            if isinstance(e, Integral):
                integrand = e.args[0]
                limits = e.args[1]
                var = limits[0]
                if var == z:
                    new_integrand = integrand.subs(z, zeta * h + b) * h
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
        return f"$z = \\zeta \\cdot h + b, \\quad dz = h \\, d\\zeta$"


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

    def apply_to(self, expr):
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

    def describe(self, header=True, final_equation=True, strip_args=False,
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
    """Build the incompressible Navier-Stokes as a ``System``.

    Returns a mutable system with ``.apply()`` for in-place transformations.

    Parameters
    ----------
    state : StateSpace
    equations : list of str, optional
        Which equations to include.  Default: all.
        Options: "continuity", "x_momentum", "y_momentum", "z_momentum".

    Usage::

        ins = FullINS(state)
        ins.z_momentum.apply({state.w: 0})
        ins.x_momentum.apply(HydrostaticPressure(state))
        ins.describe()
    """
    from zoomy_core.model.models.derived_system import System
    builder = _INSBuilder(state)
    system = System("INS", state)

    eq_builders = {
        "continuity": lambda: builder.continuity,
        "x_momentum": lambda: builder.x_momentum,
        "z_momentum": lambda: builder.z_momentum,
    }
    if state.has_y:
        eq_builders["y_momentum"] = lambda: builder.y_momentum

    for name, build_fn in eq_builders.items():
        if equations is None or name in equations:
            system.add_equation(name, build_fn())

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


def _evaluate_linear_derivatives(expr):
    """``Derivative(Add(...) | expandable-to-Add, var)`` → sum of per-term derivatives.

    Evaluates the Derivative only when it can be resolved by linearity.
    Concretely: if ``sp.expand(inner)`` is an ``Add``, distribute the
    Derivative across the summands (``.doit()`` on the Add-inner form).
    Otherwise leave the Derivative intact so conservative forms like
    ``Derivative(u**2, x)`` and ``Derivative(u*w, z)`` survive.

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
    if isinstance(expr, Derivative):
        inner = expr.args[0]
        inner_expanded = sp.expand(inner) if isinstance(inner, sp.Basic) else inner
        if isinstance(inner_expanded, sp.Add):
            inner_walked = _evaluate_linear_derivatives(inner_expanded)
            return expr.func(inner_walked, *expr.args[1:]).doit()
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
