"""Derivation operations for the clean-redesign framework.

These ops slot into the existing :class:`zoomy_core.model.operations.Operation`
spine — same ``_leaf_sp`` / ``apply_to`` / ``whole_model_op`` /
``transforms_bcs`` protocol — so an :class:`~zoomy_core.model.derivation.Equation`'s
``.apply`` and a :class:`~zoomy_core.model.derivation.Model`'s broadcast both dispatch
through them unchanged.

Plain substitutions are now just oriented :class:`~zoomy_core.model.equation.Equation`
objects (or bare ``{lhs: rhs}`` dicts) consumed by ``.apply`` — there is no
``Substitution`` class.  What remains here is the one op that does MORE than a
substitution:

``ChangeOfVariables``
    A coefficient-family rename (``a_i -> relation(q_i)``) that ALSO swaps the
    unknown family in the model's ``Q`` (``a -> q``) via
    ``model.redeclare_unknown``.  That Q-swap is exactly why it is an Operation
    and not a substitution: a substitution rewrites expressions but leaves the
    declared-unknown bookkeeping alone.

``granularity``
    Every op exposes a ``granularity`` attribute (one of ``LEAF`` / ``TERM`` /
    ``EQUATION`` / ``SYSTEM``) computed from the reused ``single_term_only`` /
    ``whole_model_op`` / ``system_level`` flags, so callers can introspect at
    which structural level an op acts.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation


__all__ = ["SolveFor", "SolveLinearSystem", "ChangeOfVariables",
           "Granularity", "granularity_of"]


# ── SolveFor ───────────────────────────────────────────────────────────────


class SolveFor(Operation):
    """Reorient the equation it is applied to into ``variable = solution``,
    IN PLACE — the pipeline-style counterpart of ``equation.solve_for(var)``
    (which returns a NEW equation and leaves the original untouched).

    After ``eq.apply(SolveFor(x))`` the row IS the oriented relation ``x = …``:
    it renders as ``x = …`` in ``describe`` and applies as the substitution
    ``x → …`` wherever the equation is handed to ``.apply``.  Use it when you
    want to rewrite a model row into solved form and keep it in the model::

        m.add_equation("omega", h*omega - (wt - ...))
        m.omega.apply(SolveFor(omega))      # m.omega is now  ω = (...)/h
    """

    def __init__(self, variable, name=None):
        self._var = variable
        super().__init__(name=name or f"solve_for[{variable}]")

    def apply_to_equation(self, eq):
        solved = eq.solve_for(self._var)        # oriented Equation
        eq.expr = solved.expr                   # residual  var - solution
        eq._as_relation = dict(solved._as_relation)
        return eq

    def __repr__(self):
        return f"SolveFor({self._var})"


# ── SolveLinearSystem ──────────────────────────────────────────────────────


class SolveLinearSystem:
    """Assemble and solve a COUPLED linear system for a family of variables.

    :class:`SolveFor` isolates ONE variable from ONE equation; many moment
    closures instead need a COUPLED solve — e.g. the continuity moments
    ``m.mass[1:N+2]`` for the vertical-velocity coefficients ``ŵ_0 … ŵ_N``.

    ``SolveLinearSystem(equations, variables).solve()`` builds ``A·v = b``
    (``A[i][j]`` = coefficient of ``variables[j]`` in ``equations[i]``, ``b`` the
    remainder), solves it with :func:`sympy.linsolve`, and returns the solution
    as a :class:`~zoomy_core.model.derivation.model.MomentFamily` whose component
    ``i`` is the oriented row ``variables[i] = solution_i``.  Opaque Galerkin
    brackets are hidden behind symbols for the solve and restored after, so it
    works both post-``ResolveBasis`` (numeric coefficients — fast) and, in
    principle, on the abstract bracket form.

    The returned family is

    * INDEXABLE  — ``sol[i]`` is the oriented row for ``variables[i]``;
    * APPLICABLE — it carries a combined ``_as_relation`` so ``eq.apply(sol)``
      substitutes every ``variables[i] -> solution_i`` at once;
    * ADDABLE    — ``model.add_equation("w_coefs", sol)``.

    Example — the SME ``w``-closure::

        h_eq = m.mass[0].solve_for(d.t(h))        # k=0 → the h-equation
        for row in m.mass[1:4]:                    # substitute ∂_t h
            row.apply(h_eq)
        sol = SolveLinearSystem(m.mass[1:4],
                                [wh(j, t, x) for j in range(3)]).solve()
        m.momentum.x.apply(sol)                    # insert ŵ_0, ŵ_1, ŵ_2 at once
        sol[0]                                     # the oriented row  ŵ_0 = …
    """

    def __init__(self, equations, variables, name="solve_linear_system"):
        self._eqs = list(equations)
        self._vars = list(variables)
        self.name = name

    def solve(self):
        from zoomy_core.model.derivation.model import MomentFamily
        from zoomy_core.model.equation import Equation
        from zoomy_core.model.operations import is_bracket

        exprs = [sp.expand(sp.sympify(getattr(e, "expr", e))) for e in self._eqs]
        vars_ = list(self._vars)
        n = len(vars_)
        # Hide opaque Galerkin brackets behind symbols so the solver sees clean
        # (rational / numeric) coefficients; restore them in the solution.
        igs = sorted({ig for ex in exprs for ig in ex.atoms(sp.Integral)
                      if is_bracket(ig)}, key=sp.srepr)
        hide = {ig: sp.Symbol(f"_Br{i}") for i, ig in enumerate(igs)}
        restore = {s: ig for ig, s in hide.items()}
        # ``variables`` are applied Functions (``ŵ_j(t,x)``), not Symbols —
        # linsolve wants Symbols, so swap in placeholders for the solve.
        ph = [sp.Symbol(f"_v{i}") for i in range(n)]
        vsub = {v: ph[i] for i, v in enumerate(vars_)}
        back = {ph[i]: vars_[i] for i in range(n)}
        H = [ex.xreplace(hide).xreplace(vsub) for ex in exprs]
        sols = sp.linsolve(H, ph)
        if not sols:
            raise ValueError(
                "SolveLinearSystem: inconsistent system (linsolve returned ∅).")
        tup = list(sols)[0]
        comps, relmap = [], {}
        for i, v in enumerate(vars_):
            si = sp.expand(tup[i].xreplace(restore).xreplace(back))
            comps.append(Equation(sp.Eq(v, si), name=f"{self.name}_{i}"))
            relmap[v] = si
        fam = MomentFamily(name=self.name, modes=list(range(n)), components=comps)
        # Combined relation so ``eq.apply(sol)`` substitutes the whole family
        # (the ``.apply`` dispatch duck-types on ``_as_relation``).
        fam._as_relation = relmap
        return fam

    def __repr__(self):
        return (f"SolveLinearSystem({len(self._eqs)} eqs, "
                f"{len(self._vars)} vars)")


# ── granularity ──────────────────────────────────────────────────────────


class Granularity:
    """Structural level an :class:`Operation` acts on.

    Mapped onto the reused operation flags:

      * ``SYSTEM``   — ``system_level`` (couples every equation; e.g. an
                        ``InvertMassMatrix``-style whole-DAE op).
      * ``EQUATION`` — ``whole_model_op`` (grows / reshapes the equation
                        dict; e.g. an outer-product ``Substitution``).
      * ``TERM``     — ``single_term_only`` (must target one additive term).
      * ``LEAF``     — the default (per-additive-term broadcast inside one
                        equation).
    """

    LEAF = "LEAF"
    TERM = "TERM"
    EQUATION = "EQUATION"
    SYSTEM = "SYSTEM"


def granularity_of(op) -> str:
    """Return the :class:`Granularity` level an op acts on, reading the
    reused operation flags (``system_level`` / ``whole_model_op`` /
    ``single_term_only``)."""
    if getattr(op, "system_level", False):
        return Granularity.SYSTEM
    if getattr(op, "whole_model_op", False):
        return Granularity.EQUATION
    if getattr(op, "single_term_only", False):
        return Granularity.TERM
    return Granularity.LEAF


# ── ChangeOfVariables ────────────────────────────────────────────────────


class ChangeOfVariables(Operation):
    """Coefficient-family change of variables that swaps the unknown family.

    ``ChangeOfVariables("a", "q", relation)`` rewrites every application
    ``a(i, *coords) -> relation(q(i, *coords))`` across the whole model AND
    calls ``model.redeclare_unknown(a -> q)`` so the state vector ``Q`` swaps
    the ``a`` family for the ``q`` family.

    The Q-swap is exactly what distinguishes this from a bare
    :class:`Substitution`: a substitution rewrites expressions but leaves the
    declared-unknown bookkeeping alone, whereas a change of variables *is* a
    redeclaration of the unknowns.

    Parameters
    ----------
    old, new : str
        Coefficient family head names (``"a"``, ``"q"``).  ``old(...)`` is the
        family currently in ``Q``; ``new`` is what replaces it.
    relation : callable
        ``relation(new_applied) -> sympy expr`` giving the substituted form,
        e.g. ``lambda q_i: q_i / h``.  Called with the *applied* ``new``
        Function so the arguments (mode index + coords) carry through.
    """

    # A change of variables grows / rewrites the whole equation set and swaps
    # Q — it is a model-level op routed through ``apply_to_model``.
    whole_model_op = True
    transforms_bcs = True

    def __init__(self, old, new, relation, name=None):
        self._old = old
        self._new = new
        self._relation = relation
        super().__init__(
            name=name or f"change_of_variables[{old}->{new}]",
            description=f"Change of variables {old}_i -> relation({new}_i)",
        )

    def apply_to_model(self, model):
        # Match the old family by NAME — reconstructing ``sp.Function(self._old)``
        # would drop the field's assumptions (``real=True``) and match nothing.
        # The real head (with assumptions) is recovered from the equations and
        # handed to ``redeclare_unknown`` so the Q swap matches too.
        old_head = next(
            (f.func for eq in model._equations.values()
             for f in eq.expr.atoms(sp.Function)
             if f.func.__name__ == self._old),
            sp.Function(self._old, real=True))
        # Mint the new head with the SAME assumptions as the old family — a real
        # ``a`` gives a real ``q``, a plain ``a`` a plain ``q``.  Hardcoding
        # ``real=True`` would create a ``q`` that ``expr.has`` cannot match
        # against an assumption-free ``Function(self._new)``.
        new_head = sp.Function(
            self._new, **getattr(old_head, "_explicit_class_assumptions", {}))

        def _rewrite(expr):
            if not hasattr(expr, "replace"):
                return expr
            return expr.replace(
                lambda e: (isinstance(e, sp.Function)
                           and e.func.__name__ == self._old),
                lambda e: self._relation(new_head(*e.args)))

        for eq in model._equations.values():
            eq.expr = _rewrite(eq.expr)
            # An ORIENTED relation (e.g. the w-reconstruction ``w̃ = …``) keeps a
            # separate ``_as_relation`` — rewrite its lhs/rhs too.
            rel = getattr(eq, "_as_relation", None)
            if rel:
                eq._as_relation = {_rewrite(k): _rewrite(v)
                                   for k, v in rel.items()}

        # Swap the unknown family in Q (and let the model recompute Qaux).  A
        # change of variables is a genuine family RENAME, so the logical Q key
        # follows the new family (``a → q``), not the old one.
        model.redeclare_unknown(old_head, new_head, rename_key=True)
        model._refresh_unknowns()
        return model

    def __repr__(self):
        return (f"ChangeOfVariables({self._old!r} -> {self._new!r}, "
                f"{self._relation!r})")
