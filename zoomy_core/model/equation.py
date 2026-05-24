"""Equation primitives — `Term`, `_TermGroup`, `Equation`.

Direct port of the framework developed in the transparent-derivation
notebooks (``thesis/notebooks/legacy/modeling/transparent_derivations/
sme_clean.py`` etc.).  Same classes, same auto-history mechanics, same
``simplify`` pipeline (sp.expand → pull var-independent factors out of
Derivative atoms → recover fluxes by pulling var-independent coeffs back
inside Derivative arguments).

Lives in its own module so the `Model` class (in `basemodel.py`) and
the derivation subclasses (`sigmaref.py`, `sme.py`, `vam.py`) can import
`Equation` without dragging in the full operations / state surface.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Expression


__all__ = [
    "Term",
    "_TermGroup",
    "Equation",
    "_pull_constants_out",
    "_recover_fluxes",
]


# ── canonical-form helpers used by `Equation.simplify` ────────────────


def _pull_constants_out(*deriv_args):
    """For a ``Derivative(inner, var)`` atom: split ``inner`` into the
    var-independent and var-dependent parts and lift the
    var-independent part outside.

    For ``inner`` of type ``Add``, distribute the Derivative
    (chain-rule for sums) — otherwise ``as_independent`` would treat
    ``const`` as an additive constant and ``0 * Derivative(rest, var)``
    would wrongly zero the term.
    """
    d = sp.Derivative(*deriv_args)
    if len(d.variables) != 1:
        return d
    var = d.variables[0]
    inner = d.args[0]
    if inner == 0:
        return sp.S.Zero
    if isinstance(inner, sp.Add):
        return sp.Add(*[_pull_constants_out(a, var) for a in inner.args])
    const, rest = inner.as_independent(var, as_Add=False)
    if rest == 1 or rest == 0:
        return sp.S.Zero
    if const == 1:
        return d
    return const * sp.Derivative(rest, var)


def _recover_fluxes(expr):
    """For each leaf term of shape ``c · Derivative(inner, var)`` where
    ``c`` is *var-independent*, pull ``c`` back inside the Derivative so
    the conservative atom ``Derivative(c · inner, var)`` survives — that's
    what the flux/hydrostatic tagger reads."""
    new_terms = []
    for term in sp.Add.make_args(expr):
        if isinstance(term, sp.Mul):
            derivs = [a for a in term.args
                      if isinstance(a, sp.Derivative)
                      and len(a.variables) == 1]
            if len(derivs) == 1:
                d = derivs[0]
                var = d.variables[0]
                coeff = sp.Mul(*[a for a in term.args if a is not d])
                if not coeff.has(var):
                    new_terms.append(
                        sp.Derivative(coeff * d.args[0], var))
                    continue
        new_terms.append(term)
    return sp.Add(*new_terms)


# ── Term / _TermGroup / Equation ─────────────────────────────────────


class Term:
    """One additive term of an :class:`Equation`."""

    __slots__ = ("expr", "tag", "_parent")

    def __init__(self, expr, tag=None, parent=None):
        self.expr = expr
        self.tag = tag
        self._parent = parent

    def apply(self, *args, **kwargs):
        new_expr = Expression(self.expr, "").apply(*args, **kwargs).expr
        pieces = list(sp.Add.make_args(sp.expand(new_expr)))
        if self._parent is None or len(pieces) == 1:
            self.expr = sp.Add(*pieces)
            return self
        idx = next(i for i, t in enumerate(self._parent._terms) if t is self)
        self._parent._terms[idx:idx + 1] = [
            Term(p, parent=self._parent) for p in pieces]
        return self

    def __repr__(self):
        tag = f", tag={self.tag!r}" if self.tag else ""
        return f"Term({self.expr}{tag})"


class _TermGroup:
    """Returned by ``eq[[i, j, k]]``.  Forwards ``apply`` to each
    term in the held sub-list — enables
    ``model.momentum_x[[0, 1, 7]].apply(ProductRule(...))`` over a
    hand-picked index list."""

    def __init__(self, terms):
        self._terms = list(terms)

    def apply(self, *args, **kwargs):
        for t in self._terms:
            t.apply(*args, **kwargs)
        return self


class Equation:
    """Linear list of :class:`Term` carrying the LHS of one PDE/algebraic
    equation.  ``apply`` delegates to ``Expression.apply`` (the library
    operation-dispatch layer) and auto-records a history entry on the
    parent :class:`Model` if one is set."""

    def __init__(self, expression, name="eq", model=None):
        self.name = name
        self._terms = [Term(t, parent=self)
                       for t in self.to_terms(expression)]
        self._model = model
        # Solver-tag groups populated by ``Model._auto_tag_equations``
        # (or by an explicit ``solver_tag`` call).  Mirrors the
        # ``Expression._solver_groups`` interface so
        # :func:`tag_extraction.collect_solver_tag` can extract operator
        # matrices directly from this Equation.
        self._solver_groups = None

    # ── solver-tag interface (mirrors Expression) ────────────────
    @property
    def solver_tags(self):
        """Dict of solver tags (canonical names) → sub-expression, or
        empty dict if none set."""
        return dict(self._solver_groups) if self._solver_groups else {}

    def get_solver_tag(self, name):
        """Return the sp.Expr for a solver tag, or None if not set.
        ``name`` is normalised through ``canonical_solver_tag``."""
        from zoomy_core.model.models.tag_catalog import canonical_solver_tag
        if not self._solver_groups:
            return None
        return self._solver_groups.get(canonical_solver_tag(name))

    def untagged_remainder(self):
        """Return ``self.expr - sum(solver_tags.values())``, simplified.
        Zero ⇒ every term has been routed to a solver tag; non-zero
        drives ``collect_solver_tag``'s ``untagged_policy``."""
        if not self._solver_groups:
            return self.expr
        tagged_sum = sum(self._solver_groups.values(), sp.S.Zero)
        return sp.expand(self.expr - tagged_sum)

    # ── term ↔ expression bridges ─────────────────────────────────
    @staticmethod
    def to_terms(expression):
        return list(sp.Add.make_args(sp.expand(expression)))

    @staticmethod
    def to_expression(terms):
        return sum((t.expr if isinstance(t, Term) else t for t in terms),
                   sp.S.Zero)

    @property
    def expr(self):
        return self.to_expression(self._terms)

    @expr.setter
    def expr(self, value):
        self._terms = [Term(t, parent=self) for t in self.to_terms(value)]

    @property
    def terms(self):
        return self._terms

    # ── indexing + iteration ─────────────────────────────────────
    def __getitem__(self, i):
        if isinstance(i, (list, tuple)):
            return _TermGroup([self._terms[k] for k in i])
        return self._terms[i]

    def __setitem__(self, i, value):
        if isinstance(value, Term):
            value._parent = self
            self._terms[i] = value
        else:
            self._terms[i] = Term(value, parent=self)

    def __iter__(self):
        return iter(self._terms)

    def __len__(self):
        return len(self._terms)

    # ── apply / simplify ─────────────────────────────────────────
    def apply(self, *args, **kwargs):
        # ``_no_history`` is set by ``Model.apply`` when it iterates over
        # every equation — that path adds a single ``target=*`` entry
        # instead of N per-equation duplicates.
        no_history = kwargs.pop("_no_history", False)
        level = kwargs.pop("level", "major")
        description = kwargs.pop("description", None)
        merged = Expression(self.expr, self.name)
        new = merged.apply(*args, **kwargs)
        self.expr = new.expr
        if not no_history and self._model is not None and args:
            op = args[0]
            self._model._history(
                getattr(op, "name", None) or type(op).__name__,
                self.name, level=level, description=description,
            )
        return self

    def simplify(self):
        """Canonical form (three sub-steps):

        1. ``sp.expand`` — flatten Mul/Add nesting and combine like terms.
        2. **Pull var-independent factors out** of every Derivative
           argument: ``Derivative(g·h, x) → g·Derivative(h, x)``,
           ``Derivative(0, x) → 0``.
        3. **Recover fluxes** — for each term ``c · Derivative(_, x)``
           where ``c`` is var-independent, pull ``c`` back inside so the
           tagger sees the conservative atom.
        """
        expr = sp.expand(self.expr)
        expr = expr.replace(sp.Derivative, _pull_constants_out)
        expr = sp.expand(expr)
        expr = _recover_fluxes(expr)
        self.expr = expr
        return self

    def __repr__(self):
        return f"Equation({self.name!r}, {len(self._terms)} terms)"
