"""Model framework — Term, Equation, Model, Symmetrize.

This is the canonical model representation used by the
``sigmaref / sme / vam / mlme / mlvam`` derivation files.  It is the
direct port of the framework developed in the transparent-derivation
notebooks (``thesis/notebooks/modeling/transparent_derivations/
sme_clean.py`` etc.) — same classes, same auto-history mechanics,
same ``simplify`` pipeline (sp.expand → pull var-independent factors
out of Derivative atoms → recover fluxes by pulling var-independent
coeffs back inside Derivative arguments).
"""

from __future__ import annotations

import itertools
import sympy as sp

from zoomy_core.model.models.legacy.ins_generator import Expression, Operation


__all__ = [
    "Term",
    "Equation",
    "Model",
    "Symmetrize",
    "_TermGroup",
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
    ``c`` is *var-independent*, pull ``c`` back inside the Derivative
    so the conservative atom ``Derivative(c · inner, var)`` survives —
    that's what the flux/hydrostatic tagger reads."""
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


# ── Term / Equation / Model ───────────────────────────────────────────


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
    ``model.equations['momentum_x'][[0, 1, 7]].apply(ProductRule(...))``
    over a hand-picked index list."""

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


class Model:
    """Ordered dictionary of :class:`Equation` + linear operation
    history.  ``apply`` routes any op across every equation and
    records one ``target=*`` history entry per call; ``multiply`` and
    ``resolve_dummy`` similarly act over the whole equation set."""

    def __init__(self, name="Model"):
        self.name = name
        self.equations = {}
        self.history = []

    # ── add_equation ─────────────────────────────────────────────────
    def add_equation(self, name, expression=None, shape=None):
        if shape is None:
            expr = expression if expression is not None else sp.S.Zero
            self.equations[name] = Equation(expr, name=name, model=self)
            self._history("add_equation", name)
            return self
        for combo in self._iter_shape(shape):
            sub = "_".join([name] + list(combo))
            self.equations[sub] = Equation(sp.S.Zero, name=sub, model=self)
            self._history("add_equation", sub)
        return self

    @staticmethod
    def _iter_shape(shape):
        if all(isinstance(c, str) for c in shape):
            for c in shape:
                yield (c,)
        else:
            for combo in itertools.product(*shape):
                yield combo

    # ── apply ────────────────────────────────────────────────────────
    def apply(self, *args, level="major", description=None, **kwargs):
        for eq in self.equations.values():
            eq.apply(*args, _no_history=True, **kwargs)
        op = args[0] if args else None
        self._history(
            getattr(op, "name", None) or type(op).__name__, "*",
            level=level, description=description,
        )
        return self

    # ── multiply (scalar / list-branching) ──────────────────────────
    def multiply(self, factor, *, level="major"):
        """Multiply every equation by ``factor`` (a scalar or a list).
        A list branches each equation into one sub-equation per element
        (outer product)."""
        if isinstance(factor, (list, tuple)):
            new_eqs = {}
            for name, eq in self.equations.items():
                for i, f in enumerate(factor):
                    sub = f"{name}_{i}"
                    new_eqs[sub] = Equation(eq.expr * f, name=sub, model=self)
            self.equations = new_eqs
            self._history("multiply", "*", level=level,
                          description=f"branched ×{len(factor)}")
            return self
        for eq in self.equations.values():
            eq.expr = eq.expr * factor
        self._history("multiply", "*", level=level,
                      description=f"× {factor}")
        return self

    # ── resolve_dummy (scalar substitution / list-branching) ────────
    def resolve_dummy(self, dummy, value, *, level="minor"):
        """Replace every ``dummy(arg)`` call across the model.

        * ``value`` callable / sympy expression → single substitution.
        * ``value`` list → branch every equation that contains the
          dummy into one sub-equation per list element.
        """
        if isinstance(value, (list, tuple)):
            new_eqs = {}
            for name, eq in self.equations.items():
                if not eq.expr.has(dummy):
                    new_eqs[name] = eq
                    continue
                for i, v in enumerate(value):
                    new_expr = self._make_replace(v)(eq.expr, dummy)
                    sub = f"{name}_{i}"
                    new_eqs[sub] = Equation(new_expr, name=sub, model=self)
            self.equations = new_eqs
            label = f"branched ×{len(value)}"
        else:
            replace = self._make_replace(value)
            for eq in self.equations.values():
                eq.expr = replace(eq.expr, dummy)
            label = "single substitution"
        self._history("resolve_dummy", "*", level=level,
                      description=f"{dummy.__name__} → {label}")
        return self

    @staticmethod
    def _make_replace(value):
        if callable(value) and not isinstance(value, sp.Basic):
            return lambda expr, dummy: expr.replace(dummy, value)
        return lambda expr, dummy: expr.replace(
            dummy, lambda *args, _v=value: _v)

    # ── history ─────────────────────────────────────────────────────
    def _history(self, op_label, target, *, level="major", description=None):
        self.history.append({"op": op_label, "target": target,
                              "level": level,
                              "description": description or ""})

    # ── display ─────────────────────────────────────────────────────
    def _repr_latex_(self):
        rows = []
        for name, eq in self.equations.items():
            rhs = sp.latex(eq.expr)
            lhs = name.replace("_", r"\_")
            rows.append(rf"\mathtt{{{lhs}}} \;&:\; {rhs} \;=\; 0")
        body = r" \\[4pt] ".join(rows)
        return rf"$$\begin{{aligned}}{body}\end{{aligned}}$$"

    def __str__(self):
        lines = [f"Model({self.name!r}) — {len(self.equations)} equations, "
                 f"{len(self.history)} ops"]
        for name, eq in self.equations.items():
            tags = sorted({t.tag for t in eq if t.tag}) or ["(untagged)"]
            lines.append(f"  {name}  :  {eq.expr}  =  0   "
                         f"[{', '.join(tags)}]")
        return "\n".join(lines)

    __repr__ = __str__

    def describe(self, *, show_history=False, include_minor=False):
        if show_history:
            print(self)
            print("\nhistory:")
            for h in self.history:
                if not include_minor and h.get("level") == "minor":
                    continue
                desc = f" — {h['description']}" if h.get("description") else ""
                print(f"  [{h['op']}] target={h['target']}{desc}")
        return self


# ── Symmetrize (general self-pair chain-rule technique) ───────────────


class Symmetrize(Operation):
    """``a → a/2 + rule(a/2)``.  When ``rule = ProductRule(inverse)``
    and the term has the self-pair shape ``c · f · ∂_v(f)``, the
    cancellation under :meth:`Equation.simplify` leaves a clean
    conservative atom ``c/2 · ∂_v(f²)``.  Used for the gravity
    self-pair fold."""

    def __init__(self, rule):
        self.rule = rule
        super().__init__(
            name="symmetrize",
            description=f"½·a + {getattr(rule, 'name', type(rule).__name__)}(½·a)",
        )

    def _leaf_sp(self, expr):
        half = expr / 2
        applied = Expression(half, "").apply(self.rule).expr
        return sp.expand(half + applied)
