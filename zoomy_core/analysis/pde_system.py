"""``PDESystem`` — the unified model representation for analysis.

A ``PDESystem`` carries:

* ``equations``: a list of sympy expressions.  Each equation is read
  as ``LHS = 0``.  Mix of differential and algebraic (constraint) is
  fine.
* ``fields``:   ordered list of state Function-call atoms — e.g.
  ``[h(t, x), u(t, x)]``.  These are what get linearised /
  plane-wave-expanded.
* ``time``:     the time symbol (typically ``sp.Symbol('t')``).
* ``space``:    list of spatial symbols, e.g. ``[x]`` or ``[x, y]``.
* ``parameters``: dict of constants known to the analysis routines —
  e.g. gravity ``{g: 9.81}``.
* ``aux_fields``: optional algebraically-closed fields (e.g. VAM's
  ``w_N`` from KBC at ξ=0 and ``p_N`` from the surface BC).  These
  are **already eliminated** in ``equations`` if the user has applied
  the closure substitutions.  Listed separately so callers can keep
  track.

The analysis package only ever reads these attributes.  No model
introspection beyond this struct is ever performed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import sympy as sp


@dataclass
class PDESystem:
    equations: List[sp.Expr]
    fields: List[Any]                              # sympy Function-call atoms
    time: sp.Symbol
    space: List[sp.Symbol]
    parameters: Dict[Any, Any] = field(default_factory=dict)
    aux_fields: List[Any] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.equations, list):
            self.equations = list(self.equations)
        if not isinstance(self.fields, list):
            self.fields = list(self.fields)
        if not isinstance(self.space, list):
            self.space = list(self.space)
        # Cheap validation: each field must be a sympy expression that
        # involves at least one of the coordinate symbols.
        coords = {self.time, *self.space}
        for f in self.fields:
            f_sym = sp.sympify(f)
            f_atoms = f_sym.free_symbols if hasattr(f_sym, "free_symbols") \
                else set()
            f_atoms |= {a for a in f_sym.atoms(sp.Symbol)}
            if not (f_atoms & coords):
                raise TypeError(
                    f"Field {f!r} does not depend on any coordinate "
                    f"symbol in {coords!r}; ``fields`` must be Function "
                    f"calls of (t, x[, y, ...])."
                )

    # Convenience helpers ----------------------------------------------------
    def n_equations(self) -> int:
        return len(self.equations)

    def n_fields(self) -> int:
        return len(self.fields)

    def with_substitutions(self, repl: Dict) -> "PDESystem":
        """Return a copy with ``xreplace(repl)`` applied to every equation."""
        return PDESystem(
            equations=[sp.expand(eq.xreplace(repl)) for eq in self.equations],
            fields=self.fields,
            time=self.time,
            space=self.space,
            parameters=self.parameters,
            aux_fields=self.aux_fields,
        )

    def __repr__(self):
        return (
            f"PDESystem(n_eq={self.n_equations()}, "
            f"n_fields={self.n_fields()}, "
            f"space={self.space}, "
            f"time={self.time})"
        )
