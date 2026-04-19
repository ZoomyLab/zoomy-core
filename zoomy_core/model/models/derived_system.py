"""
Reusable derived equation systems.

Equations live in a **tree**, not a flat dict.  A ``System`` owns a
``Zstruct`` whose leaves are ``Expression`` objects.  Intermediate
``Zstruct`` nodes group related equations: the ``continuity`` scalar
lives as a leaf directly under the root, while ``momentum`` is an
intermediate ``Zstruct(x=Expression, y=Expression, z=Expression)``.

Attribute access walks the tree::

    system.continuity            # → Expression (leaf)
    system.momentum              # → _NodeProxy over the momentum Zstruct
    system.momentum.x            # → _NodeProxy over x_momentum Expression
    system.momentum.x.apply(op)  # mutate that leaf in place

``apply`` / ``simplify`` / ``subs`` invoked on an intermediate node
recurse into every leaf below it.  Rank-changing ops (``Multiply`` with
a basis, ``outer=True``) may replace a leaf with a new ``Zstruct``
subtree; that's just another tree edit.

Boundary-condition storage (``SystemBoundaryConditions``) is pending a
dedicated refactor alongside the tree — it's not provided by this
module right now.
"""

from __future__ import annotations

import pickle
import warnings
from typing import Iterator, Tuple

import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Expression,
    KinematicBCBottom, KinematicBCSurface, HydrostaticPressure,
    Newtonian, Inviscid,
)


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

def _is_intermediate(node) -> bool:
    return isinstance(node, Zstruct)


def _iter_leaves(tree: Zstruct, prefix=()) -> Iterator[Tuple[tuple, Expression]]:
    """Yield ``(path_tuple, Expression)`` for every leaf under ``tree``."""
    for key in tree._filter_dict():
        val = getattr(tree, key)
        new_prefix = prefix + (key,)
        if _is_intermediate(val):
            yield from _iter_leaves(val, new_prefix)
        elif isinstance(val, Expression):
            yield (new_prefix, val)
        # else: skip non-Expression, non-Zstruct nodes (shouldn't occur)


def _tree_path(tree: Zstruct, path) -> Expression | Zstruct:
    node = tree
    for p in path:
        node = getattr(node, p)
    return node


def _tree_set(tree: Zstruct, path, value) -> None:
    """Set the node at ``path`` (must be non-empty)."""
    if not path:
        raise ValueError("Cannot set root via _tree_set")
    parent = tree
    for p in path[:-1]:
        parent = getattr(parent, p)
    setattr(parent, path[-1], value)


def _tree_walk_apply(tree, op_callable) -> None:
    """Apply ``op_callable(Expression) → Expression`` in place to every leaf."""
    for key in list(tree._filter_dict()):
        val = getattr(tree, key)
        if _is_intermediate(val):
            _tree_walk_apply(val, op_callable)
        elif isinstance(val, Expression):
            setattr(tree, key, op_callable(val))


# ---------------------------------------------------------------------------
# _NodeProxy — the uniform interface for leaves AND intermediate nodes
# ---------------------------------------------------------------------------

class _NodeProxy:
    """Proxy into a tree position.

    Resolves each access dynamically against the system's live tree, so
    mutations (including rank-changing ones like ``Multiply(basis)``)
    stay visible through the same proxy.

    On a **leaf** (the proxy points at an ``Expression``):
        ``.apply(op)``    → mutate the Expression in place.
        ``.simplify()``   → same.
        other attributes (``.expr``, ``.terms``, ``.tags``, ``.latex()``, …)
        delegate to the underlying Expression.

    On an **intermediate** (the proxy points at a ``Zstruct``):
        ``.apply(op)``    → recurse into every child leaf.
        ``.simplify()``   → same.
        ``.x``, ``.y``, … → child proxies.
    """

    def __init__(self, system, path):
        object.__setattr__(self, "_system", system)
        object.__setattr__(self, "_path", tuple(path))

    # -- resolution --------------------------------------------------------

    @property
    def _node(self):
        return _tree_path(self._system._tree, self._path)

    def _replace(self, new_node) -> None:
        _tree_set(self._system._tree, self._path, new_node)

    # -- attribute access --------------------------------------------------

    def __getattr__(self, name):
        node = self._node
        if _is_intermediate(node):
            children = node._filter_dict()
            if name in children:
                return _NodeProxy(self._system, self._path + (name,))
            raise AttributeError(
                f"Node {'/'.join(self._path) or 'root'} has no child {name!r}. "
                f"Available: {list(children)}"
            )
        # Leaf Expression: delegate.
        return getattr(node, name)

    def __setattr__(self, name, value):
        raise AttributeError(
            "Assignment on a _NodeProxy is not supported — use "
            "``.apply(...)`` / ``.remove()`` / ``Multiply(...)`` instead."
        )

    def __repr__(self):
        node = self._node
        if _is_intermediate(node):
            return f"_NodeProxy(path={'/'.join(self._path) or 'root'}, " \
                   f"children={list(node._filter_dict())})"
        return f"_NodeProxy({'/'.join(self._path)}): {repr(node)}"

    def __len__(self):
        node = self._node
        if _is_intermediate(node):
            return len(node._filter_dict())
        return len(node)

    def __iter__(self):
        node = self._node
        if _is_intermediate(node):
            for key in node._filter_dict():
                yield _NodeProxy(self._system, self._path + (key,))
        else:
            raise TypeError("Cannot iterate a leaf node")

    # -- operations --------------------------------------------------------

    def apply(self, *args, **kwargs):
        """Apply to this subtree in place.

        Leaf: mutate the Expression.  Intermediate: recurse into every
        leaf descendant.  Returns ``self`` for chaining.
        """
        node = self._node
        if isinstance(node, Expression):
            new_expr = node.apply(*args, **kwargs)
            self._replace(new_expr)
        elif _is_intermediate(node):
            for key in list(node._filter_dict()):
                _NodeProxy(self._system, self._path + (key,)).apply(*args, **kwargs)
        return self

    def simplify(self):
        node = self._node
        if isinstance(node, Expression):
            self._replace(node.simplify())
        elif _is_intermediate(node):
            for key in list(node._filter_dict()):
                _NodeProxy(self._system, self._path + (key,)).simplify()
        return self

    def expand(self):
        node = self._node
        if isinstance(node, Expression):
            self._replace(node.expand())
        elif _is_intermediate(node):
            for key in list(node._filter_dict()):
                _NodeProxy(self._system, self._path + (key,)).expand()
        return self

    def subs(self, *args, **kwargs):
        node = self._node
        if isinstance(node, Expression):
            self._replace(node.subs(*args, **kwargs))
        elif _is_intermediate(node):
            for key in list(node._filter_dict()):
                _NodeProxy(self._system, self._path + (key,)).subs(*args, **kwargs)
        return self

    def apply_to_term(self, index, *operations):
        node = self._node
        if not isinstance(node, Expression):
            raise TypeError("apply_to_term requires a leaf Expression")
        self._replace(node.apply_to_term(index, *operations))
        return self

    def solve_for(self, variable):
        """Solve leaf equation for ``variable``, isolate, return self.

        Only valid on leaves (an intermediate node has no single equation
        to solve).
        """
        node = self._node
        if not isinstance(node, Expression):
            raise TypeError("solve_for requires a leaf Expression")
        solutions = sp.solve(node.expr, variable)
        if not solutions:
            raise ValueError(f"Cannot solve {self._path!r} for {variable}")
        if len(solutions) > 1:
            warnings.warn(
                f"Multiple solutions for {variable} in {self._path!r}, "
                f"using first: {solutions[0]}"
            )
        solution = solutions[0]
        isolated = Expression(variable - solution, self._path[-1] if self._path else "",
                              term_tags=dict(node._term_tags),
                              tag_order=list(node._tag_order),
                              solver_groups=node._solver_groups)
        isolated._as_relation = {variable: solution}
        self._replace(isolated)
        return self

    def copy(self):
        """Deep-copy this subtree into a detached leaf / Zstruct.

        Returns a plain ``Expression`` (leaf) or ``Zstruct`` of ``Expression``
        (intermediate) — *not* a ``_NodeProxy``.  The copy is detached
        from the system, so edits don't propagate back.
        """
        return _deep_clone(self._node)

    def remove(self):
        """Remove this node from the tree.

        Leaves and intermediates both supported.  Removing the last
        child of an intermediate leaves an empty Zstruct; removing the
        intermediate itself removes the whole subtree.
        """
        if not self._path:
            raise ValueError("Cannot remove root")
        parent = _tree_path(self._system._tree, self._path[:-1])
        if hasattr(parent, self._path[-1]):
            delattr(parent, self._path[-1])

    # Legacy alias
    delete = remove

    # -- latex / repr for notebook rendering -------------------------------

    def latex(self, **kwargs):
        node = self._node
        if isinstance(node, Expression):
            return node.latex(**kwargs)
        # Intermediate: concatenate children
        parts = []
        for key in node._filter_dict():
            child = getattr(node, key)
            if isinstance(child, Expression):
                parts.append(f"\\text{{{key}}}: {child.latex(**kwargs)} = 0")
            elif _is_intermediate(child):
                child_proxy = _NodeProxy(self._system, self._path + (key,))
                parts.append(f"\\text{{{key}}}: {child_proxy.latex(**kwargs)}")
        return r"\\ ".join(parts)

    def _repr_latex_(self):
        node = self._node
        if isinstance(node, Expression):
            return node._repr_latex_()
        return f"${self.latex()}$"

    def describe(self, **kwargs):
        from zoomy_core.misc.description import Description
        node = self._node
        if isinstance(node, Expression):
            return node.describe(**kwargs)
        parts = []
        name = self._path[-1] if self._path else "system"
        parts.append(f"**{name}**")
        for key in node._filter_dict():
            child_proxy = _NodeProxy(self._system, self._path + (key,))
            parts.append(str(child_proxy.describe(**kwargs)))
        return Description("\n\n".join(parts))

    # -- tree shape --------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return isinstance(self._node, Expression)


# ---------------------------------------------------------------------------
# Deep-clone utility
# ---------------------------------------------------------------------------

def _deep_clone(node):
    """Return an independent copy of a leaf Expression or Zstruct subtree."""
    if isinstance(node, Expression):
        clone = Expression(node.expr, node.name,
                           term_tags=dict(node._term_tags),
                           tag_order=list(node._tag_order),
                           solver_groups=node._solver_groups)
        rel = getattr(node, "_as_relation", None)
        if rel is not None:
            clone._as_relation = dict(rel)
        return clone
    if _is_intermediate(node):
        new_zs = Zstruct()
        for key in node._filter_dict():
            setattr(new_zs, key, _deep_clone(getattr(node, key)))
        return new_zs
    return node


# ---------------------------------------------------------------------------
# System — tree-authoritative
# ---------------------------------------------------------------------------

class System:
    """A mutable PDE equation system stored as a tree of ``Expression`` leaves.

    Construction::

        system = System("INS", state)
        system._set("continuity", Expression(du_dx + dw_dz))
        system._set("momentum", Zstruct(
            x=Expression(...),
            z=Expression(...),
        ))

    In practice use the :func:`FullINS` builder.

    Attribute access returns a :class:`_NodeProxy`, so
    ``system.momentum.x.apply(op)`` mutates the leaf in place.
    """

    def __init__(self, name, state, tree: Zstruct | None = None, assumptions=None):
        self.name = name
        self.state = state
        self._tree: Zstruct = tree if tree is not None else Zstruct()
        self.assumptions = list(assumptions) if assumptions else []

    # -- structural mutators ----------------------------------------------

    def _set(self, name: str, value):
        """Attach ``value`` (Expression or Zstruct subtree) under ``name``.

        Lower-level than :meth:`add_equation`; used by constructors.
        """
        setattr(self._tree, name, value)

    def add_equation(self, path, expr):
        """Attach a leaf Expression at the dotted-path location.

        ``path`` may be a string (top-level) or a tuple / dotted string
        for nested locations.  Intermediate Zstructs are created on demand.
        """
        if isinstance(path, str):
            parts = tuple(path.split("."))
        else:
            parts = tuple(path)
        node = self._tree
        for p in parts[:-1]:
            if not hasattr(node, p):
                setattr(node, p, Zstruct())
            node = getattr(node, p)
        if not isinstance(expr, Expression):
            expr = Expression(expr, parts[-1])
        setattr(node, parts[-1], expr)

    def remove_equation(self, path):
        if isinstance(path, str):
            parts = tuple(path.split("."))
        else:
            parts = tuple(path)
        parent = self._tree
        for p in parts[:-1]:
            if not hasattr(parent, p):
                return
            parent = getattr(parent, p)
        if hasattr(parent, parts[-1]):
            delattr(parent, parts[-1])

    # -- attribute walk via proxy -----------------------------------------

    def __getattr__(self, name):
        # Reserved attributes on System itself.
        if name.startswith("_") or name in ("name", "state", "assumptions"):
            raise AttributeError(name)
        tree = self.__dict__.get("_tree")
        if tree is None or not hasattr(tree, name):
            raise AttributeError(
                f"System {self.__dict__.get('name', '?')!r} has no top-level "
                f"equation {name!r}. Available: "
                f"{list(tree._filter_dict()) if tree else []}"
            )
        node = getattr(tree, name)
        if isinstance(node, Expression) or _is_intermediate(node):
            return _NodeProxy(self, (name,))
        return node

    # -- iteration over leaves --------------------------------------------

    def leaves(self) -> Iterator[Tuple[tuple, Expression]]:
        yield from _iter_leaves(self._tree)

    # -- apply / simplify / subs on the whole tree ------------------------

    def apply(self, *args, **kwargs):
        """Apply to every leaf.  Returns ``self``."""
        def _op(expr: Expression) -> Expression:
            return expr.apply(*args, **kwargs)
        _tree_walk_apply(self._tree, _op)
        op = args[0] if args else None
        a_name = (getattr(op, "description", None)
                  or getattr(op, "name", None)
                  or (str(op) if op is not None else "apply"))
        self.assumptions.append(a_name)
        return self

    def simplify(self):
        _tree_walk_apply(self._tree, lambda e: e.simplify())
        return self

    def expand(self):
        _tree_walk_apply(self._tree, lambda e: e.expand())
        return self

    def subs(self, *args, **kwargs):
        _tree_walk_apply(self._tree, lambda e: e.subs(*args, **kwargs))
        return self

    def remove(self, path):
        self.remove_equation(path)
        return self

    # -- description ------------------------------------------------------

    def describe(self, header=True, assumptions=True, final_equation=True,
                 parameters=False, strip_args=False):
        from zoomy_core.misc.description import Description

        parts = []
        leaf_paths = [p for p, _ in self.leaves()]
        if header:
            leaf_names = ", ".join(".".join(p) for p in leaf_paths)
            parts.append(f"**{self.name}** ({leaf_names})\n")
        if assumptions and self.assumptions:
            parts.append("**Assumptions:** " + ", ".join(self.assumptions) + "\n")
        if final_equation:
            for path, eq in self.leaves():
                label = ".".join(path)
                use_multiline = bool(eq._term_tags)
                tex = eq.latex(strip_args=strip_args, multiline=use_multiline)
                if use_multiline:
                    parts.append(f"**{label}:**\n$$\n{tex}\n$$\n")
                else:
                    parts.append(f"**{label}:**\n$$\n{tex} = 0\n$$\n")
        if parameters:
            all_syms = set()
            for _, eq in self.leaves():
                all_syms |= eq.expr.free_symbols
            syms = sorted([s for s in all_syms if isinstance(s, sp.Symbol)], key=str)
            if syms:
                sym_str = ", ".join(f"${sp.latex(s)}$" for s in syms)
                parts.append(f"**Parameters:** {sym_str}")
        return Description("\n".join(parts))

    def _repr_markdown_(self):
        return self.describe()._repr_markdown_()

    # -- persistence ------------------------------------------------------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        leaves = list(self.leaves())
        short = ", ".join(f"{'.'.join(p)}({len(e)} terms)" for p, e in leaves)
        return f"System({self.name!r}, [{short}], assumptions={self.assumptions})"


# Alias retained for name-clarity; identical to System.
DerivedSystem = System
