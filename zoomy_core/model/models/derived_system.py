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


def _mm_escape(s: str) -> str:
    """Escape a string for a Mermaid node label.

    Mermaid labels inside ``"..."`` accept HTML, so we swap ``"`` and
    ``<``/``>`` for HTML entities that won't terminate the label early.
    """
    return (str(s)
            .replace('"', '&quot;')
            .replace("\n", "<br/>"))


class MermaidDiagram:
    """Markdown-renderable wrapper around a mermaid ``flowchart`` source.

    ``MermaidDiagram(src)._repr_markdown_()`` emits a fenced
    ``mermaid`` code block so Jupyter / any CommonMark + mermaid
    renderer displays it as an inline graph.
    """

    def __init__(self, source: str):
        self.source = source

    def __str__(self) -> str:
        return self.source

    def __repr__(self) -> str:
        head = "\n".join(self.source.splitlines()[:3])
        return f"<MermaidDiagram\n{head}\n  ...\n>"

    def _repr_markdown_(self) -> str:
        return f"```mermaid\n{self.source}\n```"


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

    def apply(self, *args, name=None, description=None, **kwargs):
        """Apply to this subtree in place.

        Leaf: mutate the Expression.  Intermediate: recurse into every
        leaf descendant.  Rank-changing Operations
        (``op.rank_changes_leaf == True``) are invoked directly on the
        leaf and replace it with whatever they return (typically a
        ``Zstruct`` of children).

        Optional kwargs mirror :class:`Operation`:

        * ``name`` — override the history label for this apply
        * ``description`` — override the tooltip / long form

        Both fall back to ``op.name`` / ``op.description`` for
        Operation conds, or to generic ``"apply"`` otherwise.  This is
        the exact same ``(name, description)`` pair used to label
        Operations; one history entry per user ``apply`` call, no
        duplicates from the internal tree walk.

        Returns ``self`` for chaining.
        """
        self._apply_internal(*args, **kwargs)
        op = args[0] if args else None
        self._system._record_history(self._path, op, name, description)
        return self

    def _apply_internal(self, *args, **kwargs):
        """Mutation without history recording (used by System.apply)."""
        node = self._node
        if isinstance(node, Expression):
            op = args[0] if args else None
            if op is not None and (
                    getattr(op, "rank_changes_leaf", False)
                    or getattr(op, "whole_leaf_op", False)):
                # Rank-changing *or* whole-leaf-inspecting ops: skip the
                # per-term dispatch in ``Expression.apply`` and hand
                # them the whole leaf directly so cross-term rules
                # (``Recombine``'s anti-product-rule, etc.) work.
                self._replace(op(node))
            else:
                new_expr = node.apply(*args, **kwargs)
                self._replace(new_expr)
        elif _is_intermediate(node):
            for key in list(node._filter_dict()):
                _NodeProxy(self._system, self._path + (key,))._apply_internal(
                    *args, **kwargs)
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

    def split_integrals(self):
        node = self._node
        if isinstance(node, Expression):
            self._replace(node.split_integrals())
        elif _is_intermediate(node):
            for key in list(node._filter_dict()):
                _NodeProxy(self._system, self._path + (key,)).split_integrals()
        return self

    def merge_integrals(self):
        node = self._node
        if isinstance(node, Expression):
            self._replace(node.merge_integrals())
        elif _is_intermediate(node):
            for key in list(node._filter_dict()):
                _NodeProxy(self._system, self._path + (key,)).merge_integrals()
        return self

    def __getitem__(self, key):
        """Index into a leaf (returns a term as Expression) or an
        intermediate (returns a child proxy by integer index).
        """
        node = self._node
        if isinstance(node, Expression):
            return node[key]
        if _is_intermediate(node) and isinstance(key, int):
            children = list(node._filter_dict())
            if not 0 <= key < len(children):
                raise IndexError(
                    f"_NodeProxy at {'/'.join(self._path) or 'root'!r} "
                    f"has {len(children)} children; index {key} out of range"
                )
            return _NodeProxy(self._system, self._path + (children[key],))
        if _is_intermediate(node) and isinstance(key, str):
            return getattr(self, key)
        raise TypeError(
            f"_NodeProxy at {'/'.join(self._path) or 'root'!r} "
            f"is not subscriptable with {key!r}"
        )

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
        # Structured derivation history: one entry per ``apply`` call
        # (either system-level or proxy-level).  Drives ``history_mermaid``
        # and any other replay/provenance tooling.  ``assumptions`` is a
        # flat list of names kept for back-compat with ``describe``.
        self.history: list[dict] = []
        # Branching metadata.  ``parent`` points at the System this
        # system was ``branch``-ed from (``None`` for the root);
        # ``branch_point`` is the ``len(parent.history)`` at the time of
        # the branch, so combined renderers can draw the fork from the
        # right node in the parent's chain.  Both are ``None`` for a
        # freshly-constructed root system.
        self.parent: "System | None" = None
        self.branch_point: int | None = None

    def branch(self, name=None):
        """Fork this system.  Returns a new ``System`` with:

        * A **deep-cloned tree** — mutations on the branch don't touch
          this system's tree (or vice versa).
        * An **empty history** — the branch starts fresh.  The shared
          prefix lives on this (the parent) system; combined renderers
          reconstruct it from the ``parent`` pointer.
        * ``parent = self`` and ``branch_point = len(self.history)``,
          so :func:`combined_history_mermaid` can draw the fork at the
          right node in the parent's history chain.

        Continue on both branches independently via ``.apply(...)``;
        their histories stay isolated.
        """
        import copy as _copy
        new = _copy.copy(self)           # share scalar metadata (state, name)
        new.name = name or f"{self.name}_branch"
        new._tree = _deep_clone(self._tree)
        new.history = []
        new.assumptions = []
        new.parent = self
        new.branch_point = len(self.history)
        return new

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

    def apply(self, *args, name=None, description=None, **kwargs):
        """Apply to every leaf.  Returns ``self``.

        Iterates over leaf paths via :class:`_NodeProxy` so rank-changing
        Operations (``Multiply(outer=True)`` etc.) can replace a leaf
        with an intermediate ``Zstruct`` in place.

        ``name``/``description`` override the history label; otherwise
        fall back to the op's own ``name``/``description``.  Exactly
        one history entry is recorded per ``System.apply`` call,
        targeted at the root.
        """
        # Snapshot paths so a rank-changing op that adds children doesn't
        # make us re-visit the freshly inserted leaves.
        leaf_paths = [p for p, _ in list(self.leaves())]
        for path in leaf_paths:
            _NodeProxy(self, path)._apply_internal(*args, **kwargs)
        op = args[0] if args else None
        self._record_history((), op, name, description)
        return self

    # -- history ----------------------------------------------------------

    def _record_history(self, target_path, op, name_override, desc_override):
        """Append a single entry to :attr:`history` (+ legacy ``assumptions``).

        Defaults aim for a human-readable label when ``name`` /
        ``description`` aren't given: Operations and Relations carry
        their own ``name``/``description``; an Expression from
        ``solve_for`` exposes its ``_as_relation`` and is labelled
        accordingly; plain dicts / tuples fall back to a count.  Any
        of these can be overridden by passing explicit
        ``name=``/``description=`` to the ``apply`` call.
        """
        op_name, op_desc = self._default_history_labels(op)
        if name_override is not None:
            op_name = name_override
        if desc_override is not None:
            op_desc = desc_override
        if op_desc is None:
            op_desc = op_name
        self.history.append({
            "step": len(self.history),
            "target": "/".join(target_path) if target_path else "root",
            "name": op_name,
            "description": op_desc,
        })
        self.assumptions.append(op_name)

    @staticmethod
    def _default_history_labels(op):
        """Return ``(name, description)`` defaults for a condition value."""
        if op is None:
            return "apply", None
        # ``_as_relation`` takes priority over ``.name`` for
        # ``solve_for`` results so "substitute p" wins over the leaf's
        # stored name ("z", "p", etc. depending on tree position).
        rel = getattr(op, "_as_relation", None)
        if isinstance(rel, dict) and rel:
            keys = list(rel.keys())
            label = (f"substitute {keys[0]}" if len(keys) == 1
                     else f"substitute ({len(keys)} relations)")
            return label, label
        name = getattr(op, "name", None)
        desc = getattr(op, "description", None)
        if name:
            return name, desc or name
        if isinstance(op, dict):
            label = (f"substitute {len(op)} rule"
                     + ("s" if len(op) != 1 else ""))
            return label, label
        if isinstance(op, (list, tuple)):
            return f"substitute ({len(op)} items)", None
        return type(op).__name__, None

    def history_mermaid(self, *, direction="LR", verbose=False,
                        group_by=None):
        """Return a :class:`MermaidDiagram` of the derivation history.

        Default (compact): one node per ``apply`` labelled with
        ``name`` only, arranged left-to-right.  Full descriptions go
        into mermaid tooltips (hover in Jupyter / mermaid-live).

        Parameters
        ----------
        direction : ``"LR"`` | ``"TD"``
            Flowchart orientation.  ``LR`` scans horizontally and keeps
            long derivations readable.
        verbose : bool
            If ``True``, stack ``name`` / ``target`` / ``description``
            inside each node (the legacy 3-line layout).  Use for
            small derivations or for offline diagrams.
        group_by : ``None`` | ``"target"``
            If ``"target"``, wrap consecutive same-target steps in a
            ``subgraph`` block so it's obvious which tree node each
            phase acted on.  Pair with ``direction="TD"`` for best
            visual effect on moderately deep trees.

        Pair with :meth:`history_table` — the table carries the full
        per-step detail; the mermaid stays a minimal navigator.
        """
        lines = [f"flowchart {direction}"]
        lines.append(f'    S0(["<b>{_mm_escape(self.name)}</b>"])')
        tooltips = []
        prev = "S0"
        current_group = None
        group_nodes: list[tuple[str, str]] = []

        def _flush_group():
            if current_group is None or not group_nodes:
                return
            lines.append(f'    subgraph G_{current_group} ["{_mm_escape(current_group)}"]')
            for node_id, _ in group_nodes:
                lines.append(f"        {node_id}")
            lines.append("    end")

        for entry in self.history:
            step = entry["step"] + 1
            node_id = f"S{step}"
            name = entry["name"]
            tgt = entry["target"]
            desc = entry["description"]
            if verbose:
                label = (f"<b>{_mm_escape(name)}</b>"
                         f"<br/><i>{_mm_escape(tgt)}</i>"
                         f"<br/>{_mm_escape(desc)}")
                shape_open, shape_close = '["', '"]'
            else:
                label = _mm_escape(name)
                shape_open, shape_close = '["', '"]'
            lines.append(f'    {node_id}{shape_open}{label}{shape_close}')
            lines.append(f'    {prev} --> {node_id}')
            # Tooltip with target + description (mermaid's ``click .. callback``
            # isn't universally rendered, but the newer ``%% tooltip``
            # inline syntax isn't either — use the "click" with a bare
            # string which mermaid-live interprets as the tooltip text).
            tip = f"{tgt} — {desc}"
            tooltips.append(f'    click {node_id} "{_mm_escape(tip)}"')
            if group_by == "target":
                if tgt != current_group:
                    _flush_group()
                    current_group = tgt
                    group_nodes = []
                group_nodes.append((node_id, tgt))
            prev = node_id

        if group_by == "target":
            _flush_group()
        lines.extend(tooltips)
        return MermaidDiagram("\n".join(lines))

    def history_table(self):
        """Return a markdown table of the derivation history.

        One row per ``apply``, columns ``# / target / name / description``.
        Complements :meth:`history_mermaid` — use the table for detail,
        the flowchart for structure.
        """
        lines = [
            "| # | target | name | description |",
            "| ---: | --- | --- | --- |",
        ]
        for entry in self.history:
            step = entry["step"] + 1
            target = entry["target"]
            name = entry["name"].replace("|", "\\|")
            desc = entry["description"].replace("|", "\\|")
            lines.append(f"| {step} | `{target}` | **{name}** | {desc} |")
        body = "\n".join(lines)

        class _MDTable:
            def __init__(self, source):
                self.source = source
            def __str__(self):
                return self.source
            def _repr_markdown_(self):
                return self.source

        return _MDTable(body)

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
                 parameters=False, strip_args=True):
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


def _history_key(entry: dict) -> tuple:
    """Canonical tuple for comparing two history entries across systems."""
    return (entry["name"], entry["target"], entry["description"])


def _lcp_length(hist_a: list, hist_b: list) -> int:
    """Longest common prefix length between two histories (by ``_history_key``)."""
    L = 0
    for ea, eb in zip(hist_a, hist_b):
        if _history_key(ea) == _history_key(eb):
            L += 1
        else:
            break
    return L


def combined_history_mermaid(*systems, direction="LR", verbose=False,
                             auto_merge=True):
    """Render a branched-derivation flowchart over multiple :class:`System`\\ s.

    Given one or more systems — some created explicitly via
    :meth:`System.branch`, some built independently by their own
    constructors — this walks each system's ancestor chain, emits
    every ancestor's history once, and draws each branch's own steps
    diverging from the parent's ``branch_point`` node.

    When ``auto_merge=True`` (default), independent root systems that
    begin with byte-identical history prefixes (same ``(name, target,
    description)`` triple per step) get automatically merged: the
    shared prefix is rendered once, and the fork happens at the first
    differing step.  This lets separately-constructed models (``SWE()``,
    ``SME2()``) that share a common derivation helper show up as
    siblings without any plumbing at construction time.

    Tradeoff: auto-merge is label-sensitive.  If two systems pass
    different ``name=`` / ``description=`` kwargs for what's
    semantically the same step, they'll render as separate chains.
    Keep your common helpers' labels consistent, or pre-link
    explicitly via :func:`link_as_branches`.

    Parameters
    ----------
    *systems : :class:`System`
        The "leaves" you want visible.  Ancestors are pulled in
        automatically so you don't need to pass the trunk explicitly.
    direction : ``"LR"`` | ``"TD"``
        Flowchart orientation.
    verbose : bool
        Three-line node labels (name / target / description) vs.
        one-line (just ``name``, with the rest in a ``click`` tooltip).
    auto_merge : bool
        Collapse identical history prefixes across independent roots.

    Returns
    -------
    :class:`MermaidDiagram`

    Example::

        # Explicit branching:
        model = FullINS(state)
        model.apply(hydrostatic_scaling(state))      # shared step
        swe  = model.branch(name="SWE")
        sme2 = model.branch(name="SME level=2")
        swe.apply(...)
        sme2.apply(...)
        combined_history_mermaid(swe, sme2)

        # Auto-merge of separately-built systems with the same prefix:
        swe  = SWE(state);  swe.build()
        sme2 = SME2(state); sme2.build()
        combined_history_mermaid(swe, sme2)
    """
    # 1. Collect every system we need to render (passed-in + ancestors),
    #    in a deterministic order: roots first (so downstream branches
    #    can find their parent's nodes already registered).
    seen: set[int] = set()
    ordered: list[System] = []

    def _collect_chain(sys: System):
        chain: list[System] = []
        cur = sys
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        for s in chain:
            if id(s) not in seen:
                seen.add(id(s))
                ordered.append(s)

    for sys in systems:
        _collect_chain(sys)

    # 2. Emit nodes.  For each history entry we mint a unique node id.
    #    Root systems (``parent is None``) get an initial "initial"
    #    node; branch systems don't — their first edge comes from the
    #    parent's branch_point node directly.  With ``auto_merge``, a
    #    root system whose history shares a prefix with a previously-
    #    rendered root is piggybacked onto that root's chain: the
    #    shared prefix points at the earlier system's node ids (via
    #    alias entries in ``node_ids``) and only the divergent tail
    #    mints new nodes.
    lines = [f"flowchart {direction}"]
    node_ids: dict[tuple[int, int], str] = {}  # (id(sys), step) -> "Nk"
    counter = [0]

    def _new_id() -> str:
        counter[0] += 1
        return f"N{counter[0]}"

    def _label(entry: dict) -> str:
        if verbose:
            return (f"<b>{_mm_escape(entry['name'])}</b>"
                    f"<br/><i>{_mm_escape(entry['target'])}</i>"
                    f"<br/>{_mm_escape(entry['description'])}")
        return _mm_escape(entry["name"])

    tooltips: list[str] = []
    rendered_roots: list[System] = []

    for sys in ordered:
        # Decide where this system starts drawing from, and which step
        # index is the first one that needs its own node.
        start_idx = 0  # first index of sys.history that needs rendering
        prev_id: str

        if sys.parent is not None:
            # Explicit branch: start from the parent's branch_point - 1
            # node.  The parent's chain was rendered earlier (we sort
            # ancestors first).
            bp = sys.branch_point or 0
            parent_key = (id(sys.parent), bp - 1) if bp > 0 else (id(sys.parent), -1)
            prev_id = node_ids.get(parent_key)
            if prev_id is None:
                prev_id = _new_id()
                node_ids[parent_key] = prev_id
                lines.append(f'    {prev_id}(["<b>{_mm_escape(sys.parent.name)}</b>"])')
            if verbose:
                fork_id = _new_id()
                lines.append(f'    {fork_id}{{"<b>branch:</b> {_mm_escape(sys.name)}"}}')
                lines.append(f'    {prev_id} --> {fork_id}')
                prev_id = fork_id
        else:
            # Independent root.  Try to piggyback on an earlier root
            # via longest-common-prefix, if ``auto_merge`` is set.
            match = None
            match_len = 0
            if auto_merge:
                for root in rendered_roots:
                    L = _lcp_length(sys.history, root.history)
                    if L > match_len:
                        match_len = L
                        match = root

            if match is not None and match_len > 0:
                # Alias the shared prefix entries to the matched root's
                # existing nodes so downstream lookups resolve.
                node_ids[(id(sys), -1)] = node_ids[(id(match), -1)]
                for i in range(match_len):
                    node_ids[(id(sys), i)] = node_ids[(id(match), i)]
                prev_id = node_ids[(id(match), match_len - 1)]
                start_idx = match_len
            else:
                initial_id = _new_id()
                node_ids[(id(sys), -1)] = initial_id
                lines.append(f'    {initial_id}(["<b>{_mm_escape(sys.name)}</b>"])')
                prev_id = initial_id
            rendered_roots.append(sys)

        for i in range(start_idx, len(sys.history)):
            entry = sys.history[i]
            node_id = _new_id()
            node_ids[(id(sys), i)] = node_id
            lines.append(f'    {node_id}["{_label(entry)}"]')
            lines.append(f'    {prev_id} --> {node_id}')
            tip = f"[{sys.name}] {entry['target']} — {entry['description']}"
            tooltips.append(f'    click {node_id} "{_mm_escape(tip)}"')
            prev_id = node_id

    lines.extend(tooltips)
    return MermaidDiagram("\n".join(lines))


def link_as_branches(*systems, reference=None):
    """Promote auto-detected sibling prefixes into real parent / branch_point metadata.

    After independently-constructed systems have been compared and
    auto-merged for rendering, you may want the link to be *durable* —
    so subsequent ``apply`` calls on one of them, or another
    ``combined_history_mermaid`` call that mixes them with new
    branches, still render correctly.  This helper performs that
    promotion:

    * Picks one system as the ``reference`` (default: the first
      passed-in system) — it keeps its history unchanged.
    * For every other system, finds the longest common history prefix
      with the reference.  If the prefix length ``L > 0``, sets
      ``sys.parent = reference`` and ``sys.branch_point = L``, then
      trims ``sys.history`` to drop the first ``L`` entries (they now
      live on the reference).
    * Systems with no matching prefix are left untouched.

    After this, ``combined_history_mermaid(sys1, sys2, ...)`` will
    render the fork via explicit ``parent`` edges (no auto-merge
    needed), and later ``apply`` calls on the non-reference systems
    record cleanly on top of the trimmed history.

    Note: this mutates the passed systems.  Pass only systems you own.

    Example::

        swe  = SWE(state);  swe.build()
        sme2 = SME2(state); sme2.build()
        link_as_branches(swe, sme2)      # promote the implicit link
        swe.apply(some_followup_step)    # now recorded as a sibling of sme2
        combined_history_mermaid(swe, sme2)
    """
    if not systems:
        return
    ref = reference if reference is not None else systems[0]
    for sys in systems:
        if sys is ref:
            continue
        if sys.parent is not None:
            # Already linked — skip.
            continue
        L = _lcp_length(sys.history, ref.history)
        if L == 0:
            continue
        sys.parent = ref
        sys.branch_point = L
        sys.history = sys.history[L:]
        # Re-number remaining steps' ``step`` fields so indices stay
        # consistent with the branch's new history length.
        for i, entry in enumerate(sys.history):
            entry["step"] = i
