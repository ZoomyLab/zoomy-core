"""The clean-redesign :class:`Model` — the derivation-graph spine.

A ``Model`` holds:

* an ordered tuple of independent ``coords`` (from
  :mod:`zoomy_core.coords`);
* ``parameters`` — a :class:`~zoomy_core.misc.misc.Zstruct` of positive
  Symbols (``model.parameters.g``);
* an **ordered** dict of equations ``_equations`` (scalar
  :class:`~zoomy_core.model.equation.Equation` rows; vector rows register one
  :class:`VectorEquation` proxy plus its flattened component rows);
* a derivation ``history`` (the op graph);
* a per-model modal-index registry placeholder ``_modal_registry``.

Unknowns follow the declared-and-present / derived contract:

* ``model.Q`` is **declared** — set from a list of applied fields
  (``[h, u, w, p]``) auto-named from each field head; a symbol enters ``Q``
  only when an op declares it an unknown, and is **dropped automatically the
  moment it appears in zero equations** (``_refresh_unknowns``).
* ``model.Qaux`` is **derived** — every field Function present in the
  equations that is *not* in ``Q``, *not* a coordinate, and *not* a parameter.

``Model`` reuses the existing operation spine: ``model.apply(op)`` broadcasts
``op`` across every equation via :meth:`Equation.apply`, except for
``whole_model_op`` / ``system_level`` ops which are routed once to the op's
``apply_to_model`` / ``__call__``.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core import coords as _coords
from zoomy_core.misc.misc import Zstruct
from zoomy_core.misc.description import Description
from zoomy_core.model.equation import Equation


__all__ = ["Model", "VectorEquation", "MomentFamily", "resolve_modes"]


# Component index → attribute name, by vector length.
#   length 2 → (x, z)   [vertical-plane models: horizontal + vertical]
#   length 3 → (x, y, z)
_COMPONENT_NAMES = {
    1: ("x",),
    2: ("x", "z"),
    3: ("x", "y", "z"),
}


class VectorEquation:
    """A vector equation — an ordered list of component :class:`Equation`s
    with named-axis access (``.x`` / ``.y`` / ``.z``) and integer/term
    indexing.

    Component rows are flattened into the model's ``_equations`` dict under
    ``"<name>_<axis>"`` so unknown-collection and operator extraction see one
    scalar row per component; the proxy is the user-facing handle
    (``model.momentum.x``).
    """

    def __init__(self, name, shape, components):
        self.name = name
        self.shape = tuple(shape)
        n = len(components)
        names = _COMPONENT_NAMES.get(n)
        if names is None:
            raise ValueError(
                f"VectorEquation {name!r}: no axis names for length {n} "
                f"(supported: {sorted(_COMPONENT_NAMES)}).")
        self._axis_names = names
        self._components = list(components)  # list[Equation]

    # ── named-axis access ────────────────────────────────────────────
    def __getattr__(self, item):
        # Only consulted when normal attribute lookup fails.
        names = self.__dict__.get("_axis_names", ())
        if item in names:
            return self._components[names.index(item)]
        raise AttributeError(item)

    # ── integer / term indexing ──────────────────────────────────────
    def __getitem__(self, i):
        if isinstance(i, tuple) and len(i) == 2:
            comp, term = i
            if isinstance(comp, str):
                comp = self._axis_names.index(comp)
            return self._components[comp][term]
        return self._components[i]

    def __iter__(self):
        return iter(self._components)

    def __len__(self):
        return len(self._components)

    @property
    def components(self):
        return list(self._components)

    def replace_axis(self, axis, replacement):
        """Replace the component held at ``axis`` (``"x"`` / index) — used by
        ``resolve_modes`` to swap a scalar component for a
        :class:`MomentFamily` after a moment-axis SHAPE BUMP, so
        ``model.momentum.x[l]`` resolves to the bumped family."""
        i = (self._axis_names.index(axis) if isinstance(axis, str) else axis)
        self._components[i] = replacement
        return replacement

    def apply(self, *args, **kwargs):
        """Broadcast across every component equation."""
        for comp in self._components:
            comp.apply(*args, **kwargs)
        return self

    def describe(self, strip_args=False):
        parts = [f"**{self.name}** (vector, {len(self)} components)"]
        for comp in self._components:
            parts.append(comp.describe(strip_args=strip_args)
                         if isinstance(comp, MomentFamily)
                         else comp.describe_line(strip_args=strip_args,
                                                 bullet=True))
        return Description("\n".join(parts))

    def __repr__(self):
        return (f"VectorEquation({self.name!r}, shape={self.shape}, "
                f"{len(self)} components)")


class MomentFamily:
    """An indexed family of moment rows — the result of a ``resolve_modes``
    SHAPE BUMP that appends a Galerkin moment axis to an equation.

    A scalar ``mass`` row ``(1,)`` bumps to ``(N+1,)``; the proxy is the
    user-facing handle ``model.mass`` and ``model.mass[l]`` is the ``l``-th
    moment row.  When the bumped equation was a *component* of a
    :class:`VectorEquation` (``model.momentum.x``), the component proxy holds
    one :class:`MomentFamily` per axis and ``model.momentum.x[l]`` is the
    ``(x, l)`` slice of the ``(2, N+1)`` tensor.

    The per-moment scalar rows are flattened into the model's ``_equations``
    dict under ``"<base>_<mode>"`` (``mass_0``, ``momentum_x_1``, …) so
    unknown-collection and operator extraction see one scalar row per moment;
    the proxy carries indexed / attribute access only.
    """

    def __init__(self, name, modes, components, index=None):
        self.name = name
        self.modes = list(modes)
        self.index = index
        self._components = list(components)   # list[Equation], in mode order

    # ── integer-mode indexing ────────────────────────────────────────
    def __getitem__(self, i):
        return self._components[i]

    def __iter__(self):
        return iter(self._components)

    def __len__(self):
        return len(self._components)

    @property
    def components(self):
        return list(self._components)

    # ── ``.eq0 / .eq1 / …`` and ``.eq_l`` aliases ────────────────────
    def __getattr__(self, item):
        # Consulted only on normal-lookup failure.  ``eqK`` → mode-K row.
        comps = self.__dict__.get("_components", [])
        if item.startswith("eq"):
            tail = item[2:]
            if tail.isdigit():
                return comps[int(tail)]
            # ``eq_l`` — the symbolic-index alias for the whole family.
            if tail in ("_l", "_" + str(self.__dict__.get("index", ""))):
                return self
        raise AttributeError(item)

    def apply(self, *args, **kwargs):
        """Broadcast across every moment row."""
        for comp in self._components:
            comp.apply(*args, **kwargs)
        return self

    def describe(self, strip_args=False):
        parts = [f"**{self.name}** (moment family, {len(self)} modes)"]
        for comp in self._components:
            parts.append(comp.describe_line(strip_args=strip_args, bullet=True))
        return Description("\n".join(parts))

    def __repr__(self):
        return (f"MomentFamily({self.name!r}, {len(self)} modes, "
                f"index={self.index})")


class Model:
    """Computational-graph spine for a symbolic derivation."""

    def __init__(self, coords=(_coords.t, _coords.x, _coords.z),
                 parameters=None, name="model"):
        self.name = name
        self._coords = tuple(coords)

        # Parameters → Zstruct of positive Symbols; remember the numeric
        # defaults the caller passed for description / later substitution.
        parameters = parameters or {}
        params = Zstruct()
        self.parameter_values = Zstruct()
        for key in parameters:
            setattr(params, key, sp.Symbol(key, positive=True))
            setattr(self.parameter_values, key, parameters[key])
        self.parameters = params

        # Ordered equation registry.  Vector rows store BOTH the proxy (under
        # the vector name) and the flattened component rows (under
        # ``<name>_<axis>``); ``_equations`` only holds scalar Equation rows.
        self._equations: dict[str, Equation] = {}
        self._vector_equations: dict[str, VectorEquation] = {}
        self._equation_shapes: dict[str, tuple] = {}
        # Moment families produced by ``resolve_modes`` (a Galerkin moment-axis
        # SHAPE BUMP).  Keyed by the proxy name a user reaches for: a scalar
        # bump registers ``{"mass": MomentFamily}``; a vector-component bump
        # registers ``{"momentum_x": MomentFamily}`` and the component proxy on
        # the VectorEquation is replaced in place so ``model.momentum.x[l]``
        # resolves to the family.
        self._moment_families: dict[str, MomentFamily] = {}

        # Derivation history + per-model modal-index registry.  The registry
        # hands out a distinct summation index per coefficient family and is
        # read back by ``separation_of_variables`` / ``ResolveBasis``.
        self.history: list[dict] = []
        from zoomy_core.derivation.modal import ModalIndexRegistry
        self._modal_registry = ModalIndexRegistry()

        # Coordinate-transformation state.  ``PDETransformation`` records the
        # σ-map here so later steps (``kinematic_bc``, depth projection) can
        # read the decorated-head map, the source vertical coord, the mapped
        # reference coord, and the explicit ``z = b + h·ζ`` relation.
        #   _field_decoration: {original Function head → decorated Function head}
        #   _sigma_from:       the physical vertical coord that was mapped (z)
        #   _vertical:         the mapped reference coord (ζ) — None pre-map
        #   coord_relations:   {z: b + h·ζ}
        self._field_decoration: dict = {}
        self._sigma_from = None
        self._vertical = None
        self.coord_relations: dict = {}

        # Declared unknowns (Q).  Maps field-head-name → applied field Function.
        self._Q: dict[str, sp.Expr] = {}
        # Heads that have appeared in some equation at least once.  A declared
        # unknown is auto-dropped from Q only once it has *appeared* and then
        # *vanished* — declaring an unknown before its equation exists (the
        # SME cell sets ``Q = [h, u, w, p]`` before the balances are added) is
        # legitimate and must not be eaten by the refresh.
        self._ever_seen_heads: set = set()

    # ── parameters helpers ───────────────────────────────────────────
    @property
    def coords(self):
        return self._coords

    @property
    def vertical(self):
        """The vertical coordinate — the mapped reference coord ``ζ`` once a
        :class:`PDETransformation` has run, else the original vertical
        (``z``: the last coord that is neither ``t`` nor a horizontal)."""
        if self._vertical is not None:
            return self._vertical
        return self._coords[-1] if self._coords else None

    @property
    def horizontal(self):
        """The horizontal spatial coords — every coord except the time coord
        (``coords[0]``) and the :attr:`vertical`."""
        vert = self.vertical
        return tuple(c for c in self._coords[1:] if c != vert)

    # ── history ──────────────────────────────────────────────────────
    def _history(self, op_name, target, level="major", description=None):
        self.history.append({
            "op": op_name,
            "target": target,
            "level": level,
            "description": description,
        })

    # ── equation registration ────────────────────────────────────────
    def add_equation(self, name, shape_or_expr=None, expr=None):
        """Register a scalar or vector equation.

        Scalar::

            model.add_equation("mass", expr)

        Vector::

            model.add_equation("momentum", (2,), [expr_x, expr_z])

        The shape argument is optional positionally — ``add_equation(name,
        expr)`` registers a scalar row.
        """
        # Disambiguate the two-arg scalar call from the three-arg vector call.
        if expr is None and shape_or_expr is not None and not (
                isinstance(shape_or_expr, tuple)
                and all(isinstance(k, int) for k in shape_or_expr)):
            # scalar: add_equation(name, expr)
            scalar_expr = shape_or_expr
            eq = Equation(scalar_expr, name=name, model=self)
            self._equations[name] = eq
            self._equation_shapes[name] = (1,)
            self._refresh_unknowns()
            return eq

        if expr is None:
            # scalar with explicit shape omitted / zero row.
            eq = Equation(sp.S.Zero if shape_or_expr is None else shape_or_expr,
                          name=name, model=self)
            self._equations[name] = eq
            self._equation_shapes[name] = (1,)
            self._refresh_unknowns()
            return eq

        # vector: add_equation(name, shape, [exprs])
        shape = shape_or_expr
        components = []
        n = len(expr)
        names = _COMPONENT_NAMES.get(n)
        if names is None:
            raise ValueError(
                f"add_equation({name!r}): no axis names for {n} components.")
        for axis, comp_expr in zip(names, expr):
            comp_name = f"{name}_{axis}"
            comp_eq = Equation(comp_expr, name=comp_name, model=self)
            self._equations[comp_name] = comp_eq
            self._equation_shapes[comp_name] = (1,)
            components.append(comp_eq)
        vec = VectorEquation(name, shape, components)
        self._vector_equations[name] = vec
        self._equation_shapes[name] = tuple(shape)
        self._refresh_unknowns()
        return vec

    def _remove_equation(self, name):
        """Drop a scalar equation (or a vector proxy + its components).

        Removing a single *component* of a vector (``momentum_z``) also drops
        the dead slot from the parent :class:`VectorEquation` so the proxy
        carries no stale reference — e.g. after the hydrostatic reduction
        eliminates the z-momentum, ``model.momentum`` exposes only the
        surviving ``x`` component.
        """
        if name in self._vector_equations:
            vec = self._vector_equations.pop(name)
            for comp in vec.components:
                comp_name = getattr(comp, "name", None)
                self._equations.pop(comp_name, None)
                self._equation_shapes.pop(comp_name, None)
            self._equation_shapes.pop(name, None)
            self._refresh_unknowns()
            return
        # A standalone scalar OR a vector component — drop the scalar row and,
        # if it was a component, prune it from the parent proxy.
        self._equations.pop(name, None)
        self._equation_shapes.pop(name, None)
        for vname, vec in list(self._vector_equations.items()):
            for i, comp in enumerate(vec._components):
                if getattr(comp, "name", None) == name:
                    del vec._components[i]
                    vec._axis_names = tuple(
                        ax for j, ax in enumerate(vec._axis_names) if j != i)
                    if vec._components:
                        self._equation_shapes[vname] = (len(vec._components),)
                        vec.shape = (len(vec._components),)
                    else:
                        self._vector_equations.pop(vname, None)
                        self._equation_shapes.pop(vname, None)
                    break
        self._refresh_unknowns()

    def _collapse_moment_family(self, base, keep):
        """Shrink the moment family ``base`` to the modes in ``keep``.

        ``[l]`` stays MOMENT-UNIFORM: a family never decays back to a bare
        scalar :class:`Equation` once it has been bumped.  When a single mode
        survives a closure (e.g. the mean continuity ``mass[0]`` after the KBC
        consumes the higher mass moments), the family keeps that one moment row
        and its flattened scalar key is RE-KEYED from ``<base>_<mode>`` to the
        bare ``<base>`` so the structural extractor / acceptance tests read
        ``model._equations[base]`` while ``model.<base>`` stays the family and
        ``model.<base>[0]`` is the surviving moment row.

        ``keep`` is the list of surviving mode indices (the caller is expected
        to have already dropped the non-kept flattened rows).  Only the
        single-survivor re-key is supported for now — that is the one closure
        shape the SME pipeline produces.
        """
        fam = self._moment_families[base]
        keep = list(keep)
        survivors = [fam.components[fam.modes.index(k)] for k in keep]
        fam._components = survivors
        fam.modes = keep
        if len(keep) == 1:
            (k,) = keep
            old_name = f"{base}_{k}"
            row = self._equations.pop(old_name)
            self._equation_shapes.pop(old_name, None)
            row.name = base
            self._equations[base] = row
            self._equation_shapes[base] = (len(keep),)
        else:
            self._equation_shapes[base] = (len(keep),)
        self._refresh_unknowns()
        return fam

    # ── attribute access to equations ────────────────────────────────
    def __getattr__(self, item):
        # Consulted only on normal-lookup failure.  Resolve vector proxies
        # first, then moment families, then scalar rows.
        vec = self.__dict__.get("_vector_equations", {})
        if item in vec:
            return vec[item]
        fam = self.__dict__.get("_moment_families", {})
        if item in fam:
            return fam[item]
        eqs = self.__dict__.get("_equations", {})
        if item in eqs:
            return eqs[item]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {item!r}")

    # ── apply (broadcast / route model-level ops) ────────────────────
    def apply(self, op, **kwargs):
        """Apply ``op``.

        * ``system_level`` ops → ``op(self)`` once (whole-DAE).
        * ``whole_model_op`` ops → ``op.apply_to_model(self)`` once.
        * everything else → broadcast across every scalar equation via
          :meth:`Equation.apply`.
        """
        if getattr(op, "system_level", False):
            result = op(self)
            self._history(getattr(op, "name", None) or type(op).__name__,
                          "*system*")
            return result if result is not None else self
        if getattr(op, "whole_model_op", False):
            op.apply_to_model(self)
            self._history(getattr(op, "name", None) or type(op).__name__,
                          "*model*")
            return self
        for eq in list(self._equations.values()):
            eq.apply(op, _no_history=True, **kwargs)
        self._history(getattr(op, "name", None) or type(op).__name__, "*all*")
        self._refresh_unknowns()
        return self

    # ── unknowns: Q (declared) / Qaux (derived) ──────────────────────
    @property
    def Q(self):
        out = Zstruct()
        for nm, field in self._Q.items():
            setattr(out, nm, field)
        out._symbolic_name = "Q"
        return out

    @Q.setter
    def Q(self, fields):
        if isinstance(fields, Zstruct):
            fields = fields.values()
        new_q = {}
        for f in fields:
            new_q[self._field_name(f)] = f
        self._Q = new_q
        self._refresh_unknowns()

    @property
    def Qaux(self):
        """Derived: every field Function present in the equations that is not
        in ``Q``, not a coordinate, and not a parameter."""
        q_heads = {self._head(f) for f in self._Q.values()}
        out = Zstruct()
        seen = set()
        for f in sorted(self._collect_fields(), key=str):
            head = self._head(f)
            if head in q_heads:
                continue
            nm = self._field_name(f)
            if nm in seen:
                continue
            seen.add(nm)
            setattr(out, nm, f)
        out._symbolic_name = "Qaux"
        return out

    def _refresh_unknowns(self):
        """Drop from ``Q`` any declared unknown that has appeared in an
        equation and then vanished from every equation.

        A declared-but-never-yet-present unknown (e.g. ``h`` declared before
        its balance is added) is kept; it is dropped only once it has been
        seen and is gone again.  (Q gains a symbol only via
        :meth:`redeclare_unknown` / the setter.)
        """
        present_heads = {self._head(f) for f in self._collect_fields()}
        self._ever_seen_heads |= present_heads
        self._Q = {
            nm: f for nm, f in self._Q.items()
            if self._head(f) in present_heads
            or self._head(f) not in self._ever_seen_heads
        }

    def redeclare_unknown(self, old_field, new_fields):
        """Transfer unknown-status from ``old_field`` to ``new_fields``.

        ``old`` leaves ``Q``; each of ``new`` enters ``Q``.  ``old_field`` /
        ``new_fields`` may be applied Functions or bare Function heads (the
        family head is what matters for the Q registry).  Not user-facing —
        ops (``ChangeOfVariables``) call this.
        """
        old_head = self._head(old_field)
        # Remove every Q entry whose head matches old.
        self._Q = {nm: f for nm, f in self._Q.items()
                   if self._head(f) != old_head}
        if not isinstance(new_fields, (list, tuple)):
            new_fields = [new_fields]
        for nf in new_fields:
            self._Q[self._field_name(nf)] = nf

    # ── field introspection ──────────────────────────────────────────
    @staticmethod
    def _is_field_application(atom):
        """True iff ``atom`` is an application of an *undefined-like* Function
        head — i.e. a user/derivation field, not a sympy builtin.

        This covers BOTH plain ``sp.Function("u")(...)`` applications
        (``AppliedUndef``) AND the dynamically-minted decorated heads a
        :class:`~zoomy_core.derivation.transformations.PDETransformation`
        creates (``type("\\tilde{u}", (sp.Function,), …)`` — a direct
        ``Function`` subclass that is NOT ``AppliedUndef``).  A builtin like
        ``sin`` has ``sympy`` machinery (``eval`` / ``_imp_``) and a base
        chain that does not end at the bare ``Function`` we mint from, so it
        is excluded.
        """
        from sympy.core.function import AppliedUndef

        if isinstance(atom, AppliedUndef):
            return True
        func = getattr(atom, "func", None)
        if func is None or not isinstance(func, sp.FunctionClass):
            return False
        # Basis machinery (opaque ``φ(k, ζ)`` / weight ``c(ζ)``) and named
        # Galerkin brackets (``Gram`` / ``Weight``) are derivation machinery,
        # never unknowns — they carry ``_is_basis_head`` / ``_is_bracket``.
        if getattr(func, "_is_basis_head", False):
            return False
        if getattr(func, "_is_bracket", False):
            return False
        # Undefined-like: a direct subclass of bare ``Function`` with no
        # evaluation rule (the minted decorated heads) — distinguished from
        # sympy builtins which carry their own ``eval``.
        return (sp.Function in getattr(func, "__bases__", ())
                and func.__dict__.get("eval") is None)

    def _collect_fields(self):
        """Every applied field Function across all equations — every
        undefined-like Function application whose head is not a parameter."""
        param_names = set(self.parameters.keys())
        fields = set()
        for eq in self._equations.values():
            for atom in eq.expr.atoms(sp.Function):
                if not self._is_field_application(atom):
                    continue
                if atom.func.__name__ in param_names:
                    continue
                fields.add(atom)
        return fields

    # ── modal-index / basis accessors (per-field Resolve) ─────────────
    def modal_index(self, target):
        """The modal-mode index assigned to a field or coefficient family.

        ``target`` may be the coefficient (head class / applied / name) OR the
        field (``u(t, x, z)`` / decorated head); both ends were registered by
        :func:`~zoomy_core.derivation.modal.separation_of_variables`, so either
        resolves to the same distinct index (``a → i``, ``aw → j``).  Used as
        ``ResolveBasis(model.modal_index(u), …)``.
        """
        idx = self._modal_registry.index_for(target)
        if idx is None:
            from zoomy_core.derivation.modal import _coeff_key
            idx = self._modal_registry.fresh(_coeff_key(target))
        return idx

    def modal_basis(self, target):
        """The opaque basis the field's separation-of-variables expansion was
        registered with (or ``None``)."""
        return self._modal_registry.basis_for(target)

    def _coord_symbols(self):
        return set(self._coords)

    @staticmethod
    def _head(field):
        """The Function head (class) of a field — applied or already a head."""
        if isinstance(field, sp.FunctionClass):
            return field
        func = getattr(field, "func", None)
        return func if func is not None else field

    @classmethod
    def _field_name(cls, field):
        """Auto-name from the field head: ``h(t,x).func.__name__ == "h"``."""
        head = cls._head(field)
        return getattr(head, "__name__", None) or str(head)

    # ── describe ─────────────────────────────────────────────────────
    def describe(self, strip_args=False, show_history=False):
        """Human-readable derivation state.  Header shows the model name +
        #equations + #ops; each equation delegates to its own
        ``describe_line``."""
        n_eq = len(self._equations)
        n_vec = len(self._vector_equations)
        n_ops = len(self.history)
        parts = [
            f"**{self.name}** — {n_eq} equation{'s' if n_eq != 1 else ''} "
            f"({n_vec} vector), {n_ops} op{'s' if n_ops != 1 else ''} "
            f"— coords {tuple(str(c) for c in self._coords)}"
        ]
        if self.parameters.length():
            kv = ", ".join(
                f"${sp.latex(self.parameters[k])} = "
                f"{getattr(self.parameter_values, k, '?')}$"
                for k in self.parameters.keys())
            parts.append(f"**Parameters:** {kv}")
        if self._Q:
            parts.append("**Q (declared):** "
                         + ", ".join(f"`{k}`" for k in self._Q))
        # Equations: render vector proxies as a group, then any remaining
        # standalone scalar rows.
        rendered = set()
        parts.append("**Equations:**")
        for vname, vec in self._vector_equations.items():
            parts.append(f"- **{vname}** (vector):")
            for comp in vec.components:
                if isinstance(comp, MomentFamily):
                    for row in comp.components:
                        parts.append("  " + row.describe_line(
                            strip_args=strip_args, bullet=True))
                        rendered.add(row.name)
                else:
                    parts.append("  " + comp.describe_line(
                        strip_args=strip_args, bullet=True))
                    rendered.add(comp.name)
        for base, fam in self._moment_families.items():
            if base in self._vector_equations or any(
                    fam is c for v in self._vector_equations.values()
                    for c in v.components):
                continue
            parts.append(f"- **{base}** (moment family):")
            for row in fam.components:
                parts.append("  " + row.describe_line(
                    strip_args=strip_args, bullet=True))
                rendered.add(row.name)
        for name, eq in self._equations.items():
            if name in rendered:
                continue
            parts.append(eq.describe_line(strip_args=strip_args, bullet=True))
        if show_history and self.history:
            parts.append("**History:**")
            for h in self.history:
                parts.append(f"- `{h['op']}` → {h['target']}")
        return Description("\n".join(parts))

    def __repr__(self):
        return (f"Model({self.name!r}, {len(self._equations)} equations, "
                f"{len(self.history)} ops)")


# ── resolve_modes: the Galerkin moment-axis SHAPE BUMP ─────────────────────


def resolve_modes(equation, *, index, modes, test_weight=None,
                  basis_cls=None, level=None, var=None, resolver=None):
    """Bump an equation's tensor shape by appending a Galerkin MOMENT AXIS.

    ``equation`` is a row carrying an ABSTRACT test index ``index`` (``l``) —
    the state after a Galerkin ``Project(c·φ(l, ζ))``-style projection.  Via the
    existing outer-product ``Substitution(over=modes, target=…)`` it is expanded
    into one row per ``l ∈ modes``, GROUPED UNDER THE PARENT as an indexed
    :class:`MomentFamily`.  Shape algebra:

    * scalar ``mass`` ``(1,)`` → ``(N+1,)`` — ``model.mass[l]`` /
      ``model.mass.eq0`` …;
    * vector component ``momentum.x`` of a ``(2,)`` vector →
      ``momentum`` shape ``(2, N+1)`` — ``model.momentum.x[l]`` ≡
      ``model.momentum["x", l]`` (the ``(x, l)`` slice).

    When ``test_weight`` / ``basis_cls`` / ``level`` are given, each bumped row
    is additionally closed by the concrete-level Galerkin
    :class:`~zoomy_core.derivation.closure.Resolve` (``φ(l, ζ) → φ(k, ζ)`` at
    the bound mode), so no opaque basis brackets remain.

    Parameters
    ----------
    equation : Equation
        The row to bump — ``model.mass`` (scalar) or ``model.momentum.x``
        (vector component).  Carries the abstract ``index``.
    index : sympy.Symbol
        The abstract Galerkin test index ``l`` to specialise per moment.
    modes : iterable[int]
        The concrete moment indices (``range(N+1)``).
    test_weight, basis_cls, level, var : optional
        If all of ``test_weight`` / ``basis_cls`` / ``level`` are given, each
        moment row is closed by ``Resolve(test_weight, basis_cls, level,
        var=var)`` after the ``l``-specialisation.
    resolver : callable, optional
        Custom per-mode closer ``resolver(row, test_weight_k, k)`` invoked
        INSTEAD of the default ``Resolve`` (e.g. to route a second-order
        viscous block conservatively while the rest resolves in place).  Only
        consulted when the ``test_weight`` / ``basis_cls`` / ``level`` closing
        path is active.
    """
    from zoomy_core.derivation.operations import Substitution

    model = equation._model
    base = equation.name                      # "mass" or "momentum_x"
    modes = list(modes)

    # Locate a parent VectorEquation if this row is a component.
    parent_vec = None
    axis = None
    for vname, vec in model._vector_equations.items():
        for ax, comp in zip(vec._axis_names, vec._components):
            if comp is equation or getattr(comp, "name", None) == base:
                parent_vec, axis = vec, ax
                break
        if parent_vec is not None:
            break

    src_expr = equation.expr

    # Build one moment row per mode via the outer-product specialisation, then
    # (optionally) close each with the concrete Resolve.
    do_close = (test_weight is not None and basis_cls is not None
                and level is not None)
    if do_close:
        from zoomy_core.derivation.closure import Resolve

    comps = []
    for k in modes:
        row_name = f"{base}_{k}"
        model.add_equation(row_name, src_expr)
        row = model._equations[row_name]
        row.apply(Substitution({index: k},
                               name=f"resolve_modes[{index}={k}]"),
                  _no_history=True)
        if do_close:
            tw = test_weight.xreplace({index: k})
            if resolver is not None:
                resolver(row, tw, k)
            else:
                row.apply(Resolve(tw, basis_cls, level, var=var),
                          _no_history=True)
        comps.append(row)

    family = MomentFamily(base, modes, comps, index=index)
    model._moment_families[base] = family

    # Shape bump: scalar (1,) → (N+1,); vector component (n,) → (n, N+1).
    if parent_vec is not None:
        # The component is being REPLACED by the family, not removed — drop
        # only its flattened scalar row and install the family in the proxy
        # slot (do NOT route through ``_remove_equation``, which would prune
        # the axis we are about to fill).
        model._equations.pop(base, None)
        model._equation_shapes.pop(base, None)
        parent_vec.replace_axis(axis, family)
        old_shape = model._equation_shapes.get(parent_vec.name,
                                               (len(parent_vec),))
        model._equation_shapes[parent_vec.name] = (*old_shape, len(modes))
        parent_vec.shape = (*parent_vec.shape, len(modes))
    else:
        # Drop the abstract-index template scalar row.
        model._remove_equation(base)
        model._equation_shapes[base] = (len(modes),)

    model._history("resolve_modes", base)
    model._refresh_unknowns()
    return family
