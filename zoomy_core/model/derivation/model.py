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
from zoomy_core.model.operations import Operation


__all__ = ["Model", "VectorEquation", "MomentFamily", "resolve_modes",
           "ResolveModes"]


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
            parts.append(str(comp.describe(strip_args=strip_args))
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


class FieldHandle:
    """Ergonomic handle for a model field — the (decorated) applied form plus
    convenience access, so a derivation never reaches into private model state.

    ``m.functions.tau_xz`` returns one of these wrapping the CURRENT applied
    form ``\\tilde{tau_xz}(t, x, ζ)`` (post-σ-map):

    * ``.head``     — the Function class (``\\tilde{tau_xz}``), for a head-level
                      ``replace`` / closure definition;
    * ``.expr``     — the canonical applied field ``\\tilde{tau_xz}(t, x, ζ)``;
    * ``.at(value)``— evaluate at a σ-boundary: ``m.functions.u.at(0)`` →
                      ``\\tilde u(t, x, 0)`` (substitutes the LAST / vertical arg);
    * calling it    — re-applies the head: ``m.functions.u(t, x, ζ)``.

    It is sympifiable (``_sympy_`` returns ``.expr``) so it drops into sympy
    arithmetic, but ``m.Q`` keeps returning raw fields — handles live on
    ``m.functions`` only, so existing ``solve_for`` / ``apply`` paths are
    unchanged."""

    def __init__(self, applied):
        self._applied = applied

    @property
    def head(self):
        return self._applied.func

    @property
    def expr(self):
        return self._applied

    def at(self, value):
        """Evaluate at a σ-coordinate value (the vertical / last argument):
        ``ũ.at(0) → ũ(t,x,0)``, ``τ̃.at(1) → τ̃(t,x,1)``."""
        return self._applied.subs(self._applied.args[-1], value)

    def __call__(self, *args):
        return self._applied.func(*args)

    def _sympy_(self):
        return self._applied

    def __repr__(self):
        return f"FieldHandle({self._applied})"


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

        # Stored ASSUMPTIONS / Relations (kinematic BCs, hydrostatic, …).  An
        # Assumption holds a ``subs_map`` (not a single ``.expr``), so it lives
        # in its own registry rather than ``_equations``.  ``PDETransformation``
        # rewrites their subs_maps alongside the equations, and ``model.<name>``
        # / ``model.apply(model.<name>)`` reach + apply them.
        self._assumptions: dict = {}
        # Names of equation rows flagged ``group="aux"`` — auxiliary relations
        # (w-reconstruction, ω-definition, …) that are derived/used but are NOT
        # part of the final PDE system handed to a SystemModel.  Oriented
        # relations in ``_assumptions`` (KinematicBCs) are auxiliary too.
        self._aux_names: set = set()
        # Field HEADS explicitly demoted to auxiliary via :meth:`move_to_aux`
        # (e.g. the bed ``b`` — given data that appears in the primary momentum
        # slope ``g h ∂_x b`` yet has no evolution equation).  A head listed here
        # is kept OUT of ``Q`` even when it is present in a primary equation, and
        # surfaces in ``Qaux`` instead.
        self._aux_fields: set = set()

        # Function groups — named slots that hold explicit DEFINITIONS of
        # derived model operators (the vertical reconstruction → ``interpolate``,
        # its Galerkin inverse → ``project``, derived BCs, …) rather than PDE
        # residuals or auxiliary fields.  ``{slot_name: {component_index: rhs}}``;
        # populated by :meth:`register_group` and parsed straight into the
        # matching ``SystemModel`` slot by ``from_model``.
        self._function_groups: dict = {}

        # Derivation history + per-model modal-index registry.  The registry
        # hands out a distinct summation index per coefficient family and is
        # read back by ``separation_of_variables`` / ``ResolveBasis``.
        self.history: list[dict] = []
        from zoomy_core.model.derivation.modal import ModalIndexRegistry
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
    def _history(self, op_name, target, level="major", description=None,
                 log_level=1):
        self.history.append({
            "op": op_name,
            "target": target,
            "level": level,
            "log_level": log_level,
            "description": description,
        })

    # ── equation registration ────────────────────────────────────────
    def add_equation(self, name, shape_or_expr=None, expr=None, *, group="model"):
        """Register a scalar or vector equation.

        Scalar::

            model.add_equation("mass", expr)

        Vector::

            model.add_equation("momentum", (2,), [expr_x, expr_z])

        The shape argument is optional positionally — ``add_equation(name,
        expr)`` registers a scalar row.

        An ORIENTED relation — a :class:`~zoomy_core.model.operations.KinematicBC`
        or any oriented ``Expression`` / solved ``Equation`` carrying
        ``_as_relation`` (a ``{lhs: rhs}`` rule) — may be passed in place of the
        expression.  It is STORED as the object; ``PDETransformation`` σ-maps
        its rule alongside the equations, and ``model.<name>`` /
        ``model.apply(model.<name>)`` reach and apply it as a substitution::

            model.add_equation("kbc_bot", KinematicBC(...))   # physical KBC
            model.apply(PDETransformation(...))               # σ-maps it: w(b)→w̃(0)
            model.apply(model.kbc_bot)                        # substitute w̃(0) = …

        ``group`` tags the row: ``"model"`` (default) for the final PDE system,
        or ``"aux"`` for auxiliary relations (``w``-reconstruction,
        ``ω``-definition, …) that are derived/used but not handed to a
        SystemModel.  ``describe(show="aux")`` lists them.  Oriented relations
        (KinematicBCs) are auxiliary regardless.
        """
        if group not in ("model", "aux"):
            raise ValueError(
                f"add_equation(group=...): 'model' | 'aux', got {group!r}.")
        if group == "aux":
            self._aux_names.add(name)
        # Store an ORIENTED relation (KinematicBC / a solved Equation — anything
        # carrying ``_as_relation``) AS the object: it holds ``{lhs: rhs}``, not
        # a bare residual.  ``PDETransformation`` σ-maps its rule alongside the
        # equations, and ``model.<name>`` / ``model.apply(model.<name>)`` reach
        # and apply it as a substitution.
        if expr is None and getattr(shape_or_expr, "_as_relation", None):
            self._assumptions[name] = shape_or_expr
            return shape_or_expr

        # Disambiguate the two-arg scalar call from the three-arg vector call.
        if expr is None and shape_or_expr is not None and not (
                isinstance(shape_or_expr, tuple)
                and all(isinstance(k, int) for k in shape_or_expr)):
            # scalar: add_equation(name, expr).  Accept an existing Equation /
            # Expression and COPY its residual — e.g. duplicate continuity for
            # the w-reconstruction: ``m.add_equation("w", m.mass)``.
            scalar_expr = (shape_or_expr.expr
                           if hasattr(shape_or_expr, "expr") else shape_or_expr)
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

    def remove(self, name):
        """Remove a named equation / vector row / assumption (KBC) from the
        model — convenience for dropping a relation once it has been consumed
        (``m.remove("kbc_top")``, ``m.remove("omega")``).  Returns ``self``."""
        if name in self._assumptions:
            del self._assumptions[name]
            self._refresh_unknowns()
        else:
            self._remove_equation(name)
        return self

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
        asmpt = self.__dict__.get("_assumptions", {})
        if item in asmpt:
            return asmpt[item]
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
        lvl = getattr(op, "log_level", 1)
        if getattr(op, "system_level", False):
            result = op(self)
            self._history(getattr(op, "name", None) or type(op).__name__,
                          "*system*", log_level=lvl)
            return result if result is not None else self
        if getattr(op, "whole_model_op", False):
            op.apply_to_model(self)
            self._history(getattr(op, "name", None) or type(op).__name__,
                          "*model*", log_level=lvl)
            return self
        for eq in list(self._equations.values()):
            eq.apply(op, _no_history=True, **kwargs)
        self._history(getattr(op, "name", None) or type(op).__name__, "*all*",
                      log_level=lvl)
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

    @staticmethod
    def _as_field_list(value):
        """A Q value is either a single field or a LIST of fields (after a
        family is split into modes, ``{u: [u_0, u_1, …]}``).  Normalise to a
        list so the Q-bookkeeping can iterate uniformly."""
        return list(value) if isinstance(value, (list, tuple)) else [value]

    @property
    def Qaux(self):
        """Derived: every field Function present in the equations that is not
        in ``Q``, not a coordinate, and not a parameter."""
        q_heads = {self._head(f)
                   for v in self._Q.values() for f in self._as_field_list(v)}
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

    def explicit_state(self):
        """One applied field per EVOLUTION (``∂_t``) row — the explicit state
        vector for a :class:`~zoomy_core.systemmodel.system_model.SystemModel`
        transition, derived from the model's OWN equations so
        ``SystemModel.from_model`` needs no hand-passed ``Q``.

        Each PRIMARY (``group="model"``) row contributes the field carried by
        its time derivative, with modal families flattened to their individual
        modes, in equation order — e.g. ``[b, h, q_0, q_1, q_2]``.  A field
        already claimed by an earlier row is skipped, so a conservative momentum
        row ``∂_t(q_k/(2k+1))`` (which, expanded, also touches ``∂_t h``) yields
        ``q_k``, not ``h`` again.  Auxiliary rows (the ``w``-reconstruction, …)
        are excluded — they become aux-update relations, not states.
        """
        t = self.coords[0]
        out, seen = [], set()
        for name, eq in self._equations.items():
            if name in self._aux_names:
                continue
            cands = []
            for der in sp.expand(eq.expr).atoms(sp.Derivative):
                if der.variables == (t,):
                    for f in der.args[0].atoms(sp.Function):
                        if self._is_field_application(f) and f not in seen:
                            cands.append(f)
            if cands:
                f = sorted(cands, key=sp.srepr)[0]
                out.append(f)
                seen.add(f)
        return out

    @property
    def functions(self):
        """Registry of EVERY field by its ORIGINAL name → :class:`FieldHandle`.

        The complement to ``Q`` (primary unknowns) and ``Qaux`` (derived): a
        single lookup for any field — primary or auxiliary, decorated or not —
        keyed by the name it was DECLARED with, so ``m.functions.tau_xz`` gives
        the σ-mapped ``\\tilde{tau_xz}`` head without touching
        ``m._field_decoration``::

            tau = m.functions.tau_xz
            mx.apply({tau.at(1): 0, tau.at(0): -lam * m.functions.u.at(0)})
            mx.apply({tau.expr: rho*nu/h * d.zeta(m.functions.u.expr)})

        The representative form per field is the BULK one (vertical arg a
        Symbol), normalised so its last argument is the canonical σ-coordinate
        ``ζ`` — so ``.expr`` reads ``\\tilde u(t, x, ζ)`` even when only the
        boundary / dummy forms are literally present in the equations."""
        zeta = sp.Symbol("zeta", real=True)
        deco_to_orig = {deco: orig.__name__
                        for orig, deco in self._field_decoration.items()}

        def _rank(f):
            # Prefer the canonical ζ form, then any Symbol/Dummy vertical arg,
            # then boundary / numeric forms — so the handle wraps the bulk field.
            last = f.args[-1] if f.args else None
            if last == zeta:
                return 2
            return 1 if isinstance(last, sp.Symbol) else 0

        reps: dict = {}
        for f in self._collect_fields():
            head = self._head(f)
            name = deco_to_orig.get(head, getattr(head, "__name__", str(head)))
            if name not in reps or _rank(f) > _rank(reps[name]):
                reps[name] = f

        out = Zstruct()
        for name in sorted(reps):
            f = reps[name]
            head = self._head(f)
            # σ-mapped fields: normalise the vertical arg to the canonical ζ so
            # ``.expr`` / ``.at`` read off ``\tilde f(t, x, ζ)`` regardless of
            # which form (``ζ`` / ``\hat ζ`` / ``0`` / ``1``) was present.
            if (head in deco_to_orig and f.args
                    and isinstance(f.args[-1], sp.Symbol)):
                f = head(*f.args[:-1], zeta)
            setattr(out, name, FieldHandle(f))
        out._symbolic_name = "functions"
        return out

    def _refresh_unknowns(self):
        """Drop from ``Q`` any declared unknown that has appeared in an
        equation and then left the PRIMARY system.

        A field stays in ``Q`` while it appears in a primary (``group="model"``)
        equation.  Two ways out, handled uniformly:

        * fully substituted out of EVERY equation (``p`` after the hydrostatic
          reduction) — gone from the primary scan, dropped;
        * eliminated from the primary equations but still DEFINED by its own
          auxiliary relation (``w̃`` after ``ω``-elimination) — also gone from
          the primary scan, dropped from ``Q`` and surfaced by :attr:`Qaux`.

        A declared-but-never-yet-present unknown (``h`` declared before its
        balance is added) is kept; it is dropped only once it has been seen
        (in ANY equation) and is gone from the primary system.  (Q gains a
        symbol only via :meth:`redeclare_unknown` / the setter.)
        """
        primary_heads = {self._head(f)
                         for f in self._collect_fields(primary_only=True)}
        self._ever_seen_heads |= {self._head(f) for f in self._collect_fields()}

        def _kept(v):
            heads = [self._head(f) for f in self._as_field_list(v)]
            # Explicitly demoted (move_to_aux) → never a state, even if it still
            # appears in a primary equation (the bed-slope ``b``).
            if any(hd in self._aux_fields for hd in heads):
                return False
            # Keep while present in the PRIMARY system, or while never seen yet
            # (declared-early case, e.g. ``h`` before its balance is added).
            return (any(hd in primary_heads for hd in heads)
                    or all(hd not in self._ever_seen_heads for hd in heads))

        self._Q = {nm: v for nm, v in self._Q.items() if _kept(v)}

    def move_to_aux(self, *fields):
        """Demote one or more PRIMARY-state fields to auxiliary.

        Removes each field's head from ``Q`` and keeps it out — even when the
        field still appears in a primary equation — so it surfaces in
        :attr:`Qaux` instead.  This is the convenient handle for a field that is
        part of the primary system but is GIVEN rather than evolved: the bed
        ``b`` appears in the momentum slope ``g h ∂_x b`` yet has no evolution
        equation, so ``m.move_to_aux(b)`` (or ``m.move_to_aux("b")``) records
        that intent explicitly.  Idempotent; returns ``self``.
        """
        for field in fields:
            self._aux_fields.add(self._resolve_head(field))
        self._refresh_unknowns()
        return self

    def register_group(self, slot, index, relation):
        """Register ``relation`` as component ``index`` of the function group
        ``slot`` — an explicit DEFINITION of a derived operator that
        ``from_model`` parses straight into the matching ``SystemModel`` slot
        (``'interpolate'`` → ``interpolate_to_3d``, ``'project'`` →
        ``project_from_3d``; the slot→attribute map is open, see
        :func:`zoomy_core.systemmodel.system_model.register_function_slot`).

        ``relation`` may be an oriented model relation (a solved ``Equation`` /
        ``FieldHandle`` carrying ``_as_relation``) or a bare sympy expression —
        e.g. the modal reconstruction ``Σ_j ŵ_j φ_j(ζ)`` built from the (aux)
        coefficients ``ŵ_j``.

        Purely ADDITIVE: it only records the definition.  The fields the
        definition references — the coefficients ``ŵ_j``, the field ``w̃`` —
        STAY in ``Q`` / ``Qaux``; membership is yours to manage (the runtime
        computes the aux coefficients and we use them to reconstruct ``w``).
        Any ``∂_x(state)`` a definition needs is still turned into a runtime
        gradient aux (``dhdx``, ``dq0dx``, …) at extraction time.

        Returns ``self``.
        """
        self._function_groups.setdefault(slot, {})[int(index)] = \
            self._relation_rhs(relation)
        return self

    @staticmethod
    def _relation_rhs(relation):
        """Extract the defining right-hand side from a registration argument:
        an oriented relation (``_as_relation`` rule), an ``Equation`` (``.rhs``),
        a ``FieldHandle`` / bare expression (sympified)."""
        rel = getattr(relation, "_as_relation", None)
        if isinstance(rel, dict) and rel:
            return sp.sympify(next(iter(rel.values())))
        if hasattr(relation, "rhs"):
            return sp.sympify(relation.rhs)
        return sp.sympify(getattr(relation, "expr", relation))

    def _resolve_head(self, field):
        """The Function head for ``field`` given as a head, an applied Function,
        or a field NAME (matched against the fields present in the equations,
        original or σ-decorated)."""
        if isinstance(field, str):
            for f in self._collect_fields():
                if self._field_name(f) == field:
                    return self._head(f)
            for orig, deco in self._field_decoration.items():
                if getattr(orig, "__name__", None) == field:
                    return deco
            raise KeyError(f"move_to_aux: no field named {field!r}")
        return self._head(field)

    def redeclare_unknown(self, old_field, new_fields, rename_key=False):
        """Transfer unknown-status from ``old_field`` to ``new_fields``.

        ``Q`` is ``{logical_name: value}``.  Two modes:

        * ``rename_key=False`` (default — a DECORATION, e.g. the σ-map
          ``u → ũ``): rewrite the VALUE, KEEP the logical key, so the handle
          survives::

              {u: u}  --σ-map-->  {u: ũ}

        * ``rename_key=True`` (a genuine FAMILY RENAME, e.g. the basis ansatz
          ``u → a`` or the change of variables ``a → q``): move the value to a
          key named after the NEW family, IN PLACE so the state-vector ORDER is
          preserved::

              {u: ũ}  --basis ansatz-->  {a: [a_0, a_1, …]}  --CoV-->  {q: …}

        A single ``new`` becomes a single value (``m.Q.q`` → that field); a list
        of ``new`` becomes a list value (``m.Q.a`` → ``[a_0, …]``,
        ``m.Q.a[1]`` → ``a_1``).  ``old_field`` / ``new_fields`` may be applied
        Functions or bare heads.  Not user-facing — ops
        (``PDETransformation`` / ``separation_of_variables`` /
        ``ChangeOfVariables``) call this.
        """
        old_head = self._head(old_field)
        if not isinstance(new_fields, (list, tuple)):
            new_fields = [new_fields]
        value = new_fields[0] if len(new_fields) == 1 else list(new_fields)

        def _is_old(v):
            return any(self._head(f) == old_head
                       for f in self._as_field_list(v))

        # The LOGICAL key(s) the old family currently occupies.
        old_keys = [nm for nm, v in self._Q.items() if _is_old(v)]
        if len(old_keys) == 1:
            # In-place swap (preserves dict ORDER): keep the key for a
            # decoration, or rename it to the new family for a rename.
            new_key = (self._field_name(new_fields[0]) if rename_key
                       else old_keys[0])
            self._Q = {(new_key if nm == old_keys[0] else nm):
                       (value if nm == old_keys[0] else v)
                       for nm, v in self._Q.items()}
        else:
            # Spread across several keys (or none yet): drop the old entries and
            # key each new field by its own family name.
            self._Q = {nm: v for nm, v in self._Q.items() if not _is_old(v)}
            for nf in new_fields:
                self._Q[self._field_name(nf)] = nf

    # ── field introspection ──────────────────────────────────────────
    @staticmethod
    def _is_field_application(atom):
        """True iff ``atom`` is an application of an *undefined-like* Function
        head — i.e. a user/derivation field, not a sympy builtin.

        This covers BOTH plain ``sp.Function("u")(...)`` applications
        (``AppliedUndef``) AND the dynamically-minted decorated heads a
        :class:`~zoomy_core.model.derivation.transformations.PDETransformation`
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

    def _collect_fields(self, primary_only=False):
        """Every applied field Function across the equations — every
        undefined-like Function application whose head is not a parameter.

        ``primary_only=True`` restricts the scan to the PRIMARY (``group="model"``)
        equations, skipping the auxiliary rows (the ``w``-reconstruction, the
        ``ω``-definition).  This is what lets a field that has been ELIMINATED
        from the primary system but still lives in its own auxiliary definition
        (``w̃``) drop out of ``Q`` — the same way a fully substituted-out field
        (``p``) does."""
        param_names = set(self.parameters.keys())
        fields = set()
        for name, eq in self._equations.items():
            if primary_only and name in self._aux_names:
                continue
            for atom in eq.expr.atoms(sp.Function):
                if not self._is_field_application(atom):
                    continue
                if atom.func.__name__ in param_names:
                    continue
                fields.add(atom)
        return fields

    # ── index registration (Galerkin test / moment index) ────────────
    def register_index(self, name):
        """Mint and REGISTER a Galerkin test / moment index Symbol on this
        model (``k = model.register_index("k")``).

        The model's :class:`~zoomy_core.model.derivation.modal.ModalIndexRegistry`
        records it, so a second ``register_index`` of the same name returns the
        SAME symbol and a clash with an auto-assigned modal (trial) index
        ``i``/``j`` raises — the user cannot accidentally create two colliding
        indices.  Returns an ``integer, non-negative`` ``sp.Symbol``."""
        return self._modal_registry.register(name)

    # ── modal-index / basis accessors (per-field Resolve) ─────────────
    def modal_index(self, target):
        """The modal-mode index assigned to a field or coefficient family.

        ``target`` may be the coefficient (head class / applied / name) OR the
        field (``u(t, x, z)`` / decorated head); both ends were registered by
        :func:`~zoomy_core.model.derivation.modal.separation_of_variables`, so either
        resolves to the same distinct index (``a → i``, ``aw → j``).  Used as
        ``ResolveBasis(model.modal_index(u), …)``.
        """
        idx = self._modal_registry.index_for(target)
        if idx is None:
            from zoomy_core.model.derivation.modal import _coeff_key
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
    def describe(self, strip_args=False, show="model", show_tags=False,
                 show_history=False, max_log_level=5):
        """Human-readable derivation state.  Header shows the model name +
        #equations + #ops; each equation delegates to its own
        ``describe_line``.

        ``show`` selects which rows are rendered:

        * ``"model"`` (default) — the final PDE rows only.
        * ``"aux"`` — only the auxiliary rows: ``group="aux"`` equations
          (``w``-reconstruction, ``ω``-definition, …) and the oriented
          relations / BCs in ``_assumptions`` (``KinematicBC`` …).
        * ``"all"`` — both, under separate headings.

        ``max_log_level`` (1–5) filters the optional history view: only ops with
        ``op.log_level <= max_log_level`` are listed, so ``max_log_level=1``
        shows just the major steps and hides level-5 bookkeeping (e.g.
        ``Simplify``)."""
        if show not in ("model", "aux", "all"):
            raise ValueError(
                f"describe(show=...): expected 'model' | 'aux' | "
                f"'all', got {show!r}.")
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
        if show in ("model", "all"):
            rendered = set()
            parts.append("\n**Equations:**")
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
                            strip_args=strip_args, bullet=True, show_tags=show_tags))
                        rendered.add(comp.name)
            for base, fam in self._moment_families.items():
                if base in self._vector_equations or any(
                        fam is c for v in self._vector_equations.values()
                        for c in v.components):
                    continue
                parts.append(f"- **{base}** (moment family):")
                for row in fam.components:
                    parts.append("  " + row.describe_line(
                        strip_args=strip_args, bullet=True, show_tags=show_tags))
                    rendered.add(row.name)
            for name, eq in self._equations.items():
                if name in rendered or name in self._aux_names:
                    continue          # aux rows render under the Auxiliary heading
                parts.append(eq.describe_line(strip_args=strip_args, bullet=True, show_tags=show_tags))
        # Auxiliary: ``group="aux"`` equation rows (w, ω, …) PLUS the oriented
        # relations / BCs in ``_assumptions`` (KinematicBC …, ``lhs = rhs``).
        if show in ("aux", "all"):
            aux_rows = [(nm, self._equations[nm]) for nm in self._equations
                        if nm in self._aux_names]
            if aux_rows or self._assumptions:
                from zoomy_core.model.operations import _StripArgsLatexPrinter
                pr = _StripArgsLatexPrinter() if strip_args else None
                tex = (lambda e: pr.doprint(e)) if pr else sp.latex
                parts.append("\n**Auxiliary:**")
                for nm, eq in aux_rows:
                    parts.append(eq.describe_line(strip_args=strip_args, bullet=True, show_tags=show_tags))
                for name, a in self._assumptions.items():
                    rules = getattr(a, "subs_map", {}) or {}
                    body = " \\\\ ".join(f"{tex(l)} = {tex(r)}"
                                         for l, r in rules.items())
                    parts.append(f"- **{name}:** ${body}$")
        if show_history and self.history:
            shown = [h for h in self.history
                     if h.get("log_level", 1) <= max_log_level]
            parts.append(f"**History** (≤ level {max_log_level}, "
                         f"{len(shown)}/{len(self.history)} ops):")
            for h in shown:
                parts.append(f"- `{h['op']}` (L{h.get('log_level', 1)}) "
                             f"→ {h['target']}")
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
    :class:`~zoomy_core.model.derivation.closure.Resolve` (``φ(l, ζ) → φ(k, ζ)`` at
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
        from zoomy_core.model.derivation.closure import Resolve

    comps = []
    for k in modes:
        row_name = f"{base}_{k}"
        model.add_equation(row_name, src_expr)
        row = model._equations[row_name]
        # Specialise the abstract test index ``l`` → concrete mode ``k`` — a
        # plain exact substitution (no Substitution class needed).
        row.apply({index: k}, _no_history=True)
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


class ResolveModes(Operation):
    """Galerkin moment-axis SHAPE BUMP, as a tracked ``.apply`` op.

    Applied to a row carrying the ABSTRACT test index ``index``
    (``model.momentum.x.apply(ResolveModes(index=k, modes=range(N+1)))``), it
    specialises ``index`` into the concrete moments ``modes`` (``k → 0..N``) and
    bumps the scalar/vector-component row into a
    :class:`MomentFamily` — the structural step previously written as the bare
    ``resolve_modes(model.momentum.x, …)`` call, now a node in the operation
    tree.  It does NOT close the φ-brackets; follow it with
    ``model.apply(ResolveIntegral(...))``."""

    def __init__(self, index, modes, name="resolve_modes"):
        self._index = index
        self._modes = list(modes)
        super().__init__(
            name=name,
            description=f"moment shape-bump {index} → {self._modes}")

    def apply_to_equation(self, equation):
        return resolve_modes(equation, index=self._index, modes=self._modes)
