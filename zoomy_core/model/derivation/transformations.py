"""The transformation + boundary-condition layer for the clean-redesign
derivation framework.

One public piece:

:class:`PDETransformation`
    The explicit-geometry σ-map ``z = b + h·ζ`` from the physical vertical
    coordinate ``z`` to the reference ``ζ ∈ [0, 1]``.  It mints a *decorated*
    head ``f̃(t, x, ζ)`` for every field ``f(t, x, z)`` (LaTeX accent, e.g.
    ``ũ``) and applies the σ chain rule **as a differential rewrite rule on
    Derivative nodes** — producing ``(1/h)∂_ζ f̃`` and the metric-correction
    terms with NO leftover ``Subs`` artifacts.  This is the explicit-geometry
    counterpart of the production
    :class:`zoomy_core.model.operations.SigmaTransform` (which keeps the field
    head and substitutes ``z → ζ·h + b`` into the argument list); here the
    decorated heads are *distinct* Function classes so the σ-mapped atoms
    cannot cross-substitute with their physical-z originals.

    It also σ-maps any stored boundary conditions: a
    :class:`~zoomy_core.model.operations.KinematicBC` added via
    ``model.add_equation`` is an oriented ``Expression`` whose ``{lhs: rhs}``
    relation is rewritten alongside the bulk equations (``w(b) → w̃(0)`` …),
    ready to apply as a substitution.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation


__all__ = ["PDETransformation"]


# ── PDETransformation ────────────────────────────────────────────────────


class PDETransformation(Operation):
    """Map the physical vertical coordinate ``z`` to the reference ``ζ`` via
    an explicit geometry ``z = b(t, x) + h(t, x)·ζ``.

    Usage::

        pde = PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))})
        model.apply(pde)
        u_tilde = pde.decorated(u)          # ũ(t, x, ζ)

    With ``ζ = (z − b) / h``, every field ``f(t, x, z)`` becomes the decorated
    ``f̃(t, x, ζ)`` and derivatives are rewritten by the chain rule:

    .. math::

        \\partial_z f &\\to \\tfrac{1}{h}\\,\\partial_\\zeta \\tilde f \\\\
        \\partial_x f &\\to \\partial_x \\tilde f
            - \\tfrac{\\partial_x(\\zeta h + b)}{h}\\,\\partial_\\zeta \\tilde f \\\\
        \\partial_t f &\\to \\partial_t \\tilde f
            - \\tfrac{\\partial_t(\\zeta h + b)}{h}\\,\\partial_\\zeta \\tilde f

    The Jacobian ``dz = h dζ`` is available via :attr:`jacobian` so a later
    depth-projection recovers the ``h`` factor.

    This is a **model-level** operation (``whole_model_op = True``): it rewrites
    the whole coordinate system and every equation, and records the σ-map
    metadata on the model (``_field_decoration`` / ``_sigma_from`` /
    ``_vertical`` / ``coord_relations``) so :func:`kinematic_bc` and later
    depth-projection can read it.  A bare equation-level apply raises.
    """

    # A coordinate transformation is a change of representation → rewrite BCs.
    transforms_bcs = True
    # Model-level: routed through ``apply_to_model`` once.
    whole_model_op = True

    _ACCENTS = {
        "tilde": r"\tilde",
        "hat": r"\hat",
        "bar": r"\bar",
        "check": r"\check",
    }

    def __init__(self, coord_map, *, decorate="tilde",
                 name="pde_transformation", description=None):
        if len(coord_map) != 1:
            raise ValueError(
                "PDETransformation supports a single coordinate replacement "
                f"z → ζ; got {len(coord_map)} entries."
            )
        ((self._z, spec),) = coord_map.items()
        self._zeta, self._b, self._h = self._parse_spec(spec)
        self._accent = self._ACCENTS.get(decorate, decorate)
        self._via_model = False
        # Head decoration cache (one decorated Function per original head).
        self._decorated: dict = {}
        # ``{head: z_slot_index}`` for every field that appears with the vertical
        # coord ``z`` SOMEWHERE in the model.  Populated by a pre-scan in
        # ``apply_to_model`` so a BOUNDARY EVALUATION ``u(t,x,b)`` (whose z-slot
        # holds an interface value, not the free ``z``) is mapped to
        # ``ũ(t,x,(b−b)/h) = ũ(t,x,0)`` — i.e. evaluated at the σ-height of that
        # interface — instead of being left untouched.
        self._vertical_heads: dict = {}

        # The coords that receive the σ chain-rule jacobian = exactly the
        # coords the geometry depends on.  The metric correction is
        # ``∂_s(ζ h + b) = ζ ∂_s h + ∂_s b``, which is identically ZERO unless
        # ``s`` is a free symbol of ``b`` or ``h``.  So we read the chain
        # coords straight off ``b``/``h`` (e.g. ``b(t,x), h(t,x)`` → ``{t,
        # x}``) — no ``horizontal=`` argument needed at the call site.
        self._chain_coords = tuple(
            (self._b.free_symbols | self._h.free_symbols) - {self._zeta}
        )

        super().__init__(
            name=name,
            description=(description or
                         f"PDE transform: {self._z} → {self._zeta}, "
                         f"decorate={decorate}, chain-rule on "
                         f"{self._chain_coords} + ∂_{self._z}"),
        )

    def _parse_spec(self, spec):
        """Read ``(ζ, b, h)`` out of the coord-map spec.

        ``spec`` is ``(ζ, relation)``; ``relation`` is an ``Eq`` or the bare
        inverse expr ``b + h·ζ``.  We normalise to the inverse form
        ``z = b + h·ζ`` and read ``b = z|_{ζ=0}``, ``h = ∂_ζ z``.
        """
        zeta, relation = spec
        if isinstance(relation, sp.Equality):
            lhs, rhs = relation.args
            # Eq(z, b+hζ)  → inverse is rhs.  Eq(ζ, (z−b)/h) → solve for z.
            if lhs == self._z:
                inverse = rhs
            elif lhs == zeta:
                sols = sp.solve(sp.Eq(zeta, rhs), self._z)
                if not sols:
                    raise ValueError(
                        f"could not solve {relation} for {self._z}.")
                inverse = sols[0]
            else:
                inverse = rhs
        else:
            inverse = sp.sympify(relation)
        # z = b + h·ζ  ⇒  h = ∂_ζ(inverse), b = inverse|_{ζ=0}.
        h = sp.diff(inverse, zeta)
        b = inverse.subs(zeta, 0)
        return zeta, b, h

    # ── geometry accessors ────────────────────────────────────────────
    @property
    def jacobian(self):
        """The vertical Jacobian ``dz/dζ = h`` (so ``dz = h dζ``)."""
        return self._h

    # ── decorated head minting ────────────────────────────────────────
    def _decorate_head(self, head):
        """A decorated copy of a Function head whose LaTeX is the accented
        original (``u`` → ``\\tilde{u}``).  Cached per original head.

        The minted head is a *distinct* Function subclass so the σ-mapped
        atom ``ũ(t, x, ζ)`` never cross-substitutes with the physical-z
        original ``u(t, x, z)``.
        """
        if head in self._decorated:
            return self._decorated[head]
        orig_name = head.__name__
        new_name = f"{self._accent}{{{orig_name}}}"
        accent = self._accent

        def _latex(expr, printer=None, exp=None):
            render = printer._print if printer is not None else sp.latex
            inner = sp.latex(sp.Symbol(orig_name))
            head_tex = rf"{accent}{{{inner}}}"
            # Honour the strip-args printer: ``\tilde{u}`` without the
            # ``(t, x, ζ)`` argument list (the bare-head form).
            if type(printer).__name__ == "_StripArgsLatexPrinter":
                s = head_tex
            else:
                args = ", ".join(str(render(a)) for a in expr.args)
                s = rf"{head_tex}\left({args}\right)"
            return s if exp is None else f"{s}^{{{exp}}}"

        new_head = type(new_name, (sp.Function,),
                        {"_latex": _latex, "_pde_decorated_from": orig_name})
        self._decorated[head] = new_head
        return new_head

    # ── decorated-field accessor (post-transform) ─────────────────────
    def decorated(self, field):
        """The σ-mapped decorated counterpart of an ORIGINAL field application.

        ``pde.decorated(u)`` where ``u = u(t, x, z)`` returns ``ũ(t, x, ζ)`` —
        the same head this transform minted, applied to the z→ζ-mapped
        argument list.  Returns ``field`` unchanged if the head was never
        decorated (the field carried no ``z``).
        """
        head = self._decorated.get(field.func)
        if head is None:
            return field
        new_args = [a.xreplace({self._z: self._zeta}) for a in field.args]
        return head(*new_args)

    # ── leaf guard (model-only) ───────────────────────────────────────
    def _leaf_sp(self, sp_expr):
        if not self._via_model:
            raise TypeError(
                "PDETransformation is a model-level operation — it rewrites "
                "the whole coordinate system (z → ζ) and every equation. "
                "Apply it with `model.apply(PDETransformation(...))`, not "
                "`eq.apply(...)` / `term.apply(...)`."
            )
        return self._rewrite(sp_expr)

    # ── model dispatch ────────────────────────────────────────────────
    def apply_to_model(self, model):
        self._via_model = True
        try:
            self._rewrite_model_coords(model)
            # Pre-scan: record every field head that appears with the vertical
            # coord ``z`` and the arg position it occupies — order-independent, so
            # a KBC's boundary evaluation ``u(t,x,b)`` is mapped even when its
            # equation is rewritten before the bulk one carrying ``u(t,x,z)``.
            for eq in model._equations.values():
                for f in eq.expr.atoms(sp.Function):
                    if isinstance(f, sp.Sum) or self._z not in f.free_symbols:
                        continue
                    for i, a in enumerate(f.args):
                        if a == self._z or self._z in a.free_symbols:
                            self._vertical_heads.setdefault(f.func, i)
                            break
            for eq in model._equations.values():
                eq.apply(self, _no_history=True)
            # Rewrite stored assumptions (KinematicBC, …) too: σ-map both sides
            # of each rule so e.g. ``{w(b): ∂_t b + u(b)∂_x b}`` becomes
            # ``{w̃(0): ∂_t b + ũ(0)∂_x b}`` — ready to apply as a substitution.
            for asmpt in getattr(model, "_assumptions", {}).values():
                new_rel = {self._rewrite(lhs): self._rewrite(rhs)
                           for lhs, rhs in asmpt.subs_map.items()}
                # Stored assumptions are oriented Expressions (KinematicBC, …):
                # rewrite the ``{lhs: rhs}`` relation AND the residual term so
                # both sides are σ-mapped (``subs_map`` is a read-only view).
                asmpt._as_relation = new_rel
                asmpt.expr = sum((lhs - rhs for lhs, rhs in new_rel.items()),
                                 sp.S.Zero)
        finally:
            self._via_model = False
        # Publish the decoration map + σ-map metadata on the model so
        # ``kinematic_bc`` / depth-projection can read it.
        model._field_decoration = dict(self._decorated)
        model._sigma_from = self._z
        model._vertical = self._zeta
        # Transfer unknown-status from each decorated Q field to its decorated
        # head.  After the σ-map the physical-z head ``u`` is gone from every
        # equation (replaced by ``ũ``), so without this Q would silently drop
        # ``u`` instead of carrying ``ũ`` forward.  ``redeclare_unknown`` keys
        # on the family head, so we map ``u → ũ`` for every Q field that was
        # decorated.
        for nm, field in list(model._Q.items()):
            new_head = self._decorated.get(model._head(field))
            if new_head is not None:
                repl = {self._z: self._zeta}
                decorated = new_head(*[a.xreplace(repl) for a in field.args])
                model.redeclare_unknown(model._head(field), decorated)
        model._refresh_unknowns()
        return model

    def _rewrite_model_coords(self, model):
        coords = getattr(model, "_coords", None)
        if coords is None or self._z not in coords:
            return
        zeta = self._zeta
        model._coords = tuple(zeta if c == self._z else c for c in coords)
        model._vertical = zeta
        model.coord_relations[self._z] = self._b + self._h * zeta

    # ── the rewrite ───────────────────────────────────────────────────
    def _rewrite(self, e):
        """Recursive coordinate-map rewrite producing decorated ζ-functions
        and ζ-derivatives, with zero ``Subs``."""
        if isinstance(e, sp.Derivative):
            inner_new = self._rewrite(e.expr)
            result = inner_new
            for var, order in e.variable_count:
                for _ in range(int(order)):
                    result = self._apply_one_derivative(result, var)
            return result
        if isinstance(e, sp.Function) and not isinstance(e, sp.Sum):
            # A known/standard sympy function head (exp, log, sin, …) applied
            # to a field expression is a COMPOSITE, not a σ-field: recurse into
            # its argument(s) and PRESERVE the head — do NOT decorate it, strip
            # its inner ζ, nor fire the boundary-evaluation branch on the
            # function node itself.  Only genuine field/unknown applications
            # (sympy ``AppliedUndef``) carry the vertical coordinate and get the
            # σ-field treatment below.  Example: ``exp(lk(z))`` →
            # ``exp(l̃k(ζ))`` (head kept, inner field σ-mapped), instead of the
            # bogus ``\tilde{exp}`` / boundary-eval of ``lk`` at ``ζ``.
            if not isinstance(e, sp.core.function.AppliedUndef):
                new_args = tuple(self._rewrite(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
                return e
            # A field application.  If it contains z, decorate the head and
            # map z → ζ in the arg list; recurse into each arg.
            if self._z in e.free_symbols:
                new_args = [self._rewrite(a.xreplace({self._z: self._zeta}))
                            for a in e.args]
                new_head = self._decorate_head(e.func)
                return new_head(*new_args)
            # BOUNDARY EVALUATION of a vertical field: ``u(t,x,Z)`` with ``Z`` an
            # interface value (no free ``z``).  Evaluate the σ-field at the height
            # of that interface, ζ = (Z − b)/h  (e.g. Z = b → ζ = 0, b+h → 1).
            if e.func in self._vertical_heads:
                slot = self._vertical_heads[e.func]
                new_args = [self._rewrite(a) for a in e.args]
                if slot < len(new_args):
                    new_args[slot] = sp.simplify(
                        (e.args[slot] - self._b) / self._h)
                return self._decorate_head(e.func)(*new_args)
            # No z dependence — leave the head, but still recurse args.
            new_args = tuple(self._rewrite(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
            return e
        if e.args:
            new_args = tuple(self._rewrite(a) for a in e.args)
            if any(n is not o for n, o in zip(new_args, e.args)):
                return e.func(*new_args)
            return e
        # Atom: free z in non-Function context → b + h·ζ.
        if e == self._z:
            return self._zeta * self._h + self._b
        return e

    def _apply_one_derivative(self, inner_expr, var):
        zeta, h, b = self._zeta, self._h, self._b
        if var == self._z:
            # ∂_z u → (1/h)·∂_ζ ũ
            return (sp.Integer(1) / h) * sp.Derivative(inner_expr, zeta)
        if var in self._chain_coords:
            # ∂_s u|_z → ∂_s ũ − (∂_s(ζh+b)/h)·∂_ζ ũ
            # Skip the jacobian correction when ``inner_expr`` has no ζ
            # dependency — its ∂_ζ is identically zero and emitting it would
            # leave a stray ``Derivative(f, ζ)`` atom (e.g. ``∂_t b`` on a bed
            # with no z → must collapse to ``∂_t b`` directly).
            if zeta not in inner_expr.free_symbols:
                return sp.Derivative(inner_expr, var)
            jacobian = sp.Derivative(zeta * h + b, var)
            return (sp.Derivative(inner_expr, var)
                    - (jacobian / h) * sp.Derivative(inner_expr, zeta))
        return sp.Derivative(inner_expr, var)

    def _repr_latex_(self):
        return (rf"$\zeta = \frac{{{sp.latex(self._z)} - {sp.latex(self._b)}}}"
                rf"{{{sp.latex(self._h)}}}, \quad "
                rf"u({sp.latex(self._z)}) \mapsto {self._accent}{{u}}"
                rf"({sp.latex(self._zeta)}), \quad "
                rf"\partial_{sp.latex(self._z)} u \mapsto "
                rf"\tfrac{{1}}{{{sp.latex(self._h)}}}\partial_\zeta "
                rf"{self._accent}{{u}}$")
