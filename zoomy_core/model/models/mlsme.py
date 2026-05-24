"""MLSME — Multi-Layer Shallow Moment Equations.

Stack of ``N_layers`` SME(N) systems coupled through hydrostatic
pressure and bottom topography:

* Layer ℓ bottom:            ``b + Σ_{m<ℓ} h_m``
* Layer ℓ top pressure:      ``ρ·g·Σ_{m>ℓ} h_m``   (0 for the surface layer)

Construction:

    >>> mlsme = MLSME(
    ...     N_layers=2,
    ...     N=2,
    ...     parameters={"g": (9.81, "positive"),
    ...                 "rho": (1.0, "positive")},
    ...     boundary_conditions=...,
    ... )
    >>> sm = SystemModel.from_model(mlsme)

In the limit ``N = 0`` (no shear moments, piecewise-constant
velocity profile), ``MLSME(N_layers=L, N=0)`` recovers the
ML-SWE system.

Internally each layer runs the full SME(N) pipeline with a
layer-specific bottom + top-pressure; per-layer equations are
merged into one flat ``self._equations`` dict with layer-suffixed
names.

Reference: ml_sme_clean.py (canonical notebook in
``thesis/notebooks/legacy/modeling/transparent_derivations/``).
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.basemodel import Model
from zoomy_core.model.state import StateSpace, MassMomentum
from zoomy_core.model.equation import Equation
from zoomy_core.model.operations import (
    Expression,
    Multiply,
    ResolveDummy,
    ProductRule,
    Integrate,
    EvaluateIntegrals,
    SigmaTransform,
    KinematicBC,
    Symmetrize,
    Legendre_shifted,
)


__all__ = ["MLSME"]


class MLSME(Model):
    """Multi-layer SME at level ``N`` over ``N_layers`` layers."""

    # Equations leave ``derive_model`` in Function form (h_layer_ℓ
    # as Function calls).  ``SystemModel.from_model`` auto-triggers
    # the Function → Symbol substitution + tagging.
    _finalize_lazy = True

    def __init__(self, N_layers: int = 2, N: int = 2, **kwargs):
        self.N_layers = N_layers
        self.N = N

        # Variable layout: per layer, [h_layer_ℓ, q_layer_ℓ_0..N].
        var_names = []
        for ell in range(1, N_layers + 1):
            var_names.append(f"h_layer_{ell}")
            for k in range(N + 1):
                var_names.append(f"q_layer_{ell}_{k}")

        kwargs.setdefault("name", f"MLSME-L={N}-{N_layers}layers")
        kwargs.setdefault("variables", var_names)
        kwargs.setdefault("parameters", {
            "g":   (9.81, "positive"),
            "rho": (1.0,  "positive"),
        })
        kwargs.setdefault("eigenvalue_mode", "numerical")
        super().__init__(**kwargs)

    # ── Derivation hook ─────────────────────────────────────────────
    def derive_model(self):
        """Run the per-layer SME pipeline for each of ``N_layers``
        layers, unify the layer-h placeholders, then merge."""
        # Shared coordinate scaffold across all layers.  Each layer's
        # pipeline still gets its own ``MassMomentum`` (h, b, u, p, …
        # are layer-local Function objects until the END substitution
        # rewires them to the layer's actual h/b/q), but ``state.t``,
        # ``state.x``, ``state.zeta_ref`` are global so MLSME's
        # ``_assert_parameter_values_supplied`` can skip ``zeta_ref``
        # cleanly via the standard ``self.state`` mechanism.
        self.state = StateSpace(dimension=2)
        t_sym, x_sym = self.state.t, self.state.x
        h_pres = [
            sp.Function(f"h_layer_{ell}_pre", positive=True)(t_sym, x_sym)
            for ell in range(1, self.N_layers + 1)
        ]
        # Global bathymetry — shared across layers.
        b_global = sp.Function("b", real=True)(t_sym, x_sym)

        rho_sym = self.parameters.rho
        g_sym = self.parameters.g

        per_layer = []
        h_actuals = []
        q_fns = []
        for ell in range(1, self.N_layers + 1):
            # Layer ℓ sits ABOVE layers 1..ℓ-1 and BELOW layers ℓ+1..N_layers.
            bottom_expr = b_global + sum(h_pres[:ell - 1], sp.S.Zero)
            # Top pressure: 0 for the surface (top) layer, otherwise
            # ρ·g·(sum of heights of layers above).
            if ell < self.N_layers:
                top_p_expr = rho_sym * g_sym * sum(h_pres[ell:], sp.S.Zero)
            else:
                top_p_expr = sp.S.Zero
            sub_model, h_layer, q_fn = self._derive_layer_sme(
                layer_label=f"_layer_{ell}",
                bottom_expr=bottom_expr,
                top_pressure_expr=top_p_expr,
            )
            per_layer.append(sub_model)
            h_actuals.append(h_layer)
            q_fns.append(q_fn)

        # Unify placeholders ↔ actuals.
        unify = dict(zip(h_pres, h_actuals))
        for sub in per_layer:
            for eq in sub.values():
                eq.expr = eq.expr.xreplace(unify)
                eq.simplify()

        # Merge per-layer equations into ``self._equations``.  Only the
        # dynamic equations matter for the SystemModel (continuity_ell_0
        # for h_layer_ell; momentum_x_ell_k for q_layer_ell_k).  The
        # scratch dict also holds momentum_z and higher-mode continuity
        # branches (continuity_ell_1..continuity_ell_{N+1}) — these are
        # pipeline residue (pressure-elimination scaffold, w-closure
        # auxiliary) and are not part of the dynamic system.  Drop them.
        for ell, sub in zip(range(1, self.N_layers + 1), per_layer):
            label = f"_layer_{ell}"
            keep = (f"continuity{label}_0",
                    *[f"momentum_x{label}_{k}" for k in range(self.N + 1)])
            for name in keep:
                if name in sub:
                    self.add_equation(name, sub[name].expr)

        # Stash bookkeeping for downstream / inspection.
        self._layer_h = h_actuals
        self._layer_q_fns = q_fns
        self._b_global = b_global

    # ── Per-layer SME pipeline ───────────────────────────────────
    def _derive_layer_sme(self, *, layer_label, bottom_expr,
                          top_pressure_expr):
        """Inline port of ml_sme_clean.py's ``derive_layer_sme``
        factory.  Runs the full SME-N pipeline for one layer with
        layer-specific bottom / top-pressure coupling; returns the
        layer's tagged equation set plus the per-layer ``h_layer``
        symbol and ``q_fn`` Function.

        The pipeline uses the GENERIC ``state.b`` / ``state.h``
        throughout (so the algebraic structure is identical to a
        single-layer SME); the layer-specific substitutions
        (``state.b → bottom_expr``, ``state.h → h_layer``) are
        applied to every equation at the END of the pipeline.
        """
        N = self.N
        state = StateSpace(dimension=2)
        src = MassMomentum(state, self.parameters)
        s, t, x = state, state.t, state.x

        # ``equations``: dict[str, Equation] populated as we go.
        # We use a plain dict (not a Model) since the per-layer set
        # is internal scaffolding — merged into MLSME at the end.
        equations: dict[str, Equation] = {}

        def _new_eq(name, expression):
            equations[name] = Equation(expression, name=name, model=None)

        def _apply_all(op):
            """Broadcast an operation across every equation in the
            scratch dict — mimics ``Model.apply`` without registering
            a parent Model."""
            for eq in equations.values():
                eq.apply(op, _no_history=True)

        # ── Setup: 3 INS equations + bottom placeholder ────────────
        _new_eq(f"continuity{layer_label}",  src.continuity.expr)
        _new_eq(f"momentum_x{layer_label}",  src.momentum.x.expr)
        _new_eq(f"momentum_z{layer_label}",  src.momentum.z.expr)
        # tau_xx → 0
        _apply_all({src.tau.xx: 0})

        # ── Pre-σ hydrostatic + p-elimination with layer top BC ──
        eq_mz = equations[f"momentum_z{layer_label}"]
        eq_mz.apply({src.w: 0,
                     src.tau.zx: 0,
                     src.tau.zz: 0,
                     src.tau.xz: 0}).simplify()
        eq_mz.apply(Integrate(s.z, s.z, src.eta, method="analytical"))
        # Layer-specific top pressure (= 0 for the surface layer,
        # ρ·g·Σ_{m>ℓ} h_m otherwise).
        eq_mz.apply({src.p.subs(s.z, src.eta): top_pressure_expr}).simplify()
        p_subst = Expression(eq_mz.expr,
                              f"momentum_z{layer_label}").solve_for(src.p)
        equations[f"momentum_x{layer_label}"].apply(p_subst).simplify()

        # ── σ-transform + KBCs ────────────────────────────────────
        _apply_all(SigmaTransform(s, src))
        _apply_all(KinematicBC(s, src.b,   src, at=sp.S.Zero))
        _apply_all(KinematicBC(s, src.eta, src, at=sp.S.One))

        # ── × h and inverse-ProductRule conservative folds ─────────
        _apply_all(Multiply(src.h))
        for eq in equations.values():
            eq.simplify()
        equations[f"momentum_x{layer_label}"][[0, 1]].apply(
            ProductRule(variables=[t, x]))
        equations[f"continuity{layer_label}"][[0]].apply(
            ProductRule(variables=[x]))
        for eq in equations.values():
            eq.simplify()

        # ── Modal ansatz ─────────────────────────────────────────
        sigma = s.zeta_ref
        u_fn = sp.Function(f"u{layer_label}", real=True)
        w_fn = sp.Function(f"w{layer_label}", real=True)
        phi_u_fn = sp.Function("phi_u", real=True)
        phi_w_fn = sp.Function("phi_w", real=True)
        psi_u    = sp.Function("psi_u", real=True)
        psi_w    = sp.Function("psi_w", real=True)
        omega    = sp.Function("omega", real=True)

        u_ansatz = sum(u_fn(k, t, x) * phi_u_fn(k, sigma)
                       for k in range(N + 1))
        w_ansatz = sum(w_fn(k, t, x) * phi_w_fn(k, sigma)
                       for k in range(N + 2))
        _apply_all({
            src.u.xreplace({s.z: sigma}): u_ansatz,
            src.w.xreplace({s.z: sigma}): w_ansatz,
        })

        # ── Galerkin projection ──────────────────────────────────
        equations[f"continuity{layer_label}"].apply(
            Multiply(psi_w(sigma) * omega(sigma)))
        equations[f"momentum_x{layer_label}"].apply(
            Multiply(psi_u(sigma) * omega(sigma)))

        # ── σ-integrate ──────────────────────────────────────────
        equations[f"continuity{layer_label}"].apply(
            Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
        equations[f"momentum_x{layer_label}"].apply(
            Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
        for eq in equations.values():
            eq.simplify()

        # ── KBC modal closure ─────────────────────────────────────
        def kbc_residual(interface, at_value):
            lhs = src.w.subs(s.z, at_value)
            rhs = (sp.Derivative(interface, s.t)
                   + src.u.subs(s.z, at_value) * sp.Derivative(interface, s.x))
            return (lhs - rhs).xreplace({
                src.w.subs(s.z, at_value): w_ansatz.subs(sigma, at_value),
                src.u.subs(s.z, at_value): u_ansatz.subs(sigma, at_value),
            })

        closure = sp.solve(
            [kbc_residual(src.b,   sp.S.Zero),
             kbc_residual(src.eta, sp.S.One)],
            [w_fn(0, t, x), w_fn(1, t, x)],
            dict=True,
        )[0]
        _apply_all(closure)

        # ── CoV u → q/h ──────────────────────────────────────────
        q_fn = sp.Function(f"q{layer_label}", real=True)
        u_modes = [u_fn(k, t, x) for k in range(N + 1)]
        q_modes = [q_fn(k, t, x) for k in range(N + 1)]
        _apply_all({u: q / src.h for u, q in zip(u_modes, q_modes)})
        for eq in equations.values():
            eq.simplify()

        # ── Resolve opaque-basis dummies to Legendre polynomials ─
        legendre_u = Legendre_shifted(level=N)
        legendre_w = Legendre_shifted(level=N + 1)

        def basis_value(b, k_arg, sigma_arg):
            if k_arg.is_Integer:
                return b.eval(int(k_arg), sigma_arg)
            return sp.Function("phi_unresolved")(k_arg, sigma_arg)

        def legendre_value(b, k):
            return lambda arg, _b=b, _k=k: _b.eval(_k, arg)

        _apply_all(ResolveDummy(omega, lambda arg: sp.S.One))
        _apply_all(ResolveDummy(
            phi_u_fn, lambda k, sg, _b=legendre_u: basis_value(_b, k, sg)))
        _apply_all(ResolveDummy(
            phi_w_fn, lambda k, sg, _b=legendre_w: basis_value(_b, k, sg)))

        # ResolveDummy with list value branches each equation that
        # contains the dummy.  We need to manually branch here since
        # _apply_all uses per-equation dispatch.
        equations = _resolve_dummy_list(equations, psi_u,
                                         [legendre_value(legendre_u, k) for k in range(N + 1)])
        equations = _resolve_dummy_list(equations, psi_w,
                                         [legendre_value(legendre_w, k) for k in range(N + 2)])

        # ── Evaluate integrals ───────────────────────────────────
        _apply_all_dict(equations, EvaluateIntegrals(state))
        for eq in equations.values():
            eq.simplify()

        # ── Gravity self-pair fold ───────────────────────────────
        g = self.parameters.g
        mom0_name = f"momentum_x{layer_label}_0"
        if mom0_name in equations:
            mom0 = equations[mom0_name]
            grav = next(
                (t for t in mom0
                 if t.expr.has(g) and t.expr.has(sp.Derivative(src.h, s.x))
                 and not t.expr.has(src.b)),
                None)
            if grav is not None:
                grav.apply(Symmetrize(ProductRule(variables=[s.x])))
                mom0.simplify()

        # ── Higher-mode w closure ────────────────────────────────
        if N >= 1:
            w_higher = [w_fn(k, t, x) for k in range(2, N + 2)]
            higher_residuals = [
                equations[f"continuity{layer_label}_{k}"].expr
                for k in range(1, N + 1)
            ]
            w_solution = sp.solve(higher_residuals, w_higher, dict=True)[0]
            _apply_all_dict(equations, w_solution)
            for eq in equations.values():
                eq.simplify()

        # ── Mass-matrix inversion ─────────────────────────────────
        for k in range(N + 1):
            eq = equations[f"momentum_x{layer_label}_{k}"]
            eq.expr = (2 * k + 1) * eq.expr
            eq.simplify()

        # ── Layer-local substitution ─────────────────────────────
        # Replace the GENERIC ``state.b`` with the layer's
        # ``bottom_expr`` (which depends on the GLOBAL bathymetry +
        # any layers below), and the GENERIC ``state.h`` with the
        # layer's own height ``h_layer = Function("h_layer_ℓ")(t, x)``.
        h_layer = sp.Function(f"h{layer_label}", positive=True)(t, x)
        sub = {src.b: bottom_expr, src.h: h_layer}
        for eq in equations.values():
            eq.expr = eq.expr.xreplace(sub)
            eq.simplify()
        return equations, h_layer, q_fn

    # ── Lazy finalization hook ────────────────────────────────────
    def _prepare_for_systemmodel(self):
        """Substitute the layer-h Functions with their Model Symbol
        equivalents and set ``_variable_map``."""
        # The merged equations reference per-layer h_layer_ℓ Function
        # calls + per-layer q_layer_ℓ_k(t, x) Function calls.  Map
        # each to its corresponding Model Symbol.
        # Pull (t, x) from the first layer's h.
        h0 = self._layer_h[0]
        t_sym, x_sym = h0.args[0], h0.args[1]
        subs = {}
        # Layer h's.
        for ell_idx, h_layer in enumerate(self._layer_h):
            ell = ell_idx + 1
            subs[h_layer] = self.variables[f"h_layer_{ell}"]
        # Layer q's.
        for ell_idx, q_fn in enumerate(self._layer_q_fns):
            ell = ell_idx + 1
            for k in range(self.N + 1):
                subs[q_fn(k, t_sym, x_sym)] = self.variables[f"q_layer_{ell}_{k}"]
        for eq in self:
            eq.expr = eq.expr.xreplace(subs)
        # Variable map: per layer, continuity_0 → h_layer row;
        # momentum_x_k → q_layer_k row.
        self._variable_map = self._build_variable_map()

    def _build_variable_map(self):
        m = {}
        row = 0
        for ell in range(1, self.N_layers + 1):
            m[f"continuity_layer_{ell}_0"] = [row]
            row += 1
            for k in range(self.N + 1):
                m[f"momentum_x_layer_{ell}_{k}"] = [row]
                row += 1
        return m


# ── Helpers ────────────────────────────────────────────────────────

def _resolve_dummy_list(equations: dict, dummy, value_list):
    """ResolveDummy(value=list) branches each equation containing the
    dummy into one sub-equation per list element.  Mimics the legacy
    ``Model.resolve_dummy`` branching path on a plain dict of
    Equations (since our per-layer scratch isn't a Model)."""
    new_eqs = {}
    for name, eq in equations.items():
        if not eq.expr.has(dummy):
            new_eqs[name] = eq
            continue
        for i, v in enumerate(value_list):
            replace = (lambda expr, dum=dummy, vv=v:
                       expr.replace(dum, vv)
                       if callable(vv) and not isinstance(vv, sp.Basic)
                       else expr.replace(dum,
                                          lambda *args, _v=vv: _v))
            new_expr = replace(eq.expr)
            sub_name = f"{name}_{i}"
            new_eqs[sub_name] = Equation(new_expr, name=sub_name, model=None)
    return new_eqs


def _apply_all_dict(equations: dict, op):
    """Broadcast an operation/substitution across every equation in
    a plain dict (used during per-layer scratch construction)."""
    for eq in equations.values():
        eq.apply(op, _no_history=True)
