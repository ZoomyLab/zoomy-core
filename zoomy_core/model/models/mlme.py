"""MLME — Multi-Layer Shallow Moment Equations.

Builds an ``N_layers``-stack ML-SME-L=N system by running the
:class:`SME` per-layer pipeline with layer-specific bottom topography
and top pressure, then merging the per-layer ``Model`` instances into
a single combined ``Model``.  See the transparent-derivation notebook
``thesis/notebooks/modeling/transparent_derivations/ml_sme_clean.ipynb``.

For ``N_layers = 2`` layers labelled 1 (bottom) and 2 (top):

* Layer 1 sees ``bottom = b``, ``top_pressure = ρ·g·h_layer_2``.
* Layer 2 sees ``bottom = b + h_layer_1``, ``top_pressure = 0``.

The standard inter-layer pressure coupling
(``g·h_layer_1·∂_x(h_layer_2)`` in layer-1 momentum_0,
``g·h_layer_2·∂_x(h_layer_1)`` in layer-2 momentum_0) then falls out
automatically.

This module re-implements the per-layer SME pipeline with the
``(bottom_expr, top_pressure_expr)`` parameters so layer-specific
state-space substitutions can be done at the end.  Each derived
sub-model gets a ``_layer_ℓ`` suffix on every equation, function and
state symbol.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.models.model import Model, Symmetrize
from zoomy_core.model.models.operations import (
    Expression,
    StateSpace,
    MassMomentum,
    Multiply,
    ProductRule,
    Integrate,
    EvaluateIntegrals,
    SigmaTransform,
    KinematicBC,
    Legendre_shifted,
)


__all__ = ["MLME", "derive_layer_sme"]


def derive_layer_sme(*, layer_label: str, bottom_expr, top_pressure_expr,
                      N: int = 2):
    """Run the full SME-L=N pipeline for one layer.  Returns
    ``(model, h_layer, q_fn)``."""
    state = StateSpace(dimension=2)
    src = MassMomentum(state)
    model = Model(f"SME-L={N}{layer_label}")
    model.add_equation(f"continuity{layer_label}",  src.continuity.expr)
    model.add_equation(f"momentum_x{layer_label}",  src.momentum.x.expr)
    model.add_equation(f"momentum_z{layer_label}",  src.momentum.z.expr)

    # Pre-σ: drop tau_xx + hydrostatic + p-elimination with the
    # layer-specific top pressure (NOT zero for inner layers).
    model.apply({state.tau.xx: 0})
    model.equations[f"momentum_z{layer_label}"].apply(
        {state.w: 0, state.tau.zx: 0,
         state.tau.zz: 0, state.tau.xz: 0}).simplify()
    model.equations[f"momentum_z{layer_label}"].apply(
        Integrate(state.z, state.z, state.eta, method="analytical"))
    model.equations[f"momentum_z{layer_label}"].apply(
        {state.p.subs(state.z, state.eta): top_pressure_expr}).simplify()
    p_subst = Expression(
        model.equations[f"momentum_z{layer_label}"].expr,
        f"momentum_z{layer_label}").solve_for(state.p)
    model.equations[f"momentum_x{layer_label}"].apply(p_subst).simplify()

    model.apply(SigmaTransform(state))
    model.apply(KinematicBC(state, interface=state.b,   at=sp.S.Zero))
    model.apply(KinematicBC(state, interface=state.eta, at=sp.S.One))

    g, h, x, t = state.g, state.h, state.x, state.t

    model.multiply(state.h)
    for eq in model.equations.values():
        eq.simplify()

    model.equations[f"momentum_x{layer_label}"][[0, 1]].apply(
        ProductRule(variables=[t, x]))
    model.equations[f"continuity{layer_label}"][[0]].apply(
        ProductRule(variables=[x]))
    for eq in model.equations.values():
        eq.simplify()

    # Modal ansatz.
    sigma = state.zeta_ref
    u_fn = sp.Function(f"u{layer_label}", real=True)
    w_fn = sp.Function(f"w{layer_label}", real=True)
    phi_u_fn = sp.Function("phi_u", real=True)
    phi_w_fn = sp.Function("phi_w", real=True)
    psi_u    = sp.Function("psi_u", real=True)
    psi_w    = sp.Function("psi_w", real=True)
    omega    = sp.Function("omega", real=True)

    u_ansatz = sum(u_fn(k, t, x) * phi_u_fn(k, sigma) for k in range(N + 1))
    w_ansatz = sum(w_fn(k, t, x) * phi_w_fn(k, sigma) for k in range(N + 2))

    model.apply({state.u.xreplace({state.z: sigma}): u_ansatz,
                 state.w.xreplace({state.z: sigma}): w_ansatz})

    model.equations[f"continuity{layer_label}"].apply(
        Multiply(psi_w(sigma) * omega(sigma)))
    model.equations[f"momentum_x{layer_label}"].apply(
        Multiply(psi_u(sigma) * omega(sigma)))

    model.equations[f"continuity{layer_label}"].apply(
        Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
    model.equations[f"momentum_x{layer_label}"].apply(
        Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
    for eq in model.equations.values():
        eq.simplify()

    # KBC closure for w_0, w_1.
    kbc_bot = KinematicBC(state, state.b,   at=sp.S.Zero)
    kbc_top = KinematicBC(state, state.eta, at=sp.S.One)

    def residual(kbc, at_value):
        (lhs, rhs), = kbc.subs_map.items()
        return (lhs - rhs).xreplace({
            state.w.subs(state.z, at_value): w_ansatz.subs(sigma, at_value),
            state.u.subs(state.z, at_value): u_ansatz.subs(sigma, at_value),
        })

    closure = sp.solve(
        [residual(kbc_bot, sp.S.Zero),
         residual(kbc_top, sp.S.One)],
        [w_fn(0, t, x), w_fn(1, t, x)],
        dict=True,
    )[0]
    model.apply(closure, level="minor",
                 description=f"KBC modal closure{layer_label}")

    # CoV u → q/h.
    q_fn = sp.Function(f"q{layer_label}", real=True)
    u_modes = [u_fn(k, t, x) for k in range(N + 1)]
    q_modes = [q_fn(k, t, x) for k in range(N + 1)]
    model.apply({u: q / state.h for u, q in zip(u_modes, q_modes)},
                level="minor", description=f"CoV u → q/h{layer_label}")
    for eq in model.equations.values():
        eq.simplify()

    # Resolve dummies.
    legendre_u = Legendre_shifted(level=N)
    legendre_w = Legendre_shifted(level=N + 1)

    def basis_value(b, k_arg, sigma_arg):
        if k_arg.is_Integer:
            return b.eval(int(k_arg), sigma_arg)
        return sp.Function("phi_unresolved")(k_arg, sigma_arg)

    def legendre_value(b, k):
        return lambda arg, _b=b, _k=k: _b.eval(_k, arg)

    model.resolve_dummy(omega, lambda arg: sp.S.One)
    model.resolve_dummy(
        phi_u_fn, lambda k, sig, _b=legendre_u: basis_value(_b, k, sig))
    model.resolve_dummy(
        phi_w_fn, lambda k, sig, _b=legendre_w: basis_value(_b, k, sig))
    model.resolve_dummy(
        psi_u, [legendre_value(legendre_u, k) for k in range(N + 1)])
    model.resolve_dummy(
        psi_w, [legendre_value(legendre_w, k) for k in range(N + 2)])

    model.apply(EvaluateIntegrals(state))
    for eq in model.equations.values():
        eq.simplify()

    # Gravity self-pair fold.
    mom0 = model.equations[f"momentum_x{layer_label}_0"]
    grav = next(
        (T for T in mom0
         if T.expr.has(g) and T.expr.has(sp.Derivative(h, x))
         and not T.expr.has(state.b)),
        None)
    if grav is not None:
        grav.apply(Symmetrize(ProductRule(variables=[x])))
        mom0.simplify()

    # Higher-mode w closure.
    w_higher = [w_fn(k, t, x) for k in range(2, N + 2)]
    residuals = [
        model.equations[f"continuity{layer_label}_{k}"].expr
        for k in range(1, N + 1)
    ]
    w_solution = sp.solve(residuals, w_higher, dict=True)[0]
    model.apply(w_solution, level="minor",
                 description=f"continuity_k → w_(k+1){layer_label}")
    for eq in model.equations.values():
        eq.simplify()

    # Mass-matrix inversion.
    for k in range(N + 1):
        eq = model.equations[f"momentum_x{layer_label}_{k}"]
        eq.expr = (2 * k + 1) * eq.expr
        eq.simplify()

    # Layer-local substitution: state.b → bottom_expr, state.h → h_layer.
    h_layer = sp.Function(f"h{layer_label}", positive=True)(t, x)
    sub = {state.b: bottom_expr, state.h: h_layer}
    for eq in model.equations.values():
        eq.expr = eq.expr.xreplace(sub)
        eq.simplify()
    return model, h_layer, q_fn


class MLME:
    """Multi-Layer SME stack."""

    def __init__(self, N_layers: int = 2, N: int = 2, *,
                 name: str | None = None):
        self.N_layers = N_layers
        self.N = N
        self.t = sp.Symbol("t", real=True)
        self.x = sp.Symbol("x", real=True)
        self.b = sp.Function("b", real=True)(self.t, self.x)
        self.g_sym   = sp.Symbol("g",   positive=True)
        self.rho_sym = sp.Symbol("rho", positive=True)
        self.model = Model(name or f"MLME-{N_layers}-layers-L={N}")
        self.layer_models = []
        self.h_layers = []
        self.q_fns = []
        self._build()

    def _build(self):
        # Pre-declared layer h placeholders.
        h_pre = [sp.Function(f"h_layer_{l + 1}", positive=True)(self.t, self.x)
                  for l in range(self.N_layers)]

        for l in range(self.N_layers):
            bottom = self.b + sum(h_pre[:l], sp.S.Zero)
            top_p = self.rho_sym * self.g_sym * sum(
                h_pre[l + 1:], sp.S.Zero)
            sub_model, h_layer, q_fn = derive_layer_sme(
                layer_label=f"_layer_{l + 1}",
                bottom_expr=bottom,
                top_pressure_expr=top_p,
                N=self.N,
            )
            self.layer_models.append(sub_model)
            self.h_layers.append(h_layer)
            self.q_fns.append(q_fn)

        # Unify the placeholder ``h_layer_ℓ`` with the actual returned
        # ``h_layer`` Functions (they share the name but may be
        # distinct sympy objects).
        unify = {h_pre[l]: self.h_layers[l] for l in range(self.N_layers)}
        for sub in self.layer_models:
            for eq in sub.equations.values():
                eq.expr = eq.expr.xreplace(unify)
                eq.simplify()

        # Merge per-layer models into one.
        for sub in self.layer_models:
            for name, eq in sub.equations.items():
                self.model.add_equation(name, eq.expr)
