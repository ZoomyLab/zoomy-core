"""MLVAM — Multi-Layer non-hydrostatic VAM.

Same per-layer factoring as :mod:`mlme` but using the VAM pipeline
(no pressure elimination, w + p kept as state, all three equations
Galerkin-projected).  Layer coupling at the bottom of each layer is
via ``bottom = b + Σ_{m<ℓ} h_m``.  Inter-layer pressure continuity
(Dirichlet condition on ``p_layer_ℓ(σ=1)``) is left as a post-build
constraint — the framework derives the equations; the user adds the
constraint if their solver needs it.
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


__all__ = ["MLVAM", "derive_layer_vam"]


def derive_layer_vam(*, layer_label: str, bottom_expr, N: int = 2):
    """Run the full VAM-L=N pipeline for one layer.  Returns
    ``(model, h_layer, q_fn)``."""
    state = StateSpace(dimension=2)
    src = MassMomentum(state)
    model = Model(f"VAM-L={N}{layer_label}")
    model.add_equation(f"continuity{layer_label}",  src.continuity.expr)
    model.add_equation(f"momentum_x{layer_label}",  src.momentum.x.expr)
    model.add_equation(f"momentum_z{layer_label}",  src.momentum.z.expr)

    model.apply({state.tau.xx: 0})

    model.apply(SigmaTransform(state))
    model.apply(KinematicBC(state, interface=state.b,   at=sp.S.Zero))
    model.apply(KinematicBC(state, interface=state.eta, at=sp.S.One))

    g, h, x, t = state.g, state.h, state.x, state.t

    model.multiply(state.h)
    for eq in model.equations.values():
        eq.simplify()

    model.equations[f"continuity{layer_label}"][[0]].apply(
        ProductRule(variables=[x]))
    model.equations[f"momentum_x{layer_label}"][[0, 1, 7]].apply(
        ProductRule(variables=[t, x]))
    model.equations[f"momentum_z{layer_label}"][[2, 3, 11]].apply(
        ProductRule(variables=[t, x]))
    for eq in model.equations.values():
        eq.simplify()

    sigma = state.zeta_ref
    u_fn = sp.Function(f"u{layer_label}", real=True)
    w_fn = sp.Function(f"w{layer_label}", real=True)
    p_fn = sp.Function(f"p{layer_label}", real=True)
    phi_u_fn = sp.Function("phi_u", real=True)
    phi_w_fn = sp.Function("phi_w", real=True)
    phi_p_fn = sp.Function("phi_p", real=True)
    psi_u    = sp.Function("psi_u", real=True)
    psi_w    = sp.Function("psi_w", real=True)
    omega    = sp.Function("omega", real=True)

    u_ansatz = sum(u_fn(k, t, x) * phi_u_fn(k, sigma) for k in range(N + 1))
    w_ansatz = sum(w_fn(k, t, x) * phi_w_fn(k, sigma) for k in range(N + 1))
    p_ansatz = sum(p_fn(k, t, x) * phi_p_fn(k, sigma) for k in range(N + 1))

    model.apply({state.u.xreplace({state.z: sigma}): u_ansatz,
                 state.w.xreplace({state.z: sigma}): w_ansatz,
                 state.p.xreplace({state.z: sigma}): p_ansatz})

    model.equations[f"continuity{layer_label}"].apply(
        Multiply(psi_u(sigma) * omega(sigma)))
    model.equations[f"momentum_x{layer_label}"].apply(
        Multiply(psi_u(sigma) * omega(sigma)))
    model.equations[f"momentum_z{layer_label}"].apply(
        Multiply(psi_w(sigma) * omega(sigma)))

    for name in (f"continuity{layer_label}",
                  f"momentum_x{layer_label}",
                  f"momentum_z{layer_label}"):
        model.equations[name].apply(
            Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
    for eq in model.equations.values():
        eq.simplify()

    # KBC modal closure (BEFORE resolve_dummy so the boundary phi_w
    # atoms produced by ``w_ansatz.subs(sigma, at_value)`` get
    # resolved together with the rest of the basis).
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
    for eq in model.equations.values():
        eq.simplify()

    # CoV u → q/h.
    q_fn = sp.Function(f"q{layer_label}", real=True)
    u_modes = [u_fn(k, t, x) for k in range(N + 1)]
    q_modes = [q_fn(k, t, x) for k in range(N + 1)]
    model.apply({u: q / state.h for u, q in zip(u_modes, q_modes)},
                level="minor", description=f"CoV u → q/h{layer_label}")
    for eq in model.equations.values():
        eq.simplify()

    # Resolve dummies + EvaluateIntegrals.
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
        phi_p_fn, lambda k, sig, _b=legendre_u: basis_value(_b, k, sig))
    model.resolve_dummy(
        psi_u, [legendre_value(legendre_u, k) for k in range(N + 1)])
    model.resolve_dummy(
        psi_w, [legendre_value(legendre_w, k) for k in range(N + 1)])

    model.apply(EvaluateIntegrals(state))
    for eq in model.equations.values():
        eq.simplify()

    # CoV w → r/h.
    r_fn = sp.Function(f"r{layer_label}", real=True)
    w_modes_all = [w_fn(k, t, x) for k in range(N + 1)]
    r_modes_all = [r_fn(k, t, x) for k in range(N + 1)]
    model.apply({w: r / state.h for w, r in zip(w_modes_all, r_modes_all)},
                level="minor", description=f"CoV w → r/h{layer_label}")
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

    # Mass-matrix inversion.
    for k in range(N + 1):
        for name in (f"momentum_x{layer_label}_{k}",
                      f"momentum_z{layer_label}_{k}"):
            if name in model.equations:
                eq = model.equations[name]
                eq.expr = (2 * k + 1) * eq.expr
                eq.simplify()

    # Layer-local substitution.
    h_layer = sp.Function(f"h{layer_label}", positive=True)(t, x)
    sub = {state.b: bottom_expr, state.h: h_layer}
    for eq in model.equations.values():
        eq.expr = eq.expr.xreplace(sub)
        eq.simplify()
    return model, h_layer, q_fn


class MLVAM:
    """Multi-Layer non-hydrostatic VAM stack."""

    def __init__(self, N_layers: int = 2, N: int = 2, *,
                 name: str | None = None):
        self.N_layers = N_layers
        self.N = N
        self.t = sp.Symbol("t", real=True)
        self.x = sp.Symbol("x", real=True)
        self.b = sp.Function("b", real=True)(self.t, self.x)
        self.model = Model(name or f"MLVAM-{N_layers}-layers-L={N}")
        self.layer_models = []
        self.h_layers = []
        self.q_fns = []
        self._build()

    def _build(self):
        h_pre = [sp.Function(f"h_layer_{l + 1}", positive=True)(self.t, self.x)
                  for l in range(self.N_layers)]
        for l in range(self.N_layers):
            bottom = self.b + sum(h_pre[:l], sp.S.Zero)
            sub_model, h_layer, q_fn = derive_layer_vam(
                layer_label=f"_layer_{l + 1}",
                bottom_expr=bottom,
                N=self.N,
            )
            self.layer_models.append(sub_model)
            self.h_layers.append(h_layer)
            self.q_fns.append(q_fn)

        unify = {h_pre[l]: self.h_layers[l] for l in range(self.N_layers)}
        for sub in self.layer_models:
            for eq in sub.equations.values():
                eq.expr = eq.expr.xreplace(unify)
                eq.simplify()

        for sub in self.layer_models:
            for name, eq in sub.equations.items():
                self.model.add_equation(name, eq.expr)
