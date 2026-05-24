"""MLVAM — Multi-Layer Vertically-Averaged Moment system (Audusse-style).

Same Audusse-Bristeau-Perthame-Sainte-Marie multi-layer construction
as :class:`MLSME`, but applied to the non-hydrostatic VAM (which
keeps ``w`` and ``p`` as state variables instead of eliminating
``p`` via hydrostatic reduction).

State vector (size ``1 + L · 3·(N+1)``):

    Q = [H,
         q_1_0..q_1_N,  r_1_0..r_1_N,  p_1_0..p_1_N,
         q_2_0..q_2_N,  r_2_0..r_2_N,  p_2_0..p_2_N,
         ...,
         q_L_0..q_L_N,  r_L_0..r_L_N,  p_L_0..p_L_N]ᵀ

with ``q_ℓ_k = h_ℓ · u_ℓ_k``, ``r_ℓ_k = h_ℓ · w_ℓ_k``, and
``h_ℓ = α_ℓ · H``.

Equations:

* one global continuity ``∂_t H + ∂_x Q = 0`` (sum of q_ℓ_0);
* per layer ℓ: ``N+1`` x-momentum equations + ``N+1`` z-momentum
  equations.  The pressure modes ``p_ℓ_k`` are state variables but
  not dynamic at this stage (no ∂_t p equation — they're set by an
  external closure such as a Poisson solver downstream).

Mass-exchange transfer terms are added to BOTH q-mode (x-momentum)
and r-mode (z-momentum) equations:

    momentum_x_layer_ℓ_k  +=  − u*_{ℓ+1/2}·G_{ℓ+1/2}
                              + (−1)^k·u*_{ℓ-1/2}·G_{ℓ-1/2}
    momentum_z_layer_ℓ_k  +=  − w*_{ℓ+1/2}·G_{ℓ+1/2}
                              + (−1)^k·w*_{ℓ-1/2}·G_{ℓ-1/2}

with the upwind interface velocities

    u*_{ℓ+1/2} = Piecewise( (u_ℓ(σ=1),    G_{ℓ+1/2} > 0),
                            (u_{ℓ+1}(σ=0), True             ) )
    w*_{ℓ+1/2} = Piecewise( (w_ℓ(σ=1),    G_{ℓ+1/2} > 0),
                            (w_{ℓ+1}(σ=0), True             ) )

and interface mass flux (closed form under fixed-α):

    G_{ℓ+1/2}  =  (Σ_{m≤ℓ} α_m) · ∂_x Q  −  Σ_{m≤ℓ} ∂_x q_m_0
    G_{1/2}    =  0    (impermeable bottom)
    G_{L+1/2}  =  0    (impermeable free surface — by Σα = 1)

Pressure-continuity across internal interfaces is NOT imposed at
this level — it's a closure choice for the downstream solver.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.basemodel import Model
from zoomy_core.model.state import StateSpace, MassMomentum
from zoomy_core.model.equation import Equation
from zoomy_core.model.operations import (
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


__all__ = ["MLVAM"]


class MLVAM(Model):
    """Multi-layer VAM at level ``N`` over ``N_layers`` layers
    (Audusse-style mass-exchange, non-hydrostatic per layer).

    Parameters
    ----------
    N_layers
        Number of layers.
    N
        Per-layer Galerkin level (number of velocity / pressure modes
        is ``N+1`` for each of u, w, p).
    alphas
        Layer fractions ``(α_1, …, α_{N_layers})``.  Default: uniform
        ``1/N_layers``.  Must sum to 1.
    """

    _finalize_lazy = True

    def __init__(self, N_layers: int = 2, N: int = 1,
                 *, alphas=None, **kwargs):
        self.N_layers = N_layers
        self.N = N

        # Layer fractions (same handling as MLSME).
        if alphas is None:
            alphas = [sp.Rational(1, N_layers)] * N_layers
        else:
            alphas = [sp.sympify(a) for a in alphas]
            if len(alphas) != N_layers:
                raise ValueError(
                    f"alphas must have length N_layers={N_layers}, "
                    f"got {len(alphas)}")
            total = sum(alphas)
            try:
                if abs(float(total) - 1.0) > 1e-12:
                    raise ValueError(
                        f"alphas must sum to 1, got {float(total)}")
            except (TypeError, ValueError):
                pass
        self._alphas = alphas

        # Variable layout: [H, (q, r, p)_ℓ_k for ℓ = 1..N_layers].
        var_names = ["H"]
        for ell in range(1, N_layers + 1):
            for k in range(N + 1):
                var_names.append(f"q_layer_{ell}_{k}")
            for k in range(N + 1):
                var_names.append(f"r_layer_{ell}_{k}")
            for k in range(N + 1):
                var_names.append(f"p_layer_{ell}_{k}")

        kwargs.setdefault("name",
                           f"MLVAM-L={N}-{N_layers}layers-Audusse")
        kwargs.setdefault("variables", var_names)
        kwargs.setdefault("parameters", {
            "g":   (9.81, "positive"),
            "rho": (1.0,  "positive"),
        })
        kwargs.setdefault("eigenvalue_mode", "numerical")
        super().__init__(**kwargs)

    # ── Derivation hook ─────────────────────────────────────────────
    def derive_model(self):
        missing = [k for k in ("g", "rho")
                   if not hasattr(self.parameters, k)]
        if missing:
            raise ValueError(
                f"symbol {missing[0]!r} required by MLVAM but not in "
                f"parameters={{...}}; supply via "
                f"`MLVAM(..., parameters={{'g': 9.81, 'rho': 1.0, ...}})`."
            )

        self.state = StateSpace(dimension=2)
        t_sym, x_sym = self.state.t, self.state.x

        H_pre = sp.Function("H_pre", positive=True)(t_sym, x_sym)
        b_global = sp.Function("b", real=True)(t_sym, x_sym)

        alphas = self._alphas

        # ── Per-layer VAM pipelines ────────────────────────────────
        per_layer = []
        layer_q_fns = []
        layer_r_fns = []
        layer_p_fns = []
        for ell in range(1, self.N_layers + 1):
            cumul_below = sum(alphas[:ell - 1], sp.S.Zero)
            h_layer_expr = alphas[ell - 1] * H_pre
            bottom_expr = b_global + cumul_below * H_pre

            sub_model, q_fn, r_fn, p_fn = self._derive_layer_vam(
                layer_label=f"_layer_{ell}",
                h_layer_expr=h_layer_expr,
                bottom_expr=bottom_expr,
            )
            per_layer.append(sub_model)
            layer_q_fns.append(q_fn)
            layer_r_fns.append(r_fn)
            layer_p_fns.append(p_fn)

        # ── Merge per-layer momentum equations into self ───────────
        for ell, sub in zip(range(1, self.N_layers + 1), per_layer):
            label = f"_layer_{ell}"
            for k in range(self.N + 1):
                for slot in (f"momentum_x{label}_{k}",
                             f"momentum_z{label}_{k}"):
                    if slot in sub:
                        self.add_equation(slot, sub[slot].expr)

        # ── Global continuity ──────────────────────────────────────
        Q_total = sum(q_fn(0, t_sym, x_sym) for q_fn in layer_q_fns)
        self.add_equation(
            "continuity_global",
            sp.Derivative(H_pre, t_sym) + sp.Derivative(Q_total, x_sym),
        )

        # ── Interface mass fluxes G_{ℓ+1/2} ────────────────────────
        G_interfaces = [sp.S.Zero]
        for ell in range(1, self.N_layers):
            cumul_alpha = sum(alphas[:ell], sp.S.Zero)
            cumul_q = sum(layer_q_fns[m - 1](0, t_sym, x_sym)
                          for m in range(1, ell + 1))
            G_interfaces.append(
                cumul_alpha * sp.Derivative(Q_total, x_sym)
                - sp.Derivative(cumul_q, x_sym)
            )
        G_interfaces.append(sp.S.Zero)

        # ── Interface velocities u*_{ℓ+1/2} and w*_{ℓ+1/2} ─────────
        def u_top(ell):
            q_fn = layer_q_fns[ell - 1]
            return sum(q_fn(k, t_sym, x_sym)
                       for k in range(self.N + 1)) / (alphas[ell - 1] * H_pre)

        def u_bot(ell):
            q_fn = layer_q_fns[ell - 1]
            return sum((-1)**k * q_fn(k, t_sym, x_sym)
                       for k in range(self.N + 1)) / (alphas[ell - 1] * H_pre)

        def w_top(ell):
            r_fn = layer_r_fns[ell - 1]
            return sum(r_fn(k, t_sym, x_sym)
                       for k in range(self.N + 1)) / (alphas[ell - 1] * H_pre)

        def w_bot(ell):
            r_fn = layer_r_fns[ell - 1]
            return sum((-1)**k * r_fn(k, t_sym, x_sym)
                       for k in range(self.N + 1)) / (alphas[ell - 1] * H_pre)

        u_interfaces = [sp.S.Zero]
        w_interfaces = [sp.S.Zero]
        for ell in range(1, self.N_layers):
            G = G_interfaces[ell]
            u_interfaces.append(sp.Piecewise(
                (u_top(ell), G > 0),
                (u_bot(ell + 1), True),
            ))
            w_interfaces.append(sp.Piecewise(
                (w_top(ell), G > 0),
                (w_bot(ell + 1), True),
            ))
        u_interfaces.append(sp.S.Zero)
        w_interfaces.append(sp.S.Zero)

        # ── Mass-exchange transfer terms (x and z momentum) ───────
        for ell in range(1, self.N_layers + 1):
            G_top_v = G_interfaces[ell]
            G_bot_v = G_interfaces[ell - 1]
            u_top_v = u_interfaces[ell]
            u_bot_v = u_interfaces[ell - 1]
            w_top_v = w_interfaces[ell]
            w_bot_v = w_interfaces[ell - 1]
            for k in range(self.N + 1):
                xname = f"momentum_x_layer_{ell}_{k}"
                if xname in self._equations:
                    transfer_x = (-u_top_v * G_top_v
                                  + (-1)**k * u_bot_v * G_bot_v)
                    eq = self._equations[xname]
                    eq.expr = eq.expr + transfer_x
                    eq.simplify()
                zname = f"momentum_z_layer_{ell}_{k}"
                if zname in self._equations:
                    transfer_z = (-w_top_v * G_top_v
                                  + (-1)**k * w_bot_v * G_bot_v)
                    eq = self._equations[zname]
                    eq.expr = eq.expr + transfer_z
                    eq.simplify()

        # Stash bookkeeping.
        self._H_pre = H_pre
        self._b_global = b_global
        self._layer_q_fns = layer_q_fns
        self._layer_r_fns = layer_r_fns
        self._layer_p_fns = layer_p_fns
        self._G_interfaces = G_interfaces
        self._u_interfaces = u_interfaces
        self._w_interfaces = w_interfaces

    # ── Per-layer VAM pipeline ─────────────────────────────────────
    def _derive_layer_vam(self, *, layer_label, h_layer_expr,
                           bottom_expr):
        """Inline port of the single-layer VAM pipeline, parameterised
        for one layer of the Audusse stack.  Runs the GENERIC VAM
        derivation; at the end substitutes the layer-specific geometry
        (``state.b → bottom_expr``, ``state.h → h_layer_expr``).
        """
        N = self.N
        state = self.state
        src = MassMomentum(state, self.parameters)
        s, t, x = state, state.t, state.x

        equations: dict[str, Equation] = {}

        def _new_eq(name, expression):
            equations[name] = Equation(expression, name=name, model=None)

        def _apply_all(op):
            for eq in equations.values():
                eq.apply(op, _no_history=True)

        # ── Setup: 3 INS equations (continuity, mom_x, mom_z) ─────
        _new_eq(f"continuity{layer_label}",  src.continuity.expr)
        _new_eq(f"momentum_x{layer_label}",  src.momentum.x.expr)
        _new_eq(f"momentum_z{layer_label}",  src.momentum.z.expr)
        _apply_all({src.tau.xx: 0})
        # VAM keeps p and w as state — no hydrostatic reduction.

        # ── σ-transform + KBCs ────────────────────────────────────
        _apply_all(SigmaTransform(s, src))
        _apply_all(KinematicBC(s, src.b,   src, at=sp.S.Zero))
        _apply_all(KinematicBC(s, src.eta, src, at=sp.S.One))

        # ── × h, conservative folds (apply broadly across all terms;
        #     for single-layer VAM the term indices were hand-picked
        #     but a full ProductRule is also correct here) ─────────
        _apply_all(Multiply(src.h))
        for eq in equations.values():
            eq.simplify()
        equations[f"continuity{layer_label}"].apply(
            ProductRule(variables=[x]))
        equations[f"momentum_x{layer_label}"].apply(
            ProductRule(variables=[t, x]))
        equations[f"momentum_z{layer_label}"].apply(
            ProductRule(variables=[t, x]))
        for eq in equations.values():
            eq.simplify()

        # ── Modal ansatz (3 fields) ──────────────────────────────
        sigma = s.zeta_ref
        u_fn = sp.Function(f"u{layer_label}", real=True)
        w_fn = sp.Function(f"w{layer_label}", real=True)
        p_fn = sp.Function(f"p{layer_label}", real=True)
        phi_u_fn = sp.Function("phi_u", real=True)
        phi_w_fn = sp.Function("phi_w", real=True)
        phi_p_fn = sp.Function("phi_p", real=True)
        psi_u    = sp.Function("psi_u", real=True)
        psi_w    = sp.Function("psi_w", real=True)
        omega    = sp.Function("omega", real=True)

        u_ansatz = sum(u_fn(k, t, x) * phi_u_fn(k, sigma)
                       for k in range(N + 1))
        w_ansatz = sum(w_fn(k, t, x) * phi_w_fn(k, sigma)
                       for k in range(N + 1))
        p_ansatz = sum(p_fn(k, t, x) * phi_p_fn(k, sigma)
                       for k in range(N + 1))
        _apply_all({
            src.u.xreplace({s.z: sigma}): u_ansatz,
            src.w.xreplace({s.z: sigma}): w_ansatz,
            src.p.xreplace({s.z: sigma}): p_ansatz,
        })

        # ── Galerkin (cont + mom_x → psi_u; mom_z → psi_w) ───────
        equations[f"continuity{layer_label}"].apply(
            Multiply(psi_u(sigma) * omega(sigma)))
        equations[f"momentum_x{layer_label}"].apply(
            Multiply(psi_u(sigma) * omega(sigma)))
        equations[f"momentum_z{layer_label}"].apply(
            Multiply(psi_w(sigma) * omega(sigma)))
        for slot in (f"continuity{layer_label}",
                     f"momentum_x{layer_label}",
                     f"momentum_z{layer_label}"):
            equations[slot].apply(
                Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
        for eq in equations.values():
            eq.simplify()

        # ── KBC modal closure (outer bottom + free surface) ──────
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

        # ── Resolve omega + branch on psi_u / psi_w ──────────────
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
            phi_u_fn,
            lambda k, sg, _b=legendre_u: basis_value(_b, k, sg)))
        _apply_all(ResolveDummy(
            phi_w_fn,
            lambda k, sg, _b=legendre_w: basis_value(_b, k, sg)))
        _apply_all(ResolveDummy(
            phi_p_fn,
            lambda k, sg, _b=legendre_u: basis_value(_b, k, sg)))

        equations = _resolve_dummy_list(
            equations, psi_u,
            [legendre_value(legendre_u, k) for k in range(N + 1)])
        equations = _resolve_dummy_list(
            equations, psi_w,
            [legendre_value(legendre_w, k) for k in range(N + 1)])

        # ── Evaluate integrals ───────────────────────────────────
        _apply_all_dict(equations, EvaluateIntegrals(state))
        for eq in equations.values():
            eq.simplify()

        # ── CoV w → r/h (z-momentum conservative form) ───────────
        r_fn = sp.Function(f"r{layer_label}", real=True)
        w_modes_all = [w_fn(k, t, x) for k in range(N + 1)]
        r_modes_all = [r_fn(k, t, x) for k in range(N + 1)]
        _apply_all_dict(equations,
                        {w: r / src.h for w, r in zip(w_modes_all, r_modes_all)})
        for eq in equations.values():
            eq.simplify()

        # ── Gravity self-pair fold (K&T flux folding) ───────────
        g = self.parameters.g
        mom0_name = f"momentum_x{layer_label}_0"
        if mom0_name in equations:
            mom0 = equations[mom0_name]
            grav = next(
                (term for term in mom0
                 if term.expr.has(g)
                 and term.expr.has(sp.Derivative(src.h, s.x))
                 and not term.expr.has(src.b)),
                None)
            if grav is not None:
                grav.apply(Symmetrize(ProductRule(variables=[s.x])))
                mom0.simplify()

        # ── Mass-matrix inversion (Legendre orthogonality) ───────
        for k in range(N + 1):
            for prefix in ("momentum_x", "momentum_z"):
                eq_name = f"{prefix}{layer_label}_{k}"
                if eq_name in equations:
                    equations[eq_name].expr = (
                        (2 * k + 1) * equations[eq_name].expr
                    )
                    equations[eq_name].simplify()

        # ── Eliminate ∂_t h via depth-mean continuity ────────────
        cont0_name = f"continuity{layer_label}_0"
        if cont0_name in equations:
            cont0_subst = {
                sp.Derivative(src.h, s.t):
                    -sp.Derivative(q_fn(0, s.t, s.x), s.x)
            }
            for name, eq in equations.items():
                if name == cont0_name:
                    continue
                eq.expr = eq.expr.xreplace(cont0_subst)
                eq.simplify()

        # ── Layer-local substitution (Audusse geometry) ──────────
        sub = {src.b: bottom_expr, src.h: h_layer_expr}
        for eq in equations.values():
            eq.expr = eq.expr.xreplace(sub)
            eq.simplify()
        return equations, q_fn, r_fn, p_fn

    # ── Lazy finalization hook ────────────────────────────────────
    def _prepare_for_systemmodel(self):
        """Substitute the H_pre placeholder + per-layer q/r/p Functions
        with their Model Symbol equivalents and set ``_variable_map``."""
        t_sym, x_sym = self.state.t, self.state.x
        subs = {self._H_pre: self.variables.H}
        for ell_idx in range(self.N_layers):
            ell = ell_idx + 1
            q_fn = self._layer_q_fns[ell_idx]
            r_fn = self._layer_r_fns[ell_idx]
            p_fn = self._layer_p_fns[ell_idx]
            for k in range(self.N + 1):
                subs[q_fn(k, t_sym, x_sym)] = self.variables[f"q_layer_{ell}_{k}"]
                subs[r_fn(k, t_sym, x_sym)] = self.variables[f"r_layer_{ell}_{k}"]
                subs[p_fn(k, t_sym, x_sym)] = self.variables[f"p_layer_{ell}_{k}"]
        for eq in self:
            eq.expr = eq.expr.xreplace(subs)
        self._variable_map = self._build_variable_map()

    def _build_variable_map(self):
        """Layout: ``[H, (q_ℓ_k, r_ℓ_k, p_ℓ_k) for ℓ=1..L, k=0..N]``.

        Continuity_global → row 0 (H).
        momentum_x_layer_ℓ_k → row ``1 + (ℓ-1)·3(N+1) + k``.
        momentum_z_layer_ℓ_k → row ``1 + (ℓ-1)·3(N+1) + (N+1) + k``.

        Pressure rows have no dynamic equation at this stage and are
        skipped — they're auxiliary closures handled by the downstream
        solver.
        """
        N = self.N
        m = {"continuity_global": [0]}
        per_layer_block = 3 * (N + 1)
        for ell in range(1, self.N_layers + 1):
            base = 1 + (ell - 1) * per_layer_block
            for k in range(N + 1):
                m[f"momentum_x_layer_{ell}_{k}"] = [base + k]
            for k in range(N + 1):
                m[f"momentum_z_layer_{ell}_{k}"] = [base + (N + 1) + k]
        return m


# ── Helpers (shared shape with mlsme.py) ──────────────────────────

def _resolve_dummy_list(equations: dict, dummy, value_list):
    """ResolveDummy(value=list) branches each equation containing the
    dummy into one sub-equation per list element.  Plain-dict
    equivalent of ResolveDummy._apply_branched."""
    new_eqs = {}
    for name, eq in equations.items():
        if not eq.expr.has(dummy):
            new_eqs[name] = eq
            continue
        for i, v in enumerate(value_list):
            if callable(v) and not isinstance(v, sp.Basic):
                new_expr = eq.expr.replace(dummy, v)
            else:
                new_expr = eq.expr.replace(dummy, lambda *args, _v=v: _v)
            sub_name = f"{name}_{i}"
            new_eqs[sub_name] = Equation(new_expr, name=sub_name, model=None)
    return new_eqs


def _apply_all_dict(equations: dict, op):
    """Broadcast an operation across every equation in a plain dict."""
    for eq in equations.values():
        eq.apply(op, _no_history=True)
