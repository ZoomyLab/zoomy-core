"""MLSME — Multi-Layer Shallow Moment Equations (Audusse-style).

Derivation in the Audusse-Bristeau-Perthame-Sainte-Marie sense:
a SINGLE coupled derivation from the 3D Euler/INS system,
partitioning the water column ``z ∈ [b, b+H]`` into ``N_layers``
*geometric* layers (NOT material columns).  Layer thicknesses are
fixed fractions of the total depth, ``h_ℓ = α_ℓ · H`` with ``α_ℓ``
constant and ``Σ_ℓ α_ℓ = 1``.

Variables (state vector size ``1 + L·(N+1)``):

    Q = [H,  q_1_0..q_1_N,  q_2_0..q_2_N,  ...,  q_L_0..q_L_N]ᵀ

Equations (``1 + L·(N+1)`` rows):

* one global continuity ``∂_t H + ∂_x Q = 0`` where ``Q = Σ q_m_0``
  (the L per-layer continuities reduce to the same global statement
  under the fixed-α closure);
* L·(N+1) momentum equations — each is the standard single-layer SME
  pipeline run with the layer-local geometry
  ``h_ℓ = α_ℓ·H``, ``bottom_ℓ = b + (Σ_{m<ℓ} α_m)·H``,
  ``top-pressure_ℓ = ρ·g·(Σ_{m>ℓ} α_m)·H``, PLUS the mass-exchange
  transfer terms:

      + u*_{ℓ+1/2} · G_{ℓ+1/2}  −  (−1)^k · u*_{ℓ-1/2} · G_{ℓ-1/2}

  carrying momentum across interfaces along with the inter-layer
  mass flux.

Interface mass flux (closed form under fixed α):

    G_{ℓ+1/2}  =  (Σ_{m≤ℓ} α_m) · ∂_x Q  −  Σ_{m≤ℓ} ∂_x q_m_0
    G_{1/2}    =  0      (impermeable bottom)
    G_{L+1/2}  =  0      (impermeable free surface — automatic since Σα = 1)

Interface velocity (upwind via Piecewise):

    u*_{ℓ+1/2}  =  Piecewise( (u_ℓ(σ=1),    G_{ℓ+1/2} > 0),
                              (u_{ℓ+1}(σ=0), True             ) )

with shifted-Legendre evaluations
``u_ℓ(σ=1) = Σ_k q_ℓ_k / h_ℓ`` and
``u_ℓ(σ=0) = Σ_k (−1)^k · q_ℓ_k / h_ℓ``.

In the limit ``N = 0`` (constant velocity profile per layer),
``MLSME(N_layers=L, N=0)`` recovers Audusse's ML-SWE
(``F = q²/h, P = α·g·H²/2``, plus the upwinded mass-exchange
transfers).
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
    """Multi-layer SME at level ``N`` over ``N_layers`` layers
    (Audusse-style mass-exchange).

    Parameters
    ----------
    N_layers
        Number of layers.
    N
        Per-layer Galerkin level (number of velocity modes is ``N+1``).
    alphas
        Layer fractions ``(α_1, …, α_{N_layers})``.  Default: uniform
        ``1/N_layers``.  Must sum to 1.
    """

    # Equations leave ``derive_model`` in Function form so the
    # ``H_pre`` and ``q_ℓ`` placeholders can be xreplace'd to Model
    # Symbols in ``_prepare_for_systemmodel``; SystemModel.from_model
    # auto-triggers that pass.
    _finalize_lazy = True

    def __init__(self, N_layers: int = 2, N: int = 2,
                 *, alphas=None, **kwargs):
        self.N_layers = N_layers
        self.N = N

        # Layer fractions — default uniform.
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
                # Symbolic alphas — trust the user.
                pass
        self._alphas = alphas

        # Variable layout: [b, H, q_ℓ_0..N for ℓ = 1..N_layers].
        # Bathymetry b is a state variable with ``∂_t b = 0``
        # (the ``bottom`` equation added in derive_model).
        var_names = ["b", "H"]
        for ell in range(1, N_layers + 1):
            for k in range(N + 1):
                var_names.append(f"q_layer_{ell}_{k}")

        kwargs.setdefault("name",
                           f"MLSME-L={N}-{N_layers}layers-Audusse")
        kwargs.setdefault("variables", var_names)
        kwargs.setdefault("parameters", {
            "g":   (9.81, "positive"),
            "rho": (1.0,  "positive"),
        })
        kwargs.setdefault("eigenvalue_mode", "numerical")
        super().__init__(**kwargs)

    # ── Derivation hook ─────────────────────────────────────────────
    def derive_model(self):
        """Drive the per-layer SME pipeline, add the global continuity
        equation, then splice in the mass-exchange transfer terms."""
        # Strict required-parameter check.
        missing = [k for k in ("g", "rho")
                   if not hasattr(self.parameters, k)]
        if missing:
            raise ValueError(
                f"symbol {missing[0]!r} required by MLSME but not in "
                f"parameters={{...}}; supply via "
                f"`MLSME(..., parameters={{'g': 9.81, 'rho': 1.0, ...}})`."
            )

        # Shared coordinate scaffold across all layers (so MLSME.state
        # exposes zeta_ref for the parameter-values assertion).
        self.state = StateSpace(dimension=2)
        t_sym, x_sym = self.state.t, self.state.x

        # Total water depth as a Function placeholder — replaced with
        # self.variables.H in ``_prepare_for_systemmodel``.
        H_pre = sp.Function("H_pre", positive=True)(t_sym, x_sym)
        b_global = sp.Function("b", real=True)(t_sym, x_sym)

        alphas = self._alphas
        rho_sym = self.parameters.rho
        g_sym = self.parameters.g

        # ── Per-layer SME pipelines ────────────────────────────────
        per_layer = []
        layer_q_fns = []
        for ell in range(1, self.N_layers + 1):
            cumul_below = sum(alphas[:ell - 1], sp.S.Zero)
            cumul_above = sum(alphas[ell:], sp.S.Zero)
            # Layer-ℓ geometry — all expressed in terms of H_pre.
            h_layer_expr = alphas[ell - 1] * H_pre
            bottom_expr = b_global + cumul_below * H_pre
            # Top pressure = weight of layers above (= 0 for surface).
            if ell < self.N_layers:
                top_p_expr = rho_sym * g_sym * cumul_above * H_pre
            else:
                top_p_expr = sp.S.Zero

            sub_model, q_fn = self._derive_layer_sme(
                layer_label=f"_layer_{ell}",
                h_layer_expr=h_layer_expr,
                bottom_expr=bottom_expr,
                top_pressure_expr=top_p_expr,
            )
            per_layer.append(sub_model)
            layer_q_fns.append(q_fn)

        # ── Merge per-layer momentum equations into self ───────────
        # Drop per-layer continuities — replaced with the single
        # global continuity below (under fixed-α, the L per-layer
        # depth-mean continuities reduce to the same statement).
        for ell, sub in zip(range(1, self.N_layers + 1), per_layer):
            label = f"_layer_{ell}"
            for k in range(self.N + 1):
                name = f"momentum_x{label}_{k}"
                if name in sub:
                    self.add_equation(name, sub[name].expr)

        # ── Bottom equation ∂_t b = 0 ─────────────────────────────
        # b_global is a Function(b)(t, x); will be substituted to
        # self.variables.b in _prepare_for_systemmodel.
        self.add_equation("bottom", sp.Derivative(b_global, t_sym))

        # ── Global continuity:  ∂_t H + ∂_x Q = 0 ──────────────────
        Q_total = sum(q_fn(0, t_sym, x_sym) for q_fn in layer_q_fns)
        self.add_equation(
            "continuity_global",
            sp.Derivative(H_pre, t_sym) + sp.Derivative(Q_total, x_sym),
        )

        # ── Interface mass fluxes G_{ℓ+1/2} ────────────────────────
        # Closed form under fixed-α:
        #   G_{ℓ+1/2} = (Σ_{m≤ℓ} α_m)·∂_x Q − Σ_{m≤ℓ} ∂_x q_m_0
        # Index convention: G[ℓ] = G_{ℓ+1/2} for ℓ = 0..N_layers,
        # so G[0] = G_{1/2} = 0 (bottom) and G[N_layers] = 0 (top).
        G_interfaces = [sp.S.Zero]   # G[0] = bottom
        for ell in range(1, self.N_layers):
            cumul_alpha = sum(alphas[:ell], sp.S.Zero)
            cumul_q = sum(layer_q_fns[m - 1](0, t_sym, x_sym)
                          for m in range(1, ell + 1))
            G_interfaces.append(
                cumul_alpha * sp.Derivative(Q_total, x_sym)
                - sp.Derivative(cumul_q, x_sym)
            )
        G_interfaces.append(sp.S.Zero)  # G[N_layers] = free surface

        # ── Interface velocities u*_{ℓ+1/2} (upwind via Piecewise) ──
        # u_ℓ(σ=1) = Σ_k q_ℓ_k / h_ℓ            (Legendre φ_k(1) = 1)
        # u_ℓ(σ=0) = Σ_k (−1)^k · q_ℓ_k / h_ℓ   (Legendre φ_k(0) = (−1)^k)
        def u_top_of_layer(ell):
            q_fn = layer_q_fns[ell - 1]
            return sum(q_fn(k, t_sym, x_sym)
                       for k in range(self.N + 1)) / (alphas[ell - 1] * H_pre)

        def u_bot_of_layer(ell):
            q_fn = layer_q_fns[ell - 1]
            return sum((-1)**k * q_fn(k, t_sym, x_sym)
                       for k in range(self.N + 1)) / (alphas[ell - 1] * H_pre)

        u_interfaces = [sp.S.Zero]   # bottom: no transfer (G=0)
        for ell in range(1, self.N_layers):
            G = G_interfaces[ell]
            u_interfaces.append(sp.Piecewise(
                (u_top_of_layer(ell), G > 0),
                (u_bot_of_layer(ell + 1), True),
            ))
        u_interfaces.append(sp.S.Zero)  # surface: no transfer (G=0)

        # ── Mass-exchange transfer terms in each momentum equation ─
        # The σ-frame Galerkin boundary terms produce, with the SHARED
        # upwind interface velocity:
        #
        #   RHS contribution to mom_x_ℓ_k:
        #       + u*_{ℓ+1/2} · G_{ℓ+1/2}  −  (−1)^k · u*_{ℓ-1/2} · G_{ℓ-1/2}
        #
        # Equations are stored in LHS = 0 form, so we add the NEGATIVE
        # of the RHS contribution:
        #
        #   transfer_ℓ_k  =  − u*_{ℓ+1/2}·G_{ℓ+1/2}  +  (−1)^k·u*_{ℓ-1/2}·G_{ℓ-1/2}
        for ell in range(1, self.N_layers + 1):
            G_top = G_interfaces[ell]      # G_{ℓ+1/2}
            G_bot = G_interfaces[ell - 1]  # G_{ℓ-1/2}
            u_top = u_interfaces[ell]
            u_bot = u_interfaces[ell - 1]
            for k in range(self.N + 1):
                name = f"momentum_x_layer_{ell}_{k}"
                if name not in self._equations:
                    continue
                transfer = (-u_top * G_top
                            + (-1)**k * u_bot * G_bot)
                eq = self._equations[name]
                eq.expr = eq.expr + transfer
                eq.simplify()

        # Stash bookkeeping for downstream / inspection / debugging.
        self._H_pre = H_pre
        self._b_global = b_global
        self._layer_q_fns = layer_q_fns
        self._G_interfaces = G_interfaces
        self._u_interfaces = u_interfaces

    # ── Per-layer SME pipeline ───────────────────────────────────
    def _derive_layer_sme(self, *, layer_label, h_layer_expr,
                          bottom_expr, top_pressure_expr):
        """Inline port of the single-layer SME pipeline, parameterised
        for one layer of the Audusse stack.  Runs the GENERIC SME
        derivation (with placeholder ``state.h`` and ``state.b``); at
        the end substitutes the layer-specific geometry
        (``state.b → bottom_expr``, ``state.h → h_layer_expr``).

        ``h_layer_expr`` here is the *Audusse-scaled* thickness
        ``α_ℓ · H_pre`` — an EXPRESSION, not a fresh placeholder
        Function, so derivatives like ``∂_x h_ℓ`` automatically
        propagate the ``α_ℓ`` factor.
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

        # ── Pre-σ hydrostatic + p-elimination ─────────────────────
        eq_mz = equations[f"momentum_z{layer_label}"]
        eq_mz.apply({src.w: 0, src.tau.zx: 0, src.tau.zz: 0,
                     src.tau.xz: 0}).simplify()
        eq_mz.apply(Integrate(s.z, s.z, src.eta, method="analytical"))
        eq_mz.apply({src.p.subs(s.z, src.eta): top_pressure_expr}).simplify()
        p_subst = Expression(eq_mz.expr,
                              f"momentum_z{layer_label}").solve_for(src.p)
        equations[f"momentum_x{layer_label}"].apply(p_subst).simplify()

        # ── σ-transform + KBCs ────────────────────────────────────
        _apply_all(SigmaTransform(s, src))
        _apply_all(KinematicBC(s, src.b,   src, at=sp.S.Zero))
        _apply_all(KinematicBC(s, src.eta, src, at=sp.S.One))

        # ── × h, inverse-ProductRule conservative folds ───────────
        _apply_all(Multiply(src.h))
        for eq in equations.values():
            eq.simplify()
        equations[f"momentum_x{layer_label}"].term[[0, 1]].apply(
            ProductRule(variables=[t, x]))
        equations[f"continuity{layer_label}"].term[[0]].apply(
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

        # ── Galerkin + σ-integrate ────────────────────────────────
        equations[f"continuity{layer_label}"].apply(
            Multiply(psi_w(sigma) * omega(sigma)))
        equations[f"momentum_x{layer_label}"].apply(
            Multiply(psi_u(sigma) * omega(sigma)))
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
            phi_u_fn,
            lambda k, sg, _b=legendre_u: basis_value(_b, k, sg)))
        _apply_all(ResolveDummy(
            phi_w_fn,
            lambda k, sg, _b=legendre_w: basis_value(_b, k, sg)))

        equations = _resolve_dummy_list(
            equations, psi_u,
            [legendre_value(legendre_u, k) for k in range(N + 1)])
        equations = _resolve_dummy_list(
            equations, psi_w,
            [legendre_value(legendre_w, k) for k in range(N + 2)])

        # ── Evaluate integrals ───────────────────────────────────
        _apply_all_dict(equations, EvaluateIntegrals(state))
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

        # ── Layer-local substitution (Audusse geometry) ──────────
        # Replace the GENERIC ``state.b`` with the layer's bottom
        # (global b + cumulative thickness of layers below) and the
        # GENERIC ``state.h`` with the layer's α-scaled thickness
        # ``α_ℓ · H_pre``.  Critically, ``h_layer_expr`` is an
        # EXPRESSION (not a fresh Function), so derivatives of h_ℓ
        # propagate the α_ℓ factor automatically.
        sub = {src.b: bottom_expr, src.h: h_layer_expr}
        for eq in equations.values():
            eq.expr = eq.expr.xreplace(sub)
            eq.simplify()
        return equations, q_fn

    # ── Lazy finalization hook ────────────────────────────────────
    def _prepare_for_systemmodel(self):
        """Substitute the H_pre, b_global placeholders + per-layer q
        Functions with their Model Symbol equivalents and set
        ``_variable_map``."""
        t_sym, x_sym = self.state.t, self.state.x
        subs = {
            self._H_pre:    self.variables.H,
            self._b_global: self.variables.b,
        }
        for ell_idx, q_fn in enumerate(self._layer_q_fns):
            ell = ell_idx + 1
            for k in range(self.N + 1):
                subs[q_fn(k, t_sym, x_sym)] = self.variables[f"q_layer_{ell}_{k}"]
        for eq in self:
            eq.expr = eq.expr.xreplace(subs)
        # Force the bottom equation to clean ``∂_t b = 0`` form.
        if "bottom" in self._equations:
            self._equations["bottom"].expr = sp.Derivative(
                self.variables.b, t_sym,
            )
        # ``∂_t b = 0`` cleanup elsewhere (no simplify — see SME).
        zero_dt_b = {sp.Derivative(self.variables.b, t_sym): sp.S.Zero}
        for name, eq in self._equations.items():
            if name == "bottom":
                continue
            eq.expr = eq.expr.xreplace(zero_dt_b)
        self._variable_map = self._build_variable_map()

    def _build_variable_map(self):
        """State layout: ``[b, H, q_layer_ℓ_k for ℓ=1..L, k=0..N]``.

        * ``bottom`` → row 0 (b)
        * ``continuity_global`` → row 1 (H)
        * ``momentum_x_layer_ℓ_k`` → row ``2 + (ℓ-1)·(N+1) + k``
        """
        m = {"bottom": [0], "continuity_global": [1]}
        row = 2
        for ell in range(1, self.N_layers + 1):
            for k in range(self.N + 1):
                m[f"momentum_x_layer_{ell}_{k}"] = [row]
                row += 1
        return m

    # ── 3D field reconstruction ──────────────────────────────────
    def interpolate_3d(self):
        """Reconstruct 3D fields from the multi-layer SME modal state.

        For each layer ℓ, the local σ-coordinate is
        ``σ_ℓ = (z − z_{ℓ-1/2}) / h_ℓ`` with
        ``z_{ℓ-1/2} = b + (Σ_{m<ℓ} α_m)·H``.  Inside layer ℓ
        (``z_{ℓ-1/2} ≤ z ≤ z_{ℓ+1/2}``):

        * ``u_3d = Σ_k (q_layer_ℓ_k / h_ℓ) · φ_k(σ_ℓ)``
        * ``v_3d = 0``
        * ``p_3d = ρ·g·(η − z)``   (hydrostatic)

        Outside all layers: zero.  Layer membership is encoded as a
        sympy ``Piecewise``.

        Returns ``Matrix([b, H, u_3d, v_3d, w_3d, p_3d])``.
        """
        from sympy import Matrix
        z = self.position[2]
        # Bathymetry is a state variable; use the state Symbol.
        b_sym = self.variables.b
        H = self.variables.H
        eta = b_sym + H
        N = self.N
        alphas = self._alphas
        # Use Legendre directly for the polynomial evaluation (basis_u
        # in this Audusse implementation is single-layer Legendre per
        # _derive_layer_sme).
        basis = Legendre_shifted(level=N)

        u_branches = []
        for ell in range(1, self.N_layers + 1):
            cumul_below = sum(alphas[:ell - 1], sp.S.Zero)
            cumul_at_top = cumul_below + alphas[ell - 1]
            z_bot = b_sym + cumul_below * H
            z_top = b_sym + cumul_at_top * H
            h_layer = alphas[ell - 1] * H
            sigma_ell = (z - z_bot) / h_layer
            u_layer = sum(
                (self.variables[f"q_layer_{ell}_{k}"] / h_layer)
                * basis.eval(k, sigma_ell)
                for k in range(N + 1)
            )
            u_branches.append((u_layer, (z >= z_bot) & (z <= z_top)))
        u_branches.append((sp.S.Zero, True))
        u_3d = sp.Piecewise(*u_branches)
        v_3d = sp.S.Zero
        w_3d = sp.S.Zero  # ML-SME doesn't carry w as state; defer.
        g = self.parameters.g
        rho = self.parameters.rho
        p_3d = rho * g * (eta - z)
        return Matrix([b_sym, H, u_3d, v_3d, w_3d, p_3d])


# ── Helpers ────────────────────────────────────────────────────────

def _resolve_dummy_list(equations: dict, dummy, value_list):
    """ResolveDummy(value=list) branches each equation containing the
    dummy into one sub-equation per list element.  Mimics
    ``ResolveDummy._apply_branched`` on a plain dict of Equations
    (the per-layer scratch dict isn't a Model, so we can't use the
    standard ``model.apply(...)`` whole-model dispatch path)."""
    import sympy as sp
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
    """Broadcast an operation / substitution across every equation in a
    plain dict (used during the per-layer scratch construction)."""
    for eq in equations.values():
        eq.apply(op, _no_history=True)
