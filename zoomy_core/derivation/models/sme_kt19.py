"""Declarative SME (Kowalski & Torrilhon 2019) builder.

Builds the Shallow Moment Equations end-to-end through the clean-redesign op
surface and returns the threaded :class:`~zoomy_core.derivation.model.Model`
ready for ``SystemModel.from_model(model, Q, Qaux)``.  The output equation set
matches K&T (4.17): ``continuity`` (``∂_t h + ∂_x(h·a_0)``) and the
``momentum_x_k`` moment rows.

Two composable surfaces live here, expressing **base-open → derived-closed**
reuse by INHERITANCE:

:class:`SME`
    The base model.  Its :meth:`SME.build` runs the full derivation pipeline
    and leaves the constitutive stress **OPEN** — the momentum rows carry the
    unresolved viscous-stress boundary atoms ``τ_xz(σ=0) / τ_xz(σ=1)`` and the
    ``∫ ẑ ∂_ẑ τ_xz dẑ`` moment integrals.  This is the physics axiom set; no
    constitutive law is committed.

:class:`SlipSME`
    INHERITS :class:`SME` and only ADDS the slip-Newton constitutive closure on
    top of the inherited open pipeline: ``super().build()`` gives the open
    model, then :meth:`SlipSME._apply_slip_closure` inserts the stress law by
    ``Substitution`` and resolves the leftover ``∫…dẑ`` moment integrals.  After
    the closure the momentum rows carry algebraic ``ν, λ`` friction terms and
    NO ``τ_xz`` / ``Integral`` atoms.

Base pipeline (all real ``.apply`` ops on one model thread):

1. **Axioms** — the GENERAL 2-D incompressible balance: continuity + the
   2-component momentum with the FULL viscous-stress tensor (``τ_xx, τ_xz,
   τ_zx, τ_zz``) and the pressure ``p`` NOT yet reduced + the bed ``∂_t b = 0``.

   * (a) **Stress reduction** — ``τ_xx→0, τ_zz→0, τ_zx→τ_xz`` (thin-film).
   * (b) **Hydrostatic pressure DERIVED** — reduce the z-momentum to ``∂_z p =
     −ρ g``, ``Integrate`` it over ``z → b+h`` (FTC: ``∂_z p → p(η)−p(z)``),
     impose the free-surface BC ``p(b+h)=0`` (a ``Substitution``), then
     ``solve_for(p)`` (PURE ALGEBRA — it no longer integrates) → ``p = ρ g
     (b+h−z)`` and substitute into the x-momentum.  The ``(1/ρ)∂_x p`` then
     DERIVES the ``g·∂_x(b+h)`` gravity flux (it is *not* hand-written), and
     ``momentum.z`` is removed.
2. **σ-map** — :class:`PDETransformation` ``z = b + h·ζ`` → decorated ζ-fields.
3. **Conservative pre-fold** — ``Multiply(h)`` (clear the ``1/h`` σ-Jacobian).
4. **Modal ansatz** — :func:`separation_of_variables` (u→a, w→aw of order N+1)
   with the opaque basis.
5. **Bind N** — ``Substitution({N_u: N})`` at the Model level, finite ``Sum``
   ``.doit()``-expanded to concrete modes.
6. **Moment families via** :func:`~zoomy_core.derivation.model.resolve_modes`
   — ``Project(c·φ_l)`` with the ABSTRACT test index ``l``, then a SHAPE BUMP
   into the moment family (scalar ``mass`` ``(1,)→(N+1,)`` → ``model.mass[l]``;
   vector ``momentum`` ``(2,)→(2,N+1)`` → ``model.momentum.x[l]``), each row
   closed by the concrete-level Galerkin
   :class:`~zoomy_core.derivation.closure.Resolve`.
7. **Kinematic-BC modal closure** — ``sp.solve`` the two KBCs for ``aw_0,
   aw_1`` (:func:`~zoomy_core.derivation.closure.kinematic_modal_closure`); the
   higher continuity moments ``model.mass[1..N]`` close ``aw_2..aw_{N+1}``;
   ``∂_t b → 0``.
8. **CoV** — :class:`ChangeOfVariables` ``a → q/h`` (conserved momentum modes).

Returns ``(model, ctx)`` where ``ctx`` carries the field handles
(``a, aw, q, h, b, …``) the caller needs for the ``Q`` / ``Qaux`` lists at the
``SystemModel.from_model`` transition.

:func:`build_sme` is a thin wrapper over ``SME(N).build()`` — kept so existing
tests / notebooks that call ``build_sme(N)`` continue to work unchanged.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core import coords as _coords
import zoomy_core.derivatives as d
from zoomy_core.derivation.model import Model, resolve_modes
from zoomy_core.derivation.transformations import PDETransformation
from zoomy_core.derivation.basis import Basis
from zoomy_core.derivation.modal import (
    separation_of_variables, reset_modal_indices, modal_bound,
)
from zoomy_core.derivation.operations import Substitution, ChangeOfVariables
from zoomy_core.derivation.closure import (
    Resolve, kinematic_modal_closure, mass_relation, Simplify,
    fold_to_conservative_form, is_conservative_diffusion,
    project_conservative_diffusion,
)
from zoomy_core.model.operations import Expression as _Expression
from zoomy_core.model.operations import (
    Legendre_shifted, Multiply, Expression, ProductRule, EvaluateIntegrals,
    Integrate,
)


__all__ = ["SME", "SlipSME", "NewtonianSME", "build_sme"]


# ── base model: open constitutive stress ──────────────────────────────────


class SME:
    """Declarative Shallow Moment Equations at moment level ``N`` — base model
    with the constitutive viscous stress left **OPEN**.

    Parameters
    ----------
    N : int
        Moment level (default ``2`` → K&T 4.17).  ``N + 1`` u-modes.
    parameters : dict, optional
        Numeric defaults for the model parameters declared on the
        :class:`~zoomy_core.derivation.model.Model`.  Defaults to
        ``{"g": 9.81, "rho": 1.0}``.  A derived closure (e.g.
        :class:`SlipSME`) declares the extra constitutive parameters it needs
        (``nu``, ``lambda``) by passing them here so their Symbol identity is
        consistent across the derivation.

    :meth:`build` runs the full pipeline and returns ``(model, ctx)`` with the
    momentum rows carrying the unresolved ``τ_xz`` boundary atoms + ``∫ ẑ ∂_ẑ
    τ_xz dẑ`` moment integrals.
    """

    DEFAULT_PARAMETERS = {"g": 9.81, "rho": 1.0}

    def __init__(self, N: int = 2, parameters=None):
        self.N = N
        self.parameters = dict(parameters) if parameters is not None \
            else dict(self.DEFAULT_PARAMETERS)

    # ── constitutive hook: τ_xx normal-stress closure ────────────────────
    def _tau_xx_closure(self, model, u, rho):
        """Substitution value for the horizontal normal stress ``τ_xx``,
        applied in the PHYSICAL momentum BEFORE the σ-map.  The base SME drops
        it (``→ 0``, thin-film); a derived model overrides this to substitute a
        constitutive law so the pipeline DERIVES its momentum contribution
        (through σ-map → ansatz → Project → Resolve) rather than hand-writing
        the answer."""
        return sp.Integer(0)

    # ── momentum resolution hook ─────────────────────────────────────────
    def _momentum_resolver(self, model, *, basis_cls, level, var, x):
        """Per-mode closer passed to ``resolve_modes`` for the momentum family.

        The base SME returns ``None`` → ``resolve_modes`` uses the default
        welded :class:`Resolve` (in-place ζ-integration) on the whole row.  A
        derived model carrying a SECOND-ORDER constitutive term (e.g.
        :class:`NewtonianSME`) overrides this to route the viscous block
        CONSERVATIVELY (so the SystemModel types it as ``diffusion_matrix``)
        while the rest of the row resolves in place."""
        return None

    # ── full pipeline (stress OPEN) ──────────────────────────────────────
    def build(self):
        """Run the SME derivation pipeline; return ``(model, ctx)`` with the
        constitutive stress left OPEN (``τ_xz`` BC atoms + moment integrals)."""
        N = self.N
        t, x, z = _coords.t, _coords.x, _coords.z
        zeta = sp.Symbol("zeta", real=True)

        model = Model(coords=(t, x, z), parameters=self.parameters)
        g, rho = model.parameters.g, model.parameters.rho

        # ── 1. Axioms: the GENERAL 2-D incompressible balance ─────────────
        # Full primitive system — continuity + the 2-component momentum with
        # the complete viscous-stress tensor and the pressure NOT yet reduced.
        # The hydrostatic pressure is DERIVED below (it is not pre-folded).
        u = sp.Function("u", real=True)(t, x, z)
        w = sp.Function("w", real=True)(t, x, z)
        p = sp.Function("p", real=True)(t, x, z)
        h = sp.Function("h", positive=True)(t, x)
        b = sp.Function("b", real=True)(t, x)
        txx = sp.Function("tau_xx", real=True)(t, x, z)
        txz = sp.Function("tau_xz", real=True)(t, x, z)
        tzx = sp.Function("tau_zx", real=True)(t, x, z)
        tzz = sp.Function("tau_zz", real=True)(t, x, z)

        model.Q = [h, u, w, p]
        model.add_equation("bottom", d.t(b))
        model.add_equation("mass", d.x(u) + d.z(w))
        model.add_equation("momentum", (2,), [
            d.t(u) + d.x(u * u) + d.z(u * w) + d.x(p) / rho
            - (d.x(txx) + d.z(txz)) / rho,                      # .x
            d.t(w) + d.x(u * w) + d.z(w * w) + d.z(p) / rho + g
            - (d.x(tzx) + d.z(tzz)) / rho,                      # .z
        ])

        # ── 1a. Stress reduction (BEFORE hydrostatic) ─────────────────────
        # Symmetrise the shear (τ_zx→τ_xz) and drop the τ_zz normal stress.
        # The τ_xx normal-stress closure is a HOOK: base SME drops it (τ_xx→0,
        # thin-film); a derived model substitutes a constitutive law (e.g.
        # NewtonianSME → τ_xx = 2ρν∂_x u) HERE, before the σ-map, so the
        # pipeline DERIVES the resulting (diffusive + σ-metric) momentum terms.
        model.apply(Substitution(
            {txx: self._tau_xx_closure(model, u, rho), tzz: 0, tzx: txz}))

        # ── 1b. Hydrostatic pressure — DERIVED, not hand-written ──────────
        # Reduce the z-momentum to the hydrostatic balance ``∂_z p = −ρ g`` by
        # dropping the vertical inertia + lateral shear, solve for ``p`` with
        # the free-surface BC ``p(z = b+h) = 0`` → ``p = ρ g (b+h−z)``, and
        # substitute into the x-momentum.  The ``(1/ρ)∂_x p`` then DERIVES the
        # ``g·∂_x(b+h)`` gravity flux (compare the old hand-folded axiom).
        model.momentum.z.apply(Substitution(
            {d.t(w): 0, d.x(u * w): 0, d.z(w * w): 0, d.x(txz): 0}))
        # Integrate the vertical balance from z up to the free surface η = b+h
        # (FTC: ∂_z p → p(η) − p(z)), impose the free-surface BC p(η)=0, then
        # ALGEBRAICALLY solve_for p — solve_for no longer integrates.
        model.momentum.z.apply(Integrate(z, z, b + h, method="analytical"))
        model.momentum.z.apply(Substitution({p.subs(z, b + h): 0}))
        p_hydro = model.momentum.z.solve_for(p)
        model.momentum.x.apply(p_hydro)
        model.momentum.z.remove()
        # ``p`` is no longer an unknown (eliminated); drop it from Q.
        model.momentum.x.apply(Simplify())

        # ── 2. σ-map ──────────────────────────────────────────────────────
        model.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))

        # ── 3. Conservative pre-fold (clear 1/h on the dynamic rows) ──────
        model.mass.apply(Multiply(h))
        model.momentum.x.apply(Multiply(h))

        # ── 4. Modal ansatz ────────────────────────────────────────────────
        basis = Basis(symbol="phi", weight="c")
        phi, c = basis.phi, basis.weight
        a = sp.Function("a")
        a_w = sp.Function("aw")
        N_u = modal_bound("N_u")
        reset_modal_indices(model)
        model.apply(separation_of_variables(u, a(t, x), basis, N_u))
        model.apply(separation_of_variables(w, a_w(t, x), basis, N_u + 1))

        # ── 5. Bind N at the Model level + expand the finite Sums ──────────
        model.apply(Substitution({N_u: N}))
        for eq in model._equations.values():
            eq.expr = eq.expr.replace(lambda e: isinstance(e, sp.Sum),
                                      lambda e: e.doit())

        n_modes = N + 1           # u-modes 0..N
        basis_lvl = N + 1         # cover the w-modes 0..N+1
        l = sp.Symbol("l", integer=True, nonnegative=True)
        kbc = kinematic_modal_closure(
            model, u_field=a, w_field=a_w, h=h, b=b,
            basis_cls=Legendre_shifted, n_u=N)
        dt_b_zero = Substitution({sp.Derivative(b, t): 0})

        # ── 6. Continuity moment family: PROJECT(c·φ_l) → resolve_modes ───
        # ``resolve_modes`` bumps the scalar ``mass`` row carrying the ABSTRACT
        # test index ``l`` into the ``(N+1,)`` moment family ``model.mass[l]``,
        # closing each moment row by the concrete Galerkin ``Resolve``.
        resolve_modes(
            model.mass, index=l, modes=range(N + 1),
            test_weight=c(zeta) * phi(l, zeta),
            basis_cls=Legendre_shifted, level=basis_lvl, var=zeta)

        # ── 6a. Higher-mode w closure off the resolved mass moments ───────
        # The higher continuities ``mass[1..N]`` (∂_t b = 0 applied) close the
        # free upper w-modes ``aw_2 .. aw_{N+1}`` via ``sp.solve`` — exactly
        # production's ``_higher_mode_w_closure``, but driven off the resolved
        # moment rows instead of hand-added ``_cont_k``.  Spent BEFORE the
        # momentum rows so the substituted w-modes carry through.
        higher_residuals = []
        higher_w = [a_w(k, t, x) for k in range(2, N + 2)]
        for k in range(1, N + 1):
            eq = model.mass[k]
            eq.apply(dt_b_zero)
            eq.apply(Simplify())
            higher_residuals.append(eq.expr)
        if higher_w:
            higher_solution = sp.solve(higher_residuals, higher_w, dict=True)[0]
            higher_w_closure = Substitution(higher_solution,
                                            name="higher_mode_w_closure")
        else:
            higher_w_closure = None

        # ── 6b. Mean continuity row (l=0): KBC closure → ∂_t b=0 → simplify
        # Close the mean moment to the single depth-averaged mass row (K&T
        # continuity).  The higher moments ``mass[1..N]`` have done their job
        # (closing the w-modes) and are dropped, COLLAPSING the moment family to
        # the single surviving moment ``mass[0] = ∂_t h + ∂_x q_0``.  ``[l]``
        # stays moment-uniform: ``model.mass`` is still a ``MomentFamily`` and
        # ``model.mass[0]`` the full moment row (NOT ``model.mass.term[0]``).
        model.mass[0].apply(kbc)
        model.mass[0].apply(dt_b_zero)
        model.mass[0].apply(Simplify())
        # Drop the higher continuity moment rows; keep ``mass[0]`` and shrink the
        # family to the single surviving mode (re-keyed ``mass_0 → mass``).
        for k in range(1, N + 1):
            model._remove_equation(f"mass_{k}")
        model._collapse_moment_family("mass", keep=[0])

        # ── 7. Momentum moment family: PROJECT(c·φ_l) → resolve_modes ─────
        # Bump the x-momentum row (abstract ``l``) into the ``(2, N+1)`` moment
        # family ``model.momentum.x[l]``, closing each row by the concrete
        # Galerkin ``Resolve``.  Then KBC + higher-w closure + ∂_t b=0 + the
        # diagonal Legendre mass-matrix inversion ×(2k+1) per row.
        resolve_modes(
            model.momentum.x, index=l, modes=range(n_modes),
            test_weight=c(zeta) * phi(l, zeta),
            basis_cls=Legendre_shifted, level=basis_lvl, var=zeta,
            resolver=self._momentum_resolver(
                model, basis_cls=Legendre_shifted, level=basis_lvl,
                var=zeta, x=x))
        for k in range(n_modes):
            eq = model.momentum.x[k]
            eq.apply(kbc)
            if higher_w_closure is not None:
                eq.apply(higher_w_closure)
            eq.apply(dt_b_zero)
            # The mean row (k=0) carries a clean conservative ``∂_t(h·a_0)``;
            # the higher rows carry the stray ``∂_t h`` the σ-metric
            # chain-rule injects via the KBC-substituted w-modes, which the
            # mass relation cancels.
            if k >= 1:
                eq.apply(mass_relation(h, a))
            # Invert the diagonal Legendre Galerkin mass matrix: ×(2k+1).
            eq.expr = (2 * k + 1) * eq.expr
            eq.apply(Simplify())

        # ── 8. Change of variables a → q/h (conserved momentum modes) ─────
        q = sp.Function("q")
        model.apply(ChangeOfVariables("a", "q", lambda qi: qi / h))
        for eq in model._equations.values():
            eq.simplify()

        # ── 8b. Conservative fold (flux/pressure bundling, NCP-preserving) ─
        # The Resolve/CoV chain leaves the spatial part fully EXPANDED
        # (``2 q_k/h·∂_x q_k − q_k²/h²·∂_x h + …``).  Re-bundle it EXACTLY as
        # K&T (4.17) / production: the conservative flux + hydrostatic-pressure
        # divergences fold into ``∂_x(F)`` / ``∂_x(g h²/2)`` units, while the
        # genuinely non-conservative couplings stay UNFOLDED so the SystemModel
        # extractor reads a non-zero ``nonconservative_matrix`` — the bed
        # coupling ``g·h·∂_x b`` and the cross-mode ``q_i/h·∂_x q_j`` terms.
        #
        # On the higher moment rows the σ-metric chain-rule injects a stray
        # ``∂_t h``; cancel it with the conserved mass relation
        # ``∂_t h → −∂_x q_0`` (the mean / mass rows are already clean) BEFORE
        # the conservative fold, exactly as production's mass-matrix-inverted
        # higher rows do.
        flux_fields = [q(k, t, x) for k in range(n_modes)]
        dt_h_to_mass = {sp.Derivative(h, t): -sp.Derivative(q(0, t, x), x)}
        for name, eq in model._equations.items():
            if name.startswith("momentum_x_"):
                # PEEL OFF the conservative diffusion atoms ``∂_x(D·∂_x q)``
                # BEFORE the doit/fold: ``.doit()`` would apply the product rule
                # and scatter the second-order flux back into source, and the
                # first-order ``fold_to_conservative_form`` is not meant for it.
                # The diffusion part is already in the K&T-conservative
                # ``∂_x(F^d)`` shape the SystemModel types as ``diffusion_matrix``
                # — keep it untouched and fold only the hyperbolic remainder.
                diff_terms, base_terms = [], []
                for term in sp.Add.make_args(eq.expr):
                    (diff_terms if is_conservative_diffusion(term, x)
                     else base_terms).append(term)
                expr = sp.expand(sp.Add(*base_terms).doit())
                if not name.endswith("_0"):
                    expr = sp.expand(expr.xreplace(dt_h_to_mass))
                expr = fold_to_conservative_form(
                    expr, flux_fields, h=h, b=b, x=x, gravity_param=g)
                eq.expr = expr + sp.Add(*diff_terms)
            else:
                eq.expr = sp.expand(eq.expr.doit())

        # Undecorate the viscous-stress head (the σ-map decorated ``tau_xz`` →
        # ``\tilde{tau_xz}``; the boundary-trace stress is a physical aux
        # field, so it carries the original head — matching the production K&T
        # form).
        def _undecorate_tau(expr):
            return expr.replace(
                lambda e: (isinstance(e, sp.Function)
                           and getattr(e.func, "_pde_decorated_from", None)
                           == "tau_xz"),
                lambda e: sp.Function("tau_xz", real=True)(*e.args),
            )
        for eq in model._equations.values():
            eq.expr = _undecorate_tau(eq.expr)

        ctx = dict(
            t=t, x=x, zeta=zeta, h=h, b=b, a=a, aw=a_w, q=q,
            basis=basis, basis_cls=Legendre_shifted, N=N, n_modes=n_modes,
            q_modes=[q(k, t, x) for k in range(n_modes)],
            aw_modes=[a_w(k, t, x) for k in range(N + 2)],
            g=g, rho=rho,
        )
        return model, ctx


# ── derived model: slip-Newton constitutive closure ───────────────────────


class SlipSME(SME):
    """SME with the slip-Newton stress closure inserted.

    INHERITS :class:`SME`.  :meth:`build` calls ``super().build()`` to obtain
    the open-stress model, then :meth:`_apply_slip_closure` resolves the
    constitutive stress in place via the slip-Newton law of K&T 2019 §4.3:

      * **Newtonian bulk:** ``τ_xz(σ) = (ρ·ν/h)·∂_σ u(σ)`` with
        ``u(σ) = Σ_{k=0}^{N} (q_k/h)·φ_k(σ)`` (shifted-Legendre ``φ_k``);
      * **free-surface BC:** ``τ_xz(σ=1) = 0``;
      * **Navier-slip bottom BC:** ``τ_xz(σ=0) = (ρ·ν/λ)·u(σ=0)``;
      * the ``∫…dẑ`` stress moment integrals are resolved by inverse product
        rule + integrate, then :class:`EvaluateIntegrals`.

    The constitutive parameters ``ν`` ("nu") and ``λ`` ("lambda") MUST be
    declared at construction so their Symbol identity is consistent with the
    rest of the derivation::

        SlipSME(N=2, parameters={"g": 9.81, "rho": 1.0,
                                 "nu": 1e-3, "lambda": 1e-2})

    After the closure the momentum rows carry algebraic ``ν, λ`` friction
    terms and NO ``τ_xz`` / ``Integral`` atoms.
    """

    def build(self):
        """Build the open SME (via ``super().build()``) then insert the
        slip-Newton stress closure; return ``(model, ctx)``."""
        model, ctx = super().build()
        self._apply_slip_closure(model, ctx)
        return model, ctx

    def _apply_slip_closure(self, model, ctx):
        """Resolve the open ``τ_xz`` BC atoms + ``∫…dẑ`` moment integrals into
        the slip-Newton algebraic friction terms on every momentum row."""
        missing = [k for k in ("nu", "lambda")
                   if k not in model.parameters.keys()]
        if missing:
            raise ValueError(
                "SlipSME: constitutive parameter "
                f"{missing[0]!r} must be declared at construction "
                "(e.g. `parameters={'g': 9.81, 'rho': 1.0, "
                "'nu': 1e-3, 'lambda': 1e-2}`)."
            )

        t, x = ctx["t"], ctx["x"]
        h, q, rho = ctx["h"], ctx["q"], ctx["rho"]
        n_u = ctx["n_modes"]
        basis = ctx["basis_cls"](level=n_u)
        nu = model.parameters.nu
        lam = model.parameters["lambda"]
        tau_xz = sp.Function("tau_xz", real=True)

        # Reconstruct u(σ) = Σ_k (q_k/h)·φ_k(σ) in concrete shifted-Legendre
        # form, mirroring production ``apply_slip_newton_friction``.
        def u_at(sig):
            return sum((q(k, t, x) / h) * basis.eval(k, sig)
                       for k in range(n_u))

        def newton(t_arg, x_arg, sig):
            # Newton's law: τ_xz = ρ·ν·∂_z u, kinematic ν.  σ-frame
            # ∂_z → (1/h)·∂_σ.  The momentum row divides τ by ρ later, so ρ
            # cancels — canonical ν·(1/h)·∂_σ u friction (no spurious 1/ρ).
            return (rho * nu / h) * sp.diff(u_at(sig), sig)

        def integral_via_product_rule(*integral_args):
            integrand, (var, lo, hi) = integral_args[0], integral_args[1]
            rewritten = Expression(integrand, "").apply(
                ProductRule(variables=[var], direction="inverse")
            ).expr
            return sum((sp.integrate(piece, (var, lo, hi))
                        for piece in sp.Add.make_args(rewritten)),
                       sp.S.Zero)

        # Navier-slip:  τ_bed = (μ/λ)·u(σ=0) = ρ·ν/λ·u(σ=0); ρ cancels the τ/ρ
        # in the momentum row → friction coefficient ν/(λ·h).
        tau_at_0_slip = (rho * nu / lam) * u_at(sp.S.Zero)
        tau_at_1_free = sp.S.Zero
        evaluate = EvaluateIntegrals()

        for name in [f"momentum_x_{k}" for k in range(n_u)]:
            eq = model._equations[name]
            expr = eq.expr
            # 1. Inverse product rule + integrate the open ``∫ ẑ ∂_ẑ τ_xz dẑ``
            #    moment integrals — leaves a residual ``∫ τ_xz dẑ`` carrying the
            #    bare stress, plus the ``[ẑ·τ_xz]`` boundary trace.
            expr = expr.replace(sp.Integral, integral_via_product_rule)
            # 2. Insert the two stress BCs (σ=0 slip, σ=1 free) then Newton's
            #    bulk law for every remaining ``τ_xz`` application.
            expr = expr.xreplace({
                tau_xz(t, x, sp.S.Zero): tau_at_0_slip,
                tau_xz(t, x, sp.S.One):  tau_at_1_free,
            })
            expr = expr.replace(tau_xz, newton)
            # 3. Resolve any leftover ``∫…dẑ`` Newton-bulk integrals.
            expr = evaluate._leaf_sp(expr)
            eq.expr = sp.expand(expr)
            eq.simplify()


# ── derived model: Newtonian normal-stress diffusive closure ───────────────


class NewtonianSME(SlipSME):
    """SlipSME PLUS the Newtonian horizontal normal-stress diffusion.

    INHERITS :class:`SlipSME` (slip-Newton ``τ_xz`` shear closure) and ADDS the
    Newtonian normal stress that the base SME drops (``τ_xx → 0``):

    .. math::

        τ_{xx} = 2 μ \\, ∂_x u = 2 ρ ν \\, ∂_x u, \\qquad μ = ρ ν.

    Its momentum contribution ``(1/ρ) ∂_x τ_{xx} = 2 ν ∂_{xx} u`` is a
    SECOND-ORDER term.  It is DERIVED, not hand-written: carried through the
    σ-map (``∂_x → Dₓ = ∂_x − m ∂_ζ``, ``m = (∂_x b + ζ ∂_x h)/h``), the modal
    ansatz, and a CONSERVATIVE Galerkin route (:meth:`_momentum_resolver` →
    :func:`~zoomy_core.derivation.closure.project_conservative_diffusion`:
    ``Multiply(c·φ_k) → ProductRule() → abstract Integrate → ResolveIntegral``)
    that keeps the genuine diffusive flux in the ``∂_x(F^d)`` shape.

    The σ-transform splits the viscous contribution into TWO physically
    distinct pieces:

    * the **second-order diffusive flux** ``∂_x(F^d_k)``, ``F^d_k = −2 ν h
      ∂_x(q_k/h) + (σ-metric flux)``, which the extractor types as the rank-4
      ``diffusion_matrix`` — a DIAGONAL ``A[q_k, q_k] = −2 ν`` plus the σ-metric
      gradient couplings ``A[q_k, h]`` and ``A[q_k, b]`` (these vanish for a
      flat geometry, leaving the pure ``−2 ν`` eddy viscosity);
    * the **σ-metric corrections that are NOT of ``A ∇Q`` form** — bilinear
      gradients ``∂_x b·∂_x q``, depth-curvature ``q·(∂_x h)²``, etc. — which
      are genuine viscous SOURCE terms (``∝ ν``, vanishing for ``∂_x b = ∂_x h
      = 0``).  The earlier hand-coded closure silently DROPPED these; the
      genuine derivation keeps them.

    The advective flux / hydrostatic pressure / NCP / mass matrix are EXACTLY
    those of the slip SME — the normal stress is purely viscous and touches
    only ``diffusion_matrix`` and the viscous part of ``source``.
    """

    def _tau_xx_closure(self, model, u, rho):
        """Newtonian horizontal normal stress ``τ_xx = 2μ ∂_x u = 2ρν ∂_x u``
        (``μ = ρν``), substituted into the PHYSICAL momentum BEFORE the σ-map.

        Nothing is hand-written: the viscous momentum term ``(1/ρ)∂_x τ_xx =
        2ν ∂_x(∂_x u)`` is carried through the σ-map (picking up the σ-metric
        ``∂_x b``/``∂_x h`` corrections), the modal ansatz, the Galerkin
        ``Project``/``Resolve`` and the conservative fold — so the resulting
        diffusive flux AND its non-conservative/source corrections are DERIVED.
        The SystemModel extractor then types the second-order
        ``∂_x(coeff·∂_x q_j)`` part as ``diffusion_matrix``."""
        nu = model.parameters.nu
        return 2 * rho * nu * d.x(u)

    def _momentum_resolver(self, model, *, basis_cls, level, var, x):
        """Route the viscous (``ν``) block of each momentum moment
        CONSERVATIVELY so the SystemModel types it as ``diffusion_matrix``.

        Per mode the row is split into its viscous terms (those carrying the
        Newtonian ``ν`` from :meth:`_tau_xx_closure`) and the hyperbolic
        remainder.  The remainder closes with the welded in-place
        :class:`Resolve` (unchanged from the base SME); the viscous block goes
        through :func:`project_conservative_diffusion`
        (``Multiply(test) → ProductRule() → abstract Integrate → ResolveIntegral``)
        which keeps the genuine diffusive flux in conservative ``∂_x(F^d)``
        form.  The SUM is bit-identical to resolving the whole row in place —
        only the diffusive part's REPRESENTATION changes (so it can be typed as
        ``diffusion_matrix`` rather than scattered into ``source``)."""
        nu = model.parameters.nu

        def _resolve(row, test_weight, k):
            visc = sp.S.Zero
            rest = sp.S.Zero
            for term in row.term:
                if term.expr.has(nu):
                    visc += term.expr
                else:
                    rest += term.expr
            rest_res = _Expression(rest, "").apply(
                Resolve(test_weight, basis_cls, level, var=var)).expr
            visc_res = project_conservative_diffusion(
                visc, test_weight, basis_cls=basis_cls, level=level,
                var=var, x=x)
            row.expr = sp.expand(rest_res) + visc_res

        return _resolve


# ── thin backward-stable wrapper ──────────────────────────────────────────


def build_sme(N: int = 2):
    """Build the declarative SME at moment level ``N`` (default 2 → K&T 4.17).

    Thin wrapper over ``SME(N).build()`` — the base model with the constitutive
    stress left OPEN.  Returns ``(model, ctx)``.  ``ctx`` is a dict of the
    symbolic handles needed downstream: ``a`` (u-mode head), ``aw`` (w-mode
    head), ``q`` (conserved momentum head), ``h``, ``b``, ``t``, ``x``,
    ``basis_cls``, ``n_modes``, ``q_modes`` (applied ``q(k,t,x)``), and the
    bare field heads for Q/Qaux.
    """
    return SME(N).build()
