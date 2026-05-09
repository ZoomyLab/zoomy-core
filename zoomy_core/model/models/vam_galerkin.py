"""VAMModelGalerkin — VAM derived via the explicit Galerkin chain.

The chain follows Escalante 2024's cont-projection formulation
(see ``thesis/chapters/derivation_vam.md`` §5.7 and §5.5–§5.6 for the
worked derivation):

  1. Inherit from :class:`VAMModel` (the base) so the solver-compatible
     state structure ``[b, h, hu_k, hw_k, hp_k]`` and operator-API
     surface stay populated for downstream consumers.
  2. Project momentum AND continuity against shifted-Legendre test
     functions; close ``W_{N_w}`` via the bottom KBC at the basis level
     (algebraic substitution, not an ``add_equation`` row); close
     ``P_{N_p}`` via the surface BC.  The ``j = 0`` continuity row
     becomes the mass evolution; ``j = 1, …, N_p`` continuity rows are
     the algebraic constraints that determine ``P_0, …, P_{N_p−1}``.

Outputs stored on the model:

* ``_chain_intermediate`` — Sum-form snapshot (post-Expand,
  pre-EvaluateIntegrals).  ``describe()`` renders ``Σ_k U_k φ_k(ζ)``.
* ``_chain_system`` — closed system: ``N_p+1`` continuity projections
  (1 mass evolution + ``N_p`` algebraic constraints) + ``M+1``
  x-momentum + ``N_w`` z-momentum, with ``W_{N_w}`` and ``P_{N_p}``
  consumed by the closures.
* ``_chain_dae`` — the same equations packaged as a
  :class:`zoomy_core.analysis.PDESystem`.
* ``_chain_dae_systemmodel`` — operator-form ``SystemModel``.

Equation count for ``(M=1, N_w=2, N_p=2)``:
  1 + 2 + 2 + 2 = **7 equations**, **7 unknowns**
  ``(h, U_0, U_1, W_0, W_1, P_0, P_1)`` after eliminating
  ``W_2 = (U_0 + U_1)∂_x b − W_0 − W_1`` (bottom KBC) and
  ``P_2 = P_1 − P_0`` (surface BC).
"""

from __future__ import annotations

import copy

import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.ins_generator import (
    AffineProjection,
    EvaluateIntegrals,
    Expand,
    FullINS,
    InterfaceKBC,
    Integrate,
    Inviscid,
    Multiply,
    ProductRule,
    StateSpace,
)
from zoomy_core.model.models.vam_model import VAMModel


class VAMModelGalerkin(VAMModel):
    """VAM derived via the explicit symbolic Galerkin chain."""

    def __init__(self, level=0, *, M=None, N_w=None, N_p=None, **kwargs):
        # ``VAMModel.__init__`` calls ``self.derive_model()`` *before*
        # ``Model.__init__`` populates ``self.level`` via param, so we
        # have to stash the chain levels on the instance up front.
        self._chain_M = M if M is not None else level
        self._chain_N_w = N_w if N_w is not None else self._chain_M + 1
        self._chain_N_p = N_p if N_p is not None else self._chain_M + 1
        super().__init__(level=level, **kwargs)

    # ------------------------------------------------------------------
    # The derivation — written linearly.  No helpers beyond what the
    # ``System`` API already provides (``apply``, ``add_equation``,
    # ``remove_equation``).
    # ------------------------------------------------------------------

    def derive_model(self):
        # Parent VAMModel populates the inherited operator-API path
        # (flux / NCP / source via the basis-matrix machinery).  We
        # leave that untouched and add the explicit chain on top.
        super().derive_model()

        M = self._chain_M
        N_w = self._chain_N_w
        N_p = self._chain_N_p

        # State + bases + coefficients.
        state = StateSpace(dimension=2)
        z = state.z
        # Use distinct basis symbol names that don't collide with
        # sympy / mpmath built-ins.  ``eta`` and ``mu`` are mpmath
        # special functions and trigger TypeError when sympy attempts
        # to evalf the test-function argument ``(z-b)/h``.  Using
        # ``phi_u``, ``phi_w``, ``phi_p`` is collision-free.
        basis_u = Legendre_shifted(level=M,   symbol="phi_u")
        basis_w = Legendre_shifted(level=N_w, symbol="phi_w")
        basis_p = Legendre_shifted(level=N_p, symbol="phi_p")
        coeffs_u = [sp.Function(f"U_{k}", real=True)(state.t, state.x)
                    for k in range(M + 1)]
        coeffs_w = [sp.Function(f"W_{k}", real=True)(state.t, state.x)
                    for k in range(N_w + 1)]
        coeffs_p = [sp.Function(f"P_{k}", real=True)(state.t, state.x)
                    for k in range(N_p + 1)]
        # Test-function arguments — uniform opaque-ζ convention.
        #
        # All test functions use ``φ_k(state.zeta)`` where
        # ``state.zeta = Function("zeta")(t, x, z)`` is an opaque sympy
        # Function.  Sympy's native chain rule fires through this head, so
        # ``ProductRule(variables=[t, x, z])`` pre-distributes
        # ``φ(ζ)·∂_t F → ∂_t(φ(ζ)·F) − F·φ'(ζ)·∂_t ζ`` before
        # ``Integrate``.  The first term has coeff=1 (t-independent), so
        # Leibniz fires correctly for the ``∂_t u`` / ``∂_t w`` integrands
        # in momentum.  The spatial chain-rule volume terms
        # ``−∫ u · ∂_x φ_j|_z dz`` for continuity (Escalante's I_j) also
        # fall out of ProductRule via ``∂_x(φ(ζ)) = φ'(ζ)·∂_x ζ``.
        # ``AffineProjection._collapse_opaque_zeta`` collapses all ζ(…)
        # atoms after the affine map z → ζ_ref·h + b.
        test_phi_u = Zstruct(
            **{f"phi_{k}": basis_u.phi[k](state.zeta)
               for k in range(M + 1)})
        test_phi_w = Zstruct(
            **{f"phi_{k}": basis_w.phi[k](state.zeta)
               for k in range(N_w)})
        test_phi_cont = Zstruct(
            **{f"phi_{k}": basis_p.phi[k](state.zeta)
               for k in range(N_p + 1)})

        # 1. 3D INS, drop viscosity, split off the hydrostatic pressure.
        sys = FullINS(state)
        sys.apply(Inviscid(state)).simplify()
        p_NH = sp.Function("p_NH", real=True)(state.t, state.x, z)
        sys.apply({state.p: state.rho * state.g * (state.eta - z) + p_NH}
                  ).simplify()

        # 2. Project continuity AND momentum against test functions.
        # The j=0 continuity projection becomes the mass evolution row;
        # j = 1, …, N_p continuity rows become the cont-projection
        # algebraic constraints (the elliptic system for the P_k).
        sys.continuity.apply(Multiply(test_phi_cont, outer=True))
        sys.momentum.x.apply(Multiply(test_phi_u, outer=True))
        sys.momentum.z.apply(Multiply(test_phi_w, outer=True))

        # 3. ProductRule on ``[t, x, z]`` — pre-distributes
        # ``φ(ζ)·∂_v F → ∂_v(φ(ζ)·F) − F·φ'(ζ)·∂_v ζ`` for all three
        # variables.  Adding ``t`` is the key change vs the old mixed
        # convention: it makes the Leibniz coefficient t-independent (=1)
        # so ``Integrate(method="auto")`` fires Leibniz on ``∂_t u`` /
        # ``∂_t w`` and produces the correct conservative-form boundary
        # corrections ``−u|_η ∂_t η`` / ``+u|_b ∂_t b``.
        sys.apply(ProductRule(variables=[state.t, state.x, z]))

        # 4. Depth-integrate (Leibniz on ∂_t / ∂_x; FT on ∂_z).
        sys.apply(Integrate(z, state.b, state.eta, method="auto"))

        # 5. Apply kinematic BCs at bottom + surface.  Surface KBC at
        # z = η substitutes ``∂_t η = ∂_t b + ∂_t h``; the ``∂_t b``
        # piece is dropped further below — after EvaluateIntegrals plus
        # ``.doit()`` — so that compound atoms like
        # ``Derivative(c·b, t)`` (where ``c`` is a basis-evaluation
        # constant from EvaluateIntegrals) have collapsed into the bare
        # ``Derivative(b, t)`` form the substitution can match.
        sys.apply(InterfaceKBC(state, state.b)).simplify()
        sys.apply(InterfaceKBC(state, state.eta)).simplify()

        # 6. Surface BC for the non-hydrostatic pressure remainder
        # (applied at the field level so the integrand sees
        # ``p_NH(η) = 0``).
        sys.apply({p_NH.subs(z, state.eta): 0}).simplify()

        # 7. Affine map z → ζ = (z−b)/h on the integration variable;
        # ``rewrite_basis_args=False`` because the test-function args
        # are already in affine form.  Then expand u / w / p_NH into
        # modes.
        sys.apply(AffineProjection(state, rewrite_basis_args=False))
        sys.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u,
                         state=state))
        sys.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w,
                         state=state))
        sys.apply(Expand(p_NH,    basis=basis_p, coefficients=coeffs_p,
                         state=state))

        # Snapshot Sum-form intermediate (paper notation) before the
        # polynomial integrals are resolved.
        self._chain_intermediate = copy.deepcopy(sys)

        # 8. Resolve the ζ integrals using the basis cache.  Leaves
        # are now polynomial in ``(h, U_k, W_k, P_k, b)``.
        sys.apply(EvaluateIntegrals(state)).simplify()

        # 9. Drop ``∂_t b`` (static bottom).  Apply ``.doit()`` first so
        # compound ``Derivative(c·b, t)`` atoms (where ``c`` is a
        # basis-evaluation constant) distribute their constants out and
        # produce the bare ``Derivative(b, t)`` atom that the
        # substitution rule matches.  ``b(t, x)`` stays a Function on
        # ``(t, x)`` — only the time derivative is zeroed, so transient
        # bottom topography can be reintroduced by removing this
        # substitution.
        sys.doit()
        sys.apply({sp.Derivative(state.b, state.t): sp.S.Zero}).simplify()

        # 10. Bottom KBC modal closure: solve
        #    Σ_k W_k φ_w_k(0) − (Σ_k U_k φ_u_k(0)) ∂_x b = 0
        # for ``W_{N_w}`` and substitute everywhere.  This consumes
        # the bottom KBC at the basis level, eliminating W_{N_w} as a
        # free unknown.  The surface KBC has already been consumed in
        # step 5's IBP boundary substitution.
        u_at_b = sum(coeffs_u[k] * basis_u.eval(k, sp.S.Zero)
                     for k in range(M + 1))
        w_at_b = sum(coeffs_w[k] * basis_w.eval(k, sp.S.Zero)
                     for k in range(N_w + 1))
        bot_kbc = w_at_b - u_at_b * sp.Derivative(state.b, state.x).doit()
        w_top_sol = sp.solve(bot_kbc, coeffs_w[N_w])[0]
        sys.apply({coeffs_w[N_w]: w_top_sol}).simplify()

        # 11. Surface BC for the non-hydrostatic pressure, applied at
        # the modal level: solve ``Σ_k φ_p_k(1) · P_k = 0`` for the
        # highest mode and substitute everywhere.
        p_at_eta = sum(coeffs_p[k] * basis_p.eval(k, sp.S.One)
                       for k in range(N_p + 1))
        p_top_sol = sp.solve(p_at_eta, coeffs_p[N_p])[0]
        sys.apply({coeffs_p[N_p]: p_top_sol}).simplify()

        # Stash + build PDESystem and SystemModel views.
        self._chain_system = sys
        self._chain_state = state
        self._chain_coeffs = {
            "u": coeffs_u, "w": coeffs_w[:N_w], "p": coeffs_p[:N_p]
        }

        from zoomy_core.analysis import PDESystem
        from zoomy_core.model.models.system_model import SystemModel

        # 12. Substitute the mass equation ``∂_t h = -∂_x(h U_0)`` into
        # the cont_j algebraic rows so they are purely algebraic in
        # ``(h, U_k, W_k, P_k, ∂_x ·)`` — no time derivatives.  This is
        # what makes them DAE algebraic constraints (zero-row in the
        # mass matrix).
        mass_expr = sys._tree.continuity.test_0.expr
        # mass = ∂_t h + ∂_x(h U_0) ⇒ ∂_t h = -∂_x(h U_0).
        dt_h_sub = {sp.Derivative(state.h, state.t):
                    -sp.Derivative(state.h * coeffs_u[0], state.x).doit()}

        # Row order: mass evolution first (continuity.test_0), then
        # x-momentum (test_0..M), z-momentum (test_0..N_w-1), then the
        # cont-projection algebraic rows (cont_j1..cont_j_{N_p}).
        ordered = [("mass", mass_expr)]
        for k in range(M + 1):
            ordered.append((f"xmom_j{k}",
                            getattr(sys._tree.momentum.x, f"test_{k}").expr))
        for k in range(N_w):
            ordered.append((f"zmom_j{k}",
                            getattr(sys._tree.momentum.z, f"test_{k}").expr))
        for k in range(1, N_p + 1):
            cont_jk = getattr(sys._tree.continuity, f"test_{k}").expr
            # ``.doit()`` distributes constants out of compound
            # ``Derivative(c·h, t)`` atoms (re-packaged by the
            # bottom-KBC and surface-BC closures' substitutions) so
            # the bare ``Derivative(h, t)`` atom matches ``dt_h_sub``.
            cont_jk_alg = sp.expand(cont_jk.doit().subs(dt_h_sub))
            ordered.append((f"cont_j{k}", cont_jk_alg))

        self._chain_dae = PDESystem(
            equations=[expr for _, expr in ordered],
            fields=[state.h] + coeffs_u + coeffs_w[:N_w] + coeffs_p[:N_p],
            time=state.t,
            space=[state.x],
            parameters={state.g: state.g, state.rho: state.rho},
        )
        self._chain_dae.equation_names = [n for n, _ in ordered]
        self._chain_dae_systemmodel = SystemModel.from_pdesystem(
            self._chain_dae)

    # ------------------------------------------------------------------
    # Display.
    # ------------------------------------------------------------------

    def describe(self, **kwargs):
        return self._chain_system.describe(**kwargs)

    def _repr_markdown_(self):
        return self._chain_system._repr_markdown_()

    def describe_chain_intermediate(self):
        """Render the Sum-form intermediate (post-Expand, pre-Evaluate)."""
        return self._chain_intermediate.describe()

    def describe_chain_closed(self):
        """Same as :meth:`describe`; kept for symmetry."""
        return self._chain_system.describe()


__all__ = ["VAMModelGalerkin"]
