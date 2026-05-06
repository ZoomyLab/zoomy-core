"""VAMModelGalerkin — VAM derived via the explicit Galerkin chain.

The class has exactly one job per the user's preferred shape:

  1. Inherit from :class:`VAMModel` (the base) so the solver-compatible
     state structure ``[b, h, hu_k, hw_k, hp_k]`` and operator-API
     surface stay populated for downstream consumers.
  2. ``derive_model`` is **linear**: a sequence of primitive
     applications + equation additions, no nested helpers.

Outputs stored on the model:

* ``_chain_intermediate`` — Sum-form snapshot (post-Expand,
  pre-EvaluateIntegrals).  ``describe()`` renders ``Σ_k U_k φ_k(ζ)``.
* ``_chain_system`` — closed system: 1 continuity (mass evolution)
  + ``M+1`` x-momentum + ``N_w+1`` z-momentum + ``kbc_top`` + ``kbc_bot``,
  with the highest pressure mode eliminated via the non-hydrostatic
  surface BC.
* ``_chain_dae`` — the same equations packaged as a
  :class:`zoomy_core.analysis.PDESystem`.
* ``_chain_dae_systemmodel`` — operator-form ``SystemModel``.

Equation count for ``(M=1, N_w=2, N_p=2)``:
  1 + 2 + 3 + 2 = **8 equations**, **8 unknowns**
  ``(h, U_0, U_1, W_0, W_1, W_2, P_0, P_1)`` after eliminating
  ``P_2 = P_1 − P_0``.
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
        basis_u = Legendre_shifted(level=M,   symbol="phi")
        basis_w = Legendre_shifted(level=N_w, symbol="eta")
        basis_p = Legendre_shifted(level=N_p, symbol="mu")
        coeffs_u = [sp.Function(f"U_{k}", real=True)(state.t, state.x)
                    for k in range(M + 1)]
        coeffs_w = [sp.Function(f"W_{k}", real=True)(state.t, state.x)
                    for k in range(N_w + 1)]
        coeffs_p = [sp.Function(f"P_{k}", real=True)(state.t, state.x)
                    for k in range(N_p + 1)]
        test_phi_u = Zstruct(
            **{f"phi_{k}": basis_u.phi[k](z) for k in range(M + 1)})
        test_phi_w = Zstruct(
            **{f"phi_{k}": basis_w.phi[k](z) for k in range(N_w + 1)})

        # 1. 3D INS, drop viscosity, split off the hydrostatic pressure.
        sys = FullINS(state)
        sys.apply(Inviscid(state)).simplify()
        p_NH = sp.Function("p_NH", real=True)(state.t, state.x, z)
        sys.apply({state.p: state.rho * state.g * (state.eta - z) + p_NH}
                  ).simplify()

        # 2. Project momentum against test functions.  Continuity is
        # left as a single scalar leaf — depth-integrating it directly
        # gives the mass evolution ``∂_t h + ∂_x(h U_0) = 0`` (no
        # need for redundant ``cont test_k`` projections).
        sys.momentum.x.apply(Multiply(test_phi_u, outer=True))
        sys.momentum.z.apply(Multiply(test_phi_w, outer=True))

        # 3. ProductRule auto-routes per term: ``coeff · ∂_z F →
        # ∂_z(coeff · F) − ∂_z(coeff) · F`` for any term containing a
        # ``∂_z`` derivative; everything else untouched.  Restricted to
        # ``z`` so we don't accidentally expand existing ``∂_x(u²)``
        # divergence forms back to chain-rule shape.
        sys.apply(ProductRule(variables=[z]))

        # 4. Depth-integrate (Leibniz on ∂_t / ∂_x; FT on ∂_z).
        sys.apply(Integrate(z, state.b, state.eta, method="auto"))

        # 5. Apply kinematic BCs at bottom + surface, drop ∂_t b
        # (static bottom).  Boundary ``u·w`` cross-terms cancel into
        # the conservative volume.
        sys.apply(InterfaceKBC(state, state.b)).simplify()
        sys.apply(InterfaceKBC(state, state.eta)).simplify()
        sys.apply({sp.Derivative(state.b, state.t): sp.S.Zero}).simplify()

        # 6. Surface BC for the non-hydrostatic pressure remainder
        # (applied at the field level so the integrand sees ``p_NH(η)=0``).
        sys.apply({p_NH.subs(z, state.eta): 0}).simplify()

        # 7. Affine map z → ζ = (z−b)/h on basis args, then expand
        # u / w / p_NH into modes.
        sys.apply(AffineProjection(state))
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

        # 9. Add the kinematic BCs as algebraic equations.  These
        # constraints close the system: continuity provides the ``h``
        # evolution, momentum projections evolve ``U_k`` / ``W_k``,
        # and the two KBCs fix the boundary ``w`` modes.
        u_at_b   = sum(coeffs_u[k] * basis_u.eval(k, sp.S.Zero)
                       for k in range(M + 1))
        u_at_eta = sum(coeffs_u[k] * basis_u.eval(k, sp.S.One)
                       for k in range(M + 1))
        w_at_b   = sum(coeffs_w[k] * basis_w.eval(k, sp.S.Zero)
                       for k in range(N_w + 1))
        w_at_eta = sum(coeffs_w[k] * basis_w.eval(k, sp.S.One)
                       for k in range(N_w + 1))
        h, b, eta = state.h, state.b, state.eta
        t, x = state.t, state.x
        # ``kbc_top`` originally carries ``∂_t η = ∂_t h`` (since
        # ``∂_t b = 0``).  Substitute via the mass equation
        # ``∂_t h = −∂_x(h·U_0)`` so the row becomes purely algebraic
        # — what Escalante calls ``kbc_top_alg``.
        sys.add_equation("kbc_top", sp.expand(
            w_at_eta
            - u_at_eta * sp.Derivative(eta, x).doit()
            + sp.Derivative(h * coeffs_u[0], x).doit()))
        sys.add_equation("kbc_bot", sp.expand(
            w_at_b - u_at_b * sp.Derivative(b, x).doit()))

        # 10. Surface BC for the non-hydrostatic pressure, applied at
        # the modal level: solve ``Σ_k φ_p_k(1) · P_k = 0`` for the
        # highest mode and substitute everywhere.  This eliminates one
        # pressure unknown without adding another row — the
        # ``surface_bc`` constraint is "consumed" rather than carried.
        p_at_eta = sum(coeffs_p[k] * basis_p.eval(k, sp.S.One)
                       for k in range(N_p + 1))
        p_top_sol = sp.solve(p_at_eta, coeffs_p[N_p])[0]
        sys.apply({coeffs_p[N_p]: p_top_sol}).simplify()

        # Stash + build PDESystem and SystemModel views.
        self._chain_system = sys
        self._chain_state = state
        self._chain_coeffs = {
            "u": coeffs_u, "w": coeffs_w, "p": coeffs_p[:N_p]
        }

        from zoomy_core.analysis import PDESystem
        from zoomy_core.model.models.system_model import SystemModel

        # Escalante-style row order: continuity (mass) first,
        # x-momentum, z-momentum, then the KBC constraints.
        ordered = [("mass", sys._tree.continuity)]
        for k in range(M + 1):
            ordered.append((f"xmom_j{k}",
                            getattr(sys._tree.momentum.x, f"test_{k}")))
        for k in range(N_w + 1):
            ordered.append((f"zmom_j{k}",
                            getattr(sys._tree.momentum.z, f"test_{k}")))
        ordered.append(("kbc_top", sys._tree.kbc_top))
        ordered.append(("kbc_bot", sys._tree.kbc_bot))
        self._chain_dae = PDESystem(
            equations=[leaf.expr for _, leaf in ordered],
            fields=[state.h] + coeffs_u + coeffs_w + coeffs_p[:N_p],
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
