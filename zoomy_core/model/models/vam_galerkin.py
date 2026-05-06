"""VAMModelGalerkin — VAM derived via the explicit Galerkin chain.

The class has exactly one job per the user's preferred shape:

  1. Inherit from :class:`VAMModel` (the base) so the solver-compatible
     state structure ``[b, h, hu_k, hw_k, hp_k]`` and operator-API
     surface stay populated for downstream consumers.
  2. ``derive_model`` is **linear**: a sequence of primitive
     applications + equation additions, no nested helpers, no big
     orchestration methods.  Everything is visible top-to-bottom.

The primitives the chain consumes — ``Multiply``, ``ProductRule``,
``Integrate``, ``InterfaceKBC``, ``AffineProjection``, ``Expand``,
``EvaluateIntegrals`` — already exist; we apply them in order.
Adding the algebraic constraints (mass / KBC / surface BC) is just
``System.add_equation(name, expr)`` calls — no new primitives, the
system's tree already supports add / remove.

Outputs stored on the model:

* ``_chain_intermediate`` — snapshot after the ``Expand`` step and
  before ``EvaluateIntegrals``.  Leaves carry ``sp.Sum`` atoms;
  ``describe()`` renders them as paper-form ``Σ_k U_k φ_k(ζ)``.
* ``_chain_system`` — the closed augmented system: 9 equations for
  ``(M=1, N_w=2, N_p=2)`` (Escalante 2024 eq (4)+(5)).
* ``_chain_dae`` — same equations packaged as a
  :class:`zoomy_core.analysis.PDESystem` so analysis routines and
  ``SystemModel.from_pdesystem`` can consume it.
* ``_chain_dae_systemmodel`` — operator-form SystemModel built from
  the chain DAE.  Mass matrix is zero on algebraic rows; flux / NCP /
  source / hydrostatic_pressure are zero placeholders awaiting
  per-term tagging.
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
    """VAM derived via the explicit symbolic Galerkin chain.

    Parameters
    ----------
    level : int
        Default basis level for ``u``-modes (``M`` defaults to ``level``).
    M, N_w, N_p : int, optional
        Asymmetric basis levels per Escalante 2024.  Defaults:
        ``M = level``, ``N_w = N_p = M + 1``.
    **kwargs
        Forwarded to :class:`VAMModel`.
    """

    def __init__(self, level=0, *, M=None, N_w=None, N_p=None, **kwargs):
        # ``VAMModel.__init__`` calls ``self.derive_model()`` *before*
        # ``Model.__init__`` populates ``self.level`` via param, so we
        # have to stash the chain levels on the instance up front.
        self._chain_M = M if M is not None else level
        self._chain_N_w = N_w if N_w is not None else self._chain_M + 1
        self._chain_N_p = N_p if N_p is not None else self._chain_M + 1
        super().__init__(level=level, **kwargs)

    # ------------------------------------------------------------------
    # The derivation, written linearly.
    # ------------------------------------------------------------------

    def derive_model(self):
        # Parent VAMModel populates the inherited operator-API path
        # (flux / NCP / source via the basis-matrix machinery) — the
        # solver consumes that; we leave it untouched and add the
        # explicit Galerkin chain on top.
        super().derive_model()

        M = self._chain_M
        N_w = self._chain_N_w
        N_p = self._chain_N_p

        # State + bases + coefficients (data; not primitives).
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

        # 1. Build the 3D INS system, drop viscosity, split pressure.
        sys = FullINS(state)
        sys.apply(Inviscid(state)).simplify()
        p_NH = sp.Function("p_NH", real=True)(state.t, state.x, z)
        sys.apply({state.p: state.rho * state.g * (state.eta - z) + p_NH}
                  ).simplify()

        # 2. Project against test functions.  Continuity gets ``N_w + 1``
        # children (the higher k's are Escalante's ``I_1`` / ``I_2``
        # constraints); x-momentum gets ``M + 1``; z-momentum gets
        # ``N_w + 1``.
        sys.continuity.apply(Multiply(test_phi_w, outer=True))
        sys.momentum.x.apply(Multiply(test_phi_u, outer=True))
        sys.momentum.z.apply(Multiply(test_phi_w, outer=True))

        # 3. Inverse product rule on the single ``∂_z`` term in each
        # leaf — converts ``phi·∂_z(uw) → ∂_z(phi·uw) − ∂_z(phi)·uw``
        # so ``Integrate``'s fundamental-theorem path handles it.
        for k in range(M + 1):
            self._inverse_product_rule_on_z(
                getattr(sys.momentum.x, f"test_{k}"), z)
        for k in range(N_w + 1):
            self._inverse_product_rule_on_z(
                getattr(sys.continuity, f"test_{k}"), z)
            self._inverse_product_rule_on_z(
                getattr(sys.momentum.z, f"test_{k}"), z)

        # 4. Depth-integrate.  Leibniz on ∂_t / ∂_x; fundamental theorem
        # on ∂_z.  Boundary atoms at z=b and z=b+h emerge.
        sys.apply(Integrate(z, state.b, state.eta, method="auto"))

        # 5. Apply kinematic BCs at the bottom and the surface, then
        # drop ∂_t b (static bottom).  This absorbs all the boundary
        # ``u·w`` cross-terms into the conservative volume — what
        # survives is ``∂_t (h U_k) + ∂_x (h U_k …)``.
        sys.apply(InterfaceKBC(state, state.b)).simplify()
        sys.apply(InterfaceKBC(state, state.eta)).simplify()
        sys.apply({sp.Derivative(state.b, state.t): sp.S.Zero}).simplify()

        # 6. Surface BC for the non-hydrostatic pressure remainder.
        sys.apply({p_NH.subs(z, state.eta): 0}).simplify()

        # 7. Affine map z → ζ = (z−b)/h on basis arguments, then
        # substitute the polynomial ansatz for u / w / p_NH.
        sys.apply(AffineProjection(state))
        sys.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u,
                         state=state))
        sys.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w,
                         state=state))
        sys.apply(Expand(p_NH,    basis=basis_p, coefficients=coeffs_p,
                         state=state))

        # Snapshot the Sum-form intermediate (paper notation) before we
        # resolve the polynomial integrals.  ``deepcopy`` is fine — the
        # System carries only sympy / Zstruct / Expression objects.
        self._chain_intermediate = copy.deepcopy(sys)

        # 8. Resolve the ζ integrals using the basis cache.  Leaves are
        # now polynomial in ``(h, U_k, W_k, P_k, b)``; no integrals or
        # held sums remain.
        sys.apply(EvaluateIntegrals(state)).simplify()

        # 9. Augment with the algebraic DAE rows Escalante 2024 eq (4)+
        # (5) carries alongside the projections.  Drop the continuity
        # projections that are redundant with these rows
        # (``test_0`` → replaced by ``mass``; ``test_M..test_{N_w}`` →
        # trivially zero / not retained).  No new primitives — every
        # equation is added with ``System.add_equation(name, expr)``.
        h, b, eta = state.h, state.b, state.eta
        t, x = state.t, state.x
        u_at_b   = sum(coeffs_u[k] * basis_u.eval(k, sp.S.Zero)
                       for k in range(M + 1))
        u_at_eta = sum(coeffs_u[k] * basis_u.eval(k, sp.S.One)
                       for k in range(M + 1))
        w_at_b   = sum(coeffs_w[k] * basis_w.eval(k, sp.S.Zero)
                       for k in range(N_w + 1))
        w_at_eta = sum(coeffs_w[k] * basis_w.eval(k, sp.S.One)
                       for k in range(N_w + 1))
        p_at_eta = sum(coeffs_p[k] * basis_p.eval(k, sp.S.One)
                       for k in range(N_p + 1))

        sys.remove_equation(("continuity", "test_0"))
        for k in range(M, N_w + 1):
            sys.remove_equation(("continuity", f"test_{k}"))

        sys.add_equation("mass", sp.expand(
            sp.Derivative(h, t) + sp.Derivative(h * coeffs_u[0], x).doit()))
        sys.add_equation("kbc_top_alg", sp.expand(
            w_at_eta
            - u_at_eta * sp.Derivative(eta, x).doit()
            + sp.Derivative(h * coeffs_u[0], x).doit()))
        sys.add_equation("kbc_bot", sp.expand(
            w_at_b - u_at_b * sp.Derivative(b, x).doit()))
        sys.add_equation("surface_bc", sp.expand(p_at_eta))

        # Stash the closed augmented system + its scaffolding.  The
        # PDESystem and SystemModel views are derived from these.
        self._chain_system = sys
        self._chain_state = state
        self._chain_coeffs = {"u": coeffs_u, "w": coeffs_w, "p": coeffs_p}

        # PDESystem view (Escalante row order) + SystemModel.
        from zoomy_core.analysis import PDESystem
        from zoomy_core.model.models.system_model import SystemModel

        ordered = [("mass", sys._tree.mass)]
        for k in range(M + 1):
            ordered.append((f"xmom_j{k}",
                            getattr(sys._tree.momentum.x, f"test_{k}")))
        for k in range(N_w + 1):
            ordered.append((f"zmom_j{k}",
                            getattr(sys._tree.momentum.z, f"test_{k}")))
        ordered.append(("kbc_top_alg", sys._tree.kbc_top_alg))
        ordered.append(("kbc_bot", sys._tree.kbc_bot))
        ordered.append(("surface_bc", sys._tree.surface_bc))
        for k in range(1, M):  # cont_j1..M-1 (Escalante eq (5) I_k)
            ordered.append((f"cont_j{k}",
                            getattr(sys._tree.continuity, f"test_{k}")))
        self._chain_dae = PDESystem(
            equations=[leaf.expr for _, leaf in ordered],
            fields=[state.h] + coeffs_u + coeffs_w + coeffs_p,
            time=state.t,
            space=[state.x],
            parameters={state.g: state.g},
        )
        self._chain_dae.equation_names = [n for n, _ in ordered]
        self._chain_dae_systemmodel = SystemModel.from_pdesystem(
            self._chain_dae)

    # ------------------------------------------------------------------
    # Tiny helper for the per-leaf ProductRule call.
    # ------------------------------------------------------------------

    @staticmethod
    def _inverse_product_rule_on_z(leaf_proxy, z):
        """Apply ``ProductRule(direction='inverse', variables=[z])`` to
        the single term of ``leaf_proxy`` whose ``Derivative`` is w.r.t.
        ``z`` — typically the ``phi · ∂_z(uw)`` advection term left over
        from ``Inviscid`` after Galerkin testing.

        ``ProductRule`` is ``single_term_only`` — the chain author picks
        the term, the operation transforms it.  No "intelligence": once
        the term is identified, the rewrite is unconditional.
        """
        leaf = leaf_proxy._node
        pr = ProductRule(direction="inverse", variables=[z])
        for i, term in enumerate(leaf.terms):
            if any(
                isinstance(d, sp.Derivative)
                and len(d.variables) == 1
                and d.variables[0] == z
                for d in term.expr.atoms(sp.Derivative)
            ):
                leaf_proxy.apply_to_term(i, pr)
                break

    # ------------------------------------------------------------------
    # Display.
    # ------------------------------------------------------------------

    def describe(self, **kwargs):
        """Render the augmented chain (closed equations + DAE rows)."""
        return self._chain_system.describe(**kwargs)

    def _repr_markdown_(self):
        return self._chain_system._repr_markdown_()

    def describe_chain_intermediate(self):
        """Render the Sum-form intermediate (post-Expand, pre-Evaluate).

        Leaves carry ``sp.Sum`` atoms — useful for showing the
        paper-notation ``Σ_k U_k φ_k(ζ)`` form.
        """
        return self._chain_intermediate.describe()

    def describe_chain_closed(self):
        """Same as :meth:`describe`; kept for symmetry."""
        return self._chain_system.describe()


__all__ = ["VAMModelGalerkin"]
