"""VAMModelGalerkin — VAM derived via the explicit Galerkin chain.

Inherits from :class:`VAMModel` to keep the existing solver-compatible
state structure (``[b, h, hu_k, hw_k, hp_k]``, conservative form,
``M = I``) and the operator-API method bodies that downstream
runtimes depend on.  Adds:

1. A `derive_model` that **also** runs the explicit symbolic chain
   from ``zoomy_core.model.models.ins_generator`` —
   ``Multiply`` (Galerkin test) → ``Integrate`` (depth) →
   ``InterfaceKBC`` × 2 → atmospheric pressure → ``AffineProjection``
   → three ``Expand``s (u/w/p) → ``EvaluateIntegrals`` — producing a
   fully closed primitive-form derived `System` saved as
   ``self._chain_system`` for inspection via ``describe``.

2. ``self._chain_intermediate`` — the same derivation snapshot
   stopped right after the three ``Expand`` calls, before
   ``EvaluateIntegrals`` resolves them.  Carries the unevaluated
   ``sp.Sum`` form of the ansatz substitution so a walkthrough can
   render the paper-notation intermediate.

Today the chain output (primitive form: ``∂_t (U_k h) + …``) and the
conservative-form operator surface (``flux / NCP / source / mass``
extracted via ``DerivedModel`` machinery and consumed by solvers) are
two **mathematically equivalent** representations of the same model.
The chain documents the derivation; the inherited operator path
delivers the canonical M=I matrices the numerical solver needs.
A separate workstream will collapse them — re-doing the operator
extraction directly from the chain's tagged equations after a
primitive→conservative substitution + author-side tagging.  Until
then: derivation is shown, operators are correct, numerics run.
"""

from __future__ import annotations

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
    level, n_layers, basis_type, eigenvalue_mode, dimension, **kwargs
        Forwarded to :class:`VAMModel`.

    Attributes (added by ``derive_model``)
    --------------------------------------
    _chain_system : :class:`DerivedSystem`
        The fully closed primitive-form system after
        ``EvaluateIntegrals`` — every leaf is closed in the basis
        coefficients ``(U_k, W_k, P_k)`` with no held integrals or
        unevaluated sums.  ``self._chain_system.describe()`` renders
        the closed equations in LaTeX.
    _chain_intermediate : :class:`DerivedSystem`
        The same chain stopped right after the three ``Expand``
        operations, before ``EvaluateIntegrals``.  Each leaf carries
        ``sp.Sum`` atoms representing the un-evaluated ansatz —
        useful for walkthroughs that want to show the
        ``Σ_k U_k φ_k`` form on paper before resolving.
    _chain_basis : dict[str, :class:`Basisfunction`]
        ``{"u": …, "w": …, "p": …}`` — the three independent bases
        used in the chain.
    _chain_coefficients : dict[str, list[sp.Function call]]
        Pre-declared coefficient functions ``[U_0, U_1, …]`` etc.
    """

    def __init__(self, level=0, *, M=None, N_w=None, N_p=None, **kwargs):
        # ``VAMModel.__init__`` calls ``self.derive_model()`` *before*
        # ``Model.__init__`` populates ``self.level`` via param — so
        # ``_build_chain``'s ``getattr(self, "level", 0)`` would silently
        # collapse every call to L=0.  Stash the chain-relevant levels
        # on the instance up front so the chain reads them correctly.
        #
        # Asymmetric basis levels per Escalante 2024:
        #   ``M``   — horizontal velocity ``u`` modes (M+1 of them).
        #   ``N_w`` — vertical velocity   ``w`` modes (N_w+1 of them).
        #   ``N_p`` — non-hydrostatic     ``p`` modes (N_p+1 of them).
        # Defaults: ``M = level``, ``N_w = N_p = M + 1``.  Escalante's
        # canonical case is ``(M=1, N_w=2, N_p=2)``.
        self._chain_M = M if M is not None else level
        self._chain_N_w = N_w if N_w is not None else self._chain_M + 1
        self._chain_N_p = N_p if N_p is not None else self._chain_M + 1
        # Backwards-compat alias used by ``_build_chain`` legacy path.
        self._chain_level = self._chain_M
        super().__init__(level=level, **kwargs)

    def derive_model(self):
        # Run the parent VAMModel's derivation first so the inherited
        # operator-API path (flux / NCP / source / mass / hydrostatic
        # pressure via the basis-matrix machinery) is fully populated
        # — that's what downstream solvers consume today.
        super().derive_model()

        # Now also run the explicit Galerkin chain on a parallel
        # ``StateSpace``, capturing both the intermediate (after Expand)
        # and final (after EvaluateIntegrals) forms.  These are stored
        # on the model for inspection / walkthroughs; they don't feed
        # the operator API yet.
        self._build_chain()

    def _build_chain(self):
        M, N_w, N_p = self._chain_M, self._chain_N_w, self._chain_N_p
        # Run the chain twice — once stop right after Expand (gives us
        # an intermediate System with un-evaluated ``sp.Sum`` atoms,
        # ``describe`` renders the paper-form Σ_k U_k φ_k(ζ)), once
        # all the way through ``EvaluateIntegrals`` (closed form).
        self._chain_intermediate = self._run_chain(M, N_w, N_p, stop="expand")
        self._chain_system = self._run_chain(M, N_w, N_p, stop="full")

    @staticmethod
    def _run_chain(M: int, N_w: int, N_p: int, *, stop: str):
        state = StateSpace(dimension=2)
        z = state.z

        # Three independent bases — different levels per field per
        # Escalante 2024.  Distinct symbols (``phi``/``eta``/``mu``)
        # so the basis-aware machinery can route per-field.
        basis_u = Legendre_shifted(level=M,   symbol="phi")
        basis_w = Legendre_shifted(level=N_w, symbol="eta")
        basis_p = Legendre_shifted(level=N_p, symbol="mu")

        coeffs_u = [sp.Function(f"U_{k}", real=True)(state.t, state.x)
                    for k in range(M + 1)]
        coeffs_w = [sp.Function(f"W_{k}", real=True)(state.t, state.x)
                    for k in range(N_w + 1)]
        coeffs_p = [sp.Function(f"P_{k}", real=True)(state.t, state.x)
                    for k in range(N_p + 1)]

        # Test-function fans, one per equation type.  ``test_phi_u``
        # has M+1 entries and is used to project x-momentum (one
        # equation per u-mode).  ``test_phi_w`` has N_w+1 entries and
        # is used to project z-momentum AND continuity (the latter
        # gives the ``I_1`` / ``I_2`` constraints Escalante writes in
        # eq (5)).
        test_phi_u = Zstruct(
            **{f"phi_{k}": basis_u.phi[k](z) for k in range(M + 1)}
        )
        test_phi_w = Zstruct(
            **{f"phi_{k}": basis_w.phi[k](z) for k in range(N_w + 1)}
        )

        sys = FullINS(state)
        sys.apply(Inviscid(state)).simplify()

        # Hydrostatic pressure split applied *before* Multiply: at this
        # stage the equations only contain ``p(t, x, z)`` in their
        # volume form (no boundary evaluations yet from Leibniz / FT),
        # so a single ``Relation`` substitution fires uniformly.  After
        # the split the chain produces the ``g·h·∂_x η`` and ``−g``
        # contributions explicitly (the latter cancels the body-force
        # gravity in z-mom), and the field ``p_NH`` carries only the
        # non-hydrostatic remainder — matching Escalante 2024 eq (4)
        # which writes hydrostatic flux as ``g·h·∂_x η`` and ``p_k`` as
        # the non-hydrostatic moments.  Atmospheric pressure ``p_atm``
        # is implicitly absorbed by leaving it out of the split.
        p_NH = sp.Function("p_NH", real=True)(state.t, state.x, z)
        sys.apply(
            {state.p: state.rho * state.g * (state.eta - z) + p_NH}
        ).simplify()

        # Project equations against test functions:
        #   continuity   ↦ N_w + 1 children (test_0..test_{N_w})
        #   x-momentum   ↦ M  + 1 children (test_0..test_M)
        #   z-momentum   ↦ N_w + 1 children (test_0..test_{N_w})
        # The continuity projections beyond k=0 are the ``I_1``, ``I_2``
        # constraints Escalante writes in eq (5).  We keep them in the
        # system as separate equations rather than substituting them
        # into momentum — the resulting system has more rows than the
        # post-resolution Escalante form, but matches the structure
        # before constraint resolution.
        sys.continuity.apply(Multiply(test_phi_w, outer=True))
        sys.momentum.x.apply(Multiply(test_phi_u, outer=True))
        sys.momentum.z.apply(Multiply(test_phi_w, outer=True))

        # Convert ``phi(z) · ∂_z F`` terms into a form ``Integrate``'s
        # fundamental-theorem path can use.  The chain author identifies
        # the term (here: the only ones with a ∂_z derivative — the
        # vertical advection ``∂_z(uw)`` in x-momentum and ``∂_z w`` in
        # continuity) and applies ProductRule(inverse) to rewrite as
        # ``∂_z(phi·F) − ∂_z(phi)·F``.
        for k in range(M + 1):
            VAMModelGalerkin._inverse_product_rule_on_z(
                getattr(sys.momentum.x, f"test_{k}"), z
            )
        for k in range(N_w + 1):
            VAMModelGalerkin._inverse_product_rule_on_z(
                getattr(sys.continuity, f"test_{k}"), z
            )
            VAMModelGalerkin._inverse_product_rule_on_z(
                getattr(sys.momentum.z, f"test_{k}"), z
            )

        sys.apply(Integrate(z, state.b, state.eta, method="auto"))

        # Kinematic BCs are applied in their natural (forward)
        # direction: ``w|_interface → ∂_t interface + u·∂_x interface``.
        # This converts the ``w(b+h)`` / ``w(b)`` atoms produced by the
        # ∂_z fundamental theorem (introduced via the earlier
        # ProductRule(inverse) on ``phi·∂_z(uw)``) into ∂_t interface
        # forms.  Combined with the Leibniz time-boundaries (which
        # already carry ``-u·phi·∂_t interface``), the ∂_t-pieces
        # cancel cleanly in ``simplify``; only the conservative volume
        # term ``∂_t ∫ u·phi dz`` (closed to ``∂_t(h U_k)`` after
        # Expand+EvaluateIntegrals) survives.
        sys.apply(InterfaceKBC(state, state.b)).simplify()
        sys.apply(InterfaceKBC(state, state.eta)).simplify()

        # Static bottom: bathymetry is time-independent.  Applied AFTER
        # KBC so any ``Derivative(b, t)`` atoms introduced by KBC@b's
        # forward substitution (``w|_b → ∂_t b + u·∂_x b``) collapse to
        # zero in one step — turning the bottom KBC into ``w|_b = u·∂_x
        # b`` exactly where it appears.  Apply-it-once chain step, not
        # a structural assumption baked into the state.
        sys.apply({sp.Derivative(state.b, state.t): sp.S.Zero}).simplify()

        # Surface BC for the non-hydrostatic remainder: ``p_NH(η) = 0``
        # (the hydrostatic part contributes ``ρ·g·(η − η) = 0`` at the
        # surface trivially).
        sys.apply({p_NH.subs(z, state.eta): 0}).simplify()
        sys.apply(AffineProjection(state))
        sys.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u,
                         state=state))
        sys.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w,
                         state=state))
        sys.apply(Expand(p_NH, basis=basis_p, coefficients=coeffs_p,
                         state=state))
        if stop == "expand":
            return sys
        sys.apply(EvaluateIntegrals(state)).simplify()
        return sys

    @staticmethod
    def _inverse_product_rule_on_z(leaf_proxy, z):
        """Apply ``ProductRule(direction='inverse', variables=[z])`` to
        the single term of ``leaf_proxy`` whose ``Derivative`` is w.r.t.
        ``z`` — typically the ``phi · ∂_z(uw)`` advection term left over
        from ``Inviscid`` after Galerkin testing.

        ``ProductRule`` is ``single_term_only`` — the chain author picks
        the term, the operation transforms it.  Here we iterate
        ``leaf_proxy._node.terms`` and call ``apply_to_term`` on the
        first match.  No "intelligence": the rewrite is unconditional
        once the term is identified.
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

    # ── Display: model.describe() shows the chain's closed form ───────

    def describe(self, **kwargs):
        """Render the closed Galerkin chain (post-EvaluateIntegrals) —
        not the inherited VAMModel system.

        ``VAMModel.derive_model`` (called via ``super().derive_model()``)
        leaves ``self._system`` populated with the parent's
        depth-integrate-then-stop chain, whose leaves carry
        un-evaluated integrals like
        ``∂_x ∫_b^{b+h} u² dz``.  The parent renders fine for
        ``flux()`` / ``source()`` etc. only because those operator
        methods bypass ``_system`` entirely and consume hand-coded
        basis matrices instead.

        For display the chain's closed primitive-form equations are
        the right object — every integral is resolved, every term is
        polynomial in ``(h, U_k, W_k, P_k)``.
        """
        return self._chain_system.describe(**kwargs)

    def _repr_markdown_(self):
        return self._chain_system._repr_markdown_()

    # ── Convenience: render the chain artefacts ────────────────────────

    def describe_chain_intermediate(self):
        """Markdown rendering of the chain intermediate (post-Expand,
        pre-EvaluateIntegrals) — leaves carry ``sp.Sum`` atoms,
        rendering as paper-form ``Σ_k U_k φ_k(ζ)``.
        """
        return self._chain_intermediate.describe()

    def describe_chain_closed(self):
        """Markdown rendering of the closed primitive-form equations
        after ``EvaluateIntegrals``.  Same as :meth:`describe`; kept
        for symmetry with ``describe_chain_intermediate``.
        """
        return self._chain_system.describe()


__all__ = ["VAMModelGalerkin"]
