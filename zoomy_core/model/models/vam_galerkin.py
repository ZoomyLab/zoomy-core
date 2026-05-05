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

    def __init__(self, level=0, **kwargs):
        # ``VAMModel.__init__`` calls ``self.derive_model()`` *before*
        # ``Model.__init__`` populates ``self.level`` via param — so
        # ``_build_chain``'s ``getattr(self, "level", 0)`` would silently
        # collapse every call to L=0.  Stash it on the instance up front
        # so the chain reads the actual requested level.
        self._chain_level = level
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
        L = self._chain_level
        # Run the chain twice — once stop right after Expand (gives us
        # an intermediate System with un-evaluated ``sp.Sum`` atoms,
        # ``describe`` renders the paper-form Σ_k U_k φ_k(ζ)), once
        # all the way through ``EvaluateIntegrals`` (closed form).  The
        # double-run is cheap (a few seconds for L≤2) and lets each
        # System carry its own ``describe`` machinery.
        self._chain_intermediate = self._run_chain(L, stop="expand")
        self._chain_system = self._run_chain(L, stop="full")

    @staticmethod
    def _run_chain(level: int, *, stop: str):
        state = StateSpace(dimension=2)
        z = state.z

        basis_u = Legendre_shifted(level=level, symbol="phi")
        basis_w = Legendre_shifted(level=level, symbol="eta")
        basis_p = Legendre_shifted(level=level, symbol="mu")

        coeffs_u = [sp.Function(f"U_{k}", real=True)(state.t, state.x)
                    for k in range(level + 1)]
        coeffs_w = [sp.Function(f"W_{k}", real=True)(state.t, state.x)
                    for k in range(level + 1)]
        coeffs_p = [sp.Function(f"P_{k}", real=True)(state.t, state.x)
                    for k in range(level + 1)]

        test_phi = Zstruct(
            **{f"phi_{k}": basis_u.phi[k](z) for k in range(level + 1)}
        )

        sys = FullINS(state)
        sys.apply(Inviscid(state)).simplify()
        sys.momentum.x.apply(Multiply(test_phi, outer=True))
        sys.momentum.z.apply(Multiply(test_phi, outer=True))

        # Convert ``phi(z) · ∂_z F`` terms into a form ``Integrate``'s
        # fundamental-theorem path can use.  The chain author identifies
        # the term (here: the only one with a ∂_z derivative — the
        # vertical advection ``∂_z(uw)`` from ``Inviscid``) and applies
        # ProductRule(inverse) to rewrite as
        # ``∂_z(phi·F) − ∂_z(phi)·F``.  The first piece is in
        # conservative form (FT-friendly); the second piece is a
        # regular volume integrand that ``Integrate`` handles directly.
        for momdir in ("x", "z"):
            for k in range(level + 1):
                leaf_proxy = getattr(getattr(sys.momentum, momdir),
                                     f"test_{k}")
                VAMModelGalerkin._inverse_product_rule_on_z(leaf_proxy, z)

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

        sys.apply({state.p.subs(z, state.eta): 0}).simplify()
        sys.apply(AffineProjection(state))
        sys.apply(Expand(state.u, basis=basis_u, coefficients=coeffs_u,
                         state=state))
        sys.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w,
                         state=state))
        sys.apply(Expand(state.p, basis=basis_p, coefficients=coeffs_p,
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
