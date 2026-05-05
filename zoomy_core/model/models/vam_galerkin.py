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
        L = getattr(self, "level", 0)
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
        sys.apply(Integrate(z, state.b, state.eta, method="auto"))
        sys.apply(InterfaceKBC(state, state.b)).simplify()
        sys.apply(InterfaceKBC(state, state.eta)).simplify()
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

    # ── Convenience: render the chain artefacts ────────────────────────

    def describe_chain_intermediate(self):
        """Markdown rendering of the chain intermediate (post-Expand,
        pre-EvaluateIntegrals) — leaves carry ``sp.Sum`` atoms,
        rendering as paper-form ``Σ_k U_k φ_k(ζ)``.
        """
        return self._chain_intermediate.describe()

    def describe_chain_closed(self):
        """Markdown rendering of the closed primitive-form equations
        after ``EvaluateIntegrals``.
        """
        return self._chain_system.describe()


__all__ = ["VAMModelGalerkin"]
