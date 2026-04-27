"""Shared building blocks for Galerkin-projection model derivations.

This package consolidates the pieces that every shallow-water-family
model (SWE, SME, ML-SWE, VAM, ML-VAM, …) duplicates: the σ-coord
coordinate symbols, the shifted Legendre basis, polynomial integration
in ``ξ ∈ [0, 1]``, and the σ-coord NS equations themselves.

Class structure (composable, not deeply inherited):

    FlowSetup                    — σ-coord NS in (t, x, ξ): continuity,
                                    x-momentum, z-momentum (each is a
                                    sympy expression returned by a
                                    method).  Stresses retained as
                                    user-supplied symbolic functions.
    HydrostaticFlow(FlowSetup)   — sets non-hydrostatic p = 0 and
                                    declares the z-momentum equation
                                    "absent" (callers should not project
                                    it).
    NonHydrostaticFlow(FlowSetup) — keeps z-momentum + non-hydrostatic
                                    pressure split.

    PolynomialAnsatz             — chooses which fields are polynomial
                                    in ξ at what degrees; provides the
                                    expanded ``u(ξ), w(ξ), p(ξ)``.
    GalerkinProjection           — projects the σ-coord momentum
                                    equations against ``φ_j(ξ)`` over
                                    [0, 1]; handles the ω(ξ)
                                    construction (depth-integrated for
                                    SME, opaque-w-as-state for VAM).

    KBCClosures                  — symbolic algebraic closures: KBC at
                                    ξ=0 (solve for ``w_N``) and surface
                                    BC ``p|_{ξ=1}=0`` (solve for ``p_N``).

Specific models become **compositions** of these pieces.  See
``tutorials/sme/sme_builder.py`` and ``tutorials/vam/escalante2024_generic.py``
for how the SME and VAM builders use them.
"""
from .coords import (
    default_coords,
    default_h,
    default_b,
)
from .basis import (
    shifted_legendre_basis,
    polynomial_integrate,
)
from .flow import (
    FlowSetup,
    HydrostaticFlow,
    NonHydrostaticFlow,
)
from .ansatz import (
    PolynomialAnsatz,
)
from .projection import (
    GalerkinProjection,
)
from .closures import (
    kbc_bottom_solve_w_N,
    surface_bc_solve_p_N,
)

__all__ = [
    # coords
    "default_coords",
    "default_h",
    "default_b",
    # basis
    "shifted_legendre_basis",
    "polynomial_integrate",
    # flow setups
    "FlowSetup",
    "HydrostaticFlow",
    "NonHydrostaticFlow",
    # ansatz + projection
    "PolynomialAnsatz",
    "GalerkinProjection",
    # closures
    "kbc_bottom_solve_w_N",
    "surface_bc_solve_p_N",
]
