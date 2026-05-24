"""ScalarPoissonGalerkin — scalar Poisson FEM weak form via the
explicit symbolic Galerkin chain.

Parallel to :class:`VAMModelGalerkin`: inherit ``Model``, write
``derive_model`` linearly as a sequence of ``.apply(...)`` calls
plus snapshots, no nested helpers.

Outputs stored on the model:

* ``_strong_form``       — Expression with ``Δu + f`` (positive sign
  convention so multiplying by ``φ`` gives a clean ``φ·Δu`` term
  for IBP).
* ``_after_multiply``    — Expression with ``φ·(Δu + f)``.
* ``_after_integrate``   — Expression with ``∫_K φ·(Δu + f) d**x**``.
* ``_after_div_thm``     — Expression with the IBP'd form
  ``-∫_K ∇φ·∇u + ∮_∂K φ·∇u·n + ∫_K φ·f``.
* ``_weak_form``         — final Expression in reference-element
  coordinates ``ξ_0, ξ_1`` with the matrix Jacobian ``|det B|``
  factored.

Each stage is built by ``.apply(...)``-ing a single Operation, so
``model.describe()`` (and every snapshot's ``.describe()``) renders the
intermediate state as LaTeX.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.basemodel import Model
from zoomy_core.model.models.ins_generator import Expression, Multiply
from zoomy_core.model.models.legacy.integrate_over_domain import IntegrateOverDomain
from zoomy_core.model.models.legacy.divergence_theorem import DivergenceTheorem
from zoomy_core.model.models.legacy.map_to_reference import MapToReferenceElement
from zoomy_core.symbolic.domains import Simplex


class ScalarPoissonGalerkin(Model):
    """Scalar Poisson on a triangle, weak form via the symbolic chain."""

    variables = ["u"]
    parameters = {}
    dimension = 2

    def __init__(self, *, vertices=None, **kwargs):
        # Symbolic vertices by default — user can pass concrete tuples.
        if vertices is None:
            x0, y0, x1, y1, x2, y2 = sp.symbols(
                "x0 y0 x1 y1 x2 y2", real=True)
            vertices = [(x0, y0), (x1, y1), (x2, y2)]
        self._vertices = vertices
        super().__init__(**kwargs)
        self.derive_model()

    def derive_model(self):
        x, y = sp.symbols("x y", real=True)
        u   = sp.Function("u",   real=True)(x, y)
        f   = sp.Function("f",   real=True)(x, y)
        phi = sp.Function("phi", real=True)(x, y)
        K = Simplex(self._vertices, coords=(x, y), name="K")
        F = (sp.diff(u, x), sp.diff(u, y))

        # 1. Strong form:  Δu + f = 0.  (Sign convention chosen so that
        #    multiplying by φ produces a clean φ·Δu = φ·∇·F term that
        #    DivergenceTheorem matches without sign juggling.)
        sys = Expression(sp.diff(u, x, 2) + sp.diff(u, y, 2) + f,
                         name="poisson")
        self._strong_form = sys

        # 2. Multiply by the test function.
        sys = sys.apply(Multiply(phi))
        self._after_multiply = sys

        # 3. Integrate over the (multi-D) domain.
        sys = sys.apply(IntegrateOverDomain(K))
        self._after_integrate = sys

        # 4. Divergence theorem (component-wise scan): IBPs every
        #    ``φ·∂_i F[i]`` component, leaves the source term ``φ·f``
        #    inside its own Integral, sums boundary contributions into
        #    one BoundaryIntegral on ∂K.
        sys = sys.apply(DivergenceTheorem(K, phi=phi, F=F, form="weighted"))
        self._after_div_thm = sys

        # 5. Affine map K → reference simplex T̂.  Substitutes coords,
        #    chain-rules first-order Derivatives via B⁻ᵀ, multiplies
        #    each volume Integrand by |det B|, and updates the boundary
        #    atom's domain to ∂T̂.
        sys = sys.apply(MapToReferenceElement(K))
        self._weak_form = sys

        # Convenience: expose the building blocks the user might want
        # to inspect or feed into a follow-up step (e.g. multi-dim
        # basis projection, once that layer exists).
        self.K = K
        self.u, self.f, self.phi = u, f, phi
        self.F = F

    def describe(self, **kwargs):
        """Render the final reference-element weak form as markdown."""
        return self._weak_form.describe(**kwargs)
