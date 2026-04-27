"""Generic linear-analysis tools for shallow-water-family PDE systems.

The unified entry point is :class:`PDESystem` — a thin wrapper around a
list of sympy equations (LHS = 0), the state fields, and the
coordinate symbols.  Every analysis routine accepts a ``PDESystem``
and never inspects model-specific attributes.

This package replaces the model-bound interfaces of
``zoomy_core.model.analysis_linear`` and the basemodel-coupled
``eigenvalues()``/``quasilinear_matrix()`` paths for the *analysis*
half of the workflow.  Models may continue to use those for
derivation; once an equation set is in hand, this package handles:

* Linearisation around a base state.
* Plane-wave dispersion analysis (`ω(k)` solutions).
* Generalised-eigenvalue ("pencil") form for systems with constraints.
* Numerical eigenvalue sampling for hyperbolicity over a parameter
  cube.

The library knows nothing about VAM, SME, ML-SWE, etc. — every
model-specific bit is in tutorials/.
"""
from .pde_system import PDESystem
from .linearisation import linearise
from .plane_wave import plane_wave_dispersion, plane_wave_matrix
from .pencil import (
    extract_quasilinear_pencil,
    generalised_eigenvalues,
    sample_generalised_eigenvalues,
    symbolic_eigenvalues_at,
)
from .reduce_pencil import reduce_singular_pencil
from .hyperbolicity import (
    is_hyperbolic_at,
    sample_hyperbolicity,
)
from .plotting import (
    plot_dispersion,
    plot_hyperbolic_region_2d,
)

__all__ = [
    "PDESystem",
    "linearise",
    "plane_wave_dispersion",
    "plane_wave_matrix",
    "extract_quasilinear_pencil",
    "generalised_eigenvalues",
    "sample_generalised_eigenvalues",
    "symbolic_eigenvalues_at",
    "reduce_singular_pencil",
    "is_hyperbolic_at",
    "sample_hyperbolicity",
    "plot_dispersion",
    "plot_hyperbolic_region_2d",
]
