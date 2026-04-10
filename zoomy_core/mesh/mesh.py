"""Re-exports LSQ utilities for backward compatibility.

New code should import from zoomy_core.mesh (BaseMesh, FVMMesh, LSQMesh).
"""

from zoomy_core.mesh.lsq_reconstruction import (  # noqa: F401
    build_monomial_indices,
    build_vandermonde,
    compute_derivatives,
    compute_gaussian_weights,
    compute_inradius_generic as _compute_inradius_generic,
    expand_neighbors,
    find_derivative_indices,
    get_physical_boundary_labels,
    get_polynomial_degree,
    get_required_monomials_count,
    least_squares_reconstruction_local,
    scale_lsq_derivative,
)
