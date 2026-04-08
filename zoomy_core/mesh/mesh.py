"""DEPRECATED — use zoomy_core.mesh.lsq_reconstruction or the mesh hierarchy directly.

This module re-exports LSQ utilities for backward compatibility with notebooks
and scripts.  The old Mesh class has been moved to zoomy_core.mesh.legacy.mesh.
New code should import from zoomy_core.mesh (BaseMesh, FVMMesh, LSQMesh) and
zoomy_core.mesh.lsq_reconstruction.
"""

# Re-export all LSQ utilities so existing `from zoomy_core.mesh.mesh import ...` works
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

# Legacy Mesh class — import from legacy subpackage
try:
    from zoomy_core.mesh.legacy.mesh import Mesh  # noqa: F401
except ImportError:
    pass
