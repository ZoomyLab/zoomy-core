"""Least-squares polynomial reconstruction for FVM derivative estimation.

Pure-numpy utilities: monomial construction, Vandermonde matrices,
weighted least-squares stencils, and cell-wise derivative computation.
"""

from __future__ import annotations

from itertools import product
from math import comb, factorial

import numpy as np


# ── Monomial helpers ──────────────────────────────────────────────────────────

def build_monomial_indices(degree: int, dim: int):
    """Build monomial multi-indices up to total degree (excluding constant)."""
    mon_indices = []
    for powers in product(range(degree + 1), repeat=dim):
        if 0 < sum(powers) <= degree:
            mon_indices.append(powers)
    return mon_indices


def scale_lsq_derivative(mon_indices):
    """Factorial scaling so LSQ coefficients equal derivatives."""
    return np.array(
        [np.prod([factorial(k) for k in mi]) for mi in mon_indices]
    )


def find_derivative_indices(full_monomials_arr, requested_derivs_arr):
    """Map requested derivative multi-indices to positions in monomial list.

    Returns shape (M,) array; -1 where a requested derivative is not found.
    """
    full_monomials_arr = np.array(full_monomials_arr)
    requested_derivs_arr = np.array(requested_derivs_arr)
    matches = np.all(
        full_monomials_arr[:, None, :] == requested_derivs_arr[None, :, :],
        axis=-1,
    )
    found = np.any(matches, axis=0)
    indices = np.argmax(matches, axis=0)
    return np.where(found, indices, -1)


def get_polynomial_degree(mon_indices):
    return max(sum(m) for m in mon_indices)


def get_required_monomials_count(degree, dim):
    return comb(int(degree + dim), int(dim))


# ── Vandermonde & weights ─────────────────────────────────────────────────────

def build_vandermonde(cell_diffs, mon_indices):
    n_neighbors, dim = cell_diffs.shape
    n_monomials = len(mon_indices)
    V = np.zeros((n_neighbors, n_monomials))
    for i, powers in enumerate(mon_indices):
        V[:, i] = np.prod(cell_diffs ** powers, axis=1)
    return V


def expand_neighbors(neighbors_list, initial_neighbors):
    expanded = set(initial_neighbors)
    for n in initial_neighbors:
        expanded.update(neighbors_list[n])
    return list(expanded)


def compute_gaussian_weights(dX, sigma=1.0):
    distances = np.linalg.norm(dX, axis=1)
    return np.exp(-((distances / sigma) ** 2))


# ── Core LSQ reconstruction ──────────────────────────────────────────────────

def least_squares_reconstruction_local(
    n_cells, dim, neighbors_list, cell_centers, lsq_degree
):
    """Build per-cell LSQ gradient operators.

    Returns
    -------
    A_glob : (n_cells, max_neighbors, n_monomials)
    neighbors_array : (n_cells, max_neighbors) int
    mon_indices : list of tuples
    """
    mon_indices = build_monomial_indices(lsq_degree, dim)
    degree = get_polynomial_degree(mon_indices)
    required_neighbors = get_required_monomials_count(degree, dim)

    neighbors_all = []
    for i_c in range(n_cells):
        current_neighbors = list(neighbors_list[i_c])
        while len(current_neighbors) < required_neighbors:
            new_neighbors = expand_neighbors(neighbors_list, current_neighbors)
            current_neighbors = list(set(new_neighbors) - {i_c})
        neighbors_all.append(current_neighbors)

    max_neighbors = max(len(nbrs) for nbrs in neighbors_all)
    A_glob = []
    neighbors_array = np.empty((n_cells, max_neighbors), dtype=int)

    for i_c in range(n_cells):
        current_neighbors = list(neighbors_all[i_c])
        n_nbr = len(current_neighbors)

        while n_nbr < max_neighbors:
            extended_neighbors = expand_neighbors(neighbors_list, current_neighbors)
            extended_neighbors = list(set(extended_neighbors) - {i_c})
            new_neighbors = [n for n in extended_neighbors if n not in current_neighbors]
            current_neighbors.extend(new_neighbors)
            n_nbr = len(current_neighbors)
            if n_nbr >= max_neighbors:
                break

        if len(current_neighbors) >= max_neighbors:
            trimmed_neighbors = current_neighbors[:max_neighbors]
        else:
            padding = [i_c] * (max_neighbors - len(current_neighbors))
            trimmed_neighbors = current_neighbors + padding

        neighbors_array[i_c, :] = trimmed_neighbors

        dX = np.zeros((max_neighbors, dim), dtype=float)
        for j, neighbor in enumerate(trimmed_neighbors):
            dX[j, :] = cell_centers[neighbor] - cell_centers[i_c]

        V = build_vandermonde(dX, mon_indices)
        weights = compute_gaussian_weights(dX)
        W = np.diag(weights)
        VW = W @ V
        alpha = dX.min() * 1e-8
        A_loc = np.linalg.pinv(VW.T @ VW + alpha * np.eye(VW.shape[1])) @ VW.T @ W
        A_glob.append(A_loc.T)

    A_glob = np.array(A_glob)
    return A_glob, neighbors_array, mon_indices


# ── Derivative computation ────────────────────────────────────────────────────

def compute_derivatives(u, mesh, derivatives_multi_index=None):
    """Cell-wise derivative estimates using the mesh LSQ stencil."""
    A_glob = mesh.lsq_gradQ
    neighbors = mesh.lsq_neighbors
    mon_indices = mesh.lsq_monomial_multi_index
    scale_factors = mesh.lsq_scale_factors

    if derivatives_multi_index is None:
        derivatives_multi_index = mon_indices
    indices = find_derivative_indices(mon_indices, derivatives_multi_index)

    out = np.zeros((A_glob.shape[0], len(derivatives_multi_index)), dtype=float)
    for i in range(A_glob.shape[0]):
        A_loc = A_glob[i]
        neighbor_idx = neighbors[i]
        u_i = u[i]
        u_neighbors = u[neighbor_idx]
        delta_u = u_neighbors - u_i
        out[i, :] = (scale_factors * (A_loc.T @ delta_u))[indices]

    return out


def get_physical_boundary_labels(filepath):
    """Extract physical boundary labels from a .msh file via meshio."""
    import meshio
    mesh = meshio.read(filepath)
    return {key: value[0] for key, value in mesh.field_data.items()}


def compute_inradius_generic(cell_center, face_centers, face_normals):
    """Inradius of a cell: shortest normal distance from center to any face."""
    inradius = np.inf
    for center, normal in zip(face_centers, face_normals):
        distance = np.abs(np.dot(center - cell_center, normal))
        inradius = min(inradius, distance)
    if inradius <= 0:
        inradius = np.array(face_centers - cell_center).min()
    return inradius
