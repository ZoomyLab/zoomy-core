"""Initial conditions carried in a mesh file (gmsh ``$NodeData`` / ``$ElementData``).

Many field solvers ship a survey mesh whose nodes carry the initial state as
gmsh node data (bathymetry, depth, velocity, …).  Historically every backend
hand-rolled the read + the node→cell interpolation + the field→state mapping
(e.g. the jax Malpasset case ``build_ic``).  This module lifts that pattern into
one backend-agnostic (pure-numpy, mesh-level) entry so *every* backend reads the
same way.

The node↔cell interpolation is defined **once** here:

* ``"nearest"`` — each cell takes the value of the mesh node nearest its centre
  (KD-tree over the vertices).  This reproduces the legacy hand-rolled readers.
* ``"cell_average"`` — each cell takes the mean of its own vertices' node values
  (mesh-connectivity average; smoother, no external node bleed).

The field→state mapping is expressed by a ``field_map`` (``state name → mesh
field name`` or ``state name → callable(node_fields) → node array``), so the
``B/H/U/V → [b, h, h·u, h·v]`` recipe is data, not code.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np

FieldSpec = Union[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]]


def _as_base_mesh(mesh_or_path):
    """Return a mesh carrying node/cell data, from a mesh object or a .msh path."""
    if isinstance(mesh_or_path, (str, bytes)) or hasattr(mesh_or_path, "__fspath__"):
        from zoomy_core.mesh.base_mesh import BaseMesh
        return BaseMesh.from_msh(str(mesh_or_path))
    return mesh_or_path


def interpolate_node_data_to_cells(
    mesh,
    node_values: np.ndarray,
    method: str = "nearest",
    query_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Interpolate a per-vertex field ``node_values`` onto cells.

    Parameters
    ----------
    mesh
        A ``BaseMesh`` (or subclass): needs ``vertex_coordinates``,
        ``cell_vertices``, ``dimension``, ``n_inner_cells`` and
        ``cell_centers_computed()``.
    node_values
        Shape ``(n_vertices,)`` (or ``(n_vertices, k)``), aligned to
        ``mesh.vertex_coordinates`` columns.
    method
        ``"nearest"`` (KD-tree over vertices; matches legacy hand-rolled
        readers) or ``"cell_average"`` (mean of each cell's own vertices).
    query_points
        Only for ``"nearest"``: points to sample, shape ``(n, dim)``.  Defaults
        to the inner cell centres.

    Returns
    -------
    np.ndarray
        Shape ``(n_inner_cells,)`` (or ``(n_inner_cells, k)``).
    """
    node_values = np.asarray(node_values, dtype=float)
    dim = mesh.dimension
    n_inner = mesh.n_inner_cells

    if method == "cell_average":
        # Mesh-connectivity average: cell value = mean of its vertices.
        cell_verts = np.asarray(mesh.cell_vertices)[:, :n_inner]  # (vpc, n_inner)
        return node_values[cell_verts].mean(axis=0)

    if method == "nearest":
        from scipy.spatial import cKDTree
        verts = np.asarray(mesh.vertex_coordinates)[:dim, :].T  # (n_vertices, dim)
        if query_points is None:
            centers = mesh.cell_centers_computed()[:dim, :n_inner].T  # (n_inner, dim)
        else:
            centers = np.asarray(query_points, dtype=float)[:, :dim]
        tree = cKDTree(verts)
        _, idx = tree.query(centers)
        return node_values[idx]

    raise ValueError(f"unknown interpolation method {method!r} "
                     "(expected 'nearest' or 'cell_average')")


def initial_conditions_from_mesh(
    mesh_or_path,
    field_map: Mapping[str, FieldSpec],
    state_names: Optional[Sequence[str]] = None,
    method: str = "nearest",
    default: float = 0.0,
    query_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a per-cell initial-state array from mesh-carried node/cell data.

    Generalises the per-backend ``build_ic``: parses the gmsh ``$NodeData`` (and
    ``$ElementData`` if present, exposed as ``mesh.node_data`` / ``mesh.cell_data``),
    interpolates each referenced field node→cell **once** (:func:`interpolate_node_data_to_cells`),
    and assembles them into the solver's state layout via ``field_map``.

    Parameters
    ----------
    mesh_or_path
        A ``BaseMesh`` carrying ``node_data`` / ``cell_data``, or a path to a
        ``.msh`` file (loaded via ``BaseMesh.from_msh``).
    field_map
        ``{state_name: mesh_field_name}`` or ``{state_name: callable(node_fields)}``.
        A callable receives the dict of per-node field arrays and returns a
        per-node array (e.g. ``lambda f: f["H"] * f["U"]`` for ``h·u``); it is
        evaluated at nodes, then interpolated to cells — so ``"nearest"`` is
        bit-identical to picking the fields at the same nearest node.
    state_names
        Ordered state variable names.  Row ``k`` of the result is
        ``state_names[k]``; entries absent from ``field_map`` are set to
        ``default``.  If ``None``, the rows follow ``field_map`` insertion order.
    method
        Node→cell interpolation, see :func:`interpolate_node_data_to_cells`.
    default
        Fill value for state rows not named in ``field_map``.
    query_points
        Optional explicit sample points for ``"nearest"`` (default: inner cell
        centres).

    Returns
    -------
    np.ndarray
        Shape ``(n_state, n_cells)`` with ``n_cells`` the number of inner cells
        (or ``len(query_points)``).
    """
    mesh = _as_base_mesh(mesh_or_path)
    node_fields: Dict[str, np.ndarray] = dict(getattr(mesh, "node_data", {}) or {})
    if not node_fields:
        raise ValueError(
            "mesh carries no node data ($NodeData); nothing to build an IC from. "
            "Load the .msh via BaseMesh.from_msh so mesh.node_data is populated.")

    if state_names is None:
        names = list(field_map.keys())
    else:
        names = [str(s) for s in state_names]
    idx = {n: k for k, n in enumerate(names)}

    if query_points is None:
        n_cells = mesh.n_inner_cells
    else:
        n_cells = len(query_points)
    Q = np.full((len(names), n_cells), float(default), dtype=float)

    for state_name, spec in field_map.items():
        if state_name not in idx:
            raise KeyError(
                f"field_map targets state {state_name!r} not in state_names "
                f"{names}")
        if callable(spec):
            node_arr = np.asarray(spec(node_fields), dtype=float)
        else:
            if spec not in node_fields:
                raise KeyError(
                    f"mesh field {spec!r} not in node data {list(node_fields)}")
            node_arr = np.asarray(node_fields[spec], dtype=float)
        Q[idx[state_name]] = interpolate_node_data_to_cells(
            mesh, node_arr, method=method, query_points=query_points)

    return Q
