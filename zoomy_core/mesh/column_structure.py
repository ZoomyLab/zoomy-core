"""Column structure for extruded meshes.

Provides vertical column mapping and numerical integration operations
for meshes built by ``BaseMesh.extrude_2d()`` or ``mesh_extrude``.

Cell indexing convention (from extrusion):
    cell_3d = iz * n_horizontal + i_horizontal

where ``iz`` is the layer index (0 = bottom, n_layers-1 = top) and
``i_horizontal`` is the horizontal cell index.
"""

from __future__ import annotations

import numpy as np


class ColumnStructure:
    """Vertical column mapping for extruded meshes.

    Precomputes column-cell mapping and layer weights for numerical
    vertical integration in sigma coordinates (zeta in [0, 1]).

    Parameters
    ----------
    n_horizontal : int
        Number of horizontal (inner) cells.
    n_layers : int
        Number of vertical layers.
    """

    def __init__(self, n_horizontal: int, n_layers: int):
        self.n_horizontal = n_horizontal
        self.n_layers = n_layers

        # Column cells: (n_horizontal, n_layers) global cell indices
        # column_cells[i_h, iz] = iz * n_horizontal + i_h
        self.column_cells = np.arange(
            n_horizontal * n_layers, dtype=int,
        ).reshape(n_layers, n_horizontal).T

        # Uniform layer weight in sigma coordinates: dzeta = 1 / n_layers
        self.d_zeta = 1.0 / n_layers

        # Layer midpoints in sigma coordinates (for function evaluation)
        self.zeta_midpoints = (np.arange(n_layers) + 0.5) / n_layers

    def integrate(self, Q: np.ndarray) -> np.ndarray:
        """Numerical vertical integral: integral_0^1 Q(zeta) dzeta.

        Midpoint rule over uniform layers.

        Parameters
        ----------
        Q : ndarray, shape (n_vars, n_3d_cells)
            Field values on the full extruded mesh.

        Returns
        -------
        ndarray, shape (n_vars, n_horizontal)
            Vertically integrated values per horizontal cell.
        """
        n_vars = Q.shape[0]
        result = np.zeros((n_vars, self.n_horizontal))
        for iz in range(self.n_layers):
            cells = self.column_cells[:, iz]
            result += Q[:, cells] * self.d_zeta
        return result

    def partial_integrate(self, Q: np.ndarray, from_top: bool = True) -> np.ndarray:
        """Running vertical integral per cell.

        Computes integral_zeta^1 Q(zeta') dzeta' (from_top=True) or
        integral_0^zeta Q(zeta') dzeta' (from_top=False).

        Parameters
        ----------
        Q : ndarray, shape (n_vars, n_3d_cells)
        from_top : bool
            If True, integrate from surface (zeta=1) downward.
            If False, integrate from bottom (zeta=0) upward.

        Returns
        -------
        ndarray, shape (n_vars, n_3d_cells)
            Running integral value at each cell.
        """
        n_vars = Q.shape[0]
        result = np.zeros_like(Q)
        running = np.zeros((n_vars, self.n_horizontal))

        if from_top:
            layer_range = range(self.n_layers - 1, -1, -1)
        else:
            layer_range = range(self.n_layers)

        for iz in layer_range:
            cells = self.column_cells[:, iz]
            running += Q[:, cells] * self.d_zeta
            result[:, cells] = running

        return result

    def scatter_to_column(self, Q_horiz: np.ndarray) -> np.ndarray:
        """Broadcast horizontal values to all layers in each column.

        Parameters
        ----------
        Q_horiz : ndarray, shape (n_vars, n_horizontal)

        Returns
        -------
        ndarray, shape (n_vars, n_3d_cells)
        """
        n_vars = Q_horiz.shape[0]
        Q_3d = np.zeros((n_vars, self.n_horizontal * self.n_layers))
        for iz in range(self.n_layers):
            cells = self.column_cells[:, iz]
            Q_3d[:, cells] = Q_horiz
        return Q_3d

    def gather_layer(self, Q: np.ndarray, iz: int) -> np.ndarray:
        """Extract values at a single layer.

        Parameters
        ----------
        Q : ndarray, shape (n_vars, n_3d_cells)
        iz : int
            Layer index (0 = bottom, n_layers-1 = top).

        Returns
        -------
        ndarray, shape (n_vars, n_horizontal)
        """
        cells = self.column_cells[:, iz]
        return Q[:, cells]
