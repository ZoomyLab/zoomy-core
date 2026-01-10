import numpy as np
from time import time as get_time
import sympy
from sympy import Matrix
import param
from typing import Callable, List

from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basefunction import Function


# --- Helper Function (Unchanged) ---
def _sympy_interpolate_data(time, timeline, data):
    assert timeline.shape[0] == data.shape[0]
    conditions = (((data[0], time <= timeline[0])),)
    for i in range(timeline.shape[0] - 1):
        t0 = timeline[i]
        t1 = timeline[i + 1]
        y0 = data[i]
        y1 = data[i + 1]
        conditions += (
            (-(time - t1) / (t1 - t0) * y0 + (time - t0) / (t1 - t0) * y1, time <= t1),
        )
    conditions += (((data[-1], time > timeline[-1])),)
    return sympy.Piecewise(*conditions)


# --- Base Class ---
class BoundaryCondition(param.Parameterized):
    """
    Default implementation. The required data for the 'ghost cell' is the data
    from the interior cell. Can be overwritten e.g. to implement periodic boundary conditions.
    """

    tag = param.String(default="bc")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        raise NotImplementedError(
            "BoundaryCondition is a virtual class. Use one of its derived classes!"
        )


# --- Derived Boundary Conditions ---


class Extrapolation(BoundaryCondition):
    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        return ZArray(Q)


class InflowOutflow(BoundaryCondition):
    prescribe_fields = param.Dict(default={})

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            # Note: eval() is risky but kept to match your specific requirement
            # If v represents a number string, direct assignment works.
            # If v is code, proceed with caution.
            try:
                val = float(v)
            except (ValueError, TypeError):
                val = eval(v)
            Qout[k] = val
        return Qout


class Lambda(BoundaryCondition):
    """
    Apply an arbitrary lambda function for boundary values.
    prescribe_fields: Dict mapping index -> callable(time, X, dX, Q, Qaux, parameters, normal)
    """

    prescribe_fields = param.Dict(default={})

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        Qout = ZArray(Q)
        for k, func in self.prescribe_fields.items():
            Qout[k] = func(time, X, dX, Q, Qaux, parameters, normal)
        return Qout


class FromData(BoundaryCondition):
    prescribe_fields = param.Dict(default={})
    timeline = param.Array(default=None)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        # Extrapolate all fields initially
        Qout = ZArray(Q)

        # Overwrite prescribed fields using interpolation
        for k, v in self.prescribe_fields.items():
            interp_func = _sympy_interpolate_data(time, self.timeline, v)
            # Linear extrapolation/ghost cell formula: Q_ghost = 2*Q_boundary - Q_interior
            Qout[k] = 2 * interp_func - Q[k]
        return Qout


class CharacteristicReflective(BoundaryCondition):
    """
    Generic characteristic reflective wall boundary condition.
    """

    R = param.Parameter(default=None)
    L = param.Parameter(default=None)
    D = param.Parameter(default=None)
    S = param.Parameter(default=None)
    M = param.Parameter(default=None)  # diagonal scaling matrix

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        q = Matrix(Q)

        # 1. Rotate Q into normal frame
        q_n = self.S @ q

        # 2. Project to characteristic space
        W_int = self.L @ q_n

        # 3. Build boundary characteristic state
        W_bc = W_int.copy()
        MW = self.M @ W_int
        for i in range(W_int.rows):
            lam = self.D[i, i]
            # Use SymPy logic for conditional
            cond = sympy.GreaterThan(-lam, 0, evaluate=False)
            W_bc[i, 0] = sympy.Function("conditional")(cond, MW[i, 0], W_int[i, 0])

        # 4. Transform back
        q_n_bc = self.R @ W_bc
        q_bc = sympy.simplify(self.S.inv() @ q_n_bc)

        out = ZArray.zeros(len(q_bc))
        # Apply depth thresholding if h (index 0 or 1) is small
        # Assuming index 0 is depth 'h' or similar variable
        for i in range(len(q_bc)):
            # This threshold logic seems specific to shallow water (h > 1e-4)
            # Consider parametrizing the index if this is a generic class
            out[i] = sympy.Function("conditional")(
                sympy.GreaterThan(q[0], 1e-4), q_bc[i, 0], q[i, 0]
            )

        return out


class Wall(BoundaryCondition):
    """
    permeability: float : 0.0 corresponds to a perfect reflection (impermeable wall)
    blending: float: 0.5 blend the reflected wall solution with the solution of the inner cell
    """

    momentum_field_indices = param.List(default=[[1, 2]])
    permeability = param.Number(default=0.0)
    wall_slip = param.Number(default=1.0)
    blending = param.Number(default=0.0)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        q = ZArray(Q)
        # Assuming normal is passed as a list or ZArray, convert to Matrix for dot product
        # Ensure we slice normal to match momentum dimension
        dim = len(self.momentum_field_indices[0])
        n_vec = Matrix(normal[:dim])

        out = ZArray(Q)  # Initialize with copy
        momentum_list_wall = []

        # Calculate reflected momentum for each set of indices
        for indices in self.momentum_field_indices:
            momentum = Matrix([q[k] for k in indices])

            normal_momentum_coef = momentum.dot(n_vec)
            transverse_momentum = momentum - normal_momentum_coef * n_vec

            momentum_wall = (
                self.wall_slip * transverse_momentum
                - (1 - self.permeability) * normal_momentum_coef * n_vec
            )
            momentum_list_wall.append(momentum_wall)

        # Apply results back to output array
        for indices, momentum_wall in zip(
            self.momentum_field_indices, momentum_list_wall
        ):
            for i, idx in enumerate(indices):
                out[idx] = (1 - self.blending) * momentum_wall[i] + self.blending * q[
                    idx
                ]
        return out


class RoughWall(Wall):
    CsW = param.Number(default=0.5)  # roughness constant
    Ks = param.Number(default=0.001)  # roughness height

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        # Calculate dynamic slip length
        slip_length = dX * sympy.ln((dX * self.CsW) / self.Ks)
        f = dX / slip_length
        wall_slip = (1 - f) / (1 + f)

        # Temporarily override wall_slip for calculation
        original_slip = self.wall_slip
        self.wall_slip = wall_slip

        res = super().compute_boundary_condition(
            time, X, dX, Q, Qaux, parameters, normal
        )

        # Restore state
        self.wall_slip = original_slip
        return res


class Periodic(BoundaryCondition):
    periodic_to_physical_tag = param.String(default="")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        return ZArray(Q)


# --- Container Class ---


class BoundaryConditions(param.Parameterized):
    """
    Container for all boundary conditions attached to a model.
    Maintains a sorted list of BCs to ensure deterministic C++ mapping.
    """

    boundary_conditions_list = param.List(default=[], item_type=BoundaryCondition)

    # Internal state for function generation
    _boundary_functions = param.List(default=[])
    _boundary_tags = param.List(default=[])

    def __init__(self, boundary_conditions=None, **params):
        """
        Initialize with a list of BoundaryCondition objects.
        Accepts both positional arg and keyword arg 'boundary_conditions'.
        """
        # 1. Handle Positional Argument
        if boundary_conditions is not None:
            params["boundary_conditions_list"] = boundary_conditions

        # 2. Handle Keyword Alias (for backward compatibility)
        elif "boundary_conditions" in params:
            params["boundary_conditions_list"] = params.pop("boundary_conditions")

        super().__init__(**params)

        # 3. Sort BCs by tag to ensure deterministic order (Crucial for C++ Enums)
        # We sort the actual list stored in self.boundary_conditions_list
        if self.boundary_conditions_list:
            self.boundary_conditions_list.sort(key=lambda bc: bc.tag)

        # 4. Cache derived lists
        self._boundary_functions = [
            bc.compute_boundary_condition for bc in self.boundary_conditions_list
        ]
        self._boundary_tags = [bc.tag for bc in self.boundary_conditions_list]

    @property
    def list_sorted_function_names(self):
        """Return list of tags, guaranteed sorted by __init__ logic."""
        return self._boundary_tags

    @property
    def boundary_conditions_list_dict(self):
        """Return dict {tag: bc_object} for easy lookup."""
        return {bc.tag: bc for bc in self.boundary_conditions_list}

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        """
        Generates the master SymPy function that dispatches to specific BCs based on 'idx'.
        """
        bc_idx = sympy.Symbol("bc_idx", integer=True)

        if not self._boundary_functions:
            # Fallback if no BCs defined (just return Q)
            bc_func_expr = ZArray(Q.get_list())
        else:
            # Create a Piecewise function: if (idx == 0) -> func0, elif (idx == 1) -> func1 ...
            conditions = []
            for i, func in enumerate(self._boundary_functions):
                res = func(
                    time,
                    X.get_list(),
                    dX,
                    Q.get_list(),
                    Qaux.get_list(),
                    parameters.get_list(),
                    normal.get_list(),
                )
                conditions.append((res, sympy.Eq(bc_idx, i)))

            bc_func_expr = sympy.Piecewise(*conditions)

        func = Function(
            name="boundary_conditions",
            args=Zstruct(
                idx=bc_idx,
                time=time,
                position=X,
                distance=dX,
                variables=Q,
                aux_variables=Qaux,
                parameters=parameters,
                normal=normal,
            ),
            definition=bc_func_expr,
        )
        return func
