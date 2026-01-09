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
        print("BoundaryCondition is a virtual class. Use one if its derived classes!")
        assert False


# --- Derived Boundary Conditions ---


class Extrapolation(BoundaryCondition):
    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        return ZArray(Q)


class InflowOutflow(BoundaryCondition):
    prescribe_fields = param.Dict(default={})

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            # Note: eval() is kept to match original behavior
            Qout[k] = eval(v)
        return Qout


class Lambda(BoundaryCondition):
    prescribe_fields = param.Dict(default={})

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            Qout[k] = v(time, X, dX, Q, Qaux, parameters, normal)
        return Qout


class FromData(BoundaryCondition):
    prescribe_fields = param.Dict(default={})
    timeline = param.Array(default=None)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        # Extrapolate all fields
        Qout = ZArray(Q)

        # Set the fields which are prescribed in boundary condition dict
        # time_start = get_time() # (Unused variable from original)
        for k, v in self.prescribe_fields.items():
            interp_func = _sympy_interpolate_data(time, self.timeline, v)
            Qout[k] = 2 * interp_func - Q[k]
        return Qout


class CharacteristicReflective(BoundaryCondition):
    """
    Generic characteristic reflective wall boundary condition.
    """

    # Using Generic Parameter for Matrices to allow flexibility (SymPy/NumPy)
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
            cond = sympy.GreaterThan(-lam, 0, evaluate=False)
            W_bc[i, 0] = sympy.Function("conditional")(cond, MW[i, 0], W_int[i, 0])

        # 4. Transform back
        q_n_bc = self.R @ W_bc
        q_bc = sympy.simplify(self.S.inv() @ q_n_bc)

        out = ZArray.zeros(len(q_bc))
        for i in range(len(q_bc)):
            out[i] = sympy.Function("conditional")(
                sympy.GreaterThan(q[1, 0], 1e-4), q_bc[i, 0], q[i, 0]
            )

        return out


class Wall(BoundaryCondition):
    """
    permeability: float : 0.0 corresponds to a perfect reflection (impermeable wall)
    blending: float: 0.5 blend the reflected wall solution with the solution of the inner cell
    """

    # Defaults are handled safely by param (no factory needed)
    momentum_field_indices = param.List(default=[[1, 2]])
    permeability = param.Number(default=0.0)
    wall_slip = param.Number(default=1.0)
    blending = param.Number(default=0.0)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        q = ZArray(Q)
        n_variables = q.shape[0]
        momentum_list = [Matrix([q[k] for k in l]) for l in self.momentum_field_indices]
        dim = momentum_list[0].shape[0]
        n = Matrix(normal[:dim])

        out = ZArray(Q)  # Initialize with Q copy
        momentum_list_wall = []

        for momentum in momentum_list:
            normal_momentum_coef = momentum.dot(n)
            transverse_momentum = momentum - normal_momentum_coef * n
            momentum_wall = (
                self.wall_slip * transverse_momentum
                - (1 - self.permeability) * normal_momentum_coef * n
            )
            momentum_list_wall.append(momentum_wall)

        for l, momentum_wall in zip(self.momentum_field_indices, momentum_list_wall):
            for i_k, k in enumerate(l):
                out[k] = (1 - self.blending) * momentum_wall[i_k] + self.blending * q[k]
        return out


class RoughWall(Wall):
    CsW = param.Number(default=0.5)  # roughness constant
    Ks = param.Number(default=0.001)  # roughness height

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        slip_length = dX * sympy.ln((dX * self.CsW) / self.Ks)
        f = dX / slip_length
        wall_slip = (1 - f) / (1 + f)

        # We can modify self temporarily or pass it locally.
        # Modifying self.wall_slip is fine in param (it's mutable).
        self.wall_slip = wall_slip

        return super().compute_boundary_condition(
            time, X, dX, Q, Qaux, parameters, normal
        )


class Periodic(BoundaryCondition):
    periodic_to_physical_tag = param.String(default="")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        return ZArray(Q)


# --- Container Class ---


# In zoomy_core/model/boundary_conditions.py


class BoundaryConditions(param.Parameterized):
    # FIXED: item_type instead of class_
    boundary_conditions_list = param.List(default=[], item_type=BoundaryCondition)

    # Internal state
    _boundary_functions = param.List(default=[])
    _boundary_tags = param.List(default=[])

    # FIXED: Added 'boundary_conditions' as a positional argument
    def __init__(self, boundary_conditions=None, **params):
        # 1. Handle Positional Arg: BoundaryConditions([...])
        if boundary_conditions is not None:
            params["boundary_conditions_list"] = boundary_conditions

        # 2. Handle Keyword Alias: BoundaryConditions(boundary_conditions=[...])
        elif "boundary_conditions" in params:
            params["boundary_conditions_list"] = params.pop("boundary_conditions")

        super().__init__(**params)

        # 3. Sort and Setup (Same as before)
        tags_unsorted = [bc.tag for bc in self.boundary_conditions_list]
        order = np.argsort(tags_unsorted)

        self.boundary_conditions_list = [
            self.boundary_conditions_list[i] for i in order
        ]

        self._boundary_functions = [
            bc.compute_boundary_condition for bc in self.boundary_conditions_list
        ]
        self._boundary_tags = [bc.tag for bc in self.boundary_conditions_list]

    # ... (get_boundary_condition_function remains the same) ...
    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        bc_idx = sympy.Symbol("bc_idx", integer=True)

        if not self._boundary_functions:
            bc_func = ZArray(Q.get_list())
        else:
            bc_func = sympy.Piecewise(
                *(
                    (
                        func(
                            time,
                            X.get_list(),
                            dX,
                            Q.get_list(),
                            Qaux.get_list(),
                            parameters.get_list(),
                            normal.get_list(),
                        ),
                        sympy.Eq(bc_idx, i),
                    )
                    for i, func in enumerate(self._boundary_functions)
                )
            )

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
            definition=bc_func,
        )
        return func
