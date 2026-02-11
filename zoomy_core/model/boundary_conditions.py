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


# --- Derived Boundary Conditions (Unchanged) ---


class Extrapolation(BoundaryCondition):
    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        return ZArray(Q)


class InflowOutflow(BoundaryCondition):
    prescribe_fields = param.Dict(default={})

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            try:
                val = float(v)
            except (ValueError, TypeError):
                val = eval(v)
            Qout[k] = val
        return Qout


class Lambda(BoundaryCondition):
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
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            interp_func = _sympy_interpolate_data(time, self.timeline, v)
            Qout[k] = 2 * interp_func - Q[k]
        return Qout


class CharacteristicReflective(BoundaryCondition):
    R = param.Parameter(default=None)
    L = param.Parameter(default=None)
    D = param.Parameter(default=None)
    S = param.Parameter(default=None)
    M = param.Parameter(default=None)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        q = Matrix(Q)
        q_n = self.S @ q
        W_int = self.L @ q_n
        W_bc = W_int.copy()
        MW = self.M @ W_int
        for i in range(W_int.rows):
            lam = self.D[i, i]
            cond = sympy.GreaterThan(-lam, 0, evaluate=False)
            W_bc[i, 0] = sympy.Function("conditional")(cond, MW[i, 0], W_int[i, 0])
        q_n_bc = self.R @ W_bc
        q_bc = sympy.simplify(self.S.inv() @ q_n_bc)
        out = ZArray.zeros(len(q_bc))
        for i in range(len(q_bc)):
            out[i] = sympy.Function("conditional")(
                sympy.GreaterThan(q[0], 1e-4), q_bc[i, 0], q[i, 0]
            )
        return out


class Wall(BoundaryCondition):
    momentum_field_indices = param.List(default=[[1, 2]])
    permeability = param.Number(default=0.0)
    wall_slip = param.Number(default=1.0)
    blending = param.Number(default=0.0)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        q = ZArray(Q)
        dim = len(self.momentum_field_indices[0])
        n_vec = Matrix(normal[:dim])
        out = ZArray(Q)
        momentum_list_wall = []
        for indices in self.momentum_field_indices:
            momentum = Matrix([q[k] for k in indices])
            normal_momentum_coef = momentum.dot(n_vec)
            transverse_momentum = momentum - normal_momentum_coef * n_vec
            momentum_wall = (
                self.wall_slip * transverse_momentum
                - (1 - self.permeability) * normal_momentum_coef * n_vec
            )
            momentum_list_wall.append(momentum_wall)
        for indices, momentum_wall in zip(
            self.momentum_field_indices, momentum_list_wall
        ):
            for i, idx in enumerate(indices):
                out[idx] = (1 - self.blending) * momentum_wall[i] + self.blending * q[
                    idx
                ]
        return out


class RoughWall(Wall):
    CsW = param.Number(default=0.5)
    Ks = param.Number(default=0.001)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        slip_length = dX * sympy.ln((dX * self.CsW) / self.Ks)
        f = dX / slip_length
        wall_slip = (1 - f) / (1 + f)
        original_slip = self.wall_slip
        self.wall_slip = wall_slip
        res = super().compute_boundary_condition(
            time, X, dX, Q, Qaux, parameters, normal
        )
        self.wall_slip = original_slip
        return res


class Periodic(BoundaryCondition):
    periodic_to_physical_tag = param.String(default="")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        return ZArray(Q)


# --- Container Class ---


class BoundaryConditions(param.Parameterized):
    boundary_conditions_list = param.List(default=[], item_type=BoundaryCondition)
    _boundary_functions = param.List(default=[])
    _boundary_tags = param.List(default=[])

    def __init__(self, boundary_conditions=None, **params):
        if boundary_conditions is not None:
            params["boundary_conditions_list"] = boundary_conditions
        elif "boundary_conditions" in params:
            params["boundary_conditions_list"] = params.pop("boundary_conditions")
        super().__init__(**params)
        if self.boundary_conditions_list:
            self.boundary_conditions_list.sort(key=lambda bc: bc.tag)
        self._boundary_functions = [
            bc.compute_boundary_condition for bc in self.boundary_conditions_list
        ]
        self._boundary_tags = [bc.tag for bc in self.boundary_conditions_list]

    @property
    def list_sorted_function_names(self):
        return self._boundary_tags

    @property
    def boundary_conditions_list_dict(self):
        return {bc.tag: bc for bc in self.boundary_conditions_list}

    # [FIX] Added 'function_name' argument with default "boundary_conditions"
    def get_boundary_condition_function(
        self,
        time,
        X,
        dX,
        Q,
        Qaux,
        parameters,
        normal,
        function_name="boundary_conditions",
    ):
        bc_idx = sympy.Symbol("bc_idx", integer=True)

        if not self._boundary_functions:
            bc_func_expr = ZArray(Q.get_list())
        else:
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

        # [FIX] Use the passed name here
        func = Function(
            name=function_name,
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
