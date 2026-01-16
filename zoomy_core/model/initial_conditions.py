import numpy as np
import param
from typing import Callable, Optional

from zoomy_core.misc.custom_types import FArray
from zoomy_core.mesh.mesh import Mesh
import zoomy_core.misc.io as io
import zoomy_core.misc.interpolation as interpolate_mesh

# --- Helper Functions for Defaults ---


def default_constant_func(n_variables: int) -> FArray:
    """Default: [1.0, 0.0, 0.0, ...]"""
    return np.array([1.0] + [0.0 for i in range(n_variables - 1)])


def default_low_state(n_variables: int) -> FArray:
    """Default Low: [1.0, 0.0, ...]"""
    return np.array([1.0 * (i == 0) for i in range(n_variables)])


def default_high_state(n_variables: int) -> FArray:
    """Default High: [2.0, 0.0, ...]"""
    return np.array([2.0 * (i == 0) for i in range(n_variables)])


def default_user_function(x: FArray) -> FArray:
    """Default user function returns 0.0"""
    return 0.0


# --- Classes ---


class InitialConditions(param.Parameterized):
    def apply(self, X, Q):
        assert False, "InitialConditions is an abstract class."
        return Q


class Constant(InitialConditions):
    constants = param.Callable(default=default_constant_func)

    def __init__(self, constants=None, **params):
        # Allow positional arg: Constant(my_func)
        if constants is not None:
            params["constants"] = constants
        super().__init__(**params)

    def apply(self, X, Q):
        n_variables = Q.shape[0]
        const_vals = self.constants(n_variables)
        for i in range(Q.shape[1]):
            Q[:, i] = const_vals
        return Q


class RP(InitialConditions):
    low = param.Callable(default=default_low_state)
    high = param.Callable(default=default_high_state)
    jump_position_x = param.Number(default=0.0)

    def apply(self, X, Q):
        assert X.shape[1] == Q.shape[1]
        n_variables = Q.shape[0]

        val_low = self.low(n_variables)
        val_high = self.high(n_variables)

        for i in range(Q.shape[1]):
            if X[0, i] < self.jump_position_x:
                Q[:, i] = val_high
            else:
                Q[:, i] = val_low
        return Q


class RP2d(InitialConditions):
    low = param.Callable(default=default_low_state)
    high = param.Callable(default=default_high_state)
    jump_position_x = param.Number(default=0.0)
    jump_position_y = param.Number(default=0.0)

    def apply(self, X, Q):
        assert X.shape[1] == Q.shape[1]
        n_variables = Q.shape[0]

        val_low = self.low(n_variables)
        val_high = self.high(n_variables)

        for i in range(Q.shape[1]):
            if X[0, i] < self.jump_position_x and X[1, i] < self.jump_position_y:
                Q[:, i] = val_high
            else:
                Q[:, i] = val_low
        return Q


class RP3d(InitialConditions):
    low = param.Callable(default=default_low_state)
    high = param.Callable(default=default_high_state)
    jump_position_x = param.Number(default=0.0)
    jump_position_y = param.Number(default=0.0)
    jump_position_z = param.Number(default=0.0)

    def apply(self, X, Q):
        assert X.shape[1] == Q.shape[1]
        n_variables = Q.shape[0]

        val_low = self.low(n_variables)
        val_high = self.high(n_variables)

        for i in range(Q.shape[1]):
            if (
                X[0, i] < self.jump_position_x
                and X[1, i] < self.jump_position_y
                and X[2, i] < self.jump_position_z
            ):
                Q[:, i] = val_high
            else:
                Q[:, i] = val_low
        return Q


class RadialDambreak(InitialConditions):
    low = param.Callable(default=default_low_state)
    high = param.Callable(default=default_high_state)
    radius = param.Number(default=0.1)

    def apply(self, X, Q):
        dim = X.shape[0]
        center = np.zeros(dim)
        for d in range(dim):
            center[d] = X[d, :].mean()

        assert X.shape[1] == Q.shape[1]
        n_variables = Q.shape[0]

        val_low = self.low(n_variables)
        val_high = self.high(n_variables)

        for i in range(Q.shape[1]):
            if np.linalg.norm(X[:, i] - center) <= self.radius:
                Q[:, i] = val_high
            else:
                Q[:, i] = val_low
        return Q


class UserFunction(InitialConditions):
    # default is None, handled in apply
    function = param.Callable(default=None)

    def __init__(self, function=None, **params):
        if function is not None:
            params["function"] = function
        super().__init__(**params)

    def apply(self, X, Q):
        assert X.shape[1] == Q.shape[1]

        func_to_use = self.function
        if func_to_use is None:
            func_to_use = lambda x: np.zeros(Q.shape[0])

        for i, x in enumerate(X.T):
            Q[:, i] = func_to_use(x)
        return Q


class RestartFromHdf5(InitialConditions):
    path_to_fields = param.String(default=None)
    mesh_new = param.ClassSelector(class_=Mesh, default=None)
    mesh_identical = param.Boolean(default=False)
    path_to_old_mesh = param.String(default=None)
    snapshot = param.Integer(default=-1)
    map_fields = param.Dict(default=None)

    def apply(self, X, Q):
        assert self.mesh_new is not None, "mesh_new must be provided for restart"
        assert self.path_to_fields is not None, "path_to_fields must be provided"
        assert X.shape[0] == Q.shape[0]

        if self.map_fields is None:
            map_fields = {i: i for i in range(Q.shape[1])}
        else:
            map_fields = self.map_fields

        mesh = Mesh.from_hdf5(self.path_to_old_mesh)
        _Q, _Qaux, time = io.load_fields_from_hdf5(
            self.path_to_fields, i_snapshot=self.snapshot
        )

        if self.mesh_identical:
            Q[:, list(map_fields.values())] = _Q[:, list(map_fields.keys())]
        else:
            assert self.path_to_old_mesh is not None
            Q[:, list(map_fields.values())] = interpolate_mesh.to_new_mesh(
                _Q, mesh, self.mesh_new
            )[:, list(map_fields.keys())]

        return Q
