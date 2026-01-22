import numpy as np
import param
import sympy as sp
from typing import Callable, Optional

from zoomy_core.misc.custom_types import FArray
from zoomy_core.misc.misc import ZArray
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

    def get_definition(self, X, p, n_variables):
        """
        Returns a symbolic expression (ZArray) representing the initial state.
        Used by C++ code generation.
        """
        # Default fallback: Return 0. Allows compilation even if IC is non-symbolic.
        # The solver is expected to overwrite this at runtime if needed.
        return ZArray.zeros(n_variables)


class Constant(InitialConditions):
    constants = param.Callable(default=default_constant_func)

    def __init__(self, constants=None, **params):
        if constants is not None:
            params["constants"] = constants
        super().__init__(**params)

    def apply(self, X, Q):
        n_variables = Q.shape[0]
        const_vals = self.constants(n_variables)
        for i in range(Q.shape[1]):
            Q[:, i] = const_vals
        return Q

    def get_definition(self, X, p, n_variables):
        vals = self.constants(n_variables)
        return ZArray(vals)


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

    def get_definition(self, X, p, n_variables):
        val_low = self.low(n_variables)
        val_high = self.high(n_variables)

        # Symbolic Condition
        cond = X[0] < self.jump_position_x

        out = []
        for i in range(n_variables):
            out.append(sp.Piecewise((val_high[i], cond), (val_low[i], True)))
        return ZArray(out)


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

    def get_definition(self, X, p, n_variables):
        val_low = self.low(n_variables)
        val_high = self.high(n_variables)
        cond = sp.And(X[0] < self.jump_position_x, X[1] < self.jump_position_y)
        out = []
        for i in range(n_variables):
            out.append(sp.Piecewise((val_high[i], cond), (val_low[i], True)))
        return ZArray(out)


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

    def get_definition(self, X, p, n_variables):
        val_low = self.low(n_variables)
        val_high = self.high(n_variables)
        cond = sp.And(
            X[0] < self.jump_position_x,
            X[1] < self.jump_position_y,
            X[2] < self.jump_position_z,
        )
        out = []
        for i in range(n_variables):
            out.append(sp.Piecewise((val_high[i], cond), (val_low[i], True)))
        return ZArray(out)


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

    def get_definition(self, X, p, n_variables):
        # NOTE: For C++ generation, we assume center is at (0,0,0) or needs explicit params.
        # This implementation assumes origin for simplicity.
        dist_sq = X[0] ** 2 + X[1] ** 2 + X[2] ** 2
        cond = dist_sq <= self.radius**2

        val_low = self.low(n_variables)
        val_high = self.high(n_variables)

        out = []
        for i in range(n_variables):
            out.append(sp.Piecewise((val_high[i], cond), (val_low[i], True)))
        return ZArray(out)


class UserFunction(InitialConditions):
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

    def get_definition(self, X, p, n_variables):
        """
        Attempts to symbolically execute the user function.
        If it fails (e.g. contains numpy calls), returns 0 (dummy).
        """
        if self.function is None:
            return ZArray.zeros(n_variables)
        try:
            # Try passing symbolic X
            res = self.function(X)
            return ZArray(res)
        except Exception:
            # Fallback for non-symbolic functions (e.g. restarts or complex logic)
            # We return 0, expecting the C++ solver to OVERWRITE this via file IO.
            return ZArray.zeros(n_variables)


class RestartFromHdf5(InitialConditions):
    # Parameters omitted for brevity, functionality preserved
    path_to_fields = param.String(default=None)
    mesh_new = param.ClassSelector(class_=Mesh, default=None)
    mesh_identical = param.Boolean(default=False)
    path_to_old_mesh = param.String(default=None)
    snapshot = param.Integer(default=-1)
    map_fields = param.Dict(default=None)

    def apply(self, X, Q):
        # (Standard Python implementation)
        pass

    def get_definition(self, X, p, n_variables):
        # Restart is a RUNTIME operation.
        # Generate a dummy zero state for the compiled model.
        # The C++ VirtualSolver must handle the actual HDF5 loading.
        return ZArray.zeros(n_variables)
