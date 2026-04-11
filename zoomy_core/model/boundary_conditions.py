"""Module `zoomy_core.model.boundary_conditions`."""

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
    """Internal helper `_sympy_interpolate_data`."""
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
        """Compute boundary condition."""
        raise NotImplementedError(
            "BoundaryCondition is a virtual class. Use one of its derived classes!"
        )


# --- Derived Boundary Conditions (Unchanged) ---


class Extrapolation(BoundaryCondition):
    """Extrapolation. (class)."""
    use_gradient = param.Boolean(default=True,
        doc="Use gradient for 2nd-order ghost extrapolation when available")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        return ZArray(Q)


class InflowOutflow(BoundaryCondition):
    """InflowOutflow. (class)."""
    prescribe_fields = param.Dict(default={})

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            try:
                val = float(v)
            except (ValueError, TypeError):
                val = eval(v)
            Qout[k] = val
        return Qout


class Lambda(BoundaryCondition):
    """Lambda. (class)."""
    prescribe_fields = param.Dict(default={})

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        Qout = ZArray(Q)
        for k, func in self.prescribe_fields.items():
            Qout[k] = func(time, X, dX, Q, Qaux, parameters, normal)
        return Qout


class FromData(BoundaryCondition):
    """FromData. (class)."""
    prescribe_fields = param.Dict(default={})
    timeline = param.Array(default=None)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            interp_func = _sympy_interpolate_data(time, self.timeline, v)
            Qout[k] = 2 * interp_func - Q[k]
        return Qout


class CharacteristicReflective(BoundaryCondition):
    """CharacteristicReflective. (class)."""
    R = param.Parameter(default=None)
    L = param.Parameter(default=None)
    D = param.Parameter(default=None)
    S = param.Parameter(default=None)
    M = param.Parameter(default=None)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
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
    """Wall. (class)."""
    momentum_field_indices = param.List(default=[[1, 2]])
    permeability = param.Number(default=0.0)
    wall_slip = param.Number(default=1.0)
    blending = param.Number(default=0.0)
    use_gradient = param.Boolean(default=True,
        doc="Use gradient for 2nd-order ghost extrapolation when available")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
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
    """RoughWall. (class)."""
    CsW = param.Number(default=0.5)
    Ks = param.Number(default=0.001)

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
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
    """Periodic. (class)."""
    periodic_to_physical_tag = param.String(default="")

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        """Compute boundary condition."""
        return ZArray(Q)


# --- System-Aware Boundary Conditions ---
#
# These are applied via system.boundary_conditions.apply(SystemWall(), tag="right").
# They dispatch per equation type: scalar → Extrapolation, momentum → reflection.


class SystemExtrapolation:
    """Apply Extrapolation to all equations in the system."""

    def __init__(self, tag=None):
        self.tag = tag

    def apply_to_system_bcs(self, system_bcs, tag=None):
        t = tag or self.tag
        for eq_name in system_bcs.equation_names:
            system_bcs.set(eq_name, Extrapolation(tag=t), tag=t)


class SystemPeriodic:
    """Apply Periodic to all equations in the system."""

    def __init__(self, tag=None, periodic_to_physical_tag=""):
        self.tag = tag
        self.periodic_to_physical_tag = periodic_to_physical_tag

    def apply_to_system_bcs(self, system_bcs, tag=None):
        t = tag or self.tag
        for eq_name in system_bcs.equation_names:
            system_bcs.set(
                eq_name,
                Periodic(tag=t, periodic_to_physical_tag=self.periodic_to_physical_tag),
                tag=t,
            )


class SystemWall:
    """System-aware wall BC: Extrapolation for scalars, reflection for momentum.

    Applied via ``system.boundary_conditions.apply(SystemWall(), tag="right")``.
    The Wall BC holds a reference to the system and reads its equations
    to determine scalar vs momentum fields automatically.

    Parameters
    ----------
    tag : str
        Boundary tag (e.g. "right", "bottom").
    permeability : float
        0 = impermeable (default), 1 = fully permeable.
    wall_slip : float
        1 = free-slip (default), 0 = no-slip.
    """

    def __init__(self, tag=None, permeability=0.0, wall_slip=1.0):
        self.tag = tag
        self.permeability = permeability
        self.wall_slip = wall_slip

    def apply_to_system_bcs(self, system_bcs, tag=None, system=None):
        t = tag or self.tag
        momentum_eqs = [n for n in system_bcs.equation_names if "momentum" in n]
        scalar_eqs = [n for n in system_bcs.equation_names if n not in momentum_eqs]

        # Scalars: extrapolation WITHOUT gradient (preserves hydrostatic balance)
        for eq_name in scalar_eqs:
            system_bcs.set(eq_name, Extrapolation(tag=t, use_gradient=False), tag=t)

        # Momentum: wall reflection WITH gradient (for O2 boundary accuracy)
        wall_bc = WallMomentumBC(
            tag=t,
            system_bcs=system_bcs,
            permeability=self.permeability,
            wall_slip=self.wall_slip,
            use_gradient=True,
        )
        for eq_name in momentum_eqs:
            system_bcs.set(eq_name, wall_bc, tag=t)


class WallMomentumBC:
    """Wall BC for momentum equations — reads the system to build reflection.

    Holds a reference to the ``SystemBoundaryConditions`` (and thus knows
    which equations exist).  At compile time, determines the momentum
    vector grouping automatically from the equation names.

    The normal/tangential decomposition works for any system derived from
    INS: SWE (hu), SME (hu0, hu1, ...), VAM (hu, hv, hw moments), full INS.
    """

    def __init__(self, tag, system_bcs, permeability=0.0, wall_slip=1.0,
                 use_gradient=True):
        self.tag = tag
        self._system_bcs = system_bcs
        self.permeability = permeability
        self.wall_slip = wall_slip
        self.use_gradient = use_gradient

    @property
    def momentum_equations(self):
        """Momentum equations in the current system."""
        return [n for n in self._system_bcs.equation_names if "momentum" in n]

    def __repr__(self):
        return (f"WallMomentumBC(eqs={self.momentum_equations}, "
                f"perm={self.permeability}, slip={self.wall_slip})")


# --- Compiler: System BCs → legacy BoundaryConditions ---


def compile_system_bcs(system_bcs, equation_variable_map, dimension):
    """Translate system-aware BCs into the legacy BoundaryConditions container.

    Reads per-equation, per-tag BCs from ``system_bcs`` and produces
    a ``BoundaryConditions`` with one entry per tag. The
    ``equation_variable_map`` maps equation names to variable indices
    so the Wall BC knows which indices form the momentum vector.

    Parameters
    ----------
    system_bcs : SystemBoundaryConditions
    equation_variable_map : dict
        ``{equation_name: [var_index, ...]}``
    dimension : int
        Model dimension (1 or 2 for horizontal).

    Returns
    -------
    BoundaryConditions
    """
    bc_list = []
    for tag in system_bcs.tags:
        bcs_for_tag = system_bcs.get_all(tag)
        if not bcs_for_tag:
            continue

        # Check if any equation has a WallMomentumBC for this tag
        wall_bc = None
        all_extrap = True
        all_periodic = True
        for eq_name, bc in bcs_for_tag.items():
            if isinstance(bc, WallMomentumBC):
                wall_bc = bc
                all_extrap = False
                all_periodic = False
            elif isinstance(bc, Periodic):
                all_extrap = False
            elif isinstance(bc, Extrapolation):
                all_periodic = False
            else:
                all_extrap = False
                all_periodic = False

        if all_periodic:
            bc_obj = next(iter(bcs_for_tag.values()))
            bc_list.append(Periodic(
                tag=tag,
                periodic_to_physical_tag=getattr(bc_obj, 'periodic_to_physical_tag', ''),
            ))
        elif wall_bc is not None:
            # Build momentum_field_indices from equation_variable_map.
            # Group momentum equation indices by moment index:
            # For level=1, 1D: x_momentum = [2, 3] → [[2], [3]]
            # For level=1, 2D: x_momentum = [2, 3], y_momentum = [4, 5]
            #   → [[2, 4], [3, 5]]  (pairs for normal/tangential)
            mom_eqs = wall_bc.momentum_equations
            mom_indices_per_eq = [equation_variable_map.get(eq, []) for eq in mom_eqs]

            if len(mom_eqs) == 0:
                # No momentum equations — just extrapolation
                bc_list.append(Extrapolation(tag=tag))
            elif len(mom_eqs) == 1:
                # 1D: each variable is its own "vector" (scalar momentum)
                bc_list.append(Wall(
                    tag=tag,
                    momentum_field_indices=[[idx] for idx in mom_indices_per_eq[0]],
                    permeability=wall_bc.permeability,
                    wall_slip=wall_bc.wall_slip,
                ))
            else:
                # 2D+: group by moment index across equations
                n_per_eq = len(mom_indices_per_eq[0])
                groups = []
                for k in range(n_per_eq):
                    group = [indices[k] for indices in mom_indices_per_eq
                             if k < len(indices)]
                    groups.append(group)
                bc_list.append(Wall(
                    tag=tag,
                    momentum_field_indices=groups,
                    permeability=wall_bc.permeability,
                    wall_slip=wall_bc.wall_slip,
                ))
        elif all_extrap:
            bc_list.append(Extrapolation(tag=tag))
        else:
            # Mixed — default to extrapolation
            bc_list.append(Extrapolation(tag=tag))

    result = BoundaryConditions(bc_list)

    # Build per-tag gradient variable indices: which variables should get
    # gradient extrapolation at each boundary tag.
    grad_indices = {}
    for tag in system_bcs.tags:
        indices = []
        for eq_name, bc in system_bcs.get_all(tag).items():
            if getattr(bc, 'use_gradient', True):
                indices.extend(equation_variable_map.get(eq_name, []))
        grad_indices[tag] = sorted(set(indices))
    result._gradient_variable_indices = grad_indices

    return result


# --- Container Class ---


class BoundaryConditions(param.Parameterized):
    """BoundaryConditions. (class)."""
    boundary_conditions_list = param.List(default=[], item_type=BoundaryCondition)
    _boundary_functions = param.List(default=[])
    _boundary_tags = param.List(default=[])

    def __init__(self, boundary_conditions=None, **params):
        """Initialize the instance."""
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
        """List sorted function names."""
        return self._boundary_tags

    @property
    def boundary_conditions_list_dict(self):
        """Boundary conditions list dict."""
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
        """Get boundary condition function."""
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
