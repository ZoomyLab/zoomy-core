"""Per-field boundary conditions — the FLAT-LIST interface, demonstrated.

THE canonical example: at ONE tag, different fields get different BCs —
momentum REFLECTS (Wall) while the water depth EXTRAPOLATES, every other slot
defaults to Extrapolation.  `on=` is resolved generically against the model's
declared state, so any future field (tracer, energy, temperature) is addressable
by name with no hard-coding.

    SME(level=2, closures=[...],
        boundary_conditions=[
            Wall("left",         on="momentum"),   # q_0,q_1,q_2 reflect
            Extrapolation("left", on="h"),         # depth zero-gradient
            Extrapolation("right"),                # on="all" → every slot
        ])
"""
import numpy as np

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.model.boundary_conditions import (
    resolve_per_field, PerFieldBoundary, Wall, Extrapolation)
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec

STATE = ["b", "h", "q_0", "q_1", "q_2"]          # SME(level=2) state


def test_resolve_per_field_momentum_reflects_depth_extrapolates():
    """The core power: one tag, Wall on momentum + Extrapolation on h."""
    bcs = resolve_per_field(
        [Wall("left", on="momentum"),
         Extrapolation("left", on="h"),
         Extrapolation("right")],
        STATE, aliases={"momentum": "q"})
    left = bcs.boundary_conditions_list_dict["left"]
    assert isinstance(left, PerFieldBoundary)

    Qin = np.array([0.1, 1.5, 0.7, 0.2, -0.1])    # [b, h, q_0, q_1, q_2]
    n_left = np.array([-1.0])
    ghost = left.face_value(Qin, np.zeros(0), n_left, 0.5, 0.0, np.zeros(0))
    assert np.isclose(ghost[0], Qin[0]),  "b should be copied"
    assert np.isclose(ghost[1], Qin[1]),  "h should EXTRAPOLATE (copied)"
    assert np.isclose(ghost[2], -Qin[2]), "q_0 should REFLECT"
    assert np.isclose(ghost[3], -Qin[3]), "q_1 should REFLECT"
    assert np.isclose(ghost[4], -Qin[4]), "q_2 should REFLECT"

    # the right tag: nothing specified → everything extrapolates (default)
    right = bcs.boundary_conditions_list_dict["right"]
    g_r = right.face_value(Qin, np.zeros(0), np.array([1.0]), 0.5, 0.0, np.zeros(0))
    assert np.allclose(g_r, Qin), "unspecified tag → all extrapolate"


def test_on_resolves_any_field_generically():
    """`on=` is pure string resolution against the declared state — a future
    field family ('T', 'k') is addressable with no code change."""
    state = ["b", "h", "q_0", "q_1", "k_0", "k_1", "T_0"]
    bcs = resolve_per_field([Wall("left", on="q"), Extrapolation("left", on="k"),
                             Extrapolation("left", on="T_0")],
                            state, aliases={})
    left = bcs.boundary_conditions_list_dict["left"]
    Qin = np.arange(len(state), dtype=float) + 1.0
    g = left.face_value(Qin, np.zeros(0), np.array([-1.0]), 0.5, 0.0, np.zeros(0))
    assert np.isclose(g[2], -Qin[2]) and np.isclose(g[3], -Qin[3]), "q_* reflect"
    assert np.isclose(g[4], Qin[4]) and np.isclose(g[6], Qin[6]), "k_*, T_0 copy"


def test_conflicting_bcs_raise():
    import pytest
    with pytest.raises(ValueError, match="conflicting"):
        resolve_per_field([Wall("left", on="h"), Extrapolation("left", on="h")],
                          STATE, aliases={})


def test_unknown_field_raises():
    import pytest
    with pytest.raises(ValueError, match="not a state field"):
        resolve_per_field([Wall("left", on="nope")], STATE, aliases={})


def test_end_to_end_per_field_wall_dambreak():
    """The flat per-field list flows through to a running numpy solve."""
    sm = SME(level=2, parameters={"nu": 1e-3},
             closures=[Newtonian(), NavierSlip(), StressFree()],
             boundary_conditions=[Wall("left", on="momentum"),
                                  Extrapolation("left", on="h"),
                                  Extrapolation("right")]).system_model
    nc = 100
    sm.initial_conditions = IC.UserFunction(
        function=lambda xv: np.array([0.0, 1.5 if float(xv[0]) < 5 else 0.75, 0, 0, 0]))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=nc)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    Q, _ = HyperbolicSolver(time_end=0.3,
                            compute_dt=timestepping.adaptive(CFL=0.45)).solve(
        mesh, nsm, write_output=False)
    h = np.asarray(Q[1, :nc], float)
    assert np.all(np.isfinite(Q)) and np.all(h > 0)
