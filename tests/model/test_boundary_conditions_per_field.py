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
from zoomy_core.systemmodel.system_model import SystemModel

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


def test_wall_is_dimension_agnostic_normal_transverse():
    """Wall must reflect only the NORMAL momentum component and keep the
    transverse — for ANY dimension (momentum components grouped into vectors per
    moment level), not reflect each scalar slot."""
    # 2-D momentum: q_x_i (x) and q_y_i (y) per moment level
    state2d = ["b", "h", "q_x_0", "q_y_0", "q_x_1", "q_y_1"]
    bcs = resolve_per_field([Wall("left", on="momentum")], state2d, aliases={"momentum": "q"})
    left = bcs.boundary_conditions_list_dict["left"]
    Qin = np.array([0.0, 1.5, 0.7, 0.3, 0.2, -0.1])     # q_x_0,q_y_0,q_x_1,q_y_1
    g = left.face_value(Qin, np.zeros(0), np.array([1.0, 0.0]), 0.5, 0.0, np.zeros(0))  # normal = +x
    assert np.isclose(g[2], -0.7) and np.isclose(g[4], -0.2), "q_x (normal) must reflect"
    assert np.isclose(g[3], 0.3) and np.isclose(g[5], -0.1), "q_y (transverse) must be kept"


def test_whole_patch_bcs_pass_through_unwrapped():
    """Periodic / Coupled are whole-patch BCs detected by TYPE at the tag level
    (resolve_periodic_bcs; preCICE/OpenFOAM). They must NOT be wrapped in a
    PerFieldBoundary, and can't be mixed with per-field BCs at the same tag."""
    from zoomy_core.model.boundary_conditions import Periodic, Coupled
    bcs = resolve_per_field(
        [Periodic("left", periodic_to_physical_tag="right"), Coupled("top"),
         Wall("front", on="momentum"), Extrapolation("front", on="h")],
        STATE, aliases={"momentum": "q"})
    d = bcs.boundary_conditions_list_dict
    assert type(d["left"]) is Periodic and type(d["top"]) is Coupled
    assert isinstance(d["front"], PerFieldBoundary)
    import pytest
    with pytest.raises(ValueError, match="whole-patch"):
        resolve_per_field([Periodic("left", periodic_to_physical_tag="right"),
                           Wall("left", on="momentum")], STATE, aliases={"momentum": "q"})


def test_named_bc_family_in_flat_per_field():
    """The named BC family is available in the new flat per-field style and each
    resolves to the right per-slot ghost: ZeroNeumann (copy), Dirichlet (value),
    Flux (normal-gradient), Lambda (callable), Wall (momentum reflect)."""
    from zoomy_core.model.boundary_conditions import (
        ZeroNeumann, Dirichlet, Flux, Lambda)
    bcs = resolve_per_field(
        [Wall("left", on="momentum"),
         Dirichlet("left", on="h", value=1.5),
         ZeroNeumann("left", on="b")],
        STATE, aliases={"momentum": "q"})            # STATE=[b,h,q_0,q_1,q_2]
    left = bcs.boundary_conditions_list_dict["left"]
    Qin = np.array([0.3, 1.0, 0.7, 0.2, -0.1])
    n = np.array([-1.0])
    g = left.face_value(Qin, np.zeros(0), n, 0.5, 0.0, np.zeros(0))
    assert np.isclose(g[0], 0.3)                      # b: ZeroNeumann → copy
    assert np.isclose(g[1], 1.5)                      # h: Dirichlet → value
    assert np.isclose(g[2], -0.7) and np.isclose(g[3], -0.2)   # q_0,q_1 reflect
    # Lambda (symbolic/codegen-path BC) is available in the flat list and
    # resolves per-field: its slot delegates to the Lambda, others extrapolate.
    lb = resolve_per_field([Lambda("right", on="q_0",
                                   prescribe_fields={2: lambda *a: 9.0})],
                           STATE, aliases={})
    right = lb.boundary_conditions_list_dict["right"]
    assert isinstance(right, PerFieldBoundary)
    assert isinstance(right._slot_bc[2], Lambda)      # q_0 served by the Lambda
    # Flux: prescribed normal gradient on its slot, value extrapolates
    fbcs = resolve_per_field([Flux("right", on="h", gradient=2.0)], STATE, aliases={})
    fr = fbcs.boundary_conditions_list_dict["right"]
    grad = fr.face_gradient(Qin, Qin, np.zeros(0), np.array([1.0]), 0.5, 0.0, np.zeros(0))
    assert np.isclose(grad[1], 2.0)                   # h gets the prescribed flux


def test_coupling_bcs_available_and_whole_patch():
    """Coupling BCs — full-state (Coupled / preCICE) and characteristic — are
    available; the whole-patch ones pass through resolve_per_field unwrapped."""
    from zoomy_core.model.boundary_conditions import (
        Coupled, Characteristic, CharacteristicWall, FromData)
    bcs = resolve_per_field([Coupled("top", mesh_name="fluidMesh")],
                            STATE, aliases={})
    assert type(bcs.boundary_conditions_list_dict["top"]) is Coupled
    # characteristic + FromData construct in the flat list (system-level resolve
    # against the SystemModel happens in resolve_and_attach)
    assert Characteristic("left").tag == "left"
    assert CharacteristicWall("left").tag == "left"
    assert FromData("left").tag == "left"


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
    sm = SystemModel.from_model(SME(level=2, parameters={"nu": 1e-3},
             closures=[Newtonian(), NavierSlip(), StressFree()],
             boundary_conditions=[Wall("left", on="momentum"),
                                  Extrapolation("left", on="h"),
                                  Extrapolation("right")]))
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
