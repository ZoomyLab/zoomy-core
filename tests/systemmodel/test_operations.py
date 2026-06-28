"""REQ-63 — reusable ``SystemModel.apply`` operations + numpy aux-application.

``register_aux`` / ``regularize_pow`` replace the per-case hand-rolled "system
surgery" (walk operators, substitute ``h^{-n}``, append to ``aux_state``,
hand-build a full-length ``update_aux_variables``, call
``refresh_derived_operators``).  The numpy FVM applies the auto-augmented
``update_aux_variables`` each step so the algebraic aux is populated at runtime.
"""
import numpy as np
import sympy as sp

from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import Extrapolation
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel import register_aux, regularize_pow


def _build_sme(level=1):
    return SME(
        level=level,
        parameters={"nu": 1e-3, "lambda_s": 0.0},
        closures=[Newtonian(), NavierSlip(), StressFree()],
        boundary_conditions=[Extrapolation("left"), Extrapolation("right")],
    ).system_model


def _kp_hinv(h, eps):
    return sp.sqrt(2) * sp.Max(h, 0) / sp.sqrt(h ** 4 + sp.Max(h, eps) ** 4)


def _uav_rows(sm):
    uav = sm.update_aux_variables
    return [uav[i, 0] for i in range(uav.shape[0])]


def test_register_aux_adds_full_length_update_row():
    sm = _build_sme(level=1)
    h = next(s for s in sm.state if str(s) == "h")
    n_aux0 = len(sm.aux_state)

    sm.apply(register_aux("hinv", _kp_hinv(h, sp.Float(1e-3)), positive=True))

    names = [str(s) for s in sm.aux_state]
    assert "hinv" in names
    rows = _uav_rows(sm)
    # full-length: one row per aux (the prefix-write covers every row)
    assert len(rows) == len(sm.aux_state) == n_aux0 + 1
    hinv_row = names.index("hinv")
    assert rows[hinv_row].has(sp.Max), "hinv row is not the KP formula"
    # source jacobian wrt aux resized to the new aux vector (refresh ran)
    assert sm.source_jacobian_wrt_aux_variables.shape[1] == len(sm.aux_state)


def test_regularize_pow_rewrites_and_refreshes():
    sm = _build_sme(level=1)
    h = next(s for s in sm.state if str(s) == "h")
    sm.apply(register_aux("hinv", _kp_hinv(h, sp.Float(1e-3)), positive=True))
    hinv = next(s for s in sm.aux_state if str(s) == "hinv")

    flux_before = sp.sympify(sm.flux[2, 0])
    assert flux_before.has(h ** -1)

    sm.apply(regularize_pow("h", "hinv"))

    flux_after = sp.sympify(sm.flux[2, 0])
    assert not flux_after.has(h ** -1) and not flux_after.has(h ** -2)
    assert flux_after.has(hinv)
    # derived operators refreshed INSIDE the op (no manual refresh): the
    # quasilinear matrix is recomputed from the substituted flux.
    ql = sm.quasilinear_matrix
    ql_syms = set().union(*[sp.sympify(ql[idx]).free_symbols
                            for idx in np.ndindex(*ql.shape)])
    assert hinv in ql_syms


def test_chaining_both_ops():
    sm = _build_sme(level=1)
    h = next(s for s in sm.state if str(s) == "h")
    sm.apply(register_aux("hinv", _kp_hinv(h, sp.Float(1e-3)), positive=True)) \
      .apply(regularize_pow("h", "hinv"))
    assert "hinv" in [str(s) for s in sm.aux_state]
    assert sp.sympify(sm.flux[2, 0]).has(
        next(s for s in sm.aux_state if str(s) == "hinv"))


def test_numpy_populates_algebraic_aux_at_runtime():
    """The numpy FVM applies ``update_aux_variables`` each step: after a short
    run the ``hinv`` aux equals ``1/h`` on the (wet) cells — not the zero it was
    initialised to."""
    sm = _build_sme(level=1)
    h = next(s for s in sm.state if str(s) == "h")
    sm.apply(register_aux("hinv", _kp_hinv(h, sp.Float(1e-3)), positive=True)) \
      .apply(regularize_pow("h", "hinv"))

    def _ic(xv):
        hval = 1.5 if float(xv[0]) < 5.0 else 0.75
        return np.array([0.0, hval] + [0.0] * (sm.n_equations - 2))

    sm.initial_conditions = IC.UserFunction(function=_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=50)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=0.02,
                              compute_dt=timestepping.adaptive(CFL=0.45))
    Q, Qaux = solver.solve(mesh, nsm, write_output=False)
    Q = np.asarray(Q, float)
    Qaux = np.asarray(Qaux, float)

    hinv_row = [str(s) for s in sm.aux_state].index("hinv")
    hcells = Q[1, :50]
    hinv_cells = Qaux[hinv_row, :50]
    assert np.any(hinv_cells != 0.0), "hinv aux never populated (still zero)"
    assert np.allclose(hinv_cells, 1.0 / hcells, atol=1e-6)
    assert np.all(np.isfinite(Q))
