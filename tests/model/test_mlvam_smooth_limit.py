"""ML-VAM(2,1) → ML-SME(2,1) in the smooth long-wave limit.

Multilayer analogue of ``test_vam_smooth_limit``: a gentle Gaussian hump
(μ = (h0/L)² ≈ 0.028) marched with identical friction must give the same
solution from the non-hydrostatic multilayer VAM (Chorin split, mean-u*
transfers) and the hydrostatic multilayer SME up to O(μ) — per-layer
discharges included.  Pins the multilayer RUNTIME chain end-to-end against
an analytic expectation.
"""
import numpy as np
import pytest

from zoomy_core.model.models import MLSME, MLVAM
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm.solver_chorin_vam_numpy import ChorinSplitVAMSolver
from zoomy_core.fvm import timestepping
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.systemmodel.system_model import SystemModel

NC, XMAX, T_END, CFL = 100, 20.0, 1.5, 0.9
PAR = {"lambda_s": 0.5, "nu": 1e-3}
H0, AMP, SIG = 0.5, 0.05, 3.0


def _bcs():
    return BoundaryConditions([Extrapolation(tag="left"),
                               Extrapolation(tag="right")])


def _ic(n):
    def ic(xv):
        out = np.zeros(n)
        out[1] = H0 + AMP * np.exp(-((float(xv[0]) - 10.0) ** 2)
                                   / (2 * SIG ** 2))
        return out
    return ic


def _build_mlsme(mesh):
    sme = MLSME(closures=[Newtonian(), NavierSlip(), StressFree()], n_layers=2, level=1, interface_velocity="mean",
                parameters=dict(PAR), boundary_conditions=_bcs())
    sm_s = SystemModel.from_model(sme)
    sm_s.initial_conditions = IC.UserFunction(function=_ic(len(sm_s.state)))
    sm_s.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    nsm = NumericalSystemModel.from_system_model(
        sm_s, reconstruction=ReconstructionSpec(order=1))
    return nsm, [str(s) for s in sm_s.state]


def _build_mlvam(mesh):
    vam = MLVAM(closures=[Newtonian(), NavierSlip(), StressFree()], n_layers=2, level=1, parameters=dict(PAR),
                boundary_conditions=_bcs())
    sm_v = SystemModel.from_model(vam)
    sm_v.initial_conditions = IC.UserFunction(function=_ic(len(sm_v.state)))
    sm_v.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    bcs = _bcs()
    sm_v.attach_boundary_conditions(bcs)
    split = vam.chorin_split(system_model=sm_v)
    split.SM_pred.attach_boundary_conditions(bcs)
    sol = ChorinSplitVAMSolver(split.SM_pred, split.SM_press, split.SM_corr,
                               pressure_solver="gmres", pressure_tol=1e-8,
                               riemann_solver="hr")
    sol.setup_simulation(mesh)
    return sol, [str(s) for s in sm_v.state]


def _mlvam_dt(sol):
    cdt = timestepping.adaptive(CFL=CFL, dimension=1)
    return min(float(cdt(sol._sim_Q, sol._sim_Qaux, sol._sim_parameters,
                         sol._sim_cell_inradius_face,
                         sol._sim_compute_max_abs_eigenvalue)), 5e-3)


def test_mlvam21_mlsme21_long_wave_one_step_twin(one_hyperbolic_step):
    """Default-tier canary: identical smooth-hump ML-SME + ML-VAM(Chorin)
    setup, ONE step each; cheap invariants only (finite, positive depth,
    bounded per-mode non-hydrostatic pressure)."""
    mesh = BaseMesh.create_1d(domain=(0.0, XMAX), n_inner_cells=NC)
    nsm, ns = _build_mlsme(mesh)
    solver = HyperbolicSolver(
        time_end=T_END, compute_dt=timestepping.adaptive(CFL=CFL, dimension=1))
    Qs = one_hyperbolic_step(solver, mesh, nsm)[:, :NC]
    sol, nv = _build_mlvam(mesh)
    sol.step(_mlvam_dt(sol))
    Qv = np.asarray(sol._sim_Q[:, :NC], float)
    assert np.all(np.isfinite(Qs)) and np.all(np.isfinite(Qv))
    assert Qs[ns.index("h")].min() > 0.0 and Qv[nv.index("h")].min() > 0.0
    p_scale = 9.81 * H0 * AMP
    for key in ("P_1_0", "P_1_1", "P_2_0", "P_2_1"):
        assert np.abs(Qv[nv.index(key)]).max() < p_scale, key


@pytest.mark.large
def test_mlvam21_matches_mlsme21_in_long_wave_limit():
    mesh = BaseMesh.create_1d(domain=(0.0, XMAX), n_inner_cells=NC)

    nsm, ns = _build_mlsme(mesh)
    solver = HyperbolicSolver(
        time_end=T_END, compute_dt=timestepping.adaptive(CFL=CFL, dimension=1))
    Qs, _ = solver.solve(mesh, nsm, write_output=False)
    Qs = np.asarray(Qs[:, :NC], float)

    sol, nv = _build_mlvam(mesh)
    t_now = 0.0
    while t_now < T_END - 1e-12:
        dt = min(_mlvam_dt(sol), T_END - t_now)
        sol.step(dt)
        t_now += dt
    Qv = np.asarray(sol._sim_Q[:, :NC], float)

    def rel(a, b, scale):
        return np.linalg.norm(a - b) / (np.linalg.norm(scale) + 1e-14)

    hs, hv = Qs[ns.index("h")], Qv[nv.index("h")]
    assert rel(hv, hs, hs - H0) < 0.10, "h deviates beyond O(mu)"
    for key in ("q_1_0", "q_2_0"):
        qs_, qv_ = Qs[ns.index(key)], Qv[nv.index(key)]
        assert rel(qv_, qs_, qs_) < 0.08, f"{key} deviates beyond O(mu)"
        assert np.corrcoef(qv_, qs_)[0, 1] > 0.99, f"{key} decorrelated"
    # mode-1 discharges (the per-layer shear) — same direction and shape
    # where there IS a signal; the TOP layer has no bed friction, so its
    # hydrostatic shear is noise-level (‖q_2_1‖ ~ 1e-4 of the q-scale) and
    # correlation is meaningless — bound it instead (the VAM's extra
    # ~1%-of-signal shear is the O(μ) non-hydrostatic correction itself)
    qscale = np.linalg.norm(Qs[ns.index("q_1_0")])
    for key in ("q_1_1", "q_2_1"):
        qs_, qv_ = Qs[ns.index(key)], Qv[nv.index(key)]
        if np.linalg.norm(qs_) > 0.05 * qscale:
            assert np.corrcoef(qv_, qs_)[0, 1] > 0.99, f"{key} decorrelated"
        else:
            assert np.linalg.norm(qv_) < 0.02 * qscale, f"{key} not small"
    # non-hydrostatic pressures stay a small correction
    p_scale = 9.81 * H0 * AMP
    for key in ("P_1_0", "P_1_1", "P_2_0", "P_2_1"):
        assert np.abs(Qv[nv.index(key)]).max() < 0.05 * p_scale, key
