"""Characteristic boundary conditions: ghost = Q + P⁻·(target − Q) with
P⁻ = R·1_{λ<0}·L over the SystemModel's opaque eigensystem kernel — only
incoming characteristics carry the target's information (the data-level
analogue of Roe upwinding; the principled coupled-boundary ghost)."""
import numpy as np

from zoomy_core.model.models import SME
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, CharacteristicFarField, CharacteristicWall)
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec


def _run(bcs, ic, t_end, nc=100):
    sm = SME(level=0, boundary_conditions=bcs).system_model
    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=nc)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=t_end,
                              compute_dt=timestepping.adaptive(CFL=0.9))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    return np.asarray(Q[:, :nc], float)


def test_farfield_lake_at_rest_stays():
    ff = [CharacteristicFarField(tag=tg, far_field=[0.0, 1.0, 0.0])
          for tg in ("left", "right")]
    Q = _run(BoundaryConditions(ff),
             lambda xv: np.array([0.0, 1.0, 0.0]), t_end=0.5)
    assert np.abs(Q[1] - 1.0).max() < 1e-12
    assert np.abs(Q[2]).max() < 1e-12


def test_farfield_lets_hump_exit():
    """A surface hump radiates out through the far-field boundaries; the
    domain relaxes back to the far-field state with small residual."""
    ff = [CharacteristicFarField(tag=tg, far_field=[0.0, 1.0, 0.0])
          for tg in ("left", "right")]

    def ic(xv):
        xx = float(xv[0])
        return np.array([0.0, 1.0 + 0.2 * np.exp(-((xx - 5.0) ** 2) / 0.5), 0.0])

    Q = _run(BoundaryConditions(ff), ic, t_end=1.5)
    assert np.all(np.isfinite(Q))
    assert np.abs(Q[1] - 1.0).max() < 0.08      # hump (0.2) mostly gone


def test_characteristic_wall_box_conserves_mass():
    wl = CharacteristicWall(tag="left", momentum_field_indices=[[2]])
    wr = CharacteristicWall(tag="right", momentum_field_indices=[[2]])

    def ic(xv):
        return np.array([0.0, 1.2 if float(xv[0]) < 5.0 else 1.0, 0.0])

    Q = _run(BoundaryConditions([wl, wr]), ic, t_end=1.0)
    assert np.all(np.isfinite(Q))
    mass = Q[1].sum() * 0.1
    assert abs(mass - 11.0) < 1e-9
