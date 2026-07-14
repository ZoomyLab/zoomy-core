"""REQ-148 — the numpy SplittingSolver must be constructible and runnable.

The class extended ``IMEXSolver`` but its ``setup_simulation`` never assigned
``self.nsm``; the base ``get_flux_operator`` /
``get_compute_max_abs_eigenvalue`` read ``self.nsm.reconstruction`` and
``self.nsm.eigenvalue_eps``, so it crashed with ``AttributeError: 'nsm'`` for
ANY model.  This smoke test pins the construct → setup → step path (the GUI
"Split (NumPy)" card runs exactly this).
"""

import numpy as np

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_splitting_numpy import FSFSplittingSolver
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics.numerical_system_model import NumericalSystemModel


def test_splitting_solver_sets_nsm_and_steps():
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=32)
    m = SWE(dimension=1, parameters={"g": 9.81})
    solver = FSFSplittingSolver(
        time_end=0.02,
        compute_dt=timestepping.adaptive(CFL=0.4),
        viscosity=1e-3,
    )
    solver.setup_simulation(mesh, m, write_output=False)
    # The one wiring the class was missing.
    assert isinstance(solver.nsm, NumericalSystemModel)
    for _ in range(3):
        solver.step(0.001)
    assert np.isfinite(solver._sim_Q).all()


def test_splitting_solver_accepts_prebuilt_nsm():
    mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=16)
    nsm = NumericalSystemModel.from_model(SWE(dimension=1, parameters={"g": 9.81}))
    solver = FSFSplittingSolver(
        time_end=0.01, compute_dt=timestepping.adaptive(CFL=0.4))
    solver.setup_simulation(mesh, nsm, write_output=False)
    assert solver.nsm is nsm
    solver.step(0.001)
    assert np.isfinite(solver._sim_Q).all()
