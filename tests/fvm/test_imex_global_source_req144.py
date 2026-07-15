"""REQ-144: the ``source_mode='global'`` implicit-source path must RUN.

``analytic_source_jvp`` reconstructed the parameter vector from
``symbolic_model.parameters.values()`` and ``float()``-ed it — but since REQ-163
those are FREE SYMBOLS, so it raised ``TypeError: Cannot convert expression to
float`` and the whole global Newton-GMRES path was dead.  The solver already
holds the NUMERIC ``parameters``; it must be threaded through.
"""
import numpy as np

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_imex_numpy import IMEXSolver
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.models.sme import SME
from zoomy_core.numerics.numerical_system_model import NumericalSystemModel


class _IMEXGlobal(IMEXSolver):
    source_mode = "global"


def test_global_implicit_source_runs_on_sme_friction():
    mesh = BaseMesh.create_2d(domain=(0.0, 10.0, 0.0, 1.0), nx=8, ny=4)
    nsm = NumericalSystemModel.from_model(
        SME(level=1, dimension=2, parameters={"g": 9.81, "n_m": 0.02}))
    solver = _IMEXGlobal(time_end=0.02, compute_dt=timestepping.adaptive(CFL=0.3))
    solver.setup_simulation(mesh, nsm, write_output=False)
    solver.step(1e-3)                       # was TypeError on the first JVP
    assert np.isfinite(solver._sim_Q).all()
