"""Closed-domain mass-conservation gate for the VAM Chorin split predictor.

All-wall box + dam break: with the split predictor's mass flux correctly
classified as a CONSERVATIVE flux it must telescope → total mass constant to
round-off.  Misclassified as NCP (the bug) it leaks.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sympy as sp

from zoomy_core.model.models import VAM
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.boundary_conditions import BoundaryConditions, Wall
from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax

nc = 80
model = VAM(closures=[Newtonian(), NavierSlip(), StressFree()], level=1)
sm = model.system_model
names = [str(s) for s in sm.state]
ih = names.index("h")
print("state:", names)


def _dam_ic(xv):
    xx = float(xv[0])
    hv = 0.30 if xx < 0.0 else 0.10        # flat bed, depth step
    out = np.zeros(len(sm.state)); out[1] = hv
    return out


sm.initial_conditions = IC.UserFunction(function=_dam_ic)
sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
# closed box: reflect BOTH horizontal momentum modes q_0(idx2), q_1(idx3)
wall = Wall(tag="left", momentum_field_indices=[[2], [3]])
wall_r = Wall(tag="right", momentum_field_indices=[[2], [3]])
bcs = BoundaryConditions([wall, wall_r])
sm.attach_boundary_conditions(bcs)

split = model.chorin_split(sp.Symbol("dt", positive=True), system_model=sm)
split.SM_pred.attach_boundary_conditions(bcs)

mesh = BaseMesh.create_1d(domain=(-1.5, 1.5), n_inner_cells=nc)
solver = ChorinSplitVAMSolverJax(split.SM_pred, split.SM_press, split.SM_corr,
                                 pressure_tol=1e-9, pressure_maxit=200)
solver.setup_simulation(mesh); solver.update_aux_variables()
dx = float(solver._rt_mesh.cell_volumes[0])
dt = 0.3 * dx / (np.sqrt(9.81 * 0.30) + 1.0)

Q, qp, qpr, qc = (solver._sim_Q, solver._sim_Qaux, solver.Qaux_press, solver.Qaux_corr)
mass0 = float(np.asarray(Q)[ih, :nc].sum()) * dx
print(f"dt={dt:.5e}  mass0={mass0:.10f}")
t = 0.0
for k in range(20):
    Q, qp, qpr, qc, tf = solver.run_jit_steps(jnp.asarray(dt), 50, Q, qp, qpr, qc, t_start=t)
    Q.block_until_ready(); t = float(tf)
    Qn = np.asarray(Q)
    mass = float(Qn[ih, :nc].sum()) * dx
    fin = bool(np.all(np.isfinite(Qn[:, :nc])))
    print(f"  t={t:6.3f}  mass={mass:.10f}  drift={mass-mass0:+.3e}  "
          f"h[{Qn[ih,:nc].min():.4f},{Qn[ih,:nc].max():.4f}]  finite={fin}")
    if not fin:
        break
print(f"FINAL drift = {mass-mass0:+.3e}  (rel {abs(mass-mass0)/mass0:.2e})")
