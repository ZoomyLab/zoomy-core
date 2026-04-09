"""GeneratedModelSolver — backward-compatible alias for ShallowWaterSolver.

The functionality that was in GeneratedModelSolver is now split:
- HyperbolicSolver: general models (any variable layout)
- ShallowWaterSolver: models with b/h (hydrostatic reconstruction, wet/dry)
"""

from zoomy_core.fvm.solver_numpy import ShallowWaterSolver
from zoomy_core.model.derivative_workflow import DerivativeAwareSolverMixin


class GeneratedModelSolver(DerivativeAwareSolverMixin, ShallowWaterSolver):
    """ShallowWaterSolver with derivative-aware auxiliary variable updates.

    Backward-compatible alias. For new code, use ShallowWaterSolver or
    HyperbolicSolver directly.
    """
    pass
