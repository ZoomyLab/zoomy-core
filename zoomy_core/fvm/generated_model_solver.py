"""Solver for free-surface flow models with derivative-aware auxiliary variables.

``GeneratedModelSolver`` = ``FreeSurfaceFlowSolver`` + ``DerivativeAwareSolverMixin``.

The mixin computes spatial derivatives (from ``model.derivative_specs``)
at each timestep and fills the ``Qaux`` array.  This enables models whose
source terms reference ``self.D.dx(self.Q.h)`` etc.
"""

from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
from zoomy_core.model.derivative_workflow import DerivativeAwareSolverMixin


class GeneratedModelSolver(DerivativeAwareSolverMixin, FreeSurfaceFlowSolver):
    """FreeSurfaceFlowSolver with derivative-aware Qaux updates.

    This is the standard solver for SWE / SME models.  For models without
    derivative-dependent source terms, ``FreeSurfaceFlowSolver`` is sufficient.
    """
    pass
