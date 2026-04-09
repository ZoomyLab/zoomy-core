"""Backward-compatible aliases for the solver hierarchy.

New code should use the solvers directly:
- ``HyperbolicSolver`` — general explicit FVM
- ``FreeSurfaceFlowSolver`` — explicit FVM for h/b models
- ``IMEXSolver`` — explicit flux + implicit source
- ``FSFIMEXSolver`` — IMEX for h/b models (replaces GeneratedModelSolver)
"""

from zoomy_core.fvm.solver_imex_numpy import FSFIMEXSolver


# Backward-compatible alias
GeneratedModelSolver = FSFIMEXSolver
