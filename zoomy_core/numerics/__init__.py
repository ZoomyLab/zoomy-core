"""Numerical sibling of the symbolic ``zoomy_core.model`` layer.

This subpackage holds the numerical setup that the FV / JAX solvers
consume.  The central type is :class:`NumericalSystemModel`, a wrapper
around a :class:`zoomy_core.systemmodel.system_model.SystemModel` that
also carries the Riemann-solver class, the LSQ reconstruction spec, the
diffusion scheme, and numerical regularization knobs.

Pipeline:  Model → SystemModel → NumericalSystemModel → Solver
"""

from zoomy_core.numerics.numerical_system_model import (
    DiffusionSpec,
    NumericalSystemModel,
    ReconstructionSpec,
    RegularizationSpec,
)

__all__ = [
    "DiffusionSpec",
    "NumericalSystemModel",
    "ReconstructionSpec",
    "RegularizationSpec",
]
