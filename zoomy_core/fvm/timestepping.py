"""Time-stepping strategies for explicit FVM solvers."""

import numpy as np


def constant(dt=0.1):
    """Fixed timestep."""
    def compute_dt(Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue):
        return dt
    return compute_dt


def adaptive(CFL=0.9, nu=0.0):
    """Adaptive CFL-based timestep.

    Parameters
    ----------
    CFL : float
        Convective CFL number.
    nu : float
        Diffusivity for diffusive CFL condition.
        If > 0, also enforces Δt < CFL * Δx² / (2 * nu * dim).
    """
    def compute_dt(Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue):
        ev_abs_max = compute_max_abs_eigenvalue(Q, Qaux, parameters)
        dt_conv = (CFL * 2 * min_inradius / ev_abs_max).min()
        if nu > 0:
            # Diffusive CFL: Δt < CFL * Δx² / (2ν)
            dx = 2 * min_inradius  # characteristic cell size
            dt_diff = CFL * dx**2 / (2 * nu)
            return min(float(dt_conv), float(dt_diff))
        return dt_conv
    return compute_dt
