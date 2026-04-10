"""Time-stepping strategies for explicit FVM solvers.

All timestep functions return a ``compute_dt`` closure that is
compatible with both NumPy and JAX (JIT-safe: no Python ``float()``
or ``min()`` on traced values).
"""

import numpy as np


def _safe_minimum(a, b):
    """Element-wise minimum that works with both NumPy arrays and JAX tracers."""
    try:
        import jax.numpy as jnp
        return jnp.minimum(a, b)
    except ImportError:
        return np.minimum(a, b)


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
        If > 0, also enforces dt < CFL * dx^2 / (2 * nu).
    """
    def compute_dt(Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue):
        ev_abs_max = compute_max_abs_eigenvalue(Q, Qaux, parameters)
        dt_conv = (CFL * 2 * min_inradius / ev_abs_max).min()
        if nu > 0:
            # Diffusive CFL: dt < CFL * dx^2 / (2*nu)
            dx = 2 * min_inradius  # characteristic cell size
            dt_diff = CFL * dx**2 / (2 * nu)
            return _safe_minimum(dt_conv, dt_diff)
        return dt_conv
    return compute_dt
