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


def adaptive(CFL=0.9, nu=0.0, dimension=2, degree=0, dt_max=None):
    """Adaptive CFL-based timestep.

    Uses the classical hyperbolic / parabolic limits with the spatial
    dimension and DG polynomial degree factored *into the denominator*
    so the ``CFL`` knob is a single safety factor ∈ (0, 1]:

    .. math::

        \\Delta t_\\text{conv} &\\le \\mathrm{CFL} \\;
            \\frac{2\\,r_\\text{in}}{d \\, (2k+1) \\, |\\lambda|_\\text{max}}, \\\\
        \\Delta t_\\text{diff} &\\le \\mathrm{CFL} \\;
            \\frac{(2\\,r_\\text{in})^2}{2 \\, d \\, \\nu}.

    Here ``r_in`` is the cell inradius (so ``2·r_in`` is the conservative
    "diameter"), ``d`` the spatial dimension, ``k`` the DG polynomial
    degree (use ``0`` for FV).  The hyperbolic CFL constant ``1/(2k+1)``
    is the SSP-RK(k+1,k+1) stability limit (Cockburn & Shu, 1991);
    the spatial dimension contributes the ``1/d`` factor in 2D/3D.

    Parameters
    ----------
    CFL : float in (0, 1]
        Safety factor on top of the theoretical limit (default 0.9).
    dt_max : float, optional
        Hard upper cap on dt — for explicit SOURCE stiffness (e.g. the
        regularized-Bingham plug viscosity or a bed-slip penalty) that
        the hyperbolic CFL cannot see.  Backend-neutral (JIT-safe):
        prefer this over wrapping ``compute_dt`` in a Python
        ``min(float(...))`` lambda, which fails under ``jax.jit``.
    nu : float
        Scalar viscosity for the parabolic CFL (default 0 → skip).
    dimension : int
        Spatial dimension of the mesh (default 2).
    degree : int
        DG polynomial degree (default 0 for FV).
    """
    deg_fac = float(2 * degree + 1)
    dim_fac = float(dimension)

    def compute_dt(Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue):
        ev_abs_max = compute_max_abs_eigenvalue(Q, Qaux, parameters)
        h = 2.0 * min_inradius                                # conservative cell size
        dt = (CFL * h / (dim_fac * deg_fac * ev_abs_max)).min()
        if nu > 0:
            dt = _safe_minimum(dt, CFL * h ** 2 / (2.0 * dim_fac * nu))
        if dt_max is not None:
            dt = _safe_minimum(dt, dt_max)
        return dt
    return compute_dt
