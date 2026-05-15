"""Chorin projection split for the VAM chain DAE (numpy backend).

Consumes the three rectangular SystemModels produced by
:func:`zoomy_core.model.splitter.split_for_pressure`:

* ``SM_pred``  — explicit predictor.  Evolution rows (mass + xmom_jk +
  zmom_jk) with the pressure modes frozen at the current step.  Solved
  with an SSP-RK explicit time stepper on the Rusanov + non-conservative
  flux machinery; no Newton, no Jacobian.
* ``SM_press`` — algebraic elliptic block on the pressure modes
  (``elliptic_jk`` rows).  Linear in ``(P_k, ∂_x P_k, ∂_xx P_k)``;
  assembled into a sparse matrix using the LSQ derivative operators
  and solved per step.
* ``SM_corr``  — closed-form algebraic corrector on the velocity modes
  (``corr_U_k``, ``corr_W_k``).  Evaluated element-wise — no solve.

The split is the structural fix for the order-2 ill-conditioning of
the monolithic :class:`DAESolver`: the hyperbolic transport block is
well-conditioned and explicit; the elliptic pressure block becomes a
linear solve with proper preconditioning; the corrector is closed
form.  Newton iteration disappears entirely.
"""
from __future__ import annotations

import time as _time
from typing import Optional

import numpy as np
import param
import sympy as sp

from zoomy_core.fvm.solver_numpy import Solver
from zoomy_core.fvm import timestepping
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.misc.logger_config import logger


_EMPTY_AUX = np.empty(0, dtype=float)


class ChorinSplitVAMSolver(Solver):
    """Chorin projection split for the VAM chain DAE.

    Consumes three pre-built sub-system models.  The splitter
    (:func:`split_for_pressure`) is the factory; the solver is
    downstream and ignorant of ``pressure_vars`` / ``dt`` / ``bottom``
    — every piece of metadata it needs (which state indices to update,
    which rows are elliptic) is carried on the sub-systems themselves
    via ``equation_to_state_index`` and ``equation_names``.
    """

    time_end = param.Number(default=0.1, bounds=(0, None),
                            doc="Simulation end time")
    method = param.Selector(default="ssprk2",
                            objects=["euler", "ssprk2"],
                            doc="Predictor explicit time-integration scheme")
    compute_dt = param.Parameter(
        default=None,
        doc="Adaptive-timestep callable; defaults to "
            "``timestepping.adaptive(CFL=0.3)``")
    reconstruction_order = param.Integer(
        default=2, bounds=(1, 2),
        doc="Spatial reconstruction order for the predictor: "
            "1 = piecewise-constant, 2 = MUSCL with η = h+b "
            "surface reconstruction.")
    limiter = param.Selector(
        default="venkatakrishnan",
        objects=["venkatakrishnan", "barth_jespersen", "minmod"],
        doc="Slope limiter for MUSCL reconstruction")
    pressure_tol = param.Number(
        default=1e-10, bounds=(0, None),
        doc="Linear-solver tolerance for the elliptic pressure block")
    pressure_maxit = param.Integer(
        default=500, bounds=(1, None),
        doc="Maximum iterations for the elliptic pressure solve")
    h_index = param.Integer(
        default=0, bounds=(0, None),
        doc="State index of the water-depth-like variable used for "
            "max-eigenvalue / CFL estimation.")

    def __init__(self, sm_pred, sm_press, sm_corr, **kwargs):
        if not isinstance(sm_pred, SystemModel):
            raise TypeError("sm_pred must be a SystemModel")
        if not isinstance(sm_press, SystemModel):
            raise TypeError("sm_press must be a SystemModel")
        if not isinstance(sm_corr, SystemModel):
            raise TypeError("sm_corr must be a SystemModel")
        super().__init__(**kwargs)
        self.sm_pred = sm_pred
        self.sm_press = sm_press
        self.sm_corr = sm_corr
        # The full chain state lives on all three sub-systems; pick one.
        self.state = list(sm_pred.state)
        self.n_state = sm_pred.n_state

        # Detect the symbolic time-step.  The splitter bakes ``dt`` into
        # SM_press and SM_corr (via ``U_corr = U_tilde - (dt/h)·T_u``);
        # SM_pred does not depend on it.
        self._dt_symbol = self._detect_dt_symbol()

        # Sanity: the three sub-systems must agree on the state vector.
        for name, sm in (("press", sm_press), ("corr", sm_corr)):
            if [str(s) for s in sm.state] != [str(s) for s in sm_pred.state]:
                raise ValueError(
                    f"SM_{name}.state disagrees with SM_pred.state — the "
                    "three sub-systems must share a common state vector."
                )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_simulation(self, mesh):
        """Build runtimes for the three sub-systems and pre-allocate
        Q / Qaux.  Topography injection / IC overrides happen on the
        caller's side (see ``test_vam_topography_dae.py`` for the
        pattern)."""
        t0 = _time.time()
        # All three sub-systems share the chain-DAE aux registry through
        # their parent — promote the mesh to an LSQMesh of the right
        # degree (degree 2 by default for the order-2 predictor).
        lsq_degree = 2 if self.reconstruction_order == 2 else 1
        # ``ensure_lsq_mesh`` looks for ``aux_registry`` on its second
        # argument; SM_pred carries the full registry.
        mesh = ensure_lsq_mesh(mesh, self.sm_pred, lsq_degree=lsq_degree)
        # Each sub-system carries the parent's indexed BC kernel
        # (propagated through ``_build_subsystem``); periodic-BC
        # resolution still needs the original ``BoundaryConditions``
        # object, which lives on the parent model — callers that need
        # periodic boundaries must hand in an already-resolved
        # ``LSQMesh``.

        # Normalise the dt Symbol: the splitter convention is
        # ``\Delta t`` (LaTeX-friendly), but Python lambdify reads the
        # backslash as a line-continuation and chokes.  Rename to a
        # Python-safe ``_dt`` throughout SM_press / SM_corr operators
        # before building runtimes.  Also register ``_dt`` as a
        # parameter on each affected sub-system so the lambdify
        # signature carries it explicitly (we bind a numeric value
        # per substep at runtime).
        if self._dt_symbol is not None:
            # Use ``dt`` (no leading underscore — ``Zstruct._filter_dict``
            # hides ``_``-prefixed keys, which would silently drop the
            # parameter from the lambdify signature).
            dt_safe = sp.Symbol("dt", positive=True)
            self._dt_symbol_safe = dt_safe
            self.sm_press = _substitute_dt(self.sm_press, self._dt_symbol,
                                           dt_safe)
            self.sm_corr = _substitute_dt(self.sm_corr, self._dt_symbol,
                                          dt_safe)
            from zoomy_core.misc.misc import Zstruct
            for sm in (self.sm_press, self.sm_corr):
                if not sm.parameters.contains("dt"):
                    new_params = Zstruct(**sm.parameters.as_dict())
                    new_params["dt"] = dt_safe
                    sm.parameters = new_params
                    new_pvals = Zstruct(**sm.parameter_values.as_dict())
                    new_pvals["dt"] = 0.0   # placeholder; set per step
                    sm.parameter_values = new_pvals
        else:
            self._dt_symbol_safe = None

        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells
        n_vars = self.n_state

        # Build a runtime for each sub-system.  ``_dt`` is a free Symbol
        # in SM_press / SM_corr — we bind it via a transient
        # ``parameter_values`` extension at step time (see ``_bind_dt``).
        self.rt_pred = NumpyRuntimeModel.from_system_model(self.sm_pred)
        self.rt_press = NumpyRuntimeModel.from_system_model(self.sm_press)
        self.rt_corr = NumpyRuntimeModel.from_system_model(self.sm_corr)

        # Parameter arrays for each runtime (pulled from each SM's
        # parameter_values).  ``dt`` is appended at step time.
        self._params_pred = np.array(
            list(self.sm_pred.parameter_values.values()), dtype=float,
        )
        self._params_press_base = np.array(
            list(self.sm_press.parameter_values.values()), dtype=float,
        )
        self._params_corr_base = np.array(
            list(self.sm_corr.parameter_values.values()), dtype=float,
        )

        # Allocate state arrays.  ``Q`` carries the full 7-state shared
        # across the three sub-systems.
        Q = np.zeros((n_vars, nc), dtype=float)

        # Each sub-system has its own ``aux_state`` from its independent
        # auto-scan.  For the first cut we allocate the largest aux row
        # count and let each runtime index into the shared array; a
        # later refactor can unify the aux pool.
        n_aux = max(
            len(self.sm_pred.aux_state),
            len(self.sm_press.aux_state),
            len(self.sm_corr.aux_state),
        )
        Qaux = np.zeros((n_aux, nc), dtype=float)

        self._sim_mesh = mesh
        self._sim_Q = Q
        self._sim_Qaux = Qaux
        self._sim_time = 0.0
        self._sim_nc = nc
        self._sim_n_cells = n_cells

        # State-index slices for the three stages.
        self._pred_state_idx = np.asarray(
            self.sm_pred.equation_to_state_index, dtype=int)
        self._press_state_idx = np.asarray(
            self.sm_press.equation_to_state_index, dtype=int)
        self._corr_state_idx = np.asarray(
            self.sm_corr.equation_to_state_index, dtype=int)

        logger.info(
            "ChorinSplitVAMSolver setup: %d state, %d aux, %d cells "
            "(pred → %s, press → %s, corr → %s) in %.2fs",
            n_vars, n_aux, nc,
            self._pred_state_idx.tolist(),
            self._press_state_idx.tolist(),
            self._corr_state_idx.tolist(),
            _time.time() - t0,
        )
        return Q

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def nc(self):
        return self._sim_nc

    @property
    def sm(self):
        """Convenience: the predictor sub-system is the canonical
        carrier of the full state."""
        return self.sm_pred

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self, dt):
        """One Chorin step: predictor → pressure → corrector.

        This is the minimum-viable wiring — predictor uses a crude
        central-flux Euler step (no Riemann solver, naive BCs); pressure
        uses ``scipy.optimize.fsolve`` on the lambdified elliptic
        residual (linear ⇒ converges in 1–2 Newton steps); corrector
        applies ``state_update`` in place.  Order-of-accuracy work
        (SSP-RK2 predictor with MUSCL, matrix-free GMRES pressure,
        well-balancing) layers on top of this.
        """
        self._step_predictor(dt)
        self._step_pressure(dt)
        self._step_corrector(dt)
        self._sim_time += dt

    def _params_with_dt(self, base, dt):
        """Return a copy of ``base`` with the last entry (``_dt``)
        replaced by the numeric ``dt``.

        ``_dt`` is appended to each dt-dependent subsystem's
        ``parameter_values`` Zstruct in ``setup_simulation`` with a
        placeholder 0.0; per substep we splice in the real dt before
        invoking the runtime.  This keeps the lambdified arg count
        constant (= len(parameter_values))."""
        out = base.copy()
        out[-1] = float(dt)
        return out

    def _step_predictor(self, dt):
        """Crude explicit Euler on SM_pred — central flux divergence
        + source.  Boundary cells: zero-gradient (copy interior)."""
        rt = self.rt_pred
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        p = self._params_pred
        nc = self._sim_nc
        dx = float(self._sim_mesh.cell_volumes[0])

        # flux: shape (n_eq=5, n_dim=1, nc) after vectorised lambdify.
        F = np.asarray(rt.flux(Q, Qaux, p), dtype=float)
        if F.ndim == 3:
            F1d = F[:, 0, :]               # (5, nc) — 1-D slab
        else:
            F1d = F
        # Central-difference flux divergence with zero-gradient ghosts.
        F_left  = np.concatenate([F1d[:, :1], F1d[:, :-1]], axis=1)
        F_right = np.concatenate([F1d[:, 1:], F1d[:, -1:]], axis=1)
        F_div   = (F_right - F_left) / (2.0 * dx)

        # source: shape (n_eq, nc)
        S = np.asarray(rt.source(Q, Qaux, p), dtype=float)
        if S.ndim == 1:
            S = S[:, None] * np.ones((1, nc))

        # Euler update on the predictor's state slots.
        e2s = self._pred_state_idx
        Q[e2s, :] = Q[e2s, :] + dt * (-F_div + S)
        self._sim_Q = Q

    def _step_pressure(self, dt):
        """Solve the algebraic elliptic block ``R(P; Q, Qaux, dt) = 0``
        for ``Q[press_e2s]`` via scipy.optimize.fsolve.  R is linear
        in P → Newton converges in 1–2 iterations from any start."""
        from scipy.optimize import fsolve
        rt = self.rt_press
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        nc = self._sim_nc
        p_full = self._params_with_dt(self._params_press_base, dt)
        e2s = self._press_state_idx
        nP = len(e2s)

        def residual_of_P(p_vec):
            Q_try = Q.copy()
            Q_try[e2s, :] = p_vec.reshape(nP, nc)
            # SM_press carries the elliptic residual in its ``source``
            # field (after auto-tagging).  Returns (nP, nc).
            R = np.asarray(rt.source(Q_try, Qaux, p_full), dtype=float)
            if R.ndim == 1:
                R = R[:, None] * np.ones((1, nc))
            return R.ravel()

        p0 = Q[e2s, :].ravel()              # warm-start from current P
        p_new, _, info, msg = fsolve(
            residual_of_P, p0, full_output=True,
            xtol=self.pressure_tol, maxfev=self.pressure_maxit,
        )
        Q[e2s, :] = p_new.reshape(nP, nc)
        self._sim_Q = Q

    def _step_corrector(self, dt):
        """Apply ``Q[corr_e2s] ← state_update(Q, Qaux, p, dt)`` in place.
        Read-then-write is atomic per row: Python evaluates the RHS
        callable fully before assigning to the LHS slice."""
        rt = self.rt_corr
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        p_full = self._params_with_dt(self._params_corr_base, dt)
        e2s = self._corr_state_idx

        new_vals = np.asarray(rt.state_update(Q, Qaux, p_full), dtype=float)
        Q[e2s, :] = new_vals
        self._sim_Q = Q

    def _detect_dt_symbol(self) -> Optional[sp.Symbol]:
        """Locate the symbolic time-step common to SM_press / SM_corr.

        The splitter bakes a free Symbol named ``dt``, ``Delta t``, or
        ``\\Delta t`` (sympy LaTeX form) into the elliptic and
        corrector operators.  Returns the Symbol, or ``None`` when the
        two sub-systems are dt-free (degenerate case, e.g. testing).
        """
        candidates = set()
        for sm in (self.sm_press, self.sm_corr):
            for atom in sm.source.free_symbols | sm.flux.free_symbols:
                if atom in sm.parameters.values():
                    continue
                if atom in sm.state:
                    continue
                if atom in sm.aux_state:
                    continue
                name = str(atom)
                if name in {"dt", "Delta t", r"\Delta t"}:
                    candidates.add(atom)
        if len(candidates) > 1:
            raise ValueError(
                "Multiple dt-like free Symbols in SM_press/SM_corr: "
                f"{[str(c) for c in candidates]}"
            )
        return next(iter(candidates)) if candidates else None


def _substitute_dt(sm, old_sym, new_sym):
    """Rename the dt Symbol throughout a sub-system's operator tensors.

    The Chorin splitter bakes a free Symbol named ``\\Delta t`` into
    ``SM_press`` / ``SM_corr``.  That LaTeX-flavoured name is sympy-
    correct but breaks Python lambdify (the backslash reads as a line
    continuation).  Substitute it for a Python-safe identifier
    everywhere it appears in the symbolic operators before lambdifying.
    """
    from zoomy_core.model.models.system_model import _to_zarray
    sub = {old_sym: new_sym}
    def _xrepl(tensor):
        if tensor is None:
            return None
        zarr = _to_zarray(tensor)
        return zarr.xreplace(sub)
    sm.flux                   = _xrepl(sm.flux)
    sm.hydrostatic_pressure   = _xrepl(sm.hydrostatic_pressure)
    sm.source                 = _xrepl(sm.source)
    sm.mass_matrix            = _xrepl(sm.mass_matrix)
    sm.nonconservative_matrix = _xrepl(sm.nonconservative_matrix)
    sm.refresh_derived_operators(eigenvalues=False)
    sm.eigenvalues            = _xrepl(sm.eigenvalues)
    sm.state_update           = _xrepl(sm.state_update)
    return sm


def _ndarray_iter(shape):
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for rest in _ndarray_iter(shape[1:]):
            yield (i,) + rest


__all__ = ["ChorinSplitVAMSolver"]
