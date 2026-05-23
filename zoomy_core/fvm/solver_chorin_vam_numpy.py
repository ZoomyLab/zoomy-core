"""Chorin projection split for the VAM chain DAE (numpy backend).

Consumes the three sub-system models produced by
:func:`zoomy_core.model.splitter.split_for_pressure`:

* ``SM_pred``  — explicit predictor.  Evolution rows (mass + xmom_jk +
  zmom_jk) with the pressure modes implicitly frozen at the current
  step (read straight from ``Q[pressure_idx]``).  Solved with
  :class:`HyperbolicSolver`'s Rusanov + non-conservative path-integral
  + indexed-BC flux machinery — same path the SWE/SME solvers use.
  Mass-conservative by construction.
* ``SM_press`` — algebraic elliptic block on the pressure modes.
  Linear in ``(P_k, ∂_x P_k, ∂_xx P_k)``; solved via
  ``scipy.optimize.fsolve`` on the lambdified residual (converges in
  1–2 Newton iterations).  Matrix-free GMRES is the next refinement.
* ``SM_corr``  — closed-form algebraic corrector on the velocity modes
  via the ``state_update`` field.  Single lambdify call + in-place
  assignment, no solve.
"""
from __future__ import annotations

import time as _time
from typing import Optional

import numpy as np
import param
import sympy as sp

from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm.riemann_solvers import (
    NonconservativeRusanov,
    PositiveNonconservativeRusanov,
)
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.model.models.system_model import SystemModel, _to_zarray
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.misc.logger_config import logger
from zoomy_core.misc.misc import Zstruct, ZArray


_EMPTY_AUX = np.empty(0, dtype=float)


def _apply_cell_limiter(u, grad, multi_index, scheme, mesh):
    """Apply a TVD limiter to a cell-centred gradient.

    Parameters
    ----------
    u : ndarray, shape ``(nc,)``
        Cell-centre values of the field being differentiated.
    grad : ndarray, shape ``(nc,)``
        Unlimited LSQ-computed gradient at cell centres.
    multi_index : tuple
        Derivative order (only first-order — ``(1,)`` in 1D — is
        limited; higher orders pass through unchanged).
    scheme : str
        One of ``"minmod"``, ``"venkatakrishnan"``, ``"barth_jespersen"``.
    mesh : object with ``cell_centers`` attribute.

    Returns
    -------
    ndarray, shape ``(nc,)``
        Limited gradient: ``φ · grad`` where ``φ ∈ [0, 1]`` is
        cell-wise.

    Convention follows MUSCL slope limiting (LeVeque, *Finite Volume
    Methods for Hyperbolic Problems*, §6.9): the limiter compares the
    LSQ-extrapolated face value against the upwind neighbour bound and
    clips the slope so the reconstruction stays monotone.

    Higher-order derivatives (e.g. ``∂_xx``) are NOT limited here —
    they live in the elliptic block's principal part and need their
    own (e.g. compact-FD) discretisation if shock-robustness is
    required.  Limiting them with this routine would zero them at
    smooth maxima and break consistency.
    """
    if max(multi_index) != 1:
        return grad
    nc = u.shape[0]
    if nc < 3:
        return grad
    # 1D cell spacing — assumes uniform grid for the cell-centred
    # forward/backward deltas.  ``mesh.cell_centers`` has shape
    # ``(dim, n_cells)`` or ``(n_cells, dim)``; handle either.
    xc = mesh.cell_centers
    if xc.ndim == 2 and xc.shape[0] <= 3:
        x1d = xc[0, :nc]
    else:
        x1d = xc[:nc, 0]
    dx = float(np.mean(np.diff(x1d)))
    if dx <= 0:
        return grad
    # Forward / backward neighbour differences.
    d_F = np.zeros(nc); d_B = np.zeros(nc)
    d_F[:-1] = (u[1:] - u[:-1]) / dx
    d_F[-1] = d_F[-2]
    d_B[1:] = (u[1:] - u[:-1]) / dx
    d_B[0] = d_B[1]
    if scheme == "minmod":
        # Cell-centred minmod: clip slope to MIN-magnitude
        # neighbour delta of consistent sign.
        eps = 1e-14
        same_sign = (d_F * d_B) > eps
        slope_lim = np.where(
            same_sign,
            np.sign(d_F) * np.minimum(np.abs(d_F), np.abs(d_B)),
            0.0,
        )
        # φ = ratio of limited slope to LSQ slope, clipped to [0, 1].
        phi = np.where(np.abs(grad) > eps,
                       np.clip(slope_lim / np.where(np.abs(grad) > eps,
                                                     grad, 1.0), 0.0, 1.0),
                       1.0)
        return phi * grad
    elif scheme == "barth_jespersen":
        # Cell-bound BJ: u_min/u_max from immediate neighbours.
        u_min = np.minimum(np.roll(u, 1), np.roll(u, -1))
        u_min = np.minimum(u_min, u)
        u_max = np.maximum(np.roll(u, 1), np.roll(u, -1))
        u_max = np.maximum(u_max, u)
        eps = 1e-14
        # Reconstructed face values at the two cell faces.
        u_F = u + grad * (dx / 2)
        u_B = u - grad * (dx / 2)
        phi = np.ones(nc)
        # Where u_F exceeds u_max or undershoots u_min, scale.
        for u_face in (u_F, u_B):
            over = u_face > u_max + eps
            under = u_face < u_min - eps
            ratio_over = np.where(np.abs(u_face - u) > eps,
                                  (u_max - u) / np.where(np.abs(u_face - u) > eps,
                                                          u_face - u, 1.0),
                                  1.0)
            ratio_under = np.where(np.abs(u_face - u) > eps,
                                   (u_min - u) / np.where(np.abs(u_face - u) > eps,
                                                           u_face - u, 1.0),
                                   1.0)
            phi = np.where(over, np.minimum(phi, ratio_over), phi)
            phi = np.where(under, np.minimum(phi, ratio_under), phi)
        phi = np.clip(phi, 0.0, 1.0)
        return phi * grad
    elif scheme == "venkatakrishnan":
        # Smooth variant of BJ — same bounds machinery but with K-eps
        # smoothing to remove the kink.  Use K = 5 (standard).
        K = 5.0
        u_min = np.minimum(np.roll(u, 1), np.roll(u, -1))
        u_max = np.maximum(np.roll(u, 1), np.roll(u, -1))
        eps2 = (K * dx) ** 3
        d_max = u_max - u
        d_min = u_min - u
        # Per-face evaluation.
        phi = np.ones(nc)
        for sign in (+1, -1):
            d_face = grad * (sign * dx / 2)
            num = d_face * (d_max if sign > 0 else d_min)
            den_sq = (d_max if sign > 0 else d_min) ** 2
            psi = ((den_sq + eps2) * d_face + 2 * d_face**2 * (
                d_max if sign > 0 else d_min)) / (
                d_face * (den_sq + 2 * d_face**2 + (
                    d_max if sign > 0 else d_min) * d_face + eps2) + 1e-30
            )
            phi = np.where(d_face != 0, np.minimum(phi, psi), phi)
        phi = np.clip(phi, 0.0, 1.0)
        return phi * grad
    else:
        raise ValueError(
            f"_apply_cell_limiter: unknown scheme {scheme!r}.  "
            "Supported: 'minmod', 'barth_jespersen', 'venkatakrishnan'."
        )


def _pad_to_square(sm: SystemModel) -> SystemModel:
    """Return a square SystemModel whose operators have ``n_state`` rows.

    Used by :class:`ChorinSplitVAMSolver` to make a rectangular sub-
    system consumable by :class:`HyperbolicSolver`'s flux machinery
    (which assumes ``n_equations == n_state``).  Rows at state indices
    *not* in ``sm.equation_to_state_index`` get all-zero operators
    (flux / source / NCP / etc.) — they contribute zero RHS to the RK,
    so the corresponding state slots stay frozen through the predictor
    substep.  This is precisely the Chorin semantics: predictor doesn't
    touch pressure modes.
    """
    n_eq = sm.n_equations
    n_st = sm.n_state
    if n_eq == n_st:
        return sm

    e2s = list(sm.equation_to_state_index)
    n_dim = sm.n_dim

    # Map: dest_state_row → src_eq_row.  Missing states get None.
    src_of_dest = {state_i: src_i for src_i, state_i in enumerate(e2s)}

    def _pad_rank2(arr, n_cols):
        """``(n_eq, n_cols)`` → ``(n_st, n_cols)`` ZArray with zero rows
        at missing dest indices.  ``arr`` may be ZArray or sp.Matrix."""
        if arr is None:
            return None
        out_rows = []
        for state_i in range(n_st):
            if state_i in src_of_dest:
                src = src_of_dest[state_i]
                row = [sp.sympify(arr[src, j]) for j in range(n_cols)]
            else:
                row = [sp.S.Zero] * n_cols
            out_rows.append(row)
        return ZArray(out_rows, shape=(n_st, n_cols))

    def _pad_rank1(arr):
        """``(n_eq,)`` → ``(n_st,)`` ZArray with zeros at missing indices."""
        if arr is None:
            return None
        out = []
        for state_i in range(n_st):
            if state_i in src_of_dest:
                src = src_of_dest[state_i]
                # arr may be rank-1 or rank-2 (n_eq, 1) — handle both.
                if len(arr.shape) == 1:
                    out.append(sp.sympify(arr[src]))
                else:
                    out.append(sp.sympify(arr[src, 0]))
            else:
                out.append(sp.S.Zero)
        return ZArray(out, shape=(n_st,))

    def _pad_rank3(arr):
        """``(n_eq, n_st_cols, n_dim)`` → ``(n_st, n_st_cols, n_dim)``."""
        if arr is None:
            return None
        n_st_cols = arr.shape[1]
        out = []
        for state_i in range(n_st):
            row = []
            if state_i in src_of_dest:
                src = src_of_dest[state_i]
                for j in range(n_st_cols):
                    inner = [sp.sympify(arr[src, j, d]) for d in range(n_dim)]
                    row.append(inner)
            else:
                for j in range(n_st_cols):
                    row.append([sp.S.Zero] * n_dim)
            out.append(row)
        return ZArray(out, shape=(n_st, n_st_cols, n_dim))

    # Pad each operator.  Source can be rank-1 (post-refactor) or (n_eq,1).
    new_flux = _pad_rank2(sm.flux, n_dim)
    new_P    = _pad_rank2(sm.hydrostatic_pressure, n_dim)
    new_B    = _pad_rank3(sm.nonconservative_matrix)
    new_S    = (_pad_rank2(sm.source, sm.source.shape[1])
                if len(sm.source.shape) == 2
                else _pad_rank1(sm.source))
    new_M    = _pad_rank2(sm.mass_matrix, n_st)
    new_E    = _pad_rank1(sm.eigenvalues) if sm.eigenvalues is not None else None

    sm_square = SystemModel(
        time=sm.time,
        space=list(sm.space),
        state=list(sm.state),
        aux_state=list(sm.aux_state),
        parameters=sm.parameters,
        parameter_values=sm.parameter_values,
        flux=new_flux,
        hydrostatic_pressure=new_P,
        nonconservative_matrix=new_B,
        source=new_S,
        mass_matrix=new_M,
        eigenvalues=new_E,
        equation_to_state_index=list(range(n_st)),
        boundary_conditions=sm.boundary_conditions,
        aux_boundary_conditions=sm.aux_boundary_conditions,
        boundary_gradients=sm.boundary_gradients,
        initial_conditions=sm.initial_conditions,
        aux_initial_conditions=sm.aux_initial_conditions,
        update_variables=sm.update_variables,
        reconstruction_variables=sm.reconstruction_variables,
        state_from_reconstruction=sm.state_from_reconstruction,
    )
    sm_square.aux_registry = list(getattr(sm, "aux_registry", []) or [])
    sm_square.equation_names = (
        [sm.equation_names[src_of_dest[i]] if i in src_of_dest
         else f"_pad_{i}"
         for i in range(n_st)]
        if getattr(sm, "equation_names", None) is not None
        else None
    )
    sm_square.history = list(sm.history) + [{
        "name": "pad_to_square",
        "description": (
            f"padded {n_eq} → {n_st} rows; "
            f"zero ops at state indices "
            f"{[i for i in range(n_st) if i not in src_of_dest]}"
        ),
    }]
    return sm_square


class ChorinSplitVAMSolver(HyperbolicSolver):
    """Chorin projection split for the VAM chain DAE.

    Consumes three pre-built sub-system models from
    :func:`split_for_pressure`.  Inherits :class:`HyperbolicSolver` so
    the predictor substep uses the same Rusanov flux + indexed-BC +
    well-balanced reconstruction machinery as every other hyperbolic
    solver in the codebase — mass-conservative by construction.

    The pressure projection and corrector substeps are Chorin-specific
    and don't duplicate any existing infrastructure (the
    ``state_update`` field on SystemModel + matrix-free linear solve
    are the new primitives this class introduces).
    """

    pressure_tol = param.Number(
        default=1e-10, bounds=(0, None),
        doc="Linear-solver tolerance for the elliptic pressure block")
    pressure_maxit = param.Integer(
        default=500, bounds=(1, None),
        doc="Maximum iterations for the elliptic pressure solve")
    time_order = param.Integer(
        default=1, bounds=(1, 2),
        doc=("Order of the time integration.  1: single pred-press-corr "
             "cycle (forward-Euler-style Chorin step).  2: SSPRK2 / Heun "
             "wrap around the full cycle — two cycles of length dt then "
             "Heun-average ``Q^{n+1} = 0.5·(Q^n + Q^(2))``.  Note the "
             "cont_jk constraints are satisfied exactly at each cycle "
             "output, the Heun average carries a small O(dt²) drift "
             "that the next step's pressure solve corrects."))
    pressure_solver = param.Selector(
        default="gmres", objects=["gmres", "lu"],
        doc=("Linear-solve backend for the elliptic pressure block.  "
             "``'gmres'`` (default): scipy matrix-free GMRES on the "
             "operator ``A·P = R(P) − b``; cheap matvec, suitable for "
             "larger meshes once a preconditioner is wired in.  "
             "``'lu'``: assemble ``A`` as a dense ``(N×N)`` matrix by "
             "applying the matvec to canonical basis vectors, then "
             "solve via :func:`numpy.linalg.solve`.  Exact to machine "
             "precision in one call — appropriate for small 1D meshes "
             "(N up to a few thousand) and for diagnostic / reference "
             "runs.  Use ``'lu'`` when comparing against the legacy "
             "VAM Chorin reference, since the elliptic operator is "
             "non-symmetric and unpreconditioned GMRES can take "
             "thousands of iterations to reach `pressure_tol`."))
    riemann_solver = param.Selector(
        default="hr", objects=["hr", "ncp"],
        doc=("Predictor Riemann route.  ``'hr'`` (default): Audusse–"
             "Bristeau–Klein hydrostatic reconstruction wrapped around "
             "NonconservativeRusanov (``PositiveNonconservativeRusanov``) "
             "— well-balancing comes from the runtime HR step + the "
             "chain's post-HR NCP entry ``B[xmom, b, x] = g·h``.  "
             "``'ncp'``: plain ``NonconservativeRusanov`` with an inline "
             "subclass adding the LAR-balance identity ``Id[h, b] = 1`` "
             "to the fluctuation dissipation.  No HR step at runtime — "
             "the chain must NOT apply ``HydrostaticReconstruction`` "
             "either, so ``g·h·∂_x(b+h)`` stays symbolically in NCP and "
             "the path-integral fluctuation supplies the bed-slope "
             "force.  Both routes operate on the same SystemModel API; "
             "only the face-flux machinery changes."))
    limited_aux_fields = param.Dict(
        default={},
        doc=("Per-aux-derivative TVD limiter map ``{aux_name: scheme}``.  "
             "Applied at runtime to LSQ-computed gradients before they "
             "enter the source / elliptic block.  Schemes: ``'minmod'``, "
             "``'venkatakrishnan'``, ``'barth_jespersen'``.  Use to "
             "stabilise pressure-projection at shocks / dam-break "
             "interfaces where the un-limited LSQ gradient overshoots "
             "(e.g. ``{'h_x': 'minmod', 'q_U0_x': 'minmod'}`` for a "
             "VAM dam-break-over-bump test).  Sister mechanism to the "
             "symbolic ``limit(Derivative, scheme)`` wrapper from "
             "``zoomy_core.model.numerics`` — that path tags aux "
             "entries as ``limited_derivative`` directly in the "
             "operator-form pipeline."))

    def __init__(self, sm_pred, sm_press, sm_corr, *,
                 reconstruction=None, **kwargs):
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
        self.state = list(sm_pred.state)
        self.n_state = sm_pred.n_state

        # Predictor reconstruction spec — seeded into the auto-built
        # NSM at ``setup_simulation`` time.  Defaults to first-order
        # constant; pass ``reconstruction=ReconstructionSpec(order=2,
        # limiter=...)`` for the primitive-WB MUSCL path.
        from zoomy_core.numerics import ReconstructionSpec
        self._reconstruction_spec = reconstruction or ReconstructionSpec()

        self._dt_symbol = self._detect_dt_symbol()

        for name, sm in (("press", sm_press), ("corr", sm_corr)):
            if [str(s) for s in sm.state] != [str(s) for s in sm_pred.state]:
                raise ValueError(
                    f"SM_{name}.state disagrees with SM_pred.state — the "
                    "three sub-systems must share a common state vector."
                )

    # ------------------------------------------------------------------
    # Numerics — Audusse well-balanced Rusanov for free-surface flow.
    # ------------------------------------------------------------------

    def _build_numerics(self, symbolic_model):
        """Predictor Riemann numerics.  Two routes, selected by
        :attr:`riemann_solver`:

        * ``"hr"`` (default) — Audusse–Bristeau–Klein hydrostatic
          reconstruction Rusanov (:class:`PositiveNonconservativeRusanov`).
          The plain ``NonconservativeRusanov`` fails lake-at-rest on a
          varying bathymetry: Rusanov's wavespeed-based dissipation
          breaks the discrete cancellation of ``g·h·∂_x η`` against
          the bottom-slope source.  Audusse HR restores well-balancing
          and handles wet/dry fronts via the ``h*`` clamp.
          ``scaled_q_indices`` are the momentum-density state indices
          (auto-detected — excludes ``h``, ``b`` and the pressure-mode
          state indices); these get rescaled by ``h*/h`` during the
          reconstruction so the velocity stays constant when ``h`` is
          clipped.  Pressure modes ``P_k`` are NOT rescaled — they are
          pressure amplitudes, not momentum densities.
        * ``"ncp"`` — plain :class:`NonconservativeRusanov` with an
          inline subclass that overrides
          :meth:`get_viscosity_identity_fluctuations` to add the
          LAR-balance entry ``Id[h, b] = 1``.  For lake-at-rest
          (``η = h + b = const`` ⇒ ``dh = -db``) the resulting h-row
          fluctuation dissipation ``s_max · (dh + db) = 0`` cancels
          exactly, the path-integral NCP fluctuation supplies the
          bed-slope force, and no Audusse face-state reconstruction is
          needed.  Requires ``h`` and ``b`` both in the conservative
          state — and the SystemModel must NOT have had
          ``HydrostaticReconstruction`` applied (the gravity term must
          still be symbolically in NCP, not split into ``g·h²/2`` flux
          + ``g·h·∂_x b`` NCP).
        """
        state_names = [str(s) for s in symbolic_model.state]

        if self.riemann_solver == "hr":
            excluded = set()
            if "h" in state_names:
                excluded.add(state_names.index("h"))
            if "b" in state_names:
                # Bathymetry as state (8-state b-promoted path): exclude
                # from h*/h rescaling — b is static topography, not a
                # momentum density that needs HR-mass-preservation scaling.
                excluded.add(state_names.index("b"))
            # Pressure-mode state indices come straight from the splitter.
            excluded.update(int(i) for i in self.sm_press.equation_to_state_index)
            scaled_q_indices = [
                i for i in range(symbolic_model.n_variables) if i not in excluded
            ]
            return PositiveNonconservativeRusanov(
                symbolic_model,
                scaled_q_indices=scaled_q_indices,
            )

        # "ncp" route: plain NCP-Rusanov with the LAR-balance identity.
        if "h" not in state_names or "b" not in state_names:
            raise ValueError(
                "ChorinSplitVAMSolver(riemann_solver='ncp') requires "
                "both ``h`` and ``b`` in the model state — found state "
                f"{state_names}.  The NCP route relies on "
                "``Id[h, b] = 1`` to balance Rusanov dissipation at "
                "lake-at-rest, and on a chain that has *not* applied "
                "HydrostaticReconstruction so that ``g·h·∂_x(b+h)`` "
                "stays in NCP."
            )
        h_idx = state_names.index("h")
        b_idx = state_names.index("b")

        class _NCPRusanovLARBalanced(NonconservativeRusanov):
            """``NonconservativeRusanov`` with LAR-balance entry
            ``Id[h, b] = 1`` added to the fluctuation dissipation.
            Parent already sets ``Id[b, b] = 0``; we add the off-
            diagonal coupling that cancels the h-row dissipation for
            ``dh = -db`` (lake-at-rest on varying bathymetry).
            """
            name = param.String(default="NCPRusanovLARBalancedV2")

            def get_viscosity_identity_fluctuations(self):
                Id = super().get_viscosity_identity_fluctuations()
                Id[h_idx, b_idx] = 1
                return Id

        return _NCPRusanovLARBalanced(symbolic_model)

    def _build_reconstruction(self, mesh, symbolic_model):
        """Primitive-variable MUSCL reconstruction on the predictor.

        At ``reconstruction_order >= 2``: wrap :class:`FreeSurfaceLSQMUSCL`
        with :class:`PrimitiveReconstruction` so the slope limiter runs
        on the model's well-balanced primitives
        (``sm.reconstruction_variables``: ``η = h+b``, ``u_k = q_Uk/h``,
        …) — physically bounded quantities — instead of the conservative
        state.  This kills the momentum overshoot at the wet/dry front
        that breaks every smooth limiter (Venkatakrishnan,
        Van Albada) on dam-break-style shocks.  See
        ``thesis/chapters/30_numerics.md`` "Primitive-variable MUSCL
        reconstruction".
        """
        if self.nsm.reconstruction.order >= 2:
            from zoomy_core.fvm.reconstruction import (
                FreeSurfaceLSQMUSCL, PrimitiveReconstruction,
            )
            from zoomy_core.fvm.solver_numpy import _var_index
            dim = symbolic_model.dimension
            h_idx = _var_index(symbolic_model, "h")
            eps_wet = self._get_dry_threshold(symbolic_model)
            # ``b`` (when state) is static — exempt from slope limiting
            # so the LSQ slope from cell-centre b passes through cleanly
            # (Audusse HR still acts on the face values).
            state_names = [str(s) for s in symbolic_model.state]
            unlimited = []
            if "b" in state_names:
                unlimited.append(state_names.index("b"))
            base = FreeSurfaceLSQMUSCL(
                mesh, dim, h_index=h_idx,
                eps_wet=eps_wet, limiter=self.nsm.reconstruction.limiter,
                unlimited_indices=unlimited or None,
            )
            # Lambdify the Model-declared WB forward / inverse maps.
            # ``state_from_reconstruction`` is in ``WB_<state_name>``
            # symbols — build that signature too.
            fwd = symbolic_model.reconstruction_variables
            inv = symbolic_model.state_from_reconstruction
            if fwd is None or inv is None:
                # Fall back to bare base limiter — no primitive transform.
                return base
            state_syms = list(symbolic_model.state)
            wb_syms = [sp.Symbol(f"WB_{s.name}", real=True)
                       for s in state_syms]
            forward_fn = sp.lambdify(
                state_syms, list(fwd), modules=["numpy"])
            inverse_fn = sp.lambdify(
                wb_syms, list(inv), modules=["numpy"])
            return PrimitiveReconstruction(base, forward_fn, inverse_fn)
        return super()._build_reconstruction(mesh, symbolic_model)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_simulation(self, mesh, write_output=False):
        """Set up the Chorin solver:

        1. Pad ``SM_pred`` to square so :class:`HyperbolicSolver` can
           drive it (the parent assumes ``n_equations == n_state``).
        2. Hand the padded predictor to the parent's
           ``setup_simulation`` — this builds the proper Rusanov flux
           operator, indexed-BC kernel, source operator, max-eigenvalue
           callable on the predictor SystemModel.
        3. Rename ``\\Delta t`` → ``dt`` in ``SM_press`` / ``SM_corr``
           (Python-safe Symbol), register as a parameter, build their
           runtimes via ``NumpyRuntimeModel.from_system_model``.
        """
        t0 = _time.time()

        # 1. Pad SM_pred to square and let the parent set up the
        #    predictor's flux machinery.  Wrap in NSM so the parent
        #    uses the reconstruction spec we were configured with.
        sm_pred_square = _pad_to_square(self.sm_pred)
        self._sm_pred_square = sm_pred_square
        nsm_pred = NumericalSystemModel.from_system_model(
            sm_pred_square, reconstruction=self._reconstruction_spec)
        super().setup_simulation(mesh, nsm_pred, write_output=write_output)

        # 2. dt rename + parameter registration on the dt-dependent subsystems.
        if self._dt_symbol is not None:
            dt_safe = sp.Symbol("dt", positive=True)
            self._dt_symbol_safe = dt_safe
            self.sm_press = _substitute_dt(self.sm_press, self._dt_symbol,
                                           dt_safe)
            self.sm_corr = _substitute_dt(self.sm_corr, self._dt_symbol,
                                          dt_safe)
            for sm in (self.sm_press, self.sm_corr):
                if not sm.parameters.contains("dt"):
                    new_params = Zstruct(**sm.parameters.as_dict())
                    new_params["dt"] = dt_safe
                    sm.parameters = new_params
                    new_pvals = Zstruct(**sm.parameter_values.as_dict())
                    new_pvals["dt"] = 0.0
                    sm.parameter_values = new_pvals
        else:
            self._dt_symbol_safe = None

        # 3. Build pressure + corrector runtimes (predictor runtime
        #    lives on ``self._sim_model`` from parent setup).
        self.rt_press = NumpyRuntimeModel.from_system_model(self.sm_press)
        self.rt_corr  = NumpyRuntimeModel.from_system_model(self.sm_corr)

        self._params_press_base = np.array(
            list(self.sm_press.parameter_values.values()), dtype=float,
        )
        self._params_corr_base = np.array(
            list(self.sm_corr.parameter_values.values()), dtype=float,
        )

        # State-index slices.
        self._pred_state_idx = np.asarray(
            self.sm_pred.equation_to_state_index, dtype=int)
        self._press_state_idx = np.asarray(
            self.sm_press.equation_to_state_index, dtype=int)
        self._corr_state_idx = np.asarray(
            self.sm_corr.equation_to_state_index, dtype=int)

        # 4. Per-subsystem aux pools.  Each sub-system has its own
        #    ``aux_state`` ordering (auto-scan picks up different
        #    atoms in different orders), so a single shared ``Qaux``
        #    breaks the row indexing — we maintain three arrays in
        #    lock-step via ``update_aux_variables``.  The predictor's
        #    Qaux is the parent's ``_sim_Qaux``.
        nc = self.nc
        self.Qaux_press = np.zeros(
            (len(self.sm_press.aux_state), nc), dtype=float)
        self.Qaux_corr = np.zeros(
            (len(self.sm_corr.aux_state), nc), dtype=float)

        # 5. Matrix-free pressure solver: identify which SM_press
        #    derivative-aux entries depend on the pressure modes
        #    being solved for.  These get recomputed via LSQ stencils
        #    on the current Krylov iterate during every matvec
        #    application — that's what makes the operator linear in P
        #    (the lambdified symbolic residual reads aux for ``P_x``,
        #    ``P_xx`` rather than embedding the LSQ stencil in the
        #    symbolic flow).
        self._press_state_idx_set = set(int(i) for i in self._press_state_idx)
        self._press_aux_recompute = []
        for entry in (self.sm_press.aux_registry or []):
            if (entry["kind"] not in ("derivative", "limited_derivative")
                    or entry.get("target_kind") != "state"
                    or entry.get("state_index") not in self._press_state_idx_set):
                continue
            scheme = (entry.get("limiter_scheme")
                      if entry["kind"] == "limited_derivative"
                      else self.limited_aux_fields.get(entry["name"]))
            self._press_aux_recompute.append({
                "row":         entry["row"],
                "state_index": entry["state_index"],
                "multi_index": entry["multi_index"],
                "limiter_scheme": scheme,
            })

        logger.info(
            "ChorinSplitVAMSolver setup: pred → %s, press → %s, corr → %s "
            "in %.2fs",
            self._pred_state_idx.tolist(),
            self._press_state_idx.tolist(),
            self._corr_state_idx.tolist(),
            _time.time() - t0,
        )
        return self._sim_Q

    # ------------------------------------------------------------------
    # Conveniences
    # ------------------------------------------------------------------

    @property
    def nc(self):
        return self._sim_mesh.n_inner_cells

    @property
    def Q(self):
        return self._sim_Q

    @property
    def Qaux(self):
        return self._sim_Qaux

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def _aux_pool_for(self, sm):
        """Return the (Qaux array, registry) for a given sub-system."""
        if sm is self.sm_pred:
            return self._sim_Qaux, sm.aux_registry or []
        if sm is self.sm_press:
            return self.Qaux_press, sm.aux_registry or []
        if sm is self.sm_corr:
            return self.Qaux_corr, sm.aux_registry or []
        raise ValueError(f"Unknown sub-system {sm}")

    def _compute_boundary_face_state(self):
        """Apply the SystemModel's indexed BC kernel to every boundary
        face; return ``(n_state, n_boundary_faces)`` face-state array.

        Used by :meth:`update_aux_variables` to give the LSQ derivative
        stencil at boundary cells the BC-correct face values — so a
        prescribed-Lambda BC (e.g. ``q_U0_face = 0.11197``) flows into
        the cell-centered ``∂_x q_U0`` estimate via the boundary-face
        virtual neighbor.  Without this, the LSQ at boundary cells
        falls back to extrapolation (Neumann-zero) which silently
        violates prescribed-value inflow BCs.
        """
        if self._n_bf == 0:
            return np.zeros((self._sim_Q.shape[0], 0))
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        p = self._sim_parameters
        mesh = self._sim_mesh
        face_centers = mesh.face_centers
        normals_arr = mesh.face_normals[:mesh.dimension, :]
        n_state = Q.shape[0]
        Q_face = np.zeros((n_state, self._n_bf))
        for i_bf in range(self._n_bf):
            fidx = self._bf_fidx[i_bf]
            inner = self._bf_cells[i_bf]
            qaux_inner = (Qaux[:, inner]
                           if Qaux.shape[0] > 0 else _EMPTY_AUX)
            Q_face[:, i_bf] = self._bc_fn(
                self._bc_indices[i_bf], self._sim_time,
                face_centers[fidx, :], self._d_face[i_bf],
                Q[:, inner], qaux_inner, p, normals_arr[:, fidx],
            )
        return Q_face

    def update_aux_variables(self):
        """Refresh derivative-aux rows by LSQ in each sub-system's
        ``Qaux`` pool.  Each sub-system has its own ``aux_state``
        ordering (auto-scan ordering depends on which atoms appear
        in that sub-system's operators) — we walk all three
        registries and fill each one's Qaux array independently.

        Function-aux rows (``b`` topography) are left untouched —
        the user sets them once via :meth:`set_function_aux` and they
        stay static through the simulation.
        """
        Q = self._sim_Q
        nc = Q.shape[1]
        n_cells = self._sim_mesh.n_cells
        mesh = self._sim_mesh
        u_full = np.zeros(n_cells)

        # BC-correct face state for state-derivative aux entries.
        # Function-aux fields (e.g. bathymetry ``b``) fall back to
        # extrapolation since their BC isn't carried by the SystemModel
        # BC kernel — but ``b`` is static so its derivatives also are.
        Q_face = self._compute_boundary_face_state()

        for sm in (self.sm_pred, self.sm_press, self.sm_corr):
            Qaux, registry = self._aux_pool_for(sm)
            for entry in registry:
                if entry["kind"] not in ("derivative", "limited_derivative"):
                    continue
                row = entry["row"]
                if row >= Qaux.shape[0]:
                    continue
                mi = entry["multi_index"]
                tk = entry["target_kind"]
                if tk == "state":
                    state_i = entry["state_index"]
                    u_full[:nc] = Q[state_i, :]
                    # FORCE extrapolation BC everywhere to match the
                    # old chain behaviour (silent extrapolation default
                    # before this session) AND the JAX-legacy convention
                    # for h/b/P — see ``apply_bc`` in
                    # ``/tmp/vam_legacy_jax.py``.  Apples-to-apples
                    # comparison with JAX requires identical BC
                    # treatment for the LSQ-recomputed aux.  Prescribed-
                    # Dirichlet inflow values are handled separately
                    # (predictor flux only); the elliptic block sees
                    # ∂_n = 0 at the boundary face.
                    u_face = "extrapolation"
                elif tk == "function":
                    u_full[:nc] = Qaux[entry["function_row"], :]
                    # Function-aux (e.g. static bathymetry ``b``) — the
                    # SystemModel doesn't carry a per-field BC kernel
                    # here, so default to extrapolation (Neumann-zero,
                    # appropriate for static topography).
                    u_face = "extrapolation"
                else:
                    continue
                d = mesh.compute_derivatives(
                    u_full, degree=max(mi),
                    derivatives_multi_index=[mi],
                    u_boundary_face=u_face,
                )
                grad_lsq = d[:nc, 0]
                # Apply TVD limiter to the LSQ gradient if the aux entry
                # is flagged as ``limited_derivative`` (via the symbolic
                # ``limit(D, scheme)`` wrapper) or if its name matches an
                # entry in ``self.limited_aux_fields`` (for ad-hoc
                # solver-level limiting without modifying the model).
                scheme = (entry.get("limiter_scheme")
                          if entry["kind"] == "limited_derivative"
                          else self.limited_aux_fields.get(entry["name"]))
                if scheme is not None:
                    grad_lsq = _apply_cell_limiter(
                        u_full[:nc], grad_lsq, mi, scheme, mesh)
                Qaux[row, :] = grad_lsq

    def set_function_aux(self, name, values):
        """Set a function-aux row (e.g. static topography ``b``) on
        every sub-system's Qaux pool at its own row index.  Use this
        once at IC time to inject bathymetry; ``update_aux_variables``
        then refreshes the derived rows (``b_x``, ``b_xx``, …) from it.
        """
        for sm in (self.sm_pred, self.sm_press, self.sm_corr):
            Qaux, registry = self._aux_pool_for(sm)
            for entry in registry:
                if entry["kind"] == "function" and entry["name"] == name:
                    Qaux[entry["row"], :] = values
                    break

    def step(self, dt):
        """One Chorin step.

        ``time_order == 1`` (default): single ``predictor → pressure →
        corrector`` cycle (forward-Euler-style Chorin step).
            - Predictor: inherited :class:`HyperbolicSolver.step` —
              proper Rusanov + NCP + indexed-BC; mass-conservative.
            - Pressure: matrix-free GMRES on the lambdified elliptic
              residual (linear in P ⇒ 1–2 iters from warm start).
            - Corrector: one lambdify call to ``state_update`` +
              atomic in-place assignment.

        ``time_order == 2``: SSPRK2 / Heun wrap on the full cycle.
        Two cycles of length ``dt`` are applied sequentially, then
        Heun-averaged: ``Q^{n+1} = 0.5·(Q^n + Q^(2))``.  The
        cont_jk constraints are satisfied exactly at each cycle
        output; the Heun average carries a small ``O(dt²)`` drift
        the next step's pressure solve corrects.
        """
        if self.time_order == 2:
            Q_n    = self._sim_Q.copy()
            Qaux_n = self._sim_Qaux.copy()
            t_n    = self._sim_time
            # Cycle 1.
            self._chorin_cycle(dt)
            # Cycle 2 (starting from Q^(1) the cycle just produced).
            self._chorin_cycle(dt)
            Q2    = self._sim_Q
            Qaux2 = self._sim_Qaux
            # Heun average.  Refresh derivative aux from the averaged
            # state so the next step sees a consistent ∂_x Q.
            self._sim_Q    = 0.5 * (Q_n + Q2)
            self._sim_Qaux = 0.5 * (Qaux_n + Qaux2)
            self._sim_time = t_n + dt
            self.update_aux_variables()
        else:
            self._chorin_cycle(dt)

    def _chorin_cycle(self, dt):
        """One ``predictor → pressure → corrector`` cycle — the
        innermost building block.  Advances ``_sim_time`` by ``dt``."""
        super().step(dt)              # predictor: parent's RK + flux + source
        self.update_aux_variables()
        self._step_pressure(dt)
        self.update_aux_variables()
        self._step_corrector(dt)
        self.update_aux_variables()
        # Parent's HyperbolicSolver.step does not advance _sim_time
        # (run_simulation does); this is the user-facing entry, so we
        # track time here.
        self._sim_time += dt

    def _params_with_dt(self, base, dt):
        """Splice numeric dt into the last slot of a parameter array
        (where ``dt`` lives by our setup_simulation convention)."""
        out = base.copy()
        out[-1] = float(dt)
        return out

    def _step_pressure(self, dt):
        """Solve the linear elliptic block for ``Q[press_e2s]``.

        Matrix-free GMRES.  The operator ``A·P`` is evaluated per
        Krylov iteration by:

        1. Stuffing the iterate ``p_vec`` into ``Q_try[press_e2s]``.
        2. **Recomputing the P-dependent aux entries** (``P_0_x``,
           ``P_1_x``, ``P_0_xx``, ``P_1_xx``) via LSQ stencils on
           ``p_vec`` — the bug fix vs. fsolve-with-frozen-aux.  The
           non-P-dependent aux entries (``h``, ``b``, ``h_x``, etc.)
           stay frozen at the start-of-step values.
        3. Evaluating ``rt.source(Q_try, Qaux_try, p_full)`` — the
           lambdified elliptic residual.

        ``b = R(P=0)`` once per step gives the data forcing; then
        ``A·P = R(P) − b`` is the linear operator.  GMRES warm-starts
        from the previous-step pressure.  No matrix assembled — same
        pattern jax-jits cleanly via ``jax.scipy.sparse.linalg.gmres``.
        """
        from scipy.sparse.linalg import LinearOperator, gmres
        rt    = self.rt_press
        Q     = self._sim_Q
        Qaux0 = self.Qaux_press
        nc    = self.nc
        mesh  = self._sim_mesh
        n_cells = mesh.n_cells
        p_full = self._params_with_dt(self._params_press_base, dt)
        e2s   = self._press_state_idx
        nP    = len(e2s)
        N     = nP * nc

        # Map state index → local pressure index (e.g. 5 → 0, 6 → 1).
        local_of_state = {int(s): k for k, s in enumerate(e2s)}

        def _refresh_pressure_aux(p_mat, Qaux_work):
            """LSQ-recompute the P-dependent derivative-aux rows from
            the current iterate ``p_mat`` (shape ``(nP, nc)``).  In-
            place on ``Qaux_work``.

            Pressure modes use ``'extrapolation'`` at boundary faces —
            the standard Chorin / projection-method BC (``∂_n P = 0``).
            The chain's SystemModel doesn't carry a separate BC kernel
            for P, so we apply the convention explicitly here.
            """
            u_full = np.zeros(n_cells)
            for entry in self._press_aux_recompute:
                state_i = entry["state_index"]
                k = local_of_state[state_i]
                u_full[:nc] = p_mat[k, :]
                mi = entry["multi_index"]
                d = mesh.compute_derivatives(
                    u_full, degree=max(mi),
                    derivatives_multi_index=[mi],
                    u_boundary_face="extrapolation",
                )
                grad_lsq = d[:nc, 0]
                scheme = entry.get("limiter_scheme")
                if scheme is not None:
                    grad_lsq = _apply_cell_limiter(
                        u_full[:nc], grad_lsq, mi, scheme, mesh)
                Qaux_work[entry["row"], :] = grad_lsq

        def _residual(p_vec):
            p_mat = p_vec.reshape(nP, nc)
            Q_try = Q.copy()
            Q_try[e2s, :] = p_mat
            Qaux_work = Qaux0.copy()
            _refresh_pressure_aux(p_mat, Qaux_work)
            R = np.asarray(rt.source(Q_try, Qaux_work, p_full), dtype=float)
            if R.ndim == 1:
                R = R[:, None] * np.ones((1, nc))
            return R.ravel()

        # Data forcing: ``b = R(P = 0)``.  Pure data — no P enters.
        b = _residual(np.zeros(N))

        # Early exit on near-zero forcing (e.g. exact lake-at-rest):
        # GMRES with only ``rtol`` set chases ``rtol·‖b‖`` which is
        # unreachable when ‖b‖ ~ machine epsilon, so it hits maxiter
        # and returns garbage.  Skip the solve and return P = warm-start
        # (which is the correct answer for ‖b‖ ≈ 0).
        b_norm = float(np.linalg.norm(b))
        if b_norm < self.pressure_tol:
            self._sim_Q = Q              # Q[e2s] unchanged (still p0)
            return

        # Linear operator: ``A·P = R(P) − b``.  Pure matvec callable.
        def apply_A(p_vec):
            return _residual(p_vec) - b

        if self.pressure_solver == "lu":
            # Dense assembly by canonical-basis application + direct solve.
            A_dense = np.empty((N, N), dtype=float)
            e_j = np.zeros(N)
            for j in range(N):
                e_j[j] = 1.0
                A_dense[:, j] = apply_A(e_j)
                e_j[j] = 0.0
            p_new = np.linalg.solve(A_dense, -b)
        else:
            A = LinearOperator((N, N), matvec=apply_A, dtype=float)
            p0 = Q[e2s, :].ravel()                      # warm start
            p_new, info = gmres(
                A, -b, x0=p0,
                rtol=self.pressure_tol,
                atol=self.pressure_tol,    # absolute floor — survives tiny ‖b‖
                maxiter=self.pressure_maxit,
            )
            if info != 0:
                logger.warning(
                    f"Chorin pressure GMRES did not fully converge "
                    f"(info={info}, ‖b‖={b_norm:.3e})"
                )
        Q[e2s, :] = p_new.reshape(nP, nc)
        self._sim_Q = Q
        # Reflect the solved pressure into Qaux_press derivative rows
        # (so the corrector reads consistent ∂_x P, ∂_xx P).
        _refresh_pressure_aux(p_new.reshape(nP, nc), self.Qaux_press)

    def _step_corrector(self, dt):
        """``Q[corr_e2s] ← state_update(Q, Qaux_corr, p, dt)`` in place."""
        rt = self.rt_corr
        Q = self._sim_Q
        Qaux = self.Qaux_corr           # SM_corr's own aux pool
        p_full = self._params_with_dt(self._params_corr_base, dt)
        e2s = self._corr_state_idx

        new_vals = np.asarray(rt.state_update(Q, Qaux, p_full), dtype=float)
        Q[e2s, :] = new_vals
        self._sim_Q = Q

    def _detect_dt_symbol(self) -> Optional[sp.Symbol]:
        """Locate the symbolic time-step common to SM_press / SM_corr."""
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
    """Rename a Symbol throughout a sub-system's operator tensors.

    Used to convert the splitter's ``\\Delta t`` placeholder into the
    Python-safe ``dt`` Symbol before lambdify (the backslash is a line
    continuation in Python source).  Element-wise xreplace via ZArray.
    """
    sub = {old_sym: new_sym}
    def _xrepl(tensor):
        if tensor is None:
            return None
        return _to_zarray(tensor).xreplace(sub)
    sm.flux                   = _xrepl(sm.flux)
    sm.hydrostatic_pressure   = _xrepl(sm.hydrostatic_pressure)
    sm.source                 = _xrepl(sm.source)
    sm.mass_matrix            = _xrepl(sm.mass_matrix)
    sm.nonconservative_matrix = _xrepl(sm.nonconservative_matrix)
    sm.refresh_derived_operators(eigenvalues=False)
    sm.eigenvalues            = _xrepl(sm.eigenvalues)
    sm.state_update           = _xrepl(sm.state_update)
    return sm


__all__ = [
    "ChorinSplitVAMSolver",
]
