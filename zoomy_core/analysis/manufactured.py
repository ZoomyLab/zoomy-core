"""Method of Manufactured Solutions (MMS) for any zoomy SystemModel.

A verification tool with a genuine ANALYTIC reference for models that have no
closed-form solution (SME, VAM, ML-*): pick a smooth field ``W*(x)``, force the
PDE to admit it exactly by adding a manufactured source, run, and measure the
error against ``W*`` under mesh refinement.  The observed order is then a
statement about the SCHEME, not about a self-reference sharing the scheme's own
truncation error.

THE ONE FORMULA.  The continuous 1-D balance law a zoomy SystemModel represents
is

    ∂_t W + A(W) ∂_x W = S(W),        A = quasilinear_matrix = ∂F/∂Q + ∂P/∂Q + B

(the flux Jacobian + the hydrostatic-pressure Jacobian + the nonconservative
matrix — exactly ``sm.quasilinear_matrix``).  For a chosen steady ``W*(x)`` to
be an EXACT steady solution of the augmented system

    ∂_t W + A(W) ∂_x W = S(W) + S_mms,

we need, at ``W = W*``,

    S_mms(x) = A(W*(x)) · dW*/dx − S(W*(x)).                        (MMS)

Every term on the right is symbolic in zoomy: ``A`` and ``S`` are model
operators, ``W*`` and ``dW*/dx`` are the manufactured expressions.  ``S_mms`` is
a pure function of the space coordinate, which the solver already binds to cell
centres (REQ-185: the source is lowered as ``(Q, Qaux, p, t, x)``).

STATIONARY-STATE ROWS (bed ``b`` with ∂_t b = 0) need no source: their row of
``A`` is zero, so (MMS) yields ``S_mms = -S`` there, which for a static bed is
also zero.  ``W*`` for such a row is simply the prescribed geometry (e.g. a
bump), and it is preserved exactly.

Aux resolution.  ``S_mms`` must be a pure function of x, so every aux the
operators reference has to be reduced to the manufactured field:

* algebraic aux (``hinv``, …) → their ``update_aux_variables`` definition;
* gradient aux (``dhdx`` = ∂h/∂x, registered ``kind='derivative'`` in
  ``sm.aux_registry``) → the exact spatial derivative of the target state;
* closure aux that no definition owns (the deviatoric stress moments
  ``\\hat{σ}_k`` the stress closure leaves open) → declared by the caller via
  ``closure_aux=`` on :func:`manufactured_source_field`, and matched by the
  aux initial condition of the run (e.g. ``σ = 0`` for an inviscid run).

Usage
-----
    from zoomy_core.analysis.manufactured import (
        manufactured_source_field, exact_cell_field)
    x = sm.space[0]
    W_star = {"h": H0 + A*sp.cos(k*x), "q_0": ..., "b": bump(x), ...}
    S_mms  = manufactured_source_field(sm, W_star, cell_centers,
                                       parameter_values=pv, closure_aux={...})
    W_cell = exact_cell_field(sm, W_star, cell_centers)
    # set IC = W*, aux IC = 0, add S_mms as a fixed inner-cell source, run,
    # then measure |Q_end - W_cell| under mesh refinement -> scheme order.

Verified (2026-07-24), SME(0) inviscid, numpy backend, genuine analytic
reference (NOT self-convergence):

* spatial-residual (truncation) order, interior, monotone field so the minmod
  limiter is inactive:  EOC = 2.000, 2.000, 2.000 — the discretization is
  cleanly 2nd order.
* time-evolved real run (IC = W* to 1e-16, then integrated), monotone ramp that
  is flat at both ends (Extrapolation exact, no boundary pollution):
  order-2 EOC = 1.99 → 2.00, order-1 EOC → 1.0.

Two confounds that MASK the order if you are not careful — both surfaced here:

* the minmod limiter clips smooth EXTREMA (first-order there), so a periodic
  field with interior extrema caps at ~1.9, never a clean 2.  Use a monotone
  field (no interior extrema) to see the true order.
* a subcritical STEADY problem with a value-fixed (Dirichlet) boundary reads
  ~1st order: the boundary treatment is low-order and, because information
  travels both ways, that error propagates across the whole domain.  Use a field
  that is flat at the boundaries (so the BC is exact) for a clean rate.

Do NOT trust a lone EOC above 2 (e.g. 2.5): that is pre-asymptotic or the error
sitting on the projection floor, not evidence of higher order.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.misc.misc import ZArray

__all__ = [
    "manufactured_source",
    "install_manufactured_source",
    "manufactured_source_field",
    "exact_cell_field",
    "mms_convergence",
]


def _state_symbol(sm, key):
    """Resolve a state entry given either its Symbol or its str name."""
    if isinstance(key, sp.Symbol):
        return key
    for s in sm.state:
        if str(s) == str(key):
            return s
    raise KeyError(
        f"manufactured: '{key}' is not a state variable; state is "
        f"{[str(s) for s in sm.state]}")


def _resolve_exact(sm, exact):
    """``{name|Symbol: expr}`` → an ordered list aligned with ``sm.state``.

    Every state variable must be given a manufactured expression (there is no
    sensible default — a missing row would silently manufacture the wrong
    balance)."""
    provided = {str(_state_symbol(sm, k)): sp.sympify(v)
                for k, v in exact.items()}
    missing = [str(s) for s in sm.state if str(s) not in provided]
    if missing:
        raise KeyError(
            f"manufactured: no expression for state {missing}. Every state "
            f"variable needs one (state is {[str(s) for s in sm.state]}).")
    return [provided[str(s)] for s in sm.state]


def _aux_substitution(sm, state_subs, coords):
    """``{aux_symbol: expr(x)}`` — each aux replaced by its
    ``update_aux_variables`` definition, itself reduced to the manufactured
    field.  Fixed-point over a few passes so an aux defined in terms of another
    aux resolves; gradient-aux (``b_x`` = ∂_x b) become the actual derivative of
    the manufactured field.  This is what makes ``S_mms`` a PURE function of x
    rather than carrying opaque ``hinv`` / gradient symbols."""
    aux = list(getattr(sm, "aux_state", []) or [])
    defs = {}
    # (1) algebraic aux from update_aux_variables (hinv, …), if present.
    uav = getattr(sm, "update_aux_variables", None)
    if uav is not None:
        for k, a in enumerate(aux):
            try:
                e = sp.sympify(uav[k, 0] if getattr(uav, "shape", None)
                               and len(uav.shape) > 1 else uav[k])
            except (IndexError, TypeError):
                continue
            defs[a] = e
    # (2) gradient aux from aux_registry (kind='derivative'): the aux IS the
    # spatial derivative of a state, so in the manufactured field it becomes the
    # exact derivative of that state's expression.  multi_index gives the
    # derivative order per coordinate (e.g. (1,) → ∂/∂x).
    state_by_name = {str(s): s for s in sm.state}
    for entry in (getattr(sm, "aux_registry", None) or []):
        if entry.get("kind") != "derivative":
            continue
        a = entry.get("aux_symbol")
        tgt = entry.get("target_name")
        mi = entry.get("multi_index", (1,))
        if a is None or tgt is None:
            continue
        tsym = state_by_name.get(tgt)
        if tsym is None or tsym not in state_subs:
            continue
        expr = state_subs[tsym]
        for d, order in enumerate(mi):
            for _ in range(int(order)):
                expr = sp.diff(expr, coords[d])
        defs[a] = expr
    if not defs:
        return {}
    sub = dict(defs)
    for _ in range(len(aux) + 1):                     # resolve aux-of-aux
        sub = {a: sp.sympify(e).xreplace(sub) for a, e in sub.items()}
    # now fold in the manufactured state field and evaluate any derivatives
    out = {}
    for a, e in sub.items():
        e = sp.sympify(e).xreplace(state_subs).doit()
        out[a] = e
    return out


def manufactured_source(sm, exact):
    """Return the manufactured source ``S_mms(x)`` (MMS) as a ``ZArray`` of
    length ``n_eq``, in terms of the space coordinate(s).

    ``exact`` maps each state variable (Symbol or name) to a smooth sympy
    expression in ``sm.space``.  See the module docstring for the formula:

        S_mms = A(W*)·∂_x W*  −  ∂_x( D(W*)·∂_x W* )  −  S(W*)

    The middle term is the DIFFUSIVE flux divergence (``diffusion_matrix`` D,
    the contract's ``+∂_x(D ∂_x Q)`` on the RHS): SME/VAM carry an intrinsic
    ``diffusion_matrix`` even without an explicit viscosity, so omitting it
    would manufacture the wrong balance and the residual would not vanish."""
    W = _resolve_exact(sm, exact)                     # aligned with sm.state
    state_subs = {s: W[i] for i, s in enumerate(sm.state)}
    aux_subs = _aux_substitution(sm, state_subs, list(sm.space))
    subs = {**state_subs, **aux_subs}
    coords = list(sm.space)
    A = sm.quasilinear_matrix                         # (n_eq, n_state, n_dim)
    S = sm.source
    D = getattr(sm, "diffusion_matrix", None)         # (n_eq, n_state, n_dim, n_dim)
    n_eq, n_st, n_dim = sm.n_equations, sm.n_state, sm.n_dim
    dWjd = {(j, d): sp.diff(W[j], coords[d])
            for j in range(n_st) for d in range(n_dim)}

    def _scalar(arr, idx):
        return sp.sympify(arr[idx])

    out = []
    for i in range(n_eq):
        # advective / nonconservative:  A(W*)·∂_x W*
        adv = sp.Integer(0)
        for d in range(n_dim):
            for j in range(n_st):
                adv += _scalar(A, (i, j, d)).xreplace(subs) * dWjd[(j, d)]
        # diffusive:  ∂_{x_d} ( D[i,j,d,e] ∂_{x_e} W_j )
        diff = sp.Integer(0)
        if D is not None:
            for d in range(n_dim):
                flux_d = sp.Integer(0)
                for e in range(n_dim):
                    for j in range(n_st):
                        Dijde = _scalar(D, (i, j, d, e))
                        if Dijde == 0:
                            continue
                        flux_d += Dijde.xreplace(subs) * dWjd[(j, e)]
                diff += sp.diff(flux_d, coords[d])
        Si = (_scalar(S, (i, 0)) if getattr(S, "shape", None) and len(S.shape) > 1
              else _scalar(S, i)).xreplace(subs)
        out.append(sp.simplify(adv - diff - Si))
    return ZArray(out).reshape(n_eq, 1)


def install_manufactured_source(sm, exact):
    """Add ``S_mms(x)`` to ``sm.source`` IN PLACE and return ``sm``.

    After this, ``W*`` is an exact steady solution of the model, so a run
    started from ``W*`` (or converged to steady state) has an analytic
    reference and its mesh-refinement error measures the scheme order."""
    S_mms = manufactured_source(sm, exact)
    base = sm.source
    n_eq = sm.n_equations
    merged = []
    for i in range(n_eq):
        b = sp.sympify(base[i, 0] if getattr(base, "shape", None)
                       and len(base.shape) > 1 else base[i])
        merged.append(b + S_mms[i, 0])
    sm.source = ZArray(merged).reshape(n_eq, 1)
    sm.refresh_derived_operators()
    return sm


def manufactured_source_field(sm, exact, cell_centers, parameter_values=None,
                              closure_aux=None):
    """Evaluate ``S_mms(x)`` at cell centres → array ``(n_eq, n_cells)``.

    A numeric companion to :func:`manufactured_source` for solvers that cannot
    take an explicit position-dependent symbolic source (e.g. because the
    symbolic source is evaluated over ghost-padded cells while the manufactured
    term is physical-cells-only).  Inject this as a FIXED additive source on the
    inner cells each step; ``W*`` is then an exact steady state and the
    steady-state error under refinement is the scheme order."""
    import numpy as np
    S_mms = manufactured_source(sm, exact)
    coords = list(sm.space)
    pv = dict(parameter_values or {})
    # bind parameter symbols by NAME (they carry assumptions) to their values
    by_name = {str(s): s for s in sm.parameters.values()}
    param_subs = {by_name[k]: v for k, v in pv.items() if k in by_name}
    # Closure aux the operators reference but that are NOT expressible from the
    # manufactured field (e.g. the deviatoric stress moments the stress closure
    # owns).  The caller declares their manufactured values — for an inviscid
    # run the stress is zero, so ``{"\\hat{sigma}_0": 0, ...}``.  Matched by
    # aux name.
    aux_by_name = {str(s): s for s in (getattr(sm, "aux_state", []) or [])}
    for k, v in (closure_aux or {}).items():
        if k in aux_by_name:
            param_subs[aux_by_name[k]] = v
    cc = np.asarray(cell_centers, float)
    nc = cc.shape[1]
    args = [cc[d, :] for d in range(len(coords))]
    out = np.zeros((sm.n_equations, nc))
    allowed = set(coords) | set(param_subs)
    for i in range(sm.n_equations):
        expr = sp.sympify(S_mms[i, 0]).xreplace(param_subs)
        leftover = {s for s in expr.free_symbols if s not in allowed}
        if leftover:
            raise NotImplementedError(
                f"manufactured_source_field: row {i} still carries unresolved "
                f"aux symbols {sorted(str(s) for s in leftover)}. Algebraic aux "
                "(update_aux_variables) and gradient aux (aux_registry "
                "kind='derivative', e.g. dhdx) are resolved automatically; "
                "what remains here are CLOSURE aux with no definition — e.g. "
                "the deviatoric stress moments \\hat{sigma}_k the stress "
                "closure leaves open. Declare their manufactured values via "
                "closure_aux={...} (and match them with the run's aux initial "
                "condition, e.g. sigma=0 for an inviscid run).")
        f = sp.lambdify(coords, expr, "numpy")
        val = f(*args)
        out[i, :] = np.broadcast_to(np.asarray(val, float), (nc,))
    return out


def mms_convergence(setup_run, grids, interior_frac=0.0):
    """Fixed-time / steady MMS convergence rate — the backend-agnostic driver.

    ``setup_run(nx)`` builds the model with ``S_mms`` injected, RUNS it (from
    ``IC = W*``), and returns ``(Q_end, W_cell)`` as ``(n_rows, nx)`` arrays:
    the evolved solution and the analytic field it is compared against.  The
    caller owns the solver, so this works for any backend.

    Returns ``dict(grids, l1, eoc)`` where ``eoc[i] = log2(l1[i-1]/l1[i])``.
    ``interior_frac`` (e.g. 0.25) drops that fraction of cells at each end before
    averaging — use it to separate an interior rate from boundary effects.

    NOTE this measures an EVOLVED error: ``setup_run`` must actually integrate in
    time, not just project ``W*``.  A zero (~1e-16) error means nothing ran.
    Read the module docstring for the extremum/boundary confounds first."""
    import numpy as np
    l1 = []
    for nx in grids:
        Q, W = (np.asarray(a, float) for a in setup_run(nx))
        d = np.abs(Q[:W.shape[0], :nx] - W[:, :nx])
        m = int(interior_frac * nx)
        l1.append(float(np.mean(d[:, m:nx - m] if m else d)))
    eoc = [float(np.log2(l1[i - 1] / l1[i])) for i in range(1, len(l1))]
    return {"grids": list(grids), "l1": l1, "eoc": eoc}


def exact_cell_field(sm, exact, cell_centers):
    """Evaluate ``W*`` at cell centres → array ``(n_state, n_cells)``.

    ``cell_centers`` is ``(n_dim, n_cells)`` (mesh convention).  Use as the
    analytic reference the numerical ``Q`` is compared against."""
    import numpy as np
    W = _resolve_exact(sm, exact)
    coords = list(sm.space)
    cc = np.asarray(cell_centers, float)
    fns = [sp.lambdify(coords, w, "numpy") for w in W]
    nc = cc.shape[1]
    out = np.zeros((len(W), nc))
    args = [cc[d, :] for d in range(len(coords))]
    for i, f in enumerate(fns):
        val = f(*args)
        out[i, :] = np.broadcast_to(np.asarray(val, float), (nc,))
    return out
