"""Moving-equilibrium (Bernoulli) well-balanced reconstruction for the moment
hierarchy (SME).  Used by HyperbolicSolver when a model sets
``equilibrium_reconstruction='bernoulli'``.

Per face, each cell state is reconstructed to the common interface bed ``b*``
preserving the discharge ``q = h·α_0`` and the per-streamline Bernoulli head
``H(s) = ½u² + g(b+h)`` (s = discharge fraction).  The numerical flux + the
moment-coupling NCP (NonconservativeRusanov) then act on the reconstructed
states (vanishing on an equilibrium where neighbours reconstruct to the SAME
state), and the solver adds the full conservative-flux-jump source
``Fc(Q) − Fc(Q*)``.  Reduces to lake-at-rest / SWE when the velocity is uniform
over depth (higher moments zero).
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import legval


def build_bernoulli_config(model, mode="bernoulli", n_sigma=200):
    """Resolve indices, basis and the conservative flux Fc=flux+pressure.

    ``mode`` selects the equilibrium reconstruction: 'audusse' (lake-at-rest:
    h*=max(0,h+b−b*), velocity preserved) or 'bernoulli' (moving: preserve the
    discharge q and the per-streamline Bernoulli head H(s))."""
    names = list(model.variables.keys())
    if "h" not in names or "b" not in names:
        raise ValueError("bernoulli WB needs 'h' and 'b' fields")
    h_idx = names.index("h")
    b_idx = names.index("b")
    q_idx = [i for i, n in enumerate(names) if n not in ("h", "b")]  # q_0..q_N
    level = len(q_idx) - 1
    sg = np.linspace(0.0, 1.0, n_sigma)        # σ EDGES spanning [0,1]
    PHI = np.array([legval(2 * sg - 1, np.eye(level + 1)[k]) for k in range(level + 1)])
    g = float(model.parameter_values["g"])
    # conservative flux Fc = flux + hydrostatic_pressure (1-D, direction 0)
    state = list(model.state)
    Fc = [sp.sympify(model.flux[r, 0]) + sp.sympify(model.hydrostatic_pressure[r, 0])
          for r in range(len(state))]
    pars = list(model.parameters)
    pv = [float(model.parameter_values[str(p)]) for p in pars]
    fc = sp.lambdify([sp.Symbol(str(s), real=True) for s in state]
                     + [sp.Symbol(str(p), real=True) for p in pars], Fc, "numpy")
    def fc_fn(Qf):                         # Qf (n_vars, nf) -> (n_vars, nf)
        rows = fc(*Qf, *pv)                # some rows are constant -> scalars
        nf = Qf.shape[1]
        return np.stack([np.broadcast_to(np.asarray(r, dtype=float), (nf,))
                         for r in rows])
    return {"h": h_idx, "b": b_idx, "q": q_idx, "PHI": PHI, "sg": sg, "g": g,
            "level": level, "fc": fc_fn, "mode": mode}


def _rtrap(y, x):                          # row-wise ∫ y dx (trapezoid), axis=1
    return np.sum(0.5 * (y[:, 1:] + y[:, :-1]) * np.diff(x, axis=1), axis=1)


def _rcumtrap(y, x):                       # row-wise cumulative trapezoid, leading 0
    c = np.cumsum(0.5 * (y[:, 1:] + y[:, :-1]) * np.diff(x, axis=1), axis=1)
    return np.concatenate([np.zeros((y.shape[0], 1)), c], axis=1)


def reconstruct(Qf, bstar, cfg):
    """Reconstruct face states Qf (n_vars, nf) to the common bed bstar (nf,).
    Dispatches on cfg['mode'] ('audusse' | 'bernoulli')."""
    if cfg["mode"] == "audusse":
        return _reconstruct_audusse(Qf, bstar, cfg)
    return _reconstruct_bernoulli(Qf, bstar, cfg)


def _reconstruct_audusse(Qf, bstar, cfg):
    """Lake-at-rest hydrostatic reconstruction: h*=max(0,h+b−b*), velocity
    preserved (q_k rescaled by h*/h), b→b*."""
    h_idx, b_idx, q_idx = cfg["h"], cfg["b"], cfg["q"]
    out = Qf.copy()
    h = Qf[h_idx]
    hstar = np.maximum(0.0, h + Qf[b_idx] - bstar)
    ratio = hstar / np.maximum(h, 1e-14)
    out[h_idx] = hstar
    out[b_idx] = bstar
    for qi in q_idx:
        out[qi] = Qf[qi] * ratio
    return out


def _reconstruct_bernoulli(Qf, bstar, cfg):
    """Moving-equilibrium reconstruction: preserve discharge q and the per-
    streamline Bernoulli head H(s).  Dry / zero-discharge faces pass through."""
    h_idx, b_idx, q_idx = cfg["h"], cfg["b"], cfg["q"]
    PHI, sg, g = cfg["PHI"], cfg["sg"], cfg["g"]
    dsg = sg[1] - sg[0] if len(sg) > 1 else 1.0
    out = Qf.copy()
    h = Qf[h_idx]; q = Qf[q_idx[0]]
    wet = (h > 1e-8) & (np.abs(q) > 1e-12)
    if not np.any(wet):
        return out
    hh = h[wet]; qq = q[wet]; bb = Qf[b_idx][wet]; bs = bstar[wet]
    eta = bb + hh
    alpha = np.array([Qf[q_idx[k]][wet] / hh for k in range(len(q_idx))])   # (L+1, nw)
    u = np.einsum("km,kn->nm", PHI, alpha)                                  # (nw, M) on σ-edges
    s = _rcumtrap(u, np.broadcast_to(sg, u.shape))                         # ∫₀^σ u dσ
    s = s / s[:, -1:]                            # discharge fraction, EXACTLY [0,1]
    H = 0.5 * u * u + g * eta[:, None]
    etas = eta.copy()
    for _ in range(60):
        disc = np.maximum(2.0 * (H - g * etas[:, None]), 1e-14)
        inv = disc ** -0.5
        f = qq * _rtrap(inv, s) - (etas - bs)
        fp = qq * _rtrap(g * disc ** -1.5, s) - 1.0
        etas = etas - f / fp
        if np.max(np.abs(f)) < 1e-13:
            break
    hstar = etas - bs
    ustar = np.sqrt(np.maximum(2.0 * (H - g * etas[:, None]), 1e-14))
    sigstar = (qq / hstar)[:, None] * _rcumtrap(1.0 / ustar, s)
    out[h_idx, wet] = hstar
    out[b_idx, wet] = bs
    out[q_idx[0], wet] = qq                       # discharge q = h·α_0 preserved exactly
    for k in range(1, len(q_idx)):
        phistar = legval(2 * sigstar - 1, np.eye(len(q_idx))[k])            # (nw, M)
        ak = (2 * k + 1) * (qq / hstar) * _rtrap(phistar, s)
        out[q_idx[k], wet] = hstar * ak
    return out
