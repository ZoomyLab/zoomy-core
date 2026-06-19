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

import logging

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import legval, leggauss
from numpy.polynomial import polynomial as npoly

_LOG = logging.getLogger(__name__)


def _gauss_on_unit(n):
    """Gauss-Legendre nodes/weights of order ``n`` mapped from [-1,1] to [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _cumulative_weight_matrix(nodes):
    """Spectral cumulative-weight matrix ``C`` for the Lagrange basis through
    ``nodes`` (on [0,1]).  ``C[j,i] = ∫₀^{σ_j} ℓ_i(σ) dσ`` so that, for any
    integrand sampled at the nodes, ``cum = thick @ C.T`` gives its cumulative
    integral at the nodes in O(n) evals (NO fine sub-grid).  Each ``ℓ_i`` is
    integrated exactly via ``numpy.polynomial`` (degree n−1 → exact)."""
    n = len(nodes)
    C = np.zeros((n, n))
    for i in range(n):
        others = np.delete(nodes, i)
        coeffs = npoly.polyfromroots(others) / np.prod(nodes[i] - others)  # ℓ_i, ascending
        anti = npoly.polyint(coeffs)                       # antiderivative, F(0)=0
        C[:, i] = npoly.polyval(nodes, anti)               # ∫₀^{σ_j} ℓ_i dσ
    return C


def build_bernoulli_config(model, mode="bernoulli", quadrature="trapezoid", n_sigma=200):
    """Resolve indices, basis and the conservative flux Fc=flux+pressure.

    ``mode`` selects the equilibrium reconstruction: 'audusse' (lake-at-rest:
    h*=max(0,h+b−b*), velocity preserved), 'bernoulli' (moving: preserve the
    discharge q and the per-streamline Bernoulli head H(s) via the discharge-
    fraction streamline label), or 'projected_bernoulli' (layered moving WB:
    σ-parameterized |u| thickness weighting + signed velocity, handles sign
    reversal over depth).

    ``quadrature`` selects how the ``projected_bernoulli`` σ-integrals (Newton on
    eps and the moment reprojection) are evaluated; it is ignored by the other
    modes:

      * 'trapezoid' (default) — uniform trapezoid over ``n_sigma`` σ-edges, the
        original committed behaviour (O(1/n²), floors L1(u) ~1e-6 for L≥4).
      * 'gauss' — Gauss-Legendre with ``n = 2·(level+1)`` nodes/weights; the
        cumulative σ*-remap uses the precomputed spectral cumulative-weight
        matrix ``C`` (O(n) integrand evals, spectral accuracy, NO fine grid).
      * 'exact' — at level 0 the SWE specific-energy cubic is solved in closed
        form (machine precision, NO quadrature); for level≥1 no elementary
        closed form exists (√ over a degree-L polynomial) so it falls back to
        'gauss' (logged once here)."""
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

    # 'exact' has a closed form only at level 0 (constant velocity → SWE cubic);
    # for level≥1 the streamline √ runs over a degree-L polynomial with no
    # elementary antiderivative, so fall back to spectral 'gauss'.
    if quadrature == "exact" and level >= 1:
        _LOG.info(
            "projected_bernoulli quadrature='exact' has no closed form for "
            "level=%d (√ over a degree-%d polynomial); falling back to 'gauss'.",
            level, level)
        quadrature = "gauss"

    # Spectral Gauss-Legendre quadrature (also the level≥1 'exact' fallback).
    gnodes = gweights = gphi = gcum = None
    if quadrature == "gauss":
        n = 2 * (level + 1)
        gnodes, gweights = _gauss_on_unit(n)
        gphi = np.array([legval(2 * gnodes - 1, np.eye(level + 1)[k])
                         for k in range(level + 1)])
        gcum = _cumulative_weight_matrix(gnodes)
    # Well-balancing source flux, lambdified PER DIRECTION; the solver projects
    # it on the face normal (Σ_d Fc_d·n_d) so the source is dimension-agnostic
    # (1-D: n=±1 recovers the direction-0 flux; 2-D: Fc_x·n_x + Fc_y·n_y).
    #
    #   * 'audusse'  -> PRESSURE jump only (hydrostatic_pressure): the momentum-
    #     only Audusse S̃ source.  Zero mass row (mass-conservative), no
    #     convective term (no aux/hinv dependence, so aux-flux models like
    #     MalpassetSWE work), and dimension-agnostic.
    #   * 'bernoulli' -> full conservative flux Fc = flux + hydrostatic_pressure:
    #     the moving-equilibrium reconstruction preserves the discharge q, so
    #     Fc's mass row cancels in the jump (conservative there too).
    state = list(model.state)
    ndim = int(model.flux.shape[1])
    pars = list(model.parameters)
    pv = [float(model.parameter_values[str(p)]) for p in pars]
    _syms = ([sp.Symbol(str(s), real=True) for s in state]
             + [sp.Symbol(str(p), real=True) for p in pars])

    def _fc_rd(r, d):
        P = sp.sympify(model.hydrostatic_pressure[r, d])
        return P if mode == "audusse" else sp.sympify(model.flux[r, d]) + P

    fc_d = [sp.lambdify(_syms, [_fc_rd(r, d) for r in range(len(state))], "numpy")
            for d in range(ndim)]

    def fc_fn(Qf, n):                      # Qf (n_vars,nf), n (ndim,nf) -> (n_vars,nf)
        nf = Qf.shape[1]
        out = np.zeros((len(state), nf))
        for d in range(ndim):
            rows = fc_d[d](*Qf, *pv)       # some rows are constant -> scalars
            Fcd = np.stack([np.broadcast_to(np.asarray(r, dtype=float), (nf,))
                            for r in rows])
            out = out + Fcd * n[d]         # project on the face normal
        return out
    return {"h": h_idx, "b": b_idx, "q": q_idx, "PHI": PHI, "sg": sg, "g": g,
            "level": level, "fc": fc_fn, "mode": mode, "quadrature": quadrature,
            "gauss_nodes": gnodes, "gauss_w": gweights, "gauss_PHI": gphi,
            "gauss_C": gcum}


def _rtrap(y, x):                          # row-wise ∫ y dx (trapezoid), axis=1
    return np.sum(0.5 * (y[:, 1:] + y[:, :-1]) * np.diff(x, axis=1), axis=1)


def _rcumtrap(y, x):                       # row-wise cumulative trapezoid, leading 0
    c = np.cumsum(0.5 * (y[:, 1:] + y[:, :-1]) * np.diff(x, axis=1), axis=1)
    return np.concatenate([np.zeros((y.shape[0], 1)), c], axis=1)


def reconstruct(Qf, bstar, cfg):
    """Reconstruct face states Qf (n_vars, nf) to the common bed bstar (nf,).
    Dispatches on cfg['mode'] ('audusse' | 'bernoulli' | 'projected_bernoulli')."""
    if cfg["mode"] == "audusse":
        return _reconstruct_audusse(Qf, bstar, cfg)
    if cfg["mode"] == "projected_bernoulli":
        return _reconstruct_projected_bernoulli(Qf, bstar, cfg)
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


def _reconstruct_projected_bernoulli(Qf, bstar, cfg):
    """Layered moving-equilibrium reconstruction (interpolate → per-z signed
    Bernoulli → project).

    Per wet column with bed b, depth h, free surface η=b+h and velocity profile
    u(σ)=Σ_k α_k P_k(2σ−1) on a σ-edge grid, each streamline carries the SAME
    full hydrostatic head H(σ)=½u²+gη.  Reconstructing to the common bed b*
    solves ONE scalar Newton for the new surface η*::

        h·∫₀¹ |u| / √(u² + 2g(η−η*)) dσ  =  η* − b*

    (σ-parameterized with |u| thickness weighting — monotone and sign-reversal
    safe, unlike the discharge-fraction label of ``_reconstruct_bernoulli``).
    The reconstructed velocity keeps its sign, u*(σ)=sign(u)·√(u²+2g(η−η*)), the
    new depth coordinate σ*(σ) is the normalized cumulative thickness, and the
    signed profile is reprojected onto Legendre moments
    q_k* = (2k+1)·h*·∫ u*(σ) P_k(2σ*−1) dσ* ≡ (2k+1)·h·∫ u(σ) P_k(2σ*−1) dσ
    (h*=η*−b*); q_0*=h*α_0* equals the original discharge by construction.
    Dry / zero-discharge columns pass through.

    Dispatches on ``cfg['quadrature']`` ('trapezoid' | 'gauss'); 'exact' (level 0
    only) is routed to the closed-form SWE cubic before reaching this kernel.
    The trapezoid path is byte-identical to the original committed behaviour.
    """
    quad = cfg["quadrature"]
    if quad == "exact":
        return _reconstruct_projected_exact(Qf, bstar, cfg)
    if quad == "gauss":
        PHI = cfg["gauss_PHI"]; w = cfg["gauss_w"]; C = cfg["gauss_C"]

        def integral(y):                       # ∫₀¹ y dσ ≈ Σ_j w_j y_j  (spectral)
            return y @ w

        def cumulative(thick):                 # cum[j]=Σ_i C[j,i] thick_i ; total=Σ_i w_i thick_i
            return thick @ C.T, thick @ w
    else:                                       # 'trapezoid' (default, byte-unchanged)
        PHI, sg = cfg["PHI"], cfg["sg"]

        def integral(y):
            return _rtrap(y, np.broadcast_to(sg, y.shape))

        def cumulative(thick):
            cumz = _rcumtrap(thick, np.broadcast_to(sg, thick.shape))
            return cumz, cumz[:, -1]
    return _projected_bernoulli_core(Qf, bstar, cfg, PHI, integral, cumulative)


def _projected_bernoulli_core(Qf, bstar, cfg, PHI, integral, cumulative):
    """Quadrature-generic projected-Bernoulli kernel.  ``PHI`` is the Legendre
    basis sampled at the quadrature abscissae; ``integral(y)`` returns the
    row-wise ∫₀¹ y dσ; ``cumulative(thick)`` returns (cumulative-at-abscissae,
    total) for the σ*-remap.  trapezoid and gauss differ only in these three."""
    h_idx, b_idx, q_idx, g = cfg["h"], cfg["b"], cfg["q"], cfg["g"]
    nq = len(q_idx)
    out = Qf.copy()
    h = Qf[h_idx]; q = Qf[q_idx[0]]
    wet = (h > 1e-8) & (np.abs(q) > 1e-12)
    if not np.any(wet):
        return out
    hh = h[wet]; bb = Qf[b_idx][wet]; bs = bstar[wet]
    eta = bb + hh
    alpha = np.array([Qf[q_idx[k]][wet] / hh for k in range(nq)])           # (L+1, nw)
    u = np.einsum("km,kn->nm", PHI, alpha)                                  # (nw, M) on abscissae
    au = np.abs(u)

    # ONE scalar Newton (shared single η* per column) on eps = η − η* > 0.
    #   disc = u² + 2g·eps = 2(H − gη*) ;  h* = h·∫|u|/√disc dσ = η*−b*
    eps = np.full(eta.shape, 1e-9)
    for _ in range(100):
        disc = u * u + 2.0 * g * eps[:, None]
        lhs = hh * integral(au / np.sqrt(disc))                            # h*+ candidate
        f = lhs - (eta - eps - bs)                                         # η*=η−eps
        dlhs = hh * integral(-g * au * disc ** -1.5)
        step = f / (dlhs + 1.0)                # d(η*−b*)/d eps = −1 ⇒ +1 term
        eps_new = eps - step
        eps = np.where(eps_new <= 0.0, 0.5 * eps, eps_new)                  # keep eps>0
        if np.max(np.abs(step)) < 1e-15:
            break

    etas = eta - eps
    hstar = etas - bs
    disc = u * u + 2.0 * g * eps[:, None]
    sqrt_disc = np.sqrt(disc)
    # new depth coordinate σ*(σ): normalized cumulative |u|/√disc thickness
    thick = au / sqrt_disc                                                  # = u/u* (well-defined)
    cum, total = cumulative(thick)
    sigstar = cum / total[:, None]                                          # in [0,1], monotone

    out[h_idx, wet] = hstar
    out[b_idx, wet] = bs
    # reproject signed profile: q_k*=(2k+1)·h*·∫u* P_k(2σ*−1)dσ* ≡ (2k+1)·h·∫u P_k(2σ*−1)dσ
    for k in range(nq):
        phistar = legval(2 * sigstar - 1, np.eye(nq)[k])                    # (nw, M)
        out[q_idx[k], wet] = (2 * k + 1) * hh * integral(u * phistar)
    return out


def _reconstruct_projected_exact(Qf, bstar, cfg):
    """Level-0 (constant-velocity) closed-form SWE reconstruction — NO quadrature.

    With u(σ)=α_0 the per-streamline head collapses to the cell specific energy
    E = ½(q/h)² + g(b+h); reconstructing to bed b* with preserved discharge q
    means the new depth h* solves ½(q/h*)² + g(h*+b*) = E, i.e. the cubic

        g·h*³ + (g·b* − E)·h*² + ½q² = 0.

    Three real roots (negative, supercritical, subcritical); the SUBCRITICAL root
    is the largest positive one.  Solved analytically (depressed-cubic trig form,
    p = −a₂²/3 ≤ 0 so the three-real-roots branch always applies)."""
    h_idx, b_idx, q_idx, g = cfg["h"], cfg["b"], cfg["q"], cfg["g"]
    out = Qf.copy()
    h = Qf[h_idx]; q = Qf[q_idx[0]]
    wet = (h > 1e-8) & (np.abs(q) > 1e-12)
    if not np.any(wet):
        return out
    hh = h[wet]; bb = Qf[b_idx][wet]; bs = bstar[wet]; qq = q[wet]
    E = 0.5 * (qq / hh) ** 2 + g * (bb + hh)               # cell Bernoulli head
    # monic cubic h³ + a2 h² + a0 = 0 (no linear term); depressed t³+p t+q3=0, h=t−a2/3
    a2 = bs - E / g
    a0 = 0.5 * qq ** 2 / g
    p = -a2 ** 2 / 3.0
    q3 = 2.0 * a2 ** 3 / 27.0 + a0
    r = np.sqrt(np.maximum(-p / 3.0, 0.0))
    arg = np.clip((3.0 * q3) / (2.0 * p) * np.sqrt(np.maximum(-3.0 / p, 0.0)), -1.0, 1.0)
    phi = np.arccos(arg)
    roots = np.stack([2.0 * r * np.cos(phi / 3.0 - 2.0 * np.pi * kk / 3.0) - a2 / 3.0
                      for kk in range(3)])                # (3, nw)
    hstar = np.max(np.where(roots > 0.0, roots, -np.inf), axis=0)   # subcritical = largest positive
    out[h_idx, wet] = hstar
    out[b_idx, wet] = bs
    out[q_idx[0], wet] = qq                               # discharge q=h·α_0 preserved exactly
    return out
