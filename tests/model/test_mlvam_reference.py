"""ML-VAM(2,1) — TERM-BY-TERM pinning against an INDEPENDENT direct-Galerkin
construction.

No published reference exists for the multilayer VAM (it is our model), so
the reference here is a from-scratch second derivation, built with plain
sympy in this file (no pipeline machinery):

* per layer the conservative σ-form balances, Galerkin-projected with the
  interface boundary terms carried explicitly: ω̃(0) = G_{ℓ-½}/ρ,
  ω̃(1) = G_{ℓ+½}/ρ (KinematicBC orientation
  ``w|_at = ∂t I + u|_at·∂x I + G/ρ``); boundary w-traces take their
  KINEMATIC values, boundary u-traces the layer's OWN modal trace;
* Escalante spaces u ∈ P₁, w,p ∈ P₂ per layer with the closure cascade:
  bottom KBC / kinematic interface conditions close ŵ₂ (upward), surface
  p=0 / downward pressure traces close p̂₂;
* ω̃ in CLOSED form (layer-mass integrated, with the G offsets);
* the layer h-equations substituted (∂t h_ℓ via layer mass incl. G);
* assembly: the Hörnschemeyer fraction closure h_ℓ = l_ℓ·h, the global
  h-equation, the mean-u* transfer swap on the x-rows (w-traces are exactly
  continuous by the closures → no w-transfer), G eliminated from the
  partial mass sums;
* friction: Navier slip at the bed (layer 1 only), Newtonian interior.

Every dynamic row and every constraint row of ``MLVAM(2,1)`` must equal
the reference rows symbolically (constraints up to one CONSISTENT
constant factor).
"""
import pytest
import sympy as sp

from zoomy_core.model.models import MLVAM

NU = 1
TOP = NU + 1
NL = 2


@pytest.fixture(scope="module")
def mlvam21():
    model = MLVAM(n_layers=NL, level=NU)
    return model, model.system_model


def _reference(sm):
    t, xx = sm.time, sm.space[0]
    Fn = lambda n: sp.Function(n, real=True)(t, xx)
    h, b = Fn("h"), Fn("b")
    hl = {1: sp.Function("h_1", positive=True)(t, xx),
          2: sp.Function("h_2", positive=True)(t, xx)}
    G1 = Fn("G_1")
    # per-layer modal functions (indexed form, as in the derivation)
    qm = {l_: [sp.Function(f"q_{l_}", real=True)(j, t, xx) for j in range(NU + 1)]
          for l_ in (1, 2)}
    rm = {l_: [sp.Function(f"r_{l_}", real=True)(j, t, xx) for j in range(NU + 1)]
          for l_ in (1, 2)}
    Pm = {l_: [sp.Function(f"P_{l_}", real=True)(j, t, xx) for j in range(NU + 1)]
          for l_ in (1, 2)}
    g = sm.parameters.g
    rho = sm.parameters.rho
    lam = sm.parameters.lambda_s
    nu = sm.parameters.nu
    l1 = sm.parameters.l_1
    l = {1: l1, 2: 1 - l1}
    Dx = lambda e: sp.Derivative(e, xx)
    Dt = lambda e: sp.Derivative(e, t)
    zeta, s_ = sp.symbols("zeta s_", nonnegative=True)
    phi = [sp.legendre(j, 2 * zeta - 1) for j in range(TOP + 1)]
    mu = [sp.Rational(1, 2 * j + 1) for j in range(TOP + 1)]

    def zint(e):
        p_ = sp.Poly(sp.expand(sp.sympify(e).doit()), zeta)
        return sum(c / (n[0] + 1) for n, c in zip(p_.monoms(), p_.coeffs()))

    H = hl[1] + hl[2]
    zb = {1: b, 2: b + hl[1]}
    G = {0: sp.S.Zero, 1: G1, 2: sp.S.Zero}
    # layer h-equations: ∂t h_ℓ + ∂x q_ℓ0 = −(G_t − G_b)/ρ
    dth_l = {1: -Dx(qm[1][0]) - (G[1] - G[0]) / rho,
             2: -Dx(qm[2][0]) - (G[2] - G[1]) / rho}
    DT_SUBS = {Dt(hl[1]): dth_l[1], Dt(hl[2]): dth_l[2]}
    zb_t = {1: sp.S.Zero, 2: dth_l[1]}

    # ── per-layer rows (h_ℓ / G-atom variables, OWN traces) ──────────────
    p_top_trace = {2: sp.S.Zero}
    raw = {}
    for ell in (2, 1):
        hh = hl[ell]
        u0, u1 = qm[ell][0] / hh, qm[ell][1] / hh
        w0, w1 = rm[ell][0] / hh, rm[ell][1] / hh
        ut = u0 + u1 * phi[1]
        w_bot = zb_t[ell] + (u0 - u1) * Dx(zb[ell]) + G[ell - 1] / rho
        w2 = w_bot - w0 + w1
        wt = w0 + w1 * phi[1] + w2 * phi[2]
        p2 = p_top_trace[ell] - Pm[ell][0] - Pm[ell][1]
        pt = Pm[ell][0] + Pm[ell][1] * phi[1] + p2 * phi[2]
        if ell == 2:
            p_top_trace[1] = sp.expand(Pm[ell][0] - Pm[ell][1] + p2)
        omega = (G[ell - 1] / rho - zeta * (dth_l[ell] + Dx(qm[ell][0]))
                 - Dx(qm[ell][1])
                 * sp.integrate(phi[1].subs(zeta, s_), (s_, 0, zeta)))
        omega_def = (wt - (zb_t[ell] + zeta * Dt(hh))
                     - ut * (Dx(zb[ell]) + zeta * Dx(hh)))
        zt = zb[ell] + hh
        zt_t = zb_t[ell] + dth_l[ell]
        w_top_kin = zt_t + ut.subs(zeta, 1) * Dx(zt) + G[ell] / rho
        ptop, pbot = p_top_trace[ell], pt.subs(zeta, 0)
        for k in range(NU + 1):
            pk = phi[k]
            pk0, pk1 = pk.subs(zeta, 0), pk.subs(zeta, 1)
            dpk = sp.diff(pk, zeta)
            interior = zint((Dt(hh * ut) + Dx(hh * ut**2 + hh * pt / rho)
                             + g * hh * Dx(b + H)) * pk)
            byparts = -zint((omega * ut
                             - (pt / rho) * (Dx(zb[ell]) + zeta * Dx(hh)))
                            * dpk)
            bdry = ((G[ell] / rho * ut.subs(zeta, 1)
                     - (ptop / rho) * Dx(zt)) * pk1
                    - (G[ell - 1] / rho * ut.subs(zeta, 0)
                       - (pbot / rho) * Dx(zb[ell])) * pk0)
            tau0 = lam * ut.subs(zeta, 0) if ell == 1 else sp.S.Zero
            fric = tau0 * pk0 / rho + zint((nu / hh) * sp.diff(ut, zeta) * dpk)
            e = sp.expand(((interior + byparts + bdry + fric) / mu[k]).doit())
            raw[f"x{ell}{k}"] = sp.expand(e.subs(DT_SUBS).doit())
            interior = zint((Dt(hh * wt) + Dx(hh * ut * wt)) * pk)
            byparts = -zint((omega * wt + pt / rho) * dpk)
            bdry = ((G[ell] / rho * w_top_kin + ptop / rho) * pk1
                    - (G[ell - 1] / rho * wt.subs(zeta, 0) + pbot / rho) * pk0)
            e = sp.expand(((interior + byparts + bdry) / mu[k]).doit())
            raw[f"z{ell}{k}"] = sp.expand(e.subs(DT_SUBS).doit())
        for k in range(1, NU + 2):
            pk = phi[k]
            pk0, pk1 = pk.subs(zeta, 0), pk.subs(zeta, 1)
            dpk = sp.diff(pk, zeta)
            interior = zint((Dt(hh) + Dx(hh * ut)) * pk)
            byparts = -zint(omega_def * dpk)
            bdry = (G[ell] / rho) * pk1 - (G[ell - 1] / rho) * pk0
            e = sp.expand((interior + byparts + bdry).doit())
            raw[f"c{ell}{k - 1}"] = sp.expand(e.subs(DT_SUBS).doit())

    # ── assembly: fractions, global h-eq, u*-swap, G elimination ─────────
    Q0 = qm[1][0] + qm[2][0]
    dth = -Dx(Q0)
    frac = {hl[1]: l[1] * h, hl[2]: l[2] * h}
    G_final = sp.expand(rho * (l[1] * Dx(Q0) - Dx(qm[1][0])))
    trace = {(1, 1): (qm[1][0] + qm[1][1]) / (l[1] * h),
             (2, 0): (qm[2][0] - qm[2][1]) / (l[2] * h)}
    ustar = (trace[(1, 1)] + trace[(2, 0)]) / 2

    def assemble(e, *, swap=None):
        e = sp.expand(sp.sympify(e).subs(frac).doit())
        e = sp.expand(e.subs(Dt(h), dth))
        if swap is not None:
            ell, k = swap
            # interface a = ℓ (top, sgn +) and a = ℓ−1 (bottom, sgn −)
            for a, side, sgn in ((ell, 1, +1), (ell - 1, 0, -1)):
                if 1 <= a <= NL - 1:
                    phik = 1 if side == 1 else (-1) ** k
                    own = trace[(ell, side)]
                    e = e + sgn * phik * (ustar - own) * G1 / rho
        e = sp.expand(e.subs(G1, G_final).doit())
        # indexed modal functions → final state names
        for l_ in (1, 2):
            for j in range(NU + 1):
                e = e.subs({qm[l_][j]: Fn(f"q_{l_}_{j}"),
                            rm[l_][j]: Fn(f"r_{l_}_{j}"),
                            Pm[l_][j]: Fn(f"P_{l_}_{j}")})
        return sp.expand(e.doit())

    rows = {"b": Dt(b), "h": Dt(h) + Dx(Q0).subs(
        {qm[l_][0]: Fn(f"q_{l_}_0") for l_ in (1, 2)})}
    for ell in (1, 2):
        for k in range(NU + 1):
            rows[f"q_{ell}_{k}"] = assemble(raw[f"x{ell}{k}"], swap=(ell, k))
            rows[f"r_{ell}_{k}"] = assemble(raw[f"z{ell}{k}"])
        for j in range(NU + 1):
            rows[f"P_{ell}_{j}"] = assemble(raw[f"c{ell}{j}"])
    return rows


def test_mlvam21_rows_match_independent_galerkin(mlvam21):
    _, sm = mlvam21
    refs = _reference(sm)
    rv = sm.reconstruct_residuals()
    names = [str(s) for s in sm.state]

    failures = []
    ratios = {}
    for i, name in enumerate(names):
        mine = sp.expand(sp.sympify(rv[i]).doit())
        ref = sp.expand(sp.sympify(refs[name]).doit())
        if name.startswith("P_"):
            d_plain = sp.simplify(mine - ref)
            if d_plain == 0:
                ratios[name] = 1
                continue
            hit = None
            for cand in (-1, 2, -2, sp.Rational(1, 2), sp.Rational(-1, 2),
                         3, -3, sp.Rational(1, 3), sp.Rational(-1, 3)):
                if sp.simplify(mine - cand * ref) == 0:
                    hit = cand
                    break
            if hit is None:
                failures.append((name, sp.simplify(mine - ref)))
            else:
                ratios[name] = hit
            continue
        diff = sp.simplify(mine - ref)
        if diff != 0:
            failures.append((name, diff))

    msg = "\n".join(f"row {n}: DIFF {d}" for n, d in failures)
    assert not failures, f"ML-VAM(2,1) vs independent Galerkin:\n{msg}"
    assert len(set(ratios.values())) == 1, f"constraint factors: {ratios}"
