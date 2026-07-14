"""MLSWE vs Aguillon, Hörnschemeyer & Sainte-Marie ("Barotropic-Baroclinic
Splitting for Multilayer Shallow Water Models with Exchanges") — TERM-BY-TERM
reference pinning against their closed system (8) with the interface
upwinding (9), under the layer-fraction constraint h_α = l_α·h.

Their (8), 1D, constant density, inviscid:

* ∂t h + ∂x(h ū) = 0,                          ū = Σ_j l_j u_j
* ∂t(h u_α) + ∂x(h u_α² + g h²/2)
      = −g h ∂x z_b + (1/l_α)(u*_{α+½} G_{α+½} − u*_{α−½} G_{α−½})
* G_{α+½} = Σ_{j≤α} l_j ∂x(h(u_j − ū)),        G_{½} = G_{N+½} = 0
* (9): u*_{α+½} = u_α if G_{α+½} ≤ 0 else u_{α+1}   (upwind), or the
  arithmetic mean (Audusse et al.) — our ``interface_velocity`` selector.

Mapping (theirs ← ours):

* our state is [b, h, q_1 … q_N] with q_α = h_α u_α = l_α·h·u_α  ⇒
  u_α = q_α/(l_α h), ū = Σ q_j / h; our momentum row α (unit ∂t q_α after
  InvertMassMatrix) equals l_α × their (h u_α)-row;
* our inlined exchange flux carries ρ and the opposite orientation,
  G_ours = −ρ·G_theirs — both cancel inside the rows, and our Piecewise
  condition (G_ours ≥ 0 → u_α) coincides exactly with their (9)
  (G_theirs ≤ 0 → u_α);
* their system is inviscid ⇒ our λ_s, ν are zeroed symbolically.

Shape, count and smoke checks are NEVER sufficient — pin every term.
"""
import pytest
import sympy as sp

from zoomy_core.model.models import MLSWE
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel.system_model import SystemModel


def _fractions(sm, N):
    ls = [getattr(sm.parameters, f"l_{j}") for j in range(1, N)]
    return ls + [1 - sum(ls)]


def _reference_rows(sm, N, ustar_of):
    """Paper system (8) rows in OUR variables/order [b, h, q_1 … q_N];
    momentum rows pre-multiplied by l_α.  ``ustar_of(a, G)`` returns the
    interface velocity u*_{a+½} given the interface index a (1 … N−1) and
    the paper-orientation exchange flux G_{a+½}."""
    t, x = sm.time, sm.space[0]
    Fn = lambda n: sp.Function(n, real=True)(t, x)
    h, b = Fn("h"), Fn("b")
    q = [Fn(f"q_{a}_0") for a in range(1, N + 1)]   # layer α, mode 0
    g = sm.parameters.g
    l = _fractions(sm, N)
    Dx = lambda e: sp.Derivative(e, x)
    Dt = lambda e: sp.Derivative(e, t)

    u = [q[a] / (l[a] * h) for a in range(N)]
    ubar = sum(q) / h

    G = {0: sp.S.Zero, N: sp.S.Zero}            # G_{½} and G_{N+½}
    for a in range(1, N):
        G[a] = sum(l[j] * Dx(h * (u[j] - ubar)) for j in range(a))

    rows = [Dt(b), Dt(h) + Dx(h * ubar)]
    for a in range(1, N + 1):
        transfer = (ustar_of(a, G[a]) * G[a]
                    - ustar_of(a - 1, G[a - 1]) * G[a - 1])
        rows.append(l[a - 1] * (
            Dt(h * u[a - 1]) + Dx(h * u[a - 1]**2 + g * h**2 / 2)
            + g * h * Dx(b) - transfer / l[a - 1]))
    return rows, u


def _assert_rows_equal(sm, refs, extra_subs=None):
    inviscid = {sm.parameters.lambda_s: 0, sm.parameters.nu: 0}
    rv = sm.reconstruct_residuals()
    for i, (mine, ref) in enumerate(zip(rv, refs)):
        mine = sp.sympify(mine).subs(inviscid)
        if extra_subs:
            mine = mine.replace(*extra_subs)
        diff = sp.simplify(sp.expand(mine.doit()) - sp.expand(ref.doit()))
        assert diff == 0, (
            f"row {i} ({sm.state[i]}) differs from Hörnschemeyer eq (8): "
            f"{diff}")


@pytest.mark.parametrize("n_layers", [2, 3])
def test_mlswe_mean_interface_matches_hornschemeyer_eq8(n_layers):
    sm = SystemModel.from_model(MLSWE(closures=[Newtonian(), NavierSlip(), StressFree()], n_layers=n_layers, interface_velocity="mean"))
    assert [str(s) for s in sm.state] == (
        ["b", "h"] + [f"q_{a}_0" for a in range(1, n_layers + 1)])

    def ustar_mean(a, G):
        if a == 0 or a == n_layers:
            return sp.S.Zero                    # multiplies G = 0
        return (u[a - 1] + u[a]) / 2

    refs, u = _reference_rows(sm, n_layers, lambda a, G: 0)
    # rebuild with the closure now that u exists
    refs, u = _reference_rows(sm, n_layers, ustar_mean)
    _assert_rows_equal(sm, refs)


@pytest.mark.parametrize("branch", ["below", "above"])
def test_mlswe_upwind_interface_matches_eq9_branches(branch):
    """Their (9) is a sign switch per interface; with N=2 there is ONE
    internal interface — pin BOTH branches: my Piecewise true-branch
    (G_ours ≥ 0 ⟺ G_theirs ≤ 0) must be the paper's u_α ('below'),
    the false-branch their u_{α+1} ('above')."""
    n_layers = 2
    sm = SystemModel.from_model(MLSWE(closures=[Newtonian(), NavierSlip(), StressFree()], n_layers=n_layers, interface_velocity="upwind"))

    take_true = branch == "below"

    def pick_branch(expr):
        def _sel(pw):
            return pw.args[0][0] if take_true else pw.args[1][0]
        return expr.replace(lambda e: isinstance(e, sp.Piecewise), _sel)

    refs_holder = {}

    def ustar_up(a, G):
        if a == 0 or a == n_layers:
            return sp.S.Zero
        u = refs_holder["u"]
        return u[a - 1] if take_true else u[a]

    refs, u = _reference_rows(sm, n_layers, lambda a, G: 0)
    refs_holder["u"] = u
    refs, u = _reference_rows(sm, n_layers, ustar_up)

    inviscid = {sm.parameters.lambda_s: 0, sm.parameters.nu: 0}
    rv = sm.reconstruct_residuals()
    for i, (mine, ref) in enumerate(zip(rv, refs)):
        mine = pick_branch(sp.sympify(mine).subs(inviscid))
        diff = sp.simplify(sp.expand(mine.doit()) - sp.expand(ref.doit()))
        assert diff == 0, (
            f"row {i} ({sm.state[i]}) [{branch}-branch] differs from "
            f"Hörnschemeyer eq (8)+(9): {diff}")
