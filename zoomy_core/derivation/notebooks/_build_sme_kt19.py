"""Generate ``sme_kt19.ipynb`` — the declarative SME → K&T (4.17) walkthrough.

Run with the zoomy env::

    python _build_sme_kt19.py

This writes ``sme_kt19.ipynb`` next to it.  The notebook threads ONE
``zoomy_core.derivation.Model`` from the local 3-D axioms to the K&T (4.17)
shallow-moment system and ends with ``SystemModel.from_model(model, Q).describe()``.
Every state is rendered with ``model.describe()`` / ``.describe()`` return
values (no ``print`` / ``display``).
"""

from __future__ import annotations

import json
import os

CELLS = []


def md(src):
    CELLS.append(("markdown", src.strip("\n")))


def code(src):
    CELLS.append(("code", src.strip("\n")))


# ── §0 — title ────────────────────────────────────────────────────────────
md(r"""
# Declarative SME → Kowalski–Torrilhon (2019)

This notebook derives the **Shallow Moment Equations** end-to-end through the
clean-redesign derivation surface and reproduces K&T (2019) eq. (4.17):

$$
\partial_t h + \partial_x q_0 = 0, \qquad
\partial_t q_0 + \partial_x\!\Big(\tfrac{g h^2}{2}
   + \tfrac{q_0^2}{h} + \tfrac{q_1^2}{3h} + \tfrac{q_2^2}{5h}\Big)
   + g h\,\partial_x b
   + \tfrac{\tau_{xz}(0) - \tau_{xz}(1)}{\rho} = 0 .
$$

One `Model` thread, real `.apply` ops only, `SystemModel.from_model(model, Q)`
at the transition.  Every state is shown with a `.describe()` return value.
""")

# ── §1 — axioms ───────────────────────────────────────────────────────────
md(r"""
## §1 — Axioms: the GENERAL 2-D incompressible balance

We start from the **full primitive system** — local continuity and the
2-component momentum carrying the **complete viscous-stress tensor**
($\tau_{xx}, \tau_{xz}, \tau_{zx}, \tau_{zz}$) and the pressure $p$ **NOT yet
reduced**.  Nothing is hand-folded: the hydrostatic pressure and the
$g\,\partial_x(b+h)$ gravity flux are *derived* in §1b.

$$
\partial_x u + \partial_z w = 0,\qquad
\partial_t u + \partial_x(u^2) + \partial_z(uw) + \tfrac1\rho\partial_x p
   - \tfrac1\rho(\partial_x\tau_{xx} + \partial_z\tau_{xz}) = 0,
$$
$$
\partial_t w + \partial_x(uw) + \partial_z(w^2) + \tfrac1\rho\partial_z p + g
   - \tfrac1\rho(\partial_x\tau_{zx} + \partial_z\tau_{zz}) = 0 .
$$
""")

code(r"""
import sympy as sp
from zoomy_core import coords
import zoomy_core.derivatives as d
from zoomy_core.derivation import (
    Model, PDETransformation, Basis, resolve_modes,
    separation_of_variables, reset_modal_indices, modal_bound,
    Substitution, ChangeOfVariables,
    Resolve, Simplify, kinematic_modal_closure, mass_relation,
    fold_to_conservative_form,
)
from zoomy_core.model.operations import Legendre_shifted, Multiply, Integrate
from zoomy_core.model.models.system_model import SystemModel

t, x, z = coords.t, coords.x, coords.z
zeta = sp.Symbol("zeta", real=True)

model = Model(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
g, rho = model.parameters.g, model.parameters.rho

u = sp.Function("u", real=True)(t, x, z)
w = sp.Function("w", real=True)(t, x, z)
p = sp.Function("p", real=True)(t, x, z)
h = sp.Function("h", positive=True)(t, x)
b = sp.Function("b", real=True)(t, x)
txx = sp.Function("tau_xx", real=True)(t, x, z)
txz = sp.Function("tau_xz", real=True)(t, x, z)
tzx = sp.Function("tau_zx", real=True)(t, x, z)
tzz = sp.Function("tau_zz", real=True)(t, x, z)

model.Q = [h, u, w, p]
model.add_equation("bottom", d.t(b))
model.add_equation("mass", d.x(u) + d.z(w))
model.add_equation("momentum", (2,), [
    d.t(u) + d.x(u * u) + d.z(u * w) + d.x(p) / rho
    - (d.x(txx) + d.z(txz)) / rho,                      # .x
    d.t(w) + d.x(u * w) + d.z(w * w) + d.z(p) / rho + g
    - (d.x(tzx) + d.z(tzz)) / rho,                      # .z
])
model.describe(strip_args=True)
""")

# ── §1b — stress reduction + DERIVED hydrostatic pressure ─────────────────
md(r"""
## §1b — Stress reduction + DERIVED hydrostatic pressure

**(a) Stress reduction** (thin-film / shallow): drop the normal stresses and
symmetrise the shear — $\tau_{xx}\to0,\ \tau_{zz}\to0,\ \tau_{zx}\to\tau_{xz}$.

**(b) Hydrostatic pressure — derived, *not* hand-written.**  Reduce the
z-momentum to the hydrostatic balance $\partial_z p = -\rho g$ by dropping the
vertical inertia and the lateral shear, **`Integrate`** it over $z\to b+h$
(FTC: $\partial_z p\to p(\eta)-p(z)$), impose the free-surface BC $p(z=b+h)=0$
(a `Substitution`), then **`solve_for`** $p$ (pure algebra) → $p = \rho
g\,(b+h-z)$, and substitute into the x-momentum.
The $\tfrac1\rho\partial_x p$ then **derives** the $g\,\partial_x(b+h)$ gravity
flux.  Finally drop the (now redundant) z-momentum row.
""")

code(r"""
# (a) stress reduction
model.apply(Substitution({txx: 0, tzz: 0, tzx: txz}))

# (b) hydrostatic: reduce z-momentum, INTEGRATE over the vertical (FTC),
#     impose the free-surface BC p(b+h)=0, then ALGEBRAICALLY solve_for p
#     (solve_for no longer integrates), substitute into x-momentum, remove z.
model.momentum.z.apply(Substitution(
    {d.t(w): 0, d.x(u * w): 0, d.z(w * w): 0, d.x(txz): 0}))
model.momentum.z.apply(Integrate(z, z, b + h, method="analytical"))
model.momentum.z.apply(Substitution({p.subs(z, b + h): 0}))
p_hydro = model.momentum.z.solve_for(p)
model.momentum.x.apply(p_hydro)
model.momentum.z.remove()
model.momentum.x.apply(Simplify())

from zoomy_core.misc.description import Description
Description(
    "**DERIVED** $p = " + sp.latex(p_hydro.subs_map[p]) + "$\n\n"
    "**momentum.x** (note the DERIVED $g\\,\\partial_x(b+h)$, not hand-folded):"
    "\n\n$" + sp.latex(sp.expand(model.momentum.x.expr)) + "$")
""")

# ── §2 — σ-map ────────────────────────────────────────────────────────────
md(r"""
## §2 — σ-map `z = b + h·ζ`

`PDETransformation` maps the physical vertical `z` to the reference `ζ∈[0,1]`,
minting decorated heads `ũ(t,x,ζ)` and rewriting `∂_z`/`∂_x` by the σ chain
rule with **zero `Subs`** artifacts.
""")

code(r"""
model.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))
model.describe(strip_args=True)
""")

# ── §3 — conservative pre-fold + modal ansatz ─────────────────────────────
md(r"""
## §3 — Conservative pre-fold + modal ansatz

`Multiply(h)` clears the `1/h` σ-Jacobian on the dynamic rows.  Then
`separation_of_variables` substitutes the **unexpanded** modal sums
`ũ = Σ_i a_i(t,x)·φ_i(ζ)` (u, order `N_u`) and `w̃ = Σ_j aw_j·φ_j` (order
`N_u+1`) with the opaque basis — Legendre-free.
""")

code(r"""
model.mass.apply(Multiply(h))
model.momentum.x.apply(Multiply(h))

basis = Basis(symbol="phi", weight="c")
phi, c = basis.phi, basis.weight
a   = sp.Function("a")
a_w = sp.Function("aw")
N_u = modal_bound("N_u")
reset_modal_indices(model)
model.apply(separation_of_variables(u,   a(t, x), basis, N_u))
model.apply(separation_of_variables(w, a_w(t, x), basis, N_u + 1))
model.mass.describe(strip_args=True)
""")

# ── §4 — bind N ───────────────────────────────────────────────────────────
md(r"""
## §4 — Bind the truncation `N_u = 2` at the Model level

Binding `N_u` to the concrete level-2 truncation collapses the abstract `Sum`
to the explicit modes `a_0, a_1, a_2` (u) and `aw_0..aw_3` (w).  This happens
at the **Model** level, before any SystemModel.
""")

code(r"""
N = 2
model.apply(Substitution({N_u: N}))
for eq in model._equations.values():
    eq.expr = eq.expr.replace(lambda e: isinstance(e, sp.Sum), lambda e: e.doit())
n_modes = N + 1
basis_lvl = N + 1
model.mass.describe(strip_args=True)
""")

# ── §5 — continuity moment family via resolve_modes ────────────────────────
md(r"""
## §5 — Continuity moment family via `resolve_modes`

`resolve_modes` PROJECTs the continuity onto the abstract test weight
`c(ζ)·φ(l,ζ)` and **bumps** the scalar `mass` row `(1,)` → `(N+1,)`, grouping
the moment rows under the parent as the indexed family `model.mass[l]` (each
row closed by the concrete-level Galerkin `Resolve`).

`[l]` is **moment access**, uniformly: `model.mass[l]` is the `l`-th MOMENT
row, never a term.  An additive TERM of a row is reached through the separate
`.term` accessor — `model.mass[0].term[i]` (single) / `.term[[i, j]]` (group).
This holds even after a closure collapses the family to a single mode: `[l]`
stays moment, so `model.mass` remains a `MomentFamily`.
""")

code(r"""
l = sp.Symbol("l", integer=True, nonnegative=True)
kbc = kinematic_modal_closure(
    model, u_field=a, w_field=a_w, h=h, b=b,
    basis_cls=Legendre_shifted, n_u=N)
dt_b_zero = Substitution({sp.Derivative(b, t): 0})

# SHAPE BUMP: mass (1,) → (N+1,) — model.mass[l] is the l-th continuity moment.
resolve_modes(model.mass, index=l, modes=range(N + 1),
              test_weight=c(zeta) * phi(l, zeta),
              basis_cls=Legendre_shifted, level=basis_lvl, var=zeta)
model.mass.describe(strip_args=True)
""")

# ── §6 — KBC + higher-w closure + mean continuity ──────────────────────────
md(r"""
## §6 — KBC closure + mass-driven higher-mode w closure

The two kinematic BCs (surface `ζ=1`, bed `ζ=0`) close the lower w-modes
`aw_0, aw_1`.  The **higher continuity moments** `model.mass[1..N]` (with
`∂_t b = 0`) close the upper w-modes `aw_2..aw_{N+1}` — driven straight off the
resolved moment family, no hand-added rows.  The mean row `model.mass[0]`
absorbs the KBCs and closes **exactly** to `∂_t h + ∂_x q_0`.
""")

code(r"""
# Higher-mode w closure off the resolved mass moments mass[1..N].
higher_residuals, higher_w = [], [a_w(k, t, x) for k in range(2, N + 2)]
for k in range(1, N + 1):
    eq = model.mass[k]
    eq.apply(dt_b_zero); eq.apply(Simplify())
    higher_residuals.append(eq.expr)
higher_w_closure = Substitution(
    sp.solve(higher_residuals, higher_w, dict=True)[0],
    name="higher_mode_w_closure")

# Mean continuity row: KBC absorption → ∂_t b = 0 → simplify.
# `[l]` is MOMENT-uniform: model.mass[0] is the FULL moment row (NOT a term).
model.mass[0].apply(kbc)
model.mass[0].apply(dt_b_zero)
model.mass[0].apply(Simplify())
# Collapse the moment family to the single surviving K&T continuity moment:
# drop the higher moments, keep mass[0]; model.mass STAYS a MomentFamily.
for k in range(1, N + 1):
    model._remove_equation(f"mass_{k}")
model._collapse_moment_family("mass", keep=[0])
model.mass.describe(strip_args=True)
""")

# ── §7 — momentum moment family via resolve_modes ──────────────────────────
md(r"""
## §7 — Momentum moment family via `resolve_modes`

`resolve_modes` bumps the x-momentum row `(2,)` → `(2, N+1)`: the moment axis
is appended to the component axis, so `model.momentum.x[l]` ≡
`model.momentum["x", l]` is the `(x, l)` slice.  Each row is closed by the
Galerkin `Resolve`, then the KBC + higher-w closure + `∂_t b = 0` + the
diagonal Legendre mass-matrix inversion `×(2k+1)` (canonical `M = I`).  The mean
row carries the clean `∂_t(h·a_0)`; the higher rows use the mass relation to
cancel the stray `∂_t h`.
""")

code(r"""
# SHAPE BUMP: momentum (2,) → (2, N+1) — model.momentum.x[l] is the (x,l) slice.
resolve_modes(model.momentum.x, index=l, modes=range(n_modes),
              test_weight=c(zeta) * phi(l, zeta),
              basis_cls=Legendre_shifted, level=basis_lvl, var=zeta)
for k in range(n_modes):
    eq = model.momentum.x[k]
    eq.apply(kbc)
    eq.apply(higher_w_closure)
    eq.apply(dt_b_zero)
    if k >= 1:
        eq.apply(mass_relation(h, a))
    eq.expr = (2 * k + 1) * eq.expr
    eq.apply(Simplify())
model.momentum.x[0].describe(strip_args=True)
""")

# ── §8 — CoV + NCP-preserving conservative fold ───────────────────────────
md(r"""
## §8 — Conserved variables `a → q/h` + NCP-preserving conservative fold

`ChangeOfVariables` swaps the modal coefficients `a_i → q_i / h` (conserved
momentum modes) and the `Q` family `a → q`.  The Resolve / CoV chain leaves the
spatial part fully **expanded** (`2 q_k/h·∂_x q_k − q_k²/h²·∂_x h + …`);
`fold_to_conservative_form` re-bundles it **exactly as K&T (4.17)**:

* the conservative **flux** + **hydrostatic-pressure** divergences fold into
  $\partial_x(F)$ / $\partial_x(g h^2/2)$ units;
* the genuinely **non-conservative** couplings stay **UNFOLDED** — the bed
  coupling $g\,h\,\partial_x b$ and the cross-mode $q_i/h\,\partial_x q_j$
  terms — because **SME is non-conservative** ($B \neq 0$).

This is the crucial difference from a greedy homotopy fold (which would fold the
bed + cross-mode couplings into the flux and report $B = 0$): the bundling must
match the way `SystemModel.from_model` will type each term.
""")

code(r"""
q = sp.Function("q")
model.apply(ChangeOfVariables("a", "q", lambda qi: qi / h))
for eq in model._equations.values():
    eq.simplify()

flux_fields = [q(k, t, x) for k in range(n_modes)]
dt_h_to_mass = {sp.Derivative(h, t): -sp.Derivative(q(0, t, x), x)}
for name, eq in model._equations.items():
    expr = sp.expand(eq.expr.doit())
    if name.startswith("momentum_x_") and not name.endswith("_0"):
        # cancel the σ-metric stray ∂_t h with the conserved mass relation
        expr = sp.expand(expr.xreplace(dt_h_to_mass))
    if name.startswith("momentum_x_"):
        expr = fold_to_conservative_form(
            expr, flux_fields, h=h, b=b, x=x, gravity_param=g)
    eq.expr = expr
# Undecorate the viscous-stress head (physical aux field).
for eq in model._equations.values():
    eq.expr = eq.expr.replace(
        lambda e: (isinstance(e, sp.Function)
                   and getattr(e.func, "_pde_decorated_from", None) == "tau_xz"),
        lambda e: sp.Function("tau_xz", real=True)(*e.args))
model.describe(strip_args=True)
""")

# ── §9 — the two K&T rows ─────────────────────────────────────────────────
md(r"""
## §9 — The K&T (4.17) rows

The threaded model now carries the K&T mass and mean-momentum rows in
conservative form.
""")

code(r"""
from zoomy_core.misc.description import Description
# `model.mass` is a MomentFamily (one surviving moment) — read the moment row
# via `[0]`; `[l]` is moment access, `.expr` lives on the moment Equation.
Description(
    "**mass:** $" + sp.latex(model.mass[0].expr) + "$\n\n"
    "**momentum_x_0:** $" + sp.latex(model.momentum_x_0.expr) + "$")
""")

# ── §10 — SystemModel transition ──────────────────────────────────────────
md(r"""
## §10 — `SystemModel.from_model(model, Q)`

`Q` is supplied **here**, at the transition — one entry per evolution row
(`b`, `h`, `q_0`, `q_1`, `q_2`).  The structural extractor reads the operator
matrices off the untagged residuals and validates `Q ∪ Qaux` covers every
field (a deliberately-incomplete `Q` raises, naming the uncovered fields).
""")

code(r"""
q0, q1, q2 = q(0, t, x), q(1, t, x), q(2, t, x)

# A deliberately-incomplete Q raises (good):
try:
    SystemModel.from_model(model, Q=[b, h], Qaux=[])
    _msg = "no error raised"
except ValueError as e:
    _msg = str(e)
Description("**Validation raised (good):**\n\n```\n" + _msg + "\n```")
""")

code(r"""
sm = SystemModel.from_model(model, Q=[b, h, q0, q1, q2])
sm.describe(full=True)
""")

# ── §10b — the F / P / B / S decomposition (SME is NON-conservative) ───────
md(r"""
### §10b — The operator split: $F$ / $P$ / $B$ / $S$

`SystemModel.from_model` feeds each (untagged) residual through the **same term
classifier production uses**, splitting

$$
\partial_t Q + \partial_x F(Q) + \partial_x P(Q)
   + \sum_j B(Q)_{:,j}\,\partial_x Q_j - S(Q) = 0 .
$$

SME is **genuinely non-conservative**: $B \neq 0$.  The defining couplings are
the bed gravity $B[q_0, b] = g h$, the depth transport $B[h, q_0] = 1$, and the
higher-mode cross terms — they must **not** be folded into the advective flux.
""")

code(r"""
from zoomy_core.misc.description import Description

def _split_report(sm):
    lines = ["| row | flux $F$ | pressure $P$ | source $S$ |",
             "|---|---|---|---|"]
    for i, s in enumerate(sm.state):
        # The open-stress aux ``tau_xz`` integrates to zero once it is exposed
        # as a constant aux Symbol — simplify the source so the empty open-SME
        # source reads as ``0`` (the stress is an auxiliary field).
        lines.append(
            f"| `{s}` | $" + sp.latex(sp.nsimplify(sm.flux[i, 0])) + "$ | $"
            + sp.latex(sm.hydrostatic_pressure[i, 0]) + "$ | $"
            + sp.latex(sp.simplify(sm.source[i, 0])) + "$ |")
    ncp = ["", "**non-conservative matrix** $B$ (non-zero entries):", ""]
    any_b = False
    for i in range(sm.n_equations):
        for j in range(sm.n_state):
            e = sm.nonconservative_matrix[i, j, 0]
            if e != 0:
                any_b = True
                ncp.append(
                    f"- $B[\\,{sp.latex(sm.state[i])},\\,"
                    f"{sp.latex(sm.state[j])}\\,] = " + sp.latex(e) + "$")
    if not any_b:
        ncp.append("- (all zero — would be the BUG)")
    return Description("\n".join(lines + ncp))

_split_report(sm)
""")

# ── §11 — base SME (open) vs SlipSME (closed) via the class API ────────────
md(r"""
## §11 — `SME` (open stress) vs `SlipSME` (slip-Newton closed)

The hand-threaded pipeline above is packaged as the composable
`SME` / `SlipSME` classes (`zoomy_core.derivation.models`).  The **base** `SME`
leaves the constitutive viscous stress **OPEN** — the momentum rows carry the
unresolved boundary atoms $\tau_{xz}(\sigma=0)$, $\tau_{xz}(\sigma=1)$ and the
moment integrals $\int \hat z\,\partial_{\hat z}\tau_{xz}\,d\hat z$, so
$\tau_{xz}$ appears as a **Qaux** field of the lifted `SystemModel`.

`SlipSME` **inherits** `SME` and only adds the slip-Newton closure
(`super().build()` → insert the three $\tau$ laws by `Substitution`, resolve the
$\int\dots d\hat z$ integrals).  After the closure the momentum rows carry
algebraic $\nu,\lambda$ friction and **no** stress atoms — there is no aux
stress field.
""")

code(r"""
from zoomy_core.derivation.models import SME, SlipSME

# Base model: constitutive stress OPEN.
open_model, octx = SME(N=2).build()
ob, oh = octx["b"], octx["h"]
sm_open = SystemModel.from_model(open_model, Q=[ob, oh] + octx["q_modes"])
sm_open.describe(full=True)
""")

code(r"""
# Derived model: SlipSME inherits SME and inserts the slip-Newton closure.
slip_model, sctx = SlipSME(
    N=2, parameters={"g": 9.81, "rho": 1.0, "nu": 1e-3, "lambda": 1e-2}
).build()
sb, sh = sctx["b"], sctx["h"]
sm_slip = SystemModel.from_model(slip_model, Q=[sb, sh] + sctx["q_modes"])
sm_slip.describe(full=True)
""")

code(r"""
# Side by side: the OPEN momentum_x_1 keeps τ_xz + ∫…dẑ; the CLOSED row carries
# algebraic ν/λ friction and NO stress atoms.
from zoomy_core.misc.description import Description
open_row = open_model.momentum_x_1.expr
slip_row = slip_model.momentum_x_1.expr
Description(
    "**open `momentum_x_1`** (stress unresolved):\n\n$"
    + sp.latex(open_row) + "$\n\n"
    "**slip-closed `momentum_x_1`** (ν, λ friction):\n\n$"
    + sp.latex(slip_row) + "$\n\n"
    "open τ atoms: "
    + str([str(a) for a in open_row.atoms(sp.Function) if "tau" in str(a.func)])
    + " · open ∫ atoms: " + str(len(open_row.atoms(sp.Integral)))
    + "  |  closed τ atoms: "
    + str([str(a) for a in slip_row.atoms(sp.Function) if "tau" in str(a.func)])
    + " · closed ∫ atoms: " + str(len(slip_row.atoms(sp.Integral))))
""")

# ── §12 — NewtonianSME extension, step by step ────────────────────────────
md(r"""
## §12 — Extension: `NewtonianSME` — the normal stress $\tau_{xx}$ as a *derived* diffusive flux

The base SME drops the horizontal extensional stress $\tau_{xx}\to 0$.
`NewtonianSME` **inherits** `SlipSME` and **keeps** it, closing it with Newton's
normal-stress law

$$
\tau_{xx} = 2\mu\,\partial_x u = 2\rho\nu\,\partial_x u,\qquad \mu=\rho\nu .
$$

Its momentum contribution $\tfrac1\rho\partial_x\tau_{xx}=2\nu\,\partial_{xx}u$ is
**second order** — a *diffusive* flux.  But it is **not** hand-written: it is
carried GENUINELY through the σ-map ($\partial_x\to D_x=\partial_x-m\,\partial_\zeta$,
$m=(\partial_x b+\zeta\,\partial_x h)/h$), the modal ansatz, and a **conservative
Galerkin route** built only from operations.  The σ-transform splits it into two
physically distinct pieces:

* a genuine **second-order diffusive flux** $\partial_x(F^d)$ → the rank-4
  `diffusion_matrix` $A$ ($F^d=A(Q)\nabla Q$): a diagonal $A[q_k,q_k]=-2\nu$ plus
  σ-metric gradient couplings $A[q_k,h]$, $A[q_k,b]$ (these vanish for a flat
  geometry, leaving the pure $-2\nu$ eddy viscosity);
* **σ-metric corrections that are NOT of $A\nabla Q$ form** (bilinear gradients
  $\partial_x b\,\partial_x q$, depth curvature $q\,(\partial_x h)^2$, …) →
  genuine viscous **source** terms ($\propto\nu$, vanishing for
  $\partial_x b=\partial_x h=0$).  The earlier *hand-coded* closure silently
  **dropped** these; the derivation keeps them.

The steps below reproduce the mechanism explicitly.  The only genuinely NEW
piece is `ResolveIntegral`, together with a **slim abstract-$\int$ projection
op** (`zoomy_core.derivation.projection.Integrate`) — *distinct* from the §1b
`model.operations.Integrate`, which evaluates the integral immediately.
Everything else is the reused op surface; the derivation itself is just
`Multiply` → `ProductRule` → abstract `Integrate` → `ResolveIntegral`.
""")

# ── §12.1 — the new resolution primitives ─────────────────────────────────
md(r"""
### §12.1 — Deferring the ζ-integral: the new `ResolveIntegral`

The moment projection of §5/§7 (`resolve_modes`) is conceptually
`Multiply(test) ∘ Integrate ∘ resolve`, welded into one step.  §12 **splits**
the last two: a slim abstract-$\int$ op
(`zoomy_core.derivation.projection.Integrate`, *not* the §1b production
`Integrate`) leaves the $\zeta$-integral as an unevaluated `sp.Integral` atom
(commuting only the outermost conservative $\partial_x$ out), and the new
`ResolveIntegral` closes each integral by a chosen method — `basis` (substitute
the concrete shifted-Legendre polynomial), `ftc`
($\int_0^1\partial_\zeta g\,d\zeta=g|_1-g|_0$), or `numerical` — with `classify`
picking the method **per integral**.  (That abstract op is used inside
`project_conservative_diffusion` in §12.3; the cell below demos `ResolveIntegral`
directly.)
""")

code(r"""
from zoomy_core.derivation import ResolveIntegral
from zoomy_core.model.operations import Expression

# (a) FTC:  ∫₀¹ ∂_ζ g dζ = g(1) − g(0)
g_ = sp.Function("g", real=True)
ftc_in = sp.Integral(sp.Derivative(g_(zeta), zeta), (zeta, 0, 1))
ftc_out = Expression(ftc_in, "").apply(ResolveIntegral(var=zeta, method="ftc")).expr

# (b) basis (Galerkin Gram):  ∫₀¹ c·φ₂·φ₂ dζ = 1/(2·2+1) = 1/5
gram_in = sp.Integral(c(zeta) * phi(2, zeta) * phi(2, zeta), (zeta, 0, 1))
gram_out = Expression(gram_in, "").apply(
    ResolveIntegral(var=zeta, method="basis",
                    basis_cls=Legendre_shifted, level=3)).expr

Description(
    "**`ResolveIntegral` — one abstract $\\int$, method per integral:**\n\n"
    "- **ftc**: $" + sp.latex(ftc_in) + " = " + sp.latex(ftc_out) + "$\n"
    "- **basis**: $" + sp.latex(gram_in) + " = " + sp.latex(gram_out)
    + "$  (shifted-Legendre Gram $\\delta_{kl}/(2l+1)$)")
""")

# ── §12.2 — inject τ_xx, carry to the pre-resolve momentum row ─────────────
md(r"""
### §12.2 — Inject $\tau_{xx}$ and carry it to the pre-resolve momentum row

Re-thread the SME pipeline of §1–§4 on a **fresh** model, changing exactly ONE
line: the stress reduction substitutes Newton's normal stress
$\tau_{xx}=2\rho\nu\,\partial_x u$ instead of $\tau_{xx}\to0$.  After the σ-map,
`Multiply(h)` and the modal ansatz, the momentum row carries the σ-mapped
$-2\nu h\,D_x^2\tilde u$ — a handful of viscous ($\nu$) terms among the
hyperbolic ones.
""")

code(r"""
nm = Model(coords=(t, x, z),
           parameters={"g": 9.81, "rho": 1.0, "nu": 1e-3, "lambda": 1e-2})
ng, nrho, nnu = nm.parameters.g, nm.parameters.rho, nm.parameters.nu
nm.Q = [h, u, w, p]
nm.add_equation("bottom", d.t(b))
nm.add_equation("mass", d.x(u) + d.z(w))
nm.add_equation("momentum", (2,), [
    d.t(u) + d.x(u * u) + d.z(u * w) + d.x(p) / nrho - (d.x(txx) + d.z(txz)) / nrho,
    d.t(w) + d.x(u * w) + d.z(w * w) + d.z(p) / nrho + ng - (d.x(tzx) + d.z(tzz)) / nrho,
])
# ★ THE ONLY CHANGE vs the base SME: keep τ_xx as Newton's normal stress.
nm.apply(Substitution({txx: 2 * nrho * nnu * d.x(u), tzz: 0, tzx: txz}))
# hydrostatic pressure (as §1b)
nm.momentum.z.apply(Substitution({d.t(w): 0, d.x(u * w): 0, d.z(w * w): 0, d.x(txz): 0}))
nm.momentum.z.apply(Integrate(z, z, b + h, method="analytical"))
nm.momentum.z.apply(Substitution({p.subs(z, b + h): 0}))
nm.momentum.x.apply(nm.momentum.z.solve_for(p)); nm.momentum.z.remove()
nm.momentum.x.apply(Simplify())
# σ-map, conservative pre-fold, modal ansatz, bind N=2 (as §2–§4)
nm.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))
nm.mass.apply(Multiply(h)); nm.momentum.x.apply(Multiply(h))
reset_modal_indices(nm)
nm.apply(separation_of_variables(u, a(t, x), basis, N_u))
nm.apply(separation_of_variables(w, a_w(t, x), basis, N_u + 1))
nm.apply(Substitution({N_u: N}))
for eq in nm._equations.values():
    eq.expr = eq.expr.replace(lambda e: isinstance(e, sp.Sum), lambda e: e.doit())

visc_terms = [tm.expr for tm in nm.momentum.x.term if tm.expr.has(nnu)]
Description(
    "**viscous terms in `momentum.x`** — the σ-mapped $-2\\nu h\\,D_x^2\\tilde u$ "
    f"({len(visc_terms)} of {len(nm.momentum.x.term)} additive terms):\n\n"
    + "\n".join(f"- $" + sp.latex(tm) + "$" for tm in visc_terms))
""")

# ── §12.3 — the conservative Galerkin route on the viscous block ───────────
md(r"""
### §12.3 — Conservative Galerkin route on the viscous block

For the test mode $k=0$, `project_conservative_diffusion` runs the viscous block
through `Multiply(c\,\phi_0)` → `ProductRule()` (move $\phi_0 h$ INTO $\partial_x$)
→ abstract `Integrate` → `ResolveIntegral(basis)`.  The bare-outer-$\partial_x$
flux stays a conservative $\partial_x(F^d_0)$ (→ `diffusion_matrix`); the σ-metric
remainder resolves in place (→ `source`).
""")

code(r"""
from zoomy_core.derivation import (
    project_conservative_diffusion, is_conservative_diffusion)

k = 0
tw = c(zeta) * phi(k, zeta)
visc = sum(visc_terms, sp.S.Zero)
visc_res = project_conservative_diffusion(
    visc, tw, basis_cls=Legendre_shifted, level=basis_lvl, var=zeta, x=x)

cons = sum((tm for tm in sp.Add.make_args(visc_res)
            if is_conservative_diffusion(tm, x)), sp.S.Zero)
resid = sp.expand(visc_res - cons)
Description(
    "**conservative diffusive flux** $\\partial_x(F^d_0)$  (→ `diffusion_matrix`):"
    "\n\n$" + sp.latex(cons) + "$\n\n"
    "**σ-metric residual**  (→ `source`):\n\n$" + sp.latex(resid) + "$")
""")

# ── §12.4 — anti-cheat: identical to the welded in-place Resolve ───────────
md(r"""
### §12.4 — Anti-cheat: the conservative route **=** the naive in-place `Resolve`

Nothing about the answer is hand-written.  Resolving the SAME viscous block with
the welded in-place `Resolve` (which integrates $\partial_x$ in place, scattering
the second order into source) gives a bit-identical result once the conservative
$\partial_x(F^d)$ is expanded — the two differ ONLY in representation.
""")

code(r"""
from zoomy_core.derivation import Resolve

naive = sp.expand(Expression(visc, "").apply(
    Resolve(tw, Legendre_shifted, level=basis_lvl, var=zeta)).expr)
delta = sp.simplify(sp.expand((visc_res - naive).doit()))
Description(
    "manual conservative route $-$ naive in-place `Resolve` $= "
    + sp.latex(delta) + "$  →  **"
    + ("identical" if delta == 0 else "DIFFERS") + "**")
""")

# ── §12.5 — packaged NewtonianSME → diffusion_matrix ───────────────────────
md(r"""
### §12.5 — Packaged `NewtonianSME` → the `diffusion_matrix`

The same route is wired into `NewtonianSME._momentum_resolver`, so building the
class and lifting to a `SystemModel` types the second-order flux into the rank-4
`diffusion_matrix` $A$ — a diagonal $A[q_k,q_k]=-2\nu$ plus the σ-metric gradient
couplings to $h$ and $b$.
""")

code(r"""
from zoomy_core.derivation.models import NewtonianSME

newt_model, nctx = NewtonianSME(
    N=2, parameters={"g": 9.81, "rho": 1.0, "nu": 1e-3, "lambda": 1e-2}
).build()
sm_newt = SystemModel.from_model(
    newt_model, Q=[nctx["b"], nctx["h"]] + nctx["q_modes"])

A = sm_newt.diffusion_matrix
lines = ["**`diffusion_matrix` $A$** — shape "
         f"`{tuple(A.shape)}` = `(n_eq, n_state, n_dim, n_dim)`, "
         "$F^d = A(Q)\\,\\partial_x Q$:", ""]
for i in range(sm_newt.n_equations):
    for j in range(sm_newt.n_state):
        e = A[i, j, 0, 0]
        if e != 0:
            lines.append(
                f"- $A[\\,{sp.latex(sm_newt.state[i])},\\,"
                f"{sp.latex(sm_newt.state[j])}\\,] = " + sp.latex(e) + "$")
Description("\n".join(lines))
""")

# ── §12.6 — invariance: hyperbolic part unchanged, source purely viscous ───
md(r"""
### §12.6 — The normal stress is *purely viscous*

Against the slip SME of §11, the Newtonian closure changes **only** the
diffusion matrix and the viscous part of the source: the advective flux $F$,
hydrostatic pressure $P$, NCP $B$ and mass matrix $M$ are **exactly** unchanged,
and every differing `source` term carries $\nu$ (vanishing at $\nu\to0$).
""")

code(r"""
def _is_zero(e):
    return sp.cancel(sp.sympify(e)) == 0

hyp_ok = all(
    _is_zero(sm_newt.flux[i, 0] - sm_slip.flux[i, 0])
    and _is_zero(sm_newt.hydrostatic_pressure[i, 0]
                 - sm_slip.hydrostatic_pressure[i, 0])
    and all(_is_zero(sm_newt.nonconservative_matrix[i, j, 0]
                     - sm_slip.nonconservative_matrix[i, j, 0])
            for j in range(sm_newt.n_state))
    and all(_is_zero(sm_newt.mass_matrix[i, j] - sm_slip.mass_matrix[i, j])
            for j in range(sm_newt.n_state))
    for i in range(sm_newt.n_equations))

nu_sym = sm_newt.parameters.nu
src_purely_viscous = True
for i in range(sm_newt.n_equations):
    diff = sp.expand(sp.sympify(sm_newt.source[i, 0])
                     - sp.sympify(sm_slip.source[i, 0]))
    if diff == 0:
        continue
    if any(not term.has(nu_sym) for term in sp.Add.make_args(diff)):
        src_purely_viscous = False
    if sp.cancel(diff.subs(nu_sym, 0)) != 0:
        src_purely_viscous = False

Description(
    "- advective $F$ / pressure $P$ / NCP $B$ / mass $M$ unchanged vs SlipSME: "
    + ("**yes**" if hyp_ok else "**NO**") + "\n"
    "- every `source` change carries $\\nu$ and vanishes at $\\nu\\to0$: "
    + ("**yes**" if src_purely_viscous else "**NO**"))
""")

# ── write notebook ────────────────────────────────────────────────────────


def _nb():
    cells = []
    for kind, src in CELLS:
        if kind == "markdown":
            cells.append({
                "cell_type": "markdown", "metadata": {},
                "source": src.splitlines(keepends=True),
            })
        else:
            cells.append({
                "cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": src.splitlines(keepends=True),
            })
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "sme_kt19.ipynb")
    with open(out, "w") as f:
        json.dump(_nb(), f, indent=1)
    print(f"wrote {out} ({len(CELLS)} cells)")
