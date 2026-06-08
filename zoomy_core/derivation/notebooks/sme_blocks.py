# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Shallow Moment Equations — single-model derivation (mass · w · momentum)
#
# One model `m`, defined once in `(t, x, z)`, then transformed and reduced in
# place: continuity is duplicated into a `w` row before `mass` is consumed into
# the `h`-evolution, the σ-map is applied once to the whole model, and `ω` is
# added at the end.

# %%
import sympy as sp
from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.derivation import (
    Model, PDETransformation, Simplify, ResolveIntegral, SolveFor, Sort, Basis,
    Split, Consolidate, ExpandSums, EvaluateSums, PullConstants, ExtractBrackets,
    AutoTag, SortByTag, ResolveModes, ResolveBasis,
    separation_of_variables, reset_modal_indices, modal_bound)
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ        # vertical (pressure) integral
from zoomy_core.derivation.projection import Integrate                 # abstract ζ-integral

t, x, z = C.t, C.x, C.z
zeta = sp.Symbol("zeta", real=True)

# %% [markdown]
# ## 1 — full system in (t, x, z)

# %%
m = Model(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
g, rho = m.parameters.g, m.parameters.rho
u = sp.Function("u", real=True)(t, x, z); w = sp.Function("w", real=True)(t, x, z)
p = sp.Function("p", real=True)(t, x, z); h = sp.Function("h", positive=True)(t, x)
b = sp.Function("b", real=True)(t, x); txz = sp.Function("tau_xz", real=True)(t, x, z)
m.Q = [h, u, w, p]
m.add_equation("bottom", d.t(b))
m.add_equation("mass", d.x(u) + d.z(w))
m.add_equation("momentum", (2,), [
    d.t(u) + d.x(u*u) + d.z(u*w) + d.x(p)/rho - d.z(txz)/rho,
    d.t(w) + d.x(u*w) + d.z(w*w) + d.z(p)/rho + g])
m.add_equation("kbc_top", KinematicBC(w=w, u=u, interface=b + h))   # physical KinematicBCs
m.add_equation("kbc_bot", KinematicBC(w=w, u=u, interface=b))
m.add_equation("w", m.mass, group="aux")    # duplicate continuity — `mass` becomes the h-eq
m.describe(show="all")

# %% [markdown]
# ## 2 — reduce z-momentum → hydrostatic pressure, eliminate p  (physical z)

# %%
m.momentum.z.apply({d.t(w): 0, d.x(u*w): 0, d.z(w*w): 0})
m.momentum.z.apply(IntegrateZ(z, z, b + h, method="analytical"))
m.momentum.z.apply({p.subs(z, b + h): 0})
m.momentum.x.apply(m.momentum.z.solve_for(p)); m.momentum.z.remove()
m.momentum.x.apply(Simplify())
m.momentum.x.describe()

# %% [markdown]
# ## 3 — σ-map the whole model (equations AND KBCs):  z = b + h·ζ
# Now `m.Q.u → ũ`, `m.Q.w → w̃`; `b → ζ=0`, `b+h → ζ=1`.

# %%
m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))
m.describe(show="all")

# %% [markdown]
# ## 4 — mass → evolution equation for h

# %%
m.mass.apply(Multiply(h))
m.mass.apply(ProductRule(variables=[zeta]))
m.mass.term[1].apply(ProductRule())
m.mass.apply(Integrate(zeta, bounds=(0, 1)))
m.mass.apply(ResolveIntegral())
m.mass.apply(m.kbc_top); m.mass.apply(m.kbc_bot)     # σ-mapped KinematicBCs
m.mass.apply({sp.Derivative(b, t): 0})
m.mass.apply(Simplify())
m.mass.apply(Sort())                                 # order: ∂_t … , flux …
m.mass.describe()

# %% [markdown]
# ## 5 — continuity (the `w` row) → vertical velocity w̃(ζ)

# %%
m.w.apply(Multiply(h))
m.w.apply(ProductRule(variables=[zeta]))
m.w.term[1].apply(ProductRule())
m.w.apply(Integrate(zeta, bounds=(0, zeta)))         # running ∫₀^ζ … d\hat{ζ}
m.w.apply(ResolveIntegral())
m.w.apply(m.kbc_bot)
m.w.apply(Simplify())
m.w.apply({sp.Derivative(b, t): 0})
m.w.apply(SolveFor(m.Q.w))                           # w̃(ζ) = …
m.remove("kbc_top"); m.remove("kbc_bot")             # KBCs consumed (mass §4, w §5)
m.w.describe()

# %% [markdown]
# ## 6 — x-momentum → hydrostatic flux + vertical-coupling operator ω

# %%
m.momentum.x.apply(Multiply(h))
m.momentum.x.term[[3, 4]].apply(ProductRule(variables=[zeta]))
ut, wt = m.Q.u, m.Q.w                                # ũ(t,x,ζ), w̃(t,x,ζ)

# vertical-coupling operator:  h·ω ≡ w̃ − ∂_t(ζh+b) − ũ·∂_x(ζh+b)
omega = sp.Function("omega", real=True)(t, x, zeta)
m.add_equation("omega", h*omega - (wt - d.t(zeta*h + b) - ut*d.x(zeta*h + b)),
               group="aux")
m.momentum.x.apply(m.omega.solve_for(wt))            # substitute w̃ into x-momentum
m.omega.apply(SolveFor(omega))                       # reorient the omega row → ω = (…)/h
m.momentum.x.apply(Simplify())

# hydrostatic pressure: Consolidate folds the self-product gh·∂_x h → ∂_x(g h²/2)
m.momentum.x.apply(Consolidate())
m.momentum.x.apply(Sort())                            # ∂_t · flux · NCP · source
m.momentum.x.describe(show_tags=True)

# %% [markdown]
# ## 7 — make ω explicit
# a) substitute `∂_t h` by the mass balance, b) substitute `w̃` by the
# w-reconstruction.  The `ũ·∂_x(ζh+b)` terms cancel and ω vanishes at both
# ends — `ω|_{ζ=0}=ω|_{ζ=1}=0` — which is what kills its boundary trace below.

# %%
m.omega.apply(m.w)                                    # b) w̃ → reconstruction
m.omega.apply(Simplify())                             # split ∂_t(ζh+b); tidy
m.omega.apply({sp.Derivative(b, t): 0})               # fixed bed
m.omega.apply(m.mass.solve_for(sp.Derivative(h, t)))  # a) ∂_t h = −∂_x(h⟨ũ⟩)
m.omega.apply(SolveFor(omega))                        # re-isolate ω
m.omega.describe()

# %% [markdown]
# ## 8 — Galerkin moment projection:  ∫₀¹ c·φ_k · (x-momentum) dζ
# Project onto the test mode `c·φ_k`.  Only the two ∂_ζ terms (`h∂_ζ(ũω)`,
# `∂_ζτ̃`) integrate by parts; the ω boundary trace drops (ω=0 at both ends).
# `Consolidate` then re-folds the IBP split `τ̃c'φ_k + τ̃cφ_k' → τ̃ ∂_ζ(cφ_k)`
# while `τ̃` is still opaque (passive factor carries no derivative), so the
# diffusion stays ONE term downstream.

# %%
basis = Basis(symbol="phi", weight="c"); c = basis.weight
k = sp.Symbol("k", integer=True, nonnegative=True); phi_k = basis.phi(k, zeta)

mx = m.momentum.x
mx.apply(Multiply(c(zeta) * phi_k))                   # c) × test mode c·φ_k
mx.apply(ProductRule(variables=[zeta]))               # e) IBP the two ∂_ζ terms
mx.apply(Integrate(zeta, bounds=(0, 1)))              # d) ∫₀¹ … dζ
mx.apply(ResolveIntegral())                           # f) FTC → boundary traces
mx.apply({omega.subs(zeta, 0): 0, omega.subs(zeta, 1): 0})   # ω|_{ζ=0,1}=0
mx.apply(Consolidate())                               # re-fold ∂_ζ(cφ_k) split
mx.apply(Sort())
mx.describe(show_tags=True)

# %% [markdown]
# ## 9 — close (Newtonian + slip) and insert the modal ansatz
# Close the stress with a **Newtonian + Navier-slip** law and insert the
# **separation ansatz** `ũ = Σ_i a_i(t,x) φ_i(ζ)`.  Field access goes through
# `m.functions` — no private state, no `.func`: `m.functions.tau_xz.at(1)` is
# the surface value, `.expr` the bulk field, `.head` the decorated head.  Both
# the boundary BCs and the bulk Newtonian law are plain `.apply({…})`
# substitutions (the bulk one is a function definition `τ̃(…,ζ)=…`, so `.apply`
# rewrites `τ̃` at every argument).

# %%
nu = sp.Symbol("nu", positive=True)            # kinematic viscosity
lam = sp.Symbol("lambda_s", positive=True)     # Navier slip length
a = sp.Function("a", real=True)                # modal coefficients a_i(t,x)
tau, uu = m.functions.tau_xz, m.functions.u    # field handles (no private state)

# Newtonian + slip closure for τ̃:
#   surface  τ̃(t,x,1) = 0            (stress-free)
#   bottom   τ̃(t,x,0) = −λ_s ũ(0)    (Navier slip)
#   bulk     τ̃(t,x,ζ) = (ρν/h) ∂_ζ ũ (Newtonian; σ-map ∂_z = (1/h)∂_ζ)
mx = m.momentum.x
mx.apply({tau.at(1): 0, tau.at(0): -lam * uu.at(0)})          # slip BCs (exact)
mx.apply({tau.expr: rho * nu / h * sp.Derivative(uu.expr, zeta)})  # bulk Newtonian

# separation ansatz ũ = Σ_i a_i(t,x) φ_i(ζ) — whole model, so the auxiliary
# `w` reconstruction is expanded/normalised by the SAME pipeline too.
reset_modal_indices(m)
N_u = modal_bound("N_u")                       # abstract truncation bound
m.apply(separation_of_variables(u, a(t, x), basis, N_u))
mx.apply(ExpandSums())                         # ũ² → Σ_i Σ_j a_i a_j φ_i φ_j
mx.apply(Sort())
m.w.apply(ExpandSums())                        # aux: same treatment for w̃(ζ)
m.w.apply(PullConstants())                     #   pull a_i / ∂_x out of the ∫₀^ζ
m.w.apply(SolveFor(m.Q.w))                     #   re-orient: displayed w̃ = … is normalised
mx.describe(show_tags=True)

# %% [markdown]
# ## 10 — eliminate ω and name the brackets
# `m.omega` is the oriented relation `ω(t,x,ζ) = …` (a function definition), so a
# plain `.apply(m.omega)` eliminates the operator — it substitutes `ω` at the
# bracket's bound dummy and alpha-renames ω's own index/dummy (`i→j`,
# `\hat ζ→\hat ξ`) so the nested integral never captures the outer sum.
#
# `ExtractBrackets` is the SHARP gate: internally it first runs `PullConstants`
# (hoist every ζ-independent factor — `a_j`, `Σ_j`, `∂_x a_j` — out of each
# `∫_ζ` / `∂_ζ`), then NAMES only the integrals whose body is now purely
# ζ-dependent (`Gram` / `Weight` / `⟨…⟩`).  A `∫₀¹` whose integrand still carries
# a `t,x`-factor is NOT a bracket and stays a plain `∫` — the `⟨…⟩` notation
# never lies about a term that is not yet a pure number.

# %%
mx.apply(m.omega)                              # ω(ζ) → reconstruction, hygienic
m.remove("omega")                              # ω consumed — drop the aux definition
mx.apply(ExpandSums())
mx.apply(ExtractBrackets(basis, var=zeta))     # PullConstants + name pure-ζ ⟨…⟩
mx.apply(Consolidate())                        # fold ∂_x(h a_j); push ⟨…⟩ into the
                                               #   pressure flux → ∂_x(g h²⟨c,φ_k⟩/2)
mx.apply(AutoTag())                            # physics tags: ω-coupling/bed-slope = NCP
for tm in mx.terms:                            # manually re-tag the pressure flux (the
    if tm.tag == "flux" and tm.expr.has(g) and not tm.expr.has(a):  # g h² flux, no a_i)
        tm.tag = "pressure_flux"
mx.apply(SortByTag())                          # order: ∂_t · flux · pressure · NCP · source
mx.describe(show_tags=True)

# %% [markdown]
# ## 11 — apply the basis (shifted Legendre) and build the explicit N=2 system
# Three structural steps promote the abstract row to a concrete vector:
#
# 1. truncate `N=2` and `EvaluateSums` — resolve the finite modal sums `Σ_i, Σ_j`;
# 2. `ResolveModes` — specialise the abstract test mode `k → 0,1,2`, promoting
#    the scalar `m.momentum.x` to the moment family `m.momentum.x[k]` (a vector);
# 3. `ResolveBasis(legendre)` — evaluate EVERY Galerkin bracket (named
#    Gram/Weight, opaque `⟨…⟩`, the nested ω-coupling double integrals, and the
#    `φ_i(0)` boundary terms) to a NUMBER against the shifted-Legendre basis —
#    fast antiderivative + per-instance cache (no `sympy.integrate` on opaque φ).

# %%
from zoomy_core.model.models.basisfunctions import Legendre_shifted

legendre = Legendre_shifted(level=2)               # concrete shifted-Legendre basis

m.momentum.x.apply({N_u: 2})                       # truncate: N = 2  (a_0, a_1, a_2)
m.momentum.x.apply(EvaluateSums())                 # resolve the finite Σ_i, Σ_j
m.momentum.x.apply(ResolveModes(index=k, modes=range(3)))   # → vector m.momentum.x[k]
m.momentum.apply(ResolveBasis(legendre, var=zeta))          # every bracket → a number

m.momentum.describe()                              # the explicit N=2 x-momentum vector

# %% [markdown]
# ## Full model

# %%
m.describe(show="all")
