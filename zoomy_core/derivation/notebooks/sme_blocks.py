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
    Model, PDETransformation, Simplify, ResolveIntegral, SolveFor, Sort)
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
m.momentum.x.apply(Sort())                            # ∂_t · flux · NCP · source
m.momentum.x.describe(show_tags=True)

# %% [markdown]
# ## Full model

# %%
m.describe(show="all")
