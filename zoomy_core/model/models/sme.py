"""Shallow Moment Equations (Kowalski & Torrilhon 2019) — the single
canonical model, derived with the declarative
:mod:`zoomy_core.model.derivation` framework (moment-projected vertical
velocity route).

``SME`` is the ONLY model class in :mod:`zoomy_core.model.models`; everything
else is archived under ``legacy/``.  It extends the empty base
:class:`zoomy_core.model.basemodel.Model` for the parameter / identity surface,
but its derivation is the declarative pipeline (the `sme_wmoments` notebook):
full 3-D system → hydrostatic → σ-map → moment-project the mass balance (h-eq +
the ŵ closure) → project & close the x-momentum → insert the shifted-Legendre
basis → conservative CoV ``û_i → q_i/h``.  The vertical reconstruction is
registered into the ``interpolate`` function group with the ŵ closure inlined
(``w(ζ) = Σ_j ŵ_j(q, ∂_x q, ∂_x h, ∂_x b) φ_j(ζ)``), so ``interpolate_to_3d`` is
self-contained.

``SME(level=2).derive_model()`` builds the declarative model; ``.system_model``
returns the runtime :class:`~zoomy_core.systemmodel.SystemModel`.
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.basemodel import Model as BaseModel
from zoomy_core.model.derivation import (
    Model as DModel, PDETransformation, Simplify, ResolveIntegral, Basis,
    Consolidate, ExpandSums, EvaluateSums, PullConstants, ExtractBrackets,
    ResolveModes, ResolveBasis, SolveLinearSystem, ChangeOfVariables,
    separation_of_variables, reset_modal_indices, modal_bound,
)
from zoomy_core.model.derivation.projection import Integrate          # abstract ζ-integral
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ       # vertical (pressure) integral
from zoomy_core.systemmodel import SystemModel

t, x, z = C.t, C.x, C.z
zeta = sp.Symbol("zeta", real=True)


class SME(BaseModel):
    """Shallow Moment Equations, modal truncation order ``level`` (``N_u``).

    ``level=2`` matches K&T (4.17): 4 dynamic equations (h, q_0, q_1, q_2)
    plus the bed ``b``.
    """

    _finalize_lazy = True               # declarative path — skip the production tag pipeline
    level = param.Integer(default=2, bounds=(0, None))

    def derive_model(self):
        """Build the declarative SME model (stored as ``self.derivation``) and
        register the vertical reconstruction.  Called by the base ``__init__``."""
        Nu = int(self.level)
        # nu (kinematic viscosity) and lambda_s (Navier slip) are MODEL
        # PARAMETERS — default 0 (inviscid / free-slip), set them for friction.
        m = DModel(coords=(t, x, z),
                   parameters={"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0})
        g, rho = m.parameters.g, m.parameters.rho
        nu, lam = m.parameters.nu, m.parameters.lambda_s
        u = sp.Function("u", real=True)(t, x, z)
        w = sp.Function("w", real=True)(t, x, z)
        p = sp.Function("p", real=True)(t, x, z)
        h = sp.Function("h", positive=True)(t, x)
        b = sp.Function("b", real=True)(t, x)
        txz = sp.Function("tau_xz", real=True)(t, x, z)

        # 1 — full system in (t, x, z)
        m.Q = [h, u, w, p]
        m.add_equation("bottom", d.t(b))
        m.add_equation("mass", d.x(u) + d.z(w))
        m.add_equation("momentum", (2,), [
            d.t(u) + d.x(u * u) + d.z(u * w) + d.x(p) / rho - d.z(txz) / rho,
            d.t(w) + d.x(u * w) + d.z(w * w) + d.z(p) / rho + g])
        m.add_equation("kbc_top", KinematicBC(w=w, u=u, interface=b + h))
        m.add_equation("kbc_bot", KinematicBC(w=w, u=u, interface=b))

        # 2 — hydrostatic: z-momentum → eliminate p
        m.momentum.z.apply({d.t(w): 0, d.x(u * w): 0, d.z(w * w): 0})
        m.momentum.z.apply(IntegrateZ(z, z, b + h, method="analytical"))
        m.momentum.z.apply({p.subs(z, b + h): 0})
        m.momentum.x.apply(m.momentum.z.solve_for(p)); m.momentum.z.remove()
        m.momentum.x.apply(Simplify())

        # 3 — σ-map the whole model: z = b + h·ζ
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))

        basis = Basis(symbol="phi", weight="c"); c = basis.weight
        k = sp.Symbol("k", integer=True, nonnegative=True); phi_k = basis.phi(k, zeta)
        legendre = Legendre_shifted(level=Nu + 2)        # need φ_{N_u+2} for the top w-mode

        # 4 — moment-project the MASS balance + kinematic BCs (pre-SoV)
        m.mass.apply(Multiply(h))
        m.mass.apply(Multiply(c(zeta) * phi_k))
        m.mass.apply(ProductRule(variables=[zeta]))
        m.mass.apply(Integrate(zeta, bounds=(0, 1)))
        m.mass.apply(ResolveIntegral())
        m.mass.apply(m.kbc_bot); m.mass.apply(m.kbc_top)
        m.mass.apply({sp.Derivative(b, t): 0})
        m.mass.apply(Simplify())

        # 5 — project the X-MOMENTUM + close the stress (before SoV)
        mx = m.momentum.x
        mx.apply(Multiply(h))
        mx.apply(Multiply(c(zeta) * phi_k))
        mx.apply(ProductRule(variables=[zeta]))
        mx.apply(Integrate(zeta, bounds=(0, 1)))
        mx.apply(ResolveIntegral())
        mx.apply(m.kbc_bot); mx.apply(m.kbc_top); mx.apply({sp.Derivative(b, t): 0})
        tau, uu = m.functions.tau_xz, m.functions.u
        mx.apply({tau.at(1): 0, tau.at(0): -lam * uu.at(0)})              # stress BCs
        mx.apply({tau.expr: rho * nu / h * sp.Derivative(uu.expr, zeta)})  # bulk Newtonian
        mx.apply(Simplify())

        # 6 — separation of variables: u → û_i (N_u), w → ŵ_j (N_u + 1)
        uh = sp.Function(r"\hat{u}", real=True); wh = sp.Function(r"\hat{w}", real=True)
        reset_modal_indices(m)
        N_u = modal_bound("N_u")
        m.apply(separation_of_variables(u, uh(t, x), basis, N_u))
        m.apply(separation_of_variables(w, wh(t, x), basis, N_u + 1))

        # 7 — basis → h-equation (k=0) and the ŵ closure (k=1…N_u+2)
        m.mass.apply(ExpandSums())
        m.mass.apply(PullConstants())
        m.mass.apply(ExtractBrackets(basis, var=zeta))
        m.mass.apply({N_u: Nu})
        m.mass.apply(EvaluateSums())
        m.mass.apply(ResolveModes(index=k, modes=range(Nu + 3)))
        m.mass.apply(ResolveBasis(legendre, var=zeta))
        h_eq = m.mass[0].solve_for(d.t(h))                # ∂_t h = −∂_x(h û_0)
        for row in m.mass[1:Nu + 3]:
            row.apply(h_eq)
        w_closure = SolveLinearSystem(
            m.mass[1:Nu + 3], [wh(j, t, x) for j in range(Nu + 2)]).solve()
        for row in m.mass[1:Nu + 3]:                      # collapse the spent moment rows
            row.apply(w_closure)

        # 8 — substitute the ŵ closure into the x-momentum, resolve
        mx.apply(ExpandSums())
        mx.apply(PullConstants())
        mx.apply(ExtractBrackets(basis, var=zeta))
        mx.apply({N_u: Nu})
        mx.apply(EvaluateSums())
        mx.apply(w_closure)
        mx.apply(ResolveModes(index=k, modes=range(Nu + 1)))
        m.momentum.x.apply(ResolveBasis(legendre, var=zeta))

        # 9 — kill loose ∂_t h, consolidate the pressure, conservative CoV û→q/h
        for kk in range(Nu + 1):
            m.momentum.x[kk].apply(h_eq)
            m.momentum.x[kk].apply(Consolidate())
        m.apply(ChangeOfVariables(r"\hat{u}", "q", lambda q_i: q_i / h))

        # 10 — vertical reconstruction → interpolate (ŵ_j inlined as their closure)
        q = m.functions.q.head
        cov = {uh(i, t, x): q(i, t, x) / h for i in range(Nu + 1)}
        u_interp = sum((q(i, t, x) / h) * sp.legendre(i, 2 * zeta - 1)
                       for i in range(Nu + 1))
        w_interp = sum(sp.expand(w_closure[j].rhs.subs(cov)) * sp.legendre(j, 2 * zeta - 1)
                       for j in range(Nu + 2))
        m.register_group("interpolate", 0, b)
        m.register_group("interpolate", 1, h)
        m.register_group("interpolate", 2, u_interp)
        m.register_group("interpolate", 4, w_interp)
        m.register_group("interpolate", 5, rho * g * h * (1 - zeta))

        # 11 — model-derived lateral wall BC: the mirror state u(ζ) → −u(ζ)
        # flips EVERY moment (odd reflection, ⟨−u φ_i⟩ = −q_i/h); h and b
        # extrapolate.  Keyed by the FIELD (no hard-coded state slots) —
        # runtime access: FromModel(tag=…, definition="wall").
        for i in range(Nu + 1):
            m.register_group("boundary:wall", q(i, t, x), -q(i, t, x))

        self.derivation = m
        self._bed = b
        return None

    @property
    def system_model(self) -> SystemModel:
        """The runtime operator-form system (conservative q-state, `b` prepended).

        Boundary conditions passed to the constructor
        (``SME(level, boundary_conditions=BoundaryConditions(...))``) are
        forwarded — the normal interface, exactly as the production models.
        ``SystemModel.attach_boundary_conditions`` remains available as the
        hook for attaching/replacing BCs on an existing SystemModel."""
        m = self.derivation
        sm = SystemModel.from_model(m, Q=[self._bed, *m.explicit_state()])
        if self.boundary_conditions is not None:
            sm.attach_boundary_conditions(
                self.boundary_conditions, aux_bcs=self.aux_boundary_conditions)
        return sm
