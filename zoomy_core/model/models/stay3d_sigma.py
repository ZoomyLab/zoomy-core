"""Stay-3D σ-coordinate model — the general 3-D mass+momentum balance, σ-mapped,
with the divergence constraint integrated against the kinematic BCs to give the
EXACT height-evolution equation, but WITHOUT any velocity ansatz / moment
projection.  The flow stays 3-D; the single unclosed object — the vertical
integral — is substituted into an auxiliary variable whose per-step value is
supplied by a backend "integrate over z" (the same machinery ``project_from_3d``
uses).

Verified derivation (``diff = 0`` against the framework, 2026-06-13):

* height equation (φ₀-moment of continuity against the two KBCs)::

      ∂_t h + ∂_x(h·U) = 0 ,        U = ∫₀¹ ũ dζ           (depth-mean velocity)

  IMPORTANT: ``U`` is the depth-MEAN (the ζ-independent ``h`` factors out of the
  column integral by itself, so the height flux is ``F[h] = U·h``).  The integral
  MUST be substituted into ``U`` BEFORE ``Simplify`` — otherwise a spurious
  ``+U·∂_x h`` term survives (``Simplify`` illegally pulls ``h`` inside
  ``∂_x(∫ũ dζ)``).

* the full system, in CONSERVATIVE σ-form, is a clean conservation law over
  ``(x, ζ)`` with the contravariant (σ-relative) vertical velocity
  ``h·ω = w̃ − ∂_t z − ũ ∂_x z``, ``z = b + h ζ``::

      ∂_t h     + ∂_x(h u)            + ∂_ζ(h ω)                            = 0
      ∂_t(h u)  + ∂_x(h u² + g h²/2)  + ∂_ζ(h u ω) + g h ∂_x b − e_x g h
                                                    − (1/ρ) ∂_ζ τ̃_xz       = 0

  ``ω(0)=ω(1)=0`` ⟺ the bed / free-surface KBCs, so the σ metric terms are
  ABSORBED into ``∂_ζ(h q ω)`` (zero boundary flux) — ζ is simply a third FLUX
  direction, no ALE/metric source class is needed.  ``U = ∫₀¹ u dζ`` (full
  column) drives the height flux; ``ω`` is a RUNNING column integral diagnosed
  from continuity — both are the same per-backend integrate-over-z (full vs
  running).

The aux is introduced with EXISTING machinery — no ``promote_to_aux`` op::

    m.add_equation("U", sp.Eq(U, ∫ũ dζ), group="aux")   # records the definition
    m.mass.apply({∫ũ dζ: U})                            # U auto-lands in Qaux

``Stay3DSigma(...).derive_model()`` builds the declarative model;
``.system_model`` returns the height-reduced ``[b, h] + U`` SystemModel (the
3-D momentum is stashed on ``self`` for the forthcoming 3-D extraction, which
needs the vertical added to the extractor's ``space``).
"""
from __future__ import annotations

import param
import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.basemodel import Model as BaseModel
from zoomy_core.model.derivation import (
    Model as DModel, PDETransformation, Simplify, ResolveIntegral)
from zoomy_core.model.derivation.projection import Integrate
from zoomy_core.model.operations import Multiply, ProductRule, KinematicBC
from zoomy_core.model.operations import Integrate as IntegrateZ
from zoomy_core.systemmodel import SystemModel

t, x, y, z = C.t, C.x, C.y, C.z
zeta = sp.Symbol("zeta", real=True)


class Stay3DSigma(BaseModel):
    """3-D mass+momentum → σ-map → column-integrated height equation, keeping the
    velocity profile UNCLOSED as a per-backend vertical integral.

    State handed to the SystemModel (v1): ``[b, h]``, with the depth-mean
    velocity ``U`` exposed as an auxiliary defined by ``U = ∫₀¹ ũ dζ`` (and the
    3-D field ``ũ`` carried as the aux it is integrated from).  The 3-D momentum
    advance becomes SystemModel state once the extractor treats ζ as a flux
    direction (stashed here as ``self._momentum_sigma_x`` / ``self._omega_def``).
    """

    _finalize_lazy = True
    dimension = param.Integer(default=2, bounds=(2, 3), doc=(
        "Total spatial dimension INCLUDING the vertical: 2 → coords (t,x,z), one "
        "horizontal (U=U_x); 3 → coords (t,x,y,z), two horizontals.  Only dim=2 "
        "is wired today; dim=3 needs the SME HAXES-loop treatment."))
    closures = param.List(default=[], doc=(
        "Stress closures applied to the RESOLVED 3-D σ-momentum (closures.py): the "
        "bulk (Newtonian) closure becomes the vertical viscous diffusion "
        "∂_ζ(ρν/h ∂_ζ ũ); the bottom (NavierSlip/RoughWall) closure sets the bed "
        "friction traction τ̃_xz(0); StressFree the free surface.  Empty → defaults "
        "to [Newtonian(), NavierSlip(), StressFree()] (slip + viscous; the friction "
        "magnitude is set by the parameters nu / lambda_s)."))

    def derive_model(self):
        if int(self.dimension) != 2:
            raise NotImplementedError(
                "Stay3DSigma: only dimension=2 (t,x,z) is wired today.")
        values = {"g": 9.81, "rho": 1.0, "nu": 0.0, "lambda_s": 0.0, "e_x": 0.0}
        user_vals = getattr(self, "parameter_values", None)
        if user_vals is not None and hasattr(user_vals, "items"):
            values.update({k: float(v) for k, v in user_vals.items()})
        from zoomy_core.model.models.equations import (
            Mass, Momentum, moment_scaling)

        m = DModel(coords=(t, x, z), parameters=values)
        g, rho, e_x = m.parameters.g, m.parameters.rho, m.parameters.e_x
        h = sp.Function("h", positive=True)(t, x)
        b = sp.Function("b", real=True)(t, x)

        # 1 — general 3-D mass + momentum (full stress tensor), shallow scaling.
        m.declare_state(h)
        m.add_equation("bottom", d.t(b))                 # ∂_t b = 0
        m.add_equation(Mass(m))                          # mints u, w
        mom = Momentum(m); m.add_equation(mom)           # mints p (+ τ closures)
        moment_scaling(m, mom)                           # drop in-plane stresses
        uvel, w, p = mom.uvel, mom.w, mom.p
        m.add_equation("kbc_top", KinematicBC(w=w, u=uvel[0], interface=b + h))
        m.add_equation("kbc_bot", KinematicBC(w=w, u=uvel[0], interface=b))

        # 2 — hydrostatic elimination of p (physical z).
        mz = m.momentum.z
        mz.apply({d.t(w): 0, d.z(w * w): 0, d.x(uvel[0] * w): 0})
        mz.apply(IntegrateZ(z, z, b + h, method="analytical"))
        mz.apply({p.subs(z, b + h): 0})                  # p(surface) = 0
        m.momentum.x.apply(mz.solve_for(p)); mz.remove()
        m.momentum.x.apply(Simplify())

        # 3 — σ-map z = b + h·ζ (decorates u→ũ, w→w̃, τ→τ̃; σ-maps the KBCs).
        m.apply(PDETransformation({z: (zeta, sp.Eq(z, b + h * zeta))}))

        # 3b — CLOSE the material/stress on the resolved σ-momentum.  Bulk
        # (Newtonian) τ̃_xz = ρν/h ∂_ζ ũ  → the vertical viscous diffusion
        # ∂_ζ(ρν/h ∂_ζ ũ); bottom (NavierSlip/RoughWall) → the bed friction
        # traction; StressFree → the free surface.  Same machinery as SME.
        from zoomy_core.model.models.closures import (
            Newtonian, NavierSlip, StressFree, apply_stress_closures)
        from zoomy_core.model.models.material import ClosureState
        clos = self.closures or [Newtonian(), NavierSlip(), StressFree()]

        def _cstate(at, *, alias=None, btag=None):
            return ClosureState(m.functions, params=m.parameters, h=h, x=x,
                                zeta=zeta, at=at, alias=alias,
                                boundary_tag=btag, horiz=[x])
        axes = [{"mx": m.momentum.x, "tau": m.functions.tau_xz, "velname": "u"}]
        apply_stress_closures(clos, m, axes, _cstate, [x])
        m.momentum.x.apply(Simplify())

        def _head(name):
            pool = m.momentum.x.expr.atoms(sp.Function)
            return next(a.func for a in pool if str(a.func) == name)
        ut = _head(r"\tilde{u}")(t, x, zeta)
        wt = _head(r"\tilde{w}")(t, x, zeta)

        # 3c — vertical-face boundary conditions (the friction lives HERE for a
        # resolved column, not in the PDE): bed = Navier slip τ̃_xz(0)=ρλ_s ũ(0)
        # (λ_s→∞ ⇒ no-slip ũ(0)=0); surface = stress-free τ̃_xz(1)=0.  Consumed by
        # the 3-D solve's ζ=0 / ζ=1 face fluxes.  τ̃_xz = ρν/h ∂_ζ ũ (Newtonian).
        rho_, nu_, lam_ = m.parameters.rho, m.parameters.nu, m.parameters.lambda_s
        dzu = sp.Derivative(ut, zeta)
        self._vertical_bcs = {
            "bed":     sp.Eq(rho_ * nu_ / h * dzu.subs(zeta, 0),
                             rho_ * lam_ * ut.subs(zeta, 0)),
            "surface": sp.Eq(dzu.subs(zeta, 1), 0),
        }

        # 3-D conservative-σ ingredients for the forthcoming 3-D extraction
        # (verified: σ-momentum(×h) ≡ ∂_t(hu)+∂_x(hu²+gh²/2)+∂_ζ(h u ω)+ghb_x−e_x g h−τ̃_z/ρ).
        zc = b + h * zeta
        self._omega_def = wt - sp.Derivative(zc, t) - ut * sp.Derivative(zc, x)   # = h·ω
        self._momentum_sigma_x = sp.expand((h * m.momentum.x.expr).doit())
        self._heads_3d = {"u": ut.func, "w": wt.func}

        # 4 — height equation: φ₀-moment of continuity against the two KBCs.
        m.mass.apply(Multiply(h))
        m.mass.apply(ProductRule(variables=[zeta]))
        m.mass.apply(Integrate(zeta, bounds=(0, 1)))
        m.mass.apply(ResolveIntegral())
        ints = list(m.mass.expr.atoms(sp.Integral))
        assert len(ints) == 1, f"expected exactly one vertical integral, got {ints}"
        vint = ints[0]                                   # = ∫₀¹ ũ dζ
        U = sp.Function("U", real=True)(t, x)            # depth-mean velocity aux
        # introduce the aux with existing machinery (NO promote_to_aux op):
        m.add_equation("U", sp.Eq(U, vint), group="aux")  # record the definition
        m.mass.apply({vint: U})                           # substitute → U ∈ Qaux
        m.mass.apply(m.kbc_bot); m.mass.apply(m.kbc_top)
        m.mass.apply({sp.Derivative(b, t): 0})
        m.mass.apply(Simplify())
        # internal self-check: the height eq is EXACTLY ∂_t h + ∂_x(h·U).
        ref = sp.Derivative(h, t).doit() + sp.diff(h * U, x)
        assert sp.simplify((m.mass.expr - ref).doit()) == 0, (
            f"height equation mismatch: {m.mass.expr} != ∂_t h + ∂_x(h U)")
        self._U_def = vint                                # U = ∫₀¹ ũ dζ

        # 5 — v1 reduced system: stash, then drop the 3-D momentum so the
        # extractor sees only the classifiable depth-reduced rows.  (The 3-D
        # momentum becomes SystemModel state once ζ is a flux direction.)
        m.remove("momentum")
        self.derivation = m
        self._bed = b
        return None

    @property
    def system_model(self) -> SystemModel:
        """Height-reduced operator system: state ``[b, h]``, flux ``F[h] = U·h``;
        ``U`` (depth-mean) auto-exposed as an auxiliary, ``ũ`` carried as the 3-D
        field it integrates from."""
        m = self.derivation
        qs = list(m.explicit_state())
        if self._bed not in qs:
            qs = [self._bed, *qs]
        sm = SystemModel.from_model(m, Q=qs, canonical_source=self)
        from zoomy_core.model.boundary_conditions import resolve_and_attach
        resolve_and_attach(sm, self.boundary_conditions,
                           aux_bcs=self.aux_boundary_conditions)
        return sm
