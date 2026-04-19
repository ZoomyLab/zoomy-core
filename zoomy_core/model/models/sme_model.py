"""SME — Shallow Moment Equations (hydrostatic).

Derivation::

    system = FullINS(state).system()
    system.equations["z_momentum"].apply({w: 0, τ_zz: 0, τ_zx: 0})
    → ∂p/∂z/ρ + g = 0  →  p = p_atm + ρg(η - z)
    system.apply({p: p_hydro})
    del system.equations["z_momentum"]
    system.apply(DepthIntegrate)
    system.apply(ApplyKinematicBCs)
    ...
"""

import sympy as sp
from sympy import Function, S

from zoomy_core.model.models.derived_model import DerivedModel


def hydrostatic_scaling(state):
    """Dict that drops w and all z-row/z-column stresses (hydrostatic assumption).

    Dimension-agnostic: works for 2D (x-z) and 3D (x-y-z).
    """
    scaling = {state.w: S.Zero}
    for key in state.tau.keys():
        if "z" in key:
            scaling[state.tau[key]] = S.Zero
    return scaling


class INSModel(DerivedModel):
    """Root: the full incompressible Navier-Stokes (all equations).

    Dimension-agnostic: works for 2D (x-z) and 3D (x-y-z).
    The ``ins_dimension`` parameter controls the INS state space:
      - 2: u, w (standard 2D vertical slice)
      - 3: u, v, w (full 3D)

    Uses numerical eigenvalues (np.linalg.eigvals on quasilinear matrix)
    because the symbolic Cardano formula produces complex intermediates
    for higher-moment models (casus irreducibilis).
    """

    eigenvalue_mode = "numerical"
    ins_dimension = 2  # override to 3 for 3D derivation

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import StateSpace, FullINS
        self._system = FullINS(StateSpace(dimension=self.ins_dimension))


class SMEModel(INSModel):
    """Shallow Moment Equations — hydrostatic, depth-integrated, Newtonian."""

    projectable = True

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            HydrostaticPressure, Newtonian, DepthIntegrate,
            ApplyKinematicBCs, StressFreeSurface,
            ZeroAtmosphericPressure, SimplifyIntegrals,
        )
        super().derive_model()
        s = self.state

        # Hydrostatic: scale z-momentum, derive pressure, apply to system
        self._system.equations["z_momentum"] = (
            self._system.equations["z_momentum"]
            .apply(hydrostatic_scaling(s))
            .simplify()
        )
        self.apply(HydrostaticPressure(s))
        self._system.remove_equation("z_momentum")

        # Depth integrate + closures
        self.apply(DepthIntegrate(s))
        self.apply(ApplyKinematicBCs(s))
        self.apply(StressFreeSurface(s))
        self.apply(ZeroAtmosphericPressure(s))
        self.apply(SimplifyIntegrals(s))
        self.apply(Newtonian(s))

    def source(self):
        return (self.newtonian_viscosity() + self.navier_slip()
                + self.gravity_body_force())


class SMEInviscid(INSModel):
    """SME without viscosity."""

    projectable = True

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            HydrostaticPressure, Inviscid, DepthIntegrate,
            ApplyKinematicBCs, StressFreeSurface,
            ZeroAtmosphericPressure, SimplifyIntegrals,
        )
        super().derive_model()
        s = self.state

        self._system.equations["z_momentum"] = (
            self._system.equations["z_momentum"]
            .apply(hydrostatic_scaling(s))
            .simplify()
        )
        self.apply(HydrostaticPressure(s))
        self._system.remove_equation("z_momentum")

        self.apply(DepthIntegrate(s))
        self.apply(ApplyKinematicBCs(s))
        self.apply(StressFreeSurface(s))
        self.apply(ZeroAtmosphericPressure(s))
        self.apply(SimplifyIntegrals(s))
        self.apply(Inviscid(s))

    def source(self):
        return self.gravity_body_force()


# =============================================================================
# Tag-driven SME: tagged scalar equations are derived from the symbolic INS
# chain (no hand-coded operators).  Coverage is partial for now — terms that
# ``project_onto_basis`` cannot fully collapse into state-symbol space
# (notably surface/bottom velocity evaluations ``u(t,x,b)`` / ``u(t,x,b+h)``)
# stay in the untagged remainder and are surfaced at flux/source time via
# ``untagged_policy="warn"``.
# =============================================================================


class SMEModelTagged(SMEModel):
    """SMEModel whose numerical operators are extracted from
    ``solver_tag`` declarations on symbolically-projected scalar equations.

    Derivation path (everything comes from incompressible Navier-Stokes):

        FullINS → hydrostatic assumption on z-momentum →
        HydrostaticPressure substitution → remove z_momentum →
        DepthIntegrate → ApplyKinematicBCs → StressFreeSurface →
        ZeroAtmosphericPressure → SimplifyIntegrals → Newtonian →
        NavierSlipBottom → NoTangentialBoundaryStress →
        Expression.project_onto_basis (per test mode) →
        substitute {α_k(t,x) → hu_k/h, h(t,x) → h_sym, b(t,x) → b_sym} →
        Expression.auto_solver_tag

    No hand-coded flux/source/NC/hp is ever read.  Anything the structural
    auto-tagger cannot classify into a canonical tag — typically un-substituted
    surface/bottom velocity evaluations that projection does not reach —
    stays in the equation's untagged remainder.  Default ``untagged_policy``
    is ``"warn"`` so the gap is visible every time an operator is queried.

    Known pre-existing ``ins_generator`` artifacts that surface here and
    affect the numerical content of ``flux()`` at this stage
    (not caused by the tagging path, but not yet repaired upstream):

      * The post-``DepthIntegrate`` convective term carries a ``2 * u^2``
        factor in the main equation expression (the per-tag ``convection``
        value itself is clean ``u^2``).  After projection the flux tag
        therefore reflects ``2 * hu^2 / h`` — double the standard SWE
        convective flux.  Root-cause investigation on the
        ``DepthIntegrate`` / ``map_with_bcs`` path is a follow-up.

      * ``project_onto_basis`` does not collapse the Newtonian viscous
        term fully — an ``Integral(Subs(Derivative(u, x), z, b+h*zeta),
        (zeta, 0))`` residue appears in the flux tag.  It evaluates but
        not into closed form.

      * The Navier-slip friction term, ``(lambda/tau) u(t, x, b)``, stays
        in the untagged remainder because ``project_onto_basis`` does not
        substitute the bottom velocity evaluation ``u(t, x, b)`` (it only
        rewrites ``Integral`` nodes).  Expanding ``u(t, x, b) =
        sum_k alpha_k * phi_k(0)`` would make the friction tag-able as a
        ``source`` term.  Not attempted here — kept untagged per the
        policy choice when this pipeline was set up.
    """

    projectable = True
    untagged_policy = "warn"

    def derive_model(self):
        from zoomy_core.model.models.ins_generator import (
            NavierSlipBottom, NoTangentialBoundaryStress,
        )
        super().derive_model()
        s = self.state
        # Close the two remaining boundary terms flagged by the depth-
        # integration Leibniz expansion.
        self.apply(NavierSlipBottom(s))
        self.apply(NoTangentialBoundaryStress(s))

    def _initialize_derived_properties(self):
        super()._initialize_derived_properties()
        self._build_operator_system()

    def _build_operator_system(self):
        from zoomy_core.model.models.ins_generator import Expression
        from zoomy_core.model.models.derived_system import System

        state = self._system.state
        t, x, z = state.t, state.x, state.z
        L = self.level
        n_mom = L + 1

        # Symbol-land state variables: [b, h, hu_0, ..., hu_L].
        b_sym = self.variables[0]
        h_sym = self.variables[1]
        hu_syms = [self.variables[2 + k] for k in range(n_mom)]

        # Basis-coefficient placeholders for project_onto_basis.  They must
        # be Functions of (t, x) so derivatives ∂_x α_k survive projection.
        alpha_funcs = [sp.Function(f"_alpha_{k}_proj")(t, x) for k in range(n_mom)]
        field_map = {"u": alpha_funcs}

        coords = [x]
        equations = {}
        variable_map = {}

        def _subs_to_state_space(eq_expr):
            sub_map = {}
            # basis coefficients α_k(t,x) → hu_k / h
            for k, a in enumerate(alpha_funcs):
                sub_map[a] = hu_syms[k] / h_sym
            # Function state-fields → Symbols
            sub_map[state.H] = h_sym
            sub_map[state.b] = b_sym
            return eq_expr.subs(sub_map)

        # continuity (scalar): project with no test mode.
        mass = self._system.equations["continuity"].project_onto_basis(
            self.basis_type, L, field_map, z, test_mode=None
        )
        mass = _subs_to_state_space(mass)
        mass = mass.auto_solver_tag(
            state_vars=self.variables, time_var=t, coords=coords
        )
        equations["continuity"] = mass
        variable_map["continuity"] = [1]

        # x-momentum: project once per test mode, producing L+1 scalar equations.
        for l in range(n_mom):
            xml = self._system.equations["x_momentum"].project_onto_basis(
                self.basis_type, L, field_map, z, test_mode=l
            )
            xml = _subs_to_state_space(xml)
            xml = xml.auto_solver_tag(
                state_vars=self.variables, time_var=t, coords=coords
            )
            ename = f"x_momentum_{l}"
            equations[ename] = xml
            variable_map[ename] = [2 + l]

        self._operator_system = System(
            "SMEModelTagged_ops", state=state, equations=equations
        )
        self._operator_variable_map = variable_map
        self._operator_coords = coords
        self._operator_state_vars = list(self.variables)

    # ---- Tag-driven numerical operators ----------------------------------

    def _collect(self, tag, *, shape_dirs=0):
        from zoomy_core.model.models.tag_extraction import collect_solver_tag
        kwargs = dict(
            variable_map=self._operator_variable_map,
            n_variables=self.n_variables,
            policy=self.untagged_policy,
        )
        if shape_dirs:
            kwargs.update(
                n_directions=self.dimension,
                coords=self._operator_coords,
                state_variables=self._operator_state_vars,
            )
        return collect_solver_tag(self._operator_system, tag, **kwargs)

    def flux(self):
        from zoomy_core.misc.misc import ZArray
        return ZArray(self._collect("flux", shape_dirs=1))

    def hydrostatic_pressure(self):
        # Lumped into `flux` by auto_solver_tag; this returns zero here.
        # Well-balanced schemes that require the hydrostatic pressure
        # separately need an explicit splitting rule — not attempted yet.
        from zoomy_core.misc.misc import ZArray
        return ZArray.zeros(self.n_variables, self.dimension)

    def nonconservative_matrix(self):
        from zoomy_core.misc.misc import ZArray
        return ZArray(self._collect("nonconservative_flux", shape_dirs=1))

    def source(self):
        """RHS-convention source = -sum(source tags).

        Tags store the LHS piece so ``sum(tags) == equation``; the solver
        consumes ``dq/dt = ... + source`` (RHS form), hence the sign flip.
        """
        from zoomy_core.misc.misc import ZArray
        raw = self._collect("source", shape_dirs=0)
        out = ZArray.zeros(self.n_variables)
        for i, s in enumerate(raw):
            out[i] = -s
        return out
