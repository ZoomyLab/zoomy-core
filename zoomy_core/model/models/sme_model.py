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
# Tag-driven SME: operator bodies extracted from solver_tags on projected
# equations. Numerically equivalent to SMEModel; kept as a distinct class
# so the hand-coded one remains a directly-derivable reference PDE model.
# =============================================================================


class SMEModelTagged(SMEModel):
    """SMEModel whose flux/source/NC/hydrostatic_pressure are extracted
    from ``solver_tag`` declarations on post-projection scalar equations.

    Numerically identical to :class:`SMEModel` (the hand-coded version).
    The post-projection equations (one per moment ``k``) are built from
    the hand-coded operators and then tagged: each equation carries five
    canonical pieces (``time_derivative``, ``flux``, ``hydrostatic_pressure``,
    ``nonconservative_flux``, ``source``) whose sum equals the full LHS.
    """

    projectable = True
    untagged_policy = "strict"

    def _initialize_derived_properties(self):
        super()._initialize_derived_properties()
        self._build_operator_system()

    def _build_operator_system(self):
        """Build tagged scalar equations for every output row.

        Convention: each equation is in LHS-equals-zero form. Each
        ``solver_tag`` value is a piece of the LHS, so
        ``sum(tags) == equation_expression`` (the ``untagged_remainder``
        invariant). ``model.source()`` negates the source tag to match
        the solver's ``dq/dt = ... + source`` convention.

        Uses the hand-coded operators on the :class:`SMEModel` MRO as the
        source of truth, so the tag system drives numerically identical
        math. For ``level=0`` this collapses to the obvious conservative
        SWE form; for ``level>0`` the flux coefficients carry basis
        matrix entries as rational numbers.
        """
        from zoomy_core.model.models.ins_generator import Expression
        from zoomy_core.model.models.derived_system import System
        from zoomy_core.model.models.derived_model import DerivedModel

        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        coords = [x] if self.dimension == 1 else [x, y]
        q = list(self.variables)
        n_vars = self.n_variables
        n_mom = self.level + 1
        dim = self.dimension

        # Pull the reference operators off the hand-coded parent once.
        F_ref  = DerivedModel.flux(self)                  # (n_vars, dim)
        Hp_ref = DerivedModel.hydrostatic_pressure(self)  # (n_vars, dim)
        B_ref  = DerivedModel.nonconservative_matrix(self)# (n_vars, n_vars, dim)
        S_ref_rhs = SMEModel.source(self)                 # (n_vars,) RHS

        equations = {}
        variable_map = {}

        def _build_row(row, name):
            time_sp = sp.Derivative(q[row], t)
            flux_sp = sum(
                (sp.Derivative(F_ref[row, i], coords[i]) for i in range(dim)),
                sp.S.Zero,
            )
            hp_sp = sum(
                (sp.Derivative(Hp_ref[row, i], coords[i]) for i in range(dim)),
                sp.S.Zero,
            )
            nc_sp = sum(
                (B_ref[row, j, i] * sp.Derivative(q[j], coords[i])
                 for i in range(dim) for j in range(n_vars)),
                sp.S.Zero,
            )
            source_lhs = -S_ref_rhs[row]
            full = time_sp + flux_sp + hp_sp + nc_sp + source_lhs
            return Expression(full, name=name).solver_tag(
                time_derivative=time_sp,
                flux=flux_sp,
                hydrostatic_pressure=hp_sp,
                nonconservative_flux=nc_sp,
                source=source_lhs,
            )

        # Mass equation (h): row 1
        equations["continuity"] = _build_row(1, "continuity")
        variable_map["continuity"] = [1]

        # x-momentum moments: rows 2..2+L
        for k in range(n_mom):
            r = 2 + k
            ename = f"x_momentum_{k}"
            equations[ename] = _build_row(r, ename)
            variable_map[ename] = [r]

        # 2D: y-momentum moments (rows 2+n_mom..2+2*n_mom) — only if present.
        if self.dimension == 2 and n_vars > 2 + n_mom:
            for k in range(n_mom):
                r = 2 + n_mom + k
                ename = f"y_momentum_{k}"
                equations[ename] = _build_row(r, ename)
                variable_map[ename] = [r]

        self._operator_system = System(
            "SMEModelTagged_ops",
            state=(self._system.state if self._system is not None else None),
            equations=equations,
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
        from zoomy_core.misc.misc import ZArray
        return ZArray(self._collect("hydrostatic_pressure", shape_dirs=1))

    def nonconservative_matrix(self):
        from zoomy_core.misc.misc import ZArray
        return ZArray(self._collect("nonconservative_flux", shape_dirs=1))

    def source(self):
        """Return RHS-convention source = -sum(source tags).

        Tags store the LHS piece (so that ``sum(tags) == equation``).
        The solver consumes ``dq/dt = ... + source`` (RHS form), hence
        the sign flip here.
        """
        from zoomy_core.misc.misc import ZArray
        raw = self._collect("source", shape_dirs=0)
        out = ZArray.zeros(self.n_variables)
        for i, s in enumerate(raw):
            out[i] = -s
        return out
