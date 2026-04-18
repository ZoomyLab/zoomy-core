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

    Numerically identical to :class:`SMEModel` (the hand-coded version),
    but the numerical operators are *read off* symbolic tagged equations
    rather than assembled from basis matrices.

    Currently implements ``level=0`` only. For ``level>0`` the user must
    extend ``_build_operator_system`` with per-mode projected equations
    and corresponding tags.
    """

    projectable = True
    untagged_policy = "strict"

    def _initialize_derived_properties(self):
        super()._initialize_derived_properties()
        if self.level != 0:
            raise NotImplementedError(
                "SMEModelTagged currently supports level=0 only; "
                "level>0 requires per-mode projected equations with "
                "manual tagging (follow-up task)."
            )
        self._build_operator_system()

    def _build_operator_system(self):
        """Store the projected, tagged equations that drive flux/source/NC.

        Convention: each equation is written in LHS-equals-zero form.
        Every ``solver_tag`` value is a piece of the LHS; the tagged
        pieces sum to the equation expression (the ``untagged_remainder``
        invariant). ``model.source()`` negates the source tag to match
        the usual ``dq/dt = ... + source`` convention used by the solver.

        For ``level=0``, projection collapses ``u(t,x,z) = alpha_0(t,x)``
        and ``alpha_0 = hu/h``.
        """
        from zoomy_core.model.models.ins_generator import Expression
        from zoomy_core.model.models.derived_system import System

        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        b = self.variables[0]
        h = self.variables[1]
        hu = self.variables[2]
        p = self._parameter_symbols

        # Level-0 source pieces (LHS convention: +friction, -gravity_body).
        # RHS form (what model.source() returns): -(LHS source) = -friction + body.
        friction_lhs   = hu / (h * p.lamda)   # opposes motion → positive on LHS
        body_lhs       = -p.g * p.ex * h      # along-slope gravity accelerates → negative on LHS
        source_lhs     = friction_lhs + body_lhs

        mass = Expression(
            sp.Derivative(h, t) + sp.Derivative(hu, x),
            name="continuity",
        ).solver_tag(
            time_derivative=sp.Derivative(h, t),
            flux=sp.Derivative(hu, x),
        )

        xmom = Expression(
            (sp.Derivative(hu, t)
             + sp.Derivative(hu**2 / h, x)
             + sp.Derivative(p.g * p.ez * h**2 / 2, x)
             + p.g * p.ez * h * sp.Derivative(b, x)
             + source_lhs),
            name="x_momentum",
        ).solver_tag(
            time_derivative=sp.Derivative(hu, t),
            flux=sp.Derivative(hu**2 / h, x),
            hydrostatic_pressure=sp.Derivative(p.g * p.ez * h**2 / 2, x),
            nonconservative_flux=p.g * p.ez * h * sp.Derivative(b, x),
            source=source_lhs,
        )

        self._operator_system = System(
            "SMEModelTagged_ops",
            state=self._system.state if self._system is not None else None,
            equations={"continuity": mass, "x_momentum": xmom},
        )
        # Row mapping: b has no equation (static); h -> continuity; hu -> x_momentum
        self._operator_variable_map = {"continuity": [1], "x_momentum": [2]}
        self._operator_coords = [x]
        self._operator_state_vars = [b, h, hu]

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
