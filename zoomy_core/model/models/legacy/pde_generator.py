"""
Automated PDE Generator for Multi-Layer SME/VAM Models.

Derives layer-averaged / moment-projected shallow water equations from
the 3D Incompressible Navier-Stokes equations using Galerkin projection
with arbitrary vertical basis functions.

Phase 1: Algebraic Formulation & Moment Projection
  1.1 — Non-integrated INS base (continuous SymPy functions)
  1.2 — Abstract Galerkin projection with weighted test functions
  1.3 — Application of kinematic & dynamic boundary conditions
Phase 2: Basis Injection & Delayed Substitution
  2.1 — Coordinate mapping to reference element
  2.2 — Heaviside windowing for multi-layer ansatz
  2.3 — Substitution into abstract projected equations
Phase 3: Custom SymPy Integration Engine
  3.1 — AST parsing for Heaviside/DiracDelta
  3.2 — Static piecewise collapse per layer
  3.3 — DiracDelta extraction via sifting property
"""

import sympy as sp
from sympy import (
    Function, Symbol, symbols, Derivative, Integral, Rational,
    Heaviside, DiracDelta, Piecewise, integrate, Add, Mul, Pow,
    oo, S, Wild, Number,
)


# ---------------------------------------------------------------------------
# Phase 1.1: Non-Integrated INS Base
# ---------------------------------------------------------------------------

class INSBase:
    """
    3D Incompressible Navier-Stokes equations in continuous form.

    State fields: u(t,x,y,z), v(t,x,y,z), w(t,x,y,z), p(t,x,y,z)
    Parameters: rho (density), nu (kinematic viscosity), g (gravity)

    The equations are:
      Continuity:  du/dx + dv/dy + dw/dz = 0
      X-momentum:  du/dt + d(uu)/dx + d(uv)/dy + d(uw)/dz = -1/rho dp/dx + nu Lap(u)
      Y-momentum:  dv/dt + d(vu)/dx + d(vv)/dy + d(vw)/dz = -1/rho dp/dy + nu Lap(v)
      Z-momentum:  dw/dt + d(wu)/dx + d(wv)/dy + d(ww)/dz = -1/rho dp/dz + nu Lap(w) - g

    For shallow water derivation we work in the hydrostatic limit
    (drop z-momentum PDE, use dp/dz = -rho*g directly).
    """

    def __init__(self, dimension=2):
        self.dimension = dimension
        self.t = Symbol("t", real=True)
        self.x = Symbol("x", real=True)
        self.z = Symbol("z", real=True)

        self.coords = [self.x]
        if dimension == 2:
            self.y = Symbol("y", real=True)
            self.coords = [self.x, self.y]
        else:
            self.y = None

        self.rho = Symbol("rho", positive=True)
        self.nu = Symbol("nu", positive=True)
        self.g = Symbol("g", positive=True)
        self.g_vec = self._gravity_vector()

        self.u = self._field("u")
        self.w = self._field("w")
        self.p = self._field("p")
        self.v = self._field("v") if dimension == 2 else None

        self.horizontal_velocities = [self.u]
        if self.v is not None:
            self.horizontal_velocities.append(self.v)

    def _field(self, name):
        args = [self.t] + self.coords + [self.z]
        return Function(name, real=True)(*args)

    def _gravity_vector(self):
        if self.dimension == 1:
            return [sp.Integer(0)]
        return [sp.Integer(0), sp.Integer(0)]

    def full_args(self):
        args = [self.t] + self.coords + [self.z]
        return args

    def continuity(self):
        expr = Derivative(self.u, self.x) + Derivative(self.w, self.z)
        if self.v is not None:
            expr += Derivative(self.v, self.y)
        return expr

    def x_momentum(self):
        t, x, z = self.t, self.x, self.z
        u, w = self.u, self.w
        expr = (
            Derivative(u, t)
            + Derivative(u * u, x)
            + Derivative(u * w, z)
            + Rational(1, 1) / self.rho * Derivative(self.p, x)
        )
        if self.v is not None:
            expr += Derivative(u * self.v, self.y)
        return expr

    def y_momentum(self):
        if self.v is None:
            return None
        t, x, y, z = self.t, self.x, self.y, self.z
        v, w = self.v, self.w
        expr = (
            Derivative(v, t)
            + Derivative(v * self.u, x)
            + Derivative(v * v, y)
            + Derivative(v * w, z)
            + Rational(1, 1) / self.rho * Derivative(self.p, y)
        )
        return expr

    def hydrostatic_pressure(self):
        return Derivative(self.p, self.z) + self.rho * self.g

    def momentum_equations(self):
        eqs = [self.x_momentum()]
        if self.v is not None:
            eqs.append(self.y_momentum())
        return eqs


# ---------------------------------------------------------------------------
# Phase 1.2: Abstract Galerkin Projection
# ---------------------------------------------------------------------------

class GalerkinProjection:
    """
    Galerkin projection of 3D INS equations onto vertical basis functions.

    Given the PDE L(u,v,w,p) = 0 over z in [b, b+H], the projected equation
    for test function phi_i(zeta) with weight c(zeta) is:

       integral_0^1 L(...) * c(zeta) * phi_i(zeta) * H  d_zeta = 0

    where zeta = (z - b) / H maps z in [b, b+H] to [0, 1].

    This class works with ABSTRACT symbols for:
      - phi_i(zeta): test function (not yet specified)
      - u, v, w, p: continuous fields
      - b, H: bathymetry and total depth
    """

    def __init__(self, ins: INSBase):
        self.ins = ins
        self.t = ins.t
        self.z = ins.z
        self.x = ins.x
        self.y = ins.y
        self.dimension = ins.dimension

        self.b = Function("b", real=True)(self.t, *ins.coords)
        self.H = Function("H", real=True)(self.t, *ins.coords)

        self.zeta = Symbol("zeta", real=True)
        self.phi_i = Function("phi_i")(self.zeta)
        self.c_weight = Function("c")(self.zeta)

    def z_of_zeta(self):
        return self.b + self.H * self.zeta

    def project_continuity(self):
        """
        Integrate continuity equation over depth with test function.

        Starting from:  du/dx + dv/dy + dw/dz = 0

        Multiply by H * c(zeta) * phi_i(zeta) and integrate over zeta in [0,1]:

          integral_0^1 [du/dx + dv/dy] * H * c * phi_i  d_zeta
          + integral_0^1 dw/dz * H * c * phi_i  d_zeta = 0

        For the dw/dz term, use dw/dz = (1/H) dw/d_zeta, so:

          integral_0^1 dw/d_zeta * c * phi_i  d_zeta

        Integration by parts on this last term gives:

          [w * c * phi_i]_0^1 - integral_0^1 w * d(c * phi_i)/d_zeta  d_zeta

        The boundary terms [w * c * phi_i] at zeta=0 and zeta=1 are where
        kinematic BCs enter (Phase 1.3).
        """
        zeta, phi_i, c_w = self.zeta, self.phi_i, self.c_weight
        H = self.H

        horizontal_div = self._horizontal_divergence_integrand()
        volume_horizontal = Integral(horizontal_div * H * c_w * phi_i, (zeta, 0, 1))

        w = Function("w")(self.t, *self.ins.coords, zeta)
        vertical_volume = -Integral(
            w * Derivative(c_w * phi_i, zeta), (zeta, 0, 1)
        )

        return ProjectedEquation(
            volume=volume_horizontal + vertical_volume,
            boundary_top=self._w_at_surface() * (c_w * phi_i).subs(zeta, 1),
            boundary_bottom=-self._w_at_bottom() * (c_w * phi_i).subs(zeta, 0),
            name="continuity",
        )

    def project_momentum(self, component_index=0):
        """
        Project the horizontal momentum equation for component `component_index`.

        component_index=0 -> x-momentum
        component_index=1 -> y-momentum (only if dimension==2)

        Starting from (x-momentum shown):
          du/dt + d(uu)/dx + d(uv)/dy + d(uw)/dz + (1/rho) dp/dx = 0

        Multiply by H * c(zeta) * phi_i(zeta), integrate over zeta in [0,1].

        The d(uw)/dz term gets integration by parts (same as continuity):
          integral dw*u/dz * H * c * phi_i  d_zeta
          = [u*w * c * phi_i]_0^1 - integral u*w * d(c*phi_i)/d_zeta  d_zeta

        The pressure gradient term becomes:
          integral (1/rho) dp/dx * H * c * phi_i  d_zeta

        Under hydrostatic assumption p(z) = p_s + rho*g*(b+H-z), the
        pressure gradient introduces the free surface gradient and
        atmospheric pressure — handled in Phase 1.3.
        """
        ins = self.ins
        zeta, phi_i, c_w = self.zeta, self.phi_i, self.c_weight
        H = self.H

        if component_index == 0:
            vel_field = "u"
            coord = self.x
            mom_eq_func = ins.x_momentum
        elif component_index == 1:
            vel_field = "v"
            coord = self.y
            mom_eq_func = ins.y_momentum
        else:
            raise ValueError("component_index must be 0 or 1")

        u_sym = Function(vel_field)(self.t, *ins.coords, zeta)
        w_sym = Function("w")(self.t, *ins.coords, zeta)

        temporal = Derivative(u_sym, self.t)
        temporal_projected = Integral(temporal * H * c_w * phi_i, (zeta, 0, 1))

        horizontal_advection = self._horizontal_advection_integrand(component_index)
        horizontal_projected = Integral(
            horizontal_advection * H * c_w * phi_i, (zeta, 0, 1)
        )

        vertical_boundary_top = (
            u_sym.subs(zeta, 1)
            * self._w_at_surface()
            * (c_w * phi_i).subs(zeta, 1)
        )
        vertical_boundary_bottom = (
            -u_sym.subs(zeta, 0)
            * self._w_at_bottom()
            * (c_w * phi_i).subs(zeta, 0)
        )
        vertical_volume = -Integral(
            u_sym * w_sym * Derivative(c_w * phi_i, zeta), (zeta, 0, 1)
        )

        pressure_integrand = self._pressure_gradient_integrand(coord)
        pressure_projected = Integral(
            pressure_integrand * H * c_w * phi_i, (zeta, 0, 1)
        )

        viscous_integrand = self._viscous_integrand(component_index)
        viscous_projected = Integral(
            viscous_integrand * H * c_w * phi_i, (zeta, 0, 1)
        )

        volume = (
            temporal_projected
            + horizontal_projected
            + vertical_volume
            + pressure_projected
            + viscous_projected
        )

        return ProjectedEquation(
            volume=volume,
            boundary_top=vertical_boundary_top,
            boundary_bottom=vertical_boundary_bottom,
            name=f"momentum_{vel_field}",
        )

    def project_all(self):
        """Return all projected equations (continuity + momentum components)."""
        eqs = [self.project_continuity()]
        for d in range(self.dimension):
            eqs.append(self.project_momentum(component_index=d))
        return eqs

    # --- Internal helpers for building integrands ---

    def _horizontal_divergence_integrand(self):
        zeta = self.zeta
        ins = self.ins
        u = Function("u")(self.t, *ins.coords, zeta)
        div = Derivative(u, self.x)
        if ins.v is not None:
            v = Function("v")(self.t, *ins.coords, zeta)
            div += Derivative(v, self.y)
        return div

    def _horizontal_advection_integrand(self, component_index):
        zeta = self.zeta
        ins = self.ins
        vel_names = ["u"]
        if ins.v is not None:
            vel_names.append("v")

        vel_field_name = vel_names[component_index]
        u_comp = Function(vel_field_name)(self.t, *ins.coords, zeta)

        adv = sp.Integer(0)
        for i, name in enumerate(vel_names):
            coord = ins.coords[i]
            u_i = Function(name)(self.t, *ins.coords, zeta)
            adv += Derivative(u_comp * u_i, coord)
        return adv

    def _pressure_gradient_integrand(self, coord):
        zeta = self.zeta
        ins = self.ins
        p = Function("p")(self.t, *ins.coords, zeta)
        return Rational(1, 1) / ins.rho * Derivative(p, coord)

    def _viscous_integrand(self, component_index):
        """
        Vertical viscous term: -nu * d2u/dz2 mapped to zeta coordinates.
        d2u/dz2 = (1/H^2) d2u/d_zeta2
        Returns: -nu/H^2 * d2u/d_zeta2  (to be multiplied by H in the integral)
        """
        zeta = self.zeta
        ins = self.ins
        vel_names = ["u", "v"] if ins.v is not None else ["u"]
        vel_field_name = vel_names[component_index]
        u_comp = Function(vel_field_name)(self.t, *ins.coords, zeta)
        return -ins.nu / self.H**2 * Derivative(u_comp, zeta, 2)

    def _w_at_surface(self):
        return Function("w_s")(self.t, *self.ins.coords)

    def _w_at_bottom(self):
        return Function("w_b")(self.t, *self.ins.coords)


# ---------------------------------------------------------------------------
# Phase 1.3: Kinematic & Dynamic Boundary Conditions
# ---------------------------------------------------------------------------

class BoundaryConditions:
    """
    Physical boundary conditions for the depth-averaged system.

    Kinematic BCs (zero relative mass flux):
      At z = b:       w_b = db/dt + u_b * db/dx + v_b * db/dy
      At z = b + H:   w_s = d(b+H)/dt + u_s * d(b+H)/dx + v_s * d(b+H)/dy

    Dynamic BCs:
      Surface: tau_s  (wind stress, atmospheric pressure gradient)
      Bottom:  tau_b  (bed friction)

    Hydrostatic pressure:
      p(x,y,z,t) = p_atm + rho * g * (b + H - z)
      => dp/dx = dp_atm/dx + rho*g * d(b+H)/dx - rho*g*dz/dx
                = dp_atm/dx + rho*g * d(b+H)/dx  (for depth-averaged)
    """

    def __init__(self, projection: GalerkinProjection):
        self.proj = projection
        self.ins = projection.ins
        self.t = projection.t
        self.x = projection.x
        self.y = projection.y
        self.b = projection.b
        self.H = projection.H
        self.dimension = projection.dimension

        self.tau_bx = Symbol("tau_bx", real=True)
        self.tau_sx = Symbol("tau_sx", real=True)
        self.p_atm = Function("p_atm")(self.t, *self.ins.coords)

        if self.dimension == 2:
            self.tau_by = Symbol("tau_by", real=True)
            self.tau_sy = Symbol("tau_sy", real=True)

    def kinematic_bottom(self):
        """w at z = b:  w_b = db/dt + u_b*db/dx [+ v_b*db/dy]"""
        b = self.b
        u_b = Function("u_b")(self.t, *self.ins.coords)
        expr = Derivative(b, self.t) + u_b * Derivative(b, self.x)
        if self.y is not None:
            v_b = Function("v_b")(self.t, *self.ins.coords)
            expr += v_b * Derivative(b, self.y)
        return expr

    def kinematic_surface(self):
        """w at z = b+H:  w_s = d(b+H)/dt + u_s*d(b+H)/dx [+ v_s*d(b+H)/dy]"""
        eta = self.b + self.H
        u_s = Function("u_s")(self.t, *self.ins.coords)
        expr = Derivative(eta, self.t) + u_s * Derivative(eta, self.x)
        if self.y is not None:
            v_s = Function("v_s")(self.t, *self.ins.coords)
            expr += v_s * Derivative(eta, self.y)
        return expr

    def hydrostatic_pressure_gradient(self, coord):
        """
        Under hydrostatic assumption:
          p = p_atm + rho * g * (b + H - z)
          dp/dx = dp_atm/dx + rho * g * d(b+H)/dx

        Projected:
          (1/rho) integral dp/dx * H * c * phi_i  d_zeta
          = integral [1/rho * dp_atm/dx + g * d(b+H)/dx] * H * c * phi_i  d_zeta
          = [1/rho * dp_atm/dx + g * d(b+H)/dx] * H * integral c * phi_i  d_zeta

        The last step holds because the pressure gradient is z-independent
        under hydrostatic assumption.
        """
        rho = self.ins.rho
        g = self.ins.g
        eta = self.b + self.H
        return Derivative(self.p_atm, coord) / rho + g * Derivative(eta, coord)

    def apply_to_continuity(self, projected_eq):
        """
        Replace abstract w_s, w_b in the projected continuity equation
        with kinematic BC expressions.
        """
        w_s = self.kinematic_surface()
        w_b = self.kinematic_bottom()
        return ProjectedEquation(
            volume=projected_eq.volume,
            boundary_top=projected_eq.boundary_top.subs(
                Function("w_s")(self.t, *self.ins.coords), w_s
            ),
            boundary_bottom=projected_eq.boundary_bottom.subs(
                Function("w_b")(self.t, *self.ins.coords), w_b
            ),
            name=projected_eq.name + "_with_kbc",
        )

    def apply_to_momentum(self, projected_eq, component_index=0):
        """
        Apply BCs to projected momentum equation:
        1. Replace w_s, w_b with kinematic BCs in boundary terms
        2. Replace pressure gradient with hydrostatic expression
        3. Add surface/bottom stress contributions
        """
        w_s = self.kinematic_surface()
        w_b = self.kinematic_bottom()

        new_boundary_top = projected_eq.boundary_top.subs(
            Function("w_s")(self.t, *self.ins.coords), w_s
        )
        new_boundary_bottom = projected_eq.boundary_bottom.subs(
            Function("w_b")(self.t, *self.ins.coords), w_b
        )

        if component_index == 0:
            tau_s = self.tau_sx
            tau_b = self.tau_bx
        else:
            tau_s = self.tau_sy
            tau_b = self.tau_by

        stress_top = tau_s / self.ins.rho
        stress_bottom = -tau_b / self.ins.rho

        return ProjectedEquation(
            volume=projected_eq.volume,
            boundary_top=new_boundary_top + stress_top,
            boundary_bottom=new_boundary_bottom + stress_bottom,
            name=projected_eq.name + "_with_bc",
            pressure_gradient=self.hydrostatic_pressure_gradient(
                self.ins.coords[component_index]
            ),
        )

    def apply_all(self, projected_equations):
        """Apply BCs to a list of projected equations from GalerkinProjection.project_all()."""
        result = []
        result.append(self.apply_to_continuity(projected_equations[0]))
        for d in range(self.dimension):
            result.append(self.apply_to_momentum(projected_equations[1 + d], d))
        return result


# ---------------------------------------------------------------------------
# Data container for projected equations
# ---------------------------------------------------------------------------

class ProjectedEquation:
    """
    Container for a single projected (depth-integrated) equation.

    Attributes:
        volume:           The volume integral terms (integrands integrated over [0,1])
        boundary_top:     Boundary contribution at zeta=1 (surface)
        boundary_bottom:  Boundary contribution at zeta=0 (bottom)
        pressure_gradient: Hydrostatic pressure gradient (if applicable)
        name:             Human-readable identifier
    """

    def __init__(self, volume, boundary_top, boundary_bottom, name="",
                 pressure_gradient=None):
        self.volume = volume
        self.boundary_top = boundary_top
        self.boundary_bottom = boundary_bottom
        self.pressure_gradient = pressure_gradient
        self.name = name

    def full_equation(self):
        expr = self.volume + self.boundary_top + self.boundary_bottom
        if self.pressure_gradient is not None:
            expr += self.pressure_gradient
        return expr

    def __repr__(self):
        return f"ProjectedEquation(name={self.name!r})"

    def pprint(self):
        print(f"--- {self.name} ---")
        print("Volume terms:")
        sp.pprint(self.volume)
        print("\nBoundary (top, zeta=1):")
        sp.pprint(self.boundary_top)
        print("\nBoundary (bottom, zeta=0):")
        sp.pprint(self.boundary_bottom)
        if self.pressure_gradient is not None:
            print("\nPressure gradient (hydrostatic):")
            sp.pprint(self.pressure_gradient)
        print()


# ---------------------------------------------------------------------------
# Phase 3: Custom SymPy Integration Engine
# ---------------------------------------------------------------------------

class PiecewiseIntegrator:
    """
    Custom integration engine that safely handles Heaviside-windowed expressions.

    SymPy's native Piecewise integration causes AST explosion and hangs for
    multi-layer expressions. This engine:

    1. Scans the expression tree for Heaviside and DiracDelta terms.
    2. For smooth volume integration: divides the domain into chunks at
       layer interfaces, statically collapses Heaviside to 0/1 per chunk,
       then integrates the resulting smooth polynomial.
    3. For DiracDelta terms: extracts coefficients and applies the sifting
       property f(z_k) / |g'(z_k)| at each interface.
    """

    def __init__(self, var, interfaces):
        """
        Parameters
        ----------
        var : Symbol
            Integration variable (typically zeta).
        interfaces : list[Expr]
            Sorted list of interface positions [z_0, z_1, ..., z_N] defining
            N layers. Must include both endpoints (bottom and top).
        """
        self.var = var
        self.interfaces = list(interfaces)
        self.n_layers = len(self.interfaces) - 1

    def integrate(self, expr):
        """
        Integrate expr over the full domain [interfaces[0], interfaces[-1]].

        Splits expr into smooth (Heaviside) and singular (DiracDelta) parts,
        handles each with the appropriate strategy, and returns the sum.
        """
        smooth_part, delta_terms = self._split_smooth_and_delta(expr)
        result = self._integrate_smooth(smooth_part)
        result += self._integrate_deltas(delta_terms)
        return sp.expand(result)

    def _split_smooth_and_delta(self, expr):
        """
        Decompose expr into:
          - smooth_part: everything that does NOT contain DiracDelta
          - delta_terms: list of additive terms that DO contain DiracDelta
        """
        expr = sp.expand(expr)
        terms = Add.make_args(expr)
        smooth_terms = []
        delta_terms = []
        for term in terms:
            if term.has(DiracDelta):
                delta_terms.append(term)
            else:
                smooth_terms.append(term)
        smooth_part = Add(*smooth_terms) if smooth_terms else S.Zero
        return smooth_part, delta_terms

    def _integrate_smooth(self, expr):
        """
        Integrate a Heaviside-containing (but DiracDelta-free) expression
        by dividing into layer chunks and statically collapsing Heaviside
        values at each chunk's midpoint.
        """
        if expr == S.Zero:
            return S.Zero
        result = S.Zero
        z = self.var
        for k in range(self.n_layers):
            z_lo = self.interfaces[k]
            z_hi = self.interfaces[k + 1]
            midpoint = (z_lo + z_hi) / 2
            collapsed = self._collapse_heavisides(expr, midpoint)
            chunk_integral = integrate(collapsed, (z, z_lo, z_hi))
            result += chunk_integral
        return result

    def _collapse_heavisides(self, expr, point):
        """
        Replace all Heaviside(f(z)) in expr with 0 or 1 by evaluating
        the sign of f at the given point.

        For Heaviside(z - z_k): if point > z_k -> 1, if point < z_k -> 0.
        At exactly z_k we use 0 (convention: Heaviside(0) = 0 in this context,
        since we handle the interface via DiracDelta).
        """
        z = self.var
        heavisides = self._find_heavisides(expr)
        subs_map = {}
        for h_expr in heavisides:
            arg = h_expr.args[0]
            val_at_point = arg.subs(z, point)
            try:
                numeric_val = float(val_at_point)
                if numeric_val > 0:
                    subs_map[h_expr] = S.One
                else:
                    subs_map[h_expr] = S.Zero
            except (TypeError, ValueError):
                subs_map[h_expr] = Heaviside(val_at_point)
        return expr.subs(subs_map)

    def _find_heavisides(self, expr):
        """Find all Heaviside sub-expressions in expr."""
        result = set()
        for sub in sp.preorder_traversal(expr):
            if isinstance(sub, Heaviside):
                result.add(sub)
        return result

    def _integrate_deltas(self, delta_terms):
        """
        Apply the sifting property to each DiracDelta term:

        integral f(z) * DiracDelta(g(z)) dz = f(z_k) / |g'(z_k)|

        where z_k is the root of g(z) = 0 within the integration domain.
        """
        if not delta_terms:
            return S.Zero
        z = self.var
        z_lo = self.interfaces[0]
        z_hi = self.interfaces[-1]
        result = S.Zero
        for term in delta_terms:
            result += self._sift_single_delta(term, z, z_lo, z_hi)
        return result

    def _sift_single_delta(self, term, z, z_lo, z_hi):
        """
        Extract the DiracDelta from a single multiplicative term,
        find its root, and apply the sifting property.
        """
        delta_expr, coefficient = self._extract_delta_and_coefficient(term, z)
        if delta_expr is None:
            return S.Zero

        arg = delta_expr.args[0]
        roots = sp.solve(arg, z)

        result = S.Zero
        for root in roots:
            try:
                root_val = sp.nsimplify(root)
                in_domain = (root_val - z_lo >= 0) and (z_hi - root_val >= 0)
                if not in_domain:
                    try:
                        in_domain = float(root_val - z_lo) >= 0 and float(z_hi - root_val) >= 0
                    except (TypeError, ValueError):
                        in_domain = True
            except (TypeError, ValueError):
                in_domain = True

            if in_domain:
                g_prime = sp.diff(arg, z)
                g_prime_at_root = g_prime.subs(z, root)
                coeff_at_root = coefficient.subs(z, root)
                if g_prime_at_root != 0:
                    result += coeff_at_root / sp.Abs(g_prime_at_root)
                else:
                    result += coeff_at_root
        return result

    def _extract_delta_and_coefficient(self, term, z):
        """
        Given a term like f(z) * DiracDelta(g(z)) * Heaviside(...) * ...,
        separate out the DiracDelta and everything else (the coefficient).

        Heaviside factors are evaluated at the delta's root location and
        folded into the coefficient.
        """
        term = sp.expand(term)
        factors = Mul.make_args(term)
        delta_expr = None
        coeff_factors = []
        for f in factors:
            if isinstance(f, DiracDelta):
                delta_expr = f
            elif isinstance(f, Pow) and isinstance(f.base, DiracDelta):
                delta_expr = f.base
            else:
                coeff_factors.append(f)
        coefficient = Mul(*coeff_factors) if coeff_factors else S.One
        return delta_expr, coefficient


# ---------------------------------------------------------------------------
# Phase 2: Basis Injection & Delayed Substitution
# ---------------------------------------------------------------------------

class LayeredAnsatz:
    """
    Constructs the multi-layer piecewise vertical ansatz using Heaviside
    windowing and substitutes it into the abstract projected equations.

    The ansatz for a horizontal velocity component is:

        u(t, x, zeta) = sum_k sum_j  u_{k,j}(t,x) * phi_j(zeta_local)
                                       * [H(zeta - zeta_{k-1}) - H(zeta - zeta_k)]

    where:
      - k indexes the layer (k = 0, ..., n_layers-1)
      - j indexes the basis function within a layer (j = 0, ..., level)
      - zeta_local = (zeta - zeta_{k-1}) / (zeta_k - zeta_{k-1})
      - phi_j is the j-th basis polynomial on [0, 1]
      - H is the Heaviside step function
    """

    def __init__(self, n_layers, basis, dimension=1):
        """
        Parameters
        ----------
        n_layers : int
            Number of vertical layers.
        basis : Basisfunction
            Basis function object from basisfunctions.py.
        dimension : int
            Horizontal dimension (1 or 2).
        """
        self.n_layers = n_layers
        self.basis = basis
        self.level = basis.level
        self.dimension = dimension

        self.zeta = Symbol("zeta", real=True)
        self.t = Symbol("t", real=True)
        self.x = Symbol("x", real=True)
        self.y = Symbol("y", real=True) if dimension == 2 else None

        self.layer_interfaces = self._default_uniform_interfaces()
        self.layer_dofs = self._create_layer_dofs()

    def _default_uniform_interfaces(self):
        """Uniform layer interfaces in [0, 1]: [0, 1/N, 2/N, ..., 1]."""
        N = self.n_layers
        return [Rational(k, N) for k in range(N + 1)]

    def _create_layer_dofs(self):
        """
        Create symbolic degrees of freedom for each layer and component.

        Returns dict:  dofs[component][layer_k][basis_j] = Symbol
        where component is 'u' (and 'v' if 2D).
        """
        components = ["u"]
        if self.dimension == 2:
            components.append("v")
        dofs = {}
        for comp in components:
            dofs[comp] = {}
            for k in range(self.n_layers):
                dofs[comp][k] = {}
                for j in range(self.level + 1):
                    name = f"{comp}_{k}_{j}"
                    dofs[comp][k][j] = Symbol(name, real=True)
        return dofs

    def local_coordinate(self, k):
        """Local coordinate within layer k: zeta_local = (zeta - zeta_k) / delta_k."""
        z_lo = self.layer_interfaces[k]
        z_hi = self.layer_interfaces[k + 1]
        return (self.zeta - z_lo) / (z_hi - z_lo)

    def layer_thickness(self, k):
        """Thickness of layer k in zeta-space."""
        return self.layer_interfaces[k + 1] - self.layer_interfaces[k]

    def window_function(self, k):
        """Heaviside window for layer k: H(zeta - zeta_k) - H(zeta - zeta_{k+1})."""
        z_lo = self.layer_interfaces[k]
        z_hi = self.layer_interfaces[k + 1]
        return Heaviside(self.zeta - z_lo) - Heaviside(self.zeta - z_hi)

    def ansatz_in_layer(self, component, k):
        """
        Expansion within a single layer k for given component:
          sum_j u_{k,j} * phi_j(zeta_local)
        """
        zeta_loc = self.local_coordinate(k)
        from sympy.abc import z as z_sym
        expr = S.Zero
        for j in range(self.level + 1):
            phi_j = self.basis.get(j)
            phi_j_local = phi_j.subs(z_sym, zeta_loc)
            expr += self.layer_dofs[component][k][j] * phi_j_local
        return expr

    def full_ansatz(self, component):
        """
        Full piecewise ansatz with Heaviside windowing:
          u(zeta) = sum_k [sum_j u_{k,j} phi_j(zeta_local)] * W_k(zeta)
        """
        expr = S.Zero
        for k in range(self.n_layers):
            expr += self.ansatz_in_layer(component, k) * self.window_function(k)
        return sp.expand(expr)

    def derivative_ansatz(self, component):
        """
        d/d_zeta of the full ansatz. Produces:
        - Smooth terms: derivatives of polynomials within each window
        - DiracDelta terms at interfaces from Heaviside derivatives
        """
        return sp.diff(self.full_ansatz(component), self.zeta)

    def substitute_into_integrand(self, integrand, component_map=None):
        """
        Replace abstract Function calls in an integrand expression with
        the concrete piecewise ansatz.

        Parameters
        ----------
        integrand : Expr
            SymPy expression potentially containing Function("u")(t, x, zeta), etc.
        component_map : dict, optional
            Maps function names to component keys, e.g. {"u": "u", "v": "v"}.
            Defaults to identity mapping.
        """
        if component_map is None:
            component_map = {"u": "u"}
            if self.dimension == 2:
                component_map["v"] = "v"

        result = integrand
        for func_name, comp_key in component_map.items():
            ansatz_expr = self.full_ansatz(comp_key)
            args_list = [self.t, self.x]
            if self.y is not None:
                args_list.append(self.y)
            args_list.append(self.zeta)
            abstract_func = Function(func_name)(*args_list)
            result = result.subs(abstract_func, ansatz_expr)
        return result

    def integrate_projected(self, integrand, test_func_index, weight_func=None):
        """
        Full pipeline: substitute ansatz, multiply by test function and weight,
        then integrate using the PiecewiseIntegrator.

        Parameters
        ----------
        integrand : Expr
            Abstract integrand (from Phase 1 projection).
        test_func_index : int
            Which test function phi_i to use (layer_k, basis_j encoded).
        weight_func : Expr, optional
            Integration weight c(zeta). Defaults to 1.
        """
        concrete = self.substitute_into_integrand(integrand)
        integrator = PiecewiseIntegrator(self.zeta, self.layer_interfaces)
        return integrator.integrate(concrete)

    def get_all_dof_symbols(self, component=None):
        """Return a flat list of all DOF symbols, optionally for one component."""
        components = [component] if component else list(self.layer_dofs.keys())
        result = []
        for comp in components:
            for k in range(self.n_layers):
                for j in range(self.level + 1):
                    result.append(self.layer_dofs[comp][k][j])
        return result


# ---------------------------------------------------------------------------
# Phase 4: Interface Routing & Physical Overrides
# ---------------------------------------------------------------------------

class InterfaceRouter:
    """
    Separates internal numerical interfaces from external physical boundaries.

    Given a list of DiracDelta terms from the PiecewiseIntegrator, this class:
    1. Checks the location (root) of each delta
    2. If at z=0 or z=1 (physical boundary): discards the symbolic jump and
       replaces it with the user-defined physical condition
    3. If at an internal interface (0 < z_k < 1): tags it as an unresolved
       numerical flux for the Riemann solver
    """

    def __init__(self, interfaces, var):
        self.interfaces = list(interfaces)
        self.var = var
        self.z_bottom = interfaces[0]
        self.z_top = interfaces[-1]
        self.internal_interfaces = interfaces[1:-1]

    def classify_delta_terms(self, expr):
        """
        Split an expression into three parts:
        - smooth: terms without DiracDelta
        - boundary_deltas: delta terms at z=0 or z=1
        - internal_deltas: delta terms at internal interfaces

        Returns
        -------
        ClassifiedTerms with attributes: smooth, boundary, internal
        """
        expr = sp.expand(expr)
        terms = Add.make_args(expr)

        smooth = []
        boundary = []
        internal = []

        for term in terms:
            if not term.has(DiracDelta):
                smooth.append(term)
                continue

            root = self._find_delta_root(term)
            if root is None:
                smooth.append(term)
                continue

            if self._is_boundary(root):
                boundary.append(InterfaceJump(term, root, "boundary"))
            else:
                boundary_flag = False
                for z_int in self.internal_interfaces:
                    if sp.simplify(root - z_int) == 0:
                        internal.append(InterfaceJump(term, root, "internal"))
                        boundary_flag = True
                        break
                if not boundary_flag:
                    internal.append(InterfaceJump(term, root, "internal"))

        smooth_expr = Add(*smooth) if smooth else S.Zero
        return ClassifiedTerms(smooth_expr, boundary, internal)

    def _find_delta_root(self, term):
        """Find the root of the DiracDelta argument in a term."""
        for sub in sp.preorder_traversal(term):
            if isinstance(sub, DiracDelta):
                arg = sub.args[0]
                roots = sp.solve(arg, self.var)
                if roots:
                    return roots[0]
        return None

    def _is_boundary(self, root):
        return (sp.simplify(root - self.z_bottom) == 0 or
                sp.simplify(root - self.z_top) == 0)

    def apply_physical_bcs(self, classified, bc_bottom=None, bc_top=None):
        """
        Replace boundary delta terms with user-defined physical conditions.

        Parameters
        ----------
        classified : ClassifiedTerms
        bc_bottom : Expr or None
            Physical boundary condition at z=0 (e.g., tau_bottom/rho for diffusion, 0 for advection)
        bc_top : Expr or None
            Physical boundary condition at z=1

        Returns the modified smooth expression with BCs applied.
        """
        result = classified.smooth
        if bc_bottom is None:
            bc_bottom = S.Zero
        if bc_top is None:
            bc_top = S.Zero

        for jump in classified.boundary:
            if sp.simplify(jump.location - self.z_bottom) == 0:
                result += bc_bottom
            elif sp.simplify(jump.location - self.z_top) == 0:
                result += bc_top
        return result


class InterfaceJump:
    """A single interface jump term extracted from the integration."""

    def __init__(self, expression, location, jump_type):
        self.expression = expression
        self.location = location
        self.jump_type = jump_type

    def __repr__(self):
        return f"InterfaceJump(type={self.jump_type}, loc={self.location})"


class ClassifiedTerms:
    """Container for terms classified by the InterfaceRouter."""

    def __init__(self, smooth, boundary, internal):
        self.smooth = smooth
        self.boundary = boundary
        self.internal = internal

    @property
    def n_internal(self):
        return len(self.internal)

    @property
    def n_boundary(self):
        return len(self.boundary)
