"""
General-purpose INS equation framework with composable projection operations.

Class hierarchy:
    StateSpace      — shared symbols (coordinates, fields, stress tensor, parameters)
    SymbolicBase    — name + display
    ├── Expression  — PDE terms with .project(), .ibp(), .apply(), .terms, [i]
    └── Relation    — lhs = rhs substitution rules with .apply_to()
        ├── Assumption  — physical conditions (kinematic BCs, hydrostatic)
        └── Material    — constitutive models (Newtonian, inviscid)

    FullINS(state)  — builds INS equations from a StateSpace

Design: the user drives every step. No hidden assumptions.
"""

import sympy as sp
from sympy import (
    Function, Symbol, Derivative, Integral, Add, Mul, S, Rational,
    Heaviside, DiracDelta,
)
import numpy as np


# ---------------------------------------------------------------------------
# StateSpace: shared symbolic state
# ---------------------------------------------------------------------------

class StateSpace:
    """
    Shared symbolic state: coordinates, velocity fields, pressure,
    stress tensor, bathymetry, and physical parameters.

    All equation systems, material models, and assumptions reference
    the same StateSpace to ensure symbol consistency.
    """

    def __init__(self, dimension=2):
        self.dim = dimension

        self.t = Symbol("t", real=True)
        self.x = Symbol("x", real=True)
        self.y = Symbol("y", real=True)
        self.z = Symbol("z", real=True)
        self.zeta = Symbol("zeta", real=True)

        args_h = [self.t, self.x]
        if dimension > 1:
            args_h.append(self.y)
        args_3d = args_h + [self.z]
        self._args_h = args_h
        self._args_3d = args_3d

        self.u = Function("u", real=True)(*args_3d)
        self.v = Function("v", real=True)(*args_3d) if dimension > 1 else S.Zero
        self.w = Function("w", real=True)(*args_3d)
        self.p = Function("p", real=True)(*args_3d)

        self.rho = Symbol("rho", positive=True)
        self.g = Symbol("g", positive=True)

        self.tau = {}
        for i in "xyz":
            for j in "xyz":
                self.tau[i + j] = Function(f"tau_{i}{j}", real=True)(*args_3d)

        self.b = Function("b", real=True)(*args_h)
        self.H = Function("H", real=True)(*args_h)
        self.eta = self.b + self.H

        self.coords_h = [self.x] + ([self.y] if dimension > 1 else [])
        self.velocities_h = [self.u] + ([self.v] if dimension > 1 else [])

    def __repr__(self):
        return f"StateSpace(dim={self.dim}, fields=[u,v,w,p], tau=9 components)"


# ---------------------------------------------------------------------------
# SymbolicBase: shared name + display
# ---------------------------------------------------------------------------

class SymbolicBase:
    """Base for all symbolic objects. Provides name and notebook display."""

    def __init__(self, name=""):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Expression: composable symbolic PDE term
# ---------------------------------------------------------------------------

class Expression(SymbolicBase):
    """
    Symbolic expression for PDE terms.

    Supports:
    - Term access: .terms, [i], len(), iteration
    - Projection: .project(test, var, domain, ...)
    - Integration by parts: .ibp(var, test_weight, domain) -> IBPResult
    - Apply conditions: .apply({old: new}, ...) or .apply(relation)
    - SymPy: .subs(), .simplify(), .expand(), .doit()
    - Notebook: _repr_latex_
    """

    def __init__(self, expr, name=""):
        super().__init__(name)
        if isinstance(expr, Expression):
            expr = expr.expr
        self.expr = sp.sympify(expr) if not isinstance(expr, sp.Basic) else expr

    @property
    def terms(self):
        expanded = sp.expand(self.expr)
        raw = Add.make_args(expanded)
        return [Expression(t, f"{self.name}[{i}]") for i, t in enumerate(raw)]

    def __getitem__(self, i):
        if isinstance(i, slice):
            ts = self.terms[i]
            combined = sum((t.expr for t in ts), S.Zero)
            return Expression(combined, self.name)
        return self.terms[i]

    def __len__(self):
        return len(Add.make_args(sp.expand(self.expr)))

    def __iter__(self):
        return iter(self.terms)

    def __add__(self, other):
        other_expr = other.expr if isinstance(other, Expression) else other
        return Expression(self.expr + other_expr, self.name)

    def __radd__(self, other):
        if other == 0:
            return self
        return Expression(other + self.expr, self.name)

    def __sub__(self, other):
        other_expr = other.expr if isinstance(other, Expression) else other
        return Expression(self.expr - other_expr, self.name)

    def __neg__(self):
        return Expression(-self.expr, self.name)

    def __mul__(self, other):
        other_expr = other.expr if isinstance(other, Expression) else other
        return Expression(self.expr * other_expr, self.name)

    def __rmul__(self, other):
        return Expression(other * self.expr, self.name)

    def project(self, test_func, var, domain=(0, 1), weight=S.One,
                scale=S.One, numerical=False, order=4):
        """Galerkin projection: integral(expr * test * weight * scale, (var, a, b))."""
        integrand = self.expr * test_func * weight * scale
        if numerical:
            result = gauss_legendre_integrate(integrand, var, domain[0], domain[1], order)
            return Expression(result, f"project_num({self.name})")
        return Expression(
            Integral(integrand, (var, domain[0], domain[1])),
            f"project({self.name})"
        )

    def ibp(self, var, test_weight, domain=(0, 1), scale=S.One):
        """
        Integration by parts on the outermost Derivative w.r.t. var.
        Returns IBPResult(integrate, boundary_upper, boundary_lower).
        """
        inner, coeff = _extract_derivative(self.expr, var)
        if inner is None:
            raise ValueError(
                f"No Derivative w.r.t. {var} found in: {self.expr}\n"
                f"Use .project() for terms without derivatives."
            )
        a, b = domain
        tw = test_weight * scale
        return IBPResult(
            integrate=Expression(
                -Integral(coeff * inner * Derivative(test_weight, var) * scale, (var, a, b)),
                f"ibp_integrate({self.name})"
            ),
            boundary_upper=Expression(
                (coeff * inner * tw).subs(var, b),
                f"ibp_upper({self.name})"
            ),
            boundary_lower=Expression(
                (coeff * inner * tw).subs(var, a),
                f"ibp_lower({self.name})"
            ),
        )

    def apply(self, *conditions):
        """
        Apply conditions: dicts, (old, new) tuples, or Relation objects.
        Relation objects use their .apply_to() method.
        """
        result = self.expr
        for cond in conditions:
            if isinstance(cond, Relation):
                result = cond.apply_to(result)
            elif isinstance(cond, dict):
                result = result.subs(cond)
            elif isinstance(cond, (list, tuple)):
                if len(cond) == 2 and isinstance(cond[0], sp.Basic):
                    result = result.subs(cond[0], cond[1])
                else:
                    for pair in cond:
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            result = result.subs(pair[0], pair[1])
            elif hasattr(cond, 'apply_to'):
                result = cond.apply_to(result)
        return Expression(result, self.name)

    def subs(self, *args, **kwargs):
        return Expression(self.expr.subs(*args, **kwargs), self.name)

    def simplify(self):
        return Expression(sp.simplify(self.expr), self.name)

    def expand(self):
        return Expression(sp.expand(self.expr), self.name)

    def doit(self):
        return Expression(self.expr.doit(), self.name)

    def has(self, *args):
        return self.expr.has(*args)

    @property
    def free_symbols(self):
        return self.expr.free_symbols

    def __repr__(self):
        short = str(self.expr)
        if len(short) > 80:
            short = short[:77] + "..."
        label = f" ({self.name})" if self.name else ""
        return f"Expression{label}: {short}"

    def _repr_latex_(self):
        return f"${sp.latex(self.expr)}$"

    def _sympy_(self):
        return self.expr

    def __eq__(self, other):
        if isinstance(other, Expression):
            return sp.simplify(self.expr - other.expr) == 0
        return sp.simplify(self.expr - other) == 0


class IBPResult:
    """
    Structured result from integration by parts.

    Attributes:
        integrate: the volume integral (Expression)
        boundary_upper: boundary term at upper limit (Expression)
        boundary_lower: boundary term at lower limit (Expression)
    """

    def __init__(self, integrate, boundary_upper, boundary_lower):
        self.integrate = integrate
        self.boundary_upper = boundary_upper
        self.boundary_lower = boundary_lower

    def apply_bcs(self, bc_lower=None, bc_upper=None):
        upper = self.boundary_upper
        lower = self.boundary_lower
        if bc_upper is not None:
            upper = upper.apply(bc_upper)
        if bc_lower is not None:
            lower = lower.apply(bc_lower)
        return IBPResult(self.integrate, upper, lower)

    def assemble(self):
        return self.integrate + self.boundary_upper - self.boundary_lower

    def __repr__(self):
        return (f"IBPResult(\n"
                f"  integrate={self.integrate},\n"
                f"  upper={self.boundary_upper},\n"
                f"  lower={self.boundary_lower}\n)")


# ---------------------------------------------------------------------------
# Relation: lhs = rhs substitution rules
# ---------------------------------------------------------------------------

class Relation(SymbolicBase):
    """
    A symbolic relation: one or more substitution rules lhs_i = rhs_i.

    Used as base for Assumption and Material. Can be applied to Expressions
    via expr.apply(relation), which calls relation.apply_to(expr).

    Displays as a system of equations in notebooks.
    """

    def __init__(self, substitutions, name=""):
        """
        Parameters
        ----------
        substitutions : dict {lhs_expr: rhs_expr} or list of (lhs, rhs) tuples
        name : str
        """
        super().__init__(name)
        if isinstance(substitutions, dict):
            self.subs_map = dict(substitutions)
        elif isinstance(substitutions, (list, tuple)):
            self.subs_map = dict(substitutions)
        else:
            raise TypeError("substitutions must be a dict or list of (lhs, rhs) tuples")

    def apply_to(self, expr):
        """Substitute all lhs -> rhs in the given SymPy expression."""
        result = expr
        for lhs, rhs in self.subs_map.items():
            result = result.subs(lhs, rhs)
        return result

    def __len__(self):
        return len(self.subs_map)

    def __repr__(self):
        lines = [f"{self.__class__.__name__}(name={self.name!r}, {len(self)} rules):"]
        for lhs, rhs in self.subs_map.items():
            lines.append(f"  {lhs} = {rhs}")
        return "\n".join(lines)

    def _repr_latex_(self):
        lines = []
        for lhs, rhs in self.subs_map.items():
            lines.append(f"{sp.latex(lhs)} = {sp.latex(rhs)}")
        body = " \\\\ ".join(lines)
        return f"$\\begin{{aligned}} {body} \\end{{aligned}}$"


class Assumption(Relation):
    """Physical assumption (kinematic BC, hydrostatic, etc.)."""
    pass


class Material(Relation):
    """Constitutive model (Newtonian, inviscid, etc.)."""
    pass


# ---------------------------------------------------------------------------
# FullINS: 3D INS equations built from a StateSpace
# ---------------------------------------------------------------------------

class FullINS:
    """
    Full 3D Incompressible Navier-Stokes equations.

    Built from a StateSpace. Does not own the symbols — references them.
    Includes continuity, x/y/z momentum with full stress tensor divergence.
    """

    def __init__(self, state: StateSpace):
        self.state = state
        self.dim = state.dim

    def _stress_divergence(self, row):
        s = self.state
        labels = ["x", "y", "z"]
        coords = [s.x, s.y, s.z]
        expr = S.Zero
        for j in range(3):
            if s.dim == 1 and labels[j] == "y":
                continue
            expr += Derivative(s.tau[row + labels[j]], coords[j])
        return expr

    @property
    def continuity(self):
        s = self.state
        expr = Derivative(s.u, s.x) + Derivative(s.w, s.z)
        if s.dim > 1:
            expr += Derivative(s.v, s.y)
        return Expression(expr, "continuity")

    @property
    def x_momentum(self):
        s = self.state
        inertia = Derivative(s.u, s.t) + Derivative(s.u * s.u, s.x) + Derivative(s.u * s.w, s.z)
        if s.dim > 1:
            inertia += Derivative(s.u * s.v, s.y)
        pressure = Rational(1, 1) / s.rho * Derivative(s.p, s.x)
        stress = Rational(1, 1) / s.rho * self._stress_divergence("x")
        return Expression(inertia + pressure - stress, "x_momentum")

    @property
    def y_momentum(self):
        s = self.state
        if s.dim < 2:
            return None
        inertia = (
            Derivative(s.v, s.t) + Derivative(s.v * s.u, s.x)
            + Derivative(s.v * s.v, s.y) + Derivative(s.v * s.w, s.z)
        )
        pressure = Rational(1, 1) / s.rho * Derivative(s.p, s.y)
        stress = Rational(1, 1) / s.rho * self._stress_divergence("y")
        return Expression(inertia + pressure - stress, "y_momentum")

    @property
    def z_momentum(self):
        s = self.state
        inertia = Derivative(s.w, s.t) + Derivative(s.w * s.u, s.x) + Derivative(s.w * s.w, s.z)
        if s.dim > 1:
            inertia += Derivative(s.w * s.v, s.y)
        pressure = Rational(1, 1) / s.rho * Derivative(s.p, s.z)
        gravity = s.g
        stress = Rational(1, 1) / s.rho * self._stress_divergence("z")
        return Expression(inertia + pressure + gravity - stress, "z_momentum")

    @property
    def equations(self):
        eqs = [self.continuity, self.x_momentum]
        if self.dim > 1:
            eqs.append(self.y_momentum)
        eqs.append(self.z_momentum)
        return eqs


# ---------------------------------------------------------------------------
# Material models library
# ---------------------------------------------------------------------------

class Newtonian(Material):
    """Newtonian fluid: tau_ij = mu * (du_i/dx_j + du_j/dx_i), mu = rho * nu."""

    def __init__(self, state: StateSpace, nu=None):
        self.nu = nu if nu is not None else Symbol("nu", positive=True)
        subs = self._build(state, self.nu)
        super().__init__(subs, name="Newtonian")

    @staticmethod
    def _build(s, nu):
        mu = nu * s.rho
        u, v, w = s.u, s.v, s.w
        x, y, z = s.x, s.y, s.z
        subs = {
            s.tau["xx"]: 2 * mu * Derivative(u, x),
            s.tau["xz"]: mu * (Derivative(u, z) + Derivative(w, x)),
            s.tau["zx"]: mu * (Derivative(w, x) + Derivative(u, z)),
            s.tau["zz"]: 2 * mu * Derivative(w, z),
        }
        if s.dim > 1:
            subs.update({
                s.tau["xy"]: mu * (Derivative(u, y) + Derivative(v, x)),
                s.tau["yx"]: mu * (Derivative(v, x) + Derivative(u, y)),
                s.tau["yy"]: 2 * mu * Derivative(v, y),
                s.tau["yz"]: mu * (Derivative(v, z) + Derivative(w, y)),
                s.tau["zy"]: mu * (Derivative(w, y) + Derivative(v, z)),
            })
        else:
            for key in ["xy", "yx", "yy", "yz", "zy"]:
                subs[s.tau[key]] = S.Zero
        return subs


class Inviscid(Material):
    """Inviscid fluid: all tau_ij = 0."""

    def __init__(self, state: StateSpace):
        subs = {v: S.Zero for v in state.tau.values()}
        super().__init__(subs, name="Inviscid")


class materials:
    """Material model library. Usage: materials.newtonian(state)"""
    newtonian = Newtonian
    inviscid = Inviscid


# ---------------------------------------------------------------------------
# Assumptions library
# ---------------------------------------------------------------------------

class KinematicBCBottom(Assumption):
    """w|_{z=b} = db/dt + u_b * db/dx [+ v_b * db/dy]"""

    def __init__(self, state: StateSpace):
        s = state
        w_at_b = s.w.subs(s.z, s.b)
        u_at_b = s.u.subs(s.z, s.b)
        rhs = Derivative(s.b, s.t) + u_at_b * Derivative(s.b, s.x)
        if s.dim > 1:
            v_at_b = s.v.subs(s.z, s.b)
            rhs += v_at_b * Derivative(s.b, s.y)
        super().__init__({w_at_b: rhs}, name="kinematic_bc_bottom")


class KinematicBCSurface(Assumption):
    """w|_{z=eta} = d(eta)/dt + u_s * d(eta)/dx [+ v_s * d(eta)/dy]"""

    def __init__(self, state: StateSpace):
        s = state
        w_at_s = s.w.subs(s.z, s.eta)
        u_at_s = s.u.subs(s.z, s.eta)
        rhs = Derivative(s.eta, s.t) + u_at_s * Derivative(s.eta, s.x)
        if s.dim > 1:
            v_at_s = s.v.subs(s.z, s.eta)
            rhs += v_at_s * Derivative(s.eta, s.y)
        super().__init__({w_at_s: rhs}, name="kinematic_bc_surface")


class HydrostaticPressure(Assumption):
    """p = p_atm + rho * g * (eta - z)"""

    def __init__(self, state: StateSpace):
        s = state
        p_atm = Function("p_atm", real=True)(s.t, *s.coords_h)
        p_hydro = p_atm + s.rho * s.g * (s.eta - s.z)
        super().__init__({s.p: p_hydro}, name="hydrostatic_pressure")


class assumptions:
    """Assumptions library. Usage: assumptions.kinematic_bc_bottom(state)"""
    kinematic_bc_bottom = KinematicBCBottom
    kinematic_bc_surface = KinematicBCSurface
    hydrostatic_pressure = HydrostaticPressure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_derivative(expr, var):
    expr = sp.sympify(expr)
    if isinstance(expr, Derivative):
        if expr.variables == (var,):
            return expr.args[0], S.One
    factors = Mul.make_args(expr)
    coeff_factors = []
    inner = None
    for f in factors:
        if isinstance(f, Derivative) and f.variables == (var,) and inner is None:
            inner = f.args[0]
        else:
            coeff_factors.append(f)
    if inner is None:
        return None, None
    coeff = Mul(*coeff_factors) if coeff_factors else S.One
    return inner, coeff


def integrate_by_parts(f, g, var, domain=(0, 1)):
    """
    Standalone IBP: integral d(f)/dvar * g dvar = [f*g]_a^b - integral f * dg/dvar dvar
    Returns IBPResult(integrate, boundary_upper, boundary_lower).
    """
    a, b = domain
    return IBPResult(
        integrate=Expression(-Integral(f * Derivative(g, var), (var, a, b)), "ibp_integrate"),
        boundary_upper=Expression((f * g).subs(var, b), "ibp_upper"),
        boundary_lower=Expression((f * g).subs(var, a), "ibp_lower"),
    )


def gauss_legendre_integrate(expr, var, a, b, order=4):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes_shifted = [(b - a) / 2 * xi + (b + a) / 2 for xi in nodes]
    weights_scaled = [(b - a) / 2 * wi for wi in weights]
    result = S.Zero
    for xi, wi in zip(nodes_shifted, weights_scaled):
        result += sp.Rational(wi).limit_denominator(10**8) * expr.subs(var, sp.nsimplify(xi))
    return result
