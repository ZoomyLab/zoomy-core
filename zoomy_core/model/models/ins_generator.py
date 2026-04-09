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

    dimension is the physical space dimension:
        2 = xz plane (1D horizontal shallow water)
        3 = xyz space (2D horizontal shallow water)

    The vertical coordinate is always z. Horizontal coordinates are
    x (and y if dimension=3). The horizontal_dim property gives the
    number of horizontal directions (1 or 2).
    """

    def __init__(self, dimension=2):
        if dimension < 2 or dimension > 3:
            raise ValueError(f"dimension must be 2 (xz) or 3 (xyz), got {dimension}")
        self.dim = dimension

        self.t = Symbol("t", real=True)
        self.x = Symbol("x", real=True)
        self.y = Symbol("y", real=True)
        self.z = Symbol("z", real=True)
        self.zeta = Symbol("zeta", real=True)

        has_y = dimension > 2

        args_h = [self.t, self.x]
        if has_y:
            args_h.append(self.y)
        args_3d = args_h + [self.z]
        self._args_h = args_h
        self._args_3d = args_3d

        self.u = Function("u", real=True)(*args_3d)
        self.v = Function("v", real=True)(*args_3d) if has_y else S.Zero
        self.w = Function("w", real=True)(*args_3d)
        self.p = Function("p", real=True)(*args_3d)

        self.rho = Symbol("rho", positive=True)
        self.g = Symbol("g", positive=True)

        self._build_stress_tensor(has_y, args_3d)

        self.b = Function("b", real=True)(*args_h)
        self.H = Function("h", real=True)(*args_h)
        self.eta = self.b + self.H

        self.coords_h = [self.x] + ([self.y] if has_y else [])
        self.velocities_h = [self.u] + ([self.v] if has_y else [])

    def _build_stress_tensor(self, has_y, args_3d):
        labels = ["x", "y", "z"] if has_y else ["x", "z"]
        self.tau = {}
        for i in labels:
            for j in labels:
                self.tau[i + j] = Function(f"tau_{i}{j}", real=True)(*args_3d)

    @property
    def horizontal_dim(self):
        return self.dim - 1

    @property
    def has_y(self):
        return self.dim > 2

    def __repr__(self):
        n_tau = len(self.tau)
        fields = "[u,v,w,p]" if self.has_y else "[u,w,p]"
        return f"StateSpace(dim={self.dim}, fields={fields}, tau={n_tau} components)"


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

    def __init__(self, expr, name="", term_groups=None):
        super().__init__(name)
        if isinstance(expr, Expression):
            expr = expr.expr
        self.expr = sp.sympify(expr) if not isinstance(expr, sp.Basic) else expr
        # Optional ordered term groups for display:
        # {"temporal": sp.Expr, "convection": sp.Expr, ...}
        # When set, latex() renders groups in this order.
        self._term_groups = term_groups

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

    def apply_to_term(self, index, *operations):
        """Apply operations to a specific term and return the full expression.

        Usage::

            xmom.apply_to_term(5, ProductRule())
            xmom.apply_to_term(5, {old: new})
        """
        terms = self.terms
        modified = terms[index].apply(*operations)
        new_terms = [t.expr if i != index else modified.expr
                     for i, t in enumerate(terms)]
        return Expression(sum(new_terms, S.Zero), self.name)

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
        Preserves term_groups if present.
        """
        def _apply_one(expr, cond):
            if isinstance(cond, Operation):
                # Operations may need the Expression (for .map, .terms etc.)
                if hasattr(cond, 'apply_to_expression'):
                    try:
                        eq = Expression(expr, self.name, term_groups=self._term_groups)
                        return cond.apply_to_expression(eq).expr
                    except NotImplementedError:
                        pass
                return cond.apply_to(expr)
            if isinstance(cond, Relation) or hasattr(cond, 'apply_to'):
                # Evaluate Subs first so BC patterns match
                if expr.has(sp.Subs):
                    subs_map = {s: s.doit() for s in expr.atoms(sp.Subs)}
                    expr = expr.subs(subs_map)
                return cond.apply_to(expr)
            elif isinstance(cond, dict):
                return expr.subs(cond)
            elif isinstance(cond, (list, tuple)):
                if len(cond) == 2 and isinstance(cond[0], sp.Basic):
                    return expr.subs(cond[0], cond[1])
                else:
                    for pair in cond:
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            expr = expr.subs(pair[0], pair[1])
                    return expr
            elif hasattr(cond, 'apply_to'):
                return cond.apply_to(expr)
            return expr

        result = self.expr
        for cond in conditions:
            result = _apply_one(result, cond)

        # Propagate term groups
        new_groups = None
        if self._term_groups:
            new_groups = {}
            for role, group_expr in self._term_groups.items():
                g = group_expr
                for cond in conditions:
                    g = _apply_one(g, cond)
                new_groups[role] = g

        # Simplify for Relation applies (not Operations — they handle their own)
        if not any(isinstance(c, Operation) for c in conditions):
            result = _simplify_preserve_integrals(result)
        return Expression(result, self.name, term_groups=new_groups)

    def simplify(self):
        """Return a new Expression with sympy simplification applied."""
        simplified = sp.simplify(self.expr)
        new_groups = None
        if self._term_groups:
            new_groups = {role: sp.simplify(g) for role, g in self._term_groups.items()}
        return Expression(simplified, self.name, term_groups=new_groups)

    def expand(self):
        """Return a new Expression with sympy expand applied."""
        expanded = sp.expand(self.expr)
        new_groups = None
        if self._term_groups:
            new_groups = {role: sp.expand(g) for role, g in self._term_groups.items()}
        return Expression(expanded, self.name, term_groups=new_groups)

    def depth_integrate(self, lower, upper, var, method="auto"):
        """
        Depth-integrate this expression over [lower, upper] w.r.t. var.

        Parameters
        ----------
        lower, upper : sympy expressions for the integration bounds
            (typically b and b+H, both functions of t and x)
        var : Symbol
            The vertical coordinate (z)
        method : str
            'auto'    : detect derivative direction and choose method
            'leibniz' : pull horizontal derivative outside:
                        int df/dx dz = d/dx[int f dz] - f(upper)*d(upper)/dx + f(lower)*d(lower)/dx
            'fundamental_theorem' : for vertical derivatives:
                        int df/dz dz = f(upper) - f(lower)
            'direct'  : keep as Integral(expr, (var, lower, upper))

        Returns
        -------
        DepthIntegralResult with .volume, .boundary_upper, .boundary_lower
        or a plain Expression for 'direct' and 'fundamental_theorem'.
        """
        expr = self.expr

        if method == "auto":
            # Detect: does the expression contain d/dz?
            inner_z, coeff_z = _extract_derivative(expr, var)
            if inner_z is not None:
                method = "fundamental_theorem"
            else:
                # Check for d/dx (or any horizontal derivative)
                method = "direct"
                for s in expr.free_symbols:
                    if s != var:
                        inner_h, coeff_h = _extract_derivative(expr, s)
                        if inner_h is not None:
                            method = "leibniz"
                            break

        if method == "fundamental_theorem":
            # int df/dz dz = f(upper) - f(lower)
            inner, coeff = _extract_derivative(expr, var)
            if inner is None:
                raise ValueError(
                    f"No Derivative w.r.t. {var} found for fundamental theorem: {expr}"
                )
            f = coeff * inner
            f_upper = sp.Subs(f, var, upper)
            f_lower = sp.Subs(f, var, lower)
            return DepthIntegralResult(
                volume=Expression(S.Zero, f"ft_volume({self.name})"),
                boundary_upper=Expression(f_upper, f"ft_upper({self.name})"),
                boundary_lower=Expression(-f_lower, f"ft_lower({self.name})"),
            )

        elif method == "leibniz":
            # int df/dx dz = d/dx[int f dz] - f(upper)*d(upper)/dx + f(lower)*d(lower)/dx
            for s in list(expr.free_symbols) + [Symbol("x"), Symbol("t")]:
                if s == var:
                    continue
                inner, coeff = _extract_derivative(expr, s)
                if inner is not None:
                    break
            else:
                raise ValueError(f"No horizontal Derivative found for Leibniz: {expr}")

            # Volume: d/dx[int f dz]
            int_f = Integral(coeff * inner, (var, lower, upper))
            volume = Derivative(int_f, s)

            # Boundary terms: use Subs to keep evaluation explicit
            f = coeff * inner
            f_at_upper = sp.Subs(f, var, upper)
            f_at_lower = sp.Subs(f, var, lower)
            bnd_upper = -f_at_upper * Derivative(upper, s)
            bnd_lower = f_at_lower * Derivative(lower, s)

            return DepthIntegralResult(
                volume=Expression(volume, f"leibniz_volume({self.name})"),
                boundary_upper=Expression(bnd_upper, f"leibniz_upper({self.name})"),
                boundary_lower=Expression(bnd_lower, f"leibniz_lower({self.name})"),
            )

        else:  # direct
            return Expression(
                Integral(expr, (var, lower, upper)),
                f"integral({self.name})",
            )

    def subs(self, *args, **kwargs):
        return Expression(self.expr.subs(*args, **kwargs), self.name)

    def simplify(self):
        """Simplify: expand + cancel, preserving Integral and Derivative(Integral) terms."""
        return Expression(_simplify_preserve_integrals(self.expr), self.name,
                          term_groups=({role: _simplify_preserve_integrals(g)
                                        for role, g in self._term_groups.items()}
                                       if self._term_groups else None))

    def expand(self):
        return Expression(sp.expand(self.expr), self.name,
                          term_groups=({role: sp.expand(g)
                                        for role, g in self._term_groups.items()}
                                       if self._term_groups else None))

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

    # ------------------------------------------------------------------
    # Per-term operations
    # ------------------------------------------------------------------

    def map(self, fn):
        """Apply fn to each term, reassemble into a single Expression.

        fn receives an Expression (single term) and must return either
        an Expression or a DepthIntegralResult.  DepthIntegralResults are
        assembled (volume + boundaries) before summing.

        Example:
            integrated = expr.map(lambda t: t.depth_integrate(b, eta, z))
        """
        results = []
        for term in self.terms:
            r = fn(term)
            if isinstance(r, DepthIntegralResult):
                results.append(r.assemble())
            elif isinstance(r, Expression):
                results.append(r)
            else:
                results.append(Expression(r, term.name))
        return sum(results, Expression(S.Zero))

    def map_with_bcs(self, fn, bcs):
        """Like map(), but collects boundary terms and applies BCs globally.

        This is the correct way to depth-integrate a full equation:
        boundary terms from ALL terms are combined first, then BCs are
        applied once (so cross-term cancellations happen properly).

        Parameters
        ----------
        fn : callable
            Applied to each term.  Must return DepthIntegralResult or Expression.
        bcs : list of Relation
            Kinematic BCs etc. applied to the combined boundary expression.

        Returns
        -------
        Expression
            The fully depth-integrated equation with BCs applied.
        """
        total_volume = Expression(S.Zero)
        total_boundary = Expression(S.Zero)

        for term in self.terms:
            r = fn(term)
            if isinstance(r, DepthIntegralResult):
                total_volume = total_volume + r.volume
                total_boundary = total_boundary + r.boundary_upper + r.boundary_lower
            elif isinstance(r, Expression):
                total_volume = total_volume + r
            else:
                total_volume = total_volume + Expression(r)

        # Apply all BCs to the combined boundary
        bnd = total_boundary
        for bc in bcs:
            bnd = bnd.apply(bc)

        # Simplify: evaluate derivatives (to combine d(H+b)/dt - db/dt → dH/dt)
        # but preserve Integrals (don't re-apply Leibniz)
        result_expr = (total_volume + bnd).expr
        result_expr = _simplify_derivatives_only(result_expr)

        return Expression(result_expr, self.name)

    # ------------------------------------------------------------------
    # Term classification
    # ------------------------------------------------------------------

    def classify(self, t=None, x=None, z=None):
        """Classify each term by its role in the PDE.

        Returns a dict: {role: Expression} where role is one of:
        'temporal', 'convective', 'diffusive', 'source'.

        Detection rules:
        - Has d/dt → temporal
        - Has d/dx (first-order) of a product → convective flux
        - Has d²/dz² or d/dz of d/dz → diffusive
        - Otherwise → source (algebraic)
        """
        roles = {
            "temporal": [],
            "convective": [],
            "diffusive": [],
            "source": [],
        }

        for term in self.terms:
            e = term.expr
            classified = False

            # Check temporal
            if t is not None and e.has(Derivative) and any(
                t in d.variables for d in e.atoms(Derivative)
            ):
                roles["temporal"].append(term)
                classified = True

            # Check diffusive (second derivatives in z)
            if not classified and z is not None:
                for d in e.atoms(Derivative):
                    if d.variables.count(z) >= 2:
                        roles["diffusive"].append(term)
                        classified = True
                        break

            # Check convective (first derivative in x)
            if not classified and x is not None:
                for d in e.atoms(Derivative):
                    if x in d.variables and d.variables.count(x) == 1:
                        roles["convective"].append(term)
                        classified = True
                        break

            if not classified:
                roles["source"].append(term)

        return {k: Expression(sum((t.expr for t in v), S.Zero), k)
                for k, v in roles.items() if v}

    @property
    def temporal(self):
        """View: only temporal (d/dt) terms."""
        c = self.classify(t=Symbol("t"))
        return c.get("temporal", Expression(S.Zero))

    @property
    def convective(self):
        """View: only convective flux (d/dx) terms."""
        c = self.classify(x=Symbol("x"))
        return c.get("convective", Expression(S.Zero))

    # ------------------------------------------------------------------
    # Basis projection
    # ------------------------------------------------------------------

    def project_onto_basis(self, basis, level, field_map, z_var,
                           lower=None, upper=None, test_mode=None):
        """
        Project a depth-integrated equation onto a polynomial basis.

        Replaces every ``Integral(f(u,...), (z, b, eta))`` by substituting
        the basis expansion ``u(z) = sum alpha_k phi_k(zeta)`` and evaluating
        the resulting integrals using the ``SymbolicIntegrator``.

        If ``test_mode`` is an integer, multiplies each integral by the test
        function phi_{test_mode}(zeta) before evaluating (Galerkin projection
        for a specific mode).  If ``test_mode=None``, returns the scalar
        integral (e.g. for the mass equation where no test function is needed).

        Parameters
        ----------
        basis : Basisfunction class (e.g. Legendre_shifted)
        level : int
        field_map : dict
            Maps the original Function name to a list of SymPy Symbols
            for the basis coefficients.
            Example: {'u': [alpha_0, alpha_1, alpha_2]}
        z_var : Symbol
            The vertical coordinate (z) that appears in the integrals.
        lower, upper : sympy expressions (optional)
            The integration bounds (b, eta).  If None, detected from
            the first Integral found.
        test_mode : int or None
            If int, project onto test function phi_{test_mode}.

        Returns
        -------
        Expression
            With all depth integrals replaced by basis matrix products.
        """
        from zoomy_core.model.models.symbolic_integrator import SymbolicIntegrator
        from zoomy_core.model.models.projected_model import get_cached_matrices

        basis_obj = basis(level=level)
        integrator = SymbolicIntegrator(basis_obj)
        matrices = get_cached_matrices(basis, level, integrator)

        M = matrices["M"]
        A = matrices["A"]
        n = level + 1
        zeta = Symbol("zeta")
        c_mean = basis_obj.mean_coefficients()

        def _replace_integral(expr):
            """Walk the expression tree, replacing Integral nodes."""
            if not isinstance(expr, sp.Basic):
                return expr

            if isinstance(expr, Integral):
                integrand = expr.args[0]
                limits = expr.args[1]
                int_var = limits[0]

                if int_var != z_var:
                    return expr

                lo, hi = limits[1], limits[2]

                # Transform to zeta-space:
                # z = lo + (hi - lo)*zeta, dz = (hi-lo)*dzeta
                h_expr = hi - lo  # water depth H
                integrand_zeta = integrand.subs(int_var, lo + h_expr * zeta)

                # Substitute basis expansion for each field
                for fname, coeffs in field_map.items():
                    expansion = sum(
                        coeffs[k] * basis_obj.eval(k, zeta)
                        for k in range(min(len(coeffs), n))
                    )
                    # Find all applications of this function and replace
                    for atom in integrand_zeta.atoms(sp.Function):
                        if atom.func.__name__ == fname:
                            integrand_zeta = integrand_zeta.subs(atom, expansion)

                # Multiply by Jacobian H and test function
                integrand_final = h_expr * integrand_zeta
                if test_mode is not None:
                    integrand_final *= basis_obj.eval(test_mode, zeta)

                # Evaluate the integral using the integrator
                result = integrator.integrate(
                    sp.expand(integrand_final) * basis_obj.weight(zeta),
                    zeta,
                    tuple(basis_obj.bounds()),
                )
                return result

            # Recurse into Derivative, Mul, Add, etc.
            if isinstance(expr, Derivative):
                new_expr = _replace_integral(expr.args[0])
                return Derivative(new_expr, *expr.args[1:])

            if expr.args:
                new_args = [_replace_integral(a) for a in expr.args]
                return expr.func(*new_args)

            return expr

        result = _replace_integral(self.expr)
        return Expression(result, f"projected({self.name})")

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def latex(self, strip_args=False, multiline=False):
        """LaTeX representation.

        Parameters
        ----------
        strip_args : bool
            ``u(t,x,z)`` → ``u``. Partial derivatives preserved.
        multiline : bool
            Render as ``\\begin{aligned}`` with one group per line
            (requires ``term_groups``).
        """
        printer = _StripArgsLatexPrinter() if strip_args else None

        def _tex(expr):
            return printer.doprint(expr) if printer else sp.latex(expr)

        if multiline and self._term_groups:
            lines = []
            first = True
            for role, g in self._term_groups.items():
                if g == S.Zero:
                    continue
                tex = _tex(g)
                if first:
                    lines.append(f"  & \\underbrace{{{tex}}}_{{{role}}}")
                    first = False
                elif tex.startswith("-"):
                    lines.append(f"  & \\underbrace{{{tex}}}_{{{role}}}")
                else:
                    lines.append(f"  & + \\underbrace{{{tex}}}_{{{role}}}")
            return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n  &= 0\n\\end{aligned}"

        if multiline and not self._term_groups:
            # No term groups — split by additive terms for multiline rendering
            from sympy import Add
            terms = Add.make_args(self.expr)
            if len(terms) > 1:
                lines = []
                for i, term in enumerate(terms):
                    tex = _tex(term)
                    if i == 0:
                        lines.append(f"  & {tex}")
                    elif tex.startswith("-"):
                        lines.append(f"  & {tex}")
                    else:
                        lines.append(f"  & + {tex}")
                return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n  &= 0\n\\end{aligned}"

        if self._term_groups:
            # Render in group order (single line) — preserves physical ordering
            parts = []
            for role, g in self._term_groups.items():
                if g == S.Zero:
                    continue
                tex = _tex(g)
                if parts and not tex.startswith("-"):
                    tex = "+ " + tex
                parts.append(tex)
            return " ".join(parts)

        return _tex(self.expr)

    def describe(self, header=True, final_equation=True, parameters=False,
                 strip_args=False):
        """Composable description of this expression.

        Returns a ``Description`` that renders as markdown in Jupyter.

        Parameters
        ----------
        header : bool
            Show expression name + term count.
        final_equation : bool
            Show the symbolic equation.
        parameters : bool
            List free symbols.
        strip_args : bool
            Display ``u`` instead of ``u(t, x, z)``.
        """
        from zoomy_core.misc.description import Description

        parts = []

        if header:
            parts.append(f"**{self.name}** ({len(self)} terms)")

        if final_equation:
            tex = self.latex(strip_args=strip_args)
            parts.append(f"\n$$\n{tex} = 0\n$$")

        if parameters:
            from sympy import Symbol
            syms = sorted([s for s in self.expr.free_symbols
                          if isinstance(s, Symbol) and not s.is_Function],
                         key=str)
            if syms:
                sym_str = ", ".join(f"${sp.latex(s)}$" for s in syms)
                parts.append(f"\n**Parameters:** {sym_str}")

        return Description("\n".join(parts))


from sympy.printing.latex import LatexPrinter as _LatexPrinter


class _StripArgsLatexPrinter(_LatexPrinter):
    """LaTeX printer that renders function calls cleanly:

    - ``u(t,x,z)`` → ``u``  (standard args stripped)
    - ``u(t,x,b+h)`` → ``u|_{z=b+h}``  (boundary evaluation shown)
    - ``Subs(u(t,x,z), z, b)`` → ``u|_{z=b}``  (via sympy Subs printing)

    Horizontal functions (b, h, p_atm with fewer args) always stripped.
    """

    # The vertical coordinate symbol — functions with this as last arg are "standard"
    _z = sp.Symbol("z", real=True)

    def _print_Function(self, expr, exp=None):
        name = expr.func.__name__
        tex = self._deal_with_super_sub(name)
        args = expr.args

        # Check if this is a 3D function (u, w, tau_xx, ...) evaluated at a boundary
        # Heuristic: if last arg is NOT z and the function has 3+ args, it's a boundary eval
        if len(args) >= 3 and args[-1] != self._z:
            z_val = args[-1]
            z_tex = self.doprint(z_val)
            tex = r"\left. %s \right|_{\substack{ z=%s }}" % (tex, z_tex)

        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex


class DepthIntegralResult:
    """
    Result of depth-integrating a term over [lower, upper].

    Attributes:
        volume:          the volume integral (Expression)
        boundary_upper:  boundary term at z=upper (Expression)
        boundary_lower:  boundary term at z=lower (Expression)

    The full integral = volume + boundary_upper - boundary_lower.
    Kinematic BCs can be applied to the boundary terms via .apply_bcs().
    """

    def __init__(self, volume, boundary_upper, boundary_lower):
        self.volume = volume
        self.boundary_upper = boundary_upper
        self.boundary_lower = boundary_lower

    def apply_bcs(self, bc_lower=None, bc_upper=None):
        upper = self.boundary_upper
        lower = self.boundary_lower
        if bc_upper is not None:
            upper = upper.apply(bc_upper)
        if bc_lower is not None:
            lower = lower.apply(bc_lower)
        return DepthIntegralResult(self.volume, upper, lower)

    def assemble(self):
        """Combine all terms: volume + boundary_upper + boundary_lower.

        The sign convention is that each component already carries its
        correct sign.  For Leibniz: upper = -f(eta)*d(eta)/dx,
        lower = +f(b)*db/dx.  For fundamental theorem: upper = +f(eta),
        lower = -f(b).
        """
        return self.volume + self.boundary_upper + self.boundary_lower

    def __repr__(self):
        return (f"DepthIntegralResult(\n"
                f"  volume={self.volume},\n"
                f"  upper={self.boundary_upper},\n"
                f"  lower={self.boundary_lower}\n)")


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
# Operation: callable transformation applied to all equations
# ---------------------------------------------------------------------------

class Operation(SymbolicBase):
    """An operation that transforms an Expression (e.g. depth integration).

    Unlike a ``Relation`` (which substitutes symbols), an ``Operation``
    applies a structural transformation.  Works with both
    ``Expression.apply()`` and ``DerivedModel.apply()``.

    Subclasses override ``apply_to(expr)`` which receives and returns
    a sympy expression, or ``apply_to_expression(expression)`` which
    receives and returns an ``Expression`` object.
    """

    def __init__(self, name="", description=None):
        super().__init__(name)
        self.description = description or name

    def apply_to(self, expr):
        """Transform a sympy expression. Override for simple operations."""
        # Default: wrap in Expression, call apply_to_expression, unwrap
        eq = Expression(expr, "")
        result = self.apply_to_expression(eq)
        return result.expr

    def apply_to_expression(self, expression):
        """Transform an Expression object. Override for operations that
        need access to .terms, .map(), etc."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement apply_to or apply_to_expression"
        )

    def _repr_latex_(self):
        return ""


class DepthIntegrate(Operation):
    """Depth-integrate all equations over [b, b+h] w.r.t. z.

    Applies Leibniz rule and fundamental theorem term-by-term.
    Boundary values (w at z=b, u at z=b+h, etc.) remain as ``Subs``
    objects.  Apply ``ApplyKinematicBCs`` to see the cancellations.
    """

    def __init__(self, state):
        super().__init__(
            name="depth_integrate",
            description="Depth integration over [b, b+h] (Leibniz rule)",
        )
        self._state = state

    def apply_to_expression(self, eq):
        s = self._state
        return eq.map(lambda t: t.depth_integrate(s.b, s.eta, s.z))

    def _repr_latex_(self):
        s = self._state
        return (
            f"$\\int_{{{sp.latex(s.b)}}}^{{{sp.latex(s.eta)}}} "
            f"(\\cdot)\\, d{sp.latex(s.z)}$"
        )


class ApplyKinematicBCs(Operation):
    """Apply kinematic BCs globally to combined boundary terms.

    Evaluates ``Subs`` boundary values, applies kinematic BCs at
    surface and bottom, and simplifies. The Leibniz boundary u-terms
    cancel with the fundamental theorem w-terms.

    Must be applied immediately after ``DepthIntegrate``.
    """

    def __init__(self, state):
        super().__init__(
            name="kinematic_bcs",
            description="Kinematic BCs (surface + bottom): w = u·∂b/∂x + ∂b/∂t",
        )
        self._kbc_s = KinematicBCSurface(state)
        self._kbc_b = KinematicBCBottom(state)

    def apply_to(self, expr):
        # Evaluate only Subs objects (NOT Derivative(Integral))
        if expr.has(sp.Subs):
            subs_map = {s: s.doit() for s in expr.atoms(sp.Subs)}
            expr = expr.subs(subs_map)
        # Apply both BCs
        for bc in [self._kbc_s, self._kbc_b]:
            expr = bc.apply_to(expr)
        # Simplify (cancels d(b+h)/dt - db/dt → dh/dt)
        return _simplify_preserve_integrals(expr)

    def _repr_latex_(self):
        parts = [self._kbc_s._repr_latex_(), self._kbc_b._repr_latex_()]
        return " \\\\ ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# FullINS: 3D INS equations built from a StateSpace
# ---------------------------------------------------------------------------

class SimplifyIntegrals(Operation):
    """Evaluate integrals with constant integrand and remove zero integrals.

    - ``∫ 0 dz → 0``
    - ``∫ c dz → c·h`` (if integrand has no z-dependence)
    - ``∂/∂x ∫ 0 dz → 0``
    """

    def __init__(self, state):
        super().__init__(
            name="simplify_integrals",
            description="Evaluate constant/zero integrals",
        )

    def apply_to(self, expr):
        from sympy import Integral, Derivative

        def _simplify_int(e):
            if isinstance(e, Integral):
                integrand = e.args[0]
                limits = e.args[1]
                var = limits[0]
                lower, upper = limits[1], limits[2]
                if integrand == S.Zero:
                    return S.Zero
                if not integrand.has(var):
                    return integrand * (upper - lower)
                return e
            if isinstance(e, Derivative):
                inner = _simplify_int(e.args[0])
                if inner == S.Zero:
                    return S.Zero
                if inner != e.args[0]:
                    return Derivative(inner, *e.args[1:])
                return e
            if e.args:
                new_args = [_simplify_int(a) for a in e.args]
                if any(n != o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        return _simplify_int(expr)


class ZetaTransform(Operation):
    """Transform vertical coordinate: z = ζ·h + b, dz = h·dζ.

    Transforms integrals: ∫_b^{b+h} f(z) dz → h·∫_0^1 f(ζ·h+b) dζ.
    """

    def __init__(self, state):
        super().__init__(
            name="zeta_transform",
            description="Coordinate transform z = ζ·h + b",
        )
        self._z = state.z
        self._b = state.b
        self._h = state.H
        self._zeta = state.zeta

    def apply_to(self, expr):
        from sympy import Integral, Derivative
        z, b, h, zeta = self._z, self._b, self._h, self._zeta

        def _transform(e):
            if isinstance(e, Integral):
                integrand = e.args[0]
                limits = e.args[1]
                var = limits[0]
                if var == z:
                    new_integrand = integrand.subs(z, zeta * h + b) * h
                    return Integral(new_integrand, (zeta, S.Zero, S.One))
                return e
            if isinstance(e, Derivative):
                inner = _transform(e.args[0])
                if inner != e.args[0]:
                    return Derivative(inner, *e.args[1:])
                return e
            if e.args:
                new_args = [_transform(a) for a in e.args]
                if any(n != o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e

        return _transform(expr)

    def _repr_latex_(self):
        return f"$z = \\zeta \\cdot h + b, \\quad dz = h \\, d\\zeta$"


class ProductRule(Operation):
    """Inverse product rule: combine ``coeff · d(f)/dx`` into ``d(F)/dx``.

    Applied to a single term (use ``expr.terms[i].apply(ProductRule())``).
    Detects the derivative in the term, checks if the remaining coefficient
    can be integrated w.r.t. the differentiated variable, and combines.

    Example: ``g·h·∂h/∂x → ∂/∂x(g·h²/2)``
    """

    def __init__(self):
        super().__init__(
            name="product_rule",
            description="Inverse product rule: coeff·∂f/∂x → ∂(F)/∂x",
        )

    def apply_to(self, expr):
        from sympy import Derivative, Mul

        # Find the derivative factor in the term
        factors = Mul.make_args(expr)
        deriv_factor = None
        other_factors = []
        for f in factors:
            if isinstance(f, Derivative) and deriv_factor is None:
                deriv_factor = f
            else:
                other_factors.append(f)

        if deriv_factor is None:
            return expr  # no derivative found, nothing to do

        inner = deriv_factor.args[0]
        var = deriv_factor.variables[0]
        coeff = sp.Mul(*other_factors) if other_factors else S.One

        # Check if coeff is proportional to inner^n
        # Common case: coeff = c * inner → term = c * inner * d(inner)/dx = c * d(inner²/2)/dx
        if coeff.has(inner):
            ratio = sp.simplify(coeff / inner)
            if not ratio.has(inner):
                flux = ratio * inner**2 / 2
                return Derivative(flux, var)

        # General case: coeff doesn't contain inner
        # term = coeff * d(inner)/dx = d(coeff * inner)/dx - d(coeff)/dx * inner
        # Only useful if d(coeff)/dx = 0 (coeff is constant w.r.t. var)
        if not coeff.has(var):
            flux = coeff * inner
            return Derivative(flux, var)

        return expr  # can't simplify

    def _repr_latex_(self):
        return ""


class FullINS:
    """
    Full 3D Incompressible Navier-Stokes equations.

    Built from a StateSpace. Call as a function to get a ``DerivedSystem``
    with mutable ``.apply()``::

        ins = FullINS(state)       # builds equations
        system = ins.system()      # returns DerivedSystem
        system.apply(Hydrostatic(state))
        system.describe()

    Or use directly for individual equations::

        ins.continuity             # Expression
        ins.x_momentum             # Expression
    """

    def __init__(self, state: StateSpace):
        self.state = state
        self.dim = state.dim

    def _stress_divergence(self, row):
        s = self.state
        labels = ["x", "y", "z"] if s.has_y else ["x", "z"]
        coord_map = {"x": s.x, "y": s.y, "z": s.z}
        expr = S.Zero
        for j in labels:
            expr += Derivative(s.tau[row + j], coord_map[j])
        return expr

    @property
    def continuity(self):
        s = self.state
        expr = Derivative(s.u, s.x) + Derivative(s.w, s.z)
        if s.has_y:
            expr += Derivative(s.v, s.y)
        return Expression(expr, "continuity")

    def _momentum(self, vel, name, gravity=S.Zero):
        """Build a momentum equation with canonical term ordering.

        Order: temporal → convection → pressure → stress → source
        """
        s = self.state
        row = name.split("_")[0]  # "x", "y", "z"

        temporal = Derivative(vel, s.t)

        convection = Derivative(vel * s.u, s.x) + Derivative(vel * s.w, s.z)
        if s.has_y:
            convection += Derivative(vel * s.v, s.y)

        pressure = Rational(1, 1) / s.rho * Derivative(s.p, {"x": s.x, "y": s.y, "z": s.z}[row])
        stress = -Rational(1, 1) / s.rho * self._stress_divergence(row)

        groups = {"temporal": temporal, "convection": convection,
                  "pressure": pressure, "stress": stress}
        if gravity != S.Zero:
            groups["source"] = gravity

        full_expr = sum(groups.values())
        return Expression(full_expr, name, term_groups=groups)

    @property
    def x_momentum(self):
        return self._momentum(self.state.u, "x_momentum")

    @property
    def y_momentum(self):
        if not self.state.has_y:
            return None
        return self._momentum(self.state.v, "y_momentum")

    @property
    def z_momentum(self):
        return self._momentum(self.state.w, "z_momentum", gravity=self.state.g)

    @property
    def equations(self):
        eqs = [self.continuity, self.x_momentum]
        if self.state.has_y:
            eqs.append(self.y_momentum)
        eqs.append(self.z_momentum)
        return eqs

    def describe(self, header=True, final_equation=True, strip_args=False,
                 **kwargs):
        """Composable description of the full INS system.

        Returns a ``Description`` that renders as markdown in Jupyter.
        """
        from zoomy_core.misc.description import Description

        parts = []
        if header:
            dim_label = "3D (x,y,z)" if self.state.has_y else "2D (x,z)"
            parts.append(f"**Incompressible Navier-Stokes** ({dim_label})")

        if final_equation:
            for eq in self.equations:
                tex = eq.latex(strip_args=strip_args)
                parts.append(f"\n**{eq.name}:**\n$$\n{tex} = 0\n$$")

        return Description("\n".join(parts))

    def _repr_markdown_(self):
        return self.describe()._repr_markdown_()

    def system(self):
        """Return a ``DerivedSystem`` with all equations.

        The system has mutable ``.apply()`` for in-place transformations::

            system = FullINS(state).system()
            system.apply(HydrostaticPressure(state))
            system.apply(DepthIntegrate(state))
            system.describe()
        """
        from zoomy_core.model.models.derived_system import DerivedSystem
        eqs = {"continuity": self.continuity, "x_momentum": self.x_momentum}
        if self.state.has_y:
            eqs["y_momentum"] = self.y_momentum
        eqs["z_momentum"] = self.z_momentum
        return DerivedSystem("INS", eqs, self.state)


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
        u, w = s.u, s.w
        x, z = s.x, s.z
        subs = {
            s.tau["xx"]: 2 * mu * Derivative(u, x),
            s.tau["xz"]: mu * (Derivative(u, z) + Derivative(w, x)),
            s.tau["zx"]: mu * (Derivative(w, x) + Derivative(u, z)),
            s.tau["zz"]: 2 * mu * Derivative(w, z),
        }
        if s.has_y:
            v, y = s.v, s.y
            subs.update({
                s.tau["xy"]: mu * (Derivative(u, y) + Derivative(v, x)),
                s.tau["yx"]: mu * (Derivative(v, x) + Derivative(u, y)),
                s.tau["yy"]: 2 * mu * Derivative(v, y),
                s.tau["yz"]: mu * (Derivative(v, z) + Derivative(w, y)),
                s.tau["zy"]: mu * (Derivative(w, y) + Derivative(v, z)),
            })
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
        if s.has_y:
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
        if s.has_y:
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


class StressFreeSurface(Assumption):
    """Stress-free surface: τ·n|_{z=η} = 0.

    For a free surface with normal n ≈ (0,0,1), this gives:
      τ_xz|_{z=η} = 0,  τ_zz|_{z=η} = 0  (and τ_yz|_{z=η} = 0 in 3D)
    """

    def __init__(self, state: StateSpace):
        s = state
        subs = {}
        # τ_xz at surface = 0
        subs[s.tau["xz"].subs(s.z, s.eta)] = S.Zero
        # τ_zx at surface = 0 (symmetric)
        if "zx" in s.tau:
            subs[s.tau["zx"].subs(s.z, s.eta)] = S.Zero
        # τ_zz at surface = 0
        subs[s.tau["zz"].subs(s.z, s.eta)] = S.Zero
        if s.has_y:
            subs[s.tau["yz"].subs(s.z, s.eta)] = S.Zero
            if "zy" in s.tau:
                subs[s.tau["zy"].subs(s.z, s.eta)] = S.Zero
        super().__init__(subs, name="stress_free_surface")


class ZeroAtmosphericPressure(Assumption):
    """p_atm = 0 (no atmospheric pressure)."""

    def __init__(self, state: StateSpace):
        p_atm = Function("p_atm", real=True)(state.t, *state.coords_h)
        super().__init__({p_atm: S.Zero}, name="p_atm=0")


class assumptions:
    """Assumptions library. Usage: assumptions.kinematic_bc_bottom(state)"""
    kinematic_bc_bottom = KinematicBCBottom
    kinematic_bc_surface = KinematicBCSurface
    hydrostatic_pressure = HydrostaticPressure
    stress_free_surface = StressFreeSurface
    zero_atmospheric_pressure = ZeroAtmosphericPressure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simplify_derivatives_only(expr):
    """Simplify Derivative sums without touching Integrals.

    Turns ``d(H+b)/dt - db/dt`` into ``dH/dt`` while preserving
    ``Derivative(Integral(...), x)`` and bare ``Integral(...)`` nodes.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    # Protect anything involving Integrals
    integral_map = {}
    counter = [0]

    def _protect(e):
        # Protect Derivative(Integral(...), ...) as a unit
        if isinstance(e, Derivative) and e.args[0].has(Integral):
            key = sp.Dummy(f"_DINT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        # Protect bare Integrals
        if isinstance(e, Integral):
            key = sp.Dummy(f"_INT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if e.args:
            new_args = [_protect(a) for a in e.args]
            return e.func(*new_args)
        return e

    protected = _protect(expr)
    # Now safe to simplify — only pure Derivative(Function, var) remain
    simplified = protected.doit() if protected.has(Derivative) else protected
    # Restore integral-containing nodes
    return simplified.subs(integral_map)


def _simplify_preserve_integrals(expr):
    """Expand + cancel while protecting Integral and Derivative(Integral) terms.

    Cancels terms like u²·d(b+h)/dx - u²·db/dx → u²·dh/dx and
    (-gρ(b+h) + gρb + gρh)·... → 0, while leaving ∫...dz terms intact.

    Uses expand + powsimp (not simplify) to avoid common denominator wrapping.
    """
    if not isinstance(expr, sp.Basic):
        return expr

    integral_map = {}
    counter = [0]

    def _protect(e):
        if isinstance(e, Derivative) and e.args[0].has(Integral):
            key = sp.Dummy(f"_DINT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if isinstance(e, Integral):
            key = sp.Dummy(f"_INT{counter[0]}")
            integral_map[key] = e
            counter[0] += 1
            return key
        if e.args:
            new_args = [_protect(a) for a in e.args]
            return e.func(*new_args)
        return e

    protected = _protect(expr)
    # Evaluate simple derivatives like d(b+h)/dt → db/dt + dh/dt
    # but NOT Derivative(Integral) (already protected as dummies)
    evaled = protected.doit() if protected.has(Derivative) else protected
    # Expand to reveal cancellations, then collect like terms
    expanded = sp.expand(evaled)
    simplified = sp.powsimp(expanded, combine="all")
    return simplified.subs(integral_map)


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
