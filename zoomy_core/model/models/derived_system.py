"""
Reusable derived equation systems.

A ``DerivedSystem`` holds a set of depth-integrated equations that
are basis-independent.  You derive it once from the INS, then reuse it
with different bases, materials, or levels without re-deriving.

Usage:
    from zoomy_core.model.models.derived_system import DerivedSystem, sme, vam

    # Pre-derived SME (hydrostatic, basis-independent)
    system = sme()

    # Try different bases
    leg_eqs = system.with_basis(Legendre_shifted, level=2, field_map={"u": alphas})
    spl_eqs = system.with_basis(SplineBasis, level=3, field_map={"u": alphas})

    # Apply material later
    viscous = system.with_material(Newtonian(state))

    # Describe
    print(system.describe())
"""

import pickle
import warnings
from typing import Dict, Optional, List

import sympy as sp

from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Expression,
    KinematicBCBottom, KinematicBCSurface, HydrostaticPressure,
    Newtonian, Inviscid,
)


class _EquationProxy:
    """Proxy for in-place operations on a single equation in a DerivedSystem.

    ``system.z_momentum`` returns this proxy. Calling ``.apply()`` on it
    mutates the equation inside the system. All other attribute access
    (e.g. ``.expr``, ``.latex()``, ``.terms``) delegates to the underlying
    Expression.
    """

    def __init__(self, system, eq_name):
        object.__setattr__(self, '_system', system)
        object.__setattr__(self, '_name', eq_name)

    @property
    def _expr(self):
        return self._system.equations[self._name]

    def apply(self, *args, **kwargs):
        """Apply operation in place — mutates the equation in the system."""
        self._system.equations[self._name] = self._expr.apply(*args, **kwargs)

    def apply_to_term(self, index, *operations):
        """Apply operation to a specific term in place."""
        self._system.equations[self._name] = self._expr.apply_to_term(index, *operations)

    def simplify(self):
        """Simplify in place."""
        self._system.equations[self._name] = self._expr.simplify()

    def solve_for(self, variable):
        """Solve this equation (= 0) for the given variable.

        Returns an Expression representing ``variable = solution``.
        If multiple solutions, uses the first and warns.
        """
        solutions = sp.solve(self._expr.expr, variable)
        if not solutions:
            raise ValueError(f"Cannot solve {self._name} for {variable}")
        if len(solutions) > 1:
            warnings.warn(
                f"Multiple solutions for {variable} in {self._name}, using first: {solutions[0]}"
            )
        return Expression(solutions[0], f"{variable}")

    def delete(self):
        """Remove this equation from the system."""
        del self._system.equations[self._name]

    def __getattr__(self, name):
        # Delegate everything else to the underlying Expression
        return getattr(self._expr, name)

    def __repr__(self):
        return repr(self._expr)

    def _repr_latex_(self):
        return self._expr._repr_latex_()

    def __len__(self):
        return len(self._expr)


class System:
    """A mutable system of symbolic PDE equations.

    Equations are ``Expression`` objects accessed by name via dot syntax
    (``system.continuity``) or dict syntax (``system.equations["continuity"]``).

    Operations are applied in place::

        system = System("INS", state)
        system.add_equation("continuity", du_dx + dw_dz)
        system.add_equation("x_momentum", ..., term_groups={...})
        system.apply(HydrostaticPressure(state))
        system.x_momentum.apply_to_term(5, ProductRule())
        system.describe()

    Attributes
    ----------
    name : str
        Human-readable name (e.g. "INS", "SME")
    equations : dict
        ``{name: Expression}``
    state : StateSpace
        The shared symbolic state
    assumptions : list of str
        Operations applied during derivation
    """

    def __init__(self, name, state, equations=None, assumptions=None):
        self.name = name
        self.state = state
        self.equations = dict(equations) if equations else {}
        self.assumptions = assumptions or []

    def add_equation(self, name, expr, term_groups=None):
        """Add an equation to the system.

        Parameters
        ----------
        name : str
            Equation name (e.g. "continuity", "x_momentum").
        expr : Expression or sympy expr
            The equation (= 0).
        term_groups : dict, optional
            Named term groups for ordered display.
        """
        if isinstance(expr, Expression):
            self.equations[name] = expr
        else:
            self.equations[name] = Expression(expr, name, term_groups=term_groups)

    def __getattr__(self, name):
        # Dot access to equations: system.x_momentum, system.continuity, etc.
        if name.startswith("_") or name in ("name", "equations", "state", "assumptions"):
            raise AttributeError(name)
        if name in self.equations:
            return _EquationProxy(self, name)
        raise AttributeError(f"No equation '{name}' in system. Available: {list(self.equations.keys())}")

    def apply(self, operation):
        """Apply an operation or relation to all equations in place."""
        self.equations = {
            k: eq.apply(operation) for k, eq in self.equations.items()
        }
        a_name = getattr(operation, 'description', None) or getattr(operation, 'name', str(operation))
        self.assumptions.append(a_name)

    def apply_to_term(self, equation_name, term_index, *operations):
        """Apply operations to a specific term of a specific equation in place.

        Usage::

            system.apply_to_term("x_momentum", 5, ProductRule())
        """
        self.equations[equation_name] = self.equations[equation_name].apply_to_term(
            term_index, *operations
        )
        for op in operations:
            a_name = getattr(op, 'description', None) or getattr(op, 'name', str(op))
            self.assumptions.append(f"{a_name} on {equation_name}[{term_index}]")

    def with_material(self, material):
        """Apply a material model. Returns a new DerivedSystem (immutable)."""
        new_eqs = {k: eq.apply(material) for k, eq in self.equations.items()}
        return System(
            f"{self.name}+{material.name}",
            self.state,
            new_eqs,
            self.assumptions + [f"material={material.name}"],
        )

    def with_assumption(self, assumption):
        """Apply an assumption. Returns a new System (immutable)."""
        new_eqs = {k: eq.apply(assumption) for k, eq in self.equations.items()}
        a_name = assumption.name if hasattr(assumption, 'name') else str(assumption)
        return System(
            self.name,
            self.state,
            new_eqs,
            self.assumptions + [a_name],
        )

    def with_basis(self, basis, level, field_map, z_var=None, test_mode=None):
        """Project all equations onto a basis.

        Parameters
        ----------
        basis : Basisfunction class
        level : int
        field_map : dict
            ``{field_name: [alpha_0, alpha_1, ...]}``
        z_var : Symbol, optional
            Vertical coordinate (default: ``self.state.z``)
        test_mode : int or None
            Galerkin test function mode

        Returns
        -------
        dict of Expression
            Projected equations with basis matrix products
        """
        z = z_var or self.state.z
        return {
            name: eq.project_onto_basis(basis, level, field_map, z, test_mode=test_mode)
            for name, eq in self.equations.items()
        }

    def describe(self, header=True, assumptions=True, final_equation=True,
                 parameters=False, strip_args=False):
        """Composable description of this equation system.

        Returns a ``Description`` that renders as markdown in Jupyter.

        Parameters
        ----------
        header : bool
            System name and equation list.
        assumptions : bool
            List assumptions applied.
        final_equation : bool
            Show equations.
        parameters : bool
            List free symbols.
        strip_args : bool
            Display ``u`` instead of ``u(t, x, z)``.
        """
        from zoomy_core.misc.description import Description
        import sympy as sp

        parts = []

        if header:
            eq_names = ", ".join(self.equations.keys())
            parts.append(f"**{self.name}** ({eq_names})\n")

        if assumptions and self.assumptions:
            parts.append("**Assumptions:** " + ", ".join(self.assumptions) + "\n")

        if final_equation:
            for eq_name, eq in self.equations.items():
                # Use multiline if term groups are available
                tex = eq.latex(strip_args=strip_args, multiline=bool(eq._term_groups))
                parts.append(f"**{eq_name}:**\n$$\n{tex} = 0\n$$\n")

        if parameters:
            all_syms = set()
            for eq in self.equations.values():
                all_syms |= eq.expr.free_symbols
            syms = sorted([s for s in all_syms if isinstance(s, sp.Symbol)], key=str)
            if syms:
                sym_str = ", ".join(f"${sp.latex(s)}$" for s in syms)
                parts.append(f"**Parameters:** {sym_str}")

        return Description("\n".join(parts))

    def _repr_markdown_(self):
        return self.describe()._repr_markdown_()

    def save(self, path):
        """Save to file for reuse without re-derivation."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load a previously saved system."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        eqs = ", ".join(f"{k}({len(v)} terms)" for k, v in self.equations.items())
        return f"System('{self.name}', [{eqs}], assumptions={self.assumptions})"


# Backward-compatible alias
DerivedSystem = System


# =========================================================================
# Pre-built factory functions
# =========================================================================

def sme(state=None, material=None):
    """Derive the depth-integrated SME (hydrostatic shallow moment equations).

    Returns a ``DerivedSystem`` that is basis-independent. Apply a basis
    later with ``.with_basis()``, or add viscosity with ``.with_material()``.

    Parameters
    ----------
    state : StateSpace, optional
        Default: ``StateSpace(dimension=2)`` (xz plane)
    material : Relation, optional
        If provided, applied before depth integration

    Returns
    -------
    DerivedSystem
    """
    state = state or StateSpace(dimension=2)
    ins = FullINS(state)
    z = state.z
    b, eta = state.b, state.eta

    kbc_b = KinematicBCBottom(state)
    kbc_s = KinematicBCSurface(state)
    hydro = HydrostaticPressure(state)

    assumptions = ["hydrostatic"]

    # x-momentum: apply hydrostatic + optional material
    xm = ins.x_momentum.apply(hydro)
    if material:
        xm = xm.apply(material)
        assumptions.append(f"material={material.name}")

    # Depth-integrate
    mass = ins.continuity.map_with_bcs(
        lambda t: t.depth_integrate(b, eta, z),
        bcs=[kbc_s, kbc_b],
    )
    xmom = xm.map_with_bcs(
        lambda t: t.depth_integrate(b, eta, z),
        bcs=[kbc_s, kbc_b],
    )

    return System("SME", state, {"mass": mass, "x_momentum": xmom}, assumptions)


def vam(state=None, material=None):
    """Derive the depth-integrated VAM (non-hydrostatic, with z-momentum).

    Returns a ``DerivedSystem`` with mass, x-momentum, and z-momentum
    equations.  Pressure is NOT substituted — it remains as an unknown.

    Parameters
    ----------
    state : StateSpace, optional
    material : Relation, optional
        Default: Inviscid

    Returns
    -------
    DerivedSystem
    """
    state = state or StateSpace(dimension=2)
    ins = FullINS(state)
    z = state.z
    b, eta = state.b, state.eta

    kbc_b = KinematicBCBottom(state)
    kbc_s = KinematicBCSurface(state)

    material = material or Inviscid(state)
    assumptions = ["non_hydrostatic", f"material={material.name}"]

    xm = ins.x_momentum.apply(material)
    zm = ins.z_momentum.apply(material)

    mass = ins.continuity.map_with_bcs(
        lambda t: t.depth_integrate(b, eta, z),
        bcs=[kbc_s, kbc_b],
    )
    xmom = xm.map_with_bcs(
        lambda t: t.depth_integrate(b, eta, z),
        bcs=[kbc_s, kbc_b],
    )
    zmom = zm.map_with_bcs(
        lambda t: t.depth_integrate(b, eta, z),
        bcs=[kbc_s, kbc_b],
    )

    return System("VAM", state,
                  {"mass": mass, "x_momentum": xmom, "z_momentum": zmom},
                  assumptions)
