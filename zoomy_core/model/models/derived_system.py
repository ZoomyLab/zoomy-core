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
from typing import Dict, Optional, List

from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, Expression,
    KinematicBCBottom, KinematicBCSurface, HydrostaticPressure,
    Newtonian, Inviscid,
)


class DerivedSystem:
    """A reusable set of depth-integrated equations, basis-independent.

    Each equation is an ``Expression`` that may contain ``Integral(...)``
    terms.  These are resolved when you call ``.with_basis()``.

    Attributes
    ----------
    name : str
        Human-readable name (e.g. "SME", "VAM")
    equations : dict
        ``{name: Expression}`` — the depth-integrated equations
    state : StateSpace
        The shared symbolic state (coordinates, fields, etc.)
    assumptions : list of str
        Names of assumptions applied during derivation
    """

    def __init__(self, name, equations, state, assumptions=None):
        self.name = name
        self.equations = equations
        self.state = state
        self.assumptions = assumptions or []

    def with_material(self, material):
        """Apply a material model to all equations.

        Returns a new ``DerivedSystem`` with the material applied.
        The original is unchanged.
        """
        new_eqs = {k: eq.apply(material) for k, eq in self.equations.items()}
        return DerivedSystem(
            f"{self.name}+{material.name}",
            new_eqs,
            self.state,
            self.assumptions + [f"material={material.name}"],
        )

    def with_assumption(self, assumption):
        """Apply an additional assumption to all equations."""
        new_eqs = {k: eq.apply(assumption) for k, eq in self.equations.items()}
        a_name = assumption.name if hasattr(assumption, 'name') else str(assumption)
        return DerivedSystem(
            self.name,
            new_eqs,
            self.state,
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
        return f"DerivedSystem('{self.name}', [{eqs}], assumptions={self.assumptions})"


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

    return DerivedSystem("SME", {"mass": mass, "x_momentum": xmom},
                         state, assumptions)


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

    return DerivedSystem("VAM", {"mass": mass, "x_momentum": xmom,
                                  "z_momentum": zmom},
                         state, assumptions)
