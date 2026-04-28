"""Solve a system of algebraic constraints jointly + substitute.

Given a PDESystem with algebraic constraint equations that determine
some fields (e.g. ``w_0..w_N`` from the continuity j>=1 + KBC bottom
projections), this module:

  1. Solves the constraint subset as a single linear system in the
     constrained fields (using ``sp.solve``).
  2. Substitutes the solution into the remaining (evolution)
     equations.

The result is a reduced PDESystem on the evolved fields only — what
the standard derivations (K&T 2019, Aguillon 2026, Escalante 2024 eq 4)
produce by construction.

Why this matters: setting constrained fields to zero (as
``w_N_as_input=True`` does in vam_builder) gives a different system,
because the constraints aren't enforced — they're just ignored.
Solving the constraints jointly produces the algebraically-equivalent
reduced system that has the SAME spectral structure as standard
formulations.
"""
from __future__ import annotations

from typing import List, Optional

import sympy as sp


def eliminate_constraints(
    differential_eqs: List[sp.Expr],
    constraint_eqs: List[sp.Expr],
    constrained_fields: List,
    *,
    apply_dt_h_rule: Optional[dict] = None,
    verbose: bool = False,
) -> List[sp.Expr]:
    """Solve ``constraint_eqs`` for ``constrained_fields`` jointly,
    substitute into ``differential_eqs``, return the reduced system.

    Args:
        differential_eqs: list of evolution equations (each = 0).
        constraint_eqs:   list of algebraic constraint equations
                          (each = 0; no time derivatives expected).
        constrained_fields: list of Function-call atoms to solve for.
                            Length must equal ``len(constraint_eqs)``.
        apply_dt_h_rule:    optional substitution dict (e.g. cont j=0
                            → ∂_t h substitution) applied to BOTH
                            constraints and differential eqs before
                            solving.  Use this to remove residual
                            ∂_t h atoms in the constraints (which
                            otherwise would prevent ``sp.solve`` from
                            finding a clean solution).
        verbose:            print the solution found.

    Returns:
        the differential_eqs with constrained_fields substituted.
    """
    if len(constraint_eqs) != len(constrained_fields):
        raise ValueError(
            f"#constraint_eqs ({len(constraint_eqs)}) must equal "
            f"#constrained_fields ({len(constrained_fields)})."
        )

    # Apply dt_h substitution if provided.
    if apply_dt_h_rule:
        def _subst(eq):
            prev = None
            cur = sp.expand(eq.doit())
            while prev != cur:
                prev = cur
                cur = sp.expand(cur.xreplace(apply_dt_h_rule).doit())
            return cur
        constraints = [_subst(c) for c in constraint_eqs]
        diff_eqs = [_subst(d) for d in differential_eqs]
    else:
        constraints = list(constraint_eqs)
        diff_eqs = list(differential_eqs)

    # Solve constraints as a linear system in constrained_fields.
    sol = sp.solve(constraints, constrained_fields)
    if not sol:
        raise ValueError(
            "sp.solve returned empty — constraints may be inconsistent "
            "or the chosen fields aren't determined by the constraints."
        )
    if isinstance(sol, list):
        # sp.solve returns a list of dicts when there are multiple solutions.
        sol = sol[0]
    if verbose:
        print(f"Eliminated {len(constrained_fields)} fields:")
        for f in constrained_fields:
            print(f"  {f} = {sp.simplify(sol[f])}")

    # Substitute into the evolution equations.
    return [sp.expand(eq.xreplace(sol).doit()) for eq in diff_eqs]
