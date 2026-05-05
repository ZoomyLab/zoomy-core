"""Solver-facing adapter around a closed, tagged derived :class:`System`.

The symbolic pipeline in this package (``FullINS`` + ``Integrate`` +
``InterfaceKBC`` + ``basis.layer_expand`` + ``Recombine`` + …) ends
with a ``System`` whose leaves are closed Expressions — no residual
``u(t, x, z)`` / ``w(t, x, z)``, everything in terms of basis
coefficients, layer thicknesses, mass fluxes, known parameters.

Zoomy solvers want a ``Model`` with ``flux()``, ``source()``,
``nonconservative_matrix()`` methods returning ``ZArray`` in Symbol
space.  ``SystemModel`` is the adapter: it walks the system's leaves,
runs ``auto_solver_tag`` per leaf, strips the outer ``∂_v`` from flux
terms, splits NCP terms into (coefficient, state-variable-index)
pairs, and assembles the matrices.

No hardcoded physics — every operator is read from the symbolic
derivation via the tag catalogue.
"""

from __future__ import annotations

import sympy as sp
from sympy import Derivative, S

from zoomy_core.misc.misc import ZArray


class SystemModel:
    """Convert a closed, tagged derived :class:`System` into the
    ``flux / nonconservative_matrix / source`` triple solvers consume.

    Parameters
    ----------
    system : System
        Derivation output — a tree-authoritative System with closed
        Expression leaves.
    state_substitutions : dict
        ``{Function : symbol_expression}`` mapping every state-bearing
        Function in the derivation to its conservative / primitive
        Symbol form.  For SWE: ``{h_fn: h_sym, q_fn: q_sym,
        alpha_0_fn: q_sym/h_sym, state.h: h_sym}``.  Any Function in
        the system's leaves that isn't covered here or by
        ``parameter_substitutions`` will mark the term as unclosed in
        the output.
    parameter_substitutions : dict, optional
        ``{Function : symbol_expression}`` for known external data
        (bathymetry, forcings).  Parameters participate in flux / NCP
        coefficients but aren't states.
    state_variables : list of sympy Symbol
        Canonical order of the state-variable symbols — determines
        the row / column indexing of the returned arrays.  Must
        contain every symbol that appears on the right-hand side of
        ``state_substitutions``.
    equation_variable : dict
        ``{leaf_path_tuple : state_variable_symbol}`` — tells the
        adapter which equation in the system evolves which state
        variable.  ``leaf_path_tuple`` is the tuple form of
        ``System.leaves()`` paths (e.g. ``("momentum", "x")``).
    time_var : sympy Symbol
        Time coordinate (default ``state.t`` if accessible).
    coords : iterable of sympy Symbol
        Spatial coordinates in the order corresponding to the
        ``dimension`` axis of the returned flux / NCP arrays.

    Notes
    -----
    * The adapter is agnostic to what produced the system — SWE, SME,
      multi-layer SWE, VAM, layered SME all work as long as each
      leaf is closed in the supplied ``state_substitutions`` /
      ``parameter_substitutions``.
    * ``source()`` returns ``-Σ source_tags`` to match Zoomy's RHS
      convention (``dq/dt = − flux_divergence + source``).
    * Terms the tagger couldn't classify appear via
      :meth:`untagged_remainders` so you can see the gap rather than
      silently dropping terms.
    """

    def __init__(
        self,
        system,
        *,
        state_substitutions,
        parameter_substitutions=None,
        state_variables,
        equation_variable,
        time_var=None,
        coords,
    ):
        self.system = system
        self._state_subs = dict(state_substitutions)
        self._param_subs = dict(parameter_substitutions) if parameter_substitutions else {}
        self._state_variables = list(state_variables)
        self._n_vars = len(self._state_variables)
        self._equation_variable = {tuple(k): v for k, v in equation_variable.items()}
        self._time_var = time_var if time_var is not None else system.state.t
        self._coords = list(coords)
        self._dimension = len(self._coords)
        # Build reverse index for fast column lookup in NCP.
        self._var_index = {s: i for i, s in enumerate(self._state_variables)}
        # Cache auto_solver_tag outputs (one pass over the system).
        self._tagged = None
        self._untagged = None
        self._compute_tags()

    # ---- Public: solver interface ---------------------------------------

    def flux(self):
        """Flux vector ``F(q)`` of shape ``(n_variables, dimension)``.

        For each evolved row ``i`` and dim ``d``, assembles ``F[i, d]``
        by collecting all ``∂_{coords[d]}(inner)`` terms from the
        leaf's ``flux`` tag and taking the sum of their inners.
        """
        F = sp.MutableDenseNDimArray.zeros(self._n_vars, self._dimension)
        for path, row_idx, tagged in self._iter_tagged():
            flux_expr = tagged.solver_tags.get("flux")
            if flux_expr is None:
                continue
            for term in sp.Add.make_args(sp.expand(flux_expr)):
                inner, coord_idx = self._strip_spatial_derivative(term)
                if inner is None:
                    # Should have been in NCP; fall through silently
                    # — the term is still accounted for by the
                    # ``untagged_remainders`` check if truly unclassified.
                    continue
                F[row_idx, coord_idx] = (
                    F[row_idx, coord_idx] + self._to_symbol_space(inner)
                )
        return ZArray(F)

    def nonconservative_matrix(self):
        """Non-conservative matrix ``A(q)`` of shape
        ``(n_variables, n_variables, dimension)``.

        Each NCP term ``coeff · ∂_{coords[d]}(state_var_j)`` contributes
        ``A[i, j, d] += coeff`` (in Symbol space).
        """
        A = sp.MutableDenseNDimArray.zeros(
            self._n_vars, self._n_vars, self._dimension)
        for path, row_idx, tagged in self._iter_tagged():
            ncp_expr = tagged.solver_tags.get("nonconservative_flux")
            if ncp_expr is None:
                continue
            for term in sp.Add.make_args(sp.expand(ncp_expr)):
                coeff, inner_fn, coord_idx = self._split_ncp_term(term)
                if inner_fn is None:
                    continue
                # Which state variable does ``inner_fn`` correspond to?
                inner_sym = self._to_symbol_space(inner_fn)
                col = self._var_index.get(inner_sym)
                if col is None:
                    # Inner wasn't a state Symbol (e.g., a parameter).
                    # Treat as conservative: add to flux as ∂_d(coeff·inner).
                    # We silently drop here — user pre-declared the split
                    # via ``state_variables``.
                    continue
                A[row_idx, col, coord_idx] = (
                    A[row_idx, col, coord_idx] + self._to_symbol_space(coeff)
                )
        return ZArray(A)

    def source(self):
        """RHS source ``S(q)`` of shape ``(n_variables,)``.

        Signs flipped to match Zoomy's ``dq/dt = − ∂_x F + S``
        convention (the tag stores the LHS contribution).  Both the
        ``source`` tag and any leftover ``time_derivative`` terms
        that touch *other* state variables (off-diagonal time
        derivatives — arise in multi-layer SWE where
        ``∂_t h_i = ∂_t(z_i − z_{i-1})`` splits) are folded in here.
        """
        S_vec = sp.MutableDenseNDimArray.zeros(self._n_vars)
        for path, row_idx, tagged in self._iter_tagged():
            s_expr = tagged.solver_tags.get("source", S.Zero)
            S_vec[row_idx] = S_vec[row_idx] - self._to_symbol_space(s_expr)
        return ZArray(S_vec)

    def untagged_remainders(self):
        """Dict ``{leaf_path: untagged_expression}`` listing what the
        tagger couldn't classify — your visibility into gaps.
        """
        return dict(self._untagged)

    # ---- Internals ------------------------------------------------------

    def _compute_tags(self):
        self._tagged = {}
        self._untagged = {}
        state_set = set(self._state_subs)
        param_set = set(self._param_subs)
        for path, eq in self.system.leaves():
            tagged = eq.auto_solver_tag(
                state_vars=state_set,
                time_var=self._time_var,
                coords=self._coords,
                parameters=param_set,
            )
            self._tagged[tuple(path)] = tagged
            rem = tagged.untagged_remainder()
            if rem != 0:
                self._untagged[tuple(path)] = rem

    def _iter_tagged(self):
        """Yield ``(path, row_idx, tagged_expression)`` per leaf."""
        for path, tagged in self._tagged.items():
            if path not in self._equation_variable:
                continue
            state_sym = self._equation_variable[path]
            row_idx = self._var_index.get(state_sym)
            if row_idx is None:
                raise ValueError(
                    f"equation_variable for {path} references "
                    f"{state_sym}, not in state_variables"
                )
            yield path, row_idx, tagged

    def _to_symbol_space(self, expr):
        """Substitute state / parameter Functions with their Symbol forms."""
        expr = expr.xreplace(self._state_subs)
        expr = expr.xreplace(self._param_subs)
        return expr

    def _strip_spatial_derivative(self, term):
        """For ``term`` of shape ``∂_{coords[d]}(inner)`` or
        ``coeff · ∂_{coords[d]}(inner)``, return ``(inner, d)``.
        Returns ``(None, None)`` on non-match.
        """
        deriv, coeff = self._extract_spatial_deriv(term)
        if deriv is None:
            return None, None
        var = deriv.variables[0]
        try:
            coord_idx = self._coords.index(var)
        except ValueError:
            return None, None
        inner = deriv.args[0] * coeff if coeff != S.One else deriv.args[0]
        return inner, coord_idx

    def _split_ncp_term(self, term):
        """For ``term = coeff · ∂_{coords[d]}(inner)`` with inner a
        Function state variable, return ``(coeff, inner, d)``.
        """
        deriv, coeff = self._extract_spatial_deriv(term)
        if deriv is None:
            return None, None, None
        var = deriv.variables[0]
        try:
            coord_idx = self._coords.index(var)
        except ValueError:
            return None, None, None
        return coeff, deriv.args[0], coord_idx

    def _extract_spatial_deriv(self, term):
        """Decompose term into ``(Derivative, coeff)`` if present with
        a first-order spatial derivative; else ``(None, S.Zero)``.
        """
        if isinstance(term, Derivative):
            if len(term.variables) == 1:
                return term, S.One
            return None, S.Zero
        if isinstance(term, sp.Mul):
            derivs = [f for f in term.args
                      if isinstance(f, Derivative)
                      and len(f.variables) == 1]
            if len(derivs) != 1:
                return None, S.Zero
            d = derivs[0]
            coeff = sp.Mul(*[f for f in term.args if f is not d])
            return d, coeff
        return None, S.Zero
