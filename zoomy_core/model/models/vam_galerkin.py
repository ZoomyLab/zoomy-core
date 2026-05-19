"""VAMModelGalerkin — VAM model derived via the explicit Galerkin chain.

The chain follows Escalante 2024's cont-projection formulation
(see ``thesis/chapters/derivation_vam.md`` §5.7 and §5.5–§5.6 for the
worked derivation):

  1. Project momentum AND continuity against shifted-Legendre test
     functions.
  2. ProductRule on ``[t, x, z]`` to expose ``∂_t / ∂_x`` boundary
     atoms via Leibniz.
  3. Integrate over ``z ∈ [b, η]``.
  4. Apply the kinematic boundary conditions at ``z=b`` and ``z=η``.
  5. Affine map ``z → ζ ∈ [0, 1]``; expand bulk fields into modes.
  6. Resolve polynomial ζ integrals via the basis cache.
  7. Close ``W_{N_w}`` via the bottom KBC at the basis level and
     ``P_{N_p}`` via the surface BC (both as algebraic substitutions).
  8. Substitute the mass equation into the cont-projection rows so
     they become purely algebraic (DAE structure).
  9. Auto-tag every row with canonical solver tags.

The result is the **System tree** ``self._chain_system`` whose leaves
carry the tagged Expression objects.  This is the *single* canonical
representation of the model.  No PDESystem, no intermediate flat
container — everything else (``flux()``, ``source()``, the
``SystemModel`` view via ``SystemModel.from_model(m)``) is derived
from the tagged tree.

For ``(M=1, N_w=2, N_p=2)`` the closed system has 7 fields
``(h, U_0, U_1, W_0, W_1, P_0, P_1)`` and 7 equations
``(mass, xmom_j0, xmom_j1, zmom_j0, zmom_j1, cont_j1, cont_j2)``
— 5 evolution rows + 2 algebraic cont-projection rows.
"""

from __future__ import annotations

import copy
from typing import Dict

import param
import sympy as sp

from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.basemodel import Model
from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.ins_generator import (
    AffineProjection,
    EvaluateIntegrals,
    Expand,
    Expression,
    FullINS,
    InterfaceKBC,
    Integrate,
    Inviscid,
    Multiply,
    ProductRule,
    StateSpace,
)
from zoomy_core.model.models.tag_extraction import (
    auto_solver_tag, collect_solver_tag,
)


class _TaggedEquationHolder:
    """Duck-typed wrapper exposing ``.equations: dict[name, Expression]``
    to ``collect_solver_tag``.

    The chain System tree stores leaves under hierarchical paths
    (``continuity.test_0``, ``momentum.x.test_k``, etc.); the operator-API
    methods on :class:`VAMModelGalerkin` flatten those into a named
    dict keyed by canonical row names (``mass``, ``xmom_j0``, …) that
    :func:`collect_solver_tag` can iterate.
    """

    def __init__(self, equations: Dict[str, Expression]):
        self.equations = equations


class VAMModelGalerkin(Model):
    """VAM derived via the explicit symbolic Galerkin chain.

    Direct subclass of :class:`Model` — no VAMModel inheritance, no
    legacy 6-state operator API.  The full state is the chain DAE's
    primitive state ``(h, U_0..U_M, W_0..W_{N_w-1}, P_0..P_{N_p-1})``.
    """

    level = param.Integer(default=0, doc="Vertical basis function order")
    quadratic_form = param.Selector(
        default="cantero_chinchilla",
        objects=["cantero_chinchilla", "escalante"],
        doc=(
            "Symbolic form of the j ≥ 1 momentum rows.  "
            "``cantero_chinchilla`` (default): the un-reduced "
            "Galerkin projection (Cantero-Chinchilla, Castro-Orgaz & "
            "Khan 2018 eq 24) with explicit ``W²`` content after "
            "modal closure of ``W_{N_w}``.  Fits the standard "
            "``flux + NCP + source`` operator decomposition.  "
            "``escalante``: additionally applies "
            "``W_k·cont_jk = 0`` constraint identities to convert "
            "``W²`` into linear-W content and clean up ``∂_t h`` "
            "cross-terms, matching Escalante 2024 JCP eq (4) on "
            "the j = 0 rows bit-for-bit (and modulo "
            "{cont_j1, cont_j2} on j = 1)."
        ),
    )

    def __init__(self, level=0, *, M=None, N_w=None, N_p=None,
                 dimension=2,
                 quadratic_form="cantero_chinchilla",
                 eigenvalue_mode="symbolic", **kwargs):
        if dimension not in (2, 3):
            raise ValueError(
                f"VAMModelGalerkin: dimension must be 2 (1D-horizontal) "
                f"or 3 (2D-horizontal); got {dimension}"
            )
        self._chain_M = M if M is not None else level
        self._chain_N_w = N_w if N_w is not None else self._chain_M + 1
        self._chain_N_p = N_p if N_p is not None else self._chain_M + 1
        self._quadratic_form = quadratic_form
        # ``self._dim`` is the chain (StateSpace) dimension; 2 = 1D-
        # horizontal, 3 = 2D-horizontal.  Stored privately because
        # ``Model.__init__`` later sets ``self.dimension`` to the
        # number of horizontal dims (``self._dim - 1``).
        self._dim = dimension

        # Build the chain System tree.  Populates self._chain_system,
        # self._chain_intermediate, self._equations, plus the helpers
        # self._state_funcs / self._state_syms / self._func_to_sym.
        self._build_chain()

        # Variable layout (Symbols, no time/space args).
        n_u_modes = self._chain_M + 1            # U_0..U_M
        n_w_modes_active = self._chain_N_w       # W_0..W_{N_w-1} (W_{N_w} closed)
        n_p_modes_active = self._chain_N_p       # P_0..P_{N_p-1} (P_{N_p} closed)
        var_names = ["h"]
        # Horizontal velocity modes: (U_k, V_k, …) one set per horizontal
        # coordinate.  ``self._dim`` is the StateSpace dimension
        # (chain dim = n_horizontal + 1 for the vertical z).
        n_horiz = self._dim - 1
        for label in ["U", "V"][:n_horiz]:
            var_names += [f"{label}_{k}" for k in range(n_u_modes)]
        var_names += [f"W_{k}" for k in range(n_w_modes_active)]
        # ``b`` enters as a state with the trivial evolution ``∂_t b = 0``
        # — the OLD JAX prototype's structure that makes gravity NCP
        # cleanly well-balancing and lets the cont_j ``b_x · q_U0``-type
        # forcing flow through the NCP path-integral instead of as a
        # cell-centre source.  See `_build_chain` for the matching
        # ``state.b`` insertion in ``_chain_state_funcs`` and the
        # ``bathymetry: ∂_t b = 0`` equation added to ``ordered``.
        var_names += ["b"]
        var_names += [f"P_{k}" for k in range(n_p_modes_active)]

        param_dict = {
            "g":   (9.81, "positive"),
            "ez":  (1.0,  "positive"),
            "rho": (1000.0, "positive"),
        }

        Model.__init__(
            self,
            init_functions=False,
            dimension=self._dim - 1,
            variables=var_names,
            aux_variables=0,
            parameters=param_dict,
            eigenvalue_mode=eigenvalue_mode,
            level=level,
            **kwargs,
        )
        # ``h`` is positive by physics; re-mint it with that assumption.
        h_old = self.variables["h"]
        h_new = sp.Symbol(h_old.name, positive=True, real=True)
        self.variables["h"] = h_new
        self._refresh_state_sym_map(h_new=h_new)

        self._initialize_functions()

    # ------------------------------------------------------------------
    # Chain construction.
    # ------------------------------------------------------------------

    def _build_chain(self):
        """Build the chain System tree + the named-equation dict.

        Stages (mirrors derive_model in escalante2024_derivation.py
        and the documented 12-step recipe in
        ``thesis/chapters/derivation_vam.md`` §5.5):

          1.  3D INS, drop viscosity, split off hydrostatic pressure.
          2.  Project continuity AND momentum onto test functions.
          3.  ProductRule on ``[t, x, z]``.
          4.  Depth-integrate (Leibniz on ∂_t / ∂_x; FT on ∂_z).
          5.  Apply kinematic BCs at bottom + surface; drop ``∂_t b``.
          6.  Surface BC for ``p_NH``.
          7.  Affine map + ansatz expansion.
          8.  Resolve ζ integrals (boundary atoms collapsed by basis
              cache).
          9.  Modal closures: solve bottom KBC for ``W_{N_w}``;
              solve surface BC for ``P_{N_p}``.
          10. Substitute mass equation into the cont-projection rows.
          11. Auto-tag every row.
        """
        M_ = self._chain_M
        N_w = self._chain_N_w
        N_p = self._chain_N_p

        state = StateSpace(dimension=self._dim)
        z = state.z
        basis_u = Legendre_shifted(level=M_, symbol="phi_u")
        basis_w = Legendre_shifted(level=N_w, symbol="phi_w")
        basis_p = Legendre_shifted(level=N_p, symbol="phi_p")

        # Horizontal coefficient args: (t, x) for 1D, (t, x, y) for 2D.
        h_args = (state.t, *state.coords_h)
        # One coefficient list per horizontal velocity component, in the
        # order (U, V, …) matching state.velocities_h.  Labels live with
        # the data so loops below stay generic.
        horiz_labels = ["U", "V"][:len(state.coords_h)]
        coeffs_horiz = [
            [sp.Function(f"{lbl}_{k}", real=True)(*h_args)
             for k in range(M_ + 1)]
            for lbl in horiz_labels
        ]
        coeffs_u = coeffs_horiz[0]    # x-momentum coefficients
        coeffs_w = [sp.Function(f"W_{k}", real=True)(*h_args)
                    for k in range(N_w + 1)]
        coeffs_p = [sp.Function(f"P_{k}", real=True)(*h_args)
                    for k in range(N_p + 1)]

        test_phi_u = Zstruct(
            **{f"phi_{k}": basis_u.phi[k](state.zeta)
               for k in range(M_ + 1)})
        test_phi_w = Zstruct(
            **{f"phi_{k}": basis_w.phi[k](state.zeta)
               for k in range(N_w)})
        test_phi_cont = Zstruct(
            **{f"phi_{k}": basis_p.phi[k](state.zeta)
               for k in range(N_p + 1)})

        # 1. 3D INS, drop viscosity, split off hydrostatic pressure.
        sys = FullINS(state)
        sys.apply(Inviscid(state)).simplify()
        p_NH = sp.Function("p_NH", real=True)(state.t, *state.coords_h, z)
        sys.apply({state.p: state.rho * state.g * (state.eta - z) + p_NH}
                  ).simplify()

        # 2. Project continuity AND momentum onto test functions.  We
        # multiply every horizontal-momentum branch by ``test_phi_u``
        # (V uses the same basis as U).
        sys.continuity.apply(Multiply(test_phi_cont, outer=True))
        for coord in state.coords_h:
            getattr(sys.momentum, str(coord)).apply(
                Multiply(test_phi_u, outer=True))
        sys.momentum.z.apply(Multiply(test_phi_w, outer=True))

        # 3. ProductRule on [t, *coords_h, z].
        sys.apply(ProductRule(variables=[state.t, *state.coords_h, z]))

        # 4. Depth-integrate.
        sys.apply(Integrate(z, state.b, state.eta, method="auto"))

        # 5. Kinematic BCs at bottom + surface; drop ∂_t b.
        sys.apply(InterfaceKBC(state, state.b)).simplify()
        sys.apply(InterfaceKBC(state, state.eta)).simplify()
        sys.apply({sp.Derivative(state.b, state.t): sp.S.Zero}).simplify()

        # 6. Surface BC for p_NH at the field level.
        sys.apply({p_NH.subs(z, state.eta): 0}).simplify()

        # 7. Affine map + ansatz expansion.
        sys.apply(AffineProjection(state, rewrite_basis_args=False))
        for vel, coeffs in zip(state.velocities_h, coeffs_horiz):
            sys.apply(Expand(vel, basis=basis_u,
                             coefficients=coeffs, state=state))
        sys.apply(Expand(state.w, basis=basis_w, coefficients=coeffs_w,
                         state=state))
        sys.apply(Expand(p_NH, basis=basis_p, coefficients=coeffs_p,
                         state=state))

        # Snapshot Sum-form intermediate.
        self._chain_intermediate = copy.deepcopy(sys)

        # 8. Resolve ζ integrals; boundary atoms collapse via basis cache.
        sys.apply(EvaluateIntegrals(state)).simplify()
        sys.apply({sp.Derivative(state.b, state.t): sp.S.Zero}).simplify()

        # 9a. Bottom KBC modal closure: solve for W_{N_w}.
        # KBC at z = b:  w(b) = Σ_d  vel_d(b) · ∂_d b.
        w_at_b = sum(coeffs_w[k] * basis_w.eval(k, sp.S.Zero)
                     for k in range(N_w + 1))
        bot_kbc = w_at_b
        for coord, coeffs_d in zip(state.coords_h, coeffs_horiz):
            vel_at_b = sum(coeffs_d[k] * basis_u.eval(k, sp.S.Zero)
                           for k in range(M_ + 1))
            bot_kbc = bot_kbc - vel_at_b * sp.Derivative(state.b, coord).doit()
        w_top_sol = sp.solve(bot_kbc, coeffs_w[N_w])[0]
        sys.apply({coeffs_w[N_w]: w_top_sol}).simplify()

        # 9b. Surface BC modal closure: solve for P_{N_p}.
        p_at_eta = sum(coeffs_p[k] * basis_p.eval(k, sp.S.One)
                       for k in range(N_p + 1))
        p_top_sol = sp.solve(p_at_eta, coeffs_p[N_p])[0]
        sys.apply({coeffs_p[N_p]: p_top_sol}).simplify()

        # Save the chain System tree and associated metadata.
        self._chain_system = sys
        self._chain_state = state
        self._chain_coeffs = {
            "u": coeffs_u, "w": coeffs_w[:N_w], "p": coeffs_p[:N_p],
        }

        # 10. Substitute mass equation into cont_jk rows.  Continuity
        # test_0 is the mass evolution; the higher cont rows would
        # otherwise carry compound ∂_t(c·h) atoms.  After substituting
        # ∂_t h = ‒∂_x(h·U_0), those rows are purely algebraic.
        mass_leaf = sys._tree.continuity.test_0
        mass_expr = mass_leaf.expr
        dt_h_relation = mass_leaf.solve_for(
            sp.Derivative(state.h, state.t))
        dt_h_sub = dt_h_relation._as_relation

        # Assemble named equations in canonical order:
        # mass; (xmom, ymom)_j0..jM; zmom_j0..jN_w-1; cont_j1..jN_p.
        ordered = [("mass", mass_expr)]
        for coord in state.coords_h:
            coord_name = str(coord)
            mom_branch = getattr(sys._tree.momentum, coord_name)
            for k in range(M_ + 1):
                ordered.append((f"{coord_name}mom_j{k}",
                                getattr(mom_branch, f"test_{k}").expr))
        for k in range(N_w):
            ordered.append((f"zmom_j{k}",
                            getattr(sys._tree.momentum.z,
                                    f"test_{k}").expr))
        # Bathymetry-as-state with trivial evolution ``∂_t b = 0``.
        # Inserted AFTER the tree-level ``∂_t b → 0`` substitutions
        # (lines ~256 and ~276 above) so this leaf's ``∂_t b`` atom
        # survives intact; the auto-tagger below routes it to a unit
        # mass-matrix entry on the b column.  Placed before cont_j
        # rows so the equation_names order matches the OLD prototype:
        # mass, xmom_j*, zmom_j*, bathymetry, cont_j*.
        ordered.append(("bathymetry", sp.Derivative(state.b, state.t)))
        for k in range(1, N_p + 1):
            cont_jk = getattr(sys._tree.continuity, f"test_{k}").expr
            cont_jk_alg = sp.expand(cont_jk.doit().subs(dt_h_sub))
            ordered.append((f"cont_j{k}", cont_jk_alg))

        # 10.5 REDUCE EVOLUTION ROWS TO ESCALANTE EQ (4) FORM.
        #
        # The Galerkin projection produces residuals that carry extras
        # the paper doesn't list in eq (4):
        #
        #  (a) Explicit ``∂_t h`` cross-terms on the j ≥ 1 momentum rows
        #      (e.g. ``−U_0·∂_t h + (U_1/3)·∂_t h`` in ``xmom_j1``)
        #      — eliminated by subtracting ``α·mass_eq`` where
        #      ``α = coeff_of_∂_t_h``.
        #
        #  (b) ``W_k`` cross-terms on the j ≥ 1 momentum rows (e.g.
        #      ``+2·U_0·W_0 + (2/3)·U_1·W_1`` in ``xmom_j1``)
        #      — eliminated by subtracting ``β_0·cont_j1 + β_1·cont_j2``
        #      where ``β_k = coeff_of_W_k / 2`` (since each ``cont_jk``
        #      contains ``+2·W_{k-1}`` linearly).
        #
        # Both reductions are zero on the solution manifold (mass_eq=0,
        # cont_jk=0), so they preserve physics but bring the symbolic
        # form to Escalante eq (4) RHS bit-for-bit on the j ≥ 1 rows.
        # The compound ``∂_t(c·h)`` and ``∂_t(c·U_k)`` atoms are
        # preserved.
        dt_h_atom = sp.Derivative(state.h, state.t)
        mass_eq_expanded = sp.expand(mass_expr.doit())

        # Build cont_jk expressions for the substitution.  ``ordered``
        # contains them under names ``cont_j1`` ... ``cont_jN_p``.
        cont_eqs = {n: sp.expand(e) for n, e in ordered
                    if n.startswith("cont_j")}

        # Save the RAW (Cantero-Chinchilla / Steffler-Jin form) ``ordered``
        # before any constraint reduction.  This is the "honest" Galerkin
        # projection with explicit ``W²`` content, matching
        # Cantero-Chinchilla, Castro-Orgaz & Khan (2018) eq (24).  Stored
        # on the model as ``self._chain_equations_raw`` for users who want
        # the un-reduced form.
        ordered_raw = list(ordered)

        # ── Stage 1: quadratic-in-W elimination ─────────────────────────
        # The j ≥ 1 z-momentum rows carry W² and W·W cross-products from
        # ``∫ w²·∂_z φ_w_j dz`` (the IBP bulk term — see derivation in
        # the project README).  Each W_k² is removed via the identity
        # ``W_k · cont_jk = 0`` (since ``cont_jk = 0`` on solutions);
        # subtracting ``α_kk · (W_k/cont_W_coeff) · cont_jk`` from a
        # row converts ``α_kk · W_k²`` into ``α_kk · W_k · (rest_of_cont)``
        # which is linear in W_k.
        #
        # For W_2 (which is closed via ``bot_KBC`` rather than cont_j_{N_w}),
        # the modal closure step has already substituted W_2 in terms of
        # the other modes.  After that substitution, ``(2/5)·W_2²``
        # becomes ``(2/5)((U_0+U_1)·∂_x b − W_0 − W_1)²``, which expands
        # into ``W_0², W_1², W_0·W_1`` content (handled by cont_j1, cont_j2)
        # plus ``(U_0+U_1)²·(∂_x b)²`` (pure spatial, kept).
        def _eliminate_quadratic_W(expr):
            """Eliminate W_i·W_j cross-products in ``expr`` using
            ``W_i · cont_jk = 0`` identities.  Iterate until quadratic
            content stabilises.  Returns the reduced expression."""
            e = sp.expand(expr)
            for _ in range(4):
                changed = False
                for k_w in range(N_w):
                    W_k = coeffs_w[k_w]
                    cont_name = f"cont_j{k_w + 1}"
                    if cont_name not in cont_eqs:
                        continue
                    cont = cont_eqs[cont_name]
                    cont_W_coeff = cont.coeff(W_k)
                    if cont_W_coeff == 0:
                        continue
                    # Diagonal W_k² term.
                    alpha = e.coeff(W_k, 2)
                    if alpha != 0:
                        ratio = alpha / cont_W_coeff
                        e = sp.expand(e - ratio * W_k * cont)
                        changed = True
                    # Off-diagonal W_k · W_j for j ≠ k.
                    for j_w in range(N_w):
                        if j_w == k_w:
                            continue
                        W_j = coeffs_w[j_w]
                        beta = e.coeff(W_k * W_j)
                        if beta == 0:
                            continue
                        # Use cont_jk (which contains W_k linearly) and
                        # multiply by W_j to absorb the cross-product.
                        ratio = beta / cont_W_coeff
                        e = sp.expand(e - ratio * W_j * cont)
                        changed = True
                if not changed:
                    break
            return e

        # ── Stage 2: linear-in-W and ∂_t h cross-term elimination ───────
        # Only applied for the ``escalante`` quadratic form.  The
        # ``cantero_chinchilla`` form (default) skips all three stages
        # and keeps the residuals in their un-reduced Galerkin form
        # (Cantero-Chinchilla, Castro-Orgaz & Khan 2018 eq 24).  See
        # the ``quadratic_form`` param docstring for the trade-offs.
        if self._quadratic_form == "escalante":
            cleaned: list = [("mass", mass_expr)]
            for name, expr in ordered[1:]:                   # skip mass
                if not name.startswith(("xmom_j", "zmom_j")):
                    cleaned.append((name, expr))
                    continue
                # IMPORTANT: do NOT call .doit() before coeff extraction
                # — see comment above (preserves compound ``∂_t(c·h)``
                # atoms).
                new_expr = sp.sympify(expr)

                # Stage 1: quadratic-in-W elimination via ``W_k·cont_jk
                # = 0`` identities (only j ≥ 1 z-momentum rows carry W²
                # content from ``∂_z(w²)``).
                new_expr = _eliminate_quadratic_W(new_expr)

                # Stage 2a: explicit ∂_t h via mass_eq.
                alpha = new_expr.coeff(dt_h_atom)
                if alpha != 0:
                    new_expr = sp.expand(
                        new_expr - alpha * mass_eq_expanded)

                # Stage 2b: linear-in-W via cont_jk.
                for k_w in range(N_w):
                    W_k = coeffs_w[k_w]
                    beta = new_expr.coeff(W_k)
                    if beta == 0:
                        continue
                    cont_name = f"cont_j{k_w + 1}"
                    if cont_name not in cont_eqs:
                        continue
                    cont_W_coeff = cont_eqs[cont_name].coeff(W_k)
                    if cont_W_coeff == 0:
                        continue
                    ratio = beta / cont_W_coeff
                    new_expr = sp.expand(
                        new_expr - ratio * cont_eqs[cont_name])
                cleaned.append((name, new_expr))

            # Stash both forms — raw and reduced.
            self._chain_equations_raw = dict(ordered_raw)
            ordered = cleaned
        else:
            # cantero_chinchilla: keep the un-reduced Galerkin output.
            self._chain_equations_raw = dict(ordered_raw)

        # State functions for tag classification.  Order matches
        # var_names: h, then (U_k, V_k, …), then W_k, then P_k.
        state_funcs = [state.h]
        for coeffs in coeffs_horiz:
            state_funcs += coeffs
        state_funcs += coeffs_w[:N_w]
        # ``b`` between W and P matches the var_names ordering above
        # and the OLD JAX prototype's state layout.
        state_funcs += [state.b]
        state_funcs += coeffs_p[:N_p]

        # 11. Auto-tag every row.
        tagged: Dict[str, Expression] = {}
        for name, expr in ordered:
            tagged_expr = auto_solver_tag(
                Expression(expr, name=name),
                state_funcs=state_funcs,
                gravity_param=state.g,
                t=state.t, x=state.x,
            )
            tagged[name] = tagged_expr

        self._equation_names = list(tagged.keys())
        self._equations = tagged
        self._equation_holder = _TaggedEquationHolder(self._equations)
        self._chain_state_funcs = state_funcs

    # ------------------------------------------------------------------
    # State-function ↔ Symbol mapping.
    # ------------------------------------------------------------------

    def _refresh_state_sym_map(self, *, h_new=None):
        """Refresh the mapping from chain Function atoms to model
        Symbols.  Called after ``Model.__init__`` populates
        ``self.variables`` (and again after any positivity re-mint of
        ``h``)."""
        state_func_names = [f.func.__name__
                            for f in self._chain_state_funcs]
        sym_for_name = {}
        for key in self.variables.keys():
            sym = self.variables[key]
            sym_for_name[key] = sym
        if h_new is not None:
            sym_for_name["h"] = h_new
        # Map Function-call atom → Symbol.
        self._func_to_sym = {
            f: sym_for_name[f.func.__name__]
            for f in self._chain_state_funcs
            if f.func.__name__ in sym_for_name
        }
        # Also store the ordered Symbol list (matches self.variables order).
        self._state_syms = [sym_for_name[n] for n in state_func_names]
        # Static topography ``b(t, x)`` becomes ``b(x)`` after
        # ``∂_t b → 0``; keep its Function-call form (it's not in state).
        # (No re-mint needed; the chain leaves' ``b`` atom is the
        # canonical bottom topography.)

    # ------------------------------------------------------------------
    # Operator-API surface — driven by tag-walks over the chain tree.
    # ------------------------------------------------------------------

    @property
    def equations(self) -> Dict[str, Expression]:
        """The chain DAE rows as a dict of tagged Expressions."""
        return self._equations

    @property
    def equation_names(self):
        """Canonical row names in mass-first / cont-last order."""
        return list(self._equation_names)

    def _coords(self):
        names = ("x", "y")[:self.dimension]
        return [sp.Symbol(n, real=True) for n in names]

    def _to_sym(self, matrix):
        """Map chain Function atoms → model Symbols inside a matrix or
        N-d array."""
        if isinstance(matrix, sp.Matrix):
            return matrix.xreplace(self._func_to_sym)
        if isinstance(matrix, (sp.MutableDenseNDimArray,
                               sp.ImmutableDenseNDimArray)):
            out = sp.MutableDenseNDimArray.zeros(*matrix.shape)
            shape = matrix.shape
            ndim = len(shape)

            def _iter(shape):
                if not shape:
                    yield ()
                    return
                for i in range(shape[0]):
                    for rest in _iter(shape[1:]):
                        yield (i,) + rest

            for idx in _iter(shape):
                entry = sp.sympify(matrix[idx])
                out[idx] = entry.xreplace(self._func_to_sym)
            return out
        return sp.sympify(matrix).xreplace(self._func_to_sym)

    def _variable_map(self):
        return {name: [i] for i, name in enumerate(self._equation_names)}

    def flux(self):
        F = collect_solver_tag(
            self._equation_holder, "flux",
            variable_map=self._variable_map(),
            n_variables=self.n_variables,
            n_directions=self.dimension,
            coords=self._coords(),
            state_variables=self._chain_state_funcs,
            policy="strict",
        )
        return self._to_sym(F)

    def hydrostatic_pressure(self):
        P = collect_solver_tag(
            self._equation_holder, "hydrostatic_pressure",
            variable_map=self._variable_map(),
            n_variables=self.n_variables,
            n_directions=self.dimension,
            coords=self._coords(),
            state_variables=self._chain_state_funcs,
            policy="strict",
        )
        return self._to_sym(P)

    def nonconservative_matrix(self):
        B = collect_solver_tag(
            self._equation_holder, "nonconservative_flux",
            variable_map=self._variable_map(),
            n_variables=self.n_variables,
            n_directions=self.dimension,
            coords=self._coords(),
            state_variables=self._chain_state_funcs,
            policy="strict",
        )
        # collect_solver_tag returns shape (n_eq, n_eq, n_dim) when
        # state_variables length matches n_eq; we want (n_eq, n_state, n_dim).
        n_eq = self.n_variables
        n_state = self.n_variables
        n_dim = self.dimension
        B_resized = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim)
        for i in range(n_eq):
            for j in range(min(n_state, B.shape[1])):
                for d in range(n_dim):
                    B_resized[i, j, d] = B[i, j, d]
        return self._to_sym(B_resized)

    def source(self):
        S = collect_solver_tag(
            self._equation_holder, "source",
            variable_map=self._variable_map(),
            n_variables=self.n_variables,
            policy="strict",
        )
        # SystemModel residual form is ``... − S(Q) = 0``, whereas the
        # auto-tagger stores source terms with their original LHS sign.
        # Negate to match the canonical form.
        S_mat = sp.Matrix(self.n_variables, 1, lambda i, _j: -S[i])
        return self._to_sym(S_mat)

    def mass_matrix(self):
        """Extract the mass matrix from the ``time_derivative`` tag on
        each equation.  ``M[i, j] = ∂(time_derivative_part_of_eq_i) /
        ∂(∂_t f_j)``.

        Compound atoms like ``Derivative(c·h, t)`` are first expanded
        via ``.doit()`` so they appear as ``c·∂_t h + h·∂_t c`` and the
        bare ``Derivative(h, t)`` atom matches.

        For algebraic rows (no time-derivative terms), the row is all
        zero — that is the DAE structure.
        """
        n = self.n_variables
        state_t = self._chain_state.t
        M = sp.zeros(n, n)
        for i, name in enumerate(self._equation_names):
            eq = self._equations[name]
            td_part = eq.get_solver_tag("time_derivative")
            if td_part is None or td_part == 0:
                continue
            td = sp.expand(td_part.doit())
            for j, fj in enumerate(self._chain_state_funcs):
                dot_j = sp.Symbol(f"_dot_{fj.func.__name__}", real=True)
                dt_atom = sp.Derivative(fj, state_t)
                replaced = td.subs(dt_atom, dot_j)
                coeff = sp.diff(replaced, dot_j)
                if coeff != 0:
                    M[i, j] = sp.expand(coeff)
        return self._to_sym(M)

    # ------------------------------------------------------------------
    # Display.
    # ------------------------------------------------------------------

    def describe(self, **kwargs):
        return self._chain_system.describe(**kwargs)

    def _repr_markdown_(self):
        return self._chain_system._repr_markdown_()

    def describe_chain_intermediate(self):
        """Render the Sum-form intermediate (post-Expand,
        pre-EvaluateIntegrals)."""
        return self._chain_intermediate.describe()

    def describe_chain_closed(self):
        """Same as :meth:`describe`; kept for symmetry."""
        return self._chain_system.describe()


__all__ = ["VAMModelGalerkin"]
