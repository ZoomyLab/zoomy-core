"""VAM — non-hydrostatic Vertically-Averaged Moment system.

Inherits :class:`SigmaReference` (and through it,
:class:`zoomy_core.model.basemodel.Model`).  Unlike SME, the pressure
is NOT eliminated via hydrostatic reduction: ``w`` and ``p`` stay as
state variables and the z-momentum equation is Galerkin-projected
alongside continuity and momentum_x.

State (matrix-extraction surface):
    ``[h, q_0..q_N, r_0..r_N, p_0..p_N]`` with ``q_k = h·u_k``,
    ``r_k = h·w_k``.  Pressure modes ``p_k`` are kept symbolic (no
    closure imposed at this level).
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.models.sigmaref import SigmaReference
from zoomy_core.model.models.basis_cache import get_basis_matrices
from zoomy_core.model.operations import (
    Multiply,
    ResolveDummy,
    ProductRule,
    Integrate,
    EvaluateIntegrals,
    Symmetrize,
    Legendre_shifted,
)


__all__ = ["VAM"]


def _default_basis_factory(level, symbol="phi"):
    """Default basis used by VAM — shifted Legendre on σ ∈ [0, 1]."""
    return Legendre_shifted(level=level, symbol=symbol)


class VAM(SigmaReference):
    """Non-hydrostatic Vertically-Averaged Moments at level ``N``.

    Parameters
    ----------
    N : int
        Moment level — number of u / w / p modes is ``N + 1``.
    basis_factory : callable, optional
        ``basis_factory(level) → Basisfunction``.  Constructs the
        u-basis (level ``N``) and the w-basis (level ``N + 1``).
        Default: ``Legendre_shifted(level=level)``.  Subclasses can
        swap this for any other basis without touching the pipeline.
    """

    # Same lazy-finalization story as SME: derive_model leaves
    # equations in Function form; substitute + tag in
    # ``_prepare_for_systemmodel``.
    _finalize_lazy = True

    def __init__(self, N: int = 2, *, basis_factory=None, **kwargs):
        self.N = N
        self.basis_factory = basis_factory or _default_basis_factory
        # u-basis carries N+1 modes; w-basis carries N+2 (one extra for
        # the optional KBC closure — VAM uses only the first N+1, but
        # the level=N+1 convention matches SME).  Distinct ``symbol``
        # per basis so the phi_fn classes are distinct and atoms route
        # to the correct ``basis.resolve_atoms`` inside EvaluateIntegrals.
        self.basis_u = self.basis_factory(N, "phi_u")
        self.basis_w = self.basis_factory(N + 1, "phi_w")
        # Indexed modal-coefficient Functions (declared before super()
        # so the derive_model pipeline can reference them).
        self.u_fn = sp.Function("u", real=True)
        self.w_fn = sp.Function("w", real=True)
        self.p_fn = sp.Function("p", real=True)
        self.q_fn = sp.Function("q", real=True)
        self.r_fn = sp.Function("r", real=True)
        # Use basis-owned phi_fn (carries ``_basis`` back-ref) so the
        # atoms stay opaque through the pipeline and resolve INSIDE
        # each Integral via ``basis.resolve_atoms`` — gives the biggest
        # cache hit on basis-product shapes.
        self.phi_u_fn = self.basis_u.phi_fn
        self.phi_w_fn = self.basis_w.phi_fn
        # Pressure ansatz uses the same family as the velocity ansatz
        # (Galerkin-on-u-test); a distinct ``phi_p_fn`` would force an
        # unnecessary second basis.  Reuse basis_u's phi_fn.
        self.phi_p_fn = self.basis_u.phi_fn
        # Galerkin test dummies — branched in _resolve_dummies.
        self.psi_u    = sp.Function("psi_u", real=True)
        self.psi_w    = sp.Function("psi_w", real=True)
        self.omega    = sp.Function("omega", real=True)

        kwargs.setdefault("name", f"VAM-L={N}")
        var_names = (
            ["h"]
            + [f"q_{k}" for k in range(N + 1)]
            + [f"r_{k}" for k in range(N + 1)]
            + [f"p_{k}" for k in range(N + 1)]
        )
        kwargs.setdefault("variables", var_names)
        kwargs.setdefault("parameters", {"g": 9.81, "rho": 1.0})
        kwargs.setdefault("eigenvalue_mode", "numerical")
        super().__init__(**kwargs)

    # ── Derivation hook ─────────────────────────────────────────────
    def derive_model(self):
        """Build reference equations (via SigmaReference) and run the
        VAM post-σ pipeline."""
        super().derive_model()
        self._multiply_h_and_fold_conservative()
        self._substitute_modal_ansatz()
        self._galerkin_project_all_three()
        self._sigma_integrate()
        self._kbc_modal_closure()
        self._cov_u_to_q()
        self._resolve_dummies()
        self._evaluate_integrals()
        self._cov_w_to_r()
        self._gravity_self_pair_fold()
        self._invert_mass_matrix()
        self._eliminate_dt_h_via_continuity_0()

    def _prepare_for_systemmodel(self):
        """Lazy finalisation hook (called by
        ``_finalize_for_systemmodel``)."""
        self._substitute_to_model_symbols()
        self._variable_map = self._build_variable_map()

    # VAM keeps w, p as state → no pre-σ hook needed.

    # ── pipeline steps ──────────────────────────────────────────────
    def _multiply_h_and_fold_conservative(self):
        s, src, t, x = self.state, self.src, self.state.t, self.state.x
        self.apply(Multiply(src.h))
        for eq in self:
            eq.simplify()
        self.continuity[[0]].apply(ProductRule(variables=[x]))
        self.momentum_x[[0, 1, 7]].apply(ProductRule(variables=[t, x]))
        self.momentum_z[[2, 3, 11]].apply(ProductRule(variables=[t, x]))
        for eq in self:
            eq.simplify()

    def _substitute_modal_ansatz(self):
        N, s, src = self.N, self.state, self.src
        sigma = s.zeta_ref
        self.u_ansatz = sum(
            self.u_fn(k, s.t, s.x) * self.phi_u_fn(k, sigma)
            for k in range(N + 1))
        self.w_ansatz = sum(
            self.w_fn(k, s.t, s.x) * self.phi_w_fn(k, sigma)
            for k in range(N + 1))
        self.p_ansatz = sum(
            self.p_fn(k, s.t, s.x) * self.phi_p_fn(k, sigma)
            for k in range(N + 1))
        self.apply({
            src.u.xreplace({s.z: sigma}): self.u_ansatz,
            src.w.xreplace({s.z: sigma}): self.w_ansatz,
            src.p.xreplace({s.z: sigma}): self.p_ansatz,
        })

    def _galerkin_project_all_three(self):
        s = self.state
        sigma = s.zeta_ref
        # Continuity + x-momentum share the u-test; z-momentum uses w-test.
        self.continuity.apply(
            Multiply(self.psi_u(sigma) * self.omega(sigma)))
        self.momentum_x.apply(
            Multiply(self.psi_u(sigma) * self.omega(sigma)))
        self.momentum_z.apply(
            Multiply(self.psi_w(sigma) * self.omega(sigma)))

    def _sigma_integrate(self):
        s = self.state
        sigma = s.zeta_ref
        for name in ("continuity", "momentum_x", "momentum_z"):
            getattr(self, name).apply(
                Integrate(sigma, sp.S.Zero, sp.S.One, method="auto"))
        for eq in self:
            eq.simplify()

    def _kbc_modal_closure(self):
        s, src = self.state, self.src
        sigma = s.zeta_ref

        def residual(interface, at_value):
            lhs = src.w.subs(s.z, at_value)
            rhs = (sp.Derivative(interface, s.t)
                   + src.u.subs(s.z, at_value) * sp.Derivative(interface, s.x))
            return (lhs - rhs).xreplace({
                src.w.subs(s.z, at_value): self.w_ansatz.subs(sigma, at_value),
                src.u.subs(s.z, at_value): self.u_ansatz.subs(sigma, at_value),
            })

        closure = sp.solve(
            [residual(src.b, sp.S.Zero), residual(src.eta, sp.S.One)],
            [self.w_fn(0, s.t, s.x), self.w_fn(1, s.t, s.x)],
            dict=True,
        )[0]
        self.apply(closure, level="minor", description="KBC modal closure")
        for eq in self:
            eq.simplify()

    def _cov_u_to_q(self):
        N, s, src = self.N, self.state, self.src
        self.u_modes = [self.u_fn(k, s.t, s.x) for k in range(N + 1)]
        self.q_modes = [self.q_fn(k, s.t, s.x) for k in range(N + 1)]
        self.apply(
            {u: q / src.h for u, q in zip(self.u_modes, self.q_modes)},
            level="minor", description="CoV u(k,t,x) → q(k,t,x)/h",
        )
        for eq in self:
            eq.simplify()

    def _resolve_dummies(self):
        """Resolve omega → 1 and branch on Galerkin test indices,
        leaving phi atoms opaque for EvaluateIntegrals to resolve
        per-Integral via ``basis.resolve_atoms``.  See SME._resolve_dummies
        for the rationale.
        """
        N = self.N

        def opaque_atom(basis, k):
            return lambda arg, _b=basis, _k=k: _b.phi_fn(_k, arg)

        # Back-compat aliases (some downstream code reads these).
        self.legendre_u = self.basis_u
        self.legendre_w = self.basis_w

        self.apply(ResolveDummy(self.omega, lambda arg: sp.S.One))
        # NB: NO ResolveDummy on phi_u_fn / phi_w_fn / phi_p_fn — they
        # all carry the basis ``_basis`` back-ref and get resolved
        # inside EvaluateIntegrals.  Galerkin test counts: VAM uses
        # N+1 test functions for both psi_u and psi_w (matching the
        # N+1 dynamic equations per block).  This deliberately uses
        # only the first N+1 modes of basis_w even though
        # basis_w.level + 1 = N + 2.
        self.apply(ResolveDummy(
            self.psi_u,
            [opaque_atom(self.basis_u, k) for k in range(N + 1)]))
        self.apply(ResolveDummy(
            self.psi_w,
            [opaque_atom(self.basis_w, k) for k in range(N + 1)]))

    def _evaluate_integrals(self):
        self.apply(EvaluateIntegrals(self.state))
        for eq in self:
            eq.simplify()

    def _cov_w_to_r(self):
        """``w_k → r_k / h`` so the vertical momentum equations carry
        conservative time-derivative atoms ``∂_t r_k``."""
        N, s, src = self.N, self.state, self.src
        self.w_modes = [self.w_fn(k, s.t, s.x) for k in range(N + 1)]
        self.r_modes = [self.r_fn(k, s.t, s.x) for k in range(N + 1)]
        self.apply(
            {w: r / src.h for w, r in zip(self.w_modes, self.r_modes)},
            level="minor", description="CoV w(k,t,x) → r(k,t,x)/h",
        )
        for eq in self:
            eq.simplify()

    def _gravity_self_pair_fold(self):
        s, src = self.state, self.src
        g = self.parameters.g
        mom0 = self.momentum_x_0
        grav = next(
            (t for t in mom0
             if t.expr.has(g) and t.expr.has(sp.Derivative(src.h, s.x))
             and not t.expr.has(src.b)),
            None)
        if grav is not None:
            grav.apply(Symmetrize(ProductRule(variables=[s.x])))
            mom0.simplify()

    def _invert_mass_matrix(self):
        """Multiply each row by the inverse diagonal of its
        corresponding basis' Galerkin mass matrix.

        * ``momentum_x_k`` ← projected via psi_u → uses ``basis_u``'s ``M``.
        * ``momentum_z_k`` ← projected via psi_w → uses ``basis_w``'s ``M``.

        For shifted Legendre (default): ``M = diag(1/(2k+1))``, so each
        row is multiplied by ``(2k+1)`` — the historical hand-rolled
        factor.  For :class:`LayeredBasis`: ``M = diag(α_ℓ/(2m+1))``
        and the per-layer ``α`` is applied automatically.
        """
        M_u = get_basis_matrices(self.basis_u)["M"]
        M_w = get_basis_matrices(self.basis_w)["M"]
        # Diagonal-only assumption (Legendre / LayeredBasis / Chebyshev).
        for label, M in (("basis_u", M_u), ("basis_w", M_w)):
            n = M.shape[0]
            for k in range(n):
                for i in range(n):
                    if i != k and M[k, i] != 0:
                        raise ValueError(
                            f"_invert_mass_matrix ({label}): mass matrix "
                            f"not diagonal — M[{k}, {i}] = {M[k, i]} ≠ 0."
                        )
        for k in range(self.N + 1):
            eq_x = self._equations.get(f"momentum_x_{k}")
            if eq_x is not None:
                eq_x.expr = (1 / M_u[k, k]) * eq_x.expr
                eq_x.simplify()
            eq_z = self._equations.get(f"momentum_z_{k}")
            if eq_z is not None:
                eq_z.expr = (1 / M_w[k, k]) * eq_z.expr
                eq_z.simplify()

    def _eliminate_dt_h_via_continuity_0(self):
        """``∂_t h → −∂_x q_0`` (depth-mean continuity) in all
        non-continuity equations."""
        s, src = self.state, self.src
        cont0_subst = {sp.Derivative(src.h, s.t):
                        -sp.Derivative(self.q_fn(0, s.t, s.x), s.x)}
        for name, eq in self._equations.items():
            if name == "continuity_0":
                continue
            eq.expr = eq.expr.xreplace(cont0_subst)
            eq.simplify()

    # ── final: Function → Symbol substitution ─────────────────────
    def _substitute_to_model_symbols(self):
        s, src = self.state, self.src
        subs = {src.h: self.variables.h}
        for k in range(self.N + 1):
            subs[self.q_fn(k, s.t, s.x)] = self.variables[f"q_{k}"]
            subs[self.r_fn(k, s.t, s.x)] = self.variables[f"r_{k}"]
            subs[self.p_fn(k, s.t, s.x)] = self.variables[f"p_{k}"]
        for eq in self:
            eq.expr = eq.expr.xreplace(subs)

    def _build_variable_map(self):
        """``{eq_name: [row_index]}`` for tag extraction.

        Layout: ``[h, q_0..q_N, r_0..r_N, p_0..p_N]``.

        * ``continuity_0`` → row 0 (h).
        * ``momentum_x_k`` → row ``1 + k`` for ``k = 0..N``.
        * ``momentum_z_k`` → row ``1 + (N+1) + k`` for ``k = 0..N``.
        * Pressure rows (``p_0..p_N``) are not dynamic equations at
          this stage — they remain auxiliary closures.
        """
        N = self.N
        m = {"continuity_0": [0]}
        for k in range(N + 1):
            m[f"momentum_x_{k}"] = [1 + k]
        for k in range(N + 1):
            mz = f"momentum_z_{k}"
            if mz in self._equations:
                m[mz] = [1 + (N + 1) + k]
        return m
