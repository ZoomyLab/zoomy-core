"""SystemModel — operator-form symbolic PDE for analysis + transformation.

`SystemModel` is the runtime/operator-form representation of a PDE
system: a flat container of symbolic ``sp.Matrix`` / sympy-tensor
operators (flux, non-conservative-matrix, source, mass matrix,
hydrostatic pressure) plus state, parameters, coordinates, and
boundary-condition kernels.  It carries no derivation history and no
equation tree — that's the `Model` class's job.

The two are **independent siblings**, not inherited.  `Model` owns
derivation; `SystemModel` owns the operator surface that solvers and
analysis consume.  ``SystemModel.from_model(m)`` extracts the
operators once by calling the model's API methods and freezes the
result; subsequent calls on the SystemModel never re-walk.

Operations on a SystemModel modify the stored matrices in place via
``apply(operation)`` — the most important being ``InvertMassMatrix``
which left-multiplies the system by ``M⁻¹`` to reach canonical
``∂_t Q + ∂_x F + B·∂_x Q − S = 0`` form.

Solvers and analysis routines accept either a ``Model`` or a
``SystemModel`` directly; internally they normalize via
``SystemModel.from_model(m)`` if a Model is passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sympy as sp


def _iter_indices(shape):
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for rest in _iter_indices(shape[1:]):
            yield (i,) + rest


@dataclass
class SystemModel:
    """Symbolic operator-form PDE system.

    Stored matrices follow the contract:

    .. math::

        M(Q) \\; \\partial_t Q
            + \\nabla \\cdot \\big(F(Q) + P(Q)\\big)
            + \\sum_d B(Q)[:,:,d] \\, \\partial_d Q
            - S(Q) = 0

    Operators are indexed ``(equation_row, state_col[, dim])``.  In
    general the system is **rectangular** — there can be fewer (or
    more) equations than state entries; this is the case for the
    splitter sub-systems, where each stage updates only a subset of
    the shared state vector.  ``equation_to_state_index[r]`` records
    which state entry equation ``r`` updates; for square systems the
    default is the identity map ``[0, 1, …, n_state-1]``.

    Shape contract:

    * ``flux``                  — ``(n_eq, n_dim)``
    * ``hydrostatic_pressure``  — ``(n_eq, n_dim)``
    * ``nonconservative_matrix``— ``(n_eq, n_state, n_dim)``
    * ``source``                — ``(n_eq, 1)``
    * ``mass_matrix``           — ``(n_eq, n_state)``
    """

    time: sp.Symbol
    space: List[sp.Symbol]
    state: List[Any]
    aux_state: List[Any]
    parameters: Dict[Any, Any]
    flux: sp.Matrix
    hydrostatic_pressure: sp.Matrix
    nonconservative_matrix: Any              # sp.MutableDenseNDimArray
    source: sp.Matrix
    mass_matrix: sp.Matrix
    equation_to_state_index: Optional[List[int]] = None
    boundary_conditions: Optional[Any] = None
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if self.equation_to_state_index is None:
            self.equation_to_state_index = list(range(self.n_equations))

    # ── Shape accessors ────────────────────────────────────────────────

    @property
    def n_equations(self) -> int:
        return self.flux.shape[0]

    @property
    def n_state(self) -> int:
        return len(self.state)

    @property
    def n_dim(self) -> int:
        return len(self.space)

    @property
    def is_square(self) -> bool:
        """True when n_equations == n_state and equations map identity-style."""
        return (self.n_equations == self.n_state
                and self.equation_to_state_index == list(range(self.n_state)))

    # ── Operator-API methods (mirror Model) ─────────────────────────────

    def quasilinear_matrix(self):
        """``∂F/∂Q + ∂P/∂Q + B`` — shape ``(n_eq, n_state, n_dim)``."""
        n_eq = self.n_equations
        n_st = self.n_state
        d = self.n_dim
        Q = sp.MutableDenseNDimArray.zeros(n_eq, n_st, d)
        for i in range(n_eq):
            for j in range(n_st):
                for k in range(d):
                    djF = sp.diff(self.flux[i, k], self.state[j])
                    djP = sp.diff(self.hydrostatic_pressure[i, k],
                                  self.state[j])
                    Q[i, j, k] = djF + djP + self.nonconservative_matrix[i, j, k]
        return Q

    def source_jacobian_wrt_state(self):
        """``∂S/∂Q`` — shape ``(n_eq, n_state)``."""
        n_eq = self.n_equations
        n_st = self.n_state
        out = sp.zeros(n_eq, n_st)
        for i in range(n_eq):
            for j in range(n_st):
                out[i, j] = sp.diff(self.source[i, 0], self.state[j])
        return out

    # ── apply / row view ───────────────────────────────────────────────

    def apply(self, operation, *, name: Optional[str] = None,
              description: Optional[str] = None):
        """Apply a system-level operation in place.  Returns self."""
        op_name = name or getattr(operation, "name", None) or \
                  type(operation).__name__
        op_desc = description or getattr(operation, "description", op_name)
        operation(self)
        self.history.append({"name": op_name, "description": op_desc})
        return self

    def __getitem__(self, i: int) -> "SystemModelRow":
        return SystemModelRow(self, i)

    # ── from_model factory ────────────────────────────────────────────

    @classmethod
    def from_model(cls, model) -> "SystemModel":
        """Build a SystemModel from a Model by reading its operator API.

        Calls ``model.flux()``, ``model.nonconservative_matrix()``,
        ``model.source()``, ``model.hydrostatic_pressure()`` once and
        freezes the matrices.  ``state`` / ``aux_state`` /
        ``parameters`` come from the model's Zstructs.
        """
        time_sym = getattr(model, "time", sp.Symbol("t", real=True))
        dim = getattr(model, "dimension", 1)
        coord_names = ["x", "y", "z"]
        space = [sp.Symbol(coord_names[d], real=True) for d in range(dim)]

        state = [model.variables[k] for k in model.variables.keys()]
        aux_state = [model.aux_variables[k]
                     for k in model.aux_variables.keys()]
        parameters: Dict[Any, Any] = {}
        for k in model._parameter_symbols.keys():
            sym = model._parameter_symbols[k]
            if hasattr(model.parameters, k):
                parameters[sym] = getattr(model.parameters, k)

        n_eq = model.n_variables

        def _to_matrix(z, n_rows, n_cols):
            """Coerce a model ZArray / sp.Matrix / 1-D ZArray into an
            sp.Matrix with the requested shape."""
            # Try direct tomatrix() first (rank-2 ZArrays).
            if hasattr(z, "tomatrix"):
                try:
                    m = z.tomatrix()
                except (ValueError, AttributeError):
                    m = None
            else:
                m = z
            if m is None:
                # Rank-1 ZArray: extract via tolist() / iteration.
                if hasattr(z, "tolist"):
                    items = list(z.tolist())
                else:
                    items = [z[i] for i in range(n_rows)]
                # Reshape to (n_rows, n_cols).
                if n_cols == 1:
                    m = sp.Matrix(items)
                else:
                    m = sp.Matrix(n_rows, n_cols,
                                  lambda i, j: items[i * n_cols + j])
                return m
            if not isinstance(m, sp.Matrix):
                m = sp.Matrix(m)
            if m.shape == (n_rows, n_cols):
                return m
            if m.shape == (n_rows * n_cols, 1) or m.shape == (1, n_rows * n_cols):
                return m.reshape(n_rows, n_cols)
            return m

        F = _to_matrix(model.flux(), n_eq, dim)
        P = _to_matrix(model.hydrostatic_pressure(), n_eq, dim)

        ncp_z = model.nonconservative_matrix()
        if hasattr(ncp_z, "todense"):
            B = sp.MutableDenseNDimArray(ncp_z.todense())
        elif hasattr(ncp_z, "tolist"):
            B = sp.MutableDenseNDimArray(ncp_z.tolist())
        else:
            B = sp.MutableDenseNDimArray(ncp_z)

        S_z = model.source()
        S_mat = _to_matrix(S_z, n_eq, 1)

        # Mass matrix: prefer ``model.mass_matrix()`` if the model
        # exposes one (chain-derived models do); otherwise default to
        # identity (canonical form for operator-API-only models).
        if hasattr(model, "mass_matrix") and callable(model.mass_matrix):
            M_mat = _to_matrix(model.mass_matrix(), n_eq, n_eq)
        else:
            M_mat = sp.eye(n_eq)

        equation_names = getattr(model, "equation_names", None)
        bcs = getattr(model, "_boundary_conditions", None)

        sm = cls(
            time=time_sym,
            space=space,
            state=state,
            aux_state=aux_state,
            parameters=parameters,
            flux=F,
            hydrostatic_pressure=P,
            nonconservative_matrix=B,
            source=S_mat,
            mass_matrix=M_mat,
            boundary_conditions=bcs,
        )
        if equation_names is not None:
            sm.equation_names = list(equation_names)
        sm.history.append({"name": "from_model",
                            "description": f"extracted from {type(model).__name__}"})
        # Auto-scan: every non-state Function atom and every Derivative
        # atom in the operator matrices becomes an aux Symbol with a
        # structured registry entry.  Solvers walk ``sm.aux_registry``
        # to compute aux values per step.
        sm.expose_aux_atoms()
        return sm

    # ── reconstruct_residuals ─────────────────────────────────────────

    def reconstruct_residuals(self):
        """Re-assemble the row-wise residuals ``M·∂_t Q + ∂_x F + ∂_x P
        + B·∂_x Q − S`` for every equation.  Returns a list of sympy
        expressions, indexed by row.

        Aux Symbols generated by ``expose_aux_atoms`` are back-
        substituted to their original ``Function`` / ``Derivative``
        atoms so the residuals display in their natural form for
        bit-for-bit fixture comparisons.

        Used in tests to compare the operator-form back against the
        original equation residuals (e.g. against Escalante eq (4)).
        """
        n_eq = self.n_equations
        n_st = self.n_state
        n_dim = self.n_dim
        t = self.time
        coords = self.space

        # Build per-state time- and spatial-derivative Functions if the
        # state is held as Symbols.  We need a coordinate-dependent
        # representation to take symbolic derivatives.
        def _as_function(sym):
            if isinstance(sym, sp.Function):
                return sym
            args = [t] + list(coords)
            return sp.Function(str(sym), real=True)(*args)

        state_funcs = [_as_function(s) for s in self.state]
        sym_to_func = dict(zip(self.state, state_funcs))
        aux_reverse = self._aux_reverse_map()

        def _restore(expr):
            """First put back the original aux atoms (which still
            reference state as Symbols), then upgrade state Symbols
            to Functions so derivatives display naturally."""
            return (sp.sympify(expr)
                    .xreplace(aux_reverse)
                    .xreplace(sym_to_func))

        residuals = []
        for i in range(n_eq):
            res = sp.S.Zero
            # M · ∂_t Q
            for j in range(n_st):
                m_ij = _restore(self.mass_matrix[i, j])
                if m_ij != 0:
                    res = res + m_ij * sp.Derivative(state_funcs[j], t)
            # ∂_x F[i] (and ∂_y if 2D)
            for d in range(n_dim):
                f_id = _restore(self.flux[i, d])
                if f_id != 0:
                    res = res + sp.Derivative(f_id, coords[d])
                p_id = _restore(self.hydrostatic_pressure[i, d])
                if p_id != 0:
                    res = res + sp.Derivative(p_id, coords[d])
            # B · ∂_x Q
            for j in range(n_st):
                for d in range(n_dim):
                    b_ijd = _restore(self.nonconservative_matrix[i, j, d])
                    if b_ijd != 0:
                        res = res + b_ijd * sp.Derivative(state_funcs[j], coords[d])
            # − S[i]
            s_i = _restore(self.source[i, 0])
            res = res - s_i
            residuals.append(sp.expand(res))
        return residuals

    # ── from_pdesystem factory (REMOVED) ──────────────────────────────


    # ── expose_aux_atoms (auto-scan) ──────────────────────────────────

    def expose_aux_atoms(self):
        """Auto-scan the operator matrices and route every
        non-state symbolic input into ``aux_state``.

        For each unique:

        * **bare** ``Function`` atom whose name is **not** a state
          variable (e.g. topography ``b(t, x, y)``, externally
          prescribed forcing fields) — substitute with an aux Symbol
          of the same name;
        * ``Derivative`` atom — substitute with an aux Symbol named
          ``{target}_{axes}`` (e.g. ``b_x``, ``h_x``, ``b_x_x``);

        and populate :attr:`aux_registry` with one structured dict per
        new aux entry:

        .. code-block:: python

            {"kind":   "function" | "derivative",
             "name":   <aux symbol name>,
             "row":    <index in self.aux_state>,
             "atom":   <original sympy atom>,
             "aux_symbol": <Symbol>,
             # derivative entries also carry:
             "target_name":  <name of the field being differentiated>,
             "target_kind":  "state" | "function" | "unknown",
             "state_index":  <int>          # if target_kind == "state"
             "function_row": <int>          # if target_kind == "function"
             "multi_index":  <tuple>        # spatial-derivative orders
                                            # in (axis_0, axis_1, …)}

        ``multi_index`` follows the SAME convention as
        :meth:`LSQMesh.compute_derivatives` so a solver can compute
        every derivative-aux entry in one call.  Time derivatives are
        skipped (they live in the mass matrix).

        Idempotent — if ``aux_registry`` is already non-empty (or all
        atoms are state) this is a no-op.
        """
        import sympy as sp
        from itertools import product as _product

        if getattr(self, "aux_registry", None) is not None:
            return

        state_names = {str(s) for s in self.state}
        space_names = {str(s): d for d, s in enumerate(self.space)}
        n_dim = len(self.space)

        def _iter_entries(M):
            if isinstance(M, sp.Matrix):
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        yield M[i, j]
            else:
                for idx in _product(*[range(s) for s in M.shape]):
                    yield M[idx]

        matrices = [self.flux, self.hydrostatic_pressure,
                    self.nonconservative_matrix, self.source,
                    self.mass_matrix]

        # ── Pass 1: collect Function atoms (excl. state). ───────────
        function_atoms = {}     # name → [atoms]
        for M in matrices:
            for entry in _iter_entries(M):
                for atom in sp.sympify(entry).atoms(sp.Function):
                    name = atom.func.__name__
                    if name in state_names:
                        continue
                    function_atoms.setdefault(name, []).append(atom)

        # ── Pass 2: collect Derivative atoms (skip time-derivs). ────
        deriv_atoms = {}    # (target_name, multi_index_tuple) → [atoms]
        for M in matrices:
            for entry in _iter_entries(M):
                for d in sp.sympify(entry).atoms(sp.Derivative):
                    target = d.args[0]
                    target_name = (target.func.__name__
                                   if isinstance(target, sp.Function)
                                   else str(target))
                    mi = [0] * n_dim
                    has_time = False
                    for var, n in d.variable_count:
                        vn = str(var)
                        if vn in space_names:
                            mi[space_names[vn]] += int(n)
                        else:
                            has_time = True
                            break
                    if has_time or all(o == 0 for o in mi):
                        continue
                    key = (target_name, tuple(mi))
                    deriv_atoms.setdefault(key, []).append(d)

        if not function_atoms and not deriv_atoms:
            self.aux_registry = []
            return

        registry = []
        sub_dict = {}
        n_aux_before = len(self.aux_state)
        new_syms = []
        function_row_of_name = {}

        # ── Function entries first (so derivative entries can ─────
        # reference them by row in the registry). ──────────────────
        for name, atoms in function_atoms.items():
            sym = sp.Symbol(name, real=True)
            for a in atoms:
                sub_dict[a] = sym
            row = n_aux_before + len(new_syms)
            new_syms.append(sym)
            function_row_of_name[name] = row
            registry.append({
                "kind": "function",
                "name": name,
                "row": row,
                "atom": atoms[0],
                "aux_symbol": sym,
            })

        # ── Derivative entries. ───────────────────────────────────
        for (target_name, mi), atoms in deriv_atoms.items():
            suffix = "_".join(
                str(self.space[d])
                for d in range(n_dim) for _ in range(mi[d])
            )
            aux_name = f"{target_name}_{suffix}"
            sym = sp.Symbol(aux_name, real=True)
            for d in atoms:
                sub_dict[d] = sym
            row = n_aux_before + len(new_syms)
            new_syms.append(sym)
            entry = {
                "kind": "derivative",
                "name": aux_name,
                "row": row,
                "atom": atoms[0],
                "aux_symbol": sym,
                "target_name": target_name,
                "multi_index": tuple(mi),
            }
            if target_name in state_names:
                entry["target_kind"] = "state"
                entry["state_index"] = next(
                    i for i, s in enumerate(self.state)
                    if str(s) == target_name
                )
            elif target_name in function_row_of_name:
                entry["target_kind"] = "function"
                entry["function_row"] = function_row_of_name[target_name]
            else:
                entry["target_kind"] = "unknown"
            registry.append(entry)

        # ── Apply substitutions to every matrix. ──────────────────
        self.flux = self.flux.xreplace(sub_dict)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(
            sub_dict)
        self.source = self.source.xreplace(sub_dict)
        self.mass_matrix = self.mass_matrix.xreplace(sub_dict)
        B = self.nonconservative_matrix
        new_B = sp.MutableDenseNDimArray.zeros(*B.shape)
        for idx in _product(*[range(s) for s in B.shape]):
            new_B[idx] = B[idx].xreplace(sub_dict)
        self.nonconservative_matrix = new_B

        self.aux_state = list(self.aux_state) + new_syms
        self.aux_registry = registry
        self.history.append({
            "name": "expose_aux_atoms",
            "description": (
                f"auto-scan: {sum(1 for r in registry if r['kind']=='function')} "
                f"function-aux, "
                f"{sum(1 for r in registry if r['kind']=='derivative')} "
                f"derivative-aux"
            ),
        })

    # ── reverse-aux substitution helper (for tests / inspection) ──

    def _aux_reverse_map(self):
        """Return ``{aux_Symbol: original_atom}`` for every entry in
        ``aux_registry`` so reconstructed residuals can be displayed in
        their original ``Derivative(…)`` form."""
        registry = getattr(self, "aux_registry", None) or []
        return {entry["aux_symbol"]: entry["atom"] for entry in registry}

    # ── (legacy) expose_functions_as_aux / expose_derivatives_as_aux ─

    def expose_functions_as_aux(self, function_names=None):
        """Replace bare :class:`~sympy.Function` atoms with auxiliary
        Symbols of the same name added to :attr:`aux_state`.

        Companion to :meth:`expose_derivatives_as_aux`.  For each
        unique Function atom whose name is in ``function_names``
        (or every free Function if ``function_names is None``), a
        fresh aux Symbol named ``{function_name}`` is substituted in
        every operator matrix.

        Use for free Functions like topography ``b(t, x, y)`` so they
        live as proper aux state (visible in VTK output, fed from a
        per-cell callable at runtime) instead of being symbolically
        deleted via ``_zero_functions_by_name``.

        The mapping ``aux_function_map: Dict[Symbol, Function]`` is
        attached to ``self`` so solvers know which aux entries are
        free-function values (vs state derivatives, vs permanent
        Zstruct aux).
        """
        import sympy as sp
        from itertools import product as _product

        names = (set(function_names)
                 if function_names is not None else None)

        def _matches(a):
            if not isinstance(a, sp.Function):
                return False
            n = a.func.__name__
            return names is None or n in names

        atoms = set()
        for M in (self.flux, self.hydrostatic_pressure,
                  self.nonconservative_matrix, self.source,
                  self.mass_matrix):
            if isinstance(M, sp.Matrix):
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        for a in M[i, j].atoms(sp.Function):
                            if _matches(a):
                                atoms.add(a)
            else:
                for idx in _product(*[range(s) for s in M.shape]):
                    for a in M[idx].atoms(sp.Function):
                        if _matches(a):
                            atoms.add(a)

        if not atoms:
            return

        # One aux Symbol per distinct function name (atoms with
        # different dummy args still map to the same conceptual field).
        sub_dict = {}
        name_to_sym = {}
        aux_function_map = getattr(self, "aux_function_map", {})
        for a in atoms:
            name = a.func.__name__
            if name not in name_to_sym:
                name_to_sym[name] = sp.Symbol(name, real=True)
            sub_dict[a] = name_to_sym[name]
            aux_function_map[name_to_sym[name]] = a

        # Apply substitution.
        self.flux = self.flux.xreplace(sub_dict)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(
            sub_dict)
        self.source = self.source.xreplace(sub_dict)
        self.mass_matrix = self.mass_matrix.xreplace(sub_dict)
        B = self.nonconservative_matrix
        new_B = sp.MutableDenseNDimArray.zeros(*B.shape)
        for idx in _product(*[range(s) for s in B.shape]):
            new_B[idx] = B[idx].xreplace(sub_dict)
        self.nonconservative_matrix = new_B

        self.aux_state = list(self.aux_state) + list(name_to_sym.values())
        self.aux_function_map = aux_function_map
        self.history.append({
            "name": "expose_functions_as_aux",
            "description": (
                f"intercepted {len(atoms)} Function atom(s) → aux "
                f"Symbols {sorted(name_to_sym.keys())}"
            ),
        })


    # ── expose_derivatives_as_aux ─────────────────────────────────────

    def expose_derivatives_as_aux(self):
        """Replace ``Derivative(target, axes...)`` atoms in every
        operator matrix with auxiliary Symbols added to
        ``self.aux_state``.

        Each unique Derivative atom found in ``flux``,
        ``hydrostatic_pressure``, ``nonconservative_matrix``,
        ``source``, or ``mass_matrix`` is replaced with a fresh aux
        Symbol named ``{target}_{axes}``  (e.g. ``h_x``, ``U_0_x_x``,
        ``b_x``).  The Symbols are appended to ``self.aux_state`` and
        recorded in ``self.aux_derivative_map: Dict[Symbol,
        Derivative]`` so a solver can compute their per-cell values
        (state derivatives via mesh + LSQ reconstruction; parameter
        Function derivatives via analytical / tabulated evaluation).

        This makes spatial derivatives — at any order — first-class
        runtime inputs without changing the
        ``(Q, Qaux, p) → ndarray`` operator-API surface.

        Mutates ``self`` in place; no-op if the matrices contain no
        Derivative atoms.
        """
        from itertools import product

        def _matrix_entries(M):
            if isinstance(M, sp.Matrix):
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        yield (i, j), M[i, j]
            else:
                shape = M.shape
                for idx in product(*[range(s) for s in shape]):
                    yield idx, M[idx]

        derivative_subs = {}
        aux_derivative_map = {}

        def _aux_name(d):
            target = d.args[0]
            target_name = (target.func.__name__
                           if isinstance(target, sp.Function)
                           else str(target))
            axes = []
            for var, n in d.variable_count:
                axes.extend([str(var)] * int(n))
            return f"{target_name}_" + "_".join(axes)

        for M in (self.flux, self.hydrostatic_pressure,
                  self.nonconservative_matrix, self.source,
                  self.mass_matrix):
            for _, entry in _matrix_entries(M):
                for d in sp.sympify(entry).atoms(sp.Derivative):
                    if d in derivative_subs:
                        continue
                    aux_sym = sp.Symbol(_aux_name(d), real=True)
                    derivative_subs[d] = aux_sym
                    aux_derivative_map[aux_sym] = d

        if not derivative_subs:
            return

        # Apply substitution to every matrix.  sp.Matrix has xreplace;
        # NDimArray (the NCP) does not, so iterate entries.
        self.flux = self.flux.xreplace(derivative_subs)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(
            derivative_subs)
        self.source = self.source.xreplace(derivative_subs)
        self.mass_matrix = self.mass_matrix.xreplace(derivative_subs)
        B = self.nonconservative_matrix
        shape = B.shape
        new_B = sp.MutableDenseNDimArray.zeros(*shape)
        for idx in product(*[range(s) for s in shape]):
            new_B[idx] = B[idx].xreplace(derivative_subs)
        self.nonconservative_matrix = new_B

        # Extend aux_state with the new derivative Symbols.
        self.aux_state = list(self.aux_state) + list(
            derivative_subs.values())
        self.aux_derivative_map = aux_derivative_map

        self.history.append({
            "name": "expose_derivatives_as_aux",
            "description": (
                f"intercepted {len(derivative_subs)} Derivative atom(s) "
                f"→ aux Symbols "
                f"{[str(s) for s in derivative_subs.values()]}"
            ),
        })


    # ── change_state_variables ────────────────────────────────────────

    def change_state_variables(self, new_state, transform):
        """Apply a state-variable change of variables in place.

        Parameters
        ----------
        new_state : list
            The new state vector (same length as ``self.state``).
            Entries that already appear in the old state stay; entries
            that replace an old state must appear in ``transform``.
        transform : dict
            Maps each OLD state entry that is being replaced to its
            expression in the NEW state variables.  Old states not
            mentioned are assumed to map identically to themselves.

        Updates ``flux``, ``hydrostatic_pressure``,
        ``nonconservative_matrix``, ``source``, ``mass_matrix``, and
        ``state`` in place.

        Mechanics.  Let ``T_i(Q_new) = transform[Q_old[i]]`` (identity
        if not in ``transform``) and ``J[i, j] = ∂T_i/∂Q_new[j]``.  In
        operator form the system is invariant under the substitution
        ``Q_old → T(Q_new)`` together with the time-derivative
        identity ``∂_t Q_old = J · ∂_t Q_new`` and the spatial-
        derivative identity ``∂_d Q_old = J · ∂_d Q_new``:

          * ``F_new = F_old(T(Q_new))``,
            ``P_new = P_old(T(Q_new))``,
            ``S_new = S_old(T(Q_new))``.
          * ``B_new[i, k, d] = Σ_j B_old[i, j, d](T) · J[j, k]``.
          * ``M_new = M_old(T(Q_new)) · J``.

        This contract preserves the residual ``M ∂_t Q + ∂_x F + ∂_x P
        + B · ∂_x Q − S`` under the change of variables.
        """
        n = self.n_state
        if len(new_state) != n:
            raise ValueError(
                f"new_state has length {len(new_state)}, expected {n}"
            )

        old_state = list(self.state)
        full_transform = {}
        for old_s in old_state:
            if old_s in transform:
                full_transform[old_s] = sp.sympify(transform[old_s])
            else:
                if old_s not in new_state:
                    raise ValueError(
                        f"old state {old_s!r} is not in new_state and "
                        f"has no entry in transform — cannot map it"
                    )
                full_transform[old_s] = old_s

        # Jacobian J[i, j] = ∂T_i / ∂new_state[j]
        J = sp.zeros(n, n)
        for i, old_s in enumerate(old_state):
            T_i = full_transform[old_s]
            for j, new_s in enumerate(new_state):
                J[i, j] = sp.diff(T_i, new_s)

        # Substitute T into operators.
        new_flux = sp.expand(self.flux.xreplace(full_transform))
        new_P = sp.expand(self.hydrostatic_pressure.xreplace(full_transform))
        new_S = sp.expand(self.source.xreplace(full_transform))

        # NCP: B_new[i, k, d] = Σ_j B_old[i, j, d](T) · J[j, k]
        n_eq = self.n_equations
        n_dim = self.n_dim
        new_B = sp.MutableDenseNDimArray.zeros(n_eq, n, n_dim)
        for d in range(n_dim):
            B_slab = sp.Matrix(
                n_eq, n,
                lambda i, j, _d=d: sp.sympify(
                    self.nonconservative_matrix[i, j, _d]
                ),
            )
            B_slab_sub = B_slab.xreplace(full_transform)
            B_slab_new = sp.expand(B_slab_sub * J)
            for i in range(n_eq):
                for k in range(n):
                    new_B[i, k, d] = B_slab_new[i, k]

        # Mass matrix: M_new = M_old(T) · J
        M_old_sub = self.mass_matrix.xreplace(full_transform)
        new_M = sp.expand(M_old_sub * J)

        self.state = list(new_state)
        self.flux = new_flux
        self.hydrostatic_pressure = new_P
        self.nonconservative_matrix = new_B
        self.source = new_S
        self.mass_matrix = new_M
        self.history.append({
            "name": "change_state_variables",
            "description": (
                "state: ["
                + ", ".join(str(s) for s in old_state)
                + "] → ["
                + ", ".join(str(s) for s in new_state)
                + "]; transform: "
                + ", ".join(f"{k}↦{v}" for k, v in transform.items())
            ),
        })
        return self

    # ── describe ──────────────────────────────────────────────────────

    def describe(self, full: bool = False) -> "SystemModelDescription":
        """Return a Description rendering the operator form.  ``full=True``
        includes symbolic flux/NCP/source/mass-matrix entries."""
        return SystemModelDescription(self, full=full)


class SystemModelRow:
    """Row view of a :class:`SystemModel` — exposes row ``i`` of
    every stored matrix."""

    def __init__(self, parent: SystemModel, i: int):
        self._parent = parent
        self._i = i

    @property
    def flux(self):
        return self._parent.flux[self._i, :]

    @property
    def hydrostatic_pressure(self):
        return self._parent.hydrostatic_pressure[self._i, :]

    @property
    def source(self):
        return self._parent.source[self._i, 0]

    @property
    def mass_matrix_row(self):
        return self._parent.mass_matrix[self._i, :]

    @property
    def nonconservative_matrix_row(self):
        n = self._parent.n_equations
        d = self._parent.n_dim
        out = sp.MutableDenseNDimArray.zeros(n, d)
        for j in range(n):
            for k in range(d):
                out[j, k] = self._parent.nonconservative_matrix[self._i, j, k]
        return out


class SystemModelDescription:
    """Markdown / plaintext description of a SystemModel.

    Renders via ``_repr_markdown_`` in Jupyter; ``str(...)`` gives a
    plain-text fallback.
    """

    def __init__(self, sm: SystemModel, *, full: bool = False):
        self._sm = sm
        self._full = full

    def _operator_block(self) -> str:
        sm = self._sm
        parts = []

        # Canonical equation form.
        parts.append("**System form:**")
        parts.append(
            r"$$"
            r"M(Q)\,\partial_t Q "
            r"+ \nabla\cdot\!\big(F(Q) + P(Q)\big)"
            r" + \sum_{d} B_{d}(Q)\,\partial_{d} Q "
            r"- S(Q) = 0"
            r"$$"
        )

        # State vector.
        Q_vec = sp.Matrix(list(sm.state))
        parts.append("**State $Q$:**")
        parts.append(f"$$\n{sp.latex(Q_vec)}\n$$")

        def _render(label, mat):
            if mat.is_zero_matrix:
                parts.append(f"**{label} $= 0$**")
            else:
                parts.append(f"**{label}:**")
                parts.append(f"$$\n{sp.latex(mat)}\n$$")

        _render("Mass matrix $M$", sm.mass_matrix)
        _render("Flux $F$", sm.flux)
        _render("Hydrostatic pressure $P$", sm.hydrostatic_pressure)
        for d in range(sm.n_dim):
            slab = sp.Matrix(sm.n_equations, sm.n_equations,
                             lambda i, j, _d=d:
                                 sm.nonconservative_matrix[i, j, _d])
            label = (f"NCP $B_{{{d}}}$ "
                     f"(direction ${sp.latex(sm.space[d])}$)")
            _render(label, slab)
        _render("Source $S$", sm.source)
        return "\n\n".join(parts)

    def _repr_markdown_(self) -> str:
        sm = self._sm
        parts = [
            f"**SystemModel** — {sm.n_equations} equations, "
            f"{sm.n_dim} spatial dimension{'s' if sm.n_dim != 1 else ''}",
            f"**State:** {', '.join(str(s) for s in sm.state)}",
        ]
        if sm.parameters:
            parts.append(
                "**Parameters:** "
                + ", ".join(f"${sp.latex(s)} = {v}$"
                            for s, v in sm.parameters.items())
            )
        if sm.history:
            parts.append(
                "**Operations:** "
                + ", ".join(h["name"] for h in sm.history)
            )
        if self._full:
            parts.append(self._operator_block())
        return "\n\n".join(parts)

    def __str__(self) -> str:
        md = self._repr_markdown_()
        return md.replace("$$", "").replace("$", "").replace("**", "")

    def __repr__(self) -> str:
        return self.__str__()


# ── System-level operations ───────────────────────────────────────────

class InvertMassMatrix:
    """System-level op: left-multiply the system by ``M⁻¹``.

    Verifies the mass matrix is constant (no state / aux-state atoms),
    inverts symbolically, and applies ``M⁻¹`` to ``flux``,
    ``hydrostatic_pressure``, ``source``, and each per-direction slab
    of ``nonconservative_matrix``.  Sets ``mass_matrix`` to identity.
    """

    name = "invert_mass_matrix"
    description = "Left-multiply by M⁻¹ to reach canonical ∂_t Q form."

    def __call__(self, sm: SystemModel):
        if not sm.is_square:
            raise ValueError(
                "InvertMassMatrix: requires a square system "
                f"(n_eq == n_state); got n_eq={sm.n_equations}, "
                f"n_state={sm.n_state}, "
                f"equation_to_state_index={sm.equation_to_state_index}."
            )
        state_atoms = set(sm.state) | set(sm.aux_state)
        if state_atoms and sm.mass_matrix.has(*state_atoms):
            raise ValueError(
                "InvertMassMatrix: mass_matrix has state-dependent "
                "entries — non-constant mass matrices are unsupported.\n"
                f"Mass matrix:\n{sm.mass_matrix}"
            )

        if sm.mass_matrix == sp.eye(sm.n_equations):
            return

        M_inv = sm.mass_matrix.inv()
        sm.flux = M_inv * sm.flux
        sm.hydrostatic_pressure = M_inv * sm.hydrostatic_pressure
        sm.source = M_inv * sm.source

        n = sm.n_equations
        d = sm.n_dim
        new_B = sp.MutableDenseNDimArray.zeros(n, n, d)
        for k in range(d):
            slab = sp.Matrix(n, n,
                             lambda i, j, _k=k:
                                 sm.nonconservative_matrix[i, j, _k])
            slab = M_inv * slab
            for i in range(n):
                for j in range(n):
                    new_B[i, j, k] = slab[i, j]
        sm.nonconservative_matrix = new_B
        sm.mass_matrix = sp.eye(n)


__all__ = [
    "SystemModel",
    "SystemModelRow",
    "SystemModelDescription",
    "InvertMassMatrix",
]
