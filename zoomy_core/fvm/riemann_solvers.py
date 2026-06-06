"""Symbolic Riemann solvers: Rusanov, positive, nonconservative variants.

Every index-dependent class (``PositiveRusanov``, ``NonconservativeRusanov``,
``PositiveNonconservativeRusanov``, …) auto-locates the named fields
``h`` and ``b`` (and ``hinv`` when present) in ``model.state`` /
``model.aux_state`` via :class:`FieldHandle` so the same Riemann
code path works whether bathymetry is part of the conservative state
(legacy SWE) or lives in ``Qaux`` (chain-DAE convention).
"""

import numpy as np
import param
import sympy as sp

from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.basefunction import SymbolicRegistrar
from zoomy_core.model.kernel_functions import (
    conditional, max_wavespeed, roe_dissipation,
)
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.transformation.to_numpy import (
    NumpyRuntimeModel, NumpyRuntimeSymbolic,
)


class FieldHandle:
    """A named field resolved to either ``Q`` (state) or ``Qaux``.

    Looks up the location at construction time via :meth:`Numerics.find_field`,
    then exposes the symbolic Symbol on the **minus / plus / state** side of
    a face so callers can write reconstruction code without knowing where
    the field lives.

    Attributes
    ----------
    name : str
        Field name (e.g. ``"h"``, ``"b"``, ``"hinv"``).
    container : {"q", "qaux"}
        Which array carries this field.
    index : int
        Index into that array.
    minus, plus, state : sympy.Symbol
        Direct references — ``minus`` and ``plus`` are the per-face
        symbolic state on the L / R side of a Rusanov face; ``state``
        is the cell-centre reference.  Using these in symbolic code
        produces the same lambdified output as ``Q[index]`` /
        ``Qaux[index]`` — a true placeholder.
    """

    __slots__ = ("name", "container", "index", "minus", "plus", "state")

    def __init__(self, name, container, index, minus, plus, state):
        if container not in {"q", "qaux"}:
            raise ValueError(
                f"FieldHandle({name}): container must be 'q' or 'qaux', "
                f"got {container!r}"
            )
        self.name = name
        self.container = container
        self.index = index
        self.minus = minus
        self.plus = plus
        self.state = state

    def access(self, q_array, qaux_array):
        """Return ``q_array[index]`` or ``qaux_array[index]``
        depending on container.  Works for symbolic ``ZArray`` and
        numeric numpy arrays alike."""
        return (q_array[self.index] if self.container == "q"
                else qaux_array[self.index])

    def assign(self, q_array, qaux_array, value):
        """In-place write into the appropriate container."""
        if self.container == "q":
            q_array[self.index] = value
        else:
            qaux_array[self.index] = value

    def __repr__(self):
        return (f"FieldHandle({self.name!r}: "
                f"{self.container}[{self.index}])")


class Numerics(param.Parameterized, SymbolicRegistrar):
    """Symbolic numerics over a :class:`SystemModel`.

    The numerics consumes a :class:`SystemModel` — the frozen
    operator-form snapshot of a derivation.  A :class:`Model` passed to
    the constructor is normalised once via
    :meth:`SystemModel.from_model`; everything internal reads the
    SystemModel's stored operators (``flux``, ``hydrostatic_pressure``,
    ``eigenvalues``, ``nonconservative_matrix``, ``quasilinear_matrix``)
    and its ``state`` / ``aux_state`` / ``parameters`` / ``normal``.

    Subclasses that depend on the location of named fields (``h``,
    ``b``, ``hinv``, …) call :meth:`find_field` to obtain a
    :class:`FieldHandle` — the search walks state then aux
    automatically, so the same numerics works whether bathymetry is in
    the state or in ``aux_state``."""

    name = param.String(default="NumericsV2")
    model = param.Parameter(
        default=None,
        doc="The SystemModel the numerics operates on (a Model is "
            "normalised via SystemModel.from_model in __init__).",
    )

    scaled_q_indices = param.List(default=None, allow_None=True)

    def __init__(self, model, **params):
        """Initialize the instance.

        ``model`` may be a :class:`SystemModel` (used directly) or a
        :class:`Model` (normalised once via
        :meth:`SystemModel.from_model`).
        """
        sm = (model if isinstance(model, SystemModel)
              else SystemModel.from_model(model))
        super().__init__(model=sm, **params)
        self.functions, self.call = Zstruct(), Zstruct()

        self.n_variables = self.model.n_equations
        self.n_aux_variables = len(self.model.aux_state)

        self.variables = ZArray(list(self.model.state))
        self.aux_variables = ZArray(list(self.model.aux_state))
        # Parameter / normal symbols (used in symbolic derivation).
        self.parameters = ZArray(list(self.model.parameters.values()))
        self.normal = ZArray(list(self.model.normal.values()))

        self.variables_minus = self._create_v("Q_minus", self.n_variables)
        self.variables_plus = self._create_v("Q_plus", self.n_variables)
        self.aux_variables_minus = self._create_v(
            "Qaux_minus", self.n_aux_variables)
        self.aux_variables_plus = self._create_v(
            "Qaux_plus", self.n_aux_variables)

        self.flux_minus = self._create_v("flux_minus", self.n_variables)
        self.flux_plus = self._create_v("flux_plus", self.n_variables)
        self.source_term = self._create_v("source_term", self.n_variables)

        # Auto-locate the standard named fields (``h``, ``b``, ``hinv``)
        # if present.  Subclasses can call ``find_field`` to register
        # additional handles.
        self._field_handles = {}
        for name in ("h", "b", "hinv"):
            h = self.find_field(name, required=False)
            if h is not None:
                self._field_handles[name] = h
        self._scaled_q_indices = self._resolve_scaled_q_indices(
            self.scaled_q_indices)

        self._initialize_functions()

    # ── Field-search API ──────────────────────────────────────────

    def find_field(self, name, *, required=True):
        """Return a :class:`FieldHandle` for the named field.

        Searches ``self.model.state`` (Q) first, then
        ``self.model.aux_state`` (Qaux).  Caches the result in
        ``self._field_handles[name]`` so repeat lookups are free.

        Parameters
        ----------
        name : str
        required : bool
            If True (default), raise ``KeyError`` when the field is
            in neither container.  Otherwise return ``None``.
        """
        cache = getattr(self, "_field_handles", None) or {}
        if name in cache:
            return cache[name]
        state_list = list(self.model.state)
        state_names = [str(s) for s in state_list]
        if name in state_names:
            i = state_names.index(name)
            h = FieldHandle(name, "q", i,
                            minus=self.variables_minus[i],
                            plus=self.variables_plus[i],
                            state=self.variables[i])
            cache[name] = h
            return h
        aux_list = list(self.model.aux_state)
        aux_names = [str(s) for s in aux_list]
        if name in aux_names:
            i = aux_names.index(name)
            h = FieldHandle(name, "qaux", i,
                            minus=self.aux_variables_minus[i],
                            plus=self.aux_variables_plus[i],
                            state=self.aux_variables[i])
            cache[name] = h
            return h
        if required:
            raise KeyError(
                f"Field {name!r} not found in model.state "
                f"({state_names}) nor model.aux_state ({aux_names})"
            )
        return None

    def has_field(self, name):
        return self.find_field(name, required=False) is not None

    def _create_v(self, name, size):
        """Internal helper `_create_v`."""
        v = ZArray([sp.Symbol(f"{name}_{i}", real=True) for i in range(size)])
        v._symbolic_name = name
        return v

    def _resolve_scaled_q_indices(self, scaled_q_indices):
        """Internal helper `_resolve_scaled_q_indices`.  Default excludes
        ``h`` and ``b`` from the depth-scaled Q rows when they live in
        the state."""
        if scaled_q_indices is not None:
            cleaned = [int(i) for i in scaled_q_indices]
            for i in cleaned:
                if i < 0 or i >= self.n_variables:
                    raise IndexError(
                        f"scaled_q_indices contains out-of-bounds index {i}."
                    )
            return cleaned

        excluded = set()
        for name in ("h", "b"):
            h = self._field_handles.get(name)
            if h is not None and h.container == "q":
                excluded.add(h.index)
        return [i for i in range(self.n_variables) if i not in excluded]

    def _eps_symbol(self):
        """Internal helper `_eps_symbol`."""
        # Symbolic eps parameter if the model declares one.
        if self.model.parameters.contains("eps"):
            return self.model.parameters.eps
        return sp.Float(1e-12)

    def _initialize_functions(self):
        """Internal helper `_initialize_functions`."""
        sig = Zstruct(
            q_minus=self.variables_minus,
            q_plus=self.variables_plus,
            aux_minus=self.aux_variables_minus,
            aux_plus=self.aux_variables_plus,
            p=self.parameters,
            normal=self.normal,
        )

        self.register_symbolic_function("numerical_flux", self.numerical_flux, sig)
        self.register_symbolic_function(
            "numerical_fluctuations", self.numerical_fluctuations, sig
        )

        eig_sig = Zstruct(
            Q=self.variables, Qaux=self.aux_variables, p=self.parameters, n=self.normal
        )
        self.register_symbolic_function(
            "local_max_abs_eigenvalue", self.local_max_eigenvalue_definition, eig_sig
        )

    def local_max_eigenvalue_definition(self):
        """
        Returns the opaque max_wavespeed function.
        The actual implementation is provided by the backend at runtime.
        """
        return max_wavespeed(
            *list(self.variables), *list(self.aux_variables),
            *list(self.parameters), *list(self.normal),
        )

    def local_max_abs_eigenvalue(self, Q=None, Qaux=None, p=None, n=None):
        """
        Called during symbolic Rusanov construction.
        Returns opaque max_wavespeed with the given state.
        """
        if Q is None:
            return self.local_max_eigenvalue_definition()
        return max_wavespeed(*list(Q), *list(Qaux), *list(p), *list(n))

    def _model_eval(self, operator_name, Q, Qaux, p, n=None):
        """Evaluate a SystemModel operator at a given state.

        ``operator_name`` is the SystemModel attribute name — ``flux``,
        ``hydrostatic_pressure``, ``eigenvalues``,
        ``nonconservative_matrix`` or ``quasilinear_matrix``.  Returns
        the operator with ``state`` / ``aux_state`` / ``parameters`` /
        ``normal`` symbols substituted by the supplied values.
        """
        definition = getattr(self.model, operator_name)
        sub_map = {}
        for sym, val in zip(self.model.state, list(Q)):
            sub_map[sym] = val
        for sym, val in zip(self.model.aux_state, list(Qaux)):
            sub_map[sym] = val
        for sym, val in zip(self.model.parameters.values(), list(p)):
            sub_map[sym] = val
        if n is not None:
            for sym, val in zip(self.model.normal.values(), list(n)):
                sub_map[sym] = val

        if isinstance(definition, sp.NDimArray):
            # ``NDimArray`` exposes ``subs`` only on the *immutable*
            # variant; ``SystemModel`` stores the nonconservative /
            # quasilinear tensors as mutable arrays, so normalise to
            # immutable before substituting (otherwise the state
            # symbols leak through unsubstituted).
            out = sp.ImmutableDenseNDimArray(definition).subs(sub_map)
        elif hasattr(definition, "subs"):
            out = definition.subs(sub_map)
        else:
            out = definition

        if isinstance(out, ZArray):
            return out
        if isinstance(out, (sp.NDimArray, sp.MatrixBase, list, tuple)):
            return ZArray(out)
        return out

    def to_runtime(self, backend="numpy"):
        """To runtime."""
        backend = backend.lower()
        if backend == "numpy":
            return NumpyRuntimeSymbolic(self)
        if backend == "ufl":
            from zoomy_core.transformation.to_ufl import UFLRuntimeSymbolic
            return UFLRuntimeSymbolic(self)
        raise ValueError(f"Unsupported runtime backend '{backend}'.")

    def to_runtime_numpy(self):
        """To runtime numpy."""
        return self.to_runtime("numpy")

    def to_runtime_ufl(self):
        """Lambdify all registered symbolic functions through the UFL
        module dict — for use by Firedrake-based backends."""
        return self.to_runtime("ufl")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.n_variables
        return ZArray(zeros)

    def numerical_fluctuations(self):
        """Numerical fluctuations."""
        zeros = ZArray.zeros(self.n_variables)
        return ZArray([zeros, zeros])


class Rusanov(Numerics):
    """Rusanov. (class)."""
    name = param.String(default="RusanovV2")

    def get_viscosity_identity_flux(self):
        """Get viscosity identity flux."""
        n = self.n_variables
        Id = ZArray.zeros(n, n)
        for i in range(n):
            Id[i, i] = 1
        # Exclude bottom-topography from Rusanov dissipation when b is part of
        # the conservative state. This preserves the stationary bed variable.
        b = self._field_handles.get("b")
        if b is not None and b.container == "q":
            Id[b.index, b.index] = 0
        return Id

    def get_viscosity_identity_fluctuations(self):
        # Conservative Rusanov variants do not use fluctuation viscosity.
        """Get viscosity identity fluctuations."""
        n = self.n_variables
        return ZArray.zeros(n, n)

    def numerical_flux(self):
        """Numerical flux."""
        return self._compute_flux(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        """Internal helper `_compute_flux`."""
        FL = self._model_eval("flux", qL, auxL, p)
        FR = self._model_eval("flux", qR, auxR, p)
        PL = self._model_eval("hydrostatic_pressure", qL, auxL, p)
        PR = self._model_eval("hydrostatic_pressure", qR, auxR, p)
        s_max = sp.Max(
            self.local_max_abs_eigenvalue(qL, auxL, p, n),
            self.local_max_abs_eigenvalue(qR, auxR, p, n),
        )
        Id = self.get_viscosity_identity_flux()
        return 0.5 * ((FL + PL) @ n + (FR + PR) @ n) - 0.5 * s_max * Id @ (qR - qL)


class HLL(Numerics):
    """HLL (Harten-Lax-van-Leer) approximate Riemann solver.

    Model-agnostic: needs only the SystemModel's ``flux``,
    ``hydrostatic_pressure`` and ``eigenvalues`` operators.  When the
    SystemModel carries a symbolic spectrum the wave-speed bounds are
    the Davis estimates — min / max over the eigenvalues of *both* face
    states.  When ``eigenvalues`` is ``None`` (the model skipped the
    spectral derivation) it falls back to ``± local_max_abs_eigenvalue``,
    i.e. HLL collapses to local Lax-Friedrichs (a valid, more diffusive
    HLL).

    The numerical flux is a single closed-form (branch-free) SymPy
    expression — clamping the wave speeds with ``Min(s_L, 0)`` /
    ``Max(s_R, 0)`` recovers the upwind branches without ``Piecewise``,
    so it codegens cleanly to every backend.
    """

    name = param.String(default="HLLV2")

    def numerical_flux(self):
        """Numerical flux."""
        return self._compute_flux(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )

    def wave_speed_bounds(self, qL, qR, auxL, auxR, p, n):
        """Return ``(s_L, s_R)`` — slowest / fastest signal speeds at the face."""
        if self.model.eigenvalues is None:
            a = sp.Max(
                self.local_max_abs_eigenvalue(qL, auxL, p, n),
                self.local_max_abs_eigenvalue(qR, auxR, p, n),
            )
            return -a, a
        eig = list(sp.flatten(self._model_eval("eigenvalues", qL, auxL, p, n)))
        eig += list(sp.flatten(self._model_eval("eigenvalues", qR, auxR, p, n)))
        return sp.Min(*eig), sp.Max(*eig)

    def _physical_flux_n(self, q, aux, p, n):
        """Normal-projected physical flux ``(F + P) @ n`` for a single state."""
        F = self._model_eval("flux", q, aux, p)
        P = self._model_eval("hydrostatic_pressure", q, aux, p)
        return (F + P) @ n

    def _state_jump(self, qL, qR):
        """``qR - qL`` with stationary fields (bed in conservative state) zeroed."""
        dq = ZArray(qR) - ZArray(qL)
        b = self._field_handles.get("b")
        if b is not None and b.container == "q":
            dq[b.index] = sp.Integer(0)
        return dq

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        """Internal helper `_compute_flux`."""
        FLn = self._physical_flux_n(qL, auxL, p, n)
        FRn = self._physical_flux_n(qR, auxR, p, n)
        sL, sR = self.wave_speed_bounds(qL, qR, auxL, auxR, p, n)
        sLm = sp.Min(sL, sp.Integer(0))
        sRp = sp.Max(sR, sp.Integer(0))
        eps = self._eps_symbol()
        inv = 1 / (sRp - sLm + eps)
        dq = self._state_jump(qL, qR)
        return (sRp * FLn - sLm * FRn + sLm * sRp * dq) * inv


class HLLC(HLL):
    """HLLC approximate Riemann solver for the free-surface family.

    Restores the contact / shear wave that HLL smears.  Requires a depth
    field ``h`` (resolved via :meth:`Numerics.find_field`); the momentum
    block is the first ``model.dimension`` depth-scaled Q rows.  Any
    further depth-scaled rows (higher moments) are advected by the
    contact wave; non-scaled rows (e.g. bed in conservative state) pass
    through unchanged.  Models without an ``h`` field should use
    :class:`HLL` instead.

    Region selection (``F_L | F_L* | F_R* | F_R``) uses the opaque
    ``conditional`` primitive, so it codegens to ``np.where`` / ternary
    expressions on every backend.
    """

    name = param.String(default="HLLCV2")

    @property
    def h_field(self):
        """Depth :class:`FieldHandle` — raises ``KeyError`` if the model
        has no ``h`` field (such models should use :class:`HLL`)."""
        return self.find_field("h")

    def _normal_velocity(self, q, aux, n, mom_idx):
        """Internal helper `_normal_velocity`."""
        h = self.h_field.access(q, aux)
        eps = self._eps_symbol()
        return sum(q[k] * n[d] for d, k in enumerate(mom_idx)) / (h + eps)

    def _star_state(self, q, aux, h, un, s_side, s_star, n, mom_idx):
        """HLLC star state on one side of the contact wave."""
        eps = self._eps_symbol()
        h_star = h * (s_side - un) / (s_side - s_star + eps)
        out = ZArray(q)
        if self.h_field.container == "q":
            out[self.h_field.index] = h_star
        # Momentum block: tangential part advected, normal part -> h* s*.
        for d, k in enumerate(mom_idx):
            u_k = q[k] / (h + eps)
            u_tan_k = u_k - un * n[d]
            out[k] = h_star * (u_tan_k + s_star * n[d])
        # Remaining depth-scaled rows (higher moments): advected, scale with depth.
        for k in self._scaled_q_indices:
            if k in mom_idx:
                continue
            out[k] = q[k] * h_star / (h + eps)
        return out

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        """Internal helper `_compute_flux`."""
        FLn = self._physical_flux_n(qL, auxL, p, n)
        FRn = self._physical_flux_n(qR, auxR, p, n)
        sL, sR = self.wave_speed_bounds(qL, qR, auxL, auxR, p, n)

        dim = self.model.n_dim
        mom_idx = list(self._scaled_q_indices)[:dim]
        eps = self._eps_symbol()

        hL = self.h_field.access(qL, auxL)
        hR = self.h_field.access(qR, auxR)
        unL = self._normal_velocity(qL, auxL, n, mom_idx)
        unR = self._normal_velocity(qR, auxR, n, mom_idx)

        # Contact-wave speed (depth-weighted HLL middle state).
        num = sL * hR * (unR - sR) - sR * hL * (unL - sL)
        den = hR * (unR - sR) - hL * (unL - sL)
        s_star = num / (den + eps)

        QLs = self._star_state(qL, auxL, hL, unL, sL, s_star, n, mom_idx)
        QRs = self._star_state(qR, auxR, hR, unR, sR, s_star, n, mom_idx)

        FLs = FLn + sL * (QLs - ZArray(qL))
        FRs = FRn + sR * (QRs - ZArray(qR))

        flux = conditional(
            sL >= 0,
            FLn,
            conditional(
                s_star >= 0,
                FLs,
                conditional(sR > 0, FRs, FRn),
            ),
        )
        return ZArray(flux)


class PositiveRusanov(Rusanov):
    """PositiveRusanov. (class).

    Hydrostatic reconstruction follows Audusse-Bristeau-Klein:
    ``h_L* = max(0, h_L + b_L − b*)``, ``b* = max(b_L, b_R)``.
    The ``h``, ``b`` and (optional) ``hinv`` fields are resolved via
    :meth:`Numerics.find_field` so the same logic works whether
    bathymetry is part of conservative state or lives in ``Qaux``.
    """

    name = param.String(default="PositiveRusanovV2")

    # Field handles are exposed as lazy properties (not set in
    # ``__init__``): ``Numerics.__init__`` eagerly builds the symbolic
    # ``numerical_flux`` — which calls ``hydrostatic_reconstruction`` —
    # *before* a subclass ``__init__`` body runs past
    # ``super().__init__()``.  Resolving on first access hits the
    # ``_field_handles`` cache that ``Numerics.__init__`` already
    # populates.  (Same pattern as :attr:`HLLC.h_field`.)

    @property
    def h_field(self):
        """Depth :class:`FieldHandle`."""
        return self.find_field("h")

    @property
    def b_field(self):
        """Bathymetry :class:`FieldHandle`."""
        return self.find_field("b")

    @property
    def hinv_field(self):
        """Optional ``1/h`` :class:`FieldHandle` (``None`` if absent)."""
        return self.find_field("hinv", required=False)

    def hydrostatic_reconstruction(self, qL, qR, auxL, auxR):
        """Hydrostatic reconstruction."""
        qLs = ZArray(qL)
        qRs = ZArray(qR)
        qauxL = ZArray(auxL)
        qauxR = ZArray(auxR)

        bL = self.b_field.access(qL, auxL)
        bR = self.b_field.access(qR, auxR)
        hL = self.h_field.access(qL, auxL)
        hR = self.h_field.access(qR, auxR)

        b_star = sp.Max(bL, bR)
        hL_star = sp.Max(0.0, hL + bL - b_star)
        hR_star = sp.Max(0.0, hR + bR - b_star)

        # Audusse momentum rescaling regularization.  Must be at
        # **floating-point scale** (just enough to avoid the ``0 / 0``
        # at truly dry faces), *not* the model's wet/dry threshold
        # ``eps`` (typically ``1e-2`` for SWE).  Using the wet/dry
        # threshold here biases the rescaling — for any face value
        # ``hL ∈ (0, eps)`` it scales the momentum down by
        # ``hL / eps`` instead of leaving it unchanged, which breaks
        # the cell-mean positivity decomposition of Xing & Zhang 2013
        # (J. Sci. Comput. 57(1):19-41, doi:10.1007/s10915-013-9695-y).
        # The ``hinv`` slot — when present — gets the same FP-scale
        # regularization for the same reason: velocity ``u = hu · hinv``
        # would otherwise be artificially compressed in thin-water cells.
        hr_eps = sp.Float(1e-14)
        hL_eff = sp.Max(hL, hr_eps)
        hR_eff = sp.Max(hR, hr_eps)

        self.b_field.assign(qLs, qauxL, b_star)
        self.b_field.assign(qRs, qauxR, b_star)
        self.h_field.assign(qLs, qauxL, hL_star)
        self.h_field.assign(qRs, qauxR, hR_star)

        for idx in self._scaled_q_indices:
            qLs[idx] = qL[idx] * hL_star / hL_eff
            qRs[idx] = qR[idx] * hR_star / hR_eff

        if self.hinv_field is not None:
            self.hinv_field.assign(qLs, qauxL, 1 / (hL_star + hr_eps))
            self.hinv_field.assign(qRs, qauxR, 1 / (hR_star + hr_eps))

        return qLs, qRs, qauxL, qauxR

    def numerical_flux(self):
        """Numerical flux."""
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
        return self._compute_flux(qLs, qRs, qauxL, qauxR, self.parameters, self.normal)

    def numerical_fluctuations(self):
        """Numerical fluctuations."""
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )

        P_mat_L_raw = self._model_eval(
            "hydrostatic_pressure",
            self.variables_minus,
            self.aux_variables_minus,
            self.parameters,
        )
        P_mat_R_raw = self._model_eval(
            "hydrostatic_pressure",
            self.variables_plus,
            self.aux_variables_plus,
            self.parameters,
        )
        P_mat_L_star = self._model_eval(
            "hydrostatic_pressure", qLs, qauxL, self.parameters
        )
        P_mat_R_star = self._model_eval(
            "hydrostatic_pressure", qRs, qauxR, self.parameters
        )

        Dm_jump = (P_mat_L_raw - P_mat_L_star) @ self.normal
        Dp_jump = (P_mat_R_star - P_mat_R_raw) @ self.normal

        base_fluct = super().numerical_fluctuations()
        Dp_base = ZArray(base_fluct[0, :])
        Dm_base = ZArray(base_fluct[1, :])

        Dp = Dp_base + Dp_jump
        Dm = Dm_base + Dm_jump

        return ZArray([Dp, Dm])


class NonconservativeRusanov(Rusanov):
    """NonconservativeRusanov. (class)."""
    name = param.String(default="NonconservativeRusanovV2")
    integration_order = param.Integer(default=3)

    def get_path_integral_states(self):
        """Get path integral states."""
        return (
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )

    def get_viscosity_identity_flux(self):
        # Nonconservative/quasilinear variants place dissipation in
        # numerical_fluctuations to avoid double counting in flux + fluctuation
        # updates.
        """Get viscosity identity flux."""
        n = self.n_variables
        return ZArray.zeros(n, n)

    def get_viscosity_identity_fluctuations(self):
        """Get viscosity identity fluctuations."""
        n = self.n_variables
        Id = ZArray.zeros(n, n)
        for i in range(n):
            Id[i, i] = 1
        # Preserve stationary bed variable when b is part of conservative state.
        b = self._field_handles.get("b")
        if b is not None and b.container == "q":
            Id[b.index, b.index] = 0
        return Id

    def numerical_fluctuations(self):
        """Numerical fluctuations."""
        qLs, qRs, qauxL, qauxR = self.get_path_integral_states()
        nc_fluct = self._compute_fluctuations(
            qLs, qRs, qauxL, qauxR, self.parameters, self.normal
        )
        out = super().numerical_fluctuations()
        return out + nc_fluct

    def _call_model_matrix(self):
        """Internal helper `_call_model_matrix`."""
        return lambda Q, Qaux, p: self._model_eval("nonconservative_matrix", Q, Qaux, p)

    def _compute_fluctuations(self, qL, qR, auxL, auxR, p, n):
        """Internal helper `_compute_fluctuations`."""
        xi_np, wi_np = np.polynomial.legendre.leggauss(self.integration_order)
        xi_np = 0.5 * (xi_np + 1)
        wi_np = 0.5 * wi_np

        dQ = qR - qL
        dAux = auxR - auxL
        n_vars = self.n_variables
        dim = len(n)

        A_int = ZArray.zeros(n_vars, n_vars)

        for xi, wi in zip(xi_np, wi_np):
            q_path = qL + xi * dQ
            aux_path = auxL + xi * dAux
            A_tensor = self._call_model_matrix()(q_path, aux_path, p)

            A_n = ZArray.zeros(n_vars, n_vars)
            for i in range(n_vars):
                for j in range(n_vars):
                    val = 0
                    for d in range(dim):
                        val += A_tensor[i, j, d] * n[d]
                    A_n[i, j] = val
            A_int += wi * A_n

        s_max = sp.Max(
            self.local_max_abs_eigenvalue(qL, auxL, p, n),
            self.local_max_abs_eigenvalue(qR, auxR, p, n),
        )

        term_advection = A_int @ dQ
        Id = self.get_viscosity_identity_fluctuations()
        term_dissipation = s_max * (Id @ dQ)

        Dp_matrix = 0.5 * (term_advection + term_dissipation)
        Dm_matrix = 0.5 * (term_advection - term_dissipation)

        return ZArray([Dp_matrix, Dm_matrix])


class WellBalancedNonconservativeRusanov(NonconservativeRusanov):
    """Path-conservative Rusanov with equilibrium-variable fluctuation
    viscosity for free-surface (lake-at-rest) well-balancing.

    Both the bed-slope ``g·h·∂_x b`` and the hydrostatic pressure
    ``g·h·∂_x h`` live in the nonconservative product (the "Malaga"
    formulation; ``hydrostatic_pressure`` is empty, ``b`` is a trivial
    conserved state with ``∂_t b = 0``).  The base
    :meth:`NonconservativeRusanov.get_viscosity_identity_fluctuations`
    already zeros the stationary-bed row.  This subclass additionally
    couples the depth-continuity row to the bed column so the Rusanov
    dissipation acts on the free-surface jump ``Δη = Δh + Δb`` instead
    of ``Δh``.  At lake-at-rest ``Δη = 0`` while ``Δh = −Δb ≠ 0``, so
    only the coupled form vanishes — giving exact well-balancing for
    both Rusanov *and* HLL-flavoured path integrals.

    For ``Q = [b, h, hu]`` the fluctuation viscosity becomes::

        [[0, 0, 0],
         [1, 1, 0],     # continuity dissipates on Δη = Δh + Δb
         [0, 0, 1]]

    Model-derived — the coupling is added only when both ``h`` and
    ``b`` resolve to conservative-state fields via
    :meth:`Numerics.find_field`.  A plain SWE model with no bed gets the
    unmodified identity (no bed-indexed term at all), as required.
    """

    name = param.String(default="WellBalancedNonconservativeRusanovV2")

    def get_viscosity_identity_fluctuations(self):
        """Get viscosity identity fluctuations (equilibrium-coupled)."""
        Id = super().get_viscosity_identity_fluctuations()
        h = self.find_field("h", required=False)
        b = self.find_field("b", required=False)
        if (h is not None and b is not None
                and h.container == "q" and b.container == "q"):
            Id[h.index, b.index] = 1
        return Id


class PositiveHLL(HLL):
    """HLL with Audusse-Bristeau-Klein hydrostatic reconstruction.

    Mirrors :class:`PositiveRusanov` but uses the sharper HLL
    two-wave numerical flux underneath instead of LF/Rusanov
    dissipation.  Recommended for free-surface dam-break / wet-dry
    simulations: the hydrostatic reconstruction enforces
    ``h_face ≥ 0`` (positivity), and HLL captures the rarefaction /
    shock fronts more accurately than Rusanov on the same mesh.

    Reconstruction (same as PositiveRusanov, Audusse-Bristeau-Klein):
    ``b* = max(b_L, b_R)``,  ``h_L* = max(0, h_L + b_L − b*)``,
    momentum scaled by ``h_L* / max(h_L, eps)``.

    The depth field ``h``, bathymetry ``b`` and (optional) ``1/h``
    inverse are resolved through :meth:`Numerics.find_field` — the
    same flux code works whether bathymetry is part of the
    conservative state or carried in ``Qaux``.
    """

    name = param.String(default="PositiveHLLV2")

    @property
    def h_field(self):
        return self.find_field("h")

    @property
    def b_field(self):
        return self.find_field("b")

    @property
    def hinv_field(self):
        return self.find_field("hinv", required=False)

    # Reuse the hydrostatic reconstruction from PositiveRusanov by
    # composition (don't multiply-inherit through Rusanov's flux).
    hydrostatic_reconstruction = PositiveRusanov.hydrostatic_reconstruction

    def numerical_flux(self):
        """HLL flux evaluated on the hydrostatically-reconstructed
        face states — positivity-preserving and well-balanced under
        the lake-at-rest steady state.

        The bed row is exactly zero by construction: HLL's
        ``_state_jump`` zeros ``dq[b]`` when ``b`` lives in the
        conservative state, and a well-posed SWE has ``F[b, :] = 0``
        symbolically.
        """
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
        return self._compute_flux(
            qLs, qRs, qauxL, qauxR, self.parameters, self.normal,
        )

    def numerical_fluctuations(self):
        """Audusse 2004 well-balancing consistency source ``S̃``.

        Audusse-Bouchut-Bristeau-Klein-Perthame 2004, SIAM J. Sci.
        Comput. 25(6):2050-2065, eq. (2.17)-(2.18):

            F_{i+1/2,L} = F_num(U*_L, U*_R) - S̃_L
            F_{i+1/2,R} = F_num(U*_L, U*_R) + S̃_R

            S̃_L = (0, ½g h_L² - ½g h*_L²)^T  =  (0, P_raw_L - P*_L)^T
            S̃_R = (0, ½g h*_R² - ½g h_R²)^T  =  (0, P*_R - P_raw_R)^T

        The HR'd numerical flux alone is not well-balanced; the per-
        cell-per-face S̃ correction restores consistency with the
        original SWE (paper, Theorem 2.5).  S̃ depends only on
        ``hydrostatic_pressure`` and is independent of the underlying
        Riemann solver — same formula already in
        :meth:`PositiveRusanov.numerical_fluctuations`.

        ``super().numerical_fluctuations()`` chains the next entry on
        the MRO.  For plain :class:`PositiveHLL` it lands at
        :meth:`Numerics.numerical_fluctuations` (returns zero).  For
        :class:`PositiveNonconservativeHLL` it lands at
        :meth:`NonconservativeRusanov.numerical_fluctuations` (the
        DLM path-integral over the NCP) — so the same override gives
        well-balanced HLL whether the model carries an NCP or not.
        """
        qLs, qRs, qauxL, qauxR = self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )

        P_mat_L_raw = self._model_eval(
            "hydrostatic_pressure",
            self.variables_minus,
            self.aux_variables_minus,
            self.parameters,
        )
        P_mat_R_raw = self._model_eval(
            "hydrostatic_pressure",
            self.variables_plus,
            self.aux_variables_plus,
            self.parameters,
        )
        P_mat_L_star = self._model_eval(
            "hydrostatic_pressure", qLs, qauxL, self.parameters
        )
        P_mat_R_star = self._model_eval(
            "hydrostatic_pressure", qRs, qauxR, self.parameters
        )

        Dm_jump = (P_mat_L_raw - P_mat_L_star) @ self.normal
        Dp_jump = (P_mat_R_star - P_mat_R_raw) @ self.normal

        base_fluct = super().numerical_fluctuations()
        Dp_base = ZArray(base_fluct[0, :])
        Dm_base = ZArray(base_fluct[1, :])

        return ZArray([Dp_base + Dp_jump, Dm_base + Dm_jump])


class PositiveNonconservativeRusanov(PositiveRusanov, NonconservativeRusanov):
    """PositiveNonconservativeRusanov. (class)."""
    name = param.String(default="PositiveNonconservativeRusanovV2")

    def get_path_integral_states(self):
        """Get path integral states."""
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )


def _make_roe_dissipation_numpy(symbolic_model):
    """Build the NumPy ``roe_dissipation`` kernel: ``|A(Q*)|·(Q_R − Q_L)``
    via numerical eigendecomposition of the full quasilinear matrix at the
    midpoint state.  Per face the n component calls share ONE eig (cached)."""
    n_vars = symbolic_model.n_variables
    n_aux = symbolic_model.n_aux_variables
    n_params = symbolic_model.n_parameters
    dim = symbolic_model.dimension
    rt = NumpyRuntimeModel.from_system_model(symbolic_model)
    ql_fn = rt.quasilinear_matrix
    keys = list(symbolic_model.variables.keys())
    b_idx = keys.index("b") if "b" in keys else None
    cache = {"key": None, "vec": None}

    def roe_diss(i, *args):
        """roe_dissipation(i, *[Q_L, Q_R, Qaux_L, Qaux_R, p, n])."""
        if cache["key"] != args:
            o = 0
            qL = np.asarray(args[o:o + n_vars], float); o += n_vars
            qR = np.asarray(args[o:o + n_vars], float); o += n_vars
            auxL = np.asarray(args[o:o + n_aux], float); o += n_aux
            auxR = np.asarray(args[o:o + n_aux], float); o += n_aux
            p = np.asarray(args[o:o + n_params], float); o += n_params
            nrm = np.asarray(args[o:o + dim], float)
            qa = 0.5 * (qL + qR)
            auxa = 0.5 * (auxL + auxR)
            ql = np.asarray(ql_fn(qa, auxa, p), float).reshape(n_vars, n_vars, dim)
            A_n = sum(ql[:, :, d] * nrm[d] for d in range(dim))
            cache["key"] = args
            cache["vec"] = _roe_abs_apply(A_n, qR - qL, b_idx)
        return float(cache["vec"][int(i)])

    return roe_diss


def _roe_abs_apply(A, dq, b_idx):
    """Return ``|A|·dq`` with ``|A| = R diag(|λ|) L`` and a Harten-Hyman
    entropy fix; fall back to scalar Rusanov dissipation ``s_max·dq`` if A is
    (near-)defective.  The stationary-bed row is never dissipated."""
    try:
        w, V = np.linalg.eig(A)
        w = np.real(w)
        V = np.real(V)
        aw = np.abs(w)
        s_max = float(aw.max()) if aw.size else 0.0
        delta = 0.1 * s_max + 1e-12                      # Harten-Hyman floor
        aw = np.where(aw < delta, 0.5 * (w * w / delta + delta), aw)
        out = V @ (aw * np.linalg.solve(V, dq))          # V diag(|λ|) V^{-1} dq
        if not np.all(np.isfinite(out)):
            raise np.linalg.LinAlgError("non-finite Roe dissipation")
    except np.linalg.LinAlgError:                        # defective -> Rusanov
        s_max = float(np.max(np.abs(np.real(np.linalg.eigvals(A)))))
        out = s_max * dq
    out = np.real(np.asarray(out, float))
    if b_idx is not None:
        out[b_idx] = 0.0
    return out


class PathConservativeRoe(PositiveNonconservativeRusanov):
    """Path-conservative Roe scheme.

    Identical to :class:`PositiveNonconservativeRusanov` (Audusse-Bristeau-
    Klein hydrostatic reconstruction + Gauss-Legendre path-integral NCP)
    EXCEPT the fluctuation dissipation: the scalar Rusanov viscosity
    ``s_max·Id·ΔQ`` is replaced by the Roe matrix dissipation
    ``|A(Q*)|·ΔQ = R|Λ|L·ΔQ``, where ``A`` is the FULL quasilinear matrix
    (``∂F/∂Q + B``) at the midpoint-averaged face state ``Q* = ½(Q_L+Q_R)``
    and ``|A|`` is formed by a runtime NUMERICAL eigendecomposition — no
    analytical eigenvectors, so the scheme is generic over any SystemModel
    (SWE, SME, VAM).  The midpoint freeze is O(jump^3)-accurate at the
    interface (vs O(jump^2) for an endpoint freeze); upwinding each
    characteristic with its own ``|λ|`` instead of a single max wave speed
    makes shocks and contacts markedly sharper than Rusanov.

    The eigendecomposition (Harten-Hyman entropy fix + defective-eigenbasis
    fallback to scalar Rusanov) runs in the backend kernel ``roe_dissipation``
    injected via :meth:`runtime_kernels`.  OpenFOAM codegen is not yet wired
    (the printer emits closed-form bodies, not Eigen calls) — numpy runtime
    for now.
    """

    name = param.String(default="PathConservativeRoeV2")

    def _compute_fluctuations(self, qL, qR, auxL, auxR, p, n):
        """Central path-integral term (as NonconservativeRusanov) with the
        scalar dissipation replaced by the Roe matrix dissipation."""
        xi_np, wi_np = np.polynomial.legendre.leggauss(self.integration_order)
        xi_np = 0.5 * (xi_np + 1)
        wi_np = 0.5 * wi_np

        dQ = qR - qL
        dAux = auxR - auxL
        n_vars = self.n_variables
        dim = len(n)

        A_int = ZArray.zeros(n_vars, n_vars)
        for xi, wi in zip(xi_np, wi_np):
            q_path = qL + xi * dQ
            aux_path = auxL + xi * dAux
            A_tensor = self._call_model_matrix()(q_path, aux_path, p)
            A_n = ZArray.zeros(n_vars, n_vars)
            for i in range(n_vars):
                for j in range(n_vars):
                    val = 0
                    for d in range(dim):
                        val += A_tensor[i, j, d] * n[d]
                    A_n[i, j] = val
            A_int += wi * A_n
        term_advection = A_int @ dQ

        # Roe matrix dissipation |A|·ΔQ — opaque numerical kernel, per row.
        flat = (list(qL) + list(qR) + list(auxL) + list(auxR)
                + list(p) + list(n))
        term_dissipation = ZArray(
            [roe_dissipation(sp.Integer(i), *flat) for i in range(n_vars)])

        Dp_matrix = 0.5 * (term_advection + term_dissipation)
        Dm_matrix = 0.5 * (term_advection - term_dissipation)
        return ZArray([Dp_matrix, Dm_matrix])

    def runtime_kernels(self, symbolic_model):
        """Backend kernels this numerics needs at runtime — merged into the
        NumpyRuntime module by the solver before lambdification."""
        return {"roe_dissipation": _make_roe_dissipation_numpy(symbolic_model)}


class PositiveNonconservativeHLL(PositiveHLL, NonconservativeRusanov):
    """HLL conservative flux + Audusse-Bristeau-Klein hydrostatic
    reconstruction + path-integral NCP fluctuations.

    Combines the sharper HLL two-wave numerical flux (vs.
    Rusanov / LF) with the same well-balanced reconstruction and
    path-integral NCP fluctuations as
    :class:`PositiveNonconservativeRusanov`.  The bed row is
    automatically excluded from the LF-style fluctuation dissipation
    via the bed mask in
    :meth:`NonconservativeRusanov.get_viscosity_identity_fluctuations`.

    The Python MRO ``(PositiveNonconservativeHLL → PositiveHLL → HLL →
    NonconservativeRusanov → Rusanov → Numerics)`` resolves
    ``numerical_flux`` to ``PositiveHLL.numerical_flux`` (hydrostatic
    reconstruction → HLL combine → bed-row mask) and
    ``numerical_fluctuations`` to ``NonconservativeRusanov.
    numerical_fluctuations`` (path-integral NCP + LF identity, bed
    masked) — exactly the split we want.
    """

    name = param.String(default="PositiveNonconservativeHLLV2")

    def get_path_integral_states(self):
        """NCP path-integral evaluated on the hydrostatically-
        reconstructed face states."""
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )


class QuasilinearRusanov(NonconservativeRusanov):
    """QuasilinearRusanov. (class)."""
    name = param.String(default="QuasilinearRusanovV2")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.n_variables
        return ZArray(zeros)

    def _call_model_matrix(self):
        """Internal helper `_call_model_matrix`."""
        return lambda Q, Qaux, p: self._model_eval("quasilinear_matrix", Q, Qaux, p)


class PositiveQuasilinearRusanov(PositiveRusanov, QuasilinearRusanov):
    """PositiveQuasilinearRusanov. (class)."""
    name = param.String(default="PositiveQuasilinearRusanovV2")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.n_variables
        return ZArray(zeros)

    def get_path_integral_states(self):
        """Get path integral states."""
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
