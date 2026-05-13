"""Symbolic Riemann solvers: Rusanov, positive, nonconservative variants.

Every index-dependent class (``PositiveRusanov``, ``NonconservativeRusanov``,
``PositiveNonconservativeRusanov``, …) auto-locates the named fields
``h`` and ``b`` (and ``hinv`` when present) in ``model.variables`` /
``model.aux_variables`` via :class:`FieldHandle` so the same Riemann
code path works whether bathymetry is part of the conservative state
(legacy SWE) or lives in ``Qaux`` (chain-DAE convention).
"""

import numpy as np
import param
import sympy as sp

from zoomy_core.misc.misc import ZArray, Zstruct
from zoomy_core.model.basefunction import SymbolicRegistrar
from zoomy_core.model.basemodel import Model
from zoomy_core.model.kernel_functions import max_wavespeed
from zoomy_core.transformation.to_numpy import NumpyRuntimeSymbolic


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
    """Numerics. (class).

    Subclasses that depend on the location of named fields (``h``,
    ``b``, ``hinv``, …) call :meth:`find_field` at the top of their
    own ``__init__`` to obtain a :class:`FieldHandle` — the search
    walks state then aux automatically, so the same numerics works
    whether bathymetry is in ``Q`` (legacy SWE) or in ``Qaux``
    (chain-DAE)."""

    name = param.String(default="NumericsV2")
    model = param.ClassSelector(class_=Model, is_instance=True)

    scaled_q_indices = param.List(default=None, allow_None=True)

    def __init__(self, model, **params):
        """Initialize the instance."""
        super().__init__(model=model, **params)
        self.functions, self.call = Zstruct(), Zstruct()

        self.variables = ZArray(self.model.variables.get_list())
        self.aux_variables = ZArray(self.model.aux_variables.get_list())
        # Use symbols for symbolic derivation (not the numeric values in model.parameters)
        self.parameters = ZArray(self.model._parameter_symbols.get_list())
        self.normal = ZArray(self.model.normal.get_list())

        self.variables_minus = self._create_v("Q_minus", self.model.n_variables)
        self.variables_plus = self._create_v("Q_plus", self.model.n_variables)
        self.aux_variables_minus = self._create_v("Qaux_minus", self.model.n_aux_variables)
        self.aux_variables_plus = self._create_v("Qaux_plus", self.model.n_aux_variables)

        self.flux_minus = self._create_v("flux_minus", self.model.n_variables)
        self.flux_plus = self._create_v("flux_plus", self.model.n_variables)
        self.source_term = self._create_v("source_term", self.model.n_variables)

        # Auto-locate the standard named fields (``h``, ``b``, ``hinv``)
        # if present.  Subclasses can call ``find_field`` to register
        # additional handles.
        self._field_handles = {}
        for name in ("h", "b", "hinv"):
            h = self.find_field(name, required=False)
            if h is not None:
                self._field_handles[name] = h
        self._scaled_q_indices = self._resolve_scaled_q_indices(self.scaled_q_indices)

        self._initialize_functions()

    # ── Field-search API ──────────────────────────────────────────

    def find_field(self, name, *, required=True):
        """Return a :class:`FieldHandle` for the named field.

        Searches ``self.model.variables`` (Q) first, then
        ``self.model.aux_variables`` (Qaux).  Caches the result in
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
        state_list = self.model.variables.get_list()
        state_names = [str(s) for s in state_list]
        if name in state_names:
            i = state_names.index(name)
            h = FieldHandle(name, "q", i,
                            minus=self.variables_minus[i],
                            plus=self.variables_plus[i],
                            state=self.variables[i])
            cache[name] = h
            return h
        aux_list = self.model.aux_variables.get_list()
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
                f"Field {name!r} not found in model.variables "
                f"({state_names}) nor model.aux_variables ({aux_names})"
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
                if i < 0 or i >= self.model.n_variables:
                    raise IndexError(
                        f"scaled_q_indices contains out-of-bounds index {i}."
                    )
            return cleaned

        excluded = set()
        for name in ("h", "b"):
            h = self._field_handles.get(name)
            if h is not None and h.container == "q":
                excluded.add(h.index)
        return [i for i in range(self.model.n_variables) if i not in excluded]

    def _eps_symbol(self):
        """Internal helper `_eps_symbol`."""
        # Use the symbolic eps from _parameter_symbols (not the numeric value)
        if hasattr(self.model._parameter_symbols, "contains") and self.model._parameter_symbols.contains("eps"):
            return self.model._parameter_symbols.eps
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

    def _model_eval(self, function_name, Q, Qaux, p, n=None):
        """Internal helper `_model_eval`."""
        definition = self.model.functions[function_name].definition
        sub_map = {}
        for sym, val in zip(self.model.variables.get_list(), list(Q)):
            sub_map[sym] = val
        for sym, val in zip(self.model.aux_variables.get_list(), list(Qaux)):
            sub_map[sym] = val
        for sym, val in zip(self.model._parameter_symbols.get_list(), list(p)):
            sub_map[sym] = val
        if n is not None:
            for sym, val in zip(self.model.normal.get_list(), list(n)):
                sub_map[sym] = val

        if hasattr(definition, "subs"):
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
        raise ValueError(f"Unsupported runtime backend '{backend}'.")

    def to_runtime_numpy(self):
        """To runtime numpy."""
        return self.to_runtime("numpy")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray(zeros)

    def numerical_fluctuations(self):
        """Numerical fluctuations."""
        zeros = ZArray.zeros(self.model.n_variables)
        return ZArray([zeros, zeros])


class Rusanov(Numerics):
    """Rusanov. (class)."""
    name = param.String(default="RusanovV2")

    def get_viscosity_identity_flux(self):
        """Get viscosity identity flux."""
        n = self.model.n_variables
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
        n = self.model.n_variables
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


class PositiveRusanov(Rusanov):
    """PositiveRusanov. (class).

    Hydrostatic reconstruction follows Audusse-Bristeau-Klein:
    ``h_L* = max(0, h_L + b_L − b*)``, ``b* = max(b_L, b_R)``.
    The ``h``, ``b`` and (optional) ``hinv`` fields are resolved via
    :meth:`Numerics.find_field` so the same logic works whether
    bathymetry is part of conservative state or lives in ``Qaux``.
    """

    name = param.String(default="PositiveRusanovV2")

    def __init__(self, model, **params):
        super().__init__(model=model, **params)
        # Cache the field handles for tight access in the reconstruction.
        self.h_field = self.find_field("h")
        self.b_field = self.find_field("b")
        self.hinv_field = self.find_field("hinv", required=False)

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

        eps = self._eps_symbol()
        hL_eff = sp.Max(hL, eps)
        hR_eff = sp.Max(hR, eps)

        self.b_field.assign(qLs, qauxL, b_star)
        self.b_field.assign(qRs, qauxR, b_star)
        self.h_field.assign(qLs, qauxL, hL_star)
        self.h_field.assign(qRs, qauxR, hR_star)

        for idx in self._scaled_q_indices:
            qLs[idx] = qL[idx] * hL_star / hL_eff
            qRs[idx] = qR[idx] * hR_star / hR_eff

        if self.hinv_field is not None:
            self.hinv_field.assign(qLs, qauxL, 1 / (hL_star + eps))
            self.hinv_field.assign(qRs, qauxR, 1 / (hR_star + eps))

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
        n = self.model.n_variables
        return ZArray.zeros(n, n)

    def get_viscosity_identity_fluctuations(self):
        """Get viscosity identity fluctuations."""
        n = self.model.n_variables
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
        n_vars = self.model.n_variables
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


class QuasilinearRusanov(NonconservativeRusanov):
    """QuasilinearRusanov. (class)."""
    name = param.String(default="QuasilinearRusanovV2")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray(zeros)

    def _call_model_matrix(self):
        """Internal helper `_call_model_matrix`."""
        return lambda Q, Qaux, p: self._model_eval("quasilinear_matrix", Q, Qaux, p)


class PositiveQuasilinearRusanov(PositiveRusanov, QuasilinearRusanov):
    """PositiveQuasilinearRusanov. (class)."""
    name = param.String(default="PositiveQuasilinearRusanovV2")

    def numerical_flux(self):
        """Numerical flux."""
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray(zeros)

    def get_path_integral_states(self):
        """Get path integral states."""
        return self.hydrostatic_reconstruction(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
        )
