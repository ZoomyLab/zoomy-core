import sympy as sp
import param
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basemodel import Model
from zoomy_core.model.basefunction import Function, SymbolicRegistrar


class Numerics(param.Parameterized, SymbolicRegistrar):
    name = param.String(default="Numerics")
    model = param.ClassSelector(class_=Model, is_instance=True)

    def __init__(self, model, **params):
        super().__init__(model=model, **params)
        self.functions, self.call = Zstruct(), Zstruct()

        self.variables_minus = self._create_v("Q_minus")
        self.variables_plus = self._create_v("Q_plus")
        self.aux_variables_minus = self._create_v("Qaux_minus")
        self.aux_variables_plus = self._create_v("Qaux_plus")

        self.dt, self.dx = sp.symbols("dt dx", real=True)
        self.flux_minus = self._create_v("flux_minus")
        self.flux_plus = self._create_v("flux_plus")
        self.source_term = self._create_v("source_term")

        self._initialize_functions()

    def _create_v(self, name):
        v = ZArray(
            [sp.Symbol(f"{name}_{i}", real=True) for i in range(self.model.n_variables)]
        )
        v._symbolic_name = name
        return v

    def _initialize_functions(self):
        flux_sig = Zstruct(
            q_minus=self.variables_minus,
            q_plus=self.variables_plus,
            aux_minus=self.aux_variables_minus,
            aux_plus=self.aux_variables_plus,
            p=self.model.parameters,
            normal=self.model.normal,
        )
        self.register_symbolic_function("numerical_flux", self.numerical_flux, flux_sig)

        eig_sig = Zstruct(
            Q=self.model.variables,
            Qaux=self.model.aux_variables,
            p=self.model.parameters,
            n=self.model.normal,
        )
        self.register_symbolic_function(
            "local_max_abs_eigenvalue", self.max_eigenvalue_definition, eig_sig
        )

    def max_eigenvalue_definition(self):
        evs = self.model.call.eigenvalues(
            self.model.variables,
            self.model.aux_variables,
            self.model.parameters,
            self.model.normal,
        )
        return sp.Max(*[sp.Abs(e) for e in evs])

    def local_max_abs_eigenvalue(self, Q=None, Qaux=None, p=None, n=None):
        if Q is None:
            return self.call.local_max_abs_eigenvalue()
        evs = self.model.call.eigenvalues(Q, Qaux, p, n)
        return sp.Max(*[sp.Abs(e) for e in evs])

    def numerical_flux(self):
        return self._compute_flux(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.model.parameters,
            self.model.normal,
        )

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        raise NotImplementedError


class Rusanov(Numerics):
    name = param.String(default="Rusanov")

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        FL = self.model.call.flux(qL, auxL, p)
        FR = self.model.call.flux(qR, auxR, p)

        s_max = sp.Max(
            self.local_max_abs_eigenvalue(qL, auxL, p, n),
            self.local_max_abs_eigenvalue(qR, auxR, p, n),
        )
        return 0.5 * (FL @ n + FR @ n) - 0.5 * s_max * (qR - qL)


class PositiveRusanov(Rusanov):
    name = param.String(default="PositiveRusanov")

    def hydrostatic_reconstruction(self, qL, qR):
        b_star = sp.Max(qL[0], qR[0])
        hL_star, hR_star = (
            sp.Max(0.0, qL[1] + qL[0] - b_star),
            sp.Max(0.0, qR[1] + qR[0] - b_star),
        )
        hL_eff, hR_eff = sp.Max(qL[1], 1e-6), sp.Max(qR[1], 1e-6)
        return ZArray([b_star, hL_star, *(qL[2:] / hL_eff) * hL_star]), ZArray(
            [b_star, hR_star, *(qR[2:] / hR_eff) * hR_star]
        )

    def numerical_flux(self):
        qLs, qRs = self.hydrostatic_reconstruction(
            self.variables_minus, self.variables_plus
        )
        return self._compute_flux(
            qLs,
            qRs,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.model.parameters,
            self.model.normal,
        )
