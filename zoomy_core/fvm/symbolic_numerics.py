import sympy as sp
import numpy as np
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

        self.variables = ZArray(self.model.variables.get_list())
        self.aux_variables = ZArray(self.model.aux_variables.get_list())
        self.parameters = ZArray(self.model.parameters.get_list())
        self.normal = ZArray(self.model.normal.get_list())

        self.variables_minus = self._create_v("Q_minus", self.model.n_variables)
        self.variables_plus = self._create_v("Q_plus", self.model.n_variables)
        self.aux_variables_minus = self._create_v("Qaux_minus", self.model.n_aux_variables)
        self.aux_variables_plus = self._create_v("Qaux_plus", self.model.n_aux_variables)

        self.flux_minus = self._create_v("flux_minus", self.model.n_variables)
        self.flux_plus = self._create_v("flux_plus", self.model.n_variables)
        self.source_term = self._create_v("source_term", self.model.n_variables)

        self._initialize_functions()

    def _create_v(self, name, size):
        v = ZArray(
            [sp.Symbol(f"{name}_{i}", real=True) for i in range(size)]
        )
        v._symbolic_name = name
        return v

    def _initialize_functions(self):
        sig = Zstruct(
            q_minus=self.variables_minus,
            q_plus=self.variables_plus,
            aux_minus=self.aux_variables_minus,
            aux_plus=self.aux_variables_plus,
            p=self.parameters,
            normal=self.normal,
        )

        self.register_symbolic_function("numerical_flux", self.numerical_flux, sig)
        self.register_symbolic_function("numerical_fluctuations", self.numerical_fluctuations, sig)

        eig_sig = Zstruct(
            Q=self.variables, Qaux=self.aux_variables, p=self.parameters, n=self.normal
        )
        self.register_symbolic_function(
            "local_max_abs_eigenvalue", self.local_max_eigenvalue_definition, eig_sig
        )

    def local_max_eigenvalue_definition(self):
        evs = self.model.call.eigenvalues(
            self.variables, self.aux_variables, self.parameters, self.normal
        )
        return sp.Max(*[sp.Abs(e) for e in evs])

    def local_max_abs_eigenvalue(self, Q=None, Qaux=None, p=None, n=None):
        if Q is None:
            return self.call.local_max_abs_eigenvalue()
        evs = self.model.call.eigenvalues(Q, Qaux, p, n)
        return sp.Max(*[sp.Abs(e) for e in evs])

    def numerical_flux(self):
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray(zeros)

    def numerical_fluctuations(self):
        # Base class now returns a flat 1D array of size 2*N
        zeros = [sp.Integer(0)] * (2 * self.model.n_variables)
        return ZArray(zeros)


class Rusanov(Numerics):
    name = param.String(default="Rusanov")
    
    def get_viscosity_identity_flux(self):
        Id = sp.Matrix(sp.Identity(self.model.n_variables))
        Id = 0 * Id
        return ZArray(Id)
    
    def get_viscosity_identity_fluctuations(self):
        Id = sp.Matrix(sp.Identity(self.model.n_variables))
        Id[0,0] = 0
        return ZArray(Id)

    def numerical_flux(self):
        return self._compute_flux(
            self.variables_minus, self.variables_plus,
            self.aux_variables_minus, self.aux_variables_plus,
            self.parameters, self.normal,
        )

    def _compute_flux(self, qL, qR, auxL, auxR, p, n):
        FL = self.model.call.flux(qL, auxL, p)
        FR = self.model.call.flux(qR, auxR, p)
        PL = self.model.call.hydrostatic_pressure(qL, auxL, p)
        PR = self.model.call.hydrostatic_pressure(qR, auxR, p)
        s_max = sp.Max(
            self.local_max_abs_eigenvalue(qL, auxL, p, n),
            self.local_max_abs_eigenvalue(qR, auxR, p, n),
        )
        Id = self.get_viscosity_identity_flux()
        return 0.5 * ((FL+PL) @ n + (FR+PR) @ n) - 0.5 * s_max * Id @ (qR - qL)


class PositiveRusanov(Rusanov):
    name = param.String(default="PositiveRusanov")

    def hydrostatic_reconstruction(self, qL, qR):
        b_star = sp.Max(qL[0], qR[0])
        hL_star, hR_star = (
            sp.Max(0.0, qL[1] + qL[0] - b_star),
            sp.Max(0.0, qR[1] + qR[0] - b_star),
        )
        eps = self.model.parameters.eps
        hL_eff, hR_eff = sp.Max(qL[1], eps), sp.Max(qR[1], eps)
        
        hL_inv = 1/(hL_star + eps)
        hR_inv = 1/(hR_star + eps)
        return ZArray([b_star, hL_star, *(qL[2:] / hL_eff) * hL_star]), ZArray(
            [b_star, hR_star, *(qR[2:] / hR_eff) * hR_star]), hL_inv, hR_inv

    def numerical_flux(self):
        qLs, qRs, hinvL, hinvR = self.hydrostatic_reconstruction(
            self.variables_minus, self.variables_plus
        )
        qauxL = ZArray(self.aux_variables_minus)
        qauxR = ZArray(self.aux_variables_plus)
        qauxL[0] = hinvL
        qauxR[0] = hinvR
        
        return self._compute_flux(
            qLs, qRs, qauxL, qauxR, self.parameters, self.normal
        )

    def numerical_fluctuations(self):
        qLs, qRs, hinvL, hinvR = self.hydrostatic_reconstruction(
            self.variables_minus, self.variables_plus
        )
        qauxL = ZArray(self.aux_variables_minus)
        qauxR = ZArray(self.aux_variables_plus)
        qauxL[0] = hinvL
        qauxR[0] = hinvR
        
        P_mat_L_raw = self.model.call.hydrostatic_pressure(self.variables_minus, self.aux_variables_minus, self.parameters)
        P_mat_R_raw = self.model.call.hydrostatic_pressure(self.variables_plus, self.aux_variables_plus, self.parameters)
        P_mat_L_star = self.model.call.hydrostatic_pressure(qLs, qauxL, self.parameters)
        P_mat_R_star = self.model.call.hydrostatic_pressure(qRs, qauxR, self.parameters)

        Dm_jump = (P_mat_L_raw - P_mat_L_star) @ self.normal
        Dp_jump = (P_mat_R_star - P_mat_R_raw) @ self.normal

        # Extract base fluctuations (which are now guaranteed to be 1D)
        base_fluct = super().numerical_fluctuations()
        n_vars = self.model.n_variables
        
        Dp_base = ZArray(base_fluct[:n_vars])
        Dm_base = ZArray(base_fluct[n_vars:])

        Dp = Dp_base + Dp_jump
        Dm = Dm_base + Dm_jump

        # Collapse the 2xN array into a flat 1D ZArray
        return ZArray([Dp, Dm]).flatten()


class NonconservativeRusanov(Rusanov):
    name = param.String(default="NonconservativeRusanov")
    integration_order = param.Integer(default=3)

    def get_path_integral_states(self):
        return self.variables_minus, self.variables_plus, self.aux_variables_minus, self.aux_variables_plus

    def numerical_fluctuations(self):
        qLs, qRs, qauxL, qauxR = self.get_path_integral_states()
        
        # nc_fluct is a flattened 1D array
        nc_fluct = self._compute_fluctuations(qLs, qRs, qauxL, qauxR, self.parameters, self.normal)
        
        # out is also a flattened 1D array
        out = super().numerical_fluctuations()
        
        # Both are 1D arrays of size 2*N, so we can directly add them!
        return out + nc_fluct
        
    def _call_model_matrix(self):
        return lambda Q, Qaux, p: self.model.call.nonconservative_matrix(Q, Qaux, p)

    def _compute_fluctuations(self, qL, qR, auxL, auxR, p, n):
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

        # Collapse the 2xN array into a flat 1D ZArray
        return ZArray([Dp_matrix, Dm_matrix]).flatten()


class PositiveNonconservativeRusanov(PositiveRusanov, NonconservativeRusanov):
    name = param.String(default="PositiveNonconservativeRusanov")

    def get_path_integral_states(self):
        qLs, qRs, hinvL, hinvR = self.hydrostatic_reconstruction(
            self.variables_minus, self.variables_plus
        )
        qauxL = ZArray(self.aux_variables_minus)
        qauxR = ZArray(self.aux_variables_plus)
        qauxL[0] = hinvL
        qauxR[0] = hinvR
        return qLs, qRs, qauxL, qauxR


class QuasilinearRusanov(NonconservativeRusanov):
    name = param.String(default="QuasilinearRusanov")
    
    def numerical_flux(self):
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray(zeros)

    def _call_model_matrix(self):
        return lambda Q, Qaux, p: self.model.call.quasilinear_matrix(Q, Qaux, p)


class PositiveQuasilinearRusanov(PositiveRusanov, QuasilinearRusanov):
    name = param.String(default="PositiveQuasilinearRusanov")

    def numerical_flux(self):
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray(zeros)

    def get_path_integral_states(self):
        qLs, qRs, hinvL, hinvR = self.hydrostatic_reconstruction(
            self.variables_minus, self.variables_plus
        )
        qauxL = ZArray(self.aux_variables_minus)
        qauxR = ZArray(self.aux_variables_plus)
        qauxL[0] = hinvL
        qauxR[0] = hinvR
        return qLs, qRs, qauxL, qauxR