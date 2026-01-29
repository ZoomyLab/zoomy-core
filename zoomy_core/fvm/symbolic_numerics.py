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
        # Signature is shared for both flux and fluctuations
        sig = Zstruct(
            q_minus=self.variables_minus,
            q_plus=self.variables_plus,
            aux_minus=self.aux_variables_minus,
            aux_plus=self.aux_variables_plus,
            p=self.parameters,
            normal=self.normal,
        )

        # 1. Register Conservative Flux (Size: n_dof)
        self.register_symbolic_function("numerical_flux", self.numerical_flux, sig)

        # 2. Register Non-Conservative Fluctuations (Size: 2 * n_dof)
        self.register_symbolic_function(
            "nonconservative_fluctuations", self.nonconservative_fluctuations, sig
        )

        eig_sig = Zstruct(
            Q=self.variables, Qaux=self.aux_variables, p=self.parameters, n=self.normal
        )
        self.register_symbolic_function(
            "local_max_abs_eigenvalue", self.max_eigenvalue_definition, eig_sig
        )

    def max_eigenvalue_definition(self):
        evs = self.model.call.eigenvalues(
            self.variables, self.aux_variables, self.parameters, self.normal
        )
        return sp.Max(*[sp.Abs(e) for e in evs])

    def local_max_abs_eigenvalue(self, Q=None, Qaux=None, p=None, n=None):
        if Q is None:
            return self.call.local_max_abs_eigenvalue()
        evs = self.model.call.eigenvalues(Q, Qaux, p, n)
        return sp.Max(*[sp.Abs(e) for e in evs])

    # --- Base Implementations (Return Zeros) ---
    def numerical_flux(self):
        # Returns [0, 0, ..., 0] (Size: n_vars)
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray(zeros)

    def nonconservative_fluctuations(self):
        # Returns [[0...], [0...]] (Size: 2, n_vars)
        zeros = [sp.Integer(0)] * self.model.n_variables
        return ZArray([ZArray(zeros), ZArray(zeros)])


class Rusanov(Numerics):
    name = param.String(default="Rusanov")

    def numerical_flux(self):
        # Override Conservative Flux
        return self._compute_flux(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )

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
        # Reconstruction applied to Flux
        qLs, qRs = self.hydrostatic_reconstruction(
            self.variables_minus, self.variables_plus
        )
        return self._compute_flux(
            qLs,
            qRs,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )


class QuasilinearRusanov(Rusanov):
    name = param.String(default="QuasilinearRusanov")
    integration_order = param.Integer(default=3)

    # Inherits numerical_flux (Conservative Rusanov) from Parent
    # Overrides nonconservative_fluctuations

    def nonconservative_fluctuations(self):
        return self._compute_fluctuations(
            self.variables_minus,
            self.variables_plus,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )

    def _compute_fluctuations(self, qL, qR, auxL, auxR, p, n):
        # 1. Setup Integration Rule
        xi_np, wi_np = np.polynomial.legendre.leggauss(self.integration_order)
        xi_np = 0.5 * (xi_np + 1)
        wi_np = 0.5 * wi_np

        dQ = qR - qL
        dAux = auxR - auxL
        n_vars = self.model.n_variables
        dim = len(n)

        # 2. Path Integral
        A_int = sp.Matrix.zeros(n_vars, n_vars)

        for xi, wi in zip(xi_np, wi_np):
            q_path = qL + xi * dQ
            aux_path = auxL + xi * dAux
            # Note: Using nonconservative_matrix from model
            A_tensor = self.model.call.nonconservative_matrix(q_path, aux_path, p)

            A_n = sp.Matrix.zeros(n_vars, n_vars)
            for i in range(n_vars):
                for j in range(n_vars):
                    val = 0
                    for d in range(dim):
                        val += A_tensor[i][j][d] * n[d]
                    A_n[i, j] = val
            A_int += wi * A_n

        # 3. Dissipation
        s_max = sp.Max(
            self.local_max_abs_eigenvalue(qL, auxL, p, n),
            self.local_max_abs_eigenvalue(qR, auxR, p, n),
        )

        dQ_vec = sp.Matrix(dQ)
        term_advection = A_int * dQ_vec
        term_dissipation = s_max * dQ_vec

        Dp_matrix = 0.5 * (term_advection + term_dissipation)
        Dm_matrix = 0.5 * (term_advection - term_dissipation)

        return ZArray([ZArray(Dp_matrix[:]), ZArray(Dm_matrix[:])])


class PositiveQuasilinearRusanov(PositiveRusanov, QuasilinearRusanov):
    name = param.String(default="PositiveQuasilinearRusanov")

    def nonconservative_fluctuations(self):
        # Reconstruction applied to Flux
        qLs, qRs = self.hydrostatic_reconstruction(
            self.variables_minus, self.variables_plus
        )   
        return self._compute_fluctuations(
            qLs,
            qRs,
            self.aux_variables_minus,
            self.aux_variables_plus,
            self.parameters,
            self.normal,
        )
