import sympy as sp
import param
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basemodel import Model
from zoomy_core.transformation.functions import conditional


class Numerics(param.Parameterized):
    """Base class for Symbolic Numerical Schemes."""

    name = param.String(default="Numerics")
    model = param.ClassSelector(class_=Model, is_instance=True)

    def __init__(self, model, **params):
        super().__init__(model=model, **params)
        self.variables = model.variables
        self.aux_variables = model.aux_variables
        self.normal = model.normal
        parameters = model.parameters
        self.variables_minus = self._create_symbolic_vector(
            "Q_minus", model.variables
        )
        self.variables_plus = self._create_symbolic_vector("Q_plus", model.variables)
        self.aux_variables_minus = self._create_symbolic_vector(
            "Qaux_minus", model.aux_variables
        )
        self.aux_variables_plus = self._create_symbolic_vector(
            "Qaux_plus", model.aux_variables
        )

    def _create_symbolic_vector(self, name, variables):
        """
        Creates a list of symbols named {name}_{i}, inheriting assumptions
        (real, positive, nonnegative) from the input 'variables' list.
        """
        symbols = []
        for i, var in enumerate(variables):
            # Start with the base assumption that physics variables are real
            assumptions = {'real': True}
            
            # Inherit specific positivity assumptions to prevent "I" (imaginary) in sqrt
            if var.is_positive:
                assumptions['positive'] = True
            elif var.is_nonnegative:
                assumptions['nonnegative'] = True
            
            # Create the new symbol inheriting these properties
            symbols.append(sp.Symbol(f"{name}_{i}", **assumptions))
                
        return symbols
    
    
    def numerical_flux(self):
        raise NotImplementedError

    def local_max_abs_eigenvalue(self):
        """
        Returns symbolic scalar max eigenvalue using the model's eigenvalues function.
        """
        Q = self.variables
        Qaux = self.aux_variables
        n = self.normal
        parameters = self.parameters
        evs = self.model._eigenvalues.eval_symbolic(Q, Qaux, parameters, n)
        return sp.Max(*[sp.Abs(e) for e in evs])
    
    def update_q(self):
        Q = self.variables
        return Q
    
    def update_qaux(self):
        Qaux = self.aux_variables
        return Qaux


class Rusanov(Numerics):
    name = param.String(default="Rusanov")

    def _compute_flux(self, qL, qR, auxL, auxR, n):
        """
        Core Rusanov logic: F = 0.5(FL + FR) - 0.5*s_max*(QR - QL)
        Can be called with original states OR reconstructed states.
        """
        # 1. Evaluate Fluxes (F(Q))
        FL_tensor = self.model._flux.eval_symbolic(qL, auxL)
        FR_tensor = self.model._flux.eval_symbolic(qR, auxR)

        # 2. Project Fluxes: F . n
        FL_n = []
        FR_n = []
        
        # Helper to sum over dimensions: F_i = sum_d(Flux_id * n_d)
        for var_idx in range(self.model.n_variables):
            val_L = 0
            val_R = 0
            for dim_idx in range(self.model.dimension):
                val_L += FL_tensor[var_idx, dim_idx] * n[dim_idx]
                val_R += FR_tensor[var_idx, dim_idx] * n[dim_idx]
            FL_n.append(val_L)
            FR_n.append(val_R)

        # 3. Compute Max Wave Speed (Reuse Helper)
        lamL = self.local_max_abs_eigenvalue(qL, auxL, n)
        lamR = self.local_max_abs_eigenvalue(qR, auxR, n)
        s_max = sp.Max(lamL, lamR)

        # 4. Assemble Rusanov Flux
        flux = []
        for i in range(self.model.n_variables):
            val = 0.5 * (FL_n[i] + FR_n[i]) - 0.5 * s_max * (qR[i] - qL[i])
            flux.append(val)

        return ZArray(flux)

    def numerical_flux(self):
        # Default behavior: use the raw +/- symbols
        qL, qR = self.variables_minus, self.variables_plus
        auxL, auxR = self.aux_variables_minus, self.aux_variables_plus
        n = self.normal.values()
        
        return self._compute_flux(qL, qR, auxL, auxR, n)
    
    def update_qaux(self):
        res = self.aux_variables
        Q = self.variables
        eps = self.parameters.epsilon
        res[0] = conditional(Q[1] < eps, 0.0, 1./Q[1])
        return res
        

class PositiveRusanov(Rusanov):
    name = param.String(default="PositiveRusanov")
    
    # Epsilon for regularization (matching Firedrake's 1e-4 or similar)

    def hydrostatic_reconstruction(self, qL, qR):
        """
        Symbolic Hydrostatic Reconstruction with regularization.
        Q = [b, h, hu, hv, ...]
        """
        # Indices: Q[0]=b (bathymetry), Q[1]=h (depth), Q[2..]=momentum
        bL, hL = qL[0], qL[1]
        bR, hR = qR[0], qR[1]

        # 1. Free surface reconstruction
        etaL = hL + bL
        etaR = hR + bR
        b_star = sp.Max(bL, bR)

        # 2. Reconstructed depths (positivity preserving)
        hL_star = sp.Max(0., etaL - b_star)
        hR_star = sp.Max(0., etaR - b_star)

        # 3. Build state vectors (convert to list for mutability)
        qL_star = list(qL)
        qR_star = list(qR)

        # Update Depth
        qL_star[1] = hL_star
        qR_star[1] = hR_star
        
        # 4. Scale Momentum: (hu)* = h* * u = h* * (hu / h_eff)
        # Use a symbolic epsilon to prevent division by zero, matching Firedrake logic.
        eps = self.parameters.epsilon
        
        # Regularize the denominator: h_eff = max(h, eps)
        hL_eff = sp.Max(hL, eps)
        hR_eff = sp.Max(hR, eps)

        for i in range(2, len(qL)):
            # Calculate velocity u using regularized depth, then scale by reconstructed depth
            # uL = qL[i] / hL_eff
            # qL_star[i] = hL_star * uL
            qL_star[i] = (qL[i] / hL_eff) * hL_star
            qR_star[i] = (qR[i] / hR_eff) * hR_star

        return ZArray(qL_star), ZArray(qR_star)

    def numerical_flux(self):
        # 1. Get Symbols
        qL_raw, qR_raw = self.variables_minus, self.variables_plus
        auxL, auxR = self.aux_variables_minus, self.aux_variables_plus
        n = self.normal.values()

        # 2. Reconstruct States (Now safe for h=0)
        qL_star, qR_star = self.hydrostatic_reconstruction(qL_raw, qR_raw)

        # 3. Reuse Rusanov Logic with Reconstructed States
        return self._compute_flux(qL_star, qR_star, auxL, auxR, n)
    