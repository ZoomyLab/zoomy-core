import sympy as sp
import param
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basemodel import Model


class Numerics(param.Parameterized):
    """Base class for Symbolic Numerical Schemes."""

    name = param.String(default="Numerics")
    model = param.ClassSelector(class_=Model, is_instance=True)

    def __init__(self, model, **params):
        super().__init__(model=model, **params)
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

    def local_max_abs_eigenvalue(self, q, aux, n):
        """
        Returns symbolic scalar max eigenvalue using the model's eigenvalues function.
        """
        evs = self.model._eigenvalues.eval_symbolic(q, aux, self.model.parameters, n)
        return sp.Max(*[sp.Abs(e) for e in evs])


class Rusanov(Numerics):
    name = param.String(default="Rusanov")

    def numerical_flux(self):
        # 1. Get Symbols
        qL, qR = self.variables_minus, self.variables_plus
        auxL, auxR = self.aux_variables_minus, self.aux_variables_plus
        n = self.model.normal.values()

        # 2. Evaluate Fluxes using eval_symbolic
        #    model.flux must be a Function returning a ZArray of shape (n_vars, dimension)

        # FL and FR are ZArrays
        FL_tensor = self.model._flux.eval_symbolic(qL, auxL)
        FR_tensor = self.model._flux.eval_symbolic(qR, auxR)

        # 3. Project Fluxes: F . n
        #    Loop over variables and dimensions
        FL_n = []
        FR_n = []

        # Convert to nested lists for easy indexing, or use ZArray indexing if robust
        # ZArray indexing [row, col] works if it's 2D
        for var_idx in range(self.model.n_variables):
            val_L = 0
            val_R = 0
            for dim_idx in range(self.model.dimension):
                # tensor[var, dim] * n[dim]
                val_L += FL_tensor[var_idx, dim_idx] * n[dim_idx]
                val_R += FR_tensor[var_idx, dim_idx] * n[dim_idx]
            FL_n.append(val_L)
            FR_n.append(val_R)

        # 4. Compute Max Wave Speed
        lamL = self.local_max_abs_eigenvalue(qL, auxL, n)
        lamR = self.local_max_abs_eigenvalue(qR, auxR, n)
        s_max = sp.Max(lamL, lamR)

        # 5. Assemble Rusanov Flux
        flux = []
        for i in range(self.model.n_variables):
            val = 0.5 * (FL_n[i] + FR_n[i]) - 0.5 * s_max * (qR[i] - qL[i])
            flux.append(val)

        return ZArray(flux)
