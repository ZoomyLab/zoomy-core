import sympy as sp
import param
from zoomy_core.model.basemodel import Model
from zoomy_core.model.basefunction import substitute_expression


class Numerics(param.Parameterized):
    """
    Base class for Symbolic Numerical Schemes.
    Defines the interface for computing numerical fluxes and time steps symbolically.
    """

    name = param.String(default="Numerics")
    model = param.ClassSelector(class_=Model, is_instance=True)

    def __init__(self, model, **params):
        super().__init__(model=model, **params)

        # Create symbolic vectors for Left (-) and Right (+) states
        # These correspond to the "Q_minus" and "Q_plus" arguments in the C++ kernels
        self.variables_minus = self._create_symbolic_vector(
            "Q_minus", model.n_variables
        )
        self.variables_plus = self._create_symbolic_vector("Q_plus", model.n_variables)

        self.aux_variables_minus = self._create_symbolic_vector(
            "Qaux_minus", model.n_aux_variables
        )
        self.aux_variables_plus = self._create_symbolic_vector(
            "Qaux_plus", model.n_aux_variables
        )

    def _create_symbolic_vector(self, name, size):
        if size == 0:
            return []
        return [sp.Symbol(f"{name}_{i}") for i in range(size)]

    def numerical_flux(self):
        """
        Returns the symbolic expression for the numerical flux F(qL, qR, n).
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def local_max_abs_eigenvalue(self, q, aux, n):
        """
        Helper: Returns symbolic scalar max eigenvalue at a specific state (q, aux).
        Fetches the eigenvalues definition from the physics model and substitutes the state variables.
        """
        # 1. Get Eigenvalues Expression from Model (defined in terms of model.variables)
        #    We assume model.eigenvalues is a Function/Callable returning the symbolic expr.
        ev_expr = self.model.eigenvalues()

        # 2. Build Substitution Map
        #    Map Model Symbols -> Local Numerics Symbols (q, aux, n)
        subs_map = {}

        # Variables
        for m_var, l_var in zip(self.model.variables.values(), q):
            subs_map[m_var] = l_var

        # Aux Variables
        for m_aux, l_aux in zip(self.model.aux_variables.values(), aux):
            subs_map[m_aux] = l_aux

        # Normal Vector (Eigenvalues depend on direction n)
        for m_n, l_n in zip(self.model.normal.values(), n):
            subs_map[m_n] = l_n

        # 3. Substitute
        evs = substitute_expression(ev_expr, subs_map)

        # 4. Compute Max(|lambda|)
        #    If evs is iterable (list/ZArray), take max of elements.
        if hasattr(evs, "__iter__"):
            return sp.Max(*[sp.Abs(e) for e in evs])
        return sp.Abs(evs)


class Rusanov(Numerics):
    """
    Concrete implementation of the Rusanov (Lax-Friedrichs) Flux.

    Formula:
      F_num = 0.5 * (F_L + F_R) - 0.5 * s_max * (Q_R - Q_L)

    Where:
      F_L   = Flux(Q_L) . n
      F_R   = Flux(Q_R) . n
      s_max = max( |lambda(Q_L)|, |lambda(Q_R)| )
    """

    name = param.String(default="Rusanov")

    def numerical_flux(self):
        # 1. Alias the states
        qL, qR = self.variables_minus, self.variables_plus
        auxL, auxR = self.aux_variables_minus, self.aux_variables_plus
        n = self.model.normal.values()  # List of normal vector symbols [n_x, n_y, ...]

        # 2. Helper to get Physical Flux projected onto Normal 'n'
        def get_flux_projected(q, aux):
            # A. Get symbolic expression from model (in terms of model.variables)
            flux_expr = self.model.flux()

            # B. Build Substitution Map
            subs_map = {}
            for m_var, l_var in zip(self.model.variables.values(), q):
                subs_map[m_var] = l_var
            for m_aux, l_aux in zip(self.model.aux_variables.values(), aux):
                subs_map[m_aux] = l_aux

            # C. Substitute to get Flux(q)
            #    This returns [F_x(q), F_y(q), ...] where F_x is a vector of eqns
            flux_val = substitute_expression(flux_expr, subs_map)

            # D. Project onto Normal: sum(F_dim * n_dim)
            #    flux_val is usually [F_x, F_y] (list of length dimension)
            #    We need result[var] = sum_dim (flux_val[dim][var] * n[dim])

            projected_flux = []
            for var_idx in range(self.model.n_variables):
                val = 0
                for dim_idx in range(self.model.dimension):
                    # Robust access: handle lists, ZArrays, or Matrices
                    f_dim = flux_val[dim_idx]

                    if hasattr(f_dim, "__getitem__"):
                        val += f_dim[var_idx] * n[dim_idx]
                    else:
                        # Fallback for scalar physics (unlikely for systems)
                        val += f_dim * n[dim_idx]

                projected_flux.append(val)
            return projected_flux

        # 3. Compute Projected Fluxes
        FL_n = get_flux_projected(qL, auxL)
        FR_n = get_flux_projected(qR, auxR)

        # 4. Compute Max Wave Speed (Rusanov parameter)
        #    We pass 'n' explicitly because eigenvalues depend on the normal direction
        lamL = self.local_max_abs_eigenvalue(qL, auxL, n)
        lamR = self.local_max_abs_eigenvalue(qR, auxR, n)
        s_max = sp.Max(lamL, lamR)

        # 5. Assemble Rusanov Flux
        #    Flux = 0.5 * (FL + FR) - 0.5 * s_max * (QR - QL)
        flux = []
        for i in range(self.model.n_variables):
            val = 0.5 * (FL_n[i] + FR_n[i]) - 0.5 * s_max * (qR[i] - qL[i])
            flux.append(val)

        return flux
