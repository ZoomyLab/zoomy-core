"""Module `zoomy_core.fvm.nonconservative_flux`."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from functools import partial

class NonconservativeFlux:
    """NonconservativeFlux. (class)."""
    def get_flux_operator(self, model):
        """Get flux operator."""
        pass


class Zero(NonconservativeFlux):
    """Zero. (class)."""
    def get_flux_operator(self, model):
        """Get flux operator."""
        def compute(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt):
            """Compute."""
            return np.zeros_like(Qi), np.zeros_like(Qi)

        return compute
    
class Rusanov(NonconservativeFlux):
    """Rusanov. (class)."""
    def __init__(self, integration_order=3, identity_matrix=None, eps=1e-10):
        """Initialize the instance."""
        self.integration_order = integration_order
        samples, weights = leggauss(integration_order)
        # shift from [-1, 1] to [0,1]
        samples = 0.5 * (samples + 1)
        weights *= 0.5
        self.wi = np.array(weights)
        self.xi = np.array(samples)
        self.Id = identity_matrix if identity_matrix else lambda n: np.eye(n)
        self.eps = eps
        
    def _get_A(self, model):
        """Internal helper `_get_A`."""
        def A(q, qaux, parameters, n):                      
            # q : (n_dof,)
            # evaluate the matrices A_d
            """A."""
            _A = model.quasilinear_matrix(q, qaux, parameters)
            return np.einsum('d...,ijd...->ij...', n, _A)
        return A
    
    def _integrate_path(self, model):
        """Internal helper `_integrate_path`."""
        compute_A = self._get_A(model)
        def compute(Qi, Qj,
                            Qauxi, Qauxj,
                            parameters,
                            normal):
            """Compute."""
            dQ     =  Qj    - Qi
            dQaux  =  Qauxj - Qauxi
            
            A_int = np.zeros((Qi.shape[0], Qi.shape[0], Qi.shape[1]))
            
            for xi, wi in zip(self.xi, self.wi):
                q_path     = Qi   + xi * dQ
                qaux_path  = Qauxi + xi * dQaux
                A = compute_A(q_path, qaux_path, parameters, normal)
                A_int += wi * A     
            return A_int
        return compute
    
    


    def get_flux_operator(self, model):
        """Get flux operator."""
        compute_path_integral = self._integrate_path(model)
        Id_single = self.Id(model.n_variables)
        
        def compute( 
            Qi,
            Qj,
            Qauxi,
            Qauxj,
            parameters,
            normal,
            Vi,
            Vj,
            Vij,
            dt
        ):
            """
            Vectorised Rusanov fluctuation.

            Shapes
            ------
            Qi, Qj            : (n_dof , N)        states for the two cells
            Qauxi, Qauxj      : (n_aux , N)
            parameters        : (n_param ,)
            normal            : (dim   , N)  or (dim,)   oriented outward for cell "i"
            Vi                : (N,)   or scalar         cell volume
            Vij               : (N,)   or scalar         face measure
            """
            A_int = compute_path_integral(Qi, Qj, Qauxi, Qauxj, parameters, normal)
            sM = np.maximum(np.abs(model.eigenvalues(Qi, Qauxi, parameters, normal)).max(axis=0), np.abs(model.eigenvalues(Qj, Qauxj, parameters, normal)).max(axis=0)) 
            Id = np.stack([Id_single]*Qi.shape[1], axis=2)
            
            dQ = Qj - Qi
            Dp = np.einsum("ij..., j...-> i...", 0.5 * (A_int + sM * Id), dQ)
            Dm = np.einsum("ij..., j...-> i...", 0.5 * (A_int - sM * Id), dQ)
            return Dp, Dm
            
        return compute
    
