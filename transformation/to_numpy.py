from typing import Callable
import numpy as np
from attrs import define, field

from zoomy_core.misc.custom_types import FArray


@define(kw_only=False, slots=True, frozen=True)
class NumpyRuntimeModel:
    """Runtime model generated from a symbolic Model."""

    # --- User-provided ---
    model: object = field()

    # --- Automatically derived ---
    name: str = field(init=False)
    dimension: int = field(init=False)
    n_variables: int = field(init=False)
    n_aux_variables: int = field(init=False)
    n_parameters: int = field(init=False)
    parameters: FArray = field(init=False)
    flux: Callable = field(init=False)
    dflux: Callable = field(init=False)
    source: Callable = field(init=False)
    source_jacobian_wrt_variables: Callable = field(init=False)
    source_jacobian_wrt_aux_variables: Callable = field(init=False)
    nonconservative_matrix: Callable = field(init=False)
    quasilinear_matrix: Callable = field(init=False)
    eigenvalues: Callable = field(init=False)
    residual: Callable = field(init=False)
    project_2d_to_3d: Callable = field(init=False)
    project_3d_to_2d: Callable = field(init=False)
    boundary_conditions: Callable = field(default=None, init=False)

    left_eigenvectors: Callable = field(default=None, init=False)
    right_eigenvectors: Callable = field(default=None, init=False)

    # --- Constants ---
    module = {
        "ones_like": np.ones_like,
        "zeros_like": np.zeros_like,
        "array": np.array,
        "squeeze": np.squeeze,
    }
    printer = "numpy"
    

    # ------------------------------------------------------------------

    def __attrs_post_init__(self):
        model = self.model

        modules = [self.module]
        if self.printer:
            modules.append(self.printer)
        
        flux=model._flux.lambdify(modules=modules)
        dflux=model._dflux.lambdify(modules=modules)
        nonconservative_matrix=model._nonconservative_matrix.lambdify(
            modules=modules
        )
        quasilinear_matrix=model._quasilinear_matrix.lambdify(
            modules=modules
        )
        eigenvalues=model._eigenvalues.lambdify(
            modules=modules
        )
        left_eigenvectors=model._left_eigenvectors.lambdify(
            modules=modules
        )
        right_eigenvectors=model._right_eigenvectors.lambdify(
            modules=modules
        )
        source=model._source.lambdify(modules=modules)
        source_jacobian_wrt_variables=model._source_jacobian_wrt_variables.lambdify(
            modules=modules
        )
        source_jacobian_wrt_aux_variables=model._source_jacobian_wrt_aux_variables.lambdify(
            modules=modules
        )
        residual=model._residual.lambdify(modules=modules)
        project_2d_to_3d=model._project_2d_to_3d.lambdify(
            modules=modules
        )
        project_3d_to_2d=model._project_3d_to_2d.lambdify(
            modules=modules
        )
        bcs = model._boundary_conditions.lambdify(modules=modules)


        # --- Assign frozen fields -------------------------------------
        object.__setattr__(self, "name", model.name)
        object.__setattr__(self, "dimension", model.dimension)
        object.__setattr__(self, "n_variables", model.n_variables)
        object.__setattr__(self, "n_aux_variables", model.n_aux_variables)
        object.__setattr__(self, "n_parameters", model.n_parameters)
        object.__setattr__(self, "parameters", model.parameter_values)

        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "dflux", dflux)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self, "source_jacobian_wrt_variables", source_jacobian_wrt_variables
        )
        object.__setattr__(
            self,
            "source_jacobian_wrt_aux_variables",
            source_jacobian_wrt_aux_variables,
        )
        object.__setattr__(self, "nonconservative_matrix", nonconservative_matrix)
        object.__setattr__(self, "quasilinear_matrix", quasilinear_matrix)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "left_eigenvectors", left_eigenvectors)
        object.__setattr__(self, "right_eigenvectors", right_eigenvectors)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "project_2d_to_3d", project_2d_to_3d)
        object.__setattr__(self, "project_3d_to_2d", project_3d_to_2d)
        object.__setattr__(self, "boundary_conditions", bcs)
