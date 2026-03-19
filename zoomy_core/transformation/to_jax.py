import jax.numpy as jnp  # type: ignore[reportMissingImports]

from zoomy_core.transformation.to_numpy import NumpyRuntimeModel


class JaxRuntimeModel(NumpyRuntimeModel):
    """
    JAX-backed runtime model.

    Provides the same interface as NumpyRuntimeModel, but compiles symbolic
    functions with JAX-friendly modules/printer for use in JAX solver paths.
    """

    module = {
        "ones_like": jnp.ones_like,
        "zeros_like": jnp.zeros_like,
        "array": jnp.array,
        "squeeze": jnp.squeeze,
    }
    printer = "jax"
