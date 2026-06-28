"""Runtime operator-form PDE system (``SystemModel``) — its own top-level
package, built from a declarative :class:`zoomy_core.model.derivation.Model`
via :meth:`SystemModel.from_model`."""
from zoomy_core.systemmodel.system_model import (
    SystemModel,
    register_function_slot,
)
from zoomy_core.systemmodel.operations import register_aux, regularize_pow

__all__ = [
    "SystemModel",
    "register_function_slot",
    "register_aux",
    "regularize_pow",
]
