"""Runtime operator-form PDE system (``SystemModel``) — its own top-level
package, built from a declarative :class:`zoomy_core.model.derivation.Model`
via :meth:`SystemModel.from_model`."""
from zoomy_core.systemmodel.system_model import (
    SystemModel,
    register_function_slot,
)

__all__ = ["SystemModel", "register_function_slot"]
