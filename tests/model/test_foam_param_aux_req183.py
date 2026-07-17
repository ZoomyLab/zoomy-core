"""REQ-183 — foam's field-level ``update_aux_variables`` must BAKE model
parameters as literals.

The emitted ``update_aux_variables(Q, Qaux, dt, mesh)`` operates on
volFields with NO parameter vector ``p`` (unlike the per-cell numpy/jax
leg), so an algebraic aux closure that references a model parameter —
``MalpassetSWE``'s KP-desingularised ``hinv`` carries ``wet_dry_eps`` —
previously tripped the free-symbol guard and raised, killing ALL codegen
for that model.  The printer now substitutes the model instance's
``parameter_values`` into the closure before the guard, so the parameter
lowers as a numeric literal.
"""

from zoomy_core.model.models.malpasset import MalpassetSWE
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.transformation.to_openfoam import FoamSystemModelPrinter


def test_parameterized_algebraic_aux_lowers_with_baked_values():
    sm = SystemModel.from_model(MalpassetSWE())
    block = FoamSystemModelPrinter(sm)._emit_update_aux_variables()
    # The hinv closure is emitted as a field-algebra assignment …
    assert "hinv" in block and "*Qaux[" in block, block
    # … with the parameter resolved to a literal, not left symbolic.
    assert "wet_dry_eps" not in block, block
