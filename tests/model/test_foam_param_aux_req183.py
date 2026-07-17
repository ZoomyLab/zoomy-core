"""REQ-183 — foam's field-level ``update_aux_variables`` must resolve model
parameters.

The emitted ``update_aux_variables(Q, Qaux, p, dt, mesh)`` operates on
volFields; an algebraic aux closure that references a model parameter —
``MalpassetSWE``'s KP-desingularised ``hinv`` carries ``wet_dry_eps`` —
previously tripped the free-symbol guard and raised, killing ALL codegen
for that model.  Final design (11b0d57, superseding the interim
literal-baking fix c6e6e57): parameters are a RUNTIME INPUT — the
interface carries the parameter vector ``p`` and the closure lowers the
symbol to its ``p[<idx>]`` slot.  The symbolic name may still appear in
the human-readable ``//`` comment; the CODE line must use ``p[...]``.
"""

from zoomy_core.model.models.malpasset import MalpassetSWE
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.transformation.to_openfoam import FoamSystemModelPrinter


def test_parameterized_algebraic_aux_lowers_via_p_vector():
    sm = SystemModel.from_model(MalpassetSWE())
    block = FoamSystemModelPrinter(sm)._emit_update_aux_variables()
    # The interface carries the parameter vector …
    assert "const Foam::List<Foam::scalar>& p," in block, block
    code = [l for l in block.splitlines()
            if not l.lstrip().startswith("//")]
    assign = [l for l in code if "*Qaux[" in l and "=" in l]
    # … the hinv closure is emitted as a field-algebra assignment …
    assert assign, block
    # … resolving the parameter through p[<idx>], never the bare symbol.
    assert any("p[" in l for l in assign), block
    assert all("wet_dry_eps" not in l for l in code), block
