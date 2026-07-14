"""``CppModel.create_code()`` must never emit a raw sympy ``Integral`` (REQ-91).

The C++ code printer cannot lower an unresolved Galerkin bracket ``Integral``
(and such brackets must never be ``.doit()``-ed — sympy's heurisch hangs).  The
``project_from_3d`` rows of every model are therefore the Integral-FREE,
fixed-node quadrature reduction (``Basisfunction.projection_rows``); SWE used to
hand-write a raw ``P3_h·Integral(P3_u, (ζ,0,1))`` depth-average, which the
printer choked on at ``PrintMethodNotImplementedError: … Integral``.

These lock:

* ``CppModel(SWE(dim)).create_code()`` succeeds for 1-D and 2-D, and
* the emitted source carries no ``Integral`` substring,

for SWE (the model that regressed) and a moment model (SME) that shares the
projection-kernel emission path.
"""

import pytest

from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.sme import SME
from zoomy_core.transformation.generic_c import CppModel
from zoomy_core.systemmodel.system_model import SystemModel


@pytest.mark.parametrize("model,label", [
    (SWE(dimension=1), "SWE 1-D"),
    (SWE(dimension=2), "SWE 2-D"),
    (SME(level=1, dimension=2), "SME(1) 2-D"),
])
def test_cppmodel_create_code_has_no_integral(model, label):
    """C++ codegen returns source with no unresolved ``Integral``."""
    code = CppModel(model).create_code()
    assert code, f"{label}: create_code returned empty source"
    assert "Integral" not in code, (
        f"{label}: emitted C++ carries an unresolved sympy Integral")


def test_swe_project_row_carries_h_factor():
    """The SWE ``project_from_3d`` moment row is the conserved ``q = h·u_mean``
    (fixed-node depth-average × ``P3_h``), not the bare mean — and is
    Integral-free arithmetic in the column samples."""
    import sympy as sp

    rows = [sp.sympify(e) for e in
            sp.flatten(SystemModel.from_model(SWE(dimension=2)).project_from_3d)]
    assert not any(r.has(sp.Integral) for r in rows), "raw Integral in rows"
    # momentum rows (res[2], res[3]) must reference the P3_h head symbol.
    P3h = sp.Symbol("P3_h", real=True)
    assert P3h in rows[2].free_symbols and P3h in rows[3].free_symbols
