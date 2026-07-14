"""The emitted ``project_from_3d`` C++ kernel must keep the physical ``×h``
(``P3_h`` → ``column[·][1]``) factor on every conserved-moment row (REQ-81).

The symbolic ``project_from_3d`` row for a moment is ``q_k = P3_h · Σ_j w_j·
P3_u(ζ_j)`` (see ``test_project_roundtrip``).  The printer's shared
``_lower_project_from_3d`` binds ``P3_h`` → the depth-constant column sample
``column[0][1]`` and the per-node velocity samples ``P3_u(ζ_j)`` →
``column[j][2]``, so the emitted moment row is ``Σ_j w_j·column[0][1]·
column[j][2]`` — NOT the bare mean ``Σ_j w_j·column[j][2]``.  Dropping the
``column[·][1]`` factor returns ``u_mean`` instead of ``q_0 = h·u_mean`` and
inflates the preCICE interface state by ``1/h``.

CSE folds the depth-constant ``column[0][1]`` into the per-node coefficient
temporaries (``t0 = w_0·column[0][1]``; ``res[2] = column[0][2]·t0 + …``), so
the factor is asserted via the temporaries the moment row references, not by a
literal ``column[0][1]`` substring on the ``res[]`` line.
"""

import re

import pytest

from zoomy_core.model.models.sme import SME
from zoomy_core.transformation.generic_c import GenericCppModel
from zoomy_core.transformation.to_openfoam import FoamSystemModelPrinter
from zoomy_core.systemmodel.system_model import SystemModel


def _project_block(code):
    """Slice the ``project_from_3d`` kernel out of an emitted code string."""
    start = code.index("project_from_3d")
    return code[start:code.index("}", start) + 1]


def _moment_row_carries_h(block, row="res[2]"):
    """True iff the emitted moment row is ``Σ_j (…·column[0][1])·column[j][2]``
    — i.e. every coefficient temporary it references carries ``column[0][1]``
    (the ``P3_h`` factor), or the row multiplies it directly."""
    line = next(l for l in block.splitlines() if row in l and "=" in l)
    if "column[0][1]" in line:
        return True
    tvars = set(re.findall(r"\bt(\d+)\b", line))
    if not tvars:
        return False
    defs = {}
    for l in block.splitlines():
        m = re.match(r"\s*(?:const\s+)?[\w:]+\s+t(\d+)\s*=", l)
        if m:
            defs[m.group(1)] = l
    return all("column[0][1]" in defs.get(t, "") for t in tvars)


@pytest.mark.parametrize("level", [0, 2])
def test_foam_project_moment_row_carries_h_factor(level):
    """Foam ``project_from_3d`` moment row keeps the ``column[·][1]`` (P3_h)
    factor — the conserved moment ``q_0 = h·u_mean``, not the bare ``u_mean``."""
    sm = SystemModel.from_model(SME(level, dimension=2))
    blocks = FoamSystemModelPrinter(sm)._emit_projection_kernels()
    block = next(b for b in blocks
                 if "project_from_3d" in b and "interpolate_to_3d" not in b)
    assert _moment_row_carries_h(block), (
        f"SME({level}) foam project_from_3d res[2] dropped the P3_h "
        f"(column[·][1]) factor:\n{block}")


@pytest.mark.parametrize("level", [0, 2])
def test_generic_c_project_moment_row_carries_h_factor(level):
    """The shared generic-C lowering (same ``_lower_project_from_3d`` kernel)
    keeps the factor too."""
    code = GenericCppModel(SME(level, dimension=2)).create_code()
    assert _moment_row_carries_h(_project_block(code)), (
        f"SME({level}) generic-C project_from_3d res[2] dropped the "
        f"column[·][1] (P3_h) factor")
