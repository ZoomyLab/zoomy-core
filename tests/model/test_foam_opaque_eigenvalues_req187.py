"""REQ-187 — the foam Numerics printer must namespace-qualify EVERY opaque
UserFunctions kernel call.

Generated kernels live in ``namespace Numerics`` while the opaque kernels
(``eigensystem``, ``eigenvalues``, ``solve``) live in ``namespace numerics``
(UserFunctions.H) — an unqualified call never resolves.  ``eigensystem`` was
qualified from day one; ``eigenvalues`` (the λ-only wave-speed kernel used by
models with OPAQUE spectra, e.g. VAM) was not, so VAM foam builds failed with
``'eigenvalues' was not declared in this scope`` while analytic-eigenvalue SWE
stayed green.
"""

import re

from zoomy_core.model.models.vam import VAM
from zoomy_core.numerics.numerical_system_model import to_numerical_system_model
from zoomy_core.transformation.to_openfoam import FoamNumericsPrinter


def test_opaque_eigenvalues_calls_are_namespace_qualified():
    nsm = to_numerical_system_model(VAM(dimension=2))
    code = FoamNumericsPrinter(nsm.build_numerics()).create_code()
    assert "numerics::eigenvalues(" in code, "opaque eigenvalues not emitted"
    # No unqualified call may survive: every `eigenvalues(` is preceded by
    # `numerics::` (word boundary keeps `numerics::eigenvalues` itself clean).
    bare = re.findall(r"(?<!numerics::)\beigenvalues\s*\(", code)
    assert not bare, f"unqualified eigenvalues calls remain: {len(bare)}"
