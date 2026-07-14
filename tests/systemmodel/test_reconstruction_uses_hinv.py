"""REQ-156 — the order-2 primitive reconstruction map must not emit a RAW 1/h.

SWE's reconstruction map ``[b, b+h, hu/h, hv/h]`` divides the momentum by the
depth.  On a dry bed that raw ``1/h`` is a division by zero — a SIGFPE on step 1
under OpenFOAM's FP trapping.  The desingularized ``hinv`` aux (``1/max(h,eps)``)
already exists and is what the flux/eigenvalues use; the wet/dry-safe defaults
(``desingularize_hinv``) must sweep it into the reconstruction map too, so every
backend is dry-safe by construction.
"""

import sympy as sp

from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics.numerical_system_model import (
    NumericalSystemModel,
    ReconstructionSpec,
)


def _reconstruction(dim):
    m = SWE(dimension=dim, parameters={"g": 9.81})
    nsm = NumericalSystemModel.from_model(
        m, reconstruction=ReconstructionSpec(order=2))
    return nsm


def _has_negative_h_power(expr, h):
    """True iff ``expr`` contains a genuine ``h**(-n)`` factor."""
    for pw in sp.sympify(expr).atoms(sp.Pow):
        if pw.base == h and pw.exp.is_number and pw.exp.is_negative:
            return True
    return False


def test_reconstruction_velocity_goes_through_hinv():
    nsm = _reconstruction(2)
    rv = list(nsm.reconstruction_variables)
    names = {str(s) for e in rv for s in sp.sympify(e).free_symbols}
    assert "hinv" in names, f"reconstruction_variables missing hinv: {rv}"


def test_reconstruction_has_no_raw_reciprocal_depth():
    for dim in (1, 2):
        nsm = _reconstruction(dim)
        h = next(s for s in nsm.state if str(s) == "h")
        for e in nsm.reconstruction_variables:
            assert not _has_negative_h_power(e, h), (
                f"reconstruction_variables carries a raw 1/h "
                f"(dim={dim}): {list(nsm.reconstruction_variables)}")
