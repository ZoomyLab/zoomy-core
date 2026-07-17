"""REQ-189: the Rusanov spectral radius must stay a LAZY ``Max`` over the
(opaque) wave spectrum, never a symbolically-evaluated reduction.

Evaluating ``Max(|λ_i|)`` over a non-orderable numerical spectrum sends sympy
into an O(n²) ``_find_localzeros → factor_terms(λ_i − λ_j)`` ordering attempt
that expands every huge flux-Jacobian-eigenvalue tree — the VAM/MLVAM/SME
1-step-twin SETUP explosion (vam_dambreak was ~210 s of pure setup).  The fix
keeps the reduction unevaluated at construction (``riemann_solvers``) and pins
``evaluate=False`` when the numpy opaque-kernel lowering rewrites the eigenvalue
atoms (``to_numpy._lower_opaque_kernels``).  ``max`` is exact + associative, so
the lowered numerics are bit-identical.

This is the timing/identity regression guard:
  * SETUP (build + numpy-lower the Rusanov numerics) stays far below the pre-fix
    minutes — the canary that the ``factor_terms`` blowup has not returned;
  * the spectrum stays behind the opaque ``eigenvalues`` kernel inside a ``Max``
    node — structurally lazy, not inlined-and-factored;
  * the numerics still lowers to a callable.
"""
import time

import sympy as sp
import pytest

from zoomy_core.model.models import VAM
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.model.kernel_functions import eigenvalues as _eig_kernel
from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
from zoomy_core.systemmodel.system_model import SystemModel


@pytest.fixture(scope="module")
def vam_sm():
    # warm-cached derivation (v7 _prebuilt) — NOT part of the timed section
    return SystemModel.from_model(
        VAM(closures=[Newtonian(), NavierSlip(), StressFree()], level=1))


def test_rusanov_setup_is_cheap(vam_sm):
    """The blowup lived in numerics construction (the ``Max`` build) AND in the
    numpy opaque-kernel xreplace lowering; both spent minutes pre-fix."""
    t0 = time.perf_counter()
    num = NonconservativeRusanov(vam_sm)      # builds numerical_flux/fluctuations
    rt = num.to_runtime_numpy()               # numpy lowering (the xreplace seam)
    setup = time.perf_counter() - t0

    assert setup < 30.0, (
        f"Rusanov setup {setup:.1f}s — spectral-radius factor_terms blowup back? "
        "(pre-fix ~210 s for VAM(1))")
    assert rt.numerical_fluctuations is not None
    assert rt.numerical_flux is not None


def test_spectrum_stays_opaque_inside_a_max(vam_sm):
    """The wave spectrum must remain the opaque ``eigenvalues`` kernel sitting
    inside a ``Max`` reduction — i.e. lazy, not inlined then factored away."""
    num = NonconservativeRusanov(vam_sm)
    fluct = sp.sympify(num.functions["numerical_fluctuations"].definition)
    assert fluct.atoms(sp.Max), "spectral-radius Max was collapsed away"
    assert fluct.atoms(_eig_kernel), "eigenvalue spectrum was expanded, not opaque"
