"""``numerical_flux`` face-wavespeed row + the ``resolve_opaque`` printer option.

Both on DERIVED models (SME / VAM families), per the test policy:

* ITEM 1 — the registered ``numerical_flux`` returns
  ``[ flux(n) | lambda_max(1) ]`` of shape ``(n+1, 1)``; the trailing row is
  ``face_max_abs_eigenvalue`` on the RAW face states (the driver-local dt
  enabler).
* ITEM 2 — ``resolve_opaque`` (printer option, default OFF):
  OFF emits registered-function calls exactly as today; ON inlines the
  registered symbolic definition into the consuming expression before the
  per-kernel cse pass, so one consumer expression = ONE CSE silo (the
  printer-side replacement for the fused ``numerical_face`` kernel).
  Genuinely external kernels (the opaque ``eigenvalues`` UserFunctions
  primitive — numerical impls per backend) have no symbolic body and MUST
  survive as calls.
"""

import numpy as np
import pytest
import sympy as sp

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.basefunction import Function
from zoomy_core.model.models.sme import SME
from zoomy_core.numerics.numerical_system_model import NumericalSystemModel
from zoomy_core.fvm.riemann_solvers import (
    NonconservativeRusanov,
    PositiveNonconservativeHLL,
)
from zoomy_core.transformation.to_amrex import AmrexNumerics

# Transitional keep (approved spec §1d): active REQ-188 Riemann-fusion seam.
# Sunset into P01/P02 once fusion lands and the goldens emit the fused form.
pytestmark = [pytest.mark.printer, pytest.mark.small, pytest.mark.fusion_wip]


def _cse_ops(rows):
    temps, red = sp.cse([sp.sympify(r) for r in rows],
                        symbols=sp.numbered_symbols("t"))
    return (sum(sp.count_ops(rhs) for _, rhs in temps)
            + sum(sp.count_ops(r) for r in red))


@pytest.fixture(scope="module")
def sme_numerics():
    nsm = NumericalSystemModel.from_model(
        SME(level=1, dimension=2), riemann=PositiveNonconservativeHLL
    ).derive()
    return nsm, nsm.build_numerics()


def _consumer(num):
    """Driver-shaped face expression: the two registered kernels invoked as
    OPAQUE proxy calls (``num.call.*``) — what a backend computes per face."""
    call_args = (num.variables_minus, num.variables_plus,
                 num.aux_variables_minus, num.aux_variables_plus,
                 num.parameters, num.normal)
    rows = (list(sp.flatten(num.call["numerical_flux"](*call_args)))
            + list(sp.flatten(num.call["numerical_fluctuations"](*call_args))))
    consumer = ZArray(rows).reshape(3 * num.n_variables + 1, 1)
    return Function(name="face_pair",
                    args=num.functions["numerical_flux"].args,
                    definition=consumer)


# ── ITEM 1: the (n+1, 1) numerical_flux layout ───────────────────────


def test_numerical_flux_layout_and_wavespeed_row(sme_numerics):
    nsm, num = sme_numerics
    n = num.n_variables
    defn = num.functions["numerical_flux"].definition
    assert defn.shape == (n + 1, 1)

    rt = num.to_runtime_numpy()
    p = np.array(list(nsm.parameter_values.values()), dtype=float)
    n_aux = len(nsm.aux_state)
    rng = np.random.default_rng(7)
    qL = np.array([0.0, 1.4, 0.35, 0.1])
    qR = np.array([0.02, 0.9, -0.2, 0.05])
    auxL = rng.uniform(0.1, 0.5, n_aux)
    auxR = rng.uniform(0.1, 0.5, n_aux)
    # SME(dimension=2) = (t, x, z): ONE horizontal direction -> 1-D normal.
    nrm = np.array([1.0])

    out = np.asarray(
        rt.numerical_flux(qL, qR, auxL, auxR, p, nrm), dtype=float
    ).reshape(-1)
    assert out.shape[0] == n + 1
    lamL = float(np.asarray(
        rt.local_max_abs_eigenvalue(qL, auxL, p, nrm), dtype=float))
    lamR = float(np.asarray(
        rt.local_max_abs_eigenvalue(qR, auxR, p, nrm), dtype=float))
    np.testing.assert_allclose(out[n], max(lamL, lamR), rtol=1e-12)


# ── ITEM 2: resolve_opaque OFF / ON ──────────────────────────────────


def test_resolve_opaque_defaults_off_and_emits_calls(sme_numerics):
    _nsm, num = sme_numerics
    pr = AmrexNumerics(num)
    assert pr.resolve_opaque is False
    body = pr._process_kernel_from_function(_consumer(num))[0]
    # today's behaviour: the registered kernels stay opaque CALLS
    assert "numerical_flux(" in body
    assert "numerical_fluctuations(" in body


def test_resolve_opaque_inlines_and_matches_fused_silo(sme_numerics):
    _nsm, num = sme_numerics
    fobj = _consumer(num)
    pr_on = AmrexNumerics(num, resolve_opaque=True)

    body = pr_on._process_kernel_from_function(fobj)[0]
    assert "numerical_flux(" not in body
    assert "numerical_fluctuations(" not in body

    # the splice is VERBATIM: resolved rows == the registered definitions
    resolved = list(sp.flatten(pr_on._resolve_opaque_calls(fobj.definition)))
    ref = (list(sp.flatten(num.functions["numerical_flux"].definition))
           + list(sp.flatten(num.functions["numerical_fluctuations"].definition)))
    assert len(resolved) == len(ref)
    assert all(a == b for a, b in zip(resolved, ref))

    # fusion parity: one consumer silo == the old fused numerical_face silo
    ops_on = _cse_ops(resolved)
    ops_face = _cse_ops(
        list(sp.flatten(num.functions["numerical_face"].definition)))
    assert ops_on == ops_face


def test_resolve_opaque_leaves_external_kernels_as_calls():
    """VAM has no closed-form spectrum — its wave speed is the opaque
    ``eigenvalues(idx, *A_flat)`` UserFunctions kernel.  It has no symbolic
    definition, so resolve_opaque must keep it a CALL."""
    from zoomy_core.model.models.vam import VAM

    nsm = NumericalSystemModel.from_model(
        VAM(level=1), riemann=NonconservativeRusanov).derive()
    num = nsm.build_numerics()
    pr_on = AmrexNumerics(num, resolve_opaque=True)
    body = pr_on._process_kernel_from_function(_consumer(num))[0]
    assert "numerical_flux(" not in body        # registered -> inlined
    assert "eigenvalues(" in body               # external -> stays a call
