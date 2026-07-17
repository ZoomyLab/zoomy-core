"""Contract tests pinning the two lowering-seam SHAPE guarantees.

Both seams exist because a symbolic operator matrix has mixed-rank rows (some
entries depend on the per-cell state, some are compile-time constants), and the
two lowering families cope with that in opposite, complementary ways:

PART A — SCALARIZATION (the scalar-C family: dmplex / amrex / openfoam / js /
glsl, all sharing ``generic_c``).  These emit one ``res[i] = <scalar>;`` line
per component.  ``GenericCppBase._process_kernel_from_function`` /
``convert_expression_body`` first run ``_expand_vector_conditionals`` then
``sp.flatten`` so the printer only ever sees SCALAR sub-expressions — it NEVER
needs an ``NDimArray`` print handler.  (This is why the old amrex
``_print_ImmutableDenseNDimArray`` override is dead code: nothing routes an
array to the printer.)  We pin that by subclassing the printer with an
``_print_ImmutableDenseNDimArray`` that raises and asserting full emission still
succeeds.

PART B — RANK-UNIFORMITY (the vectorized family: numpy / jax).  These lambdify
the WHOLE array and call it with grid-broadcast state, so every row must share
the batch rank.  ``zoomy_core.transformation.vectorize.uniform_rank`` wraps each
vector-symbol-free entry in ``zeros_like(anchor)`` / ``c*ones_like(anchor)`` so
a constant row adopts its neighbours' batch shape.  We pin that the wrap is
applied and that the lambdified array is rank-POLYMORPHIC: scalar anchor →
scalar entries, batched anchor → batched entries (jax is covered by its own
repro test; here numpy only).
"""

import numpy as np
import pytest
import sympy as sp

from zoomy_core.fvm.userfunctions import numpy_module
from zoomy_core.model.models.malpasset import MalpassetSWE
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics.numerical_system_model import NumericalSystemModel
from zoomy_core.systemmodel.operations import gate_eigenvalues_dry
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.transformation.generic_c import GenericCppModel
from zoomy_core.transformation.vectorize import uniform_rank


# =====================================================================
#  PART A — scalarization guarantee (scalar-C family)
# =====================================================================


class _ScalarizationGuardedPrinter(GenericCppModel):
    """The shared generic-C model printer with the NDimArray handlers armed
    to blow up.  If the ``_expand_vector_conditionals`` + ``sp.flatten``
    scalarization seam ever let a whole array reach the printer, emission would
    raise here instead of silently emitting a broken kernel."""

    def _print_ImmutableDenseNDimArray(self, expr):
        raise AssertionError("scalarization contract violated")

    # Any other NDimArray flavour MRO-dispatches through this alias.
    _print_NDimArray = _print_ImmutableDenseNDimArray


def _representative_systems():
    """The C-family representatives: SWE 1-D + 2-D, a genuine 3-D SME with one
    modal level, and a gated NSM (``gate_eigenvalues_dry`` wraps the eigenvalue
    spectrum in a dry-state conditional — exercises the conditional-expansion
    leg of the seam)."""
    return {
        "SWE_1d": SWE(dimension=1),
        "SWE_2d": SWE(dimension=2),
        "SME_l1_3d": SME(level=1, dimension=3),
        "gated_NSM": NumericalSystemModel.from_model(
            SWE(dimension=2), extra_operations=[gate_eigenvalues_dry()]
        ),
    }


@pytest.mark.parametrize("name", list(_representative_systems().keys()))
def test_generic_c_never_prints_ndimarray(name):
    """The generic-C lowering emits every representative system without the
    printer ever needing an ``NDimArray`` handler, and produces per-component
    ``res[i]`` assignment lines."""
    model = _representative_systems()[name]
    code = _ScalarizationGuardedPrinter(model).create_code()
    # Emission succeeded -> the armed handler was never reached -> the seam
    # scalarized every operator before printing.
    assert "res[0]" in code, f"{name}: no per-component res[] assignments emitted"
    assert code.count("res[") > 0


def test_scalarization_guard_is_actually_armed():
    """Negative control: the guard DOES fire if an array reaches the printer,
    so the green PART A tests above mean the seam works — not that the guard is
    inert."""
    printer = _ScalarizationGuardedPrinter(SWE(dimension=1))
    with pytest.raises(AssertionError, match="scalarization contract violated"):
        printer.doprint(sp.Array([[1, 2], [3, 4]]))


# =====================================================================
#  PART B — rank-uniformity guarantee (vectorized family)
# =====================================================================


def _malpasset_mixed_rank_array():
    """A flat array carrying all three entry kinds ``uniform_rank`` must
    reconcile: (i) an identically-zero row (the MalpassetSWE flux ``b``-row is
    ``[0, 0]``), (ii) a nonzero compile-time constant, (iii) state-dependent
    entries (the ``hinv*hu**2`` etc. flux rows).  Returns the array, the vector
    symbols (state + aux), and the conventional anchor (the first state
    symbol)."""
    sm = SystemModel.from_model(MalpassetSWE())
    flat = list(sp.flatten(sm.flux))          # b-row [0, 0] + hu/hv/hinv rows
    arr = sp.Array(flat + [sp.Integer(2)])    # + a nonzero constant entry
    vector_symbols = tuple(sm.state) + tuple(sm.aux_state)
    anchor = list(sm.state)[0]
    return arr, vector_symbols, anchor


def _is_anchor_wrapped(entry, anchor):
    """True iff ``entry`` is (or contains) a ``ones_like``/``zeros_like`` of the
    anchor symbol."""
    return any(
        a.func.__name__ in ("ones_like", "zeros_like") and anchor in a.args
        for a in entry.atoms(sp.Function)
    )


def test_uniform_rank_wraps_constant_and_zero_rows():
    """Every entry of the normalized array either depends on a vector symbol or
    is wrapped in ``ones_like``/``zeros_like(anchor)`` — no bare vector-free
    constant survives; active rows are left untouched."""
    arr, vector_symbols, anchor = _malpasset_mixed_rank_array()
    wrapped = uniform_rank(arr, vector_symbols, anchor)
    vset = set(vector_symbols)

    flat_in = [sp.sympify(e) for e in arr]
    flat_out = list(wrapped)
    assert len(flat_out) == len(flat_in)

    for orig, new in zip(flat_in, flat_out):
        if orig.free_symbols & vset:
            assert new == orig, "active (state-dependent) row was rewritten"
        else:
            assert _is_anchor_wrapped(new, anchor), (
                f"vector-free entry {orig} was not anchored")
        # the invariant itself: every entry references a vector symbol,
        # genuinely or through the anchor inside the wrap.
        assert (new.free_symbols & vset) or _is_anchor_wrapped(new, anchor)

    # Concrete pins for the three entry kinds.
    zeros_like = sp.Function("zeros_like")
    ones_like = sp.Function("ones_like")
    assert flat_out[0] == zeros_like(anchor)              # flux b-row col 0
    assert flat_out[1] == zeros_like(anchor)              # flux b-row col 1
    assert flat_out[-1] == sp.Integer(2) * ones_like(anchor)  # constant entry


@pytest.mark.parametrize("batch", [None, 7])
def test_uniform_rank_lambdify_is_rank_polymorphic(batch):
    """The lambdified normalized array is rank-polymorphic under numpy: called
    with scalar per-cell floats it returns scalar entries (shape == array
    shape); called with batched 1-D arrays it returns batched entries (shape ==
    array shape + batch), never a ragged/object array."""
    arr, vector_symbols, anchor = _malpasset_mixed_rank_array()
    wrapped = uniform_rank(arr, vector_symbols, anchor)
    free = sorted(wrapped.free_symbols, key=str)
    # ``[dict, "numpy"]``: the numpy runtime module for ones_like/zeros_like +
    # the NumPyPrinter that lowers the whole NDimArray to ``numpy.array([...])``
    # (exactly what ``NumpyRuntimeModel.from_system_model`` passes).
    fn = sp.lambdify(free, wrapped, modules=[numpy_module(), "numpy"])
    arr_shape = tuple(wrapped.shape)

    if batch is None:
        out = np.asarray(fn(*[1.5] * len(free)))
        assert out.shape == arr_shape            # scalar anchor -> scalar entries
    else:
        args = [np.linspace(1.0, 2.0, batch)] * len(free)
        out = np.asarray(fn(*args))
        assert out.shape == arr_shape + (batch,)  # batched anchor -> batched rows
    assert out.dtype != object, "rows failed to stack — not rank-uniform"


def test_without_uniform_rank_the_batched_array_is_ragged():
    """Negative control: the SAME array lambdified WITHOUT ``uniform_rank``
    cannot batch — the vector-free zero/constant rows stay scalar while the
    state rows go 1-D, so the array is ragged.  This is what makes the
    rank-uniformity guarantee load-bearing."""
    arr, _vector_symbols, _anchor = _malpasset_mixed_rank_array()
    free = sorted(arr.free_symbols, key=str)
    raw = sp.lambdify(free, arr, modules=[numpy_module(), "numpy"])
    n = 7
    args = [np.linspace(1.0, 2.0, n)] * len(free)
    ragged = False
    try:
        out = np.asarray(raw(*args))
        ragged = (out.dtype == object) or (out.shape != tuple(arr.shape) + (n,))
    except (ValueError, TypeError):
        ragged = True
    assert ragged, "raw array batched cleanly — uniform_rank is not load-bearing"
