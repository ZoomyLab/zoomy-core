"""The elliptic solve's tolerance and restart are EMITTED constants (mandate 6a).

A backend must not carry its own float beside the ``gmres`` call.  The restart
matters more than the tolerance here, and for a measured reason: scipy's
``maxiter`` counts RESTARTS of a Krylov space of size ``restart``, so raising
the iteration budget buys nothing once the restarted iteration stagnates, while
the restart itself is the lever.  A FIXED restart therefore degrades silently
with every refinement — the elliptic operator's condition number is O(1/h^2)
and the solve is unpreconditioned.

Hence the property these tests pin: the resolved restart MOVES WITH THE PROBLEM
SIZE.  A constant is exactly the defect it replaces.
"""

import pytest

from zoomy_core.solver.constants import (
    ELLIPTIC_RESTART_CAP,
    ELLIPTIC_RESTART_MIN,
    ELLIPTIC_RTOL,
    ConstantResolutionError,
    elliptic_constants,
    elliptic_restart,
)

pytestmark = [pytest.mark.gate, pytest.mark.small, pytest.mark.printer]


def test_restart_scales_with_problem_size():
    """THE property.  Not "is it 20" — is it a function of n at all."""
    sizes = [64, 128, 256, 512]
    restarts = [elliptic_restart(n) for n in sizes]
    assert restarts == sizes                      # full space below the cap
    assert restarts == sorted(restarts)
    assert len(set(restarts)) == len(restarts), (
        "the restart is constant across a 8x refinement — that is the "
        "silent degradation this constant exists to end")


def test_restart_is_the_full_space_below_the_cap():
    """Full (non-restarted) GMRES is the only variant with a termination
    guarantee on a non-symmetric unpreconditioned operator: <= n steps.  Below
    the memory ceiling that is what we take."""
    for n in (ELLIPTIC_RESTART_MIN + 1, 100, ELLIPTIC_RESTART_CAP):
        assert elliptic_restart(n) == n


def test_restart_is_capped_and_floored():
    assert elliptic_restart(ELLIPTIC_RESTART_CAP * 7) == ELLIPTIC_RESTART_CAP
    assert elliptic_restart(1) == ELLIPTIC_RESTART_MIN


def test_block_size_is_required_and_validated():
    """No default without a size: a caller that cannot say how big its block is
    cannot be given a size-dependent restart."""
    with pytest.raises(TypeError):
        elliptic_constants()                                # noqa: PT011
    with pytest.raises(ConstantResolutionError):
        elliptic_constants(n_unknowns=0)


def test_both_constants_are_emitted_with_provenance():
    consts = elliptic_constants(n_unknowns=256)
    assert set(consts.values) == {"c_elliptic_tol", "c_elliptic_restart"}
    assert consts.values["c_elliptic_tol"] == ELLIPTIC_RTOL
    assert consts.values["c_elliptic_restart"] == 256
    for name in consts.values:
        assert consts.provenance[name], f"{name} has no provenance"
    assert "elliptic_restart" in consts.provenance["c_elliptic_restart"]


def test_the_restart_is_emitted_as_an_int_and_the_tolerance_as_a_real():
    """The ``Assign`` that DEFINES the value is the only place a literal may
    live; a restart emitted as a float would not compile as a Krylov dimension."""
    assigns = {a.target: a for a in elliptic_constants(n_unknowns=64).assigns()}
    assert assigns["c_elliptic_restart"].ctype == "int"
    assert assigns["c_elliptic_tol"].ctype == ""


def test_an_override_is_visible_in_the_provenance():
    """Pinning a restart is allowed — reproducing the measured degradation
    deliberately is legitimate — but it must not look like the default."""
    default = elliptic_constants(n_unknowns=256)
    pinned = elliptic_constants(n_unknowns=256, restart=20)
    assert pinned.values["c_elliptic_restart"] == 20
    assert pinned.provenance["c_elliptic_restart"] != \
        default.provenance["c_elliptic_restart"]
    assert "override" in pinned.provenance["c_elliptic_restart"]


def test_an_override_cannot_exceed_the_block_size():
    """A Krylov space larger than the system is not a thing."""
    assert elliptic_constants(
        n_unknowns=32, restart=10_000).values["c_elliptic_restart"] == 32


def test_invalid_overrides_raise():
    with pytest.raises(ConstantResolutionError):
        elliptic_constants(n_unknowns=64, restart=0)
    with pytest.raises(ConstantResolutionError):
        elliptic_constants(n_unknowns=64, tol=0.0)
