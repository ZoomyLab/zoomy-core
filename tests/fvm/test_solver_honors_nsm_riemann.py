"""REQ-157 — the numpy solvers must honour the NSM's explicit Riemann choice.

The Riemann solver is owned by the :class:`NumericalSystemModel`
(``nsm.riemann``); jax and dmplex both instantiate it via
``nsm.build_numerics()``.  numpy historically hard-coded its own class and
silently ignored the NSM, so a case that built the NSM with
``PositiveNonconservativeHLL`` still ran Rusanov on numpy — a *different*
discretisation than every other backend.

These tests pin the fix and its non-breaking half: an EXPLICIT riemann is
honoured by every numpy solver, while a DEFAULTED NSM leaves each solver's own
default (plain Rusanov / positive Rusanov) untouched.
"""

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm import solver_numpy as SN
from zoomy_core.fvm.riemann_solvers import (
    NonconservativeRusanov,
    PositiveNonconservativeHLL,
    PositiveNonconservativeRusanov,
)
from zoomy_core.model.models.swe import SWE
from zoomy_core.numerics.numerical_system_model import (
    NumericalSystemModel,
    ReconstructionSpec,
)


def _numerics(solver_cls, nsm):
    solver = solver_cls(
        time_end=1.0, compute_dt=timestepping.adaptive(CFL=0.45))
    solver.nsm = nsm
    return solver._build_numerics(solver._get_symbolic_model(nsm))


def _nsm(riemann=None):
    return NumericalSystemModel.from_model(
        SWE(dimension=2, parameters={"g": 9.81}),
        riemann=riemann,
        reconstruction=ReconstructionSpec(order=1),
    )


def test_explicit_riemann_honoured_by_plain_solver():
    nsm = _nsm(riemann=PositiveNonconservativeHLL)
    assert nsm.riemann_explicit is True
    assert isinstance(
        _numerics(SN.HyperbolicSolver, nsm), PositiveNonconservativeHLL)


def test_explicit_riemann_honoured_by_free_surface_solver():
    nsm = _nsm(riemann=PositiveNonconservativeHLL)
    assert isinstance(
        _numerics(SN.FreeSurfaceFlowSolver, nsm), PositiveNonconservativeHLL)


def test_default_nsm_keeps_plain_rusanov():
    nsm = _nsm()
    assert nsm.riemann_explicit is False
    # NonconservativeRoe subclasses NonconservativeRusanov — assert exact type.
    assert type(_numerics(SN.HyperbolicSolver, nsm)) is NonconservativeRusanov


def test_default_nsm_keeps_free_surface_positive_rusanov():
    nsm = _nsm()
    assert type(
        _numerics(SN.FreeSurfaceFlowSolver, nsm)
    ) is PositiveNonconservativeRusanov
