"""Solver-unification machinery: the Procedure/Statement IR, the solver-level
argument vocabulary and the ``ExternalProcedure`` contract.

Walkers live with the other emitters, in
:mod:`zoomy_core.transformation.procedure_c` (C family) and
:mod:`zoomy_core.transformation.procedure_python` (numpy / jax).

Import note: :mod:`zoomy_core.solver.external` is exposed LAZILY (PEP 562).
Building :data:`~zoomy_core.solver.external.REQUIRED_PROCEDURES` runs the
name-collision guard, which reads ``GenericCppBase.c_functions`` — and
``generic_c`` itself imports the argument vocabulary from this package.  Eager
re-export would close that cycle at interpreter start; deferring the external
declarations to first attribute access keeps ``import
zoomy_core.transformation.generic_c`` working while the guard still runs (just
at first use, never skipped).
"""
from zoomy_core.solver.arg_slots import (
    SOLVER_ARG_KINDS,
    SOLVER_ARG_MAPPING,
    SOLVER_ARG_SLOTS,
    SOLVER_DECL_KINDS,
    solver_arg_kind,
)
from zoomy_core.solver.ir import (
    SOLVER_PREFIX,
    Assign,
    Call,
    ForEachFace,
    IfStatic,
    Procedure,
    ProcedureNameError,
    Statement,
    While,
    check_procedure_name,
    require_resolved,
    split_target,
    written_names,
)

_LAZY = {
    "REQUIRED_PROCEDURES", "ExternalProcedure", "MissingProcedureBody",
    "assert_procedure_bodies", "required_procedure_gaps",
}


def __getattr__(name):
    """Defer the ``external`` module until first use (see the module note)."""
    if name in _LAZY:
        from zoomy_core.solver import external
        return getattr(external, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY)


__all__ = [
    "SOLVER_ARG_KINDS", "SOLVER_ARG_MAPPING", "SOLVER_ARG_SLOTS",
    "SOLVER_DECL_KINDS", "solver_arg_kind", "REQUIRED_PROCEDURES",
    "ExternalProcedure", "MissingProcedureBody", "assert_procedure_bodies",
    "required_procedure_gaps", "SOLVER_PREFIX", "Assign", "Call",
    "ForEachFace", "IfStatic", "Procedure", "ProcedureNameError", "Statement",
    "While", "check_procedure_name", "require_resolved", "split_target",
    "written_names",
]
