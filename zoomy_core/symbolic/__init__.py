"""Principled symbolic-derivation primitive layer for zoomy_core.

This package replaces the semi-automatic, partially sympy-driven
pipeline in ``zoomy_core.model.models.ins_generator`` with a closed
set of well-defined mathematical primitives.  Every primitive is
ONE textbook calculus rule that fires only when explicitly invoked.
Sympy is used for arithmetic on ``Add``/``Mul``/``Pow`` and for
purely-structural ``xreplace``; sympy is *never* used to fire chain
rule, Leibniz, IBP, common-denominator, factor, or ``.doit()``.

Public surface (Phase 0):

* :func:`D`, :func:`Int`, :func:`Sub` — held-form sympy constructors
  (always ``evaluate=False``).
* :class:`PrimitiveDoesNotMatch` — exception raised when a primitive's
  preconditions fail.
* :class:`AutoEvalGuard` — context manager that bans sympy
  auto-evaluation calls inside its block (used in tests + the slim
  walkthrough verifier).

Primitives are added incrementally during Phase 0:

* ``primitives_canonical`` — purely-structural rewriters
  (``AlphaRename``, ``SplitIntegralOverAdd``,
  ``MergeIntegralsOverAdd``, ``DistributeMulOverAdd``,
  ``ZeroDerivativeOfFreeSymbol``, ``UnSubs``).
* ``primitives_calculus`` — calculus rules (``ProductRuleForward``,
  ``ProductRuleInverse``, ``ChainRule``, ``Leibniz``,
  ``FundamentalTheorem``, ``IntegrationByParts``,
  ``PolynomialIntegrate``, ``DistributeDerivativeOverAdd``,
  ``PullScalarOutOfDerivative``, ``PushScalarIntoDerivative``).
* ``primitives_change_of_variable`` — ``AffineChangeOfVariable``.
* ``primitives_basis`` — ``CanonicalizePhiDerivativeSubs``,
  ``ProjectBasisIntegrand``.
* ``primitives_substitution`` — ``Subst``, ``FunctionExpand``,
  ``SubsAtPoint``, ``SolveFor``.
* ``canonicalise`` — the fixed, finite-step ``canonicalise`` pass that
  replaces the legacy ``_simplify_preserve_integrals`` 6-iteration
  fixpoint loop.
"""

from zoomy_core.symbolic.errors import PrimitiveDoesNotMatch
from zoomy_core.symbolic.sp_safe import D, Int, Sub, held_function
from zoomy_core.symbolic.auto_eval_guard import AutoEvalGuard, AutoEvalForbidden

__all__ = [
    "D",
    "Int",
    "Sub",
    "held_function",
    "PrimitiveDoesNotMatch",
    "AutoEvalGuard",
    "AutoEvalForbidden",
]
