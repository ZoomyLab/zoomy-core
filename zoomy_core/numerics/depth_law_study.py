"""REQ-194 — one-call NSM construction over the depth-law parameter axes.

The machinery for a parameter study, not the study itself: this module makes
every depth-law choice selectable at CONSTRUCTION time so a sweep is a pure loop
with no code edit per run.  It MEASURES NOTHING and asserts no physics — which
setting is right is exactly what the sweep exists to find out.

The axes
--------

``path``          which reciprocal regularization runs (both share the SAME
                  Kurganov–Petrova definition :func:`~zoomy_core.systemmodel.
                  operations.kp_hinv`; there is no ``1/(h + eps)`` and no
                  ``1/max(eps, h)`` anywhere):

                  * ``"direct"`` — KP substituted inline wherever ``h**(-n)``
                    appears; no aux row, no added state.
                  * ``"aux"``    — ``1/h**n`` promoted to an ``hinv`` aux whose
                    update rule IS the KP expression; operators carry
                    ``hinv**n``.
                  * ``"legacy"`` — the pre-REQ ``desingularize_hinv()`` (hinv
                    aux at ``wet_dry_eps``, eigenvalues carved out onto the
                    ``1/max(1e-14, h)`` floor).  The control point.
                  * ``"none"``   — no depth regularization; bare ``1/h``.

                  ``"direct"`` and ``"aux"`` are mutually exclusive by
                  construction (one string picks one path) AND on the system
                  (applying both raises).

``eps``           the REGULARIZATION scale, default ``1e-2`` — its own quantity,
                  never ``wet_dry_eps``.  Reusing the wet/dry threshold as a
                  regularization scale is what put ``h/(h + 1e-2)`` on the
                  celerity.  Ignored by ``path="legacy"`` / ``"none"``.

``eigenvalues``   whether the spectrum is swept by the chosen path
                  (``"regularize"``) or keeps the exact ``1/h``
                  (``"exclude"``).  This is the open question: with a normalized
                  normal the SWE celerity separates cleanly
                  (``√(g·h) + hinv·(hu·n0 + hv·n1)``) so sweeping it is
                  harmless, but the SME spectrum is
                  ``hinv·(n0·q_0 ± n0·√(g·h³ + q_1²))`` — the reciprocal
                  multiplies the celerity and the radical does not factor.
                  Neither answer is baked in.

``guard``         the wet/dry eigenvalue treatment, independent of everything
                  above: ``None`` / ``"power"`` / ``"gate"`` / ``"both"``.

``normalize_normal``  impose ``|n| = 1`` before anything else.  Default ``True``
                  here (it is a fact about the normal, and it is what lets the
                  celerity separate at all), but exposed so the sweep can turn
                  it off and measure what it actually buys.

Usage
-----

::

    from zoomy_core.numerics.depth_law_study import build_nsm, AXES, iter_axes

    for combo in iter_axes():
        nsm = build_nsm(SWE(dimension=2), **combo)
        ...

Note ``build_nsm`` consumes the model FRESH each call: the NSM promotes its
SystemModel in place, so a SystemModel handed to two different axis points would
carry the first point's operators into the second.  Pass a Model (or a callable
returning one) and let this module build the SystemModel per point.
"""
from __future__ import annotations

import functools
import inspect
import itertools
from typing import Any, Callable, Iterator, Optional, Union

from zoomy_core.numerics.numerical_system_model import NumericalSystemModel
from zoomy_core.systemmodel.operations import _DEFAULT_REGULARIZATION_EPS
from zoomy_core.systemmodel.system_model import SystemModel

__all__ = [
    "AXES",
    "PATHS",
    "GUARDS",
    "EIGENVALUE_TREATMENTS",
    "DEFAULT_EPS",
    "build_nsm",
    "iter_axes",
    "axis_id",
]

#: Regularization paths.  ``"direct"`` and ``"aux"`` are the two study variants;
#: ``"legacy"`` and ``"none"`` are controls (the shipped behaviour and the
#: un-regularized system).
PATHS = ("direct", "aux", "legacy", "none")

#: Wet/dry eigenvalue guards.  NB ``"both"`` is byte-identical to ``"gate"`` —
#: ``gate_eigenvalues_dry`` carries the power guard internally (REQ-74: a
#: branchless ``conditional`` evaluates both arms, so the wet branch must stay
#: real) and the guard is idempotent.  REQ-181 pins that identity.  Both are
#: kept as separate axis points so the sweep records the composition explicitly
#: rather than assuming the identity still holds.
GUARDS = (None, "power", "gate", "both")

#: Spectrum treatment under the chosen path.
EIGENVALUE_TREATMENTS = ("regularize", "exclude")

#: Default regularization scale — the user's "start near the old working state"
#: (SWE / Malpasset shipped ``wet_dry_eps = 1e-2``).  A STUDY default on the
#: study paths only; the legacy path still reads ``wet_dry_eps``.
DEFAULT_EPS = _DEFAULT_REGULARIZATION_EPS

#: The sweep grid: the two study paths × the four guards × the two spectrum
#: treatments = 16 points.  ``eps`` is deliberately NOT part of the grid — it is
#: a continuous knob, so a sweep varies it as an outer loop over whichever of
#: these 16 structural points it cares about.
AXES = {
    "path": ("direct", "aux"),
    "guard": GUARDS,
    "eigenvalues": EIGENVALUE_TREATMENTS,
}


def iter_axes(axes: Optional[dict] = None) -> Iterator[dict]:
    """Yield one ``{axis: value}`` kwargs dict per point of the cartesian
    product of ``axes`` (default :data:`AXES`), ready to splat into
    :func:`build_nsm`."""
    axes = AXES if axes is None else axes
    keys = list(axes)
    for values in itertools.product(*(axes[k] for k in keys)):
        yield dict(zip(keys, values))


def axis_id(combo: dict) -> str:
    """Short stable label for an axis point — for test ids, filenames, table
    rows.  ``{"path": "aux", "guard": None, "eigenvalues": "exclude"}`` →
    ``"aux-noguard-evexclude"``."""
    guard = combo.get("guard") or "noguard"
    return (f"{combo.get('path', 'legacy')}-{guard}"
            f"-ev{combo.get('eigenvalues', 'regularize')}")


def build_nsm(
    model: Union[Any, Callable[[], Any]],
    *,
    path: str = "aux",
    eps: float = DEFAULT_EPS,
    guard: Optional[str] = None,
    eigenvalues: str = "regularize",
    normalize_normal: bool = True,
    riemann=None,
    **nsm_kwargs,
) -> NumericalSystemModel:
    """Build a ready :class:`NumericalSystemModel` at one point of the
    depth-law axes.

    Parameters
    ----------
    model
        A :class:`~zoomy_core.model.derivation.model.Model`, a
        :class:`~zoomy_core.systemmodel.system_model.SystemModel`, or a
        zero-argument callable returning either.  Prefer a Model or a callable:
        a SystemModel is PROMOTED IN PLACE by the NSM, so reusing one instance
        across axis points would carry the first point's operators into the
        second.  A callable is invoked once per call, which makes a sweep loop
        safe by construction.
    path
        ``"direct"`` | ``"aux"`` | ``"legacy"`` | ``"none"`` — see :data:`PATHS`.
    eps
        Regularization scale for ``"direct"`` / ``"aux"``.  NOT ``wet_dry_eps``.
    guard
        ``None`` | ``"power"`` | ``"gate"`` | ``"both"`` — see :data:`GUARDS`.
    eigenvalues
        ``"regularize"`` | ``"exclude"`` — see :data:`EIGENVALUE_TREATMENTS`.
        Ignored by ``path="legacy"`` (which carries its own historical carve-out)
        and by ``path="none"``.
    normalize_normal
        Impose ``|n| = 1`` first.  Default ``True``.
    riemann
        Riemann numerics class, forwarded verbatim; ``None`` keeps the NSM
        default (``NonconservativeRusanov``).
    **nsm_kwargs
        Anything else :meth:`NumericalSystemModel.from_system_model` accepts
        (``reconstruction``, ``diffusion``, ``dt_max``, ``extra_operations``, …).

    Returns
    -------
    NumericalSystemModel
        Already ``derive()``-d: the selected operations have run, so the
        operators, spectrum and aux rows are final and the NSM can be handed
        straight to a printer or a solver.
    """
    if path not in PATHS:
        raise ValueError(
            f"path={path!r} is not a legal regularization path; expected one "
            f"of {list(PATHS)}.")
    if guard not in GUARDS:
        raise ValueError(
            f"guard={guard!r} is not a legal eigenvalue guard; expected one "
            f"of {list(GUARDS)}.")
    if eigenvalues not in EIGENVALUE_TREATMENTS:
        raise ValueError(
            f"eigenvalues={eigenvalues!r} is not a legal spectrum treatment; "
            f"expected one of {list(EIGENVALUE_TREATMENTS)}.")

    # A bare class or a zero-arg factory is invoked here, so a sweep can pass
    # ``lambda: SWE(dimension=2)`` and get a FRESH system per axis point.
    obj = model
    if isinstance(obj, type) or inspect.isroutine(obj) or isinstance(
            obj, functools.partial):
        obj = obj()

    # ``depth_regularization=None`` is how the NSM spells "legacy default".
    axes = dict(
        riemann=riemann,
        depth_regularization=None if path == "legacy" else path,
        regularization_eps=eps,
        eigenvalue_treatment=eigenvalues,
        eigenvalue_guard=guard,
        normalize_normal=normalize_normal,
        **nsm_kwargs,
    )
    if isinstance(obj, SystemModel):
        return NumericalSystemModel.from_system_model(obj, **axes)
    return NumericalSystemModel.from_model(obj, **axes)
