"""Clean-redesign derivation framework.

The derivation spine: build a symbolic PDE derivation as a computational graph
of :class:`Model` operations.

    from zoomy_core import coords
    import zoomy_core.derivatives as d
    from zoomy_core.model.derivation import Model, Substitution, ChangeOfVariables

    t, x, z = coords.t, coords.x, coords.z
    model = Model(coords=(t, x, z), parameters={"g": 9.81, "rho": 1.0})
    model.Q = [h, u, w, p]
    model.add_equation("mass", d.x(u) + d.z(w))

``Model`` and the operations reuse the existing
:class:`zoomy_core.model.operations.Operation` /
:class:`zoomy_core.model.equation.Equation` spine — the redesign is the
model/unknown contract (declared-and-present ``Q``, derived ``Qaux``, ops that
carry their own unknown bookkeeping), not a fresh operator surface.
"""

from .model import (
    Model, VectorEquation, MomentFamily, resolve_modes, ResolveModes,
)
from .operations import (
    SolveFor,
    SolveLinearSystem,
    ChangeOfVariables,
    Granularity,
    granularity_of,
)
from .transformations import PDETransformation
from .basis import Basis
from .modal import (
    ModalIndexRegistry,
    reset_modal_indices,
    separation_of_variables,
    SeparationOfVariables,
    build_modal_sum,
    modal_bound,
    modal_index,
    test_index,
)
from .projection import (
    Gram,
    Weight,
    is_bracket,
    is_bracket_body,
    bracket_atoms,
    ExpandSums,
    EvaluateSums,
    Integrate,
    Project,
    PullConstants,
    ExtractBrackets,
    ResolveBasis,
)
from .closure import (
    Resolve,
    ResolveIntegral,
    GaussQuadrature,
    DeferQuadrature,
    ResolveNumQuad,
    InvertMassMatrix,
    FoldConservative,
    Split,
    Simplify,
    Consolidate,
    AutoTag,
    SortByTag,
    Sort,
    TAG_ORDER,
    fold_to_conservative_form,
    is_conservative_diffusion,
    project_conservative_diffusion,
)
from .system_extract import extract_system_operators

__all__ = [
    "Model",
    "VectorEquation",
    "MomentFamily",
    "resolve_modes",
    "ResolveModes",
    "SolveFor",
    "SolveLinearSystem",
    "ChangeOfVariables",
    "Granularity",
    "granularity_of",
    "PDETransformation",
    "Basis",
    "ModalIndexRegistry",
    "reset_modal_indices",
    "separation_of_variables",
    "SeparationOfVariables",
    "build_modal_sum",
    "modal_bound",
    "modal_index",
    "test_index",
    "Gram",
    "Weight",
    "is_bracket",
    "is_bracket_body",
    "bracket_atoms",
    "ExpandSums",
    "EvaluateSums",
    "Integrate",
    "Project",
    "PullConstants",
    "ExtractBrackets",
    "ResolveBasis",
    "Resolve",
    "ResolveIntegral",
    "GaussQuadrature",
    "DeferQuadrature",
    "ResolveNumQuad",
    "InvertMassMatrix",
    "FoldConservative",
    "Split",
    "Simplify",
    "Consolidate",
    "AutoTag",
    "SortByTag",
    "Sort",
    "TAG_ORDER",
    "fold_to_conservative_form",
    "is_conservative_diffusion",
    "project_conservative_diffusion",
    "extract_system_operators",
]
