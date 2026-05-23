"""Zoomy mesh hierarchy: BaseMesh → FVMMesh → LSQMesh."""

from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.mesh.lsq_mesh import LSQMesh


def ensure_lsq_mesh(mesh, model) -> LSQMesh:
    """Promote ``mesh`` to an LSQMesh sized for ``model``.

    The LSQ polynomial degree is **always** derived from the model —
    user code does not set it.  ``model`` may be a :class:`Model`, a
    :class:`SystemModel`, or a :class:`NumericalSystemModel`; in every
    case the resolution path is the same:

    1. The model is promoted to an NSM (if not already one).
    2. ``nsm.resolved_lsq_degree()`` returns the max spatial-derivative
       order required across ``sm.aux_registry``,
       ``source_derivative_specs``, and any ``additional_systems``.
    3. The mesh's cached LSQ stencil is built (or rebuilt) at that
       degree.

    Passing an already-built :class:`LSQMesh` whose cached stencil is
    of lower degree triggers a rebuild in place.

    Parameters
    ----------
    mesh : BaseMesh, FVMMesh, or LSQMesh
    model : Model, SystemModel, or NumericalSystemModel
        Required.  The degree is taken from this object — there is no
        ``lsq_degree`` kwarg.
    """
    if model is None:
        raise TypeError(
            "ensure_lsq_mesh now requires a model argument — the LSQ "
            "polynomial degree is derived from the model's "
            "NumericalSystemModel.  Pass the Model / SystemModel / NSM "
            "that the solver will consume."
        )

    # Resolve the required degree once, via the NSM.  Importing NSM
    # at module level would create a mesh → numerics cycle.
    from zoomy_core.numerics import NumericalSystemModel
    if isinstance(model, NumericalSystemModel):
        nsm = model
    else:
        nsm = NumericalSystemModel.from_system_model(model)
    required_degree = nsm.resolved_lsq_degree()

    if isinstance(mesh, LSQMesh):
        if mesh._current_lsq_degree() < required_degree:
            mesh._build_lsq_stencil(required_degree)
        return mesh

    if isinstance(mesh, FVMMesh):
        lsq = LSQMesh.from_fvm(mesh)
    elif isinstance(mesh, BaseMesh):
        lsq = LSQMesh.from_fvm(FVMMesh.from_base(mesh))
    else:
        # Fallback: old Mesh class without an LSQ hierarchy — return
        # untouched.  Solvers that hit this raise downstream.
        return mesh

    if required_degree > 1:
        lsq._build_lsq_stencil(required_degree)
    return lsq
