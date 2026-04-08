"""Zoomy mesh hierarchy: BaseMesh → FVMMesh → LSQMesh."""

from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.mesh.lsq_mesh import LSQMesh


def ensure_lsq_mesh(mesh, model=None, lsq_degree=None) -> LSQMesh:
    """Promote a mesh to LSQMesh if needed.

    Parameters
    ----------
    mesh : BaseMesh, FVMMesh, or LSQMesh
    model : optional model with derivative_specs (auto-detects lsq_degree)
    lsq_degree : explicit LSQ polynomial degree (overrides model detection)
    """
    if isinstance(mesh, LSQMesh):
        return mesh

    # Determine lsq_degree from model derivative specs if not given
    if lsq_degree is None and model is not None:
        specs = getattr(model, "derivative_specs", None)
        if specs:
            spatial_orders = [
                sum(1 for a in spec.axes if a != "t") for spec in specs
            ]
            lsq_degree = max(spatial_orders) if spatial_orders else 1
        else:
            lsq_degree = 1
    elif lsq_degree is None:
        lsq_degree = 1

    if isinstance(mesh, FVMMesh):
        return LSQMesh.from_fvm(mesh, lsq_degree)

    if isinstance(mesh, BaseMesh):
        fvm = FVMMesh.from_base(mesh)
        return LSQMesh.from_fvm(fvm, lsq_degree)

    # Fallback: assume it's the old Mesh class (already has LSQ fields)
    return mesh
