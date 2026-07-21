"""Model-DERIVED wall boundary definitions.

A free-slip wall is a GEOMETRIC statement about the momentum VECTOR, not
about a state component: the ghost mirrors the momentum about the face,

    ``q_ghost = q − 2 n (n·q)``

so the NORMAL component flips and the TANGENTIAL component is preserved
exactly.  Written in the canonical face-normal symbols ``n_d``
(:func:`zoomy_core.systemmodel.system_model.face_normal_symbols`), which
:class:`~zoomy_core.model.boundary_conditions.FromModel` substitutes per face
— so ONE registration serves every wall orientation and every mesh.

The normal-BLIND alternative (``q_d → −q_d`` in every direction) is a full
reversal: it is right only when the flow is exactly normal to the face, and
destroys the tangential momentum of any flow ALONG the wall.
"""
from __future__ import annotations

from zoomy_core.systemmodel.system_model import face_normal_symbols

__all__ = ["free_slip_wall_rows", "register_free_slip_wall"]


def free_slip_wall_rows(vectors):
    """``{state field: ghost value}`` for a free-slip wall.

    ``vectors`` is an iterable of momentum VECTORS — each a sequence of the
    per-direction fields that form one physical vector (one entry per mesh
    direction, in the model's direction order).  Every moment / layer / mode
    is its own vector: the reflection acts direction-wise, mode by mode.
    """
    rows = {}
    for vec in vectors:
        vec = list(vec)
        nrm = face_normal_symbols(len(vec))
        qn = sum(nd * qd for nd, qd in zip(nrm, vec))
        for nd, qd in zip(nrm, vec):
            rows[qd] = qd - 2 * nd * qn
    return rows


def register_free_slip_wall(m, vectors, *, name="wall"):
    """Register :func:`free_slip_wall_rows` as ``boundary:<name>`` on the
    derivation model ``m``.  ``h`` and ``b`` are left unprescribed — they
    extrapolate, which is what a wall does to a scalar."""
    for field, value in free_slip_wall_rows(vectors).items():
        m.register_group(f"boundary:{name}", field, value)
    return m
