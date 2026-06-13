"""Opaque boundary frame {n, t_α} + small_slope_scaling — Stage 1.

A boundary stress closure prescribes the traction in the local frame, built
from OPAQUE slope symbols (equations.frame_slope) rather than the physical
∂_d(interface).  small_slope_scaling resolves the frame to its n→ẑ limit by
zeroing those symbols — and must NOT touch the physical bed slope ∂_x b in the
body force (the symbol-clash that would otherwise make the closure order-
dependent).
"""
from types import SimpleNamespace

import sympy as sp

from zoomy_core import coords as C
import zoomy_core.derivatives as d
from zoomy_core.model.derivation import Model as DModel
from zoomy_core.model.models.material import ClosureState
from zoomy_core.model.models.equations import frame_slope, small_slope_scaling


class _F:                       # FieldHandle shim: bulk == trace == the symbol
    def __init__(self, e): self.e = e
    @property
    def expr(self): return self.e
    def at(self, v): return self.e


def test_frame_is_opaque_and_reduces_to_zhat_under_small_slope():
    us, ws = sp.symbols("u w", real=True)
    s = ClosureState(SimpleNamespace(u=_F(us), w=_F(ws)),
                     h=sp.Symbol("h"), x=C.x, zeta=sp.Symbol("zeta"),
                     at=0, horiz=[C.x], boundary_tag="b")
    sx = frame_slope("b", "x")
    # exact opaque frame
    assert s.normal == sp.Matrix([-sx, 1]) / sp.sqrt(1 + sx ** 2)
    assert s.tangents[0] == sp.Matrix([1, sx]) / sp.sqrt(1 + sx ** 2)
    # small-slope limit σ→0: n→ẑ, t→ê_x, u·t→u, u·n→w
    assert s.normal.subs(sx, 0) == sp.Matrix([0, 1])
    assert s.tangents[0].subs(sx, 0) == sp.Matrix([1, 0])
    assert sp.simplify(s.u_tangent[0].subs(sx, 0) - us) == 0
    assert sp.simplify(s.u_normal.subs(sx, 0) - ws) == 0


def test_frame_2d_axis_fixed_tangents():
    us, vs, ws = sp.symbols("u v w", real=True)
    s = ClosureState(SimpleNamespace(u=_F(us), v=_F(vs), w=_F(ws)),
                     h=sp.Symbol("h"), x=C.x, zeta=sp.Symbol("zeta"),
                     at=1, horiz=[C.x, C.y], boundary_tag="eta")
    sx, sy = frame_slope("eta", "x"), frame_slope("eta", "y")
    assert s.normal == sp.Matrix([-sx, -sy, 1]) / sp.sqrt(1 + sx ** 2 + sy ** 2)
    # axis-fixed: t_x in (x,z) plane, t_y in (y,z) plane
    assert s.tangents[0] == sp.Matrix([1, 0, sx]) / sp.sqrt(1 + sx ** 2)
    assert s.tangents[1] == sp.Matrix([0, 1, sy]) / sp.sqrt(1 + sy ** 2)
    # small-slope: slip components reduce to the bare horizontal velocities
    assert sp.simplify(s.u_tangent[0].subs({sx: 0, sy: 0}) - us) == 0
    assert sp.simplify(s.u_tangent[1].subs({sx: 0, sy: 0}) - vs) == 0


def test_small_slope_scaling_zeros_frame_but_keeps_topography():
    t, x, z = C.t, C.x, C.z
    m = DModel(coords=(t, x, z), parameters={"g": 9.81})
    b = sp.Function("b", real=True)(t, x)
    h = sp.Function("h", positive=True)(t, x)
    sx = frame_slope("eta", "x")
    m.declare_state(h)
    # a frame-slope term (geometry) sitting next to the topographic source
    # g h ∂_x b (physics) — same ∂_x b would alias if we'd used it for n.
    m.add_equation("test", d.t(h) + m.parameters.g * h * d.x(b) + sx * h)
    small_slope_scaling(m)
    expr = m._equations["test"].expr
    assert not any(str(s_).startswith("frameslope_") for s_ in expr.free_symbols), \
        "frame slope must resolve to 0"
    assert expr.has(sp.Derivative(b, x)), "topographic ∂_x b must survive"
