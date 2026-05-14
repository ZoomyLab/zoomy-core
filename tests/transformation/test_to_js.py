"""Tests for the SymPy → JavaScript code printer (``to_js``).

Generated kernels are syntax-checked with ``node --check`` and, where a
numeric reference is available, *executed* in node and compared against
the numpy runtime — so the JS printer is verified for correctness, not
just for parseability.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sympy import Matrix, sqrt

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.basemodel import Model
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions,
    Extrapolation,
    Wall,
)
from zoomy_core.fvm.riemann_solvers import HLL, HLLC
from zoomy_core.transformation.to_js import JsModel, JsNumerics
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel

_NODE = shutil.which("node")
requires_node = pytest.mark.skipif(_NODE is None, reason="node not available")


class MiniSWE(Model):
    """Shallow Water Equations, [h, hu, hv(if 2D)], gravity ``g``."""

    def __init__(self, dimension=2, **kwargs):
        var_names = ["h", "hu", "hv"][: dimension + 1]
        super().__init__(
            dimension=dimension,
            variables=var_names,
            parameters={"g": (9.81, "positive")},
            eigenvalue_mode="symbolic",
            **kwargs,
        )

    def flux(self):
        dim = self.dimension
        h = self.variables[0]
        hu = Matrix(self.variables[1:])
        u = hu / h
        g = self._parameter_symbols.g
        F = Matrix.zeros(self.n_variables, dim)
        F[0, :] = hu.T
        F[1:, :] = h * u * u.T + 0.5 * g * h**2 * Matrix.eye(dim)
        return ZArray(F)

    def eigenvalues(self):
        dim = self.dimension
        h = self.variables[0]
        hu = Matrix(self.variables[1:])
        n = Matrix(self.normal[:dim])
        g = self._parameter_symbols.g
        un = (hu.T * n)[0] / h
        c = sqrt(g * h)
        return ZArray([un - c, un + c] + [un] * (dim - 1))


def _swe_with_bcs(dim):
    momentum = [[1, 2]] if dim == 2 else [[1]]
    bcs = BoundaryConditions(
        [
            Wall(tag="left", momentum_field_indices=momentum),
            Extrapolation(tag="right"),
        ]
    )
    return MiniSWE(dimension=dim, boundary_conditions=bcs)


def _node_check(js: str):
    """Return (ok, output) for ``node --check`` on a JS source string."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "kernel.js"
        path.write_text(js)
        proc = subprocess.run(
            [_NODE, "--check", str(path)], capture_output=True, text=True
        )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _node_call(js: str, func: str, args: list, res_size: int) -> np.ndarray:
    """Define ``js``, call the out-parameter kernel ``func`` in node, and
    return the result it wrote into the caller-owned ``res`` array."""
    driver = (
        f"\nconst __args = JSON.parse(process.argv[2]);"
        f"\nconst __res = new Float64Array({res_size});"
        f"\n{func}(...__args, __res);"
        f"\nconsole.log(JSON.stringify(Array.from(__res)));\n"
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "kernel.js"
        path.write_text(js + driver)
        proc = subprocess.run(
            [_NODE, str(path), json.dumps(args)],
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"node failed:\n{proc.stdout}\n{proc.stderr}")
    return np.array(json.loads(proc.stdout.strip()), dtype=float)


# ── structural / syntax ──────────────────────────────────────────────


@pytest.mark.parametrize("dim", [1, 2])
def test_js_model_syntax(dim):
    js = JsModel(MiniSWE(dimension=dim)).generate()
    assert "function flux(" in js
    if _NODE:
        ok, output = _node_check(js)
        assert ok, output


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("scheme", [HLL, HLLC])
def test_js_numerics_syntax(dim, scheme):
    js = JsNumerics(scheme(_swe_with_bcs(dim))).generate()
    assert "function numerical_flux(" in js
    if _NODE:
        ok, output = _node_check(js)
        assert ok, output


@pytest.mark.parametrize("dim", [1, 2])
def test_js_boundary_conditions_syntax(dim):
    js = JsModel(_swe_with_bcs(dim)).generate_boundary_conditions()
    assert "function boundary_conditions(" in js
    assert "bc_idx" in js
    if _NODE:
        ok, output = _node_check(js)
        assert ok, output


# ── numerical correctness vs the numpy runtime ───────────────────────


@requires_node
@pytest.mark.parametrize("dim", [1, 2])
def test_js_model_flux_matches_numpy(dim):
    """Generated JS ``flux`` matches the numpy runtime."""
    m = MiniSWE(dimension=dim)
    js = JsModel(m).generate()
    mrt = NumpyRuntimeModel(m)
    p = np.array(list(m.parameters.values()), dtype=float)
    aux = np.zeros(m.n_aux_variables)
    q = np.array([1.7, 0.6, -0.25][: dim + 1])
    expect = np.asarray(mrt.flux(q, aux, p), dtype=float).ravel()
    got = _node_call(
        js, "flux", [q.tolist(), aux.tolist(), p.tolist()], expect.size
    )
    np.testing.assert_allclose(got, expect, rtol=1e-10, atol=1e-12)


@requires_node
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("scheme", [HLL, HLLC])
def test_js_numerical_flux_matches_numpy(dim, scheme):
    """Generated JS ``numerical_flux`` matches the numpy runtime of the
    same symbolic Numerics object."""
    m = MiniSWE(dimension=dim)
    numerics = scheme(m)
    js = JsNumerics(numerics).generate()
    rt = numerics.to_runtime_numpy()

    p = np.array(list(m.parameters.values()), dtype=float)
    aux = np.zeros(m.n_aux_variables)
    n = np.array([1.0] + [0.0] * (dim - 1))
    qL = np.array([2.0, 0.8, 0.3][: dim + 1])
    qR = np.array([1.1, 1.2, -0.4][: dim + 1])

    expect = np.asarray(
        rt.numerical_flux(qL, qR, aux, aux, p, n), dtype=float
    ).ravel()
    got = _node_call(
        js,
        "numerical_flux",
        [qL.tolist(), qR.tolist(), aux.tolist(), aux.tolist(),
         p.tolist(), n.tolist()],
        expect.size,
    )
    np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-9)
