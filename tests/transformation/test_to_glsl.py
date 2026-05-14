"""Tests for the SymPy → GLSL ES 3.00 code printer (``to_glsl``).

Every generated kernel is compile-validated with ``glslangValidator``
(shipped in the conda env) by wrapping it in a minimal WebGL2 fragment
shader — this catches the GLSL ES 3.00 gotchas the printer exists to
handle (no array return types, no implicit int→float, typed params).
"""

import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest
from sympy import Matrix, sqrt

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.basemodel import Model
from zoomy_core.model.models.advection_model import ScalarAdvection
from zoomy_core.transformation.to_glsl import GlslModel

_GLSLANG = shutil.which("glslangValidator")
requires_glslang = pytest.mark.skipif(
    _GLSLANG is None, reason="glslangValidator not available"
)


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


def _wrap_fragment_shader(glsl_functions: str) -> str:
    """Embed generated kernels in a minimal WebGL2 fragment shader so
    ``glslangValidator`` type-checks them."""
    return textwrap.dedent(
        """\
        #version 300 es
        precision highp float;

        {functions}

        out vec4 fragColor;
        void main() {{
            fragColor = vec4(0.0);
        }}
        """
    ).format(functions=glsl_functions)


def _glslang_validate(glsl_functions: str):
    """Run glslangValidator on a wrapped shader; return (ok, output)."""
    shader = _wrap_fragment_shader(glsl_functions)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "kernel.frag"
        path.write_text(shader)
        proc = subprocess.run(
            [_GLSLANG, str(path)],
            capture_output=True,
            text=True,
        )
    return proc.returncode == 0, proc.stdout + proc.stderr


@pytest.fixture(params=[1, 2], ids=["1d", "2d"])
def swe_glsl(request):
    return GlslModel(MiniSWE(dimension=request.param)).generate()


def test_glsl_structure(swe_glsl):
    """The generated code uses out-parameter, void-returning kernels and
    none of the JS / C++ language artefacts."""
    assert "void flux(" in swe_glsl
    assert "out float res[" in swe_glsl
    assert "Math." not in swe_glsl
    assert "std::" not in swe_glsl
    assert "Float64Array" not in swe_glsl
    # No array return types (GLSL ES 3.00 forbids them).
    assert "float[" not in swe_glsl.replace("float[](", "")  # constructors ok


@requires_glslang
def test_swe_glsl_compiles(swe_glsl):
    """Generated SWE kernels compile as GLSL ES 3.00."""
    ok, output = _glslang_validate(swe_glsl)
    assert ok, f"glslangValidator rejected the generated GLSL:\n{output}"


@requires_glslang
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_advection_glsl_compiles(dim):
    """A second, structurally different model also compiles."""
    glsl = GlslModel(ScalarAdvection(dimension=dim)).generate()
    ok, output = _glslang_validate(glsl)
    assert ok, f"glslangValidator rejected ScalarAdvection GLSL:\n{output}"


@requires_glslang
def test_integer_literals_are_floats(swe_glsl):
    """Sanity: the printer never emits a bare integer literal as a float
    operand (that would make glslangValidator fail) — covered by the
    compile test, but assert the float form is present too."""
    ok, _ = _glslang_validate(swe_glsl)
    assert ok
    # 0.5 * g * h**2 in the flux -> the 0.5 must survive as a float.
    assert "0.5" in swe_glsl or "(1.0 / 2.0)" in swe_glsl
