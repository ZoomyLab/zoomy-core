"""SME — Galerkin polynomial Shallow Moment Equations.

Inherits from :class:`Hydrostatic` and adds the polynomial Galerkin
projection at the requested ``level`` (number of u-moments retained).

Concretely, applies (in order):
  - ``Multiply(basis.phi, outer=True)`` on momentum.x (and y if 3D):
    creates ``test_0..test_L`` clones of each momentum equation.
  - ``ZetaTransform``: affine change ``z → ζ·h + b`` inside Integrals.
  - ``EvaluateIntegrals``: ``sympy.integrate`` the now-polynomial-in-ζ
    integrals.
  - ``ProjectBasisIntegrals``: extract the polynomial coefficients
    onto field amplitudes.

After this step the system has only polynomial expressions in
``(h, u_0, …, u_L)`` (and ``v_0..v_L`` for 3D), no held Integrals.
"""
from __future__ import annotations

from zoomy_core.model.models.basisfunctions import Legendre_shifted
from zoomy_core.model.models.ins_generator import (
    Multiply,
    ZetaTransform,
    EvaluateIntegrals,
    ProjectBasisIntegrals,
)
from .hydrostatic import Hydrostatic


class SME(Hydrostatic):
    step_description = (
        "Galerkin polynomial projection at order ``level``: multiply each "
        "momentum equation by the test functions φ_0..φ_L (shifted "
        "Legendre by default), affine ζ-transform, integrate, project."
    )

    def __init__(self, *, level: int = 0, basis_type=Legendre_shifted, **kwargs):
        super().__init__(**kwargs)
        self._init_kwargs = {**self._init_kwargs,
                             "level": level,
                             "basis_type": basis_type.__name__}
        cached = SME._cache_get(self._init_kwargs)
        if cached is not None:
            self._adopt_cached(cached)
            return
        self._derive_step(level=level, basis_type=basis_type)
        SME._cache_put(self._init_kwargs, self._snapshot_for_cache())

    def _derive_step(self, *, level: int, basis_type):
        s = self.state
        sys_ = self._system
        self.level = level
        self.basis = basis_type(level=level)

        # Multiply each momentum component by basis test functions.
        for axis in ("x", "y"):
            try:
                child = getattr(sys_.momentum, axis)
            except AttributeError:
                continue
            child.apply(Multiply(self.basis.phi, outer=True))

        # Affine map and integrate.
        sys_.apply(ZetaTransform(s))
        sys_.apply(EvaluateIntegrals(s))
        sys_.apply(ProjectBasisIntegrals(s, basis=self.basis))
