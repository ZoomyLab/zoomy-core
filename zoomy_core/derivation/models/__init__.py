"""Declarative-framework reference models.

These build a :class:`~zoomy_core.derivation.model.Model` end-to-end through the
clean-redesign op surface (``PDETransformation`` → ``separation_of_variables``
→ concrete-level ``Resolve`` → kinematic-BC modal closure) to a target
analytical form, ready for ``SystemModel.from_model(model, Q, Qaux)``.

:class:`SME` / :func:`build_sme` reproduce the Shallow Moment Equations of
Kowalski & Torrilhon (2019), eq. (4.17), with the constitutive viscous stress
left OPEN.  :class:`SlipSME` INHERITS :class:`SME` and inserts the slip-Newton
shear-stress closure on top of the inherited open pipeline.
:class:`NewtonianSME` INHERITS :class:`SlipSME` and additionally KEEPS the
horizontal normal stress ``τ_xx = 2 μ ∂_x u`` (which the base SME drops),
closing it as a DIFFUSIVE flux that lifts into the SystemModel's rank-4
``diffusion_matrix``.
"""

from .sme_kt19 import SME, SlipSME, NewtonianSME, build_sme

__all__ = ["SME", "SlipSME", "NewtonianSME", "build_sme"]
