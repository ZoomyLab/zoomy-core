"""The opaque modal basis for the clean-redesign derivation framework.

A :class:`Basis` is the *symbolic* counterpart of a concrete
:class:`~zoomy_core.model.models.basisfunctions.Basisfunction`.  It carries
**no polynomials** — only two opaque ``sympy`` Function heads:

* ``basis.phi`` — the trial/test family ``φ(k, ζ)`` (the index ``k`` is the
  first argument, the reference coord ``ζ`` the second).  This is the SAME
  single-class, ``(index, ζ)``-arity design the production
  :class:`Basisfunction` uses for its ``phi_fn`` — so an unexpanded modal
  ansatz reads ``Σ_k a(k, …)·φ(k, ζ)`` and a later concrete
  :class:`~zoomy_core.model.derivation.operations.ResolveBasis` can swap in a real
  polynomial basis.
* ``basis.weight`` — the test weight ``c(ζ)`` (a one-argument Function head).

The two heads are *opaque*: they participate in ``free_symbols`` /
``xreplace`` / ``Derivative`` like any other atom, and ``ExtractBrackets``
recognises ``φ`` / ``c`` products structurally to name the Galerkin brackets.

Usage::

    basis = Basis(symbol="phi", weight="c")
    zeta = coords.zeta
    k = sp.Symbol("k", integer=True, nonnegative=True)
    integrand = basis.weight(zeta) * basis.phi(k, zeta)   # c(ζ)·φ(k, ζ)
"""

from __future__ import annotations

import sympy as sp


__all__ = ["Basis"]


class Basis:
    """An opaque modal basis: a φ-family head + a weight head, no polynomials.

    Parameters
    ----------
    symbol : str
        Name of the opaque trial/test Function family (the class name in
        ``sympy``, e.g. ``"phi"``).  Two-argument ``(index, ζ)`` arity, so
        ``basis.phi(k, ζ)`` reads ``φ_k(ζ)``.
    weight : str
        Name of the one-argument weight Function ``c(ζ)`` (e.g. ``"c"``).
    """

    def __init__(self, symbol="phi", weight="c"):
        self.symbol = symbol
        self.weight_name = weight
        # ONE opaque Function class per family — exactly the production
        # ``Basisfunction.phi_fn`` shape (``(index, ζ)`` arity).  The
        # single-class design is what makes ``Σ a(k,…)·φ(k, ζ)`` a clean
        # unexpanded ``sp.Sum`` and lets ``ExtractBrackets`` match by head.
        #
        # ``_is_basis_head`` marks these as basis machinery so the model's
        # field collector never mistakes ``φ`` / ``c`` for an unknown.
        self.phi = type(symbol, (sp.Function,),
                        {"nargs": 2, "_is_basis_head": True, "_basis": self})
        self.weight = type(weight, (sp.Function,),
                           {"nargs": 1, "_is_basis_head": True, "_basis": self})

    # ── compatibility shim for the bracket/resolve machinery ───────────
    @property
    def phi_fn(self):
        """Alias for :attr:`phi` — the opaque φ-family head.

        ``ExtractBrackets`` / ``ResolveBasis`` read ``basis.phi_fn`` so the
        opaque :class:`Basis` and a production
        :class:`~zoomy_core.model.models.basisfunctions.Basisfunction`
        present the same attribute name."""
        return self.phi

    def __repr__(self):
        return f"Basis(symbol={self.symbol!r}, weight={self.weight_name!r})"
