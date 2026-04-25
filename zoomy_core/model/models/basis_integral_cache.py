"""Query-driven integral cache over a basis.

The pipeline hands an ``Integral(integrand, (var, lower, upper))`` — where
``integrand`` contains opaque basis functions ``phi_k(arg)`` and their
derivatives — to :class:`BasisIntegralCache`.  The cache either returns a
stored result or falls back to a fast, polynomial-only integration.  No
``sympy.integrate`` on the symbolic basis definitions: the opaque
``phi_k`` is first concretized against the basis polynomial (in the
basis's own ``z`` symbol), substituted at ``arg``, then integrated
via sympy's polynomial path (``expand`` + ``Poly.integrate``).

Cache key: the tuple ``(integrand, var, lower, upper)``.  sympy
expressions hash structurally — commutative ``Mul``/``Add`` arguments
collide on rearrangement — so ``phi_0(ζ)·phi_1(ζ)`` and
``phi_1(ζ)·phi_0(ζ)`` share an entry.

The cache is **per basis instance** and lives in-memory.  A new query
shape is computed once and reused.  No precomputed M/A/B/D arrays: this
module doesn't know about "matrix tags" — it's a pure functional cache
that maps a symbolic integrand to its symbolic value.
"""

import sympy as sp


_PHI_PREFIX = "phi_"


def _is_phi_call(e):
    """Return ``True`` iff ``e`` is ``phi_k(arg)``."""
    return (
        isinstance(e, sp.Function)
        and not isinstance(e, sp.Derivative)
        and getattr(e.func, "__name__", "").startswith(_PHI_PREFIX)
    )


def _phi_index(e):
    """Return ``k`` given ``phi_k(...)`` or its derivative."""
    name = e.func.__name__ if _is_phi_call(e) else e.args[0].func.__name__
    return int(name[len(_PHI_PREFIX):])


class BasisIntegralCache:
    """Memoizing symbolic integrator over a basis.

    Parameters
    ----------
    basis : :class:`~zoomy_core.model.models.basisfunctions.Basisfunction`
        The basis whose polynomial definitions back the opaque
        ``phi_k(arg)`` and ``Derivative(phi_k(arg), arg)`` functions
        used in the caller's expression tree.

    Notes
    -----
    The caller is expected to have factored out all ``var``-independent
    factors before querying.  The integrand passed in should be a
    product of ``phi_k(arg)``, ``Derivative(phi_k(arg), arg)``,
    polynomial factors in ``var``, and possibly a nested
    ``Integral(kernel, (var', ..., var))`` whose inner integrand is
    itself basis-only.  In particular, the cache will happily
    concretize every ``phi_k`` it sees — if a ``phi_k`` slips through
    with a ``var``-independent argument, that's a no-op (the poly is
    evaluated at a number / constant, which is itself a constant).
    """

    def __init__(self, basis):
        self.basis = basis
        self._cache = {}
        # The basis stores its polynomials in a conventional "z" symbol.
        # We need to know which symbol to substitute away when concretizing.
        self._z_basis = sp.Symbol("z")
        self._phi_polys = [basis.get(k) for k in range(basis.level + 1)]
        self._dphi_polys = [
            sp.diff(p, self._z_basis) for p in self._phi_polys
        ]

    # ------------------------------------------------------------------
    # Public API

    def integrate(self, integrand, var, lower, upper):
        """Return the symbolic value of ``∫_lower^upper integrand d(var)``.

        The integrand may contain opaque ``phi_k(arg)`` and
        ``Derivative(phi_k(arg), arg)`` nodes — these are substituted
        with the basis's polynomial before integration.  All other
        factors are carried through as symbolic coefficients.

        The ``(integrand, var, lower, upper)`` tuple is the cache key.
        Hits return instantly; misses invoke the polynomial integrator
        and store the result.
        """
        key = (integrand, var, lower, upper)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._compute(integrand, var, lower, upper)
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Internals

    def _compute(self, integrand, var, lower, upper):
        concrete = self._concretize(integrand)
        # Flatten any nested Integrals whose integrand is now a
        # polynomial (after concretization, any ``phi_k(arg)`` → poly;
        # inner running integrals like ``∫_0^ζ poly(ζ') dζ'`` evaluate
        # via ``doit`` to a polynomial in ζ).
        if concrete.has(sp.Integral):
            concrete = concrete.doit()
        # Polynomial integration — fast path.
        expanded = sp.expand(concrete)
        try:
            poly = sp.Poly(expanded, var)
            anti = poly.integrate()
            value = anti.as_expr().subs(var, upper) - anti.as_expr().subs(var, lower)
        except (sp.GeneratorsNeeded, sp.PolynomialError):
            # Non-polynomial in var after concretize — should be rare;
            # fall back to direct antiderivative (still no opaque phi).
            anti = sp.integrate(expanded, var)
            value = anti.subs(var, upper) - anti.subs(var, lower)
        return sp.simplify(value)

    def _concretize(self, expr):
        """Substitute every ``phi_k(arg)`` / ``Derivative(phi_k(arg), arg)``
        with the corresponding basis polynomial (evaluated at ``arg``)."""
        def _rec(e):
            # Derivative(phi_k(arg), arg) → dphi_k/dz|_{z=arg}
            if isinstance(e, sp.Derivative):
                inner = e.args[0]
                if _is_phi_call(inner):
                    # Only handle the single-variable first-order case
                    # ``Derivative(phi_k(arg), arg)``.  Higher-order or
                    # mixed partials shouldn't arise here; fall through
                    # to generic recursion if they do.
                    wrts = e.args[1:]
                    if (len(wrts) == 1
                            and (wrts[0] == inner.args[0]
                                 or (isinstance(wrts[0], tuple)
                                     and wrts[0][0] == inner.args[0]
                                     and wrts[0][1] == 1))):
                        k = _phi_index(inner)
                        arg = inner.args[0]
                        return self._dphi_polys[k].subs(self._z_basis, arg)
                # Fall through for other derivatives — recurse into args.
                new_args = tuple(_rec(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
                return e
            if _is_phi_call(e):
                k = _phi_index(e)
                arg = e.args[0]
                return self._phi_polys[k].subs(self._z_basis, arg)
            if e.args:
                new_args = tuple(_rec(a) for a in e.args)
                if any(n is not o for n, o in zip(new_args, e.args)):
                    return e.func(*new_args)
            return e
        return _rec(expr)
