"""One display call that works in every environment a case is run in.

A case is executed in at least three places and they disagree about output:

* a **Jupyter / GUI kernel** — rich output, ``display`` is an IPython builtin;
* **plain** ``python case.py`` — there is no IPython at all, so ``display`` is
  simply undefined and a cell using it dies with ``NameError``;
* a **headless worker** — importable IPython may exist, but no kernel.

On top of that, sympy renders as LaTeX only through its ``_repr_latex_`` hook,
which is not reliably active: equations came out as plain text inside a
notebook.  Routing through ``IPython.display.Math(sympy.latex(...))`` renders
via MathJax unconditionally.

So cases call :func:`show` and stop carrying their own shim::

    from zoomy_core.misc.show import show

    show(model.describe())                      # any object
    show(eq=(sympy.Symbol("h^*"), h_expr))      # a rendered equation

Keeping this here rather than pasted into each case is the same rule the rest
of the framework follows: a case selects behaviour, it does not reimplement it.
"""
from __future__ import annotations

__all__ = ["show", "in_notebook"]


def in_notebook() -> bool:
    """True when a live IPython/Jupyter kernel is driving this process.

    Importable IPython is NOT enough — a headless worker can import it without
    a kernel — so this asks ``get_ipython()`` for an actual instance."""
    try:
        from IPython import get_ipython
    except Exception:                                    # noqa: BLE001
        return False
    try:
        return get_ipython() is not None
    except Exception:                                    # noqa: BLE001
        return False


def show(obj=None, *, eq=None, precision: int = 6):
    """Render ``obj``, or the equation ``eq=(lhs, rhs)``, wherever we are.

    ``eq`` is rendered as real LaTeX in a kernel and pretty-printed otherwise.
    ``precision`` rounds sympy floats for display only — an un-rounded
    manufactured source prints 15-digit rationals and is unreadable.
    """
    if eq is not None:
        import sympy as sp
        lhs, rhs = eq
        try:
            rhs = sp.N(rhs, precision)
        except Exception:                                # noqa: BLE001
            pass
        if in_notebook():
            from IPython.display import Math, display as _d
            _d(Math(f"{sp.latex(lhs)} = {sp.latex(rhs)}"))
        else:
            print(sp.pretty(sp.Eq(lhs, rhs, evaluate=False)))
        return
    if in_notebook():
        from IPython.display import display as _d
        _d(obj)
    else:
        print(obj)
