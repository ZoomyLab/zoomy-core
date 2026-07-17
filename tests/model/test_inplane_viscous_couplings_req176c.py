"""REQ-176(4) CORRECTION — in-plane viscous h-column of the SME diffusion matrix.

The retained incompressible-Newtonian in-plane deviatoric stress
``τ_de = ρν(∂_d u_e + ∂_e u_d)`` σ-maps through the (x,z)→(x,ζ) coordinate
transform and, after the moment projection, contributes to EVERY entry of the
momentum diffusion row — not only the mode-diagonal ``A[q_k←q_k] = −2ν`` but
also an OFF-DIAGONAL h-column ``A[q_k←h]`` produced by the ζ-metric ``∂_x ζ``
couplings (``ζ = (z−b)/h`` ⇒ ``∂_x|_z`` picks up ``−(ζ ∂_x h + ∂_x b)/h ∂_ζ``).

The transform PRODUCES these couplings correctly; the pre-correction extraction
then mis-routed the foreign-state (``∂_x h``) piece to the conservative flux (a
frozen gradient-aux), leaving ``A[q_k←h] ≡ 0`` (gui, sympy-verified).  The
correction tags the viscous diffusive flux with the
:class:`~zoomy_core.model.derivation.system_extract.ViscousDiffusion` marker so
the extractor routes the WHOLE rank-4 tensor — including the foreign-state cross
pieces — into ``diffusion_matrix``.

First-principles (linearised about a uniform film, k-th moment in-plane
contribution):

    I_k = −k²[ 2ν q̂_k − 2ν ĥ (ᾱ_k + (2k+1)(Gᾱ)_k) ],
    G_kj = ∫₀¹ ζ φ_j'(ζ) φ_k(ζ) dζ        (shifted Legendre),

so the momentum-residual bracket h-coefficient is
``V[q_k←h] = −2ν(ᾱ_k + (2k+1)(Gᾱ)_k)``; for ``N=2`` (ᾱ_i ≡ q_i/h):

    k=0 : −2ν(ᾱ₀+ᾱ₁+ᾱ₂)   (= −2ν·u|_surface, since φ_i(1)=1)
    k=1 : −2ν(2ᾱ₁+3ᾱ₂)
    k=2 : −6ν ᾱ₂.

The DIFFUSION-MATRIX entry uses the opposite sign of that bracket (the same
convention as the mode-diagonal ``A[q_k←q_k] = −2ν`` vs. the ``+2ν q̂_k``
bracket term), i.e. ``A[q_k←h] = −V[q_k←h]``.
"""
import sympy as sp
import pytest

from zoomy_core.systemmodel import SystemModel
from zoomy_core.model.models.sme import SME
from zoomy_core.model.models.vam import VAM
from zoomy_core.model.models.ml_sme import MLSME
from zoomy_core.model.models.ml_vam import MLVAM
from zoomy_core.model.models.closures import (
    NewtonianInPlane, Newtonian, NavierSlip, StressFree, MeanInterface)
from zoomy_core.model.derivation.basisfunctions import Legendre_shifted
from zoomy_core.model.derivation.system_extract import ViscousDiffusion


def _sme2_retain():
    return SystemModel.from_model(SME(
        level=2, dimension=2,
        closures=[NewtonianInPlane(), Newtonian(), NavierSlip(), StressFree()]))


def _hcolumn(sm):
    """The ``∂²h`` diffusion column of each momentum row: ``A[q_k, h, 0, 0]``."""
    st = [str(s) for s in sm.state]
    sym = {str(s): s for s in sm.state}
    A = sm.diffusion_matrix
    assert A is not None
    hi = st.index("h")
    out = {}
    for i, name in enumerate(st):
        if name.startswith("q"):
            out[name] = sp.simplify(sp.sympify(A[i, hi, 0, 0]))
    return out, sym, A


def test_h_column_is_nonzero_and_matches_closed_forms():
    """SME(2)+NewtonianInPlane: the diffusion h-column ``A[q_k←h]`` is NON-zero
    and equals the hand-derived closed forms (``= −V[q_k←h]`` with ``ᾱ_i =
    q_i/h``)."""
    sm = _sme2_retain()
    hcol, sym, _ = _hcolumn(sm)
    nu = sm.parameters.nu
    h = sym["h"]
    q = [sym[f"q_{k}"] for k in range(3)]
    a = [q[k] / h for k in range(3)]                       # ᾱ_k = q_k/h
    # A[q_k←h] = −V[q_k←h] with the diffusion-matrix sign convention.
    expected = {
        "q_0": 2 * nu * (a[0] + a[1] + a[2]),
        "q_1": 2 * nu * (2 * a[1] + 3 * a[2]),
        "q_2": 6 * nu * a[2],
    }
    for name, want in expected.items():
        assert hcol[name] != 0, f"{name}: h-column is identically zero (the bug)"
        assert sp.simplify(hcol[name] - want) == 0, (
            f"A[{name}←h] = {hcol[name]}, expected {sp.simplify(want)}")


def test_h_column_matches_G_matrix_first_principles():
    """Independent of the hand-written closed forms: build ``G_kj = ∫₀¹ ζ φ_j'
    φ_k dζ`` from the shifted-Legendre basis, form ``V[q_k←h] = −2ν(ᾱ_k +
    (2k+1)(Gᾱ)_k)``, and assert the extracted diffusion column satisfies
    ``A[q_k←h] + V[q_k←h] = 0``.  Ties the code's extraction back to the
    first-principles derivation."""
    sm = _sme2_retain()
    hcol, sym, _ = _hcolumn(sm)
    nu = sm.parameters.nu
    h = sym["h"]
    a = [sym[f"q_{k}"] / h for k in range(3)]
    zeta = sp.Symbol("zeta", real=True)
    N = 2
    leg = Legendre_shifted(level=N + 2)
    phi = [sp.expand(leg.eval(i, zeta)) for i in range(N + 1)]   # concrete polys
    # φ are ζ-POLYNOMIALS with NO Derivative coefficients → direct ζ-integration
    # is exact (the "never integrate polys with Derivative coefficients" rule
    # applies to symbolic modal coefficients, not to these closed basis polys).
    G = sp.Matrix(N + 1, N + 1, lambda k, j: sp.integrate(
        zeta * sp.diff(phi[j], zeta) * phi[k], (zeta, 0, 1)))
    for k in range(N + 1):
        Ga_k = sum(G[k, j] * a[j] for j in range(N + 1))
        V = -2 * nu * (a[k] + (2 * k + 1) * Ga_k)
        assert sp.simplify(hcol[f"q_{k}"] + V) == 0, (
            f"k={k}: A[q_{k}←h]={hcol[f'q_{k}']} ≠ −V={sp.simplify(-V)}")


def test_h_row0_is_minus_2nu_surface_velocity():
    """The k=0 h-column equals ``−V = 2ν·u|_surface`` — the surface trace
    ``Σ_i ᾱ_i`` (shifted-Legendre ``φ_i(1)=1``) — while the mode-diagonal is the
    canonical ``A[q_k←q_k] = −2ν``."""
    sm = _sme2_retain()
    hcol, sym, A = _hcolumn(sm)
    st = [str(s) for s in sm.state]
    nu = sm.parameters.nu
    h = sym["h"]
    u_surface = sum(sym[f"q_{k}"] / h for k in range(3))       # φ_i(1)=1
    assert sp.simplify(hcol["q_0"] - 2 * nu * u_surface) == 0
    for k in range(3):
        i = st.index(f"q_{k}")
        assert sp.simplify(sp.sympify(A[i, i, 0, 0]) + 2 * nu) == 0


@pytest.mark.parametrize("cls", [SME, VAM, MLSME, MLVAM])
def test_default_path_has_no_viscous_marker_and_zero_h_column(cls):
    """REGRESSION IDENTITY — the correction is INERT on the default path.

    The only behavioural change (the ``ViscousDiffusion`` tag + its extraction
    branch) fires exclusively on a term that has passed through
    ``package_viscous`` — i.e. the retain-viscous path.  With DEFAULT closures
    (``ShallowInPlane`` semantics, ``τ_de = 0``) no such term exists, so the
    derived system is TERM-BY-TERM identical to the pre-correction system: no
    marker survives into the residual and the in-plane h-column is empty.  (The
    full term-by-term default system is pinned by the ``*_reference.py``
    suites; this guards specifically that MY change did not perturb it.)"""
    if cls in (MLSME, MLVAM):
        base = [Newtonian(), NavierSlip(), StressFree(), MeanInterface()]
        common = dict(n_layers=2, level=1, dimension=2)
    else:
        base = [Newtonian(), NavierSlip(), StressFree()]
        common = dict(level=2, dimension=2)
    sm = SystemModel.from_model(cls(closures=base, **common))
    R = sm.reconstruct_residuals()
    for i in range(len(sm.state)):
        assert not sp.sympify(R[i]).has(ViscousDiffusion), (
            f"{cls.__name__} row {i}: ViscousDiffusion leaked into the default path")
    A = sm.diffusion_matrix
    if A is not None:
        st = [str(s) for s in sm.state]
        hi = st.index("h")
        for i, name in enumerate(st):
            if name.startswith("q"):
                assert sp.sympify(A[i, hi, 0, 0]) == 0, (
                    f"{cls.__name__}: default path has a spurious A[{name}←h]")
