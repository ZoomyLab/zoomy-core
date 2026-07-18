"""ElderSME and KESME — turbulent SME models inheriting the dimension-agnostic
SME, WITHOUT disturbing plain SME (the turbulence hooks are no-ops on SME)."""
import sympy as sp
import pytest

from zoomy_core.model.models import SME, ElderSME, KESME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel.system_model import SystemModel


def _closes(sm):
    return sum(len(sp.sympify(sm.source[i, 0]).atoms(sp.Integral))
               for i in range(sm.n_equations)) == 0


def test_plain_sme_unaffected_by_turbulence_hooks():
    """Plain SME carries NO turbulence state and still closes — the hooks
    added for KESME are inert here."""
    sm = SystemModel.from_model(SME(level=2, closures=[Newtonian(), NavierSlip(), StressFree()]))
    st = [str(s) for s in sm.state]
    assert st == ["b", "h", "q_0", "q_1", "q_2"]
    assert not any(x in st for x in ("k", "varepsilon"))
    assert _closes(sm)


@pytest.mark.parametrize("dim", [2, 3])
def test_elder_sme(dim):
    lvl = 2 if dim == 2 else 1
    sm = SystemModel.from_model(ElderSME(level=lvl, dimension=dim,
                  parameters={"u_star": 0.3, "kappa": 0.41, "k_s": 0.01}))
    assert sm.n_dim == (1 if dim == 2 else 2)
    assert _closes(sm), "Elder ν_t(ζ) is polynomial → closes analytically"
    # no extra turbulence state — Elder closes ν_t algebraically, so neither the
    # raw fields nor the K_i/E_i transported moments appear
    st = [str(s) for s in sm.state]
    assert not any(s.startswith(("K_", "E_")) or "varepsilon" in s for s in st)


@pytest.mark.parametrize("dim", [2, 3])
def test_ke_sme(dim):
    lvl = 2 if dim == 2 else 1
    sm = SystemModel.from_model(KESME(level=lvl, dimension=dim, parameters={"k_s": 0.01}))
    assert sm.n_dim == (1 if dim == 2 else 2)
    st = [str(s) for s in sm.state]
    # Transported turbulence. k and ε are modal (separation_of_variables at N_k)
    # and then conservative: K_i = h·k̂_i, E_i = h·ε̂_i, exactly like q_i = h·û_i.
    # So the state carries the MOMENT FAMILIES K_*/E_*, not scalar k/varepsilon.
    K = [s for s in st if s.startswith("K_")]
    E = [s for s in st if s.startswith("E_")]
    assert K and E, f"no transported turbulence moments in {st}"
    assert len(K) == len(E), "k and ε must share the modal order N_k"
    assert _closes(sm), "depth-averaged ν_t=C_μ k²/ε is const in ζ → closes"
    # the k balance carries the shear production (∝ C_μ via ν_t)
    ks = sp.sympify(sm.source[st.index(K[0]), 0])
    assert ks.has(sm.parameters.C_mu), "k production missing"
    # ε balance carries the C_1/C_2 source
    es = sp.sympify(sm.source[st.index(E[0]), 0])
    assert es.has(sm.parameters.C_1) and es.has(sm.parameters.C_2)
