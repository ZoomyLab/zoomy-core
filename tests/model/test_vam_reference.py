"""VAM vs Escalante, Morales de Luna, Cantero-Chinchilla & Castro-Orgaz
(JCP 2024) — TERM-BY-TERM reference pinning.

Every dynamic row, both divergence constraints, the Chorin elliptic rows
and the corrector updates of ``VAM(level=1)`` are asserted SYMBOLICALLY
against the published system:

* eq (4)  — the five dynamic rows (h, hu0, hw0, hu1, hw1), inviscid;
* eq (5)  — the two divergence constraints I1, I2;
* §3.3    — the projection step: constraints evaluated on U − dt·T(P).

Basis / variable mapping (theirs ← ours), documented in the thesis
notebook ``derivation/vam.py``:

* their phi_1 = 1 − 2*xi = −(our phi_1 = 2*zeta − 1); phi_0, phi_2 equal
  ⇒ odd modes flip sign:  u1 = −q_1/h,  w1 = −r_1/h  (and the k=1 rows
  flip as whole rows: ours = −theirs);
* their pressure is KINEMATIC:  p_j = ±P_j/rho;
* w2 closed by the bottom KBC:  w2 = −(w0+w1) + (u0+u1)·∂x b;
* inviscid comparison: our nu, lambda_s set to 0 symbolically (their
  friction closure tau = eps·|u0+u1|(u0+u1) is a different model).

Shape, count and smoke checks are NEVER sufficient — the original
uniform-truncation VAM passed all of those while the pressure space was
wrong (secular P-drift at runtime).  This file pins every term.
"""
import pytest
import sympy as sp

from zoomy_core.model.models import VAM, newtonian_navier_slip


# ── shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vam1():
    model = VAM(material=newtonian_navier_slip(), level=1)
    sm = model.system_model
    return model, sm


@pytest.fixture(scope="module")
def reference(vam1):
    """Escalante eq (4)/(5) in OUR variables, plus mapping helpers."""
    _, sm = vam1
    t, x = sm.time, sm.space[0]

    def Fn(name):
        return sp.Function(name, real=True)(t, x)

    hf, bf = Fn("h"), Fn("b")
    q0, q1, r0, r1, P0, P1 = (Fn(n) for n in ("q_0", "q_1", "r_0", "r_1",
                                              "P_0", "P_1"))
    rho = sm.parameters.rho
    g = sm.parameters.g
    Dx = lambda e: sp.Derivative(e, x)
    Dt = lambda e: sp.Derivative(e, t)

    u0, u1, w0, w1 = q0 / hf, -q1 / hf, r0 / hf, -r1 / hf
    p0, p1 = P0 / rho, -P1 / rho
    w2 = -(w0 + w1) + (u0 + u1) * Dx(bf)

    R = {
        "h":   Dt(hf) + Dx(hf * u0),
        "q_0": Dt(hf * u0) + Dx(hf * u0**2 + hf * u1**2 / 3 + hf * p0)
               + g * hf * Dx(bf + hf) + 2 * p1 * Dx(bf),
        "r_0": Dt(hf * w0) + Dx(hf * u0 * w0 + hf * u1 * w1 / 3) - 2 * p1,
        "q_1": Dt(hf * u1) + Dx(2 * hf * u0 * u1 + hf * p1) - u0 * Dx(hf * u1)
               - (3 * p0 - p1) * Dx(hf) - 6 * (p0 - p1) * Dx(bf),
        "r_1": Dt(hf * w1)
               + Dx(hf * u0 * w1 + u1 * (hf * w0 + sp.Rational(2, 5) * hf * w2))
               + (w2 / 5 - w0) * Dx(hf * u1) + 6 * (p0 - p1),
    }
    C = {
        "P_0": hf * Dx(u0) + Dx(hf * u1) / 3 + u1 * Dx(hf) / 3
               + 2 * (w0 - u0 * Dx(bf)),
        "P_1": hf * Dx(u0) + u1 * Dx(hf) + 2 * (u1 * Dx(bf) - w1),
    }
    sign = {"h": 1, "q_0": 1, "r_0": 1, "q_1": -1, "r_1": -1}
    inviscid = {sm.parameters.lambda_s: 0, sm.parameters.nu: 0}
    return {"R": R, "C": C, "sign": sign, "inviscid": inviscid, "Fn": Fn}


# ── eq (4): every dynamic row ──────────────────────────────────────────────

def test_vam1_dynamic_rows_match_escalante_eq4(vam1, reference):
    _, sm = vam1
    rv = sm.reconstruct_residuals()
    names = [str(s) for s in sm.state]
    for name, ref in reference["R"].items():
        mine = sp.sympify(rv[names.index(name)]).subs(reference["inviscid"])
        diff = sp.simplify(sp.expand(mine.doit())
                           - reference["sign"][name] * sp.expand(ref.doit()))
        assert diff == 0, f"row {name} differs from Escalante eq (4): {diff}"


# ── eq (5): both divergence constraints (pinned at exactly −1) ─────────────

def test_vam1_constraints_match_escalante_eq5(vam1, reference):
    _, sm = vam1
    rv = sm.reconstruct_residuals()
    names = [str(s) for s in sm.state]
    for name, ref in reference["C"].items():
        mine = sp.expand(sp.sympify(rv[names.index(name)]).doit())
        diff = sp.simplify(mine + sp.expand(ref.doit()))   # ours = −theirs
        assert diff == 0, f"constraint {name} differs from eq (5): {diff}"


# ── §3.3: elliptic rows + corrector = constraints on U − dt·T(P) ───────────

def test_vam1_chorin_split_matches_escalante_projection(vam1, reference):
    model, sm = vam1
    split = model.chorin_split()
    dt = sp.Symbol("dt", positive=True)
    # the splitter generated its own dt symbol — locate it
    dts = [a for a in sp.sympify(split.SM_corr.state_update[0]).atoms(sp.Symbol)
           if str(a) == "dt"]
    dt = dts[0] if dts else dt

    rv = sm.reconstruct_residuals()
    names = [str(s) for s in sm.state]
    Fn = reference["Fn"]
    R, sign = reference["R"], reference["sign"]

    zero_P = {Fn("P_0"): 0, Fn("P_1"): 0}
    corr_map = {}
    for name in ("q_0", "q_1", "r_0", "r_1"):
        ref_row = sp.expand(R[name].doit())
        T_ref = sp.expand(ref_row - ref_row.subs(zero_P))
        corr_map[Fn(name)] = Fn(name) - dt * sign[name] * T_ref

    # elliptic rows = my (paper-pinned) constraints on the corrected state
    press_rv = split.SM_press.reconstruct_residuals()
    for m_i, cname in enumerate(("P_0", "P_1")):
        expected = sp.expand(
            sp.sympify(rv[names.index(cname)]).subs(corr_map).doit())
        got = sp.expand(sp.sympify(press_rv[m_i])
                        .subs(reference["inviscid"]).doit())
        diff = sp.simplify(got - expected)
        assert diff == 0, f"elliptic row {cname} != Escalante projection: {diff}"

    # corrector updates = the reference-anchored corrected state
    aux_rev = split.SM_corr._aux_reverse_map()
    sym2fn = {s: Fn(str(s)) for s in split.SM_corr.state}
    for k, nm in enumerate(split.SM_corr.equation_names):
        sname = nm[len("corr_"):]
        upd = sp.expand(sp.sympify(split.SM_corr.state_update[k])
                        .xreplace(aux_rev).xreplace(sym2fn).doit())
        exp_upd = sp.expand(corr_map[Fn(sname)].doit())
        diff = sp.simplify(upd - exp_upd)
        assert diff == 0, f"corrector {sname} != U − dt·T(P): {diff}"


# ── predictor: Escalante eq (12a) — the pressure-FREE hydrostatic rows ─────

def test_vam1_predictor_is_pressure_zeroed_full_rows(vam1, reference):
    """SM_pred row-by-row ≡ the full system's rows with P_0 = P_1 = 0.
    Pins the splitter's RE-extraction (term content AND source sign —
    a double sign flip here once anti-damped the predictor friction and
    inverted the dam-break shear profile)."""
    model, sm = vam1
    split = model.chorin_split()
    rv = sm.reconstruct_residuals()
    names = [str(s) for s in sm.state]
    Fn = reference["Fn"]
    zero_P = {Fn("P_0"): 0, Fn("P_1"): 0}

    pred_rv = split.SM_pred.reconstruct_residuals()
    for k, nm in enumerate(split.SM_pred.equation_names):
        sname = nm[len("pred_"):]
        expected = sp.expand(
            sp.sympify(rv[names.index(sname)]).subs(zero_P).doit())
        got = sp.expand(sp.sympify(pred_rv[k]).doit())
        diff = sp.simplify(got - expected)
        assert diff == 0, f"SM_pred row {sname} != full row at P=0: {diff}"


def test_vam1_advective_divergences_routed_as_flux(vam1):
    """The conservative parts of the k=1 rows must sit in the FLUX slot
    (compound ∂_x atoms), not be smeared into NCP — flux vs NCP routing
    is O(1) at shocks (path-dependence of nonconservative products)."""
    _, sm = vam1
    names = [str(s) for s in sm.state]
    i_q1 = names.index("q_1")
    i_r1 = names.index("r_1")
    Fq1 = sp.sympify(sm.flux[i_q1, 0])
    Fr1 = sp.sympify(sm.flux[i_r1, 0])
    by_name = {str(s): s for s in sm.state}
    q0, q1, r0, r1, P1, h = (by_name[n] for n in
                             ("q_0", "q_1", "r_0", "r_1", "P_1", "h"))
    rho = sm.parameters.rho
    assert sp.simplify(Fq1 - (2*q0*q1/h + h*P1/rho)) == 0, f"q_1 flux: {Fq1}"
    assert sp.simplify(
        Fr1 - (q0*r1/h + q1*r0/h - sp.Rational(2, 5)*q1*(r0 - r1)/h)) == 0, \
        f"r_1 flux: {Fr1}"
