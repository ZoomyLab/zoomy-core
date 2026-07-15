"""REQ-169: the Chorin split sub-systems' source jacobian-wrt-aux must have one
column per ``aux_state`` entry.

The REQ-151 aux-prepend (parent aux → prefix of every sub-system's aux_state)
GREW ``aux_state`` after ``expose_aux_atoms`` had already sized
``source_jacobian_wrt_aux_variables`` to the smaller vector, so the amrex Chorin
printer — which iterates ``c in range(len(aux_state))`` — ran off the end of the
stale jacobian (``Index (0, k) out of border``).  The split must rebuild the
derived jacobians after the aux_state mutation.
"""
import pytest

from zoomy_core.model.models.vam import VAM


@pytest.mark.parametrize("level", [1, 2])
def test_chorin_subsystem_jac_aux_matches_aux_state(level):
    res = VAM(level=level, dimension=2).chorin_split()
    for name in ("SM_pred", "SM_press", "SM_corr"):
        sm = getattr(res, name)
        J = sm.source_jacobian_wrt_aux_variables
        assert J is not None
        assert J.shape[0] == sm.n_equations, name
        assert J.shape[1] == len(sm.aux_state), (
            f"{name}: jac_aux has {J.shape[1]} cols but "
            f"{len(sm.aux_state)} aux_state entries (REQ-169 regression)")
