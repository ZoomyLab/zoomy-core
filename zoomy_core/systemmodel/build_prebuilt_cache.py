"""Regenerate the SHIPPED SystemModel cache (REQ-163).

    python -m zoomy_core.systemmodel.build_prebuilt_cache

Builds the default model configurations from scratch (cache reads disabled)
and writes their pickled SystemModels into ``zoomy_core/systemmodel/_prebuilt/``
(package data) so a fresh install gets instant first-time builds.  Commit the
regenerated files whenever a derivation or builder changes.
"""
from __future__ import annotations

import os
import pickle
import time


def default_models():
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.model.models.sme import SME
    from zoomy_core.model.models.vam import VAM
    from zoomy_core.model.models.ml_swe import MLSWE
    from zoomy_core.model.models.ml_sme import MLSME
    from zoomy_core.model.models.ml_vam import MLVAM
    from zoomy_core.model.models.sigma3d import Sigma3D
    from zoomy_core.model.models.ke_sme import KESME
    from zoomy_core.model.models.qr_kesme import QRKESME
    from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
    clo = [Newtonian(), NavierSlip(), StressFree()]
    yield "swe-1d", lambda: SWE(dimension=1)
    yield "swe-2d", lambda: SWE(dimension=2)
    for lvl in (0, 1, 2):
        yield f"sme-l{lvl}-2d", lambda lvl=lvl: SME(level=lvl, dimension=2)
    for lvl in (1, 2):
        yield f"vam-l{lvl}-2d", lambda lvl=lvl: VAM(closures=list(clo), level=lvl, dimension=2)
    # 3-D (two-horizontal, t,x,y,z) structural specs.  These are built COLD by
    # default-tier structural tests (``test_sme_2d`` SME(dim=3); ``test_vam_2d``
    # + the ``fvm`` elliptic-BC / wet-dry Chorin tests VAM(dim=3)) — ~10-20 s
    # each uncached, which is the residual default-tier floor.  Ship them so the
    # tests hit the warm cache.  Closures/parameters MUST match the test spec
    # verbatim (they are part of ``model_spec_key``; parameter VALUES are not).
    # NOTE: no ``parameters=`` here.  Values are not part of the key and are no
    # longer baked into the entry either (the artifact stores the model's
    # DEFAULTS and the case's numbers are attached per build) — the shipped
    # cache used to carry this spec's nu=0.1 / lambda_s=0.5 to every user.
    for lvl in (1, 2):
        yield f"sme-l{lvl}-3d", lambda lvl=lvl: SME(
            level=lvl, dimension=3,
            closures=[Newtonian(), NavierSlip(), StressFree()])
    yield "vam-l1-3d-navierslip", lambda: VAM(
        level=1, dimension=3, closures=[NavierSlip(), StressFree()])
    yield "vam-l1-3d-newtonian", lambda: VAM(
        level=1, dimension=3, closures=[Newtonian(), StressFree()])
    yield "mlswe-2d", lambda: MLSWE(dimension=2)
    yield "mlsme-2d", lambda: MLSME(dimension=2)
    yield "mlvam-2d", lambda: MLVAM(dimension=2)
    yield "sigma3d", lambda: Sigma3D()
    yield "kesme-2d", lambda: KESME(dimension=2)
    yield "qrkesme-2d", lambda: QRKESME(dimension=2)


def main() -> int:
    os.environ["ZOOMY_DERIVATION_REBUILD"] = "1"     # always build fresh
    from zoomy_core.systemmodel import sm_cache
    from zoomy_core.systemmodel.model_builders import _BUILDERS, build_system_model

    out = sm_cache._prebuilt_dir()
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    for label, make in default_models():
        try:
            t0 = time.time()
            model = make()
            sm = build_system_model(model)
            key = sm_cache.cache_key(model, _BUILDERS[model._system_model_kind])
            # A shipped entry obeys the same contract as a user-dir one: the
            # SYMBOLIC identity only.  ``build_system_model`` has already
            # attached this instance's BCs/ICs to the object it returned, so
            # strip them back off before pickling — otherwise every user gets
            # this script's boundary conditions and initial conditions baked in
            # (and ``sm_cache.fetch`` refuses to serve the entry at all).
            sm_cache.strip_runtime_state(sm)
            sm_cache.assert_no_runtime_state(sm, "prebuilt")
            (out / f"{key}.pkl").write_bytes(pickle.dumps(sm))
            print(f"  {label:14s} -> {key[:12]}…  ({time.time()-t0:.1f}s)")
            n_ok += 1
        except Exception as exc:
            print(f"  {label:14s} FAILED: {type(exc).__name__}: {str(exc)[:100]}")
            n_fail += 1
    print(f"prebuilt cache: {n_ok} built, {n_fail} failed -> {out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
