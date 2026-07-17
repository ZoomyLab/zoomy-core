# zoomy_core test tiers

The suite is **tiered** so the default run is a fast pre-publish gate. Two
opt-in tiers hold the expensive tests; both are **deselected by default**
(`conftest.py`), and the markers are registered in `pyproject.toml` so nothing
warns.

| tier | marker | what it holds |
|------|--------|---------------|
| **default (small)** | *(none)* | everything cheap on a warm cache, incl. the 1-step "small twins" of every large march |
| **large** | `@pytest.mark.large` | real time-march / simulation tests (VAM/ML-VAM dam-break DAE Chorin solves, multi-step MOOD/positivity marches, Σ-3D refinement studies, lake-at-rest well-balancing marches) |
| **rederive** | `@pytest.mark.rederive` | tests that clear/bypass the derivation cache or force a fresh derivation of a heavy family (double-moment SME, cold-cache `sm_cache` builds, VAM-3D/ML-VAM pressure operators) |

Every `large` march has a **1-step twin** in the default tier: identical setup,
exactly one timestep, asserting the cheap invariants (finite, `h >= 0`, bounded
mass change) — a real regression canary at ~seconds cost. The twins **add**
coverage; the large tests keep their full assertions.

## Fast-verify — the ONE entry point

`scripts/verify.py` is the single "is this change ok?" gate: it runs the default
(small) tier, prints the **wall-clock time**, and exits non-zero on failure
(so a git hook / CI can gate on it). Target: **≤ 2 minutes on a warm cache**.

```bash
micromamba run -n zoomy python scripts/verify.py            # full small tier + wall time
micromamba run -n zoomy python scripts/verify.py --changed  # only tests for changed subdirs
micromamba run -n zoomy python scripts/verify.py -k sme -x   # extra args pass to pytest
```

The small tier assumes a **warm derivation cache** (the shipped `_prebuilt` set +
the REQ-163 `sm_cache`). Every default-tier structural build — including the
`SME(dim=3)` and `VAM(dim=3)` specs that `test_sme_2d` / `test_vam_2d` / the
`fvm` elliptic-BC + wet-dry Chorin tests build — is in `_prebuilt`, so a fresh
checkout is warm. After a `CACHE_VERSION` bump or a derivation/builder edit,
**regenerate first** (otherwise the first run pays the cold symbolic cost):

```bash
python -m zoomy_core.systemmodel.build_prebuilt_cache
```

## Commands

```bash
# default (small) tier — the pre-publish gate; seconds-to-a-couple-of-minutes
micromamba run -n zoomy pytest tests/ -q

# add the time-march tier
micromamba run -n zoomy pytest tests/ -q --run-large

# add the cold-cache / fresh-derivation tier
micromamba run -n zoomy pytest tests/ -q --run-rederive

# everything
micromamba run -n zoomy pytest tests/ -q --run-large --run-rederive

# address a tier directly (an explicit -m overrides the auto-deselection)
micromamba run -n zoomy pytest tests/ -q -m large
micromamba run -n zoomy pytest tests/ -q -m rederive
```

Run the `large` and `rederive` tiers on demand (before a release, or when
touching the solvers / derivation machinery). The warm derivation cache is
assumed for the default tier; cache re-evaluation happens case-by-case via
`--run-rederive`.
