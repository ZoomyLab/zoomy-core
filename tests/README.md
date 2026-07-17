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
