# `_mass_gate.py` — 1-D closed-box mass-conservation gate for the VAM Chorin split

**Purpose.** Smoke-test that the VAM (non-hydrostatic) Chorin-split **predictor**
conserves mass. All-wall 1-D box (`Wall` both ends, reflecting both horizontal
momentum modes `q_0`, `q_1`), flat bed, depth-step dam break; tracks total mass
`Σ h·Δx` over a full slosh.

**Run.**
```
cd library/zoomy_core
JAX_PLATFORMS=cpu python _mass_gate.py
```

**Result.** Machine-zero drift (≈ −2e-16) over t≈4 s.

**Caveat — this test does NOT discriminate the real bug.** A constant-coefficient
non-conservative product still telescopes on a 1-D structured grid, so the
predictor conserves mass here **with or without** the tagger fix (verified by an
A/B git-stash run — both gave +0.0). The actual VAM over-fill is a
**multi-dimensional** failure: see `thesis/notebooks/steffler_jax/_mass_gate_dim3.py`
(dim=3 unstructured mesh) and task **0009**. Keep this script as the 1-D control —
useful to confirm the split itself is wired up, not to catch the N-D leak.

**Related:** zoomy_core `b95d6dd` (partial tagger fix), task 0009 (open N-D fix).
