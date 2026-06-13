# Design note — dimensionality + dimensional splitting (zoomy_core)

Handoff spec for redefining how zoomy_core tracks dimensionality and splits a
model into 2-D and 3-D sub-systems. Motivated by the **stay-3D σ model**
(`zoomy_core/model/models/stay3d_sigma.py`): general 3-D balance → σ-map →
column-integrated height equation, but NO velocity ansatz — the flow stays 3-D
and the vertical integral is an aux. To solve it we need one system in which
`h, b` are 2-D fields `(t,x)` and `ũ` is a 3-D field `(t,x,ζ)`, coupled.

## 0. The core problem (what is wrong today)

A field's dimensionality is encoded in its `args`: `h(t,x)` vs `ũ(t,x,ζ)`. The
derivation `Model` keeps this, and `Model.vertical` (`model.py:353`) returns the
vertical coord; the σ-map even swaps it in place (`transformations.py:251`:
`model._vertical = ζ`). **But `extract_system_operators` throws it away:**

- `system_extract.py:112-115` forces `space = model.horizontal` (drops the vertical).
- `_state_symbol` (`system_extract.py:55-63`) collapses every field to a bare
  Symbol, dropping ALL args — so `h(t,x)` and `ũ(t,x,ζ)` both become bare symbols
  and the ζ-dependence is gone before the SystemModel exists.
- The classifier raises on any surviving `∂_ζ` / `Integral` because ζ ∉ `space`.

**Answer to "does the system level retain enough info to trace that h,b are not
ζ-dependent?"** — At the derivation `Model`: YES (`field.args`, `Model.vertical`).
At the current `SystemModel`: NO (erased by `_state_symbol`). ⇒ split BEFORE
extraction, or carry per-field coords into the SystemModel. Recommend the former.

## 0b. ✅ DONE — per-field dimensionality is now carried INTO the SystemModel

**Implemented + verified 2026-06-13 (13-test extraction guard green, byte-identical).**
`SystemModel` now carries:
- `state_function_map: Optional[Dict]` — `{state_symbol: applied field}`, populated
  in `system_extract.extract_system_operators` (`{s: f for s, f in zip(state, Q)}`)
  and set on the declarative `_from_derivation_model` (production path too).
- `vertical: Optional[Any]` — `model.vertical` (the σ `ζ` / `coords[-1]`).
- `SystemModel.is_vertical_dependent(symbol) -> bool` — `vertical in
  state_function_map[symbol].args`.  `h,b → False`; a `(t,x,ζ)` field → `True`;
  SME/VAM report all-False (ζ projected out).  Zero behavioural change to existing
  models (the map is additive metadata).

Aux applied forms (e.g. the v1 `ũ` aux) are recoverable from each
`aux_registry` entry's `"atom"`; add a `state`-style aux map only if the split
needs it directly.  The split (§2) can now run AFTER extraction by partitioning
on `sm.is_vertical_dependent`.

---

(original proposal, for reference)

The cleanest way to STOP erasing the per-field signature is a **`state_function_map`**,
the exact twin of the existing `aux_function_map: Dict[Symbol, Function]`
(`system_model.py:1611, 1676`). `from_model` already receives `Q` as APPLIED
fields with `.args` intact, and `system_extract.py:150` already iterates them
(`state = [_state_symbol(f) for f in Q]`). At that same spot:

```python
state_function_map = {_state_symbol(f): f for f in Q}   # h -> h(t,x), ũ -> ũ(t,x,ζ)
aux_function_map   = {_state_symbol(f): f for f in Qaux}
```

Store both on the dataclass (beside the existing `aux_function_map`), plus
`sm.vertical = model.vertical`. Then dimensionality is recoverable forever:

```python
def is_3d(sm, s):           # s a state Symbol
    return sm.vertical in sm.state_function_map[s].args
# h, b -> 2-D ;  ũ -> 3-D
```

Properties: **additive / zero-regression** (does not touch `_state_symbol`, the
operator tensors, or any current consumer — SME/VAM/SWE carry an unused map);
**precedented** (literal twin of `aux_function_map`); **faithful** (full
signature, so `(t,x)`, `(t,x,y)`, `(t,x,ζ)`, `(t,x,y,ζ)` are uniform — the
runtime reads it to size each field: 2-D = `n_horiz` dofs, 3-D =
`n_horiz × n_layers`, and attaches the column structure to 3-D fields only).

It is **complementary** to the split, and it ENABLES BOTH: with the map present,
`split_by_dimension` can run AFTER extraction (partition state by `is_3d`) as
well as before. The map says WHO is 3-D; `space` including ζ + the per-direction
routing does the actual flux work.

## 1. Dimensionality as a DERIVED property (not a param switch)

Replace the `dimension`-param behaviour switch (the "dimension=2 no-op" sore
spot) with dimensionality DERIVED from coordinates:

- The model is born over full coords `(t, x[, y], z)`. Each field is a `Function`
  of exactly the coords it depends on.
- `Model.vertical = coords[-1]` (exists). A field is **vertical-dependent** iff
  `Model.vertical in field.args`.
- The σ-map swaps `z → ζ` in place (exists; keep). Reduced/shallow models
  (SME/VAM) are the case where the vertical has been INTEGRATED OUT — fields lose
  ζ and the vertical is removed from coords, yielding a genuinely lower-D model.
  That reduction should be explicit, not a param.
- **Requirement:** nothing downstream may assume all fields share one spatial
  domain. Preserve a per-field coordinate signature (a `{state_symbol: coords}`
  map) alongside the name-collapse `_state_symbol` does for indexing.

## 2. Dimensional split (the clean design — ride the existing splitter)

Add `split_by_dimension(model)` as a sibling of `split_for_pressure`
(`splitter.py`), run on the derivation **Model** (info intact):

- **2-D sub-model**: fields with `vertical ∉ args` (`h, b`) + the rows whose
  evolved field is 2-D (the height equation `∂_t h + ∂_x(h U) = 0`).
  `space = (t, x[, y])`.
- **3-D sub-model**: fields with `vertical ∈ args` (`ũ`, …) + the 3-D momentum.
  `space = (t, x[, y], ζ)` — **ζ is a flux direction**. The classifier already
  routes per-direction via `space.index(...)` (`system_extract.py:329-364`), so
  once ζ is in `space` and `_state_symbol`/`field_to_fn` keep the ζ arg for 3-D
  fields, `∂_ζ(·)` flux / NCP / diffusion terms route automatically.
- **Coupling** (the only cross-system data, both via the funneled
  `update_aux_variables` / aux-registry — NOT bespoke solver code):
  - `h` (2-D state) enters the 3-D momentum as an aux (pressure/geometry `g h`,
    Jacobian).
  - `U = ∫₀¹ ũ dζ` (full column) and `ω` (running column integral) from the 3-D
    field enter the 2-D height flux / 3-D vertical flux as auxes. These are
    exactly the `project_from_3d` `∫` (full + running) lowered per backend.

This is the barotropic/baroclinic split of ocean models, and structurally the
SAME predictor/pressure/corrector partition `split_for_pressure` already does.

### What `_build_subsystem` needs (`splitter.py:344`)

It already builds a rectangular SystemModel from `(name, residual)` pairs + a
`state` list. The ONE change: a sub-system must be allowed its **own `space`**
(today `coords = list(sm_parent.space)`, line 373 — inherits the parent). Pass
the sub-system's coords (2-D vs 3-D) so the 3-D sub-system carries ζ.

### The ONLY extractor change (scoped to the 3-D sub-model)

- `space` ← the sub-model's coords (include ζ when present).
- `_state_symbol` / `field_to_fn`: keep each field's own coord signature
  (don't drop the vertical for vertical-dependent fields).
- Everything else (per-direction flux/NCP/diffusion routing, mass-matrix check)
  is already dimension-general — verified: the 2-D-horizontal multi-direction
  path landed in a recent commit, and the routing keys on `space.index`.
- 2-D sub-model goes through the UNCHANGED path ⇒ SME/VAM/SWE extraction stays
  byte-identical (regression guard: `tests/model/test_sme_reference.py`,
  `test_vam_reference.py`, `test_sme_2d.py` — 13 tests green today).

## 3. Conserved variables (clarification — NOT a task)

"M = I via conserved vars" is just the existing convention, not new work: the
framework already evolves CONSERVED variables (`q`-state with `InvertMassMatrix`
giving `M = I`). For the 3-D momentum the conserved variable is the momentum
density (`h·u` per layer), so `∂_t(h u)` is already unit-coefficient and the
existing mass-matrix check passes. State = `h u` (not primitive `u`), consistent
with SWE/SME. Nothing to implement.

## 4. What the stay-3D model needs from this (the requirement list)

1. After σ-map, the model stays in `(t,x,ζ)` with per-field dimensionality intact
   (already true at the Model level — just don't erase it at extraction).
2. `split_by_dimension(model)` → a 2-D `[b, h] + U` system and a 3-D `[h u] + ω`
   system, coupled via `U, ω` (3-D→2-D) and `h` (2-D→3-D).
3. The 3-D sub-system extracts with ζ as a flux direction.
4. `U, ω` funneled through `update_aux_variables` (the `project_from_3d` `∫`,
   full + running), consumed by ONE general solver branch — no per-model override.

## 5. Verified facts this builds on (run, not asserted — 2026-06-13)

- Height eq EXACT: `∂_t h + ∂_x(h U) = 0`, `U = ∫₀¹ ũ dζ` (substitute the integral
  into `U` BEFORE `Simplify`, else a spurious `+U ∂_x h` survives).
- Aux without a new op: `m.add_equation("U", sp.Eq(U, ∫ũ dζ), group="aux")` +
  `m.mass.apply({∫ũ dζ: U})` → `U` auto-lands in `Qaux`, flux `F[h]=U·h`.
- Conservative-σ identity (`diff = 0`): `σ-mass(×h) ≡ ∂_t h + ∂_x(h u) + ∂_ζ(h ω)`;
  `σ-mom(×h) ≡ ∂_t(h u) + ∂_x(h u² + g h²/2) + ∂_ζ(h u ω) + g h ∂_x b − e_x g h −
  (1/ρ)∂_ζ τ̃_xz`; `h ω = w̃ − ∂_t z − ũ ∂_x z`; `ω(0)=ω(1)=0` ⟺ the two KBCs.
  (Repro: `/tmp/conservative_sigma_cell.py`.)
