# Protocol v2 pre-freeze preflight report

**Status: preflight only. Protocol v2 is NOT frozen. The formal 552-plan suite has NOT been launched.** This report covers manifest generation, geometry/support/feasibility checks, JS reachability/calibration, the true uniform-prior planning-baseline definition, primary/secondary/diagnostic metric declarations, and a small set of engineering-only planner feasibility/runtime checks. The Milestone 3 planner source was not modified; no check result was used to select or tune scenes, families, or gamma values.

Generated from `milestone_4/results/preflight`: `candidate_scene_manifest.json`, `feasibility_audit.json`/`.csv`, `js_reachability_table.csv`, `engineering_check.json`.

## 1. Candidate scene manifest

**12 scenes** = 4 true hotspot centers `(0.25,0.25)`, `(0.25,0.75)`, `(0.75,0.25)`, `(0.75,0.75)` x 3 obstacle layouts (A: none, B: one rectangle matching v1, C: two rectangles). Each scene carries two error-geometry direction blocks (x, y) per PROTOCOL_V2_DRAFT.md. Grid size 40x40, sigma 0.13, background 0.03, JS levels [0.05, 0.1, 0.15] nats, gammas [0.1, 0.05], initial state [0.1, 0.1, 0.0, 0.0], terminal state [0.9, 0.9, 0.0, 0.0] (fixed by the frozen planner, unchanged from v1).

**23 distinct conditions per scene per gamma**: oracle (1, shared) + diffuse/blur (3 JS levels, direction-shared) + true uniform-prior planning baseline (1, shared) + 3 direction-dependent families (spatial shift, false hotspot, false-negative suppression) x 3 JS levels x 2 direction blocks (18). **Projected plan count: 12 scenes x 2 gammas x 23 conditions = 552 plans.**

| Layout | Obstacles |
|---|---|
| A | none |
| B | rectangle [0.43,0.57] x [0.43,0.57] (matches v1) |
| C | rectangles [0.35,0.45] x [0.55,0.65] and [0.55,0.65] x [0.35,0.45] |

## 2. Geometry / support / feasibility audit

**All 12/12 candidate scenes passed** (yes). Checks: obstacles within the unit workspace; nonempty true-prior support; supported free cells form one 4-connected grid component; the fixed initial `(0.1,0.1)` and terminal `(0.9,0.9)` states clear every obstacle's collision disk; a 4-connected free-space path exists between the grid cells nearest those states; and each direction block's error geometry (false-hotspot center, suppression region) is well-defined. The connectivity checks are a **discrete grid proxy**, not a guarantee that the continuous-space SLSQP planner converges.

No scene failed any feasibility check; every candidate scene remains eligible.

## 3. JS reachability / calibration table

**252 scene x direction-block x family x JS-level rows (12 scenes x 2 blocks x 3 direction-dependent families x 3 levels, plus 12 x 3 shared blur rows).** No planner import; the same brentq-bisection calibration approach as the v1 pilot, generalized to a per-scene true center via a new `SceneErrorGenerator` (v1's `PriorErrorGenerator` intentionally rejects any non-v1 true center and is deliberately left unmodified, per PROTOCOL_V2_DRAFT.md).

**All rows matched at the existing 0.05/0.10/0.15 nat levels; no level revision is needed.**

Headroom by family (`scan_max_js_nats` is the minimum, across all scenes/blocks/levels for that family, of the maximum JS reachable anywhere in its one-dimensional parameter range; comfortably above 0.15 for every family means no scene is close to an attainability boundary):

| Family | Rows | Min scan-max JS (nats) | JS=0.15 parameter range |
|---|---:|---:|---:|
| spatial shift | 72 | 0.3857 | 0.2032 - 0.2126 |
| diffuse blur | 36 | 0.1776 | 5.8363 - 6.0776 |
| false hotspot | 72 | 0.5210 | 0.5212 - 0.5292 |
| false negative suppression | 72 | 0.3073 | 0.9144 - 0.9176 |

## 4. True uniform-prior planning baseline

Defined in `milestone_4/scene_manifest.py:uniform_prior_baseline` as a distinct planned condition per scene (shared across gamma and direction blocks, like oracle and blur): a uniform distribution over the scene's supported free-cell mask, handed to `milestone_3.planner.plan` to produce its own trajectory. This is **not** the pilot's post-hoc uniform-weighted reweighting diagnostic, which reweights an already-planned oracle-conditioned trajectory rather than planning a new one under a uniform prior; both PROTOCOL_V2_DRAFT.md and `analyze_pilot.py`'s report text now name the diagnostic "uniform-weighted diagnostic" to avoid conflating the two. Its numerical feasibility as a planned condition was confirmed by the `layoutB_uniform_g010` engineering check below (valid, collision-free, 4.4s wall-clock). It adds one condition per scene/gamma (23 vs the draft's earlier 22), raising the projected plan count from 528 to 552.

## 5. Primary / secondary / diagnostic metric definitions

**Primary:** exact full-grid detection probability at budget, and its true-prior-weighted detection-time distribution / conditional mean / median, for every scene/geometry/JS/gamma block and family (plus oracle and the true uniform-prior planning baseline). Deterministic given the planned trajectory; carries no target-sampling noise.

**Secondary (sampled):** success/timeout rate; paired outcomes (oracle-outcome x error-prior-outcome 2x2, plus both-success ΔT tail statistics); free-space coverage fraction; true-prior mass coverage; per-prior entropy (nats); trajectory length.

**Diagnostic (not primary or secondary):** the uniform-weighted reweighting of an existing planned trajectory (distinct from the true uniform-prior planning baseline in section 4).

## 6. Gamma and the family x gamma estimand

gamma=0.10 and gamma=0.05 are kept and reported separately throughout; primary contrasts are never pooled across gamma. The family x gamma interaction on the primary exact-grid endpoints is declared as part of the estimand from the outset (see PROTOCOL_V2_DRAFT.md's "Contrasts and aggregation"), not introduced post hoc from an observed pattern. No gamma value was added, dropped, or reweighted based on any preflight or engineering-check result.

## 7. Engineering-only planner feasibility / runtime checks

A fixed set of 5 checks, selected by geometry/gamma/prior-type coverage *before* any was run (one per obstacle layout, one at the tighter gamma bound, one for the true uniform-prior baseline), all on the same scene center to isolate those factors from center choice. Only solver validity and wall-clock cost were recorded -- this is not a comparison of which condition finds targets faster, and this list was not edited after seeing results.

| Check | Rationale | Valid | Wall (s) | Optimize (s) | Iterations | tf (s) |
|---|---|---|---:|---:|---:|---:|
| `layoutA_oracle_g010` | no-obstacle geometry baseline | True | 6.3 | 5.2 | 148 | 3.329 |
| `layoutB_oracle_g010` | single-rectangle geometry, matches v1's obstacle shape | True | 7.8 | 6.6 | 179 | 3.530 |
| `layoutC_oracle_g010` | two-rectangle geometry, hardest collision constraint set | True | 7.4 | 7.2 | 182 | 3.853 |
| `layoutB_oracle_g005` | tighter ergodicity bound, same scene as layoutB_oracle_g010 | True | 8.6 | 8.4 | 229 | 4.375 |
| `layoutB_uniform_g010` | true uniform-prior planning-baseline feasibility, same scene as layoutB_oracle_g010 | True | 4.4 | 4.1 | 110 | 3.995 |

All 5/5 checks were valid, solver-converged and collision-free. Mean wall-clock 6.9s (range 4.4-8.6s) per plan, JIT compilation included (the planner recompiles its JAX functions on every call; this cost is not amortized across plans in the current `plan()` implementation).

**Projected formal-suite planning time: 552 plans x 6.9s mean ≈ 3805s ≈ 1.1 hours, sequential, single process.** This covers planning only, not the sampled-episode evaluation, coverage replay, or calibration steps the formal suite would also run, and is extrapolated from only 5 of 552 conditions (oracle and the uniform-prior baseline; the three direction-dependent error families were not separately timed, though they solve the same problem structure with a different prior weighting `phik`, not a different constraint set).

## 8. Freeze recommendation

**Recommendation: do not freeze Protocol v2 yet.** Gate items 1 and 3 of PROTOCOL_V2_DRAFT.md's "Scale and freeze gates" are satisfied by this preflight (manifest/feasibility/JS-calibration preflight complete; a small geometry-selected engineering check run). Gate items 2 (saved-input pairing and evaluator semantics on synthetic cases, including full-grid evaluation cost), 4 (target-sampling precision, final seed count and targets per seed) and 5 (freezing the manifest, calibrated vectors, source/dependency hashes, planned contrasts and denominators in a dedicated commit) are **not yet done** and remain required before launching the formal 552-plan suite.

No blocking finding surfaced: every candidate scene passed feasibility, every JS level is attainable everywhere with comfortable headroom, the true uniform-prior planning baseline is numerically feasible, and per-plan planning cost is small (~7s mean in the sampled checks) relative to the 552-plan suite. Nothing here changes the 12-scene design or the 0.05/0.10/0.15 nat JS levels.

