# Milestone 4 protocol v2 — formal experiment design draft

**Status: draft, not frozen and not executed.** This proposes a finite benchmark
suite. It does not change `protocol.py`, the v1 generator, the frozen planner,
or existing results. The final scene manifest, common JS levels and evaluation
sample size must pass the gates below before a formal run is launched.

**Preflight status:** the manifest/feasibility/JS-calibration preflight and a small
set of geometry-selected engineering-only planner checks have been run; see
[PROTOCOL_V2_PREFLIGHT_REPORT.md](results/preflight/PROTOCOL_V2_PREFLIGHT_REPORT.md).
This updates the plan count below (23 conditions/scene/gamma, including the true
uniform-prior planning baseline) and gate items 1 and 3. The formal 552-plan suite
itself has **not** been launched.

## Question and motivation

At matched JS divergence, how do error structures change search success and
detection timing under the same planner and sensing protocol, and how stable
are these differences across declared scenes and error geometries?

The [pilot analysis](results/analysis/REPORT.md) motivates three design choices:

- Keep gamma=0.10 and gamma=0.05 separate: the spatial-shift success contrast
  at JS=0.10 changes sign between them in both pilot target seeds.
- Measure both area and true-mass coverage: a false hotspot at JS=0.15,
  gamma=0.05 increases visible area while losing true-prior mass coverage.
- Do not equate two target seeds with two independent planning or scene trials.
  The shared deterministic plan produces only target-sampling variation.

These exploratory observations motivate the design; they are not confirmatory
evidence for a general ranking. V1's exact scene remains a separate reference
and is excluded from the proposed primary suite.

The pilot's post-hoc analysis was subsequently extended (still zero new planning
runs) to add exact full-grid detection/timing metrics, a uniform-weighted
reweighting diagnostic, paired-outcome tail diagnostics and per-prior entropy/trajectory
length; see the [primary endpoint declaration](#primary-endpoint-declaration) below. That
extended analysis motivates the primary-endpoint choice below, but surfaced no
concrete reason to change the 12-scene/geometry/JS design itself: full-grid
detection and its area-only (uniform-weighted) diagnostic changed in the same
direction in all 24 pilot rows, and only differed in relative magnitude in one;
entropy varied smoothly across families and levels with no attainability failure.
If a future formal-suite preflight finds otherwise, document the finding and
issue a revised draft rather than changing the design silently.

## Proposed finite scene suite

Cross four Gaussian hotspot centers with three obstacle layouts, giving 12
scenes. For every scene use sigma 0.13, background density 0.03 and the same
40×40 cell-center grid, masked outside that scene's conservative collision disks.

| Dimension | Proposed values |
|---|---|
| True hotspot center `(cx, cy)` | `(0.25,0.25)`, `(0.25,0.75)`, `(0.75,0.25)`, `(0.75,0.75)` |
| Layout A | No obstacles |
| Layout B | V1 rectangle: `[0.43,0.57] × [0.43,0.57]` |
| Layout C | Two rectangles: `[0.35,0.45] × [0.55,0.65]` and `[0.55,0.65] × [0.35,0.45]` |

These coordinates are proposed, not yet validated. Preflight must check nonempty
support, connected supported free space, reachable observing poses and valid
start/terminal states. Reject infeasible geometry by these checks, not by search
outcomes. Document any replacement and issue a new draft before freezing.

Within each scene and geometry block, keep the true prior, target samples,
camera, obstacles, budget, initial/end states, planner and evaluator fixed.
Scenes vary deliberately between blocks; this is an explicit extension of v1,
not a claim that the original single-scene inputs remain identical everywhere.

## Proposed error geometry blocks

For each true center `c`, define two directions toward the workspace interior:
`u_x=(sign(0.5-cx),0)` and `u_y=(0,sign(0.5-cy))`. Use separate x and y blocks.

| Family | Definition in a geometry block with direction `u` |
|---|---|
| Spatial shift | Move Gaussian center to `c+d*u`; `d` from zero to the opposite workspace boundary (0.75 for these centers). |
| Diffuse/blur | Keep the v1 sigma-broadening formula and bounds. It is identical in x/y blocks and must not be counted as two independent plans. |
| False hotspot | Mix in a normalized Gaussian at the reflected center: `(1-cx,cy)` for x, `(cx,1-cy)` for y; sigma 0.13, no component background. |
| False-negative suppression | Suppress within a radius-0.25 disk centered at `c+0.10*u`, then renormalize. This explicitly changes v1's centered suppression geometry. |

Each family has a declared one-dimensional magnitude parameter. Direction,
false-hotspot location and suppression mask are fixed before calibrating that
parameter. Record all definitions in a machine-readable scene manifest. No
parameters may be selected by observed success or timing outcomes.

The existing v1 `PriorErrorGenerator` intentionally rejects a different truth;
it must not be relaxed silently. A separate v2 generator and validation suite
will be needed. The frozen M3 planner already accepts an environment and prior;
its implementation and solver settings remain unchanged.

## JS levels and calibration gate

Candidate common levels remain `0.05, 0.10, 0.15` nats with absolute tolerance
`1e-6`, calibrated against each scene's own true prior on its own fixed mask.
V1 success does not establish that these levels are attainable in all proposed
v2 blocks, particularly the offset suppression masks.

Before any search planning, generate a reachability table for every scene ×
geometry × family. Verify valid probabilities and zero-error identity. If any
candidate level is unattainable, do not compare family-specific substituted
levels or drop difficult conditions. Revise the common levels for the entire
suite using calibration evidence alone, document the revision, and recalibrate.
Freeze full-precision priors, parameters and hashes before evaluating search.

## Primary endpoint declaration

**Primary endpoints: exact full-grid detection probability and its true-prior-weighted
mean/median T_find, conditional on detection.** For every scene/geometry/JS/gamma block
and family (plus oracle), evaluate every supported target-grid cell under the same
visibility and observation rules used for sampled episodes. At budget this gives an exact
detection probability (the true-prior mass with a finite first-visible time) and, restricted
to detected mass, an exact true-prior-weighted mean/median T_find. Both are deterministic
functions of the frozen planner's output for that block — not draws — so they carry no
target-sampling noise and do not depend on the evaluation seed count chosen in the section
below.

**Secondary: sampled successful-only mean/median T_find and the sampled success/timeout
rate**, over the seeds/targets-per-seed fixed later in this draft. These remain fully
reported, including the paired-outcome contrasts in "Contrasts and aggregation," as a
precision-sensitivity check: they show whether a given number of target draws would recover
the exact full-grid answer, and by how much a finite-sample estimate can diverge from it.

**Diagnostic, not primary or secondary: the uniform-weighted diagnostic.** The same exact
full-grid computation, reweighted by a uniform distribution over the scene's supported
free-cell mask instead of its true prior, applied to the same true-prior-conditioned planned
trajectory, isolates which cells a trajectory physically visited from how much true-prior
probability mass it captured. Report it alongside the primary true-prior-weighted result for
every block, but do not treat it as a confirmatory or ranking metric on its own. It is not a
true uniform-prior baseline: that would require planning a fresh trajectory under a uniform
prior rather than reweighting the existing one.

**A distinct planned condition: the true uniform-prior planning baseline.** Every scene now
carries an explicit `uniform_prior_baseline` condition (`milestone_4/scene_manifest.py:uniform_prior_baseline`):
a uniform distribution over the scene's supported free-cell mask, handed to
`milestone_3.planner.plan` the same way the oracle and error priors are, producing its own
planned trajectory rather than reweighting an existing one. It is direction-independent and
shared per scene/gamma like oracle and blur, adding one condition per scene/gamma (23 total,
see "Scale and freeze gates"). Its own exact full-grid detection probability and
true-prior-weighted T_find should be reported per scene/gamma alongside oracle's, as a
planned-trajectory contrast distinct from the reweighting diagnostic above. Numerical
feasibility (solver validity, collision-free, wall-clock cost) for this condition was
confirmed by one geometry-selected engineering check (`layoutB_uniform_g010`); see
[PROTOCOL_V2_PREFLIGHT_REPORT.md](results/preflight/PROTOCOL_V2_PREFLIGHT_REPORT.md). This is a
feasibility check only, not a comparison of which condition performs better.

This ordering reverses the v1 pilot draft's original stance (which held sampled metrics
primary and full-grid evaluation supplemental). The reversal is justified directly by the
pilot's own findings, not asserted a priori:

- Observation 5 in the [pilot report](results/analysis/REPORT.md): at spatial shift
  JS=0.15, gamma=0.05, true-prior mass covered (the exact full-grid detection probability)
  changes by +1.58 percentage points relative to oracle, while the pooled 128-draw sampled
  success rate for the same saved trajectory changes by -4.69 points — opposite signs for
  the same underlying plan. The exact metric integrates the fixed distribution; the sampled
  metric is one finite draw of it. Declaring the distributional quantity primary removes
  this kind of disagreement as a source of conflicting reports on the same saved trajectory.
- Observation 6: reweighting the same saved trajectories by a uniform baseline instead of
  the true prior kept the same sign as the true-prior-weighted timing change in all 24
  pilot rows, but changed the magnitude — the true-prior-weighted change exceeded the
  uniform-weighted change in 23 of 24 rows. The true-prior-weighted (primary) and
  uniform-weighted (diagnostic) full-grid quantities are therefore not interchangeable, and
  only the former is the quantity the search task is actually specified to minimize risk
  against (true-prior-weighted, not area-weighted, detection).

Every scene/geometry/JS/gamma block report (see "Contrasts and aggregation") must present
the primary full-grid metrics first, the uniform-weighted diagnostic alongside them and
labeled as diagnostic, and the sampled metrics second and labeled as a precision-sensitivity
check against the primary. Do not silently substitute one for another, and do not compute a
formal ranking or confidence interval from the sampled metrics alone without also reporting
the exact full-grid value for the same block.

## Planner and evaluation invariants

Use the v1 frozen source commits and hashes, two gamma bounds (0.10/0.05),
50 nodes, 600 maximum iterations, the same deterministic initial guess and no
warm starts. Keep initial/end states `(0.1,0.1,0,0)` / `(0.9,0.9,0,0)`.
Keep range 0.25, FoV 90°, zero-phase pan `pi*t/2`, budget 15 seconds and
0.05-second observations. Preserve collision validation, reference node timing
and terminal hold exactly. Target coordinates never reach planning.

Proposed evaluation sampling is seeds **100–109**, **256 targets per seed per
scene**, using the unchanged `default_rng(seed).choice(..., p=true_prior)` rule.
These are candidate precision settings, not a power calculation. Share each
scene's exact sampled target arrays across every family, level, geometry and
gamma. Store and pair using `(scene_id, seed, episode)`. Do not assume identical
coordinates across scenes merely because the seed is identical.

Evaluate all supported target-grid cells using the same visibility and
observation rules, weighted by true-prior mass and, as a diagnostic, by a
uniform distribution over the supported mask; see "Primary endpoint
declaration" above for why the true-prior-weighted full-grid result — not the
sampled success/timeout and conditional timing metrics below — is the primary
reported output. This is not continuous-space ground truth or an extra
independent replicate; it separates sampling discrepancies from trajectory
effects using the same saved trajectory the sampled metrics are computed from.

**Estimand.** The primary contrasts are the exact full-grid detection probability and
true-prior-weighted T_find, per (family, gamma) cell, never pooled across gamma (see item 3
below). The family × gamma interaction on these primary endpoints is therefore part of the
declared estimand from the outset, not a post-hoc addition motivated by an observed pattern.

## Contrasts and aggregation

1. For every scene/geometry/JS/gamma block, report all four families plus oracle and the
   true uniform-prior planning baseline (see "Primary endpoint declaration"):
   exact full-grid detection probability and true-prior-weighted mean/median
   T_find (primary); the uniform-weighted diagnostic of both (for the four
   error families and oracle); success/timeout rate and successful-only
   mean/median T_find (secondary, sampled); coverage fraction; trajectory
   length; prior entropy; and planner validity.
2. Reconstruct oracle-versus-error pairs and all six unordered family-pair
   comparisons at fixed JS and gamma. Report the exact full-grid detection
   probability and weighted-T_find difference for every comparison first.
   Alongside it, as a distinct target-level contrast (not a substitute for the
   primary full-grid difference), reconstruct the paired outcome table as an
   explicit 2x2 (oracle outcome x error-prior outcome) of success/timeout
   counts, and report both-success paired mean/median ΔT plus its p75/p90/p95
   and max, and the ΔT>0/=0/<0 proportions, all with the common-success count
   as their stated denominator.
3. Do not pool gamma values, for either the primary full-grid metrics or the
   secondary sampled metrics: gamma indexes a distinct deterministic plan, so
   an exact full-grid value at one gamma is not exchangeable with the other,
   and the pilot's spatial-shift contrast changing sign between gammas shows
   sampled contrasts are not either. Average the two geometry-block contrasts
   inside each scene, then give each of the 12 scenes equal weight. Retain the
   full scene/block matrix and show the range and sign consistency of
   contrasts. Reused oracle/blur plans and shared targets are dependencies,
   not extra n.
4. Pool conditional time summaries from raw detections when describing a
   within-scene target set. Do not average seed medians, or replace missing
   successful-only times with zero. Do not interpret a common-success timing
   contrast as an all-target effect.
5. Primary claims are limited to this **declared finite suite**. The proposed
   12 scenes are a designed set, not random independent samples from a defined
   scene population. Do not attach across-scene population confidence intervals
   or significance claims by treating individual target draws as scene trials.
   A broader generalization study would first need a scene-sampling distribution
   and a separately justified number of independent scenes.

The first formal suite report will emphasize effect sizes, finite-suite
contrasts and per-scene variation, without a post-hoc significance ranking, led
by the primary exact full-grid contrasts. The secondary sampled contrasts
reveal which of those primary results a given evaluation sample size would
actually have recovered, i.e. how sensitive the sampled estimate is to the
target draws — not the reverse. If inferential tests are later required,
specify estimands, independent sampling units, sample-size justification and
multiplicity treatment in a further protocol revision before obtaining the
relevant confirmatory data.

## Invalid plans and missing conditions

Retain all attempts and solver diagnostics. An invalid plan is a planner failure,
not a search timeout; do not assign it T_find=15 or a success rate of zero.
Only valid pairs enter conditional search comparisons, and every report must
show the planned and available condition denominators.

For the initial formal benchmark, a failed plan blocks the claim of a complete
matched suite. Do not silently rank families on different retained scene sets,
change solver settings for one family, choose the best retry, or drop scenes
after viewing outcomes. Diagnose failures and either report the incomplete
suite explicitly or freeze a revised common solver/protocol and rerun all
affected comparisons under that revision.

## Scale and freeze gates

With two geometry blocks, each scene has, per gamma: 18 distinct shift/hotspot/
suppression priors (3 families × 3 levels × 2 geometries), three blur priors,
one oracle and one true uniform-prior planning baseline, or 23 distinct plans.
The proposed suite therefore has **12 × 2 × 23 = 552 plans**. Sharing
blur/oracle/uniform-prior-baseline plans across both direction blocks avoids
artificial replication. Target seeds change evaluation workload, not this plan
count.

1. **Done.** Implement v2 scene/error manifests and run calibration/geometry
   preflight only, before viewing any new search outcomes: see
   [PROTOCOL_V2_PREFLIGHT_REPORT.md](results/preflight/PROTOCOL_V2_PREFLIGHT_REPORT.md).
   All 12 candidate scenes passed the geometry/support/feasibility audit; all 252
   scene/direction-block/family/level JS-calibration rows matched at the existing
   0.05/0.10/0.15 nat levels, so no level revision was needed.
2. Validate saved-input pairing and unchanged evaluator semantics on synthetic
   cases. Measure evaluation cost, including the primary full-grid calculation.
3. **Partially done.** A separately labelled engineering check for solver
   feasibility/runtime was run on 5 blocks selected by geometry/gamma/prior-type
   criteria fixed before running (see the preflight report): one plan per
   obstacle layout, one at the tighter gamma bound, one confirming the true
   uniform-prior planning baseline is feasible. All 5 were valid and
   collision-free, wall-clock 4.4-8.6s each. This does not itself justify or
   tune the formal suite's per-scene/per-family selection, and does not
   constitute a runtime check of the full 552-plan suite at scale.
4. Review target-sampling precision requirements and resources; fix final seed
   count and targets per seed, and record their justification. No optional
   stopping based on whether a desired ranking has appeared.
5. Freeze the manifest, calibrated vectors, source/dependency hashes, planned
   contrasts, denominators and failure handling in a dedicated commit. Only
   then launch the formal suite into a new output directory.

The analysis accompanying this draft performs **zero new planning runs**, aside
from the 5 declared engineering-only checks in gate item 3 above.
