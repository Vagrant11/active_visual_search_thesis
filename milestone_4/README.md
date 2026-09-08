# Milestone 4: Controlled prior-error design

Four parameterized errors change only the prior supplied to the frozen
Milestone 3 planner. The [fixed protocol](PROTOCOL.md) specifies the distributions,
geometry, sampling, camera, planner and metrics. Milestone 3's current code,
figures, CSVs and scoped conclusion are committed at `6e0bce5`.

## Calibration first

From the repository root, generate priors and calibration artifacts without
importing or running the planner:

```bash
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python -m milestone_4.calibrate_priors
```

The shared levels are **0.05, 0.10, 0.15 nats**, with absolute JS tolerance
`1e-6`. The example level 0.30 is unattainable for the specified blur family:
its uniform free-cell limit is about 0.17805 nats. This is recorded explicitly
in [example_levels_audit.csv](results/calibration/example_levels_audit.csv).

The [calibration table](results/calibration/calibration.csv) records each family,
requested and actual JS, parameter, resolved geometry, numerical range, status,
normalization and blocked mass. All 12 rows match; the largest absolute error
is approximately **6.2e-14 nats**.

| Parameter | JS 0.05 | JS 0.10 | JS 0.15 |
|---|---:|---:|---:|
| Spatial shift: distance `d` westward | 0.114118 | 0.166445 | 0.211541 |
| Diffuse/blur: sigma multiplier `k` | 2.014885 | 3.171297 | 5.908290 |
| False hotspot: mixture mass `alpha` | 0.252402 | 0.402660 | 0.524946 |
| False-negative suppression: fraction removed inside region `alpha` | 0.732938 | 0.848722 | 0.908207 |

These parameters are rounded for display; use the full precision CSV or saved
vectors for evaluation. In the suppression row, `alpha` applies inside the
fixed region, not to the whole prior's probability mass.

Other calibration artifacts:

- [priors.npz](results/calibration/priors.npz): points, fixed true prior, free mask
  and 12 predicted priors, keyed by the CSV's `prior_id`.
- [protocol.json](results/calibration/protocol.json): fixed settings, error
  geometry, package versions, frozen source hashes and probability-vector hashes.
- [calibrated_priors.png](results/calibration/calibrated_priors.png) and
  [PDF](results/calibration/calibrated_priors.pdf): all families and levels on a
  shared color scale; the cross marks the true hotspot.

`--levels 0.05 0.15 0.30 --output /private/tmp/m4_example_calibration` writes the
example's available vectors and unmatched rows, then exits nonzero. It never
silently clips a requested JS level or passes an unmatched prior to planning.
Calibration reruns replace their own output artifacts; use a new `--output`
to retain an alternative design. M3 output paths are rejected.

## Small pilot

The pilot is a separate explicit command. It loads and verifies the saved
calibration against the fixed protocol before importing the unchanged planner:

```bash
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python -m milestone_4.run_pilot
```

This runs four families × three JS levels × two gamma values, plus one oracle
reference per gamma: **26 plans**. Each is evaluated on two paired target sets
(seeds 7 and 11, 64 targets each). The deterministic trajectory is reused across
target seeds. Seed 7 and the true prior exactly reproduce the frozen M3 inputs.
No formal experiment batch or scene expansion is run.

The default output is `results/pilot/`. The runner requires an empty directory;
use `--output milestone_4/results/pilot_repeat` for a repeat. It preserves
per-plan checkpoints and exits nonzero if any plan or final audit is invalid.

The saved pilot passed: **26/26 valid plans**, **3,328 episode records** and
**3,072 paired outcomes**. Maximum equality residual is below `9e-13`, maximum
inequality violation below `3e-10`. Pairing, episode counts and fixed endpoint
checks all pass in [pilot_audit.json](results/pilot/pilot_audit.json). The saved
oracle metrics for seed 7 also reproduce M3. These checks validate the pipeline;
they do not establish which error family is generally more harmful.

- `summary.csv`: per-condition, per-seed success/timeout counts and rates,
  successful-only mean/median detection time, coverage and secondary capped-time
  metrics. An invalid plan has zero evaluated episodes and blank search metrics.
- `episodes.csv` and `paired_outcomes.csv`: individual detections/timeouts and
  same-target oracle-versus-error comparisons.
- `coverage_timeseries.csv` and `coverage_summary.csv`: the original visibility
  union and true-prior mass coverage, once per plan.
- `planner_diagnostics.csv`, `trajectory_*.npz`, `trajectories_gamma_*.png/.pdf`:
  solver checks, states/controls and spatial trajectory comparisons.
- `inputs.npz`, `config.json`, `pilot_audit.json`: exact paired inputs, protocol
  snapshot and validation results.

## Pilot analysis and protocol v2 draft

Read the [pilot analysis report](results/analysis/REPORT.md) for per-gamma
comparisons, separate target-seed results, pooled episode statistics, paired
timing and spatial visibility gains/losses. Generate it from the saved pilot:

```bash
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python -m milestone_4.analyze_pilot
```

This command imports no planner and performs zero planning runs. It replays
visibility on the existing state trajectories and checks all 7,826 saved
coverage time samples and 3,328 episode detection records. Input files remain
unchanged; hashes and checks are saved in
[analysis_audit.json](results/analysis/analysis_audit.json).
It requires a complete valid v1 pilot and rejects missing/duplicate episodes,
inconsistent pairing, invalid plans or mismatched saved metrics. Reruns replace
analysis artifacts; `--output` can preserve an alternative analysis directory.

The outputs in `results/analysis/` include:

- `metrics_by_seed.csv` and `metrics_pooled.csv`: success/timeout and
  successful-only timing statistics, coverage and oracle contrasts. Pooled
  means and medians are calculated from raw episodes, not seed-summary averages.
- `paired_summary_by_seed.csv`, `paired_summary_pooled.csv`, and
  `paired_time_differences.csv`: four-outcome counts and error-minus-oracle
  timing differences restricted to the same targets found by both policies.
  Each common-success denominator is explicit; the subset differs by comparison.
- `mechanism_summary.csv` and `visibility_replay.npz`: gained/lost visible
  cells and true probability mass, plus incremental coverage after terminal
  arrival `(N-1)*tf/N`. Infinity encodes unseen cells in the NPZ only; CSV
  timeout detection times remain blank.
- `metrics_gamma_*`, `paired_gamma_*`, `coverage_maps_gamma_*`, and
  `mechanism_gamma_*` figures in PNG/PDF: metric curves, paired outcomes/timing,
  spatial visibility comparisons and coverage versus sampled success.

Two descriptive findings illustrate why these distinctions matter. At JS=0.10,
spatial shift changes pooled success relative to oracle by +5.47 percentage
points for gamma=0.10, but -13.28 points for gamma=0.05. At JS=0.15, gamma=0.05,
false hotspot adds 15.08 points of visible free-cell coverage while losing 6.54
points of true-prior mass coverage. These are conditional pilot observations,
not a general family ranking.

Under this grid distribution, ideal detector and shared observation times,
true-prior mass covered is the distribution's budgeted detection probability.
Its agreement with sampled success is not independent evidence that coverage
is the sole causal pathway. Two target seeds also do not provide replication
across scenes or optimization trials.

The [protocol v2 draft](PROTOCOL_V2_DRAFT.md) proposes a finite 12-scene suite,
two error-geometry blocks, common-JS preflight, paired contrasts and explicit
failure handling. It is **not frozen or executed**. Scene feasibility, JS
reachability, resources and target-sampling precision must be checked before
freezing a formal experiment manifest. Existing v1 protocol/code remain fixed.

## Validation

```bash
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python -m unittest discover milestone_4/tests -v
```

Tests cover matched JS, zero-error identity, legal support, distinct error
structures, unreachable-level rejection, unchanged M3 inputs, artifact/protocol
tampering and pairing across seeds (including zero-time detection and timeouts).
The pilot is the integration check against the actual planner. Its purpose is
to validate the controlled design and outputs before considering expansion.

## Frozen Protocol v2 formal suite

The authoritative [frozen manifest](results/frozen/PROTOCOL_V2_FROZEN_MANIFEST.json)
contains 12 scenes × 2 gamma values × 23 distinct prior conditions = **552 plans**.
The formal runner loads the saved prior vectors and checks their hashes, source
bytes, environment versions, JS values and full condition list before planning:

```bash
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python -m milestone_4.run_formal
```

Repeat the same command to resume. Completed valid and invalid attempts are
verified and skipped. A saved solver result resumes evaluation without planning;
an attempt interrupted before a durable solver result is recorded as interrupted
and is never retried. A lock prevents overlapping coordinators and remains held
by an orphaned solver worker until that worker exits. Each plan uses a fresh
process to bound JAX compilation memory; planner inputs, initialization and solver
settings stay frozen. `--validate-only` checks inputs with zero planner calls.

`results/formal/` preserves the manifest snapshot, source/input hashes, paired
inputs, progress, and one directory per condition under `plans/`. Each completed
plan has its attempt record, solver log/diagnostics, trajectory when returned by
the solver, exact primary metrics, uniform reweighting diagnostics, sampled
metrics and raw episodes. Valid plans also save every grid cell's first-visible
time. Invalid plans have no search outcomes; missing metrics remain null. Saved
artifact hashes protect checkpoint reuse.

After all attempts are recorded, run the full audit and then generate the report:

```bash
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python -m milestone_4.audit_formal
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python -m milestone_4.analyze_formal
```

These commands do not plan or tune. A faithfully recorded invalid attempt can
pass the integrity audit while preventing a claim of a complete valid matched
suite. Primary contrasts use exact detection probability and conditional weighted
detection time, average the two direction contrasts within scene, weight scenes
equally and keep gamma values separate. Reports include planned/available
condition denominators. Sampled metrics use only the frozen five seeds and 256
targets per seed as sanity checks; no population inference or post-hoc tuning is
part of this run.
