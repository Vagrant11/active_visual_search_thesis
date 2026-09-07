# Milestone 3: Prior-to-search evaluation

This milestone connects a supplied spatial prior to a planned trajectory, a
target-independent camera scan, and first-detection evaluation. The first
experiment compares an oracle distribution with a spatially biased distribution
on exactly the same 64 target realizations and two ergodic bounds.

## Run

From the repository root, using the existing `erg` conda environment:

```bash
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python -m milestone_3.run_experiment

/opt/miniconda3/envs/erg/bin/python -m unittest discover milestone_3/tests -v
/opt/miniconda3/envs/erg/bin/python -m unittest discover milestone_2/tests -v
```

The default run takes about 40 seconds after imports on the development machine.
Use `--help` for options. For example, `--seed 11 --episodes 128 --output
milestone_3/results_seed11` retains the default results while changing targets.
Reusing an output directory replaces its generated experiment files.

To regenerate diagnostics from a completed run without calling the planner:

```bash
/opt/miniconda3/envs/erg/bin/python -m milestone_3.diagnostics \
  --results milestone_3/results --gamma 0.1
```

## Experiment definition

- Workspace: unit square; one stationary rectangular occluder.
- Robot state: `(px, py, vx, vy)` with the reference forward-Euler double-integrator
  dynamics; controls are accelerations with component bounds `[-1, 1]`.
- Start/end: `(0.1, 0.1, 0, 0)` and `(0.9, 0.9, 0, 0)`, identical for every run.
- Added constraints: hard workspace bounds, speed norm <= 1, positive duration,
  and collision avoidance for every complete straight segment between nodes.
  Obstacles are conservatively enclosed by circles with 0.015 extra clearance.
  Visibility still uses the actual rectangles from Milestone 2.
- Camera: range 0.25, 90-degree FoV, independent pan `theta(t) = pi*t/2` modulo
  `2*pi`, initially pointing along +x. No target coordinates enter planning or
  camera execution. The pan has fixed angular speed; it is not optimized.
- Detector: ideal, using range, FoV and unoccluded line of sight.
- Targets: sampled from a normalized probability vector on a 40x40 cell-center
  grid. Cells inside the conservative collision disks have zero probability.
  The remaining free space is connected; every supported target has a reachable
  observing pose. Targets may already be visible at time zero, which is success.
- Oracle prior: Gaussian hotspot centered at `(0.75, 0.70)`, sigma 0.13, plus
  background density 0.03, masked and normalized. This is the true sampling
  distribution, not a delta function revealing the realized target.
- Biased prior: the same construction centered at `(0.25, 0.70)`.
- Fixed priors: no posterior updates or replanning. The target is sampled only
  for evaluation and is never passed to `plan`.

The experiment measures Jensen-Shannon divergence in nats on these probability
vectors. It is **not yet a matched-error comparison of multiple error families**,
nor a calibration or learned-model experiment.

## Reference planner adapter

`planner.py` imports the authors' `experiments/comparison_study/build_solver.py`.
It reuses the reference objective, dynamics equalities, 8x8 Fourier basis,
frequency weights and ergodic/control inequalities. After construction it
replaces the builder's default uniform Fourier coefficients with the supplied
prior coefficients. The primary objective remains minimum terminal time with
`E <= gamma`; the upstream workspace penalty is retained alongside hard bounds.

The optimization backend is **SciPy SLSQP with JAX 64-bit derivatives**, not the
upstream augmented-Lagrangian iteration. This deliberate local change follows
Milestone 1's unsuccessful/nonfinite stopping reports. The upstream source is
unchanged. Results are local numerical solutions, not global-optimality proofs.
Every prior/gamma uses the same initial guess, with no warm start between runs.

A plan enters search evaluation only when the solver reports success, all
values are finite, maximum equality residual <= 1e-5, maximum inequality
violation <= 1e-6, and explicit rectangle segment checks find no collision.
Solver success, feasibility and task success are separate output fields.
An invalid plan produces diagnostics but no target-search statistics; the CLI
writes the available outputs and exits nonzero if any plan is invalid.

## Timing and metrics

The reference has `N` states but uses `dt=tf/N`. Therefore state `i` is at
`i*tf/N`, and the final state is at `(N-1)*tf/N`. Execution linearly interpolates
the position nodes and holds the zero-velocity terminal pose for the last
interval. This is execution of the discretized path, not an exact continuous
double-integrator rollout. After `tf`, the robot remains at that pose and the
camera continues scanning until the common 15-second budget. If `tf` exceeds
the budget, evaluation uses only the path prefix.

Observations use a common 0.05-second grid plus the budget endpoint. Detection
times are sampled first visibility, not exact continuous event times; brief
visibility windows can be missed. Position, speed and acceleration constraints
are planner discretization checks; segment collision checks cover the entire
executed position path.

`evaluate(policy, environment, targets)` returns per-target rows and a summary:

- `t_find`: first detection time, or blank/None for a timeout.
- `success_rate`: fraction of all episodes detected within budget.
- `mean_capped_time`, `median_capped_time`: statistics of `min(T_find, budget)`;
  unsuccessful episodes contribute the full budget.
- `*_t_find_success_only`: explicitly conditional on successful episodes.
- `path_length_until_stop`: distance until detection or budget, integrated along
  original trajectory knots rather than chords across observation samples.

The capped mean is not an estimate of unrestricted expected first-detection
time. The median capped time is also not an unrestricted median when it reaches
the budget. Planning wall time is reported separately, both with compilation
and optimization only; it is not added to simulated search time.

## Outputs and initial result

`results/` contains configuration and dependency versions (`config.json`), paired
targets and prior grids (`inputs.npz`), all four state/control trajectories,
per-episode records (`episodes.csv`), aggregates (`summary.csv`), solver checks
(`planner_diagnostics.csv`) and the PNG/PDF comparison figure. It also contains
`diagnostic_summary.csv` (success/failure counts and uncensored successful-only
Tfind statistics), `fairness_audit.json` (paired-target and fixed-protocol
checks), and `diagnostic_cases_gamma_*.png/.pdf`. The latter shows representative
Oracle/Biased successes and failures: target, robot trajectory, obstacle, scan
FoV samples, and the first-detection pose/FoV when one exists.
`paired_outcomes.csv` and `paired_target_outcomes_gamma_*.png/.pdf` make the
same-target comparison explicit: each target is labelled as found by both,
Oracle only, Biased only, or neither.

`coverage_timeseries.csv`, `coverage_summary.csv`, and
`coverage_diagnostics.png/.pdf` quantify the mechanism without replanning. They
measure the union of target-grid cells visible at the standard observation times
under the same range, FoV, and occlusion predicate as detection. The reported
free-cell fraction is a discrete-grid coverage estimate; the true-prior mass is
the oracle target-sampling probability contained in that visible union.

Default run, seed 7, one scene, 64 paired targets, budget 15 seconds:

| Prior | gamma | Optimized tf | Success rate | Mean capped time | Median capped time |
|---|---:|---:|---:|---:|---:|
| Oracle | 0.10 | 3.73 | 68.75% | 5.95 | 2.00 |
| Biased | 0.10 | 6.33 | 84.38% | 5.84 | 3.65 |
| Oracle | 0.05 | 4.86 | 78.13% | 4.91 | 1.88 |
| Biased | 0.05 | 7.50 | 89.06% | 5.76 | 3.88 |

All four solves pass validation. Maximum equality residual is below 6e-13 and
maximum inequality violation below 9e-11 in the saved run.

Oracle achieves lower median detection time here, while biased trajectories
achieve higher budgeted success. At gamma=0.10, the biased prior even has a
slightly lower capped mean. These are single-scene pilot observations, not
evidence that incorrect priors generally help. The position-visitation ergodic
objective does not account for FoV orientation or visibility, and fixed camera
phase, fixed terminal pose and post-plan holding can all influence these results.
The oracle prior therefore does not imply an oracle search policy.

The coverage diagnostics support a narrower mechanism statement: **this
particular spatially biased prior induces broader coverage in this setting**.
At gamma=0.10, free-cell coverage rises from 34.99% to 73.69%, and true-prior
mass covered from 70.51% to 86.16%; at gamma=0.05, the corresponding values are
37.60% to 75.07% and 76.77% to 91.32%. These measurements do not establish that
incorrect priors generally induce broader coverage.

This Milestone 3 snapshot preserves the current figures, CSVs and saved runs;
subsequent controlled-error work belongs in `milestone_4/`.

## Next experiment

Milestone 4 first fixes the ground-truth distribution, target sampling, camera,
obstacles, budget, planner and evaluator, changing only the predicted prior
passed to the planner. Implement spatial shift, diffuse/blur, false hotspot and
false-negative suppression, numerically matched at shared attainable JS levels
in nats. Validate calibration before a small paired pilot; defer formal batches
and multi-scene expansion until those checks pass.
