# Milestone 4 pilot analysis

This is a descriptive analysis of saved trajectories and paired targets.
No planner was imported or run. Four error families share JS levels 0.05, 0.10 and 0.15 nats.
The analysis verifies the saved calibration, episode summaries, pairings and every coverage time sample.

## How to read these results

- Each condition has 64 targets per seed (7 and 11); pooled results use all 128 raw episodes.
  Pooled successful-only means/medians are recomputed from detections, not averaged across seed summaries.
- Delta means error minus oracle. A positive success delta is a gain; a positive time delta is slower.
- Successful-only summaries can compare different target subsets. Paired timing uses only targets found
  by both policies, with its denominator reported; that common subset also changes across error comparisons.
- Coverage is measured once per deterministic plan. The two seeds are evaluation samples, not independent
  scenes or optimizer replicates. No confidence intervals, p-values or general family ranking are reported.
- On this fixed target grid with the ideal detector and shared observation times, true-prior mass covered
  equals the detection probability under the discrete true distribution. Agreement with sampled success
  is therefore an expected sampling relationship, not independent proof of a causal mediation mechanism.

## Observations in this pilot

1. **Compare the gamma strata separately.** For spatial shift at JS=0.10, pooled success
   differs from oracle by +5.47 percentage points at gamma=0.10,
   and -13.28 points at gamma=0.05.
   Both target seeds agree with their respective pooled contrast signs: True.
2. **Separate visible area from true probability.** For false hotspot at JS=0.15, gamma=0.05,
   free-cell coverage changes by +15.08 points, whereas true-prior
   mass covered changes by -6.54 points. Pooled success changes by
   -13.28 points. The maps show both new regions and lost regions.
3. **Inspect the JS response without imposing monotonicity.** For spatial
   shift at gamma=0.05, success across JS=0.05/0.10/0.15 is 67.19%/62.50%/71.09%.
4. **Common-success timing also changes.** Across the 24 pooled error-versus-oracle comparisons,
   mean paired time differences range from +0.053 to +1.286 seconds.
   Each comparison uses its own both-success subset; this is not a claim that every target is slower.
5. **Distribution-level coverage and sampled success are distinct.** Spatial shift at JS=0.15, gamma=0.05
   changes true-prior mass covered by +1.58 points while pooled sampled
   success changes by -4.69 points.
   These are different quantities: one integrates the fixed distribution, the other uses 128 draws.
   This motivates separating target-sampling uncertainty from scenario/geometry variation.

## gamma = 0.1

Oracle: success 83/128 (64.84%), successful-only mean/median 2.016/1.350 s, free-cell coverage 34.99%, true-prior mass covered 70.51%.

| Family | JS | Success % | Delta success pp | Successful-only mean / median s | Both-success n | Paired mean / median delta s | Coverage % | True mass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial shift | 0.05 | 70.31 | +5.47 | 2.396 / 2.050 | 82 | 0.392 / 0.050 | 39.82 | 74.37 |
| Spatial shift | 0.1 | 70.31 | +5.47 | 2.599 / 3.050 | 80 | 0.595 / 0.100 | 43.15 | 76.56 |
| Spatial shift | 0.15 | 74.22 | +9.38 | 3.075 / 3.250 | 80 | 1.135 / 0.200 | 51.44 | 82.08 |
| Diffuse / blur | 0.05 | 75.00 | +10.16 | 2.154 / 1.725 | 83 | 0.240 / 0.050 | 38.25 | 76.60 |
| Diffuse / blur | 0.1 | 74.22 | +9.38 | 2.743 / 2.900 | 83 | 0.883 / 0.700 | 40.80 | 76.58 |
| Diffuse / blur | 0.15 | 71.88 | +7.03 | 3.198 / 3.100 | 82 | 1.286 / 1.700 | 43.99 | 75.29 |
| False hotspot | 0.05 | 67.19 | +2.34 | 2.305 / 1.525 | 81 | 0.270 / 0.050 | 39.10 | 73.17 |
| False hotspot | 0.1 | 64.84 | +0.00 | 2.680 / 3.600 | 77 | 0.710 / 0.100 | 46.87 | 72.76 |
| False hotspot | 0.15 | 65.62 | +0.78 | 3.148 / 3.650 | 78 | 1.170 / 0.750 | 49.87 | 73.12 |
| False-negative suppression | 0.05 | 71.09 | +6.25 | 2.266 / 1.850 | 83 | 0.331 / 0.050 | 37.34 | 73.87 |
| False-negative suppression | 0.1 | 73.44 | +8.59 | 2.514 / 2.675 | 83 | 0.607 / 0.450 | 38.90 | 74.76 |
| False-negative suppression | 0.15 | 71.88 | +7.03 | 2.712 / 2.825 | 81 | 0.811 / 0.700 | 40.60 | 74.40 |

![Metric comparisons](metrics_gamma_0.1.png)

![Paired outcomes and timing](paired_gamma_0.1.png)

![Visibility gains and losses](coverage_maps_gamma_0.1.png)

![Coverage and sampled success](mechanism_gamma_0.1.png)

## gamma = 0.05

Oracle: success 97/128 (75.78%), successful-only mean/median 2.170/1.500 s, free-cell coverage 37.60%, true-prior mass covered 76.77%.

| Family | JS | Success % | Delta success pp | Successful-only mean / median s | Both-success n | Paired mean / median delta s | Coverage % | True mass % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial shift | 0.05 | 67.19 | -8.59 | 2.561 / 1.700 | 79 | 0.406 / 0.000 | 37.47 | 69.64 |
| Spatial shift | 0.1 | 62.50 | -13.28 | 2.630 / 1.575 | 73 | 0.482 / 0.050 | 38.71 | 65.76 |
| Spatial shift | 0.15 | 71.09 | -4.69 | 3.064 / 3.800 | 83 | 1.075 / 0.350 | 50.91 | 78.34 |
| Diffuse / blur | 0.05 | 80.47 | +4.69 | 2.477 / 1.850 | 97 | 0.267 / 0.050 | 41.32 | 82.31 |
| Diffuse / blur | 0.1 | 78.12 | +2.34 | 2.895 / 3.325 | 91 | 0.710 / 0.800 | 45.23 | 78.58 |
| Diffuse / blur | 0.15 | 64.84 | -10.94 | 3.437 / 3.650 | 75 | 1.231 / 1.650 | 47.85 | 70.84 |
| False hotspot | 0.05 | 73.44 | -2.34 | 2.501 / 1.425 | 86 | 0.281 / 0.000 | 45.89 | 77.39 |
| False hotspot | 0.1 | 66.41 | -9.38 | 2.590 / 1.450 | 76 | 0.399 / 0.000 | 49.93 | 72.82 |
| False hotspot | 0.15 | 62.50 | -13.28 | 3.110 / 1.725 | 71 | 0.987 / 0.050 | 52.68 | 70.23 |
| False-negative suppression | 0.05 | 79.69 | +3.91 | 2.266 / 1.850 | 97 | 0.053 / 0.050 | 39.75 | 78.99 |
| False-negative suppression | 0.1 | 76.56 | +0.78 | 2.651 / 3.150 | 90 | 0.480 / 0.625 | 42.75 | 78.05 |
| False-negative suppression | 0.15 | 71.88 | -3.91 | 2.883 / 3.300 | 83 | 0.723 / 0.800 | 44.45 | 74.08 |

![Metric comparisons](metrics_gamma_0.05.png)

![Paired outcomes and timing](paired_gamma_0.05.png)

![Visibility gains and losses](coverage_maps_gamma_0.05.png)

![Coverage and sampled success](mechanism_gamma_0.05.png)

## Mechanism diagnostics

`mechanism_summary.csv` separates newly visible cells from cells lost relative to oracle, weighting each
by the fixed true prior. It also splits mass first detected before versus at/after terminal-pose arrival.
Arrival is `(N-1)*tf/N`, not `tf`, under the frozen execution rule. The latter mass is incremental
coverage during the terminal hold, not a counterfactual estimate of the benefit of holding.

`visibility_replay.npz` retains first-visible times (infinity for unseen cells), allowing the spatial
gain/loss maps and decomposition to be checked without planning. The replay matches all saved coverage
time samples and episode detections. Area gain alone need not imply probability-mass gain.

## Seed sensitivity

`metrics_by_seed.csv` and `paired_summary_by_seed.csv` retain separate seed outcomes. The plots show
seed 7 dashed, seed 11 dotted and pooled solid. With only two target seeds, their spread is descriptive
and is not an uncertainty estimate across scenes or error geometries.

## Limits and next experiment

These comparisons establish condition-specific behavior under the frozen planner, camera and budget.
They do not establish universal error-family ordering, or coverage as the only pathway affecting timing.
The next deliverable is the [protocol v2 draft](../../PROTOCOL_V2_DRAFT.md): a prespecified scenario and
error-geometry set, per-block JS calibration, and paired contrasts with scenes as the unit of generalization.
Its feasibility and precision checks must precede freezing the formal experiment manifest.
