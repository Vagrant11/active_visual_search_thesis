# Stored-plan replay results

Constraints for all three tasks: frozen protocol/results and M3 read-only; no planner; no existing result overwrite. New files only under milestone_4/results/mechanism/.

546 valid formal plans + 26 valid pilot plans. Six invalid formal plans listed separately. No pilot uniform-prior plan exists; none was created.

## Definitions

- Grid: the stored 40×40 unit-workspace prior quadrature grid. The ergodic metric is spectral, not a grid-occupancy entropy.
- entropy_node_nats: Shannon entropy of the cell histogram of all 50 stored positions with weight 1/50 (the solver ck quadrature). Maximum is ln(50), not ln(1600). No support remasking of camera positions.
- entropy_dwell_tf_nats: exact piecewise-linear cell residence over [0,tf], including the last tf/N terminal hold.
- entropy_dwell_15s_nats: same residence over [0,15], including terminal hold or truncation. These are separate diagnostic definitions, not interchangeable estimates.
- Boundary cells: half-open intervals, x/y=1 assigned to the last cell; solver excursions <=1e-6 are clipped. Zero-probability cells contribute zero to entropy.
- path_length: sum of distances between stored nodes, unit-workspace length. camera_path_length_15s also reports execution truncated at the observation budget.
- visible_area: supported grid cells observed by 15 s × 1/1600, in unit-workspace area. This is camera-visible area, not the area occupied by the camera center. visible_free_fraction has supported area as denominator.
- ergodic_metric: NumPy replay of the source 8×8 cosine basis and spectral weights; every value checked against stored diagnostics to absolute tolerance 1e-9.
- Task 1 summaries weight valid plans equally, separating pilot/formal. Deltas are family minus oracle paired by scene and gamma; JS/direction details are retained. Missing invalid runs are excluded with counts.
- Task 3 averages x/y within scene before giving scenes equal weight. Incomplete x/y sets remain in per-scene output, explicitly flagged, and are excluded from across-scene summaries, matching the frozen analysis.
- Positive/negative net counts use tolerance 1e-10 pp. Masses are probabilities; gained/lost/net are percentage points.

## Task 1: paired mean deltas (family minus oracle)

| Dataset | gamma | Family | n plans | Δtf s | ΔH nodes nats | ΔH dwell tf nats |
|---|---:|---|---:|---:|---:|---:|
| formal | 0.05 | uniform_prior_baseline | 12 | -1.199961 | +0.028598 | +0.048512 |
| formal | 0.05 | diffuse_blur | 36 | -1.033414 | +0.020039 | +0.035826 |
| formal | 0.05 | spatial_shift | 72 | -0.398091 | -0.016007 | -0.045298 |
| formal | 0.05 | false_hotspot | 72 | -0.408320 | +0.034454 | +0.082400 |
| formal | 0.05 | false_negative_suppression | 71 | -1.081793 | +0.010193 | -0.004970 |
| formal | 0.1 | uniform_prior_baseline | 12 | -0.457055 | +0.080040 | +0.008035 |
| formal | 0.1 | diffuse_blur | 36 | -0.578551 | +0.047321 | -0.048094 |
| formal | 0.1 | spatial_shift | 68 | -0.189487 | +0.024018 | -0.054080 |
| formal | 0.1 | false_hotspot | 72 | -0.233730 | +0.059608 | +0.033879 |
| formal | 0.1 | false_negative_suppression | 71 | -0.802542 | -0.003894 | -0.108830 |
| pilot | 0.1 | spatial_shift | 3 | +0.126147 | +0.046210 | +0.144336 |
| pilot | 0.1 | diffuse_blur | 3 | +0.104530 | +0.073936 | +0.179675 |
| pilot | 0.1 | false_hotspot | 3 | +0.721268 | +0.064694 | +0.353794 |
| pilot | 0.1 | false_negative_suppression | 3 | -0.230430 | +0.046210 | +0.053655 |
| pilot | 0.05 | spatial_shift | 3 | +0.074758 | +0.006977 | +0.014245 |
| pilot | 0.05 | diffuse_blur | 3 | -0.179450 | +0.069405 | +0.193890 |
| pilot | 0.05 | false_hotspot | 3 | +0.421378 | +0.065917 | +0.302355 |
| pilot | 0.05 | false_negative_suppression | 3 | -0.579152 | +0.020930 | +0.090868 |

Means/medians of every requested diagnostic: diagnostic_summary.csv; JS strata: diagnostic_summary_by_js.csv. Gamma source quotes: gamma_semantics.md.

## Task 2: pilot net visibility differences

Both gamma=0.05 figures already existed in the original directories. Both gamma slices were rendered in pilot_figures using the original plotting function / unchanged coverage-map statements. Colour meanings, annotations and styling are unchanged.

| Family | JS | gained .10 pp | lost .10 pp | net .10 pp | gained .05 pp | lost .05 pp | net .05 pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial_shift | 0.05 | 5.4709 | 1.6117 | 3.8592 | 4.4192 | 11.5471 | -7.1279 |
| spatial_shift | 0.1 | 7.3463 | 1.2924 | 6.0539 | 5.5474 | 16.5484 | -11.0010 |
| spatial_shift | 0.15 | 12.4986 | 0.9213 | 11.5774 | 8.1846 | 6.6075 | 1.5771 |
| diffuse_blur | 0.05 | 6.7453 | 0.6479 | 6.0974 | 5.5442 | 0.0000 | 5.5442 |
| diffuse_blur | 0.1 | 6.1267 | 0.0493 | 6.0774 | 6.7380 | 4.9263 | 1.8118 |
| diffuse_blur | 0.15 | 6.1444 | 1.3647 | 4.7798 | 7.4021 | 13.3313 | -5.9293 |
| false_hotspot | 0.05 | 3.2428 | 0.5800 | 2.6628 | 7.5642 | 6.9401 | 0.6241 |
| false_hotspot | 0.1 | 6.1704 | 3.9135 | 2.2569 | 8.8102 | 12.7517 | -3.9414 |
| false_hotspot | 0.15 | 5.7788 | 3.1680 | 2.6108 | 9.3964 | 15.9330 | -6.5366 |
| false_negative_suppression | 0.05 | 3.8253 | 0.4574 | 3.3678 | 3.5944 | 1.3665 | 2.2279 |
| false_negative_suppression | 0.1 | 4.9331 | 0.6817 | 4.2514 | 5.6707 | 4.3816 | 1.2891 |
| false_negative_suppression | 0.15 | 5.8162 | 1.9201 | 3.8961 | 6.6468 | 9.3311 | -2.6843 |

## Task 3: complete-direction scene results

| gamma | Family | JS | complete / 12 | positive | negative | mean net pp | median net pp | positive all 12 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.05 | uniform_prior_baseline | None | 12/12 | 1 | 11 | -25.1225 | -29.1860 | False |
| 0.05 | diffuse_blur | 0.05 | 12/12 | 8 | 4 | +0.0718 | +2.8862 | False |
| 0.05 | diffuse_blur | 0.1 | 12/12 | 4 | 8 | -8.9874 | -10.6827 | False |
| 0.05 | diffuse_blur | 0.15 | 12/12 | 0 | 12 | -17.5792 | -21.1757 | False |
| 0.05 | spatial_shift | 0.05 | 12/12 | 3 | 9 | -2.4200 | -3.3249 | False |
| 0.05 | spatial_shift | 0.1 | 12/12 | 0 | 12 | -5.7801 | -5.0819 | False |
| 0.05 | spatial_shift | 0.15 | 12/12 | 0 | 12 | -7.0825 | -6.4557 | False |
| 0.05 | false_hotspot | 0.05 | 12/12 | 4 | 8 | -1.9529 | -0.7381 | False |
| 0.05 | false_hotspot | 0.1 | 12/12 | 3 | 9 | -3.9714 | -1.9456 | False |
| 0.05 | false_hotspot | 0.15 | 12/12 | 1 | 11 | -10.4075 | -12.5230 | False |
| 0.05 | false_negative_suppression | 0.05 | 12/12 | 4 | 8 | -2.2594 | -1.4908 | False |
| 0.05 | false_negative_suppression | 0.1 | 11/12 | 6 | 5 | -2.1973 | +0.0764 | False |
| 0.05 | false_negative_suppression | 0.15 | 12/12 | 5 | 7 | -3.4280 | -3.8560 | False |
| 0.1 | uniform_prior_baseline | None | 12/12 | 2 | 10 | -28.8804 | -29.1190 | False |
| 0.1 | diffuse_blur | 0.05 | 12/12 | 4 | 8 | -3.9599 | -2.8679 | False |
| 0.1 | diffuse_blur | 0.1 | 12/12 | 3 | 9 | -12.9095 | -8.3632 | False |
| 0.1 | diffuse_blur | 0.15 | 12/12 | 2 | 10 | -21.5389 | -17.1431 | False |
| 0.1 | spatial_shift | 0.05 | 11/12 | 5 | 6 | -1.9495 | -1.4549 | False |
| 0.1 | spatial_shift | 0.1 | 10/12 | 2 | 8 | -8.7998 | -6.3432 | False |
| 0.1 | spatial_shift | 0.15 | 11/12 | 2 | 9 | -15.8553 | -12.9767 | False |
| 0.1 | false_hotspot | 0.05 | 12/12 | 5 | 7 | -5.3378 | -2.5493 | False |
| 0.1 | false_hotspot | 0.1 | 12/12 | 4 | 8 | -9.5810 | -9.6178 | False |
| 0.1 | false_hotspot | 0.15 | 12/12 | 0 | 12 | -15.6609 | -12.8229 | False |
| 0.1 | false_negative_suppression | 0.05 | 12/12 | 3 | 9 | -4.8857 | -6.0501 | False |
| 0.1 | false_negative_suppression | 0.1 | 12/12 | 3 | 9 | -7.1515 | -9.8094 | False |
| 0.1 | false_negative_suppression | 0.15 | 11/12 | 2 | 9 | -7.8044 | -11.4178 | False |

Flag: the gamma=0.10 pilot pattern of positive net mass for every error family/JS holds across all 12 scenes: **False**.
This flag concerns exact true-prior mass, not sampled target success percentages. No causal or hypothesis interpretation is supplied.

## Reproduction

Run from repository root with PYTHONDONTWRITEBYTECODE=1 and /opt/miniconda3/envs/erg/bin/python -B milestone_4/results/mechanism/replay_analysis.py. Each run creates a new timestamped directory and refuses to reuse an existing directory.
Unit tests: /opt/miniconda3/envs/erg/bin/python -B -m unittest discover -s milestone_4/results/mechanism -p test_replay.py -v.
