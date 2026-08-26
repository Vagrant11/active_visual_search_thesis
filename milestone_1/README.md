# Milestone 1: Gamma Sweep Reproduction

This directory turns the first control-side reproduction into a repeatable
experiment. It uses the public time-optimal ergodic search implementation from
`external/time_optimal_ergodic_search/experiments/comparison_study/build_solver.py`.

## Goal

Reproduce the reference behavior:

```text
target distribution phi -> optimizer -> trajectory
change gamma -> optimized terminal time and trajectory change
```

Here `gamma` is the ergodicity upper bound. Smaller `gamma` means a stricter
ergodic constraint, so the trajectory must better match the spatial statistics
of `phi`. In the reference setup `phi` is uniform over the unit square.

## Run

Use the conda environment that already has JAX installed:

```bash
cd /Users/leofang/Desktop/active_visual_search_thesis
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python milestone_1/run_gamma_sweep.py
```

## Outputs

```text
milestone_1/
├── run_gamma_sweep.py
├── results/
│   ├── gamma_sweep.csv
│   └── gamma_sweep_solutions.pkl
├── figures/
│   ├── gamma_vs_tf.pdf
│   ├── gamma_vs_tf.png
│   ├── trajectories.pdf
│   └── trajectories.png
└── README.md
```

The CSV records:

```text
gamma
optimized_tf
measured_ergodicity
solver_success
constraint_satisfied
solver_message
max_abs_control
time_normalized_control_cost
runtime_seconds
```

## Interpretation

The key check is not only whether the solver prints `done in ... iterations`.
The current augmented Lagrangian implementation can print `unsuccessful` when
its numerical stopping tolerance is not reached, even when the final trajectory
satisfies the ergodic constraint:

```text
solver reports unsuccessful
does not necessarily mean
E(x, phi, tf) > gamma
```

For reporting, use `constraint_satisfied` and `measured_ergodicity <= gamma` to
separate feasibility from the solver's stopping criterion.

The expected qualitative result is:

```text
smaller gamma -> stricter ergodicity requirement -> larger optimized tf
```

Current reproduction output:

```text
gamma   optimized_tf   measured_ergodicity   solver_success   constraint_satisfied
0.100   4.602          0.09999               False            True
0.080   4.958          0.07164               False            True
0.050   5.456          0.04132               True             True
0.020   6.646          0.01352               False            True
0.010   7.454          0.00927               False            True
0.005   9.792          0.00476               False            True
```

As `gamma` is tightened, the optimized trajectory takes longer and covers more
of the unit square before reaching the terminal state. This is the expected
time-optimal ergodic search behavior: a stricter match to `phi` costs time.

This completes the control-side base needed before moving to biased or noisy
visual-search priors.
