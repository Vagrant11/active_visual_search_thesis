# Active Visual Search Thesis

This repository contains Leo Fang's thesis experiments for active visual search
with ergodic planning.

The public time-optimal ergodic search implementation by Dong, Berger, and
Abraham is kept as a git submodule under:

```text
external/time_optimal_ergodic_search/
```

This keeps the thesis code separate from the upstream reference implementation.

## Milestones

```text
milestone_1/
    Reproduces the qualitative gamma-vs-optimized-time behavior using the
    authors' public time-optimal ergodic planner.

milestone_2/
    Defines a simple 2D visibility simulator and computes first detection time
    for a target along a known camera trajectory.

milestone_3/
    Connects the reference time-optimal ergodic formulation to target-blind
    camera execution and paired oracle-versus-biased-prior evaluation.
```

## Setup

After cloning this repository, initialize the upstream baseline code:

```bash
git submodule update --init --recursive
```

Milestone 1 currently uses the local conda environment that has JAX installed:

```bash
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python milestone_1/run_gamma_sweep.py
```

Run the new [Milestone 3 experiment](milestone_3/README.md):

```bash
env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp/xdg-cache \
  /opt/miniconda3/envs/erg/bin/python -m milestone_3.run_experiment
```
