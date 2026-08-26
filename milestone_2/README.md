# Milestone 2: Visibility Simulator

Milestone 2 starts the thesis-specific visual-search side without using CLIP,
VLMs, neural detectors, or 3D rendering.

The first target is deliberately small:

```text
given a trajectory and a target, determine the first time the target is visible
```

The ideal detector is:

```text
detected(t) = 1 if target is visible from x(t), otherwise 0
```

Visibility requires:

```text
1. target is inside sensing radius
2. target is inside camera FOV
3. camera-target line of sight is not blocked by an obstacle
```

## Run

```bash
cd /Users/leofang/Desktop/active_visual_search_thesis
python milestone_2/run_demo.py
```

## Outputs

```text
milestone_2/figures/visibility_demo.pdf
milestone_2/figures/visibility_demo.png
```

This creates the task-level metric used later:

```text
T_find = first time that target enters the visible set V(x(t))
```

Milestone 3 will connect this simulator to planned trajectories and controlled
prior-error experiments.
