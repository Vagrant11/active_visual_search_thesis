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

Run the visibility checks:

```bash
python -m unittest discover milestone_2/tests
```

## Outputs

```text
milestone_2/figures/visibility_demo.pdf
milestone_2/figures/visibility_demo.png
```

Current demo output:

```text
T_find = 4.6
first_detection_sample = (4.6, 0.5541772151898734, 0.35402008058244666, 0.7494535789350105)
```

This creates the task-level metric used later:

```text
T_find = first time that target enters the visible set V(x(t))
```

## Checkpoint Status

M2.1 visibility is covered by unit tests for:

```text
range limit
field-of-view limit
occlusion by an axis-aligned rectangle
clear line of sight
```

M2.2 first detection time is covered by tests for both visible and never-visible
trajectories. The demo figure shows the obstacle, target, trajectory, first
detection point, and camera FOV at detection.

Milestone 3 will connect this simulator to planned trajectories and controlled
prior-error experiments.
