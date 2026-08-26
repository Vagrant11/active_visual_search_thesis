"""A hand-built scene for testing first detection time."""

from __future__ import annotations

from math import atan2

import numpy as np

from environment import Scene
from visibility import CameraModel, RectangleObstacle


def build_scene(num_steps: int = 80) -> Scene:
    workspace = (0.0, 0.0, 1.0, 1.0)
    target = (0.84, 0.62)
    obstacles = [
        RectangleObstacle(0.38, 0.42, 0.62, 0.70),
    ]
    camera = CameraModel(sensing_radius=0.45, fov_degrees=70.0)

    xs = np.linspace(0.10, 0.88, num_steps)
    ys = 0.18 + 0.18 * np.sin(np.linspace(0.0, np.pi, num_steps))
    trajectory = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        t = float(i) / 10.0
        theta = atan2(target[1] - y, target[0] - x)
        trajectory.append((t, float(x), float(y), float(theta)))

    return Scene(
        workspace=workspace,
        target=target,
        obstacles=obstacles,
        camera=camera,
        trajectory=trajectory,
    )
