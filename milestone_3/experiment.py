"""Target distributions, target-blind camera execution, and paired evaluation."""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "milestone_2"))
from visibility import CameraModel, RectangleObstacle, is_visible, segment_intersects_rect


@dataclass(frozen=True)
class Environment:
    obstacles: tuple = (RectangleObstacle(0.43, 0.43, 0.57, 0.57),)
    camera: CameraModel = CameraModel(0.25, 90.0)
    budget: float = 15.0
    observation_dt: float = 0.05
    scan_rate: float = np.pi / 2
    clearance: float = 0.015

    @property
    def collision_disks(self):
        # Conservative enclosing circles give smooth segment-clearance constraints.
        return np.array([
            ((o.xmin + o.xmax) / 2, (o.ymin + o.ymax) / 2,
             np.hypot(o.xmax - o.xmin, o.ymax - o.ymin) / 2 + self.clearance)
            for o in self.obstacles
        ]).reshape(-1, 3)


def make_priors(environment, grid_size=40):
    axis = (np.arange(grid_size) + 0.5) / grid_size
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    free = np.ones(len(points), dtype=bool)
    for cx, cy, radius in environment.collision_disks:
        free &= np.linalg.norm(points - [cx, cy], axis=1) > radius

    def density(center):
        values = np.exp(-np.sum((points - center) ** 2, axis=1) / (2 * 0.13**2))
        values = (values + 0.03) * free
        return values / values.sum()

    return points, {"oracle": density([0.75, 0.70]), "biased": density([0.25, 0.70])}


def js_divergence(p, q):
    """Discrete Jensen-Shannon divergence in nats, with exact zero handling."""
    p, q = np.asarray(p), np.asarray(q)
    midpoint = (p + q) / 2
    def kl(a):
        keep = a > 0
        return np.sum(a[keep] * np.log(a[keep] / midpoint[keep]))
    return float((kl(p) + kl(q)) / 2)


def camera_trajectory(states, tf, environment):
    """Execute reference Euler nodes and a target-independent constant-rate pan.

    Upstream uses N states with dt=tf/N, hence the last node is at tf-dt.
    Hold the terminal (zero-velocity) pose for the last interval and thereafter
    until the common budget. Observations occur on one shared fixed time grid.
    """
    states = np.asarray(states)
    if not np.isfinite(states).all() or not np.isfinite(tf) or tf <= 0:
        raise ValueError("Cannot execute nonfinite states or nonpositive duration")
    if environment.budget <= 0 or environment.observation_dt <= 0:
        raise ValueError("Budget and observation interval must be positive")
    times = np.arange(0, environment.budget, environment.observation_dt)
    times = np.append(times, environment.budget)
    node_times = np.arange(len(states)) * tf / len(states)
    xy = np.column_stack([np.interp(times, node_times, states[:, i]) for i in (0, 1)])
    theta = (environment.scan_rate * times + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack((times, xy, theta))


def trajectory_is_collision_free(states, environment):
    xy = np.asarray(states)[:, :2]
    return not any(segment_intersects_rect(tuple(a), tuple(b), obstacle)
                   for a, b in zip(xy[:-1], xy[1:]) for obstacle in environment.obstacles)


def evaluate(policy, environment, targets):
    """Evaluate a planned policy on paired targets; hidden targets never reach it.

    Failed plans are reported separately and excluded from task statistics.
    Timeout detection times are None; capped time is min(T_find, budget).
    """
    if not policy["diagnostics"]["valid"]:
        return [], {"status": "invalid_plan", "episodes": 0}
    trajectory = camera_trajectory(policy["states"], policy["tf"], environment)
    # Integrate length at original knots, not chords between observation samples.
    nodes = np.asarray(policy["states"])[:, :2]
    node_times = np.arange(len(nodes)) * policy["tf"] / len(nodes)
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
    rows = []
    for episode, target in enumerate(targets):
        found = next((float(t) for t, x, y, theta in trajectory
                      if is_visible((x, y, theta), tuple(target), list(environment.obstacles),
                                    environment.camera)), None)
        stop = found if found is not None else environment.budget
        rows.append({"episode": episode, "target_x": float(target[0]),
                     "target_y": float(target[1]), "success": found is not None,
                     "t_find": found, "capped_time": stop,
                     "path_length_until_stop": float(np.interp(stop, node_times, lengths))})
    successful = [r["t_find"] for r in rows if r["success"]]
    capped = [r["capped_time"] for r in rows]
    return rows, {
        "status": "evaluated", "episodes": len(rows),
        "success_rate": len(successful) / len(rows) if rows else None,
        "mean_capped_time": float(np.mean(capped)) if rows else None,
        "median_capped_time": float(np.median(capped)) if rows else None,
        "mean_t_find_success_only": float(np.mean(successful)) if successful else None,
        "median_t_find_success_only": float(np.median(successful)) if successful else None,
        "mean_path_length_until_stop": float(np.mean([r["path_length_until_stop"] for r in rows])) if rows else None,
    }
