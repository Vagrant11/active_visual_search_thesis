"""Visibility utilities for the simplified 2D active visual search simulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin


EPS = 1e-9


@dataclass(frozen=True)
class CameraModel:
    """Planar pinhole-style camera model."""

    sensing_radius: float
    fov_degrees: float

    @property
    def half_fov_radians(self) -> float:
        return self.fov_degrees * pi / 360.0


@dataclass(frozen=True)
class RectangleObstacle:
    """Axis-aligned rectangular occluder."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains_point(self, point: tuple[float, float]) -> bool:
        x, y = point
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


def wrap_to_pi(angle: float) -> float:
    """Map an angle to [-pi, pi]."""
    return (angle + pi) % (2.0 * pi) - pi


def point_in_range(camera_xy: tuple[float, float], target: tuple[float, float], radius: float) -> bool:
    dx = target[0] - camera_xy[0]
    dy = target[1] - camera_xy[1]
    return hypot(dx, dy) <= radius + EPS


def point_in_fov(
    camera_pose: tuple[float, float, float],
    target: tuple[float, float],
    camera: CameraModel,
) -> bool:
    x, y, theta = camera_pose
    bearing = atan2(target[1] - y, target[0] - x)
    return abs(wrap_to_pi(bearing - theta)) <= camera.half_fov_radians + EPS


def segment_intersects_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: RectangleObstacle,
) -> bool:
    """Return True when the line segment from start to end crosses rect."""
    if rect.contains_point(start) or rect.contains_point(end):
        return True

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    t_min = 0.0
    t_max = 1.0

    for start_coord, delta, lower, upper in (
        (start[0], dx, rect.xmin, rect.xmax),
        (start[1], dy, rect.ymin, rect.ymax),
    ):
        if abs(delta) < EPS:
            if start_coord < lower or start_coord > upper:
                return False
            continue

        inv_delta = 1.0 / delta
        t1 = (lower - start_coord) * inv_delta
        t2 = (upper - start_coord) * inv_delta
        t_enter = min(t1, t2)
        t_exit = max(t1, t2)
        t_min = max(t_min, t_enter)
        t_max = min(t_max, t_exit)
        if t_min > t_max:
            return False

    return True


def line_of_sight_clear(
    camera_xy: tuple[float, float],
    target: tuple[float, float],
    obstacles: list[RectangleObstacle],
) -> bool:
    return not any(segment_intersects_rect(camera_xy, target, obstacle) for obstacle in obstacles)


def is_visible(
    camera_pose: tuple[float, float, float],
    target: tuple[float, float],
    obstacles: list[RectangleObstacle],
    camera: CameraModel,
) -> bool:
    """Check range, field of view, and occlusion for an ideal detector."""
    camera_xy = (camera_pose[0], camera_pose[1])
    return (
        point_in_range(camera_xy, target, camera.sensing_radius)
        and point_in_fov(camera_pose, target, camera)
        and line_of_sight_clear(camera_xy, target, obstacles)
    )


def first_detection_time(
    trajectory: list[tuple[float, float, float, float]],
    target: tuple[float, float],
    obstacles: list[RectangleObstacle],
    camera: CameraModel,
) -> tuple[float | None, tuple[float, float, float, float] | None]:
    """Return the first trajectory sample that sees the target."""
    for sample in trajectory:
        t, x, y, theta = sample
        if is_visible((x, y, theta), target, obstacles, camera):
            return t, sample
    return None, None


def fov_boundary_points(
    camera_pose: tuple[float, float, float],
    camera: CameraModel,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Endpoint rays used only for plotting the camera field of view."""
    x, y, theta = camera_pose
    angles = (theta - camera.half_fov_radians, theta + camera.half_fov_radians)
    return tuple(
        (x + camera.sensing_radius * cos(angle), y + camera.sensing_radius * sin(angle))
        for angle in angles
    )
