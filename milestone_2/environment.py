"""Simple 2D scene containers for Milestone 2."""

from __future__ import annotations

from dataclasses import dataclass

from visibility import CameraModel, RectangleObstacle


@dataclass(frozen=True)
class Scene:
    workspace: tuple[float, float, float, float]
    target: tuple[float, float]
    obstacles: list[RectangleObstacle]
    camera: CameraModel
    trajectory: list[tuple[float, float, float, float]]
