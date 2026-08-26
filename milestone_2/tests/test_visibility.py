"""Tests for the Milestone 2 visibility definition."""

from __future__ import annotations

import sys
import unittest
from math import pi
from pathlib import Path


MILESTONE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MILESTONE_DIR))

from visibility import (  # noqa: E402
    CameraModel,
    RectangleObstacle,
    first_detection_time,
    is_visible,
    line_of_sight_clear,
    point_in_fov,
    point_in_range,
    segment_intersects_rect,
)


class VisibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = CameraModel(sensing_radius=1.0, fov_degrees=90.0)
        self.target = (0.8, 0.0)

    def test_target_inside_sensing_radius(self) -> None:
        self.assertTrue(point_in_range((0.0, 0.0), self.target, radius=1.0))

    def test_target_outside_sensing_radius_is_not_visible(self) -> None:
        camera = CameraModel(sensing_radius=0.5, fov_degrees=90.0)
        self.assertFalse(is_visible((0.0, 0.0, 0.0), self.target, [], camera))

    def test_target_inside_field_of_view(self) -> None:
        self.assertTrue(point_in_fov((0.0, 0.0, 0.0), self.target, self.camera))

    def test_target_outside_field_of_view_is_not_visible(self) -> None:
        self.assertFalse(is_visible((0.0, 0.0, pi), self.target, [], self.camera))

    def test_clear_line_of_sight_is_visible(self) -> None:
        self.assertTrue(is_visible((0.0, 0.0, 0.0), self.target, [], self.camera))

    def test_occluding_rectangle_blocks_visibility(self) -> None:
        obstacle = RectangleObstacle(0.3, -0.1, 0.5, 0.1)
        self.assertTrue(segment_intersects_rect((0.0, 0.0), self.target, obstacle))
        self.assertFalse(line_of_sight_clear((0.0, 0.0), self.target, [obstacle]))
        self.assertFalse(is_visible((0.0, 0.0, 0.0), self.target, [obstacle], self.camera))

    def test_non_occluding_rectangle_does_not_block_visibility(self) -> None:
        obstacle = RectangleObstacle(0.3, 0.3, 0.5, 0.5)
        self.assertFalse(segment_intersects_rect((0.0, 0.0), self.target, obstacle))
        self.assertTrue(line_of_sight_clear((0.0, 0.0), self.target, [obstacle]))
        self.assertTrue(is_visible((0.0, 0.0, 0.0), self.target, [obstacle], self.camera))

    def test_first_detection_time_returns_first_visible_sample(self) -> None:
        trajectory = [
            (0.0, 0.0, 0.0, pi),
            (1.0, 0.0, 0.0, pi / 2.0),
            (2.0, 0.0, 0.0, 0.0),
            (3.0, 0.2, 0.0, 0.0),
        ]
        t_find, sample = first_detection_time(trajectory, self.target, [], self.camera)
        self.assertEqual(t_find, 2.0)
        self.assertEqual(sample, trajectory[2])

    def test_first_detection_time_returns_none_when_never_visible(self) -> None:
        trajectory = [
            (0.0, 0.0, 0.0, pi),
            (1.0, 0.0, 0.0, pi),
        ]
        t_find, sample = first_detection_time(trajectory, self.target, [], self.camera)
        self.assertIsNone(t_find)
        self.assertIsNone(sample)


if __name__ == "__main__":
    unittest.main()
