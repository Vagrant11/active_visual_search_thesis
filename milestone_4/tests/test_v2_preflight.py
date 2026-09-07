"""Gate 2: evaluator semantics on synthetic cases, generalized to v2 geometries.

No planner import, zero planning. `milestone_3.experiment.evaluate`/`is_visible`
are frozen and reused unchanged for v2; these tests check that reuse is sound
on synthetic trajectories/targets/obstacles, including the two-rectangle
Layout C geometry that v1 never exercised (v1 has exactly one obstacle).
"""

import unittest

import numpy as np

from milestone_3.experiment import Environment, evaluate
from visibility import is_visible, line_of_sight_clear, CameraModel, RectangleObstacle
from milestone_4.scene_manifest import LAYOUTS, scenes, scene_environment, scene_grid, gaussian_density, SIGMA, BACKGROUND
from milestone_4.analyze_pilot import first_seen_grid


def straight_line_policy(start, end, tf):
    states = np.array([[*start, 0.0, 0.0], [*end, 0.0, 0.0]])
    return {"states": states, "tf": tf, "diagnostics": {"valid": True}}


class EvaluatorSemanticsSyntheticTests(unittest.TestCase):
    """No obstacles: exact, hand-derivable outcomes."""

    def test_close_aligned_target_found_at_t_zero(self):
        # theta(0) = 0 (camera faces +x at t=0); target on that bearing, well within range.
        policy = straight_line_policy((0.1, 0.1), (0.9, 0.9), tf=4.0)
        environment = Environment(obstacles=())
        rows, summary = evaluate(policy, environment, [(0.15, 0.1)])
        self.assertTrue(rows[0]["success"])
        self.assertEqual(rows[0]["t_find"], 0.0)
        self.assertEqual(rows[0]["capped_time"], 0.0)
        self.assertEqual(summary["success_rate"], 1.0)

    def test_always_out_of_range_target_times_out(self):
        # Perpendicular distance from (0.95, 0.05) to the y=x path is ~0.636, past sensing_radius=0.25.
        policy = straight_line_policy((0.1, 0.1), (0.9, 0.9), tf=4.0)
        environment = Environment(obstacles=())
        rows, summary = evaluate(policy, environment, [(0.95, 0.05)])
        self.assertFalse(rows[0]["success"])
        self.assertIsNone(rows[0]["t_find"])
        self.assertEqual(rows[0]["capped_time"], environment.budget)
        self.assertEqual(summary["success_rate"], 0.0)

    def test_invalid_plan_returns_no_episodes(self):
        policy = {"diagnostics": {"valid": False}}
        rows, summary = evaluate(policy, Environment(obstacles=()), [(0.5, 0.5)])
        self.assertEqual(rows, [])
        self.assertEqual(summary["status"], "invalid_plan")


class LayoutCOcclusionTests(unittest.TestCase):
    """Layout C (two rectangles) is new to v2; v1 only ever had one obstacle."""

    def setUp(self):
        self.obstacles = list(LAYOUTS["C"])
        self.camera = CameraModel(0.25, 90.0)

    def test_line_clear_of_both_rectangles_is_visible(self):
        # Straight down the (0,0)-(1,1) diagonal never enters either off-diagonal rectangle.
        self.assertTrue(line_of_sight_clear((0.1, 0.1), (0.9, 0.9), self.obstacles))

    def test_occluded_by_first_rectangle_only(self):
        # [0.35,0.45] x [0.55,0.65]: a vertical segment straight through its middle.
        self.assertFalse(line_of_sight_clear((0.40, 0.50), (0.40, 0.70), self.obstacles))

    def test_occluded_by_second_rectangle_only(self):
        # [0.55,0.65] x [0.35,0.45]: a horizontal segment straight through its middle.
        self.assertFalse(line_of_sight_clear((0.50, 0.40), (0.70, 0.40), self.obstacles))

    def test_path_between_the_two_rectangles_stays_clear(self):
        # The gap along y=x between the two off-diagonal rectangles is never blocked.
        for x in np.linspace(0.46, 0.54, 5):
            self.assertTrue(line_of_sight_clear((x, x), (x + 0.02, x + 0.02), self.obstacles))

    def test_is_visible_respects_both_range_and_occlusion(self):
        pose = (0.40, 0.50, 0.0)  # facing +x
        # In FOV/bearing but occluded by rectangle 1.
        self.assertFalse(is_visible(pose, (0.40, 0.70), self.obstacles, self.camera))
        # Out of range regardless of occlusion.
        self.assertFalse(is_visible(pose, (0.40, 0.90), self.obstacles, self.camera))


class FullGridReplaySanityTests(unittest.TestCase):
    """The exact primary-endpoint replay (first_seen_grid) runs cleanly on every v2 layout."""

    def test_monotone_nondecreasing_coverage_on_every_layout(self):
        for scene in scenes():
            environment = scene_environment(scene)
            points, free = scene_grid(environment)
            true_prior = gaussian_density(points, free, scene.center, SIGMA, BACKGROUND)
            support = true_prior > 0
            policy = straight_line_policy((0.1, 0.1), (0.9, 0.9), tf=4.0)
            first = first_seen_grid(policy, environment, points, support)
            # Reconstruct cumulative seen-count at each observation time; must never decrease.
            times = sorted(set(first[np.isfinite(first)]))
            counts = [int(np.sum(first <= t)) for t in times]
            self.assertEqual(counts, sorted(counts), f"non-monotone coverage on {scene.scene_id}")
            self.assertTrue(np.isfinite(first[support]).any(), f"nothing ever seen on {scene.scene_id}")


class PairKeySyntheticTests(unittest.TestCase):
    """v2 pairing extends v1's (seed, episode) key with scene_id; scenes must not be conflated."""

    def test_same_seed_episode_different_scene_are_distinct_pairs(self):
        rows = [
            {"scene_id": "sceneA", "seed": 7, "episode": 0, "target_x": 0.2, "target_y": 0.2},
            {"scene_id": "sceneB", "seed": 7, "episode": 0, "target_x": 0.8, "target_y": 0.8},
        ]
        keys = [(r["scene_id"], r["seed"], r["episode"]) for r in rows]
        self.assertEqual(len(set(keys)), len(rows))
        by_key = {k: r for k, r in zip(keys, rows)}
        self.assertNotEqual(by_key[("sceneA", 7, 0)]["target_x"], by_key[("sceneB", 7, 0)]["target_x"])

    def test_dropping_scene_id_would_wrongly_collide_pairs(self):
        # Demonstrates why PROTOCOL_V2_DRAFT.md requires scene_id in the pair key.
        rows = [
            {"scene_id": "sceneA", "seed": 7, "episode": 0, "target_x": 0.2, "target_y": 0.2},
            {"scene_id": "sceneB", "seed": 7, "episode": 0, "target_x": 0.8, "target_y": 0.8},
        ]
        v1_style_keys = [(r["seed"], r["episode"]) for r in rows]
        self.assertEqual(len(set(v1_style_keys)), 1)


if __name__ == "__main__":
    unittest.main()
