import unittest
import numpy as np

from milestone_3.experiment import (Environment, camera_trajectory, evaluate, js_divergence,
                                    make_priors, trajectory_is_collision_free)
from milestone_3.diagnostics import coverage_series, diagnostic_summary, paired_outcomes
from visibility import CameraModel


class ExperimentTests(unittest.TestCase):
    def test_priors_normalized_and_oracle_is_not_realized_target(self):
        env = Environment()
        points, priors = make_priors(env)
        for p in priors.values():
            self.assertAlmostEqual(p.sum(), 1.0)
            self.assertTrue((p >= 0).all())
            self.assertGreater(np.count_nonzero(p), 1)
            for cx, cy, r in env.collision_disks:
                self.assertTrue((p[np.linalg.norm(points - [cx, cy], axis=1) <= r] == 0).all())
        self.assertEqual(js_divergence(priors['oracle'], priors['oracle']), 0)
        self.assertGreater(js_divergence(priors['oracle'], priors['biased']), 0)

    def test_js_handles_disjoint_support(self):
        self.assertAlmostEqual(js_divergence([1, 0], [0, 1]), np.log(2))

    def test_reference_node_times_and_terminal_hold(self):
        env = Environment(budget=3, observation_dt=0.5)
        states = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
        trajectory = camera_trajectory(states, 2, env)
        self.assertEqual(trajectory[1, 1], 0.5)
        self.assertEqual(trajectory[2, 1], 1)
        self.assertEqual(trajectory[-1, 1], 1)
        self.assertAlmostEqual(trajectory[1, 3], env.scan_rate * 0.5)

    def test_timeout_and_zero_time_detection_are_distinct(self):
        env = Environment(obstacles=(), budget=1, observation_dt=0.1)
        policy = {'states': np.array([[0.1, 0.1, 0, 0], [0.1, 0.1, 0, 0]]),
                  'tf': 1, 'diagnostics': {'valid': True}}
        rows, summary = evaluate(policy, env, [[0.2, 0.1], [0.9, 0.9]])
        self.assertEqual(rows[0]['t_find'], 0)
        self.assertIsNone(rows[1]['t_find'])
        self.assertEqual(summary['success_rate'], 0.5)
        self.assertEqual(summary['mean_capped_time'], 0.5)
        # Evaluating different hidden targets cannot mutate the planned camera motion.
        before = camera_trajectory(policy['states'], policy['tf'], env)
        evaluate(policy, env, [[0.4, 0.5]])
        np.testing.assert_array_equal(before, camera_trajectory(policy['states'], policy['tf'], env))

    def test_invalid_plan_excluded(self):
        rows, summary = evaluate({'diagnostics': {'valid': False}}, Environment(), [[0, 0]])
        self.assertEqual(rows, [])
        self.assertEqual(summary['status'], 'invalid_plan')

    def test_segment_collision_even_when_endpoints_free(self):
        self.assertFalse(trajectory_is_collision_free(np.array([[0.1, 0.5], [0.9, 0.5]]), Environment()))
        self.assertTrue(trajectory_is_collision_free(np.array([[0.1, 0.2], [0.9, 0.2]]), Environment()))

    def test_length_preserves_corners_between_observations(self):
        env = Environment(obstacles=(), budget=1.5, observation_dt=1.5)
        policy = {'states': np.array([[0, 0], [1, 0], [1, 1]]), 'tf': 3,
                  'diagnostics': {'valid': True}}
        rows, _ = evaluate(policy, env, [[9, 9]])
        self.assertAlmostEqual(rows[0]['path_length_until_stop'], 1.5)
        self.assertEqual(rows[0]['capped_time'], 1.5)

    def test_camera_keeps_scanning_after_terminal_time(self):
        env = Environment(obstacles=(), budget=3, observation_dt=0.05)
        policy = {'states': np.array([[0.5, 0.5], [0.5, 0.5]]), 'tf': 0.2,
                  'diagnostics': {'valid': True}}
        rows, _ = evaluate(policy, env, [[0.3, 0.5]])
        self.assertTrue(rows[0]['success'])
        self.assertGreater(rows[0]['t_find'], policy['tf'])
        self.assertEqual(rows[0]['path_length_until_stop'], 0)

    def test_diagnostics_separate_timeouts_and_pair_same_targets(self):
        rows = [
            {"prior": "oracle", "gamma": 0.1, "episode": 0, "target_x": .1, "target_y": .2,
             "success": True, "t_find": 1.0},
            {"prior": "oracle", "gamma": 0.1, "episode": 1, "target_x": .3, "target_y": .4,
             "success": False, "t_find": None},
            {"prior": "biased", "gamma": 0.1, "episode": 0, "target_x": .1, "target_y": .2,
             "success": False, "t_find": None},
            {"prior": "biased", "gamma": 0.1, "episode": 1, "target_x": .3, "target_y": .4,
             "success": False, "t_find": None},
        ]
        summary = {row["prior"]: row for row in diagnostic_summary(rows)}
        self.assertEqual(summary["oracle"]["timeout_count"], 1)
        self.assertEqual(summary["oracle"]["mean_t_find_success_only"], 1.0)
        self.assertEqual(summary["biased"]["successes"], 0)
        paired = {row["episode"]: row["outcome"] for row in paired_outcomes(rows)}
        self.assertEqual(paired, {0: "oracle_only", 1: "both_timeout"})

    def test_coverage_uses_visibility_union_and_true_prior_mass(self):
        env = Environment(obstacles=(), camera=CameraModel(1, 360), budget=.1, observation_dt=.1)
        policy = {"states": np.array([[.5, .5], [.5, .5]]), "tf": .1}
        rows = coverage_series(policy, env, np.array([[.4, .5], [.6, .5]]), np.array([.25, .75]))
        self.assertEqual(rows[-1]["visible_free_cells"], 2)
        self.assertEqual(rows[-1]["visible_free_fraction"], 1.0)
        self.assertEqual(rows[-1]["oracle_probability_mass_covered"], 1.0)


if __name__ == '__main__':
    unittest.main()
