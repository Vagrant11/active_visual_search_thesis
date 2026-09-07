import copy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

from milestone_3.experiment import Environment, evaluate
from milestone_4.analyze_pilot import (aggregate, coverage_comparison, first_seen_grid,
                                      load_and_validate, pair_episodes, paired_stats, stats)
from milestone_4.protocol import ROOT
from visibility import CameraModel


class AnalysisTests(unittest.TestCase):
    def test_pooling_uses_success_counts_and_raw_median(self):
        # Seed 7 finds one target; seed 11 finds three. Averaging seed means
        # would give 4.5, and averaging seed medians would give 3, both wrong.
        records = []
        for seed, values in ((7, [0, None, None]), (11, [3, 6, 18])):
            for i, t in enumerate(values):
                records.append(dict(prior_id="oracle", family="oracle", target_js_nats=0., gamma=.1,
                                    seed=seed, episode=i, success=t is not None, t_find=t))
        cover = [dict(prior_id="oracle", gamma=.1, visible_free_fraction=.5, oracle_probability_mass_covered=.8)]
        metrics, _ = aggregate({"episodes": records, "coverage_summary": cover}, [], {"target_sampling": {"seeds": [7, 11]}})
        pooled = next(r for r in metrics if r["seed_scope"] == "pooled")
        self.assertEqual(pooled["episodes"], 6)
        self.assertEqual(pooled["success_count"], 4)
        self.assertEqual(pooled["mean_t_find_success_only"], 6.75)
        self.assertEqual(pooled["median_t_find_success_only"], 4.5)
        self.assertEqual(pooled["oracle_probability_mass_covered"], .8)

    def test_pairing_common_success_zero_timeout_and_seed_keys(self):
        records = []
        for seed, oracle_times, error_times in ((7, [0, 2, None, None], [1, None, 4, None]),
                                               (11, [3, None, None, None], [2, None, None, None])):
            for key, times in (("oracle", oracle_times), ("shift", error_times)):
                for i, t in enumerate(times):
                    records.append(dict(prior_id=key, family=key, gamma=.1, target_js_nats=.05,
                                        seed=seed, episode=i, target_x=i/10, target_y=.2,
                                        success=t is not None, t_find=t))
        pairs = pair_episodes(records)
        result = paired_stats(pairs)
        self.assertEqual(result["both_success_count"], 2)
        self.assertEqual(result["both_timeout_count"], 4)
        self.assertEqual(result["delta_success_rate"], 0)
        self.assertEqual(result["both_success_mean_delta_t_find"], 0)
        self.assertEqual(result["error_faster_count_both_success"], 1)
        self.assertEqual(result["error_slower_count_both_success"], 1)
        self.assertEqual(pairs[0]["oracle_t_find"], 0)
        self.assertIsNone(next(p for p in pairs if p["outcome"] == "oracle_only")["delta_t_find_both_success"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pair_episodes(records + [records[0]])
        changed = copy.deepcopy(records)
        changed[4]["target_x"] = .9
        with self.assertRaisesRegex(ValueError, "coordinates differ"):
            pair_episodes(changed)
        with self.assertRaisesRegex(ValueError, "Missing paired oracle"):
            pair_episodes([r for r in records if r["prior_id"] != "oracle"])

    def test_no_success_is_missing_not_zero(self):
        result = stats([{"success": False, "t_find": None}])
        self.assertEqual(result["timeout_rate"], 1)
        self.assertIsNone(result["mean_t_find_success_only"])
        pairs = paired_stats([{"outcome": "both_timeout", "delta_t_find_both_success": None}])
        self.assertIsNone(pairs["both_success_mean_delta_t_find"])

    def test_visibility_replay_matches_evaluator_including_range_boundary(self):
        env = Environment(obstacles=(), camera=CameraModel(.25, 90), budget=3., observation_dt=.05)
        policy = {"states": np.array([[.5, .5, 0, 0], [.5, .5, 0, 0]]), "tf": .2,
                  "diagnostics": {"valid": True}}
        points = np.array([[.5 + .25 + 0.5e-9, .5], [.3, .5], [.95, .95]])
        first = first_seen_grid(policy, env, points, np.ones(3, dtype=bool))
        rows, _ = evaluate(policy, env, points)
        self.assertEqual(first[0], 0)
        self.assertGreater(first[1], .2)
        self.assertTrue(np.isinf(first[2]))
        for i, row in enumerate(rows):
            self.assertEqual(None if np.isinf(first[i]) else first[i], row["t_find"])

    def test_area_gain_can_lose_true_mass_and_hold_starts_at_last_node(self):
        truth = np.array([.6, .15, .15, .1])
        first = np.array([np.inf, 0., 1., np.inf])
        oracle = np.array([0., np.inf, np.inf, np.inf])
        row = coverage_comparison(first, oracle, truth, terminal_arrival=1.)
        self.assertEqual(row["gained_free_cells"], 2)
        self.assertEqual(row["lost_free_cells"], 1)
        self.assertAlmostEqual(row["delta_true_prior_mass"], -.3)
        self.assertAlmostEqual(row["true_mass_first_seen_before_terminal_arrival"], .15)
        self.assertAlmostEqual(row["true_mass_first_seen_at_or_after_terminal_arrival"], .15)

    def test_complete_saved_pilot_replays_without_planner(self):
        _, tables, pairs, _, _, policies, _ = load_and_validate(ROOT / "milestone_4/results/pilot")
        self.assertEqual(len(policies), 26)
        self.assertEqual(len(tables["coverage_timeseries"]), 7826)
        self.assertEqual(len(pairs), 3072)
        self.assertNotIn("milestone_3.planner", sys.modules)

    def test_changed_episode_or_invalid_plan_is_rejected(self):
        from milestone_4.analyze_pilot import read_csv
        from milestone_4.calibrate_priors import write_csv
        source = ROOT / "milestone_4/results/pilot"
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "pilot"
            shutil.copytree(source, out)
            rows = read_csv(out / "episodes.csv")
            rows[0]["target_x"] = .999
            write_csv(out / "episodes.csv", rows)
            with self.assertRaisesRegex(ValueError, "target mismatch"):
                load_and_validate(out)
            shutil.copyfile(source / "episodes.csv", out / "episodes.csv")
            rows = read_csv(out / "planner_diagnostics.csv")
            rows[0]["valid"] = False
            write_csv(out / "planner_diagnostics.csv", rows)
            with self.assertRaisesRegex(ValueError, "complete valid pilot"):
                load_and_validate(out)


if __name__ == "__main__":
    unittest.main()
