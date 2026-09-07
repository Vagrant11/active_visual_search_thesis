import unittest

import numpy as np

from milestone_3.experiment import js_divergence
from milestone_4.calibrate_priors import calibrate_suite
from milestone_4.prior_errors import ErrorDesign, FAMILIES, PriorErrorGenerator, validate_prior
from milestone_4.protocol import LEVELS, ROOT, paired_targets, protocol_manifest


class PriorErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator, cls.rows, cls.priors, cls.audit = calibrate_suite()

    def test_all_twelve_match_js_and_keep_common_support(self):
        self.assertEqual(len(self.rows), 12)
        for row in self.rows:
            with self.subTest(row=row["prior_id"]):
                p = self.priors[row["prior_id"]]
                validate_prior(p, self.generator.support)
                self.assertEqual(row["status"], "matched")
                self.assertLessEqual(abs(js_divergence(self.generator.true_prior, p) - row["target_js_nats"]), 1e-6)

    def test_zero_error_recovers_unchanged_truth_for_all_families(self):
        for family in FAMILIES:
            row, p = self.generator.calibrate(family, 0)
            np.testing.assert_array_equal(p, self.generator.true_prior)
            self.assertEqual(row["actual_js_nats"], 0)

    def test_blur_cannot_match_point_three_and_is_not_clipped(self):
        row = next(r for r in self.audit if r["family"] == "diffuse_blur" and r["target_js_nats"] == .30)
        self.assertEqual(row["status"], "unattainable_in_scan")
        self.assertIsNone(row["parameter_value"])
        self.assertIsNone(row["actual_js_nats"])
        self.assertLess(row["scan_max_js_nats"], .18)
        self.assertAlmostEqual(self.generator.metadata()["blur_uniform_limit_js_nats"], .17805172492048166)

    def test_error_structures_are_distinct_at_matched_js(self):
        g = self.generator
        for level in LEVELS:
            priors = [self.priors[r["prior_id"]] for r in self.rows if r["target_js_nats"] == level]
            for i, p in enumerate(priors):
                for q in priors[i + 1:]:
                    self.assertGreater(np.linalg.norm(p - q), 1e-3)
        shifted = g.generate("spatial_shift", .5)
        self.assertLess(g.points[np.argmax(shifted), 0], .3)
        blurred = g.generate("diffuse_blur", 2)
        self.assertEqual(np.argmax(blurred), np.argmax(g.true_prior))
        suppressed = g.generate("false_negative_suppression", 1)
        self.assertTrue((suppressed[g.suppressed] == 0).all())
        ratio = suppressed[g.support & ~g.suppressed] / g.true_prior[g.support & ~g.suppressed]
        np.testing.assert_allclose(ratio, ratio[0])
        mixed = g.generate("false_hotspot", .4)
        np.testing.assert_allclose(mixed, .6*g.true_prior + .4*g.false_hotspot)

    def test_invalid_inputs_rejected(self):
        for value in (-.1, np.nan, np.inf, 1.1):
            with self.assertRaises(ValueError):
                self.generator.generate("false_hotspot", value)
        for value in (-.1, np.nan, np.inf, .7):
            with self.assertRaises(ValueError):
                self.generator.calibrate("spatial_shift", value)
        for p in ([1, -1], [np.nan, 1], [0, 0], [0, 1]):
            with self.assertRaises(ValueError):
                validate_prior(p, [True, False])
        with self.assertRaises(ValueError):
            PriorErrorGenerator(ErrorDesign(sigma=.2))

    def test_ground_truth_and_seed_seven_match_frozen_milestone_three(self):
        g = self.generator
        with np.load(ROOT / "milestone_3/results/inputs.npz") as archived:
            np.testing.assert_array_equal(g.points, archived["points"])
            np.testing.assert_array_equal(g.true_prior, archived["oracle"])
            indices, targets = paired_targets(g.points, g.true_prior, 7)
            np.testing.assert_array_equal(indices, archived["target_indices"])
            np.testing.assert_array_equal(targets, archived["targets"])
        manifest = protocol_manifest()
        self.assertEqual(manifest["planner"]["gammas"], [.1, .05])


if __name__ == "__main__":
    unittest.main()
