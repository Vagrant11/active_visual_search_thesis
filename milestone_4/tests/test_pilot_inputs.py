import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from milestone_4.protocol import ROOT
from milestone_4.run_pilot import load_calibration, paired_outcomes


class PilotInputTests(unittest.TestCase):
    def test_saved_calibration_can_be_loaded_without_planning(self):
        _, rows, priors, manifest = load_calibration(ROOT / "milestone_4/results/calibration")
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(priors), 12)
        self.assertTrue(manifest["calibration_passed"])

    def test_changed_protocol_or_prior_is_rejected(self):
        source = ROOT / "milestone_4/results/calibration"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ("protocol.json", "priors.npz", "calibration.csv"):
                shutil.copyfile(source / name, path / name)
            manifest = json.loads((path / "protocol.json").read_text())
            manifest["environment"]["budget"] = 16
            (path / "protocol.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                load_calibration(path)
            shutil.copyfile(source / "protocol.json", path / "protocol.json")
            with np.load(path / "priors.npz") as data:
                arrays = {key: data[key].copy() for key in data.files}
            arrays["spatial_shift_js_0.05"] = arrays["true_prior"].copy()
            np.savez(path / "priors.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "integrity check"):
                load_calibration(path)

    def test_pairing_uses_seed_and_retains_zero_time_and_timeout(self):
        base = {"gamma": .1, "episode": 0, "target_x": .2, "target_y": .3}
        rows = [dict(base, prior_id="oracle", family="oracle", target_js_nats=0,
                     seed=7, success=True, t_find=0.0),
                dict(base, prior_id="oracle", family="oracle", target_js_nats=0,
                     seed=11, success=False, t_find=None),
                dict(base, prior_id="error", family="spatial_shift", target_js_nats=.05,
                     seed=7, success=False, t_find=None),
                dict(base, prior_id="error", family="spatial_shift", target_js_nats=.05,
                     seed=11, success=True, t_find=2.0)]
        pairs = paired_outcomes(rows)
        self.assertEqual([r["outcome"] for r in pairs], ["oracle_only", "error_only"])
        self.assertEqual(pairs[0]["oracle_t_find"], 0.0)
        self.assertIsNone(pairs[0]["error_t_find"])
        self.assertEqual(paired_outcomes(rows[2:]), [])
        rows[2]["target_x"] = .8
        with self.assertRaisesRegex(ValueError, "coordinates differ"):
            paired_outcomes(rows)


if __name__ == "__main__":
    unittest.main()
