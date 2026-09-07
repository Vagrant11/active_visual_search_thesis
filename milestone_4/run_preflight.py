"""Protocol v2 preflight, part 1/2: manifest, feasibility, JS calibration.

No planner import, zero planning runs. Generates the candidate scene/error
manifest, the geometry/support/feasibility audit, and the JS reachability
table across every scene x direction-block x family x level combination.
Run: python -m milestone_4.run_preflight
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .protocol import LEVELS, ROOT, protocol_manifest
from .scene_manifest import (DIRECTION_DEPENDENT_FAMILIES, SceneErrorGenerator, candidate_manifest,
                             directions, scene_feasibility, scenes, uniform_prior_baseline)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def js_reachability_table():
    rows = []
    for scene in scenes():
        blur_results = {}
        for direction_name, direction in directions(scene.center).items():
            generator = SceneErrorGenerator(scene, direction_name, direction)
            for family in DIRECTION_DEPENDENT_FAMILIES:
                scan = generator.parameter_scan(family)
                for level in LEVELS:
                    row, prior = generator.calibrate(family, level, scan=scan)
                    rows.append({"scene_id": scene.scene_id, "direction": direction_name, **row})
            scan = generator.parameter_scan("diffuse_blur")
            for level in LEVELS:
                row, prior = generator.calibrate("diffuse_blur", level, scan=scan)
                blur_results.setdefault(level, []).append((direction_name, row, prior))
        for level, entries in blur_results.items():
            (_, row0, prior0) = entries[0]
            for direction_name, row, prior in entries[1:]:
                same_status = row["status"] == row0["status"]
                same_prior = (prior is None) == (prior0 is None) and (prior is None or np.allclose(prior, prior0, atol=1e-12))
                if not (same_status and same_prior):
                    raise RuntimeError(f"diffuse_blur unexpectedly direction-dependent: {scene.scene_id}, JS={level}")
            rows.append({"scene_id": scene.scene_id, "direction": "shared", **row0})
    return rows


def feasibility_audit():
    return [scene_feasibility(scene) for scene in scenes()]


def uniform_baseline_summary():
    rows = []
    for scene in scenes():
        prior = uniform_prior_baseline(scene)
        rows.append({"scene_id": scene.scene_id, "support_size": int((prior > 0).sum()),
                     "min_probability": float(prior[prior > 0].min()) if (prior > 0).any() else None,
                     "max_probability": float(prior.max()), "probability_sum": float(prior.sum())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/preflight")
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT / "milestone_3" or ROOT / "milestone_3" in out.parents:
        parser.error("Milestone 3 is frozen")
    out.mkdir(parents=True, exist_ok=True)

    # Fails rather than silently preflighting against a modified frozen M3 source.
    v1_manifest = protocol_manifest()

    manifest = candidate_manifest()
    (out / "candidate_scene_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    feasibility = feasibility_audit()
    write_csv(out / "feasibility_audit.csv", feasibility)
    (out / "feasibility_audit.json").write_text(json.dumps(
        {"scenes": feasibility, "all_scenes_passed": all(r["passed"] for r in feasibility)},
        indent=2, allow_nan=False) + "\n")

    reachability = js_reachability_table()
    write_csv(out / "js_reachability_table.csv", reachability)

    uniform = uniform_baseline_summary()
    write_csv(out / "uniform_prior_baseline_summary.csv", uniform)

    unattainable = [r for r in reachability if r["status"] != "matched"]
    summary = {
        "frozen_m3_commit": v1_manifest["frozen_milestone_3_commit"],
        "scene_count": manifest["scene_count"],
        "conditions_per_scene_per_gamma": manifest["conditions_per_scene_per_gamma"],
        "projected_plan_count": manifest["projected_plan_count"],
        "feasibility_all_scenes_passed": all(r["passed"] for r in feasibility),
        "js_reachability_rows": len(reachability),
        "js_reachability_all_matched": not unattainable,
        "js_reachability_unattainable_rows": unattainable,
        "planner_imported": False, "planner_calls": 0,
    }
    (out / "preflight_manifest_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Preflight (manifest/feasibility/calibration) written to {out}")


if __name__ == "__main__":
    main()
