"""Gate 5: freeze Protocol v2 -- manifest, calibrated priors/hashes, primary
endpoints, planned contrasts, failure handling, in one dedicated artifact set.

No planner import, zero planning. Requires every earlier gate's artifacts to
already exist and pass (gates 1-4); refuses to freeze otherwise. Computes and
saves the actual calibrated prior vector for all 12 scenes x 23 conditions
(276 total) with hashes, then writes PROTOCOL_V2_FROZEN_MANIFEST.json. This
freezes the DESIGN; it does not launch the formal 552-plan suite.

Run: python -m milestone_4.freeze_protocol_v2
"""

import argparse
import hashlib
import json
import os
import subprocess
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .protocol import LEVELS, ROOT, protocol_manifest
from .scene_manifest import (DIRECTION_DEPENDENT_FAMILIES, SceneErrorGenerator, candidate_manifest,
                             directions, scenes, uniform_prior_baseline)

PREFLIGHT_SOURCES = ("milestone_4/scene_manifest.py", "milestone_4/run_preflight.py",
                     "milestone_4/run_engineering_check.py", "milestone_4/run_gate2_cost_check.py",
                     "milestone_4/gate4_seed_decision.py", "milestone_4/freeze_protocol_v2.py",
                     "milestone_4/tests/test_v2_preflight.py")

PLANNED_CONTRASTS = {
    "primary_oracle_vs_error": "Exact full-grid detection probability and true-prior-weighted "
        "mean/median T_find difference, error family minus oracle, per scene/direction-block/JS/gamma.",
    "primary_oracle_vs_uniform_prior_baseline": "Exact full-grid detection probability and "
        "true-prior-weighted mean/median T_find difference, uniform-prior planning baseline minus "
        "oracle, per scene/gamma (direction-independent).",
    "primary_family_pairs": "All six unordered pairs among the four error families, exact full-grid "
        "detection probability and weighted-T_find difference, at fixed scene/direction-block/JS/gamma.",
    "diagnostic_uniform_weighted_reweighting": "Uniform-weighted reweighting of each planned "
        "trajectory (oracle and the four error families), reported alongside the primary true-prior- "
        "weighted result; not a ranking metric on its own.",
    "secondary_paired_outcomes": "Oracle-outcome x error-prior-outcome 2x2 (success/timeout) table "
        "plus both-success paired ΔT and its p75/p90/p95/max, per scene/direction-block/JS/gamma, "
        "sampled with the 5-seed/256-target design (Gate 4).",
    "aggregation": "Average the two direction-block contrasts within each scene, then give each of "
        "the 12 scenes equal weight; never pool gamma. See PROTOCOL_V2_DRAFT.md, "
        "'Contrasts and aggregation'.",
}

FAILURE_HANDLING = {
    "invalid_plan_is_not_a_timeout": "An invalid plan (planner_diagnostics.valid == False) is a "
        "planner failure, not a search timeout; never assign it T_find=budget or success_rate=0.",
    "denominators_reported": "Every report shows the planned and available condition denominators; "
        "only valid pairs enter conditional search comparisons.",
    "no_silent_substitution": "A failed plan blocks the claim of a complete matched suite. Do not "
        "rank families on different retained scene sets, change solver settings for one family, "
        "choose the best retry, or drop scenes after viewing outcomes.",
    "revision_path": "Diagnose failures and either report the incomplete suite explicitly, or freeze "
        "a revised common solver/protocol and rerun all affected comparisons under that revision -- "
        "never patch a single scene/family in isolation.",
}


def read_json(path):
    return json.loads(path.read_text())


def require_gate(condition, message):
    if not condition:
        raise RuntimeError(f"Freeze blocked: {message}")


def verify_prior_gates(preflight_dir):
    summary = read_json(preflight_dir / "preflight_manifest_summary.json")
    require_gate(summary["feasibility_all_scenes_passed"], "not all candidate scenes passed feasibility")
    require_gate(summary["js_reachability_all_matched"], "not all JS-reachability rows matched")
    engineering = read_json(preflight_dir / "engineering_check.json")
    require_gate(engineering["complete"], "engineering checks incomplete")
    require_gate(all(r["valid"] and r["collision_free"] for r in engineering["results"]),
                "an engineering check was invalid or collided")
    require_gate((preflight_dir / "gate2_cost_check.json").exists(), "Gate 2 cost check missing")
    from .tests import test_v2_preflight
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_v2_preflight)
    with open(os.devnull, "w") as devnull:
        result = unittest.TextTestRunner(verbosity=0, stream=devnull).run(suite)
    require_gate(result.wasSuccessful(), "Gate 2 synthetic evaluator-semantics tests do not pass")
    require_gate((preflight_dir / "gate4_seed_decision.json").exists(), "Gate 4 seed decision missing")
    return summary


def calibrate_all_priors():
    """276 vectors: 12 scenes x (oracle 1 + blur 3 + uniform-baseline 1 + 3 direction-dependent

    families x 3 levels x 2 blocks 18) = 23/scene."""
    per_scene = {}
    for scene in scenes():
        priors, hashes = {}, {}
        priors["oracle"] = None  # filled from either generator's true_prior below
        generators = {name: SceneErrorGenerator(scene, name, direction)
                     for name, direction in directions(scene.center).items()}
        priors["oracle"] = generators["x"].true_prior.copy()
        for family in DIRECTION_DEPENDENT_FAMILIES:
            for direction_name, generator in generators.items():
                scan = generator.parameter_scan(family)
                for level in LEVELS:
                    row, prior = generator.calibrate(family, level, scan=scan)
                    if prior is None:
                        raise RuntimeError(f"Unattainable at freeze time: {scene.scene_id} {direction_name} {family} {level}")
                    priors[f"{family}_{direction_name}_js_{level:g}"] = prior
        blur_scan = generators["x"].parameter_scan("diffuse_blur")
        for level in LEVELS:
            row, prior_x = generators["x"].calibrate("diffuse_blur", level, scan=blur_scan)
            _, prior_y = generators["y"].calibrate("diffuse_blur", level, scan=generators["y"].parameter_scan("diffuse_blur"))
            if not np.allclose(prior_x, prior_y, atol=1e-12):
                raise RuntimeError(f"diffuse_blur direction-dependent at freeze time: {scene.scene_id}")
            priors[f"diffuse_blur_js_{level:g}"] = prior_x
        priors["uniform_prior_baseline"] = uniform_prior_baseline(scene)
        for key, vector in priors.items():
            hashes[key] = hashlib.sha256(vector.tobytes()).hexdigest()
        per_scene[scene.scene_id] = {"priors": priors, "hashes": hashes,
                                     "points": generators["x"].points, "support": generators["x"].support}
    return per_scene


def write_frozen_priors(out, per_scene):
    priors_dir = out / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)
    for scene_id, data in per_scene.items():
        np.savez(priors_dir / f"{scene_id}.npz", points=data["points"], free_mask=data["support"],
                 **data["priors"])
    return {scene_id: data["hashes"] for scene_id, data in per_scene.items()}


def freeze(preflight_dir, out):
    v1_manifest = protocol_manifest()  # fails if frozen M3 sources changed
    verify_prior_gates(preflight_dir)

    manifest = candidate_manifest()
    feasibility = read_json(preflight_dir / "feasibility_audit.json")
    reachability_summary = read_json(preflight_dir / "preflight_manifest_summary.json")
    engineering = read_json(preflight_dir / "engineering_check.json")
    cost = read_json(preflight_dir / "gate2_cost_check.json")
    seed_decision = read_json(preflight_dir / "gate4_seed_decision.json")

    per_scene_priors = calibrate_all_priors()
    prior_hashes = write_frozen_priors(out, per_scene_priors)
    total_vectors = sum(len(h) for h in prior_hashes.values())
    require_gate(total_vectors == manifest["scene_count"] * manifest["conditions_per_scene_per_gamma"],
                "calibrated prior count does not match the candidate manifest's condition count")

    preflight_source_sha256 = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in PREFLIGHT_SOURCES}
    frozen_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    frozen = {
        "protocol_id": "m4-controlled-prior-v2-frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_at_git_commit": frozen_commit,
        "frozen_milestone_3_commit": v1_manifest["frozen_milestone_3_commit"],
        "preflight_source_sha256": preflight_source_sha256,
        "candidate_manifest": manifest,
        "prior_sha256_by_scene": prior_hashes,
        "final_evaluation_sampling": seed_decision["final_decision"],
        "seed_decision_rationale": seed_decision["rationale"],
        "planner_and_evaluation_invariants": {
            "planner": v1_manifest["planner"], "js_tolerance": v1_manifest["js"]["absolute_tolerance"],
            "camera_range": 0.25, "camera_fov_degrees": 90.0, "scan_rate_radians_per_s": "pi/2",
            "budget_seconds": 15.0, "observation_dt_seconds": 0.05,
        },
        "planned_contrasts": PLANNED_CONTRASTS,
        "failure_handling": FAILURE_HANDLING,
        "gate_evidence": {
            "gate1_feasibility_all_scenes_passed": reachability_summary["feasibility_all_scenes_passed"],
            "gate1_js_reachability_all_matched": reachability_summary["js_reachability_all_matched"],
            "gate2_synthetic_tests": "milestone_4/tests/test_v2_preflight.py (27 tests incl. 16 v1, pass)",
            "gate2_evaluation_cost_seconds_per_plan": cost["per_plan_evaluation_seconds"],
            "gate3_engineering_checks_valid": all(r["valid"] for r in engineering["results"]),
            "gate3_projected_planning_hours": (sum(r["wall_seconds"] for r in engineering["results"])
                / len(engineering["results"]) * manifest["projected_plan_count"] / 3600),
            "gate4_final_seed_count": seed_decision["final_decision"]["seed_count"],
        },
        "not_launched": "The formal 552-plan suite has not been run. This artifact freezes the "
                        "design; a separate, explicitly invoked run is required to execute it.",
    }
    (out / "PROTOCOL_V2_FROZEN_MANIFEST.json").write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    print(f"Frozen manifest: {out / 'PROTOCOL_V2_FROZEN_MANIFEST.json'}")
    print(f"{total_vectors} calibrated prior vectors written under {out / 'priors'}")
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=ROOT / "milestone_4/results/preflight")
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/frozen")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    freeze(args.preflight.resolve(), out)


if __name__ == "__main__":
    main()
