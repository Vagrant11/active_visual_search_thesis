"""Independent, read-only audit of the frozen Protocol v2 formal run.

No planner is imported or called.  The audit replays saved trajectories, checks
every sampled target against its seed and scene, and recomputes the frozen
constraint residuals with NumPy.  A correctly recorded failure can pass the
integrity audit while preventing a claim of a complete matched suite.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from milestone_3.experiment import camera_trajectory, js_divergence, trajectory_is_collision_free
from .analyze_pilot import (entropy_nats, first_seen_grid, read_csv, stats,
                            trajectory_length, weighted_detection_stats)
from .protocol import ROOT, protocol_manifest
from .scene_manifest import (FAMILIES, Scene, SceneErrorGenerator, candidate_manifest,
                             gaussian_density, scene_environment, scene_grid)
from visibility import RectangleObstacle


DEFAULT_MANIFEST = ROOT / "milestone_4/results/frozen/PROTOCOL_V2_FROZEN_MANIFEST.json"
EXACT_KEYS = ("exact_detection_probability", "exact_weighted_mean_t_find",
              "exact_weighted_median_t_find", "uniform_detection_probability",
              "uniform_weighted_mean_t_find", "uniform_weighted_median_t_find")
LABEL_KEYS = ("scene_id", "prior_id", "family", "direction", "target_js_nats", "gamma")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number_equal(left, right, atol=1e-10):
    if left is None or right is None:
        return left is None and right is None
    return bool(np.isfinite(left) and np.isfinite(right)
                and np.isclose(left, right, rtol=1e-10, atol=atol))


def expected_plans(manifest):
    candidate = manifest["candidate_manifest"]
    scenes = candidate["scenes"]
    require(len(scenes) == candidate["scene_count"] == 12, "Expected exactly 12 frozen scenes")
    require(len({s["scene_id"] for s in scenes}) == 12, "Duplicate frozen scene IDs")
    require(candidate["conditions_per_scene_per_gamma"] == 23, "Expected 23 conditions per scene/gamma")
    require(candidate["gammas"] == manifest["planner_and_evaluation_invariants"]["planner"]["gammas"],
            "Candidate and frozen planner gamma lists disagree")
    result = {}
    for scene in scenes:
        conditions = scene["conditions"]
        require(len(conditions) == scene["conditions_per_gamma"] == 23,
                f"Wrong frozen condition count: {scene['scene_id']}")
        require(len({c["prior_id"] for c in conditions}) == 23, "Duplicate frozen prior IDs")
        for gamma in candidate["gammas"]:
            for condition in conditions:
                plan_id = f"{scene['scene_id']}__gamma_{gamma:g}__{condition['prior_id']}"
                result[plan_id] = {"scene_id": scene["scene_id"], "gamma": gamma,
                                   **condition, "plan_id": plan_id}
    require(len(result) == candidate["projected_plan_count"] == 552, "Expected exactly 552 plans")
    return result


def scene_from_manifest(spec):
    return Scene(spec["scene_id"], tuple(spec["center"]), spec["layout"],
                 tuple(RectangleObstacle(**o) for o in spec["obstacles"]))


def audit_inputs(manifest, source, manifest_path):
    """Validate frozen arrays, calibrated parameters and saved paired draws."""
    require(json.loads(json.dumps(candidate_manifest())) == manifest["candidate_manifest"],
            "Current scene/error geometry differs from frozen candidate manifest")
    sampling = manifest["final_evaluation_sampling"]
    require(sampling["seed_count"] == len(sampling["seeds"]), "Seed count mismatch")
    require(sampling["total_draws_per_scene_per_condition"] ==
            sampling["seed_count"] * sampling["targets_per_seed"], "Draw denominator mismatch")
    calibration_path = manifest_path.parent.parent / "preflight/js_reachability_table.csv"
    calibration = {}
    with calibration_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["scene_id"], row["direction"], row["family"], float(row["target_js_nats"])
            require(key not in calibration, f"Duplicate calibration row: {key}")
            calibration[key] = row
    require(len(calibration) == 252, "Expected 252 frozen calibration parameter rows")
    result = {}
    for spec in manifest["candidate_manifest"]["scenes"]:
        scene_id = spec["scene_id"]
        scene = scene_from_manifest(spec)
        environment = scene_environment(scene)
        points, free = scene_grid(environment)
        truth = gaussian_density(points, free, scene.center,
                                 manifest["candidate_manifest"]["sigma"],
                                 manifest["candidate_manifest"]["background"])
        with np.load(manifest_path.parent / "priors" / f"{scene_id}.npz", allow_pickle=False) as data:
            frozen = {k: data[k].copy() for k in data.files}
        with np.load(source / "inputs" / f"{scene_id}.npz", allow_pickle=False) as data:
            saved = {k: data[k].copy() for k in data.files}
        require(set(frozen) == {"points", "free_mask"} | {c["prior_id"] for c in spec["conditions"]},
                f"Frozen NPZ condition keys differ: {scene_id}")
        require(np.array_equal(points, frozen["points"]), f"Frozen grid changed: {scene_id}")
        require(np.array_equal(free, frozen["free_mask"]), f"Frozen support changed: {scene_id}")
        require(np.array_equal(truth, frozen["oracle"]), f"Frozen truth differs from geometry: {scene_id}")
        for key, vector in frozen.items():
            require(key in saved and np.array_equal(vector, saved[key]),
                    f"Saved input differs from frozen input: {scene_id}/{key}")
        generators = {d: SceneErrorGenerator(scene, d, tuple(vector))
                      for d, vector in spec["directions"].items()}
        for condition in spec["conditions"]:
            key = condition["prior_id"]
            vector = saved[key]
            require(array_sha256(vector) == manifest["prior_sha256_by_scene"][scene_id][key],
                    f"Frozen prior SHA-256 mismatch: {scene_id}/{key}")
            require(np.isfinite(vector).all() and np.all(vector >= 0)
                    and np.isclose(vector.sum(), 1, rtol=0, atol=1e-12)
                    and np.all(vector[~free] == 0), f"Invalid prior probabilities: {scene_id}/{key}")
            actual = js_divergence(truth, vector)
            target = condition["target_js_nats"]
            if target is not None:
                require(abs(actual - target) <= manifest["planner_and_evaluation_invariants"]["js_tolerance"],
                        f"JS target mismatch: {scene_id}/{key}")
            if condition["family"] in FAMILIES:
                direction = condition["direction"] or "shared"
                row = calibration[scene_id, direction, condition["family"], target]
                generated = generators[condition["direction"] or "x"].generate(
                    condition["family"], float(row["parameter_value"]))
                require(row["status"] == "matched" and row["valid_prior"] == "True"
                        and number_equal(actual, float(row["actual_js_nats"]))
                        and np.allclose(generated, vector, rtol=0, atol=1e-15),
                        f"Saved prior does not match calibrated error geometry: {scene_id}/{key}")
            elif condition["family"] == "uniform_prior_baseline":
                require(np.array_equal(vector, free / free.sum()), "Uniform baseline was substituted")
        expected_saved_keys = set(frozen)
        for seed in sampling["seeds"]:
            indices = np.random.default_rng(seed).choice(len(points), sampling["targets_per_seed"], p=truth)
            i_key, t_key = f"target_indices_seed_{seed}", f"targets_seed_{seed}"
            expected_saved_keys.update((i_key, t_key))
            require(np.array_equal(saved[i_key], indices), f"Target seed/index mismatch: {scene_id}/{seed}")
            require(np.array_equal(saved[t_key], points[indices]), f"Target coordinate mismatch: {scene_id}/{seed}")
        require(set(saved) == expected_saved_keys, f"Unexpected or missing saved input keys: {scene_id}")
        result[scene_id] = {"arrays": saved, "environment": environment}
    require({p.stem for p in (source / "inputs").glob("*.npz")} == set(result),
            "Unexpected or missing scene input files")
    return result


def recompute_diagnostics(policy, environment, points, probabilities, gamma, initial, terminal):
    """Independent NumPy replay of the frozen planner's feasibility formulas."""
    states, controls, tf = policy["states"], policy["controls"], policy["tf"]
    require(states.ndim == 2 and states.shape[1] == 4 and controls.shape == (len(states), 2),
            "Trajectory/control shape mismatch")
    if not all(np.isfinite(v).all() for v in (states, controls, np.array(tf))):
        return {"finite": False, "collision_free": False, "feasible": False}
    derivative = np.column_stack((states[:-1, 2:], controls[:-1]))
    eq_error = float(np.max(np.abs(np.vstack((states[0] - initial,
                       states[1:] - (states[:-1] + tf / len(states) * derivative), states[-1] - terminal)))))
    kmesh = np.meshgrid(np.arange(8), np.arange(8))
    k = np.stack([v.ravel() for v in kmesh]).T * np.pi
    norm_components = np.ones_like(k)
    nonzero = k != 0
    norm_components[nonzero] = (2 * k[nonzero] + np.sin(2 * k[nonzero])) / (4 * k[nonzero])
    hk = np.sqrt(np.prod(norm_components, axis=1))
    ck = np.mean(np.prod(np.cos(states[:, None, :2] * k[None, :, :]), axis=2), axis=0) / hk
    phik = probabilities @ np.prod(np.cos(points[:, None, :] * k[None, :, :]), axis=2)
    phik = phik / phik[0] / hk
    lam = (1 + np.linalg.norm(k / np.pi, axis=1) ** 2) ** -1.5
    ergodicity = float(np.sum(lam * (ck - phik) ** 2))
    xy = states[:, :2]
    inequalities = [ergodicity - gamma, -tf, np.max(np.abs(controls) - 1),
                    np.max(-xy), np.max(xy - 1), np.max(np.sum(states[:, 2:] ** 2, axis=1) - 1), 0.1 - tf]
    for cx, cy, radius in environment.collision_disks:
        a, delta = xy[:-1], xy[1:] - xy[:-1]
        fraction = np.clip(np.sum((np.array([cx, cy]) - a) * delta, axis=1)
                           / (np.sum(delta ** 2, axis=1) + 1e-15), 0, 1)
        closest = a + fraction[:, None] * delta
        inequalities.append(np.max(radius ** 2 - np.sum((closest - (cx, cy)) ** 2, axis=1)))
    ineq_error = float(max(0, *inequalities))
    finite = bool(np.isfinite([eq_error, ineq_error, ergodicity]).all())
    collision_free = finite and trajectory_is_collision_free(states, environment)
    feasible = bool(finite and tf >= 0.1 and eq_error <= 1e-5 and ineq_error <= 1e-6 and collision_free)
    return {"finite": finite, "collision_free": collision_free, "feasible": feasible,
            "max_equality_residual": eq_error, "max_inequality_violation": ineq_error,
            "ergodicity": ergodicity}


def validate_episodes(episodes, first, arrays, policy, environment, sampling):
    expected = {(seed, episode) for seed in sampling["seeds"]
                for episode in range(sampling["targets_per_seed"])}
    actual = [(r["seed"], r["episode"]) for r in episodes]
    require(len(actual) == len(set(actual)) and set(actual) == expected,
            "Sampled episodes are duplicated, missing or use unexpected seeds")
    nodes = policy["states"][:, :2]
    node_times = np.arange(len(nodes)) * policy["tf"] / len(nodes)
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
    for row in episodes:
        seed, episode = row["seed"], row["episode"]
        index = arrays[f"target_indices_seed_{seed}"][episode]
        target = arrays["points"][index]
        require(np.array_equal([row["target_x"], row["target_y"]], target),
                f"Paired target coordinates differ: seed={seed}, episode={episode}")
        found = float(first[index]) if np.isfinite(first[index]) else None
        require(row["success"] is (found is not None) and number_equal(row["t_find"], found),
                f"Sampled outcome differs from grid replay: seed={seed}, episode={episode}")
        stop = found if found is not None else environment.budget
        require(number_equal(row["capped_time"], stop), "Incorrect sampled capped time")
        require(number_equal(row["path_length_until_stop"], float(np.interp(stop, node_times, lengths))),
                "Incorrect sampled trajectory length at detection/timeout")


def validate_plan(path, expected, inputs, manifest, replay=True):
    result = read_json(path / "result.json")
    for key in (*LABEL_KEYS, "plan_id"):
        require(result.get(key) == expected[key], f"Condition label mismatch: {key}")
    require(result.get("attempt_count") == 1, "Retries are forbidden in the frozen suite")
    require(result.get("status") in ("valid", "invalid", "interrupted", "exception"), "Unknown final status")
    require(isinstance(result.get("valid"), bool), "Missing boolean plan validity")
    require(result["valid"] == (result["status"] == "valid"), "Status and validity disagree")
    require((path / "started.json").is_file(), "Missing pre-solver attempt checkpoint")
    artifacts = result.get("artifact_sha256", {})
    required = {"diagnostics.json", "exact_metrics.json", "episodes.csv", "sampled_metrics.json"}
    if result["status"] in ("valid", "invalid"):
        required.update(("trajectory.npz", "solver_result.json"))
    if result["valid"]:
        required.add("first_seen_grid.npz")
    require(required <= set(artifacts), f"Missing artifact hashes: {sorted(required - set(artifacts))}")
    for name, digest in artifacts.items():
        target = (path / name).resolve()
        require(target.is_relative_to(path.resolve()), "Artifact path escapes plan directory")
        require(target.is_file() and sha256(target) == digest, f"Artifact SHA-256 mismatch: {name}")
    diagnostics = read_json(path / "diagnostics.json")
    require(diagnostics.get("valid") is result["valid"], "Diagnostics and result validity disagree")
    exact = read_json(path / "exact_metrics.json")
    sampled = read_json(path / "sampled_metrics.json")
    episodes = read_csv(path / "episodes.csv") if (path / "episodes.csv").stat().st_size else []
    require(isinstance(sampled, list), "Sampled metrics must contain a per-seed list")
    sampling = manifest["final_evaluation_sampling"]
    require(len(sampled) == sampling["seed_count"] and {r["seed"] for r in sampled} == set(sampling["seeds"]),
            "Missing, duplicate or unexpected sampled summary seeds")
    arrays, environment = inputs["arrays"], inputs["environment"]
    actual_js = js_divergence(arrays["oracle"], arrays[expected["prior_id"]])
    require(number_equal(result.get("actual_js_nats"), actual_js), "Recorded JS differs from saved prior")
    require(number_equal(exact.get("prior_entropy_nats"), entropy_nats(arrays[expected["prior_id"]])),
            "Prior entropy differs from frozen input")
    policy = None
    if (path / "trajectory.npz").exists():
        with np.load(path / "trajectory.npz", allow_pickle=False) as data:
            policy = {"states": data["states"].copy(), "controls": data["controls"].copy(),
                      "tf": float(data["tf"]), "diagnostics": diagnostics}
        planner = manifest["planner_and_evaluation_invariants"]["planner"]
        require(policy["states"].shape == (planner["nodes"], 4), "Wrong planner node count")
        if result["status"] in ("valid", "invalid"):
            computed = recompute_diagnostics(policy, environment, arrays["points"],
                        arrays[expected["prior_id"]], expected["gamma"],
                        np.asarray(planner["initial_state"]), np.asarray(planner["terminal_state"]))
            for key, value in computed.items():
                require(diagnostics.get(key) == value if isinstance(value, bool)
                        else number_equal(diagnostics.get(key), value, atol=1e-9),
                        f"Saved solver diagnostics disagree with independent residual replay: {key}")
            require(result["valid"] == bool(diagnostics["solver_success"] and computed["feasible"]),
                    "Validity does not follow frozen solver-success/feasibility rule")
    if not result["valid"]:
        require(not episodes, "Invalid plan has search episodes (failure treated as timeout)")
        for key in (*EXACT_KEYS, "visible_free_fraction", "oracle_probability_mass_covered"):
            require(key in exact and exact[key] is None, f"Invalid plan has a search metric: {key}")
        require(not (path / "first_seen_grid.npz").exists(), "Invalid plan has detection-grid outcomes")
        for row in sampled:
            require(row.get("episodes") == 0, "Invalid plan has sampled episode denominator")
            for key in ("success_count", "timeout_count", "success_rate", "timeout_rate",
                        "mean_t_find_success_only", "median_t_find_success_only"):
                require(row.get(key) is None, f"Invalid plan has a sampled search metric: {key}")
        return {**result, "episodes_audited": 0, "replayed": False}
    require(policy is not None, "Valid plan has no saved trajectory")
    with np.load(path / "first_seen_grid.npz", allow_pickle=False) as data:
        first = data["first_seen"].copy()
    require(first.shape == arrays["oracle"].shape and not np.isnan(first).any(), "Invalid first-seen grid")
    require(np.all(np.isposinf(first[~arrays["free_mask"]])), "Unsupported cells have detection times")
    observed = camera_trajectory(policy["states"], policy["tf"], environment)[:, 0]
    require(np.isin(first[np.isfinite(first)], observed).all(), "First detection is not on frozen observation grid")
    if replay:
        replayed = first_seen_grid(policy, environment, arrays["points"], arrays["free_mask"])
        require(np.array_equal(first, replayed), "Saved first-seen grid differs from trajectory visibility replay")
    uniform = arrays["free_mask"] / arrays["free_mask"].sum()
    expected_metrics = {}
    for prefix, weights in (("exact", arrays["oracle"]), ("uniform", uniform)):
        expected_metrics.update({f"{prefix}_{key}": value for key, value in weighted_detection_stats(first, weights).items()})
    expected_metrics.update(trajectory_length=trajectory_length(policy["states"]),
                            visible_free_fraction=float(np.isfinite(first).sum() / arrays["free_mask"].sum()),
                            oracle_probability_mass_covered=expected_metrics["exact_detection_probability"])
    for key, value in expected_metrics.items():
        require(key in exact and number_equal(exact[key], value), f"Exact-grid metric mismatch: {key}")
    validate_episodes(episodes, first, arrays, policy, environment, sampling)
    for row in sampled:
        records = [r for r in episodes if r["seed"] == row["seed"]]
        expected_summary = stats(records)
        for key, value in expected_summary.items():
            require(key in row and number_equal(row[key], value), f"Sampled summary mismatch: seed={row['seed']}/{key}")
    return {**result, "episodes_audited": len(episodes), "replayed": replay}


def pairing_denominators(manifest, records):
    """Count planned and available paired contrasts without dropping conditions."""
    valid = {(r["scene_id"], r["gamma"], r["prior_id"]) for r in records if r["valid"]}
    counts = {kind: {"planned": 0, "available": 0} for kind in
              ("oracle_vs_error_direction_blocks", "family_pairs_direction_blocks", "oracle_vs_uniform")}
    for scene in manifest["candidate_manifest"]["scenes"]:
        for gamma in manifest["candidate_manifest"]["gammas"]:
            def available(prior):
                return (scene["scene_id"], gamma, prior) in valid
            counts["oracle_vs_uniform"]["planned"] += 1
            counts["oracle_vs_uniform"]["available"] += int(available("oracle") and available("uniform_prior_baseline"))
            for direction in scene["directions"]:
                for level in manifest["candidate_manifest"]["js_levels"]:
                    priors = [f"{family}_{direction}_js_{level:g}" if family != "diffuse_blur"
                              else f"{family}_js_{level:g}" for family in FAMILIES]
                    for prior in priors:
                        counts["oracle_vs_error_direction_blocks"]["planned"] += 1
                        counts["oracle_vs_error_direction_blocks"]["available"] += int(available("oracle") and available(prior))
                    for left, right in combinations(priors, 2):
                        counts["family_pairs_direction_blocks"]["planned"] += 1
                        counts["family_pairs_direction_blocks"]["available"] += int(available(left) and available(right))
    return counts


def verify_provenance(source, config, manifest, manifest_path):
    require(config["frozen_manifest"] == manifest, "Run manifest snapshot differs from frozen manifest")
    require(config["frozen_manifest_sha256"] == sha256(manifest_path), "Frozen manifest SHA-256 changed")
    protocol_manifest()  # Checks frozen M3 byte content and reference-submodule commit/tree.
    for name, digest in manifest["preflight_source_sha256"].items():
        require(sha256(ROOT / name) == digest, f"Frozen preflight source changed: {name}")
    for field in ("source_sha256", "frozen_artifact_sha256"):
        require(bool(config.get(field)), f"Missing pre-run provenance: {field}")
        for name, digest in config[field].items():
            path = Path(name)
            path = path if path.is_absolute() else ROOT / path
            require(sha256(path) == digest, f"Pre/post content SHA-256 changed: {name}")
    post_path = source / "postrun_provenance.json"
    require(post_path.is_file(), "Missing explicit post-run provenance checkpoint")
    post = read_json(post_path)
    for field in ("source_sha256", "frozen_artifact_sha256", "versions"):
        require(post.get(field) == config.get(field), f"Pre/post provenance mismatch: {field}")
    require(post.get("frozen_manifest_sha256") == config["frozen_manifest_sha256"],
            "Post-run frozen manifest SHA-256 mismatch")


def audit(source, out=None, replay=True):
    source = Path(source).resolve()
    config = read_json(source / "config.json")
    manifest_path = Path(config.get("frozen_manifest_path", DEFAULT_MANIFEST))
    manifest_path = manifest_path if manifest_path.is_absolute() else ROOT / manifest_path
    manifest = read_json(manifest_path)
    checks, errors = {}, []

    def check(name, function):
        try:
            value = function()
            checks[name] = True
            return value
        except (AssertionError, KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
            checks[name] = False
            errors.append({"check": name, "error": str(error)})
            return None

    expected = check("manifest_condition_design", lambda: expected_plans(manifest)) or {}
    check("pre_post_frozen_source_and_artifact_hashes", lambda: verify_provenance(source, config, manifest, manifest_path))
    configured = config.get("expected_plans", [])
    check("run_condition_manifest", lambda: require(
        len(configured) == len(expected) and
        {r["plan_id"]: {k: r[k] for k in (*LABEL_KEYS, "plan_id")} for r in configured} == expected,
        "Run expected-plan manifest differs from frozen conditions"))
    inputs = check("frozen_inputs_calibration_geometry_and_seed_pairing",
                   lambda: audit_inputs(manifest, source, manifest_path))
    directories = {p.name: p for p in (source / "plans").iterdir() if p.is_dir()} if (source / "plans").exists() else {}
    check("all_and_only_frozen_plan_directories", lambda: require(set(directories) == set(expected),
          f"Missing {len(set(expected) - set(directories))}; extra {len(set(directories) - set(expected))} plan directories"))
    records = []
    for index, (plan_id, label) in enumerate(expected.items(), 1):
        if plan_id not in directories:
            continue
        if inputs is not None:
            row = check(f"plan:{plan_id}", lambda: validate_plan(directories[plan_id], label,
                                      inputs[label["scene_id"]], manifest, replay))
            if row is not None:
                records.append(row)
        if index % 50 == 0 or index == len(expected):
            print(f"Formal audit: {index}/{len(expected)} conditions inspected; {len(errors)} integrity errors", flush=True)
    terminal_count = sum((p / "result.json").is_file() for p in directories.values())
    valid_count = sum(r["valid"] for r in records)
    execution_complete = terminal_count == len(expected) == 552 and set(directories) == set(expected)
    integrity_passed = bool(checks) and all(checks.values()) and len(records) == len(expected)
    pair_counts = pairing_denominators(manifest, records)
    groups = []
    for gamma in manifest["candidate_manifest"]["gammas"]:
        for family in ("oracle", "uniform_prior_baseline", *FAMILIES):
            planned = [r for r in expected.values() if r["gamma"] == gamma and r["family"] == family]
            available = [r for r in records if r["gamma"] == gamma and r["family"] == family and r["valid"]]
            groups.append({"gamma": gamma, "family": family, "planned": len(planned), "available_valid": len(available)})
    report = {"protocol_id": manifest["protocol_id"], "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(source), "frozen_manifest_sha256": sha256(manifest_path),
              "passed": integrity_passed, "integrity_passed": integrity_passed,
              "execution_complete": execution_complete,
              "complete_matched_suite": integrity_passed and execution_complete and valid_count == 552,
              "planned_plans": len(expected), "terminal_records": terminal_count,
              "audited_plans": len(records), "valid_plans": valid_count,
              "invalid_or_failed_plans": len(records) - valid_count,
              "status_counts": dict(Counter(r["status"] for r in records)),
              "sampled_episodes_audited": sum(r["episodes_audited"] for r in records),
              "exact_grid_replays": sum(r["replayed"] for r in records),
              "pairing_denominators": pair_counts, "condition_denominators": groups,
              "failures": [{k: r.get(k) for k in (*LABEL_KEYS, "plan_id", "status")}
                           for r in records if not r["valid"]],
              "checks": checks, "errors": errors,
              "failure_handling": manifest["failure_handling"],
              "note": "Integrity passing means recorded artifacts follow the frozen design. Any invalid, "
                      "exception or interrupted plan prevents a complete matched-suite claim; no retries "
                      "or replacement conditions were performed by this audit."}
    if out is not None:
        out = Path(out)
        destination = out if out.suffix == ".json" else out / "formal_audit.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "milestone_4/results/formal")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.source, args.output or args.source)
    print(json.dumps({k: report[k] for k in ("integrity_passed", "execution_complete", "complete_matched_suite",
                     "planned_plans", "valid_plans", "invalid_or_failed_plans", "errors")}, indent=2))
    raise SystemExit(0 if report["integrity_passed"] else 1)


if __name__ == "__main__":
    main()
