"""Protocol v2 preflight, part 2/2: engineering-only planner feasibility/runtime checks.

This is the ONLY v2 preflight script that imports the frozen Milestone 3
planner, and it does not modify it. A small, FIXED set of checks is declared
below, selected purely by geometry/gamma/prior-type coverage before any of them
were run:

  1. cx0.25_cy0.25_layoutA, oracle, gamma=0.10  -- no-obstacle baseline
  2. cx0.25_cy0.25_layoutB, oracle, gamma=0.10  -- one rectangle (v1-like obstacle)
  3. cx0.25_cy0.25_layoutC, oracle, gamma=0.10  -- two rectangles (hardest obstacle)
  4. cx0.25_cy0.25_layoutB, oracle, gamma=0.05  -- tighter ergodic bound, same scene as #2
  5. cx0.25_cy0.25_layoutB, uniform_prior_baseline, gamma=0.10 -- confirms the true
     uniform-prior planning baseline (a fresh planned trajectory, not the pilot's
     post-hoc reweighting diagnostic) is numerically feasible

These are not a comparison of which condition "performs better": only solver
validity and wall-clock runtime are recorded, and this list must not be edited
after seeing results. This script never launches the formal 552-plan suite.

Run: python -m milestone_4.run_engineering_check
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .protocol import MAXITER, NODES, ROOT, protocol_manifest
from .scene_manifest import scenes, uniform_prior_baseline, scene_environment, scene_grid, gaussian_density, SIGMA, BACKGROUND

CHECKS = (
    {"check_id": "layoutA_oracle_g010", "scene_id": "cx0.25_cy0.25_layoutA", "prior": "oracle", "gamma": 0.10,
     "rationale": "no-obstacle geometry baseline"},
    {"check_id": "layoutB_oracle_g010", "scene_id": "cx0.25_cy0.25_layoutB", "prior": "oracle", "gamma": 0.10,
     "rationale": "single-rectangle geometry, matches v1's obstacle shape"},
    {"check_id": "layoutC_oracle_g010", "scene_id": "cx0.25_cy0.25_layoutC", "prior": "oracle", "gamma": 0.10,
     "rationale": "two-rectangle geometry, hardest collision constraint set"},
    {"check_id": "layoutB_oracle_g005", "scene_id": "cx0.25_cy0.25_layoutB", "prior": "oracle", "gamma": 0.05,
     "rationale": "tighter ergodicity bound, same scene as layoutB_oracle_g010"},
    {"check_id": "layoutB_uniform_g010", "scene_id": "cx0.25_cy0.25_layoutB", "prior": "uniform_prior_baseline",
     "gamma": 0.10, "rationale": "true uniform-prior planning-baseline feasibility, same scene as layoutB_oracle_g010"},
)


def scene_by_id(scene_id):
    return next(s for s in scenes() if s.scene_id == scene_id)


def scene_prior(scene, name):
    environment = scene_environment(scene)
    points, free = scene_grid(environment)
    if name == "oracle":
        return points, gaussian_density(points, free, scene.center, SIGMA, BACKGROUND)
    if name == "uniform_prior_baseline":
        return points, uniform_prior_baseline(scene)
    raise ValueError(f"Engineering check does not define prior: {name}")


def run_checks(check_ids=None):
    checks = [c for c in CHECKS if check_ids is None or c["check_id"] in check_ids]
    if not checks:
        raise ValueError("No matching checks")
    # Reuses the same frozen-source guard as the v1 pilot before any planner import.
    protocol_manifest()
    from milestone_3.planner import plan

    results = []
    for check in checks:
        scene = scene_by_id(check["scene_id"])
        points, probabilities = scene_prior(scene, check["prior"])
        environment = scene_environment(scene)
        print(f"Engineering check {check['check_id']}: {check['rationale']} ...", flush=True)
        started = perf_counter()
        policy = plan(points, probabilities, check["gamma"], environment, NODES, MAXITER)
        wall_seconds = perf_counter() - started
        diagnostics = policy["diagnostics"]
        results.append({**check, "wall_seconds": wall_seconds,
                        "optimization_seconds": diagnostics["optimization_seconds"],
                        "planning_seconds_including_compile": diagnostics["planning_seconds_including_compile"],
                        "valid": diagnostics["valid"], "solver_success": diagnostics["solver_success"],
                        "iterations": diagnostics["iterations"],
                        "max_equality_residual": diagnostics["max_equality_residual"],
                        "max_inequality_violation": diagnostics["max_inequality_violation"],
                        "collision_free": diagnostics["collision_free"], "tf": policy["tf"]})
        print(f"  valid={diagnostics['valid']}; wall={wall_seconds:.1f}s; "
              f"optimize={diagnostics['optimization_seconds']:.1f}s; tf={policy['tf']:.3f}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/preflight")
    parser.add_argument("--checks", nargs="+", default=None,
                        help="Subset of check_id values to run (default: all declared checks)")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    existing = {}
    check_path = out / "engineering_check.json"
    if check_path.exists():
        existing = {r["check_id"]: r for r in json.loads(check_path.read_text())["results"]}
    results = run_checks(args.checks)
    for r in results:
        existing[r["check_id"]] = r
    ordered = [existing[c["check_id"]] for c in CHECKS if c["check_id"] in existing]
    payload = {"declared_checks": list(CHECKS), "results": ordered,
              "complete": {c["check_id"] for c in CHECKS} <= set(existing),
              "note": "Engineering-only feasibility/runtime checks; not a formal-suite result, "
                      "not used to select or tune research conditions."}
    check_path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=list) + "\n")
    print(f"Engineering check results written to {check_path}")


if __name__ == "__main__":
    main()
