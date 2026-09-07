"""Gate 2, part 2: measure evaluation cost, including the primary full-grid calculation.

No planner import, zero planning. Times `milestone_3.experiment.evaluate` (the
sampled evaluator) and `milestone_4.analyze_pilot.first_seen_grid` (the exact
full-grid replay behind the primary endpoints) on a synthetic, non-optimized
trajectory of realistic shape/scale (the planner's own deterministic initial
guess, at the mean tf observed across the Gate-3 engineering checks) so the
projection does not require any further planner call.

Run: python -m milestone_4.run_gate2_cost_check
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from milestone_3.experiment import evaluate
from .analyze_pilot import first_seen_grid
from .protocol import NODES, ROOT
from .scene_manifest import BACKGROUND, SIGMA, gaussian_density, scene_environment, scene_grid, scenes

# Same deterministic initial-guess waypoints as milestone_3.planner.plan; reused here
# purely for a realistic-shape, realistic-scale trajectory, not as a planner call.
WAYPOINTS = np.array([[0.1, 0.1], [0.8, 0.2], [0.8, 0.8], [0.9, 0.9]])
MEAN_ENGINEERING_CHECK_TF = 3.816429855  # mean tf across the 5 Gate-3 checks (engineering_check.json)


def synthetic_policy(nodes=NODES, tf=MEAN_ENGINEERING_CHECK_TF):
    xy = np.column_stack([np.interp(np.linspace(0, 3, nodes), np.arange(4), WAYPOINTS[:, i]) for i in (0, 1)])
    states = np.column_stack((xy, np.zeros_like(xy)))
    return {"states": states, "tf": tf, "diagnostics": {"valid": True}}


def measure(seeds, targets_per_seed, projected_plan_count):
    scene = next(s for s in scenes() if s.scene_id == "cx0.25_cy0.25_layoutB")  # same scene as engineering checks
    environment = scene_environment(scene)
    points, free = scene_grid(environment)
    true_prior = gaussian_density(points, free, scene.center, SIGMA, BACKGROUND)
    support = true_prior > 0
    policy = synthetic_policy()

    started = perf_counter()
    first = first_seen_grid(policy, environment, points, support)
    full_grid_seconds = perf_counter() - started
    detection_rate = float(true_prior[np.isfinite(first)].sum())

    all_targets = []
    for seed in seeds:
        indices = np.random.default_rng(seed).choice(len(points), size=targets_per_seed, p=true_prior)
        all_targets.append(points[indices])
    started = perf_counter()
    total_episodes, total_success = 0, 0
    for targets in all_targets:
        rows, summary = evaluate(policy, environment, targets)
        total_episodes += len(rows)
        total_success += sum(r["success"] for r in rows)
    sampled_eval_seconds = perf_counter() - started

    per_plan_seconds = full_grid_seconds + sampled_eval_seconds
    return {
        "scene_id": scene.scene_id, "grid_points": len(points),
        "seeds": list(seeds), "targets_per_seed": targets_per_seed,
        "total_sampled_targets": total_episodes,
        "full_grid_replay_seconds": full_grid_seconds,
        "full_grid_exact_detection_rate": detection_rate,
        "sampled_evaluate_seconds": sampled_eval_seconds,
        "sampled_success_rate": total_success / total_episodes if total_episodes else None,
        "per_plan_evaluation_seconds": per_plan_seconds,
        "projected_plan_count": projected_plan_count,
        "projected_total_evaluation_seconds": per_plan_seconds * projected_plan_count,
        "projected_total_evaluation_hours": per_plan_seconds * projected_plan_count / 3600,
        "note": "Timed on a synthetic (non-optimized) trajectory of realistic shape/scale; "
                "no planner call. Evaluation cost is dominated by grid/target count and "
                "trajectory sample count, not path optimality.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/preflight")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
    parser.add_argument("--targets-per-seed", type=int, default=256)
    parser.add_argument("--projected-plan-count", type=int, default=552)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = measure(args.seeds, args.targets_per_seed, args.projected_plan_count)
    (out / "gate2_cost_check.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Written to {out / 'gate2_cost_check.json'}")


if __name__ == "__main__":
    main()
