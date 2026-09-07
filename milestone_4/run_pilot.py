"""Explicit small pilot using saved calibration and the unchanged M3 planner."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from milestone_3.experiment import evaluate, js_divergence
from milestone_3.diagnostics import coverage_series
from .calibrate_priors import prior_id, write_csv
from .prior_errors import FAMILIES, PriorErrorGenerator, validate_prior
from .protocol import (EPISODES, GAMMAS, JS_TOLERANCE, LEVELS, MAXITER, NODES, ROOT,
                       SEEDS, paired_targets, protocol_manifest)


def load_calibration(path):
    manifest = json.loads((path / "protocol.json").read_text())
    expected = json.loads(json.dumps(protocol_manifest()))
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Calibration protocol mismatch: {key}")
    if not manifest.get("calibration_passed"):
        raise ValueError("Calibration contains unmatched conditions")
    generator = PriorErrorGenerator()
    if manifest["prior_errors"] != json.loads(json.dumps(generator.metadata())):
        raise ValueError("Error design differs from the fixed protocol")
    with (path / "calibration.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = {prior_id(f, level) for f in FAMILIES for level in LEVELS}
    if len(rows) != len(expected_ids) or {r["prior_id"] for r in rows} != expected_ids:
        raise ValueError("Calibration must contain exactly the 12 fixed conditions")
    priors = {}
    with np.load(path / "priors.npz", allow_pickle=False) as data:
        for key, values in (("points", generator.points), ("true_prior", generator.true_prior),
                            ("free_mask", generator.support)):
            if not np.array_equal(data[key], values):
                raise ValueError(f"Calibration changed fixed input: {key}")
        for row in rows:
            key = row["prior_id"]
            family, level = row["family"], float(row["target_js_nats"])
            if family not in FAMILIES or level not in LEVELS or prior_id(family, level) != key:
                raise ValueError(f"Condition label mismatch: {key}")
            prior = validate_prior(data[key], generator.support).copy()
            actual = js_divergence(generator.true_prior, prior)
            if (row["status"] != "matched" or row["valid_prior"] != "True"
                    or abs(actual - level) > JS_TOLERANCE
                    or abs(actual - float(row["actual_js_nats"])) > 1e-12
                    or hashlib.sha256(prior.tobytes()).hexdigest() != manifest["prior_sha256"][key]
                    or not np.allclose(prior, generator.generate(family, float(row["parameter_value"])), rtol=0, atol=1e-15)):
                raise ValueError(f"Calibration integrity check failed: {key}")
            priors[key] = prior
    return generator, rows, priors, manifest


def paired_outcomes(episodes):
    """Pair every error condition with oracle, using gamma, seed and episode ID."""
    oracle = {(r["gamma"], r["seed"], r["episode"]): r for r in episodes if r["prior_id"] == "oracle"}
    rows = []
    for r in episodes:
        if r["prior_id"] == "oracle":
            continue
        baseline = oracle.get((r["gamma"], r["seed"], r["episode"]))
        if baseline is None:  # Invalid baseline has no task metrics or pairs.
            continue
        if (r["target_x"], r["target_y"]) != (baseline["target_x"], baseline["target_y"]):
            raise ValueError("Paired target coordinates differ")
        outcome = ("both_success" if baseline["success"] and r["success"] else
                   "oracle_only" if baseline["success"] else
                   "error_only" if r["success"] else "both_timeout")
        rows.append({k: r[k] for k in ("prior_id", "family", "target_js_nats", "gamma", "seed",
                                      "episode", "target_x", "target_y")}
                    | {"oracle_success": baseline["success"], "error_success": r["success"],
                       "oracle_t_find": baseline["t_find"], "error_t_find": r["t_find"], "outcome": outcome})
    return rows


def plot_trajectories(out, generator, rows, priors, policies):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for gamma in GAMMAS:
        fig, axes = plt.subplots(4, 3, figsize=(11, 12), layout="constrained")
        for ax, row in zip(axes.flat, rows):
            key = row["prior_id"]
            policy = policies[key, gamma]
            ax.imshow(np.ma.masked_where(~generator.support, priors[key]).reshape(40, 40),
                      extent=(0, 1, 0, 1), origin="lower", cmap="YlOrRd",
                      vmin=0, vmax=max(p.max() for p in priors.values()))
            for o in generator.environment.obstacles:
                ax.add_patch(Rectangle((o.xmin, o.ymin), o.xmax-o.xmin, o.ymax-o.ymin, color="#333333"))
            baseline = policies["oracle", gamma]["states"]
            ax.plot(baseline[:, 0], baseline[:, 1], "--", color="#777777", label="oracle")
            states = policy["states"]
            ax.plot(states[:, 0], states[:, 1], color="#007c91", label="error prior")
            ax.plot(.75, .70, "+", color="black")
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
                   title=f"{row['family'].replace('_', ' ')}; JS={float(row['target_js_nats']):g}\n"
                         f"tf={policy['tf']:.2f}s; valid={policy['diagnostics']['valid']}")
            ax.title.set_fontsize(9)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Milestone 4 pilot: gamma={gamma:g}; fixed camera protocol")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"trajectories_gamma_{gamma:g}.{ext}", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT / "milestone_4/results/calibration")
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/pilot")
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT / "milestone_3" or ROOT / "milestone_3" in out.parents:
        parser.error("Milestone 3 is frozen")
    if out.exists() and any(out.iterdir()):
        parser.error("Pilot output must be empty; choose a new directory to preserve previous runs")
    generator, calibration, priors, manifest = load_calibration(args.calibration)
    # Delayed until all input checks pass. Calibration alone never imports it.
    from milestone_3.planner import plan

    out.mkdir(parents=True, exist_ok=True)
    manifest["run_type"] = "small pilot; single fixed scene; no formal inference"
    manifest["calibration_directory"] = str(args.calibration.resolve().relative_to(ROOT)) if args.calibration.resolve().is_relative_to(ROOT) else str(args.calibration.resolve())
    (out / "config.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    targets = {seed: paired_targets(generator.points, generator.true_prior, seed) for seed in SEEDS}
    np.savez(out / "inputs.npz", points=generator.points, true_prior=generator.true_prior, **priors,
             **{f"target_indices_seed_{seed}": pair[0] for seed, pair in targets.items()},
             **{f"targets_seed_{seed}": pair[1] for seed, pair in targets.items()})
    conditions = [{"prior_id": "oracle", "family": "oracle", "target_js_nats": 0.0,
                   "actual_js_nats": 0.0}] + calibration
    episodes, summaries, diagnostics, coverage, coverage_summary, policies = [], [], [], [], [], {}

    def checkpoint():
        for filename, records in (("episodes", episodes), ("summary", summaries),
                                  ("planner_diagnostics", diagnostics), ("coverage_timeseries", coverage),
                                  ("coverage_summary", coverage_summary), ("paired_outcomes", paired_outcomes(episodes))):
            write_csv(out / f"{filename}.csv", records)

    for gamma in GAMMAS:
        for condition in conditions:
            key = condition["prior_id"]
            probabilities = generator.true_prior if key == "oracle" else priors[key]
            label = {"prior_id": key, "family": condition["family"],
                     "target_js_nats": float(condition["target_js_nats"]), "gamma": gamma}
            print(f"Planning {key}, gamma={gamma:g} ...", flush=True)
            policy = plan(generator.points, probabilities, gamma, generator.environment, NODES, MAXITER)
            policies[key, gamma] = policy
            diagnostics.append({**label, **policy["diagnostics"]})
            np.savez(out / f"trajectory_{key}_gamma_{gamma:g}.npz", states=policy["states"],
                     controls=policy["controls"], tf=policy["tf"])
            final_coverage = {"visible_free_fraction": None, "oracle_probability_mass_covered": None}
            if policy["diagnostics"]["valid"]:
                series = coverage_series(policy, generator.environment, generator.points, generator.true_prior)
                coverage.extend({**label, **r} for r in series)
                coverage_summary.append({**label, **series[-1]})
                final_coverage = {k: series[-1][k] for k in final_coverage}
            for seed, (_, sampled_targets) in targets.items():
                records, summary = evaluate(policy, generator.environment, sampled_targets)
                episodes.extend({**label, "seed": seed, **r} for r in records)
                defaults = dict.fromkeys(("success_rate", "mean_capped_time", "median_capped_time",
                                         "mean_t_find_success_only", "median_t_find_success_only",
                                         "mean_path_length_until_stop"))
                success_count = sum(r["success"] for r in records) if records else None
                summaries.append({**label, "seed": seed, "actual_js_nats": float(condition["actual_js_nats"]),
                                  "optimized_tf": policy["tf"], **defaults, **summary,
                                  "success_count": success_count,
                                  "timeout_count": len(records) - success_count if records else None,
                                  "timeout_rate": 1 - summary["success_rate"] if records else None,
                                  **final_coverage})
            checkpoint()
            print(f"  valid={policy['diagnostics']['valid']}; tf={policy['tf']:.4f}; "
                  f"eq={policy['diagnostics']['max_equality_residual']:.2g}; "
                  f"ineq={policy['diagnostics']['max_inequality_violation']:.2g}", flush=True)

    pairs = paired_outcomes(episodes)
    target_match = all(np.array_equal([r["target_x"], r["target_y"]], targets[r["seed"]][1][r["episode"]]) for r in episodes)
    complete = all(sum(r["prior_id"] == key and r["gamma"] == gamma and r["seed"] == seed for r in episodes)
                   == (EPISODES if policy["diagnostics"]["valid"] else 0)
                   for (key, gamma), policy in policies.items() for seed in SEEDS)
    valid_count = sum(p["diagnostics"]["valid"] for p in policies.values())
    expected_pairs = sum(EPISODES * len(SEEDS) for (key, gamma), p in policies.items()
                         if key != "oracle" and p["diagnostics"]["valid"] and policies["oracle", gamma]["diagnostics"]["valid"])
    valid_policies = [p for p in policies.values() if p["diagnostics"]["valid"]]
    checks = {"all_26_plans_valid": len(policies) == 26 and valid_count == 26,
              "paired_targets_match_saved_inputs": target_match,
              "episode_counts_match_plan_validity": complete,
              "paired_outcome_count_correct": len(pairs) == expected_pairs,
              "initial_states_match": bool(valid_policies) and all(np.allclose(p["states"][0], [.1, .1, 0, 0], atol=1e-5) for p in valid_policies),
              "terminal_states_match": bool(valid_policies) and all(np.allclose(p["states"][-1], [.9, .9, 0, 0], atol=1e-5) for p in valid_policies)}
    audit = {"passed": all(checks.values()), "checks": checks, "valid_plans": valid_count,
             "total_plans": len(policies), "episodes": len(episodes), "paired_outcomes": len(pairs),
             "note": "Seeds vary evaluation targets only; each deterministic plan is reused across target seeds."}
    (out / "pilot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    plot_trajectories(out, generator, calibration, priors, policies)
    print(json.dumps(audit), flush=True)
    if not audit["passed"]:
        raise SystemExit("Pilot checks failed; inspect diagnostics before expansion. No invalid-plan search metrics reported.")


if __name__ == "__main__":
    main()
