"""Run from repository root: python -m milestone_3.run_experiment."""

import argparse
import csv
from dataclasses import asdict
import json
from importlib.metadata import version
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .experiment import Environment, evaluate, js_divergence, make_priors
from .diagnostics import write_diagnostics
from .planner import plan


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path, environment, points, priors, targets, runs, episodes):
    fig, axes = plt.subplots(2, len(runs), figsize=(4 * len(runs), 7), squeeze=False)
    for col, (name, gamma, policy) in enumerate(runs):
        ax = axes[0, col]
        ax.scatter(points[:, 0], points[:, 1], c=priors[name], cmap="YlOrRd", s=10,
                   vmin=0, vmax=max(p.max() for p in priors.values()))
        for obstacle in environment.obstacles:
            ax.add_patch(Rectangle((obstacle.xmin, obstacle.ymin), obstacle.xmax-obstacle.xmin,
                                   obstacle.ymax-obstacle.ymin, color="#333333"))
        states = policy["states"]
        ax.plot(states[:, 0], states[:, 1], color="#007c91", lw=1.5)
        ax.scatter(targets[:, 0], targets[:, 1], s=8, marker="x", color="#444444", alpha=0.4)
        ax.scatter(*states[0, :2], marker="o", color="green")
        ax.scatter(*states[-1, :2], marker="s", color="blue")
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
               title=f"{name}, gamma={gamma:g}\ntf={policy['tf']:.2f}, valid={policy['diagnostics']['valid']}")
        rows = [r for r in episodes if r["prior"] == name and r["gamma"] == gamma]
        ax = axes[1, col]
        if rows:
            times = np.linspace(0, environment.budget, 301)
            cdf = [sum(r["success"] and r["t_find"] <= t for r in rows) / len(rows) for t in times]
            ax.step(times, cdf, where="post")
        else:
            ax.text(0.5, 0.5, "Invalid plan: no task evaluation", ha="center", transform=ax.transAxes)
        ax.set(xlim=(0, environment.budget), ylim=(0, 1.02), xlabel="Search time (s)",
               ylabel="Fraction of all targets found")
        ax.grid(alpha=0.2)
    fig.suptitle("Milestone 3: paired targets, constant-rate camera scan, ideal detector")
    fig.tight_layout()
    fig.savefig(path / "prior_comparison.png", dpi=160)
    fig.savefig(path / "prior_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.05])
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--budget", type=float, default=15.0)
    parser.add_argument("--observation-dt", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if (args.episodes < 1 or not np.isfinite([args.budget, args.observation_dt]).all()
            or args.budget <= 0 or args.observation_dt <= 0):
        parser.error("episodes, budget and observation-dt must be positive")
    if args.nodes < 8 or args.maxiter < 1 or any(not np.isfinite(g) or g <= 0 for g in args.gammas):
        parser.error("nodes >= 8, maxiter >= 1 and finite positive gammas required")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    environment = Environment(budget=args.budget, observation_dt=args.observation_dt)
    points, priors = make_priors(environment)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(points), size=args.episodes, p=priors["oracle"])
    targets = points[indices]  # Identical target realizations for every plan.
    np.savez(out / "inputs.npz", points=points, targets=targets, target_indices=indices, **priors)
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1] / "external/time_optimal_ergodic_search"),
         "rev-parse", "HEAD"], text=True).strip()
    config = {**vars(args), "output": str(out), "environment": asdict(environment),
              "versions": {name: version(name) for name in ("jax", "numpy", "scipy", "matplotlib")},
              "upstream_commit": upstream_commit, "protocol": "fixed prior; pan theta(t)=pi*t/2; hold endpoint until budget",
              "notes": "single scene; discrete target-cell probabilities; oracle knows distribution, not target"}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    episodes, summaries, diagnostics, runs = [], [], [], []
    for gamma in args.gammas:
        for name, probabilities in priors.items():
            print(f"Planning {name}, gamma={gamma:g} ...", flush=True)
            policy = plan(points, probabilities, gamma, environment, args.nodes, args.maxiter)
            rows, summary = evaluate(policy, environment, targets)
            distance = js_divergence(probabilities, priors["oracle"])
            label = {"prior": name, "gamma": gamma}
            episodes.extend([{**label, **row} for row in rows])
            # Uniform schema also for invalid plans.
            summary_defaults = dict.fromkeys(["success_rate", "mean_capped_time", "median_capped_time",
                "mean_t_find_success_only", "median_t_find_success_only", "mean_path_length_until_stop"])
            summaries.append({**label, "js_divergence_nats": distance, "optimized_tf": policy["tf"],
                              **summary_defaults, **summary})
            diagnostics.append({**label, **policy["diagnostics"]})
            np.savez(out / f"trajectory_{name}_gamma_{gamma:g}.npz", states=policy["states"],
                     controls=policy["controls"], tf=policy["tf"])
            runs.append((name, gamma, policy))
            print(json.dumps({**label, **summary, **policy["diagnostics"]}), flush=True)
    write_csv(out / "episodes.csv", episodes)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "planner_diagnostics.csv", diagnostics)
    plot_results(out, environment, points, priors, targets, runs, episodes)
    policies = {(name, gamma): policy for name, gamma, policy in runs}
    write_diagnostics(out, environment, policies, episodes, config, points, priors["oracle"])
    print(f"Results: {out}")
    if any(not row["valid"] for row in diagnostics):
        raise SystemExit("Some plans failed validation; see planner_diagnostics.csv. No metrics reported for those plans.")


if __name__ == "__main__":
    main()
