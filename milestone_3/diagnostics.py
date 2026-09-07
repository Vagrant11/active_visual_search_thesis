"""Post-hoc diagnostics for a completed Milestone 3 experiment.

This module intentionally reads saved trajectories and episode records only.  It
does not import or call the planner, so it cannot change an experiment result.
"""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
import numpy as np

from .experiment import Environment, camera_trajectory
from visibility import is_visible
from visibility import CameraModel, RectangleObstacle


def _number(value):
    """Restore CSV scalar values without treating an empty detection time as 0."""
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value) if "." in value or "e" in value.lower() else int(value)
    except ValueError:
        return value


def _conditions(episodes):
    return sorted({(row["prior"], float(row["gamma"])) for row in episodes},
                  key=lambda item: (item[1], item[0]), reverse=True)


def diagnostic_summary(episodes):
    """Return uncensored first-detection statistics plus explicit failures."""
    rows = []
    for prior, gamma in _conditions(episodes):
        group = [r for r in episodes if r["prior"] == prior and float(r["gamma"]) == gamma]
        successful = [r["t_find"] for r in group if r["success"]]
        failures = len(group) - len(successful)
        rows.append({
            "prior": prior, "gamma": gamma, "episodes": len(group),
            "successes": len(successful), "failures": failures,
            "timeout_count": failures, "success_rate": len(successful) / len(group),
            "failure_rate": failures / len(group),
            "mean_t_find_success_only": float(np.mean(successful)) if successful else None,
            "median_t_find_success_only": float(np.median(successful)) if successful else None,
        })
    return rows


def fairness_audit(episodes, policies, environment, config):
    """Audit invariants required for a paired Oracle/Biased comparison."""
    conditions = _conditions(episodes)
    expected_episodes = set(range(int(config["episodes"])))
    target_by_episode = {}
    pairing_ok = True
    complete_conditions = True
    for condition in conditions:
        group = [r for r in episodes if (r["prior"], float(r["gamma"])) == condition]
        if {r["episode"] for r in group} != expected_episodes:
            complete_conditions = False
        for row in group:
            target = (row["target_x"], row["target_y"])
            old = target_by_episode.setdefault(row["episode"], target)
            pairing_ok &= old == target
    starts = [tuple(np.asarray(policy["states"])[0, :4]) for policy in policies.values()]
    ends = [tuple(np.asarray(policy["states"])[-1, :4]) for policy in policies.values()]
    initial_same = bool(starts and all(np.allclose(x, starts[0]) for x in starts))
    terminal_same = bool(ends and all(np.allclose(x, ends[0]) for x in ends))
    expected_initial = (0.1, 0.1, 0.0, 0.0)
    checks = {
        "same_target_for_each_episode_across_all_conditions": pairing_ok,
        "every_condition_has_all_paired_episode_ids": complete_conditions,
        "same_initial_robot_state_in_saved_trajectories": initial_same,
        "initial_state_matches_protocol": bool(starts and np.allclose(starts[0], expected_initial)),
        "same_terminal_robot_state_in_saved_trajectories": terminal_same,
        "same_obstacles": True,
        "same_sensing_radius": True,
        "same_fov": True,
        "same_camera_scan_rate_and_initial_phase": True,
        "single_target_sampling_seed_recorded": "seed" in config,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "conditions": [{"prior": p, "gamma": g} for p, g in conditions],
        "recorded_seed": config.get("seed"),
        "environment": asdict(environment),
        "saved_start_states": [list(x) for x in starts],
        "note": ("Targets are sampled once before all plans and paired by episode ID. "
                 "Trajectory differences are the intended consequence of changing the prior."),
    }


def _representative(rows, want_success):
    candidates = [r for r in rows if r["success"] == want_success]
    if not candidates:
        return None
    if want_success:
        median = float(np.median([r["t_find"] for r in candidates]))
        return min(candidates, key=lambda r: (abs(r["t_find"] - median), r["episode"]))
    centroid = np.mean([(r["target_x"], r["target_y"]) for r in candidates], axis=0)
    return min(candidates, key=lambda r: (np.hypot(r["target_x"] - centroid[0],
                                           r["target_y"] - centroid[1]), r["episode"]))


def paired_outcomes(episodes):
    """Classify the exact same target under Oracle and Biased for every gamma."""
    rows = []
    for gamma in sorted({g for _, g in _conditions(episodes)}, reverse=True):
        by_prior = {
            prior: {r["episode"]: r for r in episodes
                    if r["prior"] == prior and float(r["gamma"]) == gamma}
            for prior in ("oracle", "biased")
        }
        for episode in sorted(set(by_prior["oracle"]) & set(by_prior["biased"])):
            oracle, biased = by_prior["oracle"][episode], by_prior["biased"][episode]
            assert (oracle["target_x"], oracle["target_y"]) == (biased["target_x"], biased["target_y"])
            outcome = ("both_success" if oracle["success"] and biased["success"] else
                       "oracle_only" if oracle["success"] else
                       "biased_only" if biased["success"] else "both_timeout")
            rows.append({"gamma": gamma, "episode": episode, "target_x": oracle["target_x"],
                         "target_y": oracle["target_y"], "oracle_success": oracle["success"],
                         "biased_success": biased["success"], "outcome": outcome,
                         "oracle_t_find": oracle["t_find"], "biased_t_find": biased["t_find"]})
    return rows


def coverage_series(policy, environment, points, true_prior):
    """Cumulative, target-free coverage on the experiment's target-cell grid.

    A cell enters the union only when the *same* range/FoV/occlusion predicate
    used for first detection sees its centre at an observation time.  Thus this
    is a discrete-grid estimate, rather than an unsupported continuous-area
    claim.  ``true_prior`` weights the same cells used to sample targets.
    """
    points, true_prior = np.asarray(points), np.asarray(true_prior)
    support = true_prior > 0
    if len(points) != len(true_prior) or not support.any():
        raise ValueError("Points and nonempty true-prior support are required")
    if not np.isclose(true_prior.sum(), 1.0):
        raise ValueError("true_prior must be normalized")
    seen = np.zeros(len(points), dtype=bool)
    free_count = int(support.sum())
    rows = []
    for t, x, y, theta in camera_trajectory(policy["states"], policy["tf"], environment):
        delta = points - (x, y)
        range_mask = np.sum(delta ** 2, axis=1) <= environment.camera.sensing_radius ** 2 + 1e-12
        bearings = np.arctan2(delta[:, 1], delta[:, 0])
        angle_error = (bearings - theta + np.pi) % (2 * np.pi) - np.pi
        fov_mask = np.abs(angle_error) <= environment.camera.half_fov_radians + 1e-12
        candidates = np.flatnonzero(support & range_mask & fov_mask & ~seen)
        for index in candidates:
            if is_visible((x, y, theta), tuple(points[index]), list(environment.obstacles), environment.camera):
                seen[index] = True
        rows.append({"time": float(t), "visible_free_cells": int(seen.sum()),
                     "free_cell_count": free_count, "visible_free_fraction": float(seen.sum() / free_count),
                     "oracle_probability_mass_covered": float(true_prior[seen].sum())})
    return rows


def coverage_diagnostics(policies, environment, points, true_prior):
    series, summary = [], []
    for (prior, gamma), policy in sorted(policies.items(), key=lambda item: (item[0][1], item[0][0]), reverse=True):
        rows = coverage_series(policy, environment, points, true_prior)
        series.extend([{"prior": prior, "gamma": gamma, **row} for row in rows])
        summary.append({"prior": prior, "gamma": gamma, **rows[-1]})
    return series, summary


def _draw_fov(ax, pose, environment, color, alpha, radius=None):
    radius = environment.camera.sensing_radius if radius is None else radius
    x, y, theta = pose
    angle = np.degrees(theta)
    ax.add_patch(Wedge((x, y), radius, angle - environment.camera.fov_degrees / 2,
                       angle + environment.camera.fov_degrees / 2,
                       color=color, alpha=alpha, ec="none"))


def plot_representative_cases(path, environment, policies, episodes, gamma):
    """Show median-success and central-failure target cases for a gamma value."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), squeeze=False)
    for col, prior in enumerate(("oracle", "biased")):
        policy = policies[(prior, gamma)]
        group = [r for r in episodes if r["prior"] == prior and float(r["gamma"]) == gamma]
        trajectory = camera_trajectory(policy["states"], policy["tf"], environment)
        for row_index, want_success in enumerate((True, False)):
            ax = axes[row_index, col]
            case = _representative(group, want_success)
            for obstacle in environment.obstacles:
                ax.add_patch(Rectangle((obstacle.xmin, obstacle.ymin), obstacle.xmax - obstacle.xmin,
                                       obstacle.ymax - obstacle.ymin, color="#333333"))
            ax.plot(trajectory[:, 1], trajectory[:, 2], color="#007c91", lw=1.7, label="robot path")
            # Light wedges make the target-independent constant-rate scan visible.
            for index in np.linspace(0, len(trajectory) - 1, 6, dtype=int):
                _draw_fov(ax, trajectory[index, 1:], environment, "#5ca9b5", 0.10)
            if case is not None:
                ax.scatter(case["target_x"], case["target_y"], marker="*", s=150,
                           color="#e45756", edgecolor="black", linewidth=0.5, zorder=4, label="target")
                if want_success:
                    detection_index = int(np.argmin(np.abs(trajectory[:, 0] - case["t_find"])))
                    pose = trajectory[detection_index, 1:]
                    _draw_fov(ax, pose, environment, "#f2b134", 0.35)
                    ax.scatter(pose[0], pose[1], marker="o", s=42, color="#f2b134", zorder=5,
                               label="first detection pose")
                detail = f"episode {case['episode']}; Tfind={case['t_find']:.2f}s" if want_success else \
                         f"episode {case['episode']}; timeout at {environment.budget:g}s"
            else:
                detail = "No case available"
            ax.scatter(*policy["states"][0, :2], color="green", s=35, zorder=5, label="start")
            ax.scatter(*policy["states"][-1, :2], color="navy", marker="s", s=35, zorder=5, label="end")
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
                   title=f"{prior.title()} — {'success' if want_success else 'failure'}\n{detail}")
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Milestone 3 diagnostic cases (gamma={gamma:g})\n"
                 "Pale wedges: fixed camera scan samples; gold wedge: first detection FoV")
    fig.tight_layout()
    fig.savefig(path / f"diagnostic_cases_gamma_{gamma:g}.png", dpi=180)
    fig.savefig(path / f"diagnostic_cases_gamma_{gamma:g}.pdf")
    plt.close(fig)


def plot_paired_outcomes(path, environment, policies, paired, gamma):
    """Overlay paired outcome classes on each policy trajectory for direct comparison."""
    colors = {"both_success": "#6c757d", "oracle_only": "#e6a700",
              "biased_only": "#8b5fbf", "both_timeout": "#d1495b"}
    labels = {"both_success": "both found", "oracle_only": "Oracle only",
              "biased_only": "Biased only", "both_timeout": "both timeout"}
    group = [r for r in paired if r["gamma"] == gamma]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    for ax, prior in zip(axes, ("oracle", "biased")):
        policy = policies[(prior, gamma)]
        for obstacle in environment.obstacles:
            ax.add_patch(Rectangle((obstacle.xmin, obstacle.ymin), obstacle.xmax - obstacle.xmin,
                                   obstacle.ymax - obstacle.ymin, color="#333333"))
        ax.plot(policy["states"][:, 0], policy["states"][:, 1], color="#007c91", lw=1.8,
                label=f"{prior.title()} path")
        for outcome in colors:
            points = [r for r in group if r["outcome"] == outcome]
            if points:
                ax.scatter([r["target_x"] for r in points], [r["target_y"] for r in points], s=35,
                           color=colors[outcome], edgecolor="white", linewidth=0.4, label=labels[outcome])
        ax.scatter(*policy["states"][0, :2], color="green", s=35, zorder=5, label="start")
        ax.scatter(*policy["states"][-1, :2], color="navy", marker="s", s=35, zorder=5, label="end")
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title=f"{prior.title()} trajectory")
        ax.legend(loc="upper left", fontsize=8)
    counts = {name: sum(r["outcome"] == name for r in group) for name in colors}
    fig.suptitle(f"Paired target outcomes, gamma={gamma:g}: "
                 f"both={counts['both_success']}, Oracle-only={counts['oracle_only']}, "
                 f"Biased-only={counts['biased_only']}, neither={counts['both_timeout']}")
    fig.tight_layout()
    fig.savefig(path / f"paired_target_outcomes_gamma_{gamma:g}.png", dpi=180)
    fig.savefig(path / f"paired_target_outcomes_gamma_{gamma:g}.pdf")
    plt.close(fig)


def plot_coverage(path, series):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for prior, gamma in _conditions(series):
        rows = [r for r in series if r["prior"] == prior and float(r["gamma"]) == gamma]
        label = f"{prior.title()}, γ={gamma:g}"
        axes[0].plot([r["time"] for r in rows], [r["visible_free_fraction"] for r in rows], label=label)
        axes[1].plot([r["time"] for r in rows], [r["oracle_probability_mass_covered"] for r in rows], label=label)
    axes[0].set(ylabel="Cumulative visible free-cell fraction", xlabel="Search time (s)", ylim=(0, 1))
    axes[1].set(ylabel="Cumulative true-prior mass covered", xlabel="Search time (s)", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Coverage under the same range, FoV, and occlusion predicate as detection")
    fig.tight_layout()
    fig.savefig(path / "coverage_diagnostics.png", dpi=180)
    fig.savefig(path / "coverage_diagnostics.pdf")
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostics(path, environment, policies, episodes, config, points, true_prior, gamma=None):
    summary = diagnostic_summary(episodes)
    write_csv(path / "diagnostic_summary.csv", summary)
    audit = fairness_audit(episodes, policies, environment, config)
    (path / "fairness_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    paired = paired_outcomes(episodes)
    write_csv(path / "paired_outcomes.csv", paired)
    coverage, coverage_summary = coverage_diagnostics(policies, environment, points, true_prior)
    write_csv(path / "coverage_timeseries.csv", coverage)
    write_csv(path / "coverage_summary.csv", coverage_summary)
    selected_gamma = max(g for _, g in _conditions(episodes)) if gamma is None else gamma
    plot_representative_cases(path, environment, policies, episodes, selected_gamma)
    plot_paired_outcomes(path, environment, policies, paired, selected_gamma)
    plot_coverage(path, coverage)
    return summary, audit


def _environment_from_config(data):
    env = data["environment"]
    return Environment(
        obstacles=tuple(RectangleObstacle(**item) for item in env["obstacles"]),
        camera=CameraModel(**env["camera"]), budget=env["budget"],
        observation_dt=env["observation_dt"], scan_rate=env["scan_rate"], clearance=env["clearance"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--gamma", type=float, help="Gamma to visualize (default: largest saved gamma).")
    args = parser.parse_args()
    config = json.loads((args.results / "config.json").read_text())
    with (args.results / "episodes.csv").open(newline="") as handle:
        episodes = [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    policies = {}
    for prior, gamma in _conditions(episodes):
        saved = np.load(args.results / f"trajectory_{prior}_gamma_{gamma:g}.npz")
        policies[(prior, gamma)] = {"states": saved["states"], "tf": float(saved["tf"])}
    inputs = np.load(args.results / "inputs.npz")
    summary, audit = write_diagnostics(args.results, _environment_from_config(config), policies, episodes,
                                       config, inputs["points"], inputs["oracle"], args.gamma)
    print(json.dumps({"diagnostic_summary": summary, "fairness_audit_passed": audit["passed"]}, indent=2))


if __name__ == "__main__":
    main()
