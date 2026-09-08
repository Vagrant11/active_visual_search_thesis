"""Report the frozen v2 formal suite from audited, saved artifacts only.

No planner calls, calibration, resampling, confidence intervals, or ranking.
Run after audit: python -m milestone_4.analyze_formal
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .analyze_pilot import (COLORS, NAMES, paired_stats, read_csv, require,
                            save_figure, stats)
from .prior_errors import FAMILIES
from .protocol import ROOT


PRIMARY = ("exact_detection_probability", "exact_weighted_mean_t_find",
           "exact_weighted_median_t_find")
DIAGNOSTIC = ("uniform_detection_probability", "uniform_weighted_mean_t_find",
              "uniform_weighted_median_t_find")
METRICS = PRIMARY + DIAGNOSTIC
CONTRAST_KEY = ("contrast_type", "family_left", "family_right", "target_js_nats", "gamma")
LABEL = ("scene_id", "prior_id", "family", "direction", "target_js_nats", "gamma", "plan_id")
PRETTY = dict(zip(FAMILIES, NAMES)) | {"oracle": "Oracle", "uniform_prior_baseline": "Uniform-prior baseline"}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def difference(left, right):
    return None if left is None or right is None else float(left - right)


def describe(values):
    values = [float(v) for v in values if v is not None]
    return {"available_scenes": len(values),
            "mean": float(np.mean(values)) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "negative_scenes": sum(v < -1e-10 for v in values),
            "zero_scenes": sum(abs(v) <= 1e-10 for v in values),
            "positive_scenes": sum(v > 1e-10 for v in values)}


def pair_saved_episodes(error, oracle, scene_id):
    """Pair by scene/seed/episode, including coordinate verification."""
    indexed = {}
    for row in oracle:
        key = (scene_id, row["seed"], row["episode"])
        require(key not in indexed, f"Duplicate oracle target: {key}")
        indexed[key] = row
    result, seen = [], set()
    for row in error:
        key = (scene_id, row["seed"], row["episode"])
        require(key in indexed and key not in seen, f"Missing or duplicate target pair: {key}")
        seen.add(key)
        ref = indexed[key]
        require((row["target_x"], row["target_y"]) == (ref["target_x"], ref["target_y"]),
                f"Paired target coordinates differ: {key}")
        outcome = ("both_success" if ref["success"] and row["success"] else
                   "oracle_only" if ref["success"] else "error_only" if row["success"] else "both_timeout")
        result.append({"scene_id": scene_id, "seed": row["seed"], "episode": row["episode"],
                       "outcome": outcome, "delta_t_find_both_success":
                       row["t_find"] - ref["t_find"] if outcome == "both_success" else None})
    require(seen == set(indexed), f"Unequal target sets in {scene_id}")
    return result


def load_results(source, manifest):
    """Require one terminal record per manifest plan, including invalid attempts."""
    rows, paths = {}, {}
    actual_dirs = {p.parent.name for p in (source / "plans").glob("*/result.json")}
    expected_dirs = set()
    for scene in manifest["candidate_manifest"]["scenes"]:
        for gamma in manifest["candidate_manifest"]["gammas"]:
            for condition in scene["conditions"]:
                pid = f"{scene['scene_id']}__gamma_{gamma:g}__{condition['prior_id']}"
                path = source / "plans" / pid
                expected_dirs.add(pid)
                record = json.loads((path / "result.json").read_text())
                for key, value in {**condition, "scene_id": scene["scene_id"], "gamma": gamma,
                                   "plan_id": pid}.items():
                    require(record[key] == value, f"Manifest condition mismatch: {pid}/{key}")
                require(record["attempt_count"] == 1, f"Unexpected retry: {pid}")
                require(record["valid"] == (record["status"] == "valid"), f"Validity mismatch: {pid}")
                exact = json.loads((path / "exact_metrics.json").read_text())
                row = {key: record[key] for key in LABEL} | {
                    "actual_js_nats": record.get("actual_js_nats"),
                    "valid": record["valid"], "status": record["status"],
                    "attempt_count": record["attempt_count"]}
                row.update(exact)
                if not row["valid"]:
                    require(all(row.get(k) is None for k in METRICS),
                            f"Invalid plan has numerical search metrics: {pid}")
                rows[scene["scene_id"], gamma, condition["prior_id"]] = row
                paths[pid] = path
    require(actual_dirs == expected_dirs, "Completed plan set differs from manifest")
    return rows, paths


def primary_contrasts(manifest, rows):
    """Left minus right; reused plans yield block contrasts, never extra trials."""
    output = []
    candidate = manifest["candidate_manifest"]
    for scene in candidate["scenes"]:
        sid = scene["scene_id"]
        conditions = {(r["family"], r["direction"], r["target_js_nats"]): r["prior_id"]
                      for r in scene["conditions"]}
        for gamma in candidate["gammas"]:
            def emit(left, right, direction, level, kind):
                a, b = rows[sid, gamma, left], rows[sid, gamma, right]
                valid = a["valid"] and b["valid"]
                output.append({"scene_id": sid, "gamma": gamma, "direction": direction,
                               "target_js_nats": level, "contrast_type": kind,
                               "family_left": a["family"], "family_right": b["family"],
                               "prior_left": left, "prior_right": right,
                               "planned_pairs": 1, "available_valid_pairs": int(valid),
                               "left_valid": a["valid"], "right_valid": b["valid"],
                               **{f"delta_{metric}": difference(a.get(metric), b.get(metric))
                                  if valid else None for metric in METRICS}})
            emit("uniform_prior_baseline", "oracle", "shared", None, "baseline_minus_oracle")
            for level in candidate["js_levels"]:
                for direction in scene["directions"]:
                    ids = {family: conditions[family, None if family == "diffuse_blur" else direction, level]
                           for family in FAMILIES}
                    for family in FAMILIES:
                        emit(ids[family], "oracle", direction, level, "error_minus_oracle")
                    for a, b in combinations(FAMILIES, 2):
                        emit(ids[a], ids[b], direction, level, "family_pair")
    return output


def aggregate_contrasts(blocks, manifest):
    """Require all declared directions before averaging a scene's contrast.

    Timing has an additional detection-availability denominator: a valid pair
    can have zero detected mass, which leaves conditional timing undefined.
    """
    scene_groups, summary_groups = defaultdict(list), defaultdict(list)
    for row in blocks:
        scene_groups[tuple(row[k] for k in CONTRAST_KEY) + (row["scene_id"],)].append(row)
    scene_rows = []
    directions = {s["scene_id"]: len(s["directions"]) for s in manifest["candidate_manifest"]["scenes"]}
    for key, group in scene_groups.items():
        label = dict(zip(CONTRAST_KEY + ("scene_id",), key))
        expected = 1 if label["contrast_type"] == "baseline_minus_oracle" else directions[label["scene_id"]]
        require(len(group) == expected, f"Missing planned direction contrast: {key}")
        available = sum(r["available_valid_pairs"] for r in group)
        row = label | {"planned_direction_pairs": expected, "available_valid_direction_pairs": available,
                       "complete_valid_direction_set": available == expected}
        for metric in METRICS:
            values = [r[f"delta_{metric}"] for r in group]
            row[f"delta_{metric}"] = (float(np.mean(values)) if available == expected
                                       and all(v is not None for v in values) else None)
        scene_rows.append(row)
        summary_groups[key[:-1]].append(row)
    summary = []
    expected_scenes = manifest["candidate_manifest"]["scene_count"]
    for key, group in summary_groups.items():
        require(len(group) == expected_scenes, f"Missing planned scene contrast: {key}")
        retained = [r["scene_id"] for r in group if r["complete_valid_direction_set"]]
        row = dict(zip(CONTRAST_KEY, key)) | {
            "planned_scenes": expected_scenes, "available_valid_scenes": len(retained),
            "planned_direction_pairs": sum(r["planned_direction_pairs"] for r in group),
            "available_valid_direction_pairs": sum(r["available_valid_direction_pairs"] for r in group),
            "retained_scene_ids": ";".join(retained),
            "complete_finite_suite_contrast": len(retained) == expected_scenes,
            "interpretation": "finite_suite" if len(retained) == expected_scenes else "conditional_on_valid_pairs"}
        for metric in METRICS:
            values = [r[f"delta_{metric}"] for r in group]
            row.update({f"{metric}_{name}": value for name, value in describe(values).items()})
            row[f"{metric}_scene_ids"] = ";".join(r["scene_id"] for r in group if r[f"delta_{metric}"] is not None)
        summary.append(row)
    return scene_rows, summary


def gamma_interactions(scene_rows, manifest):
    """Difference of paired scene contrasts at the two declared gammas."""
    gammas = manifest["candidate_manifest"]["gammas"]
    require(len(gammas) == 2, "The frozen interaction assumes its two gamma values")
    high, low = max(gammas), min(gammas)
    groups = defaultdict(dict)
    keys = tuple(k for k in CONTRAST_KEY if k != "gamma") + ("scene_id",)
    for row in scene_rows:
        groups[tuple(row[k] for k in keys)][row["gamma"]] = row
    output = []
    for key, group in groups.items():
        require(set(group) == set(gammas), "Unpaired gamma condition")
        output.append(dict(zip(keys, key)) | {"gamma_left": low, "gamma_right": high,
            "planned_gamma_pairs": 1,
            "available_valid_gamma_pairs": int(all(r["complete_valid_direction_set"] for r in group.values())),
            **{f"interaction_{metric}": difference(group[low][f"delta_{metric}"], group[high][f"delta_{metric}"])
               for metric in PRIMARY}})
    summary_groups = defaultdict(list)
    for row in output:
        summary_groups[tuple(row[k] for k in keys[:-1])].append(row)
    summary = []
    for key, group in summary_groups.items():
        row = dict(zip(keys[:-1], key)) | {"gamma_left": low, "gamma_right": high,
            "planned_scenes": len(group), "available_valid_scenes": sum(r["available_valid_gamma_pairs"] for r in group)}
        for metric in PRIMARY:
            row.update({f"{metric}_{name}": value for name, value in describe([r[f"interaction_{metric}"] for r in group]).items()})
            row[f"{metric}_scene_ids"] = ";".join(r["scene_id"] for r in group if r[f"interaction_{metric}"] is not None)
        summary.append(row)
    return output, summary


def sampled_tables(manifest, rows, paths):
    """Pool raw detections across seeds within each plan; never average medians."""
    sampling = manifest["final_evaluation_sampling"]
    seeds = sampling["seeds"]
    samples, pairs = [], []
    for scene in manifest["candidate_manifest"]["scenes"]:
        sid = scene["scene_id"]
        for gamma in manifest["candidate_manifest"]["gammas"]:
            oracle = rows[sid, gamma, "oracle"]
            oracle_episodes = read_csv(paths[oracle["plan_id"]] / "episodes.csv") if oracle["valid"] else []
            for condition in scene["conditions"]:
                row = rows[sid, gamma, condition["prior_id"]]
                label = {k: row[k] for k in LABEL}
                episodes = read_csv(paths[row["plan_id"]] / "episodes.csv") if row["valid"] else []
                paired = pair_saved_episodes(episodes, oracle_episodes, sid) if row["valid"] and oracle["valid"] else []
                for scope in [*seeds, "pooled"]:
                    subset = episodes if scope == "pooled" else [r for r in episodes if r["seed"] == scope]
                    planned = sampling["total_draws_per_scene_per_condition"] if scope == "pooled" else sampling["targets_per_seed"]
                    values = stats(subset)
                    samples.append(label | {"seed_scope": str(scope), "plan_valid": row["valid"],
                        "planned_episodes": planned, "available_episodes": len(subset), **values,
                        "exact_detection_probability": row.get("exact_detection_probability"),
                        "exact_weighted_mean_t_find": row.get("exact_weighted_mean_t_find"),
                        "exact_weighted_median_t_find": row.get("exact_weighted_median_t_find"),
                        "sampled_minus_exact_detection_probability": difference(values["success_rate"], row.get("exact_detection_probability")),
                        "sampled_minus_exact_mean_t_find": difference(values["mean_t_find_success_only"], row.get("exact_weighted_mean_t_find")),
                        "sampled_minus_exact_median_t_find": difference(values["median_t_find_success_only"], row.get("exact_weighted_median_t_find"))})
                    if row["family"] == "oracle":
                        continue
                    subpairs = paired if scope == "pooled" else [r for r in paired if r["seed"] == scope]
                    for direction in ([row["direction"]] if row["direction"] is not None else
                                      ["shared"] if row["family"] == "uniform_prior_baseline" else scene["directions"]):
                        pairs.append(label | {"direction": direction, "seed_scope": str(scope),
                            "planned_pairs": planned, "available_pairs": len(subpairs),
                            "both_plans_valid": row["valid"] and oracle["valid"], **paired_stats(subpairs)})
    return samples, pairs


def plot_results(out, manifest, conditions, scenes, summary, samples, interaction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidate = manifest["candidate_manifest"]
    gammas, levels = candidate["gammas"], candidate["js_levels"]
    names = dict(zip(FAMILIES, NAMES))
    specifications = [(PRIMARY[0], "Detection probability difference (pp)", 100),
                      (PRIMARY[1], "Conditional mean time difference (s)", 1),
                      (PRIMARY[2], "Conditional median time difference (s)", 1)]
    for gamma in gammas:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, (metric, title, scale) in zip(axes, specifications):
            ax.axhline(0, color="0.4", lw=.8)
            for family, color in zip(FAMILIES, COLORS):
                entries = [next(r for r in summary if r["gamma"] == gamma and r["family_left"] == family
                                and r["contrast_type"] == "error_minus_oracle" and r["target_js_nats"] == level) for level in levels]
                # Missing any scene leaves the finite-suite figure blank for that point.
                values = [r[f"{metric}_mean"] * scale if r[f"{metric}_available_scenes"] == candidate["scene_count"] else np.nan for r in entries]
                ax.plot(levels, values, "o-", color=color, label=names[family])
                for level in levels:
                    local = [r[f"delta_{metric}"] * scale for r in scenes if r["gamma"] == gamma
                             and r["family_left"] == family and r["contrast_type"] == "error_minus_oracle"
                             and r["target_js_nats"] == level and r[f"delta_{metric}"] is not None]
                    ax.scatter([level] * len(local), local, color=color, alpha=.17, s=12)
            baseline = next(r for r in summary if r["gamma"] == gamma and r["contrast_type"] == "baseline_minus_oracle")
            if baseline[f"{metric}_available_scenes"] == candidate["scene_count"]:
                ax.axhline(baseline[f"{metric}_mean"] * scale, color="black", ls="--", lw=1, label="Uniform-prior baseline")
            ax.set(xlabel="Matched JS divergence (nats)", ylabel=title, xticks=levels)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Frozen v2: condition minus oracle, gamma={gamma:g}\nLines require all 12 scenes; dots are direction-averaged scene contrasts")
        fig.tight_layout()
        save_figure(fig, out, f"primary_contrasts_gamma_{gamma:g}")

        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        for ax, (metric, title, scale) in zip(axes, specifications):
            matrix = np.full((candidate["scene_count"], len(FAMILIES) * len(levels)), np.nan)
            columns = [(f, level) for f in FAMILIES for level in levels]
            for i, scene in enumerate(candidate["scenes"]):
                for j, (family, level) in enumerate(columns):
                    row = next(r for r in scenes if r["scene_id"] == scene["scene_id"] and r["gamma"] == gamma
                               and r["family_left"] == family and r["target_js_nats"] == level and r["contrast_type"] == "error_minus_oracle")
                    if row[f"delta_{metric}"] is not None:
                        matrix[i, j] = row[f"delta_{metric}"] * scale
            bound = max(float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0, 1e-8)
            shown = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
            ax.set_xticks(range(len(columns)), [f"{['Shift', 'Blur', 'Hotspot', 'Suppress'][FAMILIES.index(f)]}\n{level:g}" for f, level in columns], rotation=65)
            ax.set_yticks(range(candidate["scene_count"]), [s["scene_id"].replace("cx", "").replace("_cy", ",").replace("_layout", " / ") for s in candidate["scenes"]], fontsize=8)
            ax.set_title(title, fontsize=10)
            fig.colorbar(shown, ax=ax, shrink=.7)
        fig.suptitle(f"All declared scenes, gamma={gamma:g}; gray/blank cells have unavailable contrasts")
        fig.tight_layout()
        save_figure(fig, out, f"scene_matrix_gamma_{gamma:g}")

    fig, axes = plt.subplots(1, len(gammas), figsize=(10, 4))
    for ax, gamma in zip(np.atleast_1d(axes), gammas):
        selected = [r for r in samples if r["gamma"] == gamma and r["seed_scope"] == "pooled" and r["plan_valid"]]
        ax.scatter([r["exact_detection_probability"] * 100 for r in selected],
                   [r["success_rate"] * 100 for r in selected], s=12, alpha=.6)
        ax.plot([0, 100], [0, 100], "k--", lw=.8)
        ax.set(xlabel="Exact detection probability (%)", ylabel="Sampled success (%)",
               title=f"gamma={gamma:g}: {len(selected)}/{len(conditions)//len(gammas)} valid plans")
    fig.suptitle("Sampled sanity check: 5 seeds × 256 targets per plan")
    fig.tight_layout()
    save_figure(fig, out, "sampled_sanity_check")

    fig, axes = plt.subplots(1, len(gammas), figsize=(11, 4))
    for ax, gamma in zip(np.atleast_1d(axes), gammas):
        for family, color in zip(FAMILIES, COLORS):
            selected = [r for r in conditions if r["gamma"] == gamma and r["family"] == family and r["valid"]]
            ax.scatter([r["uniform_detection_probability"] * 100 for r in selected],
                       [r["exact_detection_probability"] * 100 for r in selected], s=15, alpha=.6, color=color, label=names[family])
        ax.set(xlabel="Uniform-weighted detection / area covered (%)", ylabel="True-prior-weighted detection (%)", title=f"gamma={gamma:g}")
    axes[0].legend(fontsize=7)
    fig.suptitle("Uniform reweighting diagnostic of saved trajectories")
    fig.tight_layout()
    save_figure(fig, out, "uniform_weighted_diagnostic")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (metric, title, scale) in zip(axes, specifications):
        ax.axhline(0, color="0.4", lw=.8)
        for family, color in zip(FAMILIES, COLORS):
            selected = [next(r for r in interaction if r["family_left"] == family and r["contrast_type"] == "error_minus_oracle" and r["target_js_nats"] == level) for level in levels]
            ax.plot(levels, [r[f"{metric}_mean"] * scale if r[f"{metric}_available_scenes"] == candidate["scene_count"] else np.nan for r in selected], "o-", color=color, label=names[family])
        ax.set(xlabel="Matched JS divergence (nats)", ylabel=title, xticks=levels)
    axes[0].legend(fontsize=7)
    fig.suptitle("Declared family × gamma interaction: (error − oracle) at gamma=0.05 minus gamma=0.10")
    fig.tight_layout()
    save_figure(fig, out, "family_gamma_interaction")


def format_number(value, scale=1, digits=3):
    return "NA" if value is None else f"{value * scale:.{digits}f}"


def write_report(out, manifest, audit, conditions, summary, samples, interactions):
    candidate = manifest["candidate_manifest"]
    valid = sum(r["valid"] for r in conditions)
    total = candidate["projected_plan_count"]
    lines = ["# Frozen Protocol v2 formal results", "",
        f"Execution records: **{len(conditions)}/{total}**. Valid plans: **{valid}/{total}**. "
        f"Invalid/interrupted/exception plans: **{total-valid}/{total}**. Artifact audit: **passed**.", "",
        ("The declared matched finite suite is complete." if valid == total else
         "**The execution ledger is complete, but the matched valid suite is incomplete.** Failed plans are planner failures, not search timeouts. Conditional comparisons below retain only valid pairs; no family ranking is made."), "",
        f"Design: {candidate['scene_count']} scenes × {len(candidate['gammas'])} gamma values × "
        f"{candidate['conditions_per_scene_per_gamma']} distinct plans. Oracle, blur, and uniform-prior baseline plans are shared across direction blocks, not independent replications.", "",
        "Exact full-grid detection probability and true-prior-weighted conditional mean/median detection time are primary. "
        "These values integrate the frozen supported target grid and contain no target-sampling noise. "
        "They are not continuous-space ground truth. Conditional detection times describe detected mass; "
        "faster times alone do not establish better search when detection probability changes.", "",
        "Each contrast is left minus right. The two direction contrasts are averaged inside a scene; "
        "the 12 declared scenes then receive equal weight. A missing direction leaves that scene's average unavailable. "
        "Timing summaries additionally require detected mass in both plans. Every available denominator and retained scene set is saved in the CSV tables. "
        "A mean over fewer than 12 scenes is explicitly conditional; figures with finite-suite mean lines require all 12 scenes. "
        "No population confidence intervals, significance tests, or post-hoc ranking are reported for this designed finite suite.", "",
        "## Primary exact-grid contrasts", "",
        "Mean differences and scene ranges below follow the frozen direction-then-scene aggregation. "
        "The N column is valid scenes / 12; mean-time and median-time columns each state their own available scene count. "
        "Positive probability differences favor the left condition; positive timing differences mean slower conditional detection.", ""]
    for gamma in candidate["gammas"]:
        lines += [f"### gamma={gamma:g}", "",
                  "| Left − right | JS (nats) | N | Δ detection (pp) [scene range] | Δ mean T (s), N | Δ median T (s), N |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in (r for r in summary if r["gamma"] == gamma):
            p, mean, median = PRIMARY
            lines.append(f"| {PRETTY[row['family_left']]} − {PRETTY[row['family_right']]} | "
                f"{format_number(row['target_js_nats'], digits=2)} | {row['available_valid_scenes']}/{row['planned_scenes']} | "
                f"{format_number(row[p+'_mean'],100)} [{format_number(row[p+'_min'],100)}, {format_number(row[p+'_max'],100)}] | "
                f"{format_number(row[mean+'_mean'])}, {row[mean+'_available_scenes']}/12 | "
                f"{format_number(row[median+'_mean'])}, {row[median+'_available_scenes']}/12 |")
        lines += ["", f"![Primary contrasts](figures/primary_contrasts_gamma_{gamma:g}.png)", "",
                  f"![Per-scene variation](figures/scene_matrix_gamma_{gamma:g}.png)", ""]
    lines += ["## Declared family × gamma interaction", "",
        "Interaction = (left − right at gamma=0.05) − (left − right at gamma=0.10), "
        "paired within the same scene before equal-scene averaging. Both gamma values must be available. "
        "The complete interaction table includes all six family pairs and the uniform-prior baseline.", "",
        "| Error family | JS | Δ probability interaction (pp) | N | Δ mean-time interaction (s) | N |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in interactions:
        if row["contrast_type"] == "error_minus_oracle":
            p, mean = PRIMARY[:2]
            lines.append(f"| {PRETTY[row['family_left']]} | {row['target_js_nats']:g} | "
                         f"{format_number(row[p+'_mean'],100)} | {row[p+'_available_scenes']}/12 | "
                         f"{format_number(row[mean+'_mean'])} | {row[mean+'_available_scenes']}/12 |")
    lines += ["", "![Gamma interaction](figures/family_gamma_interaction.png)", "",
              "## Uniform-weighted diagnostic", "",
              "Every saved valid trajectory also has uniform-weighted detection probability and conditional timing. "
              "This reweights the same trajectory over free cells and is a diagnostic, not an independent ranking metric. "
              "The separately planned uniform-prior baseline appears in the primary contrasts above.", "",
              "![Uniform diagnostic](figures/uniform_weighted_diagnostic.png)", "",
              "## Sampled precision-sensitivity check", "",
              "Seeds 100–104, 256 targets per seed, 1,280 draws per scene/condition. "
              "Targets are paired by (scene_id, seed, episode) across priors, directions, levels and gamma. "
              "Successful-only summaries pool raw detections within each plan; they never average seed medians. "
              "The paired tables include oracle-success/error-success, oracle-success/error-timeout, "
              "oracle-timeout/error-success, and both-timeout counts; both-success ΔT mean, median, "
              "p75/p90/p95/max and negative/zero/positive proportions use the explicit common-success denominator.", ""]
    for gamma in candidate["gammas"]:
        selected = [r for r in samples if r["gamma"] == gamma and r["seed_scope"] == "pooled" and r["plan_valid"]]
        errors = [abs(r["sampled_minus_exact_detection_probability"]) for r in selected]
        lines.append(f"- gamma={gamma:g}: {len(selected)}/{total//len(candidate['gammas'])} valid plans; "
                     f"median / maximum absolute sampled–exact probability discrepancy: "
                     f"{format_number(float(np.median(errors)) if errors else None,100)} / "
                     f"{format_number(max(errors) if errors else None,100)} percentage points. "
                     "This is a sampling diagnostic, not a formal effect or ranking.")
    lines += ["", "![Sampled sanity check](figures/sampled_sanity_check.png)", "",
              "## Validity and provenance", "",
              f"Frozen protocol: `{manifest['protocol_id']}`; frozen M3 commit: `{manifest['frozen_milestone_3_commit']}`.", "",
              "All attempted conditions remain in `condition_metrics.csv`; invalid search metrics are null. "
              "No solver retry, per-condition substitution, condition deletion, or protocol revision is part of this analysis. "
              "See the formal suite's `formal_audit.json` for condition, frozen-source/prior hashes, pairing and validity verification.", ""]
    failed = [r for r in conditions if not r["valid"]]
    if failed:
        lines += ["| Failed plan | Status |", "|---|---|"]
        lines += [f"| `{r['plan_id']}` | {r['status']} |" for r in failed]
        lines += [""]
    lines += ["## Saved outputs", "",
              "- `condition_metrics.csv`: every distinct planned condition, exact primary and uniform diagnostic metrics, coverage, entropy, trajectory length, validity.",
              "- `primary_block_contrasts.csv`: all oracle/error, uniform/oracle and six family-pair contrasts at each declared block.",
              "- `primary_scene_contrasts.csv`, `primary_contrast_summary.csv`: direction-then-scene aggregation, ranges, signs, denominators and retained scene IDs.",
              "- `gamma_interactions_by_scene.csv`, `gamma_interaction_summary.csv`: the declared paired family × gamma contrasts.",
              "- `sampled_sanity_check.csv`: each seed and pooled sampled metrics beside exact values.",
              "- `sampled_oracle_pairs.csv`: per-block 2×2 counts and paired common-success ΔT tails.",
              "- `figures/`: PNG and PDF figures. Raw trajectories, solver diagnostics, grid first-detection distributions and sampled episodes remain in the formal suite's per-plan directories.", "",
              "No post-hoc tuning or protocol changes were performed. Further changes require a separately frozen revision.", ""]
    (out / "FORMAL_RESULTS_SUMMARY.md").write_text("\n".join(lines))


def analyze(source, out, frozen):
    manifest_bytes = frozen.read_bytes()
    manifest = json.loads(manifest_bytes)
    audit = json.loads((source / "formal_audit.json").read_text())
    require(audit.get("passed") is True, "A passing complete artifact audit is required before formal analysis")
    rows, paths = load_results(source, manifest)
    require(len(rows) == manifest["candidate_manifest"]["projected_plan_count"], "Incomplete formal ledger")
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    blocks = primary_contrasts(manifest, rows)
    scenes, summary = aggregate_contrasts(blocks, manifest)
    interaction_scenes, interactions = gamma_interactions(scenes, manifest)
    samples, pairs = sampled_tables(manifest, rows, paths)
    conditions = list(rows.values())
    tables = {"condition_metrics": conditions, "primary_block_contrasts": blocks,
              "primary_scene_contrasts": scenes, "primary_contrast_summary": summary,
              "gamma_interactions_by_scene": interaction_scenes, "gamma_interaction_summary": interactions,
              "sampled_sanity_check": samples, "sampled_oracle_pairs": pairs}
    for name, records in tables.items():
        write_csv(out / f"{name}.csv", records)
    plot_results(figures, manifest, conditions, scenes, summary, samples, interactions)
    write_report(out, manifest, audit, conditions, summary, samples, interactions)
    metadata = {"protocol_id": manifest["protocol_id"],
                "frozen_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "formal_audit_sha256": hashlib.sha256((source / "formal_audit.json").read_bytes()).hexdigest(),
                "planned_plans": manifest["candidate_manifest"]["projected_plan_count"],
                "recorded_plans": len(conditions), "valid_plans": sum(r["valid"] for r in conditions),
                "complete_valid_matched_suite": all(r["valid"] for r in conditions),
                "table_rows": {name: len(records) for name, records in tables.items()},
                "planner_calls": 0, "protocol_modified": False,
                "files_sha256": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(out.rglob("*")) if p.is_file() and p.name != "analysis_manifest.json"}}
    write_json(out / "analysis_manifest.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "milestone_4/results/formal")
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/formal/analysis")
    parser.add_argument("--frozen", type=Path, default=ROOT / "milestone_4/results/frozen/PROTOCOL_V2_FROZEN_MANIFEST.json")
    args = parser.parse_args()
    output = args.output.resolve()
    frozen_root = ROOT / "milestone_4/results/frozen"
    require(output != frozen_root and frozen_root not in output.parents, "Frozen artifacts cannot be overwritten")
    require(output != ROOT / "milestone_3" and ROOT / "milestone_3" not in output.parents, "Milestone 3 is frozen")
    print(json.dumps(analyze(args.source.resolve(), output, args.frozen.resolve()), indent=2))


if __name__ == "__main__":
    main()
