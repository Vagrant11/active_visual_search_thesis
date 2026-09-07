"""Analyze saved M4 pilot artifacts; never import or run the planner."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from milestone_3.experiment import camera_trajectory, js_divergence
from visibility import EPS, is_visible
from .calibrate_priors import prior_id, write_csv
from .prior_errors import FAMILIES, validate_prior
from .protocol import ROOT, fixed_inputs, protocol_manifest

LABEL = ("prior_id", "family", "target_js_nats", "gamma")
OUTCOMES = ("both_success", "oracle_only", "error_only", "both_timeout")
# Explicit 2x2 (oracle outcome x error-prior outcome) reading of OUTCOMES, so the
# discordant/concordant structure is visible in column names, not just prose.
OUTCOME_2X2 = {"both_success": ("success", "success"), "oracle_only": ("success", "timeout"),
               "error_only": ("timeout", "success"), "both_timeout": ("timeout", "timeout")}
COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
NAMES = ("Spatial shift", "Diffuse / blur", "False hotspot", "False-negative suppression")
CALIBRATION_PRIORS_PATH = ROOT / "milestone_4/results/calibration/priors.npz"


def entropy_nats(p):
    """Standard discrete entropy -sum(p*log(p)) over the support, in nats."""
    p = np.asarray(p, dtype=float)
    support = p > 0
    return float(-np.sum(p[support] * np.log(p[support])))


def trajectory_length(states):
    """Physical path length: sum of consecutive Euclidean distances between planned (x, y) nodes."""
    xy = np.asarray(states)[:, :2]
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def weighted_detection_stats(first, weights):
    """Exact full-grid detection probability and weighted mean/median T_find, given detection.

    `weights` need not sum to 1 over the detected subset alone; the reported rate is the
    weighted mass detected (e.g. true-prior mass covered, or uniform-mass fraction of the
    support ever visible), and the mean/median are conditional on detection so that cells
    never seen do not silently enter the timing statistic. The rate is always returned
    alongside so the denominator behind the conditional mean/median is explicit.
    """
    first = np.asarray(first, dtype=float)
    weights = np.asarray(weights, dtype=float)
    seen = np.isfinite(first)
    rate = float(weights[seen].sum()) if seen.any() else 0.0
    if not seen.any() or rate <= 0:
        return {"detection_probability": rate, "weighted_mean_t_find": None, "weighted_median_t_find": None}
    w, t = weights[seen], first[seen]
    order = np.argsort(t)
    t_sorted, cum = t[order], np.cumsum(w[order])
    median = float(t_sorted[np.searchsorted(cum, 0.5 * cum[-1])])
    return {"detection_probability": rate, "weighted_mean_t_find": float(np.sum(w * t) / w.sum()),
            "weighted_median_t_find": median}


def load_prior_entropy(inputs, config):
    """Per-prior entropy from the calibration artifact, cross-checked against pilot inputs.

    This does not run the planner or generator; it only loads and validates the already-saved
    `results/calibration/priors.npz` against hashes/arrays the pilot itself was already
    validated against in `load_and_validate`.
    """
    with np.load(CALIBRATION_PRIORS_PATH, allow_pickle=False) as data:
        cal = {k: data[k].copy() for k in data.files}
    require(np.array_equal(cal["points"], inputs["points"]), "Calibration points differ from pilot inputs")
    require(np.array_equal(cal["true_prior"], inputs["true_prior"]), "Calibration true prior differs from pilot inputs")
    require(np.array_equal(cal["free_mask"], inputs["true_prior"] > 0), "Calibration free mask differs from true-prior support")
    entropy = {"oracle": entropy_nats(cal["true_prior"])}
    for key in sorted(k for k in cal if k not in ("points", "true_prior", "free_mask")):
        require(hashlib.sha256(cal[key].tobytes()).hexdigest() == config["prior_sha256"][key],
                f"Calibration prior hash changed: {key}")
        entropy[key] = entropy_nats(cal[key])
    require(set(entropy) == set(config["prior_sha256"]) | {"oracle"}, "Calibration priors do not cover all pilot conditions")
    return entropy


def read_csv(path):
    def scalar(value):
        if value == "":
            return None
        if value in ("True", "False"):
            return value == "True"
        try:
            return float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            return value
    with path.open(newline="") as handle:
        return [{k: scalar(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_unique(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(key not in result, f"Duplicate record: {key}")
        result[key] = row
    return result


def same_number(a, b):
    return a is b if a is None or b is None else bool(np.isclose(a, b, atol=1e-10, rtol=1e-10))


def stats(records):
    """Pool raw episodes, never average seed-specific conditional means/medians."""
    times = [r["t_find"] for r in records if r["success"]]
    n = len(records)
    return {"episodes": n, "success_count": len(times), "timeout_count": n - len(times),
            "success_rate": len(times) / n if n else None,
            "timeout_rate": (n - len(times)) / n if n else None,
            "mean_t_find_success_only": float(np.mean(times)) if times else None,
            "median_t_find_success_only": float(np.median(times)) if times else None}


def pair_episodes(episodes):
    indexed = index_unique(episodes, ("prior_id", "gamma", "seed", "episode"))
    result = []
    for r in episodes:
        if r["prior_id"] == "oracle":
            continue
        key = ("oracle", r["gamma"], r["seed"], r["episode"])
        require(key in indexed, f"Missing paired oracle: {key}")
        oracle = indexed[key]
        require((r["target_x"], r["target_y"]) == (oracle["target_x"], oracle["target_y"]),
                "Paired target coordinates differ")
        outcome = ("both_success" if oracle["success"] and r["success"] else
                   "oracle_only" if oracle["success"] else "error_only" if r["success"] else "both_timeout")
        result.append({k: r[k] for k in (*LABEL, "seed", "episode", "target_x", "target_y")}
                      | {"oracle_success": oracle["success"], "error_success": r["success"],
                         "oracle_t_find": oracle["t_find"], "error_t_find": r["t_find"], "outcome": outcome,
                         "delta_t_find_both_success": r["t_find"] - oracle["t_find"] if outcome == "both_success" else None})
    return result


def paired_stats(records):
    counts = Counter(r["outcome"] for r in records)
    delta = [r["delta_t_find_both_success"] for r in records if r["outcome"] == "both_success"]
    n = len(records)
    both_success = counts["both_success"]
    faster, equal, slower = (sum(d < -1e-10 for d in delta), sum(abs(d) <= 1e-10 for d in delta),
                             sum(d > 1e-10 for d in delta))
    def rate(count):
        return count / both_success if both_success else None
    return {"pairs": n, **{f"{k}_count": counts[k] for k in OUTCOMES},
            # Same four counts, named as an explicit oracle-outcome x error-outcome 2x2 table.
            **{f"oracle_{o}_error_{e}_count": counts[k] for k, (o, e) in OUTCOME_2X2.items()},
            "delta_success_rate": (counts["error_only"] - counts["oracle_only"]) / n if n else None,
            "both_success_mean_delta_t_find": float(np.mean(delta)) if delta else None,
            "both_success_median_delta_t_find": float(np.median(delta)) if delta else None,
            "both_success_p75_delta_t_find": float(np.percentile(delta, 75)) if delta else None,
            "both_success_p90_delta_t_find": float(np.percentile(delta, 90)) if delta else None,
            "both_success_p95_delta_t_find": float(np.percentile(delta, 95)) if delta else None,
            "both_success_max_delta_t_find": float(np.max(delta)) if delta else None,
            "error_faster_count_both_success": faster,
            "equal_time_count_both_success": equal,
            "error_slower_count_both_success": slower,
            "error_faster_rate_both_success": rate(faster),
            "equal_time_rate_both_success": rate(equal),
            "error_slower_rate_both_success": rate(slower)}


def first_seen_grid(policy, environment, points, support):
    """Replay visibility on saved nodes, preserving zero-time and budget detections.

    Broad prefilters use the visibility predicate's EPS. The predicate itself
    decides visibility; no optimizer or dynamics rollout is invoked.
    """
    first = np.full(len(points), np.inf)
    for t, x, y, theta in camera_trajectory(policy["states"], policy["tf"], environment):
        delta = points - (x, y)
        near = np.sum(delta**2, axis=1) <= (environment.camera.sensing_radius + EPS)**2
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        angle = (bearing - theta + np.pi) % (2*np.pi) - np.pi
        candidates = np.flatnonzero(support & ~np.isfinite(first) & near
                                    & (np.abs(angle) <= environment.camera.half_fov_radians + EPS))
        for i in candidates:
            if is_visible((x, y, theta), tuple(points[i]), list(environment.obstacles), environment.camera):
                first[i] = t
    return first


def coverage_comparison(first, oracle_first, true_prior, terminal_arrival):
    seen, oracle_seen = np.isfinite(first), np.isfinite(oracle_first)
    gained, lost = seen & ~oracle_seen, oracle_seen & ~seen
    union = seen | oracle_seen
    before = seen & (first < terminal_arrival)
    after = seen & ~before
    return {"gained_free_cells": int(gained.sum()), "lost_free_cells": int(lost.sum()),
            "gained_true_prior_mass": float(true_prior[gained].sum()),
            "lost_true_prior_mass": float(true_prior[lost].sum()),
            "delta_true_prior_mass": float(true_prior[gained].sum() - true_prior[lost].sum()),
            "visible_union_jaccard": float((seen & oracle_seen).sum() / union.sum()) if union.any() else None,
            "true_mass_first_seen_before_terminal_arrival": float(true_prior[before].sum()),
            "true_mass_first_seen_at_or_after_terminal_arrival": float(true_prior[after].sum())}


def load_and_validate(source):
    config = json.loads((source / "config.json").read_text())
    expected = json.loads(json.dumps(protocol_manifest()))
    for key, value in expected.items():
        require(config.get(key) == value, f"Fixed protocol mismatch: {key}")
    require(json.loads((source / "pilot_audit.json").read_text())["passed"], "Saved pilot audit failed")
    tables = {name: read_csv(source / f"{name}.csv") for name in
              ("episodes", "summary", "paired_outcomes", "coverage_summary", "coverage_timeseries", "planner_diagnostics")}
    env, points, truth = fixed_inputs()
    with np.load(source / "inputs.npz", allow_pickle=False) as data:
        inputs = {k: data[k].copy() for k in data.files}
    require(np.array_equal(points, inputs["points"]) and np.array_equal(truth, inputs["true_prior"]),
            "Saved ground truth or points changed")
    seeds = config["target_sampling"]["seeds"]
    count = config["target_sampling"]["episodes_per_seed"]
    gammas, levels = config["planner"]["gammas"], config["js"]["levels"]
    ids = ["oracle"] + [prior_id(f, level) for f in FAMILIES for level in levels]
    labels = {"oracle": ("oracle", 0.)} | {prior_id(f, l): (f, l) for f in FAMILIES for l in levels}
    for key in ids[1:]:
        p = validate_prior(inputs[key], truth > 0)
        require(hashlib.sha256(p.tobytes()).hexdigest() == config["prior_sha256"][key], f"Prior hash changed: {key}")
        require(abs(js_divergence(truth, p) - labels[key][1]) <= config["js"]["absolute_tolerance"], "JS mismatch")
    for seed in seeds:
        indices = np.random.default_rng(seed).choice(len(points), size=count, p=truth)
        require(np.array_equal(inputs[f"target_indices_seed_{seed}"], indices)
                and np.array_equal(inputs[f"targets_seed_{seed}"], points[indices]), "Target sampling mismatch")
    for rows in tables.values():
        for r in rows:
            require(r["prior_id"] in labels and r["gamma"] in gammas, "Unexpected condition")
            require((r["family"], r["target_js_nats"]) == labels[r["prior_id"]], "Condition label mismatch")
    diagnostics = index_unique(tables["planner_diagnostics"], ("prior_id", "gamma"))
    require(set(diagnostics) == {(p, g) for p in ids for g in gammas}, "Missing plan diagnostics")
    for d in diagnostics.values():
        require(all(d[k] is True for k in ("valid", "feasible", "finite", "solver_success", "collision_free"))
                and 0 <= d["max_equality_residual"] <= 1e-5
                and 0 <= d["max_inequality_violation"] <= 1e-6,
                "Analysis requires a complete valid pilot; invalid plans cannot be silently omitted")
    episodes = index_unique(tables["episodes"], ("prior_id", "gamma", "seed", "episode"))
    require(set(episodes) == {(p, g, s, i) for p in ids for g in gammas for s in seeds for i in range(count)},
            "Missing or unexpected episode records")
    for r in episodes.values():
        target = inputs[f"targets_seed_{r['seed']}"][r["episode"]]
        require(np.array_equal(target, [r["target_x"], r["target_y"]]), "Episode target mismatch")
        require(isinstance(r["success"], bool), "Success must be a boolean")
        t = r["t_find"]
        require((t is not None and np.isfinite(t) and 0 <= t <= env.budget) if r["success"] else t is None,
                "Detection/timeout encoding mismatch")
        require(same_number(r["capped_time"], t if r["success"] else env.budget), "Capped time mismatch")
    pairs = pair_episodes(tables["episodes"])
    saved_pairs = index_unique(tables["paired_outcomes"], ("prior_id", "gamma", "seed", "episode"))
    require(len(saved_pairs) == len(pairs), "Missing paired outcomes")
    for r in pairs:
        old = saved_pairs.get(tuple(r[k] for k in ("prior_id", "gamma", "seed", "episode")))
        require(old is not None and all(r[k] == v for k, v in old.items()), "Saved pairing differs from episodes")
    cover = index_unique(tables["coverage_summary"], ("prior_id", "gamma"))
    require(set(cover) == set(diagnostics), "Missing coverage conditions")
    series = defaultdict(list)
    for row in tables["coverage_timeseries"]:
        series[row["prior_id"], row["gamma"]].append(row)
    require(set(series) == set(cover), "Coverage time-series conditions differ")
    policies, firsts = {}, {}
    for key, gamma in diagnostics:
        with np.load(source / f"trajectory_{key}_gamma_{gamma:g}.npz", allow_pickle=False) as data:
            policy = {k: data[k].copy() for k in data.files}
        policy["tf"] = float(policy["tf"])
        require(policy["states"].shape == (config["planner"]["nodes"], 4)
                and np.isfinite(policy["states"]).all() and np.isfinite(policy["controls"]).all(), "Invalid saved trajectory")
        require(np.allclose(policy["states"][0], config["planner"]["initial_state"], atol=1e-5)
                and np.allclose(policy["states"][-1], config["planner"]["terminal_state"], atol=1e-5), "Endpoint mismatch")
        policies[key, gamma] = policy
        first = first_seen_grid(policy, env, points, truth > 0)
        firsts[key, gamma] = first
        times = camera_trajectory(policy["states"], policy["tf"], env)[:, 0]
        saved = sorted(series[key, gamma], key=lambda r: r["time"])
        require(len(saved) == len(times), "Coverage time grid differs")
        for t, row in zip(times, saved):
            mask = first <= t
            require(same_number(t, row["time"]) and int(mask.sum()) == row["visible_free_cells"]
                    and row["free_cell_count"] == int((truth > 0).sum())
                    and same_number(mask.sum() / (truth > 0).sum(), row["visible_free_fraction"])
                    and same_number(truth[mask].sum(), row["oracle_probability_mass_covered"]),
                    f"Replayed visibility differs from saved coverage: {key}, {gamma}, {t}")
        require(saved[-1] == cover[key, gamma], "Coverage summary differs from last time sample")
        for seed in seeds:
            for episode, i in enumerate(inputs[f"target_indices_seed_{seed}"]):
                r = episodes[key, gamma, seed, episode]
                replayed = float(first[i]) if np.isfinite(first[i]) else None
                require(same_number(replayed, r["t_find"]), "Replayed detection differs from saved episode")
    summaries = index_unique(tables["summary"], ("prior_id", "gamma", "seed"))
    require(set(summaries) == {(p, g, s) for p in ids for g in gammas for s in seeds}, "Missing summary rows")
    for (key, gamma, seed), saved in summaries.items():
        group = [episodes[key, gamma, seed, i] for i in range(count)]
        require(saved["status"] == "evaluated", "Invalid summary status")
        for field, value in stats(group).items():
            require(same_number(saved[field], value), f"Saved summary disagrees with episodes: {field}")
        require(same_number(saved["optimized_tf"], policies[key, gamma]["tf"]), "Saved terminal time differs")
        for field in ("visible_free_fraction", "oracle_probability_mass_covered"):
            require(same_number(saved[field], cover[key, gamma][field]), "Saved summary coverage differs")
    return config, tables, pairs, env, inputs, policies, firsts


def aggregate(tables, pairs, config, extra_by_key_gamma=None, entropy_by_prior=None):
    """extra_by_key_gamma/entropy_by_prior are per-condition (not per-seed) diagnostics —
    exact full-grid metrics, trajectory length and prior entropy — merged identically into
    every seed_scope row, the same way visible_free_fraction/oracle_probability_mass_covered
    already are below. Both default to empty so unit tests that build minimal tables/config
    without these optional artifacts still run unchanged."""
    extra_by_key_gamma, entropy_by_prior = extra_by_key_gamma or {}, entropy_by_prior or {}
    metrics, paired = [], []
    coverage = index_unique(tables["coverage_summary"], ("prior_id", "gamma"))
    groups, pair_groups = defaultdict(list), defaultdict(list)
    for r in tables["episodes"]:
        groups[r["prior_id"], r["gamma"]].append(r)
    for r in pairs:
        pair_groups[r["prior_id"], r["gamma"]].append(r)
    for (key, gamma), records in groups.items():
        for scope in [*config["target_sampling"]["seeds"], "pooled"]:
            subset = records if scope == "pooled" else [r for r in records if r["seed"] == scope]
            baseline = groups["oracle", gamma]
            baseline = baseline if scope == "pooled" else [r for r in baseline if r["seed"] == scope]
            values, oracle = stats(subset), stats(baseline)
            label = {k: records[0][k] for k in LABEL} | {"seed_scope": str(scope)}
            metrics.append({**label, **values, "delta_success_rate": values["success_rate"] - oracle["success_rate"],
                            "delta_mean_t_find_success_only": (values["mean_t_find_success_only"] - oracle["mean_t_find_success_only"]
                                if values["success_count"] and oracle["success_count"] else None),
                            **{k: coverage[key, gamma][k] for k in ("visible_free_fraction", "oracle_probability_mass_covered")},
                            **extra_by_key_gamma.get((key, gamma), {}),
                            **({"entropy_nats": entropy_by_prior[key]} if key in entropy_by_prior else {})})
            if key != "oracle":
                raw = pair_groups[key, gamma]
                raw = raw if scope == "pooled" else [r for r in raw if r["seed"] == scope]
                paired.append({**label, **paired_stats(raw)})
    return metrics, paired


def save_figure(fig, out, stem):
    import matplotlib.pyplot as plt
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_results(out, config, metrics, paired, mechanism, firsts, inputs, policies, env):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    levels = config["js"]["levels"]
    scopes = [str(s) for s in config["target_sampling"]["seeds"]] + ["pooled"]
    style = dict(zip(scopes, ("--", ":", "-")))
    fields = [("success_rate", "Success (%)", 100), ("timeout_rate", "Timeout (%)", 100),
              ("mean_t_find_success_only", "Successful-only mean T_find (s)", 1),
              ("median_t_find_success_only", "Successful-only median T_find (s)", 1),
              ("visible_free_fraction", "Free-cell coverage (%)", 100),
              ("oracle_probability_mass_covered", "True-prior mass covered (%)", 100)]
    for gamma in config["planner"]["gammas"]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, (field, title, scale) in zip(axes.flat, fields):
            show_scopes = ["pooled"] if field in ("visible_free_fraction", "oracle_probability_mass_covered") else scopes
            for family, color in zip(FAMILIES, COLORS):
                for scope in show_scopes:
                    rows = sorted([r for r in metrics if r["gamma"] == gamma and r["family"] == family and r["seed_scope"] == scope],
                                  key=lambda r: r["target_js_nats"])
                    ax.plot(levels, [scale*r[field] if r[field] is not None else np.nan for r in rows],
                            color=color, ls=style[scope], marker="o" if scope == "pooled" else None,
                            lw=2 if scope == "pooled" else 1, alpha=1 if scope == "pooled" else .6)
            for scope in show_scopes:
                baseline = next(r for r in metrics if r["gamma"] == gamma and r["family"] == "oracle" and r["seed_scope"] == scope)
                if baseline[field] is not None:
                    ax.axhline(scale*baseline[field], color="black", ls=style[scope], lw=1, alpha=.65)
            ax.set(title=title, xlabel="JS divergence (nats)", xticks=levels)
            ax.grid(alpha=.2)
        handles = [Line2D([], [], color=c, label=n) for c, n in zip(COLORS, NAMES)]
        handles += [Line2D([], [], color="black", label="Oracle baseline")]
        handles += [Line2D([], [], color="gray", ls=style[s], label=f"seed {s}" if s != "pooled" else "Pooled episodes") for s in scopes]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9)
        fig.suptitle(f"Pilot descriptive comparisons | gamma={gamma:g} | coverage is measured once per plan")
        fig.tight_layout(rect=(0, .10, 1, .95))
        save_figure(fig, out, f"metrics_gamma_{gamma:g}")

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        rows = [r for r in paired if r["gamma"] == gamma and r["seed_scope"] == "pooled"]
        rows.sort(key=lambda r: (FAMILIES.index(r["family"]), r["target_js_nats"]))
        bottom = np.zeros(len(rows))
        outcome_colors = ("#86bcb6", "#e07b54", "#6699cc", "#dddddd")
        for outcome, color in zip(OUTCOMES, outcome_colors):
            values = np.array([r[f"{outcome}_count"] / r["pairs"] * 100 for r in rows])
            axes[0].bar(np.arange(len(rows)), values, bottom=bottom, color=color, label=outcome.replace("_", " "))
            bottom += values
        short_names = dict(zip(FAMILIES, ("Shift", "Blur", "Hotspot", "Suppress")))
        axes[0].set(xticks=np.arange(len(rows)), xticklabels=[f"{short_names[r['family']]}\n{r['target_js_nats']:g}" for r in rows],
                    ylabel="Paired targets (%)", title="Pooled paired outcomes")
        axes[0].tick_params(axis="x", labelsize=7, labelrotation=50)
        axes[0].legend(fontsize=8, loc="lower left")
        for ax, field, title in zip(axes[1:], ("both_success_mean_delta_t_find", "both_success_median_delta_t_find"),
                                   ("Same-target mean time difference", "Same-target median time difference")):
            for family, color in zip(FAMILIES, COLORS):
                for scope in scopes:
                    group = sorted([r for r in paired if r["gamma"] == gamma and r["family"] == family and r["seed_scope"] == scope],
                                   key=lambda r: r["target_js_nats"])
                    ax.plot(levels, [r[field] if r[field] is not None else np.nan for r in group], color=color, ls=style[scope],
                            marker="o" if scope == "pooled" else None, alpha=1 if scope == "pooled" else .6)
            ax.axhline(0, color="black", lw=1)
            ax.set(title=title, xlabel="JS (nats)", ylabel="Error - oracle T_find (s); both success only", xticks=levels)
            ax.grid(alpha=.2)
        fig.suptitle(f"gamma={gamma:g} | positive time difference = error slower | common-success subsets differ by comparison")
        fig.legend(handles=[Line2D([], [], color=c, label=n) for c, n in zip(COLORS, NAMES)]
                   + [Line2D([], [], color="gray", ls=style[s], label=f"seed {s}" if s != "pooled" else "Pooled episodes") for s in scopes],
                   loc="lower center", ncol=4, fontsize=8)
        fig.tight_layout(rect=(0, .12, 1, .93))
        save_figure(fig, out, f"paired_gamma_{gamma:g}")

        fig, axes = plt.subplots(4, 3, figsize=(11, 13))
        cmap = ListedColormap(["#eeeeee", "#a7c6b7", "#e07b54", "#6699cc"])
        oracle_seen = np.isfinite(firsts["oracle", gamma])
        for i, family in enumerate(FAMILIES):
            for j, level in enumerate(levels):
                key = prior_id(family, level)
                seen = np.isfinite(firsts[key, gamma])
                category = np.zeros(len(seen), dtype=int)
                category[seen & oracle_seen] = 1
                category[oracle_seen & ~seen] = 2
                category[seen & ~oracle_seen] = 3
                ax = axes[i, j]
                ax.imshow(np.ma.masked_where(inputs["true_prior"] == 0, category).reshape(40, 40),
                          extent=(0, 1, 0, 1), origin="lower", cmap=cmap, vmin=0, vmax=3)
                for o in env.obstacles:
                    ax.add_patch(Rectangle((o.xmin, o.ymin), o.xmax-o.xmin, o.ymax-o.ymin, color="#333333"))
                for name, ls, color in (("oracle", "--", "#555555"), (key, "-", "#111111")):
                    xy = policies[name, gamma]["states"][:, :2]
                    ax.plot(xy[:, 0], xy[:, 1], ls=ls, color=color, lw=.8)
                ax.plot(.75, .70, "+", color="#8e165e", markersize=10)
                row = next(r for r in mechanism if r["prior_id"] == key and r["gamma"] == gamma)
                ax.set(title=f"{NAMES[i]}; JS={level:g}\ntrue mass: +{row['gained_true_prior_mass']*100:.1f} / -{row['lost_true_prior_mass']*100:.1f} pp",
                       xlim=(0, 1), ylim=(0, 1), aspect="equal")
                ax.title.set_fontsize(9)
        fig.legend(handles=[Patch(color=c, label=s) for c, s in zip(cmap.colors, ("Neither visible", "Both visible", "Oracle only (lost)", "Error only (gained)"))],
                   loc="lower center", ncol=4, fontsize=9)
        fig.suptitle(f"Visible-cell gains/losses at 15 s | gamma={gamma:g}\nSolid path: error; dashed: oracle; + true hotspot; cell colors are visibility, not density")
        fig.tight_layout(rect=(0, .04, 1, .94))
        save_figure(fig, out, f"coverage_maps_gamma_{gamma:g}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for family, color in zip(FAMILIES, COLORS):
            rows = [r for r in mechanism if r["gamma"] == gamma and r["family"] == family]
            for row in rows:
                x = row["delta_true_prior_mass"]*100
                axes[0].scatter(x, row["delta_success_rate_pooled"]*100, color=color)
                axes[0].annotate(f"{row['target_js_nats']:g}", (x, row["delta_success_rate_pooled"]*100), fontsize=8)
                axes[1].scatter(row["delta_visible_free_fraction"]*100, x, color=color)
        limits = axes[0].get_xlim()
        axes[0].plot(limits, limits, ":", color="gray", label="Equal changes")
        axes[0].set(xlabel="Change in true-prior mass covered (pp)", ylabel="Change in pooled success rate (pp)")
        axes[1].set(xlabel="Change in free-cell coverage (pp)", ylabel="Change in true-prior mass covered (pp)")
        for ax in axes:
            ax.axhline(0, color="black", lw=.7)
            ax.axvline(0, color="black", lw=.7)
            ax.grid(alpha=.2)
        fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=n) for c, n in zip(COLORS, NAMES)],
                   loc="lower center", ncol=4, fontsize=8)
        fig.suptitle(f"Coverage and sampled outcomes | gamma={gamma:g}\nTrue-prior mass covered equals grid-distribution detection probability; no causal fit")
        fig.tight_layout(rect=(0, .08, 1, .9))
        save_figure(fig, out, f"mechanism_gamma_{gamma:g}")


def write_report(out, config, metrics, paired, mechanism, prior_entropy_table):
    lines = ["# Milestone 4 pilot analysis", "", "This is a descriptive analysis of saved trajectories and paired targets.",
             "No planner was imported or run. Four error families share JS levels 0.05, 0.10 and 0.15 nats.",
             "The analysis verifies the saved calibration, episode summaries, pairings and every coverage time sample.", "",
             "## How to read these results", "",
             "- Each condition has 64 targets per seed (7 and 11); pooled results use all 128 raw episodes.",
             "  Pooled successful-only means/medians are recomputed from detections, not averaged across seed summaries.",
             "- Delta means error minus oracle. A positive success delta is a gain; a positive time delta is slower.",
             "- Successful-only summaries can compare different target subsets. Paired timing uses only targets found",
             "  by both policies, with its denominator reported; that common subset also changes across error comparisons.",
             "- Coverage is measured once per deterministic plan. The two seeds are evaluation samples, not independent",
             "  scenes or optimizer replicates. No confidence intervals, p-values or general family ranking are reported.",
             "- On this fixed target grid with the ideal detector and shared observation times, true-prior mass covered",
             "  equals the detection probability under the discrete true distribution. Agreement with sampled success",
             "  is therefore an expected sampling relationship, not independent proof of a causal mediation mechanism.",
             "- **Primary endpoints for this pilot and the proposed formal suite are the exact full-grid metrics**:",
             "  exact detection probability (true-prior mass with finite first-visible time; identical to true-prior",
             "  mass covered above) and its true-prior-weighted mean/median T_find, conditional on detection. Sampled",
             "  successful-only mean/median and the 128-draw success rate are retained as a secondary, precision-sensitivity",
             "  check against target-draw noise (see observation 6 and the protocol v2 draft's primary-endpoint section).",
             "- **Uniform-weighted diagnostic**: the same exact full-grid computation reweighted by a uniform distribution",
             "  over the supported free-cell mask instead of the true prior, applied to the same oracle-conditioned saved",
             "  trajectories -- not a new planner run under a uniform prior. Its detection rate is exactly",
             "  `visible_free_fraction` (the visited fraction of supported cells); its weighted mean/median T_find is new",
             "  and isolates *where the trajectory physically went* from *how much true-prior probability mass it",
             "  captured*. A true uniform-prior baseline would require planning a fresh trajectory under a uniform prior;",
             "  that is deferred to the protocol v2 preflight/formal design, not computed here.",
             "- **Paired outcomes are a 2x2 table** of (oracle succeeded/timed out) x (error-prior policy succeeded/timed",
             "  out): `oracle_success_error_success_count`, `oracle_success_error_timeout_count`,",
             "  `oracle_timeout_error_success_count`, `oracle_timeout_error_timeout_count` in the paired summary CSVs",
             "  (equal to the legacy `both_success/oracle_only/error_only/both_timeout` columns, kept for compatibility).",
             "- Paired timing now also reports p75/p90/p95/max of the both-success ΔT distribution and the ΔT>0/=0/<0",
             "  proportions (not just counts), alongside the existing mean/median.",
             "- `prior_entropy.csv` and the `trajectory_length`/`entropy_nats` columns in `mechanism_summary.csv` record",
             "  discrete Shannon entropy (nats) of each calibrated prior and the physical path length of each planned",
             "  trajectory (sum of consecutive node-to-node distances), for cross-reference against coverage and success",
             "  in the same row.", ""]
    def metric(key, gamma):
        return next(r for r in metrics if r["prior_id"] == key and r["gamma"] == gamma and r["seed_scope"] == "pooled")
    def fmt_signed(value):
        return "NA" if value is None else f"{value:+.3f}"
    def fmt(value):
        return "NA" if value is None else f"{value:.3f}"
    shift_lo, shift_hi = metric("spatial_shift_js_0.1", .1), metric("spatial_shift_js_0.1", .05)
    hotspot = next(r for r in mechanism if r["prior_id"] == "false_hotspot_js_0.15" and r["gamma"] == .05)
    small_shift = next(r for r in mechanism if r["prior_id"] == "spatial_shift_js_0.15" and r["gamma"] == .05)
    non_oracle_mechanism = [r for r in mechanism if r["prior_id"] != "oracle"]
    both_positive_count = sum(1 for r in non_oracle_mechanism
                              if r["delta_exact_weighted_mean_t_find"] > 0 and r["delta_uniform_weighted_mean_t_find"] > 0)
    exact_exceeds_uniform_count = sum(1 for r in non_oracle_mechanism
                                      if r["delta_exact_weighted_mean_t_find"] > r["delta_uniform_weighted_mean_t_find"])
    exception = min(non_oracle_mechanism, key=lambda r: r["delta_exact_weighted_mean_t_find"] - r["delta_uniform_weighted_mean_t_find"])
    timing = [r["both_success_mean_delta_t_find"] for r in paired if r["seed_scope"] == "pooled"
              and r["both_success_mean_delta_t_find"] is not None]
    same_seed_signs = all(np.sign(r["delta_success_rate"]) == np.sign(metric(r["prior_id"], r["gamma"])["delta_success_rate"])
                          for r in metrics if r["prior_id"] == "spatial_shift_js_0.1" and r["seed_scope"] != "pooled")
    lines += ["## Observations in this pilot", "",
              f"1. **Compare the gamma strata separately.** For spatial shift at JS=0.10, pooled success",
              f"   differs from oracle by {shift_lo['delta_success_rate']*100:+.2f} percentage points at gamma=0.10,",
              f"   and {shift_hi['delta_success_rate']*100:+.2f} points at gamma=0.05.",
              f"   Both target seeds agree with their respective pooled contrast signs: {same_seed_signs}.",
              f"2. **Separate visible area from true probability.** For false hotspot at JS=0.15, gamma=0.05,",
              f"   free-cell coverage changes by {hotspot['delta_visible_free_fraction']*100:+.2f} points, whereas true-prior",
              f"   mass covered changes by {hotspot['delta_true_prior_mass']*100:+.2f} points. Pooled success changes by",
              f"   {hotspot['delta_success_rate_pooled']*100:+.2f} points. The maps show both new regions and lost regions.",
              "3. **Inspect the JS response without imposing monotonicity.** For spatial",
              "   shift at gamma=0.05, success across JS=0.05/0.10/0.15 is "
              + "/".join(f"{metric(prior_id('spatial_shift', level), .05)['success_rate']*100:.2f}%" for level in config["js"]["levels"]) + ".",
              f"4. **Common-success timing also changes.** Across the {len(timing)} pooled error-versus-oracle comparisons,",
              f"   mean paired time differences range from {min(timing):+.3f} to {max(timing):+.3f} seconds.",
              "   Each comparison uses its own both-success subset; this is not a claim that every target is slower.",
              "5. **Distribution-level coverage and sampled success are distinct.** Spatial shift at JS=0.15, gamma=0.05",
              f"   changes true-prior mass covered by {small_shift['delta_true_prior_mass']*100:+.2f} points while pooled sampled",
              f"   success changes by {small_shift['delta_success_rate_pooled']*100:+.2f} points.",
              "   These are different quantities: one integrates the fixed distribution, the other uses 128 draws.",
              "   This motivates separating target-sampling uncertainty from scenario/geometry variation.",
              "6. **True-prior- and uniform-weighted timing agree in sign but not in magnitude.** Across all "
              f"{len(non_oracle_mechanism)} pooled error-versus-oracle mechanism rows (12 conditions x 2 gammas), the",
              f"   true-prior-weighted and uniform-weighted changes in mean T_find are both positive (slower) in "
              f"{both_positive_count} of {len(non_oracle_mechanism)} rows: this is not a sign-flip finding like observation 2's",
              "   coverage/mass contrast. The true-prior-weighted change exceeds the uniform-weighted change in "
              f"{exact_exceeds_uniform_count} of {len(non_oracle_mechanism)} rows -- e.g. for false hotspot at JS=0.15, gamma=0.05, the",
              f"   true-prior-weighted mean T_find changes by {fmt_signed(hotspot['delta_exact_weighted_mean_t_find'])} s versus "
              f"{fmt_signed(hotspot['delta_uniform_weighted_mean_t_find'])} s uniform-weighted (same trajectories, reweighted",
              f"   only by which cells were physically visited). The exception is {dict(zip(FAMILIES, NAMES))[exception['family']].lower()} at "
              f"JS={exception['target_js_nats']:g}, gamma={exception['gamma']:g}, where the uniform-weighted change "
              f"({fmt_signed(exception['delta_uniform_weighted_mean_t_find'])} s) exceeds the true-prior-weighted change",
              f"   ({fmt_signed(exception['delta_exact_weighted_mean_t_find'])} s). This is a descriptive pattern in this",
              "   pilot's rows, not a mechanism claim.", ""]
    for gamma in config["planner"]["gammas"]:
        oracle = next(r for r in metrics if r["gamma"] == gamma and r["prior_id"] == "oracle" and r["seed_scope"] == "pooled")
        lines += [f"## gamma = {gamma:g}", "",
                  f"Oracle: success {oracle['success_count']}/128 ({oracle['success_rate']*100:.2f}%), "
                  f"successful-only mean/median {oracle['mean_t_find_success_only']:.3f}/{oracle['median_t_find_success_only']:.3f} s, "
                  f"free-cell coverage {oracle['visible_free_fraction']*100:.2f}%, true-prior mass covered {oracle['oracle_probability_mass_covered']*100:.2f}%.", "",
                  "| Family | JS | Success % | Delta success pp | Successful-only mean / median s | Both-success n | Paired mean / median delta s | Coverage % | True mass % |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for family, name in zip(FAMILIES, NAMES):
            for level in config["js"]["levels"]:
                key = prior_id(family, level)
                r = next(r for r in metrics if r["prior_id"] == key and r["gamma"] == gamma and r["seed_scope"] == "pooled")
                p = next(r for r in paired if r["prior_id"] == key and r["gamma"] == gamma and r["seed_scope"] == "pooled")
                lines.append(f"| {name} | {level:g} | {r['success_rate']*100:.2f} | {r['delta_success_rate']*100:+.2f} | "
                             f"{fmt(r['mean_t_find_success_only'])} / {fmt(r['median_t_find_success_only'])} | {p['both_success_count']} | "
                             f"{fmt(p['both_success_mean_delta_t_find'])} / {fmt(p['both_success_median_delta_t_find'])} | "
                             f"{r['visible_free_fraction']*100:.2f} | {r['oracle_probability_mass_covered']*100:.2f} |")
        oracle_exact = next(m for m in mechanism if m["prior_id"] == "oracle" and m["gamma"] == gamma)
        lines += ["", "**Exact full-grid diagnostics (primary endpoint, columns 3-5; uniform-weighted diagnostic, columns 6-8).**",
                  "Detection probability/rate is the true-prior (resp. uniform) mass with a finite first-visible time;",
                  "weighted mean/median T_find are conditional on that detection. Trajectory length is the summed",
                  "node-to-node path distance of the planned trajectory.", "",
                  f"Oracle: exact detection {oracle_exact['exact_detection_probability']*100:.2f}%, exact weighted mean/median "
                  f"{fmt(oracle_exact['exact_weighted_mean_t_find'])}/{fmt(oracle_exact['exact_weighted_median_t_find'])} s; "
                  f"uniform detection {oracle_exact['uniform_detection_probability']*100:.2f}%, uniform weighted mean/median "
                  f"{fmt(oracle_exact['uniform_weighted_mean_t_find'])}/{fmt(oracle_exact['uniform_weighted_median_t_find'])} s; "
                  f"trajectory length {oracle_exact['trajectory_length']:.3f}.", "",
                  "| Family | JS | Exact detect % | Exact weighted mean / median s | Uniform detect % | Uniform weighted mean / median s | Trajectory length | Delta length |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for family, name in zip(FAMILIES, NAMES):
            for level in config["js"]["levels"]:
                key = prior_id(family, level)
                m = next(m for m in mechanism if m["prior_id"] == key and m["gamma"] == gamma)
                lines.append(f"| {name} | {level:g} | {m['exact_detection_probability']*100:.2f} | "
                             f"{fmt(m['exact_weighted_mean_t_find'])} / {fmt(m['exact_weighted_median_t_find'])} | "
                             f"{m['uniform_detection_probability']*100:.2f} | "
                             f"{fmt(m['uniform_weighted_mean_t_find'])} / {fmt(m['uniform_weighted_median_t_find'])} | "
                             f"{m['trajectory_length']:.3f} | {m['delta_trajectory_length']:+.3f} |")
        lines += ["", "**Paired discordant outcomes (2x2: oracle outcome x error-prior outcome) and both-success ΔT tails.**",
                  "ΔT = error T_find - oracle T_find on the common-success subset; positive is slower. Proportions use",
                  "the \"oracle ok, error ok\" (both-success) count from this same row as their denominator.", "",
                  "| Family | JS | Oracle ok, error ok | Oracle ok, error timeout | Oracle timeout, error ok | Both timeout | ΔT>0 % | ΔT=0 % | ΔT<0 % | p75 | p90 | p95 | max s |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for family, name in zip(FAMILIES, NAMES):
            for level in config["js"]["levels"]:
                key = prior_id(family, level)
                p = next(r for r in paired if r["prior_id"] == key and r["gamma"] == gamma and r["seed_scope"] == "pooled")
                def pct(value):
                    return "NA" if value is None else f"{value*100:.1f}"
                lines.append(f"| {name} | {level:g} | {p['oracle_success_error_success_count']} | "
                             f"{p['oracle_success_error_timeout_count']} | {p['oracle_timeout_error_success_count']} | "
                             f"{p['oracle_timeout_error_timeout_count']} | {pct(p['error_slower_rate_both_success'])} | "
                             f"{pct(p['equal_time_rate_both_success'])} | {pct(p['error_faster_rate_both_success'])} | "
                             f"{fmt(p['both_success_p75_delta_t_find'])} | {fmt(p['both_success_p90_delta_t_find'])} | "
                             f"{fmt(p['both_success_p95_delta_t_find'])} | {fmt(p['both_success_max_delta_t_find'])} |")
        lines += ["", f"![Metric comparisons](metrics_gamma_{gamma:g}.png)", "",
                  f"![Paired outcomes and timing](paired_gamma_{gamma:g}.png)", "",
                  f"![Visibility gains and losses](coverage_maps_gamma_{gamma:g}.png)", "",
                  f"![Coverage and sampled success](mechanism_gamma_{gamma:g}.png)", ""]
    lines += ["## Mechanism diagnostics", "",
              "`mechanism_summary.csv` separates newly visible cells from cells lost relative to oracle, weighting each",
              "by the fixed true prior. It also splits mass first detected before versus at/after terminal-pose arrival.",
              "Arrival is `(N-1)*tf/N`, not `tf`, under the frozen execution rule. The latter mass is incremental",
              "coverage during the terminal hold, not a counterfactual estimate of the benefit of holding.", "",
              "`visibility_replay.npz` retains first-visible times (infinity for unseen cells), allowing the spatial",
              "gain/loss maps and decomposition to be checked without planning. The replay matches all saved coverage",
              "time samples and episode detections. Area gain alone need not imply probability-mass gain.", "",
              "## Seed sensitivity", "",
              "`metrics_by_seed.csv` and `paired_summary_by_seed.csv` retain separate seed outcomes. The plots show",
              "seed 7 dashed, seed 11 dotted and pooled solid. With only two target seeds, their spread is descriptive",
              "and is not an uncertainty estimate across scenes or error geometries.", "",
              "## Prior entropy", "",
              "Discrete Shannon entropy (nats, natural log, matching the JS units used elsewhere) of each calibrated",
              "prior and the true/oracle prior, from `results/calibration/priors.npz`. Entropy is a property of the",
              "prior alone and does not vary with gamma; `prior_entropy.csv` and the `entropy_nats` column of",
              "`mechanism_summary.csv` carry the same values for cross-reference against coverage and success.", "",
              "| Prior | Family | JS | Entropy (nats) |", "|---|---|---:|---:|",
              f"| oracle | oracle | 0.0 | {next(r['entropy_nats'] for r in prior_entropy_table if r['prior_id'] == 'oracle'):.4f} |"]
    for family, name in zip(FAMILIES, NAMES):
        for level in config["js"]["levels"]:
            key = prior_id(family, level)
            e = next(r["entropy_nats"] for r in prior_entropy_table if r["prior_id"] == key)
            lines.append(f"| {key} | {name} | {level:g} | {e:.4f} |")
    lines += ["", "## Limits and next experiment", "",
              "These comparisons establish condition-specific behavior under the frozen planner, camera and budget.",
              "They do not establish universal error-family ordering, or coverage as the only pathway affecting timing.",
              "The exact full-grid detection probability and weighted T_find are the declared primary endpoints for the",
              "proposed formal suite (see the protocol v2 draft's primary-endpoint section); sampled successful-only",
              "statistics remain a secondary precision-sensitivity check, motivated directly by observation 6 above.",
              "The next deliverable is the [protocol v2 draft](../../PROTOCOL_V2_DRAFT.md): a prespecified scenario and",
              "error-geometry set, per-block JS calibration, and paired contrasts with scenes as the unit of generalization.",
              "Its feasibility and precision checks must precede freezing the formal experiment manifest.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def analyze(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    protected = (ROOT / "milestone_3", ROOT / "milestone_4/results/pilot", ROOT / "milestone_4/results/calibration", source)
    require(not any(out == p or p in out.parents or out in p.parents for p in protected),
            "Analysis output must be separate from input and frozen experiment directories")
    files = sorted(p for p in source.iterdir() if p.is_file())
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    config, tables, pairs, env, inputs, policies, firsts = load_and_validate(source)

    # Exact full-grid diagnostics (step 1/5): primary endpoints computed by reweighting the
    # already-replayed visibility (`firsts`), never by planning. True-prior weights give the
    # exact detection probability/weighted T_find; uniform weights over the supported free-cell
    # mask give a diagnostic baseline isolating physical area coverage from true-prior mass.
    support = inputs["true_prior"] > 0
    uniform_weights = support.astype(float) / support.sum()
    exact_by_key_gamma, traj_len_by_key_gamma = {}, {}
    cover = index_unique(tables["coverage_summary"], ("prior_id", "gamma"))
    for (key, gamma), first in firsts.items():
        true_stats = weighted_detection_stats(first, inputs["true_prior"])
        uniform_stats = weighted_detection_stats(first, uniform_weights)
        # Sanity checks tying the new exact metric to the coverage fields already validated
        # against the saved pilot in load_and_validate: the true-prior-weighted detection rate
        # must equal the already-checked oracle_probability_mass_covered, and the uniform-weighted
        # rate must equal the already-checked visible_free_fraction (uniform mass ever visited
        # equals the visited fraction of supported free cells by construction).
        require(same_number(true_stats["detection_probability"], cover[key, gamma]["oracle_probability_mass_covered"]),
                f"Exact true-prior detection probability disagrees with validated coverage: {key}, {gamma}")
        require(same_number(uniform_stats["detection_probability"], cover[key, gamma]["visible_free_fraction"]),
                f"Uniform-weighted detection probability disagrees with validated free-cell coverage: {key}, {gamma}")
        exact_by_key_gamma[key, gamma] = {
            "exact_detection_probability": true_stats["detection_probability"],
            "exact_weighted_mean_t_find": true_stats["weighted_mean_t_find"],
            "exact_weighted_median_t_find": true_stats["weighted_median_t_find"],
            "uniform_detection_probability": uniform_stats["detection_probability"],
            "uniform_weighted_mean_t_find": uniform_stats["weighted_mean_t_find"],
            "uniform_weighted_median_t_find": uniform_stats["weighted_median_t_find"]}
        traj_len_by_key_gamma[key, gamma] = trajectory_length(policies[key, gamma]["states"])

    entropy_by_prior = load_prior_entropy(inputs, config)
    extra_by_key_gamma = {key: {**exact_by_key_gamma[key], "trajectory_length": traj_len_by_key_gamma[key]}
                          for key in firsts}
    metrics, paired = aggregate(tables, pairs, config, extra_by_key_gamma=extra_by_key_gamma, entropy_by_prior=entropy_by_prior)
    mechanism = []
    for (key, gamma), policy in policies.items():
        row = cover[key, gamma]
        arrival = (len(policy["states"]) - 1) * policy["tf"] / len(policy["states"])
        pooled = next(r for r in metrics if r["prior_id"] == key and r["gamma"] == gamma and r["seed_scope"] == "pooled")
        exact, oracle_exact = exact_by_key_gamma[key, gamma], exact_by_key_gamma["oracle", gamma]
        def delta_exact(field):
            return (exact[field] - oracle_exact[field]) if exact[field] is not None and oracle_exact[field] is not None else None
        mechanism.append({k: row[k] for k in LABEL} | {"terminal_arrival_time": arrival, "optimized_tf": policy["tf"],
                          "visible_free_fraction": row["visible_free_fraction"],
                          "true_prior_mass_covered": row["oracle_probability_mass_covered"],
                          "delta_visible_free_fraction": row["visible_free_fraction"] - cover["oracle", gamma]["visible_free_fraction"],
                          "delta_success_rate_pooled": pooled["delta_success_rate"],
                          **coverage_comparison(firsts[key, gamma], firsts["oracle", gamma], inputs["true_prior"], arrival),
                          **exact, "trajectory_length": traj_len_by_key_gamma[key, gamma],
                          "delta_trajectory_length": traj_len_by_key_gamma[key, gamma] - traj_len_by_key_gamma["oracle", gamma],
                          "delta_exact_weighted_mean_t_find": delta_exact("exact_weighted_mean_t_find"),
                          "delta_uniform_weighted_mean_t_find": delta_exact("uniform_weighted_mean_t_find"),
                          "entropy_nats": entropy_by_prior[key]})
    prior_entropy_table = [{"prior_id": pid, **({"family": "oracle", "target_js_nats": 0.0} if pid == "oracle" else
                            {"family": pid.rsplit("_js_", 1)[0], "target_js_nats": float(pid.rsplit("_js_", 1)[1])}),
                            "entropy_nats": value} for pid, value in entropy_by_prior.items()]
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("metrics_by_seed", [r for r in metrics if r["seed_scope"] != "pooled"]),
                       ("metrics_pooled", [r for r in metrics if r["seed_scope"] == "pooled"]),
                       ("paired_summary_by_seed", [r for r in paired if r["seed_scope"] != "pooled"]),
                       ("paired_summary_pooled", [r for r in paired if r["seed_scope"] == "pooled"]),
                       ("paired_time_differences", pairs), ("mechanism_summary", mechanism),
                       ("prior_entropy", prior_entropy_table)):
        write_csv(out / f"{name}.csv", rows)
    np.savez(out / "visibility_replay.npz", points=inputs["points"], true_prior=inputs["true_prior"],
             **{f"{key}_gamma_{gamma:g}": times for (key, gamma), times in firsts.items()})
    plot_results(out, config, metrics, paired, mechanism, firsts, inputs, policies, env)
    write_report(out, config, metrics, paired, mechanism, prior_entropy_table)
    require(all(hashlib.sha256(p.read_bytes()).hexdigest() == before[p.name] for p in files), "Input files changed during analysis")
    require("milestone_3.planner" not in sys.modules, "Planner unexpectedly imported")
    audit = {"passed": True, "planner_imported": False, "planner_calls": 0,
             "input_files_unchanged": True, "input_sha256": before,
             "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "source_directory": str(source), "plans": len(policies), "episodes": len(tables["episodes"]),
             "paired_outcomes": len(pairs), "replayed_coverage_samples": len(tables["coverage_timeseries"]),
             "checks": ["frozen protocol and prior hashes", "complete valid-plan diagnostics", "paired target identities",
                        "summary statistics recomputed from raw episodes", "paired outcomes reconstructed from episodes",
                        "every saved coverage time sample reproduced", "every episode detection reproduced from saved trajectory",
                        "exact true-prior-weighted detection rate cross-checked against validated oracle_probability_mass_covered",
                        "exact uniform-weighted detection rate cross-checked against validated visible_free_fraction",
                        "calibration priors.npz hashes and true prior/points cross-checked against validated pilot inputs"],
             "interpretation": "descriptive single-scene analysis; no formal ranking or causal mediation claim"}
    (out / "analysis_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "milestone_4/results/pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/analysis")
    args = parser.parse_args()
    audit = analyze(args.results, args.output)
    print(f"Analysis passed: {audit['plans']} saved plans, {audit['episodes']} episodes; zero planner calls.")
    print(f"Report: {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
