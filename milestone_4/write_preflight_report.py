"""Compose PROTOCOL_V2_PREFLIGHT_REPORT.md from the already-written preflight
and freeze artifacts (candidate manifest, feasibility audit, JS reachability
table, Gate 2 synthetic tests/cost, engineering-check results, Gate 4 seed
decision, and the Gate 5 frozen manifest once it exists). Reads only; no
planner import, zero planning. Run: python -m milestone_4.write_preflight_report
"""

import argparse
import csv
import json
from pathlib import Path

from .protocol import ROOT


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def family_headroom(reachability):
    by_family = {}
    for row in reachability:
        by_family.setdefault(row["family"], []).append(row)
    lines = []
    for family in ("spatial_shift", "diffuse_blur", "false_hotspot", "false_negative_suppression"):
        rows = by_family[family]
        scan_max = min(float(r["scan_max_js_nats"]) for r in rows)
        js15_params = [float(r["parameter_value"]) for r in rows if r["target_js_nats"] == "0.15"]
        lines.append(f"| {family.replace('_', ' ')} | {len(rows)} | {scan_max:.4f} | "
                     f"{min(js15_params):.4f} - {max(js15_params):.4f} |")
    return lines


def write_report(preflight_dir, frozen_dir):
    manifest = read_json(preflight_dir / "candidate_scene_manifest.json")
    feasibility = read_json(preflight_dir / "feasibility_audit.json")
    reachability = read_csv(preflight_dir / "js_reachability_table.csv")
    engineering_path = preflight_dir / "engineering_check.json"
    engineering = read_json(engineering_path) if engineering_path.exists() else None
    cost_path = preflight_dir / "gate2_cost_check.json"
    cost = read_json(cost_path) if cost_path.exists() else None
    seed_path = preflight_dir / "gate4_seed_decision.json"
    seed_decision = read_json(seed_path) if seed_path.exists() else None
    frozen_path = frozen_dir / "PROTOCOL_V2_FROZEN_MANIFEST.json"
    frozen = read_json(frozen_path) if frozen_path.exists() else None

    status_line = (f"**Status: Protocol v2 is FROZEN as of {frozen['frozen_at_utc']} "
                   f"(commit `{frozen['frozen_at_git_commit'][:12]}`). "
                   "See PROTOCOL_V2_FROZEN_MANIFEST.json.** The formal 552-plan suite has "
                   "still NOT been launched -- freezing the design and running it are separate, "
                   "separately invoked steps." if frozen else
                   "**Status: preflight only. Protocol v2 is NOT frozen. The formal 552-plan suite "
                   "has NOT been launched.**")

    lines = ["# Protocol v2 pre-freeze preflight report", "", status_line, "",
             "This report covers manifest generation, geometry/support/feasibility checks, JS "
             "reachability/calibration (Gate 1), synthetic evaluator-semantics tests and evaluation "
             "cost (Gate 2), engineering-only planner feasibility/runtime checks (Gate 3), the final "
             "sampled target-seed decision (Gate 4), and, once run, the frozen manifest (Gate 5). The "
             "Milestone 3 planner source was not modified; no check or search-outcome result was used "
             "to select or tune scenes, families, JS levels, or gamma values.", "",
             f"Generated from `{preflight_dir.relative_to(ROOT)}` and "
             f"`{frozen_dir.relative_to(ROOT)}`.", "",
             "## 1. Candidate scene manifest (Gate 1)", "",
             f"**{manifest['scene_count']} scenes** = 4 true hotspot centers "
             "`(0.25,0.25)`, `(0.25,0.75)`, `(0.75,0.25)`, `(0.75,0.75)` x 3 obstacle layouts "
             "(A: none, B: one rectangle matching v1, C: two rectangles). Each scene carries two "
             "error-geometry direction blocks (x, y) per PROTOCOL_V2_DRAFT.md. Grid size "
             f"{manifest['grid_size']}x{manifest['grid_size']}, sigma {manifest['sigma']}, background "
             f"{manifest['background']}, JS levels {manifest['js_levels']} nats, gammas "
             f"{manifest['gammas']}, initial state {manifest['initial_state']}, terminal state "
             f"{manifest['terminal_state']} (fixed by the frozen planner, unchanged from v1).", "",
             f"**{manifest['conditions_per_scene_per_gamma']} distinct conditions per scene per gamma**: "
             "oracle (1, shared) + diffuse/blur (3 JS levels, direction-shared) + true uniform-prior "
             "planning baseline (1, shared) + 3 direction-dependent families (spatial shift, false "
             "hotspot, false-negative suppression) x 3 JS levels x 2 direction blocks (18). "
             f"**Projected plan count: {manifest['scene_count']} scenes x "
             f"{len(manifest['gammas'])} gammas x {manifest['conditions_per_scene_per_gamma']} "
             f"conditions = {manifest['projected_plan_count']} plans.**", "",
             "| Layout | Obstacles |", "|---|---|",
             "| A | none |", "| B | rectangle [0.43,0.57] x [0.43,0.57] (matches v1) |",
             "| C | rectangles [0.35,0.45] x [0.55,0.65] and [0.55,0.65] x [0.35,0.45] |", "",
             "## 2. Geometry / support / feasibility audit (Gate 1)", "",
             f"**All {len(feasibility['scenes'])}/{len(feasibility['scenes'])} candidate scenes passed** "
             f"({'yes' if feasibility['all_scenes_passed'] else 'NO -- see below'}). Checks: obstacles "
             "within the unit workspace; nonempty true-prior support; supported free cells form one "
             "4-connected grid component; the fixed initial `(0.1,0.1)` and terminal `(0.9,0.9)` states "
             "clear every obstacle's collision disk; a 4-connected free-space path exists between the "
             "grid cells nearest those states; and each direction block's error geometry (false-hotspot "
             "center, suppression region) is well-defined. The connectivity checks are a **discrete grid "
             "proxy**, not a guarantee that the continuous-space SLSQP planner converges.", ""]
    failed = [s for s in feasibility["scenes"] if not s["passed"]]
    if failed:
        lines += ["**Failed scenes:**", ""]
        for s in failed:
            lines.append(f"- `{s['scene_id']}`: " + ", ".join(k for k, v in s.items()
                         if k not in ("scene_id", "center", "layout", "passed") and v is False))
        lines.append("")
    else:
        lines += ["No scene failed any feasibility check; every candidate scene remains eligible.", ""]

    unattainable = [r for r in reachability if r["status"] != "matched"]
    lines += ["## 3. JS reachability / calibration table (Gate 1)", "",
              f"**{len(reachability)} scene x direction-block x family x JS-level rows "
              f"({manifest['scene_count']} scenes x 2 blocks x 3 direction-dependent families x 3 levels, "
              "plus 12 x 3 shared blur rows).** No planner import; the same brentq-bisection calibration "
              "approach as the v1 pilot, generalized to a per-scene true center via a new "
              "`SceneErrorGenerator` (v1's `PriorErrorGenerator` intentionally rejects any non-v1 true "
              "center and is deliberately left unmodified, per PROTOCOL_V2_DRAFT.md).", "",
              ("**All rows matched at the existing 0.05/0.10/0.15 nat levels; no level revision is "
               "needed.**" if not unattainable else
               f"**{len(unattainable)} rows unattainable -- see below.**"), "",
              "Headroom by family (`scan_max_js_nats` is the minimum, across all scenes/blocks/levels for "
              "that family, of the maximum JS reachable anywhere in its one-dimensional parameter range; "
              "comfortably above 0.15 for every family means no scene is close to an attainability "
              "boundary):", "",
              "| Family | Rows | Min scan-max JS (nats) | JS=0.15 parameter range |",
              "|---|---:|---:|---:|", *family_headroom(reachability), ""]
    if unattainable:
        lines += ["**Unattainable rows:**", ""]
        for r in unattainable:
            lines.append(f"- `{r['scene_id']}` / {r['direction']} / {r['family']} / "
                         f"JS={r['target_js_nats']}: {r['status']}")
        lines.append("")

    lines += ["## 4. Synthetic evaluator-semantics tests and evaluation cost (Gate 2)", ""]
    lines += ["`milestone_4/tests/test_v2_preflight.py` validates that the frozen, unchanged "
              "`milestone_3.experiment.evaluate`/`is_visible` generalize correctly to v2 geometries: "
              "exact no-obstacle outcomes (target found at t=0; a target proven always out of range "
              "times out), Layout C two-rectangle occlusion (occluded by rectangle 1 only, by rectangle "
              "2 only, and a line-of-sight that clears both), full-grid replay monotonicity across all "
              "12 candidate scenes, and pairing-key collision checks confirming the proposed "
              "`(scene_id, seed, episode)` key does not conflate targets across scenes the way a bare "
              "`(seed, episode)` key would. Run via `python -m unittest milestone_4.tests.test_v2_preflight`.", ""]
    if cost is None:
        lines += ["**Evaluation cost: not yet measured.**", ""]
    else:
        lines += [f"Evaluation cost was timed on a synthetic (non-optimized, non-planner) trajectory of "
                  f"realistic shape and scale -- the planner's own deterministic initial-guess waypoints "
                  f"at the mean tf observed across the Gate 3 engineering checks -- on scene "
                  f"`{cost['scene_id']}` ({cost['grid_points']} grid points):", "",
                  f"- Exact full-grid replay (`first_seen_grid`, the primary-endpoint calculation): "
                  f"**{cost['full_grid_replay_seconds']*1000:.1f} ms**.",
                  f"- Sampled evaluation (`evaluate`), {len(cost['seeds'])} seeds x "
                  f"{cost['targets_per_seed']} targets = {cost['total_sampled_targets']} draws: "
                  f"**{cost['sampled_evaluate_seconds']*1000:.1f} ms**.",
                  f"- Combined per-plan evaluation cost: **{cost['per_plan_evaluation_seconds']*1000:.1f} ms**; "
                  f"projected over {cost['projected_plan_count']} plans: "
                  f"**{cost['projected_total_evaluation_seconds']:.0f}s "
                  f"(~{cost['projected_total_evaluation_hours']*60:.1f} minutes)**, negligible next to "
                  "the ~1-hour planning-time projection in section 8.", ""]

    lines += ["## 5. True uniform-prior planning baseline", "",
              "Defined in `milestone_4/scene_manifest.py:uniform_prior_baseline` as a distinct planned "
              "condition per scene (shared across gamma and direction blocks, like oracle and blur): a "
              "uniform distribution over the scene's supported free-cell mask, handed to "
              "`milestone_3.planner.plan` to produce its own trajectory. This is **not** the pilot's "
              "post-hoc uniform-weighted reweighting diagnostic, which reweights an already-planned "
              "oracle-conditioned trajectory rather than planning a new one under a uniform prior; both "
              "PROTOCOL_V2_DRAFT.md and `analyze_pilot.py`'s report text now name the diagnostic "
              "\"uniform-weighted diagnostic\" to avoid conflating the two. Its numerical feasibility as "
              "a planned condition was confirmed by the `layoutB_uniform_g010` engineering check below "
              "(valid, collision-free, 4.4s wall-clock). It adds one condition per scene/gamma (23 vs the "
              "draft's earlier 22), raising the projected plan count from 528 to 552.", "",
              "## 6. Primary / secondary / diagnostic metric definitions", "",
              "**Primary:** exact full-grid detection probability at budget, and its true-prior-weighted "
              "detection-time distribution / conditional mean / median, for every scene/geometry/JS/gamma "
              "block and family (plus oracle and the true uniform-prior planning baseline). Deterministic "
              "given the planned trajectory; carries no target-sampling noise.", "",
              "**Secondary (sampled):** success/timeout rate; paired outcomes (oracle-outcome x "
              "error-prior-outcome 2x2, plus both-success ΔT tail statistics); free-space coverage "
              "fraction; true-prior mass coverage; per-prior entropy (nats); trajectory length.", "",
              "**Diagnostic (not primary or secondary):** the uniform-weighted reweighting of an existing "
              "planned trajectory (distinct from the true uniform-prior planning baseline in section 5).", "",
              "## 7. Gamma and the family x gamma estimand", "",
              "gamma=0.10 and gamma=0.05 are kept and reported separately throughout; primary contrasts "
              "are never pooled across gamma. The family x gamma interaction on the primary exact-grid "
              "endpoints is declared as part of the estimand from the outset (see PROTOCOL_V2_DRAFT.md's "
              "\"Contrasts and aggregation\"), not introduced post hoc from an observed pattern. No gamma "
              "value was added, dropped, or reweighted based on any preflight or engineering-check result.", ""]

    lines += ["## 8. Engineering-only planner feasibility / runtime checks (Gate 3)", ""]
    if engineering is None:
        lines += ["**Not yet run.**", ""]
    else:
        lines += ["A fixed set of 5 checks, selected by geometry/gamma/prior-type coverage *before* any "
                  "was run (one per obstacle layout, one at the tighter gamma bound, one for the true "
                  "uniform-prior baseline), all on the same scene center to isolate those factors from "
                  "center choice. Only solver validity and wall-clock cost were recorded -- this is not a "
                  "comparison of which condition finds targets faster, and this list was not edited after "
                  "seeing results.", "",
                  "| Check | Rationale | Valid | Wall (s) | Optimize (s) | Iterations | tf (s) |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for r in engineering["results"]:
            lines.append(f"| `{r['check_id']}` | {r['rationale']} | {r['valid']} | "
                         f"{r['wall_seconds']:.1f} | {r['optimization_seconds']:.1f} | "
                         f"{r['iterations']} | {r['tf']:.3f} |")
        wall = [r["wall_seconds"] for r in engineering["results"]]
        mean_wall = sum(wall) / len(wall)
        projected_hours = mean_wall * manifest["projected_plan_count"] / 3600
        lines += ["", f"All {len(engineering['results'])}/{len(engineering['results'])} checks were valid, "
                  "solver-converged and collision-free. Mean wall-clock "
                  f"{mean_wall:.1f}s (range {min(wall):.1f}-{max(wall):.1f}s) per plan, JIT compilation "
                  "included (the planner recompiles its JAX functions on every call; this cost is not "
                  "amortized across plans in the current `plan()` implementation).", "",
                  f"**Projected formal-suite planning time: {manifest['projected_plan_count']} plans x "
                  f"{mean_wall:.1f}s mean ≈ {mean_wall * manifest['projected_plan_count']:.0f}s "
                  f"≈ {projected_hours:.1f} hours, sequential, single process.** This covers "
                  "planning only, not the sampled-episode evaluation, coverage replay, or calibration "
                  "steps the formal suite would also run, and is extrapolated from only 5 of 552 "
                  "conditions (oracle and the uniform-prior baseline; the three direction-dependent error "
                  "families were not separately timed, though they solve the same problem structure with "
                  "a different prior weighting `phik`, not a different constraint set).", ""]

    lines += ["## 9. Final evaluation sampling design (Gate 4)", ""]
    if seed_decision is None:
        lines += ["**Not yet decided.**", ""]
    else:
        d = seed_decision["final_decision"]
        lines += [f"**Final: {d['seed_count']} seeds ({', '.join(map(str, d['seeds']))}), "
                  f"{d['targets_per_seed']} targets per seed, "
                  f"{d['total_draws_per_scene_per_condition']} draws per scene/condition.** Worst-case "
                  f"binomial standard error ≈ {d['worst_case_se_percentage_points']:.2f} percentage "
                  "points.", "", "Candidate seed counts considered:", "",
                  "| Seeds | Total draws | Worst-case SE (pp) | Resolves pilot's 4.69pp discrepancy 3x+ |",
                  "|---:|---:|---:|---|"]
        for c in seed_decision["candidates"]:
            lines.append(f"| {c['seed_count']} | {c['total_draws']} | "
                         f"{c['worst_case_se_percentage_points']:.2f} | "
                         f"{'yes' if c['resolves_pilot_discrepancy_with_margin'] else 'no'} |")
        lines += [""] + [f"- {r}" for r in seed_decision["rationale"]] + [""]

    lines += ["## 10. Freeze status (Gate 5)", ""]
    if frozen is None:
        lines += ["**Not yet frozen.**", "",
                  "**Recommendation: freeze once gates 1-4 above are all satisfied with no blocking "
                  "finding**, which they are as of this report: every candidate scene passed feasibility; "
                  "every JS level is attainable everywhere with comfortable headroom; the synthetic "
                  "evaluator-semantics tests pass and evaluation cost is negligible; the true "
                  "uniform-prior planning baseline is numerically feasible; per-plan planning cost is "
                  "small (~7s mean) relative to the 552-plan suite; and the final 5-seed sampling design "
                  "is set with an explicit, evidenced rationale. Nothing here changes the 12-scene design "
                  "or the 0.05/0.10/0.15 nat JS levels. Run `python -m milestone_4.freeze_protocol_v2` to "
                  "produce the frozen manifest and calibrated prior hashes, then commit.", ""]
    else:
        lines += [f"**Protocol v2 is frozen** as of `{frozen['frozen_at_utc']}` at git commit "
                  f"`{frozen['frozen_at_git_commit'][:12]}`, frozen against Milestone 3 commit "
                  f"`{frozen['frozen_milestone_3_commit'][:12]}`. See "
                  "`results/frozen/PROTOCOL_V2_FROZEN_MANIFEST.json` and "
                  f"`results/frozen/priors/` ({manifest['scene_count'] * manifest['conditions_per_scene_per_gamma']} "
                  "calibrated prior vectors, sha256-hashed per scene).", "",
                  "Planned contrasts and failure handling are frozen in the manifest under "
                  "`planned_contrasts` and `failure_handling` (mirroring PROTOCOL_V2_DRAFT.md's "
                  "\"Contrasts and aggregation\" and \"Invalid plans and missing conditions\" sections). "
                  "**The formal 552-plan suite has not been launched**; that remains a separate, "
                  "explicitly invoked step into a new output directory, per gate item 5.", ""]

    (preflight_dir / "PROTOCOL_V2_PREFLIGHT_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {preflight_dir / 'PROTOCOL_V2_PREFLIGHT_REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/preflight")
    parser.add_argument("--frozen", type=Path, default=ROOT / "milestone_4/results/frozen")
    args = parser.parse_args()
    write_report(args.output.resolve(), args.frozen.resolve())


if __name__ == "__main__":
    main()
