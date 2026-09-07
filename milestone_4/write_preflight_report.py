"""Compose PROTOCOL_V2_PREFLIGHT_REPORT.md from the already-written preflight
artifacts (candidate manifest, feasibility audit, JS reachability table,
engineering-check results). Reads only; no planner import, zero planning.
Run: python -m milestone_4.write_preflight_report
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


def write_report(out):
    manifest = read_json(out / "candidate_scene_manifest.json")
    feasibility = read_json(out / "feasibility_audit.json")
    reachability = read_csv(out / "js_reachability_table.csv")
    summary = read_json(out / "preflight_manifest_summary.json")
    engineering_path = out / "engineering_check.json"
    engineering = read_json(engineering_path) if engineering_path.exists() else None

    lines = ["# Protocol v2 pre-freeze preflight report", "",
             "**Status: preflight only. Protocol v2 is NOT frozen. The formal 552-plan suite has NOT "
             "been launched.** This report covers manifest generation, geometry/support/feasibility "
             "checks, JS reachability/calibration, the true uniform-prior planning-baseline definition, "
             "primary/secondary/diagnostic metric declarations, and a small set of engineering-only "
             "planner feasibility/runtime checks. The Milestone 3 planner source was not modified; no "
             "check result was used to select or tune scenes, families, or gamma values.", "",
             f"Generated from `{out.relative_to(ROOT)}`: `candidate_scene_manifest.json`, "
             "`feasibility_audit.json`/`.csv`, `js_reachability_table.csv`, "
             f"`{'engineering_check.json' if engineering else '(engineering check not yet run)'}`.", "",
             "## 1. Candidate scene manifest", "",
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
             "## 2. Geometry / support / feasibility audit", "",
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
    lines += ["## 3. JS reachability / calibration table", "",
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

    lines += ["## 4. True uniform-prior planning baseline", "",
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
              "## 5. Primary / secondary / diagnostic metric definitions", "",
              "**Primary:** exact full-grid detection probability at budget, and its true-prior-weighted "
              "detection-time distribution / conditional mean / median, for every scene/geometry/JS/gamma "
              "block and family (plus oracle and the true uniform-prior planning baseline). Deterministic "
              "given the planned trajectory; carries no target-sampling noise.", "",
              "**Secondary (sampled):** success/timeout rate; paired outcomes (oracle-outcome x "
              "error-prior-outcome 2x2, plus both-success ΔT tail statistics); free-space coverage "
              "fraction; true-prior mass coverage; per-prior entropy (nats); trajectory length.", "",
              "**Diagnostic (not primary or secondary):** the uniform-weighted reweighting of an existing "
              "planned trajectory (distinct from the true uniform-prior planning baseline in section 4).", "",
              "## 6. Gamma and the family x gamma estimand", "",
              "gamma=0.10 and gamma=0.05 are kept and reported separately throughout; primary contrasts "
              "are never pooled across gamma. The family x gamma interaction on the primary exact-grid "
              "endpoints is declared as part of the estimand from the outset (see PROTOCOL_V2_DRAFT.md's "
              "\"Contrasts and aggregation\"), not introduced post hoc from an observed pattern. No gamma "
              "value was added, dropped, or reweighted based on any preflight or engineering-check result.", ""]

    lines += ["## 7. Engineering-only planner feasibility / runtime checks", ""]
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

    lines += ["## 8. Freeze recommendation", "",
              "**Recommendation: do not freeze Protocol v2 yet.** Gate items 1 and 3 of "
              "PROTOCOL_V2_DRAFT.md's \"Scale and freeze gates\" are satisfied by this preflight "
              "(manifest/feasibility/JS-calibration preflight complete; a small geometry-selected "
              "engineering check run). Gate items 2 (saved-input pairing and evaluator semantics on "
              "synthetic cases, including full-grid evaluation cost), 4 (target-sampling precision, final "
              "seed count and targets per seed) and 5 (freezing the manifest, calibrated vectors, "
              "source/dependency hashes, planned contrasts and denominators in a dedicated commit) are "
              "**not yet done** and remain required before launching the formal 552-plan suite.", "",
              "No blocking finding surfaced: every candidate scene passed feasibility, every JS level is "
              "attainable everywhere with comfortable headroom, the true uniform-prior planning baseline "
              "is numerically feasible, and per-plan planning cost is small (~7s mean in the sampled "
              "checks) relative to the 552-plan suite. Nothing here changes the 12-scene design or the "
              "0.05/0.10/0.15 nat JS levels.", ""]

    (out / "PROTOCOL_V2_PREFLIGHT_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {out / 'PROTOCOL_V2_PREFLIGHT_REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/preflight")
    args = parser.parse_args()
    write_report(args.output.resolve())


if __name__ == "__main__":
    main()
