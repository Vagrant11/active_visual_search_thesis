"""Gate 4: finalize the sampled target-seed count now that exact-grid metrics
are the declared primary endpoint and sampled metrics are secondary/diagnostic
only (a precision-sensitivity sanity check, never a confirmatory ranking or CI
source -- PROTOCOL_V2_DRAFT.md, "Primary endpoint declaration").

No planner import, zero planning. Computes a worst-case binomial standard
error for candidate seed counts against the largest exact-vs-sampled
discrepancy actually observed in the pilot (Observation 5: 4.69 percentage
points at n=128 draws), and against the measured Gate-2 evaluation cost, to
make the final count an evidenced choice rather than an unjustified round
number. Run: python -m milestone_4.gate4_seed_decision
"""

import argparse
import json
from math import sqrt
from pathlib import Path

from .protocol import ROOT

PILOT_OBSERVED_DISCREPANCY_PP = 4.69  # Observation 5, results/analysis/REPORT.md
TARGETS_PER_SEED = 256
CANDIDATE_SEED_COUNTS = (2, 3, 5, 10)


def worst_case_se_pp(n):
    return 100 * sqrt(0.5 * 0.5 / n)  # binomial SE upper bound at p=0.5, in percentage points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/preflight")
    args = parser.parse_args()
    out = args.output.resolve()
    cost_path = out / "gate2_cost_check.json"
    cost = json.loads(cost_path.read_text()) if cost_path.exists() else None

    candidates = []
    for seeds in CANDIDATE_SEED_COUNTS:
        n = seeds * TARGETS_PER_SEED
        se = worst_case_se_pp(n)
        candidates.append({"seed_count": seeds, "targets_per_seed": TARGETS_PER_SEED, "total_draws": n,
                           "worst_case_se_percentage_points": se,
                           "resolves_pilot_discrepancy_with_margin": se * 3 < PILOT_OBSERVED_DISCREPANCY_PP})

    final_seed_count = 5
    final_n = final_seed_count * TARGETS_PER_SEED
    per_plan_cost_at_10_seeds = cost["per_plan_evaluation_seconds"] if cost else None
    decision = {
        "pilot_observed_discrepancy_percentage_points": PILOT_OBSERVED_DISCREPANCY_PP,
        "candidates": candidates,
        "final_decision": {
            "seed_count": final_seed_count, "seeds": list(range(100, 100 + final_seed_count)),
            "targets_per_seed": TARGETS_PER_SEED, "total_draws_per_scene_per_condition": final_n,
            "worst_case_se_percentage_points": worst_case_se_pp(final_n),
        },
        "rationale": [
            "Exact full-grid detection probability/weighted T_find are the primary endpoints and carry "
            "zero target-sampling noise; sampled success/timeout is declared secondary, used only as a "
            "precision-sensitivity sanity check against the exact value, never for a formal ranking or "
            "confidence interval on its own (PROTOCOL_V2_DRAFT.md, 'Primary endpoint declaration').",
            "Measured evaluation cost is negligible regardless of seed count: the full candidate 10-seed "
            "design costs ~0.26s/plan (see gate2_cost_check.json), ~2.4 minutes across the whole 552-plan "
            "suite. Cost therefore does not drive this choice in either direction.",
            "At least 3 independent seeds are needed to see a seed-to-seed spread at all; the pilot's own "
            "2 seeds (7, 11) could only show two conditions differ, not characterize a distribution.",
            f"5 seeds x {TARGETS_PER_SEED} targets = {final_n} draws gives a worst-case binomial SE of "
            f"{worst_case_se_pp(final_n):.2f} percentage points, resolving the pilot's largest observed "
            f"exact-vs-sampled discrepancy ({PILOT_OBSERVED_DISCREPANCY_PP} points, Observation 5) with "
            "more than 3x margin -- comfortably enough for a sanity check.",
            "5 seeds is deliberately smaller than the draft's earlier 10-seed candidate: since cost is "
            "immaterial, the remaining reason to pick a number is to keep the sampled design's scale "
            "visibly matched to its declared secondary/sanity-check role, rather than an n large enough "
            "to look like it could support a confirmatory ranking, which the protocol explicitly forbids "
            "computing from sampled metrics alone.",
        ],
    }
    (out / "gate4_seed_decision.json").write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    print(json.dumps(decision, indent=2))
    print(f"Written to {out / 'gate4_seed_decision.json'}")


if __name__ == "__main__":
    main()
