"""Fixed protocol shared by calibration and the explicitly invoked pilot."""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import numpy as np

from milestone_3.experiment import Environment, make_priors

ROOT = Path(__file__).resolve().parents[1]
FROZEN_M3_COMMIT = "6e0bce5"
LEVELS = (0.05, 0.10, 0.15)
EXAMPLE_LEVELS = (0.05, 0.15, 0.30)
JS_TOLERANCE = 1e-6
GAMMAS = (0.10, 0.05)
SEEDS = (7, 11)
EPISODES = 64
NODES = 50
MAXITER = 600
SOURCES = ("milestone_3/experiment.py", "milestone_3/planner.py",
           "milestone_3/diagnostics.py", "milestone_2/visibility.py")


def fixed_inputs():
    environment = Environment()
    points, priors = make_priors(environment, grid_size=40)
    return environment, points, priors["oracle"]


def paired_targets(points, true_prior, seed, episodes=EPISODES):
    indices = np.random.default_rng(seed).choice(len(points), size=episodes, p=true_prior)
    return indices, points[indices]


def protocol_manifest():
    # Fail rather than silently run a modified planner or evaluator.
    for source in SOURCES:
        frozen = subprocess.check_output(["git", "show", f"{FROZEN_M3_COMMIT}:{source}"], cwd=ROOT)
        if (ROOT / source).read_bytes() != frozen:
            raise RuntimeError(f"Fixed protocol source changed: {source}")
    environment, points, true_prior = fixed_inputs()
    upstream_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT / "external/time_optimal_ergodic_search", text=True).strip()
    frozen_config = json.loads(subprocess.check_output(
        ["git", "show", f"{FROZEN_M3_COMMIT}:milestone_3/results/config.json"], cwd=ROOT, text=True))
    if upstream_commit != frozen_config["upstream_commit"]:
        raise RuntimeError("Reference planner submodule commit changed from frozen Milestone 3")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "*.py"],
                      cwd=ROOT / "external/time_optimal_ergodic_search", check=False).returncode:
        raise RuntimeError("Reference planner submodule has modified Python sources")
    return {
        "protocol_id": "m4-controlled-prior-v1",
        "frozen_milestone_3_commit": subprocess.check_output(
            ["git", "rev-parse", FROZEN_M3_COMMIT], cwd=ROOT, text=True).strip(),
        "only_treatment": "predicted prior passed to milestone_3.planner.plan",
        "environment": asdict(environment),
        "ground_truth": {"grid_size": 40, "center": [0.75, 0.70], "sigma": 0.13,
                         "background_density": 0.03, "mask": "outside collision disks"},
        "target_sampling": {"method": "numpy.default_rng(seed).choice with oracle probabilities",
                            "seeds": list(SEEDS), "episodes_per_seed": EPISODES,
                            "pair_key": ["seed", "episode"], "shared_across_all_conditions": True},
        "planner": {"callable": "milestone_3.planner.plan", "gammas": list(GAMMAS),
                    "nodes": NODES, "maxiter": MAXITER, "warm_start": False,
                    "initial_state": [0.1, 0.1, 0, 0], "terminal_state": [0.9, 0.9, 0, 0]},
        "evaluation": {"callable": "milestone_3.experiment.evaluate",
                       "coverage_callable": "milestone_3.diagnostics.coverage_series",
                       "invalid_plans": "excluded; reported separately; pilot exits nonzero",
                       "metrics": ["success_rate", "timeout_rate", "mean_t_find_success_only",
                                   "median_t_find_success_only", "visible_free_fraction",
                                   "oracle_probability_mass_covered", "paired_outcomes"]},
        "js": {"units": "nats", "levels": list(LEVELS), "absolute_tolerance": JS_TOLERANCE,
               "reference": "unchanged oracle on masked 40x40 cell-center grid"},
        "source_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in SOURCES},
        "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "true_prior_sha256": hashlib.sha256(true_prior.tobytes()).hexdigest(),
        "versions": {name: version(name) for name in ("numpy", "scipy", "jax", "matplotlib")},
        "upstream_commit": upstream_commit,
    }
