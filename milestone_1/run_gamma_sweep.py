"""
Milestone 1: reproduce the gamma-vs-optimized-time sweep for
time-optimal ergodic search.

This script intentionally stays on the control side:
    target distribution phi -> time-optimal ergodic planner -> trajectory

It uses the repository's existing public implementation in
experiments/comparison_study/build_solver.py and records both the solver's
reported stopping status and the measured ergodic constraint value.
"""

from __future__ import annotations

import contextlib
import csv
import io
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "external" / "time_optimal_ergodic_search"
COMPARISON_DIR = BASELINE_ROOT / "experiments" / "comparison_study"
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(COMPARISON_DIR))

from build_solver import build_erg_time_opt_solver  # noqa: E402
from time_opt_erg_lib.ergodic_metric import ErgodicMetric  # noqa: E402
from time_opt_erg_lib.fourier_utils import BasisFunc, get_ck, get_phik  # noqa: E402
from time_opt_erg_lib.target_distribution import TargetDistribution  # noqa: E402


N = 200
X0 = onp.array([0.1, 0.1, 0.0, 0.0])
XF = onp.array([0.9, 0.9, 0.0, 0.0])
TF_INIT = 10.0
GAMMAS = [0.1, 0.08, 0.05, 0.02, 0.01, 0.005]
PLOT_GAMMAS = [0.1, 0.05, 0.01, 0.005]
MAX_ITER = 10000
EPS = 1e-8

OUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUT_DIR / "results"
FIGURES_DIR = OUT_DIR / "figures"


def make_init_sol(n: int = N, x0: onp.ndarray = X0, xf: onp.ndarray = XF, tf: float = TF_INIT):
    """Straight-line state initial guess, zero control, scalar tf decision variable."""
    x = onp.linspace(x0, xf, n, endpoint=True)
    u = onp.zeros((n, xf.shape[0] // 2))
    return {"x": x, "u": u, "tf": np.array(tf)}


_basis = BasisFunc(n_basis=[8, 8])
_target = TargetDistribution()
_erg_metric = ErgodicMetric(_basis)
_phik = get_phik(_target.evals, _basis)


def ergodicity(sol: dict, n: int = N) -> float:
    """Measured ergodic metric E(x, phi, tf) for a solution."""
    tf = float(sol["tf"])
    ck = get_ck(onp.asarray(sol["x"])[:, :2], _basis, tf, tf / n)
    return float(_erg_metric(ck, _phik))


def control_cost(sol: dict, n: int = N) -> float:
    """Time-normalized integral control cost."""
    u = onp.asarray(sol["u"])
    tf = float(sol["tf"])
    return float(onp.sum(u**2) * (tf / n) / tf)


def run_solve(solver, args: dict) -> tuple[bool, str, float]:
    """Run the existing solver and capture its printed status without changing it."""
    stdout = io.StringIO()
    start = perf_counter()
    with contextlib.redirect_stdout(stdout):
        solver.solve(args=args, max_iter=MAX_ITER, eps=EPS)
    elapsed = perf_counter() - start
    message = " ".join(stdout.getvalue().strip().split())
    return "done in" in message, message, elapsed


def write_results(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gamma",
        "optimized_tf",
        "measured_ergodicity",
        "solver_success",
        "constraint_satisfied",
        "solver_message",
        "max_abs_control",
        "time_normalized_control_cost",
        "runtime_seconds",
    ]
    with (RESULTS_DIR / "gamma_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_gamma_vs_tf(rows: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["gamma"])
    gamma = [row["gamma"] for row in ordered]
    tf = [row["optimized_tf"] for row in ordered]
    erg = [row["measured_ergodicity"] for row in ordered]

    fig, ax1 = plt.subplots(figsize=(4.2, 2.8))
    ax1.plot(gamma, tf, marker="o", color="#1f77b4", linewidth=1.7)
    ax1.set_xlabel(r"Ergodicity upper bound $\gamma$")
    ax1.set_ylabel(r"Optimized terminal time $t_f^*$ (s)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linewidth=0.4, alpha=0.45)

    ax2 = ax1.twinx()
    ax2.plot(gamma, erg, marker="s", color="#d62728", linewidth=1.2, linestyle="--")
    ax2.set_ylabel(r"Measured ergodicity $\mathcal{E}$", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gamma_vs_tf.pdf")
    fig.savefig(FIGURES_DIR / "gamma_vs_tf.png", dpi=200)
    plt.close(fig)


def plot_trajectories(solutions: dict[float, dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    grid_x, grid_y = onp.meshgrid(onp.linspace(0, 1, 80), onp.linspace(0, 1, 80))
    phi = onp.ones_like(grid_x)

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.6), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, gamma in zip(axes, PLOT_GAMMAS):
        sol = solutions[gamma]
        traj = onp.asarray(sol["x"])
        ax.imshow(
            phi,
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap="Greys",
            vmin=0,
            vmax=1.35,
            alpha=0.28,
        )
        ax.plot(traj[:, 0], traj[:, 1], color="#006d77", linewidth=1.4)
        ax.scatter(traj[0, 0], traj[0, 1], color="#2ca02c", s=22, label="start")
        ax.scatter(traj[-1, 0], traj[-1, 1], color="#d62728", s=22, label="goal")
        ax.set_title(
            rf"$\gamma={gamma:g}$, $t_f^*={float(sol['tf']):.2f}$ s",
            fontsize=10,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linewidth=0.3, alpha=0.35)

    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.supxlabel("x")
    fig.supylabel("y")
    fig.suptitle(r"Optimized trajectories over the reference uniform $\phi$", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "trajectories.pdf")
    fig.savefig(FIGURES_DIR / "trajectories.png", dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    args = {"N": N, "x0": X0, "xf": XF, "erg_ub": GAMMAS[0]}
    solver = build_erg_time_opt_solver(make_init_sol(), args, step_size=1e-3, c=1.0)

    rows = []
    solutions = {}
    for gamma in GAMMAS:
        args.update({"erg_ub": gamma})
        solver_success, solver_message, runtime_seconds = run_solve(solver, args)
        sol = deepcopy(solver.get_solution())
        measured_erg = ergodicity(sol)
        optimized_tf = float(sol["tf"])
        max_abs_control = float(onp.max(onp.abs(onp.asarray(sol["u"]))))
        constraint_satisfied = measured_erg <= gamma + 1e-5
        solutions[gamma] = sol

        row = {
            "gamma": gamma,
            "optimized_tf": optimized_tf,
            "measured_ergodicity": measured_erg,
            "solver_success": solver_success,
            "constraint_satisfied": constraint_satisfied,
            "solver_message": solver_message,
            "max_abs_control": max_abs_control,
            "time_normalized_control_cost": control_cost(sol),
            "runtime_seconds": runtime_seconds,
        }
        rows.append(row)
        print(
            "gamma={gamma:<6g} tf={tf:>7.3f} E={erg:.5f} "
            "solver_success={success} constraint_satisfied={constraint}".format(
                gamma=gamma,
                tf=optimized_tf,
                erg=measured_erg,
                success=solver_success,
                constraint=constraint_satisfied,
            )
        )

    write_results(rows)
    with (RESULTS_DIR / "gamma_sweep_solutions.pkl").open("wb") as f:
        pickle.dump(solutions, f)
    plot_gamma_vs_tf(rows)
    plot_trajectories(solutions)
    print(f"\nwrote {RESULTS_DIR / 'gamma_sweep.csv'}")
    print(f"wrote {FIGURES_DIR / 'gamma_vs_tf.pdf'}")
    print(f"wrote {FIGURES_DIR / 'trajectories.pdf'}")


if __name__ == "__main__":
    main()
