"""Run a simple Milestone 2 visibility demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Wedge

from scenes.simple_scene import build_scene
from visibility import first_detection_time, fov_boundary_points, is_visible


OUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUT_DIR / "figures"


def plot_scene(scene, first_detection_sample) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    xmin, ymin, xmax, ymax = scene.workspace
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.35)

    for obstacle in scene.obstacles:
        ax.add_patch(
            Rectangle(
                (obstacle.xmin, obstacle.ymin),
                obstacle.xmax - obstacle.xmin,
                obstacle.ymax - obstacle.ymin,
                facecolor="#303030",
                edgecolor="black",
                linewidth=0.8,
            )
        )

    xs = [sample[1] for sample in scene.trajectory]
    ys = [sample[2] for sample in scene.trajectory]
    visible_flags = [
        is_visible((sample[1], sample[2], sample[3]), scene.target, scene.obstacles, scene.camera)
        for sample in scene.trajectory
    ]

    ax.plot(xs, ys, color="#006d77", linewidth=1.4)
    ax.scatter(xs, ys, c=["#2ca02c" if flag else "#7f8c8d" for flag in visible_flags], s=12)
    ax.scatter([scene.target[0]], [scene.target[1]], marker="*", color="#d62728", s=120, label="target")

    if first_detection_sample is not None:
        _, x, y, theta = first_detection_sample
        left, right = fov_boundary_points((x, y, theta), scene.camera)
        ax.add_patch(
            Wedge(
                (x, y),
                scene.camera.sensing_radius,
                (theta - scene.camera.half_fov_radians) * 180.0 / 3.141592653589793,
                (theta + scene.camera.half_fov_radians) * 180.0 / 3.141592653589793,
                facecolor="#ffd166",
                alpha=0.22,
                edgecolor="#e0a800",
                linewidth=0.8,
            )
        )
        ax.add_patch(
            Polygon([(x, y), left, right], closed=True, facecolor="none", edgecolor="#e0a800", linewidth=0.8)
        )
        ax.scatter([x], [y], color="#ff7f0e", s=45, label="first detection")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "visibility_demo.pdf")
    fig.savefig(FIGURES_DIR / "visibility_demo.png", dpi=200)
    plt.close(fig)


def main() -> None:
    scene = build_scene()
    t_find, sample = first_detection_time(
        scene.trajectory,
        scene.target,
        scene.obstacles,
        scene.camera,
    )

    print(f"T_find = {t_find}")
    print(f"first_detection_sample = {sample}")
    plot_scene(scene, sample)
    print(f"wrote {FIGURES_DIR / 'visibility_demo.pdf'}")


if __name__ == "__main__":
    main()
