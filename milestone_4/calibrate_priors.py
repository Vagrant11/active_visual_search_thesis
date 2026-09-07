"""Generate calibration artifacts only: python -m milestone_4.calibrate_priors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .prior_errors import FAMILIES, PriorErrorGenerator
from .protocol import EXAMPLE_LEVELS, LEVELS, ROOT, protocol_manifest


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prior_id(family, level):
    return f"{family}_js_{level:g}"


def calibrate_suite(levels=LEVELS):
    levels = tuple(levels)
    if not levels or len({prior_id("level", x) for x in levels}) != len(levels):
        raise ValueError("Require nonempty, distinct JS levels")
    generator = PriorErrorGenerator()
    rows, priors, audit = [], {}, []
    for family in FAMILIES:
        scan = generator.parameter_scan(family)
        for level in levels:
            row, prior = generator.calibrate(family, level, scan=scan)
            key = prior_id(family, level)
            rows.append({"prior_id": key, **row})
            if prior is not None:
                priors[key] = prior
        for level in EXAMPLE_LEVELS:
            row, _ = generator.calibrate(family, level, scan=scan)
            audit.append(row)
    return generator, rows, priors, audit


def plot_priors(out, generator, rows, priors, levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(len(FAMILIES), len(levels), figsize=(4 * len(levels), 12), squeeze=False,
                             layout="constrained")
    vmax = max([generator.true_prior.max()] + [p.max() for p in priors.values()])
    for i, family in enumerate(FAMILIES):
        for j, level in enumerate(levels):
            ax = axes[i, j]
            key = prior_id(family, level)
            row = next(r for r in rows if r["prior_id"] == key)
            if key in priors:
                values = np.ma.masked_where(~generator.support, priors[key]).reshape(40, 40)
                ax.imshow(values, extent=(0, 1, 0, 1), origin="lower", cmap="YlOrRd", vmin=0, vmax=vmax)
                ax.set_title(f"{family.replace('_', ' ')}\nJS={row['actual_js_nats']:.5f}; {row['parameter_name']}={row['parameter_value']:.4f}", fontsize=9)
            else:
                ax.set_title(f"{family}\nJS={level:g}: unattainable", fontsize=9)
            ax.plot(.75, .70, "+", color="#007c91", markersize=12)
            for o in generator.environment.obstacles:
                ax.add_patch(Rectangle((o.xmin, o.ymin), o.xmax-o.xmin, o.ymax-o.ymin, color="#333333"))
            for cx, cy, radius in generator.environment.collision_disks:
                ax.add_patch(plt.Circle((cx, cy), radius, fill=False, color="#777777", linestyle="--"))
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    from matplotlib.cm import ScalarMappable
    fig.colorbar(ScalarMappable(norm=plt.Normalize(0, vmax), cmap="YlOrRd"), ax=axes.ravel().tolist(),
                 shrink=.65, label="Predicted probability per cell (shared scale)")
    fig.suptitle("Milestone 4: matched JS errors; + marks the fixed true hotspot")
    for extension in ("png", "pdf"):
        fig.savefig(out / f"calibrated_priors.{extension}", dpi=160)
    plt.close(fig)


def write_calibration(out, levels=LEVELS):
    out = Path(out).resolve()
    if out == ROOT / "milestone_3" or ROOT / "milestone_3" in out.parents:
        raise ValueError("Milestone 3 is frozen; choose a Milestone 4 output directory")
    manifest = protocol_manifest()
    generator, rows, priors, audit = calibrate_suite(levels)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "calibration.csv", rows)
    write_csv(out / "example_levels_audit.csv", audit)
    np.savez(out / "priors.npz", points=generator.points, true_prior=generator.true_prior,
             free_mask=generator.support, **priors)
    manifest["js"]["levels"] = list(levels)
    manifest["prior_errors"] = generator.metadata()
    manifest["calibration_passed"] = all(r["status"] == "matched" for r in rows)
    manifest["prior_sha256"] = {key: hashlib.sha256(p.tobytes()).hexdigest() for key, p in priors.items()}
    (out / "protocol.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    plot_priors(out, generator, rows, priors, levels)
    return generator, rows, priors, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "milestone_4/results/calibration")
    parser.add_argument("--levels", type=float, nargs="+", default=LEVELS)
    args = parser.parse_args()
    _, rows, _, manifest = write_calibration(args.output, tuple(args.levels))
    for row in rows:
        print(f"{row['prior_id']}: {row['status']}; parameter={row['parameter_value']}; JS={row['actual_js_nats']}")
    if not manifest["calibration_passed"]:
        raise SystemExit("Unmatched levels; see calibration.csv. These priors must not enter a matched pilot.")
    print(f"Calibration only; planner was not imported. Results: {args.output}")


if __name__ == "__main__":
    main()
