"""Four one-parameter error families; no planner imports or target access."""

from dataclasses import asdict, dataclass
import json

import numpy as np
from scipy.optimize import brentq

from milestone_3.experiment import js_divergence
from .protocol import JS_TOLERANCE, fixed_inputs

FAMILIES = ("spatial_shift", "diffuse_blur", "false_hotspot", "false_negative_suppression")


@dataclass(frozen=True)
class ErrorDesign:
    true_center: tuple = (0.75, 0.70)
    sigma: float = 0.13
    background: float = 0.03
    shift_direction: tuple = (-1.0, 0.0)
    max_shift: float = 0.75
    max_sigma_scale: float = 1000.0
    false_center: tuple = (0.25, 0.70)
    false_sigma: float = 0.13
    suppression_radius: float = 0.25


def validate_prior(probabilities, support):
    p = np.asarray(probabilities, dtype=float)
    support = np.asarray(support, dtype=bool)
    if (p.ndim != 1 or p.shape != support.shape or not np.isfinite(p).all()
            or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)
            or np.any(p[~support] != 0)):
        raise ValueError("Prior must be finite, nonnegative, normalized and zero on blocked cells")
    return p


class PriorErrorGenerator:
    def __init__(self, design=ErrorDesign()):
        self.design = design
        self.environment, self.points, self.true_prior = fixed_inputs()
        self.support = self.true_prior > 0
        # Ground truth is fixed; these are recorded explicitly, never fitted to targets.
        if (design.true_center != (0.75, 0.70) or design.sigma != 0.13
                or design.background != 0.03):
            raise ValueError("Ground-truth center, sigma and background are fixed by protocol")
        values = [*design.shift_direction, design.max_shift, design.max_sigma_scale,
                  *design.false_center, design.false_sigma, design.suppression_radius]
        if (not np.isfinite(values).all() or len(design.shift_direction) != 2
                or len(design.false_center) != 2
                or not np.isclose(np.linalg.norm(design.shift_direction), 1)
                or design.max_shift <= 0 or design.max_sigma_scale <= 1
                or design.false_sigma <= 0 or design.suppression_radius <= 0):
            raise ValueError("Invalid error-family geometry or parameter bounds")
        end = np.array(design.true_center) + design.max_shift * np.array(design.shift_direction)
        if np.any(end < 0) or np.any(end > 1) or np.any(np.array(design.false_center) < 0) or np.any(np.array(design.false_center) > 1):
            raise ValueError("Hotspot centers must remain in the unit workspace")
        self.suppressed = (np.linalg.norm(self.points - design.true_center, axis=1)
                           <= design.suppression_radius) & self.support
        if not self.suppressed.any() or np.all(self.suppressed[self.support]):
            raise ValueError("Suppression region must cover a nonempty proper subset of free cells")
        self.false_hotspot = self._gaussian(design.false_center, design.false_sigma, 0.0)

    def _normalize(self, weights):
        weights = np.asarray(weights, dtype=float) * self.support
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("Cannot normalize empty or nonfinite weights")
        return validate_prior(weights / weights.sum(), self.support)

    def _gaussian(self, center, sigma, background):
        weights = np.exp(-np.sum((self.points - center)**2, axis=1) / (2 * sigma**2))
        return self._normalize(weights + background)

    def bounds(self, family):
        d = self.design
        return {"spatial_shift": ("distance_d", 0.0, d.max_shift),
                "diffuse_blur": ("sigma_scale", 1.0, d.max_sigma_scale),
                "false_hotspot": ("mixture_alpha", 0.0, 1.0),
                "false_negative_suppression": ("suppression_alpha", 0.0, 1.0)}[family]

    def generate(self, family, parameter):
        _, lo, hi = self.bounds(family)
        if not np.isfinite(parameter) or not lo <= parameter <= hi:
            raise ValueError(f"{family} parameter must be finite and in [{lo}, {hi}]")
        if parameter == lo:
            return self.true_prior.copy()
        d = self.design
        if family == "spatial_shift":
            center = np.array(d.true_center) + parameter * np.array(d.shift_direction)
            return self._gaussian(center, d.sigma, d.background)
        if family == "diffuse_blur":
            return self._gaussian(d.true_center, d.sigma * parameter, d.background)
        if family == "false_hotspot":
            return self._normalize((1 - parameter) * self.true_prior + parameter * self.false_hotspot)
        return self._normalize(self.true_prior * (1 - parameter * self.suppressed))

    def resolved_parameters(self, family, parameter):
        d = self.design
        if family == "spatial_shift":
            return {"center": (np.array(d.true_center) + parameter * np.array(d.shift_direction)).tolist(),
                    "direction": list(d.shift_direction), "sigma": d.sigma, "background": d.background}
        if family == "diffuse_blur":
            return {"center": list(d.true_center), "sigma": d.sigma * parameter,
                    "variance_multiplier": parameter**2, "background": d.background}
        if family == "false_hotspot":
            return {"false_center": list(d.false_center), "false_sigma": d.false_sigma,
                    "hotspot_component_background": 0.0}
        return {"region_center": list(d.true_center), "region_radius": d.suppression_radius,
                "region_true_mass": float(self.true_prior[self.suppressed].sum()),
                "removed_mass_before_normalization": float(parameter * self.true_prior[self.suppressed].sum())}

    def parameter_scan(self, family):
        _, lo, hi = self.bounds(family)
        grid = np.geomspace(lo, hi, 513) if family == "diffuse_blur" else np.linspace(lo, hi, 513)
        distances = np.array([js_divergence(self.true_prior, self.generate(family, a)) for a in grid])
        return grid, distances

    def calibrate(self, family, target_js, tolerance=JS_TOLERANCE, scan=None):
        if not np.isfinite(target_js) or not 0 <= target_js <= np.log(2):
            raise ValueError("JS target must lie in [0, log(2)] nats")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Tolerance must be finite and positive")
        name, lo, hi = self.bounds(family)
        grid, distances = self.parameter_scan(family) if scan is None else scan
        row = {"family": family, "target_js_nats": float(target_js), "parameter_name": name,
               "parameter_value": None, "actual_js_nats": None, "absolute_error_nats": None,
               "tolerance_nats": tolerance, "status": "unattainable_in_scan", "valid_prior": False,
               "parameter_lower_bound": lo, "parameter_upper_bound": hi,
               "scan_max_js_nats": float(distances.max()), "probability_sum": None,
               "minimum_probability": None, "blocked_probability_mass": None,
               "resolved_parameters_json": ""}
        # Scan first: masks/boundaries can make a family nonmonotone. Do not assume
        # the far endpoint brackets every attainable JS value or silently clip it.
        crossing = np.flatnonzero((distances[:-1] - target_js) * (distances[1:] - target_js) <= 0)
        if target_js == 0:
            parameter = lo
        elif len(crossing):
            i = crossing[0]
            parameter = brentq(lambda a: js_divergence(self.true_prior, self.generate(family, a)) - target_js,
                               grid[i], grid[i + 1], xtol=1e-12, rtol=1e-14)
        else:
            return row, None
        prior = self.generate(family, parameter)
        actual = js_divergence(self.true_prior, prior)
        error = abs(actual - target_js)
        if error > tolerance:
            raise RuntimeError(f"Calibration failed tolerance: {family}, {target_js}, {actual}")
        row.update(parameter_value=float(parameter), actual_js_nats=actual, absolute_error_nats=error,
                   status="matched", valid_prior=True, probability_sum=float(prior.sum()),
                   minimum_probability=float(prior.min()), blocked_probability_mass=float(prior[~self.support].sum()),
                   resolved_parameters_json=json.dumps(self.resolved_parameters(family, parameter), sort_keys=True))
        return row, prior

    def metadata(self):
        return {"design": asdict(self.design), "scan_points_per_family": 513,
                "root_solver": "scipy.optimize.brentq; first scanned crossing",
                "blur_uniform_limit_js_nats": js_divergence(self.true_prior, self.support / self.support.sum())}
