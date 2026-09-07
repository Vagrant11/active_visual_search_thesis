"""Protocol v2 scene/error-geometry manifest: no planner import, zero planning.

Builds the proposed 12-scene x 2-direction-block suite from PROTOCOL_V2_DRAFT.md
(4 hotspot centers x 3 obstacle layouts; x/y error-geometry blocks per scene) and
the four error families, plus the true uniform-prior planning-baseline condition.
Also provides discrete geometry/support/feasibility checks used by the preflight
before any calibration or planning is attempted for a scene.
"""

from dataclasses import dataclass
import json

import numpy as np

from milestone_3.experiment import Environment, js_divergence
from visibility import RectangleObstacle
from .protocol import JS_TOLERANCE, LEVELS

GRID_SIZE = 40
SIGMA = 0.13
BACKGROUND = 0.03
MAX_SHIFT = 0.75
MAX_SIGMA_SCALE = 1000.0
FALSE_SIGMA = 0.13
SUPPRESSION_RADIUS = 0.25
SUPPRESSION_OFFSET = 0.10
INITIAL_STATE = (0.1, 0.1, 0.0, 0.0)
TERMINAL_STATE = (0.9, 0.9, 0.0, 0.0)
FAMILIES = ("spatial_shift", "diffuse_blur", "false_hotspot", "false_negative_suppression")
DIRECTION_DEPENDENT_FAMILIES = ("spatial_shift", "false_hotspot", "false_negative_suppression")
CENTERS = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75))
LAYOUTS = {
    "A": (),
    "B": (RectangleObstacle(0.43, 0.43, 0.57, 0.57),),
    "C": (RectangleObstacle(0.35, 0.55, 0.45, 0.65), RectangleObstacle(0.55, 0.35, 0.65, 0.45)),
}


@dataclass(frozen=True)
class Scene:
    scene_id: str
    center: tuple
    layout: str
    obstacles: tuple


def scenes():
    return tuple(Scene(f"cx{cx:.2f}_cy{cy:.2f}_layout{layout}", (cx, cy), layout, obstacles)
                 for cx, cy in CENTERS for layout, obstacles in LAYOUTS.items())


def scene_environment(scene):
    return Environment(obstacles=scene.obstacles)


def sign(value):
    return 1.0 if value > 0 else (-1.0 if value < 0 else 0.0)


def directions(center):
    cx, cy = center
    return {"x": (sign(0.5 - cx), 0.0), "y": (0.0, sign(0.5 - cy))}


def scene_grid(environment, grid_size=GRID_SIZE):
    axis = (np.arange(grid_size) + 0.5) / grid_size
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    free = np.ones(len(points), dtype=bool)
    for cx, cy, radius in environment.collision_disks:
        free &= np.linalg.norm(points - [cx, cy], axis=1) > radius
    return points, free


def gaussian_density(points, free, center, sigma, background):
    weights = np.exp(-np.sum((points - center) ** 2, axis=1) / (2 * sigma ** 2))
    weights = (weights + background) * free
    total = weights.sum()
    if total <= 0:
        raise ValueError("Degenerate density: zero free probability mass")
    return weights / total


def grid_neighbors(index, grid_size):
    row, col = divmod(index, grid_size)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = row + dr, col + dc
        if 0 <= r < grid_size and 0 <= c < grid_size:
            yield r * grid_size + c


def connected_component(mask, grid_size, seed_index):
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask)
    if not mask[seed_index]:
        return visited
    stack = [seed_index]
    visited[seed_index] = True
    while stack:
        i = stack.pop()
        for j in grid_neighbors(i, grid_size):
            if mask[j] and not visited[j]:
                visited[j] = True
                stack.append(j)
    return visited


def is_fully_connected(mask, grid_size):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return False
    visited = connected_component(mask, grid_size, int(np.flatnonzero(mask)[0]))
    return bool(np.array_equal(visited, mask))


def nearest_index(points, xy):
    return int(np.argmin(np.sum((points - xy) ** 2, axis=1)))


def state_feasible(state_xy, environment):
    for cx, cy, radius in environment.collision_disks:
        if np.hypot(state_xy[0] - cx, state_xy[1] - cy) <= radius:
            return False
    return True


def obstacles_within_workspace(obstacles):
    return all(0 <= o.xmin < o.xmax <= 1 and 0 <= o.ymin < o.ymax <= 1 for o in obstacles)


def scene_feasibility(scene):
    """Discrete geometry/support/feasibility checks; no planner, no target draws.

    The start/terminal-connectivity check is a coarse 4-connected grid proxy for
    whether a collision-free path can exist around the obstacles -- it is not a
    guarantee that the continuous-space planner will find or converge to one.
    """
    environment = scene_environment(scene)
    points, free = scene_grid(environment)
    result = {"scene_id": scene.scene_id, "center": scene.center, "layout": scene.layout,
              "obstacles_within_workspace": obstacles_within_workspace(scene.obstacles)}
    try:
        true_prior = gaussian_density(points, free, scene.center, SIGMA, BACKGROUND)
        support = true_prior > 0
        result["nonempty_support"] = bool(support.any())
        result["support_connected"] = is_fully_connected(support, GRID_SIZE)
    except ValueError as error:
        result["nonempty_support"] = False
        result["support_connected"] = False
        result["density_error"] = str(error)
    result["start_state_feasible"] = state_feasible(INITIAL_STATE[:2], environment)
    result["terminal_state_feasible"] = state_feasible(TERMINAL_STATE[:2], environment)
    start_index = nearest_index(points, INITIAL_STATE[:2])
    terminal_index = nearest_index(points, TERMINAL_STATE[:2])
    if free[start_index] and free[terminal_index]:
        component = connected_component(free, GRID_SIZE, start_index)
        result["start_terminal_connected_via_free_space"] = bool(component[terminal_index])
    else:
        result["start_terminal_connected_via_free_space"] = False
    geometry_errors = []
    for direction_name, direction in directions(scene.center).items():
        try:
            SceneErrorGenerator(scene, direction_name, direction)
        except ValueError as error:
            geometry_errors.append(f"{direction_name}: {error}")
    result["error_geometry_valid"] = not geometry_errors
    if geometry_errors:
        result["error_geometry_errors"] = "; ".join(geometry_errors)
    result["passed"] = all(result[k] for k in (
        "obstacles_within_workspace", "nonempty_support", "support_connected",
        "start_state_feasible", "terminal_state_feasible",
        "start_terminal_connected_via_free_space", "error_geometry_valid"))
    return result


class SceneErrorGenerator:
    """Per-scene, per-direction-block analogue of prior_errors.PriorErrorGenerator.

    Deliberately separate from the v1 generator, which intentionally rejects any
    ground truth but the fixed v1 center: v2 scenes vary the true center and
    obstacle layout by design (see PROTOCOL_V2_DRAFT.md), so reusing v1's class
    would require relaxing its truth check, which the draft says not to do
    silently.
    """

    def __init__(self, scene, direction_name, direction):
        self.scene, self.direction_name, self.direction = scene, direction_name, direction
        self.environment = scene_environment(scene)
        self.points, self.free = scene_grid(self.environment)
        self.true_prior = gaussian_density(self.points, self.free, scene.center, SIGMA, BACKGROUND)
        self.support = self.true_prior > 0
        cx, cy = scene.center
        self.false_center = (1 - cx, cy) if direction_name == "x" else (cx, 1 - cy)
        if not (0 <= self.false_center[0] <= 1 and 0 <= self.false_center[1] <= 1):
            raise ValueError("False-hotspot center must remain in the unit workspace")
        end = np.array(scene.center) + MAX_SHIFT * np.array(direction)
        if np.any(end < 0) or np.any(end > 1):
            raise ValueError("Spatial-shift endpoint must remain in the unit workspace")
        self.suppression_center = np.array(scene.center) + SUPPRESSION_OFFSET * np.array(direction)
        self.suppressed = (np.linalg.norm(self.points - self.suppression_center, axis=1)
                           <= SUPPRESSION_RADIUS) & self.support
        if not self.suppressed.any() or np.all(self.suppressed[self.support]):
            raise ValueError("Suppression region must cover a nonempty proper subset of free cells")
        self.false_hotspot = self._gaussian(self.false_center, FALSE_SIGMA, 0.0)

    def _normalize(self, weights):
        weights = np.asarray(weights, dtype=float) * self.support
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("Cannot normalize empty or nonfinite weights")
        return weights / weights.sum()

    def _gaussian(self, center, sigma, background):
        weights = np.exp(-np.sum((self.points - center) ** 2, axis=1) / (2 * sigma ** 2))
        return self._normalize(weights + background)

    def bounds(self, family):
        return {"spatial_shift": ("distance_d", 0.0, MAX_SHIFT),
                "diffuse_blur": ("sigma_scale", 1.0, MAX_SIGMA_SCALE),
                "false_hotspot": ("mixture_alpha", 0.0, 1.0),
                "false_negative_suppression": ("suppression_alpha", 0.0, 1.0)}[family]

    def generate(self, family, parameter):
        _, lo, hi = self.bounds(family)
        if not np.isfinite(parameter) or not lo <= parameter <= hi:
            raise ValueError(f"{family} parameter must be finite and in [{lo}, {hi}]")
        if parameter == lo:
            return self.true_prior.copy()
        if family == "spatial_shift":
            center = np.array(self.scene.center) + parameter * np.array(self.direction)
            return self._gaussian(center, SIGMA, BACKGROUND)
        if family == "diffuse_blur":
            return self._gaussian(self.scene.center, SIGMA * parameter, BACKGROUND)
        if family == "false_hotspot":
            return self._normalize((1 - parameter) * self.true_prior + parameter * self.false_hotspot)
        return self._normalize(self.true_prior * (1 - parameter * self.suppressed))

    def parameter_scan(self, family):
        _, lo, hi = self.bounds(family)
        grid = np.geomspace(lo, hi, 513) if family == "diffuse_blur" else np.linspace(lo, hi, 513)
        distances = np.array([js_divergence(self.true_prior, self.generate(family, a)) for a in grid])
        return grid, distances

    def calibrate(self, family, target_js, tolerance=JS_TOLERANCE, scan=None):
        if not np.isfinite(target_js) or not 0 <= target_js <= np.log(2):
            raise ValueError("JS target must lie in [0, log(2)] nats")
        name, lo, hi = self.bounds(family)
        grid, distances = self.parameter_scan(family) if scan is None else scan
        row = {"family": family, "target_js_nats": float(target_js), "parameter_name": name,
               "parameter_value": None, "actual_js_nats": None, "absolute_error_nats": None,
               "tolerance_nats": tolerance, "status": "unattainable_in_scan", "valid_prior": False,
               "scan_max_js_nats": float(distances.max())}
        crossing = np.flatnonzero((distances[:-1] - target_js) * (distances[1:] - target_js) <= 0)
        if target_js == 0:
            parameter = lo
        elif len(crossing):
            from scipy.optimize import brentq
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
                   status="matched", valid_prior=True)
        return row, prior


def uniform_prior_baseline(scene):
    """True uniform-prior planning-baseline condition: uniform over the scene's

    supported free-cell mask. Distinct from the pilot's post-hoc uniform-weighted
    diagnostic (a reweighting of an existing oracle-conditioned trajectory): this
    prior is meant to be handed to the planner itself, producing its own planned
    trajectory, the same way the oracle/error priors are. Defining it here does
    not run the planner.
    """
    environment = scene_environment(scene)
    points, free = scene_grid(environment)
    true_prior = gaussian_density(points, free, scene.center, SIGMA, BACKGROUND)
    support = true_prior > 0
    return (support.astype(float) / support.sum())


def scene_condition_count():
    """23 distinct priors per scene per gamma: oracle(1) + blur(3, direction-shared)

    + uniform-prior baseline(1, direction-shared) + 3 direction-dependent families
    x 3 JS levels x 2 direction blocks (18)."""
    return 1 + len(LEVELS) + 1 + len(DIRECTION_DEPENDENT_FAMILIES) * len(LEVELS) * 2


def candidate_manifest():
    scene_list = scenes()
    entries = []
    for scene in scene_list:
        conditions = [{"prior_id": "oracle", "family": "oracle", "direction": None, "target_js_nats": 0.0},
                      {"prior_id": "uniform_prior_baseline", "family": "uniform_prior_baseline",
                       "direction": None, "target_js_nats": None}]
        for level in LEVELS:
            conditions.append({"prior_id": f"diffuse_blur_js_{level:g}", "family": "diffuse_blur",
                               "direction": None, "target_js_nats": level})
        for direction_name in ("x", "y"):
            for family in DIRECTION_DEPENDENT_FAMILIES:
                for level in LEVELS:
                    conditions.append({"prior_id": f"{family}_{direction_name}_js_{level:g}",
                                       "family": family, "direction": direction_name, "target_js_nats": level})
        entries.append({"scene_id": scene.scene_id, "center": list(scene.center), "layout": scene.layout,
                        "obstacles": [{"xmin": o.xmin, "ymin": o.ymin, "xmax": o.xmax, "ymax": o.ymax}
                                     for o in scene.obstacles],
                        "directions": {k: list(v) for k, v in directions(scene.center).items()},
                        "conditions": conditions, "conditions_per_gamma": len(conditions)})
    assert all(e["conditions_per_gamma"] == scene_condition_count() for e in entries)
    return {"protocol_id": "m4-controlled-prior-v2-candidate", "scenes": entries,
            "scene_count": len(scene_list), "gammas": [0.10, 0.05],
            "conditions_per_scene_per_gamma": scene_condition_count(),
            "projected_plan_count": len(scene_list) * 2 * scene_condition_count(),
            "js_levels": list(LEVELS), "initial_state": list(INITIAL_STATE),
            "terminal_state": list(TERMINAL_STATE), "grid_size": GRID_SIZE, "sigma": SIGMA,
            "background": BACKGROUND, "suppression_radius": SUPPRESSION_RADIUS,
            "suppression_offset": SUPPRESSION_OFFSET, "max_shift": MAX_SHIFT,
            "primary_endpoints": ["exact_grid_weighted_detection_probability_at_budget",
                                  "exact_grid_weighted_detection_time_distribution_conditional_mean_median"],
            "secondary_endpoints": ["sampled_success_timeout_rate", "paired_outcomes",
                                    "free_space_coverage", "true_prior_mass_coverage",
                                    "entropy_nats", "trajectory_length"],
            "diagnostic_endpoints": ["uniform_weighted_reweighting_of_saved_trajectory"],
            "estimand_note": ("Primary contrasts are reported per (family, gamma) cell without pooling "
                              "gamma; family x gamma interaction on the primary exact-grid endpoints is "
                              "part of the declared estimand, not a post-hoc addition."),
            "uniform_prior_baseline_note": ("A distinct planned condition per scene/gamma: the planner "
                                            "receives a uniform prior over the scene's supported free-cell "
                                            "mask and produces its own trajectory, unlike the pilot's "
                                            "uniform-weighted diagnostic which reweights an existing "
                                            "oracle-conditioned trajectory.")}
