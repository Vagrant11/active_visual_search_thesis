"""Local adapter for the authors' objective, dynamics and spectral constraints.

Uses SLSQP with JAX derivatives instead of the reference augmented-Lagrangian
iteration, and adds hard workspace, speed and continuous segment-clearance
constraints. The upstream submodule is unchanged.
"""

import importlib.util
from pathlib import Path
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize

from .experiment import trajectory_is_collision_free

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/time_optimal_ergodic_search"
sys.path.insert(0, str(UPSTREAM))
# Upstream tracks Python bytecode; importing must not dirty those artifacts.
_previous_bytecode_setting = sys.dont_write_bytecode
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "m3_reference_builder", UPSTREAM / "experiments/comparison_study/build_solver.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
from time_opt_erg_lib.fourier_utils import BasisFunc, get_phik, get_ck
from time_opt_erg_lib.ergodic_metric import ErgodicMetric
sys.dont_write_bytecode = _previous_bytecode_setting


def plan(points, probabilities, gamma, environment, nodes=50, maxiter=600):
    if nodes < 8 or not 0 < gamma or maxiter < 1:
        raise ValueError("Require nodes >= 8, gamma > 0 and maxiter >= 1")
    started = perf_counter()
    basis = BasisFunc([8, 8])
    # An identical deterministic initial guess for every prior and gamma.
    waypoints = np.array([[0.1, 0.1], [0.8, 0.2], [0.8, 0.8], [0.9, 0.9]])
    xy = np.column_stack([np.interp(np.linspace(0, 3, nodes), np.arange(4), waypoints[:, i])
                          for i in (0, 1)])
    initial = {"x": jnp.array(np.column_stack((xy, np.zeros_like(xy)))),
               "u": jnp.zeros((nodes, 2)), "tf": jnp.array(10.0)}
    args = {"N": nodes, "x0": jnp.array([0.1, 0.1, 0., 0.]),
            "xf": jnp.array([0.9, 0.9, 0., 0.]), "erg_ub": gamma}
    solver = reference.build_erg_time_opt_solver(initial, args)
    # The builder installs uniform phik: override AFTER construction.
    args["phik"] = get_phik((jnp.array(probabilities), jnp.array(points)), basis)
    flat, unravel = ravel_pytree(initial)
    disks = environment.collision_disks

    def inequalities(z):
        params = unravel(z)
        x, tf = params["x"], params["tf"]
        xy = x[:, :2]
        extra = [solver.ineq_constr(params, args), (-xy).ravel(), (xy - 1).ravel(),
                 jnp.sum(x[:, 2:] ** 2, axis=1) - 1.0, jnp.array([0.1 - tf])]
        for cx, cy, radius in disks:
            a, delta = xy[:-1], xy[1:] - xy[:-1]
            fraction = jnp.clip(jnp.sum((jnp.array([cx, cy]) - a) * delta, axis=1)
                                / (jnp.sum(delta**2, axis=1) + 1e-15), 0, 1)
            closest = a + fraction[:, None] * delta
            extra.append(radius**2 - jnp.sum((closest - jnp.array([cx, cy]))**2, axis=1))
        return -jnp.concatenate(extra)  # scipy convention: >= 0

    loss = jax.jit(lambda z: solver.loss(unravel(z), args))
    eq = jax.jit(lambda z: solver.eq_constr(unravel(z), args).ravel())
    ineq = jax.jit(inequalities)
    gradient, eq_jac, ineq_jac = map(jax.jit, (jax.grad(loss), jax.jacfwd(eq), jax.jacfwd(ineq)))
    # Compile before separately timing optimization; total includes compilation.
    for function in (loss, eq, ineq, gradient, eq_jac, ineq_jac):
        function(flat).block_until_ready()
    optimize_started = perf_counter()
    result = minimize(loss, np.asarray(flat), jac=gradient, method="SLSQP",
                      constraints=[{"type": "eq", "fun": eq, "jac": eq_jac},
                                   {"type": "ineq", "fun": ineq, "jac": ineq_jac}],
                      options={"maxiter": maxiter, "ftol": 1e-9})
    optimization_seconds = perf_counter() - optimize_started
    sol = {k: np.asarray(v) for k, v in unravel(result.x).items()}
    tf = float(sol["tf"])
    eq_error = float(np.max(np.abs(eq(result.x))))
    ineq_error = float(max(0, -np.min(ineq(result.x))))
    ergodicity = float(ErgodicMetric(basis)(get_ck(jnp.array(sol["x"][:, :2]), basis, tf, tf / nodes), args["phik"]))
    finite = bool(all(np.isfinite(v).all() for v in sol.values())
                  and np.isfinite([eq_error, ineq_error, ergodicity]).all())
    collision_free = finite and trajectory_is_collision_free(sol["x"], environment)
    feasible = bool(finite and tf >= 0.1 and eq_error <= 1e-5 and ineq_error <= 1e-6 and collision_free)
    diagnostics = {"valid": bool(result.success and feasible), "feasible": feasible,
                   "solver_success": bool(result.success), "solver_message": str(result.message),
                   "iterations": int(result.nit), "finite": finite,
                   "max_equality_residual": eq_error, "max_inequality_violation": ineq_error,
                   "collision_free": bool(collision_free), "ergodicity": ergodicity,
                   "optimization_seconds": optimization_seconds,
                   "planning_seconds_including_compile": perf_counter() - started,
                   "backend": "scipy-SLSQP / upstream formulation"}
    return {"states": sol["x"], "controls": sol["u"], "tf": tf, "diagnostics": diagnostics}
