# Gamma call-site audit

Gamma is passed unchanged as erg_ub. The constraint is E - gamma <= 0.
The adapter negates inequalities for SciPy (>= 0); this changes sign, not scale.
Workspace mapping uses [0,1] on both axes and is the identity here.
Fourier coefficients are normalized (dt/tf=1/N, prior total mass, hk); gamma itself is not renormalized.

## milestone_4/run_formal.py

```python
270:     try:
271:         from milestone_3.planner import plan
272:         scene = next(s for s in config['frozen_manifest']['candidate_manifest']['scenes'] if s['scene_id'] == label['scene_id'])
273:         settings = config['frozen_manifest']['planner_and_evaluation_invariants']['planner']
274:         with np.load(output/'inputs'/f"{label['scene_id']}.npz", allow_pickle=False) as data:
275:             require(array_sha(data[label['prior_id']]) == expected_prior_hash, 'Worker input prior changed')
276:             policy = plan(data['points'], data[label['prior_id']], label['gamma'], environment_for(scene), settings['nodes'], settings['maxiter'])
277:         save_solver_result(directory, policy)
```

## milestone_4/run_pilot.py

```python
146:             key = condition["prior_id"]
147:             probabilities = generator.true_prior if key == "oracle" else priors[key]
148:             label = {"prior_id": key, "family": condition["family"],
149:                      "target_js_nats": float(condition["target_js_nats"]), "gamma": gamma}
150:             print(f"Planning {key}, gamma={gamma:g} ...", flush=True)
151:             policy = plan(generator.points, probabilities, gamma, generator.environment, NODES, MAXITER)
152:             policies[key, gamma] = policy
153:             diagnostics.append({**label, **policy["diagnostics"]})
154:             np.savez(out / f"trajectory_{key}_gamma_{gamma:g}.npz", states=policy["states"],
```

## milestone_3/planner.py

```python
39:         raise ValueError("Require nodes >= 8, gamma > 0 and maxiter >= 1")
40:     started = perf_counter()
41:     basis = BasisFunc([8, 8])
42:     # An identical deterministic initial guess for every prior and gamma.
43:     waypoints = np.array([[0.1, 0.1], [0.8, 0.2], [0.8, 0.8], [0.9, 0.9]])
44:     xy = np.column_stack([np.interp(np.linspace(0, 3, nodes), np.arange(4), waypoints[:, i])
45:                           for i in (0, 1)])
46:     initial = {"x": jnp.array(np.column_stack((xy, np.zeros_like(xy)))),
47:                "u": jnp.zeros((nodes, 2)), "tf": jnp.array(10.0)}
48:     args = {"N": nodes, "x0": jnp.array([0.1, 0.1, 0., 0.]),
49:             "xf": jnp.array([0.9, 0.9, 0., 0.]), "erg_ub": gamma}
50:     solver = reference.build_erg_time_opt_solver(initial, args)
51:     # The builder installs uniform phik: override AFTER construction.
52:     args["phik"] = get_phik((jnp.array(probabilities), jnp.array(points)), basis)
53:     flat, unravel = ravel_pytree(initial)
54:     disks = environment.collision_disks
68:         return -jnp.concatenate(extra)  # scipy convention: >= 0
```

## external/time_optimal_ergodic_search/experiments/comparison_study/build_solver.py

```python
104:     def ineq_constr(params, args):
105:         """ inequality constraints"""
106:         x = params['x']
107:         u = params['u']
108:         phik = args['phik']
109:         tf = params['tf']
110:         N = args['N']
111:         dt = tf/N
112:         e = emap(x)
113:         # _cbf_ineq = [vmap(_cbf_ineq, in_axes=(0,0,None, None))(x, u, args['alpha'], dt).flatten() 
114:         #            for _cbf_ineq in cbf_constr]
115:         ck = get_ck(e, basis, tf, dt)
116:         _erg_ineq = [np.array([erg_metric(ck, phik) - args['erg_ub'], -tf])]
117:         _ctrl_box = [(np.abs(u) - 1.).flatten()]
118:         return np.concatenate(_erg_ineq + _ctrl_box)# + _cbf_ineq)
```

## external/time_optimal_ergodic_search/time_opt_erg_lib/fourier_utils.py

```python
12: def get_ck(trajectory, basis, tf, dt):
13:     ck = np.sum(vmap(basis.fk_vmap)(trajectory), axis=0)
14:     ck = ck / basis.hk_list
15:     ck = ck * dt / tf
16:     return ck
17: 
18: def get_phik(vals, basis):
19:     _phi, _x = vals 
20:     phik = np.dot(_phi, vmap(basis.fk_vmap)(_x))
21:     phik = phik/phik[0]
22:     phik = phik/basis.hk_list
23:     return phik
24: 
```

## external/time_optimal_ergodic_search/time_opt_erg_lib/ergodic_metric.py

```python
6:     def __init__(self, basis) -> None:
7:         self.basis = basis
8:         self.lamk = (1.+np.linalg.norm(basis.k_list/np.pi,axis=1)**2)**(-(basis.n+1)/2.)
9:         # self.lamk = 1.0
10:         # lamk = np.exp(-0.8 * np.linalg.norm(k, axis=1))
11:         # lamk = np.ones((len(k), 1))
12:     def __call__(self, ck, phik):
13:         # return (self.lamk * (ck - phik)**2).flatten()
14:         return np.sum(self.lamk * (ck - phik)**2)
```
