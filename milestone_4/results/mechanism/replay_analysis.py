"""Pure stored-plan replay. No planner import; all new output stays beside this file.

Constraints for tasks 1, 2 and 3: M3, frozen protocols and all existing results
are read-only. Never plan, regenerate inputs, or overwrite existing results.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import ast
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(BASE / 'matplotlib_cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(BASE / 'cache'))
import numpy as np
from milestone_3.experiment import Environment
from milestone_4 import analyze_pilot as pilot_analysis
from milestone_4.run_pilot import plot_trajectories
from visibility import RectangleObstacle

RESULTS = ROOT / 'milestone_4/results'
FAMILIES = ('spatial_shift', 'diffuse_blur', 'false_hotspot', 'false_negative_suppression')
METRICS = ('tf_s', 'path_length', 'visible_area', 'entropy_node_nats',
           'entropy_dwell_tf_nats', 'entropy_dwell_15s_nats', 'ergodic_metric')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entropy(p):
    p = np.asarray(p, dtype=float)
    if np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError('Invalid occupancy weights')
    p = p[p > 0] / p.sum()
    return float(-np.dot(p, np.log(p)))


def cell_ids(xy, n=40):
    xy = np.asarray(xy)
    if not np.isfinite(xy).all() or np.any(xy < -1e-6) or np.any(xy > 1 + 1e-6):
        raise ValueError('Trajectory outside unit workspace')
    ij = np.minimum((np.clip(xy, 0, 1) * n).astype(int), n - 1)
    return ij[..., 1] * n + ij[..., 0]


def node_occupancy(states, n=40):
    # get_ck uses dt/tf = 1/N, including the terminal node.
    return np.bincount(cell_ids(states[:, :2], n), minlength=n*n) / len(states)


def dwell_occupancy(states, tf, horizon, n=40):
    """Exact linear-segment cell residence, then terminal hold; no resampling."""
    if tf <= 0 or horizon <= 0:
        raise ValueError('Nonpositive time')
    xy = np.asarray(states)[:, :2]
    cell_ids(xy, n)
    knots = np.arange(len(xy)) * tf / len(xy)
    mass = np.zeros(n*n)
    edges = np.arange(1, n) / n
    for i, (a, b) in enumerate(zip(xy[:-1], xy[1:])):
        duration = min(knots[i+1], horizon) - knots[i]
        if duration <= 0:
            break
        b = a + (b-a) * duration / (knots[i+1]-knots[i])
        delta = b-a
        cuts = [0., 1.]
        for d in range(2):
            if delta[d] != 0:
                cuts.extend(t for t in (edges-a[d])/delta[d] if 0 < t < 1)
        cuts = np.unique(cuts)
        for low, high in zip(cuts[:-1], cuts[1:]):
            mass[int(cell_ids(a + (low+high)/2*delta, n))] += duration*(high-low)
    mass[int(cell_ids(xy[-1], n))] += max(0., horizon-knots[-1])
    if not np.isclose(mass.sum(), horizon, atol=1e-10):
        raise AssertionError('Residence durations do not sum to horizon')
    return mass / horizon


def spectral_basis(xy):
    k = np.stack([v.ravel() for v in np.meshgrid(np.arange(8), np.arange(8))], axis=1)
    # For k*pi, the source hk formula is 1 for zero and sqrt(1/2) otherwise.
    hk = np.sqrt(np.prod(np.where(k == 0, 1., .5), axis=1))
    values = np.prod(np.cos(np.asarray(xy)[:, None, :] * k[None, :, :] * np.pi), axis=2) / hk
    return values, (1 + np.sum(k*k, axis=1))**-1.5


def achieved_metric(states, points, prior):
    node_basis, lam = spectral_basis(states[:, :2])
    grid_basis, _ = spectral_basis(points)
    ck = node_basis.mean(axis=0)
    phik = np.sum((prior / prior.sum())[:, None] * grid_basis, axis=0)
    return float(np.dot(lam, (ck-phik)**2))


def visibility(first, oracle_first, truth, budget=15.):
    seen = np.isfinite(first) & (first <= budget)
    oracle = np.isfinite(oracle_first) & (oracle_first <= budget)
    gained = float(truth[seen & ~oracle].sum()) * 100
    lost = float(truth[oracle & ~seen].sum()) * 100
    error_mass, oracle_mass = float(truth[seen].sum()), float(truth[oracle].sum())
    if not np.isclose(gained-lost, (error_mass-oracle_mass)*100, atol=1e-10):
        raise AssertionError('Mass decomposition failed')
    return dict(error_visible_mass=error_mass, oracle_visible_mass=oracle_mass,
                gained_pp=gained, lost_pp=lost, net_pp=gained-lost)


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def summarise(rows, keys, metrics):
    result = []
    for key, group in grouped(rows, keys).items():
        row = dict(zip(keys, key), n=len(group))
        for metric in metrics:
            vals = [r[metric] for r in group]
            row[metric+'_mean'] = float(np.mean(vals))
            row[metric+'_median'] = float(np.median(vals))
        result.append(row)
    return result


def diagnostics(label, policy, points, prior, truth, first, saved_metric):
    states, tf = policy['states'], float(policy['tf'])
    n = int(round(np.sqrt(len(points))))
    axis = (np.arange(n)+.5)/n
    xx, yy = np.meshgrid(axis, axis)
    np.testing.assert_allclose(points, np.column_stack((xx.ravel(), yy.ravel())), atol=1e-14)
    metric = achieved_metric(states, points, prior)
    np.testing.assert_allclose(metric, saved_metric, rtol=0, atol=1e-9)
    seen = np.isfinite(first) & (first <= 15) & (truth > 0)
    length = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()
    times = np.arange(len(states)) * tf / len(states)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))]
    return dict(label, tf_s=tf, path_length=float(length),
                camera_path_length_15s=float(np.interp(15., times, lengths)),
                visible_cells=int(seen.sum()), supported_cells=int((truth > 0).sum()),
                visible_area=float(seen.sum()/n**2), visible_free_fraction=float(seen.sum()/(truth > 0).sum()),
                entropy_node_nats=entropy(node_occupancy(states, n)),
                entropy_dwell_tf_nats=entropy(dwell_occupancy(states, tf, tf, n)),
                entropy_dwell_15s_nats=entropy(dwell_occupancy(states, tf, 15., n)),
                ergodic_metric=metric, ergodic_metric_saved=saved_metric,
                ergodic_metric_abs_difference=abs(metric-saved_metric),
                ergodic_constraint_residual=metric-label['gamma'], nodes=len(states),
                terminal_arrival_s=float(times[-1]))


def load_all():
    records, excluded, context = [], [], {}
    manifest = read_json(RESULTS/'frozen/PROTOCOL_V2_FROZEN_MANIFEST.json')
    for scene in manifest['candidate_manifest']['scenes']:
        data = load_npz(RESULTS/'frozen/priors'/f"{scene['scene_id']}.npz")
        for prior_id, expected in manifest['prior_sha256_by_scene'][scene['scene_id']].items():
            assert hashlib.sha256(data[prior_id].tobytes()).hexdigest() == expected
        for gamma in (.05, .1):
            for condition in scene['conditions']:
                plan_id = f"{scene['scene_id']}__gamma_{gamma:g}__{condition['prior_id']}"
                directory = RESULTS/'formal/plans'/plan_id
                result = read_json(directory/'result.json')
                label = dict(dataset='formal', scene=scene['scene_id'], gamma=gamma,
                             family=condition['family'], js=condition['target_js_nats'],
                             direction=condition['direction'], prior_id=condition['prior_id'], plan_id=plan_id)
                if not result['valid']:
                    excluded.append(dict(label, status=result['status'], reason=read_json(directory/'diagnostics.json')['solver_message']))
                    continue
                for name, expected in result['artifact_sha256'].items():
                    assert sha(directory/name) == expected, (plan_id, name)
                policy = load_npz(directory/'trajectory.npz')
                first = load_npz(directory/'first_seen_grid.npz')['first_seen']
                diag = read_json(directory/'diagnostics.json')
                exact = read_json(directory/'exact_metrics.json')
                row = diagnostics(label, policy, data['points'], data[condition['prior_id']], data['oracle'], first, diag['ergodicity'])
                np.testing.assert_allclose(row['path_length'], exact['trajectory_length'], atol=1e-12)
                np.testing.assert_allclose(data['oracle'][np.isfinite(first)].sum(), exact['exact_detection_probability'], atol=1e-12)
                records.append(row)
                context['formal', plan_id] = (policy, first, data, scene)
    inputs = load_npz(RESULTS/'pilot/inputs.npz')
    replay = load_npz(RESULTS/'analysis/visibility_replay.npz')
    np.testing.assert_array_equal(inputs['points'], replay['points'])
    np.testing.assert_array_equal(inputs['true_prior'], replay['true_prior'])
    config = read_json(RESULTS/'pilot/config.json')
    for key, expected in config['prior_sha256'].items():
        assert hashlib.sha256(inputs[key].tobytes()).hexdigest() == expected
    for d in pilot_analysis.read_csv(RESULTS/'pilot/planner_diagnostics.csv'):
        key, gamma = d['prior_id'], d['gamma']
        plan_id = f'{key}_gamma_{gamma:g}'
        label = dict(dataset='pilot', scene='pilot_cx0.75_cy0.70_center_obstacle', gamma=gamma,
                     family=d['family'], js=d['target_js_nats'], direction=None, prior_id=key, plan_id=plan_id)
        if not d['valid']:
            excluded.append(dict(label, status='invalid', reason=d['solver_message']))
            continue
        policy = load_npz(RESULTS/'pilot'/f'trajectory_{plan_id}.npz')
        policy['diagnostics'] = d
        first = replay[plan_id]
        # Independently replay all 26 pilot plans with the original visibility layer.
        np.testing.assert_array_equal(first, pilot_analysis.first_seen_grid(policy, Environment(), inputs['points'], inputs['true_prior'] > 0))
        prior = inputs['true_prior'] if key == 'oracle' else inputs[key]
        records.append(diagnostics(label, policy, inputs['points'], prior, inputs['true_prior'], first, d['ergodicity']))
        context['pilot', plan_id] = (policy, first, inputs, None)
    assert sum(r['dataset'] == 'formal' for r in records) == 546
    assert sum(r['dataset'] == 'pilot' for r in records) == 26
    assert len(excluded) == 6
    return records, excluded, context


def task1(out, records, excluded):
    """Read-only sources; replay only; exclusive new files under mechanism."""
    write_csv(out/'trajectory_diagnostics.csv', records)
    write_csv(out/'excluded_plans.csv', excluded)
    summary = summarise(records, ['dataset', 'gamma', 'family'], METRICS)
    write_csv(out/'diagnostic_summary.csv', summary)
    write_csv(out/'diagnostic_summary_by_js.csv', summarise(records, ['dataset', 'gamma', 'family', 'js'], METRICS))
    oracle = {(r['dataset'], r['scene'], r['gamma']):r for r in records if r['family'] == 'oracle'}
    paired = []
    for r in records:
        if r['family'] == 'oracle':
            continue
        base = oracle[r['dataset'], r['scene'], r['gamma']]
        paired.append(dict({k:r[k] for k in ['dataset', 'scene', 'gamma', 'family', 'js', 'direction', 'plan_id']},
                           **{f'delta_{m}':r[m]-base[m] for m in METRICS}))
    write_csv(out/'diagnostic_oracle_pairs.csv', paired)
    deltas = summarise(paired, ['dataset', 'gamma', 'family'], ['delta_'+m for m in METRICS])
    for d in deltas:
        group = [r for r in paired if all(r[k] == d[k] for k in ['dataset', 'gamma', 'family'])]
        for m in ('tf_s', 'entropy_node_nats', 'entropy_dwell_tf_nats', 'entropy_dwell_15s_nats'):
            d['positive_delta_'+m+'_count'] = sum(r['delta_'+m] > 1e-10 for r in group)
    write_csv(out/'diagnostic_oracle_deltas.csv', deltas)
    for d in deltas:
        print(f"{d['dataset']:6} gamma={d['gamma']:.2f} {d['family']:27} n={d['n']:3} "
              f"delta_tf={d['delta_tf_s_mean']:+.6f} s delta_Hnode={d['delta_entropy_node_nats_mean']:+.6f} nats "
              f"delta_Hdwell_tf={d['delta_entropy_dwell_tf_nats_mean']:+.6f} nats", flush=True)
    return deltas


def task2(out, records, context):
    """Read-only sources; original plotting statements; new mechanism output only."""
    figures = out/'pilot_figures'
    figures.mkdir()
    selected = [r for r in records if r['dataset'] == 'pilot']
    inputs = context['pilot', selected[0]['plan_id']][2]
    policies = {(r['prior_id'], r['gamma']):context['pilot', r['plan_id']][0] for r in selected}
    firsts = {(r['prior_id'], r['gamma']):context['pilot', r['plan_id']][1] for r in selected}
    config = read_json(RESULTS/'pilot/config.json')
    mechanism = []
    comparison = []
    for family in FAMILIES:
        for js in (.05, .1, .15):
            key = f'{family}_js_{js:g}'
            row = dict(family=family, js=js)
            for gamma in (.1, .05):
                v = visibility(firsts[key, gamma], firsts['oracle', gamma], inputs['true_prior'])
                row.update({f'{k}_gamma_{gamma:g}':value for k, value in v.items()})
                mechanism.append(dict(prior_id=key, gamma=gamma, gained_true_prior_mass=v['gained_pp']/100,
                                      lost_true_prior_mass=v['lost_pp']/100))
            comparison.append(row)
    write_csv(out/'pilot_gains_losses_comparison.csv', comparison)
    generator = SimpleNamespace(support=inputs['true_prior'] > 0, environment=Environment())
    priors = {r['prior_id']:inputs[r['prior_id']] for r in selected if r['prior_id'] != 'oracle'}
    calibration = pilot_analysis.read_csv(RESULTS/'calibration/calibration.csv')
    plot_trajectories(figures, generator, calibration, priors, policies)
    # Reuse the exact coverage-map AST block of the original function, without
    # invoking its unrelated metrics/paired/scatter plots. No style edits.
    source = (ROOT/'milestone_4/analyze_pilot.py').read_text()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plot_results')
    loop = next(n for n in fn.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'gamma')
    start = next(i for i, n in enumerate(loop.body) if 'plt.subplots(4, 3' in ast.get_source_segment(source, n))
    end = next(i for i, n in enumerate(loop.body) if 'save_figure(fig, out, f"coverage_maps_gamma_' in ast.get_source_segment(source, n))
    imports = [n for n in fn.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    block = ast.Module(body=imports + loop.body[start:end+1], type_ignores=[])
    namespace = dict(vars(pilot_analysis), out=figures, levels=config['js']['levels'],
                     inputs=inputs, env=Environment(), policies=policies, firsts=firsts, mechanism=mechanism)
    for gamma in (.1, .05):
        namespace['gamma'] = gamma
        exec(compile(block, str(ROOT/'milestone_4/analyze_pilot.py'), 'exec'), namespace)
    return comparison


def scene_visibility(blocks):
    rows = []
    keys = ['gamma', 'family', 'js', 'scene']
    fields = ['error_visible_mass', 'oracle_visible_mass', 'gained_pp', 'lost_pp', 'net_pp']
    for key, group in grouped(blocks, keys).items():
        expected = 2 if group[0]['family'] in ('spatial_shift', 'false_hotspot', 'false_negative_suppression') else 1
        row = dict(zip(keys, key), valid_directions=len(group), expected_directions=expected,
                   complete=len(group) == expected)
        # Preserve available-only values but exclude incomplete pairs from suite summaries.
        row.update({f:float(np.mean([r[f] for r in group])) for f in fields})
        rows.append(row)
    return rows


def task3(out, records, excluded, context):
    """Read-only frozen suite; saved first-visible times; no planner calls."""
    blocks = []
    oracle = {(r['scene'], r['gamma']):r for r in records if r['dataset'] == 'formal' and r['family'] == 'oracle'}
    for r in records:
        if r['dataset'] != 'formal' or r['family'] == 'oracle':
            continue
        first = context['formal', r['plan_id']][1]
        base = context['formal', oracle[r['scene'], r['gamma']]['plan_id']][1]
        truth = context['formal', r['plan_id']][2]['oracle']
        blocks.append(dict({k:r[k] for k in ['gamma', 'family', 'js', 'scene', 'direction', 'plan_id']},
                           **visibility(first, base, truth)))
    write_csv(out/'suite_visibility_by_direction.csv', blocks)
    scenes = scene_visibility(blocks)
    write_csv(out/'suite_visibility_by_scene.csv', scenes)
    fields = ['error_visible_mass', 'oracle_visible_mass', 'gained_pp', 'lost_pp', 'net_pp']
    complete = [r for r in scenes if r['complete']]
    summary = summarise(complete, ['gamma', 'family', 'js'], fields)
    for row in summary:
        group = [r for r in complete if all(r[k] == row[k] for k in ['gamma', 'family', 'js'])]
        row.update(total_scenes=12, incomplete_or_missing_scenes=12-len(group),
                   positive_scenes=sum(r['net_pp'] > 1e-10 for r in group),
                   negative_scenes=sum(r['net_pp'] < -1e-10 for r in group),
                   zero_scenes=sum(abs(r['net_pp']) <= 1e-10 for r in group),
                   positive_in_all_12=len(group) == 12 and all(r['net_pp'] > 1e-10 for r in group))
    write_csv(out/'suite_visibility_summary.csv', summary)
    # Check against the frozen suite's existing complete-direction aggregation.
    old = pilot_analysis.read_csv(RESULTS/'formal/analysis/primary_scene_contrasts.csv')
    checked = 0
    for r in complete:
        kind = 'baseline_minus_oracle' if r['family'] == 'uniform_prior_baseline' else 'family_minus_oracle'
        matches = [o for o in old if o['scene_id'] == r['scene'] and o['gamma'] == r['gamma']
                   and o['family_left'] == r['family'] and o['family_right'] == 'oracle'
                   and (r['family'] == 'uniform_prior_baseline' or o['target_js_nats'] == r['js'])]
        assert len(matches) == 1, (kind, r)
        np.testing.assert_allclose(r['net_pp'], matches[0]['delta_exact_detection_probability']*100, atol=1e-10)
        checked += 1
    write_json(out/'suite_crosscheck.json', dict(complete_scene_contrasts_checked=checked, passed=True))
    return summary


def gamma_note(out):
    selections = {
        'milestone_4/run_formal.py': [(270, 277)],
        'milestone_4/run_pilot.py': [(146, 154)],
        'milestone_3/planner.py': [(39, 54), (68, 68)],
        'external/time_optimal_ergodic_search/experiments/comparison_study/build_solver.py': [(104, 118)],
        'external/time_optimal_ergodic_search/time_opt_erg_lib/fourier_utils.py': [(12, 24)],
        'external/time_optimal_ergodic_search/time_opt_erg_lib/ergodic_metric.py': [(6, 17)]}
    lines = ['# Gamma call-site audit', '',
             'Gamma is passed unchanged as erg_ub. The constraint is E - gamma <= 0.',
             'The adapter negates inequalities for SciPy (>= 0); this changes sign, not scale.',
             'Workspace mapping uses [0,1] on both axes and is the identity here.',
             'Fourier coefficients are normalized (dt/tf=1/N, prior total mass, hk); gamma itself is not renormalized.', '']
    for name, ranges in selections.items():
        source = (ROOT/name).read_text().splitlines()
        lines += [f'## {name}', '', '```python']
        for start, end in ranges:
            lines += [f'{i}: {source[i-1]}' for i in range(start, min(end, len(source))+1)]
        lines += ['```', '']
    with (out/'gamma_semantics.md').open('x') as f:
        f.write('\n'.join(lines))


def source_audit(out):
    before = read_json(BASE/'source_inventory_before.json')
    changed = [p for p, digest in before.items() if not (ROOT/p).is_file() or sha(ROOT/p) != digest]
    current = {str(p.relative_to(ROOT)) for base in ['milestone_3', 'milestone_4', 'milestone_2', 'external/time_optimal_ergodic_search']
               for p in (ROOT/base).rglob('*') if p.is_file() and BASE not in p.parents and '.git' not in p.parts}
    added = sorted(current-set(before))
    audit = dict(original_files_checked=len(before), changed_original_files=changed,
                 added_files_outside_mechanism=added, planner_imported='milestone_3.planner' in sys.modules,
                 planner_calls=0, passed=not changed and not added and 'milestone_3.planner' not in sys.modules)
    write_json(out/'source_integrity_audit.json', audit)
    assert audit['passed'], audit


def spot_checks(out, context):
    """Independent scalar sums and visibility-layer replay of three formal plans."""
    cases = [('cx0.25_cy0.25_layoutA', .05, 'oracle'),
             ('cx0.75_cy0.75_layoutB', .1, 'diffuse_blur_js_0.15'),
             ('cx0.25_cy0.75_layoutC', .05, 'false_hotspot_x_js_0.1')]
    checks = []
    for scene, gamma, prior in cases:
        plan_id = f'{scene}__gamma_{gamma:g}__{prior}'
        policy, first, data, scene_spec = context['formal', plan_id]
        env = Environment(obstacles=tuple(RectangleObstacle(**o) for o in scene_spec['obstacles']))
        replayed = pilot_analysis.first_seen_grid(policy, env, data['points'], data['oracle'] > 0)
        np.testing.assert_array_equal(first, replayed)
        path = sum(float(np.hypot(*(b[:2]-a[:2]))) for a, b in zip(policy['states'][:-1], policy['states'][1:]))
        mass = sum(float(w) for w, t in zip(data['oracle'], first) if np.isfinite(t))
        stored = read_json(RESULTS/'formal/plans'/plan_id/'exact_metrics.json')
        np.testing.assert_allclose([path, mass], [stored['trajectory_length'], stored['exact_detection_probability']], atol=1e-12)
        # Python cell counts give an independent node-entropy calculation.
        from collections import Counter
        from math import log
        counts = Counter((min(39, int(max(0., x)*40)), min(39, int(max(0., y)*40))) for x, y in policy['states'][:, :2])
        node_entropy = -sum((count/len(policy['states']))*log(count/len(policy['states'])) for count in counts.values())
        np.testing.assert_allclose(node_entropy, entropy(node_occupancy(policy['states'])), atol=1e-12)
        checks.append(dict(plan_id=plan_id, tf_s=float(policy['tf']), scalar_path_length=path,
                           scalar_true_mass=mass, scalar_node_entropy_nats=node_entropy,
                           node_cell_counts=sorted(counts.values()), all_1600_first_seen_times_identical=True))
    write_json(out/'three_plan_spot_checks.json', checks)


def check_plot_pixels(out):
    from PIL import Image
    checks = []
    for prefix, original in [('coverage_maps', 'analysis'), ('trajectories', 'pilot')]:
        for gamma in (.05, .1):
            name = f'{prefix}_gamma_{gamma:g}.png'
            with Image.open(out/'pilot_figures'/name) as a, Image.open(RESULTS/original/name) as b:
                identical = np.array_equal(np.asarray(a), np.asarray(b))
            checks.append(dict(file=name, original=str(RESULTS/original/name), pixels_identical=identical))
            assert identical, name
    write_json(out/'plot_pixel_comparison.json', checks)


def report(out, records, deltas, comparison, suite):
    lines = ['# Stored-plan replay results', '',
             'Constraints for all three tasks: frozen protocol/results and M3 read-only; no planner; no existing result overwrite. New files only under milestone_4/results/mechanism/.', '',
             '546 valid formal plans + 26 valid pilot plans. Six invalid formal plans listed separately. No pilot uniform-prior plan exists; none was created.', '',
             '## Definitions', '',
             '- Grid: the stored 40×40 unit-workspace prior quadrature grid. The ergodic metric is spectral, not a grid-occupancy entropy.',
             '- entropy_node_nats: Shannon entropy of the cell histogram of all 50 stored positions with weight 1/50 (the solver ck quadrature). Maximum is ln(50), not ln(1600). No support remasking of camera positions.',
             '- entropy_dwell_tf_nats: exact piecewise-linear cell residence over [0,tf], including the last tf/N terminal hold.',
             '- entropy_dwell_15s_nats: same residence over [0,15], including terminal hold or truncation. These are separate diagnostic definitions, not interchangeable estimates.',
             '- Boundary cells: half-open intervals, x/y=1 assigned to the last cell; solver excursions <=1e-6 are clipped. Zero-probability cells contribute zero to entropy.',
             '- path_length: sum of distances between stored nodes, unit-workspace length. camera_path_length_15s also reports execution truncated at the observation budget.',
             '- visible_area: supported grid cells observed by 15 s × 1/1600, in unit-workspace area. This is camera-visible area, not the area occupied by the camera center. visible_free_fraction has supported area as denominator.',
             '- ergodic_metric: NumPy replay of the source 8×8 cosine basis and spectral weights; every value checked against stored diagnostics to absolute tolerance 1e-9.',
             '- Task 1 summaries weight valid plans equally, separating pilot/formal. Deltas are family minus oracle paired by scene and gamma; JS/direction details are retained. Missing invalid runs are excluded with counts.',
             '- Task 3 averages x/y within scene before giving scenes equal weight. Incomplete x/y sets remain in per-scene output, explicitly flagged, and are excluded from across-scene summaries, matching the frozen analysis.',
             '- Positive/negative net counts use tolerance 1e-10 pp. Masses are probabilities; gained/lost/net are percentage points.', '',
             '## Task 1: paired mean deltas (family minus oracle)', '',
             '| Dataset | gamma | Family | n plans | Δtf s | ΔH nodes nats | ΔH dwell tf nats |',
             '|---|---:|---|---:|---:|---:|---:|']
    for d in deltas:
        lines.append(f"| {d['dataset']} | {d['gamma']} | {d['family']} | {d['n']} | {d['delta_tf_s_mean']:+.6f} | {d['delta_entropy_node_nats_mean']:+.6f} | {d['delta_entropy_dwell_tf_nats_mean']:+.6f} |")
    lines += ['', 'Means/medians of every requested diagnostic: diagnostic_summary.csv; JS strata: diagnostic_summary_by_js.csv. Gamma source quotes: gamma_semantics.md.', '',
              '## Task 2: pilot net visibility differences', '',
              'Both gamma=0.05 figures already existed in the original directories. Both gamma slices were rendered in pilot_figures using the original plotting function / unchanged coverage-map statements. Colour meanings, annotations and styling are unchanged.', '',
              '| Family | JS | gained .10 pp | lost .10 pp | net .10 pp | gained .05 pp | lost .05 pp | net .05 pp |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison:
        vals = [r[f'{f}_gamma_{g:g}'] for g in (.1, .05) for f in ['gained_pp', 'lost_pp', 'net_pp']]
        lines.append(f"| {r['family']} | {r['js']} | " + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
    lines += ['', '## Task 3: complete-direction scene results', '',
              '| gamma | Family | JS | complete / 12 | positive | negative | mean net pp | median net pp | positive all 12 |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for r in suite:
        lines.append(f"| {r['gamma']} | {r['family']} | {r['js']} | {r['n']}/12 | {r['positive_scenes']} | {r['negative_scenes']} | {r['net_pp_mean']:+.4f} | {r['net_pp_median']:+.4f} | {r['positive_in_all_12']} |")
    all_positive = all(r['positive_in_all_12'] for r in suite if r['family'] in FAMILIES and r['gamma'] == .1)
    lines += ['', f'Flag: the gamma=0.10 pilot pattern of positive net mass for every error family/JS holds across all 12 scenes: **{all_positive}**.',
              'This flag concerns exact true-prior mass, not sampled target success percentages. No causal or hypothesis interpretation is supplied.', '',
              '## Reproduction', '',
              'Run from repository root with PYTHONDONTWRITEBYTECODE=1 and /opt/miniconda3/envs/erg/bin/python -B milestone_4/results/mechanism/replay_analysis.py. Each run creates a new timestamped directory and refuses to reuse an existing directory.',
              'Unit tests: /opt/miniconda3/envs/erg/bin/python -B -m unittest discover -s milestone_4/results/mechanism -p test_replay.py -v.', '']
    with (out/'REPORT.md').open('x') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    out = (BASE/args.run_name).resolve()
    if out.parent != BASE:
        raise ValueError('Output must be a new direct child of mechanism')
    out.mkdir(exist_ok=False)
    print('Output:', out, flush=True)
    print('Loading and verifying stored plans', flush=True)
    records, excluded, context = load_all()
    print('Task 1: diagnostics', flush=True)
    deltas = task1(out, records, excluded)
    gamma_note(out)
    print('Task 2: original pilot plots and gamma comparison', flush=True)
    comparison = task2(out, records, context)
    print('Task 3: suite visibility statistics', flush=True)
    suite = task3(out, records, excluded, context)
    print('Verifying three formal visibility replays and four rendered PNGs', flush=True)
    spot_checks(out, context)
    check_plot_pixels(out)
    report(out, records, deltas, comparison, suite)
    source_audit(out)
    write_json(out/'run_manifest.json', dict(script_sha256=sha(__file__), numpy=np.__version__,
                                           valid_formal=546, valid_pilot=26, invalid_formal=6,
                                           max_metric_replay_abs_difference=max(r['ergodic_metric_abs_difference'] for r in records)))
    print('Completed:', out, flush=True)


if __name__ == '__main__':
    main()
