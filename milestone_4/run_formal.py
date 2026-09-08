"""Execute exactly the frozen v2 suite, once per plan, with durable resume.

The coordinator launches one fresh solver process per manifest condition to bound
JAX compilation memory. This changes no planner inputs or solver options. An
attempt marker is durable before launch; a lost attempt is never retried. A
saved solver result can be evaluated after interruption without solving again.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np

from milestone_3.experiment import Environment, evaluate, js_divergence
from visibility import RectangleObstacle
from .analyze_pilot import entropy_nats, first_seen_grid, trajectory_length, weighted_detection_stats
from .protocol import ROOT, protocol_manifest
from .scene_manifest import candidate_manifest, scene_grid, gaussian_density

DEFAULT_MANIFEST = ROOT / 'milestone_4/results/frozen/PROTOCOL_V2_FROZEN_MANIFEST.json'
DEFAULT_OUTPUT = ROOT / 'milestone_4/results/formal'
EPISODE_FIELDS = ['seed', 'episode', 'target_x', 'target_y', 'success', 't_find', 'capped_time', 'path_length_until_stop']
METRIC_FIELDS = ['exact_detection_probability', 'exact_weighted_mean_t_find', 'exact_weighted_median_t_find',
                 'uniform_detection_probability', 'uniform_weighted_mean_t_find', 'uniform_weighted_median_t_find',
                 'visible_free_fraction', 'oracle_probability_mass_covered', 'trajectory_length']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def claim_worker(directory):
    """Durably consume the sole solver invocation, including failed invocations.

    Exclusive creation also prevents two manually launched workers from solving
    the same condition. An incomplete claim after a crash still forbids retry.
    """
    with (directory/'worker_started.json').open('x') as handle:
        json.dump(dict(started_at_utc=now(), pid=os.getpid(), attempt_count=1), handle)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_attempt(directory, label, manifest_hash, prior_hash):
    saved = read_json(directory/'started.json')
    require(all(saved.get(k) == v for k, v in label.items()), 'Attempt condition differs from manifest')
    require(saved.get('attempt_count') == 1, 'Attempt count differs from frozen no-retry rule')
    require(saved.get('frozen_manifest_sha256') == manifest_hash, 'Attempt manifest hash changed')
    require(saved.get('prior_sha256') == prior_hash, 'Attempt prior hash changed')


def atomic_npz(path, **arrays):
    import io
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


def atomic_csv(path, rows, fields=EPISODE_FIELDS):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def require(value, message):
    if not value:
        raise ValueError(message)


def environment_for(scene):
    return Environment(obstacles=tuple(RectangleObstacle(**o) for o in scene['obstacles']))


def expand_plans(manifest):
    return [dict(scene_id=s['scene_id'], gamma=g, **c,
                 plan_id=f"{s['scene_id']}__gamma_{g:g}__{c['prior_id']}")
            for s in manifest['candidate_manifest']['scenes']
            for g in manifest['planner_and_evaluation_invariants']['planner']['gammas']
            for c in s['conditions']]


def load_frozen(manifest_path=DEFAULT_MANIFEST):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    require(manifest['protocol_id'] == 'm4-controlled-prior-v2-frozen', 'Not frozen Protocol v2')
    base = protocol_manifest()  # Checks the actual frozen M3 source bytes and upstream commit.
    for path, expected in manifest['preflight_source_sha256'].items():
        require(sha(ROOT / path) == expected, f'Frozen source hash changed: {path}')
    # These reused metric/protocol helpers were present before freeze, but not
    # individually included in the manifest source map. Guard their freeze bytes.
    for path in ('milestone_4/analyze_pilot.py', 'milestone_4/protocol.py', 'milestone_4/prior_errors.py', 'milestone_4/calibrate_priors.py'):
        expected = subprocess.check_output(['git', 'show', f"{manifest['frozen_at_git_commit']}:{path}"], cwd=ROOT)
        require((ROOT / path).read_bytes() == expected, f'Frozen helper changed: {path}')
    require(json.loads(json.dumps(candidate_manifest())) == manifest['candidate_manifest'], 'Candidate design changed')
    inv = manifest['planner_and_evaluation_invariants']
    require(base['planner'] == inv['planner'], 'Planner settings changed')
    env = Environment()
    require((env.camera.sensing_radius, env.camera.fov_degrees, env.budget, env.observation_dt) ==
            (inv['camera_range'], inv['camera_fov_degrees'], inv['budget_seconds'], inv['observation_dt_seconds']),
            'Evaluation settings changed')
    require(inv['scan_rate_radians_per_s'] == 'pi/2' and env.scan_rate == np.pi/2, 'Scan changed')
    sampling = manifest['final_evaluation_sampling']
    require(sampling['seed_count'] == len(sampling['seeds']) == len(set(sampling['seeds'])), 'Sampling mismatch')
    require(sampling['total_draws_per_scene_per_condition'] == sampling['seed_count'] * sampling['targets_per_seed'], 'Draw count mismatch')
    expected_versions = json.loads(subprocess.check_output(
        ['git', 'show', f"{manifest['frozen_at_git_commit']}:milestone_4/results/calibration/protocol.json"], cwd=ROOT))['versions']
    require(base['versions'] == expected_versions, f'Dependency versions differ from frozen environment: {base["versions"]}')
    inputs = {}
    for scene in manifest['candidate_manifest']['scenes']:
        sid = scene['scene_id']
        with np.load(manifest_path.parent / 'priors' / f'{sid}.npz', allow_pickle=False) as npz:
            data = {k: npz[k].copy() for k in npz.files}
        points, support = scene_grid(environment_for(scene), manifest['candidate_manifest']['grid_size'])
        require(np.array_equal(data['points'], points) and np.array_equal(data['free_mask'], support), f'Grid/support mismatch: {sid}')
        expected = manifest['prior_sha256_by_scene'][sid]
        require(set(data) == set(expected) | {'points', 'free_mask'}, f'Prior keys mismatch: {sid}')
        require(set(expected) == {c['prior_id'] for c in scene['conditions']}, f'Condition keys mismatch: {sid}')
        truth = gaussian_density(points, support, scene['center'], manifest['candidate_manifest']['sigma'], manifest['candidate_manifest']['background'])
        require(np.array_equal(truth, data['oracle']), f'Oracle mismatch: {sid}')
        for c in scene['conditions']:
            p = data[c['prior_id']]
            require(array_sha(p) == expected[c['prior_id']], f'Prior hash mismatch: {sid}/{c["prior_id"]}')
            require(np.isfinite(p).all() and (p >= 0).all() and abs(p.sum()-1) < 1e-12 and (p[~support] == 0).all(), f'Invalid frozen prior: {sid}')
            if c['target_js_nats'] is not None:
                require(abs(js_divergence(truth, p)-c['target_js_nats']) <= inv['js_tolerance'], f'JS mismatch: {sid}/{c["prior_id"]}')
        for seed in sampling['seeds']:
            indices = np.random.default_rng(seed).choice(len(points), size=sampling['targets_per_seed'], p=truth)
            data[f'target_indices_seed_{seed}'] = indices
            data[f'targets_seed_{seed}'] = points[indices]
        inputs[sid] = data
    plans = expand_plans(manifest)
    require(len(plans) == len({p['plan_id'] for p in plans}) == manifest['candidate_manifest']['projected_plan_count'] == 552, 'Plan count mismatch')
    return manifest, inputs, base


def provenance(manifest_path, manifest, base):
    source_paths = set(base['source_sha256']) | set(manifest['preflight_source_sha256']) | {
        'milestone_4/analyze_pilot.py', 'milestone_4/protocol.py', 'milestone_4/run_formal.py',
        'milestone_4/prior_errors.py', 'milestone_4/calibrate_priors.py'}
    upstream = ROOT / 'external/time_optimal_ergodic_search'
    source_paths.update(str(p.relative_to(ROOT)) for p in upstream.rglob('*.py'))
    artifacts = [manifest_path, *sorted((manifest_path.parent/'priors').glob('*.npz'))]
    return dict(source_sha256={p: sha(ROOT/p) for p in sorted(source_paths)},
                frozen_artifact_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in artifacts},
                versions=base['versions'], upstream_commit=base['upstream_commit'])


def verify_hashes(directory, hashes):
    for path, digest in hashes.items():
        require(sha(Path(directory)/path) == digest, f'Checkpoint artifact changed: {path}')


def write_evaluation(directory, label, data, scene, manifest, policy, status):
    valid = policy is not None and policy['diagnostics']['valid']
    metrics = dict.fromkeys(METRIC_FIELDS)
    metrics['prior_entropy_nats'] = entropy_nats(data[label['prior_id']])
    episodes, sampled = [], []
    if valid:
        first = first_seen_grid(policy, environment_for(scene), data['points'], data['free_mask'])
        atomic_npz(directory/'first_seen_grid.npz', first_seen=first)
        for prefix, weights in [('exact', data['oracle']), ('uniform', data['free_mask']/data['free_mask'].sum())]:
            metrics.update({f'{prefix}_{k}': v for k, v in weighted_detection_stats(first, weights).items()})
        metrics.update(trajectory_length=trajectory_length(policy['states']),
                       visible_free_fraction=metrics['uniform_detection_probability'],
                       oracle_probability_mass_covered=metrics['exact_detection_probability'])
    for seed in manifest['final_evaluation_sampling']['seeds']:
        if valid:
            rows, summary = evaluate(policy, environment_for(scene), data[f'targets_seed_{seed}'])
            episodes.extend(dict(seed=seed, **r) for r in rows)
            count = sum(r['success'] for r in rows)
            summary.update(success_count=count, timeout_count=len(rows)-count, timeout_rate=1-summary['success_rate'])
        else:
            summary = dict(status='invalid_plan', episodes=0, success_count=None, timeout_count=None,
                           success_rate=None, timeout_rate=None, mean_t_find_success_only=None,
                           median_t_find_success_only=None, mean_capped_time=None, median_capped_time=None,
                           mean_path_length_until_stop=None)
        sampled.append(dict(seed=seed, planned_episodes=manifest['final_evaluation_sampling']['targets_per_seed'], **summary))
    atomic_json(directory/'exact_metrics.json', metrics)
    atomic_json(directory/'sampled_metrics.json', sampled)
    atomic_csv(directory/'episodes.csv', episodes)
    artifacts = {p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name not in ('result.json',) and not p.name.endswith('.tmp')}
    result = dict(label, actual_js_nats=js_divergence(data['oracle'], data[label['prior_id']]),
                  status=status, valid=bool(valid), attempt_count=1, completed_at_utc=now(),
                  artifact_sha256=artifacts)
    atomic_json(directory/'result.json', result)
    return result


def save_solver_result(directory, policy):
    atomic_npz(directory/'trajectory.npz', states=policy['states'], controls=policy['controls'], tf=policy['tf'])
    # JSON has no NaN/Inf; the raw trajectory preserves any nonfinite solver values.
    diag = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in policy['diagnostics'].items()}
    atomic_json(directory/'diagnostics.json', diag)
    atomic_json(directory/'solver_result.json', dict(diagnostics=diag, trajectory_sha256=sha(directory/'trajectory.npz'), saved_at_utc=now()))


def load_solver_result(directory):
    saved = read_json(directory/'solver_result.json')
    require(sha(directory/'trajectory.npz') == saved['trajectory_sha256'], 'Saved solver trajectory changed')
    require(read_json(directory/'diagnostics.json') == saved['diagnostics'], 'Saved diagnostics changed')
    with np.load(directory/'trajectory.npz', allow_pickle=False) as p:
        return dict(states=p['states'], controls=p['controls'], tf=float(p['tf']), diagnostics=saved['diagnostics'])


def worker(output, plan_id):
    config = read_json(output/'config.json')
    manifest_path = Path(config['frozen_manifest_path'])
    require(sha(manifest_path) == config['frozen_manifest_sha256'], 'Worker manifest changed')
    require(read_json(manifest_path) == config['frozen_manifest'], 'Worker embedded manifest changed')
    label = next(p for p in config['expected_plans'] if p['plan_id'] == plan_id)
    directory = output/'plans'/plan_id
    require((directory/'started.json').exists() and not any((directory/name).exists() for name in
            ('solver_result.json', 'result.json', 'exception.json', 'worker_started.json')),
            'Worker requires a fresh marked attempt; retry is forbidden')
    expected_prior_hash = config['frozen_manifest']['prior_sha256_by_scene'][label['scene_id']][label['prior_id']]
    validate_attempt(directory, label, config['frozen_manifest_sha256'], expected_prior_hash)
    claim_worker(directory)
    try:
        from milestone_3.planner import plan
        scene = next(s for s in config['frozen_manifest']['candidate_manifest']['scenes'] if s['scene_id'] == label['scene_id'])
        settings = config['frozen_manifest']['planner_and_evaluation_invariants']['planner']
        with np.load(output/'inputs'/f"{label['scene_id']}.npz", allow_pickle=False) as data:
            require(array_sha(data[label['prior_id']]) == expected_prior_hash, 'Worker input prior changed')
            policy = plan(data['points'], data[label['prior_id']], label['gamma'], environment_for(scene), settings['nodes'], settings['maxiter'])
        save_solver_result(directory, policy)
    except Exception as e:
        atomic_json(directory/'exception.json', dict(type=type(e).__name__, message=str(e), traceback=traceback.format_exc(), at_utc=now()))
        raise


def complete_attempt(directory, label, data, scene, manifest):
    if (directory/'result.json').exists():
        result = read_json(directory/'result.json')
        require(all(result.get(k) == v for k, v in label.items()), 'Checkpoint condition differs from manifest')
        verify_hashes(directory, result['artifact_sha256'])
        return result
    if (directory/'solver_result.json').exists():
        policy = load_solver_result(directory)
        return write_evaluation(directory, label, data, scene, manifest, policy, 'valid' if policy['diagnostics']['valid'] else 'invalid')
    status = 'exception' if (directory/'exception.json').exists() else 'interrupted'
    atomic_json(directory/'diagnostics.json', dict(valid=False, solver_success=False, status=status,
        solver_message='Attempt ended without a durable solver result; no retry or search metric imputation.', trajectory_available=(directory/'trajectory.npz').exists()))
    return write_evaluation(directory, label, data, scene, manifest, None, status)


def run(manifest_path, output):
    manifest, inputs, base = load_frozen(manifest_path)
    prov = provenance(manifest_path, manifest, base)
    output.mkdir(parents=True, exist_ok=True)
    # Inherited by a worker so an orphan still prevents concurrent resume.
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output/'config.json'
        if config_path.exists():
            config = read_json(config_path)
            require(config['frozen_manifest_sha256'] == sha(manifest_path), 'Resume manifest changed')
            require(config['frozen_manifest'] == manifest, 'Resume embedded manifest changed')
            for k, v in prov.items():
                require(config[k] == v, f'Resume provenance changed: {k}')
            require(config['expected_plans'] == expand_plans(manifest), 'Resume plan list changed')
        else:
            require(not (output/'plans').exists(), 'Unrecognized existing plans; refuse overwrite')
            config = dict(frozen_manifest_path=str(manifest_path), frozen_manifest_sha256=sha(manifest_path),
                          frozen_manifest=manifest, expected_plans=expand_plans(manifest), **prov,
                          started_at_utc=now(), python=sys.version, executable=sys.executable, platform=platform.platform(),
                          git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                          git_status_before=subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True),
                          execution='Serial fresh worker per condition; frozen planner; one attempt; no warm start or retry')
            atomic_json(config_path, config)
        for sid, data in inputs.items():
            path = output/'inputs'/f'{sid}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    require(set(saved.files) == set(data) and all(np.array_equal(saved[k], v) for k,v in data.items()), f'Saved inputs changed: {sid}')
            else:
                atomic_npz(path, **data)
        results = []
        started = perf_counter()
        for i, label in enumerate(config['expected_plans'], 1):
            directory = output/'plans'/label['plan_id']
            scene = next(s for s in manifest['candidate_manifest']['scenes'] if s['scene_id'] == label['scene_id'])
            if not (directory/'started.json').exists():
                require(not directory.exists(), f'Unrecognized attempt directory: {directory}')
                atomic_json(directory/'started.json', dict(label, started_at_utc=now(), attempt_count=1,
                            frozen_manifest_sha256=config['frozen_manifest_sha256'], prior_sha256=array_sha(inputs[label['scene_id']][label['prior_id']])))
                print(f'[{i}/552] START {label["plan_id"]}', flush=True)
                with (directory/'solver.log').open('wb') as log:
                    process = subprocess.run([sys.executable, '-m', 'milestone_4.run_formal', '--output', str(output), '--worker', label['plan_id']],
                                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                atomic_json(directory/'worker_exit.json', dict(returncode=process.returncode, ended_at_utc=now()))
            validate_attempt(directory, label, config['frozen_manifest_sha256'],
                             array_sha(inputs[label['scene_id']][label['prior_id']]))
            result = complete_attempt(directory, label, inputs[label['scene_id']], scene, manifest)
            results.append(result)
            counts = Counter(r['status'] for r in results)
            atomic_json(output/'progress.json', dict(updated_at_utc=now(), expected_plans=552, completed_plans=len(results),
                        counts=dict(counts), current_plan_id=label['plan_id'], session_elapsed_seconds=perf_counter()-started))
            print(f'[{i}/552] {result["status"].upper()} counts={dict(counts)} elapsed={perf_counter()-started:.1f}s', flush=True)
        _, _, base_after = load_frozen(manifest_path)
        post = provenance(manifest_path, manifest, base_after)
        require(post == prov, 'Source or frozen inputs changed during execution')
        atomic_json(output/'postrun_provenance.json', dict(**post, frozen_manifest_sha256=sha(manifest_path), completed_at_utc=now()))
        print('All 552 attempts recorded. Run audit_formal, then analyze_formal.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--worker', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args.output.resolve(), args.worker)
    elif args.validate_only:
        manifest, _, _ = load_frozen(args.manifest)
        print(f'Frozen inputs verified: {len(expand_plans(manifest))} plans; zero planner calls.')
    else:
        output = args.output.resolve()
        require(output.is_relative_to(ROOT/'milestone_4/results') and output.name not in ('frozen','pilot','preflight','analysis','calibration'), 'Use a separate formal results directory')
        run(args.manifest.resolve(), output)


if __name__ == '__main__':
    main()
