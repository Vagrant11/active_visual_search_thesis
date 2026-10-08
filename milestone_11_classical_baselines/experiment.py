"""M11：仅用已存 val 数据的传统搜索基线，不调用优化器或训练模型。"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(HERE / '.cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(HERE / '.cache')
os.environ['MPLBACKEND'] = 'Agg'

import ast
import copy
import csv
import json
import time
from dataclasses import replace, asdict
import numpy as np
from scipy.sparse.csgraph import shortest_path
from milestone_3.experiment import Environment, camera_trajectory
from visibility import CameraModel, RectangleObstacle, EPS, is_visible, segment_intersects_rect

M9 = ROOT / 'milestone_9_diffusion_obstacle'
GUIDE = M9 / 'guidance'
CAP = 120.0
MARGIN = 1e-4
POLYGON_N = 64
ARRIVAL_OFFSET = 1e-6


def extract(path, name, scope):
    # 执行原函数的 AST，避免旧模块顶层设置缓存目录或启动其他流程。
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


SCOPE = dict(np=np, EPS=EPS, is_visible=is_visible, camera_trajectory=camera_trajectory)
HARNESS = ROOT / 'milestone_4/analyze_pilot.py'
first_seen_grid = extract(HARNESS, 'first_seen_grid', SCOPE)
weighted_detection_stats = extract(HARNESS, 'weighted_detection_stats', SCOPE)
geometry_audit = extract(ROOT / 'milestone_5_init_sensitivity/common.py', 'geometry_audit',
                         dict(np=np, segment_intersects_rect=segment_intersects_rect))
geometry_metrics = extract(ROOT / 'milestone_7_diffusion_pilot/pilot.py', 'geometry_metrics', dict(np=np))


def make_observer():
    # 在线后验更新原样复用 first_seen_grid 的单次观测循环体，不另写可见性模型。
    original = next(n for n in ast.parse(HARNESS.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'first_seen_grid')
    loop = next(n for n in original.body if isinstance(n, ast.For))
    template = ast.parse('def observe(first,t,x,y,theta,environment,points,support):\n pass').body[0]
    template.body = copy.deepcopy(loop.body)
    scope = dict(SCOPE)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[template], type_ignores=[])), str(HARNESS), 'exec'), scope)
    return scope['observe']


observe = make_observer()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def json_write(path, value):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def inventory():
    # 仅 stat 旧文件，不打开 train/test 结果的内容。
    return {str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns]
            for directory in list(ROOT.glob('milestone_*')) + [ROOT / 'external']
            if directory.is_dir() and directory != HERE
            for p in directory.rglob('*') if p.is_file()}


def actual_limits():
    # 从实际约束表达式读取数值；表达式改变则断言失败，避免悄悄沿用旧限制。
    tree = ast.parse((ROOT / 'milestone_3/planner.py').read_text())
    speed = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)
             and ast.unparse(n.left) == 'jnp.sum(x[:, 2:] ** 2, axis=1)']
    tree = ast.parse((ROOT / 'external/time_optimal_ergodic_search/experiments/comparison_study/build_solver.py').read_text())
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_erg_time_opt_solver')
    control = [n for n in ast.walk(builder) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)
               and ast.unparse(n.left) == 'np.abs(u)']
    assert len(speed) == len(control) == 1
    return np.sqrt(ast.literal_eval(speed[0].right)), ast.literal_eval(control[0].right)


class Router:
    """安全圆外切多边形的可见图；所有候选连线检查完整线段。"""
    def __init__(self, environment, points):
        assert len(environment.collision_disks) == 1
        self.center = environment.collision_disks[0, :2]
        self.radius = environment.collision_disks[0, 2]
        angle = np.arange(POLYGON_N) * 2 * np.pi / POLYGON_N
        radius = (self.radius + MARGIN) / np.cos(np.pi / POLYGON_N)
        self.vertices = self.center + radius * np.column_stack((np.cos(angle), np.sin(angle)))
        graph = self.connections(self.vertices, self.vertices)
        np.fill_diagonal(graph, 0)
        self.dist, self.pred = shortest_path(graph, directed=False, return_predecessors=True)
        self.points = points
        self.to_grid = self.connections(self.vertices, points)
        self.via_grid = np.min(self.dist[:, :, None] + self.to_grid[None, :, :], axis=1)

    def connections(self, a, b):
        a = np.atleast_2d(a)[:, None, :]
        b = np.atleast_2d(b)[None, :, :]
        delta = b - a
        squared = np.sum(delta * delta, axis=-1)
        fraction = np.clip(np.sum((self.center-a)*delta, axis=-1) / np.maximum(squared, 1e-30), 0, 1)
        closest = a + fraction[..., None] * delta
        clear = np.sum((closest-self.center)**2, axis=-1) >= (self.radius+MARGIN)**2 - 1e-14
        return np.where(clear, np.sqrt(squared), np.inf)

    def distances(self, p):
        direct = self.connections(p, self.points)[0]
        to_vertices = self.connections(p, self.vertices)[0]
        return np.minimum(direct, np.min(to_vertices[:, None] + self.via_grid, axis=0))

    def route(self, p, q):
        if np.isfinite(self.connections(p, q)[0, 0]): return [np.asarray(q)]
        source = self.connections(p, self.vertices)[0]
        target = self.connections(self.vertices, q)[:, 0]
        costs = source[:, None] + self.dist + target[None, :]
        i, j = np.unravel_index(np.argmin(costs), costs.shape)
        assert np.isfinite(costs[i, j])
        path = [j]
        while path[-1] != i:
            path.append(int(self.pred[i, path[-1]]))
            assert path[-1] >= 0
        return [self.vertices[k].copy() for k in reversed(path)] + [np.asarray(q)]


def velocity_profile(length, direction, h, vmax, umax, initial_speed=0.):
    # 在每个方向使用实际分量加速度上限，寻找最少离散时间步的梯形/三角形包络。
    # v0固定；末速度为0；Euler 位移为 h*sum(v[:-1])。
    a = umax / np.max(np.abs(direction))
    n = max(1, int(np.ceil(initial_speed/(a*h))))
    while True:
        k = np.arange(n+1)
        lower = np.maximum(initial_speed-a*h*k, 0)
        upper = np.minimum(np.minimum(initial_speed+a*h*k, vmax), a*h*(n-k))
        if upper[0] >= initial_speed-1e-12 and h*upper[:-1].sum() >= length-1e-14:
            break
        n += 1
    d0, d1 = h*lower[:-1].sum(), h*upper[:-1].sum()
    assert length >= d0-1e-10, (length, d0)
    mix = np.clip((length-d0)/max(d1-d0, 1e-30), 0, 1)
    velocity = lower + mix*(upper-lower)
    velocity[0] = initial_speed
    velocity[-1] = 0
    assert abs(h*velocity[:-1].sum()-length) < 1e-9
    return velocity


class Search:
    def __init__(self, environment, points, support, initial, router):
        self.env, self.points, self.support, self.router = environment, points, support, router
        self.h = environment.observation_dt
        self.vmax, self.umax = actual_limits()
        self.states = [initial.copy()]
        self.first = np.full(len(points), np.inf)
        self.events = []
        self.decisions = 0
        self.observe()

    @property
    def t(self): return (len(self.states)-1)*self.h

    @property
    def p(self): return self.states[-1][:2]

    @property
    def v(self): return self.states[-1][2:]

    def observe(self):
        theta = (self.env.scan_rate*self.t+np.pi) % (2*np.pi)-np.pi
        observe(self.first, self.t, *self.p, theta, self.env, self.points, self.support)

    def step(self, next_velocity):
        # 与 planner 相同：x[k+1]=x[k]+h*v[k]，u[k]=(v[k+1]-v[k])/h。
        self.states.append(np.r_[self.p+self.h*self.v, next_velocity])
        self.observe()

    def brake(self):
        speed = np.linalg.norm(self.v)
        if speed < 1e-12: return
        direction = self.v / speed
        a = self.umax / np.max(np.abs(direction))
        while speed > 1e-12:
            speed = max(0., speed-a*self.h)
            self.step(speed*direction)

    def leg(self, q, vmax, target=None, weights=None, ratio=False):
        delta = q-self.p
        length = np.linalg.norm(delta)
        if length < 1e-10: return False
        direction = delta/length
        speed = np.linalg.norm(self.v)
        assert speed < 1e-10 or np.linalg.norm(self.v/speed-direction) < 1e-8
        profile = velocity_profile(length, direction, self.h, vmax, self.umax, speed)
        for vnext in profile[1:]:
            self.step(vnext*direction)
            if self.t >= CAP:
                self.brake()  # 仅保存安全制动尾段；评测仍截在120s，避免瞬停假象。
                return True
            if target is not None and np.isfinite(self.first[target]):
                self.events.append(dict(time=self.t, event='target_observed', target=int(target)))
                # 提前观测后立即重选；若下一目标仍沿当前直线且有制动余量，保持运动。
                candidate = self.choose(weights, ratio)
                if candidate is not None and np.linalg.norm(self.v) > 1e-10:
                    route = self.router.route(self.p, self.points[candidate] + np.array([ARRIVAL_OFFSET, 0.]))
                    next_delta = route[0]-self.p
                    next_length = np.linalg.norm(next_delta)
                    a = self.umax/np.max(np.abs(direction))
                    speeds = np.maximum(np.linalg.norm(self.v)-a*self.h*np.arange(1+int(np.ceil(np.linalg.norm(self.v)/(a*self.h)))), 0)
                    stopping = self.h*speeds.sum()
                    if next_length >= stopping+1e-10 and np.linalg.norm(next_delta/next_length-direction) < 1e-8:
                        self.leg(route[0], vmax, candidate, weights, ratio)
                        return True
                self.brake()
                return True
        assert np.linalg.norm(self.p-q) < 1e-8
        return False

    def travel(self, q, vmax, target=None, weights=None, ratio=False):
        for corner in self.router.route(self.p, q):
            if self.leg(corner, vmax, target, weights, ratio): return

    def wait_scan(self, target=None):
        assert np.linalg.norm(self.v) < 1e-10
        steps = int(np.ceil((2*np.pi/self.env.scan_rate)/self.h))
        for _ in range(steps):
            if self.t >= CAP or (target is not None and np.isfinite(self.first[target])): break
            self.step(np.zeros(2))

    def choose(self, weights, ratio):
        self.decisions += 1
        remaining = weights.copy()
        remaining[np.isfinite(self.first) | ~self.support] = 0
        mass = remaining.sum()
        if mass <= 0: return None
        remaining /= mass
        if ratio:
            distances = self.router.distances(self.p)
            score = remaining / np.maximum(distances, 1e-12)
        else: score = remaining
        return int(np.argmax(score))

    def greedy(self, weights, ratio):
        while self.t < CAP:
            target = self.choose(weights, ratio)
            if target is None: break
            self.events.append(dict(time=self.t, event='select_target', target=target))
            # 相机与目标精确共点时方位未定义；停靠点偏移1e-6（远小于网格间距），
            # 是否观察到仍完全由原harness决定，不把“到达”直接当成“发现”。
            q = self.points[target] + np.array([ARRIVAL_OFFSET, 0.])
            if np.linalg.norm(q-self.p) < 1e-8:
                # 已到最高分格子，但相机尚未朝向它：原地等待首次可见。
                self.wait_scan(target)
            else:
                self.travel(q, self.vmax, target, weights, ratio)

    def lawn(self, conservative):
        R = self.env.camera.sensing_radius
        nrows = int(np.ceil(1/R))
        spacing = 1/nrows
        period = 2*np.pi/self.env.scan_rate
        speed = min(self.vmax, .9*2*np.sqrt(R**2-(spacing/2)**2)/period) if conservative else self.vmax
        for row in range(nrows):
            y = (row+.5)*spacing
            for x in ((0., 1.) if row % 2 == 0 else (1., 0.)):
                if self.t >= CAP: return
                self.travel(np.array([x, y]), speed)
                if conservative: self.wait_scan()
        # 同一固定蛇形路线之后，几何漏扫补全；不读取φ，按避障最短路程选择。
        self.events.append(dict(time=self.t, event='snake_complete'))
        while self.t < CAP and np.any(self.support & ~np.isfinite(self.first)):
            distances = self.router.distances(self.p)
            distances[~self.support | np.isfinite(self.first)] = np.inf
            target = int(np.argmin(distances))
            self.events.append(dict(time=self.t, event='geometry_cleanup', target=target))
            self.travel(self.points[target] + np.array([ARRIVAL_OFFSET, 0.]), speed)
            self.wait_scan(target)

    def finish(self):
        assert np.linalg.norm(self.v) < 1e-9
        states = np.asarray(self.states)
        controls = np.vstack((np.diff(states[:, 2:], axis=0)/self.h, np.zeros(2)))
        return states, controls, len(states)*self.h


def audit(states, tf, environment, initial):
    xy = states[:, :2]
    # 本实验不指定终点，保留零末速检查；起点与初速度仍须满足原设置。
    terminal = np.r_[xy[-1], 0., 0.]
    metrics = geometry_metrics(xy, tf, initial, terminal)
    geom = geometry_audit(xy, environment, np.array([[0., 1.], [0., 1.]]))
    radius = environment.collision_disks[0, 2]
    clearance = geom['min_collision_disk_clearance']
    violation = max(0., radius**2-(radius+clearance)**2)
    metrics.update(rectangle_collision=geom['intersects_rectangle'], min_circle_clearance=clearance,
                   safety_circle_violation_squared=violation, circle_fail=violation > 1e-6)
    metrics['physical_pass'] = bool(not (metrics['out_of_bounds_tolerance'] or metrics['endpoint_fail']
                or metrics['dynamics_fail'] or geom['intersects_rectangle'] or metrics['circle_fail']))
    return metrics


def replay(states, tf, environment, points, support, horizon):
    return first_seen_grid(dict(states=states, tf=tf), replace(environment, budget=float(horizon)), points, support)


def compute_metrics(first, weights, reference):
    stats = weighted_detection_stats(first, weights)
    seen = np.flatnonzero(np.isfinite(first))
    order = seen[np.argsort(first[seen])]
    cumulative = np.cumsum(weights[order])
    reached = np.flatnonzero(cumulative >= reference-1e-12)
    match_time = float(first[order[reached[0]]]) if len(reached) else None
    return stats, match_time


def main():
    started = time.perf_counter()
    assert not (HERE/'run_started.json').exists(), '结果已存在；禁止覆盖/自动重跑。'
    before = inventory()
    json_write(HERE/'protected_before.json', before)
    json_write(HERE/'run_started.json', dict(unix_time=time.time(), pid=os.getpid()))
    data = load(GUIDE/'val_inputs.npz')
    assert set(data['split']) == {'val'} and data['valid'].all()
    seeds = np.unique(data['phi_seed'])
    assert len(seeds) == 20 and len(data['phi_seed']) == 100
    config = json.loads((M9/'PROTOCOL.json').read_text())['environment']
    env = Environment(obstacles=tuple(RectangleObstacle(**o) for o in config['obstacles']),
                      camera=CameraModel(**config['camera']), budget=config['budget'],
                      observation_dt=config['observation_dt'], scan_rate=config['scan_rate'], clearance=config['clearance'])
    points, support, initial = data['grid_points'], data['free_mask'], data['initial_state']
    vmax, umax = actual_limits()
    protocol = dict(only_val=True, test_used=False, val_phi=seeds, cap_seconds=CAP, environment=asdict(env),
        disks=env.collision_disks, velocity_norm_limit=vmax, control_component_limit=umax,
        routing=dict(vertices=POLYGON_N, radial_margin=MARGIN, metric='shortest visibility-graph length; deterministic ties',
            observation_arrival_offset=[ARRIVAL_OFFSET,0.], reason='avoid undefined bearing when target equals camera position; still require original visibility'),
        lawn=dict(row_spacing=env.camera.sensing_radius, fast='minimum discrete trapezoid duration; no scan wait at row ends',
            conservative='0.9*2*sqrt(R^2-(spacing/2)^2)/pan_period; full rotation at row endpoints',
            cleanup='both: nearest geometrically unseen free grid cell, move then wait until seen; no phi access'),
        greedy='target persists until observed/reached; observe every dt; reselect, brake only for direction change or arrival; no distance-step stops',
        greedy_ratio='remaining normalized cell probability / visibility-graph shortest route length; denominator floor 1e-12',
        dynamics='Euler dt=observation_dt; exact leg distance; v/a bounded; zero velocity at non-collinear corners; nonzero-speed straight continuation allowed',
        end_constraint='classical free endpoint; ETO/NN/diffusion existing fixed endpoint; disclose asymmetry',
        common_time='T_phi=mean of five ETO tf; no time rescaling; truncate or hold+pan',
        reference_probability='mean of five ETO Pdet(T_phi)',
        guided_setting='eta0.5_raw_smooth_scale; all 8 samples; raw eta0.5 auxiliary row',
        timing='new classical construction+online observation+decision; shared graph separately; old methods saved timing; evaluator excluded',
        statistics='equal phi weight; report censored reach rate and reached-only time; physics failures retained and separately labeled')
    json_write(HERE/'PROTOCOL.json', protocol)
    (HERE/'trajectories').mkdir()
    setup_start = time.perf_counter()
    router = Router(env, points)
    setup_seconds = time.perf_counter()-setup_start
    # 先验证离散最短时间速度曲线和路由，再生成正式样本。
    check_primitives(router, env, initial)
    json_write(HERE/'primitive_checks.json', dict(passed=True, graph_setup_seconds=setup_seconds))
    classic = {}
    shared = {}
    for method in ('lawn_fast', 'lawn_conservative'):
        begin = time.perf_counter()
        search = Search(env, points, support, initial, router)
        search.lawn(method == 'lawn_conservative')
        states, controls, tf = search.finish()
        seconds = time.perf_counter()-begin
        diagnostics = audit(states, tf, env, initial)
        assert diagnostics['physical_pass'], (method, diagnostics)
        save_classical(method, states, controls, tf, search, seconds, diagnostics)
        shared[method] = dict(states=states, tf=tf, generation_seconds=seconds, source=f'trajectories/{method}.npz')
        print(method, f'{seconds:.2f}s compute, path {tf:.2f}s, seen {np.isfinite(search.first).sum()}/{support.sum()}', flush=True)
    for seed in seeds:
        source = int(np.flatnonzero(data['phi_seed'] == seed)[0])
        weights = data['phi_grid'][source].ravel()
        for method in ('greedy_probability', 'greedy_probability_per_distance'):
            begin = time.perf_counter()
            search = Search(env, points, support, initial, router)
            search.greedy(weights, method.endswith('per_distance'))
            states, controls, tf = search.finish()
            seconds = time.perf_counter()-begin
            diagnostics = audit(states, tf, env, initial)
            assert diagnostics['physical_pass'], (seed, method, diagnostics)
            name = f'{method}_phi{seed:03d}'
            save_classical(name, states, controls, tf, search, seconds, diagnostics)
            classic[(int(seed), method)] = dict(states=states, tf=tf, generation_seconds=seconds, source=f'trajectories/{name}.npz')
            print(seed, method, f'{seconds:.2f}s compute, path {tf:.2f}s, seen {np.isfinite(search.first).sum()}/{support.sum()}', flush=True)
    evaluate(data, env, router, shared, classic, setup_seconds)
    after = inventory()
    changed = sorted(k for k in set(before)|set(after) if before.get(k) != after.get(k))
    assert not changed, changed
    json_write(HERE/'verification.json', dict(old_files_unchanged=True, protected_file_count=len(before),
        test_used=False, val_phi_count=20, classical_paths=42, classical_evaluation_rows=80, total_seconds=time.perf_counter()-started))
    json_write(HERE/'COMPLETE.json', dict(seconds=time.perf_counter()-started, unix_time=time.time()))


def save_classical(name, states, controls, tf, search, seconds, diagnostics):
    path = HERE/'trajectories'/name
    # 在线更新必须与原检测harness在完全相同的观测时间上逐格一致。
    replayed = replay(states, tf, search.env, search.points, search.support, search.t)
    np.testing.assert_allclose(replayed, search.first, rtol=0, atol=1e-10)
    residual = states[1:]-states[:-1]-search.h*np.column_stack((states[:-1, 2:], controls[:-1]))
    assert np.abs(residual).max() < 1e-10
    np.savez_compressed(path.with_suffix('.npz'), states=states, controls=controls, tf=tf,
        first_seen_grid=search.first, times=np.arange(len(states))*search.h)
    json_write(path.with_suffix('.json'), dict(generation_seconds=seconds, decisions=search.decisions,
        events=search.events, dynamics_residual=float(np.abs(residual).max()), diagnostics=diagnostics))


def check_primitives(router, env, initial):
    vmax, umax = actual_limits()
    h = env.observation_dt
    for length in (.0001, .01, .125, .5, 1., 2.):
        for direction in (np.array([1., 0.]), np.array([1., 1.])/np.sqrt(2)):
            v = velocity_profile(length, direction, h, vmax, umax)
            assert v.max() <= vmax+1e-12 and np.max(np.abs(np.diff(v)[:, None]*direction/h)) <= umax+1e-12
            assert abs(h*v[:-1].sum()-length) < 1e-10 and v[0] == v[-1] == 0
    for q in ([.9, .9], [.1, .9], [.9, .1], [.5, .8]):
        xy = np.vstack((initial[:2], router.route(initial[:2], np.array(q))))
        geom = geometry_audit(xy, env, np.array([[0, 1], [0, 1]]))
        assert not geom['intersects_rectangle'] and geom['min_collision_disk_clearance'] >= MARGIN-1e-10


def evaluate(data, env, router, shared, classic, setup_seconds):
    original = load(M9/'val_samples.npz')
    base = load(GUIDE/'base/val_samples.npz')
    post = load(GUIDE/'post/val_samples.npz')
    points, support, initial = data['grid_points'], data['free_mask'], data['initial_state']
    rows, references, archived = [], [], {}
    for seed in np.unique(data['phi_seed']):
        indices = np.flatnonzero(data['phi_seed'] == seed)
        weights = data['phi_grid'][indices[0]].ravel()
        common_time = float(data['tf'][indices].mean())
        eto = []
        eto_first = []
        for j in indices:
            states = np.column_stack((data['positions'][j], data['velocities'][j]))
            first = replay(states, float(data['tf'][j]), env, points, support, common_time)
            eto_first.append(first)
            eto.append(dict(method='ETO', sample_id=int(data['init_id'][j]), states=states, tf=float(data['tf'][j]),
                generation_seconds=float(data['planning_seconds_including_compile'][j]), source=f'M8 phi{seed} init{data["init_id"][j]}'))
        reference = float(np.mean([weighted_detection_stats(f, weights)['detection_probability'] for f in eto_first]))
        references.append(dict(phi_seed=int(seed), common_time=common_time, target_probability=reference))
        candidates = eto + [dict(method=k, sample_id=0, **v) for k, v in shared.items()]
        candidates += [dict(method=k, sample_id=0, **v) for (s, k), v in classic.items() if s == seed]
        for archive, source_path, method, setting in (
            (original, 'M9/val_samples.npz', 'nearest', None),
            (base, 'M9/guidance/base/val_samples.npz', 'diffusion_guided_raw', 'diff_eta0.5_raw'),
            (post, 'M9/guidance/post/val_samples.npz', 'diffusion_guided_smooth_scale', 'diff_eta0.5_raw_smooth_scale')):
            mask = archive['phi_seed'] == seed
            mask &= (archive['method'] == 'nearest_neighbor') if setting is None else (archive['setting'] == setting)
            for j in np.flatnonzero(mask):
                values = archive['values'][j]
                xy, tf = values[:-1].reshape(-1, 2), float(values[-1])
                velocity = np.vstack((np.diff(xy, axis=0)/(tf/len(xy)), np.zeros(2)))
                candidates.append(dict(method=method, sample_id=int(archive['sample_id'][j]), states=np.column_stack((xy, velocity)),
                    tf=tf, generation_seconds=float(archive['generation_seconds'][j]), source=f'{source_path}:row={j}'))
        assert len(candidates) == 26, [(v['method'], v['sample_id']) for v in candidates]
        for item in candidates:
            states, tf = item['states'], item['tf']
            begin = time.perf_counter()
            first_common = replay(states, tf, env, points, support, common_time)
            # 固定末端位置扫描一整圈后不会新增可见格子；保持原0.05s时钟。
            horizon = min(CAP, tf+2*np.pi/env.scan_rate+env.observation_dt)
            first_cap = replay(states, tf, env, points, support, horizon)
            first_cap[first_cap > CAP] = np.inf
            # 额外末次观测只在确实的120s截止点发生；horizon捷径不能额外增加不同相位观测。
            extra = np.isfinite(first_cap) & (np.abs(first_cap-horizon) < 1e-9)
            if horizon < CAP:
                assert not extra.any(), '终点扫描后仍新增观测，不能缩短回放。'
            common_stats, _ = compute_metrics(first_common, weights, reference)
            cap_stats, match = compute_metrics(first_cap, weights, reference)
            diagnostics = audit(states, tf, env, initial)
            row = dict(phi_seed=int(seed), method=item['method'], sample_id=item['sample_id'], source=item['source'],
                common_time=common_time, reference_probability=reference,
                detection_probability_common=common_stats['detection_probability'],
                conditional_mean_t_find_common=common_stats['weighted_mean_t_find'],
                detection_probability_120=cap_stats['detection_probability'], reach_time=match,
                reached=match is not None, restricted_reach_time=min(CAP, match) if match is not None else CAP,
                generation_seconds=item['generation_seconds'], shared_graph_setup_seconds=setup_seconds if item['method'].startswith(('lawn', 'greedy')) else 0.,
                evaluation_seconds=time.perf_counter()-begin, **diagnostics)
            rows.append(row)
            key = f'{seed}_{item["method"]}_{item["sample_id"]}'
            archived[key+'_first_common'] = first_common
            archived[key+'_first_120'] = first_cap
            archived[key+'_states'] = states
            if item['method'] == 'ETO':
                # 旧 own-tf 指标作为适配回归检查；主表只使用新的共同时间指标。
                j = indices[item['sample_id']]
                old_first = replay(states, tf, env, points, support, tf)
                np.testing.assert_allclose(old_first, data['first_seen_grid'][j].ravel(), rtol=0, atol=1e-10)
        print(f'evaluated phi {seed}, T={common_time:.4f}, reference P={reference:.5f}, {len(rows)}/520 rows', flush=True)
        csv_write(HERE/'results.csv', rows)
    np.savez_compressed(HERE/'replays.npz', **archived)
    csv_write(HERE/'references.csv', references)
    json_write(HERE/'results.json', rows)
    # 验证截短末端扫描回放与完整120s回放完全一致，不以假定替代检查。
    first_seed = int(data['phi_seed'][0])
    for method in ('ETO', 'nearest', 'diffusion_guided_raw', 'diffusion_guided_smooth_scale'):
        row = next(r for r in rows if r['phi_seed'] == first_seed and r['method'] == method)
        key = f'{first_seed}_{method}_{row["sample_id"]}'
        full = replay(archived[key+'_states'], row['tf'], env, points, support, CAP)
        np.testing.assert_allclose(full, archived[key+'_first_120'], rtol=0, atol=1e-10)
    assert len(rows) == 520
    json_write(HERE/'replay_checks.json', dict(ETO_own_tf_reproduced=100, terminal_scan_shortcut_checked=4,
        classical_online_replay_checked=42, result_count=520))
    from report import report
    report(data, rows, references, archived, env)


def evaluate_existing():
    # 评测故障恢复：只读42条已落盘新轨迹，不重新生成策略或重算历史方法。
    assert not (HERE/'COMPLETE.json').exists()
    before = json.loads((HERE/'protected_before.json').read_text())
    data = load(GUIDE/'val_inputs.npz')
    assert set(data['split']) == {'val'}
    config = json.loads((HERE/'PROTOCOL.json').read_text())['environment']
    env = Environment(obstacles=tuple(RectangleObstacle(**o) for o in config['obstacles']),
        camera=CameraModel(**config['camera']), budget=config['budget'], observation_dt=config['observation_dt'],
        scan_rate=config['scan_rate'], clearance=config['clearance'])
    shared, classic = {}, {}
    for method in ('lawn_fast', 'lawn_conservative', 'greedy_probability', 'greedy_probability_per_distance'):
        seeds = [None] if method.startswith('lawn') else np.unique(data['phi_seed'])
        for seed in seeds:
            name = method if seed is None else f'{method}_phi{seed:03d}'
            saved = load(HERE/'trajectories'/f'{name}.npz')
            meta = json.loads((HERE/'trajectories'/f'{name}.json').read_text())
            entry = dict(states=saved['states'], tf=float(saved['tf']), generation_seconds=meta['generation_seconds'],
                         source=f'trajectories/{name}.npz')
            if seed is None: shared[method] = entry
            else: classic[(int(seed),method)] = entry
    setup = json.loads((HERE/'primitive_checks.json').read_text())['graph_setup_seconds']
    evaluate(data, env, None, shared, classic, setup)
    after = inventory()
    changed = sorted(k for k in set(before)|set(after) if before.get(k) != after.get(k))
    assert not changed, changed
    elapsed = time.time()-json.loads((HERE/'run_started.json').read_text())['unix_time']
    json_write(HERE/'verification.json', dict(old_files_unchanged=True, protected_file_count=len(before),
        test_used=False, val_phi_count=20, classical_paths=42, classical_evaluation_rows=80,
        recovered_evaluation_without_regeneration=True, wall_seconds_including_recovery=elapsed))
    json_write(HERE/'COMPLETE.json', dict(seconds=elapsed, unix_time=time.time()))


if __name__ == '__main__':
    if '--evaluate-existing' in sys.argv: evaluate_existing()
    else: main()
