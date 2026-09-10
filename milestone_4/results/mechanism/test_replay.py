"""Analytic and independent-reference tests; no solver calls or result writes."""
import sys
sys.dont_write_bytecode = True
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_analysis as r


class ReplayTests(unittest.TestCase):
    def test_entropy_known_distributions_and_zeros(self):
        self.assertAlmostEqual(r.entropy([1, 0, 0]), 0.)
        self.assertAlmostEqual(r.entropy([.25]*4), np.log(4))
        self.assertAlmostEqual(r.entropy([1, 3]), -.25*np.log(.25)-.75*np.log(.75))
        with self.assertRaises(ValueError):
            r.entropy([0, 0])

    def test_node_weights_and_boundary(self):
        states = np.array([[0, 0], [.5, .5], [1, 1], [1, 1]])
        np.testing.assert_allclose(r.node_occupancy(states, 2), [.25, 0, 0, .75])
        with self.assertRaises(ValueError):
            r.cell_ids([[1.01, .1]])

    def test_exact_dwell_crossings_terminal_hold_and_truncation(self):
        states = np.array([[0., .25], [1., .25]])
        # tf=2: motion 0..1 s; terminal hold 1..2 s.
        np.testing.assert_allclose(r.dwell_occupancy(states, 2., 2., 2), [.25, .75, 0, 0])
        np.testing.assert_allclose(r.dwell_occupancy(states, 2., .25, 2), [1., 0, 0, 0])
        np.testing.assert_allclose(r.dwell_occupancy(states, 2., 4., 2), [.125, .875, 0, 0])

    def test_diagonal_corner_crossing_and_stationary_path(self):
        np.testing.assert_allclose(r.dwell_occupancy(np.array([[0., 0.], [1., 1.]]), 2., 2., 2), [.25, 0, 0, .75])
        p = r.dwell_occupancy(np.array([[.2, .2], [.2, .2], [.2, .2]]), 3., 15.)
        self.assertAlmostEqual(r.entropy(p), 0.)

    def test_visibility_budget_infinity_and_mass_identity(self):
        truth = np.array([.1, .2, .3, .4])
        result = r.visibility(np.array([0., np.inf, 15., 16.]), np.array([0., 2., np.inf, np.inf]), truth)
        self.assertAlmostEqual(result['gained_pp'], 30.)
        self.assertAlmostEqual(result['lost_pp'], 20.)
        self.assertAlmostEqual(result['net_pp'], 10.)
        self.assertAlmostEqual(result['error_visible_mass'], .4)

    def test_scene_pairing_and_incomplete_directions(self):
        blocks = []
        for scene, values in [('a', [2., -4.]), ('b', [8.])]:
            for value in values:
                blocks.append(dict(gamma=.1, family='spatial_shift', js=.05, scene=scene,
                                   error_visible_mass=.5, oracle_visible_mass=.5-value/100,
                                   gained_pp=max(value, 0), lost_pp=max(-value, 0), net_pp=value))
        a, b = r.scene_visibility(blocks)
        self.assertTrue(a['complete'])
        self.assertEqual(a['net_pp'], -1)
        self.assertFalse(b['complete'])
        self.assertEqual(b['valid_directions'], 1)

    def test_ergodic_metric_zero_when_prior_is_node_measure(self):
        xy = np.array([[.1, .2], [.3, .9], [.8, .4]])
        self.assertAlmostEqual(r.achieved_metric(xy, xy, np.ones(3)/3), 0.)

    def test_metric_independent_upstream_implementation(self):
        # Only the pure Fourier/metric modules; no planner/builder/optimizer.
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        sys.path.insert(0, str(r.ROOT/'external/time_optimal_ergodic_search'))
        from time_opt_erg_lib.fourier_utils import BasisFunc, get_ck, get_phik
        from time_opt_erg_lib.ergodic_metric import ErgodicMetric
        rng = np.random.default_rng(4712)
        xy, points = rng.random((50, 2)), rng.random((100, 2))
        prior = rng.random(100); prior /= prior.sum()
        basis = BasisFunc([8, 8])
        metric = ErgodicMetric(basis)(get_ck(jnp.array(xy), basis, 7., 7./50), get_phik((jnp.array(prior), jnp.array(points)), basis))
        self.assertAlmostEqual(r.achieved_metric(xy, points, prior), float(metric), places=12)

    def test_three_stored_plans_independent_reference(self):
        cases = [('cx0.25_cy0.25_layoutA', .05, 'oracle'),
                 ('cx0.75_cy0.75_layoutB', .1, 'diffuse_blur_js_0.15'),
                 ('cx0.25_cy0.75_layoutC', .05, 'false_hotspot_x_js_0.1')]
        for scene, gamma, prior in cases:
            with self.subTest(scene=scene, prior=prior):
                directory = r.RESULTS/'formal/plans'/f'{scene}__gamma_{gamma:g}__{prior}'
                data = r.load_npz(r.RESULTS/'frozen/priors'/f'{scene}.npz')
                trajectory = r.load_npz(directory/'trajectory.npz')
                first = r.load_npz(directory/'first_seen_grid.npz')['first_seen']
                saved = r.read_json(directory/'exact_metrics.json')
                path = sum(float(np.hypot(*(b[:2]-a[:2]))) for a, b in zip(trajectory['states'][:-1], trajectory['states'][1:]))
                self.assertAlmostEqual(path, saved['trajectory_length'], places=12)
                self.assertAlmostEqual(sum(float(w) for w, t in zip(data['oracle'], first) if np.isfinite(t)), saved['exact_detection_probability'], places=12)
                self.assertAlmostEqual(r.achieved_metric(trajectory['states'], data['points'], data[prior]), r.read_json(directory/'diagnostics.json')['ergodicity'], places=9)
                manifest = r.read_json(r.RESULTS/'frozen/PROTOCOL_V2_FROZEN_MANIFEST.json')
                spec = next(s for s in manifest['candidate_manifest']['scenes'] if s['scene_id'] == scene)
                env = r.Environment(obstacles=tuple(r.RectangleObstacle(**o) for o in spec['obstacles']))
                np.testing.assert_array_equal(first, r.pilot_analysis.first_seen_grid(trajectory, env, data['points'], data['oracle'] > 0))

    def test_planner_not_imported(self):
        self.assertNotIn('milestone_3.planner', sys.modules)


if __name__ == '__main__':
    unittest.main()
