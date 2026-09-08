"""Checkpoint/failure tests use synthetic trajectories; zero solver calls."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from milestone_4 import run_formal as r

class FormalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.inputs, _ = r.load_frozen()
        cls.label = r.expand_plans(cls.manifest)[0]
        cls.scene = cls.manifest['candidate_manifest']['scenes'][0]
        cls.data = cls.inputs[cls.label['scene_id']]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def policy(self, valid):
        states = np.zeros((50,4)); states[:,:2] = [.1,.1]
        return dict(states=states, controls=np.zeros((50,2)), tf=3., diagnostics={'valid':valid,'solver_success':valid})

    def complete(self):
        return r.complete_attempt(self.path,self.label,self.data,self.scene,self.manifest)

    def test_manifest_expands_exact_distinct_plans(self):
        plans=r.expand_plans(self.manifest)
        self.assertEqual(len(plans),552)
        self.assertEqual(len({p['plan_id'] for p in plans}),552)

    def test_worker_claim_cannot_be_reused(self):
        r.claim_worker(self.path)
        with self.assertRaises(FileExistsError): r.claim_worker(self.path)

    def test_interruption_is_terminal_missing_metrics_not_timeout(self):
        result=self.complete()
        self.assertEqual(result['status'],'interrupted')
        self.assertIsNone(r.read_json(self.path/'exact_metrics.json')['exact_detection_probability'])
        sampled=r.read_json(self.path/'sampled_metrics.json')
        self.assertTrue(all(x['episodes']==0 and x['success_rate'] is None for x in sampled))
        with patch.object(r,'write_evaluation',side_effect=AssertionError('must skip')):
            self.assertEqual(self.complete(),result)

    def test_invalid_solver_is_preserved_and_never_evaluated(self):
        r.save_solver_result(self.path,self.policy(False))
        with patch.object(r,'evaluate',side_effect=AssertionError('invalid')):
            result=self.complete()
        self.assertEqual(result['status'],'invalid')
        self.assertTrue((self.path/'trajectory.npz').exists())
        with patch.object(r,'write_evaluation',side_effect=AssertionError('must skip')):
            self.assertEqual(self.complete(),result)

    def test_valid_saved_policy_resumes_evaluation_and_then_skips(self):
        r.save_solver_result(self.path,self.policy(True))
        result=self.complete()
        self.assertEqual(result['status'],'valid')
        self.assertEqual(sum(x['episodes'] for x in r.read_json(self.path/'sampled_metrics.json')),1280)
        with patch.object(r,'write_evaluation',side_effect=AssertionError('must skip')):
            self.assertEqual(self.complete(),result)

    def test_checkpoint_tampering_rejected(self):
        self.complete()
        (self.path/'episodes.csv').write_text('corrupted')
        with self.assertRaises(ValueError): self.complete()

    def test_saved_solver_tampering_rejected(self):
        r.save_solver_result(self.path,self.policy(True))
        (self.path/'trajectory.npz').write_bytes(b'corrupt')
        with self.assertRaises(ValueError): self.complete()

    def test_attempt_label_and_hash_checked(self):
        saved=dict(self.label,attempt_count=1,frozen_manifest_sha256='manifest',prior_sha256='prior')
        r.atomic_json(self.path/'started.json',saved)
        r.validate_attempt(self.path,self.label,'manifest','prior')
        with self.assertRaises(ValueError): r.validate_attempt(self.path,self.label,'changed','prior')
        wrong=dict(self.label,gamma=.2)
        with self.assertRaises(ValueError): r.validate_attempt(self.path,wrong,'manifest','prior')

    def test_resume_rejects_changed_embedded_manifest_before_worker(self):
        manifest_path=r.DEFAULT_MANIFEST
        m,inputs,base=r.load_frozen(manifest_path)
        prov=r.provenance(manifest_path,m,base)
        changed=copy.deepcopy(m)
        changed['planner_and_evaluation_invariants']['planner']['maxiter']=601
        config=dict(frozen_manifest_sha256=r.sha(manifest_path),frozen_manifest=changed,
                    expected_plans=r.expand_plans(m),**prov)
        r.atomic_json(self.path/'config.json',config)
        with self.assertRaisesRegex(ValueError,'embedded manifest'):
            r.run(manifest_path,self.path)

if __name__=='__main__': unittest.main()
