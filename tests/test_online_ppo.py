"""Synthetic online fragments and real hidden worker; no gameplay or capture."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional PyTorch unavailable")
import torch

from playmodel.atomic_io import atomic_json
from playmodel.learning import online_ppo as online
from playmodel.learning.full_run import FullRunRecorder
from playmodel.learning.recurrent_ppo import ModelConfig, RecurrentActorCritic


class OnlinePPOTests(unittest.TestCase):
    def setUp(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.output = self.root / "jobs"
        torch.manual_seed(17)
        self.model = RecurrentActorCritic(ModelConfig(context_dim=4, candidate_dim=3,
            hidden_size=16, visual_size=16, candidate_hidden_size=8))

    def proof(self, name, observed):
        path = self.root / name
        path.write_bytes(name.encode())
        return dict(frame_ref=str(path), frame_sha256=online._sha(path), observed_at_ns=observed,
            available_at_ns=observed+1, verified_at_ns=observed+2, verified=True,
            independent_of_policy=True, verifier_id="synthetic", clock_domain="perf_counter_ns_same_host")

    def fragment(self, *, split="train", next_delta=0., bad_hidden=False, kind="truncated"):
        recorder = FullRunRecorder(self.model, "episode/version0", split=split, session_ids=("episode",))
        for index, phase in enumerate((1, 3, 0, 0)):
            count = 3 if phase else 1
            legal = torch.zeros(1, 9, dtype=torch.bool)
            legal[:, :count if phase else 9] = True
            inputs = (torch.randint(0, 256, (1,3,96,96), dtype=torch.uint8), torch.zeros(1,4),
                      torch.tensor([phase]), torch.rand(1,count,3), legal)
            before = recorder.hidden.clone()
            reset = torch.tensor([index == 0])
            with torch.no_grad():
                result = self.model.step(*inputs, hidden=before, reset=reset)
                action, logp = result.sample(generator=torch.Generator().manual_seed(index))
            tensors = dict(zip(("images","context","phase","candidates","legal_mask"), inputs))
            tensors.update(hidden_before=before, next_hidden=result.next_hidden, reset=reset,
                           actions=action, old_log_probs=logp, old_values=result.value)
            evidence = self.proof(f"frame-{index}", (index+1)*1_000_000_000)
            evidence.update(decided_at_ns=evidence["observed_at_ns"]+3, sent_at_ns=evidence["observed_at_ns"]+4,
                action_origin="policy", transmitted=True, actual_action=int(action), acknowledged=None,
                game_application_verified=phase != 0)
            recorder.append_decision(tensors, evidence=evidence)
        ending = self.proof("ending", 6_000_000_000)
        ending["kind"] = kind
        with torch.no_grad():
            bootstrap = self.model.step(*inputs, hidden=recorder.hidden).value.item()
        if kind == "truncated":
            ending.update(online.save_online_bootstrap(self.root/"bootstrap.pt", inputs,
                recorder.hidden + (1 if bad_hidden else 0), behavior_version=self.model.policy_version(),
                observed_at_ns=ending["observed_at_ns"], available_at_ns=ending["available_at_ns"]))
        report = recorder.finish(self.root/"fragment", kind=kind, evidence=ending,
                                 next_value=bootstrap+next_delta, chunk_steps=2, burn_in=1)
        self.manifest = Path(report["manifest_path"])
        return report

    def request(self):
        return online._request(self.manifest, root=self.repo, output=self.output,
                               device="cpu", seed=0, stop_file=None)

    def start(self, **kwargs):
        return online.start_online_job(self.manifest, root=self.repo, output=self.output,
                                       device="cpu", seed=0, **kwargs)

    def completed(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(online.subprocess, "Popen", return_value=process):
            job = self.start()
        with patch.object(torch, "set_num_interop_threads"):
            online._run_job(job.directory)
        return job

    def test_truncated_fragment_retains_menu_and_combat_with_true_bootstrap(self):
        report = self.fragment()
        _, _, chunks, flat, indices, header = online._load_fragment(self.manifest)
        self.assertFalse(report["full_run_complete"])
        self.assertEqual(flat.phase[:,0].tolist(), [1,3,0,0])
        self.assertEqual(header["phase_counts"], {"0":2,"1":1,"3":1})
        self.assertTrue(flat.truncated[-1,0])
        self.assertEqual(sorted(indices[chunks.valid][indices[chunks.valid] >= 0].unique().tolist()), [0,1,2,3])

    def test_evaluation_and_ineligible_fragments_cannot_launch(self):
        self.fragment(split="evaluation")
        with patch.object(online.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "verified train fragment"):
                self.start()
        spawn.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_fragment_validation_and_optimizer_share_device(self):
        self.fragment()
        request = online._request(self.manifest, root=self.repo, output=self.output,
                                  device='cuda', seed=0, stop_file=None)
        directory = self.root/'cuda-training'
        directory.mkdir()
        report = online._train(request, directory, lambda **kwargs: None)
        self.assertGreater(report['optimizer_steps'], 0)
        self.assertTrue(report['checkpoint_reload_verified'])

    def test_bootstrap_value_mismatch_is_rejected(self):
        self.fragment(next_delta=.1)
        with self.assertRaisesRegex(ValueError, "bootstrap value"):
            online._load_fragment(self.manifest)

    def test_bootstrap_hidden_mismatch_is_rejected(self):
        self.fragment(bad_hidden=True)
        with self.assertRaisesRegex(ValueError, "bootstrap source/time/hidden"):
            online._load_fragment(self.manifest)

    def test_death_uses_independent_reward_and_zero_bootstrap(self):
        self.fragment(kind="death")
        _, _, _, flat, _, _ = online._load_fragment(self.manifest)
        self.assertEqual(float(flat.rewards[-1,0]), -1.)
        self.assertEqual(float(flat.next_values[-1,0]), 0.)
        self.assertTrue(bool(flat.terminated[-1,0]))

    def test_job_is_hidden_and_duplicate_launch_attaches(self):
        self.fragment()
        process = Mock()
        process.poll.return_value = None
        with patch.object(online.subprocess, "Popen", return_value=process) as spawn:
            first, second = self.start(), self.start()
        spawn.assert_called_once()
        self.assertEqual(first.directory, second.directory)
        self.assertEqual(spawn.call_args.kwargs["creationflags"], getattr(subprocess,"CREATE_NO_WINDOW",0)
                         | getattr(subprocess,"BELOW_NORMAL_PRIORITY_CLASS",0))
        self.assertEqual(spawn.call_args.kwargs["env"]["OMP_NUM_THREADS"], "1")
        first.close()
        process.terminate.assert_not_called()

    def test_saved_candidate_recovers_without_second_optimizer(self):
        self.fragment()
        job = self.completed()
        first = job.poll()
        (job.directory/"result.json").unlink()
        with patch.object(torch,"set_num_interop_threads"), \
                patch("playmodel.learning.recurrent_ppo.ppo_update") as update:
            online._run_job(job.directory)
        update.assert_not_called()
        self.assertEqual(job.poll()["checkpoint"], first["checkpoint"])

    def test_result_numbers_come_from_checkpoint_and_stale_apply_is_rejected(self):
        self.fragment()
        job = self.completed()
        expected = job.poll()
        result_path = job.directory/"result.json"
        document = json.loads(result_path.read_text())
        document.update(final_kl_within_target=not expected["final_kl_within_target"], optimizer_steps=-1)
        atomic_json(result_path, document)
        actual = job.poll()
        self.assertEqual(actual["final_kl_within_target"], expected["final_kl_within_target"])
        self.assertEqual(actual["optimizer_steps"], expected["optimizer_steps"])
        with self.assertRaisesRegex(online.TrainingJobError, "stale online candidate"):
            job.load_candidate("other-actor")
        if actual["final_kl_within_target"]:
            model, report = job.load_candidate(self.model.policy_version())
            self.assertEqual(model.policy_version(), report["candidate_version"])
            self.assertEqual(self.model.policy_version(), report["source_version"])

    def test_changed_manifest_and_bootstrap_original_are_rejected(self):
        self.fragment()
        request = self.request()
        with (self.root/"bootstrap.pt").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "bootstrap evidence changed"):
            online._load_fragment(self.manifest)
        with self.manifest.open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            online._verify(request)

    def test_stop_before_launch_is_bounded_and_no_optimizer_starts(self):
        self.fragment()
        stop = self.root/"STOP"
        stop.touch()
        with patch.object(online.subprocess,"Popen") as spawn:
            job = self.start(stop_file=stop)
        spawn.assert_not_called()
        with self.assertRaises(online.TrainingCancelled):
            job.poll()

    def test_candidate_consumption_rechecks_original_evidence(self):
        self.fragment()
        job = self.completed()
        self.assertIsNotNone(job.poll())
        (self.root/"frame-0").write_bytes(b"modified after optimization")
        with self.assertRaisesRegex(online.TrainingJobError, "original evidence changed"):
            job.poll()

    def test_real_worker_trains_fragment_while_coordinator_remains_free(self):
        self.fragment()
        original = self.model.policy_version()
        job = self.start()
        try:
            self.assertIsNone(job.poll())
            result = job.wait(timeout=30)
            self.assertEqual(result["source_version"], original)
            self.assertGreater(result["optimizer_steps"], 0)
            self.assertNotEqual(result["candidate_version"], original)
            self.assertGreater(result["optimization_finished_at_ns"], result["optimization_started_at_ns"])
            self.assertEqual(result["phase_counts"], {"0":2,"1":1,"3":1})
            self.assertEqual(self.model.policy_version(), original)
            job.process.wait(timeout=5)
        finally:
            if job.process.poll() is None:
                job.close(cancel=True)


if __name__ == "__main__":
    unittest.main()
