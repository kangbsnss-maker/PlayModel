"""Durability/provenance checks without game input or an expensive PPO workload."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional PyTorch unavailable")
import torch

from playmodel.atomic_io import atomic_json
from playmodel.learning import recurrent_training_worker as worker
from playmodel.learning.recurrent_ppo import ModelConfig, RecurrentActorCritic, load_checkpoint, save_checkpoint
from playmodel.learning.runtime_contract import contract_fields


class RecurrentTrainingWorkerTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = Path(__file__).resolve().parents[1]
        self.output = self.directory / "output"
        self.output.mkdir()
        self.source = self.directory / "source.pt"
        torch.manual_seed(3)
        self.model = RecurrentActorCritic(ModelConfig(context_dim=4, candidate_dim=3,
            hidden_size=16, visual_size=16, candidate_hidden_size=8))
        save_checkpoint(self.model, self.source, contract_fields())
        self.manifest_path = self.directory / "full-run.json"
        self.header = dict(run_id="training-session", split="train", training_eligible=True,
                           full_run_complete=True, burn_in=1, behavior_version=self.model.policy_version(),
                           **contract_fields())
        atomic_json(self.manifest_path, self.header)
        self.training = dict(self.header, manifest_path=str(self.manifest_path))
        self.process = Mock()
        self.process.poll.return_value = None

    def request(self, stop_file=None):
        return worker._request(self.source, self.training, root=self.root, output=self.output,
                               device="cpu", seed=7, stop_file=stop_file)

    def start(self, **kwargs):
        return worker.start_training_job(self.source, self.training, root=self.root,
            output=self.output, device="cpu", seed=7, **kwargs)

    def candidate(self, *, source_version=None, name="candidate-saved", ppo_fields=None):
        model, _ = load_checkpoint(self.source)
        with torch.no_grad():
            next(model.parameters()).add_(.01)
        report = dict(source_version=source_version or self.model.policy_version(),
                      candidate_version=model.policy_version(), optimizer_steps=1)
        report.update(ppo_fields or {})
        path = self.output / name / "candidate.pt"
        save_checkpoint(model, path, dict(full_run_manifest=str(self.manifest_path),
            ppo=report, **contract_fields()))
        return dict(report, checkpoint=str(path), checkpoint_reload_verified=True)

    def make_job(self, request=None):
        request = request or self.request()
        directory = self.output / "training-jobs" / request["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "request.json", request)
        return worker.TrainingJob(directory, request)

    def save_result(self, job, candidate):
        atomic_json(job.directory / "result.json", dict(schema=worker.SCHEMA,
            request_sha256=job.request["request_sha256"], candidate=candidate,
            checkpoint_sha256=worker._sha(candidate["checkpoint"])))

    def test_manifest_and_source_identity_are_enforced_before_optimizer(self):
        with patch("playmodel.learning.recurrent_ppo.ppo_update") as update:
            for key, value in (("split", "evaluation"), ("training_eligible", False),
                               ("full_run_complete", False)):
                invalid = dict(self.training, **{key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    worker.train_candidate_once(self.source, invalid, output=self.output, device="cpu", seed=7)
            atomic_json(self.manifest_path, dict(self.header, behavior_version="wrong-source"))
            with self.assertRaisesRegex(ValueError, "source policy"):
                worker.train_candidate_once(self.source, self.training, output=self.output, device="cpu", seed=7)
        update.assert_not_called()

    def test_completed_checkpoint_recovers_without_optimizer_and_preserves_partial(self):
        partial = self.output / "candidate-000-partial" / "candidate.pt"
        partial.parent.mkdir()
        partial.write_bytes(b"interrupted checkpoint")
        wrong = self.candidate(source_version="unrelated-source", name="candidate-001-wrong")
        correct = self.candidate(name="candidate-002-correct")
        hashes = {path: worker._sha(path) for path in (partial, Path(wrong["checkpoint"]), Path(correct["checkpoint"]))}
        with patch("playmodel.learning.full_run.load_full_run", return_value=([], None, None, self.header)) as loader, \
                patch("playmodel.learning.recurrent_ppo.ppo_update") as update:
            result = worker.train_candidate_once(self.source, self.training, output=self.output, device="cpu", seed=7)
        loader.assert_called_once()
        update.assert_not_called()
        self.assertEqual(result["checkpoint"], correct["checkpoint"])
        self.assertTrue(result["recovered_completed_update"])
        self.assertEqual(hashes, {path: worker._sha(path) for path in hashes})

    def test_update_records_actual_optimizer_times_and_preserves_source(self):
        original = worker._sha(self.source)
        notifications = []

        def update(model, chunks, config, *, optimizer, trajectory, transition_indices):
            self.assertEqual(config.minibatch_sequences, 2)
            self.assertEqual(config.gamma, .997)
            source_version = model.policy_version()
            for _ in range(2):
                optimizer.zero_grad()
                next(model.parameters()).sum().backward()
                optimizer.step()
            return dict(source_version=source_version, candidate_version=model.policy_version(), optimizer_steps=2)

        with patch("playmodel.learning.full_run.load_full_run", return_value=([], None, None, self.header)), \
                patch("playmodel.learning.recurrent_ppo.ppo_update", side_effect=update):
            result = worker.train_candidate_once(self.source, self.training, output=self.output, device="cpu", seed=7,
                                                status_callback=lambda **row: notifications.append(row))
        self.assertEqual(worker._sha(self.source), original)
        self.assertLessEqual(result["optimization_started_at_ns"], result["optimization_finished_at_ns"])
        self.assertGreater(result["training_elapsed_seconds"], 0)
        self.assertEqual(notifications[-1], {"phase": "training_save", "optimizer_steps": 2})
        restored, metadata = load_checkpoint(result["checkpoint"])
        self.assertEqual(restored.policy_version(), result["candidate_version"])
        self.assertEqual(metadata["source_checkpoint_sha256"], original)
        self.assertEqual(metadata["full_run_manifest_sha256"], worker._sha(self.manifest_path))
        self.assertEqual(metadata["ppo_config"], self.request()["ppo_config"])

    def test_start_is_hidden_low_priority_and_duplicate_attaches(self):
        with patch.object(worker.subprocess, "Popen", return_value=self.process) as spawn:
            first, second = self.start(), self.start()
        spawn.assert_called_once()
        self.assertEqual(first.directory, second.directory)
        self.assertIsNone(second.process)
        _, kwargs = spawn.call_args
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["creationflags"], getattr(subprocess, "CREATE_NO_WINDOW", 0)
                         | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
        self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
        self.assertEqual(first.describe()["status_path"], str(first.directory / "state.json"))
        first.close()
        self.process.terminate.assert_not_called()
        self.assertFalse((first.directory / "cancel.json").exists())

    def test_os_lock_prevents_duplicate_optimizer_even_with_stale_state(self):
        job = self.make_job()
        atomic_json(job.directory / "state.json", {"status": "failed", "attempt": 1})
        with worker._exclusive(job.directory / "optimizer.lock"), \
                patch.object(worker.subprocess, "Popen") as spawn, \
                patch.object(worker, "train_candidate_once") as update:
            self.assertEqual(self.start().directory, job.directory)
            self.assertEqual(worker._run_job(job.directory), 0)
        spawn.assert_not_called()
        update.assert_not_called()

    def test_explicit_resume_archives_cancelled_attempt_and_recovers_checkpoint(self):
        job = self.make_job()
        candidate = self.candidate()
        atomic_json(job.directory / "state.json", {"status": "cancelled", "attempt": 1})
        atomic_json(job.directory / "cancel.json", {"reason": "user"})
        job.release_for_combat(sent_at_ns=time.perf_counter_ns(), run_id="previous-evaluation", session_id="old-combat")
        with patch.object(worker.subprocess, "Popen", return_value=self.process):
            resumed = self.start()
        self.assertEqual(resumed.describe()["attempt"], 2)
        self.assertTrue((job.directory / "attempt-0001-state.json").exists())
        self.assertTrue((job.directory / "attempt-0001-cancel.json").exists())
        self.assertFalse((job.directory / "cancel.json").exists())
        self.assertTrue((job.directory / "attempt-0001-combat-gate.json").exists())
        self.assertFalse((job.directory / "combat-gate.json").exists())
        with patch.object(torch, "set_num_interop_threads"), \
                patch("playmodel.learning.full_run.load_full_run", return_value=([], None, None, self.header)), \
                patch("playmodel.learning.recurrent_ppo.ppo_update") as update:
            self.assertEqual(worker._run_job(job.directory), 0)
        update.assert_not_called()
        attached = worker.TrainingJob(job.directory, job.request)
        self.assertEqual(attached.wait(timeout=.1)["checkpoint"], candidate["checkpoint"])
        self.assertEqual(attached.describe()["status"], "completed")

    def test_result_consumption_rejects_original_change_and_forged_candidate_provenance(self):
        job = self.make_job()
        candidate = self.candidate()
        self.save_result(job, candidate)
        self.assertEqual(job.wait(timeout=.1)["candidate_version"], candidate["candidate_version"])
        atomic_json(self.manifest_path, dict(self.header, tampered=True))
        with self.assertRaisesRegex(worker.TrainingJobError, "manifest changed"):
            job.wait(timeout=.1)
        atomic_json(self.manifest_path, self.header)
        wrong = self.candidate(source_version="wrong", name="candidate-forged")
        self.save_result(job, wrong)
        with self.assertRaisesRegex(worker.TrainingJobError, "provenance"):
            job.wait(timeout=.1)

    def test_request_pins_manifest_source_code_and_optimizer_settings(self):
        job = self.make_job()
        changed = deepcopy(job.request)
        changed["ppo_config"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "request changed"):
            worker._verify_request(changed)
        with patch.object(worker.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(worker.TrainingJobError, "settings/code changed"):
                worker.start_training_job(self.source, self.training, root=self.root, output=self.output,
                                          device="cpu", seed=8)
        spawn.assert_not_called()
        with self.source.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "source checkpoint changed"):
            worker._verify_request(job.request)

    def test_result_rejects_edited_numerical_gate_and_optimizer_evidence(self):
        job = self.make_job()
        candidate = self.candidate(ppo_fields={"final_kl_within_target": False,
            "combat_gate": {"sent_at_ns": 123, "run_id": "evaluation", "session_id": "combat"}})
        checkpoint_sha = worker._sha(candidate["checkpoint"])
        self.save_result(job, candidate)
        self.assertFalse(job.wait(timeout=.1)["final_kl_within_target"])
        for key, tampered in (("final_kl_within_target", True), ("optimizer_steps", 0),
                              ("combat_gate", {"sent_at_ns": 1, "run_id": "evaluation", "session_id": "combat"})):
            with self.subTest(key=key):
                self.save_result(job, dict(candidate, **{key: tampered}))
                with self.assertRaisesRegex(worker.TrainingJobError, "provenance"):
                    job.wait(timeout=.1)
                self.assertEqual(worker._sha(candidate["checkpoint"]), checkpoint_sha)

    def test_completed_result_survives_code_change_without_retraining(self):
        request = self.request()
        request["code_files"][worker.CODE_FILES[0]] = "previous-code-hash"
        body = {k: v for k, v in request.items() if k not in ("job_id", "request_sha256")}
        request["request_sha256"] = worker._digest(body)
        job = self.make_job(request)
        candidate = self.candidate()
        self.save_result(job, candidate)
        with patch.object(worker.subprocess, "Popen") as spawn:
            attached = self.start()
            result = attached.wait(timeout=.1)
        spawn.assert_not_called()
        self.assertEqual(attached.request, request)
        self.assertEqual(result["checkpoint"], candidate["checkpoint"])
        (job.directory / "result.json").unlink()
        with self.assertRaisesRegex(worker.TrainingJobError, "fresh training required"):
            self.start()

    def test_real_hidden_child_persists_evidence_failure(self):
        # A real process exercises CLI/bootstrap/atomic state, while the invalid
        # source identity stops before PPO or any gameplay could run.
        atomic_json(self.manifest_path, dict(self.header, behavior_version="unrelated-source"))
        job = self.start()
        try:
            with self.assertRaisesRegex(worker.TrainingJobError, "source policy.*log="):
                job.wait(timeout=20)
            self.assertEqual(job.describe()["status"], "failed")
            self.assertTrue((job.directory / "worker.log").exists())
            job.process.wait(timeout=5)
            self.assertFalse(worker._occupied(job.directory / "optimizer.lock"))
        finally:
            if job.process.poll() is None:
                job.close(cancel=True)

    def test_stop_prevents_launch_and_cancel_uses_owned_handle_only(self):
        stop = self.directory / "STOP"
        stop.touch()
        with patch.object(worker.subprocess, "Popen") as spawn:
            job = self.start(stop_file=stop)
        spawn.assert_not_called()
        with self.assertRaises(worker.TrainingCancelled):
            job.wait(timeout=.1)
        stop.unlink()
        self.process.wait.side_effect = [subprocess.TimeoutExpired("owned-worker", 8), None]
        owned = worker.TrainingJob(job.directory, job.request, self.process)
        owned.close(cancel=True)
        self.process.terminate.assert_called_once()

    def test_child_failure_is_persisted_and_wait_reports_log(self):
        job = self.make_job()
        with patch.object(torch, "set_num_interop_threads"), \
                patch.object(worker, "train_candidate_once", side_effect=ValueError("invalid rollout evidence")):
            with self.assertRaisesRegex(ValueError, "invalid rollout"):
                worker._run_job(job.directory)
        self.assertFalse(worker._occupied(job.directory / "optimizer.lock"))
        with self.assertRaisesRegex(worker.TrainingJobError, "invalid rollout evidence.*log="):
            job.wait(timeout=.1)

    def test_sync_gate_runs_after_evidence_and_before_any_optimizer(self):
        order = []
        gate = {"sent_at_ns": time.perf_counter_ns(), "run_id": "evaluation", "session_id": "combat"}

        def load(*args, **kwargs):
            order.append("evidence")
            return [], None, None, self.header

        def release():
            order.append("actual_send_gate")
            return gate

        def update(model, *args, **kwargs):
            order.append("optimizer")
            return dict(source_version=model.policy_version(), candidate_version=model.policy_version(), optimizer_steps=0)

        with patch("playmodel.learning.full_run.load_full_run", side_effect=load), \
                patch("playmodel.learning.recurrent_ppo.ppo_update", side_effect=update):
            result = worker.train_candidate_once(self.source, self.training, output=self.output, device="cpu", seed=7,
                                                optimization_gate=release)
        self.assertEqual(order, ["evidence", "actual_send_gate", "optimizer"])
        self.assertEqual(result["combat_gate"], gate)
        _, metadata = load_checkpoint(result["checkpoint"])
        self.assertEqual(metadata["ppo"]["combat_gate"], gate)
        self.assertIsNone(result["optimization_started_at_ns"])

    def test_worker_waits_for_first_actual_send_and_freezes_gate(self):
        request = worker._request(self.source, self.training, root=self.root, output=self.output,
                                 device="cpu", seed=7, stop_file=None, wait_for_combat=True)
        job = self.make_job(request)
        candidate = self.candidate()
        entered, optimized = threading.Event(), threading.Event()
        errors, gates = [], []

        def train(*args, optimization_gate, **kwargs):
            entered.set()
            gates.append(optimization_gate())
            optimized.set()
            return candidate

        def run():
            try:
                worker._run_job(job.directory)
            except BaseException as error:
                errors.append(error)

        with patch.object(torch, "set_num_interop_threads"), patch.object(worker, "train_candidate_once", side_effect=train):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertFalse(optimized.wait(.15))
                sent = time.perf_counter_ns()
                expected = job.release_for_combat(sent_at_ns=sent, run_id="source-evaluation", session_id="first-combat")
                self.assertEqual(job.release_for_combat(sent_at_ns=sent, run_id="source-evaluation", session_id="first-combat"), expected)
                with self.assertRaisesRegex(worker.TrainingJobError, "already frozen"):
                    job.release_for_combat(sent_at_ns=sent, run_id="other-evaluation", session_id="other-combat")
                self.assertTrue(optimized.wait(3))
            finally:
                if not optimized.is_set():
                    job.close(cancel=True)
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(gates, [expected])
        self.assertEqual(job.wait(timeout=.1)["checkpoint"], candidate["checkpoint"])

    def test_stop_interrupts_combat_gate_before_optimizer(self):
        stop = self.directory / "STOP"
        request = worker._request(self.source, self.training, root=self.root, output=self.output,
                                 device="cpu", seed=7, stop_file=stop, wait_for_combat=True)
        job = self.make_job(request)
        errors, optimized = [], []
        entered = threading.Event()

        def train(*args, optimization_gate, **kwargs):
            entered.set()
            optimization_gate()
            optimized.append(True)
            raise AssertionError("STOP should prevent optimization")

        def run():
            try:
                worker._run_job(job.directory)
            except BaseException as error:
                errors.append(error)

        with patch.object(torch, "set_num_interop_threads"), patch.object(worker, "train_candidate_once", side_effect=train):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
            finally:
                stop.touch()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(optimized, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], worker.TrainingCancelled)
        self.assertEqual(job.describe()["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
