"""Neural math/contract tests; synthetic transitions are not gameplay evidence."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional PyTorch is not installed")

import torch

from playmodel.learning.recurrent_ppo import (
    ModelConfig, PPOConfig, RecurrentActorCritic, RolloutBatch,
    generalized_advantage_estimate, load_checkpoint, ppo_update,
    pretrain_encoder, reconstruction_loss, recurrent_outputs, save_checkpoint,
)


class RecurrentPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(123)
        self.config = ModelConfig(context_dim=4, candidate_dim=3, hidden_size=32,
                                  visual_size=32, candidate_hidden_size=16, phase_count=3)
        self.model = RecurrentActorCritic(self.config)

    def rollout(self):
        time, batch, count = 4, 2, 3
        images = torch.rand(time, batch, 3, 96, 96)
        context = torch.rand(time, batch, 4) * 2 - 1
        phase = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 0]])
        candidates = torch.rand(time, batch, count, 3) * 2 - 1
        legal = torch.zeros(time, batch, 9, dtype=torch.bool)
        for step in range(time):
            for column in range(batch):
                legal[step, column, :9 if phase[step, column] == 0 else count] = True
        legal[0, 0, 7:] = False
        legal[1, 0, 2] = False
        valid = torch.ones(time, batch, dtype=torch.bool)
        valid[-1, 1] = False
        reset = torch.zeros_like(valid)
        reset[0] = True
        terminated = torch.zeros_like(valid)
        terminated[-1, 0] = True
        truncated = torch.zeros_like(valid)
        truncated[-2, 1] = True
        empty = torch.zeros(time, batch)
        rollout = RolloutBatch(images, context, phase, candidates, legal,
            torch.zeros(time, batch, dtype=torch.long), empty.clone(), empty.clone(),
            torch.tensor([[.1, -.1], [.2, .3], [0., .5], [1., 0.]]), empty.clone(),
            terminated, truncated, valid, reset, torch.full((time, batch), .2),
            self.model.initial_hidden(batch), self.model.policy_version(), "synthetic-rollout")
        with torch.no_grad():
            logits, values, _ = recurrent_outputs(self.model, rollout)
            actions = torch.multinomial(logits.softmax(-1).reshape(-1, 9), 1,
                                        generator=torch.Generator().manual_seed(88)).reshape(time, batch)
            logs = logits.log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
            next_values = torch.zeros_like(values)
            next_values[:-1] = values[1:]
            next_values[-2, 1] = .125  # synthetic true-final-observation bootstrap
            next_values[-1, 0] = 987  # true terminals must ignore bootstrap
        return replace(rollout, actions=actions, old_log_probs=logs, old_values=values,
                       next_values=next_values)

    def test_masks_zero_probability_and_masked_candidate_invariance(self):
        images = torch.rand(2, 3, 96, 96)
        context = torch.zeros(2, 4)
        phase = torch.tensor([0, 1])
        candidates = torch.rand(2, 3, 3)
        mask = torch.tensor([[True] * 9, [True, True] + [False] * 7])
        first = self.model.step(images, context, phase, candidates, mask)
        changed = candidates.clone()
        changed[0] = float("nan")  # movement ignores all candidates
        changed[1, 2] = float("nan")  # illegal candidate must be entirely absent
        second = self.model.step(images, context, phase, changed, mask)
        self.assertTrue(torch.equal(first.logits, second.logits))
        self.assertTrue(torch.equal(first.value, second.value))
        self.assertTrue(torch.equal(first.probabilities[~mask], torch.zeros_like(first.probabilities[~mask])))
        for _ in range(10):
            actions, logs = first.sample()
            self.assertTrue(mask.gather(-1, actions[:, None]).all())
            self.assertTrue(torch.isfinite(logs).all())
        with self.assertRaisesRegex(ValueError, "no legal"):
            self.model.step(images, context, phase, candidates, torch.zeros_like(mask))

    def test_candidate_permutation_equivariance(self):
        images = torch.rand(1, 3, 96, 96)
        context = torch.zeros(1, 4)
        phase = torch.ones(1, dtype=torch.long)
        candidates = torch.rand(1, 3, 3)
        mask = torch.tensor([[True] * 3 + [False] * 6])
        order = torch.tensor([2, 0, 1])
        before = self.model.step(images, context, phase, candidates, mask)
        after = self.model.step(images, context, phase, candidates[:, order], mask)
        self.assertTrue(torch.allclose(before.logits[:, order], after.logits[:, :3], atol=1e-7))
        self.assertTrue(torch.equal(before.value, after.value))
        self.assertTrue(torch.equal(before.next_hidden, after.next_hidden))

    def test_hidden_reset_and_burn_in_detach(self):
        rollout = self.rollout()
        arguments = (rollout.images[0], rollout.context[0], rollout.phase[0],
                     rollout.candidates[0], rollout.legal_mask[0])
        first = self.model.step(*arguments, torch.ones(2, 32), torch.ones(2, dtype=torch.bool))
        second = self.model.step(*arguments, torch.zeros(2, 32), torch.zeros(2, dtype=torch.bool))
        self.assertTrue(torch.equal(first.next_hidden, second.next_hidden))
        images = rollout.images.clone().requires_grad_()
        _, values, _ = recurrent_outputs(self.model, replace(rollout, images=images), burn_in=2)
        values[2:].sum().backward()
        self.assertEqual(float(images.grad[:2].abs().sum()), 0)
        self.assertGreater(float(images.grad[2, 0].abs().sum()), 0)
        self.assertEqual(float(images.grad[-1, 1].abs().sum()), 0)

    def test_gae_terminal_truncation_and_elapsed_discount(self):
        rewards = torch.tensor([[1.], [2.], [3.]])
        values = torch.tensor([[.5], [.6], [.7]])
        next_values = torch.tensor([[.6], [10.], [200.]])
        terminated = torch.tensor([[False], [False], [True]])
        truncated = torch.tensor([[False], [True], [False]])
        valid = torch.ones(3, 1, dtype=torch.bool)
        elapsed = torch.ones(3, 1)
        advantage, returns = generalized_advantage_estimate(rewards, values, next_values,
            terminated, truncated, valid, elapsed, gamma=.9, gae_lambda=.8)
        expected = torch.tensor([[1.04 + .9 * .8 * 10.4], [10.4], [2.3]])
        self.assertTrue(torch.allclose(advantage, expected, atol=1e-6))
        self.assertTrue(torch.allclose(returns, expected + values, atol=1e-6))
        timed, _ = generalized_advantage_estimate(rewards[:1], values[:1], next_values[:1],
            terminated[:1], truncated[:1], valid[:1], elapsed[:1] * 2,
            gamma=.9, gae_lambda=.8, discount_time_unit_seconds=1)
        self.assertAlmostEqual(float(timed[0, 0]), 1 + .9 ** 2 * .6 - .5, places=6)

    def test_ppo_changes_weights_and_rejects_offpolicy_reuse(self):
        rollout = self.rollout()
        before = self.model.policy_version()
        report = ppo_update(self.model, rollout, PPOConfig(epochs=2, minibatch_sequences=2))
        self.assertNotEqual(before, self.model.policy_version())
        self.assertEqual(report["optimizer_steps"], 2)
        self.assertFalse(report["performance_improvement_proven"])
        self.assertEqual(report["deployment_status"], "unapproved_candidate")
        with self.assertRaisesRegex(ValueError, "off-policy"):
            ppo_update(self.model, rollout)

    def test_padding_perturbations_cannot_change_update(self):
        rollout = self.rollout()
        other_model = deepcopy(self.model)
        changed = {}
        for name, tensor in rollout.__dict__.items():
            if not isinstance(tensor, torch.Tensor) or name in {"initial_hidden", "valid"}:
                continue
            value = tensor.clone()
            if value.is_floating_point():
                value[-1, 1] = float("nan")
            elif value.dtype == torch.long:
                value[-1, 1] = -999
            else:
                value[-1, 1] = ~value[-1, 1]
            changed[name] = value
        other = replace(rollout, **changed)
        config = PPOConfig(epochs=2, minibatch_sequences=2)
        ppo_update(self.model, rollout, config)
        ppo_update(other_model, other, config)
        self.assertEqual(self.model.policy_version(), other_model.policy_version())

    def test_invalid_rollouts_fail_before_weight_update(self):
        rollout = self.rollout()
        bad_logs = rollout.old_log_probs.clone()
        bad_logs[0, 0] += .1
        bad_bootstrap = rollout.next_values.clone()
        bad_bootstrap[0, 0] += 1
        bad_reset = rollout.reset.clone()
        bad_reset[1, 0] = True
        cases = (replace(rollout, behavior_version="other-policy"),
                 replace(rollout, split="evaluation"),
                 replace(rollout, old_log_probs=bad_logs),
                 replace(rollout, next_values=bad_bootstrap),
                 replace(rollout, old_values=rollout.old_values.clone().requires_grad_()),
                 replace(rollout, reset=bad_reset))
        before = self.model.policy_version()
        for invalid in cases:
            with self.subTest(invalid=invalid.behavior_version), self.assertRaises(ValueError):
                ppo_update(self.model, invalid)
            self.assertEqual(self.model.policy_version(), before)
            self.assertTrue(self.model.training)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_rollout_can_update_same_policy_on_cuda(self):
        rollout = self.rollout()
        self.model.to("cuda")
        report = ppo_update(self.model, rollout.to("cuda"), PPOConfig(epochs=1, minibatch_sequences=2))
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertTrue(report["final_kl_within_target"])

    def test_kl_early_stop_stops_repeated_updates(self):
        report = ppo_update(self.model, self.rollout(), PPOConfig(
            epochs=4, minibatch_sequences=2, learning_rate=.01, target_kl=1e-10))
        self.assertTrue(report["kl_early_stopped"])
        self.assertEqual(report["optimizer_steps"], 1)

    def test_checkpoint_roundtrip_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            save_checkpoint(self.model, path, {"data_split": "synthetic-test", "approved": False})
            restored, metadata = load_checkpoint(path)
            self.assertEqual(restored.policy_version(), self.model.policy_version())
            self.assertEqual(metadata["data_split"], "synthetic-test")
            rollout = self.rollout()
            with torch.no_grad():
                before = recurrent_outputs(self.model, rollout)
                after = recurrent_outputs(restored, rollout)
            for first, second in zip(before, after):
                self.assertTrue(torch.equal(first, second))
            with self.assertRaises(FileExistsError):
                save_checkpoint(self.model, path)
            payload = torch.load(path, weights_only=True)
            payload["state_dict"]["movement_head.bias"][0] += .1
            torch.save(payload, Path(directory) / "tampered.pt")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_checkpoint(Path(directory) / "tampered.pt")

    def test_old_frames_pretrain_only_encoder_and_decoder(self):
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        images = torch.randint(0, 256, (2, 3, 96, 96), dtype=torch.uint8)
        report = pretrain_encoder(self.model, [images], steps=2)
        changed = [name for name, value in self.model.state_dict().items()
                   if not torch.equal(value, before[name])]
        self.assertTrue(any(name.startswith("encoder.") for name in changed))
        self.assertTrue(all(name.startswith(("encoder.", "reconstruction_decoder.")) for name in changed))
        self.assertFalse(report["actions_used_as_labels"])
        self.assertTrue(torch.isfinite(reconstruction_loss(self.model, images)))
        self.assertEqual(self.model.reconstruct(images).shape, images.shape)


if __name__ == "__main__":
    unittest.main()
