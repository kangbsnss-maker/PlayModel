"""Optional PyTorch recurrent PPO for local visual movement and menu decisions.

No game I/O, OCR truth inference, reward invention, or automatic deployment.
Rollout ``valid`` means a real, collector-verified policy transition; padding is
false. The collector must preserve observation/action/effect provenance and use
one unchanged policy for the rollout. Old demonstration or replay frames belong
only in ``pretrain_encoder``, never in on-policy PPO transitions.

Phase 0 has nine movement actions. Other phases score variable candidates with
the same recurrent state. Action width is max(9, K); masks are predecision facts.
Images are RGB uint8 or floats in [0,1]; context/candidates are finite normalized
features in [-1,1]. Unknown effects need explicit caller-provided feature flags.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .runtime_contract import RUNTIME_CONTRACT

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError("recurrent_ppo requires the optional local PyTorch dependency") from exc


SCHEMA = "playmodel.recurrent-ppo.v1"
BUILD_STATE_CONTEXT_SCHEMA = "brotato-observed-build-v2"
BUILD_STATE_CONTEXT_DIM = 64


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 96
    context_dim: int = 16
    candidate_dim: int = 16
    hidden_size: int = 128
    visual_size: int = 128
    phase_count: int = 5
    phase_embedding_size: int = 8
    candidate_hidden_size: int = 64

    def __post_init__(self):
        if self.image_size != 96:
            raise ValueError("this architecture uses 96x96 RGB inputs")
        for name, value in asdict(self).items():
            if type(value) is not int or not 1 <= value <= 1024:
                raise ValueError(f"invalid positive model dimension: {name}")
        if self.phase_count < 2:
            raise ValueError("movement and choice phases are required")


@dataclass
class PolicyOutput:
    logits: Tensor
    value: Tensor
    next_hidden: Tensor

    @property
    def probabilities(self) -> Tensor:
        return self.logits.softmax(dim=-1)

    def sample(self, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        actions = torch.multinomial(self.probabilities, 1, generator=generator).squeeze(-1)
        return actions, self.logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def _finite(tensor: Tensor, name: str, *, low: float | None = None, high: float | None = None) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains nonfinite values")
    if low is not None and (tensor < low).any():
        raise ValueError(f"{name} below {low}")
    if high is not None and (tensor > high).any():
        raise ValueError(f"{name} above {high}")


def _rgb(images: Tensor, size: int) -> Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, size, size):
        raise ValueError("images must have shape [batch,3,96,96]")
    if images.dtype == torch.uint8:
        return images.float() / 255
    if not images.is_floating_point():
        raise ValueError("images require uint8 or floating point")
    _finite(images, "images", low=0, high=1)
    return images.float()


class RecurrentActorCritic(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 6 * 6, c.visual_size), nn.ReLU())
        self.phase_embedding = nn.Embedding(c.phase_count, c.phase_embedding_size)
        self.memory = nn.GRUCell(c.visual_size + c.context_dim + c.phase_embedding_size, c.hidden_size)
        self.movement_head = nn.Linear(c.hidden_size, 9)
        self.candidate_encoder = nn.Sequential(nn.Linear(c.candidate_dim, c.candidate_hidden_size),
                                               nn.Tanh(), nn.Linear(c.candidate_hidden_size, c.candidate_hidden_size))
        self.choice_query = nn.Linear(c.hidden_size, c.candidate_hidden_size)
        self.candidate_bias = nn.Linear(c.candidate_hidden_size, 1)
        self.value_head = nn.Linear(c.hidden_size, 1)
        # Auxiliary image decoder is saved for reproducible reconstruction
        # diagnostics, but never participates in policy inference or PPO loss.
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(c.visual_size, 64 * 6 * 6), nn.ReLU(),
            nn.Unflatten(1, (64, 6, 6)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 8, 4, 2), nn.Sigmoid())
        # A small initial policy scale permits exploration without claiming a
        # hand-coded action table or a pretrained understanding of game rules.
        nn.init.orthogonal_(self.movement_head.weight, gain=.01)
        nn.init.zeros_(self.movement_head.bias)
        nn.init.orthogonal_(self.choice_query.weight, gain=.01)
        nn.init.zeros_(self.choice_query.bias)
        nn.init.zeros_(self.candidate_bias.weight)
        nn.init.zeros_(self.candidate_bias.bias)

    def initial_hidden(self, batch_size: int, *, device=None) -> Tensor:
        return torch.zeros(batch_size, self.config.hidden_size,
                           device=device if device is not None else next(self.parameters()).device)

    def reconstruct(self, images: Tensor) -> Tensor:
        """Auxiliary image reconstruction, not a predicted game outcome."""
        return self.reconstruction_decoder(self.encoder(_rgb(images, self.config.image_size)))

    def step(self, images: Tensor, context: Tensor, phase: Tensor, candidates: Tensor,
             legal_mask: Tensor, hidden: Tensor | None = None,
             reset: Tensor | None = None) -> PolicyOutput:
        c = self.config
        images = _rgb(images, c.image_size)
        batch = images.shape[0]
        if context.shape != (batch, c.context_dim) or not context.is_floating_point():
            raise ValueError("context shape/dtype mismatch")
        _finite(context, "context", low=-1, high=1)
        if phase.shape != (batch,) or phase.dtype != torch.long or (phase < 0).any() or (phase >= c.phase_count).any():
            raise ValueError("phase requires in-range int64 [batch]")
        if (candidates.ndim != 3 or candidates.shape[0] != batch
                or candidates.shape[2] != c.candidate_dim or not candidates.is_floating_point()
                or not 1 <= candidates.shape[1] <= 64):
            raise ValueError("candidate shape/dtype mismatch; require 1..64 slots")
        count = candidates.shape[1]
        width = max(9, count)
        if legal_mask.shape != (batch, width) or legal_mask.dtype != torch.bool:
            raise ValueError("legal_mask requires bool [batch,max(9,K)]")
        movement = phase == 0
        structural = torch.arange(width, device=phase.device)[None, :] < torch.where(movement, 9, count)[:, None]
        if (legal_mask & ~structural).any() or not legal_mask.any(-1).all():
            raise ValueError("invalid phase mask or no legal action; caller must abstain")
        candidate_legal = legal_mask[:, :count] & (~movement[:, None])
        # Illegal candidates carry no information or gradient, including NaNs.
        clean_candidates = torch.where(candidate_legal[:, :, None], candidates, torch.zeros_like(candidates))
        _finite(clean_candidates, "legal candidates", low=-1, high=1)
        if hidden is None:
            hidden = self.initial_hidden(batch, device=images.device)
        if hidden.shape != (batch, c.hidden_size) or not hidden.is_floating_point():
            raise ValueError("hidden state shape/dtype mismatch")
        _finite(hidden, "hidden")
        if reset is None:
            reset = torch.zeros(batch, dtype=torch.bool, device=images.device)
        if reset.shape != (batch,) or reset.dtype != torch.bool:
            raise ValueError("reset requires bool [batch]")
        hidden = torch.where(reset[:, None], torch.zeros_like(hidden), hidden)
        encoded = self.encoder(images)
        recurrent_input = torch.cat((encoded, context.float(), self.phase_embedding(phase)), dim=-1)
        next_hidden = self.memory(recurrent_input, hidden)
        movement_logits = self.movement_head(next_hidden)
        candidate_keys = self.candidate_encoder(clean_candidates)
        query = self.choice_query(next_hidden)
        choice_logits = (candidate_keys * query[:, None, :]).sum(-1) / math.sqrt(c.candidate_hidden_size)
        choice_logits = choice_logits + self.candidate_bias(candidate_keys).squeeze(-1)
        movement_logits = F.pad(movement_logits, (0, width - 9))
        choice_logits = F.pad(choice_logits, (0, width - count))
        logits = torch.where(movement[:, None], movement_logits, choice_logits)
        return PolicyOutput(logits.masked_fill(~legal_mask, -torch.inf),
                            self.value_head(next_hidden).squeeze(-1), next_hidden)

    def policy_version(self) -> str:
        digest = hashlib.sha256(json.dumps(asdict(self.config), sort_keys=True).encode())
        for name, tensor in sorted(self.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            # NumPy is an optional fast path; the exact-storage fallback preserves
            # this module's ability to operate with PyTorch alone.
            try:
                raw = value.numpy().tobytes()
            except RuntimeError:
                raw = bytes(value.clone().untyped_storage())
            digest.update(raw)
        return digest.hexdigest()


@dataclass(frozen=True)
class RolloutBatch:
    images: Tensor                 # [T,B,3,96,96], uint8 or [0,1] float
    context: Tensor                # [T,B,C]
    phase: Tensor                  # [T,B], int64
    candidates: Tensor             # [T,B,K,F], slot identity fixed per observation
    legal_mask: Tensor             # [T,B,max(9,K)], bool, before sampling
    actions: Tensor                # [T,B], int64, actual sampled/applied action
    old_log_probs: Tensor          # [T,B], from immutable behavior policy
    old_values: Tensor             # [T,B], same behavior policy
    rewards: Tensor                # [T,B], independently verified collector reward
    next_values: Tensor            # [T,B], true next/final observation, not reset observation
    terminated: Tensor             # [T,B], true environment terminal => no bootstrap
    truncated: Tensor              # [T,B], time limit => bootstrap, stop GAE trace
    valid: Tensor                  # [T,B], real verified transition vs padding
    reset: Tensor                  # [T,B], reset BEFORE this observation
    elapsed_seconds: Tensor        # [T,B], explicit decision-to-next-observation time
    initial_hidden: Tensor         # [B,H], logged behavior state before first observation
    behavior_version: str
    rollout_id: str
    split: str = "train"
    runtime_contract: str = RUNTIME_CONTRACT

    def to(self, device) -> RolloutBatch:
        return RolloutBatch(**{name: value.to(device) if isinstance(value, Tensor) else value
                               for name, value in self.__dict__.items()})

    def columns(self, indices: Tensor) -> RolloutBatch:
        return RolloutBatch(**{name: (value[:, indices] if name != "initial_hidden" else value[indices])
                               if isinstance(value, Tensor) else value
                               for name, value in self.__dict__.items()})


@dataclass(frozen=True)
class PPOConfig:
    epochs: int = 4
    minibatch_sequences: int = 4
    learning_rate: float = 3e-4
    clip_ratio: float = .2
    value_clip: float = .2
    value_coefficient: float = .5
    entropy_coefficient: float = .01
    max_grad_norm: float = .5
    target_kl: float = .02
    gamma: float = .99
    gae_lambda: float = .95
    discount_time_unit_seconds: float | None = None
    burn_in: int = 0
    normalize_advantages: bool = True
    seed: int = 0

    def __post_init__(self):
        for name in ("epochs", "minibatch_sequences"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.burn_in) is not int or self.burn_in < 0:
            raise ValueError("burn_in must be nonnegative")
        for name in ("learning_rate", "clip_ratio", "value_clip", "max_grad_norm", "target_kl"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")
        for name in ("gamma", "gae_lambda"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"invalid {name}")
        for name in ("value_coefficient", "entropy_coefficient"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"invalid {name}")
        if (self.discount_time_unit_seconds is not None
                and (not math.isfinite(self.discount_time_unit_seconds) or self.discount_time_unit_seconds <= 0)):
            raise ValueError("discount_time_unit_seconds must be positive or None")


def _validate_rollout(model: RecurrentActorCritic, rollout: RolloutBatch, config: PPOConfig) -> Tensor:
    if rollout.runtime_contract != RUNTIME_CONTRACT:
        raise ValueError("legacy or incompatible runtime rollout cannot enter current PPO")
    if rollout.split != "train":
        raise ValueError("evaluation/replay samples cannot enter PPO")
    if not rollout.rollout_id or rollout.behavior_version != model.policy_version():
        raise ValueError("off-policy rollout or missing rollout ID")
    if rollout.valid.ndim != 2 or rollout.valid.dtype != torch.bool or not rollout.valid.any():
        raise ValueError("valid must be nonempty bool [time,batch]")
    time, batch = rollout.valid.shape
    c = model.config
    count = rollout.candidates.shape[2] if rollout.candidates.ndim == 4 else 0
    shapes = {"images": (time, batch, 3, c.image_size, c.image_size),
              "context": (time, batch, c.context_dim), "candidates": (time, batch, count, c.candidate_dim),
              "legal_mask": (time, batch, max(9, count)), "initial_hidden": (batch, c.hidden_size)}
    for name, value in rollout.__dict__.items():
        if isinstance(value, Tensor):
            if tuple(value.shape) != shapes.get(name, (time, batch)):
                raise ValueError(f"rollout shape mismatch: {name}")
            if value.device != rollout.images.device:
                raise ValueError("all rollout tensors must be on the same device")
            if value.requires_grad:
                raise ValueError("recorded rollout tensors must be detached from autograd")
    for name in ("terminated", "truncated", "reset", "legal_mask"):
        if getattr(rollout, name).dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
    for name in ("actions", "phase"):
        if getattr(rollout, name).dtype != torch.long:
            raise ValueError(f"{name} must be int64")
    for name in ("old_log_probs", "old_values", "rewards", "next_values", "elapsed_seconds"):
        value = getattr(rollout, name)
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        _finite(value[rollout.valid], name)
    _finite(rollout.initial_hidden, "initial_hidden")
    if ((rollout.terminated & rollout.truncated) & rollout.valid).any():
        raise ValueError("a transition cannot be both terminated and truncated")
    if (rollout.elapsed_seconds[rollout.valid] <= 0).any():
        raise ValueError("elapsed transition time must be positive")
    if config.burn_in >= time:
        raise ValueError("burn_in leaves no optimization steps")
    for column in range(batch):
        positions = torch.where(rollout.valid[:, column])[0]
        if positions.numel() == 0:
            continue
        first, last = int(positions[0]), int(positions[-1])
        if positions.numel() != last - first + 1:
            raise ValueError("padding may not interrupt a recurrent sequence")
        if not bool(rollout.reset[first, column]) and (
                config.burn_in == 0 or not rollout.valid[:config.burn_in, column].all()):
            raise ValueError("mid-episode sequences require real burn-in observations or an initial reset")
        for step in range(first + 1, last + 1):
            ended = bool(rollout.terminated[step - 1, column] or rollout.truncated[step - 1, column])
            if ended != bool(rollout.reset[step, column]):
                raise ValueError("hidden resets must exactly match preceding episode boundaries")
        selected = rollout.actions[positions, column]
        if (selected < 0).any() or (selected >= max(9, count)).any():
            raise ValueError("action index out of bounds")
        if not rollout.legal_mask[positions, column].gather(-1, selected[:, None]).all():
            raise ValueError("recorded action was illegal under predecision mask")
    learning = rollout.valid.clone()
    learning[:config.burn_in] = False
    if not learning.any():
        raise ValueError("no valid optimization transitions after burn-in")
    return learning


def recurrent_outputs(model: RecurrentActorCritic, rollout: RolloutBatch,
                      *, burn_in: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """Reconstruct hidden state on each epoch; burn-in has no backward graph.

    Invalid padding is sanitized *before* the network and cannot alter memory.
    Hidden state at the burn-in boundary is detached for truncated BPTT. For a
    chunk starting mid-episode the logged initial state is an approximation after
    weights change; burn-in reduces, and does not magically remove, that error.
    """
    hidden = rollout.initial_hidden
    logits, values, states = [], [], []
    for step in range(rollout.valid.shape[0]):
        valid = rollout.valid[step]
        def clean(value: Tensor) -> Tensor:
            mask = valid.reshape(valid.shape[0], *((1,) * (value.ndim - 1)))
            return torch.where(mask, value, torch.zeros_like(value))
        legal = clean(rollout.legal_mask[step])
        legal[:, 0] |= ~valid
        if step == burn_in:
            hidden = hidden.detach()
        with torch.set_grad_enabled(torch.is_grad_enabled() and step >= burn_in):
            output = model.step(clean(rollout.images[step]), clean(rollout.context[step]),
                                clean(rollout.phase[step]), clean(rollout.candidates[step]),
                                legal, hidden, clean(rollout.reset[step]))
            hidden = torch.where(valid[:, None], output.next_hidden, hidden)
        logits.append(output.logits)
        values.append(output.value)
        states.append(hidden)
    return torch.stack(logits), torch.stack(values), torch.stack(states)


def generalized_advantage_estimate(rewards: Tensor, values: Tensor, next_values: Tensor,
                                  terminated: Tensor, truncated: Tensor, valid: Tensor,
                                  elapsed_seconds: Tensor, *, gamma: float = .99,
                                  gae_lambda: float = .95,
                                  discount_time_unit_seconds: float | None = None) -> tuple[Tensor, Tensor]:
    """Terminal: bootstrap zero. Truncation: bootstrap true final state, stop trace.

    Default gamma/lambda apply per recorded transition, independent of wall time.
    An explicit time unit instead uses gamma**(elapsed/unit) and lambda**(elapsed/unit).
    ``next_values`` at truncation must NOT be the value of a reset/new-run frame.
    """
    shape = valid.shape
    if valid.ndim != 2 or any(value.shape != shape for value in (
            rewards, values, next_values, terminated, truncated, elapsed_seconds)):
        raise ValueError("GAE inputs require matching [time,batch] shapes")
    if valid.dtype != torch.bool or terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("GAE validity and terminal masks must be boolean")
    if not 0 < gamma <= 1 or not 0 < gae_lambda <= 1:
        raise ValueError("invalid discount factors")
    if discount_time_unit_seconds is not None and (not math.isfinite(discount_time_unit_seconds) or discount_time_unit_seconds <= 0):
        raise ValueError("invalid discount time unit")
    for name, value in (("rewards", rewards), ("values", values), ("next_values", next_values), ("elapsed_seconds", elapsed_seconds)):
        _finite(value[valid], name)
    if (elapsed_seconds[valid] <= 0).any() or ((terminated & truncated) & valid).any():
        raise ValueError("invalid transition duration or terminal masks")
    with torch.no_grad():
        rewards, values, next_values = [torch.where(valid, x, torch.zeros_like(x)) for x in (rewards, values, next_values)]
        elapsed = torch.where(valid, elapsed_seconds, torch.ones_like(elapsed_seconds))
        exponent = elapsed / discount_time_unit_seconds if discount_time_unit_seconds else torch.ones_like(elapsed)
        discount = torch.pow(gamma, exponent)
        trace_discount = torch.pow(gae_lambda, exponent)
        delta = rewards + discount * torch.where(terminated, torch.zeros_like(next_values), next_values) - values
        advantages = torch.zeros_like(values)
        carry = torch.zeros_like(values[0])
        for step in reversed(range(shape[0])):
            continuation = ~(terminated[step] | truncated[step])
            if step + 1 < shape[0]:
                continuation &= valid[step + 1]
            carry = delta[step] + discount[step] * trace_discount[step] * continuation * carry
            carry = torch.where(valid[step], carry, torch.zeros_like(carry))
            advantages[step] = carry
        returns = torch.where(valid, advantages + values, torch.zeros_like(values))
    return advantages, returns


def ppo_update(model: RecurrentActorCritic, rollout: RolloutBatch,
               config: PPOConfig | None = None, *, optimizer=None,
               trajectory: RolloutBatch | None = None,
               transition_indices: Tensor | None = None) -> dict[str, Any]:
    """Train on one fixed-version rollout, using whole recurrent sequences.

    Numeric guards are not promotion gates. The caller must use independent
    full runs and a pinned comparison protocol before claiming improvement.
    """
    config = config or PPOConfig()
    if next(model.parameters()).device != rollout.images.device:
        raise ValueError("move model and rollout to the same training device")
    learning = _validate_rollout(model, rollout, config)
    source_version = rollout.behavior_version
    was_training = model.training
    # There is no dropout/batchnorm: checks are deterministic in either mode.
    # Avoid changing caller state when a corrupt rollout is rejected below.
    with torch.no_grad():
        old_logits, old_values, _ = recurrent_outputs(model, rollout, burn_in=config.burn_in)
        safe_actions = torch.where(rollout.valid, rollout.actions, torch.zeros_like(rollout.actions))
        recomputed = old_logits.log_softmax(-1).gather(-1, safe_actions.unsqueeze(-1)).squeeze(-1)
        if not torch.allclose(recomputed[rollout.valid], rollout.old_log_probs[rollout.valid], atol=2e-5, rtol=2e-5):
            raise ValueError("logged old probabilities do not match behavior policy/mask/hidden state")
        if not torch.allclose(old_values[rollout.valid], rollout.old_values[rollout.valid], atol=2e-5, rtol=2e-5):
            raise ValueError("logged old values do not match behavior policy")
        # Consecutive ordinary transitions must bootstrap the recorded next
        # observation. A truncated final observation is deliberately separate.
        for step in range(rollout.valid.shape[0] - 1):
            contiguous = (rollout.valid[step] & rollout.valid[step + 1]
                          & ~rollout.terminated[step] & ~rollout.truncated[step])
            if not torch.allclose(rollout.next_values[step][contiguous], rollout.old_values[step + 1][contiguous], atol=2e-5, rtol=2e-5):
                raise ValueError("bootstrap differs from the recorded next observation")
    advantages, returns, target_report = _advantage_targets(
        model, rollout, config, learning, trajectory, transition_indices)
    if config.normalize_advantages:
        selected = advantages[learning]
        deviation = selected.std(unbiased=False)
        # A one-sample or constant-advantage rollout must still have a policy
        # gradient; subtracting its mean would erase all actor learning.
        if selected.numel() > 1 and deviation > 1e-8:
            advantages = (advantages - selected.mean()) / (deviation + 1e-8)
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    expected_parameters = {id(p) for p in model.parameters()}
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != expected_parameters:
        raise ValueError("optimizer does not own exactly this model")
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    reports = []
    stopped = False
    updates = 0
    model.train()
    try:
        for epoch in range(config.epochs):
            order = torch.randperm(rollout.valid.shape[1], generator=generator).tolist()
            for start in range(0, len(order), config.minibatch_sequences):
                indices = torch.tensor(order[start:start + config.minibatch_sequences], device=rollout.images.device)
                mask = learning[:, indices]
                if not mask.any():
                    continue
                batch = rollout.columns(indices)
                logits, values, _ = recurrent_outputs(model, batch, burn_in=config.burn_in)
                selected_logits = logits[mask]
                log_probs = selected_logits.log_softmax(-1)
                action_log_probs = log_probs.gather(-1, batch.actions[mask].unsqueeze(-1)).squeeze(-1)
                old_log_probs = batch.old_log_probs[mask]
                log_ratio = action_log_probs - old_log_probs
                ratio = log_ratio.exp()
                approximate_kl = ((ratio - 1) - log_ratio).mean()
                if not torch.isfinite(approximate_kl):
                    raise ValueError("nonfinite PPO KL divergence")
                if approximate_kl.item() > config.target_kl:
                    stopped = True
                    break
                advantage = advantages[:, indices][mask]
                policy_loss = -torch.minimum(ratio * advantage,
                    ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantage).mean()
                baseline = batch.old_values[mask]
                clipped_value = baseline + (values[mask] - baseline).clamp(-config.value_clip, config.value_clip)
                targets = returns[:, indices][mask]
                value_loss = .5 * torch.maximum((values[mask] - targets).square(),
                                                (clipped_value - targets).square()).mean()
                legal = batch.legal_mask[mask]
                safe_log_probs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
                entropy = -(log_probs.exp() * safe_log_probs).sum(-1).mean()
                loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite PPO objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                reports.append({"epoch": epoch, "loss": loss.item(), "policy_loss": policy_loss.item(),
                                "value_loss": value_loss.item(), "entropy": entropy.item(),
                                "approximate_kl": approximate_kl.item(), "gradient_norm": float(grad_norm)})
            if stopped:
                break
    finally:
        model.train(was_training)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("PPO produced invalid parameters; discard candidate")
    with torch.no_grad():
        final_logits, _, _ = recurrent_outputs(model, rollout, burn_in=config.burn_in)
        final_logs = final_logits[learning].log_softmax(-1).gather(
            -1, rollout.actions[learning].unsqueeze(-1)).squeeze(-1)
        final_log_ratio = final_logs - rollout.old_log_probs[learning]
        final_kl = float((final_log_ratio.exp() - 1 - final_log_ratio).mean())
        if not math.isfinite(final_kl):
            raise ValueError("nonfinite final KL; discard candidate")
    return {"schema": SCHEMA, "runtime_contract": RUNTIME_CONTRACT,
            "source_version": source_version, "candidate_version": model.policy_version(),
            "rollout_id": rollout.rollout_id, "optimized_transitions": int(learning.sum()),
            "optimizer_steps": updates, "kl_early_stopped": stopped, "updates": reports,
            "final_approximate_kl": final_kl, "final_kl_within_target": final_kl <= config.target_kl,
            **target_report,
            "discount_semantics": "per_transition" if config.discount_time_unit_seconds is None else "elapsed_time",
            "deployment_status": "unapproved_candidate", "performance_improvement_proven": False}


def _advantage_targets(model, rollout, config, learning, trajectory, indices):
    if (trajectory is None) != (indices is None):
        raise ValueError("full trajectory and transition indices must be supplied together")
    source = trajectory if trajectory is not None else rollout
    if trajectory is not None:
        if (source.valid.shape[1] != 1 or not source.valid.all()
                or source.behavior_version != rollout.behavior_version
                or source.rollout_id != rollout.rollout_id or source.split != rollout.split):
            raise ValueError("full trajectory must be one complete, same-version chronological run")
        _validate_rollout(model, source, replace(config, burn_in=0))
        if indices.shape != rollout.valid.shape or indices.dtype != torch.long:
            raise ValueError("transition indices must be int64 [chunk_time,chunk_batch]")
        cpu_indices = indices.detach().cpu()
        cpu_valid = rollout.valid.cpu()
        selected = cpu_indices[cpu_valid]
        if ((selected < 0).any() or (selected >= source.valid.shape[0]).any()
                or (cpu_indices[~cpu_valid] != -1).any()):
            raise ValueError("invalid source indices or nonempty padding indices")
        learned = cpu_indices[learning.cpu()].sort().values
        if not torch.equal(learned, torch.arange(source.valid.shape[0])):
            raise ValueError("optimization must cover each original transition exactly once")
        # Only verified original data can generate targets. Arbitrary cached
        # advantage tensors are never accepted as a substitute for the trajectory.
        for name, value in rollout.__dict__.items():
            if isinstance(value, Tensor) and name != "initial_hidden":
                expected = getattr(source, name)[selected.to(source.images.device), 0].cpu()
                actual = value[rollout.valid].detach().cpu()
                if not torch.equal(actual, expected):
                    raise ValueError(f"chunk data differs from full trajectory: {name}")
        for step in range(source.valid.shape[0] - 1):
            if not bool(source.terminated[step, 0] or source.truncated[step, 0]):
                if not torch.allclose(source.next_values[step], source.old_values[step + 1], atol=2e-5, rtol=2e-5):
                    raise ValueError("full trajectory bootstrap seam mismatch")
    advantage, returns = generalized_advantage_estimate(
        source.rewards, source.old_values, source.next_values, source.terminated,
        source.truncated, source.valid, source.elapsed_seconds, gamma=config.gamma,
        gae_lambda=config.gae_lambda, discount_time_unit_seconds=config.discount_time_unit_seconds)
    if trajectory is not None:
        safe = cpu_indices.clamp(min=0).to(source.images.device)
        advantage = advantage[safe, 0].to(rollout.images.device)
        returns = returns[safe, 0].to(rollout.images.device)
        advantage = torch.where(rollout.valid, advantage, torch.zeros_like(advantage))
        returns = torch.where(rollout.valid, returns, torch.zeros_like(returns))
    provenance = {"gamma": config.gamma, "gae_lambda": config.gae_lambda,
                  "discount_time_unit_seconds": config.discount_time_unit_seconds,
                  "behavior_version": source.behavior_version, "rollout_id": source.rollout_id}
    for name in ("rewards", "old_values", "next_values", "terminated", "truncated", "valid", "elapsed_seconds"):
        value = getattr(source, name)
        provenance[name] = torch.where(source.valid, value, torch.zeros_like(value)).detach().cpu().tolist()
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return advantage, returns, {
        "gae_scope": "full_chronological_trajectory" if trajectory is not None else "chunk_bootstrap",
        "advantage_source_sha256": digest}


def pretrain_encoder(model: RecurrentActorCritic, batches: Iterable[Tensor], *, steps: int = 100,
                     learning_rate: float = 1e-3, device="cpu") -> dict[str, Any]:
    """Image-only autoencoder warmup; old actions/outcomes are never targets.

    Batches must come only from the declared training split. This function does
    not decide dataset membership and does not turn reconstruction into game-rule
    knowledge. The auxiliary decoder is saved for evaluation; policy/GRU heads
    stay unchanged and never invoke this decoder when choosing an action.
    """
    if type(steps) is not int or steps <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive pretraining steps and learning rate required")
    model.to(device)
    source_version = model.policy_version()
    decoder = model.reconstruction_decoder
    parameters = [*model.encoder.parameters(), *decoder.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    iterator = iter(batches)
    losses = []
    examples = 0
    previous_mode = model.training
    model.train()
    try:
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise ValueError("pretraining iterable is empty or not restartable") from exc
            images = _rgb(batch.to(device), model.config.image_size)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = decoder(model.encoder(images))
            loss = F.mse_loss(reconstructed, images)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite reconstruction loss")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            examples += images.shape[0]
    finally:
        model.train(previous_mode)
    return {"objective": "image_reconstruction_only", "steps": steps, "examples_seen": examples,
            "first_loss": losses[0], "last_loss": losses[-1], "source_version": source_version,
            "candidate_version": model.policy_version(), "actions_used_as_labels": False,
            "deployment_status": "unapproved_candidate", "performance_improvement_proven": False}


def reconstruction_loss(model: RecurrentActorCritic, images: Tensor) -> Tensor:
    """Caller controls eval/no_grad and train/evaluation dataset separation."""
    images = _rgb(images, model.config.image_size)
    return F.mse_loss(model.reconstruct(images), images)


def save_checkpoint(model: RecurrentActorCritic, path: Path | str,
                    metadata: dict[str, Any] | None = None) -> None:
    metadata = dict(metadata or {})
    # Limit metadata to plain JSON-compatible primitives before torch serialization.
    json.dumps(metadata, allow_nan=False)
    payload = {"schema": SCHEMA, "config": asdict(model.config), "version": model.policy_version(),
               "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
               "metadata": metadata, "deployment_status": "unapproved_candidate"}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    with path.open("xb") as stream:
        stream.write(buffer.getvalue())


def load_checkpoint(path: Path | str, *, device="cpu") -> tuple[RecurrentActorCritic, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("deployment_status") != "unapproved_candidate":
        raise ValueError("unsupported recurrent checkpoint")
    model = RecurrentActorCritic(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()) or model.policy_version() != payload.get("version"):
        raise ValueError("checkpoint weights/version mismatch")
    metadata = payload.get("metadata", {})
    json.dumps(metadata, allow_nan=False)
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    return model.to(device).eval(), metadata


def migrate_build_state_checkpoint(source: Path | str, target: Path | str) -> dict[str, Any]:
    """Create a distinct context64 warm start; no optimizer or rollout is used.

    The recurrent input is [visual, context, phase]. Extra context columns must
    be inserted BEFORE the phase embedding, whose learned columns move intact.
    The new inputs initially have zero effect and need fresh on-policy learning.
    Source metadata is ancestry, never evidence that this new policy was trained.
    """
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or target.exists():
        raise FileExistsError("Migration requires a new target; checkpoints are immutable")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    original, source_metadata = load_checkpoint(source, device="cpu")
    if original.config.context_dim != 16:
        raise ValueError("Build-state migration supports only legacy context16 checkpoints")
    if original.config.candidate_dim != 16 or original.config.phase_count < 4:
        raise ValueError("Build-state migration requires candidate16 and combat/shop phases")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
        raise ValueError("Source checkpoint changed during migration")
    migrated = RecurrentActorCritic(replace(original.config, context_dim=BUILD_STATE_CONTEXT_DIM))
    old_state = original.state_dict()
    new_state = {name: value.detach().clone() for name, value in old_state.items()}
    old_weight = old_state["memory.weight_ih"]
    prefix = original.config.visual_size + original.config.context_dim
    phase_start = original.config.visual_size + BUILD_STATE_CONTEXT_DIM
    expanded = torch.zeros_like(migrated.memory.weight_ih)
    expanded[:, :prefix] = old_weight[:, :prefix]
    expanded[:, phase_start:] = old_weight[:, prefix:]
    new_state["memory.weight_ih"] = expanded
    migrated.load_state_dict(new_state, strict=True)
    migrated.eval()
    source_version, target_version = original.policy_version(), migrated.policy_version()
    if source_version == target_version:
        raise ValueError("Expanded checkpoint must have a distinct behavior version")
    metadata = {
        "context_schema": BUILD_STATE_CONTEXT_SCHEMA,
        "context_layout": [
            {"name": "legacy_context", "start": 0, "width": 16},
            {"name": "observed_stat_values", "start": 16, "width": 16},
            {"name": "observed_stat_known", "start": 32, "width": 16},
            {"name": "weapon_lexical", "start": 48, "width": 8},
            {"name": "build_status", "start": 56, "width": 8}],
        "training_performed": False,
        "deployment_approved": False,
        "warm_start": {
            "schema": "playmodel.context-expansion.v1",
            "source_checkpoint": str(source), "source_sha256": source_sha,
            "source_policy_version": source_version, "target_policy_version": target_version,
            "source_context_dim": 16, "target_context_dim": BUILD_STATE_CONTEXT_DIM,
            "feature_schema": BUILD_STATE_CONTEXT_SCHEMA,
            "method": "preserve_existing_weights_insert_zero_context_columns",
            "additional_features_trained": False, "training_performed": False,
            "old_rollouts_reusable_for_ppo": False},
        "lineage": {"source_metadata": source_metadata}}
    # save_checkpoint uses exclusive create, including races after the initial check.
    save_checkpoint(migrated, target, metadata)
    reloaded, restored_metadata = load_checkpoint(target, device="cpu")
    if reloaded.policy_version() != target_version or restored_metadata != metadata:
        raise ValueError("Migrated checkpoint failed reload verification")
    return {"source_checkpoint": str(source), "checkpoint": str(target),
            "source_sha256": source_sha, "checkpoint_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source_policy_version": source_version, "policy_version": target_version,
            "context_dim": BUILD_STATE_CONTEXT_DIM, "context_schema": BUILD_STATE_CONTEXT_SCHEMA,
            "reload_verified": True, "training_performed": False,
            "additional_features_trained": False, "deployment_approved": False}
