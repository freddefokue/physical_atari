"""Minimal online PPO agent for streaming continual Atari runs."""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - exercised indirectly in dependency-missing tests
    import torch
    import torch.nn.functional as F
    from torch import nn

    _TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency path
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = exc


@dataclass
class PPOConfig:
    """Configuration for :class:`PPOAgent`."""

    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    train_interval: int = 128
    batch_size: int = 32
    epochs: int = 4
    reward_clip: float = 1.0
    obs_size: int = 84
    frame_stack: int = 4
    grayscale: bool = True
    normalize_advantages: bool = True
    deterministic_actions: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0.0, 1.0]")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0.0, 1.0]")
        if self.clip_range <= 0.0:
            raise ValueError("clip_range must be > 0")
        if self.ent_coef < 0.0:
            raise ValueError("ent_coef must be >= 0")
        if self.vf_coef < 0.0:
            raise ValueError("vf_coef must be >= 0")
        if self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be >= 0")
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be > 0")
        if self.train_interval <= 0:
            raise ValueError("train_interval must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.reward_clip < 0.0:
            raise ValueError("reward_clip must be >= 0")
        if self.obs_size <= 0:
            raise ValueError("obs_size must be > 0")
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be > 0")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _PPOModel(nn.Module):
    def __init__(self, in_channels: int, action_space_n: int, obs_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, int(obs_size), int(obs_size), dtype=torch.float32)
            feature_dim = int(self.features(dummy).shape[1])
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(256, int(action_space_n))
        self.value_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(self.features(x))
        logits = self.policy_head(h)
        values = self.value_head(h)
        return logits, values


class PPOAgent:
    """
    Practical streaming PPO baseline with robust safety checks.

    Designed for benchmark observability and correctness, not maximal throughput.
    """

    def __init__(
        self,
        action_space_n: int,
        seed: int = 0,
        config: Optional[PPOConfig] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "agent=ppo requires torch. Install torch (CPU or CUDA build) or use --agent random/repeat."
            ) from _TORCH_IMPORT_ERROR
        if action_space_n <= 0:
            raise ValueError("action_space_n must be > 0")

        self.action_space_n = int(action_space_n)
        self.seed = int(seed)
        self.config = config or PPOConfig()

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self._rng = np.random.default_rng(self.seed)
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu")

        channels_per_frame = 1 if self.config.grayscale else 3
        in_channels = int(self.config.frame_stack * channels_per_frame)
        self._model = _PPOModel(in_channels=in_channels, action_space_n=self.action_space_n, obs_size=self.config.obs_size).to(
            self.device
        )
        self._optim = torch.optim.Adam(self._model.parameters(), lr=float(self.config.learning_rate))

        self._start_time_s = time.monotonic()
        self._decision_steps = 0
        self._train_steps = 0
        self._train_updates = 0
        self._last_action_idx = 0
        self._last_policy_entropy: Optional[float] = None
        self._last_approx_kl: Optional[float] = None
        self._last_policy_loss: Optional[float] = None
        self._last_value_loss: Optional[float] = None
        self._last_total_loss: Optional[float] = None
        self._nan_guard_trigger_count = 0

        self._resize_rows: Optional[np.ndarray] = None
        self._resize_cols: Optional[np.ndarray] = None
        self._source_hw: Optional[Tuple[int, int]] = None
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=int(self.config.frame_stack))
        self._reset_buffer_next_obs = False

        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._prev_logprob: Optional[float] = None
        self._prev_probs: Optional[np.ndarray] = None
        self._prev_value: Optional[float] = None

        self._rollout_obs: List[np.ndarray] = []
        self._rollout_actions: List[int] = []
        self._rollout_logprobs: List[float] = []
        self._rollout_values: List[float] = []
        self._rollout_rewards: List[float] = []
        self._rollout_dones: List[float] = []

    def get_config(self) -> Dict[str, Any]:
        return {
            **self.config.as_dict(),
            "action_space_n": int(self.action_space_n),
            "seed": int(self.seed),
            "device_resolved": str(self.device),
        }

    def get_stats(self) -> Dict[str, Any]:
        elapsed_s = max(time.monotonic() - self._start_time_s, 1e-9)
        return {
            "decision_steps": int(self._decision_steps),
            "train_steps": int(self._train_steps),
            "train_updates": int(self._train_updates),
            "buffer_fill": int(len(self._rollout_rewards)),
            "rollout_progress": float(len(self._rollout_rewards) / max(1, int(self.config.rollout_steps))),
            "last_action_idx": int(self._last_action_idx),
            "policy_entropy": None if self._last_policy_entropy is None else float(self._last_policy_entropy),
            "approx_kl": None if self._last_approx_kl is None else float(self._last_approx_kl),
            "last_policy_loss": None if self._last_policy_loss is None else float(self._last_policy_loss),
            "last_value_loss": None if self._last_value_loss is None else float(self._last_value_loss),
            "last_total_loss": None if self._last_total_loss is None else float(self._last_total_loss),
            "nan_guard_trigger_count": int(self._nan_guard_trigger_count),
            "steps_per_sec": float(self._decision_steps / elapsed_s),
        }

    def _clip_reward(self, reward: float) -> float:
        if not np.isfinite(reward):
            self._nan_guard_trigger_count += 1
            return 0.0
        if self.config.reward_clip <= 0.0:
            return float(reward)
        lim = float(self.config.reward_clip)
        return float(np.clip(float(reward), -lim, lim))

    def _preprocess_obs(self, obs_rgb: np.ndarray) -> np.ndarray:
        if obs_rgb.dtype != np.uint8:
            obs_rgb = np.asarray(obs_rgb, dtype=np.uint8)
        height, width = int(obs_rgb.shape[0]), int(obs_rgb.shape[1])
        source_hw = (height, width)
        if self._source_hw != source_hw or self._resize_rows is None or self._resize_cols is None:
            self._source_hw = source_hw
            self._resize_rows = np.linspace(0, height - 1, self.config.obs_size, dtype=np.int32)
            self._resize_cols = np.linspace(0, width - 1, self.config.obs_size, dtype=np.int32)
        sampled = obs_rgb[self._resize_rows][:, self._resize_cols]
        if self.config.grayscale:
            gray = sampled.astype(np.uint16).sum(axis=2) // 3
            return gray.astype(np.uint8)[None, :, :]
        return np.transpose(sampled, (2, 0, 1)).astype(np.uint8)

    def _build_state(self, obs_rgb: np.ndarray) -> np.ndarray:
        frame = self._preprocess_obs(obs_rgb)
        if self._reset_buffer_next_obs:
            self._frame_buffer.clear()
            self._reset_buffer_next_obs = False
        if not self._frame_buffer:
            for _ in range(int(self.config.frame_stack)):
                self._frame_buffer.append(frame.copy())
        else:
            self._frame_buffer.append(frame.copy())
            while len(self._frame_buffer) < int(self.config.frame_stack):
                self._frame_buffer.append(frame.copy())
        return np.concatenate(list(self._frame_buffer), axis=0)

    def _policy(self, state_u8: np.ndarray) -> Tuple[int, float, float, float, np.ndarray]:
        state_t = torch.from_numpy(state_u8).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        with torch.no_grad():
            logits_t, value_t = self._model(state_t)
        logits = logits_t.detach().cpu().numpy()[0]
        value = float(value_t.squeeze(1).detach().cpu().item())

        if (not np.all(np.isfinite(logits))) or (not np.isfinite(value)):
            self._nan_guard_trigger_count += 1
            return 0, 0.0, 0.0, 0.0, np.ones((self.action_space_n,), dtype=np.float32) / float(self.action_space_n)

        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        denom = float(np.sum(probs))
        if (not np.isfinite(denom)) or denom <= 0.0:
            self._nan_guard_trigger_count += 1
            return 0, 0.0, 0.0, 0.0, np.ones((self.action_space_n,), dtype=np.float32) / float(self.action_space_n)
        probs = probs / denom
        if not np.all(np.isfinite(probs)):
            self._nan_guard_trigger_count += 1
            return 0, 0.0, 0.0, 0.0, np.ones((self.action_space_n,), dtype=np.float32) / float(self.action_space_n)

        if self.config.deterministic_actions:
            action_idx = int(np.argmax(probs))
        else:
            action_idx = int(self._rng.choice(self.action_space_n, p=probs))
        action_idx = int(np.clip(action_idx, 0, self.action_space_n - 1))
        p_action = float(max(probs[action_idx], 1e-8))
        logprob = float(np.log(p_action))
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
        return action_idx, logprob, value, entropy, probs.astype(np.float32, copy=False)

    def _append_transition(self, reward: float, done: bool, applied_action_idx: Optional[int]) -> None:
        if self._prev_state is None or self._prev_action is None or self._prev_value is None:
            return
        action_idx = int(self._prev_action)
        if applied_action_idx is not None and 0 <= int(applied_action_idx) < self.action_space_n:
            action_idx = int(applied_action_idx)

        if self._prev_probs is not None and self._prev_probs.shape[0] == self.action_space_n:
            logprob = float(np.log(max(float(self._prev_probs[action_idx]), 1e-8)))
        elif self._prev_logprob is not None and action_idx == int(self._prev_action):
            logprob = float(self._prev_logprob)
        else:
            self._nan_guard_trigger_count += 1
            logprob = 0.0

        self._rollout_obs.append(self._prev_state.copy())
        self._rollout_actions.append(int(action_idx))
        self._rollout_logprobs.append(float(logprob))
        self._rollout_values.append(float(self._prev_value))
        self._rollout_rewards.append(float(reward))
        self._rollout_dones.append(1.0 if done else 0.0)

    def _clear_rollout(self) -> None:
        self._rollout_obs.clear()
        self._rollout_actions.clear()
        self._rollout_logprobs.clear()
        self._rollout_values.clear()
        self._rollout_rewards.clear()
        self._rollout_dones.clear()

    def _maybe_train(self, bootstrap_value: float) -> None:
        num_steps = len(self._rollout_rewards)
        if num_steps < int(self.config.rollout_steps):
            return
        if self._decision_steps % int(self.config.train_interval) != 0:
            return
        obs_np = np.asarray(self._rollout_obs, dtype=np.uint8)
        actions_np = np.asarray(self._rollout_actions, dtype=np.int64)
        old_logprobs_np = np.asarray(self._rollout_logprobs, dtype=np.float32)
        values_np = np.asarray(self._rollout_values, dtype=np.float32)
        rewards_np = np.asarray(self._rollout_rewards, dtype=np.float32)
        dones_np = np.asarray(self._rollout_dones, dtype=np.float32)

        advantages = np.zeros_like(rewards_np, dtype=np.float32)
        gae = 0.0
        next_value = float(bootstrap_value)
        for idx in range(num_steps - 1, -1, -1):
            non_terminal = 1.0 - float(dones_np[idx])
            delta = float(rewards_np[idx]) + float(self.config.gamma) * next_value * non_terminal - float(values_np[idx])
            gae = delta + float(self.config.gamma) * float(self.config.gae_lambda) * non_terminal * gae
            advantages[idx] = float(gae)
            next_value = float(values_np[idx])
        returns_np = advantages + values_np

        obs_t = torch.from_numpy(obs_np).to(self.device, dtype=torch.float32) / 255.0
        actions_t = torch.from_numpy(actions_np).to(self.device, dtype=torch.int64)
        old_logprobs_t = torch.from_numpy(old_logprobs_np).to(self.device, dtype=torch.float32)
        returns_t = torch.from_numpy(returns_np).to(self.device, dtype=torch.float32)
        advantages_t = torch.from_numpy(advantages).to(self.device, dtype=torch.float32)

        if self.config.normalize_advantages and int(num_steps) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        batch_size = min(int(self.config.batch_size), int(num_steps))
        losses_total: List[float] = []
        losses_policy: List[float] = []
        losses_value: List[float] = []
        kls: List[float] = []
        entropies: List[float] = []

        for _ in range(int(self.config.epochs)):
            perm = self._rng.permutation(num_steps)
            for start in range(0, num_steps, batch_size):
                batch_idx = perm[start : start + batch_size]
                b = torch.as_tensor(batch_idx, device=self.device, dtype=torch.int64)
                logits_b, values_b = self._model(obs_t[b])
                if (not torch.isfinite(logits_b).all()) or (not torch.isfinite(values_b).all()):
                    self._nan_guard_trigger_count += 1
                    self._clear_rollout()
                    return

                dist_b = torch.distributions.Categorical(logits=logits_b)
                new_logprob_b = dist_b.log_prob(actions_t[b])
                entropy_b = dist_b.entropy().mean()
                values_b = values_b.squeeze(1)

                ratio = torch.exp(new_logprob_b - old_logprobs_t[b])
                surr1 = ratio * advantages_t[b]
                surr2 = torch.clamp(ratio, 1.0 - float(self.config.clip_range), 1.0 + float(self.config.clip_range)) * advantages_t[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_b, returns_t[b])
                loss = policy_loss + float(self.config.vf_coef) * value_loss - float(self.config.ent_coef) * entropy_b

                if (not torch.isfinite(loss)) or (not torch.isfinite(new_logprob_b).all()) or (not torch.isfinite(entropy_b)):
                    self._nan_guard_trigger_count += 1
                    self._clear_rollout()
                    return

                self._optim.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.max_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(self._model.parameters(), float(self.config.max_grad_norm))
                self._optim.step()

                self._train_steps += 1
                losses_total.append(float(loss.detach().cpu().item()))
                losses_policy.append(float(policy_loss.detach().cpu().item()))
                losses_value.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy_b.detach().cpu().item()))
                kl = (old_logprobs_t[b] - new_logprob_b).mean()
                kls.append(float(kl.detach().cpu().item()))

        self._train_updates += 1
        self._last_total_loss = float(np.mean(losses_total)) if losses_total else None
        self._last_policy_loss = float(np.mean(losses_policy)) if losses_policy else None
        self._last_value_loss = float(np.mean(losses_value)) if losses_value else None
        self._last_policy_entropy = float(np.mean(entropies)) if entropies else self._last_policy_entropy
        self._last_approx_kl = float(np.mean(kls)) if kls else self._last_approx_kl
        self._clear_rollout()

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        done = bool(terminated) or bool(truncated)
        self._decision_steps += 1

        reward_clipped = self._clip_reward(float(reward))
        state_u8 = self._build_state(np.asarray(obs_rgb))
        info_map = info if isinstance(info, dict) else {}
        has_prev_applied_action = bool(info_map.get("has_prev_applied_action", False))
        prev_applied_action_idx_raw = info_map.get("prev_applied_action_idx")
        prev_applied_action_idx: Optional[int] = None
        if has_prev_applied_action and prev_applied_action_idx_raw is not None:
            prev_applied_action_idx = int(prev_applied_action_idx_raw)
        self._append_transition(reward=reward_clipped, done=done, applied_action_idx=prev_applied_action_idx)

        action_idx, logprob, value, entropy, probs = self._policy(state_u8)
        if not np.isfinite(logprob) or not np.isfinite(value):
            self._nan_guard_trigger_count += 1
            action_idx, logprob, value, entropy = 0, 0.0, 0.0, 0.0
            probs = np.ones((self.action_space_n,), dtype=np.float32) / float(self.action_space_n)
        if action_idx < 0 or action_idx >= self.action_space_n:
            self._nan_guard_trigger_count += 1
            action_idx = 0
        self._last_action_idx = int(action_idx)
        self._last_policy_entropy = float(entropy)

        bootstrap_value = 0.0 if done else float(value)
        self._maybe_train(bootstrap_value=bootstrap_value)

        if done:
            # Do not bootstrap a new transition from terminal/truncation frames.
            self._prev_state = None
            self._prev_action = None
            self._prev_logprob = None
            self._prev_probs = None
            self._prev_value = None
            self._reset_buffer_next_obs = True
        else:
            self._prev_state = state_u8
            self._prev_action = int(action_idx)
            self._prev_logprob = float(logprob)
            self._prev_probs = np.asarray(probs, dtype=np.float32)
            self._prev_value = float(value)

        return int(action_idx)
