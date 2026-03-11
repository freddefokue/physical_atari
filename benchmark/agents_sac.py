"""Benchmark-local discrete SAC agent for streaming Atari."""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

import numpy as np

try:  # pragma: no cover - exercised via dependency-missing tests
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
class SACAgentConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    feature_dim: int = 512
    actor_hidden_dim: int = 256
    value_hidden_dim: int = 256
    frame_skip: int = 4
    n_stack: int = 4
    obs_height: int = 128
    obs_width: int = 128
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    gradient_steps: int = 1
    train_freq: int = 1
    tau: float = 0.005
    target_entropy_scale: float = 0.5
    eval_mode: bool = False
    load_file: Optional[str] = None
    gpu: int = 0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0.0, 1.0]")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        if self.actor_hidden_dim <= 0 or self.value_hidden_dim <= 0:
            raise ValueError("hidden dims must be > 0")
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be > 0")
        if self.n_stack <= 0:
            raise ValueError("n_stack must be > 0")
        if self.obs_height <= 0 or self.obs_width <= 0:
            raise ValueError("obs_height and obs_width must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")
        if self.gradient_steps <= 0:
            raise ValueError("gradient_steps must be > 0")
        if self.train_freq <= 0:
            raise ValueError("train_freq must be > 0")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0.0, 1.0]")
        if self.target_entropy_scale <= 0.0:
            raise ValueError("target_entropy_scale must be > 0")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], seed: int) -> None:
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(int(seed))
        self._obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self._next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        self._obs[self._ptr] = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr] = int(action)
        self._rewards[self._ptr] = float(reward)
        self._dones[self._ptr] = 1.0 if done else 0.0
        self._ptr = (self._ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = self._rng.integers(0, self.size, size=int(batch_size))
        obs = torch.from_numpy(self._obs[idx]).to(device=device, dtype=torch.float32) / 255.0
        next_obs = torch.from_numpy(self._next_obs[idx]).to(device=device, dtype=torch.float32) / 255.0
        actions = torch.from_numpy(self._actions[idx]).to(device=device, dtype=torch.int64)
        rewards = torch.from_numpy(self._rewards[idx]).to(device=device, dtype=torch.float32)
        dones = torch.from_numpy(self._dones[idx]).to(device=device, dtype=torch.float32)
        return obs, actions, rewards, next_obs, dones


class _Encoder(nn.Module):
    def __init__(self, n_stack: int, feature_dim: int, input_size: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_stack, input_size, input_size, dtype=torch.float32)
            conv_dim = int(self.conv(dummy).shape[1])
        self.fc = nn.Sequential(nn.Linear(conv_dim, feature_dim), nn.ReLU(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class _PolicyHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, num_actions))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _QHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, num_actions))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SACAgent:
    """Benchmark-local discrete SAC with benchmark-aware frame boundaries."""

    def __init__(
        self,
        *,
        data_dir: str,
        seed: int,
        num_actions: int,
        total_frames: int,
        config: Optional[SACAgentConfig] = None,
    ) -> None:
        del data_dir, total_frames
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "agent=sac requires torch. Install torch (CPU/CUDA build) or use --agent random/--agent repeat."
            ) from _TORCH_IMPORT_ERROR
        if num_actions <= 0:
            raise ValueError("num_actions must be > 0")

        self.seed = int(seed)
        self.num_actions = int(num_actions)
        self.config = config or SACAgentConfig()

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if torch.cuda.is_available() and int(self.config.gpu) >= 0:
            self.device = torch.device(f"cuda:{int(self.config.gpu)}")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.encoder = _Encoder(int(self.config.n_stack), int(self.config.feature_dim), int(self.config.obs_height)).to(self.device)
        self.actor = _PolicyHead(int(self.config.feature_dim), int(self.config.actor_hidden_dim), self.num_actions).to(self.device)
        self.q1 = _QHead(int(self.config.feature_dim), int(self.config.value_hidden_dim), self.num_actions).to(self.device)
        self.q2 = _QHead(int(self.config.feature_dim), int(self.config.value_hidden_dim), self.num_actions).to(self.device)
        self.q1_target = _QHead(int(self.config.feature_dim), int(self.config.value_hidden_dim), self.num_actions).to(self.device)
        self.q2_target = _QHead(int(self.config.feature_dim), int(self.config.value_hidden_dim), self.num_actions).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False

        self.q_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=float(self.config.learning_rate),
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(self.config.learning_rate))
        self.target_entropy = float(-np.log(1.0 / float(self.num_actions)) * float(self.config.target_entropy_scale))
        self.min_alpha = 1e-4
        self.max_alpha = 10.0
        self.log_alpha_min = float(np.log(self.min_alpha))
        self.log_alpha_max = float(np.log(self.max_alpha))
        self.log_alpha = torch.tensor([np.log(0.2)], requires_grad=True, device=self.device, dtype=torch.float32)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(self.config.learning_rate))

        self.replay = _ReplayBuffer(
            capacity=int(self.config.buffer_size),
            obs_shape=(int(self.config.n_stack), int(self.config.obs_height), int(self.config.obs_width)),
            seed=self.seed,
        )

        self._resize_rows: Optional[np.ndarray] = None
        self._resize_cols: Optional[np.ndarray] = None
        self._source_hw: Optional[Tuple[int, int]] = None
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=int(self.config.n_stack))
        self._accumulated_reward = 0.0
        self._last_obs: Optional[np.ndarray] = None
        self._last_action = 0

        self.step_count = 0
        self.training_step = 0
        self._decision_steps = 0
        self._start_time_s = time.monotonic()
        self.last_total_loss: Optional[float] = None
        self.last_actor_loss: Optional[float] = None
        self.last_q_loss: Optional[float] = None
        self.last_alpha: float = float(self._alpha(detach=True).item())
        self.last_entropy: Optional[float] = None

        if self.config.load_file is not None and os.path.exists(self.config.load_file):
            self.load_model(self.config.load_file)

    def get_config(self) -> Dict[str, Any]:
        return {
            **self.config.as_dict(),
            "device_resolved": str(self.device),
            "seed": int(self.seed),
            "num_actions": int(self.num_actions),
        }

    def get_stats(self) -> Dict[str, Any]:
        elapsed_s = max(time.monotonic() - self._start_time_s, 1e-9)
        return {
            "decision_steps": int(self._decision_steps),
            "step_count": int(self.step_count),
            "training_step": int(self.training_step),
            "replay_size": int(self.replay.size),
            "last_action_idx": int(self._last_action),
            "last_total_loss": None if self.last_total_loss is None else float(self.last_total_loss),
            "last_actor_loss": None if self.last_actor_loss is None else float(self.last_actor_loss),
            "last_q_loss": None if self.last_q_loss is None else float(self.last_q_loss),
            "alpha": float(self.last_alpha),
            "policy_entropy": None if self.last_entropy is None else float(self.last_entropy),
            "steps_per_sec": float(self._decision_steps / elapsed_s),
        }

    @staticmethod
    def _boundary_done(boundary: Any) -> bool:
        if isinstance(boundary, Mapping):
            return bool(boundary.get("end_of_episode_pulse", boundary.get("terminated", False) or boundary.get("truncated", False)))
        return bool(boundary)

    def _preprocess_obs(self, obs_rgb: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_rgb)
        if obs.dtype != np.uint8:
            obs = obs.astype(np.uint8)
        if obs.ndim != 3 or obs.shape[-1] < 3:
            raise ValueError(f"expected RGB observation, got shape {obs.shape}")

        height, width = int(obs.shape[0]), int(obs.shape[1])
        source_hw = (height, width)
        if self._source_hw != source_hw or self._resize_rows is None or self._resize_cols is None:
            self._source_hw = source_hw
            self._resize_rows = np.linspace(0, height - 1, int(self.config.obs_height), dtype=np.int32)
            self._resize_cols = np.linspace(0, width - 1, int(self.config.obs_width), dtype=np.int32)

        sampled = obs[self._resize_rows][:, self._resize_cols]
        gray = sampled.astype(np.uint16).sum(axis=2) // 3
        return np.asarray(gray, dtype=np.uint8)

    def _append_frame(self, frame_u8: np.ndarray) -> None:
        self._frame_buffer.append(np.asarray(frame_u8, dtype=np.uint8).copy())
        while len(self._frame_buffer) < int(self.config.n_stack):
            self._frame_buffer.append(np.asarray(frame_u8, dtype=np.uint8).copy())

    def _reset_buffer(self, frame_u8: np.ndarray) -> None:
        self._frame_buffer.clear()
        for _ in range(int(self.config.n_stack)):
            self._frame_buffer.append(np.asarray(frame_u8, dtype=np.uint8).copy())

    def _clear_episode_state(self) -> None:
        self._frame_buffer.clear()
        self._accumulated_reward = 0.0
        self._last_obs = None
        self._last_action = 0

    def _current_obs_stack(self) -> np.ndarray:
        return np.stack(list(self._frame_buffer), axis=0)

    def _policy(self, obs_stack_u8: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        obs_t = torch.from_numpy(obs_stack_u8[None, ...]).to(self.device, dtype=torch.float32) / 255.0
        with torch.no_grad():
            features = self.encoder(obs_t)
            logits = self.actor(features)
            probs = torch.softmax(logits, dim=1)
        probs_np = probs.squeeze(0).cpu().numpy()
        if bool(self.config.eval_mode):
            action = int(np.argmax(probs_np))
        else:
            action = int(np.random.choice(self.num_actions, p=probs_np))
        return action, probs_np, logits.squeeze(0).cpu().numpy()

    def _soft_update_targets(self) -> None:
        tau = float(self.config.tau)
        with torch.no_grad():
            for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
            for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def _alpha(self, *, detach: bool) -> torch.Tensor:
        alpha = self.log_alpha.clamp(self.log_alpha_min, self.log_alpha_max).exp()
        if detach:
            alpha = alpha.detach()
        return alpha

    def _train_batch(self) -> None:
        if self.replay.size < max(int(self.config.learning_starts), int(self.config.batch_size)):
            return

        obs, actions, rewards, next_obs, dones = self.replay.sample(int(self.config.batch_size), self.device)

        with torch.no_grad():
            next_features = self.encoder(next_obs)
            next_logits = self.actor(next_features)
            next_log_probs = torch.log_softmax(next_logits, dim=1)
            next_probs = torch.softmax(next_logits, dim=1)
            target_q = torch.min(self.q1_target(next_features), self.q2_target(next_features))
            alpha = self._alpha(detach=True)
            target_v = torch.sum(next_probs * (target_q - alpha * next_log_probs), dim=1)
            td_target = rewards + float(self.config.gamma) * (1.0 - dones) * target_v

        features = self.encoder(obs)
        q1_values = self.q1(features).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_values = self.q2(features).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1_values, td_target) + F.mse_loss(q2_values, td_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=10.0)
        self.q_optimizer.step()

        current_features = self.encoder(obs)
        logits = self.actor(current_features.detach())
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        with torch.no_grad():
            q_min = torch.min(self.q1(current_features), self.q2(current_features))
            alpha = self._alpha(detach=True)
        actor_loss = torch.sum(probs * (alpha * log_probs - q_min), dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        with torch.no_grad():
            entropy = -(probs * log_probs).sum(dim=1).mean()

        target_entropy_t = torch.as_tensor(float(self.target_entropy), device=self.device, dtype=torch.float32)
        alpha_loss = self._alpha(detach=False) * (entropy.detach() - target_entropy_t)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.log_alpha.data.clamp_(self.log_alpha_min, self.log_alpha_max)

        self.last_alpha = float(self._alpha(detach=True).item())
        self.last_entropy = float(entropy.item())
        self.last_actor_loss = float(actor_loss.item())
        self.last_q_loss = float(q_loss.item())
        self.last_total_loss = float(actor_loss.item() + q_loss.item())
        self.training_step += 1
        self._soft_update_targets()

    def frame(self, obs_rgb, reward, boundary) -> int:
        transition_obs = obs_rgb
        next_action_obs = obs_rgb
        has_reset_obs = False
        if isinstance(boundary, Mapping):
            transition_obs = boundary.get("transition_obs_rgb", obs_rgb)
            next_action_obs = boundary.get("reset_obs_rgb", obs_rgb)
            has_reset_obs = "reset_obs_rgb" in boundary

        self.step_count += 1
        clipped_reward = float(np.clip(float(reward), -1.0, 1.0))
        done = self._boundary_done(boundary)
        transition_frame = self._preprocess_obs(np.asarray(transition_obs))
        self._append_frame(transition_frame)
        self._accumulated_reward += clipped_reward

        if self.step_count % int(self.config.frame_skip) != 0:
            if done:
                reset_source = next_action_obs if has_reset_obs else transition_obs
                reset_frame = self._preprocess_obs(np.asarray(reset_source))
                self._clear_episode_state()
                self._append_frame(reset_frame)
                self._accumulated_reward += clipped_reward
            return int(self._last_action)

        obs_stack = self._current_obs_stack()

        if self._last_obs is not None:
            self.replay.add(
                obs=self._last_obs,
                action=int(self._last_action),
                reward=float(self._accumulated_reward),
                next_obs=obs_stack,
                done=bool(done),
            )
            if not bool(self.config.eval_mode) and self.replay.size >= max(int(self.config.learning_starts), int(self.config.batch_size)):
                if (self.step_count // int(self.config.frame_skip)) % int(self.config.train_freq) == 0:
                    for _ in range(int(self.config.gradient_steps)):
                        self._train_batch()

        if done and isinstance(boundary, Mapping) and "reset_obs_rgb" in boundary:
            reset_frame = self._preprocess_obs(np.asarray(next_action_obs))
            self._reset_buffer(reset_frame)
            obs_stack = self._current_obs_stack()

        action_idx, _probs, _logits = self._policy(obs_stack)
        self._last_obs = np.asarray(obs_stack, dtype=np.uint8).copy()
        self._last_action = int(action_idx)
        self._accumulated_reward = 0.0
        self._decision_steps += 1

        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"SAC produced out-of-bounds action {action_idx} for action_space={self.num_actions}")
        return int(action_idx)

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        boundary: Dict[str, Any] = {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "end_of_episode_pulse": bool(terminated) or bool(truncated),
        }
        if isinstance(info, Mapping):
            for key in ("transition_obs_rgb", "reset_obs_rgb", "boundary_cause", "termination_reason", "env_termination_reason"):
                if key in info:
                    boundary[key] = info[key]
        return self.frame(obs_rgb, reward, boundary)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if "encoder" in checkpoint:
                self.encoder.load_state_dict(checkpoint["encoder"])
                self.actor.load_state_dict(checkpoint["actor"])
                self.q1.load_state_dict(checkpoint["q1"])
                self.q2.load_state_dict(checkpoint["q2"])
                self.q1_target.load_state_dict(checkpoint.get("q1_target", checkpoint["q1"]))
                self.q2_target.load_state_dict(checkpoint.get("q2_target", checkpoint["q2"]))
                if "log_alpha" in checkpoint:
                    self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
                    self.log_alpha.data.clamp_(self.log_alpha_min, self.log_alpha_max)
                self.last_alpha = float(self._alpha(detach=True).item())
                return
            if "cnn" in checkpoint:
                self.encoder.load_state_dict(checkpoint["cnn"])
                self.actor.load_state_dict(checkpoint["actor"])
                self.q1.load_state_dict(checkpoint["q1"])
                self.q2.load_state_dict(checkpoint["q2"])
                self.q1_target.load_state_dict(checkpoint.get("q1_target", checkpoint["q1"]))
                self.q2_target.load_state_dict(checkpoint.get("q2_target", checkpoint["q2"]))
                if "log_alpha" in checkpoint:
                    self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
                    self.log_alpha.data.clamp_(self.log_alpha_min, self.log_alpha_max)
                self.last_alpha = float(self._alpha(detach=True).item())
                return
            if "model_state_dict" in checkpoint:
                self.encoder.load_state_dict(checkpoint["model_state_dict"])
                return
        raise ValueError("Unsupported SAC checkpoint format")

    def save_model(self, path: str) -> None:
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )
