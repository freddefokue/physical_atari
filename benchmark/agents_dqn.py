"""Benchmark-local RoboAtari-style DQN agent."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

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
class DQNAgentConfig:
    stack_size: int = 4
    obs_height: int = 84
    obs_width: int = 84
    buffer_size: int = 100_000
    batch_size: int = 32
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    train_start: int = 50_000
    train_freq: int = 4
    target_update_freq: int = 10_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_frames: int = 1_000_000
    grad_clip: Optional[float] = 10.0
    load_file: Optional[str] = None
    gpu: int = 0

    def __post_init__(self) -> None:
        if self.stack_size <= 0:
            raise ValueError("stack_size must be > 0")
        if self.obs_height <= 0 or self.obs_width <= 0:
            raise ValueError("obs_height and obs_width must be > 0")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0.0, 1.0]")
        if self.train_start < 0:
            raise ValueError("train_start must be >= 0")
        if self.train_freq <= 0:
            raise ValueError("train_freq must be > 0")
        if self.target_update_freq <= 0:
            raise ValueError("target_update_freq must be > 0")
        if self.epsilon_decay_frames <= 0:
            raise ValueError("epsilon_decay_frames must be > 0")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _ReplayBuffer:
    def __init__(self, capacity: int, stack_size: int, obs_shape: Tuple[int, int], seed: int) -> None:
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(int(seed))
        self._states = np.zeros((self.capacity, int(stack_size), *obs_shape), dtype=np.uint8)
        self._next_states = np.zeros((self.capacity, int(stack_size), *obs_shape), dtype=np.uint8)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    @property
    def size(self) -> int:
        return int(self._size)

    def add(self, state, action, reward, next_state, done) -> None:
        self._states[self._ptr] = state
        self._next_states[self._ptr] = next_state
        self._actions[self._ptr] = int(action)
        self._rewards[self._ptr] = float(reward)
        self._dones[self._ptr] = 1.0 if done else 0.0
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = self._rng.integers(0, self._size, size=int(batch_size))
        states = torch.from_numpy(self._states[idx]).to(device=device, dtype=torch.float32) / 255.0
        actions = torch.from_numpy(self._actions[idx]).to(device=device, dtype=torch.int64)
        rewards = torch.from_numpy(self._rewards[idx]).to(device=device, dtype=torch.float32)
        next_states = torch.from_numpy(self._next_states[idx]).to(device=device, dtype=torch.float32) / 255.0
        dones = torch.from_numpy(self._dones[idx]).to(device=device, dtype=torch.float32)
        return states, actions, rewards, next_states, dones


class _QNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    """Benchmark-local online DQN using raw-frame Carmack boundaries."""

    def __init__(
        self,
        *,
        data_dir: str,
        seed: int,
        num_actions: int,
        total_frames: int,
        config: Optional[DQNAgentConfig] = None,
    ) -> None:
        del data_dir, total_frames
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "agent=dqn requires torch. Install torch (CPU/CUDA build) or use --agent random/--agent repeat."
            ) from _TORCH_IMPORT_ERROR
        if num_actions <= 0:
            raise ValueError("num_actions must be > 0")

        self.seed = int(seed)
        self.num_actions = int(num_actions)
        self.config = config or DQNAgentConfig()

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

        self.network = _QNetwork(in_channels=int(self.config.stack_size), num_actions=self.num_actions).to(self.device)
        self.target_network = _QNetwork(in_channels=int(self.config.stack_size), num_actions=self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=float(self.config.learning_rate))
        self.replay = _ReplayBuffer(
            capacity=int(self.config.buffer_size),
            stack_size=int(self.config.stack_size),
            obs_shape=(int(self.config.obs_height), int(self.config.obs_width)),
            seed=self.seed,
        )

        self._resize_rows: Optional[np.ndarray] = None
        self._resize_cols: Optional[np.ndarray] = None
        self._source_hw: Optional[Tuple[int, int]] = None
        self._frame_stack: Optional[np.ndarray] = None
        self._last_state: Optional[np.ndarray] = None
        self._last_action = 0

        self.frame_count = 0
        self.training_steps = 0
        self.epsilon = float(self.config.epsilon_start)
        self.last_loss = 0.0
        self.loss_ema: Optional[float] = None
        self.last_avg_q = 0.0
        self.last_max_q = 0.0
        self._decision_steps = 0
        self._start_time_s = time.monotonic()

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
            "frame_count": int(self.frame_count),
            "training_steps": int(self.training_steps),
            "replay_size": int(self.replay.size),
            "epsilon": float(self.epsilon),
            "last_action_idx": int(self._last_action),
            "last_loss": float(self.last_loss),
            "loss_ema": None if self.loss_ema is None else float(self.loss_ema),
            "last_avg_q": float(self.last_avg_q),
            "last_max_q": float(self.last_max_q),
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

    def _stack_from_frame(self, frame_u8: np.ndarray) -> np.ndarray:
        if self._frame_stack is None:
            self._frame_stack = np.repeat(frame_u8[None, :, :], int(self.config.stack_size), axis=0)
        else:
            self._frame_stack = np.roll(self._frame_stack, shift=-1, axis=0)
            self._frame_stack[-1] = frame_u8
        return np.asarray(self._frame_stack, dtype=np.uint8)

    def _reset_stack(self, frame_u8: np.ndarray) -> np.ndarray:
        self._frame_stack = np.repeat(frame_u8[None, :, :], int(self.config.stack_size), axis=0)
        return np.asarray(self._frame_stack, dtype=np.uint8)

    def _epsilon_for_frame(self) -> float:
        frac = min(float(self.frame_count) / float(self.config.epsilon_decay_frames), 1.0)
        return float(self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start))

    def _select_action(self, state_u8: np.ndarray) -> int:
        self.epsilon = self._epsilon_for_frame()
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))

        state_t = torch.from_numpy(state_u8[None, ...]).to(self.device, dtype=torch.float32) / 255.0
        with torch.no_grad():
            q_values = self.network(state_t)
            self.last_avg_q = float(q_values.mean().item())
            self.last_max_q = float(q_values.max().item())
            action = int(torch.argmax(q_values, dim=1).item())
        return int(action)

    def _train_step(self) -> None:
        if self.replay.size < max(int(self.config.train_start), int(self.config.batch_size)):
            return
        if self.frame_count % int(self.config.train_freq) != 0:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(int(self.config.batch_size), self.device)
        q_values = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.network(next_states)
            next_actions = torch.argmax(next_q_online, dim=1)
            next_q_target = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + float(self.config.gamma) * next_q_target * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), float(self.config.grad_clip))
        self.optimizer.step()

        self.last_loss = float(loss.item())
        self.loss_ema = self.last_loss if self.loss_ema is None else 0.95 * float(self.loss_ema) + 0.05 * self.last_loss
        self.training_steps += 1

        if self.training_steps % int(self.config.target_update_freq) == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def frame(self, obs_rgb, reward, boundary) -> int:
        transition_obs = obs_rgb
        next_action_obs = obs_rgb
        if isinstance(boundary, Mapping):
            transition_obs = boundary.get("transition_obs_rgb", obs_rgb)
            next_action_obs = boundary.get("reset_obs_rgb", obs_rgb)

        transition_frame = self._preprocess_obs(np.asarray(transition_obs))
        transition_state = self._stack_from_frame(transition_frame)
        done = self._boundary_done(boundary)

        if self._last_state is not None:
            self.replay.add(
                state=self._last_state,
                action=int(self._last_action),
                reward=float(reward),
                next_state=transition_state,
                done=bool(done),
            )
            self._train_step()

        if done and isinstance(boundary, Mapping) and "reset_obs_rgb" in boundary:
            reset_frame = self._preprocess_obs(np.asarray(next_action_obs))
            current_state = self._reset_stack(reset_frame)
        else:
            current_state = transition_state

        action_idx = self._select_action(current_state)
        self._last_state = np.asarray(current_state, dtype=np.uint8).copy()
        self._last_action = int(action_idx)
        self._decision_steps += 1
        self.frame_count += 1

        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"DQN produced out-of-bounds action {action_idx} for action_space={self.num_actions}")
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
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)
