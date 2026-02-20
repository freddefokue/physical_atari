"""Minimal online DQN baseline agent for streaming continual Atari runs."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TinyDQNConfig:
    """Configuration for :class:`TinyDQNAgent`."""

    gamma: float = 0.99
    lr: float = 1e-4
    buffer_size: int = 10_000
    batch_size: int = 32
    train_every_decisions: int = 4
    target_update_decisions: int = 250
    replay_min_size: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frames: int = 200_000
    obs_size: int = 84
    use_replay: bool = True
    device: str = "cpu"
    grad_clip_norm: float = 10.0
    train_log_interval: int = 500
    decision_interval: int = 1

    def __post_init__(self) -> None:
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0.0, 1.0]")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.train_every_decisions <= 0:
            raise ValueError("train_every_decisions must be > 0")
        if self.target_update_decisions <= 0:
            raise ValueError("target_update_decisions must be > 0")
        if self.replay_min_size < 0:
            raise ValueError("replay_min_size must be >= 0")
        if self.obs_size <= 0:
            raise ValueError("obs_size must be > 0")
        if self.eps_decay_frames <= 0:
            raise ValueError("eps_decay_frames must be > 0")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.train_log_interval < 0:
            raise ValueError("train_log_interval must be >= 0")
        if self.decision_interval <= 0:
            raise ValueError("decision_interval must be > 0")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class ReplayBuffer:
    """Simple fixed-size replay buffer storing uint8 observations."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(int(seed))
        self._obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self._next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return int(self._size)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self._obs[self._ptr] = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr] = int(action)
        self._rewards[self._ptr] = float(reward)
        self._dones[self._ptr] = 1.0 if done else 0.0
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self._rng.integers(0, self._size, size=int(batch_size))
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_obs[idx],
            self._dones[idx],
        )


class TinyQNet(nn.Module):
    """Small CNN producing Q-values over discrete action indices."""

    def __init__(self, action_space_n: int, obs_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, int(obs_size), int(obs_size), dtype=torch.float32)
            feature_dim = int(self.features(dummy).shape[1])
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, int(action_space_n)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TinyDQNAgent:
    """
    Minimal online DQN baseline for streaming runners.

    The agent is called every frame but learns on decision-interval transitions:
    it accumulates reward between decision boundaries and assigns that transition
    to the runner-provided `prev_applied_action_idx` so credit remains aligned
    with delayed, actually-executed actions. Intervals that never receive a
    valid applied-action label are skipped.
    """

    def __init__(
        self,
        action_space_n: int,
        seed: int = 0,
        config: Optional[TinyDQNConfig] = None,
    ) -> None:
        if action_space_n <= 0:
            raise ValueError("action_space_n must be > 0")
        self.action_space_n = int(action_space_n)
        self.seed = int(seed)
        self.config = config or TinyDQNConfig()

        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        self._online = TinyQNet(self.action_space_n, self.config.obs_size).to(self.device)
        self._target = TinyQNet(self.action_space_n, self.config.obs_size).to(self.device)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._optim = torch.optim.Adam(self._online.parameters(), lr=float(self.config.lr))

        obs_shape = (1, int(self.config.obs_size), int(self.config.obs_size))
        self._replay = ReplayBuffer(capacity=self.config.buffer_size, obs_shape=obs_shape, seed=self.seed)

        self._last_action_idx = 0
        self._frame_counter = 0
        self._decision_counter = 0
        self._finalized_transition_counter = 0
        self._train_steps = 0
        self._last_epsilon = float(self.config.eps_start)
        self._agent_start_time_s = time.monotonic()
        self._train_start_time_s = self._agent_start_time_s
        self._last_train_log_time_s = self._agent_start_time_s
        self._last_train_log_step = 0

        self._pending_decision_obs_u8: Optional[np.ndarray] = None
        self._pending_interval_reward = 0.0
        self._pending_interval_action_idx: Optional[int] = None
        self._pending_interval_done = False

        self._resize_rows: Optional[np.ndarray] = None
        self._resize_cols: Optional[np.ndarray] = None
        self._source_hw: Optional[Tuple[int, int]] = None

    @property
    def replay_size(self) -> int:
        return len(self._replay)

    @property
    def decision_steps(self) -> int:
        return int(self._decision_counter)

    @property
    def finalized_transition_counter(self) -> int:
        return int(self._finalized_transition_counter)

    @property
    def train_steps(self) -> int:
        return int(self._train_steps)

    @property
    def replay_min_size(self) -> int:
        return int(self.config.replay_min_size)

    @property
    def current_epsilon(self) -> float:
        return float(self._last_epsilon)

    def get_config(self) -> Dict[str, object]:
        return self.config.as_dict()

    def get_stats(self) -> Dict[str, object]:
        now_s = time.monotonic()
        elapsed_s = max(now_s - self._agent_start_time_s, 1e-9)
        return {
            "decision_steps": int(self._decision_counter),
            "finalized_transition_counter": int(self._finalized_transition_counter),
            "train_steps": int(self._train_steps),
            "replay_size": int(self.replay_size),
            "replay_min_size": int(self.config.replay_min_size),
            "current_epsilon": float(self.current_epsilon),
            "decision_steps_per_sec": float(self._decision_counter / elapsed_s),
            "train_steps_per_sec": float(self._train_steps / elapsed_s),
        }

    def _epsilon_for_frame(self, frame_idx: int) -> float:
        progress = min(1.0, max(0.0, float(frame_idx) / float(self.config.eps_decay_frames)))
        eps = float(self.config.eps_start + progress * (self.config.eps_end - self.config.eps_start))
        return float(max(0.0, min(1.0, eps)))

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
        gray = sampled.astype(np.uint16).sum(axis=2) // 3
        resized = gray.astype(np.uint8)
        return resized[None, :, :]

    def _select_action(self, obs_u8: np.ndarray, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return int(self._rng.integers(0, self.action_space_n))

        obs_t = torch.from_numpy(obs_u8)
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
        obs_t = obs_t.to(self.device, dtype=torch.float32) / 255.0
        with torch.no_grad():
            q_values = self._online(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    def _maybe_train(self) -> None:
        if not self.config.use_replay:
            return
        if self.replay_size < max(self.config.replay_min_size, self.config.batch_size):
            return
        if self._finalized_transition_counter % self.config.train_every_decisions != 0:
            return

        obs_u8, actions, rewards, next_obs_u8, dones = self._replay.sample(self.config.batch_size)

        obs_t = torch.from_numpy(obs_u8).to(self.device, dtype=torch.float32) / 255.0
        next_obs_t = torch.from_numpy(next_obs_u8).to(self.device, dtype=torch.float32) / 255.0
        actions_t = torch.from_numpy(actions).to(self.device, dtype=torch.int64)
        rewards_t = torch.from_numpy(rewards).to(self.device, dtype=torch.float32)
        dones_t = torch.from_numpy(dones).to(self.device, dtype=torch.float32)

        q = self._online(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self._target(next_obs_t).max(dim=1).values
            target_q = rewards_t + float(self.config.gamma) * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q, target_q)
        self._optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self._online.parameters(), float(self.config.grad_clip_norm))
        self._optim.step()
        self._train_steps += 1
        if self.config.train_log_interval > 0 and (self._train_steps % self.config.train_log_interval == 0):
            q_mean = float(q.detach().mean().item())
            target_q_mean = float(target_q.detach().mean().item())
            td_abs_mean = float((q.detach() - target_q.detach()).abs().mean().item())
            now_s = time.monotonic()
            delta_time_s = max(now_s - self._last_train_log_time_s, 1e-9)
            delta_steps = max(self._train_steps - self._last_train_log_step, 1)
            train_sps_window = float(delta_steps / delta_time_s)
            total_train_elapsed_s = max(now_s - self._train_start_time_s, 1e-9)
            train_sps_total = float(self._train_steps / total_train_elapsed_s)
            print(
                "[tinydqn] "
                f"train_step={self._train_steps} "
                f"replay_size={self.replay_size} "
                f"finalized_transitions={self._finalized_transition_counter} "
                f"epsilon={self.current_epsilon:.6f} "
                f"loss={float(loss.detach().item()):.6f} "
                f"q_mean={q_mean:.6f} "
                f"target_q_mean={target_q_mean:.6f} "
                f"td_abs_mean={td_abs_mean:.6f} "
                f"train_sps={train_sps_window:.2f} "
                f"train_sps_total={train_sps_total:.2f}",
                flush=True,
            )
            self._last_train_log_time_s = now_s
            self._last_train_log_step = int(self._train_steps)

    def _maybe_sync_target(self) -> None:
        """Sync target network on decision cadence, independent of training cadence."""

        if self._decision_counter <= 0:
            return
        if self._decision_counter % self.config.target_update_decisions != 0:
            return
        self._target.load_state_dict(self._online.state_dict())

    def step(self, obs_rgb: np.ndarray, reward: float, terminated: bool, truncated: bool, info: Dict[str, object]) -> int:
        done = bool(terminated or truncated)
        if "is_decision_frame" in info:
            is_decision_frame = bool(info.get("is_decision_frame"))
        else:
            is_decision_frame = bool(self._frame_counter % int(self.config.decision_interval) == 0)
        if is_decision_frame:
            # Count decisions at interval start so training cadence matches
            # the interval being finalized on this boundary.
            self._decision_counter += 1

        if "has_prev_applied_action" not in info or "prev_applied_action_idx" not in info:
            raise KeyError(
                "TinyDQNAgent requires info['has_prev_applied_action'] and "
                "info['prev_applied_action_idx'] for delay-aligned credit assignment."
            )
        has_prev_applied_action = bool(info["has_prev_applied_action"])
        prev_applied_action_idx = int(info["prev_applied_action_idx"])

        obs_u8: Optional[np.ndarray] = None

        if self._pending_decision_obs_u8 is not None:
            self._pending_interval_reward += float(reward)
            if (
                has_prev_applied_action
                and self._pending_interval_action_idx is None
                and 0 <= prev_applied_action_idx < self.action_space_n
            ):
                self._pending_interval_action_idx = int(prev_applied_action_idx)
            if done:
                self._pending_interval_done = True

            if is_decision_frame or done:
                obs_u8 = self._preprocess_obs(obs_rgb)
                transition_action = (
                    int(self._pending_interval_action_idx)
                    if self._pending_interval_action_idx is not None
                    else None
                )
                if self.config.use_replay and transition_action is not None and 0 <= transition_action < self.action_space_n:
                    self._replay.add(
                        obs=self._pending_decision_obs_u8,
                        action=transition_action,
                        reward=float(self._pending_interval_reward),
                        next_obs=obs_u8,
                        done=bool(self._pending_interval_done),
                    )
                    self._finalized_transition_counter += 1
                    self._maybe_train()
                self._pending_decision_obs_u8 = None
                self._pending_interval_reward = 0.0
                self._pending_interval_action_idx = None
                self._pending_interval_done = False

        if not is_decision_frame:
            self._frame_counter += 1
            return int(self._last_action_idx)

        if obs_u8 is None:
            obs_u8 = self._preprocess_obs(obs_rgb)

        frame_idx = int(info.get("global_frame_idx", self._frame_counter))
        epsilon = self._epsilon_for_frame(frame_idx)
        self._last_epsilon = float(epsilon)
        action_idx = self._select_action(obs_u8, epsilon)
        self._last_action_idx = int(action_idx)

        self._pending_decision_obs_u8 = obs_u8
        self._pending_interval_reward = 0.0
        self._pending_interval_action_idx = None
        self._pending_interval_done = False

        self._maybe_sync_target()
        self._frame_counter += 1
        return int(self._last_action_idx)
