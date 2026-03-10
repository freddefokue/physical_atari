"""Benchmark-local RoboAtari-style Rainbow DQN agent."""

from __future__ import annotations

import math
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
class RainbowDQNConfig:
    stack_size: int = 4
    obs_height: int = 84
    obs_width: int = 84
    buffer_size: int = 100_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99
    train_start: int = 50_000
    train_freq: int = 1
    target_update_freq: int = 2_000
    epsilon_start: float = 0.0
    epsilon_end: float = 0.0
    epsilon_decay_frames: int = 1_000_000
    grad_clip: Optional[float] = 10.0
    n_step: int = 3
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    priority_alpha: float = 0.5
    priority_beta: float = 0.4
    priority_beta_increment: float = 1e-6
    priority_eps: float = 1e-6
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
        if self.n_step <= 0:
            raise ValueError("n_step must be > 0")
        if self.num_atoms <= 1:
            raise ValueError("num_atoms must be > 1")
        if self.v_max <= self.v_min:
            raise ValueError("v_max must be > v_min")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        return noise.sign().mul_(noise.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class _SumTree:
    def __init__(self, capacity: int):
        self.capacity = 1
        while self.capacity < int(capacity):
            self.capacity *= 2
        self.tree = torch.zeros(2 * self.capacity, dtype=torch.float32)

    def update(self, idx: int, priority: float):
        tree_idx = int(idx) + self.capacity
        self.tree[tree_idx] = float(priority)
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]
            tree_idx //= 2

    def total(self) -> torch.Tensor:
        return self.tree[1]

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if float(self.tree[left].item()) >= float(prefixsum):
                idx = left
            else:
                prefixsum -= float(self.tree[left].item())
                idx = left + 1
        return int(idx - self.capacity)


class _PrioritizedReplay:
    def __init__(
        self,
        capacity: int,
        stack_size: int,
        obs_shape: Tuple[int, int],
        alpha: float,
        beta: float,
        beta_increment: float,
        priority_eps: float,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.stack_size = int(stack_size)
        self.obs_shape = obs_shape
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_increment = float(beta_increment)
        self.priority_eps = float(priority_eps)
        self.device = device

        self.states = np.zeros((self.capacity, self.stack_size, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((self.capacity, self.stack_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)
        self.tree = _SumTree(self.capacity)
        self.max_priority = 1.0
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = bool(done)
        self.tree.update(self.ptr, self.max_priority ** self.alpha)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = []
        priorities = []
        segment = float(self.tree.total().item()) / float(batch_size)
        for i in range(int(batch_size)):
            a = segment * i
            b = segment * (i + 1)
            value = float(np.random.uniform(a, b))
            idx = min(self.find_prefixsum_idx(value), self.size - 1)
            indices.append(idx)
            priorities.append(float(self.tree.tree[idx + self.tree.capacity].item()))

        indices_np = np.asarray(indices, dtype=np.int64)
        priorities_np = np.asarray(priorities, dtype=np.float32)
        probs = priorities_np / max(float(self.tree.total().item()), 1e-12)
        weights = (self.size * probs) ** (-self.beta)
        weights /= max(float(weights.max()), 1e-12)
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            indices_np,
            torch.from_numpy(self.states[indices_np]).to(self.device, dtype=torch.float32) / 255.0,
            torch.from_numpy(self.actions[indices_np]).to(self.device, dtype=torch.int64),
            torch.from_numpy(self.rewards[indices_np]).to(self.device, dtype=torch.float32),
            torch.from_numpy(self.next_states[indices_np]).to(self.device, dtype=torch.float32) / 255.0,
            torch.from_numpy(self.dones[indices_np].astype(np.float32)).to(self.device, dtype=torch.float32),
            torch.from_numpy(weights).to(self.device, dtype=torch.float32),
        )

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        return self.tree.find_prefixsum_idx(prefixsum)

    def update_priorities(self, indices: np.ndarray, priorities: torch.Tensor):
        priorities_np = priorities.detach().cpu().numpy()
        for idx, priority in zip(indices, priorities_np):
            priority_value = float(priority) + self.priority_eps
            self.tree.update(int(idx), priority_value ** self.alpha)
            self.max_priority = max(self.max_priority, priority_value)


class _RainbowNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, num_atoms: int) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_atoms = int(num_atoms)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc_input_dim = 64 * 7 * 7
        self.value_stream = nn.Sequential(_NoisyLinear(self.fc_input_dim, 512), nn.ReLU(), _NoisyLinear(512, self.num_atoms))
        self.adv_stream = nn.Sequential(
            _NoisyLinear(self.fc_input_dim, 512),
            nn.ReLU(),
            _NoisyLinear(512, self.num_actions * self.num_atoms),
        )

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, _NoisyLinear):
                module.reset_noise()

    def forward(self, x: torch.Tensor):
        features = self.conv(x).view(x.size(0), -1)
        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.adv_stream(features).view(-1, self.num_actions, self.num_atoms)
        q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        log_probs = F.log_softmax(q_atoms, dim=2)
        probs = torch.exp(log_probs)
        return probs, log_probs

    def q_values(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        probs, _ = self.forward(x)
        return torch.sum(probs * support.view(1, 1, -1), dim=2)


class RainbowDQNAgent:
    """Benchmark-local online Rainbow DQN using raw-frame Carmack boundaries."""

    def __init__(
        self,
        *,
        data_dir: str,
        seed: int,
        num_actions: int,
        total_frames: int,
        config: Optional[RainbowDQNConfig] = None,
    ) -> None:
        del data_dir, total_frames
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "agent=rainbow_dqn requires torch. Install torch (CPU/CUDA build) or use --agent random/--agent repeat."
            ) from _TORCH_IMPORT_ERROR
        if num_actions <= 0:
            raise ValueError("num_actions must be > 0")

        self.seed = int(seed)
        self.num_actions = int(num_actions)
        self.config = config or RainbowDQNConfig()

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

        self.network = _RainbowNetwork(int(self.config.stack_size), self.num_actions, int(self.config.num_atoms)).to(self.device)
        self.target_network = _RainbowNetwork(int(self.config.stack_size), self.num_actions, int(self.config.num_atoms)).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=float(self.config.learning_rate))
        self.replay = _PrioritizedReplay(
            capacity=int(self.config.buffer_size),
            stack_size=int(self.config.stack_size),
            obs_shape=(int(self.config.obs_height), int(self.config.obs_width)),
            alpha=float(self.config.priority_alpha),
            beta=float(self.config.priority_beta),
            beta_increment=float(self.config.priority_beta_increment),
            priority_eps=float(self.config.priority_eps),
            device=self.device,
        )

        self.support = torch.linspace(float(self.config.v_min), float(self.config.v_max), int(self.config.num_atoms), device=self.device)
        self.delta_z = float(self.config.v_max - self.config.v_min) / float(self.config.num_atoms - 1)
        self._resize_rows: Optional[np.ndarray] = None
        self._resize_cols: Optional[np.ndarray] = None
        self._source_hw: Optional[Tuple[int, int]] = None
        self._frame_stack: Optional[np.ndarray] = None
        self._last_state: Optional[np.ndarray] = None
        self._last_action = 0
        self._n_step_buffer: Deque[Tuple[np.ndarray, int, float]] = deque(maxlen=int(self.config.n_step))

        self.frame_count = 0
        self.training_steps = 0
        self.epsilon = float(self.config.epsilon_start)
        self.last_loss = 0.0
        self.loss_ema: Optional[float] = None
        self.last_avg_q = 0.0
        self.last_max_q = 0.0
        self.last_td_error = 0.0
        self.last_grad_norm = 0.0
        self._decision_steps = 0
        self._start_time_s = time.monotonic()

        if self.config.load_file is not None:
            if not os.path.exists(self.config.load_file):
                raise FileNotFoundError(f"Rainbow DQN checkpoint not found: {self.config.load_file}")
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
            "last_td_error": float(self.last_td_error),
            "last_grad_norm": float(self.last_grad_norm),
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
            self.network.reset_noise()
            q_values = self.network.q_values(state_t, self.support)
            self.last_avg_q = float(q_values.mean().item())
            self.last_max_q = float(q_values.max().item())
            action = int(torch.argmax(q_values, dim=1).item())
        return int(action)

    def _store_n_step(self, next_state: np.ndarray, done: bool) -> None:
        if not self._n_step_buffer:
            return
        cumulative_reward = 0.0
        for idx, (_, _, reward) in enumerate(self._n_step_buffer):
            cumulative_reward += (float(self.config.gamma) ** idx) * float(reward)
        state, action, _ = self._n_step_buffer[0]
        self.replay.add(state, action, cumulative_reward, next_state, done)

    def _append_transition(self, reward: float, next_state: np.ndarray, done: bool) -> None:
        if self._last_state is None:
            return
        self._n_step_buffer.append((self._last_state.copy(), int(self._last_action), float(reward)))
        if len(self._n_step_buffer) == int(self.config.n_step):
            self._store_n_step(next_state, done)
            self._n_step_buffer.popleft()
        if done:
            while self._n_step_buffer:
                self._store_n_step(next_state, done)
                self._n_step_buffer.popleft()

    def _train_step(self) -> None:
        if self.replay.size < max(int(self.config.train_start), int(self.config.batch_size)):
            return
        if self.frame_count % int(self.config.train_freq) != 0:
            return

        indices, states, actions, rewards, next_states, dones, weights = self.replay.sample(int(self.config.batch_size))
        self.network.reset_noise()
        self.target_network.reset_noise()

        with torch.no_grad():
            next_probs, _ = self.network(next_states)
            next_q_values = torch.sum(next_probs * self.support.view(1, 1, -1), dim=2)
            next_actions = torch.argmax(next_q_values, dim=1)

            target_probs, _ = self.target_network(next_states)
            target_probs = target_probs[torch.arange(int(self.config.batch_size)), next_actions]
            tz = rewards.unsqueeze(1) + (float(self.config.gamma) ** int(self.config.n_step)) * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)
            tz = tz.clamp(float(self.config.v_min), float(self.config.v_max))
            b = (tz - float(self.config.v_min)) / self.delta_z
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)

            proj_dist = torch.zeros_like(target_probs)
            offset = torch.arange(
                0,
                int(self.config.batch_size) * int(self.config.num_atoms),
                int(self.config.num_atoms),
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(1)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_probs * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_probs * (b - l.float())).view(-1))

        self.network.reset_noise()
        probs, log_probs = self.network(states)
        log_p = log_probs[torch.arange(int(self.config.batch_size)), actions]
        sample_losses = -(proj_dist * log_p).sum(dim=1)
        loss = (sample_losses * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        total_norm_sq = 0.0
        for param in self.network.parameters():
            if param.grad is not None:
                grad_norm = float(param.grad.detach().data.norm(2).item())
                total_norm_sq += grad_norm * grad_norm
        self.last_grad_norm = float(math.sqrt(total_norm_sq))
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), float(self.config.grad_clip))
        self.optimizer.step()

        self.last_loss = float(loss.item())
        self.loss_ema = self.last_loss if self.loss_ema is None else 0.95 * float(self.loss_ema) + 0.05 * self.last_loss
        self.last_td_error = float(sample_losses.detach().mean().item())
        self.training_steps += 1
        self.replay.update_priorities(indices, sample_losses.detach().abs())

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

        self._append_transition(float(reward), transition_state, done)
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
            raise ValueError(
                f"Rainbow DQN produced out-of-bounds action {action_idx} for action_space={self.num_actions}"
            )
        return int(action_idx)

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        boundary: Dict[str, Any] = {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "end_of_episode_pulse": bool(terminated) or bool(truncated),
        }
        if isinstance(info, Mapping):
            if "end_of_episode_pulse" in info:
                boundary["end_of_episode_pulse"] = bool(info["end_of_episode_pulse"])
            for key in (
                "transition_obs_rgb",
                "reset_obs_rgb",
                "boundary_cause",
                "termination_reason",
                "env_termination_reason",
                "has_prev_applied_action",
                "prev_applied_action_idx",
                "global_frame_idx",
                "is_decision_frame",
            ):
                if key in info:
                    boundary[key] = info[key]
        return self.frame(obs_rgb, reward, boundary)

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        if isinstance(state_dict, dict) and "network" in state_dict and isinstance(state_dict["network"], Mapping):
            state_dict = state_dict["network"]
        elif isinstance(state_dict, dict) and "model_state_dict" in state_dict and isinstance(state_dict["model_state_dict"], Mapping):
            state_dict = state_dict["model_state_dict"]
        if not isinstance(state_dict, Mapping):
            raise ValueError("Unsupported Rainbow DQN checkpoint format")
        self.network.load_state_dict(state_dict)
        self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)
