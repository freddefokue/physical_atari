import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import Agent


@dataclass
class DQNConfig:
    frame_stack: int = 4
    learning_rate: float = 1e-4
    gamma: float = 0.99
    buffer_capacity: int = 100_000
    batch_size: int = 32
    learning_starts: int = 5_000
    train_freq: int = 4
    target_update_freq: int = 10_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 1_000_000
    max_grad_norm: float = 10.0


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple[int, int, int], seed: int):
        self.capacity = capacity
        self.state_shape = state_shape
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        batch = {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }
        return batch

    def __len__(self):
        return self.size


class DQNNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = x / 255.0
        return self.net(x)


class DQNAgent(Agent):
    def __init__(
        self,
        seed: int = 0,
        config: Optional[DQNConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(seed=seed)
        self.config = config or DQNConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net: Optional[DQNNetwork] = None
        self.target_net: Optional[DQNNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.replay: Optional[ReplayBuffer] = None
        self.num_actions: Optional[int] = None

        self.total_steps = 0
        self.learn_steps = 0
        self.frame_stack = self.config.frame_stack
        self._frame_stack = collections.deque(maxlen=self.frame_stack)
        self._current_state: Optional[np.ndarray] = None
        self._needs_reset = True
        self._last_action: Optional[int] = None

    def begin_task(self, game_id, cycle, action_space):
        if self.num_actions is None or self.num_actions != action_space.n:
            self.num_actions = action_space.n
            self._build_networks()
        self._frame_stack.clear()
        self._current_state = None
        self._needs_reset = True
        self._last_action = None

    def _build_networks(self):
        assert self.num_actions is not None
        self.policy_net = DQNNetwork(self.frame_stack, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(self.frame_stack, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        state_shape = (self.frame_stack, 84, 84)
        self.replay = ReplayBuffer(self.config.buffer_capacity, state_shape, seed=int(self.rng.integers(1 << 30)))

    def _ensure_state_initialized(self, obs):
        if not self._needs_reset and self._current_state is not None:
            return
        processed = self._preprocess(obs)
        self._frame_stack.clear()
        for _ in range(self.frame_stack):
            self._frame_stack.append(processed)
        self._current_state = np.stack(self._frame_stack, axis=0)
        self._needs_reset = False

    def _preprocess(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            obs = obs.get("pixel")
        obs_array = np.asarray(obs)
        frame = Image.fromarray(obs_array).convert("L").resize((84, 84), Image.BILINEAR)
        return np.array(frame, dtype=np.uint8)

    def _append_next_state(self, obs) -> np.ndarray:
        processed = self._preprocess(obs)
        self._frame_stack.append(processed)
        return np.stack(self._frame_stack, axis=0)

    def act(self, obs, action_space) -> int:
        self._ensure_state_initialized(obs)
        assert self._current_state is not None
        epsilon = self._epsilon()
        if self.rng.random() < epsilon:
            action = int(action_space.sample())
        else:
            state_tensor = torch.from_numpy(self._current_state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
        self._last_action = action
        return action

    def observe(self, obs, action, reward, next_obs, terminated, truncated, info):
        if self.policy_net is None or self.replay is None:
            return
        self._ensure_state_initialized(obs)
        assert self._current_state is not None
        done = float(terminated or truncated)
        state = self._current_state.copy()
        if done:
            next_state = np.zeros_like(state, dtype=np.uint8)
            self._needs_reset = True
            self._frame_stack.clear()
            self._current_state = None
        else:
            next_state = self._append_next_state(next_obs)
            self._current_state = next_state

        self.replay.add(state, action, float(reward), next_state, done)
        self.total_steps += 1

        if self.total_steps >= self.config.learning_starts:
            if self.total_steps % self.config.train_freq == 0:
                self._train_step()
            if self.learn_steps % self.config.target_update_freq == 0 and self.learn_steps > 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def end_episode(self, episode_return: float) -> None:
        self._last_action = None

    def _epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.config.eps_decay)
        return self.config.eps_start + frac * (self.config.eps_end - self.config.eps_start)

    def _train_step(self):
        assert self.replay is not None and self.policy_net is not None and self.target_net is not None
        if len(self.replay) < self.config.batch_size:
            return
        batch = self.replay.sample(self.config.batch_size)
        states = torch.from_numpy(batch["states"]).float().to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).float().to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).float().to(self.device)
        dones = torch.from_numpy(batch["dones"]).float().to(self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target = rewards + (1.0 - dones) * self.config.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.learn_steps += 1
