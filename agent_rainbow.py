# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gym-based Atari Rainbow agent using CleanRL-style wrappers and benchmark runner integration."""

from __future__ import annotations

import argparse
import collections
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from benchmark_runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    CycleConfig,
    EnvironmentHandle,
    FrameRunnerContext,
    FrameRunnerResult,
    GameSpec,
    GameResult,
)

# Import shared utilities from common module
from common import (
    FRAME_SKIP,
    ATARI_CANONICAL_ACTIONS,
    PROGRESS_POINTS,
    DEFAULT_CONTINUAL_GAMES,
    DEFAULT_CONTINUAL_CYCLES,
    DEFAULT_CONTINUAL_CYCLE_FRAMES,
    make_atari_env,
    update_progress_graphs,
    write_continual_summary,
    create_logger,
)


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.zeros(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        n_atoms: int,
        v_min: float,
        v_max: float,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.num_actions = num_actions
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 3136

        self.value_head = nn.Sequential(
            NoisyLinear(conv_out, 512),
            nn.ReLU(),
            NoisyLinear(512, n_atoms),
        )
        self.advantage_head = nn.Sequential(
            NoisyLinear(conv_out, 512),
            nn.ReLU(),
            NoisyLinear(512, n_atoms * num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x / 255.0)
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(h).view(-1, self.num_actions, self.n_atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=-1)

    def reset_noise(self):
        for module in [self.value_head, self.advantage_head]:
            for layer in module:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()


PrioritizedBatch = collections.namedtuple(
    "PrioritizedBatch",
    ["observations", "actions", "rewards", "next_observations", "dones", "indices", "weights"],
)


class SumSegmentTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size, dtype=np.float32)

    def _propagate(self, idx: int):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = self.tree[parent * 2 + 1] + self.tree[parent * 2 + 2]
            parent = (parent - 1) // 2

    def update(self, idx: int, value: float):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def total(self) -> float:
        return self.tree[0]

    def retrieve(self, value: float) -> int:
        idx = 0
        while idx * 2 + 1 < self.tree_size:
            left = idx * 2 + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - (self.capacity - 1)


class MinSegmentTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.full(self.tree_size, float("inf"), dtype=np.float32)

    def _propagate(self, idx: int):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = min(self.tree[parent * 2 + 1], self.tree[parent * 2 + 2])
            parent = (parent - 1) // 2

    def update(self, idx: int, value: float):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def min(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_space: gym.Space,
        device: torch.device,
        *,
        n_step: int,
        gamma: float,
        alpha: float,
        beta: float,
        eps: float,
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.n_step = n_step
        self.gamma = gamma

        # This logic is extracted so we can reset it later
        self.observation_space = observation_space
        self.init_buffers()

    def init_buffers(self):
        """Full initialization - allocates new arrays. Used at construction time."""
        obs_shape = self.observation_space.shape
        self.buffer_obs = np.zeros((self.capacity,) + obs_shape, dtype=np.uint8)
        self.buffer_next_obs = np.zeros((self.capacity,) + obs_shape, dtype=np.uint8)
        self.buffer_actions = np.zeros(self.capacity, dtype=np.int64)
        self.buffer_rewards = np.zeros(self.capacity, dtype=np.float32)
        self.buffer_dones = np.zeros(self.capacity, dtype=np.bool_)

        self.sum_tree = SumSegmentTree(self.capacity)
        self.min_tree = MinSegmentTree(self.capacity)
        self.max_priority = 1.0

        self.pos = 0
        self.size = 0
        self.n_step_buffer: deque = deque(maxlen=self.n_step)

    def reset_counters(self):
        """
        Lightweight reset - just resets logical state without memory reallocation.
        Used when switching games in continual learning to avoid training on old data.
        The old data in arrays will be overwritten as new experiences come in.
        """
        # Reset segment trees (these are small, O(capacity) ints/floats)
        self.sum_tree.tree.fill(0.0)
        self.min_tree.tree.fill(float("inf"))
        self.max_priority = 1.0
        
        # Reset counters
        self.pos = 0
        self.size = 0
        
        # Clear n-step buffer
        self.n_step_buffer.clear()

    def __len__(self) -> int:
        return self.size

    def _aggregate_n_step(self):
        reward = 0.0
        next_obs = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]
        for idx, (_, _, r, n_obs, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma**idx) * r
            if d:
                next_obs = n_obs
                done = True
                break
        return reward, next_obs, done

    def add(self, obs, action: int, reward: float, next_obs, done: bool):
        obs_np = np.asarray(obs, dtype=np.uint8)
        next_obs_np = np.asarray(next_obs, dtype=np.uint8)
        self.n_step_buffer.append((obs_np, action, reward, next_obs_np, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_obs, done = self._aggregate_n_step()
        obs, action = self.n_step_buffer[0][:2]

        idx = self.pos
        self.buffer_obs[idx] = obs
        self.buffer_next_obs[idx] = next_obs
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = reward
        self.buffer_dones[idx] = done

        priority = self.max_priority**self.alpha
        self.sum_tree.update(idx, priority)
        self.min_tree.update(idx, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> PrioritizedBatch:
        if self.size < batch_size:
            raise ValueError("Not enough samples in buffer to sample a batch.")

        indices = []
        segment = self.sum_tree.total() / batch_size
        for i in range(batch_size):
            value = random.uniform(segment * i, segment * (i + 1))
            idx = self.sum_tree.retrieve(value)
            indices.append(idx)

        obs = torch.from_numpy(self.buffer_obs[indices]).to(self.device)
        next_obs = torch.from_numpy(self.buffer_next_obs[indices]).to(self.device)
        actions = torch.from_numpy(self.buffer_actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.buffer_rewards[indices]).to(self.device)
        dones = torch.from_numpy(self.buffer_dones[indices].astype(np.float32)).to(self.device)

        probs = np.array([self.sum_tree.tree[idx + self.capacity - 1] for idx in indices], dtype=np.float32)
        weights = (self.size * probs / self.sum_tree.total()) ** -self.beta
        weights = weights / weights.max()
        weights = torch.from_numpy(weights.astype(np.float32)).to(self.device)

        return PrioritizedBatch(obs, actions, rewards, next_obs, dones, indices, weights.unsqueeze(1))

    def update_priorities(self, indices: Sequence[int], priorities: np.ndarray):
        priorities = np.abs(priorities) + self.eps
        self.max_priority = max(self.max_priority, priorities.max())
        for idx, priority in zip(indices, priorities):
            priority = float(priority) ** self.alpha
            self.sum_tree.update(idx, priority)
            self.min_tree.update(idx, priority)




@dataclass
class AgentConfig:
    env_id: str = "BreakoutNoFrameskip-v4"
    total_steps: int = 1_000_000
    buffer_size: int = 1_000_000
    learning_rate: float = 1.5e-4
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 8_000
    batch_size: int = 32
    exploration_fraction: float = 0.10
    start_e: float = 1.0
    end_e: float = 0.01
    learning_starts: int = 80_000
    per_game_learning_starts: int = 1_000
    n_step: int = 3
    prioritized_replay_alpha: float = 0.5
    prioritized_replay_beta: float = 0.4
    prioritized_replay_beta_end: float = 1.0
    prioritized_replay_eps: float = 1e-6
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    train_frequency: int = 4
    seed: int = 1
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "physical-atari"
    wandb_entity: Optional[str] = None
    log_file: str = ""
    continual: bool = False
    continual_games: str = ""
    continual_cycles: int = DEFAULT_CONTINUAL_CYCLES
    continual_cycle_frames: int = DEFAULT_CONTINUAL_CYCLE_FRAMES

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")


class Agent:
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: AgentConfig):
        if observation_space.shape is None or len(observation_space.shape) != 3:
            raise ValueError("Observation space must be 3D (frames, height, width).")

        self.config = config
        self.device = config.device
        self.action_space = action_space
        self.global_step = 0
        self.training_updates = 0
        self.last_obs: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self._game_learning_start_step: int = 0
        self.game_frame_counters: Dict[str, int] = {}

        obs_shape = observation_space.shape
        frame_stack = FRAME_SKIP
        if obs_shape[0] == frame_stack:
            self.channels_last = False
            in_channels = obs_shape[0]
        elif obs_shape[-1] == frame_stack:
            self.channels_last = True
            in_channels = obs_shape[-1]
        else:
            raise ValueError("Unable to infer stacked frame dimension from observation shape.")

        # --- FIX: Action Space Mismatch ---
        # If continual, we force the network to handle the full Atari action set
        # to prevent shape mismatches when switching between games.
        if config.continual:
            self.num_actions = ATARI_CANONICAL_ACTIONS
        else:
            self.num_actions = action_space.n
        # ----------------------------------

        # Use self.num_actions instead of action_space.n
        self.q_network = NoisyDuelingDistributionalNetwork(
            in_channels,
            self.num_actions,
            config.n_atoms,
            config.v_min,
            config.v_max,
        ).to(self.device)
        self.target_network = NoisyDuelingDistributionalNetwork(
            in_channels,
            self.num_actions,
            config.n_atoms,
            config.v_min,
            config.v_max,
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate, eps=1.5e-4)

        self.replay_buffer = PrioritizedReplayBuffer(
            config.buffer_size,
            observation_space,
            self.device,
            n_step=config.n_step,
            gamma=config.gamma,
            alpha=config.prioritized_replay_alpha,
            beta=config.prioritized_replay_beta,
            eps=config.prioritized_replay_eps,
        )

    def act(self, observation) -> int:
        obs_np = np.array(observation, copy=False)
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.channels_last:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)

        with torch.no_grad():
            dist = self.q_network(obs_tensor)
            q_values = torch.sum(dist * self.q_network.support, dim=2)
            action = int(torch.argmax(q_values, dim=1).item())

        self.last_obs = obs_np
        self.last_action = action
        return action

    def step(self, next_observation, reward: float, done: bool, info: Optional[Dict] = None):
        if self.last_obs is None or self.last_action is None:
            return None

        next_obs_np = np.array(next_observation, copy=False)
        self.replay_buffer.add(self.last_obs, self.last_action, reward, next_obs_np, done)
        self.global_step += 1

        train_log = None
        learning_ready = self.global_step >= max(self.config.learning_starts, self._game_learning_start_step)
        if (
            learning_ready
            and self.global_step % self.config.train_frequency == 0
            and len(self.replay_buffer) >= self.config.batch_size
        ):
            train_log = self.train_function()
        if self.global_step % self.config.target_network_frequency == 0:
            self.update_target()

        if done:
            self.last_obs = None
            self.last_action = None
        else:
            self.last_obs = next_obs_np
            self.last_action = None

        return train_log

    def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()
        if self.channels_last:
            return obs.permute(0, 3, 1, 2)
        return obs

    def _update_per_beta(self):
        progress = min(1.0, self.global_step / max(1, self.config.total_steps))
        beta = self.config.prioritized_replay_beta + progress * (
            self.config.prioritized_replay_beta_end - self.config.prioritized_replay_beta
        )
        self.replay_buffer.beta = float(min(1.0, beta))

    def train_function(self):
        self._update_per_beta()
        # Note: We reset noise every training step in Rainbow
        self.q_network.reset_noise()
        self.target_network.reset_noise()

        batch = self.replay_buffer.sample(self.config.batch_size)
        obs = self._prepare_obs(batch.observations)
        next_obs = self._prepare_obs(batch.next_observations)
        actions = batch.actions.long()
        rewards = batch.rewards.unsqueeze(1)
        dones = batch.dones.unsqueeze(1)
        weights = batch.weights

        with torch.no_grad():
            next_dist = self.target_network(next_obs)
            support = self.target_network.support
            next_q_online = torch.sum(self.q_network(next_obs) * support, dim=2)
            best_actions = torch.argmax(next_q_online, dim=1)
            next_pmfs = next_dist[torch.arange(self.config.batch_size, device=self.device), best_actions]

            gamma_n = self.config.gamma ** self.config.n_step
            next_atoms = rewards + gamma_n * support * (1 - dones)
            tz = next_atoms.clamp(self.q_network.v_min, self.q_network.v_max)
            b = (tz - self.q_network.v_min) / self.q_network.delta_z
            l = b.floor().clamp(0, self.config.n_atoms - 1)
            u = b.ceil().clamp(0, self.config.n_atoms - 1)

            d_m_l = (u.float() + (l == b).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs

            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        dist = self.q_network(obs)
        actions_expanded = actions.unsqueeze(1).unsqueeze(-1).expand(-1, -1, self.config.n_atoms)
        pred_dist = dist.gather(1, actions_expanded).squeeze(1)
        log_pred = torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5))
        loss_per_sample = -(target_pmfs * log_pred).sum(dim=1)
        loss = (loss_per_sample * weights.squeeze()).mean()

        self.replay_buffer.update_priorities(batch.indices, loss_per_sample.detach().cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_updates += 1

        q_values = torch.sum(pred_dist * self.q_network.support, dim=1)
        return {
            "loss": loss.item(),
            "q_value": q_values.mean().item(),
            "beta": self.replay_buffer.beta,
        }

    def update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(
                    self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data
                )

    def save_model(self, path: str):
        torch.save(self.q_network.state_dict(), path)

    def start_new_game(self):
        self._game_learning_start_step = self.global_step + max(0, self.config.per_game_learning_starts)
        
        # --- FIX: Buffer Pollution (Performance Optimized) ---
        # Reset buffer counters and trees instead of reallocating arrays.
        # This avoids deallocating/reallocating potentially gigabytes of memory.
        if self.config.continual:
            print(f"  [Agent] Resetting Prioritized Replay Buffer counters for new game...")
            # Use lightweight reset that clears logical state without memory reallocation.
            # Also clears n-step buffer to prevent cross-game transition leaks.
            self.replay_buffer.reset_counters()
        # -----------------------------


def agent_rainbow_frame_runner(
    agent: Agent,
    handle: EnvironmentHandle,
    *,
    context: FrameRunnerContext,
    writer: Optional[SummaryWriter] = None,
) -> FrameRunnerResult:
    if handle.backend != 'gym' or handle.gym is None:
        raise TypeError("agent_rainbow_frame_runner requires a Gym environment handle.")

    env = handle.gym
    config = agent.config
    frames_per_step = max(1, handle.frames_per_step)

    # Use consistent logging from common module
    logger = create_logger(log_file=getattr(config, "log_file", None))
    log = logger.log

    max_step_budget = max(1, context.frame_budget // frames_per_step)
    if config.total_steps > 0:
        remaining_steps = max(0, config.total_steps - agent.global_step)
        steps_to_run = min(max_step_budget, remaining_steps)
    else:
        steps_to_run = max_step_budget

    game_name = getattr(handle.spec, "name", context.name)
    scheduled_frames = steps_to_run * frames_per_step

    if steps_to_run <= 0:
        log(f"Skipping game '{game_name}' because no steps remain.")
        params = list(agent.q_network.parameters())
        episode_graph = torch.full((PROGRESS_POINTS,), -999.0, dtype=torch.float32)
        parms_graph = torch.zeros((PROGRESS_POINTS, len(params)), dtype=torch.float32)
        return FrameRunnerResult(context.last_model_save, [], [], episode_graph, parms_graph)

    log(
        f"Starting game '{game_name}' with target {scheduled_frames} frames "
        f"(cycle budget {context.frame_budget})."
    )

    reset_seed = handle.spec.seed if handle.spec.seed is not None else config.seed
    obs, info = env.reset(seed=reset_seed)
    agent.start_new_game()

    episode_scores: List[float] = []
    episode_end: List[int] = []
    params = list(agent.q_network.parameters())
    episode_graph = torch.full((PROGRESS_POINTS,), -999.0, dtype=torch.float32)
    parms_graph = torch.zeros((PROGRESS_POINTS, len(params)), dtype=torch.float32)
    frames_consumed = 0
    frames_since_reward = 0
    running_episode_score = 0.0
    start_time = time.time()

    last_model_save = context.last_model_save

    for _ in range(steps_to_run):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        running_episode_score += reward
        if reward != 0:
            frames_since_reward = 0
        else:
            frames_since_reward += frames_per_step

        timed_out = (
            context.max_frames_without_reward > 0 and frames_since_reward >= context.max_frames_without_reward
        )
        done_for_agent = done or timed_out

        train_log = agent.step(next_obs, reward, done_for_agent, info)
        frames_consumed += frames_per_step
        agent.game_frame_counters[game_name] = agent.game_frame_counters.get(game_name, 0) + frames_per_step

        if train_log and agent.global_step % 1000 == 0:
            sps = int(agent.global_step / max(1e-9, time.time() - start_time))
            log(
                f"step={agent.global_step} loss={train_log['loss']:.4f} "
                f"q={train_log['q_value']:.2f}"
            )
            log(f"SPS: {sps}")
            if writer:
                writer.add_scalar("losses/td_loss", train_log["loss"], agent.global_step)
                writer.add_scalar("losses/q_values", train_log["q_value"], agent.global_step)
                writer.add_scalar("charts/SPS", sps, agent.global_step)
                beta = train_log.get("beta")
                if beta is not None:
                    writer.add_scalar("charts/beta", beta, agent.global_step)

        recorded_episode = False
        if info and "episode" in info:
            episode = info["episode"]
            episode_scores.append(float(episode["r"]))
            episode_end.append(context.frame_offset + frames_consumed)
            log(f"global_step={agent.global_step}, episodic_return={episode['r']}")
            recorded_episode = True
            if writer:
                writer.add_scalar("charts/episodic_return", episode["r"], agent.global_step)
                writer.add_scalar(f"charts/episodic_return/{game_name}", episode["r"], agent.global_step)
                
                # Track cumulative frames for this specific game
                current_game_frames = agent.game_frame_counters.get(game_name, 0)
                writer.add_scalar(f"charts/frames_per_game/{game_name}", current_game_frames, agent.global_step)

                length = episode.get("l")
                if length is not None:
                    writer.add_scalar(
                        "charts/episodic_length_frames", length * FRAME_SKIP, agent.global_step
                    )

        if timed_out and not recorded_episode:
            log(f"Resetting due to {frames_since_reward} frames without reward.")
            episode_scores.append(running_episode_score)
            episode_end.append(context.frame_offset + frames_consumed)

        if context.save_incremental_models:
            global_frame = context.frame_offset + frames_consumed
            checkpoint_slot = global_frame // 500_000
            if checkpoint_slot > last_model_save and checkpoint_slot > 0:
                model_path = os.path.join(context.data_dir, f"{context.name}_{global_frame}.model")
                agent.save_model(model_path)
                log(f"Saved incremental model to {model_path}")
                last_model_save = checkpoint_slot

        if done_for_agent:
            obs, info = env.reset()
            running_episode_score = 0.0
            frames_since_reward = 0
        else:
            obs = next_obs

        update_progress_graphs(
            episode_graph,
            parms_graph,
            params,
            context.frame_offset,
            frames_consumed,
            context.frame_budget,
            context.average_frames,
            episode_scores,
            episode_end,
            context.graph_total_frames,
        )

        if frames_consumed >= context.frame_budget:
            break

    return FrameRunnerResult(last_model_save, episode_scores, episode_end, episode_graph, parms_graph)


def _parse_continual_game_ids(config: AgentConfig) -> List[str]:
    """
    Parse the list of games for continual learning.
    
    Allows any number of games >= 2 for flexibility in ablations and debugging.
    Default is 8 games if not specified.
    """
    if config.continual_games.strip():
        game_ids = [game.strip() for game in config.continual_games.split(",") if game.strip()]
    else:
        game_ids = list(DEFAULT_CONTINUAL_GAMES)
    
    if len(game_ids) < 2:
        raise ValueError(f"Continual mode requires at least 2 games for meaningful evaluation, received {len(game_ids)}.")
    
    return game_ids


def _validate_continual_envs(
    game_ids: Sequence[str],
    config: AgentConfig,
    base_shape: Sequence[int],
    min_actions: int,
):
    for env_id in game_ids[1:]:
        preview_env = make_atari_env(env_id, config.seed)
        try:
            obs_shape = preview_env.observation_space.shape
            if obs_shape != base_shape:
                raise ValueError(
                    f"Observation shape mismatch for {env_id}: expected {base_shape}, observed {obs_shape}."
                )
            action_space = getattr(preview_env.action_space, "n", None)
            if action_space is None:
                raise ValueError(f"{env_id} does not expose a discrete action space.")
            if action_space < min_actions:
                raise ValueError(
                    f"{env_id} exposes only {action_space} actions, but the agent was initialized with {min_actions}."
                )
        finally:
            preview_env.close()


def _build_continual_benchmark_config(
    game_ids: Sequence[str],
    frame_budget_per_game: int,
    config: AgentConfig,
) -> BenchmarkConfig:
    cycles: List[CycleConfig] = []
    for cycle_index in range(config.continual_cycles):
        cycle_games: List[GameSpec] = []
        for env_id in game_ids:
            cycle_games.append(
                GameSpec(
                    name=env_id,
                    frame_budget=frame_budget_per_game,
                    seed=config.seed + cycle_index,
                    backend='gym',
                    env_id=env_id,
                    sticky_prob=0.0,
                    params={
                        'noop_max': 30,
                        'frame_skip': FRAME_SKIP,
                        'frame_stack': FRAME_SKIP,
                        'resize_shape': (84, 84),
                        'grayscale': True,
                    },
                )
            )
        cycles.append(CycleConfig(cycle_index=cycle_index, games=cycle_games))
    description = (
        f"Continual benchmark: {config.continual_cycles} cycle(s) x {len(game_ids)} game(s) "
        f"(frames per game per cycle={config.continual_cycle_frames})"
    )
    return BenchmarkConfig(cycles=cycles, description=description)




def _run_single_environment(config: AgentConfig, writer: SummaryWriter, run_name: str, run_dir: str):
    env: Optional[gym.Env] = None
    try:
        env = make_atari_env(config.env_id, config.seed)
        agent = Agent(env.observation_space, env.action_space, config)
        frames_per_step = FRAME_SKIP
        frame_budget_frames = max(config.total_steps, 1) * frames_per_step
        game_spec = GameSpec(
            name=config.env_id,
            frame_budget=frame_budget_frames,
            seed=config.seed,
            backend='gym',
            env_id=config.env_id,
        )
        handle = EnvironmentHandle(
            backend='gym',
            spec=game_spec,
            frames_per_step=frames_per_step,
            gym=env,
        )
        context = FrameRunnerContext(
            name=run_name,
            data_dir=run_dir,
            rank=0,
            average_frames=100_000,
            max_frames_without_reward=0,
            lives_as_episodes=1,
            save_incremental_models=False,
            last_model_save=-1,
            frame_budget=frame_budget_frames,
            frame_offset=0,
            graph_total_frames=frame_budget_frames,
            delay_frames=0,
        )
        agent_rainbow_frame_runner(agent, handle, context=context, writer=writer)
    finally:
        if env is not None:
            env.close()


def _run_continual_mode(config: AgentConfig, writer: SummaryWriter, run_dir: str):
    game_ids = _parse_continual_game_ids(config)
    if config.continual_cycle_frames <= 0:
        raise ValueError("--continual-cycle-frames must be a positive integer.")
    if config.continual_cycles <= 0:
        raise ValueError("--continual-cycles must be a positive integer.")

    print(
        f"Starting continual mode: {config.continual_cycles} cycles x {len(game_ids)} games "
        f"({config.continual_cycle_frames} frames per game per cycle)."
    )
    frame_budget_per_game = config.continual_cycle_frames
    frames_needed = config.continual_cycle_frames * len(game_ids) * config.continual_cycles
    required_steps = math.ceil(frames_needed / max(1, FRAME_SKIP))
    if config.total_steps != required_steps:
        print(
            f"Adjusting total_steps from {config.total_steps} to {required_steps} "
            "so the continual benchmark exactly matches the requested frame budget."
        )
        config.total_steps = required_steps

    base_env = make_atari_env(game_ids[0], config.seed)
    try:
        agent = Agent(base_env.observation_space, base_env.action_space, config)
        base_shape = base_env.observation_space.shape
        min_actions = base_env.action_space.n
    finally:
        base_env.close()

    _validate_continual_envs(game_ids, config, base_shape, min_actions)
    benchmark_config = _build_continual_benchmark_config(game_ids, frame_budget_per_game, config)

    def frame_runner_with_writer(agent_instance, handle, *, context):
        return agent_rainbow_frame_runner(agent_instance, handle, context=context, writer=writer)

    runner = BenchmarkRunner(
        agent,
        benchmark_config,
        frame_runner=frame_runner_with_writer,
        data_dir=run_dir,
        rank=0,
        default_seed=config.seed,
        # --- FIX: Ensure Runner forces full action set for safety ---
        use_canonical_full_actions=True, 
        # ------------------------------------------------------------
        average_frames=100_000,
        max_frames_without_reward=0,
        lives_as_episodes=1,
        save_incremental_models=False,
    )
    results = runner.run()
    summary_path = os.path.join(run_dir, "continual_summary.json")
    write_continual_summary(results, summary_path)


def parse_args() -> AgentConfig:
    parser = argparse.ArgumentParser(description="Gym Atari DQN agent with CleanRL wrappers.")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=6.25e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--target-network-frequency", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.01)
    parser.add_argument("--learning-starts", type=int, default=80_000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable CUDA (default: enabled).",
    )
    parser.add_argument(
        "--track",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Track with Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="physical-atari",
        help="The wandb's project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="The entity (team) of wandb's project.",
    )
    parser.add_argument(
        "--continual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the continual benchmark (8 games x 3 cycles, 400k frames per cycle).",
    )
    parser.add_argument(
        "--continual-games",
        type=str,
        default="",
        help="Comma-separated Gym env ids to use in continual mode (defaults to a preset list).",
    )
    parser.add_argument(
        "--continual-cycles",
        type=int,
        default=DEFAULT_CONTINUAL_CYCLES,
        help="Number of cycles to run in continual mode.",
    )
    parser.add_argument(
        "--continual-cycle-frames",
        type=int,
        default=DEFAULT_CONTINUAL_CYCLE_FRAMES,
        help="Frame budget per game per cycle in continual mode.",
    )
    parser.add_argument(
        "--per-game-learning-starts",
        type=int,
        default=1_000,
        help="Steps of experience to collect at the start of each game before training.",
    )
    parser.add_argument("--n-step", type=int, default=3, help="Number of steps for n-step returns.")
    parser.add_argument("--n-atoms", type=int, default=51, help="Number of atoms for distributional RL.")
    parser.add_argument("--v-min", type=float, default=-10.0, help="Minimum value support for distributional RL.")
    parser.add_argument("--v-max", type=float, default=10.0, help="Maximum value support for distributional RL.")
    parser.add_argument(
        "--prioritized-alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for prioritized replay.",
    )
    parser.add_argument(
        "--prioritized-beta",
        type=float,
        default=0.4,
        help="Initial beta parameter for prioritized replay.",
    )
    parser.add_argument(
        "--prioritized-beta-end",
        type=float,
        default=1.0,
        help="Final beta value for prioritized replay annealing.",
    )
    parser.add_argument(
        "--prioritized-eps",
        type=float,
        default=1e-6,
        help="Epsilon to add to priorities for numerical stability.",
    )
    args = parser.parse_args()
    return AgentConfig(
        env_id=args.env_id,
        total_steps=args.total_steps,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        target_network_frequency=args.target_network_frequency,
        batch_size=args.batch_size,
        exploration_fraction=args.exploration_fraction,
        start_e=args.start_e,
        end_e=args.end_e,
        learning_starts=args.learning_starts,
        train_frequency=args.train_frequency,
        seed=args.seed,
        cuda=args.cuda,
        continual=args.continual,
        continual_games=args.continual_games,
        continual_cycles=args.continual_cycles,
        continual_cycle_frames=args.continual_cycle_frames,
        per_game_learning_starts=args.per_game_learning_starts,
        n_step=args.n_step,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        prioritized_replay_alpha=args.prioritized_alpha,
        prioritized_replay_beta=args.prioritized_beta,
        prioritized_replay_beta_end=args.prioritized_beta_end,
        prioritized_replay_eps=args.prioritized_eps,
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
    )


def main():
    config = parse_args()
    log_root = './results_final_rainbow/smokey'
    os.makedirs(log_root, exist_ok=True)
    run_name = f"{config.env_id}__agent_final_rainbow__{config.seed}__{int(time.time())}"
    
    if config.track:
        import wandb

        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=vars(config),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    run_dir = os.path.join(log_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    config.log_file = os.path.join(run_dir, 'agent.log')
    config_path = os.path.join(run_dir, 'hyperparameters.json')
    with open(config_path, 'w', encoding='utf-8') as cfg:
        json.dump(vars(config), cfg, indent=2)
    print(f'Logging run to {run_dir}')

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.log_file:
        with open(config.log_file, "w", encoding="utf-8") as log_f:
            log_f.write("")

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in vars(config).items()),
    )

    try:
        if config.continual:
            _run_continual_mode(config, writer, run_dir)
        else:
            _run_single_environment(config, writer, run_name, run_dir)
    finally:
        writer.close()


if __name__ == "__main__":
    main()