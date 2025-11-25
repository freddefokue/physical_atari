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

"""Gym-based Atari BBF agent with BenchmarkRunner integration."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
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
    GameResult,
    GameSpec,
)
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

FRAME_SKIP = 4
PROGRESS_POINTS = 1000
DEFAULT_CONTINUAL_GAMES = (
    "BreakoutNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "RiverraidNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
)
DEFAULT_CONTINUAL_CYCLES = 3
DEFAULT_CONTINUAL_CYCLE_FRAMES = 400_000


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, FRAME_SKIP)
    env.action_space.seed(seed)
    return env


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs


class ImpalaCNN(nn.Module):
    def __init__(self, in_channels: int, channel_scale: int = 4):
        super().__init__()
        widths = [32 * channel_scale, 64 * channel_scale, 64 * channel_scale]
        layers: List[nn.Module] = []
        for width in widths:
            layers.append(nn.Conv2d(in_channels, width, kernel_size=3, padding=1))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(ResidualBlock(width))
            layers.append(ResidualBlock(width))
            in_channels = width
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, n_atoms: int, v_min: float, v_max: float):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_actions = num_actions
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        self.encoder = ImpalaCNN(in_channels, channel_scale=4)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            x = self.encoder(dummy)
            self.out_features = x.shape[1] * x.shape[2] * x.shape[3]

        self.fc_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.out_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_atoms),
        )
        self.fc_advantage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.out_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_atoms * num_actions),
        )

        self.proj_dim = 512
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.out_features, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
        )
        self.dynamics_model = nn.Sequential(
            nn.Linear(self.proj_dim + num_actions, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x / 255.0)
        value = self.fc_value(x).view(-1, 1, self.n_atoms)
        advantage = self.fc_advantage(x).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=2)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(x / 255.0))


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
        obs_shape: Sequence[int],
        device: torch.device,
        alpha: float,
        beta: float,
        eps: float,
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.buffer_obs = np.zeros((capacity,) + tuple(obs_shape), dtype=np.uint8)
        self.buffer_actions = np.zeros(capacity, dtype=np.int64)
        self.buffer_rewards = np.zeros(capacity, dtype=np.float32)
        self.buffer_dones = np.zeros(capacity, dtype=np.bool_)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)

    def add(self, obs, action, reward, next_obs, done):
        idx = self.pos
        self.buffer_obs[idx] = obs
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = reward
        self.buffer_dones[idx] = done

        priority = self.max_priority**self.alpha
        self.sum_tree.update(idx, priority)
        self.min_tree.update(idx, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int):
        indices: List[int] = []
        p_total = self.sum_tree.total()
        segment = p_total / batch_size

        for i in range(batch_size):
            while True:
                a = segment * i
                b = segment * (i + 1)
                value = np.random.uniform(a, b)
                idx = self.sum_tree.retrieve(value)

                if self.size < self.capacity:
                    if idx + seq_len > self.pos:
                        continue
                else:
                    if idx + seq_len > self.capacity and (idx + seq_len) % self.capacity >= self.pos:
                        continue
                if idx + seq_len > self.capacity:
                    continue
                indices.append(idx)
                break

        b_obs = np.array([self.buffer_obs[i : i + seq_len] for i in indices])
        b_actions = np.array([self.buffer_actions[i : i + seq_len] for i in indices])
        b_rewards = np.array([self.buffer_rewards[i : i + seq_len] for i in indices])
        b_dones = np.array([self.buffer_dones[i : i + seq_len] for i in indices])

        samples = {
            "observations": torch.from_numpy(b_obs).to(self.device),
            "actions": torch.from_numpy(b_actions).to(self.device),
            "rewards": torch.from_numpy(b_rewards).to(self.device),
            "dones": torch.from_numpy(b_dones.astype(np.float32)).to(self.device),
        }

        probs = np.array([self.sum_tree.tree[idx + self.capacity - 1] for idx in indices], dtype=np.float32)
        weights = (self.size * probs / p_total) ** -self.beta
        weights = weights / weights.max()
        samples["weights"] = torch.from_numpy(weights).to(self.device).unsqueeze(1)
        samples["indices"] = indices
        return samples

    def update_priorities(self, indices: Sequence[int], priorities: np.ndarray):
        priorities = np.abs(priorities) + self.eps
        self.max_priority = max(self.max_priority, priorities.max())
        for idx, priority in zip(indices, priorities):
            priority = float(priority) ** self.alpha
            self.sum_tree.update(idx, priority)
            self.min_tree.update(idx, priority)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    if duration <= 0:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def shrink_and_perturb(model: QNetwork, target_model: QNetwork, alpha: float = 0.5):
    random_model = ImpalaCNN(model.encoder.network[0].weight.shape[1]).to(model.encoder.network[0].weight.device)
    with torch.no_grad():
        for param, rand_param in zip(model.encoder.parameters(), random_model.parameters()):
            param.data.mul_(alpha).add_(rand_param.data * (1.0 - alpha))
    target_model.load_state_dict(model.state_dict())


class Augmentation(nn.Module):
    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        is_seq = False
        if x.ndim == 5:
            is_seq = True
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)

        b, _, h, w = x.shape
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
        eps = 1.0 / (h + 2 * self.pad)
        x = x + torch.rand_like(x) * eps

        shifted = []
        for i in range(b):
            y = random.randint(0, 2 * self.pad)
            x_shift = random.randint(0, 2 * self.pad)
            shifted.append(x[i, :, y : y + h, x_shift : x_shift + w])
        out = torch.stack(shifted)
        if is_seq:
            out = out.view(b, t, out.shape[1], out.shape[2], out.shape[3])
        return out


@dataclass
class AgentConfig:
    env_id: str = "BreakoutNoFrameskip-v4"
    total_steps: int = 1_000_000
    buffer_size: int = 1_000_000
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    gamma_initial: float = 0.97
    gamma_final: float = 0.997
    n_step_max: int = 10
    n_step_min: int = 3
    n_step_decay_steps: int = 10_000
    reset_grad_interval: int = 40_000
    replay_ratio: int = 8
    spr_weight: float = 2.0
    jumps: int = 5
    max_grad_norm: float = 10.0
    tau: float = 0.995
    batch_size: int = 32
    learning_starts: int = 2_000
    exploration_fraction: float = 0.02
    start_e: float = 1.0
    end_e: float = 0.01
    prioritized_replay_alpha: float = 0.5
    prioritized_replay_beta: float = 0.4
    prioritized_replay_eps: float = 1e-6
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    per_game_learning_starts: int = 1_000
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
        self.config = config
        self.device = config.device
        self.action_space = action_space
        self.global_step = 0
        self.global_grad_step = 0
        self.last_obs: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self._game_learning_start_step = 0
        self.game_frame_counters: Dict[str, int] = {}

        obs_shape = observation_space.shape
        if obs_shape[0] != FRAME_SKIP:
            raise ValueError("Expected channel-first stacked observations.")

        self.q_network = QNetwork(obs_shape[0], action_space.n, config.n_atoms, config.v_min, config.v_max).to(self.device)
        self.target_network = QNetwork(obs_shape[0], action_space.n, config.n_atoms, config.v_min, config.v_max).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1.5e-4,
        )
        self.augmentation = Augmentation().to(self.device)

        self.replay_buffer = PrioritizedReplayBuffer(
            config.buffer_size,
            obs_shape,
            self.device,
            config.prioritized_replay_alpha,
            config.prioritized_replay_beta,
            config.prioritized_replay_eps,
        )

    def start_new_game(self):
        self._game_learning_start_step = self.global_step + max(0, self.config.per_game_learning_starts)

    def act(self, observation) -> int:
        obs_np = np.array(observation, copy=False)
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        epsilon = linear_schedule(
            self.config.start_e,
            self.config.end_e,
            int(self.config.exploration_fraction * max(1, self.config.total_steps)),
            self.global_step,
        )
        if random.random() < epsilon or self.global_step < self.config.learning_starts:
            action = self.action_space.sample()
        else:
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
        if learning_ready and self.replay_buffer.size >= self.config.batch_size:
            for _ in range(self.config.replay_ratio):
                train_log = self._train_once()

        if done:
            self.last_obs = None
            self.last_action = None
        else:
            self.last_obs = next_obs_np
            self.last_action = None

        return train_log

    def _train_once(self):
        self.global_grad_step += 1
        if self.global_grad_step % self.config.reset_grad_interval == 0:
            shrink_and_perturb(self.q_network, self.target_network)
            self.optimizer = optim.AdamW(
                self.q_network.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=1.5e-4,
            )

        steps_since_reset = self.global_grad_step % self.config.reset_grad_interval
        frac = min(1.0, steps_since_reset / max(1, self.config.n_step_decay_steps))
        current_n_step = max(self.config.n_step_min, int(self.config.n_step_max * (0.3**frac)))
        current_gamma = self.config.gamma_initial * ((self.config.gamma_final / self.config.gamma_initial) ** frac)

        seq_len = current_n_step + self.config.jumps + 1
        batch = self.replay_buffer.sample(self.config.batch_size, seq_len)

        observations = batch["observations"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        dones = batch["dones"]
        weights = batch["weights"]

        with torch.no_grad():
            b_rewards = rewards
            b_dones = dones

            n_rewards = torch.zeros(self.config.batch_size, device=self.device)
            bootstrap_mask = torch.ones(self.config.batch_size, device=self.device)
            for i in range(current_n_step):
                r = b_rewards[:, i]
                d = b_dones[:, i]
                n_rewards += r * (current_gamma**i) * bootstrap_mask
                bootstrap_mask *= (1 - d)

            bootstrap_obs = observations[:, current_n_step]
            bootstrap_obs = bootstrap_obs.float()
            next_dist = self.target_network(bootstrap_obs)
            support = self.target_network.support
            online_dist = self.q_network(bootstrap_obs)
            next_q_online = torch.sum(online_dist * support, dim=2)
            best_actions = torch.argmax(next_q_online, dim=1)
            next_pmfs = next_dist[torch.arange(self.config.batch_size, device=self.device), best_actions]

            gamma_n = current_gamma**current_n_step
            next_atoms = n_rewards.unsqueeze(1) + gamma_n * support * bootstrap_mask.unsqueeze(1)
            tz = next_atoms.clamp(self.q_network.v_min, self.q_network.v_max)
            b = (tz - self.q_network.v_min) / self.q_network.delta_z
            l = b.floor().clamp(0, self.config.n_atoms - 1)
            u = b.ceil().clamp(0, self.config.n_atoms - 1)

            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), (u[i].float() + (l[i] == b[i]).float() - b[i]) * next_pmfs[i])
                target_pmfs[i].index_add_(0, u[i].long(), (b[i] - l[i].float()) * next_pmfs[i])

        curr_obs = observations[:, 0].float()
        curr_obs_aug = self.augmentation(curr_obs)
        dist = self.q_network(curr_obs_aug)
        action_idx = actions[:, 0].unsqueeze(1).unsqueeze(2).expand(-1, 1, self.config.n_atoms)
        pred_dist = dist.gather(1, action_idx).squeeze(1)
        log_pred = torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5))
        loss_q = -(target_pmfs * log_pred).sum(dim=1)

        target_obs_seq = observations[:, 1 : self.config.jumps + 1].float()
        with torch.no_grad():
            b, k, c, h, w = target_obs_seq.shape
            target_flat = target_obs_seq.reshape(b * k, c, h, w)
            target_reps = self.target_network.get_representation(target_flat).view(b, k, -1)

        z0 = self.q_network.get_representation(curr_obs_aug)
        pred_reps: List[torch.Tensor] = []
        curr_z = z0
        for k in range(self.config.jumps):
            action_one_hot = F.one_hot(actions[:, k].squeeze(), num_classes=self.action_space.n).float()
            model_input = torch.cat([curr_z, action_one_hot], dim=1)
            next_z_hat = self.q_network.dynamics_model(model_input)
            pred_reps.append(self.q_network.predictor(next_z_hat))
            curr_z = next_z_hat
        pred_stack = torch.stack(pred_reps, dim=1)

        mask = torch.ones(self.config.batch_size, self.config.jumps, device=self.device)
        for k in range(1, self.config.jumps):
            mask[:, k] = mask[:, k - 1] * (1 - b_dones[:, k - 1])

        sim = F.cosine_similarity(pred_stack, target_reps, dim=-1)
        spr_loss = -(sim * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        total_loss = (loss_q + self.config.spr_weight * spr_loss) * weights.squeeze()
        final_loss = total_loss.mean()

        new_priorities = loss_q.detach().cpu().numpy()
        self.replay_buffer.update_priorities(batch["indices"], new_priorities)

        self.optimizer.zero_grad()
        final_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
                target_param.data.mul_(self.config.tau).add_(param.data * (1.0 - self.config.tau))

        q_values = torch.sum(pred_dist * self.q_network.support, dim=1)
        return {
            "loss": loss_q.mean().item(),
            "spr_loss": spr_loss.mean().item(),
            "q_value": q_values.mean().item(),
            "n_step": current_n_step,
            "gamma": current_gamma,
        }

    def save_model(self, path: str):
        torch.save(self.q_network.state_dict(), path)


def _moving_average(scores: Sequence[float], end_frames: Sequence[int], window: int, current_frame: int) -> float:
    if not scores:
        return -999.0
    cutoff = current_frame - window
    total = 0.0
    count = 0
    for score, frame_idx in zip(reversed(scores), reversed(end_frames)):
        if frame_idx < cutoff:
            break
        total += score
        count += 1
    if count == 0:
        return -999.0
    return total / count


def _update_progress_graphs(
    episode_graph: torch.Tensor,
    parms_graph: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    frame_offset: int,
    frames_consumed: int,
    frame_budget: int,
    average_frames: int,
    episode_scores: Sequence[float],
    episode_end: Sequence[int],
    graph_total_frames: Optional[int],
):
    total_frames = graph_total_frames or (frame_offset + frame_budget)
    total_frames = max(total_frames, frame_offset + frame_budget)
    current_frame = frame_offset + frames_consumed
    previous_frame = max(frame_offset, current_frame - FRAME_SKIP)
    points = episode_graph.shape[0]
    prev_bucket = min((previous_frame * points) // total_frames, points - 1)
    curr_bucket = min((current_frame * points) // total_frames, points - 1)
    if curr_bucket == prev_bucket:
        return
    avg_score = _moving_average(episode_scores, episode_end, average_frames, current_frame)
    for bucket in range(prev_bucket + 1, curr_bucket + 1):
        episode_graph[bucket] = avg_score
        with torch.no_grad():
            for idx, param in enumerate(params):
                parms_graph[bucket, idx] = torch.norm(param.detach()).item()


def agent_bbf_frame_runner(
    agent: Agent,
    handle: EnvironmentHandle,
    *,
    context: FrameRunnerContext,
    writer: Optional[SummaryWriter] = None,
) -> FrameRunnerResult:
    if handle.backend != "gym" or handle.gym is None:
        raise TypeError("agent_bbf_frame_runner requires a Gym environment handle.")

    env = handle.gym
    config = agent.config
    frames_per_step = max(1, handle.frames_per_step)

    log_file = getattr(config, "log_file", "")

    def log(message: str):
        print(message)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as log_f:
                log_f.write(message + "\n")

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
                f"spr={train_log.get('spr_loss', 0):.4f} q={train_log['q_value']:.2f}"
            )
            log(f"SPS: {sps}")
            if writer:
                writer.add_scalar("losses/td_loss", train_log["loss"], agent.global_step)
                writer.add_scalar("losses/spr_loss", train_log.get("spr_loss", 0.0), agent.global_step)
                writer.add_scalar("charts/n_step", train_log.get("n_step", 0), agent.global_step)
                writer.add_scalar("charts/gamma", train_log.get("gamma", 0.0), agent.global_step)
                writer.add_scalar("charts/q_values", train_log["q_value"], agent.global_step)
                writer.add_scalar("charts/SPS", sps, agent.global_step)

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
                current_game_frames = agent.game_frame_counters.get(game_name, 0)
                writer.add_scalar(
                    f"charts/frames_per_game/{game_name}",
                    current_game_frames,
                    agent.global_step,
                )
                length = episode.get("l")
                if length is not None:
                    writer.add_scalar(
                        "charts/episodic_length_frames",
                        length * FRAME_SKIP,
                        agent.global_step,
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

        _update_progress_graphs(
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
    if config.continual_games.strip():
        game_ids = [game.strip() for game in config.continual_games.split(",") if game.strip()]
    else:
        game_ids = list(DEFAULT_CONTINUAL_GAMES)
    if len(game_ids) != 8:
        raise ValueError(f"Continual mode requires exactly 8 games, received {len(game_ids)}.")
    return game_ids


def _validate_continual_envs(
    game_ids: Sequence[str],
    config: AgentConfig,
    base_shape: Sequence[int],
    min_actions: int,
):
    for env_id in game_ids[1:]:
        preview_env = make_env(env_id, config.seed)
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
                    backend="gym",
                    env_id=env_id,
                    sticky_prob=0.25,
                    params={
                        "noop_max": 30,
                        "frame_skip": FRAME_SKIP,
                        "frame_stack": FRAME_SKIP,
                        "resize_shape": (84, 84),
                        "grayscale": True,
                    },
                )
            )
        cycles.append(CycleConfig(cycle_index=cycle_index, games=cycle_games))
    description = (
        f"Continual benchmark: {config.continual_cycles} cycle(s) x {len(game_ids)} game(s) "
        f"(frames per game per cycle={config.continual_cycle_frames})"
    )
    return BenchmarkConfig(cycles=cycles, description=description)


def _write_continual_summary(results: Sequence[GameResult], path: str):
    summary_records = []
    for result in results:
        total_score = float(sum(result.episode_scores))
        mean_score = float(np.mean(result.episode_scores)) if result.episode_scores else 0.0
        summary_records.append(
            {
                "cycle_index": result.cycle_index,
                "game_index": result.game_index,
                "game": result.spec.name,
                "frame_offset": result.frame_offset,
                "frame_budget": result.frame_budget,
                "episodes": len(result.episode_scores),
                "total_episode_score": total_score,
                "mean_episode_score": mean_score,
            }
        )
    final_cycle_index = results[-1].cycle_index if results else -1
    final_cycle_total = sum(
        float(sum(result.episode_scores))
        for result in results
        if result.cycle_index == final_cycle_index
    )
    payload = {
        "final_cycle_index": final_cycle_index,
        "final_cycle_total_episode_score": final_cycle_total,
        "records": summary_records,
    }
    with open(path, "w", encoding="utf-8") as summary_file:
        json.dump(payload, summary_file, indent=2)
    print(f"Wrote continual summary to {path}")


def _run_single_environment(config: AgentConfig, writer: SummaryWriter, run_name: str, run_dir: str):
    env: Optional[gym.Env] = None
    try:
        env = make_env(config.env_id, config.seed)
        agent = Agent(env.observation_space, env.action_space, config)
        frames_per_step = FRAME_SKIP
        frame_budget_frames = max(config.total_steps, 1) * frames_per_step
        game_spec = GameSpec(
            name=config.env_id,
            frame_budget=frame_budget_frames,
            seed=config.seed,
            backend="gym",
            env_id=config.env_id,
        )
        handle = EnvironmentHandle(
            backend="gym",
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
        agent_bbf_frame_runner(agent, handle, context=context, writer=writer)
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

    base_env = make_env(game_ids[0], config.seed)
    try:
        agent = Agent(base_env.observation_space, base_env.action_space, config)
        base_shape = base_env.observation_space.shape
        min_actions = base_env.action_space.n
    finally:
        base_env.close()

    _validate_continual_envs(game_ids, config, base_shape, min_actions)
    benchmark_config = _build_continual_benchmark_config(game_ids, frame_budget_per_game, config)

    def frame_runner_with_writer(agent_instance, handle, *, context):
        return agent_bbf_frame_runner(agent_instance, handle, context=context, writer=writer)

    runner = BenchmarkRunner(
        agent,
        benchmark_config,
        frame_runner=frame_runner_with_writer,
        data_dir=run_dir,
        rank=0,
        default_seed=config.seed,
        average_frames=100_000,
        max_frames_without_reward=0,
        lives_as_episodes=1,
        save_incremental_models=False,
    )
    results = runner.run()
    summary_path = os.path.join(run_dir, "continual_summary.json")
    _write_continual_summary(results, summary_path)


def parse_args() -> AgentConfig:
    parser = argparse.ArgumentParser(description="BBF Atari agent with CleanRL components.")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gamma-initial", type=float, default=0.97)
    parser.add_argument("--gamma-final", type=float, default=0.997)
    parser.add_argument("--n-step-max", type=int, default=10)
    parser.add_argument("--n-step-min", type=int, default=3)
    parser.add_argument("--n-step-decay-steps", type=int, default=10_000)
    parser.add_argument("--reset-grad-interval", type=int, default=40_000)
    parser.add_argument("--replay-ratio", type=int, default=8)
    parser.add_argument("--spr-weight", type=float, default=2.0)
    parser.add_argument("--jumps", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.02)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.01)
    parser.add_argument("--prioritized-alpha", type=float, default=0.5)
    parser.add_argument("--prioritized-beta", type=float, default=0.4)
    parser.add_argument("--prioritized-eps", type=float, default=1e-6)
    parser.add_argument("--n-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--per-game-learning-starts", type=int, default=1_000)
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
        help="The wandb project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="The wandb entity/team.",
    )
    parser.add_argument(
        "--continual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the continual benchmark.",
    )
    parser.add_argument(
        "--continual-games",
        type=str,
        default="",
        help="Comma-separated Gym env ids to use in continual mode.",
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
    args = parser.parse_args()
    return AgentConfig(
        env_id=args.env_id,
        total_steps=args.total_steps,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma_initial=args.gamma_initial,
        gamma_final=args.gamma_final,
        n_step_max=args.n_step_max,
        n_step_min=args.n_step_min,
        n_step_decay_steps=args.n_step_decay_steps,
        reset_grad_interval=args.reset_grad_interval,
        replay_ratio=args.replay_ratio,
        spr_weight=args.spr_weight,
        jumps=args.jumps,
        max_grad_norm=args.max_grad_norm,
        tau=args.tau,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        exploration_fraction=args.exploration_fraction,
        start_e=args.start_e,
        end_e=args.end_e,
        prioritized_replay_alpha=args.prioritized_alpha,
        prioritized_replay_beta=args.prioritized_beta,
        prioritized_replay_eps=args.prioritized_eps,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        per_game_learning_starts=args.per_game_learning_starts,
        seed=args.seed,
        cuda=args.cuda,
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
        log_file="",
        continual=args.continual,
        continual_games=args.continual_games,
        continual_cycles=args.continual_cycles,
        continual_cycle_frames=args.continual_cycle_frames,
    )


def main():
    config = parse_args()
    log_root = "./results_final_bbf/smokey"
    os.makedirs(log_root, exist_ok=True)
    run_name = f"{config.env_id}__agent_bbf__{config.seed}__{int(time.time())}"

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
    config.log_file = os.path.join(run_dir, "agent.log")
    config_path = os.path.join(run_dir, "hyperparameters.json")
    with open(config_path, "w", encoding="utf-8") as cfg:
        json.dump(vars(config), cfg, indent=2)
    print(f"Logging run to {run_dir}")

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