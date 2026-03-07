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

"""
BBF (Bigger, Better, Faster) Agent wrapped for the Continual Learning Benchmark.
Original Paper: https://arxiv.org/abs/2305.19452

This implementation wraps the correct CleanRL BBF logic to fit the benchmark runner interface.
"""

from __future__ import annotations

import argparse
from collections import deque
import inspect
import math
import os
import random
import time
from dataclasses import dataclass
import json
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity, record_function
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # Optional runtime dependency.
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def add_text(self, *args, **kwargs):
            del args, kwargs

        def add_scalar(self, *args, **kwargs):
            del args, kwargs

        def close(self):
            return None

try:
    # Import Benchmark interfaces
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
except ModuleNotFoundError:
    BenchmarkConfig = BenchmarkRunner = CycleConfig = EnvironmentHandle = object  # type: ignore
    FrameRunnerContext = FrameRunnerResult = GameSpec = GameResult = object  # type: ignore

# Always-needed defaults (single-game and core agent logic must not depend on common).
FRAME_SKIP = 4
ATARI_CANONICAL_ACTIONS = 18
PROGRESS_POINTS = 10
DEFAULT_CONTINUAL_GAMES = ("ALE/Breakout-v5", "ALE/Pong-v5")
DEFAULT_CONTINUAL_CYCLES = 1
DEFAULT_CONTINUAL_CYCLE_FRAMES = 100_000

# Optional continual-mode helpers from common.
try:
    from common import (
        make_atari_env,
        update_progress_graphs,
        write_continual_summary,
        create_logger,
    )
except (ModuleNotFoundError, ImportError):
    make_atari_env = None

    def update_progress_graphs(*args, **kwargs):  # type: ignore
        return None

    def write_continual_summary(*args, **kwargs):  # type: ignore
        return None

    class _FallbackLogger:
        def log(self, msg: str) -> None:
            print(msg)

    def create_logger(*args, **kwargs):  # type: ignore
        return _FallbackLogger()

try:
    from cleanrl_utils.atari_wrappers import (
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
    )
except ModuleNotFoundError:
    class _IdentityWrapper:
        def __new__(cls, env, *args, **kwargs):
            del args, kwargs
            return env

    ClipRewardEnv = EpisodicLifeEnv = FireResetEnv = MaxAndSkipEnv = NoopResetEnv = _IdentityWrapper


def make_env(env_id, seed, idx, capture_video=False, run_name="", full_action_space=False):
    """Create a thunk for SyncVectorEnv (matches bbf_atari.py exactly).
    
    Args:
        full_action_space: If True, use all 18 Atari actions (for continual learning).
                          If False, use game-specific action space.
    """
    def thunk():
        # Gymnasium 1.0+ ALE configuration:
        # repeat_action_probability=0.0 ensures NO STICKY ACTIONS (Standard Atari 100k)
        # full_action_space=True gives all 18 actions for continual learning
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", frameskip=1, 
                          repeat_action_probability=0.0, full_action_space=full_action_space)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0,
                          full_action_space=full_action_space)
            
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
            
        # Official BBF uses reward clipping (standard for Atari 100k)
        env = ClipRewardEnv(env) 
        
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        # Keep this outermost so vector `final_info` carries episodic-life episode stats.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        env.action_space.seed(seed)
        return env
    return thunk


def make_eval_env(
    env_id: str,
    seed: int,
    *,
    full_action_space: bool = False,
    sticky_prob: float = 0.0,
    clip_rewards: bool = False,
):
    """Evaluation environment with true episodic returns (no EpisodicLife)."""
    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=sticky_prob,
        full_action_space=full_action_space,
    )
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    if clip_rewards:
        env = ClipRewardEnv(env)

    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env.action_space.seed(seed)
    return env


# =============================================================================
# =============================================================================
# BBF NETWORK + REPLAY CORE (faithful to official JAX BBF semantics)
# =============================================================================


def renormalize_spatial(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.view(tensor.shape[0], -1)
    max_value = flat.max(dim=-1, keepdim=True).values
    min_value = flat.min(dim=-1, keepdim=True).values
    flat = (flat - min_value) / (max_value - min_value + 1e-5)
    return flat.view_as(tensor)


class RandomShift(nn.Module):
    """DrQ-style random crop + intensity augmentation used by BBF."""

    def __init__(self, pad: int = 4, intensity_scale: float = 0.05):
        super().__init__()
        self.pad = int(pad)
        self.intensity_scale = float(intensity_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)

        b, c, h, w = x.shape
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
            max_off = 2 * self.pad + 1
            off_y = torch.randint(0, max_off, (b,), device=x.device)
            off_x = torch.randint(0, max_off, (b,), device=x.device)
            out = torch.empty((b, c, h, w), device=x.device, dtype=x.dtype)
            for i in range(b):
                out[i] = x[i, :, off_y[i]:off_y[i] + h, off_x[i]:off_x[i] + w]
        else:
            out = x

        if self.training:
            noise = torch.randn(b, 1, 1, 1, device=out.device, dtype=out.dtype).clamp_(-2.0, 2.0)
            out = out * (1.0 + self.intensity_scale * noise)
        return out


class ResidualBlock(nn.Module):
    """Pre-activation residual block used by Impala encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x


class ImpalaStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    def __init__(self, in_channels: int, width_scale: int = 4, num_blocks: int = 2):
        super().__init__()
        dims = [16, 32, 32]
        widths = [int(d * width_scale) for d in dims]
        self.stages = nn.ModuleList([
            ImpalaStage(in_channels, widths[0], num_blocks=num_blocks),
            ImpalaStage(widths[0], widths[1], num_blocks=num_blocks),
            ImpalaStage(widths[1], widths[2], num_blocks=num_blocks),
        ])
        self.out_channels = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return F.relu(x)


class ConvTransitionModel(nn.Module):
    """MuZero/SPR convolutional transition model over spatial latent."""

    def __init__(self, num_actions: int, latent_dim: int, renormalize: bool):
        super().__init__()
        self.num_actions = int(num_actions)
        self.latent_dim = int(latent_dim)
        self.renormalize = bool(renormalize)
        self.conv1 = nn.Conv2d(self.latent_dim + self.num_actions, self.latent_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1)

    def forward_one(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        b, _, h, w = latent.shape
        a_onehot = F.one_hot(action.long(), num_classes=self.num_actions).to(latent.dtype)
        a_onehot = a_onehot.view(b, self.num_actions, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([latent, a_onehot], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.renormalize:
            x = renormalize_spatial(x)
        return x

    def rollout(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.numel() == 0:
            return latent.new_zeros((latent.shape[0], 0, *latent.shape[1:]))
        preds = []
        x = latent
        for t in range(actions.shape[1]):
            x = self.forward_one(x, actions[:, t])
            preds.append(x)
        return torch.stack(preds, dim=1)


class LinearHead(nn.Module):
    """Rainbow linear head matching BBF's single hidden projection path."""

    def __init__(self, in_dim: int, num_actions: int, num_atoms: int, dueling: bool):
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_atoms = int(num_atoms)
        self.dueling = bool(dueling)
        self.advantage = nn.Linear(in_dim, self.num_actions * self.num_atoms)
        self.value = nn.Linear(in_dim, self.num_atoms) if self.dueling else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        adv = self.advantage(x).view(b, self.num_actions, self.num_atoms)
        if not self.dueling:
            return adv
        val = self.value(x).view(b, 1, self.num_atoms)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class BBFNetwork(nn.Module):
    """PyTorch port of official BBF RainbowDQNNetwork with SPR rollout path."""

    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        args,
        obs_hw: Sequence[int] = (84, 84),
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_atoms = int(args.n_atoms)
        self.dueling = bool(args.dueling)
        self.distributional = bool(args.distributional)
        self.renormalize = bool(args.renormalize)

        self.register_buffer("support", torch.linspace(args.v_min, args.v_max, self.num_atoms))
        self.delta_z = (args.v_max - args.v_min) / (self.num_atoms - 1)

        self.encoder = ImpalaEncoder(
            in_channels,
            width_scale=int(args.impala_width),
            num_blocks=int(args.impala_blocks),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, int(obs_hw[0]), int(obs_hw[1]))
            latent = self.encoder(dummy)
        self.spatial_shape = latent.shape[1:]
        self.representation_dim = int(np.prod(self.spatial_shape))
        latent_dim = self.spatial_shape[0]

        self.transition_model = ConvTransitionModel(
            num_actions=self.num_actions,
            latent_dim=latent_dim,
            renormalize=self.renormalize,
        )

        self.projection = nn.Linear(self.representation_dim, int(args.hidden_dim))
        self.predictor = nn.Linear(int(args.hidden_dim), int(args.hidden_dim))
        self.head = LinearHead(int(args.hidden_dim), self.num_actions, self.num_atoms, self.dueling)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        latent = self.encoder(x)
        if self.renormalize:
            latent = renormalize_spatial(latent)
        return latent

    def flatten_spatial_latent(self, spatial_latent: torch.Tensor) -> torch.Tensor:
        if spatial_latent.ndim == 4:
            return spatial_latent.reshape(spatial_latent.shape[0], -1)
        if spatial_latent.ndim == 5:
            return spatial_latent.reshape(spatial_latent.shape[0], spatial_latent.shape[1], -1)
        raise ValueError(f"Unexpected latent rank: {spatial_latent.ndim}")

    def encode_project(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        representation = self.flatten_spatial_latent(latent)
        return self.projection(representation)

    def spr_rollout(self, spatial_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pred_latents = self.transition_model.rollout(spatial_latent, actions)
        b, t = pred_latents.shape[0], pred_latents.shape[1]
        rep = self.flatten_spatial_latent(pred_latents).reshape(b * t, -1)
        proj = self.projection(rep)
        pred = self.predictor(proj)
        return pred.view(b, t, -1)

    def compute_outputs(
        self,
        x: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        do_rollout: bool = False,
    ) -> Dict[str, torch.Tensor]:
        spatial_latent = self.encode(x)
        representation = self.flatten_spatial_latent(spatial_latent)
        projected = self.projection(representation)
        hidden = F.relu(projected)
        logits = self.head(hidden)

        if self.distributional:
            probabilities = F.softmax(logits, dim=-1)
            q_values = (probabilities * self.support).sum(dim=-1)
        else:
            probabilities = torch.ones_like(logits)
            q_values = logits.squeeze(-1)

        spr_predictions = None
        if do_rollout:
            if actions is None:
                raise ValueError("actions must be provided when do_rollout=True")
            spr_predictions = self.spr_rollout(spatial_latent, actions)

        return {
            "q_values": q_values,
            "logits": logits,
            "probabilities": probabilities,
            "spr_predictions": spr_predictions,
            "representation": representation,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_outputs(x)["probabilities"]


class SumTree:
    """Deterministic priority sum-tree used for prioritized replay."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.max_priority = 1.0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def set(self, idx: int, priority: float) -> None:
        tree_idx = idx + self.capacity - 1
        change = float(priority) - self.tree[tree_idx]
        self.tree[tree_idx] = float(priority)
        self._propagate(tree_idx, change)
        self.max_priority = max(self.max_priority, float(priority))

    def get(self, s: float) -> int:
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)

    def stratified_sample(self, batch_size: int) -> np.ndarray:
        if self.total <= 0:
            raise RuntimeError("Cannot sample from an empty sum tree")
        segment = self.total / batch_size
        out = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            out[i] = self.get(np.random.uniform(low, high))
        return out

    def get_priority(self, indices: np.ndarray) -> np.ndarray:
        return self.tree[indices + self.capacity - 1]

    def reset_priorities(self, size: int) -> None:
        for i in range(int(size)):
            self.set(i, self.max_priority)

    @property
    def total(self) -> float:
        return float(self.tree[0])


def invalid_range(cursor: int, replay_capacity: int, stack_size: int, update_horizon: int) -> np.ndarray:
    return np.array(
        [
            (cursor - update_horizon + i) % replay_capacity
            for i in range(stack_size + update_horizon)
        ],
        dtype=np.int64,
    )


class SubsequenceReplayBuffer:
    """Replay buffer matching official BBF subsequence sampling semantics."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        stack_size: int,
        replay_capacity: int,
        batch_size: int,
        subseq_len: int,
        update_horizon: int,
        gamma: float,
        max_sample_attempts: int = 1000,
    ):
        self._stack_size = int(stack_size)
        self._batch_size = int(batch_size)
        self._subseq_len = int(subseq_len)
        self._update_horizon = int(update_horizon)
        self._gamma = float(gamma)
        self._capacity = int(replay_capacity)
        self._max_sample_attempts = int(max_sample_attempts)

        obs_shape = tuple(observation_shape)
        if len(obs_shape) == 3 and obs_shape[0] == self._stack_size:
            self.channels_last = False
            self._frame_shape = obs_shape[1:]
        elif len(obs_shape) == 3 and obs_shape[-1] == self._stack_size:
            self.channels_last = True
            self._frame_shape = obs_shape[:-1]
        elif len(obs_shape) == 2:
            self.channels_last = False
            self._frame_shape = obs_shape
        else:
            raise ValueError(f"Unsupported observation shape: {obs_shape}")

        self.observations = np.empty((self._capacity, *self._frame_shape), dtype=np.uint8)
        self.actions = np.empty((self._capacity,), dtype=np.int32)
        self.rewards = np.empty((self._capacity,), dtype=np.float32)
        self.terminals = np.empty((self._capacity,), dtype=np.uint8)
        self.episode_end_without_terminal = np.zeros((self._capacity,), dtype=np.uint8)

        self.add_count = 0
        self.total_steps = 0
        self.invalid_range = np.zeros((self._stack_size + self._update_horizon,), dtype=np.int64)
        self._cumulative_discount_vector = np.array(
            [math.pow(self._gamma, n) for n in range(self._update_horizon + 1)],
            dtype=np.float32,
        )

    @property
    def size(self) -> int:
        return min(self.add_count, self._capacity)

    def is_full(self) -> bool:
        return self.add_count >= self._capacity

    def cursor(self) -> int:
        return int(self.add_count % self._capacity)

    def num_elements(self) -> int:
        return int(self.size)

    def reset_counters(self) -> None:
        self.add_count = 0
        self.total_steps = 0
        self.invalid_range = np.zeros((self._stack_size + self._update_horizon,), dtype=np.int64)

    def _extract_latest_frame(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs)
        if arr.ndim == 3 and arr.shape[0] == self._stack_size:
            frame = arr[-1]
        elif arr.ndim == 3 and arr.shape[-1] == self._stack_size:
            frame = arr[..., -1]
        elif arr.ndim == 2:
            frame = arr
        else:
            raise ValueError(f"Unexpected observation for replay: shape={arr.shape}")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        return frame

    def add(
        self,
        obs,
        action: int,
        reward: float,
        terminal: bool,
        *,
        episode_end: bool,
        priority: Optional[float] = None,
    ) -> None:
        del priority
        idx = self.cursor()
        self.observations[idx] = self._extract_latest_frame(obs)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.terminals[idx] = 1 if terminal else 0
        self.episode_end_without_terminal[idx] = 1 if (episode_end and not terminal) else 0

        self.add_count += 1
        self.total_steps += 1
        self.invalid_range = invalid_range(self.cursor(), self._capacity, self._stack_size, self._update_horizon)

    def _parallel_get_stack(self, abs_indices: np.ndarray, censor_before: np.ndarray) -> np.ndarray:
        offsets = np.arange(-self._stack_size + 1, 1, dtype=np.int64)[:, None]
        stack_abs = abs_indices[None, :] + offsets
        mask = stack_abs >= censor_before[None, :]

        stack = self.observations[stack_abs % self._capacity]
        while mask.ndim < stack.ndim:
            mask = np.expand_dims(mask, -1)
        stack = stack * mask.astype(stack.dtype)
        stack = np.moveaxis(stack, 0, -1)  # (N, H, W, stack)
        if not self.channels_last:
            stack = np.moveaxis(stack, -1, 1)  # (N, stack, H, W)
        return stack

    def _is_valid_transition(self, abs_index: int, update_horizon: int, subseq_len: int):
        if not self.is_full():
            if abs_index >= self.add_count - update_horizon - subseq_len:
                return False, 0
            if abs_index < self._stack_size - 1:
                return False, 0

        slot = abs_index % self._capacity
        if slot in set(self.invalid_range.tolist()):
            return False, 0

        stack_abs = np.arange(abs_index - self._stack_size + 1, abs_index + 1, dtype=np.int64)
        stack_terminals = self.terminals[stack_abs % self._capacity][:-1]
        if stack_terminals.any():
            ep_start = abs_index - self._stack_size + int(np.argmax(stack_terminals)) + 2
        else:
            ep_start = 0

        for k in range(update_horizon):
            slot_k = (abs_index + k) % self._capacity
            if self.episode_end_without_terminal[slot_k]:
                return False, 0

        return True, ep_start

    def _uniform_sample_indices(self, batch_size: int, update_horizon: int, subseq_len: int):
        if self.is_full():
            min_abs = self.add_count - self._capacity + self._stack_size - 1
        else:
            min_abs = self._stack_size - 1
        max_abs = self.add_count - update_horizon - subseq_len
        if max_abs <= min_abs:
            raise RuntimeError(
                f"Cannot sample: need more than stack_size({self._stack_size}) + update_horizon({update_horizon}) + subseq_len({subseq_len})"
            )

        abs_indices = np.random.randint(min_abs, max_abs, size=batch_size, dtype=np.int64)
        censor_before = np.zeros(batch_size, dtype=np.int64)

        attempts_left = self._max_sample_attempts
        for i in range(batch_size):
            valid, ep_start = self._is_valid_transition(int(abs_indices[i]), update_horizon, subseq_len)
            while not valid and attempts_left > 0:
                abs_indices[i] = np.random.randint(min_abs, max_abs, dtype=np.int64)
                valid, ep_start = self._is_valid_transition(int(abs_indices[i]), update_horizon, subseq_len)
                attempts_left -= 1
            if not valid:
                raise RuntimeError("Max sample attempts exhausted while sampling replay")
            censor_before[i] = ep_start

        return abs_indices, censor_before

    def sample_transition_batch(
        self,
        *,
        batch_size: Optional[int] = None,
        update_horizon: Optional[int] = None,
        subseq_len: Optional[int] = None,
        gamma: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        batch_size = self._batch_size if batch_size is None else int(batch_size)
        subseq_len = self._subseq_len if subseq_len is None else int(subseq_len)
        update_horizon = self._update_horizon if update_horizon is None else int(update_horizon)

        abs_indices, censor_before = self._uniform_sample_indices(batch_size, update_horizon, subseq_len)

        if gamma is None:
            cumulative_discount_vector = self._cumulative_discount_vector
        else:
            cumulative_discount_vector = np.array(
                [math.pow(float(gamma), n) for n in range(update_horizon + 1)],
                dtype=np.float32,
            )

        state_abs = abs_indices[:, None] + np.arange(subseq_len, dtype=np.int64)[None, :]
        flat_state_abs = state_abs.reshape(-1)
        flat_censor = censor_before[:, None].repeat(subseq_len, axis=1).reshape(-1)

        traj_abs = (
            np.arange(-1, update_horizon - 1, dtype=np.int64)[:, None] + flat_state_abs[None, :]
        )
        trajectory_terminals = self.terminals[traj_abs % self._capacity].astype(np.float32)
        trajectory_terminals[0, :] = 0.0

        is_terminal_transition = trajectory_terminals.any(axis=0).astype(np.uint8)
        valid_mask = (1.0 - trajectory_terminals).cumprod(axis=0)
        trajectory_discount = valid_mask * cumulative_discount_vector[:update_horizon, None]
        trajectory_rewards = self.rewards[(traj_abs + 1) % self._capacity]
        returns = np.cumsum(trajectory_discount * trajectory_rewards, axis=0)

        horizon_idx = update_horizon - 1
        returns = returns[horizon_idx, np.arange(flat_state_abs.shape[0])]
        next_abs = flat_state_abs + horizon_idx

        state = self._parallel_get_stack(flat_state_abs, flat_censor)
        next_state = self._parallel_get_stack(next_abs, flat_censor)

        slots = flat_state_abs % self._capacity
        next_slots = next_abs % self._capacity

        action = self.actions[slots].reshape(batch_size, subseq_len)
        reward = self.rewards[slots].reshape(batch_size, subseq_len)
        terminal = is_terminal_transition.reshape(batch_size, subseq_len)
        same_trajectory = self.terminals[slots].reshape(batch_size, subseq_len).copy()
        same_trajectory[0, :] = 0
        same_trajectory = (1 - same_trajectory).cumprod(axis=1).astype(np.uint8)

        indices = slots.reshape(batch_size, subseq_len)[:, 0].astype(np.int32)
        discount = np.full((batch_size, subseq_len), cumulative_discount_vector[horizon_idx + 1], dtype=np.float32)

        return {
            "state": state.reshape(batch_size, subseq_len, *state.shape[1:]),
            "action": action,
            "reward": reward,
            "return": returns.reshape(batch_size, subseq_len),
            "discount": discount,
            "next_state": next_state.reshape(batch_size, subseq_len, *next_state.shape[1:]),
            "next_action": self.actions[next_slots].reshape(batch_size, subseq_len),
            "next_reward": self.rewards[next_slots].reshape(batch_size, subseq_len),
            "terminal": terminal,
            "same_trajectory": same_trajectory,
            "indices": indices,
            "sampling_probabilities": np.ones((batch_size,), dtype=np.float32),
        }

    def set_priority(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        del indices, priorities

    def reset_priorities(self) -> None:
        pass


class PrioritizedSubsequenceReplayBuffer(SubsequenceReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sum_tree = SumTree(self._capacity)

    def add(
        self,
        obs,
        action: int,
        reward: float,
        terminal: bool,
        *,
        episode_end: bool,
        priority: Optional[float] = None,
    ) -> None:
        idx = self.cursor()
        super().add(
            obs,
            action,
            reward,
            terminal,
            episode_end=episode_end,
            priority=priority,
        )
        if priority is None:
            priority = self.sum_tree.max_priority
        self.sum_tree.set(idx, float(priority))

    def _priority_sample_indices(self, batch_size: int, update_horizon: int, subseq_len: int):
        slots = self.sum_tree.stratified_sample(batch_size)

        abs_indices = np.empty(batch_size, dtype=np.int64)
        censor_before = np.zeros(batch_size, dtype=np.int64)
        attempts_left = self._max_sample_attempts

        for i in range(batch_size):
            slot = int(slots[i])
            if self.is_full():
                base = (self.add_count // self._capacity) * self._capacity
                abs_idx = slot + base
                if abs_idx >= self.add_count:
                    abs_idx -= self._capacity
            else:
                abs_idx = slot

            valid, ep_start = self._is_valid_transition(abs_idx, update_horizon, subseq_len)
            while not valid and attempts_left > 0:
                slot = int(self.sum_tree.stratified_sample(1)[0])
                if self.is_full():
                    base = (self.add_count // self._capacity) * self._capacity
                    abs_idx = slot + base
                    if abs_idx >= self.add_count:
                        abs_idx -= self._capacity
                else:
                    abs_idx = slot
                valid, ep_start = self._is_valid_transition(abs_idx, update_horizon, subseq_len)
                attempts_left -= 1
            if not valid:
                raise RuntimeError("Max sample attempts exhausted in prioritized sampling")

            abs_indices[i] = abs_idx
            censor_before[i] = ep_start
            slots[i] = abs_idx % self._capacity

        return abs_indices, censor_before, slots.astype(np.int32)

    def sample_transition_batch(
        self,
        *,
        batch_size: Optional[int] = None,
        update_horizon: Optional[int] = None,
        subseq_len: Optional[int] = None,
        gamma: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        batch_size = self._batch_size if batch_size is None else int(batch_size)
        subseq_len = self._subseq_len if subseq_len is None else int(subseq_len)
        update_horizon = self._update_horizon if update_horizon is None else int(update_horizon)

        abs_indices, censor_before, _ = self._priority_sample_indices(batch_size, update_horizon, subseq_len)

        if gamma is None:
            cumulative_discount_vector = self._cumulative_discount_vector
        else:
            cumulative_discount_vector = np.array(
                [math.pow(float(gamma), n) for n in range(update_horizon + 1)],
                dtype=np.float32,
            )

        state_abs = abs_indices[:, None] + np.arange(subseq_len, dtype=np.int64)[None, :]
        flat_state_abs = state_abs.reshape(-1)
        flat_censor = censor_before[:, None].repeat(subseq_len, axis=1).reshape(-1)

        traj_abs = (
            np.arange(-1, update_horizon - 1, dtype=np.int64)[:, None] + flat_state_abs[None, :]
        )
        trajectory_terminals = self.terminals[traj_abs % self._capacity].astype(np.float32)
        trajectory_terminals[0, :] = 0.0

        is_terminal_transition = trajectory_terminals.any(axis=0).astype(np.uint8)
        valid_mask = (1.0 - trajectory_terminals).cumprod(axis=0)
        trajectory_discount = valid_mask * cumulative_discount_vector[:update_horizon, None]
        trajectory_rewards = self.rewards[(traj_abs + 1) % self._capacity]
        returns = np.cumsum(trajectory_discount * trajectory_rewards, axis=0)

        horizon_idx = update_horizon - 1
        returns = returns[horizon_idx, np.arange(flat_state_abs.shape[0])]
        next_abs = flat_state_abs + horizon_idx

        state = self._parallel_get_stack(flat_state_abs, flat_censor)
        next_state = self._parallel_get_stack(next_abs, flat_censor)

        slots = flat_state_abs % self._capacity
        next_slots = next_abs % self._capacity

        action = self.actions[slots].reshape(batch_size, subseq_len)
        reward = self.rewards[slots].reshape(batch_size, subseq_len)
        terminal = is_terminal_transition.reshape(batch_size, subseq_len)
        same_trajectory = self.terminals[slots].reshape(batch_size, subseq_len).copy()
        same_trajectory[0, :] = 0
        same_trajectory = (1 - same_trajectory).cumprod(axis=1).astype(np.uint8)

        indices = slots.reshape(batch_size, subseq_len)[:, 0].astype(np.int32)
        sampling_probabilities = self.sum_tree.get_priority(indices).astype(np.float32)
        discount = np.full((batch_size, subseq_len), cumulative_discount_vector[horizon_idx + 1], dtype=np.float32)

        return {
            "state": state.reshape(batch_size, subseq_len, *state.shape[1:]),
            "action": action,
            "reward": reward,
            "return": returns.reshape(batch_size, subseq_len),
            "discount": discount,
            "next_state": next_state.reshape(batch_size, subseq_len, *next_state.shape[1:]),
            "next_action": self.actions[next_slots].reshape(batch_size, subseq_len),
            "next_reward": self.rewards[next_slots].reshape(batch_size, subseq_len),
            "terminal": terminal,
            "same_trajectory": same_trajectory,
            "indices": indices,
            "sampling_probabilities": sampling_probabilities,
        }

    def set_priority(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int32).reshape(-1)
        priorities = np.asarray(priorities, dtype=np.float32).reshape(-1)
        for idx, priority in zip(indices, priorities):
            self.sum_tree.set(int(idx), float(priority))

    def reset_priorities(self) -> None:
        self.sum_tree.reset_priorities(self.size)


def linearly_decaying_epsilon(
    decay_period: float,
    step: int,
    warmup_steps: float,
    epsilon: float,
) -> float:
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / max(1.0, decay_period)
    bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
    return float(epsilon + bonus)


def exponential_decay_scheduler(
    decay_period: int,
    warmup_steps: int,
    initial_value: float,
    final_value: float,
    reverse: bool = False,
):
    if reverse:
        initial_value = 1 - initial_value
        final_value = 1 - final_value

    start = np.log(initial_value)
    end = np.log(final_value)

    if decay_period == 0:
        return lambda x: initial_value if x < warmup_steps else final_value

    def scheduler(step: int) -> float:
        steps_left = decay_period + warmup_steps - step
        bonus_frac = steps_left / decay_period
        bonus = np.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (start - end) + end
        new_value = float(np.exp(new_value))
        if reverse:
            new_value = 1 - new_value
        return new_value

    return scheduler


@dataclass
class AgentConfig:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "physical-atari"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = False
    profile: bool = False
    torch_profile: bool = False
    torch_profile_steps: int = 50

    env_id: str = "ALE/Breakout-v5"
    total_steps: int = 100_000
    num_envs: int = 1
    buffer_size: int = 200_000
    full_action_space: bool = False
    continual_reset_on_game_switch: bool = False
    allow_resets_in_continual: bool = False
    continual_sticky_prob: float = 0.25
    continual_delay_frames: int = 0

    learning_rate: float = 0.0001
    encoder_learning_rate: float = 0.0001
    weight_decay: float = 0.1

    noisy: bool = False
    dueling: bool = True
    double_dqn: bool = True
    distributional: bool = True
    use_target_network: bool = True
    target_action_selection: bool = True

    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0

    # Official BBF default architecture
    impala_width: int = 4
    impala_blocks: int = 2
    hidden_dim: int = 2048
    renormalize: bool = True

    # Replay/update schedule (official BBF semantics)
    replay_ratio: int = 64
    batches_to_group: int = 2
    batch_size: int = 32
    learning_starts: int = 2000
    update_horizon: int = 3
    max_update_horizon: int = 10
    gamma: float = 0.997
    min_gamma: float = 0.97
    cycle_steps: int = 10_000

    # Target update
    target_update_period: int = 1
    target_update_tau: float = 0.005
    max_target_update_tau: Optional[float] = None

    # Exploration
    epsilon_train: float = 0.0
    epsilon_eval: float = 0.001
    epsilon_decay_period: int = 2001

    # Resets
    reset_every: int = 20_000
    no_resets_after: int = 100_000
    reset_offset: int = 1
    reset_target: bool = True
    reset_head: bool = True
    reset_projection: bool = True
    reset_encoder: bool = False
    reset_noise: bool = True
    reset_priorities: bool = False
    reset_interval_scaling: Optional[str] = None
    shrink_perturb_keys: str = "encoder,transition_model"
    shrink_factor: float = 0.5
    perturb_factor: float = 0.5
    offline_update_frac: float = 0.0

    # SPR
    jumps: int = 5
    spr_weight: float = 5.0
    data_augmentation: bool = True

    # Replay scheme
    use_per: bool = True

    # Logging / compatibility fields
    log_file: str = ""
    log_every: int = 100
    channels_last_memory: bool = False
    use_amp: bool = False
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"

    continual: bool = False
    continual_games: str = ""
    continual_cycles: int = DEFAULT_CONTINUAL_CYCLES
    continual_cycle_frames: int = DEFAULT_CONTINUAL_CYCLE_FRAMES
    per_game_learning_starts: int = 2000

    eval_episodes: int = 10
    eval_epsilon: float = 0.001
    eval_sticky_prob: float = 0.0
    eval_clip_rewards: bool = False
    eval_select_interval: int = 25_000
    eval_select_start: int = 25_000
    eval_select_episodes: int = 2

    # Legacy CLI fields kept for compatibility with existing scripts.
    initial_gamma: float = 0.97
    final_gamma: float = 0.997
    initial_n_step: int = 10
    final_n_step: int = 3
    anneal_duration: int = 10_000
    reset_interval: int = 20_000
    reset_warmup_steps: int = 0
    reset_disable_last_steps: int = 0
    start_e: float = 0.0
    end_e: float = 0.0
    exploration_fraction: float = 0.0
    per_alpha: float = 0.5
    per_beta: float = 0.5
    per_eps: float = 1e-6
    tau: float = 0.005

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")

    def __post_init__(self) -> None:
        # Keep legacy parser flags wired into active BBF fields.
        self.reset_every = int(self.reset_interval)
        self.target_update_tau = float(self.tau)


class Agent:
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: AgentConfig):
        self.config = config
        self.device = config.device

        if config.continual:
            self.num_actions = ATARI_CANONICAL_ACTIONS
        else:
            self.num_actions = action_space.n

        obs_shape = tuple(observation_space.shape)
        if len(obs_shape) == 2:
            self.frame_shape = obs_shape
        elif len(obs_shape) == 3 and obs_shape[0] in (1, FRAME_SKIP):
            self.frame_shape = obs_shape[1:]
        elif len(obs_shape) == 3 and obs_shape[-1] in (1, FRAME_SKIP):
            self.frame_shape = obs_shape[:2]
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")

        self.in_channels = FRAME_SKIP
        self.q_network = BBFNetwork(self.in_channels, self.num_actions, config, obs_hw=self.frame_shape).to(self.device)
        self.target_network = BBFNetwork(self.in_channels, self.num_actions, config, obs_hw=self.frame_shape).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.aug = RandomShift(pad=4).to(self.device)

        self.optimizer = self._build_optimizer()

        self.shrink_perturb_keys = tuple(
            s.strip() for s in str(config.shrink_perturb_keys).lower().split(",") if s.strip()
        )

        if config.max_update_horizon is None:
            self.max_update_horizon = int(config.update_horizon)
            self.update_horizon_scheduler = lambda _: int(config.update_horizon)
        else:
            self.max_update_horizon = int(config.max_update_horizon)
            n_schedule = exponential_decay_scheduler(
                int(config.cycle_steps),
                0,
                1.0,
                float(config.update_horizon) / float(self.max_update_horizon),
            )
            self.update_horizon_scheduler = lambda step: int(np.round(n_schedule(step) * self.max_update_horizon))

        if config.min_gamma is None or int(config.cycle_steps) <= 1:
            self.gamma_scheduler = lambda _: float(config.gamma)
        else:
            self.gamma_scheduler = exponential_decay_scheduler(
                int(config.cycle_steps),
                0,
                float(config.min_gamma),
                float(config.gamma),
                reverse=True,
            )

        if config.max_target_update_tau is None:
            self.target_update_tau_scheduler = lambda _: float(config.target_update_tau)
        else:
            self.target_update_tau_scheduler = exponential_decay_scheduler(
                int(config.cycle_steps),
                0,
                float(config.max_target_update_tau),
                float(config.target_update_tau),
            )

        self.seq_len = int(config.jumps) + 1
        replay_kwargs = dict(
            observation_shape=self.frame_shape,
            stack_size=FRAME_SKIP,
            replay_capacity=int(config.buffer_size),
            batch_size=int(config.batch_size),
            subseq_len=self.seq_len,
            update_horizon=self.max_update_horizon,
            gamma=float(config.gamma),
        )
        if config.use_per:
            self.replay_buffer = PrioritizedSubsequenceReplayBuffer(**replay_kwargs)
        else:
            self.replay_buffer = SubsequenceReplayBuffer(**replay_kwargs)

        self._set_replay_settings()

        self.training_steps = 0
        self.grad_steps = 0
        self.cycle_grad_steps = 0
        self.global_step = 0
        self.grad_step = 0

        self.cumulative_resets = 0
        self.next_reset = int(config.reset_every) + int(config.reset_offset)

        self.last_obs: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self._act_state: Optional[np.ndarray] = None
        self.learning_ready_step = int(config.learning_starts)

        self.game_frame_counters: Dict[str, int] = {}
        self.profiler = None
        self.profiler_active = False

    def _set_replay_settings(self) -> None:
        n_envs = 1
        self._num_updates_per_train_step = max(1, int(self.config.replay_ratio) * n_envs // int(self.config.batch_size))
        self.update_period = max(1, int(self.config.batch_size) // max(1, int(self.config.replay_ratio)) * n_envs)
        self._batches_to_group = min(int(self.config.batches_to_group), self._num_updates_per_train_step)
        if self._num_updates_per_train_step % self._batches_to_group != 0:
            raise ValueError("replay ratio / batch grouping mismatch")
        self._num_updates_per_train_step = max(1, self._num_updates_per_train_step // self._batches_to_group)

    def _step_profiler(self) -> None:
        if not self.profiler_active or self.profiler is None:
            return
        try:
            self.profiler.step()
        except Exception as exc:
            print(f"[WARN] profiler.step() failed; disabling profiler. ({exc})")
            self.profiler_active = False

    def _build_optimizer(self) -> optim.Optimizer:
        cfg = self.config
        enc_decay, enc_no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, param in self.q_network.named_parameters():
            if not param.requires_grad:
                continue
            is_encoder = name.startswith("encoder") or name.startswith("transition_model")
            is_no_decay = (param.ndim <= 1) or name.endswith(".bias")

            if is_encoder and is_no_decay:
                enc_no_decay.append(param)
            elif is_encoder:
                enc_decay.append(param)
            elif is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)

        groups = []
        if enc_decay:
            groups.append(
                {
                    "params": enc_decay,
                    "lr": float(cfg.encoder_learning_rate),
                    "weight_decay": float(cfg.weight_decay),
                }
            )
        if enc_no_decay:
            groups.append(
                {
                    "params": enc_no_decay,
                    "lr": float(cfg.encoder_learning_rate),
                    "weight_decay": 0.0,
                }
            )
        if head_decay:
            groups.append(
                {
                    "params": head_decay,
                    "lr": float(cfg.learning_rate),
                    "weight_decay": float(cfg.weight_decay),
                }
            )
        if head_no_decay:
            groups.append(
                {
                    "params": head_no_decay,
                    "lr": float(cfg.learning_rate),
                    "weight_decay": 0.0,
                }
            )

        adamw_kwargs = {"eps": 1.5e-4}
        if "fused" in inspect.signature(optim.AdamW).parameters:
            adamw_kwargs["fused"] = self.device.type == "cuda"
        return optim.AdamW(groups, **adamw_kwargs)

    def _prefix_match(self, name: str, prefixes: Sequence[str]) -> bool:
        return any(name == p or name.startswith(f"{p}.") for p in prefixes)

    def _copy_tensor_dict(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in module.state_dict().items()}

    def _blend_shrink_perturb(
        self,
        old_state: Dict[str, torch.Tensor],
        random_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not self.shrink_perturb_keys:
            return old_state
        out = {}
        for k, old_v in old_state.items():
            if self._prefix_match(k, self.shrink_perturb_keys):
                out[k] = (
                    float(self.config.shrink_factor) * old_v
                    + float(self.config.perturb_factor) * random_state[k]
                )
            else:
                out[k] = old_v
        return out

    def _reset_interval(self) -> int:
        cfg = self.config
        scale = cfg.reset_interval_scaling
        if scale is None or not scale:
            return int(cfg.reset_every)
        if str(scale).lower() == "linear":
            return int(cfg.reset_every * (1 + self.cumulative_resets))
        if "epoch" in str(scale):
            epochs = float(str(scale).replace("epochs:", ""))
            denom = max(1, int(cfg.batch_size) * self._num_updates_per_train_step * self._batches_to_group)
            steps = epochs * self.replay_buffer.num_elements() / denom
            return int(steps) + int(cfg.reset_every)
        scale_f = float(scale)
        return int(cfg.reset_every * (scale_f ** self.cumulative_resets))

    def reset_weights(self, *, force: bool = False) -> bool:
        self.cumulative_resets += 1
        interval = self._reset_interval()

        self.next_reset = int(interval) + int(self.training_steps)
        if not force and self.next_reset > int(self.config.no_resets_after) + int(self.config.reset_offset):
            return False

        old_online = self._copy_tensor_dict(self.q_network)
        old_target = self._copy_tensor_dict(self.target_network)

        fresh_online = BBFNetwork(self.in_channels, self.num_actions, self.config).to(self.device)
        fresh_target = BBFNetwork(self.in_channels, self.num_actions, self.config).to(self.device)
        random_online = self._copy_tensor_dict(fresh_online)
        random_target = self._copy_tensor_dict(fresh_target)

        old_online = self._blend_shrink_perturb(old_online, random_online)
        old_target = self._blend_shrink_perturb(old_target, random_target)

        keys_to_copy: List[str] = []
        if not self.config.reset_projection:
            keys_to_copy.append("projection")
            if self.config.spr_weight > 0:
                keys_to_copy.append("predictor")
        if not self.config.reset_encoder:
            keys_to_copy.append("encoder")
            if self.config.spr_weight > 0:
                keys_to_copy.append("transition_model")
        if not self.config.reset_head:
            keys_to_copy.append("head")

        new_online = {k: v.detach().clone() for k, v in random_online.items()}
        for k in new_online:
            if self._prefix_match(k, keys_to_copy):
                new_online[k] = old_online[k]
        self.q_network.load_state_dict(new_online)

        if self.config.reset_target:
            new_target = {k: v.detach().clone() for k, v in random_target.items()}
            for k in new_target:
                if self._prefix_match(k, keys_to_copy):
                    new_target[k] = old_target[k]
            self.target_network.load_state_dict(new_target)

        kept_state = {}
        old_optim_state = self.optimizer.state
        for name, param in self.q_network.named_parameters():
            if self._prefix_match(name, keys_to_copy) and param in old_optim_state:
                kept_state[name] = {
                    k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in old_optim_state[param].items()
                }

        self.optimizer = self._build_optimizer()
        param_by_name = dict(self.q_network.named_parameters())
        for name, state in kept_state.items():
            if name in param_by_name:
                self.optimizer.state[param_by_name[name]] = state

        self.cycle_grad_steps = 0

        if self.config.reset_priorities:
            self.replay_buffer.reset_priorities()

        if self.replay_buffer.add_count > self.config.learning_starts:
            offline_steps = int(interval * float(self.config.offline_update_frac) * self._num_updates_per_train_step)
            for _ in range(max(0, offline_steps)):
                self._training_step_update(offline=True)

        return True

    def start_new_game(self):
        self.last_obs = None
        self.last_action = None
        self._act_state = None
        self.learning_ready_step = self.training_steps + int(self.config.per_game_learning_starts)

        if self.config.continual and self.config.continual_reset_on_game_switch:
            self.replay_buffer.reset_counters()
            self.reset_weights(force=True)

    def _current_epsilon(self, eval_mode: bool = False) -> float:
        if eval_mode:
            return float(self.config.epsilon_eval)
        return linearly_decaying_epsilon(
            float(self.config.epsilon_decay_period),
            int(self.training_steps),
            float(self.config.learning_starts),
            float(self.config.epsilon_train),
        )

    def _extract_frame(self, observation) -> np.ndarray:
        obs_np = np.asarray(observation)
        if obs_np.ndim == 2:
            frame = obs_np
        elif obs_np.ndim == 3 and obs_np.shape[0] == FRAME_SKIP:
            frame = obs_np[-1]
        elif obs_np.ndim == 3 and obs_np.shape[-1] == FRAME_SKIP:
            frame = obs_np[..., -1]
        elif obs_np.ndim == 3 and obs_np.shape[0] == 1:
            frame = obs_np[0]
        elif obs_np.ndim == 3 and obs_np.shape[-1] == 1:
            frame = obs_np[..., 0]
        else:
            raise ValueError(f"Unexpected observation shape: {obs_np.shape}")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        return frame

    def _build_act_state(self, observation) -> np.ndarray:
        frame = self._extract_frame(observation)
        if self._act_state is None:
            self._act_state = np.zeros((FRAME_SKIP, *self.frame_shape), dtype=np.uint8)
        self._act_state = np.roll(self._act_state, -1, axis=0)
        self._act_state[-1] = frame
        return self._act_state

    def _prepare_obs_for_network(self, stacked_state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(stacked_state, device=self.device, dtype=torch.uint8).unsqueeze(0)

    def act(self, observation) -> int:
        stacked_state = self._build_act_state(observation)
        current_frame = stacked_state[-1]
        epsilon = self._current_epsilon(eval_mode=False)
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            obs_t = self._prepare_obs_for_network(stacked_state)
            with torch.no_grad():
                net = self.target_network if self.config.target_action_selection else self.q_network
                q_values = net.compute_outputs(obs_t)["q_values"]
                action = int(torch.argmax(q_values, dim=1).item())

        self.last_obs = current_frame
        self.last_action = int(action)
        return int(action)

    @staticmethod
    def _project_distribution(
        target_support: torch.Tensor,
        next_probabilities: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        v_min = float(support[0].item())
        v_max = float(support[-1].item())
        n_atoms = int(support.shape[0])
        delta_z = (v_max - v_min) / (n_atoms - 1)

        tz = target_support.clamp(v_min, v_max)
        b = (tz - v_min) / delta_z
        l = b.floor().long().clamp_(0, n_atoms - 1)
        u = b.ceil().long().clamp_(0, n_atoms - 1)

        batch_size = target_support.shape[0]
        target = torch.zeros_like(next_probabilities)
        offset = (torch.arange(batch_size, device=target.device) * n_atoms).unsqueeze(1)

        target.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (next_probabilities * (u.float() - b)).view(-1),
        )
        target.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (next_probabilities * (b - l.float())).view(-1),
        )

        same_bucket = (u == l)
        if same_bucket.any():
            target.view(-1).index_add_(
                0,
                (l + offset)[same_bucket],
                next_probabilities[same_bucket],
            )

        return target

    def _process_states(self, raw_states: torch.Tensor, augment: bool) -> torch.Tensor:
        b, t, c, h, w = raw_states.shape
        x = raw_states.reshape(b * t, c, h, w)
        x = x.float().div_(255.0)
        if augment:
            x = self.aug(x)
        return x.view(b, t, c, h, w)

    def _process_single_states(self, raw_states: torch.Tensor, augment: bool) -> torch.Tensor:
        x = raw_states.float().div_(255.0)
        if augment:
            x = self.aug(x)
        return x

    def _soft_update_target(self, tau: float) -> None:
        if tau >= 1.0 or tau < 0.0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            return
        with torch.no_grad():
            for p, tp in zip(self.q_network.parameters(), self.target_network.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def _train_batch_chunk(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        tau: float,
    ) -> Dict[str, torch.Tensor]:
        rf = record_function if self.profiler_active else (lambda _: nullcontext())
        self.q_network.train()
        self.target_network.train()

        with rf("aug"):
            states = self._process_states(batch["state"], augment=self.config.data_augmentation)
            next_states = self._process_single_states(batch["next_state"][:, 0], augment=self.config.data_augmentation)

        with rf("batch/prep"):
            actions = batch["action"].long()
            n_step_returns = batch["return"][:, 0].float()
            terminals = batch["terminal"][:, 0].float()
            same_traj_mask = batch["same_trajectory"][:, 1:].float()
            loss_weights = batch["loss_weights"].float()
            cumulative_gamma = batch["discount"][:, 0].float()

        current_state = states[:, 0]
        use_spr = self.config.spr_weight > 0

        with rf("fwd/online"):
            online_out = self.q_network.compute_outputs(
                current_state,
                actions=actions[:, :-1],
                do_rollout=use_spr,
            )
            logits = online_out["logits"]

        with rf("fwd/target_c51"), torch.no_grad():
            if self.config.use_target_network or not self.config.double_dqn:
                target_next_logits = self.target_network.compute_outputs(next_states)["logits"]
            else:
                target_next_logits = None

            if (not self.config.use_target_network) or self.config.double_dqn:
                online_next_logits = self.q_network.compute_outputs(next_states)["logits"]
            else:
                online_next_logits = None

            backup_logits = target_next_logits if self.config.use_target_network else online_next_logits
            select_logits = online_next_logits if self.config.double_dqn else target_next_logits

            select_prob = F.softmax(select_logits, dim=-1)
            select_q = (select_prob * self.q_network.support).sum(dim=-1)
            next_action = torch.argmax(select_q, dim=-1)

            backup_prob = F.softmax(backup_logits, dim=-1)
            next_probabilities = backup_prob[torch.arange(backup_prob.shape[0], device=self.device), next_action]

            gamma_terminal = cumulative_gamma * (1.0 - terminals)
            target_support = n_step_returns.unsqueeze(1) + gamma_terminal.unsqueeze(1) * self.q_network.support.unsqueeze(0)
            target_dist = self._project_distribution(target_support, next_probabilities, self.q_network.support)

        with rf("loss/c51"):
            chosen_logits = logits[torch.arange(logits.shape[0], device=self.device), actions[:, 0]]
            dqn_loss = -(target_dist * F.log_softmax(chosen_logits, dim=-1)).sum(dim=-1)

        with rf("loss/spr"):
            if use_spr:
                spr_pred = online_out["spr_predictions"]
                with torch.no_grad():
                    future_states = states[:, 1:]
                    b, j, c, h, w = future_states.shape
                    target_proj = self.target_network.encode_project(future_states.reshape(b * j, c, h, w))
                    spr_target = target_proj.view(b, j, -1)

                spr_pred = F.normalize(spr_pred, p=2, dim=-1)
                spr_target = F.normalize(spr_target, p=2, dim=-1)
                spr_loss_t = ((spr_pred - spr_target) ** 2).sum(dim=-1)
                spr_loss = (spr_loss_t * same_traj_mask).mean(dim=1)
            else:
                spr_loss = torch.zeros_like(dqn_loss)

        with rf("loss/total"):
            total_loss_per_sample = dqn_loss + float(self.config.spr_weight) * spr_loss
            loss = (total_loss_per_sample * loss_weights).mean()

        with rf("bwd"):
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

        with rf("optim/step"):
            nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
            self.optimizer.step()

        with rf("target/update"):
            if self.grad_steps % int(self.config.target_update_period) == 0:
                self._soft_update_target(float(tau))

        avg_q = online_out["q_values"].mean()

        return {
            "loss": loss.detach(),
            "dqn_loss": dqn_loss.detach(),
            "spr_loss": spr_loss.detach(),
            "avg_q": avg_q.detach(),
        }

    def _training_step_update(self, offline: bool = False):
        del offline
        rf = record_function if self.profiler_active else (lambda _: nullcontext())
        curr_n = int(self.update_horizon_scheduler(self.cycle_grad_steps))
        curr_gamma = float(self.gamma_scheduler(self.cycle_grad_steps))
        tau = float(self.target_update_tau_scheduler(self.cycle_grad_steps))

        with rf("rb/sample"):
            replay_batch = self.replay_buffer.sample_transition_batch(
                batch_size=int(self.config.batch_size) * self._batches_to_group,
                update_horizon=curr_n,
                gamma=curr_gamma,
                subseq_len=self.seq_len,
            )

        indices = replay_batch["indices"]
        sampling_probabilities = replay_batch["sampling_probabilities"]

        if self.config.use_per:
            probs = np.asarray(sampling_probabilities, dtype=np.float32)
            loss_weights_np = 1.0 / np.sqrt(probs + 1e-10)
            loss_weights_np /= (np.max(loss_weights_np) + 1e-10)
        else:
            loss_weights_np = np.ones((indices.shape[0],), dtype=np.float32)

        with rf("batch/tensorize"):
            torch_batch = {
                k: torch.as_tensor(v, device=self.device)
                for k, v in replay_batch.items()
                if k != "sampling_probabilities"
            }
            torch_batch["loss_weights"] = torch.as_tensor(loss_weights_np, device=self.device)

        all_dqn_losses = []
        all_losses = []
        all_spr = []
        all_avg_q = []

        bsz = int(self.config.batch_size)
        for i in range(self._batches_to_group):
            sl = slice(i * bsz, (i + 1) * bsz)
            chunk = {
                k: (v[sl] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == bsz * self._batches_to_group else v)
                for k, v in torch_batch.items()
            }
            with rf("batch/chunk"):
                out = self._train_batch_chunk(chunk, tau=tau)

            all_dqn_losses.append(out["dqn_loss"].detach().cpu().numpy())
            all_losses.append(float(out["loss"].item()))
            all_spr.append(float(out["spr_loss"].mean().item()))
            all_avg_q.append(float(out["avg_q"].item()))

            self.grad_steps += 1
            self.grad_step = self.grad_steps
            self.cycle_grad_steps += 1
            if self.profiler_active:
                # Keep profiler steps aligned with optimizer updates for detailed traces.
                self._step_profiler()

        if self.config.use_per:
            with rf("prio/update"):
                priorities = np.sqrt(np.concatenate(all_dqn_losses, axis=0) + 1e-10)
                self.replay_buffer.set_priority(indices.astype(np.int32), priorities.astype(np.float32))

        return {
            "loss": float(np.mean(all_losses)),
            "spr_loss": float(np.mean(all_spr)),
            "avg_q": float(np.mean(all_avg_q)),
            "gamma": float(curr_gamma),
            "n_step": int(curr_n),
        }

    def step(
        self,
        next_observation,
        reward: float,
        terminated: bool,
        truncated: bool = False,
        info: Optional[Dict] = None,
    ):
        del info
        if self.last_obs is None or self.last_action is None:
            return None

        episode_end = bool(terminated or truncated)
        terminal = bool(terminated)

        self.replay_buffer.add(
            self.last_obs,
            self.last_action,
            float(reward),
            terminal,
            episode_end=episode_end,
            priority=None,
        )

        train_stats = None
        if (
            self.replay_buffer.add_count > int(self.config.learning_starts)
            and self.training_steps >= self.learning_ready_step
            and self.training_steps % self.update_period == 0
        ):
            for _ in range(self._num_updates_per_train_step):
                train_stats = self._training_step_update(offline=False)

        allow_resets = not (self.config.continual and not self.config.allow_resets_in_continual)
        if allow_resets and int(self.config.reset_every) > 0 and self.training_steps > self.next_reset:
            self.reset_weights(force=False)

        self.training_steps += 1
        self.global_step = self.training_steps

        if episode_end:
            self.last_obs = None
            self.last_action = None
            self._act_state = None
        else:
            self.last_obs = self._extract_frame(next_observation)

        return train_stats

    def save_model(self, path: str):
        torch.save(self.q_network.state_dict(), path)
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_policy(
    agent: Agent,
    policy_network: BBFNetwork,
    config: AgentConfig,
    *,
    episodes: int,
    epsilon: float,
    seed: int,
    tag: str,
):
    """Run no-learning evaluation and return summary metrics."""
    if episodes <= 0:
        return None

    rng = random.Random(seed)
    env = make_eval_env(
        config.env_id,
        seed=seed,
        full_action_space=config.full_action_space,
        sticky_prob=config.eval_sticky_prob,
        clip_rewards=config.eval_clip_rewards,
    )

    was_training = policy_network.training
    policy_network.eval()
    returns: List[float] = []
    lengths: List[int] = []
    obs, _ = env.reset(seed=seed)
    eval_state = np.zeros((FRAME_SKIP, *agent.frame_shape), dtype=np.uint8)
    first_frame = agent._extract_frame(obs)
    eval_state = np.roll(eval_state, -1, axis=0)
    eval_state[-1] = first_frame
    ep_return = 0.0
    ep_length = 0

    print(
        f"\n[EVAL:{tag}] Starting evaluation: episodes={episodes}, "
        f"epsilon={epsilon:.4f}, sticky_prob={config.eval_sticky_prob:.3f}, "
        f"clip_rewards={config.eval_clip_rewards}"
    )
    while len(returns) < episodes:
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_t = torch.as_tensor(eval_state, device=agent.device, dtype=torch.uint8).unsqueeze(0)
            q_values = policy_network.compute_outputs(obs_t)["q_values"]
            action = int(torch.argmax(q_values, dim=1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)
        ep_length += 1

        if terminated or truncated:
            returns.append(ep_return)
            lengths.append(ep_length)
            print(
                f"[EVAL:{tag}] Episode {len(returns)}/{episodes} "
                f"Return={ep_return:.0f} Length={ep_length}"
            )
            obs, _ = env.reset()
            eval_state.fill(0)
            frame = agent._extract_frame(obs)
            eval_state = np.roll(eval_state, -1, axis=0)
            eval_state[-1] = frame
            ep_return = 0.0
            ep_length = 0
        else:
            frame = agent._extract_frame(obs)
            eval_state = np.roll(eval_state, -1, axis=0)
            eval_state[-1] = frame

    env.close()
    if was_training:
        policy_network.train()

    returns_np = np.asarray(returns, dtype=np.float32)
    lengths_np = np.asarray(lengths, dtype=np.float32)
    summary = {
        "episodes": int(episodes),
        "mean_return": float(np.mean(returns_np)),
        "median_return": float(np.median(returns_np)),
        "max_return": float(np.max(returns_np)),
        "min_return": float(np.min(returns_np)),
        "mean_length": float(np.mean(lengths_np)),
    }
    print(
        f"[EVAL:{tag}] Summary: mean={summary['mean_return']:.2f}, "
        f"median={summary['median_return']:.2f}, min={summary['min_return']:.2f}, "
        f"max={summary['max_return']:.2f}, mean_len={summary['mean_length']:.1f}"
    )
    return summary


# =============================================================================
# FRAME RUNNER FOR BENCHMARK
# =============================================================================

def agent_bbf_frame_runner(
    agent: Agent,
    handle: EnvironmentHandle,
    *,
    context: FrameRunnerContext,
    writer: Optional[SummaryWriter] = None,
) -> FrameRunnerResult:
    
    config = agent.config
    frames_per_step = handle.frames_per_step

    # Initialize profiler on first game if enabled (continual mode)
    if config.torch_profile and agent.profiler is None:
        profiler_dir = os.path.join(os.path.dirname(config.log_file), "profiler")
        os.makedirs(profiler_dir, exist_ok=True)
        agent.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=max(1, config.learning_starts // 10),
                warmup=5,
                active=config.torch_profile_steps,
                repeat=1
            ),
            on_trace_ready=tensorboard_trace_handler(profiler_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        agent.profiler.start()
        agent.profiler_active = True
        print(f"*** Torch Profiler ENABLED - will profile {config.torch_profile_steps} steps ***")
        print(f"*** Profiler output: {profiler_dir} ***")

    # Signal new game (optional for continual learning)
    if not (config.continual and not config.continual_reset_on_game_switch):
        agent.start_new_game()

    logger = create_logger(log_file=getattr(config, "log_file", None))
    log = logger.log

    game_name = getattr(handle.spec, "name", context.name)
    env_id = handle.spec.env_id if hasattr(handle.spec, 'env_id') else game_name
    max_step_budget = context.frame_budget // frames_per_step

    log(f"Starting BBF on '{game_name}' (env_id={env_id}) for {max_step_budget} steps.")

    # Use the environment created by BenchmarkRunner (already has correct wrappers)
    if handle.gym is None:
        raise ValueError(f"No gym environment in handle for game '{game_name}'. Ensure backend='gym' in GameSpec.")
    env = handle.gym
    reset_seed = handle.spec.seed if handle.spec.seed is not None else config.seed
    obs, info = env.reset(seed=reset_seed)
    
    episode_scores: List[float] = []
    episode_end: List[int] = []
    
    params = list(agent.q_network.parameters())
    episode_graph = torch.full((PROGRESS_POINTS,), -999.0, dtype=torch.float32)
    parms_graph = torch.zeros((PROGRESS_POINTS, len(params)), dtype=torch.float32)

    frames_consumed = 0
    frames_since_reward = 0
    running_episode_score = 0.0
    start_time = time.time()
    training_start_time = None
    training_start_step = None
    env_start_time = time.time()
    env_start_step = agent.global_step
    last_model_save = context.last_model_save
    
    # Best model tracking for this game
    best_return = float('-inf')
    best_model_path = f"best_model_{game_name.replace('/', '_').replace(' ', '_')}.pt"

    delay_frames = max(0, int(context.delay_frames))
    action_delay = deque([0] * delay_frames) if delay_frames > 0 else None

    for _ in range(max_step_budget):
        action = agent.act(obs)
        if action_delay is not None:
            action_delay.append(action)
            action_to_env = action_delay.popleft()
        else:
            action_to_env = action
        next_obs, reward, terminated, truncated, info = env.step(action_to_env)

        running_episode_score += reward
        if reward != 0:
            frames_since_reward = 0
        else:
            frames_since_reward += frames_per_step

        # Check for reward timeout (treat as truncation, not termination)
        reward_timed_out = (context.max_frames_without_reward > 0 and frames_since_reward >= context.max_frames_without_reward)
        if reward_timed_out:
            truncated = True

        # Check if this step will exhaust the frame budget (treat as truncation)
        budget_exhausted = (frames_consumed + frames_per_step) >= context.frame_budget
        if budget_exhausted and not terminated:
            truncated = True

        train_log = agent.step(next_obs, reward, terminated, truncated, info)
        
        frames_consumed += frames_per_step
        agent.game_frame_counters[game_name] = agent.game_frame_counters.get(game_name, 0) + frames_per_step

        if train_log:
            # Track when training actually starts (after warmup)
            if training_start_time is None:
                training_start_time = time.time()
                training_start_step = agent.global_step
            
            sps = None
            if agent.grad_step % 100 == 0:
                # SPS based on training time only (excludes warmup)
                training_elapsed = time.time() - training_start_time
                training_steps = agent.global_step - training_start_step
                sps = int(training_steps / (training_elapsed + 1e-5))
                ep_count = len(episode_scores)
                log(f"step={agent.global_step} loss={train_log['loss']:.4f} spr={train_log['spr_loss']:.4f} avg_q={train_log['avg_q']:.2f} gamma={train_log['gamma']:.4f} sps={sps} eps={ep_count}")
            if writer and sps is not None:
                writer.add_scalar("losses/total", train_log['loss'], agent.global_step)
                writer.add_scalar("losses/spr_loss", train_log['spr_loss'], agent.global_step)
                writer.add_scalar("charts/avg_q", train_log['avg_q'], agent.global_step)
                writer.add_scalar("charts/gamma", train_log['gamma'], agent.global_step)
                writer.add_scalar("charts/n_step", train_log['n_step'], agent.global_step)
                writer.add_scalar("charts/SPS", sps, agent.global_step)
        elif agent.global_step % 100 == 0:
            # Env-only SPS before training starts (e.g., replay_ratio=0)
            env_elapsed = time.time() - env_start_time
            env_steps = agent.global_step - env_start_step
            sps = int(env_steps / (env_elapsed + 1e-5))
            log(f"step={agent.global_step} sps={sps} (env-only)")
            if writer:
                writer.add_scalar("charts/SPS", sps, agent.global_step)

        if info and "episode" in info:
            episode = info["episode"]
            ep_return = float(episode["r"])
            ep_length = int(episode["l"])
            episode_scores.append(ep_return)
            episode_end.append(context.frame_offset + frames_consumed)
            log(f"--> Episode Done. Step={agent.global_step}, Return={ep_return:.0f}, Length={ep_length}")
            if writer:
                writer.add_scalar("charts/episodic_return", ep_return, agent.global_step)
                writer.add_scalar("charts/episodic_length", ep_length, agent.global_step)
                writer.add_scalar(f"charts/episodic_return/{game_name}", ep_return, agent.global_step)

            # Save best model
            if ep_return > best_return:
                best_return = ep_return
                agent.save_model(best_model_path)
                log(f"    ★ New best! Saved model with return {best_return:.0f}")
                if writer:
                    writer.add_scalar("charts/best_return", best_return, agent.global_step)
                    writer.add_scalar(f"charts/best_return/{game_name}", best_return, agent.global_step)

        # Reset environment if episode ended (either terminated or truncated)
        episode_ended = terminated or truncated
        if episode_ended:
            obs, info = env.reset()
            running_episode_score = 0.0
            frames_since_reward = 0
            if action_delay is not None:
                action_delay = deque([0] * delay_frames)
        else:
            obs = next_obs
            
        update_progress_graphs(
            episode_graph, parms_graph, params, 
            context.frame_offset, frames_consumed, context.frame_budget, 
            context.average_frames, episode_scores, episode_end, context.graph_total_frames
        )

        if frames_consumed >= context.frame_budget:
            if not terminated:
                log(f"Game '{game_name}' truncated at frame budget ({frames_consumed} frames, {len(episode_scores)} episodes)")
            break

    # Note: env cleanup is handled by BenchmarkRunner in its finally block

    return FrameRunnerResult(last_model_save, episode_scores, episode_end, episode_graph, parms_graph)


# =============================================================================
# MAIN & CLI
# =============================================================================

def _parse_continual_game_ids(config: AgentConfig) -> List[str]:
    if config.continual_games.strip():
        game_ids = [game.strip() for game in config.continual_games.split(",") if game.strip()]
    else:
        game_ids = list(DEFAULT_CONTINUAL_GAMES)
    
    if len(game_ids) < 2:
        raise ValueError(f"Continual mode requires at least 2 games, received {len(game_ids)}.")
    
    return game_ids


def _build_continual_benchmark_config(game_ids: Sequence[str], frame_budget: int, config: AgentConfig) -> BenchmarkConfig:
    cycles = []
    for cycle_index in range(config.continual_cycles):
        cycle_games = []
        for env_id in game_ids:
            cycle_games.append(
                GameSpec(
                    name=env_id,
                    frame_budget=frame_budget,
                    sticky_prob=config.continual_sticky_prob,
                    delay_frames=config.continual_delay_frames,
                    seed=config.seed + cycle_index,
                    backend='gym',
                    env_id=env_id,
                    params={
                        'noop_max': 30,
                        'frame_skip': FRAME_SKIP,
                        'resize_shape': (84, 84),
                        'grayscale': True,
                        'use_canonical_full_actions': True,
                    },
                )
            )
        cycles.append(CycleConfig(cycle_index=cycle_index, games=cycle_games))
    return BenchmarkConfig(cycles=cycles, description="BBF Continual")


def parse_args() -> AgentConfig:
    parser = argparse.ArgumentParser(description="BBF Agent for Continual Learning Benchmark")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__)[:-3])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project-name", type=str, default="physical-atari")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-profile-steps", type=int, default=50)

    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--total-steps", "--total-timesteps", dest="total_steps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument(
        "--full-action-space",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the full 18-action Atari set (single-game mode).",
    )
    parser.add_argument(
        "--continual-reset-on-game-switch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reset BBF state when switching games in continual mode (oracle baseline).",
    )
    parser.add_argument(
        "--allow-resets-in-continual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow BBF periodic hard resets during continual runs.",
    )
    parser.add_argument(
        "--continual-sticky-prob",
        type=float,
        default=0.25,
        help="Sticky action probability for continual benchmark games.",
    )
    parser.add_argument(
        "--continual-delay-frames",
        type=int,
        default=0,
        help="Action delay (frames) for continual benchmark games.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--encoder-learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--impala-width", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-ratio", type=int, default=64, help="BBF replay ratio.")
    parser.add_argument("--batches-to-group", type=int, default=2, help="Grouped batches per replay update.")
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--min-gamma", type=float, default=0.97)
    parser.add_argument("--update-horizon", type=int, default=3)
    parser.add_argument("--max-update-horizon", type=int, default=10)
    parser.add_argument("--cycle-steps", type=int, default=10_000)
    parser.add_argument("--initial-gamma", type=float, default=0.97)
    parser.add_argument("--final-gamma", type=float, default=0.997)
    parser.add_argument("--initial-n-step", type=int, default=10)
    parser.add_argument("--final-n-step", type=int, default=3)
    parser.add_argument("--anneal-duration", type=int, default=10_000)
    parser.add_argument("--reset-interval", type=int, default=20_000, help="Environment steps between periodic resets.")
    parser.add_argument("--no-resets-after", type=int, default=100_000)
    parser.add_argument("--shrink-factor", type=float, default=0.5)
    parser.add_argument("--perturb-factor", type=float, default=0.5)
    parser.add_argument("--shrink-perturb-keys", type=str, default="encoder,transition_model")
    parser.add_argument("--offline-update-frac", type=float, default=0.0)
    parser.add_argument("--reset-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-priorities", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--reset-disable-last-steps",
        type=int,
        default=0,
        help="Disable periodic hard resets in the final N env steps to avoid end-of-run collapse.",
    )
    parser.add_argument("--spr-weight", type=float, default=5.0)
    parser.add_argument("--jumps", type=int, default=5, help="Number of SPR prediction jumps")
    parser.add_argument("--n-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-update-period", type=int, default=1)
    parser.add_argument("--target-action-selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-e", type=float, default=0.0)
    parser.add_argument("--end-e", type=float, default=0.0)
    parser.add_argument("--exploration-fraction", type=float, default=0.0)
    parser.add_argument("--epsilon-train", type=float, default=0.0)
    parser.add_argument("--epsilon-decay-period", type=int, default=2001)
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--use-per", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-alpha", type=float, default=0.5)
    parser.add_argument("--per-beta", type=float, default=0.5)
    parser.add_argument("--per-eps", type=float, default=1e-6)
    parser.add_argument("--data-augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--channels-last-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile-mode", type=str, default="reduce-overhead")

    # Continual benchmark
    parser.add_argument("--continual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continual-games", type=str, default="")
    parser.add_argument("--continual-cycles", type=int, default=DEFAULT_CONTINUAL_CYCLES)
    parser.add_argument("--continual-cycle-frames", type=int, default=DEFAULT_CONTINUAL_CYCLE_FRAMES)
    parser.add_argument("--per-game-learning-starts", type=int, default=2000)

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of post-training evaluation episodes (single-game mode only). Set 0 to disable.",
    )
    parser.add_argument("--eval-epsilon", type=float, default=0.001, help="Epsilon for post-training evaluation.")
    parser.add_argument(
        "--eval-sticky-prob",
        type=float,
        default=0.0,
        help="Sticky action probability used in post-training evaluation.",
    )
    parser.add_argument(
        "--eval-clip-rewards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to clip rewards during post-training evaluation.",
    )
    parser.add_argument(
        "--eval-select-interval",
        type=int,
        default=25_000,
        help="Run eval-based checkpoint selection every N env steps in single-game mode. Set 0 to disable.",
    )
    parser.add_argument(
        "--eval-select-start",
        type=int,
        default=25_000,
        help="First env step at which eval-based checkpoint selection can run.",
    )
    parser.add_argument(
        "--eval-select-episodes",
        type=int,
        default=2,
        help="Episodes per eval-selection pass.",
    )
    
    args = parser.parse_args()
    return AgentConfig(**vars(args))


def main():
    config = parse_args()

    if config.num_envs != 1:
        raise ValueError("agent_bbf.py currently supports num_envs=1 (to match BBF sequence handling).")

    log_root = './results_final_bbf'
    os.makedirs(log_root, exist_ok=True)
    env_suffix = config.env_id.split("/")[-1] if "/" in config.env_id else config.env_id
    run_name = f"{env_suffix}__{config.exp_name}__{config.seed}__{int(time.time())}"
    
    if config.track:
        import wandb
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            config=vars(config),
            name=run_name,
            monitor_gym=True,
            save_code=True
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
    torch.backends.cudnn.deterministic = config.torch_deterministic
    torch.backends.cudnn.benchmark = not config.torch_deterministic
    if config.cuda and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    # Log hyperparameters to TensorBoard
    writer.add_text("hyperparameters", 
        "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(config).items()]))

    # Print run configuration
    print(f"\n{'='*60}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Environment:    {config.env_id}")
    print(f"  Replay Ratio:   {config.replay_ratio}")
    print(f"  Reset Interval: {config.reset_interval} (env steps)")
    print(f"  No Resets After:{config.no_resets_after} (env steps)")
    print(f"  Total Steps:    {config.total_steps}")
    print(f"  Seed:           {config.seed}")
    print(f"  Learning Rate:  {config.learning_rate}")
    print(f"  Full Actions:   {config.full_action_space}")
    print(f"  Continual:      {config.continual}")
    if config.continual:
        print(f"  Games:          {config.continual_games}")
        print(f"  Cycles:         {config.continual_cycles}")
        print(f"  Cycle Frames:   {config.continual_cycle_frames}")
        print(f"  Sticky Prob:    {config.continual_sticky_prob}")
        print(f"  Delay Frames:   {config.continual_delay_frames}")
        print(f"  Reset on Switch:{config.continual_reset_on_game_switch}")
        print(f"  Allow Resets:   {config.allow_resets_in_continual}")
    else:
        print(f"  Eval Episodes:  {config.eval_episodes}")
        print(f"  Eval Epsilon:   {config.eval_epsilon}")
        print(f"  Eval Sticky:    {config.eval_sticky_prob}")
        print(f"  Eval Sel Every: {config.eval_select_interval}")
        print(f"  Eval Sel Start: {config.eval_select_start}")
        print(f"  Eval Sel Eps:   {config.eval_select_episodes}")
    print(f"{'='*60}\n")

    if config.continual:
        if make_atari_env is None:
            raise RuntimeError(
                "Continual mode requires the original 'common' package helpers "
                "(make_atari_env, update_progress_graphs, write_continual_summary, create_logger) "
                "on this branch. Single-game mode does not require them."
            )
        # Create dummy env to get observation/action space for continual mode
        dummy_env = make_atari_env(config.env_id, config.seed)
        agent = Agent(dummy_env.observation_space, dummy_env.action_space, config)
        dummy_env.close()
        game_ids = _parse_continual_game_ids(config)
        
        # Adjust total_steps to match benchmark
        frames_needed = config.continual_cycle_frames * len(game_ids) * config.continual_cycles
        required_steps = math.ceil(frames_needed / FRAME_SKIP)
        if config.total_steps != required_steps:
            print(f"Adjusting total_steps from {config.total_steps} to {required_steps}")
            config.total_steps = required_steps
        
        bench_config = _build_continual_benchmark_config(game_ids, config.continual_cycle_frames, config)
        
        def runner_wrapper(agent_inst, handle, *, context):
            return agent_bbf_frame_runner(agent_inst, handle, context=context, writer=writer)

        runner = BenchmarkRunner(
            agent,
            bench_config,
            frame_runner=runner_wrapper,
            data_dir=run_dir,
            rank=0,
            default_seed=config.seed,
            use_canonical_full_actions=True,
            average_frames=100_000
        )
        results = runner.run()
        
        # Write continual summary
        summary_path = os.path.join(run_dir, "continual_summary.json")
        write_continual_summary(results, summary_path)
        if agent.profiler is not None:
            try:
                agent.profiler.stop()
            except Exception as exc:
                print(f"*** Torch Profiler stop failed: {exc} ***")
            agent.profiler_active = False
            agent.profiler = None
            print(f"*** View results: tensorboard --logdir={run_dir}/profiler ***")
    else:
        # Single game mode - CleanRL-style training loop with SyncVectorEnv
        num_envs = config.num_envs  # BBF uses single env for sequence correctness
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    config.env_id,
                    config.seed + i,
                    i,
                    config.capture_video,
                    run_name,
                    full_action_space=config.full_action_space,
                )
                for i in range(num_envs)
            ]
        )

        # Reinitialize agent with correct observation space from vectorized env
        agent = Agent(envs.single_observation_space, envs.single_action_space, config)

        if config.torch_profile and agent.profiler is None:
            profiler_dir = os.path.join(run_dir, "profiler")
            os.makedirs(profiler_dir, exist_ok=True)
            agent.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=max(1, config.learning_starts // 10),
                    warmup=5,
                    active=config.torch_profile_steps,
                    repeat=1
                ),
                on_trace_ready=tensorboard_trace_handler(profiler_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            agent.profiler.start()
            agent.profiler_active = True
            print(f"*** Torch Profiler ENABLED - will profile {config.torch_profile_steps} steps ***")
            print(f"*** Profiler output: {profiler_dir} ***")
        
        obs, _ = envs.reset(seed=config.seed)
        start_time = time.time()
        training_start_time = None
        training_start_step = None
        env_start_time = time.time()
        env_start_step = 0
        all_returns = []
        
        print("Starting Training Loop...")

        # Checkpoint tracking:
        # - train_best_return: best training episodic-life return (diagnostic only)
        # - best_eval_score: best checkpoint score on the eval protocol (used for model selection)
        train_best_return = float('-inf')
        train_best_step = -1
        best_eval_score = float("-inf")
        best_eval_step = -1
        best_model_path = os.path.join(run_dir, "best_model.pt")
        next_eval_select_step = None
        if config.eval_select_interval > 0 and config.eval_select_episodes > 0:
            next_eval_select_step = config.eval_select_start if config.eval_select_start > 0 else config.eval_select_interval

        def run_eval_checkpoint_selection(current_step: int, tag_suffix: str):
            nonlocal best_eval_score, best_eval_step
            if config.eval_select_episodes <= 0:
                return
            eval_summary = evaluate_policy(
                agent,
                agent.q_network,
                config,
                episodes=config.eval_select_episodes,
                epsilon=config.eval_epsilon,
                seed=config.seed + 30_000 + current_step,
                tag=f"select_{tag_suffix}",
            )
            if eval_summary is None:
                return
            score = eval_summary["mean_return"]
            writer.add_scalar("eval/select_mean_return", score, current_step)
            if score > best_eval_score:
                best_eval_score = score
                best_eval_step = current_step
                agent.save_model(best_model_path)
                print(
                    f"    ★ New eval-best checkpoint at step {current_step}: "
                    f"mean return {best_eval_score:.2f}"
                )
                writer.add_scalar("eval/select_best_mean_return", best_eval_score, current_step)

        for global_step in range(config.total_steps):
            if global_step <= config.learning_starts and global_step % 100 == 0:
                print(f"Warmup: Filling Replay Buffer {global_step}/{config.learning_starts}")

            action = agent.act(obs[0])
            next_obs, rewards, terminations, truncations, infos = envs.step(np.array([action], dtype=np.int64))

            train_stats = agent.step(
                next_obs[0],
                float(rewards[0]),
                bool(terminations[0]),
                bool(truncations[0]),
                infos,
            )
            obs = next_obs

            episode_infos: List[Dict[str, object]] = []
            final_infos = infos.get("final_info")
            if final_infos is not None:
                final_info_mask = infos.get("_final_info")
                if final_info_mask is None:
                    for info in final_infos:
                        if info:
                            episode_infos.append(info)
                else:
                    for has_info, info in zip(final_info_mask, final_infos):
                        if has_info and info:
                            episode_infos.append(info)
            elif "episode" in infos:
                # Fallback for vector wrappers that surface `episode` directly with a mask.
                episode_data = infos["episode"]
                episode_mask = infos.get("_episode")
                if isinstance(episode_data, dict) and episode_mask is not None:
                    for env_idx, has_episode in enumerate(episode_mask):
                        if has_episode:
                            episode_infos.append(
                                {
                                    "episode": {
                                        key: value[env_idx]
                                        for key, value in episode_data.items()
                                    }
                                }
                            )

            for info in episode_infos:
                if "episode" not in info:
                    continue
                episode = info["episode"]
                ep_return = float(episode["r"])
                ep_length = int(episode["l"])
                all_returns.append(ep_return)
                print(f"--> Episode Done. Step={global_step}, Return={ep_return:.0f}, Length={ep_length}")
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episodic_length", ep_length, global_step)
                if ep_return > train_best_return:
                    train_best_return = ep_return
                    train_best_step = global_step
                    writer.add_scalar("charts/train_best_return", train_best_return, global_step)

            if train_stats:
                if training_start_time is None:
                    training_start_time = time.time()
                    training_start_step = global_step
                    print(f"*** Training started at step {global_step} ***")

                if agent.grad_step % 100 == 0:
                    training_elapsed = time.time() - training_start_time
                    training_steps = global_step - training_start_step
                    sps = int(training_steps / (training_elapsed + 1e-5))
                    remaining_steps = config.total_steps - global_step
                    eta_min = (remaining_steps / (sps + 1e-5)) / 60
                    epsilon = agent._current_epsilon(eval_mode=False)

                    print(
                        f"Step: {global_step} | Grad: {agent.grad_step} | "
                        f"Loss: {train_stats['loss']:.3f} | SPR: {train_stats['spr_loss']:.3f} | "
                        f"AvgQ: {train_stats['avg_q']:.2f} | Eps: {epsilon:.3f} | "
                        f"SPS: {sps} | ETA: {eta_min:.1f} min"
                    )
                    writer.add_scalar("losses/total_loss", train_stats["loss"], agent.grad_step)
                    writer.add_scalar("losses/spr_loss", train_stats["spr_loss"], agent.grad_step)
                    writer.add_scalar("charts/avg_q", train_stats["avg_q"], agent.grad_step)
                    writer.add_scalar("charts/n_step", train_stats["n_step"], agent.grad_step)
                    writer.add_scalar("charts/gamma", train_stats["gamma"], agent.grad_step)
                    writer.add_scalar("charts/SPS", sps, agent.grad_step)
            elif global_step % 100 == 0:
                env_elapsed = time.time() - env_start_time
                env_steps = global_step - env_start_step
                sps = int(env_steps / (env_elapsed + 1e-5))
                print(f"Step: {global_step} | SPS: {sps} (env-only)")
                writer.add_scalar("charts/SPS", sps, global_step)

            if (
                next_eval_select_step is not None
                and global_step >= next_eval_select_step
                and global_step >= config.eval_select_start
            ):
                run_eval_checkpoint_selection(global_step, f"step{global_step}")
                next_eval_select_step += config.eval_select_interval
        
        envs.close()

        # Ensure we always have at least one eval-based checkpoint candidate.
        if config.eval_select_episodes > 0 and best_eval_score == float("-inf"):
            run_eval_checkpoint_selection(config.total_steps, "final")

        # Stop and flush profiler output.
        if agent.profiler is not None:
            try:
                agent.profiler.stop()
            except Exception as exc:
                print(f"*** Torch Profiler stop failed: {exc} ***")
            agent.profiler_active = False
            agent.profiler = None
            print(f"*** Torch Profiler completed at end of training ***")
            print(f"*** View results: tensorboard --logdir={run_dir}/profiler ***")

        # Final summary (CleanRL style)
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE - FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"  Run Name:       {run_name}")
        print(f"  Replay Ratio:   {config.replay_ratio}")
        print(f"  Reset Interval: {config.reset_interval} (env steps)")
        print(f"  Seed:           {config.seed}")
        print(f"  Environment:    {config.env_id}")
        print(f"  Total Time:     {total_time/60:.1f} minutes")
        print(f"  Grad Steps:     {agent.grad_step}")
        if all_returns:
            print(f"  Episodes:       {len(all_returns)}")
            print(f"  Max Return:     {max(all_returns):.0f}")
            print(f"  Mean Return:    {np.mean(all_returns):.1f}")
            print(f"  Last 5:         {[int(r) for r in all_returns[-5:]]}")
        if train_best_return > float("-inf"):
            print(f"  Train Best:     {train_best_return:.0f} @ step {train_best_step}")
        if best_eval_score > float("-inf"):
            print(f"  Eval Best:      {best_eval_score:.2f} @ step {best_eval_step}")
            print(f"  Best Model:     {best_model_path}")
        print(f"{'='*60}\n")
        
        # Log final metrics
        if all_returns:
            writer.add_scalar("final/max_return", max(all_returns), 0)
            writer.add_scalar("final/mean_return", np.mean(all_returns), 0)
            writer.add_scalar("final/episodes", len(all_returns), 0)
            writer.add_scalar("final/total_time_minutes", total_time/60, 0)

        # Separate no-learning evaluation to avoid conflating training noise/resets with final performance.
        if config.eval_episodes > 0:
            final_eval = evaluate_policy(
                agent,
                agent.q_network,
                config,
                episodes=config.eval_episodes,
                epsilon=config.eval_epsilon,
                seed=config.seed + 10_000,
                tag="final",
            )
            if final_eval is not None:
                writer.add_scalar("eval/final_mean_return", final_eval["mean_return"], config.total_steps)
                writer.add_scalar("eval/final_median_return", final_eval["median_return"], config.total_steps)
                writer.add_scalar("eval/final_max_return", final_eval["max_return"], config.total_steps)

            if os.path.exists(best_model_path):
                best_eval_net = BBFNetwork(agent.in_channels, agent.num_actions, config).to(agent.device)
                best_eval_net.load_state_dict(torch.load(best_model_path, map_location=agent.device))
                best_eval = evaluate_policy(
                    agent,
                    best_eval_net,
                    config,
                    episodes=config.eval_episodes,
                    epsilon=config.eval_epsilon,
                    seed=config.seed + 20_000,
                    tag="best",
                )
                if best_eval is not None:
                    writer.add_scalar("eval/best_mean_return", best_eval["mean_return"], config.total_steps)
                    writer.add_scalar("eval/best_median_return", best_eval["median_return"], config.total_steps)
                    writer.add_scalar("eval/best_max_return", best_eval["max_return"], config.total_steps)

    if config.save_model:
        final_model_path = os.path.join(run_dir, f"{config.exp_name}_final.pt")
        agent.save_model(final_model_path)
        print(f"Saved final model to {final_model_path}")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
