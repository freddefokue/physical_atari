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
from torch.amp import autocast, GradScaler
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

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

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


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
            
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
            
        # Official BBF uses reward clipping (standard for Atari 100k)
        env = ClipRewardEnv(env) 
        
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        # Using FrameStackObservation (Gymnasium 1.0+) - Outputs (4, 84, 84)
        env = gym.wrappers.FrameStackObservation(env, 4)
        
        env.action_space.seed(seed)
        return env
    return thunk


# =============================================================================
# NETWORK ARCHITECTURE (from CleanRL BBF)
# =============================================================================

class RandomShift(nn.Module):
    """Vectorized random shift augmentation using grid_sample (no Python loops)."""
    def __init__(self, pad=4, intensity_scale=0.05):
        super().__init__()
        self.pad = pad
        self.intensity_scale = intensity_scale

    def forward(self, x):
        # Input: (B, C, H, W) - uint8 or float
        b, c, h, w = x.shape 
        x = x.float() / 255.0  # Normalize to [0, 1] for grid_sample
        
        # Random shifts in pixels, converted to normalized coordinates [-1, 1]
        shift_y = (torch.randint(0, 2 * self.pad + 1, (b,), device=x.device) - self.pad).float()
        shift_x = (torch.randint(0, 2 * self.pad + 1, (b,), device=x.device) - self.pad).float()
        
        # Convert pixel shifts to normalized grid coordinates
        shift_y_norm = shift_y / h * 2
        shift_x_norm = shift_x / w * 2
        
        # Build affine transformation matrix for translation only
        theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = 1.0  # scale x
        theta[:, 1, 1] = 1.0  # scale y
        theta[:, 0, 2] = shift_x_norm  # translate x
        theta[:, 1, 2] = shift_y_norm  # translate y
        
        # Generate sampling grid and apply transformation
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        out = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        # Intensity augmentation (from official BBF/DrQ)
        if self.training:
            noise = torch.randn(b, 1, 1, 1, device=x.device).clamp(-2.0, 2.0)
            out = out * (1.0 + self.intensity_scale * noise)
        
        # Scale back to [0, 255] range
        return out * 255.0


class ResidualBlock(nn.Module):
    """Pre-activation residual block (official BBF style)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # Pre-activation: ReLU before conv (better gradient flow)
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x  # No ReLU after add


class ImpalaBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Official BBF uses stride=1 conv followed by MaxPool (not strided conv)
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_c)
        self.res2 = ResidualBlock(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class BBFNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, args):
        super().__init__()
        self.n_atoms = args.n_atoms
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.delta_z = (args.v_max - args.v_min) / (args.n_atoms - 1)
        self.n_actions = num_actions
        self.register_buffer("support", torch.linspace(args.v_min, args.v_max, args.n_atoms))
        
        w = args.impala_width
        self.encoder = nn.Sequential(
            ImpalaBlock(in_channels, 16 * w),
            ImpalaBlock(16 * w, 32 * w),
            ImpalaBlock(32 * w, 32 * w),
            nn.ReLU()
        )
        self.enc_out_dim = 32 * w * 11 * 11
        
        self.latent_dim = 512
        self.rep_head = nn.Linear(self.enc_out_dim, self.latent_dim)
        
        self.value_head = nn.Sequential(
            nn.Linear(self.latent_dim, 512), nn.ReLU(), nn.Linear(512, self.n_atoms)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(self.latent_dim, 512), nn.ReLU(), nn.Linear(512, self.n_atoms * self.n_actions)
        )
        
        self.proj_dim = 256
        self.projection = nn.Sequential(
            nn.Linear(self.latent_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )
        self.action_emb = nn.Embedding(self.n_actions, self.latent_dim)
        self.trans_fc = nn.Linear(self.latent_dim * 2, self.latent_dim)

    def encode(self, x):
        x = x / 255.0
        h = self.encoder(x)
        # Latent renormalization (MuZero-style, from official BBF)
        h = self._renormalize(h)
        h = h.view(h.size(0), -1)
        return self.rep_head(h)
    
    def _renormalize(self, x):
        """Min-max normalize spatial latent to [0, 1] (official BBF)."""
        shape = x.shape
        x = x.view(x.size(0), -1)
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-5)
        return x.view(shape)

    def forward(self, x):
        z = self.encode(x)
        return self.get_dist(z)

    def get_dist(self, z):
        value = self.value_head(z).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(z).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)
        return q_dist

    def transition(self, z, action):
        a_emb = self.action_emb(action)
        x = torch.cat([z, a_emb], dim=1)
        return z + F.relu(self.trans_fc(x))

    def rollout_latents(self, z0, actions_seq):
        zs = [z0]
        curr_z = z0
        for i in range(actions_seq.size(1)):
            curr_z = self.transition(curr_z, actions_seq[:, i])
            zs.append(curr_z)
        return torch.stack(zs, dim=1)


# =============================================================================
# REPLAY BUFFERS (from CleanRL BBF)
# =============================================================================

class SumTree:
    """Binary tree for O(log n) prioritized sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.max_priority = 1.0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def update(self, idx, priority):
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        self.max_priority = max(self.max_priority, priority)
    
    def get(self, s):
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
    
    @property
    def total(self):
        return self.tree[0]
    
    def __getitem__(self, idx):
        return self.tree[idx + self.capacity - 1]
    
    def reset(self):
        """Reset tree for new game in continual learning."""
        self.tree.fill(0.0)
        self.max_priority = 1.0


class SequenceReplayBuffer:
    """Uniform sampling replay buffer with GPU storage for fast gathering."""
    def __init__(self, capacity, obs_shape, device, max_seq_len, batch_size=32):
        self.capacity = capacity
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.obs_shape = obs_shape

        # GPU tensors for storage (eliminates CPU gather + H2D copy bottleneck)
        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.empty((capacity,), dtype=torch.int64, device=device)
        self.rewards = torch.empty((capacity,), dtype=torch.float32, device=device)
        self.dones = torch.empty((capacity,), dtype=torch.bool, device=device)

        # Pre-allocated weights buffer (uniform weights for this buffer type)
        self._weights_gpu = torch.ones(batch_size, dtype=torch.float32, device=device)

        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, done, priority=None):
        # Small H2D transfer per step (~28KB, unavoidable but tiny compared to batch gather)
        self.obs[self.pos].copy_(torch.as_tensor(obs, device=self.device))
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = bool(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_gpu(self, obs_t, action, reward, done, priority=None):
        # Expect obs_t on correct device with uint8 dtype to avoid extra copies.
        if obs_t.device.type != self.device.type:
            raise ValueError(f"add_gpu expects obs on {self.device}, got {obs_t.device}")
        if self.device.index is not None and obs_t.device.index != self.device.index:
            raise ValueError(f"add_gpu expects obs on {self.device}, got {obs_t.device}")
        if obs_t.dtype != torch.uint8:
            obs_t = obs_t.to(dtype=torch.uint8)
        self.obs[self.pos].copy_(obs_t)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = bool(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, seq_len):
        if self.size < seq_len:
            raise ValueError(f"Not enough data: size={self.size}, seq_len={seq_len}")

        # Zero-Bias Modulo Sampling to handle circular wrapping
        cursor = self.pos if self.size == self.capacity else 0
        max_logical = self.size - seq_len + 1
        logical_starts = np.random.randint(0, max_logical, size=batch_size)
        physical_starts = (cursor + logical_starts) % self.capacity

        seq_indices = (physical_starts[:, None] + np.arange(seq_len)) % self.capacity

        # Upload indices to GPU (tiny: batch_size * seq_len * 8 bytes)
        seq_indices_gpu = torch.from_numpy(seq_indices).to(self.device)

        # Fast GPU gather (no H2D copy needed - data already on GPU)
        obs_batch = self.obs[seq_indices_gpu]
        actions_batch = self.actions[seq_indices_gpu]
        rewards_batch = self.rewards[seq_indices_gpu]
        dones_batch = self.dones[seq_indices_gpu]

        # Ensure weights buffer is correct size
        if self._weights_gpu.shape[0] != batch_size:
            self._weights_gpu = torch.ones(batch_size, dtype=torch.float32, device=self.device)

        return (
            obs_batch,
            actions_batch,
            rewards_batch,
            dones_batch,
            physical_starts,
            self._weights_gpu[:batch_size],
        )

    def update_priorities(self, indices, priorities):
        pass  # No-op for uniform buffer

    def reset_counters(self):
        """Reset buffer for new game in continual learning (no memory reallocation)."""
        self.pos = 0
        self.size = 0


class PrioritizedSequenceReplayBuffer:
    """Prioritized Experience Replay buffer with GPU storage for fast gathering."""
    def __init__(self, capacity, obs_shape, device, max_seq_len, alpha=0.5, beta=0.5, eps=1e-6, batch_size=32):
        self.capacity = capacity
        self.device = device
        self.max_seq_len = max_seq_len
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.obs_shape = obs_shape
        self.batch_size = batch_size

        # GPU tensors for storage (eliminates CPU gather + H2D copy bottleneck)
        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.empty((capacity,), dtype=torch.int64, device=device)
        self.rewards = torch.empty((capacity,), dtype=torch.float32, device=device)
        self.dones = torch.empty((capacity,), dtype=torch.bool, device=device)

        # Pre-allocated weights buffer
        self._weights_gpu = torch.empty(batch_size, dtype=torch.float32, device=device)

        # Priority tree stays on CPU (small, needs CPU operations)
        self.tree = SumTree(capacity)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, done, priority=None):
        # Small H2D transfer per step (~28KB, unavoidable but tiny compared to batch gather)
        self.obs[self.pos].copy_(torch.as_tensor(obs, device=self.device))
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = bool(done)

        # New transitions get max priority (official BBF style: no alpha exponent)
        if priority is None:
            priority = self.tree.max_priority
        self.tree.update(self.pos, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_gpu(self, obs_t, action, reward, done, priority=None):
        # Expect obs_t on correct device with uint8 dtype to avoid extra copies.
        if obs_t.device.type != self.device.type:
            raise ValueError(f"add_gpu expects obs on {self.device}, got {obs_t.device}")
        if self.device.index is not None and obs_t.device.index != self.device.index:
            raise ValueError(f"add_gpu expects obs on {self.device}, got {obs_t.device}")
        if obs_t.dtype != torch.uint8:
            obs_t = obs_t.to(dtype=torch.uint8)
        self.obs[self.pos].copy_(obs_t)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = bool(done)

        # New transitions get max priority (official BBF style: no alpha exponent)
        if priority is None:
            priority = self.tree.max_priority
        self.tree.update(self.pos, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, seq_len):
        if self.size < seq_len:
            raise ValueError(f"Not enough data: size={self.size}, seq_len={seq_len}")

        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Stratified sampling for better coverage (on CPU - tree is on CPU)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = np.random.uniform(low, high)
            idx = self.tree.get(s)

            cursor = self.pos if self.size == self.capacity else 0
            max_valid = self.size - seq_len

            # Retry if invalid index
            attempts = 0
            while attempts < 10:
                if self.size == self.capacity:
                    logical_idx = (idx - cursor) % self.capacity
                    if logical_idx <= max_valid:
                        break
                else:
                    if idx <= max_valid:
                        break
                s = np.random.uniform(low, high)
                idx = self.tree.get(s)
                attempts += 1

            indices[i] = idx
            priorities[i] = self.tree[idx]

        seq_indices = (indices[:, None] + np.arange(seq_len)) % self.capacity

        # Upload indices to GPU (tiny: batch_size * seq_len * 8 bytes)
        seq_indices_gpu = torch.from_numpy(seq_indices).to(self.device)

        # Fast GPU gather (no H2D copy needed - data already on GPU)
        obs_batch = self.obs[seq_indices_gpu]
        actions_batch = self.actions[seq_indices_gpu]
        rewards_batch = self.rewards[seq_indices_gpu]
        dones_batch = self.dones[seq_indices_gpu]

        # Compute importance sampling weights (Official BBF: 1/sqrt(prob), no beta)
        probs = priorities / (self.tree.total + 1e-10)
        weights = 1.0 / np.sqrt(probs + 1e-10)
        weights = weights / (weights.max() + 1e-10)

        # Ensure weights buffer is correct size and upload weights
        if self._weights_gpu.shape[0] != batch_size:
            self._weights_gpu = torch.empty(batch_size, dtype=torch.float32, device=self.device)
        self._weights_gpu.copy_(torch.from_numpy(weights.astype(np.float32)))

        return (
            obs_batch,
            actions_batch,
            rewards_batch,
            dones_batch,
            indices,
            self._weights_gpu[:batch_size],
        )

    def update_priorities(self, indices, losses):
        """Update priorities based on losses (Official BBF: sqrt(loss), no alpha)."""
        priorities = np.sqrt(losses + self.eps)
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)

    def reset_counters(self):
        """Reset buffer for new game in continual learning (no memory reallocation)."""
        self.pos = 0
        self.size = 0
        self.tree.reset()


# =============================================================================
# SCHEDULE FUNCTIONS
# =============================================================================

def get_linear_schedule(start, end, t, duration):
    if t >= duration:
        return end
    return start + (end - start) * (t / duration)


def get_exponential_schedule(start, end, t, duration):
    """Exponential interpolation in log-space (BBF paper specification)."""
    if t >= duration:
        return end
    return start * (end / start) ** (t / duration)


# =============================================================================
# AGENT CONFIG
# =============================================================================

@dataclass
class AgentConfig:
    # General run settings (align with bbf_atari.py CLI)
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

    # Environment / training shape
    env_id: str = "ALE/Breakout-v5"
    total_steps: int = 100_000  # Alias for total_timesteps
    num_envs: int = 1
    buffer_size: int = 120_000
    
    learning_rate: float = 0.0001
    weight_decay: float = 0.1
    impala_width: int = 4
    
    replay_ratio: int = 8
    batch_size: int = 32
    
    initial_gamma: float = 0.97
    final_gamma: float = 0.997
    initial_n_step: int = 10
    final_n_step: int = 3
    anneal_duration: int = 10_000
    
    reset_interval: int = 40_000
    shrink_factor: float = 0.5
    reset_warmup_steps: int = 2000
    
    # Official paper value for SPR weight
    spr_weight: float = 5.0
    jumps: int = 5
    
    # Official value support (with reward clipping)
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    tau: float = 0.005

    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 2000

    # Prioritized Experience Replay (PER)
    use_per: bool = True
    per_alpha: float = 0.5
    per_beta: float = 0.5
    per_eps: float = 1e-6

    # Mixed Precision Training
    use_amp: bool = True  # Enable automatic mixed precision (FP16) for faster training

    log_file: str = ""
    
    continual: bool = False
    continual_games: str = ""
    continual_cycles: int = DEFAULT_CONTINUAL_CYCLES
    continual_cycle_frames: int = DEFAULT_CONTINUAL_CYCLE_FRAMES
    per_game_learning_starts: int = 2000

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")


# =============================================================================
# AGENT CLASS
# =============================================================================

class Agent:
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: AgentConfig):
        self.config = config
        self.device = config.device
        self.action_space = action_space
        
        self.global_step = 0
        self.grad_step = 0
        self.steps_since_reset = 0
        
        self.last_obs: Optional[np.ndarray] = None
        self.last_obs_t: Optional[torch.Tensor] = None
        self.last_action: Optional[int] = None
        self.learning_ready_step = config.learning_starts

        # Fix action space for continual learning
        if config.continual:
            self.num_actions = ATARI_CANONICAL_ACTIONS
        else:
            self.num_actions = action_space.n

        # Infer input channels
        obs_shape = observation_space.shape
        if obs_shape[0] == FRAME_SKIP:
            self.channels_last = False
            self.in_channels = obs_shape[0]
        elif obs_shape[-1] == FRAME_SKIP:
            self.channels_last = True
            self.in_channels = obs_shape[-1]
        else:
            raise ValueError("Unable to infer stacked frame dimension.")

        # Networks
        self.q_network = BBFNetwork(self.in_channels, self.num_actions, config).to(self.device)
        self.target_network = BBFNetwork(self.in_channels, self.num_actions, config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with weight decay excluding biases (official BBF excludes 1D params)
        decay_params = []
        no_decay_params = []
        for name, param in self.q_network.named_parameters():
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        self.optimizer = optim.AdamW([
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=config.learning_rate, eps=1.5e-4)
        
        self.aug = RandomShift(pad=4).to(self.device)

        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler('cuda', enabled=config.use_amp and self.device.type == 'cuda')
        if config.use_amp and self.device.type == 'cuda':
            print("*** MIXED PRECISION (AMP) ENABLED - Using FP16 for faster training ***")

        # Replay Buffer - choose based on use_per
        # GPU storage eliminates CPU gather + H2D copy bottleneck
        self.seq_req = config.jumps + 1 + config.initial_n_step
        if config.use_per:
            print("Using Prioritized Experience Replay (PER) with GPU storage")
            self.replay_buffer = PrioritizedSequenceReplayBuffer(
                config.buffer_size,
                observation_space.shape,
                self.device,
                self.seq_req,
                alpha=config.per_alpha,
                beta=config.per_beta,
                eps=config.per_eps,
                batch_size=config.batch_size
            )
        else:
            print("Using Uniform Experience Replay with GPU storage")
            self.replay_buffer = SequenceReplayBuffer(
                config.buffer_size,
                observation_space.shape,
                self.device,
                self.seq_req,
                batch_size=config.batch_size
            )

        self.game_frame_counters: Dict[str, int] = {}

        # Profiler state (for continual mode, managed externally)
        self.profiler: Optional[torch.profiler.profile] = None
        self.profiler_active: bool = False

    def _get_epsilon(self):
        duration = self.config.exploration_fraction * self.config.total_steps
        if self.global_step >= duration:
            return self.config.end_e
        return self.config.start_e + (self.config.end_e - self.config.start_e) * (self.global_step / duration)

    def _hard_reset(self):
        """
        BBF Reset Strategy (from official implementation):
        - Shrink-perturb ONLY: encoder, transition model (trans_fc, action_emb)
        - Fully reset: heads (value, advantage), projection, predictor, rep_head
        """
        print(f"  [Agent] Hard Reset (shrink={self.config.shrink_factor})...")
        with torch.no_grad():
            fresh = BBFNetwork(self.in_channels, self.num_actions, self.config).to(self.device)
            alpha = self.config.shrink_factor
            
            for (name, p), (_, p_new) in zip(self.q_network.named_parameters(), fresh.named_parameters()):
                # Shrink-perturb only encoder and transition model
                if "encoder" in name or "trans_fc" in name or "action_emb" in name:
                    p.data.copy_(alpha * p.data + (1.0 - alpha) * p_new.data)
                else:
                    # Fully reset: heads, projection, predictor, rep_head
                    p.data.copy_(p_new.data)
                
                # Always clear optimizer state (momentum etc.) for ALL params
                if p in self.optimizer.state:
                    del self.optimizer.state[p]
            
            self.target_network.load_state_dict(self.q_network.state_dict())

    def start_new_game(self):
        """Called by BenchmarkRunner when a new game starts."""
        print(f"[Agent] New Game - global_step={self.global_step} grad_step={self.grad_step}")
        print("[Agent] Resetting buffer and performing hard reset.")

        # Reset buffer
        self.replay_buffer.reset_counters()

        # Reset annealing
        self.steps_since_reset = 0
        self.last_obs_t = None

        # Shrink & Perturb
        self._hard_reset()

        # Set learning threshold
        self.learning_ready_step = self.global_step + self.config.per_game_learning_starts
        print(f"[Agent] learning_ready_step set to {self.learning_ready_step} (current + {self.config.per_game_learning_starts})")

    def act(self, observation) -> int:
        # Early epsilon check to avoid unnecessary GPU work
        epsilon = self._get_epsilon()
        if random.random() < epsilon:
            self.last_obs = np.array(observation, copy=False)
            self.last_obs_t = None
            self.last_action = random.randint(0, self.num_actions - 1)
            return self.last_action
        
        obs_np = np.array(observation, copy=False)
        obs_t = torch.as_tensor(obs_np, device=self.device, dtype=torch.uint8)
        obs_tensor = obs_t.unsqueeze(0)
        if self.channels_last:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            
        with torch.no_grad():
            q_dist = self.target_network(obs_tensor)  # Use TARGET network (matches bbf_atari.py)
            q_values = torch.sum(q_dist * self.target_network.support, dim=2)
            action = int(torch.argmax(q_values, dim=1).item())
            
        self.last_obs = obs_np
        self.last_obs_t = obs_t
        self.last_action = action
        return action

    def step(self, next_observation, reward: float, terminated: bool, truncated: bool = False, info: Optional[Dict] = None):
        """Process a transition and optionally train.

        Args:
            next_observation: The observation after taking the action
            reward: Reward received
            terminated: True if episode ended due to MDP rules (no bootstrapping)
            truncated: True if episode ended due to time limit/budget (should bootstrap)
            info: Optional info dict from environment
        """
        if self.last_obs is None or self.last_action is None:
            return None

        # For replay buffer: only use `terminated` to decide bootstrapping
        # truncated episodes should bootstrap (the episode didn't really end)
        if self.last_obs_t is not None:
            self.replay_buffer.add_gpu(self.last_obs_t, self.last_action, reward, terminated)
        else:
            self.replay_buffer.add(self.last_obs, self.last_action, reward, terminated)
        self.global_step += 1

        self.last_obs = np.array(next_observation, copy=False)
        self.last_obs_t = None

        # Clean up agent state if episode ended (either way)
        if terminated or truncated:
            self.last_obs = None
            self.last_action = None
        
        # Training
        train_stats = None
        min_samples = self.seq_req + 1

        # Diagnostic logging every 1000 steps
        if self.global_step % 1000 == 0:
            print(f"[DIAG] step={self.global_step} buffer_size={self.replay_buffer.size} "
                  f"min_samples={min_samples} learning_ready_step={self.learning_ready_step} "
                  f"grad_step={self.grad_step}")

        if self.replay_buffer.size > min_samples and self.global_step >= self.learning_ready_step:
            # One-time message when training starts
            if self.grad_step == 0:
                print(f"[TRAINING STARTED] step={self.global_step} buffer_size={self.replay_buffer.size}")

            # Check for reset interval (BBF periodic resets)
            # Reset based on GRADIENT STEPS (paper: "reset every 40k gradient steps")
            if self.grad_step > 0 and self.grad_step % self.config.reset_interval == 0:
                print(f"  [Agent] Hard Reset at grad step {self.grad_step} (env step {self.global_step})")
                self._hard_reset()
                self.steps_since_reset = 0
                
                # Offline warmup after reset
                print(f"  [Agent] Offline Warmup ({self.config.reset_warmup_steps} steps)...")
                for _ in range(self.config.reset_warmup_steps):
                    self.grad_step += 1
                    self._train_batch(self.config.initial_gamma, self.config.initial_n_step)
                    self._step_profiler()

            # Replay ratio loop
            for _ in range(self.config.replay_ratio):
                self.grad_step += 1
                self.steps_since_reset += 1

                curr_gamma = get_exponential_schedule(
                    self.config.initial_gamma, self.config.final_gamma,
                    self.steps_since_reset, self.config.anneal_duration
                )
                curr_n = int(round(get_exponential_schedule(
                    self.config.initial_n_step, self.config.final_n_step,
                    self.steps_since_reset, self.config.anneal_duration
                )))

                train_stats = self._train_batch(curr_gamma, curr_n)
                self._step_profiler()

        return train_stats

    def _step_profiler(self):
        """Step the profiler and stop it after scheduled steps complete."""
        if not self.profiler_active or self.profiler is None:
            return

        # Calculate total steps needed for schedule to complete
        # wait + warmup + active = total steps before on_trace_ready fires
        wait_steps = max(1, self.config.learning_starts // 10)
        total_profiler_steps = wait_steps + 5 + self.config.torch_profile_steps

        # Must call step() exactly total_profiler_steps times to complete the cycle
        # Only stop AFTER completing (use > not >=)
        if self.grad_step > total_profiler_steps:
            # Schedule has completed, on_trace_ready already fired
            self.profiler_active = False
            self.profiler = None
            print(f"*** Torch Profiler completed after {total_profiler_steps} grad steps ***")
            print(f"*** View with: tensorboard --logdir=<run_dir>/profiler ***")
            return

        self.profiler.step()

    def _train_batch(self, curr_gamma, curr_n):
        """Training logic with AMP support (matches bbf_atari.py exactly)."""
        args = self.config
        use_amp = args.use_amp and self.device.type == 'cuda'
        fetch_len = args.jumps + 1 + curr_n

        data_obs, data_act, data_rew, data_done, indices, weights = self.replay_buffer.sample(args.batch_size, fetch_len)

        if self.channels_last:
            data_obs = data_obs.permute(0, 1, 4, 2, 3)

        # --- FORWARD PASS WITH MIXED PRECISION ---
        # AMP Strategy: FP16 for convolutions (fast), FP32 for C51/SPR (accurate)
        with autocast(device_type='cuda', enabled=use_amp):
            # --- AUGMENTATION (obs_0) ---
            obs_0 = self.aug(data_obs[:, 0])

            # --- ENCODE (online) - Keep FP16 for speed ---
            z0 = self.q_network.encode(obs_0)

            # --- SPR ROLLOUT ---
            rollout_acts = data_act[:, :args.jumps]
            z_seq = self.q_network.rollout_latents(z0, rollout_acts)

        # --- SPR TARGETS (target network) - FORCE FP32 for normalization ---
        with torch.no_grad(), autocast(device_type='cuda', enabled=False):
            # Stack obs for jumps 1..jumps: (batch, jumps, C, H, W) -> (batch*jumps, C, H, W)
            obs_jumps = data_obs[:, 1:args.jumps+1]
            B, J = obs_jumps.shape[0], obs_jumps.shape[1]
            obs_jumps_flat = obs_jumps.reshape(-1, *obs_jumps.shape[2:])

            # Augmentation + encode in FP16 (convolutions are fast)
            with autocast(device_type='cuda', enabled=use_amp):
                obs_jumps_aug = self.aug(obs_jumps_flat)
                t_z_flat = self.target_network.encode(obs_jumps_aug)

            # Projection + normalization in FP32 (critical for numerical stability)
            t_z_flat = t_z_flat.float()
            t_p_flat = self.target_network.projection(t_z_flat)
            t_p_norm = F.normalize(t_p_flat, dim=1)

            # Reshape: (batch*jumps, proj_dim) -> (batch, jumps, proj_dim)
            target_latents = t_p_norm.view(B, J, -1)

        # --- SPR LOSS - FORCE FP32 for normalization ---
        with autocast(device_type='cuda', enabled=False):
            # Get predicted latents for all jumps: z_seq[:, 1:] has shape (batch, jumps, latent_dim)
            z_jumps = z_seq[:, 1:args.jumps+1].float()  # Force FP32
            z_jumps_flat = z_jumps.reshape(-1, z_jumps.shape[-1])

            # Projection + predictor + normalization in FP32
            p_flat = self.q_network.projection(z_jumps_flat)
            pred_flat = self.q_network.predictor(p_flat)
            pred_norm = F.normalize(pred_flat, dim=1)

            # Reshape back
            pred_latents = pred_norm.view(B, J, -1)

            # Compute MSE loss
            spr_loss_per_jump = ((pred_latents - target_latents) ** 2).sum(dim=2)

            # Apply cumulative done mask
            spr_done_mask = 1.0 - data_done[:, :args.jumps].float()
            cumulative_mask = torch.cumprod(spr_done_mask, dim=1)

            # Match original: mean over batch for each jump, then sum over jumps
            spr_loss = (spr_loss_per_jump * cumulative_mask).mean(dim=0).sum()

        # --- C51 TARGET COMPUTATION - FORCE FP32 for distributional RL ---
        avg_q = 0.0
        with torch.no_grad(), autocast(device_type='cuda', enabled=False):
            gammas = torch.pow(curr_gamma, torch.arange(curr_n, device=self.device))
            not_done = 1.0 - data_done.float()
            mask_seq = torch.cat([torch.ones((args.batch_size, 1), device=self.device), not_done[:, :-1]], dim=1)
            reward_mask = torch.cumprod(mask_seq, dim=1)[:, :curr_n]
            n_step_rew = torch.sum(data_rew[:, :curr_n] * gammas * reward_mask, dim=1)

            obs_n = data_obs[:, curr_n]
            bootstrap_mask = torch.prod(not_done[:, :curr_n], dim=1)

            # Forward passes in FP16 (fast), then convert to FP32 for distribution ops
            with autocast(device_type='cuda', enabled=use_amp):
                next_dist = self.target_network(obs_n)
                online_n_dist = self.q_network(obs_n)

            # All distribution operations in FP32
            next_dist = next_dist.float()
            online_n_dist = online_n_dist.float()
            next_sup = self.target_network.support.float()

            # Compute Q-values and best action in FP32
            avg_q = (online_n_dist * self.q_network.support.float()).sum(2).mean().item()
            best_act = (online_n_dist * next_sup).sum(2).argmax(1)
            next_pmf = next_dist[torch.arange(args.batch_size), best_act]

            # C51 categorical projection in FP32 (critical!)
            gamma_n = curr_gamma ** curr_n
            Tz = n_step_rew.unsqueeze(1) + bootstrap_mask.unsqueeze(1) * gamma_n * next_sup.unsqueeze(0)
            Tz = Tz.clamp(min=args.v_min, max=args.v_max)
            b = (Tz - args.v_min) / self.q_network.delta_z
            l = b.floor().clamp(0, args.n_atoms - 1).long()
            u = b.ceil().clamp(0, args.n_atoms - 1).long()

            target_pmf = torch.zeros_like(next_pmf)
            if not hasattr(self, "_c51_offset") or self._c51_offset.device != self.device or self._c51_offset.shape[0] != args.batch_size:
                self._c51_offset = (torch.arange(args.batch_size, device=self.device, dtype=torch.long) * args.n_atoms).unsqueeze(1)
            offset = self._c51_offset
            target_pmf.view(-1).index_add_(0, (l + offset).view(-1), (next_pmf * (u.float() - b)).view(-1))
            target_pmf.view(-1).index_add_(0, (u + offset).view(-1), (next_pmf * (b - l.float())).view(-1))

        # --- C51 LOSS - FORCE FP32 ---
        with autocast(device_type='cuda', enabled=False):
            z0_fp32 = z0.float()
            curr_dist = self.q_network.get_dist(z0_fp32).float()
            log_p = torch.log(curr_dist[torch.arange(args.batch_size), data_act[:, 0]] + 1e-8)

            # Per-sample C51 loss (for priority updates)
            c51_loss_per_sample = -torch.sum(target_pmf * log_p, dim=1)

            # Apply importance sampling weights (PER correction)
            weighted_c51_loss = (c51_loss_per_sample * weights).mean()
            total_loss = weighted_c51_loss + args.spr_weight * spr_loss

        # --- BACKWARD (with gradient scaling for AMP) ---
        self.optimizer.zero_grad()
        if self.scaler is not None and use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # --- OPTIMIZER STEP (with gradient scaling for AMP) ---
        if self.scaler is not None and use_amp:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
            self.optimizer.step()

        # Soft update target
        for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

        # Update priorities based on C51 loss (Official BBF uses sqrt(loss))
        losses = c51_loss_per_sample.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, losses)

        # Diagnostic: confirm training is happening
        if self.grad_step % 1000 == 0:
            print(f"[TRAIN] grad_step={self.grad_step} loss={total_loss.item():.4f} spr={spr_loss.item():.4f} avg_q={avg_q:.2f}")

        return {
            "loss": total_loss.item(),
            "spr_loss": spr_loss.item() if isinstance(spr_loss, torch.Tensor) else spr_loss,
            "avg_q": avg_q,
            "gamma": curr_gamma,
            "n_step": curr_n
        }

    def save_model(self, path: str):
        torch.save(self.q_network.state_dict(), path)


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
            with_stack=False,  # Disabled due to PyTorch bug with Python stack tracing
        )
        agent.profiler.start()
        agent.profiler_active = True
        print(f"*** Torch Profiler ENABLED - will profile {config.torch_profile_steps} steps ***")
        print(f"*** Profiler output: {profiler_dir} ***")

    # Signal new game
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
    last_model_save = context.last_model_save
    
    # Best model tracking for this game
    best_return = float('-inf')
    best_model_path = f"best_model_{game_name.replace('/', '_').replace(' ', '_')}.pt"

    for _ in range(max_step_budget):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

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
                    seed=config.seed + cycle_index,
                    backend='gym',
                    env_id=env_id,
                    params={
                        'noop_max': 30, 'frame_skip': FRAME_SKIP, 'resize_shape': (84, 84), 'grayscale': True
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
    parser.add_argument("--buffer-size", type=int, default=120_000)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--impala-width", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-ratio", type=int, default=8, help="Number of gradient steps per env step")
    parser.add_argument("--initial-gamma", type=float, default=0.97)
    parser.add_argument("--final-gamma", type=float, default=0.997)
    parser.add_argument("--initial-n-step", type=int, default=10)
    parser.add_argument("--final-n-step", type=int, default=3)
    parser.add_argument("--anneal-duration", type=int, default=10_000)
    parser.add_argument("--reset-interval", type=int, default=40_000, help="Gradient steps between periodic resets (paper: 40k)")
    parser.add_argument("--shrink-factor", type=float, default=0.5)
    parser.add_argument("--reset-warmup-steps", type=int, default=2000)
    parser.add_argument("--spr-weight", type=float, default=5.0)
    parser.add_argument("--jumps", type=int, default=5, help="Number of SPR prediction jumps")
    parser.add_argument("--n-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.01)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--use-per", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-alpha", type=float, default=0.5)
    parser.add_argument("--per-beta", type=float, default=0.5)
    parser.add_argument("--per-eps", type=float, default=1e-6)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)

    # Continual benchmark
    parser.add_argument("--continual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continual-games", type=str, default="")
    parser.add_argument("--continual-cycles", type=int, default=DEFAULT_CONTINUAL_CYCLES)
    parser.add_argument("--continual-cycle-frames", type=int, default=DEFAULT_CONTINUAL_CYCLE_FRAMES)
    parser.add_argument("--per-game-learning-starts", type=int, default=2000)
    
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
    print(f"  Reset Interval: {config.reset_interval} (grad steps)")
    print(f"  Total Steps:    {config.total_steps}")
    print(f"  Seed:           {config.seed}")
    print(f"  Learning Rate:  {config.learning_rate}")
    print(f"  Continual:      {config.continual}")
    if config.continual:
        print(f"  Games:          {config.continual_games}")
        print(f"  Cycles:         {config.continual_cycles}")
        print(f"  Cycle Frames:   {config.continual_cycle_frames}")
    print(f"{'='*60}\n")

    if config.continual:
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
    else:
        # Single game mode - CleanRL-style training loop with SyncVectorEnv
        num_envs = config.num_envs  # BBF uses single env for sequence correctness
        envs = gym.vector.SyncVectorEnv(
            [make_env(config.env_id, config.seed + i, i, config.capture_video, run_name) for i in range(num_envs)]
        )
        
        # Reinitialize agent with correct observation space from vectorized env
        agent = Agent(envs.single_observation_space, envs.single_action_space, config)
        
        obs, _ = envs.reset(seed=config.seed)
        start_time = time.time()
        training_start_time = None
        training_start_step = None
        all_returns = []
        
        print("Starting Training Loop...")

        # Best model checkpointing
        best_return = float('-inf')
        best_model_path = os.path.join(run_dir, "best_model.pt")

        # Setup torch.profiler if enabled (uses agent's profiler attributes)
        if config.torch_profile:
            profiler_dir = os.path.join(run_dir, "profiler")
            os.makedirs(profiler_dir, exist_ok=True)
            # Schedule: wait for buffer to fill, warmup GPU, then profile
            agent.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=max(1, config.learning_starts // 10),  # Wait for some buffer filling
                    warmup=5,
                    active=config.torch_profile_steps,
                    repeat=1
                ),
                on_trace_ready=tensorboard_trace_handler(profiler_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,  # Disabled due to PyTorch bug with Python stack tracing
            )
            agent.profiler.start()
            agent.profiler_active = True
            print(f"*** Torch Profiler ENABLED - will profile {config.torch_profile_steps} steps ***")
            print(f"*** Profiler output: {profiler_dir} ***")
        
        for global_step in range(config.total_steps):
            # Warmup logging
            if global_step <= config.learning_starts and global_step % 100 == 0:
                print(f"Warmup: Filling Replay Buffer {global_step}/{config.learning_starts}")
            
            # --- ACTION SELECTION (vectorized) ---
            obs_t = torch.as_tensor(obs, device=agent.device, dtype=torch.uint8)
            obs_t_net = obs_t
            if agent.channels_last:
                obs_t_net = obs_t_net.permute(0, 3, 1, 2)

            epsilon = agent._get_epsilon()
            if random.random() < epsilon:
                actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
            else:
                with torch.no_grad():
                    q_dist = agent.target_network(obs_t_net)
                    q_values = torch.sum(q_dist * agent.target_network.support, dim=2)
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
            # --- ENV STEP ---
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
            # --- EPISODE LOGGING (matches bbf_atari.py exactly) ---
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_return = float(info['episode']['r'])
                        all_returns.append(ep_return)
                        print(f"--> Episode Done. Step={global_step}, Return={ep_return:.0f}, Length={info['episode']['l']}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        
                        # Save best model
                        if ep_return > best_return:
                            best_return = ep_return
                            agent.save_model(best_model_path)
                            print(f"    ★ New best! Saved model with return {best_return:.0f}")
                            writer.add_scalar("charts/best_return", best_return, global_step)
            elif "episode" in infos:
                ep_return = float(infos['episode']['r'])
                all_returns.append(ep_return)
                print(f"--> Episode Done. Step={global_step}, Return={ep_return:.0f}, Length={infos['episode']['l']}")
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
                
                # Save best model
                if ep_return > best_return:
                    best_return = ep_return
                    agent.save_model(best_model_path)
                    print(f"    ★ New best! Saved model with return {best_return:.0f}")
                    writer.add_scalar("charts/best_return", best_return, global_step)
            
            # --- BUFFER ADD (vectorized) ---
            # Only use `terminated` for buffer's done flag (for correct bootstrapping)
            # Truncated episodes should bootstrap, terminated episodes should not
            for idx in range(num_envs):
                agent.replay_buffer.add_gpu(obs_t[idx], actions[idx], rewards[idx], terminations[idx])
            
            agent.global_step += 1
            obs = next_obs  # SyncVectorEnv auto-resets
            
            # --- TRAINING ---
            train_stats = None
            min_samples = agent.seq_req + 1
            
            if agent.replay_buffer.size > min_samples and agent.global_step >= agent.learning_ready_step:
                # Check for reset interval (BBF periodic resets)
                # Reset based on GRADIENT STEPS (paper: "reset every 40k gradient steps")
                if agent.grad_step > 0 and agent.grad_step % config.reset_interval == 0:
                    print(f"*** HARD RESET at grad step {agent.grad_step} (env step {agent.global_step}) ***")
                    agent._hard_reset()
                    agent.steps_since_reset = 0
                    
                    # Offline warmup after reset
                    print(f"*** Offline Warmup ({config.reset_warmup_steps} steps) ***")
                    for _ in range(config.reset_warmup_steps):
                        agent.grad_step += 1
                        agent._train_batch(config.initial_gamma, config.initial_n_step)
                        agent._step_profiler()

                # Replay ratio loop
                for _ in range(config.replay_ratio):
                    agent.grad_step += 1
                    agent.steps_since_reset += 1
                    
                    curr_gamma = get_exponential_schedule(
                        config.initial_gamma, config.final_gamma,
                        agent.steps_since_reset, config.anneal_duration
                    )
                    curr_n = int(round(get_exponential_schedule(
                        config.initial_n_step, config.final_n_step,
                        agent.steps_since_reset, config.anneal_duration
                    )))
                    
                    train_stats = agent._train_batch(curr_gamma, curr_n)
                    agent._step_profiler()  # Profiler stepping handled by agent

            # Training stats logging (CleanRL style)
            if train_stats:
                # Track when training actually starts (after warmup)
                if training_start_time is None:
                    training_start_time = time.time()
                    training_start_step = global_step
                    print(f"*** Training started at step {global_step} ***")
                
                if agent.grad_step % 100 == 0:
                    # SPS based on training time only (excludes warmup)
                    training_elapsed = time.time() - training_start_time
                    training_steps = global_step - training_start_step
                    sps = int(training_steps / (training_elapsed + 1e-5))
                    remaining_steps = config.total_steps - global_step
                    eta_min = (remaining_steps / (sps + 1e-5)) / 60
                    
                    print(f"Step: {global_step} | Grad: {agent.grad_step} | Loss: {train_stats['loss']:.3f} | SPR: {train_stats['spr_loss']:.3f} | "
                          f"AvgQ: {train_stats['avg_q']:.2f} | Eps: {epsilon:.2f} | SPS: {sps} | ETA: {eta_min:.1f} min")
                    
                    writer.add_scalar("losses/total_loss", train_stats['loss'], agent.grad_step)
                    writer.add_scalar("losses/spr_loss", train_stats['spr_loss'], agent.grad_step)
                    writer.add_scalar("charts/avg_q", train_stats['avg_q'], agent.grad_step)
                    writer.add_scalar("charts/n_step", train_stats['n_step'], agent.grad_step)
                    writer.add_scalar("charts/gamma", train_stats['gamma'], agent.grad_step)
                    writer.add_scalar("charts/SPS", sps, agent.grad_step)
        
        envs.close()

        # Mark profiler as inactive if still running (it will auto-cleanup)
        if agent.profiler_active and agent.profiler is not None:
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
        print(f"  Reset Interval: {config.reset_interval} (grad steps)")
        print(f"  Seed:           {config.seed}")
        print(f"  Environment:    {config.env_id}")
        print(f"  Total Time:     {total_time/60:.1f} minutes")
        print(f"  Grad Steps:     {agent.grad_step}")
        if all_returns:
            print(f"  Episodes:       {len(all_returns)}")
            print(f"  Max Return:     {max(all_returns):.0f}")
            print(f"  Mean Return:    {np.mean(all_returns):.1f}")
            print(f"  Last 5:         {[int(r) for r in all_returns[-5:]]}")
        if best_return > float('-inf'):
            print(f"  Best Model:     {best_model_path}")
            print(f"  Best Return:    {best_return:.0f}")
        print(f"{'='*60}\n")
        
        # Log final metrics
        if all_returns:
            writer.add_scalar("final/max_return", max(all_returns), 0)
            writer.add_scalar("final/mean_return", np.mean(all_returns), 0)
            writer.add_scalar("final/episodes", len(all_returns), 0)
            writer.add_scalar("final/total_time_minutes", total_time/60, 0)

    if config.save_model:
        final_model_path = os.path.join(run_dir, f"{config.exp_name}_final.pt")
        agent.save_model(final_model_path)
        print(f"Saved final model to {final_model_path}")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
