# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rainbow/#rainbow_ataripy
import os
import random
import time
import math
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import ale_py 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# =============================================================================
# PROFILING UTILITIES
# =============================================================================
class Timer:
    """Lightweight profiler for diagnosing runtime bottlenecks."""
    def __init__(self, enabled=True, sync_cuda=True):
        self.enabled = enabled
        self.sync_cuda = sync_cuda
        self.times = {}
        self.counts = {}
        self._stack = []
    
    def start(self, name):
        if not self.enabled:
            return
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._stack.append((name, time.perf_counter()))
    
    def stop(self):
        if not self.enabled or not self._stack:
            return
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        name, start = self._stack.pop()
        elapsed = time.perf_counter() - start
        if name not in self.times:
            self.times[name] = 0.0
            self.counts[name] = 0
        self.times[name] += elapsed
        self.counts[name] += 1
    
    def report(self):
        if not self.times:
            print("No timing data collected.")
            return
        
        total = sum(self.times.values())
        print(f"\n{'='*70}")
        print(f"{'PROFILING REPORT':^70}")
        print(f"{'='*70}")
        print(f"{'Operation':<30} {'Total (s)':>10} {'%':>7} {'Avg (ms)':>10} {'Calls':>10}")
        print(f"{'-'*70}")
        
        for name, t in sorted(self.times.items(), key=lambda x: -x[1]):
            avg_ms = (t / self.counts[name]) * 1000
            pct = (t / total) * 100
            print(f"{name:<30} {t:>10.2f} {pct:>6.1f}% {avg_ms:>10.2f} {self.counts[name]:>10}")
        
        print(f"{'-'*70}")
        print(f"{'TOTAL':<30} {total:>10.2f} {'100.0':>6}%")
        print(f"{'='*70}\n")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "BBF-Atari"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = False
    
    # Profiling
    profile: bool = False  # Enable detailed timing profiling
    torch_profile: bool = False  # Enable PyTorch profiler (GPU + CPU) - outputs to TensorBoard
    torch_profile_steps: int = 50  # Number of training steps to profile
    
    # Mixed Precision Training
    use_amp: bool = True  # Enable automatic mixed precision (FP16) for faster training

    env_id: str = "ALE/Breakout-v5"
    total_timesteps: int = 100_000
    
    # Enforce num_envs=1 for BBF sequence correctness
    num_envs: int = 1 
    
    replay_ratio: int = 8
    batch_size: int = 32
    
    # Reduced buffer size to prevent OOM on standard RAM (needs ~3.5GB)
    buffer_size: int = 120_000
    
    learning_rate: float = 0.0001
    weight_decay: float = 0.1
    impala_width: int = 4
    
    initial_gamma: float = 0.97
    final_gamma: float = 0.997
    initial_n_step: int = 10
    final_n_step: int = 3
    anneal_duration: int = 10_000

    # Hard reset every N gradient steps (BBF paper: 40k grad steps)
    reset_interval: int = 40_000
    shrink_factor: float = 0.5
    # Warmup training after reset to recover policy (NOT in original paper, but helps stability)
    reset_warmup_steps: int = 2000
    
    spr_weight: float = 5.0  # Official paper value
    jumps: int = 5
    
    # Value support (official uses -10/10 with reward clipping)
    n_atoms: int = 51
    v_min: float = -10
    v_max: float = 10
    tau: float = 0.005

    start_e: float = 1.0
    end_e: float = 0.01
    # Paper uses 10% exploration (10k steps out of 100k)
    exploration_fraction: float = 0.10
    learning_starts: int = 2000

    # Prioritized Experience Replay (PER)
    use_per: bool = True
    per_alpha: float = 0.5  # Priority exponent (0 = uniform, 1 = full prioritization)
    per_beta: float = 0.5   # Importance sampling exponent (fixed, not annealed in BBF)
    per_eps: float = 1e-6   # Small constant to avoid zero priorities


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        # Gymnasium 1.0+ ALE configuration:
        # repeat_action_probability=0.0 ensures NO STICKY ACTIONS (Standard Atari 100k)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
            
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
        # shift of `pad` pixels = pad/dim in normalized coords (since grid is [-1,1])
        shift_y = (torch.randint(0, 2 * self.pad + 1, (b,), device=x.device) - self.pad).float()
        shift_x = (torch.randint(0, 2 * self.pad + 1, (b,), device=x.device) - self.pad).float()
        
        # Convert pixel shifts to normalized grid coordinates
        # grid_sample expects coordinates in [-1, 1], so shift/dim * 2
        shift_y_norm = shift_y / h * 2
        shift_x_norm = shift_x / w * 2
        
        # Build affine transformation matrix for translation only
        # [1, 0, tx]
        # [0, 1, ty]
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
    def __init__(self, env, args):
        super().__init__()
        self.n_atoms = args.n_atoms
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.delta_z = (args.v_max - args.v_min) / (args.n_atoms - 1)
        self.n_actions = env.single_action_space.n
        self.register_buffer("support", torch.linspace(args.v_min, args.v_max, args.n_atoms))
        
        w = args.impala_width
        self.encoder = nn.Sequential(
            ImpalaBlock(4, 16 * w),
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
        """Sample a leaf index given a value s in [0, total_priority]."""
        idx = 0
        while idx < self.capacity - 1:  # While not a leaf
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)  # Convert tree idx to data idx
    
    @property
    def total(self):
        return self.tree[0]
    
    def __getitem__(self, idx):
        return self.tree[idx + self.capacity - 1]


class SequenceReplayBuffer:
    """Uniform sampling replay buffer."""
    def __init__(self, capacity, obs_shape, device, max_seq_len):
        self.capacity = capacity
        self.device = device
        self.max_seq_len = max_seq_len
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity), dtype=np.int64)
        self.rewards = np.zeros((capacity), dtype=np.float32)
        self.dones = np.zeros((capacity), dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, done, priority=None):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
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
        
        return (
            torch.as_tensor(self.obs[seq_indices], device=self.device),
            torch.as_tensor(self.actions[seq_indices], device=self.device),
            torch.as_tensor(self.rewards[seq_indices], device=self.device),
            torch.as_tensor(self.dones[seq_indices], device=self.device),
            physical_starts,  # Return indices for priority updates
            torch.ones(batch_size, device=self.device),  # Uniform weights
        )
    
    def update_priorities(self, indices, priorities):
        pass  # No-op for uniform buffer


class PrioritizedSequenceReplayBuffer:
    """Prioritized Experience Replay buffer with sequence support (Official BBF style)."""
    def __init__(self, capacity, obs_shape, device, max_seq_len, alpha=0.5, beta=0.5, eps=1e-6):
        self.capacity = capacity
        self.device = device
        self.max_seq_len = max_seq_len
        # Note: alpha/beta kept for API compatibility but not used (official BBF doesn't use them)
        self.eps = eps
        
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity), dtype=np.int64)
        self.rewards = np.zeros((capacity), dtype=np.float32)
        self.dones = np.zeros((capacity), dtype=np.bool_)
        self.tree = SumTree(capacity)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, done, priority=None):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        
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
        
        # Stratified sampling for better coverage
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = np.random.uniform(low, high)
            idx = self.tree.get(s)
            
            # Ensure valid sequence start (not too close to current write position)
            # and handle circular buffer edge cases
            cursor = self.pos if self.size == self.capacity else 0
            max_valid = self.size - seq_len
            
            # Retry if invalid index
            attempts = 0
            while attempts < 10:
                if self.size == self.capacity:
                    # Full buffer: check we don't cross the write cursor
                    logical_idx = (idx - cursor) % self.capacity
                    if logical_idx <= max_valid:
                        break
                else:
                    # Not full: just check bounds
                    if idx <= max_valid:
                        break
                s = np.random.uniform(low, high)
                idx = self.tree.get(s)
                attempts += 1
            
            indices[i] = idx
            priorities[i] = self.tree[idx]
        
        # Build sequences from start indices
        seq_indices = (indices[:, None] + np.arange(seq_len)) % self.capacity
        
        # Compute importance sampling weights (Official BBF: 1/sqrt(prob), no beta)
        probs = priorities / (self.tree.total + 1e-10)
        weights = 1.0 / np.sqrt(probs + 1e-10)
        weights = weights / (weights.max() + 1e-10)  # Normalize
        
        return (
            torch.as_tensor(self.obs[seq_indices], device=self.device),
            torch.as_tensor(self.actions[seq_indices], device=self.device),
            torch.as_tensor(self.rewards[seq_indices], device=self.device),
            torch.as_tensor(self.dones[seq_indices], device=self.device),
            indices,  # Return indices for priority updates
            torch.as_tensor(weights, device=self.device, dtype=torch.float32),
        )
    
    def update_priorities(self, indices, losses):
        """Update priorities based on losses (Official BBF: sqrt(loss), no alpha)."""
        priorities = np.sqrt(losses + self.eps)
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)

def get_linear_schedule(start, end, t, duration):
    if t >= duration: return end
    return start + (end - start) * (t / duration)

def get_exponential_schedule(start, end, t, duration):
    """Exponential interpolation in log-space (BBF paper specification)."""
    if t >= duration:
        return end
    # Interpolate in log-space: exp(log(start) + alpha * (log(end) - log(start)))
    return start * (end / start) ** (t / duration)

def hard_reset(model, optimizer, args):
    """
    BBF Reset Strategy (from official implementation):
    - Shrink-perturb ONLY: encoder, transition model (trans_fc, action_emb)
    - Fully reset: heads (value, advantage), projection, predictor, rep_head
    """
    with torch.no_grad():
        fresh = type(model)(model.env_ref, args).to(model.device_ref)
        alpha = args.shrink_factor
        
        for (name, p), (_, p_new) in zip(model.named_parameters(), fresh.named_parameters()):
            # Shrink-perturb only encoder and transition model
            if "encoder" in name or "trans_fc" in name or "action_emb" in name:
                p.data.copy_(alpha * p.data + (1.0 - alpha) * p_new.data)
            else:
                # Fully reset: heads, projection, predictor, rep_head
                p.data.copy_(p_new.data)

            # Always clear optimizer state (momentum etc.) for ALL params
            if p in optimizer.state:
                del optimizer.state[p]

def train_batch(q_network, target_network, optimizer, rb, aug, args, curr_gamma, curr_n, timer=None, scaler=None):
    """Encapsulated training logic for main loop and warmup."""
    device = next(q_network.parameters()).device
    use_amp = args.use_amp and device.type == 'cuda'

    # --- SAMPLE FROM BUFFER ---
    if timer: timer.start("1_buffer_sample")
    fetch_len = args.jumps + 1 + curr_n
    data_obs, data_act, data_rew, data_done, indices, weights = rb.sample(args.batch_size, fetch_len)
    if timer: timer.stop()

    # --- FORWARD PASS WITH MIXED PRECISION ---
    # AMP Strategy: FP16 for convolutions (fast), FP32 for C51/SPR (accurate)
    with autocast(device_type='cuda', enabled=use_amp):
        # --- AUGMENTATION (obs_0) ---
        if timer: timer.start("2_augment_obs0")
        obs_0 = aug(data_obs[:, 0])
        if timer: timer.stop()

        # --- ENCODE (online) - Keep FP16 for speed ---
        if timer: timer.start("3_encode_online")
        z0 = q_network.encode(obs_0)
        if timer: timer.stop()

        # --- SPR ROLLOUT ---
        if timer: timer.start("4_spr_rollout")
        rollout_acts = data_act[:, :args.jumps]
        z_seq = q_network.rollout_latents(z0, rollout_acts)
        if timer: timer.stop()

    # --- SPR TARGETS (target network) - FORCE FP32 for normalization ---
    if timer: timer.start("5_spr_targets")
    with torch.no_grad(), autocast(device_type='cuda', enabled=False):
        # Stack obs for jumps 1..jumps: (batch, jumps, 4, 84, 84) -> (batch*jumps, 4, 84, 84)
        obs_jumps = data_obs[:, 1:args.jumps+1]
        B, J = obs_jumps.shape[0], obs_jumps.shape[1]
        obs_jumps_flat = obs_jumps.reshape(-1, *obs_jumps.shape[2:])

        # Augmentation + encode in FP16 (convolutions are fast)
        with autocast(device_type='cuda', enabled=use_amp):
            obs_jumps_aug = aug(obs_jumps_flat)
            t_z_flat = target_network.encode(obs_jumps_aug)

        # Projection + normalization in FP32 (critical for numerical stability)
        t_z_flat = t_z_flat.float()
        t_p_flat = target_network.projection(t_z_flat)
        t_p_norm = F.normalize(t_p_flat, dim=1)

        # Reshape: (batch*jumps, proj_dim) -> (batch, jumps, proj_dim)
        target_latents = t_p_norm.view(B, J, -1)
    if timer: timer.stop()

    # --- SPR LOSS - FORCE FP32 for normalization ---
    if timer: timer.start("6_spr_loss")
    with autocast(device_type='cuda', enabled=False):
        # Get predicted latents for all jumps: z_seq[:, 1:] has shape (batch, jumps, latent_dim)
        z_jumps = z_seq[:, 1:args.jumps+1].float()  # Force FP32
        z_jumps_flat = z_jumps.reshape(-1, z_jumps.shape[-1])

        # Projection + predictor + normalization in FP32
        p_flat = q_network.projection(z_jumps_flat)
        pred_flat = q_network.predictor(p_flat)
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
    if timer: timer.stop()

    # --- C51 TARGET COMPUTATION - FORCE FP32 for distributional RL ---
    if timer: timer.start("7_c51_target")
    avg_q = 0.0
    with torch.no_grad(), autocast(device_type='cuda', enabled=False):
        gammas = torch.pow(curr_gamma, torch.arange(curr_n, device=data_obs.device))
        not_done = 1.0 - data_done.float()
        mask_seq = torch.cat([torch.ones((args.batch_size, 1), device=data_obs.device), not_done[:, :-1]], dim=1)
        reward_mask = torch.cumprod(mask_seq, dim=1)[:, :curr_n]
        n_step_rew = torch.sum(data_rew[:, :curr_n] * gammas * reward_mask, dim=1)

        obs_n = data_obs[:, curr_n]
        bootstrap_mask = torch.prod(not_done[:, :curr_n], dim=1)

        # Forward passes in FP16 (fast), then convert to FP32 for distribution ops
        with autocast(device_type='cuda', enabled=use_amp):
            next_dist = target_network(obs_n)
            online_n_dist = q_network(obs_n)

        # All distribution operations in FP32
        next_dist = next_dist.float()
        online_n_dist = online_n_dist.float()
        next_sup = target_network.support.float()

        # Compute Q-values and best action in FP32
        avg_q = (online_n_dist * q_network.support.float()).sum(2).mean().item()
        best_act = (online_n_dist * next_sup).sum(2).argmax(1)
        next_pmf = next_dist[torch.arange(args.batch_size), best_act]

        # C51 categorical projection in FP32 (critical!)
        gamma_n = curr_gamma ** curr_n
        Tz = n_step_rew.unsqueeze(1) + bootstrap_mask.unsqueeze(1) * gamma_n * next_sup.unsqueeze(0)
        Tz = Tz.clamp(min=args.v_min, max=args.v_max)
        b = (Tz - args.v_min) / q_network.delta_z
        l = b.floor().clamp(0, args.n_atoms - 1).long()
        u = b.ceil().clamp(0, args.n_atoms - 1).long()

        target_pmf = torch.zeros_like(next_pmf)
        offset = torch.linspace(0, (args.batch_size - 1) * args.n_atoms, args.batch_size, device=data_obs.device).long().unsqueeze(1)
        target_pmf.view(-1).index_add_(0, (l + offset).view(-1), (next_pmf * (u.float() - b)).view(-1))
        target_pmf.view(-1).index_add_(0, (u + offset).view(-1), (next_pmf * (b - l.float())).view(-1))
    if timer: timer.stop()

    # --- C51 LOSS - FORCE FP32 ---
    if timer: timer.start("8_c51_loss")
    with autocast(device_type='cuda', enabled=False):
        z0_fp32 = z0.float()
        curr_dist = q_network.get_dist(z0_fp32).float()
        log_p = torch.log(curr_dist[torch.arange(args.batch_size), data_act[:, 0]] + 1e-8)

        # Per-sample C51 loss (for priority updates)
        c51_loss_per_sample = -torch.sum(target_pmf * log_p, dim=1)

        # Apply importance sampling weights (PER correction)
        weighted_c51_loss = (c51_loss_per_sample * weights).mean()
        total_loss = weighted_c51_loss + args.spr_weight * spr_loss
    if timer: timer.stop()
    
    # --- BACKWARD (with gradient scaling for AMP) ---
    if timer: timer.start("9_backward")
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()
    if timer: timer.stop()
    
    # --- OPTIMIZER STEP (with gradient scaling for AMP) ---
    if timer: timer.start("10_optimizer_step")
    if scaler is not None:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
        optimizer.step()
    if timer: timer.stop()
    
    # --- TARGET UPDATE ---
    if timer: timer.start("11_target_update")
    for param, target_param in zip(q_network.parameters(), target_network.parameters()):
        target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
    if timer: timer.stop()
        
    # --- PRIORITY UPDATE ---
    if timer: timer.start("12_priority_update")
    losses = c51_loss_per_sample.detach().cpu().numpy()
    rb.update_priorities(indices, losses)
    if timer: timer.stop()
        
    return total_loss, spr_loss, avg_q, weights

if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "Simple SequenceBuffer only supports num_envs=1"
    
    run_name = f"{args.env_id.split('/')[-1]}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Log configuration to TensorBoard
    config_str = f"""
## Run Configuration
- **Replay Buffer**: {'PER (Prioritized)' if args.use_per else 'Uniform'}
- **Replay Ratio**: {args.replay_ratio}
- **Seed**: {args.seed}
- **Environment**: {args.env_id}
- **Reset Interval**: {args.reset_interval}
- **Learning Rate**: {args.learning_rate}
- **Weight Decay**: {args.weight_decay}
- **Batch Size**: {args.batch_size}
- **Buffer Size**: {args.buffer_size}
- **SPR Weight**: {args.spr_weight}
- **Jumps**: {args.jumps}
- **Initial γ**: {args.initial_gamma} → **Final γ**: {args.final_gamma}
- **Initial n-step**: {args.initial_n_step} → **Final n-step**: {args.final_n_step}
"""
    writer.add_text("config/summary", config_str, 0)
    
    # Also log as hparams for TensorBoard comparison
    writer.add_hparams(
        {
            "replay_buffer": "PER" if args.use_per else "Uniform",
            "replay_ratio": args.replay_ratio,
            "seed": args.seed,
            "reset_interval": args.reset_interval,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        {},  # Empty metrics dict - will be filled at end
        run_name=f"hparams/{run_name}"
    )
    
    print(f"\n{'='*60}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Replay Buffer: {'PER (Prioritized)' if args.use_per else 'Uniform'}")
    print(f"  Replay Ratio:  {args.replay_ratio}")
    print(f"  Seed:          {args.seed}")
    print(f"  Environment:   {args.env_id}")
    print(f"  Reset Interval:{args.reset_interval}")
    print(f"{'='*60}\n")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic  # Enable cuDNN autotuning for speed

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    
    q_network = BBFNetwork(envs, args).to(device)
    q_network.env_ref = envs
    q_network.device_ref = device
    target_network = BBFNetwork(envs, args).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # Weight decay excluding biases (official BBF excludes 1D params)
    decay_params = []
    no_decay_params = []
    for name, param in q_network.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=args.learning_rate, eps=1.5e-4)
    aug = RandomShift(pad=4).to(device)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler('cuda', enabled=args.use_amp and device.type == 'cuda')
    if args.use_amp and device.type == 'cuda':
        print("*** MIXED PRECISION (AMP) ENABLED - Using FP16 for faster training ***")
    
    # Initialize profiling timer
    timer = Timer(enabled=args.profile, sync_cuda=True)
    if args.profile:
        print("*** PROFILING ENABLED - This will add overhead due to CUDA synchronization ***")
    
    seq_req = args.jumps + 1 + args.initial_n_step
    if args.use_per:
        print("Using Prioritized Experience Replay (PER)")
        rb = PrioritizedSequenceReplayBuffer(
            args.buffer_size,
            envs.single_observation_space.shape,
            device,
            seq_req,
            alpha=args.per_alpha,
            beta=args.per_beta,
            eps=args.per_eps,
        )
    else:
        print("Using Uniform Experience Replay")
        rb = SequenceReplayBuffer(args.buffer_size, envs.single_observation_space.shape, device, seq_req)

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    
    grad_step = 0
    steps_since_reset = 0
    all_returns = []  # Track all episode returns for final summary
    
    # Best model checkpointing
    best_return = float('-inf')
    best_model_path = f"runs/{run_name}/{args.exp_name}_best.cleanrl_model"
    
    # Accurate SPS tracking (excludes warmup)
    training_start_time = None
    training_start_step = 0
    
    print("Starting Training Loop...")
    
    # PyTorch Profiler setup
    torch_profiler = None
    torch_profile_step = 0
    if args.torch_profile:
        print(f"*** TORCH PROFILER ENABLED ***")
        print(f"    Will skip warmup steps and profile {args.torch_profile_steps} training gradient steps")
        print(f"    Output: runs/{run_name}/torch_profiler/")

        # Calculate wait steps: we need to wait for learning_starts, then do warmup
        # Since step() is called once per gradient update (replay_ratio times per env step)
        # After learning_starts, we'll have done 0 gradient steps
        wait_steps = 10  # Wait a few gradient steps after learning_starts
        warmup_steps = 5  # Then warmup for a few steps

        torch_profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=args.torch_profile_steps,
                repeat=1
            ),
            on_trace_ready=tensorboard_trace_handler(f"runs/{run_name}/torch_profiler"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False  # Disable stack tracking to avoid state issues
        )
        torch_profiler.__enter__()

    for global_step in range(args.total_timesteps):
        
        if global_step <= args.learning_starts:
            if global_step % 100 == 0:
                print(f"Warmup: Filling Replay Buffer {global_step}/{args.learning_starts}")
        
        epsilon = get_linear_schedule(args.start_e, args.end_e, global_step, args.exploration_fraction * args.total_timesteps)

        # --- ACTION SELECTION ---
        timer.start("env_action_select")
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                q_dist = target_network(torch.Tensor(obs).to(device))
                q_values = torch.sum(q_dist * target_network.support, dim=2)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
        timer.stop()

        # --- ENV STEP ---
        timer.start("env_step")
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        timer.stop()
        
        # LOGGING (ROBUST)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    # Handle both scalar and array returns (NumPy 1.25+ compatibility)
                    ep_return_raw = info['episode']['r']
                    ep_return = float(ep_return_raw.item() if hasattr(ep_return_raw, 'item') else ep_return_raw)
                    ep_length_raw = info['episode']['l']
                    ep_length = int(ep_length_raw.item() if hasattr(ep_length_raw, 'item') else ep_length_raw)

                    all_returns.append(ep_return)
                    print(f"--> Episode Done. Step={global_step}, Return={ep_return:.0f}, Length={ep_length}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)

                    # Save best model
                    if ep_return > best_return:
                        best_return = ep_return
                        torch.save(q_network.state_dict(), best_model_path)
                        print(f"    ★ New best! Saved model with return {best_return:.0f}")
                        writer.add_scalar("charts/best_return", best_return, global_step)
        elif "episode" in infos:
            # Handle both scalar and array returns (NumPy 1.25+ compatibility)
            ep_return_raw = infos['episode']['r']
            ep_return = float(ep_return_raw.item() if hasattr(ep_return_raw, 'item') else ep_return_raw)
            ep_length_raw = infos['episode']['l']
            ep_length = int(ep_length_raw.item() if hasattr(ep_length_raw, 'item') else ep_length_raw)

            all_returns.append(ep_return)
            print(f"--> Episode Done. Step={global_step}, Return={ep_return:.0f}, Length={ep_length}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)

            # Save best model
            if ep_return > best_return:
                best_return = ep_return
                torch.save(q_network.state_dict(), best_model_path)
                print(f"    ★ New best! Saved model with return {best_return:.0f}")
                writer.add_scalar("charts/best_return", best_return, global_step)

        # --- BUFFER ADD ---
        timer.start("env_buffer_add")
        for idx in range(args.num_envs):
            done_flag = terminations[idx] or truncations[idx]
            rb.add(obs[idx], actions[idx], rewards[idx], done_flag)
        timer.stop()
            
        obs = next_obs

        if global_step > args.learning_starts:
            # Initialize training timer once (right when training starts)
            if training_start_time is None:
                training_start_time = time.time()
                training_start_step = global_step
                print(f"*** Training started at step {global_step} ***")
            
            # --- RESET CHECK & WARMUP ---
            # Reset based on GRAD STEPS (as per BBF paper: every 40k gradient steps)
            if grad_step > 0 and grad_step % args.reset_interval == 0:
                print(f"*** HARD RESET at grad step {grad_step} (env step {global_step}) ***")
                hard_reset(q_network, optimizer, args)
                target_network.load_state_dict(q_network.state_dict())
                steps_since_reset = 0
                
                print(f"*** Offline Warmup ({args.reset_warmup_steps} steps) ***")
                for _ in range(args.reset_warmup_steps):
                    grad_step += 1
                    _ = train_batch(q_network, target_network, optimizer, rb, aug, args, args.initial_gamma, args.initial_n_step, timer, scaler)

            # --- STANDARD TRAINING LOOP ---
            for _ in range(args.replay_ratio):
                grad_step += 1
                steps_since_reset += 1
                
                curr_gamma = get_exponential_schedule(args.initial_gamma, args.final_gamma, steps_since_reset, args.anneal_duration)
                curr_n = int(round(get_exponential_schedule(args.initial_n_step, args.final_n_step, steps_since_reset, args.anneal_duration)))
                
                total_loss, spr_loss, avg_q, is_weights = train_batch(q_network, target_network, optimizer, rb, aug, args, curr_gamma, curr_n, timer, scaler)
                
                # PyTorch Profiler step (must be called consistently for state machine)
                if torch_profiler is not None:
                    torch_profile_step += 1
                    try:
                        torch_profiler.step()
                        # Auto-stops after schedule completes (wait + warmup + active)
                        # Add buffer to ensure schedule completes
                        if torch_profile_step >= (10 + 5 + args.torch_profile_steps + 10):
                            torch_profiler.__exit__(None, None, None)
                            print(f"\n*** TORCH PROFILER COMPLETE - Captured {args.torch_profile_steps} active steps ***")
                            print(f"    View with: tensorboard --logdir=runs/{run_name}/torch_profiler")
                            torch_profiler = None
                    except Exception as e:
                        print(f"\n*** TORCH PROFILER ERROR: {e} ***")
                        print("    Disabling profiler and continuing training...")
                        try:
                            torch_profiler.__exit__(None, None, None)
                        except:
                            pass
                        torch_profiler = None

                if grad_step % 100 == 0:
                    # SPS based on training time only (excludes warmup/setup)
                    training_elapsed = time.time() - training_start_time
                    training_steps = global_step - training_start_step
                    # Require at least 1 second elapsed to avoid division issues
                    if training_elapsed > 1.0:
                        sps = int(training_steps / training_elapsed)
                        remaining_steps = args.total_timesteps - global_step
                        eta_min = (remaining_steps / sps) / 60 if sps > 0 else 0
                    else:
                        sps = 0
                        eta_min = 0
                    print(f"Step: {global_step} | Grad: {grad_step} | Loss: {total_loss.item():.3f} | SPR: {spr_loss.item():.3f} | AvgQ: {avg_q:.2f} | Eps: {epsilon:.2f} | SPS: {sps} | ETA: {eta_min:.1f} min")

                    writer.add_scalar("losses/total_loss", total_loss.item(), grad_step)
                    writer.add_scalar("losses/spr_loss", spr_loss.item(), grad_step)
                    writer.add_scalar("charts/n_step", curr_n, grad_step)
                    writer.add_scalar("charts/gamma", curr_gamma, grad_step)
                    writer.add_scalar("charts/avg_q", avg_q, grad_step)
                    writer.add_scalar("charts/SPS", sps, grad_step)

                    # Log IS weights (useful for confirming PER is active)
                    if args.use_per:
                        writer.add_scalar("losses/is_weight_mean", is_weights.mean().item(), grad_step)
                        writer.add_scalar("losses/is_weight_std", is_weights.std().item(), grad_step)

                    # AMP diagnostics: Monitor gradient norm and loss scale
                    if args.use_amp:
                        writer.add_scalar("debug/amp_scale", scaler.get_scale(), grad_step)
                        # Compute gradient norm (unscaled)
                        total_norm = 0.0
                        for p in q_network.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        writer.add_scalar("debug/grad_norm", total_norm, grad_step)

    # SAVE MODEL AT THE END
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # PROFILING REPORT
    if args.profile:
        timer.report()
    
    # Cleanup torch profiler if still active
    if torch_profiler is not None:
        try:
            torch_profiler.__exit__(None, None, None)
            print(f"\n*** TORCH PROFILER COMPLETE - {torch_profile_step} steps captured ***")
            print(f"    View with: tensorboard --logdir=runs/{run_name}/torch_profiler")
        except Exception as e:
            print(f"\n*** TORCH PROFILER CLEANUP WARNING: {e} ***")

    # FINAL SUMMARY
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Run Name:      {run_name}")
    print(f"  Replay Buffer: {'PER (Prioritized)' if args.use_per else 'Uniform'}")
    print(f"  Replay Ratio:  {args.replay_ratio}")
    print(f"  Seed:          {args.seed}")
    print(f"  Environment:   {args.env_id}")
    print(f"  Total Time:    {total_time/60:.1f} minutes")
    print(f"  Grad Steps:    {grad_step}")
    if all_returns:
        print(f"  Episodes:      {len(all_returns)}")
        print(f"  Max Return:    {max(all_returns):.0f}")
        print(f"  Mean Return:   {np.mean(all_returns):.1f}")
        print(f"  Last 5:        {[int(r) for r in all_returns[-5:]]}")
    if best_return > float('-inf'):
        print(f"  Best Model:    {best_model_path}")
        print(f"  Best Return:   {best_return:.0f}")
    print(f"{'='*60}\n")
    
    # Log final metrics to TensorBoard
    if all_returns:
        writer.add_scalar("final/max_return", max(all_returns), 0)
        writer.add_scalar("final/mean_return", np.mean(all_returns), 0)
        writer.add_scalar("final/episodes", len(all_returns), 0)
        writer.add_scalar("final/total_time_minutes", total_time/60, 0)

    envs.close()
    writer.close()