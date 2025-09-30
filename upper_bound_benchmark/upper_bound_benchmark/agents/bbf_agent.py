# coding=utf-8
# BBF Agent - PyTorch implementation
# Based on: Schwarzer et al. "Bigger, Better, Faster: Human-level Atari with human-level efficiency" (2023)
# Official code: https://github.com/google-research/google-research/tree/master/bigger_better_faster

import collections
import copy
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import Agent
from .bbf_networks import BBFNetwork


class MetricsTracker:
    """Tracks and logs training metrics to CSV file."""
    
    def __init__(self, output_dir: str, log_interval: int = 1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.metrics_file = self.output_dir / "training_metrics.csv"
        self.episode_file = self.output_dir / "episode_metrics.csv"
        
        # Initialize CSV files with headers
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'env_steps', 'learn_steps', 'td_loss', 'spr_loss', 'total_loss',
                'mean_q', 'max_q', 'gamma', 'n_step', 'epsilon', 'time_elapsed'
            ])
        
        with open(self.episode_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['env_steps', 'episode_num', 'episode_return', 'episode_length'])
        
        self.start_time = time.time()
        self.episode_count = 0
        
    def log_training_step(self, metrics: dict):
        """Log training step metrics to CSV."""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('env_steps', 0),
                metrics.get('learn_steps', 0),
                f"{metrics.get('td_loss', 0):.4f}",
                f"{metrics.get('spr_loss', 0):.4f}",
                f"{metrics.get('total_loss', 0):.4f}",
                f"{metrics.get('mean_q', 0):.3f}",
                f"{metrics.get('max_q', 0):.3f}",
                f"{metrics.get('gamma', 0):.4f}",
                metrics.get('n_step', 0),
                f"{metrics.get('epsilon', 0):.4f}",
                f"{time.time() - self.start_time:.1f}"
            ])
    
    def log_episode(self, env_steps: int, episode_return: float, episode_length: int):
        """Log episode metrics to CSV."""
        self.episode_count += 1
        with open(self.episode_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([env_steps, self.episode_count, f"{episode_return:.2f}", episode_length])


@dataclass
class BBFConfig:
    """BBF hyperparameters matching the paper's configuration."""
    # Network architecture
    frame_stack: int = 4
    width_scale: int = 4  # BBF uses 4x width scaling
    hidden_dim: int = 2048
    num_atoms: int = 51  # C51 distributional RL
    v_min: float = -10.0
    v_max: float = 10.0
    
    # Logging
    log_interval: int = 1000  # Log metrics every N env steps
    verbose: bool = True  # Print to console
    
    # Learning
    learning_rate: float = 1e-4
    encoder_learning_rate: float = 1e-4
    weight_decay: float = 0.1  # AdamW weight decay
    gamma_min: float = 0.97  # Starting gamma (anneals to gamma_max)
    gamma_max: float = 0.997  # Final gamma
    update_horizon_min: int = 3  # Starting n-step (anneals from max to min)
    update_horizon_max: int = 10  # Max n-step
    
    # Replay and training
    buffer_capacity: int = 200_000
    batch_size: int = 32
    replay_ratio: int = 2  # Gradient updates per env step (2 for RR=2, 8 for RR=8)
    learning_starts: int = 2_000  # min_replay_history from config
    
    # Target network (EMA)
    target_update_tau: float = 0.005  # EMA coefficient for target network
    target_action_selection: bool = True  # Use target network for action selection
    
    # Resets (SR-SPR style)
    reset_every: int = 20_000  # Reset every N steps (gradient steps)
    shrink_factor: float = 0.5  # BBF uses harder resets
    perturb_factor: float = 0.5
    cycle_steps: int = 10_000  # Annealing happens over this many steps after each reset
    no_resets_after: int = 100_000  # Stop resetting after this many steps
    
    # SPR self-supervised learning
    spr_weight: float = 5.0  # Weight for SPR loss
    jumps: int = 5  # Number of future steps to predict
    
    # Data augmentation
    data_augmentation: bool = True
    aug_prob: float = 1.0  # Probability of applying augmentation
    
    # Prioritized replay
    replay_scheme: str = 'prioritized'  # 'uniform' or 'prioritized'
    priority_exponent: float = 0.6  # Alpha in prioritized replay
    importance_sampling_exponent: float = 0.4  # Beta in prioritized replay
    
    # Architecture flags
    dueling: bool = True
    double_dqn: bool = True
    distributional: bool = True
    
    # Epsilon greedy (BBF uses epsilon=0 with no noisy nets)
    eps_start: float = 0.0
    eps_end: float = 0.0
    eps_decay: int = 1
    
    # Output directory for metrics
    output_dir: Optional[str] = None


def random_crop_aug(images, pad=4):
    """Random crop data augmentation (DrQ style).
    
    Args:
        images: Batch of images (B, C, H, W)
        pad: Padding size
        
    Returns:
        Augmented images
    """
    batch_size, channels, height, width = images.shape
    assert height == width, "Only square images supported"
    
    # Pad
    padded = F.pad(images, (pad, pad, pad, pad), mode='replicate')
    
    # Random crop back to original size
    crop_max = height + 2 * pad - height
    w_crop = np.random.randint(0, crop_max + 1)
    h_crop = np.random.randint(0, crop_max + 1)
    
    cropped = padded[:, :, h_crop:h_crop + height, w_crop:w_crop + width]
    
    # Intensity augmentation
    noise_scale = 0.05
    noise = 1.0 + (noise_scale * torch.clamp(torch.randn((batch_size, 1, 1, 1), device=images.device), -2.0, 2.0))
    augmented = cropped * noise
    
    return augmented


class PrioritizedReplayBuffer:
    """Prioritized replay buffer for subsequence sampling (for SPR).
    
    Stores sequences of transitions to enable multi-step returns and SPR rollouts.
    """
    
    def __init__(self, capacity, state_shape, jumps, seed):
        self.capacity = capacity
        self.state_shape = state_shape
        self.jumps = jumps
        self.sequence_length = jumps + 1  # We need jumps+1 frames for jumps predictions
        
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        
        self.priorities = np.ones((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)
        
    def add(self, state, action, reward, done):
        """Add a single transition."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        
        # New transitions get max priority
        if self.size > 0:
            self.priorities[self.pos] = self.priorities[:self.size].max()
        else:
            self.priorities[self.pos] = 1.0
            
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size, priority_exponent, importance_sampling_exponent):
        """Sample a batch of sequences.
        
        Returns:
            Dictionary with states, actions, rewards, dones, and importance weights
        """
        # Can't sample sequences that cross episode boundaries or buffer wrap
        valid_indices = []
        for i in range(self.size - self.sequence_length):
            # Check if sequence is valid (no done flags except possibly the last)
            if not np.any(self.dones[i:i+self.sequence_length-1]):
                valid_indices.append(i)
                
        if len(valid_indices) < batch_size:
            return None
            
        valid_indices = np.array(valid_indices)
        
        # Compute sampling probabilities from priorities
        priorities = self.priorities[valid_indices] ** priority_exponent
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = self.rng.choice(valid_indices, size=batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (self.size * probs[np.searchsorted(valid_indices, indices)]) ** (-importance_sampling_exponent)
        weights = weights / weights.max()  # Normalize
        
        # Extract sequences
        states_seq = []
        actions_seq = []
        rewards_seq = []
        dones_seq = []
        
        for idx in indices:
            states_seq.append(self.states[idx:idx+self.sequence_length])
            actions_seq.append(self.actions[idx:idx+self.sequence_length])
            rewards_seq.append(self.rewards[idx:idx+self.sequence_length])
            dones_seq.append(self.dones[idx:idx+self.sequence_length])
            
        batch = {
            'states': np.array(states_seq),  # (B, T, C, H, W)
            'actions': np.array(actions_seq),  # (B, T)
            'rewards': np.array(rewards_seq),  # (B, T)
            'dones': np.array(dones_seq),  # (B, T)
            'weights': weights.astype(np.float32),  # (B,)
            'indices': indices,  # For updating priorities
        }
        
        return batch
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions."""
        self.priorities[indices] = priorities
        
    def __len__(self):
        return self.size


class BBFAgent(Agent):
    """BBF agent implementation in PyTorch.
    
    Implements all key components from the BBF paper:
    - Impala CNN with 4x width scaling
    - Periodic shrink-and-perturb resets
    - Annealing n-step and gamma
    - EMA target network
    - SPR self-supervised learning
    - AdamW with weight decay
    - C51 distributional RL
    - Prioritized replay
    """
    
    def __init__(
        self,
        seed: int = 0,
        config: Optional[BBFConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(seed=seed)
        self.config = config or BBFConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net: Optional[BBFNetwork] = None
        self.target_net: Optional[BBFNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.replay: Optional[PrioritizedReplayBuffer] = None
        self.num_actions: Optional[int] = None
        
        self.total_steps = 0  # Environment steps
        self.learn_steps = 0  # Gradient steps
        self.last_reset_step = 0
        
        self.frame_stack = self.config.frame_stack
        self._frame_stack = collections.deque(maxlen=self.frame_stack)
        self._current_state: Optional[np.ndarray] = None
        self._needs_reset = True
        
        # C51 support
        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.num_atoms,
            device=self.device
        )
        self.delta_z = (self.config.v_max - self.config.v_min) / (self.config.num_atoms - 1)
        
        # Metrics tracking
        self.metrics_tracker: Optional[MetricsTracker] = None
        if self.config.output_dir:
            self.metrics_tracker = MetricsTracker(self.config.output_dir, self.config.log_interval)
        
        # For tracking recent metrics
        self.recent_td_loss = []
        self.recent_spr_loss = []
        self.recent_q_values = []
        self.episode_length = 0
        self.episode_return = 0.0
        
    def begin_task(self, game_id, cycle, action_space):
        if self.num_actions is None or self.num_actions != action_space.n:
            self.num_actions = action_space.n
            self._build_networks()
        self._frame_stack.clear()
        self._current_state = None
        self._needs_reset = True
        
    def _build_networks(self):
        assert self.num_actions is not None
        
        # Policy network
        self.policy_net = BBFNetwork(
            in_channels=self.frame_stack,
            num_actions=self.num_actions,
            num_atoms=self.config.num_atoms,
            width_scale=self.config.width_scale,
            hidden_dim=self.config.hidden_dim,
            dueling=self.config.dueling,
            distributional=self.config.distributional,
        ).to(self.device)
        
        # Target network (EMA)
        self.target_net = BBFNetwork(
            in_channels=self.frame_stack,
            num_actions=self.num_actions,
            num_atoms=self.config.num_atoms,
            width_scale=self.config.width_scale,
            hidden_dim=self.config.hidden_dim,
            dueling=self.config.dueling,
            distributional=self.config.distributional,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer: AdamW with weight decay
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=0.00015,  # From BBF config
        )
        
        # Replay buffer
        state_shape = (self.frame_stack, 84, 84)
        self.replay = PrioritizedReplayBuffer(
            capacity=self.config.buffer_capacity,
            state_shape=state_shape,
            jumps=self.config.jumps,
            seed=int(self.rng.integers(1 << 30))
        )
        
    def _preprocess(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            obs = obs.get("pixel")
        obs_array = np.asarray(obs)
        frame = Image.fromarray(obs_array).convert("L").resize((84, 84), Image.BILINEAR)
        return np.array(frame, dtype=np.uint8)
        
    def _ensure_state_initialized(self, obs):
        if not self._needs_reset and self._current_state is not None:
            return
        processed = self._preprocess(obs)
        self._frame_stack.clear()
        for _ in range(self.frame_stack):
            self._frame_stack.append(processed)
        self._current_state = np.stack(self._frame_stack, axis=0)
        self._needs_reset = False
        
    def _append_next_state(self, obs) -> np.ndarray:
        processed = self._preprocess(obs)
        self._frame_stack.append(processed)
        return np.stack(self._frame_stack, axis=0)
        
    def act(self, obs, action_space) -> int:
        self._ensure_state_initialized(obs)
        assert self._current_state is not None
        
        # BBF uses epsilon=0 (no exploration noise, relies on training dynamics)
        epsilon = 0.0
        
        if self.rng.random() < epsilon:
            action = int(action_space.sample())
        else:
            state_tensor = torch.from_numpy(self._current_state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Use target network for action selection if specified
                net = self.target_net if self.config.target_action_selection else self.policy_net
                outputs = net(state_tensor / 255.0)
                
                if self.config.distributional:
                    # C51: compute Q-values from distribution
                    probs = F.softmax(outputs['q_logits'], dim=-1)
                    q_values = (probs * self.support).sum(dim=-1)
                else:
                    q_values = outputs['q_logits']
                    
                action = int(torch.argmax(q_values, dim=1).item())
                
        return action
        
    def observe(self, obs, action, reward, next_obs, terminated, truncated, info):
        if self.policy_net is None or self.replay is None:
            return
            
        self._ensure_state_initialized(obs)
        assert self._current_state is not None
        
        done = float(terminated or truncated)
        state = self._current_state.copy()
        
        self.episode_length += 1
        self.episode_return += reward
        
        if done:
            self._needs_reset = True
            self._frame_stack.clear()
            self._current_state = None
        else:
            next_state = self._append_next_state(next_obs)
            self._current_state = next_state
            
        # Add to replay buffer
        self.replay.add(state, action, float(reward), done)
        self.total_steps += 1
        
        # Train
        if self.total_steps >= self.config.learning_starts:
            for _ in range(self.config.replay_ratio):
                self._train_step()
        
        # Log metrics periodically
        if self.total_steps % self.config.log_interval == 0 and self.total_steps > 0:
            self._log_metrics()
                
    def end_episode(self, episode_return: float) -> None:
        # Log episode metrics
        if self.metrics_tracker:
            self.metrics_tracker.log_episode(self.total_steps, episode_return, self.episode_length)
        
        if self.config.verbose:
            print(f"  Episode finished: Return={episode_return:.1f}, Length={self.episode_length}, Steps={self.total_steps}")
        
        # Reset episode tracking
        self.episode_length = 0
        self.episode_return = 0.0
        
    def _get_current_gamma(self):
        """Get current gamma based on annealing schedule."""
        steps_since_reset = self.learn_steps - self.last_reset_step
        if steps_since_reset >= self.config.cycle_steps:
            return self.config.gamma_max
        # Exponential annealing from gamma_min to gamma_max
        progress = steps_since_reset / self.config.cycle_steps
        log_min = np.log(1 - self.config.gamma_min)
        log_max = np.log(1 - self.config.gamma_max)
        log_gamma = log_min + progress * (log_max - log_min)
        return 1 - np.exp(log_gamma)
        
    def _get_current_n_step(self):
        """Get current n-step based on annealing schedule."""
        steps_since_reset = self.learn_steps - self.last_reset_step
        if steps_since_reset >= self.config.cycle_steps:
            return self.config.update_horizon_min
        # Linear annealing from max to min
        progress = steps_since_reset / self.config.cycle_steps
        n = self.config.update_horizon_max - progress * (self.config.update_horizon_max - self.config.update_horizon_min)
        return int(np.round(n))
        
    def _should_reset(self):
        """Check if we should perform a network reset."""
        if self.learn_steps >= self.config.no_resets_after:
            return False
        if self.learn_steps - self.last_reset_step >= self.config.reset_every:
            return True
        return False
        
    def _perform_reset(self):
        """Perform shrink-and-perturb reset on encoder and transition model."""
        print(f"Performing reset at step {self.learn_steps}")
        
        # Create new random weights
        new_net = BBFNetwork(
            in_channels=self.frame_stack,
            num_actions=self.num_actions,
            num_atoms=self.config.num_atoms,
            width_scale=self.config.width_scale,
            hidden_dim=self.config.hidden_dim,
            dueling=self.config.dueling,
            distributional=self.config.distributional,
        ).to(self.device)
        
        # Interpolate encoder and transition_model parameters
        with torch.no_grad():
            for (name, param), (_, new_param) in zip(
                self.policy_net.named_parameters(),
                new_net.named_parameters()
            ):
                if 'encoder' in name or 'transition_model' in name:
                    # Shrink and perturb: interpolate between current and random
                    param.data = (
                        self.config.shrink_factor * param.data +
                        self.config.perturb_factor * new_param.data
                    )
        
        # Reset optimizer state for those parameters
        # (In practice, this is complex; JAX implementation handles this differently)
        
        self.last_reset_step = self.learn_steps
        
    def _train_step(self):
        if len(self.replay) < self.config.batch_size:
            return
            
        # Sample batch
        batch = self.replay.sample(
            self.config.batch_size,
            self.config.priority_exponent,
            self.config.importance_sampling_exponent
        )
        
        if batch is None:
            return
            
        # Check if reset is needed
        if self._should_reset():
            self._perform_reset()
            
        # Convert to tensors
        states = torch.from_numpy(batch['states']).float().to(self.device) / 255.0  # (B, T, C, H, W)
        actions = torch.from_numpy(batch['actions']).long().to(self.device)  # (B, T)
        rewards = torch.from_numpy(batch['rewards']).float().to(self.device)  # (B, T)
        dones = torch.from_numpy(batch['dones']).float().to(self.device)  # (B, T)
        weights = torch.from_numpy(batch['weights']).float().to(self.device)  # (B,)
        
        # Apply data augmentation
        if self.config.data_augmentation:
            batch_size, seq_len = states.shape[:2]
            states_flat = states.reshape(batch_size * seq_len, *states.shape[2:])
            states_flat = random_crop_aug(states_flat)
            states = states_flat.reshape(batch_size, seq_len, *states.shape[2:])
        
        # Get current n-step and gamma
        n_step = self._get_current_n_step()
        gamma = self._get_current_gamma()
        
        # Compute losses and Q-values for logging
        td_loss, spr_loss, priorities, mean_q = self._compute_loss(
            states, actions, rewards, dones, n_step, gamma
        )
        
        # Combined loss
        loss = td_loss + self.config.spr_weight * spr_loss
        loss = (loss * weights).mean()
        
        # Track metrics
        self.recent_td_loss.append(td_loss.mean().item())
        self.recent_spr_loss.append(spr_loss.mean().item())
        self.recent_q_values.append(mean_q)
        
        # Keep only recent values
        if len(self.recent_td_loss) > 100:
            self.recent_td_loss.pop(0)
            self.recent_spr_loss.pop(0)
            self.recent_q_values.pop(0)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        self.replay.update_priorities(batch['indices'], priorities.detach().cpu().numpy())
        
        # Update target network (EMA)
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1 - self.config.target_update_tau)
                target_param.data.add_(self.config.target_update_tau * policy_param.data)
        
        self.learn_steps += 1
        
    def _compute_loss(self, states, actions, rewards, dones, n_step, gamma):
        """Compute TD loss and SPR loss.
        
        Args:
            states: (B, T, C, H, W)
            actions: (B, T)
            rewards: (B, T)
            dones: (B, T)
            n_step: Current n-step value
            gamma: Current discount factor
            
        Returns:
            td_loss, spr_loss, priorities
        """
        batch_size = states.shape[0]
        
        # Current state and action
        current_states = states[:, 0]  # (B, C, H, W)
        current_actions = actions[:, 0]  # (B,)
        
        # Compute n-step return
        n_step = min(n_step, states.shape[1] - 1)
        n_step_rewards = torch.zeros(batch_size, device=self.device)
        gamma_n = 1.0
        
        for t in range(n_step):
            n_step_rewards += gamma_n * rewards[:, t]
            gamma_n *= gamma
            if torch.any(dones[:, t]):
                break
                
        # Next state for n-step backup
        next_states = states[:, min(n_step, states.shape[1]-1)]  # (B, C, H, W)
        next_dones = dones[:, min(n_step-1, states.shape[1]-2)]  # (B,)
        
        # Forward pass
        outputs = self.policy_net(current_states, return_latent=True, return_projection=True)
        q_logits = outputs['q_logits']  # (B, A, num_atoms)
        latent = outputs['latent']  # (B, latent_dim)
        
        # Get Q-values for current actions
        if self.config.distributional:
            current_q_logits = q_logits[range(batch_size), current_actions]  # (B, num_atoms)
        else:
            current_q_values = q_logits[range(batch_size), current_actions]  # (B,)
            
        # Target Q-values
        with torch.no_grad():
            target_outputs = self.target_net(next_states)
            target_q_logits = target_outputs['q_logits']  # (B, A, num_atoms)
            
            if self.config.double_dqn:
                # Double DQN: select action with online network
                online_outputs = self.policy_net(next_states)
                if self.config.distributional:
                    online_probs = F.softmax(online_outputs['q_logits'], dim=-1)
                    online_q = (online_probs * self.support).sum(dim=-1)
                else:
                    online_q = online_outputs['q_logits']
                next_actions = online_q.argmax(dim=1)
            else:
                if self.config.distributional:
                    target_probs = F.softmax(target_q_logits, dim=-1)
                    target_q = (target_probs * self.support).sum(dim=-1)
                else:
                    target_q = target_q_logits
                next_actions = target_q.argmax(dim=1)
                
            if self.config.distributional:
                # C51: project distribution
                target_q_logits_next = target_q_logits[range(batch_size), next_actions]  # (B, num_atoms)
                target_probs = F.softmax(target_q_logits_next, dim=-1)
                
                # Compute projected distribution
                tz = n_step_rewards.unsqueeze(1) + (1 - next_dones.unsqueeze(1)) * (gamma ** n_step) * self.support
                tz = tz.clamp(self.config.v_min, self.config.v_max)
                b = (tz - self.config.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()
                
                # Distribute probability
                m = torch.zeros_like(target_probs)
                for i in range(batch_size):
                    for j in range(self.config.num_atoms):
                        m[i, l[i, j]] += target_probs[i, j] * (u[i, j] - b[i, j])
                        m[i, u[i, j]] += target_probs[i, j] * (b[i, j] - l[i, j])
                        
                # Cross-entropy loss
                td_loss = -(m * F.log_softmax(current_q_logits, dim=-1)).sum(dim=-1)
                priorities = td_loss + 1e-6
            else:
                # Standard DQN
                target_q_values = target_q_logits[range(batch_size), next_actions]
                target = n_step_rewards + (1 - next_dones) * (gamma ** n_step) * target_q_values
                td_loss = F.smooth_l1_loss(current_q_values, target, reduction='none')
                priorities = td_loss.abs() + 1e-6
                
        # SPR loss
        spr_loss = self._compute_spr_loss(latent, states, actions)
        
        # Calculate mean Q-value for logging
        with torch.no_grad():
            if self.config.distributional:
                probs = F.softmax(q_logits, dim=-1)
                q_values = (probs * self.support).sum(dim=-1)
            else:
                q_values = q_logits
            mean_q = q_values.mean().item()
        
        return td_loss, spr_loss, priorities, mean_q
        
    def _compute_spr_loss(self, latent, states, actions):
        """Compute SPR self-supervised loss.
        
        Args:
            latent: Current latent state (B, latent_dim)
            states: Future states (B, T, C, H, W)
            actions: Actions taken (B, T)
            
        Returns:
            SPR loss
        """
        batch_size = states.shape[0]
        jumps = min(self.config.jumps, states.shape[1] - 1)
        
        if jumps == 0:
            return torch.tensor(0.0, device=self.device)
            
        # Rollout predictions
        predicted_projections = self.policy_net.spr_rollout(latent, actions[:, :jumps])  # (B, jumps, proj_dim)
        
        # Target projections from future states
        with torch.no_grad():
            future_states = states[:, 1:jumps+1]  # (B, jumps, C, H, W)
            future_states_flat = future_states.reshape(batch_size * jumps, *future_states.shape[2:])
            target_outputs = self.target_net(future_states_flat, return_projection=True)
            target_projections = target_outputs['projection']  # (B*jumps, proj_dim)
            target_projections = target_projections.reshape(batch_size, jumps, -1)  # (B, jumps, proj_dim)
            
        # Cosine similarity loss
        predicted_norm = F.normalize(predicted_projections, dim=-1)
        target_norm = F.normalize(target_projections, dim=-1)
        similarity = (predicted_norm * target_norm).sum(dim=-1)  # (B, jumps)
        
        spr_loss = -similarity.mean()
        
        return spr_loss
    
    def _log_metrics(self):
        """Log training metrics to console and CSV file."""
        if not self.recent_td_loss:
            return
        
        # Calculate averages
        avg_td_loss = np.mean(self.recent_td_loss)
        avg_spr_loss = np.mean(self.recent_spr_loss)
        avg_total_loss = avg_td_loss + self.config.spr_weight * avg_spr_loss
        avg_q = np.mean(self.recent_q_values)
        max_q = np.max(self.recent_q_values)
        
        # Get current hyperparameters
        current_gamma = self._get_current_gamma()
        current_n_step = self._get_current_n_step()
        current_epsilon = self.config.eps_start  # BBF uses eps=0
        
        # Console logging
        if self.config.verbose:
            print(f"\n[Step {self.total_steps:,}] Training Metrics:")
            print(f"  Loss: TD={avg_td_loss:.4f}, SPR={avg_spr_loss:.4f}, Total={avg_total_loss:.4f}")
            print(f"  Q-values: Mean={avg_q:.2f}, Max={max_q:.2f}")
            print(f"  Hyperparams: γ={current_gamma:.4f}, n-step={current_n_step}, ε={current_epsilon:.3f}")
            print(f"  Training: Learn steps={self.learn_steps:,}, Buffer size={len(self.replay):,}")
        
        # CSV logging
        if self.metrics_tracker:
            metrics = {
                'env_steps': self.total_steps,
                'learn_steps': self.learn_steps,
                'td_loss': avg_td_loss,
                'spr_loss': avg_spr_loss,
                'total_loss': avg_total_loss,
                'mean_q': avg_q,
                'max_q': max_q,
                'gamma': current_gamma,
                'n_step': current_n_step,
                'epsilon': current_epsilon,
            }
            self.metrics_tracker.log_training_step(metrics)

