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

"""Gym-based Atari DQN agent using CleanRL-style wrappers and replay buffer."""

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
    GameSpec,
    GameResult,
)
from cleanrl_utils.buffers import ReplayBuffer

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
    nvtx_range,
    ProfileSections,
)


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x / 255.0)


@dataclass
class AgentConfig:
    env_id: str = "BreakoutNoFrameskip-v4"
    total_steps: int = 1_000_000
    buffer_size: int = 1_000_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1_000
    batch_size: int = 32
    exploration_fraction: float = 0.10
    start_e: float = 1.0
    end_e: float = 0.01
    learning_starts: int = 80_000
    per_game_learning_starts: int = 1_000
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

        # --- FIX: Action Space Mismatch ---
        # If continual, we force the network to handle the full Atari action set
        # to prevent shape mismatches when switching between games with different action counts.
        if config.continual:
            self.num_actions = ATARI_CANONICAL_ACTIONS
        else:
            self.num_actions = action_space.n
        # ----------------------------------

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

        # Use self.num_actions instead of action_space.n
        self.q_network = QNetwork(in_channels, self.num_actions).to(self.device)
        self.target_network = QNetwork(in_channels, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.frame_skip = FRAME_SKIP
        self.game_frame_counters: Dict[str, int] = {}

        self.replay_buffer = ReplayBuffer(
            buffer_size=config.buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
            n_envs=1,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

    def act(self, observation) -> int:
        # --- FIX: Early epsilon check ---
        # Check epsilon BEFORE creating tensors to avoid unnecessary GPU work
        epsilon = self._current_epsilon()
        if random.random() < epsilon or self.global_step < self.config.learning_starts:
            self.last_obs = np.array(observation, copy=False)
            self.last_action = random.randint(0, self.num_actions - 1)
            return self.last_action
        
        # Only create tensor when we need the network
        obs_np = np.array(observation, copy=False)
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.channels_last:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)

        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        self.last_obs = obs_np
        self.last_action = action
        return action

    def step(self, next_observation, reward: float, done: bool, info: Optional[Dict] = None):
        if self.last_obs is None or self.last_action is None:
            return None

        next_obs_np = np.array(next_observation, copy=False)
        obs_batch = np.expand_dims(self.last_obs, axis=0)
        next_obs_batch = np.expand_dims(next_obs_np, axis=0)
        action_batch = np.array([[self.last_action]], dtype=np.int64)
        reward_batch = np.array([reward], dtype=np.float32)
        done_batch = np.array([done], dtype=np.float32)
        info_list = [info or {}]

        self.replay_buffer.add(obs_batch, next_obs_batch, action_batch, reward_batch, done_batch, info_list)
        self.global_step += 1

        train_log = None
        learning_ready = self.global_step >= max(self.config.learning_starts, self._game_learning_start_step)
        if learning_ready and self.global_step % self.config.train_frequency == 0:
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

    def _current_epsilon(self) -> float:
        decay_steps = max(1, int(self.config.exploration_fraction * self.config.total_steps))
        progress = min(1.0, self.global_step / decay_steps)
        return self.config.start_e + progress * (self.config.end_e - self.config.start_e)

    def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            return obs.permute(0, 3, 1, 2)
        return obs

    def train_function(self):
        samples = self.replay_buffer.sample(self.config.batch_size)
        obs = self._prepare_obs(samples.observations)
        next_obs = self._prepare_obs(samples.next_observations)
        rewards = samples.rewards.flatten()
        dones = samples.dones.flatten()
        actions = samples.actions.long().squeeze(1)

        with torch.no_grad():
            target_max, _ = self.target_network(next_obs).max(dim=1)
            td_target = rewards + self.config.gamma * target_max * (1 - dones)

        current_q = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(td_target, current_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_updates += 1

        return {"loss": loss.item(), "q_value": current_q.mean().item()}

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
        # Reset buffer counters instead of reallocating the entire buffer.
        # This avoids deallocating/reallocating potentially gigabytes of memory.
        if self.config.continual:
            print(f"  [Agent] Resetting Replay Buffer counters for new game...")
            # Reset the logical state without memory reallocation
            self.replay_buffer.pos = 0
            self.replay_buffer.full = False
        # -----------------------------


def agent_dqn_frame_runner(
    agent: Agent,
    handle: EnvironmentHandle,
    *,
    context: FrameRunnerContext,
    writer: Optional[SummaryWriter] = None,
) -> FrameRunnerResult:
    if handle.backend != 'gym' or handle.gym is None:
        raise TypeError("agent_dqn_frame_runner requires a Gym environment handle.")

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
                
                # Track cumulative frames for this specific game to allow overlaying cycles
                current_game_frames = agent.game_frame_counters.get(game_name, 0)
                writer.add_scalar(f"charts/frames_per_game/{game_name}", current_game_frames, agent.global_step)
                
                # Log cycle index if implicit in context or runner
                # We don't have explicit cycle index in context, but we can log it if available or just rely on game/time
                
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
        agent_dqn_frame_runner(agent, handle, context=context, writer=writer)
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
        return agent_dqn_frame_runner(agent_instance, handle, context=context, writer=writer)

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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--target-network-frequency", type=int, default=1_000)
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
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
    )


def main():
    config = parse_args()
    log_root = './results_final_dqn/smokey_continual'
    os.makedirs(log_root, exist_ok=True)
    run_name = f"{config.env_id}__agent_final_dqn__{config.seed}__{int(time.time())}"
    
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