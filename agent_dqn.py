"""Vanilla DQN agent compatible with the continual benchmark harness."""

from __future__ import annotations

import argparse
import collections
import json
import os
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_runner import BenchmarkConfig, BenchmarkRunner, CycleConfig, GameSpec


@dataclass
class DQNConfig:
    """Configuration bundle for the DQN agent."""

    frame_skip: int = 1
    stack_size: int = 4
    obs_height: int = 84
    obs_width: int = 84
    replay_capacity: int = 200_000
    replay_start_size: int = 20_000
    batch_size: int = 32
    gamma: float = 0.99
    train_frequency: int = 4
    target_update_interval: int = 8_000
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 1_000_000
    max_grad_norm: float = 10.0


class ReplayBuffer:
    """Simple cyclic replay buffer storing pre-processed frames."""

    def __init__(self, capacity: int, state_shape: Tuple[int, int, int]):
        self.capacity = capacity
        self.state_shape = state_shape
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        state_bytes = (state.clamp(0.0, 1.0) * 255).to(dtype=torch.uint8).cpu().numpy()
        next_state_bytes = (next_state.clamp(0.0, 1.0) * 255).to(dtype=torch.uint8).cpu().numpy()

        self.states[self.position] = state_bytes
        self.next_states[self.position] = next_state_bytes
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        indices = np.random.choice(self.size, batch_size, replace=False)

        states = torch.from_numpy(self.states[indices]).to(device=device, dtype=torch.float32) / 255.0
        next_states = (
            torch.from_numpy(self.next_states[indices]).to(device=device, dtype=torch.float32) / 255.0
        )
        actions = torch.from_numpy(self.actions[indices]).to(device=device, dtype=torch.int64)
        rewards = torch.from_numpy(self.rewards[indices]).to(device=device, dtype=torch.float32)
        dones = torch.from_numpy(self.dones[indices].astype(np.float32)).to(device=device, dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:  # pragma: no cover - convenience wrapper
        return self.size


class DQNNetwork(nn.Module):
    """Classic convolutional torso followed by a fully connected value head."""

    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DQNAgent:
    """Vanilla DQN agent with ε-greedy exploration and replay buffer training."""

    def __init__(self, data_dir: str, seed: int, num_actions: int, total_frames: int, **kwargs: Any) -> None:
        self.data_dir = data_dir
        self.seed = seed
        self.num_actions = num_actions
        self.total_frames = total_frames

        self.config = DQNConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.obs_queue: Deque[torch.Tensor] = collections.deque(maxlen=self.config.stack_size)
        state_shape = (self.config.stack_size, self.config.obs_height, self.config.obs_width)
        self.replay_buffer = ReplayBuffer(self.config.replay_capacity, state_shape)

        self.online_net = DQNNetwork(self.config.stack_size, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(self.config.stack_size, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)

        self.grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=self.device, dtype=torch.float32).view(
            3, 1, 1
        )
        self.zero_frame = torch.zeros(
            (self.config.obs_height, self.config.obs_width), device=self.device, dtype=torch.float32
        )

        self.frame_index = 0
        self.gradient_steps = 0
        self.prev_state: Optional[torch.Tensor] = None
        self.prev_action: int = 0
        self.epsilon = self.config.epsilon_start

        self.train_losses: List[float] = []
        self.epsilon_history: List[float] = []
        self.q_value_history: List[float] = []

        self.loss_last_step: float = 0.0

    # --------------------------------
    # Returns the selected action index
    # --------------------------------
    def frame(self, observation_rgb8: np.ndarray, reward: float, end_of_episode: int) -> int:
        frame_tensor = self._preprocess_observation(observation_rgb8)
        self.obs_queue.append(frame_tensor)

        done = bool(end_of_episode)

        current_state = self._get_state_tensor()

        if self.prev_state is not None:
            self.replay_buffer.push(self.prev_state, self.prev_action, reward, current_state, done)

        action = self._select_action(current_state)

        if len(self.replay_buffer) >= self.config.replay_start_size:
            if self.frame_index % self.config.train_frequency == 0:
                self._train_step()

        if done:
            self.obs_queue.clear()
            for _ in range(self.config.stack_size):
                self.obs_queue.append(frame_tensor)
            self.prev_state = None
        else:
            self.prev_state = current_state.detach()

        self.prev_action = action
        self.frame_index += 1
        return action

    def save_model(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.online_net.state_dict(), filename)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        frame = torch.from_numpy(observation).to(self.device, dtype=torch.float32)
        frame = frame.permute(2, 0, 1)  # C, H, W
        frame = (frame * self.grayscale_weights).sum(dim=0, keepdim=True)
        frame = frame.unsqueeze(0)
        frame = F.interpolate(
            frame,
            size=(self.config.obs_height, self.config.obs_width),
            mode='bilinear',
            align_corners=False,
        )
        frame = frame.squeeze(0) / 255.0
        return frame.squeeze(0)

    def _get_state_tensor(self) -> torch.Tensor:
        if not self.obs_queue:
            for _ in range(self.config.stack_size):
                self.obs_queue.append(self.zero_frame)

        frames = list(self.obs_queue)
        if len(frames) < self.config.stack_size:
            pad = [frames[0]] * (self.config.stack_size - len(frames))
            frames = pad + frames
        else:
            frames = frames[-self.config.stack_size :]

        state = torch.stack(frames, dim=0)
        return state

    def _epsilon(self) -> float:
        decay_ratio = min(1.0, self.frame_index / float(self.config.epsilon_decay_steps))
        epsilon = self.config.epsilon_start + decay_ratio * (self.config.epsilon_final - self.config.epsilon_start)
        return float(max(self.config.epsilon_final, epsilon))

    def _select_action(self, state: torch.Tensor) -> int:
        self.epsilon = self._epsilon()
        if np.random.random() < self.epsilon:
            action = int(np.random.randint(self.num_actions))
        else:
            with torch.no_grad():
                q_values = self.online_net(state.unsqueeze(0))
                action = int(q_values.argmax(dim=1).item())
                self.q_value_history.append(float(q_values.max().item()))
        self.epsilon_history.append(self.epsilon)
        return action

    def _train_step(self) -> None:
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size, self.device
        )

        q_values = self.online_net(states)
        state_action_values = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(dim=1).values
            targets = rewards + (1.0 - dones) * self.config.gamma * max_next_q_values

        loss = F.smooth_l1_loss(state_action_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.loss_last_step = float(loss.item())
        self.train_losses.append(self.loss_last_step)

        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())


def run_training_frames(
    agent: DQNAgent,
    ale,
    action_set,
    *,
    name: str,
    data_dir: str,
    rank: int,
    delay_frames: int,
    average_frames: int,
    max_frames_without_reward: int,
    lives_as_episodes: int,
    save_incremental_models: bool,
    last_model_save: int,
    frame_budget: int,
    frame_offset: int = 0,
    graph_total_frames: Optional[int] = None,
):
    episode_scores: List[float] = []
    episode_end: List[int] = []
    episode_graph = torch.full((1000,), -999.0)
    parms_graph = torch.zeros((1000, 1))

    cumulative_reward = 0.0
    frames_without_reward = 0
    previous_lives = ale.lives()
    delayed_actions = [0] * delay_frames
    action = 0

    if graph_total_frames is None:
        graph_total_frames = frame_offset + frame_budget
    else:
        graph_total_frames = max(graph_total_frames, frame_offset + frame_budget)

    for local_frame in range(frame_budget):
        global_frame = frame_offset + local_frame
        global_next = global_frame + 1

        delayed_actions.append(action)
        env_action = int(action_set[delayed_actions.pop(0)])
        reward = ale.act(env_action)
        cumulative_reward += reward

        if reward != 0:
            frames_without_reward = 0
        else:
            frames_without_reward += 1

        end_of_episode = 0
        if lives_as_episodes and ale.lives() < previous_lives:
            previous_lives = ale.lives()
            end_of_episode = 1
        if ale.game_over() or frames_without_reward >= max_frames_without_reward:
            ale.reset_game()
            previous_lives = ale.lives()
            frames_without_reward = 0
            episode_scores.append(cumulative_reward)
            episode_end.append(global_frame)
            cumulative_reward = 0.0
            end_of_episode = 1

        observation = ale.getScreenRGB()
        action = agent.frame(observation, reward, end_of_episode)

        if save_incremental_models and global_next // 500_000 != last_model_save:
            last_model_save = global_next // 500_000
            filename = os.path.join(data_dir, f'{name}_{global_next}.model')
            agent.save_model(filename)

        points = episode_graph.shape[0]
        if (
            global_frame * points // graph_total_frames
            != global_next * points // graph_total_frames
        ):
            i = global_frame * points // graph_total_frames
            i = min(i, points - 1)
            if agent.train_losses:
                episode_graph[i] = agent.train_losses[-1]
            parms_graph[i, 0] = agent.epsilon

    return last_model_save, episode_scores, episode_end, episode_graph, parms_graph


def _build_continual_config(
    cycle_frames: int,
    game_order: List[str],
    seed: int,
    frame_budget_override: Optional[List[int]] = None,
) -> BenchmarkConfig:
    if frame_budget_override is not None:
        if len(frame_budget_override) != len(game_order):
            raise ValueError('Custom frame budgets must match number of games.')
        if sum(frame_budget_override) != cycle_frames:
            raise ValueError('Custom frame budgets must sum to cycle frame budget.')
        game_frame_budgets = frame_budget_override
    else:
        if cycle_frames % len(game_order) != 0:
            raise ValueError('cycle_frames must be divisible by number of games without overrides.')
        frame_budget = cycle_frames // len(game_order)
        game_frame_budgets = [frame_budget] * len(game_order)

    cycles: List[CycleConfig] = []
    for cycle_index in range(3):
        cycle_games: List[GameSpec] = []
        for game_index, game_name in enumerate(game_order):
            cycle_games.append(
                GameSpec(
                    name=game_name,
                    frame_budget=game_frame_budgets[game_index],
                    sticky_prob=0.25,
                    delay_frames=6,
                    seed=seed + cycle_index,
                    params={'use_canonical_full_actions': True},
                )
            )
        cycles.append(CycleConfig(cycle_index=cycle_index, games=cycle_games))

    return BenchmarkConfig(
        cycles=cycles,
        description=f'Continual DQN benchmark: 3 cycles x {len(game_order)} games (cycle frames={cycle_frames})',
    )


def main() -> None:  # noqa: C901 - command-line parsing wrapper
    data_dir = './results/dqn_continual_run'
    os.makedirs(data_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description='Run the vanilla DQN agent on the continual benchmark.')
    parser.add_argument('--cycle_frames', type=int, default=150_000)
    parser.add_argument(
        '--game_frame_budgets',
        type=str,
        help='Comma-separated frame budgets for each game in the cycle (optional).',
    )
    parser.add_argument('--rank', type=int, default=0, help='Rank/index for multi-process runs.')
    parser.add_argument('--mode', type=str, default='continual', choices=['continual'])
    args = parser.parse_args()

    if args.mode != 'continual':
        raise ValueError('Only continual mode is supported in this script.')

    game_order = ['centipede', 'ms_pacman', 'atlantis']

    provided_budgets: Optional[List[int]] = None
    if args.game_frame_budgets:
        provided_budgets = [int(value.strip()) for value in args.game_frame_budgets.split(',') if value.strip()]

    benchmark_config = _build_continual_config(args.cycle_frames, game_order, args.rank, provided_budgets)

    num_actions = len(BenchmarkRunner._canonical_action_set())
    total_frames = benchmark_config.total_frames

    agent = DQNAgent(
        data_dir,
        seed=args.rank,
        num_actions=num_actions,
        total_frames=total_frames,
    )

    runner = BenchmarkRunner(
        agent,
        benchmark_config,
        frame_runner=run_training_frames,
        data_dir=data_dir,
        rank=args.rank,
        default_seed=args.rank,
        reduce_action_set=0,
        use_canonical_full_actions=True,
        average_frames=100_000,
        max_frames_without_reward=18_000,
        lives_as_episodes=1,
        save_incremental_models=False,
    )

    results = runner.run()

    summary_records: List[Dict[str, Any]] = []
    for result in results:
        total_score = float(sum(result.episode_scores))
        mean_score = float(np.mean(result.episode_scores)) if result.episode_scores else 0.0
        summary_records.append(
            {
                'cycle_index': result.cycle_index,
                'game_index': result.game_index,
                'game': result.spec.name,
                'frame_offset': result.frame_offset,
                'frame_budget': result.frame_budget,
                'episodes': len(result.episode_scores),
                'total_episode_score': total_score,
                'mean_episode_score': mean_score,
            }
        )

    summary_path = os.path.join(data_dir, 'continual_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump(
            {
                'description': benchmark_config.description,
                'records': summary_records,
            },
            fp,
            indent=2,
        )

    print(f'Wrote summary to {summary_path}')


if __name__ == '__main__':
    main()


