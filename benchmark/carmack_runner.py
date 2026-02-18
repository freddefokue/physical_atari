"""Single-game runner mode that emulates agent_delay_target.py control-loop semantics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence

import numpy as np

from benchmark.runner import EnvStep


class CarmackEnv(Protocol):
    """Environment interface needed by Carmack-compatible single-game runner."""

    action_set: Sequence[int]

    def reset(self) -> np.ndarray:
        """Reset environment and return initial RGB frame."""

    def step(self, action_idx: int) -> EnvStep:
        """Advance one frame with local action index."""

    def lives(self) -> int:
        """Return current lives count."""


class CarmackAgent(Protocol):
    """Carmack agent interface: receives post-action transition each frame."""

    def frame(self, obs_rgb, reward, end_of_episode) -> int:
        """Consume post-step tuple and return the next policy action index."""


@dataclass
class CarmackRunnerConfig:
    """Configuration for Carmack-compatible single-game loop semantics."""

    total_frames: int
    delay_frames: int = 0
    default_action_idx: int = 0
    include_timestamps: bool = True
    lives_as_episodes: bool = True
    max_frames_without_reward: int = 18_000
    reset_on_life_loss: bool = False

    def __post_init__(self) -> None:
        if self.total_frames <= 0:
            raise ValueError("total_frames must be > 0")
        if self.delay_frames < 0:
            raise ValueError("delay_frames must be >= 0")
        if self.default_action_idx < 0:
            raise ValueError("default_action_idx must be >= 0")
        if self.max_frames_without_reward <= 0:
            raise ValueError("max_frames_without_reward must be > 0")


class CarmackCompatRunner:
    """Runner emulating the order/boundary behavior in agent_delay_target.py main loop."""

    def __init__(
        self,
        env: CarmackEnv,
        agent: CarmackAgent,
        config: CarmackRunnerConfig,
        event_writer=None,
        episode_writer=None,
        time_fn=time.time,
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config
        self.event_writer = event_writer
        self.episode_writer = episode_writer
        self.time_fn = time_fn

        self._num_actions = len(self.env.action_set)
        if self._num_actions <= 0:
            raise ValueError("env.action_set must not be empty")
        if self.config.default_action_idx >= self._num_actions:
            raise ValueError("default_action_idx is outside env.action_set bounds")

    def _validate_action_idx(self, action_idx: Any) -> int:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self._num_actions:
            raise ValueError(f"action_idx {action_idx} out of bounds [0, {self._num_actions - 1}]")
        return action_idx

    def run(self) -> Dict[str, Any]:
        self.env.reset()
        previous_lives = int(self.env.lives())

        delayed_actions = deque([int(self.config.default_action_idx)] * int(self.config.delay_frames))
        taken_action = int(self.config.default_action_idx)

        episode_idx = 0
        episode_return = 0.0
        episode_length = 0
        episodes_completed = 0
        frames_without_reward = 0

        life_loss_pulses = 0
        reset_count = 0
        game_over_resets = 0
        timeout_resets = 0
        life_loss_resets = 0

        for frame_idx in range(int(self.config.total_frames)):
            decided_action_idx = int(taken_action)
            delayed_actions.append(int(decided_action_idx))
            applied_action_idx = int(delayed_actions.popleft()) if self.config.delay_frames > 0 else int(decided_action_idx)
            step = self.env.step(applied_action_idx)

            reward = float(step.reward)
            episode_return += reward
            episode_length += 1

            if reward != 0.0:
                frames_without_reward = 0
            else:
                frames_without_reward += 1

            end_of_episode = 0
            pulse_reason: Optional[str] = None
            reset_performed = False
            termination_reason = str(step.termination_reason) if step.termination_reason else None

            if bool(self.config.lives_as_episodes) and int(step.lives) < previous_lives:
                previous_lives = int(step.lives)
                life_loss_pulses += 1
                end_of_episode = 1
                pulse_reason = "life_loss"

            timeout_reached = frames_without_reward >= int(self.config.max_frames_without_reward)
            frames_without_reward_before_reset = int(frames_without_reward)
            env_done = bool(step.terminated or step.truncated or timeout_reached)
            reset_due_to_life_loss = bool(end_of_episode and self.config.reset_on_life_loss)
            should_reset = bool(env_done or reset_due_to_life_loss)

            if timeout_reached:
                end_of_episode = 1
                pulse_reason = "no_reward_timeout"
            elif step.terminated:
                end_of_episode = 1
                pulse_reason = termination_reason or "terminated"
            elif step.truncated:
                end_of_episode = 1
                pulse_reason = termination_reason or "truncated"

            event_episode_idx = int(episode_idx)
            event_episode_return = float(episode_return)
            event_episode_length = int(episode_length)

            if should_reset:
                if timeout_reached:
                    timeout_resets += 1
                elif step.terminated:
                    game_over_resets += 1
                elif reset_due_to_life_loss:
                    life_loss_resets += 1
                if self.episode_writer is not None:
                    self.episode_writer.write(
                        {
                            "episode_idx": int(episode_idx),
                            "episode_return": float(episode_return),
                            "length": int(episode_length),
                            "termination_reason": str(pulse_reason or "terminated"),
                            "end_frame_idx": int(frame_idx),
                            "ended_by_reset": True,
                        }
                    )
                episodes_completed += 1
                episode_idx += 1
                episode_return = 0.0
                episode_length = 0

                obs_for_agent = self.env.reset()
                previous_lives = int(self.env.lives())
                frames_without_reward = 0
                reset_performed = True
                reset_count += 1
            else:
                obs_for_agent = step.obs_rgb

            next_action = self._validate_action_idx(self.agent.frame(obs_for_agent, reward, int(end_of_episode)))
            taken_action = int(next_action)

            event = {
                "frame_idx": int(frame_idx),
                "applied_action_idx": int(applied_action_idx),
                "decided_action_idx": int(decided_action_idx),
                "next_policy_action_idx": int(next_action),
                "reward": float(reward),
                "terminated": bool(step.terminated),
                "truncated": bool(step.truncated),
                "lives": int(step.lives),
                "episode_idx": int(event_episode_idx),
                "episode_return": float(event_episode_return),
                "episode_length": int(event_episode_length),
                "end_of_episode_pulse": bool(end_of_episode),
                "pulse_reason": pulse_reason,
                "reset_performed": bool(reset_performed),
                "frames_without_reward": int(frames_without_reward_before_reset),
            }
            if self.config.include_timestamps:
                event["wallclock_time"] = float(self.time_fn())
            if self.event_writer is not None:
                self.event_writer.write(event)

        return {
            "frames": int(self.config.total_frames),
            "episodes_completed": int(episodes_completed),
            "last_episode_idx": int(episode_idx),
            "last_episode_return": float(episode_return),
            "last_episode_length": int(episode_length),
            "life_loss_pulses": int(life_loss_pulses),
            "reset_count": int(reset_count),
            "game_over_resets": int(game_over_resets),
            "timeout_resets": int(timeout_resets),
            "life_loss_resets": int(life_loss_resets),
        }
