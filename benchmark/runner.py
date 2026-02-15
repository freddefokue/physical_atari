"""Streaming benchmark runner with frame-skip and action-delay semantics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class EnvStep:
    """Result of one environment frame step."""

    obs_rgb: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    lives: int
    termination_reason: Optional[str] = None


class StreamingEnv(Protocol):
    """Environment interface needed by the benchmark runner."""

    action_set: Sequence[int]

    def reset(self) -> np.ndarray:
        """Reset environment and return initial RGB frame."""

    def step(self, action_idx: int) -> EnvStep:
        """Advance one frame with a provided action index."""

    def lives(self) -> int:
        """Return current lives count."""


class StreamingAgent(Protocol):
    """Agent interface for frame-by-frame streaming."""

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        """Return an action index for this frame."""


@dataclass
class RunnerConfig:
    """
    Benchmark runner configuration.

    - `total_frames`: total environment frames to run.
    - `frame_skip`: update `decided_action_idx` every N frames.
    - `delay_frames`: action latency queue length in frames.
    - `default_action_idx`: queue initialization action index.
    - `include_timestamps`: include wallclock timestamps in event logs.
    """

    total_frames: int
    frame_skip: int = 4
    delay_frames: int = 0
    default_action_idx: int = 0
    include_timestamps: bool = True

    def __post_init__(self) -> None:
        if self.total_frames <= 0:
            raise ValueError("total_frames must be > 0")
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be > 0")
        if self.delay_frames < 0:
            raise ValueError("delay_frames must be >= 0")
        if self.default_action_idx < 0:
            raise ValueError("default_action_idx must be >= 0")


class BenchmarkRunner:
    """Single-game streaming runner with frame-skip and delay queue mechanics."""

    def __init__(
        self,
        env: StreamingEnv,
        agent: StreamingAgent,
        config: RunnerConfig,
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

        self._delay_queue = deque()
        self._obs_rgb = None
        self._lives = 0
        self._episode_idx = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._decided_action_idx = self.config.default_action_idx

    def _validate_action_idx(self, action_idx: Any) -> int:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self._num_actions:
            raise ValueError(f"action_idx {action_idx} out of bounds [0, {self._num_actions - 1}]")
        return action_idx

    def _reset_episode_state(self) -> None:
        self._obs_rgb = self.env.reset()
        self._lives = int(self.env.lives())
        self._episode_return = 0.0
        self._episode_length = 0
        self._decided_action_idx = self.config.default_action_idx
        self._delay_queue = deque([self.config.default_action_idx] * self.config.delay_frames)

    def _apply_delay(self, decided_action_idx: int) -> int:
        if self.config.delay_frames == 0:
            return decided_action_idx
        self._delay_queue.append(decided_action_idx)
        return int(self._delay_queue.popleft())

    def run(self) -> dict:
        """Execute the benchmark run and return a summary dictionary."""
        self._episode_idx = 0
        self._reset_episode_state()

        prev_reward = 0.0
        prev_terminated = False
        prev_truncated = False
        episodes_completed = 0

        for frame_idx in range(self.config.total_frames):
            info = {
                "frame_idx": frame_idx,
                "episode_idx": self._episode_idx,
                "lives": self._lives,
                "frame_skip": self.config.frame_skip,
                "delay_frames": self.config.delay_frames,
            }
            candidate_action_idx = self.agent.step(
                self._obs_rgb,
                prev_reward,
                prev_terminated,
                prev_truncated,
                info,
            )

            if frame_idx % self.config.frame_skip == 0:
                self._decided_action_idx = self._validate_action_idx(candidate_action_idx)

            applied_action_idx = self._apply_delay(self._decided_action_idx)
            step = self.env.step(applied_action_idx)

            self._obs_rgb = step.obs_rgb
            self._lives = int(step.lives)
            self._episode_return += float(step.reward)
            self._episode_length += 1

            event = {
                "frame_idx": frame_idx,
                "applied_action_idx": int(applied_action_idx),
                "decided_action_idx": int(self._decided_action_idx),
                "reward": float(step.reward),
                "terminated": bool(step.terminated),
                "truncated": bool(step.truncated),
                "lives": self._lives,
                "episode_idx": self._episode_idx,
                "episode_return": float(self._episode_return),
            }
            if self.config.include_timestamps:
                event["wallclock_time"] = float(self.time_fn())
            if self.event_writer is not None:
                self.event_writer.write(event)

            prev_reward = float(step.reward)
            prev_terminated = bool(step.terminated)
            prev_truncated = bool(step.truncated)

            if step.terminated or step.truncated:
                reason = step.termination_reason
                if reason is None:
                    reason = "terminated" if step.terminated else "truncated"
                if self.episode_writer is not None:
                    self.episode_writer.write(
                        {
                            "episode_idx": self._episode_idx,
                            "episode_return": float(self._episode_return),
                            "length": int(self._episode_length),
                            "termination_reason": reason,
                            "end_frame_idx": frame_idx,
                        }
                    )
                episodes_completed += 1
                self._episode_idx += 1
                self._reset_episode_state()

        return {
            "frames": int(self.config.total_frames),
            "episodes_completed": int(episodes_completed),
            "last_episode_idx": int(self._episode_idx),
            "last_episode_return": float(self._episode_return),
            "last_episode_length": int(self._episode_length),
        }
