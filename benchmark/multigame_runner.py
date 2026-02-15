"""Multi-game continual streaming runner with deterministic visit switching."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np

from benchmark.runner import EnvStep, StreamingAgent
from benchmark.schedule import Schedule


class MultiGameEnv(Protocol):
    """Environment interface required by `MultiGameRunner`."""

    action_set: Sequence[int]

    def load_game(self, game_id: str) -> Sequence[int]:
        """Load a game ROM and return the resulting local action set."""

    def reset(self) -> np.ndarray:
        """Reset game state and return first RGB observation."""

    def step(self, action_idx: int) -> EnvStep:
        """Advance one frame using local action index."""

    def lives(self) -> int:
        """Return current lives for active game."""


@dataclass
class MultiGameRunnerConfig:
    """Runtime configuration for continual multi-game streaming."""

    decision_interval: int = 4
    delay_frames: int = 0
    default_action_idx: int = 0
    include_timestamps: bool = True
    global_action_set: Sequence[int] = field(default_factory=lambda: tuple(range(18)))

    def __post_init__(self) -> None:
        if self.decision_interval <= 0:
            raise ValueError("decision_interval must be > 0")
        if self.delay_frames < 0:
            raise ValueError("delay_frames must be >= 0")
        if self.default_action_idx < 0:
            raise ValueError("default_action_idx must be >= 0")
        if not self.global_action_set:
            raise ValueError("global_action_set must not be empty")
        if self.default_action_idx >= len(self.global_action_set):
            raise ValueError("default_action_idx must be within global_action_set bounds")


class MultiGameRunner:
    """Continual runner: one frame stream across scheduled game visits."""

    def __init__(
        self,
        env: MultiGameEnv,
        agent: StreamingAgent,
        schedule: Schedule,
        config: MultiGameRunnerConfig,
        event_writer=None,
        episode_writer=None,
        segment_writer=None,
        time_fn=time.time,
    ) -> None:
        self.env = env
        self.agent = agent
        self.schedule = schedule
        self.config = config
        self.event_writer = event_writer
        self.episode_writer = episode_writer
        self.segment_writer = segment_writer
        self.time_fn = time_fn

        self._global_action_set = [int(action) for action in self.config.global_action_set]
        self._num_global_actions = len(self._global_action_set)

        self._local_action_set: Sequence[int] = []
        self._local_action_to_idx: Dict[int, int] = {}
        self._local_default_idx = 0

        self._delay_queue = deque()
        self._decided_action_idx = int(self.config.default_action_idx)
        self._decision_phase = 0

        self._obs_rgb: Optional[np.ndarray] = None
        self._lives = 0

        self._episode_id = 0
        self._episode_start_global_frame_idx = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._true_terminal_episodes = 0

        self._segment_id = 0
        self._segment_start_global_frame_idx = 0
        self._segment_return = 0.0
        self._segment_length = 0

    def _validate_action_idx(self, action_idx: Any) -> int:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self._num_global_actions:
            raise ValueError(f"action_idx {action_idx} out of bounds [0, {self._num_global_actions - 1}]")
        return action_idx

    def _reset_control_state(self) -> None:
        """Reset latency queue and decision phase at environment boundaries."""

        self._decided_action_idx = int(self.config.default_action_idx)
        self._delay_queue = deque([self.config.default_action_idx] * self.config.delay_frames)
        self._decision_phase = 0

    def _start_segment(self, global_frame_idx: int) -> None:
        self._segment_start_global_frame_idx = int(global_frame_idx)
        self._segment_return = 0.0
        self._segment_length = 0

    def _start_episode(self, global_frame_idx: int) -> None:
        self._episode_start_global_frame_idx = int(global_frame_idx)
        self._episode_return = 0.0
        self._episode_length = 0

    def _apply_delay(self, decided_action_idx: int) -> int:
        if self.config.delay_frames == 0:
            return int(decided_action_idx)
        self._delay_queue.append(int(decided_action_idx))
        return int(self._delay_queue.popleft())

    def _set_local_action_set(self, action_set: Sequence[int]) -> None:
        self._local_action_set = [int(a) for a in action_set]
        if not self._local_action_set:
            raise ValueError("local action_set must not be empty")
        self._local_action_to_idx = {action: idx for idx, action in enumerate(self._local_action_set)}

        default_global_ale_action = int(self._global_action_set[self.config.default_action_idx])
        self._local_default_idx = self._local_action_to_idx.get(default_global_ale_action, 0)

    def _map_global_to_local(self, global_action_idx: int) -> Tuple[int, int]:
        global_action_idx = int(global_action_idx)
        ale_action = int(self._global_action_set[global_action_idx])
        local_idx = self._local_action_to_idx.get(ale_action, self._local_default_idx)
        return int(local_idx), int(self._local_action_set[local_idx])

    def _build_agent_info(self, global_frame_idx: int) -> Dict[str, Any]:
        """Agent-facing info intentionally excludes schedule/task identity fields."""

        return {
            "lives": int(self._lives),
            "decision_interval": int(self.config.decision_interval),
            "delay_frames": int(self.config.delay_frames),
            "action_space_n": int(self._num_global_actions),
            "episode_id": int(self._episode_id),
            "segment_id": int(self._segment_id),
            "is_decision_frame": bool(self._decision_phase == 0),
            "global_frame_idx": int(global_frame_idx),
        }

    @staticmethod
    def _segment_ended_by(terminated: bool, truncated: bool) -> Optional[str]:
        if truncated:
            return "truncated"
        if terminated:
            return "terminated"
        return None

    def _write_episode(self, game_id: str, end_global_frame_idx: int, ended_by: str) -> None:
        if self.episode_writer is None:
            return
        self.episode_writer.write(
            {
                "game_id": str(game_id),
                "episode_id": int(self._episode_id),
                "start_global_frame_idx": int(self._episode_start_global_frame_idx),
                "end_global_frame_idx": int(end_global_frame_idx),
                "length": int(self._episode_length),
                "return": float(self._episode_return),
                "ended_by": str(ended_by),
            }
        )

    def _write_segment(self, game_id: str, end_global_frame_idx: int, ended_by: str) -> None:
        if self.segment_writer is None:
            return
        self.segment_writer.write(
            {
                "game_id": str(game_id),
                "segment_id": int(self._segment_id),
                "start_global_frame_idx": int(self._segment_start_global_frame_idx),
                "end_global_frame_idx": int(end_global_frame_idx),
                "length": int(self._segment_length),
                "return": float(self._segment_return),
                "ended_by": str(ended_by),
            }
        )

    def run(self) -> Dict[str, Any]:
        """Execute the scheduled continual stream and return a run summary."""

        if len(self.schedule) == 0:
            raise ValueError("schedule must contain at least one visit")

        prev_reward = 0.0
        prev_terminated = False
        prev_truncated = False

        global_frame_idx = 0
        segments_completed = 0
        episodes_completed = 0
        true_terminal_episodes_completed = 0

        self._episode_id = 0
        self._segment_id = 0
        self._true_terminal_episodes = 0

        for visit in self.schedule:
            local_actions = self.env.load_game(visit.game_id)
            self._set_local_action_set(local_actions)

            self._obs_rgb = self.env.reset()
            self._lives = int(self.env.lives())
            self._reset_control_state()
            self._start_episode(global_frame_idx)
            self._start_segment(global_frame_idx)

            for visit_frame_idx in range(visit.visit_frames):
                is_decision_frame = self._decision_phase == 0

                candidate_action_idx = self.agent.step(
                    self._obs_rgb,
                    prev_reward,
                    prev_terminated,
                    prev_truncated,
                    self._build_agent_info(global_frame_idx),
                )

                if is_decision_frame:
                    self._decided_action_idx = self._validate_action_idx(candidate_action_idx)
                self._decision_phase = (self._decision_phase + 1) % self.config.decision_interval

                applied_action_idx = self._apply_delay(self._decided_action_idx)
                applied_local_idx, applied_ale_action = self._map_global_to_local(applied_action_idx)

                # Exactly one environment step per frame.
                step = self.env.step(applied_local_idx)
                self._obs_rgb = step.obs_rgb
                self._lives = int(step.lives)

                is_visit_last_frame = visit_frame_idx == (visit.visit_frames - 1)
                terminated = bool(step.terminated)
                truncated = bool(step.truncated or is_visit_last_frame)

                self._episode_return += float(step.reward)
                self._episode_length += 1
                self._segment_return += float(step.reward)
                self._segment_length += 1

                event = {
                    "frame_idx": int(global_frame_idx),
                    "global_frame_idx": int(global_frame_idx),
                    "game_id": str(visit.game_id),
                    "visit_idx": int(visit.visit_idx),
                    "cycle_idx": int(visit.cycle_idx),
                    "visit_frame_idx": int(visit_frame_idx),
                    "episode_id": int(self._episode_id),
                    "segment_id": int(self._segment_id),
                    "is_decision_frame": bool(is_decision_frame),
                    "decided_action_idx": int(self._decided_action_idx),
                    "applied_action_idx": int(applied_action_idx),
                    "applied_action_idx_local": int(applied_local_idx),
                    "applied_ale_action": int(applied_ale_action),
                    "reward": float(step.reward),
                    "terminated": terminated,
                    "truncated": truncated,
                    "lives": int(self._lives),
                    "episode_return_so_far": float(self._episode_return),
                    "segment_return_so_far": float(self._segment_return),
                    "true_terminal_episodes_so_far": int(self._true_terminal_episodes),
                }
                if self.config.include_timestamps:
                    event["wallclock_time"] = float(self.time_fn())
                if self.event_writer is not None:
                    self.event_writer.write(event)

                prev_reward = float(step.reward)
                prev_terminated = terminated
                prev_truncated = truncated

                if terminated or truncated:
                    ended_by = self._segment_ended_by(terminated, truncated)
                    assert ended_by is not None
                    self._write_episode(visit.game_id, global_frame_idx, ended_by)
                    self._write_segment(visit.game_id, global_frame_idx, ended_by)
                    episodes_completed += 1
                    segments_completed += 1
                    self._episode_id += 1
                    self._segment_id += 1

                    if terminated:
                        true_terminal_episodes_completed += 1
                        self._true_terminal_episodes += 1

                    if not is_visit_last_frame:
                        self._obs_rgb = self.env.reset()
                        self._lives = int(self.env.lives())
                        self._reset_control_state()
                        self._start_episode(global_frame_idx + 1)
                        self._start_segment(global_frame_idx + 1)

                global_frame_idx += 1

        return {
            "frames": int(global_frame_idx),
            "episodes_completed": int(episodes_completed),
            "segments_completed": int(segments_completed),
            "true_terminal_episodes": int(true_terminal_episodes_completed),
            "last_episode_id": int(self._episode_id),
            "last_segment_id": int(self._segment_id),
            "visits_completed": int(len(self.schedule)),
            "total_scheduled_frames": int(self.schedule.total_frames),
        }
