"""Carmack-compatible multi-game runner with post-step control-loop semantics."""

from __future__ import annotations

import inspect
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from benchmark.runner import EnvStep
from benchmark.schedule import Schedule

CARMACK_MULTI_RUN_PROFILE = "carmack_compat"
CARMACK_MULTI_RUN_SCHEMA_VERSION = "carmack_multi_v1"


class CarmackMultiGameEnv(Protocol):
    """Environment interface required by Carmack-compatible multi-game runner."""

    action_set: Sequence[int]

    def load_game(self, game_id: str) -> Sequence[int]:
        """Load a game ROM and return the resulting local action set."""

    def reset(self) -> np.ndarray:
        """Reset game state and return first RGB observation."""

    def step(self, action_idx: int) -> EnvStep:
        """Advance one frame using local action index."""

    def lives(self) -> int:
        """Return current lives for active game."""


class CarmackMultiGameAgent(Protocol):
    """Agent contract for post-step Carmack-compatible multi-game control."""

    def frame(self, obs_rgb, reward, boundary) -> int:
        """Consume post-step tuple and return next action index."""


class _LegacyEndOfEpisodeFrameAdapter:
    """Backward-compatible adapter for frame(obs, reward, end_of_episode)."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def frame(self, obs_rgb, reward, boundary) -> int:
        end_of_episode = 0
        if isinstance(boundary, Mapping):
            end_of_episode = int(bool(boundary.get("end_of_episode_pulse", False)))
        else:
            end_of_episode = int(bool(boundary))
        return int(self._agent.frame(obs_rgb, float(reward), int(end_of_episode)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


class _LegacyObsRewardFrameAdapter:
    """Backward-compatible adapter for frame(obs, reward)."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def frame(self, obs_rgb, reward, boundary) -> int:
        del boundary
        return int(self._agent.frame(obs_rgb, float(reward)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


class _LegacyObsOnlyFrameAdapter:
    """Backward-compatible adapter for frame(obs)."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def frame(self, obs_rgb, reward, boundary) -> int:
        del reward, boundary
        return int(self._agent.frame(obs_rgb))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


class FrameFromStepAdapter:
    """
    Adapter for step-agents used under Carmack-compatible post-step runner mode.

    The adapter computes `is_decision_frame` only when boundary payload does not
    provide one, so explicit boundary metadata can override local cadence.
    """

    def __init__(self, step_agent: Any, decision_interval: int = 1) -> None:
        self._step_agent = step_agent
        self._decision_interval = max(1, int(decision_interval))
        self._frame_counter = 0

    def frame(self, obs_rgb, reward, boundary) -> int:
        if isinstance(boundary, Mapping):
            terminated = bool(boundary.get("terminated", False))
            truncated = bool(boundary.get("truncated", False))
            end_of_episode_pulse = bool(boundary.get("end_of_episode_pulse", False))
            has_prev_applied_action = bool(boundary.get("has_prev_applied_action", False))
            prev_applied_action_idx = int(boundary.get("prev_applied_action_idx", 0))
            global_frame_idx = int(boundary.get("global_frame_idx", self._frame_counter))
            if "is_decision_frame" in boundary:
                is_decision_frame = bool(boundary.get("is_decision_frame"))
            else:
                is_decision_frame = bool(self._frame_counter % self._decision_interval == 0)
        else:
            terminated = bool(boundary)
            truncated = False
            end_of_episode_pulse = bool(boundary)
            has_prev_applied_action = False
            prev_applied_action_idx = 0
            global_frame_idx = int(self._frame_counter)
            is_decision_frame = bool(self._frame_counter % self._decision_interval == 0)

        out = self._step_agent.step(
            obs_rgb=obs_rgb,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info={
                "end_of_episode_pulse": bool(end_of_episode_pulse),
                "has_prev_applied_action": bool(has_prev_applied_action),
                "prev_applied_action_idx": int(prev_applied_action_idx),
                "global_frame_idx": int(global_frame_idx),
                "is_decision_frame": bool(is_decision_frame),
            },
        )
        self._frame_counter += 1
        return int(out)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._step_agent, name)


def _adapt_carmack_frame_agent(agent: Any) -> Any:
    """Wrap legacy frame signatures for Carmack-compatible boundary payloads."""

    frame_fn = getattr(agent, "frame", None)
    if not callable(frame_fn):
        return agent
    try:
        sig = inspect.signature(frame_fn)
    except (TypeError, ValueError):
        return agent

    params = list(sig.parameters.values())
    if any(param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for param in params):
        return agent
    if len(params) < 3:
        if len(params) == 2:
            return _LegacyObsRewardFrameAdapter(agent)
        if len(params) == 1:
            return _LegacyObsOnlyFrameAdapter(agent)
        return agent
    third = params[2]
    if third.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
        return agent

    name = third.name.lower()
    if name in {"boundary", "boundary_payload", "boundary_info"}:
        return agent
    if name in {"end_of_episode", "done", "terminal", "episode_end"}:
        return _LegacyEndOfEpisodeFrameAdapter(agent)
    if third.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        return _LegacyEndOfEpisodeFrameAdapter(agent)
    return agent


@dataclass
class CarmackMultiGameRunnerConfig:
    """Runtime config for Carmack-compatible multi-game continual runner."""

    decision_interval: int = 1
    delay_frames: int = 0
    default_action_idx: int = 0
    include_timestamps: bool = True
    global_action_set: Sequence[int] = field(default_factory=lambda: tuple(range(18)))
    episode_log_interval: int = 0
    train_log_interval: int = 0
    reset_delay_queue_on_reset: bool = True
    reset_delay_queue_on_visit_switch: bool = True

    def __post_init__(self) -> None:
        if self.decision_interval != 1:
            raise ValueError("CarmackMultiGameRunnerConfig requires decision_interval == 1 (agent-owned cadence)")
        if self.delay_frames < 0:
            raise ValueError("delay_frames must be >= 0")
        if self.default_action_idx < 0:
            raise ValueError("default_action_idx must be >= 0")
        if not self.global_action_set:
            raise ValueError("global_action_set must not be empty")
        if self.default_action_idx >= len(self.global_action_set):
            raise ValueError("default_action_idx must be within global_action_set bounds")
        if self.episode_log_interval < 0:
            raise ValueError("episode_log_interval must be >= 0")
        if self.train_log_interval < 0:
            raise ValueError("train_log_interval must be >= 0")


class CarmackMultiGameRunner:
    """Post-step continual multi-game runner with Carmack-compatible semantics."""

    _BOUNDARY_CAUSE_VISIT_SWITCH = "visit_switch"
    _BOUNDARY_CAUSE_TERMINATED = "terminated"
    _BOUNDARY_CAUSE_TRUNCATED = "truncated"

    def __init__(
        self,
        env: CarmackMultiGameEnv,
        agent: CarmackMultiGameAgent,
        schedule: Schedule,
        config: CarmackMultiGameRunnerConfig,
        event_writer=None,
        episode_writer=None,
        segment_writer=None,
        time_fn=time.time,
    ) -> None:
        self.env = env
        self.agent = _adapt_carmack_frame_agent(agent)
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
        self._last_applied_action_idx_global: Optional[int] = None
        self._has_prev_applied_action = False

        self._obs_rgb: Optional[np.ndarray] = None
        self._lives = 0

        self._episode_id = 0
        self._episode_start_global_frame_idx = 0
        self._episode_return = 0.0
        self._episode_length = 0

        self._segment_id = 0
        self._segment_start_global_frame_idx = 0
        self._segment_return = 0.0
        self._segment_length = 0

        # Seed delay queue once so persist mode behaves as true queue persistence,
        # not as an accidental no-delay path on first use.
        self._delay_queue = deque([self.config.default_action_idx] * self.config.delay_frames)

    def _validate_action_idx(self, action_idx: Any) -> int:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self._num_global_actions:
            raise ValueError(f"action_idx {action_idx} out of bounds [0, {self._num_global_actions - 1}]")
        return action_idx

    def _reset_control_state(self, *, reset_delay_queue: bool) -> None:
        self._decided_action_idx = int(self.config.default_action_idx)
        if reset_delay_queue or len(self._delay_queue) == 0:
            self._delay_queue = deque([self.config.default_action_idx] * self.config.delay_frames)
        self._last_applied_action_idx_global = None
        self._has_prev_applied_action = False

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

    def _boundary_payload(
        self,
        *,
        terminated: bool,
        truncated: bool,
        global_frame_idx: int,
        applied_action_idx: int,
    ) -> Dict[str, Any]:
        return {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "end_of_episode_pulse": bool(terminated or truncated),
            # In post-step mode, boundary payload must reference the action that
            # produced the current (obs, reward) transition.
            "has_prev_applied_action": True,
            "prev_applied_action_idx": int(applied_action_idx),
            "global_frame_idx": int(global_frame_idx),
        }

    @classmethod
    def _resolve_boundary(
        cls,
        *,
        step: EnvStep,
        is_visit_last_frame: bool,
    ) -> Dict[str, Any]:
        boundary_terminated = bool(step.terminated)
        boundary_truncated = bool(step.truncated or is_visit_last_frame)
        boundary_cause: Optional[str]
        reset_cause: Optional[str]

        if is_visit_last_frame:
            boundary_cause = cls._BOUNDARY_CAUSE_VISIT_SWITCH
            reset_cause = cls._BOUNDARY_CAUSE_VISIT_SWITCH
        elif bool(step.truncated):
            boundary_cause = cls._BOUNDARY_CAUSE_TRUNCATED
            reset_cause = cls._BOUNDARY_CAUSE_TRUNCATED
        elif bool(step.terminated):
            boundary_cause = cls._BOUNDARY_CAUSE_TERMINATED
            reset_cause = cls._BOUNDARY_CAUSE_TERMINATED
        else:
            boundary_cause = None
            reset_cause = None

        reset_performed = bool(reset_cause is not None)
        return {
            "terminated": bool(boundary_terminated),
            "truncated": bool(boundary_truncated),
            "boundary_cause": boundary_cause,
            "reset_cause": reset_cause,
            "end_of_episode_pulse": bool(boundary_terminated or boundary_truncated),
            "reset_performed": bool(reset_performed),
            "reset_in_visit": bool(reset_performed and not is_visit_last_frame),
        }

    @staticmethod
    def _segment_ended_by(terminated: bool, truncated: bool) -> Optional[str]:
        if truncated:
            return "truncated"
        if terminated:
            return "terminated"
        return None

    def _write_episode(
        self,
        game_id: str,
        end_global_frame_idx: int,
        ended_by: str,
        boundary_cause: Optional[str],
        *,
        visit_idx: int,
        cycle_idx: int,
        episode_fps: float,
        rolling_avg: float,
    ) -> None:
        if self.config.episode_log_interval > 0 and (self._episode_id % self.config.episode_log_interval == 0):
            stats: Dict[str, Any] = {}
            stats_fn = getattr(self.agent, "get_stats", None)
            if callable(stats_fn):
                try:
                    payload = stats_fn()
                    if isinstance(payload, dict):
                        stats = payload
                except Exception:  # pragma: no cover - defensive
                    stats = {}
            common_suffix = (
                f'avg {rolling_avg:4.1f} '
                f'game {game_id} cycle {cycle_idx} visit {visit_idx} '
                f'ended_by {ended_by} boundary {boundary_cause}'
            )
            common_prefix = (
                f'frame:{end_global_frame_idx:7d} {episode_fps:4.0f}/s '
                f'eps {self._episode_id:3d},{self._episode_length:5d}={int(self._episode_return):5d} '
            )
            train_steps = stats.get("train_updates", stats.get("train_steps_estimate", None))
            train_steps_str = f'u {train_steps} ' if train_steps is not None else ''
            if "policy_entropy" in stats:
                def _f(v: Any) -> str:
                    return "n/a" if v is None else f"{float(v):.3f}"
                print(
                    f'0:ppo_multigame {common_prefix}'
                    f'{train_steps_str}'
                    f'ploss {_f(stats.get("last_policy_loss"))} '
                    f'vloss {_f(stats.get("last_value_loss"))} '
                    f'ent {_f(stats.get("policy_entropy"))} '
                    f'kl {_f(stats.get("approx_kl"))} '
                    f'{common_suffix}',
                    flush=True,
                )
            else:
                err_avg = float(stats.get("avg_error_ema", 0.0))
                err_max = float(stats.get("max_error_ema", 0.0))
                loss = float(stats.get("train_loss_ema", 0.0))
                target = float(stats.get("target_ema", 0.0))
                print(
                    f'0:delay_multigame {common_prefix}'
                    f'{train_steps_str}'
                    f'err {err_avg:.1f} {err_max:.1f} '
                    f'loss {loss:.1f} targ {target:.1f} '
                    f'{common_suffix}',
                    flush=True,
                )
        if self.episode_writer is None:
            return
        self.episode_writer.write(
            {
                "multi_run_profile": CARMACK_MULTI_RUN_PROFILE,
                "multi_run_schema_version": CARMACK_MULTI_RUN_SCHEMA_VERSION,
                "game_id": str(game_id),
                "episode_id": int(self._episode_id),
                "start_global_frame_idx": int(self._episode_start_global_frame_idx),
                "end_global_frame_idx": int(end_global_frame_idx),
                "length": int(self._episode_length),
                "return": float(self._episode_return),
                "ended_by": str(ended_by),
                "boundary_cause": None if boundary_cause is None else str(boundary_cause),
            }
        )

    def _write_segment(self, game_id: str, end_global_frame_idx: int, ended_by: str, boundary_cause: Optional[str]) -> None:
        if self.segment_writer is None:
            return
        self.segment_writer.write(
            {
                "multi_run_profile": CARMACK_MULTI_RUN_PROFILE,
                "multi_run_schema_version": CARMACK_MULTI_RUN_SCHEMA_VERSION,
                "game_id": str(game_id),
                "segment_id": int(self._segment_id),
                "start_global_frame_idx": int(self._segment_start_global_frame_idx),
                "end_global_frame_idx": int(end_global_frame_idx),
                "length": int(self._segment_length),
                "return": float(self._segment_return),
                "ended_by": str(ended_by),
                "boundary_cause": None if boundary_cause is None else str(boundary_cause),
            }
        )

    def run(self) -> Dict[str, Any]:
        """Execute scheduled continual stream and return run summary."""

        if len(self.schedule) == 0:
            raise ValueError("schedule must contain at least one visit")

        visits = self.schedule.visits
        global_frame_idx = 0
        segments_completed = 0
        episodes_completed = 0

        self._episode_id = 0
        self._segment_id = 0

        boundary_cause_counts: Dict[str, int] = {}
        reset_cause_counts: Dict[str, int] = {}
        reset_count = 0
        run_start_time = float(self.time_fn())
        rolling_average_frames = 100_000
        rolling_avg = -999.0
        episode_scores: list[float] = []
        episode_end: list[int] = []
        environment_start_time = float(self.time_fn())

        taken_action = int(self.config.default_action_idx)
        for visit_idx, visit in enumerate(visits):
            local_actions = self.env.load_game(visit.game_id)
            self._set_local_action_set(local_actions)
            self._obs_rgb = self.env.reset()
            self._lives = int(self.env.lives())
            self._reset_control_state(reset_delay_queue=bool(self.config.reset_delay_queue_on_visit_switch))
            self._start_episode(global_frame_idx)
            self._start_segment(global_frame_idx)

            for visit_frame_idx in range(int(visit.visit_frames)):
                decided_action_idx = int(taken_action)
                applied_action_idx = self._apply_delay(decided_action_idx)
                applied_local_idx, applied_ale_action = self._map_global_to_local(applied_action_idx)

                step = self.env.step(applied_local_idx)
                self._lives = int(step.lives)

                is_visit_last_frame = bool(visit_frame_idx == (int(visit.visit_frames) - 1))
                boundary = self._resolve_boundary(step=step, is_visit_last_frame=is_visit_last_frame)

                self._episode_return += float(step.reward)
                self._episode_length += 1
                self._segment_return += float(step.reward)
                self._segment_length += 1

                boundary_payload = self._boundary_payload(
                    terminated=bool(boundary["terminated"]),
                    truncated=bool(boundary["truncated"]),
                    global_frame_idx=int(global_frame_idx),
                    applied_action_idx=int(applied_action_idx),
                )

                event = {
                    "multi_run_profile": CARMACK_MULTI_RUN_PROFILE,
                    "multi_run_schema_version": CARMACK_MULTI_RUN_SCHEMA_VERSION,
                    "frame_idx": int(global_frame_idx),
                    "global_frame_idx": int(global_frame_idx),
                    "game_id": str(visit.game_id),
                    "visit_idx": int(visit.visit_idx),
                    "cycle_idx": int(visit.cycle_idx),
                    "visit_frame_idx": int(visit_frame_idx),
                    "episode_id": int(self._episode_id),
                    "segment_id": int(self._segment_id),
                    "is_decision_frame": True,
                    "decided_action_idx": int(decided_action_idx),
                    "applied_action_idx": int(applied_action_idx),
                    "next_policy_action_idx": None,
                    "applied_action_idx_local": int(applied_local_idx),
                    "applied_ale_action": int(applied_ale_action),
                    "reward": float(step.reward),
                    "terminated": bool(boundary["terminated"]),
                    "truncated": bool(boundary["truncated"]),
                    "env_terminated": bool(step.terminated),
                    "env_truncated": bool(step.truncated),
                    "lives": int(self._lives),
                    "episode_return_so_far": float(self._episode_return),
                    "segment_return_so_far": float(self._segment_return),
                    "end_of_episode_pulse": bool(boundary["end_of_episode_pulse"]),
                    "boundary_cause": boundary["boundary_cause"],
                    "reset_cause": boundary["reset_cause"],
                    "reset_performed": bool(boundary["reset_performed"]),
                    "env_termination_reason": None if step.termination_reason is None else str(step.termination_reason),
                }

                if bool(boundary["end_of_episode_pulse"]):
                    cause = boundary.get("boundary_cause")
                    if cause is not None:
                        boundary_cause_counts[str(cause)] = int(boundary_cause_counts.get(str(cause), 0) + 1)
                if bool(boundary["reset_performed"]):
                    reset_count += 1
                    cause = boundary.get("reset_cause")
                    if cause is not None:
                        reset_cause_counts[str(cause)] = int(reset_cause_counts.get(str(cause), 0) + 1)

                if self.config.include_timestamps:
                    event["wallclock_time"] = float(self.time_fn())

                if bool(boundary["end_of_episode_pulse"]):
                    ended_by = self._segment_ended_by(bool(boundary["terminated"]), bool(boundary["truncated"]))
                    assert ended_by is not None
                    self._write_segment(visit.game_id, global_frame_idx, ended_by, boundary.get("boundary_cause"))
                    segments_completed += 1
                    self._segment_id += 1

                    episode_scores.append(float(self._episode_return))
                    episode_end.append(int(global_frame_idx))
                    now = float(self.time_fn())
                    episode_fps = float(self._episode_length / max(now - environment_start_time, 1e-9))
                    environment_start_time = float(now)
                    count = 0
                    total = 0.0
                    for j in range(len(episode_scores) - 1, -1, -1):
                        if episode_end[j] < int(global_frame_idx) - int(rolling_average_frames):
                            break
                        count += 1
                        total += float(episode_scores[j])
                    rolling_avg = -999.0 if count == 0 else float(total / count)
                    self._write_episode(
                        visit.game_id,
                        global_frame_idx,
                        ended_by,
                        boundary.get("boundary_cause"),
                        visit_idx=int(visit.visit_idx),
                        cycle_idx=int(visit.cycle_idx),
                        episode_fps=float(episode_fps),
                        rolling_avg=float(rolling_avg),
                    )
                    episodes_completed += 1
                    self._episode_id += 1

                if bool(boundary["reset_in_visit"]):
                    self._obs_rgb = self.env.reset()
                    self._lives = int(self.env.lives())
                    self._reset_control_state(reset_delay_queue=bool(self.config.reset_delay_queue_on_reset))
                    self._start_episode(global_frame_idx + 1)
                    self._start_segment(global_frame_idx + 1)
                    obs_for_agent = self._obs_rgb
                else:
                    self._obs_rgb = step.obs_rgb
                    obs_for_agent = self._obs_rgb

                next_action = self._validate_action_idx(self.agent.frame(obs_for_agent, float(step.reward), boundary_payload))
                taken_action = int(next_action)
                event["next_policy_action_idx"] = int(next_action)

                if self.config.train_log_interval > 0 and ((global_frame_idx + 1) % int(self.config.train_log_interval) == 0):
                    stats: Dict[str, Any] = {}
                    stats_fn = getattr(self.agent, "get_stats", None)
                    if callable(stats_fn):
                        try:
                            payload = stats_fn()
                            if isinstance(payload, dict):
                                stats = payload
                        except Exception:  # pragma: no cover - defensive
                            stats = {}
                    elapsed = max(float(self.time_fn()) - run_start_time, 1e-6)
                    fps = float(global_frame_idx + 1) / elapsed
                    fr = global_frame_idx + 1
                    gm = visit.game_id
                    vi = visit.visit_idx
                    ci = visit.cycle_idx
                    if "policy_entropy" in stats:
                        def _ft(v: Any) -> str:
                            return "n/a" if v is None else f"{float(v):.3f}"
                        ra = rolling_avg if rolling_avg > -999.0 else 0.0
                        print(
                            f'0:ppo_multigame frame:{fr:7d} {fps:4.0f}/s '
                            f'eps {self._episode_id:3d} ret {int(self._episode_return):5d} avg {ra:4.1f} '
                            f'u {stats.get("train_updates", 0)} '
                            f'ploss {_ft(stats.get("last_policy_loss"))} '
                            f'vloss {_ft(stats.get("last_value_loss"))} '
                            f'ent {_ft(stats.get("policy_entropy"))} '
                            f'kl {_ft(stats.get("approx_kl"))} '
                            f'game {gm} cycle {ci} visit {vi}',
                            flush=True,
                        )
                    else:
                        msg = (
                            f"[train] frame={fr} game={gm} visit_idx={vi} fps={fps:.1f}"
                        )
                        for key in (
                            "frame_count", "u", "episode_number",
                            "train_loss_ema", "avg_error_ema", "max_error_ema",
                            "target_ema", "train_steps_estimate",
                        ):
                            if key not in stats:
                                continue
                            value = stats[key]
                            if isinstance(value, int):
                                msg += f" {key}={value}"
                            else:
                                try:
                                    msg += f" {key}={float(value):.3f}"
                                except (TypeError, ValueError):
                                    msg += f" {key}={value}"
                        print(msg, flush=True)

                if self.event_writer is not None:
                    self.event_writer.write(event)

                self._last_applied_action_idx_global = int(applied_action_idx)
                self._has_prev_applied_action = True
                global_frame_idx += 1

                if bool(boundary["end_of_episode_pulse"]) and not bool(boundary["reset_in_visit"]):
                    # Visit switch (or end-of-visit boundary) starts a new episode/segment on next global frame.
                    self._start_episode(global_frame_idx)
                    self._start_segment(global_frame_idx)

            # Keep visit-level control-state explicit at switches.
            if visit_idx + 1 < len(visits):
                self._reset_control_state(reset_delay_queue=bool(self.config.reset_delay_queue_on_visit_switch))

        return {
            "runner_mode": CARMACK_MULTI_RUN_PROFILE,
            "multi_run_profile": CARMACK_MULTI_RUN_PROFILE,
            "multi_run_schema_version": CARMACK_MULTI_RUN_SCHEMA_VERSION,
            "frames": int(global_frame_idx),
            "episodes_completed": int(episodes_completed),
            "segments_completed": int(segments_completed),
            "last_episode_id": int(self._episode_id),
            "last_segment_id": int(self._segment_id),
            "visits_completed": int(len(self.schedule)),
            "total_scheduled_frames": int(self.schedule.total_frames),
            "boundary_cause_counts": {str(k): int(v) for k, v in boundary_cause_counts.items()},
            "reset_cause_counts": {str(k): int(v) for k, v in reset_cause_counts.items()},
            "reset_count": int(reset_count),
        }
