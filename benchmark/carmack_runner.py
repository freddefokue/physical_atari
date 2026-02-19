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
    progress_log_interval_frames: int = 0
    pulse_log_interval: int = 0
    reset_log_interval: int = 1
    rolling_average_frames: int = 100_000
    log_rank: int = 0
    log_name: str = "delay"

    def __post_init__(self) -> None:
        if self.total_frames <= 0:
            raise ValueError("total_frames must be > 0")
        if self.delay_frames < 0:
            raise ValueError("delay_frames must be >= 0")
        if self.default_action_idx < 0:
            raise ValueError("default_action_idx must be >= 0")
        if self.max_frames_without_reward <= 0:
            raise ValueError("max_frames_without_reward must be > 0")
        if self.progress_log_interval_frames < 0:
            raise ValueError("progress_log_interval_frames must be >= 0")
        if self.pulse_log_interval < 0:
            raise ValueError("pulse_log_interval must be >= 0")
        if self.reset_log_interval < 0:
            raise ValueError("reset_log_interval must be >= 0")
        if self.rolling_average_frames <= 0:
            raise ValueError("rolling_average_frames must be > 0")


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
        pulse_count = 0
        reset_count = 0
        game_over_resets = 0
        timeout_resets = 0
        life_loss_resets = 0
        episode_scores = []
        episode_end = []
        environment_start = 0
        environment_start_time = float(self.time_fn())
        avg_for_log = 0.0
        average_bins = 1000
        start_time = float(self.time_fn())
        last_log_time = float(start_time)
        last_logged_frame = 0
        last_train_steps = 0

        def _agent_stats() -> Dict[str, Any]:
            stats_fn = getattr(self.agent, "get_stats", None)
            if callable(stats_fn):
                try:
                    payload = stats_fn()
                    if isinstance(payload, dict):
                        return payload
                except Exception:  # pragma: no cover - defensive
                    pass
            return {}

        for frame_idx in range(int(self.config.total_frames)):
            # Mirror agent_delay_target.py cadence: update rolling average only when the
            # episode graph bucket advances, not at every reset.
            if (int(frame_idx) * average_bins // int(self.config.total_frames)) != (
                (int(frame_idx) + 1) * average_bins // int(self.config.total_frames)
            ):
                count = 0
                total = 0.0
                for j in range(len(episode_scores) - 1, -1, -1):
                    if episode_end[j] < int(frame_idx) - int(self.config.rolling_average_frames):
                        break
                    count += 1
                    total += float(episode_scores[j])
                avg_for_log = -999.0 if count == 0 else float(total / count)

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

            if end_of_episode:
                pulse_count += 1
                if self.config.pulse_log_interval > 0 and (pulse_count % self.config.pulse_log_interval == 0):
                    print(
                        "[pulse] "
                        f"frame={frame_idx} "
                        f"pulse_idx={pulse_count} "
                        f"reason={pulse_reason} "
                        f"reward={reward:.3f} "
                        f"episode_return={episode_return:.3f} "
                        f"episode_length={episode_length} "
                        f"reset={should_reset}",
                        flush=True,
                    )

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
                episode_end.append(int(frame_idx))
                episode_scores.append(float(event_episode_return))
                if self.config.reset_log_interval > 0 and (episodes_completed % self.config.reset_log_interval == 0):
                    now = float(self.time_fn())
                    frames = int(frame_idx - environment_start)
                    frame_rate = float(frames / max(now - environment_start_time, 1e-9))
                    environment_start_time = now
                    environment_start = int(frame_idx)
                    stats = _agent_stats()
                    err_avg = float(stats.get("avg_error_ema", 0.0))
                    err_max = float(stats.get("max_error_ema", 0.0))
                    loss = float(stats.get("train_loss_ema", 0.0))
                    targ = float(stats.get("target_ema", 0.0))
                    print(
                        f"{int(self.config.log_rank)}:{self.config.log_name} "
                        f"frame:{int(frame_idx):7} "
                        f"{frame_rate:4.0f}/s "
                        f"eps {int(episodes_completed - 1):3},{int(frames):5}={int(event_episode_return):5} "
                        f"err {err_avg:.1f} {err_max:.1f} "
                        f"loss {loss:.1f} "
                        f"targ {targ:.1f} "
                        f"avg {avg_for_log:4.1f}",
                        flush=True,
                    )
            else:
                obs_for_agent = step.obs_rgb

            next_action = self._validate_action_idx(self.agent.frame(obs_for_agent, reward, int(end_of_episode)))
            taken_action = int(next_action)

            if self.config.progress_log_interval_frames > 0 and (
                (frame_idx + 1) % self.config.progress_log_interval_frames == 0
                or (frame_idx + 1) == int(self.config.total_frames)
            ):
                now = float(self.time_fn())
                delta_t = max(now - last_log_time, 1e-9)
                total_t = max(now - start_time, 1e-9)
                delta_frames = int(frame_idx + 1 - last_logged_frame)
                fps = float(delta_frames / delta_t)
                fps_total = float((frame_idx + 1) / total_t)
                stats = _agent_stats()
                train_steps = None
                train_step_fields = (
                    stats.get("train_steps_estimate"),
                    stats.get("train_steps"),
                )
                for value in train_step_fields:
                    if isinstance(value, int):
                        train_steps = int(value)
                        break
                if train_steps is None:
                    frame_count = stats.get("frame_count")
                    frame_skip = stats.get("frame_skip")
                    if isinstance(frame_count, int) and isinstance(frame_skip, int) and frame_skip > 0:
                        train_steps = int(frame_count // frame_skip)
                if train_steps is None:
                    train_steps = int(last_train_steps)
                train_delta = int(max(train_steps - last_train_steps, 0))
                train_sps = float(train_delta / delta_t)
                train_sps_total = float(train_steps / total_t)
                last_train_steps = int(train_steps)
                last_logged_frame = int(frame_idx + 1)
                last_log_time = float(now)

                msg = (
                    "[train] "
                    f"frame={frame_idx + 1} "
                    f"fps={fps:.2f} "
                    f"fps_total={fps_total:.2f} "
                    f"train_steps={train_steps} "
                    f"train_sps={train_sps:.2f} "
                    f"train_sps_total={train_sps_total:.2f} "
                    f"episode_return={event_episode_return:.3f} "
                    f"episode_length={event_episode_length} "
                    f"resets={episodes_completed}"
                )
                if "train_loss_ema" in stats:
                    msg += f" loss={float(stats['train_loss_ema']):.6f}"
                if "avg_error_ema" in stats:
                    msg += f" err_avg={float(stats['avg_error_ema']):.6f}"
                if "max_error_ema" in stats:
                    msg += f" err_max={float(stats['max_error_ema']):.6f}"
                if "target_ema" in stats:
                    msg += f" target={float(stats['target_ema']):.6f}"
                print(msg, flush=True)

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
