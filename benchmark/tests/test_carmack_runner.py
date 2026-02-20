from __future__ import annotations

from dataclasses import dataclass
import os
import re

import numpy as np
import pytest

from benchmark.carmack_runner import (
    CARMACK_SINGLE_RUN_PROFILE,
    CARMACK_SINGLE_RUN_SCHEMA_VERSION,
    CarmackCompatRunner,
    CarmackRunnerConfig,
)
from benchmark.run_single_game import _FrameFromStepAdapter
from benchmark.runner import EnvStep


class MemoryWriter:
    def __init__(self) -> None:
        self.rows = []

    def write(self, record) -> None:
        self.rows.append(dict(record))


class RecordingFrameAgent:
    def __init__(self, action_idx: int = 1) -> None:
        self.action_idx = int(action_idx)
        self.calls = []

    def frame(self, obs_rgb, reward, end_of_episode) -> int:
        marker = int(obs_rgb[0, 0, 0])
        self.calls.append(
            {
                "obs_marker": marker,
                "reward": float(reward),
                "end_of_episode": int(end_of_episode),
            }
        )
        return int(self.action_idx)


class RecordingFrameAgentWithStats(RecordingFrameAgent):
    def get_stats(self):
        return {
            "avg_error_ema": 1.2,
            "max_error_ema": 3.4,
            "train_loss_ema": 0.5,
            "target_ema": -2.0,
        }


class RecordingFrameAgentWithMalformedStats(RecordingFrameAgent):
    def get_stats(self):
        return {
            "avg_error_ema": "bad",
            "max_error_ema": object(),
            "train_loss_ema": "nan-ish",
            "target_ema": None,
        }


class RecordingBoundaryFrameAgent:
    def __init__(self, action_idx: int = 1) -> None:
        self.action_idx = int(action_idx)
        self.calls = []

    def frame(self, obs_rgb, reward, boundary) -> int:
        marker = int(obs_rgb[0, 0, 0])
        payload = dict(boundary) if isinstance(boundary, dict) else {"raw": boundary}
        payload["obs_marker"] = marker
        payload["reward"] = float(reward)
        self.calls.append(payload)
        return int(self.action_idx)


class RecordingLegacyUnknownNameAgent:
    """Legacy scalar end-of-episode contract but non-standard parameter name."""

    def __init__(self, action_idx: int = 1) -> None:
        self.action_idx = int(action_idx)
        self.calls = []

    def frame(self, obs_rgb, reward, flag) -> int:
        marker = int(obs_rgb[0, 0, 0])
        self.calls.append(
            {
                "obs_marker": marker,
                "reward": float(reward),
                "flag": int(flag),
            }
        )
        return int(self.action_idx)


class RecordingTwoArgFrameAgent:
    """Legacy/malformed signature: frame(obs, reward)."""

    def __init__(self, action_idx: int = 1) -> None:
        self.action_idx = int(action_idx)
        self.calls = []

    def frame(self, obs_rgb, reward) -> int:
        marker = int(obs_rgb[0, 0, 0])
        self.calls.append({"obs_marker": marker, "reward": float(reward)})
        return int(self.action_idx)


class RecordingObsOnlyFrameAgent:
    """Legacy/malformed signature: frame(obs)."""

    def __init__(self, action_idx: int = 1) -> None:
        self.action_idx = int(action_idx)
        self.calls = []

    def frame(self, obs_rgb) -> int:
        marker = int(obs_rgb[0, 0, 0])
        self.calls.append({"obs_marker": marker})
        return int(self.action_idx)


class RecordingVariadicFrameAgent:
    """Variadic signature should remain pass-through and receive full payload."""

    def __init__(self, action_idx: int = 1) -> None:
        self.action_idx = int(action_idx)
        self.calls = []

    def frame(self, *args) -> int:
        self.calls.append(tuple(args))
        return int(self.action_idx)


class RecordingStepAgentWithMalformedStats:
    def __init__(self, action_idx: int = 1) -> None:
        self.action_idx = int(action_idx)
        self.calls = 0

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        del obs_rgb, reward, terminated, truncated, info
        self.calls += 1
        return int(self.action_idx)

    def get_stats(self):
        return {
            "avg_error_ema": object(),
            "max_error_ema": "bad",
            "train_loss_ema": None,
            "target_ema": float("nan"),
            "train_steps_estimate": "not_an_int",
        }


class RecordingBoundarySequenceAgent:
    """Deterministic agent for golden-trace compatibility checks."""

    def __init__(self, action_sequence):
        self._seq = [int(x) for x in action_sequence]
        self._idx = 0
        self.calls = []

    def frame(self, obs_rgb, reward, boundary) -> int:
        marker = int(obs_rgb[0, 0, 0])
        self.calls.append(
            {
                "obs_marker": marker,
                "reward": float(reward),
                "boundary": dict(boundary),
            }
        )
        out = int(self._seq[self._idx % len(self._seq)])
        self._idx += 1
        return out


@dataclass
class ScriptedEnv:
    action_set: list
    rewards: list
    lives_seq: list
    terminated_at: set
    truncated_at: set

    def __post_init__(self) -> None:
        self.step_count = 0
        self.reset_count = 0
        self._lives = int(self.lives_seq[0] if self.lives_seq else 3)

    def reset(self):
        self.reset_count += 1
        self._lives = 3
        return np.full((210, 160, 3), fill_value=200 + self.reset_count, dtype=np.uint8)

    def lives(self) -> int:
        return int(self._lives)

    def step(self, action_idx: int) -> EnvStep:
        del action_idx
        idx = self.step_count
        self.step_count += 1

        reward = float(self.rewards[idx] if idx < len(self.rewards) else 0.0)
        self._lives = int(self.lives_seq[idx] if idx < len(self.lives_seq) else self._lives)
        terminated = idx in self.terminated_at
        truncated = idx in self.truncated_at
        reason = "scripted_end" if terminated else ("time_limit" if truncated else None)
        obs = np.full((210, 160, 3), fill_value=10 + self.step_count, dtype=np.uint8)
        return EnvStep(obs_rgb=obs, reward=reward, terminated=terminated, truncated=truncated, lives=self._lives, termination_reason=reason)


def run_with_memory(env, agent, config: CarmackRunnerConfig, *, time_fn=None):
    events = MemoryWriter()
    episodes = MemoryWriter()
    runner_kwargs = {
        "env": env,
        "agent": agent,
        "config": config,
        "event_writer": events,
        "episode_writer": episodes,
    }
    if time_fn is not None:
        runner_kwargs["time_fn"] = time_fn
    summary = CarmackCompatRunner(**runner_kwargs).run()
    return summary, events.rows, episodes.rows


def test_carmack_runner_calls_agent_after_env_step():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0, 0.0],
        lives_seq=[3, 3, 3],
        terminated_at=set(),
        truncated_at=set(),
    )
    agent = RecordingFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(total_frames=3, delay_frames=0, include_timestamps=False)
    _, events, _ = run_with_memory(env, agent, config)

    assert len(agent.calls) == 3
    assert agent.calls[0]["obs_marker"] == 11
    assert [row["applied_action_idx"] for row in events] == [0, 1, 1]


def test_carmack_runner_life_loss_pulse_without_reset():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0, 0.0, 0.0],
        lives_seq=[3, 2, 2, 2],
        terminated_at=set(),
        truncated_at=set(),
    )
    agent = RecordingFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(
        total_frames=4,
        delay_frames=0,
        include_timestamps=False,
        lives_as_episodes=True,
        max_frames_without_reward=999,
        reset_on_life_loss=False,
    )
    summary, events, episodes = run_with_memory(env, agent, config)

    assert summary["life_loss_pulses"] == 1
    assert summary["reset_count"] == 0
    assert summary["episodes_completed"] == 0
    assert len(episodes) == 0
    assert [row["pulse_reason"] for row in events] == [None, "life_loss", None, None]
    assert [call["end_of_episode"] for call in agent.calls] == [0, 1, 0, 0]


def test_carmack_runner_game_over_resets_and_uses_reset_obs_for_agent():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[1.0, 2.0, 3.0, 4.0],
        lives_seq=[3, 3, 3, 3],
        terminated_at={2},
        truncated_at=set(),
    )
    agent = RecordingFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(total_frames=4, delay_frames=0, include_timestamps=False, max_frames_without_reward=999)
    summary, events, episodes = run_with_memory(env, agent, config)

    assert summary["game_over_resets"] == 1
    assert summary["episodes_completed"] == 1
    assert len(episodes) == 1
    assert episodes[0]["episode_return"] == 6.0
    assert episodes[0]["end_frame_idx"] == 2
    assert events[2]["pulse_reason"] == "scripted_end"
    assert events[2]["reset_performed"] is True
    assert agent.calls[2]["obs_marker"] == 202
    assert agent.calls[2]["end_of_episode"] == 1


def test_carmack_runner_timeout_resets():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0, 0.0, 0.0, 0.0],
        lives_seq=[3, 3, 3, 3, 3],
        terminated_at=set(),
        truncated_at=set(),
    )
    agent = RecordingFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(
        total_frames=5,
        delay_frames=0,
        include_timestamps=False,
        max_frames_without_reward=2,
    )
    summary, events, episodes = run_with_memory(env, agent, config)

    assert summary["timeout_resets"] == 2
    assert summary["truncated_resets"] == 0
    assert summary["episodes_completed"] == 2
    assert len(episodes) == 2
    assert [row["pulse_reason"] for row in events] == [None, "no_reward_timeout", None, "no_reward_timeout", None]
    assert [row["frames_without_reward"] for row in events] == [1, 2, 1, 2, 1]
    assert [row["truncated"] for row in events] == [False, True, False, True, False]
    assert [row["env_truncated"] for row in events] == [False, False, False, False, False]


def test_carmack_runner_consumes_final_transition():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[7.0],
        lives_seq=[3],
        terminated_at=set(),
        truncated_at=set(),
    )
    agent = RecordingFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(total_frames=1, delay_frames=0, include_timestamps=False, max_frames_without_reward=999)
    _, _, _ = run_with_memory(env, agent, config)

    assert len(agent.calls) == 1
    assert agent.calls[0]["reward"] == 7.0


def test_carmack_runner_reset_log_matches_delay_target_style(capsys):
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[1.0, 2.0, 0.0],
        lives_seq=[3, 3, 3],
        terminated_at={1},
        truncated_at=set(),
    )
    agent = RecordingFrameAgentWithStats(action_idx=1)
    config = CarmackRunnerConfig(
        total_frames=3,
        delay_frames=0,
        include_timestamps=False,
        progress_log_interval_frames=0,
        pulse_log_interval=0,
        reset_log_interval=1,
        log_rank=5,
        log_name="delay_breakout0_4_16384",
    )
    run_with_memory(env, agent, config)
    out = capsys.readouterr().out

    assert "[train]" not in out
    assert "[pulse]" not in out
    assert re.search(r"5:delay_breakout0_4_16384 frame:\s*\d+\s+\d+/s eps\s+\d+,\s*\d+=\s*\d+", out)
    assert "err 1.2 3.4" in out
    assert "loss 0.5" in out
    assert "targ -2.0" in out


def test_carmack_runner_malformed_stats_do_not_crash_logging(capsys):
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0],
        lives_seq=[3, 3],
        terminated_at={1},
        truncated_at=set(),
    )
    agent = RecordingFrameAgentWithMalformedStats(action_idx=1)
    config = CarmackRunnerConfig(
        total_frames=2,
        delay_frames=0,
        include_timestamps=False,
        progress_log_interval_frames=1,
        pulse_log_interval=0,
        reset_log_interval=1,
        max_frames_without_reward=999,
    )
    summary, _, _ = run_with_memory(env, agent, config)
    out = capsys.readouterr().out

    assert summary["frames"] == 2
    assert summary["episodes_completed"] == 1
    assert "[train]" in out


def test_carmack_runner_new_boundary_payload_timeout_marks_truncated():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0, 0.0],
        lives_seq=[3, 3, 3],
        terminated_at=set(),
        truncated_at=set(),
    )
    agent = RecordingBoundaryFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(total_frames=3, delay_frames=0, include_timestamps=False, max_frames_without_reward=2)
    summary, events, episodes = run_with_memory(env, agent, config)

    assert summary["timeout_resets"] == 1
    assert len(episodes) == 1
    assert events[1]["pulse_reason"] == "no_reward_timeout"
    assert events[1]["reset_performed"] is True
    assert agent.calls[1]["terminated"] is False
    assert agent.calls[1]["truncated"] is True
    assert agent.calls[1]["end_of_episode_pulse"] is True
    assert "boundary_cause" not in agent.calls[1]


def test_carmack_runner_new_boundary_payload_life_loss_pulse_without_reset():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0, 0.0],
        lives_seq=[3, 2, 2],
        terminated_at=set(),
        truncated_at=set(),
    )
    agent = RecordingBoundaryFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(
        total_frames=3,
        delay_frames=0,
        include_timestamps=False,
        lives_as_episodes=True,
        reset_on_life_loss=False,
        max_frames_without_reward=999,
    )
    summary, events, episodes = run_with_memory(env, agent, config)

    assert summary["life_loss_pulses"] == 1
    assert summary["reset_count"] == 0
    assert len(episodes) == 0
    assert events[1]["pulse_reason"] == "life_loss"
    assert events[1]["reset_performed"] is False
    assert agent.calls[1]["terminated"] is False
    assert agent.calls[1]["truncated"] is False
    assert agent.calls[1]["end_of_episode_pulse"] is True
    assert "boundary_cause" not in agent.calls[1]


def test_carmack_runner_legacy_adapter_handles_unknown_third_arg_name():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0, 0.0],
        lives_seq=[3, 3, 3],
        terminated_at={1},
        truncated_at=set(),
    )
    agent = RecordingLegacyUnknownNameAgent(action_idx=1)
    config = CarmackRunnerConfig(total_frames=3, delay_frames=0, include_timestamps=False, max_frames_without_reward=999)
    _, events, _ = run_with_memory(env, agent, config)

    assert len(agent.calls) == 3
    assert [call["flag"] for call in agent.calls] == [0, 1, 0]
    assert events[1]["terminated"] is True
    assert events[1]["truncated"] is False


def test_carmack_runner_legacy_adapter_handles_two_arg_frame_signature():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0, 0.0],
        lives_seq=[3, 3, 3],
        terminated_at={1},
        truncated_at=set(),
    )
    agent = RecordingTwoArgFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(total_frames=3, delay_frames=0, include_timestamps=False, max_frames_without_reward=999)
    summary, events, _ = run_with_memory(env, agent, config)

    assert summary["frames"] == 3
    assert len(agent.calls) == 3
    assert events[1]["terminated"] is True


def test_carmack_runner_legacy_adapter_handles_obs_only_frame_signature():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0],
        lives_seq=[3, 3],
        terminated_at={1},
        truncated_at=set(),
    )
    agent = RecordingObsOnlyFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(total_frames=2, delay_frames=0, include_timestamps=False, max_frames_without_reward=999)
    summary, _, _ = run_with_memory(env, agent, config)

    assert summary["frames"] == 2
    assert len(agent.calls) == 2


def test_carmack_runner_variadic_frame_signature_remains_passthrough():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0],
        lives_seq=[3, 3],
        terminated_at={1},
        truncated_at=set(),
    )
    agent = RecordingVariadicFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(total_frames=2, delay_frames=0, include_timestamps=False, max_frames_without_reward=999)
    summary, events, _ = run_with_memory(env, agent, config)

    assert summary["frames"] == 2
    assert len(agent.calls) == 2
    first_call = agent.calls[0]
    assert len(first_call) == 3
    assert isinstance(first_call[2], dict)
    assert set(first_call[2].keys()) == {"terminated", "truncated", "end_of_episode_pulse"}
    assert events[1]["terminated"] is True


def test_carmack_runner_step_adapter_malformed_stats_do_not_crash_logging(capsys):
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 0.0, 0.0],
        lives_seq=[3, 3, 3],
        terminated_at={2},
        truncated_at=set(),
    )
    step_agent = RecordingStepAgentWithMalformedStats(action_idx=1)
    agent = _FrameFromStepAdapter(step_agent)
    config = CarmackRunnerConfig(
        total_frames=3,
        delay_frames=0,
        include_timestamps=False,
        progress_log_interval_frames=1,
        pulse_log_interval=0,
        reset_log_interval=1,
        max_frames_without_reward=999,
    )
    summary, _, _ = run_with_memory(env, agent, config)
    out = capsys.readouterr().out

    assert summary["frames"] == 3
    assert summary["episodes_completed"] == 1
    assert "[train]" in out


def test_carmack_runner_golden_trace_actions_and_boundaries():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        lives_seq=[3, 3, 3, 2, 2, 2],
        terminated_at={1},
        truncated_at={4},
    )
    agent = RecordingBoundarySequenceAgent(action_sequence=[1, 2, 3, 1, 2, 3])
    config = CarmackRunnerConfig(
        total_frames=6,
        delay_frames=1,
        include_timestamps=False,
        lives_as_episodes=True,
        reset_on_life_loss=False,
        max_frames_without_reward=99,
        progress_log_interval_frames=0,
        pulse_log_interval=0,
        reset_log_interval=0,
    )
    summary, events, episodes = run_with_memory(env, agent, config)

    expected_events = [
        {
            "frame_idx": 0,
            "decided_action_idx": 0,
            "applied_action_idx": 0,
            "next_policy_action_idx": 1,
            "terminated": False,
            "truncated": False,
            "boundary_cause": None,
            "reset_cause": None,
            "end_of_episode_pulse": False,
            "reset_performed": False,
        },
        {
            "frame_idx": 1,
            "decided_action_idx": 1,
            "applied_action_idx": 0,
            "next_policy_action_idx": 2,
            "terminated": True,
            "truncated": False,
            "boundary_cause": "scripted_end",
            "reset_cause": "terminated",
            "end_of_episode_pulse": True,
            "reset_performed": True,
        },
        {
            "frame_idx": 2,
            "decided_action_idx": 2,
            "applied_action_idx": 1,
            "next_policy_action_idx": 3,
            "terminated": False,
            "truncated": False,
            "boundary_cause": None,
            "reset_cause": None,
            "end_of_episode_pulse": False,
            "reset_performed": False,
        },
        {
            "frame_idx": 3,
            "decided_action_idx": 3,
            "applied_action_idx": 2,
            "next_policy_action_idx": 1,
            "terminated": False,
            "truncated": False,
            "boundary_cause": "life_loss",
            "reset_cause": None,
            "end_of_episode_pulse": True,
            "reset_performed": False,
        },
        {
            "frame_idx": 4,
            "decided_action_idx": 1,
            "applied_action_idx": 3,
            "next_policy_action_idx": 2,
            "terminated": False,
            "truncated": True,
            "boundary_cause": "time_limit",
            "reset_cause": "truncated",
            "end_of_episode_pulse": True,
            "reset_performed": True,
        },
        {
            "frame_idx": 5,
            "decided_action_idx": 2,
            "applied_action_idx": 1,
            "next_policy_action_idx": 3,
            "terminated": False,
            "truncated": False,
            "boundary_cause": "life_loss",
            "reset_cause": None,
            "end_of_episode_pulse": True,
            "reset_performed": False,
        },
    ]

    actual_events = []
    for row in events:
        actual_events.append(
            {
                "frame_idx": row["frame_idx"],
                "decided_action_idx": row["decided_action_idx"],
                "applied_action_idx": row["applied_action_idx"],
                "next_policy_action_idx": row["next_policy_action_idx"],
                "terminated": row["terminated"],
                "truncated": row["truncated"],
                "boundary_cause": row["boundary_cause"],
                "reset_cause": row["reset_cause"],
                "end_of_episode_pulse": row["end_of_episode_pulse"],
                "reset_performed": row["reset_performed"],
            }
        )
    assert actual_events == expected_events

    assert episodes == [
        {
            "single_run_profile": CARMACK_SINGLE_RUN_PROFILE,
            "single_run_schema_version": CARMACK_SINGLE_RUN_SCHEMA_VERSION,
            "episode_idx": 0,
            "episode_return": 2.0,
            "length": 2,
            "termination_reason": "scripted_end",
            "end_frame_idx": 1,
            "ended_by_reset": True,
        },
        {
            "single_run_profile": CARMACK_SINGLE_RUN_PROFILE,
            "single_run_schema_version": CARMACK_SINGLE_RUN_SCHEMA_VERSION,
            "episode_idx": 1,
            "episode_return": 0.0,
            "length": 3,
            "termination_reason": "time_limit",
            "end_frame_idx": 4,
            "ended_by_reset": True,
        },
    ]

    assert summary["frames"] == 6
    assert summary["episodes_completed"] == 2
    assert summary["pulse_count"] == 4
    assert summary["boundary_cause_counts"] == {"scripted_end": 1, "life_loss": 2, "time_limit": 1}
    assert summary["reset_cause_counts"] == {"terminated": 1, "truncated": 1}
    assert summary["life_loss_pulses"] == 2
    assert summary["game_over_resets"] == 1
    assert summary["truncated_resets"] == 1
    assert summary["timeout_resets"] == 0
    assert summary["life_loss_resets"] == 0

    assert all("boundary_cause" not in call["boundary"] for call in agent.calls)


def test_carmack_runner_delay_queue_policy_persist_vs_reset():
    env_kwargs = dict(
        action_set=list(range(4)),
        rewards=[0.0, 0.0, 0.0, 0.0],
        lives_seq=[3, 3, 3, 3],
        terminated_at={2},
        truncated_at=set(),
    )

    # Persist policy (default): delayed actions from pre-reset queue carry over.
    env_persist = ScriptedEnv(**env_kwargs)
    agent_persist = RecordingBoundarySequenceAgent(action_sequence=[1, 2, 3, 1, 2])
    config_persist = CarmackRunnerConfig(
        total_frames=4,
        delay_frames=2,
        default_action_idx=0,
        include_timestamps=False,
        max_frames_without_reward=999,
        reset_delay_queue_on_reset=False,
        progress_log_interval_frames=0,
        pulse_log_interval=0,
        reset_log_interval=0,
    )
    _, events_persist, _ = run_with_memory(env_persist, agent_persist, config_persist)

    # Reset policy: queue is re-seeded to default on reset.
    env_reset = ScriptedEnv(**env_kwargs)
    agent_reset = RecordingBoundarySequenceAgent(action_sequence=[1, 2, 3, 1, 2])
    config_reset = CarmackRunnerConfig(
        total_frames=4,
        delay_frames=2,
        default_action_idx=0,
        include_timestamps=False,
        max_frames_without_reward=999,
        reset_delay_queue_on_reset=True,
        progress_log_interval_frames=0,
        pulse_log_interval=0,
        reset_log_interval=0,
    )
    _, events_reset, _ = run_with_memory(env_reset, agent_reset, config_reset)

    # After reset at frame 2, frame 3 applied action differs by policy.
    assert events_persist[2]["reset_performed"] is True
    assert events_reset[2]["reset_performed"] is True
    assert events_persist[3]["applied_action_idx"] == 1
    assert events_reset[3]["applied_action_idx"] == 0


def test_carmack_runner_boundary_precedence_table():
    # timeout dominates all
    timeout = CarmackCompatRunner._resolve_boundary_and_reset(
        step_terminated=True,
        step_truncated=True,
        life_loss_pulse=True,
        timeout_reached=True,
        reset_on_life_loss=True,
        termination_reason="custom_term",
    )
    assert timeout["boundary_cause"] == "no_reward_timeout"
    assert timeout["reset_cause"] == "no_reward_timeout"
    assert timeout["end_of_episode_pulse"] is True
    assert timeout["should_reset"] is True
    assert timeout["terminated"] is True
    assert timeout["truncated"] is True

    # terminated dominates truncated + life_loss
    terminated = CarmackCompatRunner._resolve_boundary_and_reset(
        step_terminated=True,
        step_truncated=True,
        life_loss_pulse=True,
        timeout_reached=False,
        reset_on_life_loss=True,
        termination_reason="scripted_end",
    )
    assert terminated["boundary_cause"] == "scripted_end"
    assert terminated["reset_cause"] == "terminated"
    assert terminated["end_of_episode_pulse"] is True
    assert terminated["should_reset"] is True
    assert terminated["terminated"] is True
    assert terminated["truncated"] is True

    # truncated dominates life_loss
    truncated = CarmackCompatRunner._resolve_boundary_and_reset(
        step_terminated=False,
        step_truncated=True,
        life_loss_pulse=True,
        timeout_reached=False,
        reset_on_life_loss=True,
        termination_reason="time_limit",
    )
    assert truncated["boundary_cause"] == "time_limit"
    assert truncated["reset_cause"] == "truncated"
    assert truncated["end_of_episode_pulse"] is True
    assert truncated["should_reset"] is True
    assert truncated["terminated"] is False
    assert truncated["truncated"] is True

    # life-loss pulse without reset
    life_loss_no_reset = CarmackCompatRunner._resolve_boundary_and_reset(
        step_terminated=False,
        step_truncated=False,
        life_loss_pulse=True,
        timeout_reached=False,
        reset_on_life_loss=False,
        termination_reason=None,
    )
    assert life_loss_no_reset["boundary_cause"] == "life_loss"
    assert life_loss_no_reset["reset_cause"] is None
    assert life_loss_no_reset["end_of_episode_pulse"] is True
    assert life_loss_no_reset["should_reset"] is False
    assert life_loss_no_reset["terminated"] is False
    assert life_loss_no_reset["truncated"] is False

    # life-loss with reset
    life_loss_reset = CarmackCompatRunner._resolve_boundary_and_reset(
        step_terminated=False,
        step_truncated=False,
        life_loss_pulse=True,
        timeout_reached=False,
        reset_on_life_loss=True,
        termination_reason=None,
    )
    assert life_loss_reset["boundary_cause"] == "life_loss"
    assert life_loss_reset["reset_cause"] == "life_loss_reset"
    assert life_loss_reset["end_of_episode_pulse"] is True
    assert life_loss_reset["should_reset"] is True
    assert life_loss_reset["terminated"] is False
    assert life_loss_reset["truncated"] is True


def test_carmack_runner_cadence_summary_stats():
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0] * 6,
        lives_seq=[3] * 6,
        terminated_at=set(),
        truncated_at=set(),
    )
    # decided action sequence is [0, 1, 1, 2, 2, 2] because frame i uses action from frame i-1.
    agent = RecordingBoundarySequenceAgent(action_sequence=[1, 1, 2, 2, 2, 1])
    config = CarmackRunnerConfig(
        total_frames=6,
        delay_frames=0,
        include_timestamps=False,
        lives_as_episodes=False,
        max_frames_without_reward=999,
        progress_log_interval_frames=0,
        pulse_log_interval=0,
        reset_log_interval=0,
    )
    summary, events, _ = run_with_memory(env, agent, config)

    assert [row["decided_action_idx"] for row in events] == [0, 1, 1, 2, 2, 2]
    assert [row["applied_action_idx"] for row in events] == [0, 1, 1, 2, 2, 2]
    assert [row["decided_action_changed"] for row in events] == [False, True, False, True, False, False]
    assert [row["applied_action_changed"] for row in events] == [False, True, False, True, False, False]
    assert [row["decided_applied_mismatch"] for row in events] == [False, False, False, False, False, False]
    assert [row["applied_action_hold_run_length"] for row in events] == [1, 1, 2, 1, 2, 3]
    assert summary["decided_action_change_count"] == 2
    assert summary["applied_action_change_count"] == 2
    assert summary["decided_applied_mismatch_count"] == 0
    assert summary["decided_action_change_rate"] == 0.4
    assert summary["applied_action_change_rate"] == 0.4
    assert summary["decided_applied_mismatch_rate"] == 0.0
    assert summary["applied_action_hold_run_count"] == 3
    assert summary["applied_action_hold_run_mean"] == 2.0
    assert summary["applied_action_hold_run_max"] == 3
    assert summary["pulse_count"] == 0
    assert summary["boundary_cause_counts"] == {}
    assert summary["reset_cause_counts"] == {}


def test_carmack_runner_emits_required_schema_fields_and_types():
    env = ScriptedEnv(
        action_set=list(range(3)),
        rewards=[1.0, 0.0],
        lives_seq=[3, 3],
        terminated_at={0},
        truncated_at=set(),
    )
    agent = RecordingBoundaryFrameAgent(action_idx=1)
    config = CarmackRunnerConfig(
        total_frames=2,
        delay_frames=0,
        include_timestamps=False,
        max_frames_without_reward=999,
        progress_log_interval_frames=0,
        pulse_log_interval=0,
        reset_log_interval=0,
    )
    _, events, episodes = run_with_memory(env, agent, config)

    assert len(events) == 2
    event = events[0]
    assert event["single_run_profile"] == CARMACK_SINGLE_RUN_PROFILE
    assert event["single_run_schema_version"] == CARMACK_SINGLE_RUN_SCHEMA_VERSION
    assert isinstance(event["frame_idx"], int)
    assert isinstance(event["applied_action_idx"], int)
    assert isinstance(event["decided_action_idx"], int)
    assert isinstance(event["next_policy_action_idx"], int)
    assert isinstance(event["decided_action_changed"], bool)
    assert isinstance(event["applied_action_changed"], bool)
    assert isinstance(event["decided_applied_mismatch"], bool)
    assert isinstance(event["applied_action_hold_run_length"], int)
    assert isinstance(event["reward"], float)
    assert isinstance(event["terminated"], bool)
    assert isinstance(event["truncated"], bool)
    assert isinstance(event["env_terminated"], bool)
    assert isinstance(event["env_truncated"], bool)
    assert isinstance(event["lives"], int)
    assert isinstance(event["episode_idx"], int)
    assert isinstance(event["episode_return"], float)
    assert isinstance(event["episode_length"], int)
    assert isinstance(event["end_of_episode_pulse"], bool)
    assert "pulse_reason" in event
    assert "boundary_cause" in event
    assert "reset_cause" in event
    assert isinstance(event["reset_performed"], bool)
    assert isinstance(event["frames_without_reward"], int)

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["single_run_profile"] == CARMACK_SINGLE_RUN_PROFILE
    assert episode["single_run_schema_version"] == CARMACK_SINGLE_RUN_SCHEMA_VERSION
    assert isinstance(episode["episode_idx"], int)
    assert isinstance(episode["episode_return"], float)
    assert isinstance(episode["length"], int)
    assert isinstance(episode["termination_reason"], str)
    assert isinstance(episode["end_frame_idx"], int)
    assert isinstance(episode["ended_by_reset"], bool)


class _IncrementClock:
    def __init__(self, start: float, step: float) -> None:
        self._t = float(start)
        self._step = float(step)

    def __call__(self) -> float:
        current = float(self._t)
        self._t += self._step
        return current


def _run_determinism_trace(time_fn):
    env = ScriptedEnv(
        action_set=list(range(4)),
        rewards=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        lives_seq=[3, 3, 3, 2, 2, 2],
        terminated_at={1},
        truncated_at={4},
    )
    agent = RecordingBoundarySequenceAgent(action_sequence=[1, 2, 3, 1, 2, 3])
    config = CarmackRunnerConfig(
        total_frames=6,
        delay_frames=1,
        include_timestamps=True,
        lives_as_episodes=True,
        reset_on_life_loss=False,
        max_frames_without_reward=99,
        progress_log_interval_frames=0,
        pulse_log_interval=0,
        reset_log_interval=0,
    )
    return run_with_memory(env, agent, config, time_fn=time_fn)


def _normalize_events(rows):
    normalized = []
    for row in rows:
        payload = dict(row)
        payload.pop("wallclock_time", None)
        normalized.append(payload)
    return normalized


_SUMMARY_INT_KEYS = (
    "frames",
    "episodes_completed",
    "last_episode_idx",
    "last_episode_length",
    "pulse_count",
    "life_loss_pulses",
    "reset_count",
    "game_over_resets",
    "truncated_resets",
    "timeout_resets",
    "life_loss_resets",
    "decided_action_change_count",
    "applied_action_change_count",
    "decided_applied_mismatch_count",
    "applied_action_hold_run_count",
    "applied_action_hold_run_max",
)


@pytest.mark.skipif(
    os.getenv("BENCHMARK_STRICT_RUNTIME_TESTS", "0") != "1",
    reason="strict determinism tier is same-runtime only; enable with BENCHMARK_STRICT_RUNTIME_TESTS=1",
)
def test_carmack_runner_determinism_strict_tier_same_runtime_only():
    summary_a, events_a, _ = _run_determinism_trace(time_fn=_IncrementClock(start=100.0, step=0.01))
    summary_b, events_b, _ = _run_determinism_trace(time_fn=_IncrementClock(start=100.0, step=0.01))

    assert _normalize_events(events_a) == _normalize_events(events_b)
    assert [row.get("wallclock_time") for row in events_a] == [row.get("wallclock_time") for row in events_b]
    for key in _SUMMARY_INT_KEYS:
        assert summary_a[key] == summary_b[key]


def test_carmack_runner_determinism_portable_tier_bounded_drift():
    summary_a, events_a, _ = _run_determinism_trace(time_fn=_IncrementClock(start=100.0, step=0.01))
    summary_b, events_b, _ = _run_determinism_trace(time_fn=_IncrementClock(start=900.0, step=0.015))

    # Portable tier: semantics must match after normalization.
    assert _normalize_events(events_a) == _normalize_events(events_b)
    for key in _SUMMARY_INT_KEYS:
        assert summary_a[key] == summary_b[key]

    # Bounded drift check: wallclock pacing can differ but should remain bounded.
    dts_a = [float(events_a[i]["wallclock_time"] - events_a[i - 1]["wallclock_time"]) for i in range(1, len(events_a))]
    dts_b = [float(events_b[i]["wallclock_time"] - events_b[i - 1]["wallclock_time"]) for i in range(1, len(events_b))]
    assert len(dts_a) == len(dts_b)
    assert max(abs(a - b) for a, b in zip(dts_a, dts_b)) <= 0.01
