from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from benchmark.carmack_runner import CarmackCompatRunner, CarmackRunnerConfig
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


def run_with_memory(env, agent, config: CarmackRunnerConfig):
    events = MemoryWriter()
    episodes = MemoryWriter()
    summary = CarmackCompatRunner(env=env, agent=agent, config=config, event_writer=events, episode_writer=episodes).run()
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
    assert summary["episodes_completed"] == 2
    assert len(episodes) == 2
    assert [row["pulse_reason"] for row in events] == [None, "no_reward_timeout", None, "no_reward_timeout", None]
    assert [row["frames_without_reward"] for row in events] == [1, 2, 1, 2, 1]


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
