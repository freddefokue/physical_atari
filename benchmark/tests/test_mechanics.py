from __future__ import annotations

import json

import numpy as np

from benchmark.agents import FakeSequenceAgent, RepeatActionAgent
from benchmark.logging_utils import JsonlWriter
from benchmark.runner import BenchmarkRunner, EnvStep, RunnerConfig


class MemoryWriter:
    def __init__(self) -> None:
        self.rows = []

    def write(self, record) -> None:
        assert isinstance(record, dict)
        self.rows.append(record)


class MockEnv:
    def __init__(self, num_actions=16, episode_length=None, truncate_length=None, reward_per_step=0.0):
        self.action_set = list(range(num_actions))
        self.episode_length = episode_length
        self.truncate_length = truncate_length
        self.reward_per_step = float(reward_per_step)
        self.reset_count = 0
        self._episode_step = 0
        self._lives = 3
        self._obs = np.zeros((210, 160, 3), dtype=np.uint8)

    def lives(self) -> int:
        return self._lives

    def reset(self):
        self.reset_count += 1
        self._episode_step = 0
        self._lives = 3
        return self._obs

    def step(self, action_idx: int) -> EnvStep:
        del action_idx
        self._episode_step += 1
        terminated = False
        truncated = False
        if self.episode_length is not None and self._episode_step >= self.episode_length:
            terminated = True
            self._lives = max(0, self._lives - 1)
        elif self.truncate_length is not None and self._episode_step >= self.truncate_length:
            truncated = True

        reason = None
        if terminated:
            reason = "scripted_end"
        elif truncated:
            reason = "time_limit"

        return EnvStep(
            obs_rgb=self._obs,
            reward=self.reward_per_step,
            terminated=terminated,
            truncated=truncated,
            lives=self._lives,
            termination_reason=reason,
        )


class BoundaryAgent:
    """Returns action 1,2,3... per frame-skip window using frame_idx."""

    def __init__(self, frame_skip: int) -> None:
        self.frame_skip = frame_skip

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        del obs_rgb, reward, terminated, truncated
        return 1 + (info["frame_idx"] // self.frame_skip)


class StreamingCallAgent:
    """Captures step-call cadence and frame indexes passed to agent."""

    def __init__(self, action_idx: int = 0) -> None:
        self.action_idx = action_idx
        self.call_count = 0
        self.seen_frame_idx = []

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        del obs_rgb, reward, terminated, truncated
        self.call_count += 1
        self.seen_frame_idx.append(info["frame_idx"])
        return self.action_idx


def run_with_memory(env, agent, config: RunnerConfig):
    events = MemoryWriter()
    episodes = MemoryWriter()
    BenchmarkRunner(env=env, agent=agent, config=config, event_writer=events, episode_writer=episodes).run()
    return events.rows, episodes.rows


def test_delay_queue_correctness():
    env = MockEnv(num_actions=16)
    agent = FakeSequenceAgent(start=1)
    config = RunnerConfig(total_frames=8, frame_skip=1, delay_frames=3, default_action_idx=0, include_timestamps=False)
    events, _ = run_with_memory(env, agent, config)

    applied = [row["applied_action_idx"] for row in events]
    assert applied == [0, 0, 0, 1, 2, 3, 4, 5]


def test_frameskip_correctness():
    frame_skip = 4
    env = MockEnv(num_actions=16)
    agent = BoundaryAgent(frame_skip=frame_skip)
    config = RunnerConfig(total_frames=12, frame_skip=frame_skip, delay_frames=0, include_timestamps=False)
    events, _ = run_with_memory(env, agent, config)

    decided = [row["decided_action_idx"] for row in events]
    applied = [row["applied_action_idx"] for row in events]
    assert decided == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert applied == decided


def test_combined_delay_and_frameskip():
    frame_skip = 4
    delay = 6
    env = MockEnv(num_actions=32)
    agent = BoundaryAgent(frame_skip=frame_skip)
    config = RunnerConfig(
        total_frames=16,
        frame_skip=frame_skip,
        delay_frames=delay,
        default_action_idx=0,
        include_timestamps=False,
    )
    events, _ = run_with_memory(env, agent, config)

    decided = [row["decided_action_idx"] for row in events]
    applied = [row["applied_action_idx"] for row in events]
    assert applied == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
    for t in range(delay):
        assert applied[t] == 0
    for t in range(delay, config.total_frames):
        assert applied[t] == decided[t - delay]
    for t in range(1, config.total_frames):
        if t % frame_skip != 0:
            assert decided[t] == decided[t - 1]


def test_agent_is_called_every_frame_streaming():
    total_frames = 17
    frame_skip = 4
    env = MockEnv(num_actions=8)
    agent = StreamingCallAgent(action_idx=0)
    config = RunnerConfig(
        total_frames=total_frames,
        frame_skip=frame_skip,
        delay_frames=3,
        include_timestamps=False,
    )
    run_with_memory(env, agent, config)

    assert agent.call_count == total_frames
    assert agent.seen_frame_idx == list(range(total_frames))


def test_logging_completeness(tmp_path):
    env = MockEnv(num_actions=8)
    agent = RepeatActionAgent(action_idx=0)
    config = RunnerConfig(total_frames=100, frame_skip=1, delay_frames=0, include_timestamps=False)

    events_path = tmp_path / "events.jsonl"
    episodes_path = tmp_path / "episodes.jsonl"
    with JsonlWriter(events_path) as events_writer, JsonlWriter(episodes_path) as episodes_writer:
        BenchmarkRunner(
            env=env,
            agent=agent,
            config=config,
            event_writer=events_writer,
            episode_writer=episodes_writer,
        ).run()

    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 100
    required_fields = {
        "frame_idx",
        "applied_action_idx",
        "decided_action_idx",
        "reward",
        "terminated",
        "truncated",
        "lives",
        "episode_idx",
    }
    parsed = [json.loads(line) for line in lines]
    for row in parsed:
        assert required_fields.issubset(set(row.keys()))
        assert isinstance(row["lives"], int)
    assert [row["frame_idx"] for row in parsed] == list(range(100))
    episode_idx = [row["episode_idx"] for row in parsed]
    assert episode_idx == sorted(episode_idx)


def test_reset_reinitializes_delay_queue_and_episode_counters():
    env = MockEnv(num_actions=64, episode_length=3, reward_per_step=1.0)
    agent = FakeSequenceAgent(start=1)
    config = RunnerConfig(total_frames=9, frame_skip=1, delay_frames=2, default_action_idx=0, include_timestamps=False)
    events, episodes = run_with_memory(env, agent, config)

    assert env.reset_count == 4
    assert len(episodes) == 3

    episode_starts = [idx for idx, row in enumerate(events) if idx == 0 or row["episode_idx"] != events[idx - 1]["episode_idx"]]
    assert episode_starts == [0, 3, 6]

    # After each reset, the delay queue should be [0, 0], so first two applied actions are 0.
    for start in episode_starts:
        assert events[start]["applied_action_idx"] == 0
        assert events[start + 1]["applied_action_idx"] == 0
        assert events[start]["episode_return"] == 1.0
        assert events[start + 1]["episode_return"] == 2.0

    expected_end_frames = [2, 5, 8]
    for idx, row in enumerate(episodes):
        assert row["episode_idx"] == idx
        assert row["length"] == 3
        assert row["episode_return"] == 3.0
        assert row["termination_reason"] == "scripted_end"
        assert row["end_frame_idx"] == expected_end_frames[idx]

    for ep_idx in range(3):
        ep_events = [row for row in events if row["episode_idx"] == ep_idx]
        assert len(ep_events) == 3
        assert ep_events[-1]["terminated"] is True
        assert ep_events[-1]["truncated"] is False
        assert ep_events[-1]["episode_return"] == episodes[ep_idx]["episode_return"]


def test_truncation_plumbing_and_episode_logging():
    env = MockEnv(num_actions=8, truncate_length=4, reward_per_step=0.5)
    agent = RepeatActionAgent(action_idx=0)
    config = RunnerConfig(total_frames=8, frame_skip=1, delay_frames=0, include_timestamps=False)
    events, episodes = run_with_memory(env, agent, config)

    assert len(episodes) == 2
    assert [row["end_frame_idx"] for row in episodes] == [3, 7]
    for row in episodes:
        assert row["length"] == 4
        assert row["episode_return"] == 2.0
        assert row["termination_reason"] == "time_limit"

    assert events[3]["truncated"] is True
    assert events[3]["terminated"] is False
    assert events[7]["truncated"] is True
    assert events[7]["terminated"] is False
    assert all(row["lives"] == 3 for row in events)
