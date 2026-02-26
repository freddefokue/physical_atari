from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from benchmark.carmack_multigame_runner import (
    CARMACK_MULTI_RUN_PROFILE,
    CARMACK_MULTI_RUN_SCHEMA_VERSION,
    CarmackMultiGameRunner,
    CarmackMultiGameRunnerConfig,
    FrameFromStepAdapter,
)
from benchmark.runner import EnvStep
from benchmark.schedule import Schedule, ScheduleConfig


class MemoryWriter:
    def __init__(self) -> None:
        self.rows = []

    def write(self, record) -> None:
        self.rows.append(dict(record))


@dataclass
class MockMultiGameEnv:
    action_sets: Dict[str, List[int]]
    terminated_steps: Dict[str, Sequence[int]] | None = None

    def __post_init__(self) -> None:
        self.current_game = None
        self.action_set = []
        self._obs = np.zeros((210, 160, 3), dtype=np.uint8)
        self._lives = 3
        self._episode_step = 0

    def load_game(self, game_id: str):
        self.current_game = str(game_id)
        self.action_set = list(self.action_sets[self.current_game])
        self._episode_step = 0
        self._lives = 3
        return list(self.action_set)

    def reset(self):
        self._episode_step = 0
        self._lives = 3
        return self._obs

    def lives(self) -> int:
        return int(self._lives)

    def step(self, action_idx: int) -> EnvStep:
        del action_idx
        self._episode_step += 1
        terminated = False
        if self.terminated_steps is not None:
            terminated = int(self._episode_step) in set(self.terminated_steps.get(self.current_game, ()))
        if terminated:
            self._lives = max(0, self._lives - 1)
        return EnvStep(
            obs_rgb=self._obs,
            reward=0.0,
            terminated=bool(terminated),
            truncated=False,
            lives=int(self._lives),
            termination_reason="scripted_end" if terminated else None,
        )


class RecordingFrameAgent:
    def __init__(self, action_idx: int = 1) -> None:
        self.action_idx = int(action_idx)
        self.calls = []

    def frame(self, obs_rgb, reward, boundary) -> int:
        del obs_rgb
        self.calls.append(
            {
                "reward": float(reward),
                "boundary": dict(boundary),
            }
        )
        return int(self.action_idx)


def _run_with_memory(env, agent, schedule, config):
    events = MemoryWriter()
    episodes = MemoryWriter()
    segments = MemoryWriter()
    summary = CarmackMultiGameRunner(
        env=env,
        agent=agent,
        schedule=schedule,
        config=config,
        event_writer=events,
        episode_writer=episodes,
        segment_writer=segments,
    ).run()
    return summary, events.rows, episodes.rows, segments.rows


def test_carmack_multigame_runner_emits_required_boundary_payload():
    schedule = Schedule(
        ScheduleConfig(games=["a"], base_visit_frames=3, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8))}, terminated_steps={"a": [2]})
    agent = RecordingFrameAgent(action_idx=1)
    config = CarmackMultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=0,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
    )
    summary, events, _, _ = _run_with_memory(env, agent, schedule, config)

    assert summary["frames"] == 3
    assert len(agent.calls) == 3
    first_boundary = agent.calls[0]["boundary"]
    assert set(first_boundary.keys()) == {
        "terminated",
        "truncated",
        "end_of_episode_pulse",
        "has_prev_applied_action",
        "prev_applied_action_idx",
        "global_frame_idx",
    }
    assert first_boundary["has_prev_applied_action"] is True
    assert [call["boundary"]["prev_applied_action_idx"] for call in agent.calls] == [
        row["applied_action_idx"] for row in events
    ]
    assert agent.calls[1]["boundary"]["end_of_episode_pulse"] is True
    assert events[1]["boundary_cause"] == "terminated"
    assert events[1]["env_termination_reason"] == "scripted_end"
    assert events[0]["multi_run_profile"] == CARMACK_MULTI_RUN_PROFILE
    assert events[0]["multi_run_schema_version"] == CARMACK_MULTI_RUN_SCHEMA_VERSION


def test_carmack_multigame_runner_visit_switch_emits_truncated_boundary():
    schedule = Schedule(
        ScheduleConfig(games=["a", "b"], base_visit_frames=2, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8)), "b": list(range(8))})
    agent = RecordingFrameAgent(action_idx=1)
    config = CarmackMultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=0,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
    )
    summary, events, episodes, segments = _run_with_memory(env, agent, schedule, config)

    switch_frames = [row["global_frame_idx"] for row in events if row["boundary_cause"] == "visit_switch"]
    assert switch_frames == [1, 3]
    assert events[1]["reset_performed"] is True
    assert events[1]["reset_cause"] == "visit_switch"
    assert [row["ended_by"] for row in episodes] == ["truncated", "truncated"]
    assert [row["ended_by"] for row in segments] == ["truncated", "truncated"]
    assert summary["boundary_cause_counts"].get("visit_switch") == 2
    assert summary["reset_cause_counts"].get("visit_switch") == 2


def test_carmack_multigame_step_adapter_supplies_required_info_fields():
    class _StepAgent:
        def __init__(self) -> None:
            self.calls = []

        def step(self, obs_rgb, reward, terminated, truncated, info):
            del obs_rgb, reward, terminated, truncated
            self.calls.append(dict(info))
            return 1

    schedule = Schedule(
        ScheduleConfig(games=["a"], base_visit_frames=4, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8))})
    step_agent = _StepAgent()
    agent = FrameFromStepAdapter(step_agent, decision_interval=3)
    config = CarmackMultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=0,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
    )
    _, _, _, _ = _run_with_memory(env, agent, schedule, config)

    assert len(step_agent.calls) == 4
    assert all("has_prev_applied_action" in call for call in step_agent.calls)
    assert all("prev_applied_action_idx" in call for call in step_agent.calls)
    assert [bool(call["is_decision_frame"]) for call in step_agent.calls] == [True, False, False, True]


def test_carmack_multigame_delay_queue_persistence_across_visit_switch():
    schedule = Schedule(
        ScheduleConfig(games=["a", "b"], base_visit_frames=3, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8)), "b": list(range(8))})
    agent = RecordingFrameAgent(action_idx=1)

    # Persist queue on visit switch.
    config_persist = CarmackMultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=2,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
        reset_delay_queue_on_visit_switch=False,
    )
    _, events_persist, _, _ = _run_with_memory(env, agent, schedule, config_persist)

    # Reset queue on visit switch.
    env_reset = MockMultiGameEnv(action_sets={"a": list(range(8)), "b": list(range(8))})
    agent_reset = RecordingFrameAgent(action_idx=1)
    config_reset = CarmackMultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=2,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
        reset_delay_queue_on_visit_switch=True,
    )
    _, events_reset, _, _ = _run_with_memory(env_reset, agent_reset, schedule, config_reset)

    # Frame 3 is first frame of second visit.
    assert events_persist[3]["applied_action_idx"] == 1
    assert events_reset[3]["applied_action_idx"] == 0


def test_carmack_multigame_carries_last_decision_across_visit_switch():
    schedule = Schedule(
        ScheduleConfig(games=["a", "b"], base_visit_frames=1, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8)), "b": list(range(8))})

    class _SequenceFrameAgent:
        def __init__(self) -> None:
            self._actions = [5, 6]
            self._idx = 0

        def frame(self, obs_rgb, reward, boundary) -> int:
            del obs_rgb, reward, boundary
            action = self._actions[min(self._idx, len(self._actions) - 1)]
            self._idx += 1
            return int(action)

    agent = _SequenceFrameAgent()
    config = CarmackMultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=0,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
    )
    _, events, _, _ = _run_with_memory(env, agent, schedule, config)

    # The action produced on the last frame of visit 0 should become the
    # decided action on the first frame of visit 1 (not overwritten to default).
    assert events[0]["next_policy_action_idx"] == 5
    assert events[1]["decided_action_idx"] == 5
