from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from benchmark.multigame_runner import MultiGameRunner, MultiGameRunnerConfig
from benchmark.runner import EnvStep
from benchmark.schedule import Schedule, ScheduleConfig


class MemoryWriter:
    def __init__(self) -> None:
        self.rows = []

    def write(self, record) -> None:
        assert isinstance(record, dict)
        self.rows.append(record)


class CountingAgent:
    """Returns 1,2,3... (wrapped) and captures info payloads."""

    def __init__(self) -> None:
        self._next = 1
        self.seen_infos = []

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        del obs_rgb, reward, terminated, truncated
        self.seen_infos.append(dict(info))
        action_space_n = int(info["action_space_n"])
        action = self._next % action_space_n
        self._next += 1
        return action


@dataclass
class MockMultiGameEnv:
    action_sets: Dict[str, List[int]]
    episode_lengths: Optional[Dict[str, int]] = None
    truncated_steps: Optional[Dict[str, Sequence[int]]] = None
    reward_per_step: float = 0.0

    def __post_init__(self) -> None:
        self.current_game = None
        self.action_set = []
        self._obs = np.zeros((210, 160, 3), dtype=np.uint8)
        self._lives = 3
        self._episode_step = 0
        self.applied_actions = []

    def load_game(self, game_id: str):
        if game_id not in self.action_sets:
            raise ValueError(f"Unknown game {game_id}")
        self.current_game = game_id
        self.action_set = list(self.action_sets[game_id])
        self._lives = 3
        self._episode_step = 0
        return list(self.action_set)

    def reset(self):
        self._episode_step = 0
        self._lives = 3
        return self._obs

    def lives(self) -> int:
        return self._lives

    def step(self, action_idx: int) -> EnvStep:
        if action_idx < 0 or action_idx >= len(self.action_set):
            raise ValueError("action_idx out of bounds")

        self.applied_actions.append(int(action_idx))
        self._episode_step += 1
        terminated = False
        if self.episode_lengths is not None:
            game_limit = self.episode_lengths.get(self.current_game)
            if game_limit is not None and self._episode_step >= game_limit:
                terminated = True
                self._lives = max(0, self._lives - 1)
        truncated = False
        if self.truncated_steps is not None:
            game_steps = self.truncated_steps.get(self.current_game, ())
            if self._episode_step in game_steps:
                truncated = True

        return EnvStep(
            obs_rgb=self._obs,
            reward=float(self.reward_per_step),
            terminated=terminated,
            truncated=truncated,
            lives=self._lives,
            termination_reason="scripted_end" if terminated else None,
        )


def run_multigame(env, agent, schedule, config):
    events = MemoryWriter()
    episodes = MemoryWriter()
    segments = MemoryWriter()
    summary = MultiGameRunner(
        env=env,
        agent=agent,
        schedule=schedule,
        config=config,
        event_writer=events,
        episode_writer=episodes,
        segment_writer=segments,
    ).run()
    return summary, events.rows, episodes.rows, segments.rows


def test_schedule_determinism():
    cfg = ScheduleConfig(
        games=["ms_pacman", "centipede", "qbert"],
        base_visit_frames=100,
        num_cycles=3,
        seed=7,
        jitter_pct=0.1,
        min_visit_frames=15,
    )
    schedule_a = Schedule(cfg)
    schedule_b = Schedule(cfg)
    schedule_c = Schedule(
        ScheduleConfig(
            games=cfg.games,
            base_visit_frames=100,
            num_cycles=3,
            seed=8,
            jitter_pct=0.1,
            min_visit_frames=15,
        )
    )

    assert schedule_a.as_records() == schedule_b.as_records()
    assert schedule_a.as_records() != schedule_c.as_records()



def test_visit_boundary_truncation_occurs_only_on_last_frame():
    schedule = Schedule(
        ScheduleConfig(
            games=["a", "b"],
            base_visit_frames=3,
            num_cycles=1,
            seed=0,
            jitter_pct=0.0,
            min_visit_frames=1,
        )
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8)), "b": list(range(8))})
    agent = CountingAgent()
    config = MultiGameRunnerConfig(decision_interval=1, delay_frames=0, include_timestamps=False, global_action_set=tuple(range(8)))

    _, events, episodes, segments = run_multigame(env, agent, schedule, config)

    assert len(events) == 6
    truncated_frames = [row["global_frame_idx"] for row in events if row["truncated"]]
    assert truncated_frames == [2, 5]
    for row in events:
        if row["global_frame_idx"] not in (2, 5):
            assert row["truncated"] is False

    assert len(segments) == 2
    assert [row["ended_by"] for row in segments] == ["truncated", "truncated"]
    assert len(episodes) == 2
    assert [row["ended_by"] for row in episodes] == ["truncated", "truncated"]



def test_episode_and_segment_counters_follow_spec():
    schedule = Schedule(
        ScheduleConfig(games=["a"], base_visit_frames=5, num_cycles=1, seed=1, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(
        action_sets={"a": list(range(8))},
        episode_lengths={"a": 2},
        reward_per_step=1.0,
    )
    agent = CountingAgent()
    config = MultiGameRunnerConfig(decision_interval=1, delay_frames=0, include_timestamps=False, global_action_set=tuple(range(8)))

    summary, events, episodes, segments = run_multigame(env, agent, schedule, config)

    assert [row["episode_id"] for row in events] == [0, 0, 1, 1, 2]
    assert [row["segment_id"] for row in events] == [0, 0, 1, 1, 2]

    assert len(episodes) == 3
    assert [row["episode_id"] for row in episodes] == [0, 1, 2]
    assert [row["ended_by"] for row in episodes] == ["terminated", "terminated", "truncated"]

    assert len(segments) == 3
    assert [row["ended_by"] for row in segments] == ["terminated", "terminated", "truncated"]

    assert summary["episodes_completed"] == 3
    assert summary["segments_completed"] == 3


def test_episode_return_resets_across_visit_switches():
    schedule = Schedule(
        ScheduleConfig(games=["a", "b"], base_visit_frames=3, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(
        action_sets={"a": list(range(8)), "b": list(range(8))},
        episode_lengths=None,
        reward_per_step=1.0,
    )
    agent = CountingAgent()
    config = MultiGameRunnerConfig(decision_interval=1, delay_frames=0, include_timestamps=False, global_action_set=tuple(range(8)))

    _, events, episodes, _ = run_multigame(env, agent, schedule, config)

    assert [row["episode_id"] for row in events] == [0, 0, 0, 1, 1, 1]
    assert [row["episode_return_so_far"] for row in events] == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    assert [row["ended_by"] for row in episodes] == ["truncated", "truncated"]



def test_delay_queue_resets_on_switch_or_reset():
    schedule = Schedule(
        ScheduleConfig(games=["a", "b"], base_visit_frames=4, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(16)), "b": list(range(16))})
    agent = CountingAgent()
    config = MultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=2,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(16)),
    )

    _, events, _, _ = run_multigame(env, agent, schedule, config)

    segment_starts = [idx for idx, row in enumerate(events) if idx == 0 or row["segment_id"] != events[idx - 1]["segment_id"]]
    assert segment_starts == [0, 4]
    for start in segment_starts:
        assert events[start]["applied_action_idx"] == 0
        assert events[start + 1]["applied_action_idx"] == 0



def test_forbidden_info_keys_not_passed_to_agent():
    schedule = Schedule(
        ScheduleConfig(games=["a", "b"], base_visit_frames=3, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8)), "b": list(range(8))})
    agent = CountingAgent()
    config = MultiGameRunnerConfig(decision_interval=1, delay_frames=0, include_timestamps=False, global_action_set=tuple(range(8)))

    run_multigame(env, agent, schedule, config)

    forbidden = {
        "game_id",
        "visit_idx",
        "cycle_idx",
        "visit_frame_idx",
        "frames_remaining",
        "visit_frames",
        "episode_id",
        "segment_id",
    }
    assert agent.seen_infos
    for info in agent.seen_infos:
        assert forbidden.isdisjoint(set(info.keys()))
        assert {
            "lives",
            "is_decision_frame",
            "action_space_n",
            "default_action_idx",
            "prev_applied_action_idx",
            "has_prev_applied_action",
        }.issubset(set(info.keys()))



def test_is_decision_frame_changes_only_by_decision_interval():
    schedule = Schedule(
        ScheduleConfig(games=["a"], base_visit_frames=8, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(16))})
    agent = CountingAgent()
    config = MultiGameRunnerConfig(
        decision_interval=4,
        delay_frames=0,
        include_timestamps=False,
        global_action_set=tuple(range(16)),
    )

    _, events, _, _ = run_multigame(env, agent, schedule, config)
    assert [row["is_decision_frame"] for row in events] == [True, False, False, False, True, False, False, False]


def test_prev_applied_action_delay0():
    schedule = Schedule(
        ScheduleConfig(games=["a"], base_visit_frames=6, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8))})
    agent = CountingAgent()
    config = MultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=0,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
    )

    _, events, _, _ = run_multigame(env, agent, schedule, config)
    assert len(agent.seen_infos) == len(events) == 6
    assert agent.seen_infos[0]["has_prev_applied_action"] is False
    assert agent.seen_infos[0]["prev_applied_action_idx"] == 0
    for frame_idx in range(1, len(events)):
        assert agent.seen_infos[frame_idx]["has_prev_applied_action"] is True
        assert agent.seen_infos[frame_idx]["prev_applied_action_idx"] == events[frame_idx - 1]["applied_action_idx"]


def test_prev_applied_action_delayK():
    schedule = Schedule(
        ScheduleConfig(games=["a"], base_visit_frames=8, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8))})
    agent = CountingAgent()
    config = MultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=3,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
    )

    _, events, _, _ = run_multigame(env, agent, schedule, config)
    assert [row["applied_action_idx"] for row in events] == [0, 0, 0, 1, 2, 3, 4, 5]
    assert agent.seen_infos[0]["has_prev_applied_action"] is False
    assert agent.seen_infos[0]["prev_applied_action_idx"] == 0
    for frame_idx in range(1, len(events)):
        assert agent.seen_infos[frame_idx]["has_prev_applied_action"] is True
        assert agent.seen_infos[frame_idx]["prev_applied_action_idx"] == events[frame_idx - 1]["applied_action_idx"]


def test_prev_applied_action_resets_on_truncation_switch():
    schedule = Schedule(
        ScheduleConfig(games=["a", "b"], base_visit_frames=5, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8)), "b": list(range(8))})
    agent = CountingAgent()
    config = MultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=0,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
    )

    _, events, _, _ = run_multigame(env, agent, schedule, config)
    assert len(events) == 10
    assert agent.seen_infos[0]["has_prev_applied_action"] is False
    assert agent.seen_infos[5]["has_prev_applied_action"] is False
    assert agent.seen_infos[5]["prev_applied_action_idx"] == 0


def test_prev_applied_action_resets_on_termination_reset_mid_visit():
    schedule = Schedule(
        ScheduleConfig(games=["a"], base_visit_frames=6, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8))}, episode_lengths={"a": 3})
    agent = CountingAgent()
    config = MultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=0,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
    )

    _, events, _, _ = run_multigame(env, agent, schedule, config)
    terminated_indices = [idx for idx, row in enumerate(events) if row["terminated"]]
    assert terminated_indices
    first_terminated = terminated_indices[0]
    next_frame = first_terminated + 1
    assert next_frame < len(events)
    assert agent.seen_infos[next_frame]["has_prev_applied_action"] is False
    assert agent.seen_infos[next_frame]["prev_applied_action_idx"] == 0


def test_env_truncation_mid_visit_is_treated_as_terminated_only():
    schedule = Schedule(
        ScheduleConfig(games=["a"], base_visit_frames=5, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8))}, truncated_steps={"a": [2]})
    agent = CountingAgent()
    config = MultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=0,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
    )

    _, events, episodes, segments = run_multigame(env, agent, schedule, config)
    assert len(events) == 5
    assert events[1]["terminated"] is True
    assert events[1]["truncated"] is False
    assert [row["global_frame_idx"] for row in events if row["truncated"]] == [4]
    assert episodes[-1]["ended_by"] == "truncated"
    assert segments[-1]["ended_by"] == "truncated"
    assert all(row["ended_by"] == "terminated" for row in episodes[:-1])
    assert all(row["ended_by"] == "terminated" for row in segments[:-1])


def test_episode_return_logging_emits_at_configured_interval(capsys):
    schedule = Schedule(
        ScheduleConfig(games=["a"], base_visit_frames=5, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    env = MockMultiGameEnv(action_sets={"a": list(range(8))}, episode_lengths={"a": 2}, reward_per_step=1.0)
    agent = CountingAgent()
    config = MultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=0,
        default_action_idx=0,
        include_timestamps=False,
        global_action_set=tuple(range(8)),
        episode_log_interval=2,
    )

    run_multigame(env, agent, schedule, config)
    captured = capsys.readouterr()
    assert "[episode] episode_id=0 game=a return=2.000000" in captured.out
    assert "[episode] episode_id=1 " not in captured.out
    assert "[episode] episode_id=2 game=a return=1.000000" in captured.out
