from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.score_run import score_run


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")


def _make_base_config() -> dict:
    return {
        "games": ["a", "b"],
        "total_scheduled_frames": 40,
        "schedule": [
            {"visit_idx": 0, "cycle_idx": 0, "game_id": "a", "visit_frames": 10},
            {"visit_idx": 1, "cycle_idx": 0, "game_id": "b", "visit_frames": 10},
            {"visit_idx": 2, "cycle_idx": 1, "game_id": "a", "visit_frames": 10},
            {"visit_idx": 3, "cycle_idx": 1, "game_id": "b", "visit_frames": 10},
        ],
    }


def _make_segments() -> list:
    return [
        {"game_id": "a", "segment_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 3, "return": 1.0, "ended_by": "terminated"},
        {"game_id": "a", "segment_id": 1, "start_global_frame_idx": 4, "end_global_frame_idx": 7, "return": 2.0, "ended_by": "terminated"},
        {"game_id": "a", "segment_id": 2, "start_global_frame_idx": 8, "end_global_frame_idx": 9, "return": 0.0, "ended_by": "truncated"},
        {"game_id": "b", "segment_id": 3, "start_global_frame_idx": 10, "end_global_frame_idx": 13, "return": 10.0, "ended_by": "terminated"},
        {"game_id": "b", "segment_id": 4, "start_global_frame_idx": 14, "end_global_frame_idx": 17, "return": 12.0, "ended_by": "terminated"},
        {"game_id": "b", "segment_id": 5, "start_global_frame_idx": 18, "end_global_frame_idx": 19, "return": 0.0, "ended_by": "truncated"},
        {"game_id": "a", "segment_id": 6, "start_global_frame_idx": 20, "end_global_frame_idx": 23, "return": 3.0, "ended_by": "terminated"},
        {"game_id": "a", "segment_id": 7, "start_global_frame_idx": 24, "end_global_frame_idx": 27, "return": 4.0, "ended_by": "terminated"},
        {"game_id": "a", "segment_id": 8, "start_global_frame_idx": 28, "end_global_frame_idx": 29, "return": 0.0, "ended_by": "truncated"},
        {"game_id": "b", "segment_id": 9, "start_global_frame_idx": 30, "end_global_frame_idx": 33, "return": 14.0, "ended_by": "terminated"},
        {"game_id": "b", "segment_id": 10, "start_global_frame_idx": 34, "end_global_frame_idx": 37, "return": 16.0, "ended_by": "terminated"},
        {"game_id": "b", "segment_id": 11, "start_global_frame_idx": 38, "end_global_frame_idx": 39, "return": 0.0, "ended_by": "truncated"},
    ]


def _make_episodes() -> list:
    return [
        {"game_id": "a", "episode_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 3, "length": 4, "return": 1.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 1, "start_global_frame_idx": 4, "end_global_frame_idx": 7, "length": 4, "return": 2.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 2, "start_global_frame_idx": 8, "end_global_frame_idx": 9, "length": 2, "return": 0.0, "ended_by": "truncated"},
        {"game_id": "b", "episode_id": 3, "start_global_frame_idx": 10, "end_global_frame_idx": 13, "length": 4, "return": 10.0, "ended_by": "terminated"},
        {"game_id": "b", "episode_id": 4, "start_global_frame_idx": 14, "end_global_frame_idx": 17, "length": 4, "return": 12.0, "ended_by": "terminated"},
        {"game_id": "b", "episode_id": 5, "start_global_frame_idx": 18, "end_global_frame_idx": 19, "length": 2, "return": 0.0, "ended_by": "truncated"},
        {"game_id": "a", "episode_id": 6, "start_global_frame_idx": 20, "end_global_frame_idx": 23, "length": 4, "return": 3.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 7, "start_global_frame_idx": 24, "end_global_frame_idx": 27, "length": 4, "return": 4.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 8, "start_global_frame_idx": 28, "end_global_frame_idx": 29, "length": 2, "return": 0.0, "ended_by": "truncated"},
        {"game_id": "b", "episode_id": 9, "start_global_frame_idx": 30, "end_global_frame_idx": 33, "length": 4, "return": 14.0, "ended_by": "terminated"},
        {"game_id": "b", "episode_id": 10, "start_global_frame_idx": 34, "end_global_frame_idx": 37, "length": 4, "return": 16.0, "ended_by": "terminated"},
        {"game_id": "b", "episode_id": 11, "start_global_frame_idx": 38, "end_global_frame_idx": 39, "length": 2, "return": 0.0, "ended_by": "truncated"},
    ]


def _make_events() -> list:
    reward_by_frame = {
        7: 1.0,
        8: 1.0,
        9: 1.0,
        10: 4.0,
        11: 4.0,
        12: 4.0,
        17: 3.0,
        18: 3.0,
        19: 3.0,
        26: 2.0,
        27: 2.0,
        28: 2.0,
        29: 2.0,
        30: 1.0,
        31: 1.0,
        32: 1.0,
        36: 5.0,
        37: 5.0,
        38: 5.0,
        39: 5.0,
    }
    rows = []
    for idx in range(40):
        if idx < 10:
            game_id = "a"
        elif idx < 20:
            game_id = "b"
        elif idx < 30:
            game_id = "a"
        else:
            game_id = "b"
        rows.append(
            {
                "global_frame_idx": idx,
                "game_id": game_id,
                "reward": float(reward_by_frame.get(idx, 0.0)),
                "wallclock_time": idx * 0.1,
            }
        )
    return rows


def test_score_run_metrics(tmp_path):
    run_dir = tmp_path / "run"
    _write_json(run_dir / "config.json", _make_base_config())
    _write_jsonl(run_dir / "segments.jsonl", _make_segments())
    _write_jsonl(run_dir / "episodes.jsonl", _make_episodes())
    _write_jsonl(run_dir / "events.jsonl", _make_events())

    summary = score_run(run_dir, window_frames=4, bottom_k_frac=0.5, revisit_frames=3)

    assert summary["per_game_scores"]["a"] == pytest.approx(2.0)
    assert summary["per_game_scores"]["b"] == pytest.approx(5.0)
    assert summary["per_game_episode_counts"] == {"a": 2, "b": 2}
    assert summary["per_game_visit_frames"] == {"a": 10, "b": 10}

    assert summary["mean_score"] == pytest.approx(3.5)
    assert summary["bottom_k_score"] == pytest.approx(2.0)
    assert summary["final_score"] == pytest.approx(2.75)

    assert summary["forgetting_index_mean"] == pytest.approx(1.5)
    assert summary["forgetting_index_median"] == pytest.approx(1.5)
    assert summary["per_game_forgetting"]["a"] == pytest.approx(1.0)
    assert summary["per_game_forgetting"]["b"] == pytest.approx(2.0)

    assert summary["plasticity_mean"] == pytest.approx(0.0)
    assert summary["plasticity_median"] == pytest.approx(0.0)
    assert summary["per_game_plasticity"]["a"] == pytest.approx(1.0)
    assert summary["per_game_plasticity"]["b"] == pytest.approx(-1.0)

    assert summary["fps"] == pytest.approx(10.0)
    assert summary["frames"] == 40


def test_score_run_last_cycle_fallback_uses_last_visit_per_game(tmp_path):
    run_dir = tmp_path / "run_last_visit_fallback"
    config = {
        "games": ["a", "b"],
        "total_scheduled_frames": 8,
        "schedule": [
            {"visit_idx": 0, "game_id": "a", "visit_frames": 2},
            {"visit_idx": 1, "game_id": "b", "visit_frames": 2},
            {"visit_idx": 2, "game_id": "a", "visit_frames": 2},
            {"visit_idx": 3, "game_id": "a", "visit_frames": 2},
        ],
    }
    segments = [
        {"game_id": "a", "segment_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 1, "return": 1.0, "ended_by": "terminated"},
        {"game_id": "b", "segment_id": 1, "start_global_frame_idx": 2, "end_global_frame_idx": 3, "return": 10.0, "ended_by": "terminated"},
        {"game_id": "a", "segment_id": 2, "start_global_frame_idx": 4, "end_global_frame_idx": 5, "return": 2.0, "ended_by": "terminated"},
        {"game_id": "a", "segment_id": 3, "start_global_frame_idx": 6, "end_global_frame_idx": 7, "return": 3.0, "ended_by": "terminated"},
    ]
    episodes = [
        {"game_id": "a", "episode_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 1, "length": 2, "return": 1.0, "ended_by": "terminated"},
        {"game_id": "b", "episode_id": 1, "start_global_frame_idx": 2, "end_global_frame_idx": 3, "length": 2, "return": 10.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 2, "start_global_frame_idx": 4, "end_global_frame_idx": 5, "length": 2, "return": 2.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 3, "start_global_frame_idx": 6, "end_global_frame_idx": 7, "length": 2, "return": 3.0, "ended_by": "terminated"},
    ]
    events = [
        {"global_frame_idx": 0, "game_id": "a", "reward": 1.0},
        {"global_frame_idx": 1, "game_id": "a", "reward": 1.0},
        {"global_frame_idx": 2, "game_id": "b", "reward": 10.0},
        {"global_frame_idx": 3, "game_id": "b", "reward": 10.0},
        {"global_frame_idx": 4, "game_id": "a", "reward": 2.0},
        {"global_frame_idx": 5, "game_id": "a", "reward": 2.0},
        {"global_frame_idx": 6, "game_id": "a", "reward": 3.0},
        {"global_frame_idx": 7, "game_id": "a", "reward": 3.0},
    ]

    _write_json(run_dir / "config.json", config)
    _write_jsonl(run_dir / "segments.jsonl", segments)
    _write_jsonl(run_dir / "episodes.jsonl", episodes)
    _write_jsonl(run_dir / "events.jsonl", events)

    summary = score_run(run_dir, window_frames=2, bottom_k_frac=0.5, revisit_frames=1)

    assert summary["notes"]["cycle_selection"] == "approx_last_visit_per_game"
    assert summary["per_game_scores"]["a"] == pytest.approx(3.0)
    assert summary["per_game_scores"]["b"] == pytest.approx(10.0)
    assert summary["per_game_episode_counts"] == {"a": 1, "b": 1}
    assert summary["per_game_visit_frames"] == {"a": 2, "b": 2}


def test_score_run_works_without_terminated_episodes(tmp_path):
    run_dir = tmp_path / "run_no_terminated_episodes"
    config = {
        "games": ["a"],
        "total_scheduled_frames": 5,
        "schedule": [{"visit_idx": 0, "cycle_idx": 0, "game_id": "a", "visit_frames": 5}],
    }
    segments = [{"game_id": "a", "segment_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 4, "return": 15.0, "ended_by": "truncated"}]
    episodes = [{"game_id": "a", "episode_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 4, "length": 5, "return": 15.0, "ended_by": "truncated"}]
    events = [
        {"global_frame_idx": 0, "game_id": "a", "reward": 1.0},
        {"global_frame_idx": 1, "game_id": "a", "reward": 2.0},
        {"global_frame_idx": 2, "game_id": "a", "reward": 3.0},
        {"global_frame_idx": 3, "game_id": "a", "reward": 4.0},
        {"global_frame_idx": 4, "game_id": "a", "reward": 5.0},
    ]

    _write_json(run_dir / "config.json", config)
    _write_jsonl(run_dir / "segments.jsonl", segments)
    _write_jsonl(run_dir / "episodes.jsonl", episodes)
    _write_jsonl(run_dir / "events.jsonl", events)

    summary = score_run(run_dir, window_frames=3, bottom_k_frac=1.0, revisit_frames=2)

    assert summary["per_game_scores"]["a"] == pytest.approx(4.0)
    assert summary["mean_score"] == pytest.approx(4.0)
    assert summary["bottom_k_score"] == pytest.approx(4.0)
    assert summary["final_score"] == pytest.approx(4.0)
    assert summary["per_game_episode_counts"] == {"a": 0}
    assert summary["per_game_visit_frames"] == {"a": 5}
    assert summary["plasticity_mean"] == pytest.approx(3.0)
    assert summary["plasticity_median"] == pytest.approx(3.0)
    assert summary["per_game_plasticity"]["a"] == pytest.approx(3.0)
    assert summary["notes"]["games_with_zero_last_cycle_episodes"] == ["a"]


def test_score_run_window_frames_clamped_to_visit_length(tmp_path):
    run_dir = tmp_path / "run_window_clamp"
    config = {
        "games": ["a"],
        "total_scheduled_frames": 3,
        "schedule": [{"visit_idx": 0, "cycle_idx": 0, "game_id": "a", "visit_frames": 3}],
    }
    segments = [{"game_id": "a", "segment_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 2, "return": 6.0, "ended_by": "truncated"}]
    episodes = []
    events = [
        {"global_frame_idx": 0, "game_id": "a", "reward": 1.0},
        {"global_frame_idx": 1, "game_id": "a", "reward": 2.0},
        {"global_frame_idx": 2, "game_id": "a", "reward": 3.0},
    ]

    _write_json(run_dir / "config.json", config)
    _write_jsonl(run_dir / "segments.jsonl", segments)
    _write_jsonl(run_dir / "episodes.jsonl", episodes)
    _write_jsonl(run_dir / "events.jsonl", events)

    summary = score_run(run_dir, window_frames=10, bottom_k_frac=1.0, revisit_frames=10)

    assert summary["per_game_scores"]["a"] == pytest.approx(2.0)
    assert summary["per_game_visit_frames"] == {"a": 3}


def test_score_run_sparse_rewards_treat_missing_frames_as_zero(tmp_path):
    run_dir = tmp_path / "run_sparse_rewards"
    config = {
        "games": ["a"],
        "total_scheduled_frames": 5,
        "schedule": [{"visit_idx": 0, "cycle_idx": 0, "game_id": "a", "visit_frames": 5}],
    }
    segments = [{"game_id": "a", "segment_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 4, "return": 10.0, "ended_by": "truncated"}]
    episodes = []
    events = [
        {"global_frame_idx": 0, "game_id": "a", "reward": 4.0},
        {"global_frame_idx": 4, "game_id": "a", "reward": 6.0},
    ]

    _write_json(run_dir / "config.json", config)
    _write_jsonl(run_dir / "segments.jsonl", segments)
    _write_jsonl(run_dir / "episodes.jsonl", episodes)
    _write_jsonl(run_dir / "events.jsonl", events)

    summary = score_run(run_dir, window_frames=2, bottom_k_frac=1.0, revisit_frames=3)

    assert summary["per_game_scores"]["a"] == pytest.approx(3.0)
    assert summary["per_game_plasticity"]["a"] == pytest.approx((6.0 / 3.0) - (4.0 / 3.0))
