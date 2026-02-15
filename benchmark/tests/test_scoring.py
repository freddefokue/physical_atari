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
        {"game_id": "b", "episode_id": 2, "start_global_frame_idx": 10, "end_global_frame_idx": 13, "length": 4, "return": 10.0, "ended_by": "terminated"},
        {"game_id": "b", "episode_id": 3, "start_global_frame_idx": 14, "end_global_frame_idx": 17, "length": 4, "return": 12.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 4, "start_global_frame_idx": 20, "end_global_frame_idx": 23, "length": 4, "return": 3.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 5, "start_global_frame_idx": 24, "end_global_frame_idx": 27, "length": 4, "return": 4.0, "ended_by": "terminated"},
        {"game_id": "b", "episode_id": 6, "start_global_frame_idx": 30, "end_global_frame_idx": 33, "length": 4, "return": 14.0, "ended_by": "terminated"},
        {"game_id": "b", "episode_id": 7, "start_global_frame_idx": 34, "end_global_frame_idx": 37, "length": 4, "return": 16.0, "ended_by": "terminated"},
    ]


def _make_events() -> list:
    rows = []
    for idx in range(40):
        rows.append({"global_frame_idx": idx, "game_id": "a" if idx < 20 else "b", "reward": 0.0, "wallclock_time": idx * 0.1})
    return rows


def test_score_run_metrics(tmp_path):
    run_dir = tmp_path / "run"
    _write_json(run_dir / "config.json", _make_base_config())
    _write_jsonl(run_dir / "segments.jsonl", _make_segments())
    _write_jsonl(run_dir / "episodes.jsonl", _make_episodes())
    _write_jsonl(run_dir / "events.jsonl", _make_events())

    summary = score_run(run_dir, window_episodes=2, bottom_k_frac=0.5, revisit_episodes=1)

    assert summary["per_game_scores"]["a"] == pytest.approx(3.5)
    assert summary["per_game_scores"]["b"] == pytest.approx(15.0)
    assert summary["per_game_episode_counts"] == {"a": 2, "b": 2}

    assert summary["mean_score"] == pytest.approx(9.25)
    assert summary["bottom_k_score"] == pytest.approx(3.5)
    assert summary["final_score"] == pytest.approx(6.375)

    assert summary["forgetting_index_mean"] == pytest.approx(-1.5)
    assert summary["forgetting_index_median"] == pytest.approx(-1.5)
    assert summary["per_game_forgetting"]["a"] == pytest.approx(-1.0)
    assert summary["per_game_forgetting"]["b"] == pytest.approx(-2.0)

    assert summary["plasticity_mean"] == pytest.approx(1.5)
    assert summary["plasticity_median"] == pytest.approx(1.5)
    assert summary["per_game_plasticity"]["a"] == pytest.approx(1.0)
    assert summary["per_game_plasticity"]["b"] == pytest.approx(2.0)

    assert summary["fps"] == pytest.approx(10.0)
    assert summary["frames"] == 40


def test_score_run_falls_back_to_segments_when_episodes_are_ambiguous(tmp_path):
    run_dir = tmp_path / "run_ambiguous"
    config = {
        "games": ["a"],
        "total_scheduled_frames": 6,
        "schedule": [{"visit_idx": 0, "cycle_idx": 0, "game_id": "a", "visit_frames": 6}],
    }
    segments = [
        {"game_id": "a", "segment_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 1, "return": 1.0, "ended_by": "terminated"},
        {"game_id": "a", "segment_id": 1, "start_global_frame_idx": 2, "end_global_frame_idx": 3, "return": 2.0, "ended_by": "terminated"},
        {"game_id": "a", "segment_id": 2, "start_global_frame_idx": 4, "end_global_frame_idx": 5, "return": 0.0, "ended_by": "truncated"},
    ]
    episodes = [
        {"game_id": "a", "episode_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 1, "length": 2, "return": 50.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 1, "start_global_frame_idx": 2, "end_global_frame_idx": 3, "length": 2, "return": 60.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 2, "start_global_frame_idx": 4, "end_global_frame_idx": 5, "length": 2, "return": 999.0, "ended_by": "truncated"},
    ]

    _write_json(run_dir / "config.json", config)
    _write_jsonl(run_dir / "segments.jsonl", segments)
    _write_jsonl(run_dir / "episodes.jsonl", episodes)
    _write_jsonl(run_dir / "events.jsonl", [{"global_frame_idx": i, "reward": 0.0} for i in range(6)])

    summary = score_run(run_dir, window_episodes=2, bottom_k_frac=1.0, revisit_episodes=1)

    assert summary["per_game_scores"]["a"] == pytest.approx(1.5)
    assert summary["notes"]["episodes_jsonl_truncated_entries_detected"] is True
    assert "segments_terminated_fallback" in summary["notes"]["episode_source"]


def test_score_run_last_cycle_fallback_uses_last_visit_per_game(tmp_path):
    run_dir = tmp_path / "run_last_visit_fallback"
    config = {
        "games": ["a", "b"],
        "total_scheduled_frames": 8,
        # no cycle_idx on purpose
        "schedule": [
            {"visit_idx": 0, "game_id": "a", "visit_frames": 2},
            {"visit_idx": 1, "game_id": "b", "visit_frames": 2},
            {"visit_idx": 2, "game_id": "a", "visit_frames": 2},
            {"visit_idx": 3, "game_id": "a", "visit_frames": 2},
        ],
    }
    segments = [
        {"game_id": "a", "segment_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 1, "return": 1.0, "ended_by": "terminated"},
        {"game_id": "b", "segment_id": 1, "start_global_frame_idx": 2, "end_global_frame_idx": 3, "return": 100.0, "ended_by": "terminated"},
        {"game_id": "a", "segment_id": 2, "start_global_frame_idx": 4, "end_global_frame_idx": 5, "return": 2.0, "ended_by": "terminated"},
        {"game_id": "a", "segment_id": 3, "start_global_frame_idx": 6, "end_global_frame_idx": 7, "return": 3.0, "ended_by": "terminated"},
    ]
    episodes = [
        {"game_id": "a", "episode_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 1, "length": 2, "return": 1.0, "ended_by": "terminated"},
        {"game_id": "b", "episode_id": 1, "start_global_frame_idx": 2, "end_global_frame_idx": 3, "length": 2, "return": 100.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 2, "start_global_frame_idx": 4, "end_global_frame_idx": 5, "length": 2, "return": 2.0, "ended_by": "terminated"},
        {"game_id": "a", "episode_id": 3, "start_global_frame_idx": 6, "end_global_frame_idx": 7, "length": 2, "return": 3.0, "ended_by": "terminated"},
    ]

    _write_json(run_dir / "config.json", config)
    _write_jsonl(run_dir / "segments.jsonl", segments)
    _write_jsonl(run_dir / "episodes.jsonl", episodes)
    _write_jsonl(run_dir / "events.jsonl", [{"global_frame_idx": i, "reward": 0.0} for i in range(8)])

    summary = score_run(run_dir, window_episodes=1, bottom_k_frac=0.5, revisit_episodes=1)

    assert summary["notes"]["cycle_selection"] == "approx_last_visit_per_game"
    assert summary["per_game_scores"]["a"] == pytest.approx(3.0)
    assert summary["per_game_scores"]["b"] == pytest.approx(100.0)
    assert summary["per_game_episode_counts"] == {"a": 1, "b": 1}


def test_score_run_marks_unassigned_episodes(tmp_path):
    run_dir = tmp_path / "run_unassigned"
    config = {
        "games": ["a"],
        "total_scheduled_frames": 4,
        "schedule": [{"visit_idx": 0, "cycle_idx": 0, "game_id": "a", "visit_frames": 4}],
    }
    segments = [
        {"game_id": "a", "segment_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 3, "return": 1.0, "ended_by": "terminated"},
    ]
    episodes = [
        {"game_id": "a", "episode_id": 0, "start_global_frame_idx": 0, "end_global_frame_idx": 3, "length": 4, "return": 1.0, "ended_by": "terminated"},
        # outside visit range -> should be unassigned and excluded
        {"game_id": "a", "episode_id": 1, "start_global_frame_idx": 100, "end_global_frame_idx": 101, "length": 2, "return": 999.0, "ended_by": "terminated"},
    ]

    _write_json(run_dir / "config.json", config)
    _write_jsonl(run_dir / "segments.jsonl", segments)
    _write_jsonl(run_dir / "episodes.jsonl", episodes)
    _write_jsonl(run_dir / "events.jsonl", [{"global_frame_idx": i, "reward": 0.0} for i in range(4)])

    summary = score_run(run_dir, window_episodes=10, bottom_k_frac=1.0, revisit_episodes=1)

    assert summary["per_game_scores"]["a"] == pytest.approx(1.0)
    assert summary["notes"]["unassigned_episode_count"] == 1
    assert len(summary["notes"]["unassigned_episode_examples"]) == 1
