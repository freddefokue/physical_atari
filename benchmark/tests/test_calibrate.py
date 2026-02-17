from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.calibrate import aggregate_summary, evaluate_smoke_expectations, merge_config, resolve_config_path


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
        fh.write("\n")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")


def _fake_success_run(tmp_path: Path, name: str, *, frames: int, final: float, mean: float, bottom: float) -> dict:
    run_dir = tmp_path / name
    config = {
        "games": ["a"],
        "total_scheduled_frames": int(frames),
        "schedule": [{"visit_idx": 0, "cycle_idx": 0, "game_id": "a", "visit_frames": int(frames)}],
    }
    events = [{"global_frame_idx": idx, "reward": 0.0} for idx in range(frames)]
    episodes = [
        {
            "game_id": "a",
            "episode_id": 0,
            "start_global_frame_idx": 0,
            "end_global_frame_idx": max(0, frames - 1),
            "length": int(frames),
            "return": 1.0,
            "ended_by": "terminated",
        }
    ]
    score = {
        "final_score": float(final),
        "mean_score": float(mean),
        "bottom_k_score": float(bottom),
        "fps": 100.0,
        "frames": int(frames),
        "forgetting_index_mean": 0.0,
        "plasticity_mean": 0.0,
    }

    _write_json(run_dir / "config.json", config)
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_jsonl(run_dir / "episodes.jsonl", episodes)
    _write_json(run_dir / "score.json", score)

    return {
        "status": "success",
        "run_dir": str(run_dir),
        "score": score,
    }


def test_merge_config_overrides_nested_dicts():
    base = {
        "num_cycles": 1,
        "agent_config": {"lr": 1e-4, "gamma": 0.99},
        "scoring_defaults": {"window_episodes": 20, "bottom_k_frac": 0.25},
    }
    overrides = {
        "num_cycles": 3,
        "agent_config": {"lr": 3e-4},
        "scoring_defaults": {"revisit_episodes": 5},
    }

    merged = merge_config(base, overrides)

    assert merged["num_cycles"] == 3
    assert merged["agent_config"]["lr"] == pytest.approx(3e-4)
    assert merged["agent_config"]["gamma"] == pytest.approx(0.99)
    assert merged["scoring_defaults"] == {
        "window_episodes": 20,
        "bottom_k_frac": 0.25,
        "revisit_episodes": 5,
    }


def test_aggregate_summary_stats_math():
    run_results = [
        {
            "agent": "random",
            "seed": 0,
            "status": "success",
            "score": {"final_score": 2.0, "mean_score": 3.0, "bottom_k_score": 1.0, "fps": 100.0, "frames": 10},
        },
        {
            "agent": "random",
            "seed": 1,
            "status": "success",
            "score": {"final_score": 4.0, "mean_score": 5.0, "bottom_k_score": 2.0, "fps": 120.0, "frames": 14},
        },
        {
            "agent": "repeat",
            "seed": 0,
            "status": "failed",
            "score": None,
        },
    ]

    summary = aggregate_summary(run_results, agents=["random", "repeat"])

    random_stats = summary["per_agent"]["random"]
    assert random_stats["counts"] == {"requested": 2, "success": 2, "failed": 0, "skipped": 0}
    assert random_stats["final_score_stats"]["mean"] == pytest.approx(3.0)
    assert random_stats["final_score_stats"]["median"] == pytest.approx(3.0)
    assert random_stats["final_score_stats"]["min"] == pytest.approx(2.0)
    assert random_stats["final_score_stats"]["max"] == pytest.approx(4.0)
    assert random_stats["fps_mean"] == pytest.approx(110.0)
    assert random_stats["frames_mean"] == pytest.approx(12.0)

    repeat_stats = summary["per_agent"]["repeat"]
    assert repeat_stats["counts"] == {"requested": 1, "success": 0, "failed": 1, "skipped": 0}
    assert repeat_stats["final_score_stats"] is None


def test_smoke_expectations_pass(tmp_path):
    repeat_run = _fake_success_run(tmp_path, "repeat_seed0", frames=6, final=1.0, mean=2.0, bottom=1.0)
    repeat_run.update({"agent": "repeat", "seed": 0})

    random_run = _fake_success_run(tmp_path, "random_seed0", frames=6, final=2.0, mean=3.0, bottom=1.5)
    random_run.update({"agent": "random", "seed": 0})

    results = [
        repeat_run,
        random_run,
        {
            "agent": "tinydqn",
            "seed": 0,
            "status": "skipped",
            "reason": "torch_not_installed",
            "run_dir": None,
            "score": None,
        },
    ]

    expectations = evaluate_smoke_expectations(results)
    assert expectations["passed"] is True
    assert expectations["errors"] == []


def test_smoke_expectations_fail_when_random_underperforms_repeat(tmp_path):
    repeat_run = _fake_success_run(tmp_path, "repeat_seed0", frames=6, final=5.0, mean=6.0, bottom=5.0)
    repeat_run.update({"agent": "repeat", "seed": 0})

    random_run = _fake_success_run(tmp_path, "random_seed0", frames=6, final=2.0, mean=3.0, bottom=2.0)
    random_run.update({"agent": "random", "seed": 0})

    expectations = evaluate_smoke_expectations([repeat_run, random_run])
    assert expectations["passed"] is False
    assert any("RandomAgent mean final_score" in msg for msg in expectations["errors"])


def test_resolve_config_path_is_repo_relative_when_no_override():
    path = resolve_config_path(None, "configs/v1_smoke.json")
    assert path.exists()
    assert path.name == "v1_smoke.json"
