from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.contract import BENCHMARK_CONTRACT_VERSION, compute_contract_hash, resolve_scoring_defaults
from benchmark.validate_contract import validate_contract


def _make_config() -> dict:
    return {
        "games": ["ms_pacman", "centipede"],
        "schedule": [
            {"visit_idx": 0, "cycle_idx": 0, "game_id": "ms_pacman", "visit_frames": 100},
            {"visit_idx": 1, "cycle_idx": 0, "game_id": "centipede", "visit_frames": 100},
        ],
        "decision_interval": 4,
        "delay": 2,
        "sticky": 0.25,
        "life_loss_termination": True,
        "full_action_space": True,
        "default_action_idx": 0,
        "action_mapping_policy": {
            "global_action_set": list(range(18)),
        },
        "scoring_defaults": {
            "window_episodes": 20,
            "bottom_k_frac": 0.25,
            "revisit_episodes": 5,
            "final_score_weights": [0.5, 0.5],
        },
    }


def test_contract_hash_stable_across_key_reordering():
    config_a = _make_config()
    config_b = {
        "default_action_idx": 0,
        "full_action_space": True,
        "action_mapping_policy": {"global_action_set": list(range(18))},
        "sticky": 0.25,
        "delay": 2,
        "decision_interval": 4,
        "schedule": [
            {"game_id": "ms_pacman", "visit_frames": 100, "cycle_idx": 0, "visit_idx": 0},
            {"visit_frames": 100, "visit_idx": 1, "game_id": "centipede", "cycle_idx": 0},
        ],
        "life_loss_termination": True,
        "games": ["ms_pacman", "centipede"],
        "scoring_defaults": {
            "bottom_k_frac": 0.25,
            "revisit_episodes": 5,
            "final_score_weights": [0.5, 0.5],
            "window_episodes": 20,
        },
    }

    assert compute_contract_hash(config_a) == compute_contract_hash(config_b)


def test_contract_hash_changes_when_contract_input_changes():
    config_a = _make_config()
    config_b = _make_config()
    config_b["decision_interval"] = 8

    assert compute_contract_hash(config_a) != compute_contract_hash(config_b)


def test_validate_contract_passes_for_real_smoke_run(tmp_path):
    source_run = Path("runs/random_30k_smoke/multigame_seed0_20260215_140650")
    if not source_run.exists():
        pytest.skip(f"missing fixture run directory: {source_run}")

    config_path = source_run / "config.json"
    score_path = source_run / "score.json"
    events_path = source_run / "events.jsonl"
    episodes_path = source_run / "episodes.jsonl"
    segments_path = source_run / "segments.jsonl"
    if (
        not config_path.exists()
        or not score_path.exists()
        or not events_path.exists()
        or not episodes_path.exists()
        or not segments_path.exists()
    ):
        pytest.skip(f"missing required source artifacts in {source_run}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    with score_path.open("r", encoding="utf-8") as fh:
        score = json.load(fh)

    config["scoring_defaults"] = resolve_scoring_defaults(config)
    config["benchmark_contract_version"] = BENCHMARK_CONTRACT_VERSION
    config["benchmark_contract_hash"] = compute_contract_hash(config)

    score["benchmark_contract_version"] = config["benchmark_contract_version"]
    score["benchmark_contract_hash"] = config["benchmark_contract_hash"]

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "score.json").open("w", encoding="utf-8") as fh:
        json.dump(score, fh, indent=2, sort_keys=True)
        fh.write("\n")

    sample_lines = 5
    with events_path.open("r", encoding="utf-8") as src, (run_dir / "events.jsonl").open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if idx >= sample_lines:
                break
            dst.write(line)
    with episodes_path.open("r", encoding="utf-8") as src, (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if idx >= sample_lines:
                break
            dst.write(line)
    with segments_path.open("r", encoding="utf-8") as src, (run_dir / "segments.jsonl").open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if idx >= sample_lines:
                break
            dst.write(line)

    result = validate_contract(run_dir, sample_event_lines=sample_lines)
    assert result["ok"] is True, result["errors"]
    assert result["errors"] == []


def test_validate_contract_fails_on_stale_hash(tmp_path):
    config = _make_config()
    config["benchmark_contract_version"] = BENCHMARK_CONTRACT_VERSION
    config["benchmark_contract_hash"] = "0" * 64
    score = {
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "benchmark_contract_hash": "0" * 64,
    }

    run_dir = tmp_path / "run_bad_hash"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "score.json").open("w", encoding="utf-8") as fh:
        json.dump(score, fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=0)
    assert result["ok"] is False
    assert any("canonical contract input" in error for error in result["errors"])


def test_validate_contract_fails_on_missing_score_keys(tmp_path):
    config = _make_config()
    config["benchmark_contract_version"] = BENCHMARK_CONTRACT_VERSION
    config["benchmark_contract_hash"] = compute_contract_hash(config)
    score = {
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "benchmark_contract_hash": config["benchmark_contract_hash"],
    }

    run_dir = tmp_path / "run_bad_score_schema"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "score.json").open("w", encoding="utf-8") as fh:
        json.dump(score, fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=0)
    assert result["ok"] is False
    assert any("score.json missing key: final_score" in error for error in result["errors"])
