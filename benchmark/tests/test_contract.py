from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.carmack_runner import CARMACK_SINGLE_RUN_PROFILE, CARMACK_SINGLE_RUN_SCHEMA_VERSION
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


def _make_carmack_config() -> dict:
    return {
        "runner_mode": CARMACK_SINGLE_RUN_PROFILE,
        "single_run_profile": CARMACK_SINGLE_RUN_PROFILE,
        "single_run_schema_version": CARMACK_SINGLE_RUN_SCHEMA_VERSION,
        "game": "breakout",
        "seed": 0,
        "frames": 100,
        "default_action_idx": 0,
        "runner_config": {
            "runner_mode": CARMACK_SINGLE_RUN_PROFILE,
            "single_run_schema_version": CARMACK_SINGLE_RUN_SCHEMA_VERSION,
            "action_cadence_mode": "agent_owned",
            "frame_skip_enforced": 1,
            "total_frames": 100,
            "delay_frames": 0,
        },
    }


def _make_carmack_event_row() -> dict:
    return {
        "single_run_profile": CARMACK_SINGLE_RUN_PROFILE,
        "single_run_schema_version": CARMACK_SINGLE_RUN_SCHEMA_VERSION,
        "frame_idx": 0,
        "applied_action_idx": 0,
        "decided_action_idx": 0,
        "next_policy_action_idx": 1,
        "decided_action_changed": False,
        "applied_action_changed": False,
        "decided_applied_mismatch": False,
        "applied_action_hold_run_length": 1,
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "env_terminated": False,
        "env_truncated": False,
        "lives": 3,
        "episode_idx": 0,
        "episode_return": 0.0,
        "episode_length": 1,
        "end_of_episode_pulse": False,
        "pulse_reason": None,
        "boundary_cause": None,
        "reset_cause": None,
        "reset_performed": False,
        "frames_without_reward": 1,
    }


def _make_carmack_episode_row() -> dict:
    return {
        "single_run_profile": CARMACK_SINGLE_RUN_PROFILE,
        "single_run_schema_version": CARMACK_SINGLE_RUN_SCHEMA_VERSION,
        "episode_idx": 0,
        "episode_return": 1.0,
        "length": 10,
        "termination_reason": "terminated",
        "end_frame_idx": 9,
        "ended_by_reset": True,
    }


def _make_carmack_summary() -> dict:
    return {
        "runner_mode": CARMACK_SINGLE_RUN_PROFILE,
        "single_run_profile": CARMACK_SINGLE_RUN_PROFILE,
        "single_run_schema_version": CARMACK_SINGLE_RUN_SCHEMA_VERSION,
        "frames": 100,
        "episodes_completed": 1,
        "last_episode_idx": 1,
        "last_episode_return": 0.0,
        "last_episode_length": 0,
        "pulse_count": 1,
        "boundary_cause_counts": {"terminated": 1},
        "reset_cause_counts": {"terminated": 1},
        "life_loss_pulses": 0,
        "reset_count": 1,
        "game_over_resets": 1,
        "truncated_resets": 0,
        "timeout_resets": 0,
        "life_loss_resets": 0,
        "decided_action_change_count": 1,
        "decided_action_change_rate": 0.01,
        "applied_action_change_count": 1,
        "applied_action_change_rate": 0.01,
        "decided_applied_mismatch_count": 0,
        "decided_applied_mismatch_rate": 0.0,
        "applied_action_hold_run_count": 5,
        "applied_action_hold_run_mean": 2.0,
        "applied_action_hold_run_max": 3,
    }


def test_validate_contract_passes_for_carmack_single_run_schema(tmp_path):
    run_dir = tmp_path / "run_carmack_ok"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_config(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_event_row(), sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_summary(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is True, result["errors"]
    assert result["errors"] == []


def test_validate_contract_fails_for_carmack_bad_event_type(tmp_path):
    run_dir = tmp_path / "run_carmack_bad"
    run_dir.mkdir(parents=True, exist_ok=True)

    bad_event = _make_carmack_event_row()
    bad_event["decided_action_changed"] = "no"

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_config(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(bad_event, sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_summary(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is False
    assert any("events.jsonl row has invalid type/value for 'decided_action_changed'" in err for err in result["errors"])


def test_validate_contract_fails_for_carmack_missing_run_summary(tmp_path):
    run_dir = tmp_path / "run_carmack_missing_summary"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_config(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_event_row(), sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_episode_row(), sort_keys=True) + "\n")

    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is False
    assert any("run_summary.json not found" in err for err in result["errors"])


def test_validate_contract_fails_for_carmack_bad_summary_type(tmp_path):
    run_dir = tmp_path / "run_carmack_bad_summary"
    run_dir.mkdir(parents=True, exist_ok=True)

    bad_summary = _make_carmack_summary()
    bad_summary["frames"] = "100"

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_config(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_event_row(), sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(bad_summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is False
    assert any("run_summary.json key 'frames' must be int" in err for err in result["errors"])


def test_validate_contract_fails_for_carmack_bad_event_semantics(tmp_path):
    run_dir = tmp_path / "run_carmack_bad_semantics"
    run_dir.mkdir(parents=True, exist_ok=True)

    bad_event = _make_carmack_event_row()
    bad_event["reset_performed"] = True
    bad_event["reset_cause"] = None

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_config(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(bad_event, sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_summary(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is False
    assert any("reset_performed must match" in err for err in result["errors"])


def test_validate_contract_fails_for_carmack_sample_vs_summary_inconsistency(tmp_path):
    run_dir = tmp_path / "run_carmack_bad_consistency"
    run_dir.mkdir(parents=True, exist_ok=True)

    event = _make_carmack_event_row()
    event["end_of_episode_pulse"] = True
    event["boundary_cause"] = "terminated"
    event["reset_cause"] = "terminated"
    event["reset_performed"] = True
    event["terminated"] = True

    summary = _make_carmack_summary()
    summary["pulse_count"] = 0
    summary["reset_count"] = 0
    summary["boundary_cause_counts"] = {}
    summary["reset_cause_counts"] = {}

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_config(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is False
    assert any("pulse_count is smaller than sampled event pulse count" in err for err in result["errors"])


def test_validate_contract_fails_for_carmack_bad_config_combination(tmp_path):
    run_dir = tmp_path / "run_carmack_bad_config"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = _make_carmack_config()
    config["frame_skip"] = 4

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_event_row(), sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_summary(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is False
    assert any("config.json frame_skip must be int 1" in err for err in result["errors"])
