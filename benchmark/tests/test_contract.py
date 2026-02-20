from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.carmack_multigame_runner import CARMACK_MULTI_RUN_PROFILE, CARMACK_MULTI_RUN_SCHEMA_VERSION
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


def _make_multigame_score(contract_hash: str) -> dict:
    return {
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "benchmark_contract_hash": contract_hash,
        "final_score": 0.0,
        "mean_score": 0.0,
        "bottom_k_score": 0.0,
        "forgetting_index_mean": None,
        "forgetting_index_median": None,
        "plasticity_mean": None,
        "plasticity_median": None,
        "fps": 10.0,
        "frames": 4,
        "per_game_scores": {"breakout": 0.0},
        "per_game_episode_counts": {"breakout": 2},
        "per_game_forgetting": {},
        "per_game_plasticity": {},
    }


def _make_carmack_multigame_config() -> dict:
    config = {
        "games": ["breakout"],
        "schedule": [
            {"visit_idx": 0, "cycle_idx": 0, "game_id": "breakout", "visit_frames": 4},
        ],
        "decision_interval": 1,
        "delay": 0,
        "sticky": 0.0,
        "life_loss_termination": False,
        "full_action_space": False,
        "default_action_idx": 0,
        "runner_mode": CARMACK_MULTI_RUN_PROFILE,
        "multi_run_profile": CARMACK_MULTI_RUN_PROFILE,
        "multi_run_schema_version": CARMACK_MULTI_RUN_SCHEMA_VERSION,
        "action_mapping_policy": {"global_action_set": list(range(4))},
        "runner_config": {
            "decision_interval": 1,
            "delay_frames": 0,
            "default_action_idx": 0,
            "episode_log_interval": 0,
            "include_timestamps": False,
            "runner_mode": CARMACK_MULTI_RUN_PROFILE,
            "action_cadence_mode": "agent_owned",
            "frame_skip_enforced": 1,
            "multi_run_schema_version": CARMACK_MULTI_RUN_SCHEMA_VERSION,
            "reset_delay_queue_on_reset": True,
            "reset_delay_queue_on_visit_switch": True,
        },
        "scoring_defaults": {
            "window_episodes": 20,
            "bottom_k_frac": 0.25,
            "revisit_episodes": 5,
            "final_score_weights": [0.5, 0.5],
        },
        "total_scheduled_frames": 4,
    }
    config["benchmark_contract_version"] = BENCHMARK_CONTRACT_VERSION
    config["benchmark_contract_hash"] = compute_contract_hash(config)
    return config


def _make_carmack_multigame_event_row() -> dict:
    return {
        "multi_run_profile": CARMACK_MULTI_RUN_PROFILE,
        "multi_run_schema_version": CARMACK_MULTI_RUN_SCHEMA_VERSION,
        "frame_idx": 0,
        "global_frame_idx": 0,
        "game_id": "breakout",
        "visit_idx": 0,
        "cycle_idx": 0,
        "visit_frame_idx": 0,
        "episode_id": 0,
        "segment_id": 0,
        "is_decision_frame": True,
        "decided_action_idx": 0,
        "applied_action_idx": 0,
        "next_policy_action_idx": 1,
        "applied_action_idx_local": 0,
        "applied_ale_action": 0,
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "env_terminated": False,
        "env_truncated": False,
        "lives": 3,
        "episode_return_so_far": 0.0,
        "segment_return_so_far": 0.0,
        "end_of_episode_pulse": False,
        "boundary_cause": None,
        "reset_cause": None,
        "reset_performed": False,
        "env_termination_reason": None,
    }


def _make_carmack_multigame_episode_row() -> dict:
    return {
        "multi_run_profile": CARMACK_MULTI_RUN_PROFILE,
        "multi_run_schema_version": CARMACK_MULTI_RUN_SCHEMA_VERSION,
        "game_id": "breakout",
        "episode_id": 0,
        "start_global_frame_idx": 0,
        "end_global_frame_idx": 3,
        "length": 4,
        "return": 0.0,
        "ended_by": "truncated",
        "boundary_cause": "visit_switch",
    }


def _make_carmack_multigame_segment_row() -> dict:
    return {
        "multi_run_profile": CARMACK_MULTI_RUN_PROFILE,
        "multi_run_schema_version": CARMACK_MULTI_RUN_SCHEMA_VERSION,
        "game_id": "breakout",
        "segment_id": 0,
        "start_global_frame_idx": 0,
        "end_global_frame_idx": 3,
        "length": 4,
        "return": 0.0,
        "ended_by": "truncated",
        "boundary_cause": "visit_switch",
    }


def _make_carmack_multigame_summary() -> dict:
    return {
        "frames": 4,
        "episodes_completed": 1,
        "segments_completed": 1,
        "last_episode_id": 1,
        "last_segment_id": 1,
        "visits_completed": 1,
        "total_scheduled_frames": 4,
        "boundary_cause_counts": {"visit_switch": 1},
        "reset_cause_counts": {"visit_switch": 1},
        "reset_count": 1,
        "agent_stats": {},
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


def test_validate_contract_passes_for_carmack_multigame_schema(tmp_path):
    run_dir = tmp_path / "run_carmack_multigame_ok"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = _make_carmack_multigame_config()
    score = _make_multigame_score(config["benchmark_contract_hash"])
    e0 = _make_carmack_multigame_event_row()
    e1 = dict(e0)
    e1["frame_idx"] = 1
    e1["global_frame_idx"] = 1
    e1["visit_frame_idx"] = 1
    e1["applied_action_idx"] = 1
    e1["next_policy_action_idx"] = 2
    e1["episode_return_so_far"] = 0.5
    e1["segment_return_so_far"] = 0.5
    e2 = dict(e1)
    e2["frame_idx"] = 2
    e2["global_frame_idx"] = 2
    e2["visit_frame_idx"] = 2
    e2["applied_action_idx"] = 2
    e2["next_policy_action_idx"] = 3
    e2["episode_return_so_far"] = 1.0
    e2["segment_return_so_far"] = 1.0
    e3 = dict(e2)
    e3["frame_idx"] = 3
    e3["global_frame_idx"] = 3
    e3["visit_frame_idx"] = 3
    e3["applied_action_idx"] = 3
    e3["next_policy_action_idx"] = 0
    e3["end_of_episode_pulse"] = True
    e3["truncated"] = True
    e3["boundary_cause"] = "visit_switch"
    e3["reset_cause"] = "visit_switch"
    e3["reset_performed"] = True

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "score.json").open("w", encoding="utf-8") as fh:
        json.dump(score, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        for row in (e0, e1, e2, e3):
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_multigame_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "segments.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_multigame_segment_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_multigame_summary(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=4, validation_mode="full")
    assert result["ok"] is True, result["errors"]
    assert result["errors"] == []


def test_validate_contract_fails_for_carmack_multigame_bad_boundary_cause(tmp_path):
    run_dir = tmp_path / "run_carmack_multigame_bad_cause"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = _make_carmack_multigame_config()
    score = _make_multigame_score(config["benchmark_contract_hash"])
    bad_event = _make_carmack_multigame_event_row()
    bad_event["end_of_episode_pulse"] = True
    bad_event["truncated"] = True
    bad_event["boundary_cause"] = "scripted_end"
    bad_event["reset_cause"] = "truncated"
    bad_event["reset_performed"] = True
    bad_event["env_truncated"] = True

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "score.json").open("w", encoding="utf-8") as fh:
        json.dump(score, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(bad_event, sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_multigame_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "segments.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_multigame_segment_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_multigame_summary(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is False
    assert any("boundary_cause must be one of" in err for err in result["errors"])


def test_validate_contract_fails_for_carmack_multigame_reset_consistency(tmp_path):
    run_dir = tmp_path / "run_carmack_multigame_bad_reset"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = _make_carmack_multigame_config()
    score = _make_multigame_score(config["benchmark_contract_hash"])
    bad_event = _make_carmack_multigame_event_row()
    bad_event["end_of_episode_pulse"] = True
    bad_event["truncated"] = True
    bad_event["boundary_cause"] = "visit_switch"
    bad_event["reset_cause"] = "visit_switch"
    bad_event["reset_performed"] = False

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "score.json").open("w", encoding="utf-8") as fh:
        json.dump(score, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(bad_event, sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_multigame_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "segments.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_multigame_segment_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_multigame_summary(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is False
    assert any("reset_performed must match" in err for err in result["errors"])


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


def _make_carmack_runtime_fingerprint(config: dict) -> dict:
    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "fingerprint_schema_version": "runtime_fingerprint_v1",
        "runner_mode": CARMACK_SINGLE_RUN_PROFILE,
        "single_run_profile": CARMACK_SINGLE_RUN_PROFILE,
        "single_run_schema_version": CARMACK_SINGLE_RUN_SCHEMA_VERSION,
        "game": str(config["game"]),
        "seed": int(config["seed"]),
        "seed_policy": "global_seed_python_numpy_ale",
        "frames": int(config["frames"]),
        "config_sha256_algorithm": "sha256",
        "config_sha256_scope": "config_without_runtime_fingerprint",
        "config_sha256": config_sha,
        "python_version": "3.12.0",
        "ale_py_version": "0.10.1",
        "rom_sha256": "0" * 64,
        "platform": "linux-test",
        "machine": "x86_64",
        "processor": "x86_64",
        "python_implementation": "CPython",
        "python_executable": "/usr/bin/python3",
        "numpy_version": "1.0.0",
        "torch_version": "2.0.0",
        "cuda_available": False,
        "rom_path": "/tmp/fake_rom.bin",
    }


def _write_carmack_runtime_fingerprint_from_run_dir(run_dir: Path, *, drop_keys: tuple[str, ...] = ()) -> None:
    with (run_dir / "config.json").open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    fingerprint = _make_carmack_runtime_fingerprint(config)
    for key in drop_keys:
        fingerprint.pop(key, None)
    config_with_fp = dict(config)
    config_with_fp["runtime_fingerprint"] = dict(fingerprint)
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config_with_fp, fh, indent=2, sort_keys=True)
        fh.write("\n")

    run_summary_path = run_dir / "run_summary.json"
    if run_summary_path.exists():
        with run_summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        summary_with_fp = dict(summary)
        summary_with_fp["runtime_fingerprint"] = dict(fingerprint)
        with run_summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary_with_fp, fh, indent=2, sort_keys=True)
            fh.write("\n")

    with (run_dir / "runtime_fingerprint.json").open("w", encoding="utf-8") as fh:
        json.dump(fingerprint, fh, indent=2, sort_keys=True)
        fh.write("\n")


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

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
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

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
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

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
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

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
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

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
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

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
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

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is False
    assert any("config.json frame_skip must be int 1" in err for err in result["errors"])


def test_validate_contract_full_mode_catches_unsampled_bad_row(tmp_path):
    run_dir = tmp_path / "run_carmack_full_mode"
    run_dir.mkdir(parents=True, exist_ok=True)

    events = [_make_carmack_event_row(), _make_carmack_event_row()]
    events[1]["decided_action_changed"] = "bad_type"

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_config(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        for row in events:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_summary(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
    sample_result = validate_contract(run_dir, sample_event_lines=1, validation_mode="sample")
    assert sample_result["ok"] is True, sample_result["errors"]

    full_result = validate_contract(run_dir, sample_event_lines=1, validation_mode="full")
    assert full_result["ok"] is False
    assert any("events.jsonl row has invalid type/value for 'decided_action_changed'" in err for err in full_result["errors"])


def test_validate_contract_stratified_mode_catches_rare_boundary_cause_row(tmp_path):
    run_dir = tmp_path / "run_carmack_stratified_mode"
    run_dir.mkdir(parents=True, exist_ok=True)

    normal = _make_carmack_event_row()
    terminated = _make_carmack_event_row()
    terminated["frame_idx"] = 1
    terminated["end_of_episode_pulse"] = True
    terminated["boundary_cause"] = "terminated"
    terminated["reset_cause"] = "terminated"
    terminated["reset_performed"] = True
    terminated["terminated"] = True
    terminated["env_terminated"] = True
    terminated["truncated"] = False

    timeout_bad = _make_carmack_event_row()
    timeout_bad["frame_idx"] = 2
    timeout_bad["end_of_episode_pulse"] = True
    timeout_bad["boundary_cause"] = "no_reward_timeout"
    timeout_bad["reset_cause"] = "no_reward_timeout"
    timeout_bad["reset_performed"] = True
    timeout_bad["truncated"] = False
    timeout_bad["env_truncated"] = False

    summary = _make_carmack_summary()
    summary["pulse_count"] = 2
    summary["reset_count"] = 2
    summary["boundary_cause_counts"] = {"terminated": 1, "no_reward_timeout": 1}
    summary["reset_cause_counts"] = {"terminated": 1, "no_reward_timeout": 1}

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_config(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        for row in (normal, terminated, timeout_bad):
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
    sample_result = validate_contract(run_dir, sample_event_lines=1, validation_mode="sample")
    assert sample_result["ok"] is True, sample_result["errors"]

    stratified_result = validate_contract(
        run_dir,
        sample_event_lines=3,
        validation_mode="stratified",
        stratified_seed=123,
    )
    assert stratified_result["ok"] is False
    assert any("no_reward_timeout reset requires truncated=true" in err for err in stratified_result["errors"])


def test_validate_contract_stratified_mode_requires_positive_budget(tmp_path):
    run_dir = tmp_path / "run_carmack_stratified_zero_budget"
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

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
    result = validate_contract(run_dir, sample_event_lines=0, validation_mode="stratified")
    assert result["ok"] is False
    assert any("stratified validation requires sample_event_lines > 0" in err for err in result["errors"])


def test_validate_contract_stratified_mode_fails_when_budget_too_small(tmp_path):
    run_dir = tmp_path / "run_carmack_stratified_small_budget"
    run_dir.mkdir(parents=True, exist_ok=True)

    normal = _make_carmack_event_row()
    second = _make_carmack_event_row()
    second["frame_idx"] = 1

    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_config(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(normal, sort_keys=True) + "\n")
        fh.write(json.dumps(second, sort_keys=True) + "\n")
    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_carmack_episode_row(), sort_keys=True) + "\n")
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_make_carmack_summary(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    _write_carmack_runtime_fingerprint_from_run_dir(run_dir)
    result = validate_contract(run_dir, sample_event_lines=1, validation_mode="stratified")
    assert result["ok"] is False
    assert any("stratified validation budget too small" in err for err in result["errors"])


def test_validate_contract_runtime_fingerprint_missing_informational_keys_warns(tmp_path):
    run_dir = tmp_path / "run_carmack_fingerprint_warn"
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
    _write_carmack_runtime_fingerprint_from_run_dir(run_dir, drop_keys=("torch_version",))

    result = validate_contract(run_dir, sample_event_lines=1)
    assert result["ok"] is True
    assert result["errors"] == []
    assert any("runtime_fingerprint missing informational key: torch_version" in warn for warn in result["warnings"])


def test_validate_contract_fail_on_warnings_rejects_informational_warning(tmp_path):
    run_dir = tmp_path / "run_carmack_fingerprint_warn_fail"
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
    _write_carmack_runtime_fingerprint_from_run_dir(run_dir, drop_keys=("torch_version",))

    result = validate_contract(run_dir, sample_event_lines=1, fail_on_warnings=True)
    assert result["ok"] is False
    assert result["errors"] == []
    assert any("runtime_fingerprint missing informational key: torch_version" in warn for warn in result["warnings"])
