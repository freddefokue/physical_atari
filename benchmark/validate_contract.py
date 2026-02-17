"""Validation helpers for benchmark v1 contract tags and core artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from benchmark.contract import BENCHMARK_CONTRACT_VERSION, compute_contract_hash


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _check_config(config: Mapping[str, Any], errors: List[str]) -> None:
    if not isinstance(config.get("benchmark_contract_version"), str):
        errors.append("config.json missing string key: benchmark_contract_version")
    if not isinstance(config.get("benchmark_contract_hash"), str):
        errors.append("config.json missing string key: benchmark_contract_hash")

    games = config.get("games")
    if not isinstance(games, list) or not games or any(not isinstance(game, str) for game in games):
        errors.append("config.json games must be a non-empty list[str]")

    schedule = config.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        errors.append("config.json schedule must be a non-empty list")
    else:
        required_schedule_keys = {"visit_idx", "cycle_idx", "game_id", "visit_frames"}
        for idx, row in enumerate(schedule[:5]):
            if not isinstance(row, dict):
                errors.append(f"config.json schedule[{idx}] must be an object")
                continue
            missing = required_schedule_keys - set(row.keys())
            if missing:
                errors.append(f"config.json schedule[{idx}] missing keys: {sorted(missing)}")
                continue
            if not _is_int(row["visit_idx"]) or not _is_int(row["cycle_idx"]) or not _is_int(row["visit_frames"]):
                errors.append(f"config.json schedule[{idx}] has invalid integer fields")
            if not isinstance(row["game_id"], str):
                errors.append(f"config.json schedule[{idx}].game_id must be string")

    if not _is_int(config.get("decision_interval")):
        errors.append("config.json decision_interval must be int")

    delay = config.get("delay")
    runner_cfg = config.get("runner_config")
    if not _is_int(delay):
        if not isinstance(runner_cfg, dict) or not _is_int(runner_cfg.get("delay_frames")):
            errors.append("config.json delay (or runner_config.delay_frames) must be int")

    if not _is_number(config.get("sticky")):
        errors.append("config.json sticky must be numeric")
    if not isinstance(config.get("life_loss_termination"), bool):
        errors.append("config.json life_loss_termination must be bool")
    if not isinstance(config.get("full_action_space"), bool):
        errors.append("config.json full_action_space must be bool")
    if not _is_int(config.get("default_action_idx")):
        errors.append("config.json default_action_idx must be int")

    action_mapping_policy = config.get("action_mapping_policy")
    if not isinstance(action_mapping_policy, dict):
        errors.append("config.json action_mapping_policy must be object")
    else:
        global_action_set = action_mapping_policy.get("global_action_set")
        if not isinstance(global_action_set, list) or any(not _is_int(action) for action in global_action_set):
            errors.append("config.json action_mapping_policy.global_action_set must be list[int]")

    scoring_defaults = config.get("scoring_defaults")
    if not isinstance(scoring_defaults, dict):
        errors.append("config.json scoring_defaults must be object")
    else:
        if not _is_int(scoring_defaults.get("window_episodes")):
            errors.append("config.json scoring_defaults.window_episodes must be int")
        if not _is_number(scoring_defaults.get("bottom_k_frac")):
            errors.append("config.json scoring_defaults.bottom_k_frac must be numeric")
        if not _is_int(scoring_defaults.get("revisit_episodes")):
            errors.append("config.json scoring_defaults.revisit_episodes must be int")
        weights = scoring_defaults.get("final_score_weights")
        if (
            not isinstance(weights, Sequence)
            or isinstance(weights, (str, bytes))
            or len(weights) != 2
            or any(not _is_number(weight) for weight in weights)
        ):
            errors.append("config.json scoring_defaults.final_score_weights must be [number, number]")


def _check_score_tags(config: Mapping[str, Any], score: Mapping[str, Any], errors: List[str]) -> None:
    config_version = config.get("benchmark_contract_version")
    config_hash = config.get("benchmark_contract_hash")
    score_version = score.get("benchmark_contract_version")
    score_hash = score.get("benchmark_contract_hash")

    if not isinstance(score_version, str):
        errors.append("score.json missing string key: benchmark_contract_version")
    if not isinstance(score_hash, str):
        errors.append("score.json missing string key: benchmark_contract_hash")

    if score_version != config_version:
        errors.append("score.json benchmark_contract_version does not match config.json")
    if score_hash != config_hash:
        errors.append("score.json benchmark_contract_hash does not match config.json")


def _check_score_schema(score: Mapping[str, Any], errors: List[str]) -> None:
    numeric_or_none_keys = (
        "final_score",
        "mean_score",
        "bottom_k_score",
        "forgetting_index_mean",
        "forgetting_index_median",
        "plasticity_mean",
        "plasticity_median",
        "fps",
    )
    for key in numeric_or_none_keys:
        if key not in score:
            errors.append(f"score.json missing key: {key}")
            continue
        value = score.get(key)
        if value is not None and not _is_number(value):
            errors.append(f"score.json key '{key}' must be numeric or null")

    if "frames" not in score:
        errors.append("score.json missing key: frames")
    elif not _is_int(score.get("frames")):
        errors.append("score.json key 'frames' must be int")

    object_keys = (
        "per_game_scores",
        "per_game_episode_counts",
        "per_game_forgetting",
        "per_game_plasticity",
    )
    for key in object_keys:
        if key not in score:
            errors.append(f"score.json missing key: {key}")
            continue
        if not isinstance(score.get(key), dict):
            errors.append(f"score.json key '{key}' must be object")


def _check_config_hash_integrity(config: Mapping[str, Any], errors: List[str]) -> None:
    version = config.get("benchmark_contract_version")
    observed = config.get("benchmark_contract_hash")
    if not isinstance(version, str) or not isinstance(observed, str):
        return
    if version != BENCHMARK_CONTRACT_VERSION:
        errors.append(
            f"config.json benchmark_contract_version must be '{BENCHMARK_CONTRACT_VERSION}' for this validator"
        )
        return
    expected = compute_contract_hash(config)
    if observed != expected:
        errors.append("config.json benchmark_contract_hash does not match canonical contract input")


def _check_jsonl_schema_sample(
    path: Path,
    sample_lines: int,
    errors: List[str],
    *,
    record_name: str,
    required_keys: Sequence[str],
    field_validators: Optional[Mapping[str, Callable[[Any], bool]]] = None,
) -> None:
    if sample_lines <= 0:
        return
    if not path.exists():
        errors.append(f"{record_name} not found")
        return

    checked = 0
    required = set(required_keys)
    for row in _iter_jsonl(path):
        if not isinstance(row, dict):
            errors.append(f"{record_name} contains a non-object row")
            return
        missing = required - set(row.keys())
        if missing:
            errors.append(f"{record_name} row missing required keys: {sorted(missing)}")
            return
        if field_validators:
            for key, validator in field_validators.items():
                if key not in row:
                    continue
                if not validator(row[key]):
                    errors.append(f"{record_name} row has invalid type/value for '{key}'")
                    return
        checked += 1
        if checked >= sample_lines:
            break

    if checked == 0:
        errors.append(f"{record_name} had no readable rows in sample")


def _check_events_sample(events_path: Path, sample_lines: int, errors: List[str]) -> None:
    _check_jsonl_schema_sample(
        events_path,
        sample_lines,
        errors,
        record_name="events.jsonl",
        required_keys=(
            "global_frame_idx",
            "game_id",
            "visit_idx",
            "cycle_idx",
            "visit_frame_idx",
            "episode_id",
            "segment_id",
            "is_decision_frame",
            "decided_action_idx",
            "applied_action_idx",
            "reward",
            "terminated",
            "truncated",
        ),
        field_validators={
            "global_frame_idx": _is_int,
            "game_id": lambda value: isinstance(value, str),
            "visit_idx": _is_int,
            "cycle_idx": _is_int,
            "visit_frame_idx": _is_int,
            "episode_id": _is_int,
            "segment_id": _is_int,
            "is_decision_frame": lambda value: isinstance(value, bool),
            "decided_action_idx": _is_int,
            "applied_action_idx": _is_int,
            "reward": _is_number,
            "terminated": lambda value: isinstance(value, bool),
            "truncated": lambda value: isinstance(value, bool),
        },
    )


def _check_episodes_sample(episodes_path: Path, sample_lines: int, errors: List[str]) -> None:
    _check_jsonl_schema_sample(
        episodes_path,
        sample_lines,
        errors,
        record_name="episodes.jsonl",
        required_keys=(
            "game_id",
            "episode_id",
            "start_global_frame_idx",
            "end_global_frame_idx",
            "length",
            "return",
            "ended_by",
        ),
        field_validators={
            "game_id": lambda value: isinstance(value, str),
            "episode_id": _is_int,
            "start_global_frame_idx": _is_int,
            "end_global_frame_idx": _is_int,
            "length": _is_int,
            "return": _is_number,
            "ended_by": lambda value: str(value) in {"terminated", "truncated"},
        },
    )


def _check_segments_sample(segments_path: Path, sample_lines: int, errors: List[str]) -> None:
    _check_jsonl_schema_sample(
        segments_path,
        sample_lines,
        errors,
        record_name="segments.jsonl",
        required_keys=(
            "game_id",
            "segment_id",
            "start_global_frame_idx",
            "end_global_frame_idx",
            "length",
            "return",
            "ended_by",
        ),
        field_validators={
            "game_id": lambda value: isinstance(value, str),
            "segment_id": _is_int,
            "start_global_frame_idx": _is_int,
            "end_global_frame_idx": _is_int,
            "length": _is_int,
            "return": _is_number,
            "ended_by": lambda value: str(value) in {"terminated", "truncated"},
        },
    )


def validate_contract(run_dir: Path, sample_event_lines: int = 0) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    errors: List[str] = []

    config_path = run_dir / "config.json"
    score_path = run_dir / "score.json"
    events_path = run_dir / "events.jsonl"
    episodes_path = run_dir / "episodes.jsonl"
    segments_path = run_dir / "segments.jsonl"

    if not config_path.exists():
        return {"ok": False, "errors": ["config.json not found"]}
    if not score_path.exists():
        return {"ok": False, "errors": ["score.json not found"]}

    try:
        config = _load_json(config_path)
    except Exception as exc:  # pragma: no cover - invalid JSON path
        return {"ok": False, "errors": [f"failed to load config.json: {exc}"]}

    try:
        score = _load_json(score_path)
    except Exception as exc:  # pragma: no cover - invalid JSON path
        return {"ok": False, "errors": [f"failed to load score.json: {exc}"]}

    _check_config(config, errors)
    _check_config_hash_integrity(config, errors)
    _check_score_schema(score, errors)
    _check_score_tags(config, score, errors)
    sample_lines = int(sample_event_lines)
    _check_events_sample(events_path, sample_lines, errors)
    _check_episodes_sample(episodes_path, sample_lines, errors)
    _check_segments_sample(segments_path, sample_lines, errors)

    return {"ok": len(errors) == 0, "errors": errors}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark contract tags for a run directory.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory path.")
    parser.add_argument(
        "--sample-event-lines",
        type=int,
        default=0,
        help="Sample N lines from events/episodes/segments logs for required-key checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_contract(Path(args.run_dir), sample_event_lines=int(args.sample_event_lines))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
