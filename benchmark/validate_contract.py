"""Validation helpers for benchmark v1 contract tags and core artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from benchmark.carmack_multigame_runner import CARMACK_MULTI_RUN_PROFILE, CARMACK_MULTI_RUN_SCHEMA_VERSION
from benchmark.carmack_runner import CARMACK_SINGLE_RUN_PROFILE, CARMACK_SINGLE_RUN_SCHEMA_VERSION
from benchmark.contract import BENCHMARK_CONTRACT_VERSION, compute_contract_hash

VALIDATION_MODE_SAMPLE = "sample"
VALIDATION_MODE_STRATIFIED = "stratified"
VALIDATION_MODE_FULL = "full"
RUNTIME_FINGERPRINT_SCHEMA_VERSION = "runtime_fingerprint_v1"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)


def _is_optional_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _check_multigame_config(config: Mapping[str, Any], errors: List[str]) -> None:
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


def _check_carmack_single_run_config(config: Mapping[str, Any], errors: List[str]) -> None:
    runner_mode = config.get("runner_mode")
    if str(runner_mode) != CARMACK_SINGLE_RUN_PROFILE:
        errors.append(f"config.json runner_mode must be '{CARMACK_SINGLE_RUN_PROFILE}'")
    if config.get("single_run_profile") != CARMACK_SINGLE_RUN_PROFILE:
        errors.append(f"config.json single_run_profile must be '{CARMACK_SINGLE_RUN_PROFILE}'")
    if config.get("single_run_schema_version") != CARMACK_SINGLE_RUN_SCHEMA_VERSION:
        errors.append(f"config.json single_run_schema_version must be '{CARMACK_SINGLE_RUN_SCHEMA_VERSION}'")

    if not isinstance(config.get("game"), str):
        errors.append("config.json game must be string")
    if not _is_int(config.get("seed")):
        errors.append("config.json seed must be int")
    if not _is_int(config.get("frames")):
        errors.append("config.json frames must be int")
    elif int(config.get("frames")) <= 0:
        errors.append("config.json frames must be > 0")
    frame_skip = config.get("frame_skip")
    if frame_skip is not None:
        if not _is_int(frame_skip) or int(frame_skip) != 1:
            errors.append("config.json frame_skip must be int 1 for carmack_compat")
    max_frames_without_reward = config.get("max_frames_without_reward")
    if max_frames_without_reward is not None:
        if not _is_int(max_frames_without_reward) or int(max_frames_without_reward) <= 0:
            errors.append("config.json max_frames_without_reward must be int > 0")
    if not _is_int(config.get("default_action_idx")):
        errors.append("config.json default_action_idx must be int")

    runner_cfg = config.get("runner_config")
    if not isinstance(runner_cfg, dict):
        errors.append("config.json runner_config must be object")
        return
    if runner_cfg.get("runner_mode") != CARMACK_SINGLE_RUN_PROFILE:
        errors.append(f"config.json runner_config.runner_mode must be '{CARMACK_SINGLE_RUN_PROFILE}'")
    if runner_cfg.get("single_run_schema_version") != CARMACK_SINGLE_RUN_SCHEMA_VERSION:
        errors.append(
            f"config.json runner_config.single_run_schema_version must be '{CARMACK_SINGLE_RUN_SCHEMA_VERSION}'"
        )
    if runner_cfg.get("action_cadence_mode") != "agent_owned":
        errors.append("config.json runner_config.action_cadence_mode must be 'agent_owned'")
    if not _is_int(runner_cfg.get("frame_skip_enforced")) or int(runner_cfg.get("frame_skip_enforced")) != 1:
        errors.append("config.json runner_config.frame_skip_enforced must be int 1")
    if not _is_int(runner_cfg.get("total_frames")):
        errors.append("config.json runner_config.total_frames must be int")
    elif _is_int(config.get("frames")) and int(runner_cfg.get("total_frames")) != int(config.get("frames")):
        errors.append("config.json runner_config.total_frames must match config.json frames")
    if not _is_int(runner_cfg.get("delay_frames")):
        errors.append("config.json runner_config.delay_frames must be int")
    elif int(runner_cfg.get("delay_frames")) < 0:
        errors.append("config.json runner_config.delay_frames must be >= 0")


def _check_carmack_multigame_config(config: Mapping[str, Any], errors: List[str]) -> None:
    if str(config.get("runner_mode")) != CARMACK_MULTI_RUN_PROFILE:
        errors.append(f"config.json runner_mode must be '{CARMACK_MULTI_RUN_PROFILE}'")
    if str(config.get("multi_run_profile")) != CARMACK_MULTI_RUN_PROFILE:
        errors.append(f"config.json multi_run_profile must be '{CARMACK_MULTI_RUN_PROFILE}'")
    if str(config.get("multi_run_schema_version")) != CARMACK_MULTI_RUN_SCHEMA_VERSION:
        errors.append(f"config.json multi_run_schema_version must be '{CARMACK_MULTI_RUN_SCHEMA_VERSION}'")

    if not _is_int(config.get("decision_interval")):
        errors.append("config.json decision_interval must be int")
    elif int(config.get("decision_interval")) != 1:
        errors.append("config.json decision_interval must be int 1 for carmack_compat multi-game")

    runner_cfg = config.get("runner_config")
    if not isinstance(runner_cfg, dict):
        errors.append("config.json runner_config must be object")
        return
    if str(runner_cfg.get("runner_mode")) != CARMACK_MULTI_RUN_PROFILE:
        errors.append(f"config.json runner_config.runner_mode must be '{CARMACK_MULTI_RUN_PROFILE}'")
    if str(runner_cfg.get("multi_run_schema_version")) != CARMACK_MULTI_RUN_SCHEMA_VERSION:
        errors.append(
            f"config.json runner_config.multi_run_schema_version must be '{CARMACK_MULTI_RUN_SCHEMA_VERSION}'"
        )
    if runner_cfg.get("action_cadence_mode") != "agent_owned":
        errors.append("config.json runner_config.action_cadence_mode must be 'agent_owned'")
    if not _is_int(runner_cfg.get("frame_skip_enforced")) or int(runner_cfg.get("frame_skip_enforced")) != 1:
        errors.append("config.json runner_config.frame_skip_enforced must be int 1")
    if not _is_int(runner_cfg.get("delay_frames")):
        errors.append("config.json runner_config.delay_frames must be int")
    elif int(runner_cfg.get("delay_frames")) < 0:
        errors.append("config.json runner_config.delay_frames must be >= 0")


def _check_carmack_runtime_fingerprint(
    fingerprint: Mapping[str, Any],
    config: Mapping[str, Any],
    errors: List[str],
    warnings: List[str],
    *,
    source_name: str = "runtime_fingerprint",
) -> None:
    required_str_keys = (
        "fingerprint_schema_version",
        "runner_mode",
        "single_run_profile",
        "single_run_schema_version",
        "game",
        "seed_policy",
        "config_sha256_algorithm",
        "config_sha256_scope",
        "python_version",
        "ale_py_version",
        "rom_sha256",
        "config_sha256",
    )
    for key in required_str_keys:
        if not isinstance(fingerprint.get(key), str):
            errors.append(f"{source_name} missing string key: {key}")

    required_int_keys = ("seed", "frames")
    for key in required_int_keys:
        if not _is_int(fingerprint.get(key)):
            errors.append(f"{source_name} missing int key: {key}")

    if fingerprint.get("fingerprint_schema_version") != RUNTIME_FINGERPRINT_SCHEMA_VERSION:
        errors.append(
            f"{source_name} fingerprint_schema_version must be '{RUNTIME_FINGERPRINT_SCHEMA_VERSION}'"
        )
    if fingerprint.get("runner_mode") != CARMACK_SINGLE_RUN_PROFILE:
        errors.append(f"{source_name} runner_mode must be '{CARMACK_SINGLE_RUN_PROFILE}'")
    if fingerprint.get("single_run_profile") != CARMACK_SINGLE_RUN_PROFILE:
        errors.append(f"{source_name} single_run_profile must be '{CARMACK_SINGLE_RUN_PROFILE}'")
    if fingerprint.get("single_run_schema_version") != CARMACK_SINGLE_RUN_SCHEMA_VERSION:
        errors.append(
            f"{source_name} single_run_schema_version must be '{CARMACK_SINGLE_RUN_SCHEMA_VERSION}'"
        )
    if fingerprint.get("config_sha256_algorithm") != "sha256":
        errors.append(f"{source_name} config_sha256_algorithm must be 'sha256'")
    if fingerprint.get("config_sha256_scope") != "config_without_runtime_fingerprint":
        errors.append(f"{source_name} config_sha256_scope must be 'config_without_runtime_fingerprint'")

    if isinstance(fingerprint.get("game"), str) and isinstance(config.get("game"), str):
        if str(fingerprint.get("game")) != str(config.get("game")):
            errors.append(f"{source_name} game does not match config.json")
    if _is_int(fingerprint.get("seed")) and _is_int(config.get("seed")):
        if int(fingerprint.get("seed")) != int(config.get("seed")):
            errors.append(f"{source_name} seed does not match config.json")
    if _is_int(fingerprint.get("frames")) and _is_int(config.get("frames")):
        if int(fingerprint.get("frames")) != int(config.get("frames")):
            errors.append(f"{source_name} frames does not match config.json")

    config_without_fingerprint = dict(config)
    config_without_fingerprint.pop("runtime_fingerprint", None)
    expected_config_sha = _stable_payload_sha256(config_without_fingerprint)
    observed_config_sha = fingerprint.get("config_sha256")
    if isinstance(observed_config_sha, str):
        if observed_config_sha != expected_config_sha:
            errors.append(f"{source_name} config_sha256 does not match config.json canonical hash")

    rom_sha = fingerprint.get("rom_sha256")
    if isinstance(rom_sha, str):
        if re.fullmatch(r"[0-9a-f]{64}", rom_sha) is None:
            errors.append(f"{source_name} rom_sha256 must be a lowercase 64-char hex digest")

    informational_keys = (
        "platform",
        "machine",
        "processor",
        "python_implementation",
        "python_executable",
        "numpy_version",
        "torch_version",
        "cuda_available",
        "rom_path",
    )
    for key in informational_keys:
        if key not in fingerprint:
            warnings.append(f"{source_name} missing informational key: {key}")


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
    validation_mode: str = VALIDATION_MODE_SAMPLE,
    stratified_seed: int = 0,
    stratified_selectors: Optional[Sequence[Callable[[Mapping[str, Any]], Optional[str]]]] = None,
) -> None:
    _collect_jsonl_schema_sample(
        path,
        sample_lines,
        errors,
        record_name=record_name,
        required_keys=required_keys,
        field_validators=field_validators,
        validation_mode=validation_mode,
        stratified_seed=stratified_seed,
        stratified_selectors=stratified_selectors,
    )


def _row_schema_error(
    row: Any,
    *,
    record_name: str,
    required_keys: Sequence[str],
    field_validators: Optional[Mapping[str, Callable[[Any], bool]]] = None,
) -> Optional[str]:
    if not isinstance(row, dict):
        return f"{record_name} contains a non-object row"
    required = set(required_keys)
    missing = required - set(row.keys())
    if missing:
        return f"{record_name} row missing required keys: {sorted(missing)}"
    if field_validators:
        for key, validator in field_validators.items():
            if key not in row:
                continue
            if not validator(row[key]):
                return f"{record_name} row has invalid type/value for '{key}'"
    return None


def _row_frame_idx_or_default(row: Mapping[str, Any]) -> int:
    frame_idx = row.get("frame_idx")
    if _is_int(frame_idx):
        return int(frame_idx)
    global_frame_idx = row.get("global_frame_idx")
    if _is_int(global_frame_idx):
        return int(global_frame_idx)
    return -1


def _stable_row_rank(*, seed: int, row_idx: int, row: Mapping[str, Any]) -> str:
    basis = f"{int(seed)}:{int(row_idx)}:{_row_frame_idx_or_default(row)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _select_sample_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_mode: str,
    sample_lines: int,
    stratified_seed: int,
    stratified_selectors: Optional[Sequence[Callable[[Mapping[str, Any]], Optional[str]]]] = None,
) -> List[int]:
    total = len(rows)
    if total == 0:
        return []
    if validation_mode == VALIDATION_MODE_FULL:
        return list(range(total))
    if sample_lines <= 0:
        return []
    if validation_mode == VALIDATION_MODE_SAMPLE:
        return list(range(min(sample_lines, total)))

    mandatory_indices: set[int] = {0, total - 1}
    if stratified_selectors:
        for selector in stratified_selectors:
            first_by_label: Dict[str, int] = {}
            for row_idx, row in enumerate(rows):
                label = selector(row)
                if label is None or label in first_by_label:
                    continue
                first_by_label[label] = row_idx
            mandatory_indices.update(first_by_label.values())

    if len(mandatory_indices) > sample_lines:
        raise ValueError(
            "stratified validation budget too small: "
            f"need at least {len(mandatory_indices)} rows for mandatory strata, got {sample_lines}"
        )
    if len(mandatory_indices) == sample_lines:
        return sorted(mandatory_indices)

    ranked_remaining: List[tuple[str, int]] = []
    for row_idx, row in enumerate(rows):
        if row_idx in mandatory_indices:
            continue
        ranked_remaining.append((_stable_row_rank(seed=stratified_seed, row_idx=row_idx, row=row), row_idx))
    ranked_remaining.sort()
    needed = sample_lines - len(mandatory_indices)
    chosen = {row_idx for _, row_idx in ranked_remaining[:needed]}
    return sorted(mandatory_indices | chosen)


def _collect_jsonl_schema_sample(
    path: Path,
    sample_lines: int,
    errors: List[str],
    *,
    record_name: str,
    required_keys: Sequence[str],
    field_validators: Optional[Mapping[str, Callable[[Any], bool]]] = None,
    validation_mode: str = VALIDATION_MODE_SAMPLE,
    stratified_seed: int = 0,
    stratified_selectors: Optional[Sequence[Callable[[Mapping[str, Any]], Optional[str]]]] = None,
) -> List[Mapping[str, Any]]:
    if not path.exists():
        errors.append(f"{record_name} not found")
        return []

    requires_full_read = validation_mode in {VALIDATION_MODE_STRATIFIED, VALIDATION_MODE_FULL}
    if validation_mode == VALIDATION_MODE_STRATIFIED and sample_lines <= 0:
        errors.append("stratified validation requires sample_event_lines > 0")
        return []
    if not requires_full_read and sample_lines <= 0:
        return []

    rows: List[Mapping[str, Any]] = []
    if requires_full_read:
        for row in _iter_jsonl(path):
            if not isinstance(row, dict):
                errors.append(f"{record_name} contains a non-object row")
                return []
            rows.append(dict(row))
    else:
        checked = 0
        for row in _iter_jsonl(path):
            if not isinstance(row, dict):
                errors.append(f"{record_name} contains a non-object row")
                return []
            rows.append(dict(row))
            checked += 1
            if checked >= sample_lines:
                break

    if not rows:
        errors.append(f"{record_name} had no readable rows in validation scope")
        return []

    try:
        selected_indices = _select_sample_indices(
            rows,
            validation_mode=validation_mode,
            sample_lines=sample_lines,
            stratified_seed=stratified_seed,
            stratified_selectors=stratified_selectors,
        )
    except ValueError as exc:
        errors.append(str(exc))
        return []
    sampled_rows: List[Mapping[str, Any]] = []
    for row_idx in selected_indices:
        row = rows[row_idx]
        schema_error = _row_schema_error(
            row,
            record_name=record_name,
            required_keys=required_keys,
            field_validators=field_validators,
        )
        if schema_error is not None:
            errors.append(schema_error)
            return []
        sampled = dict(row)
        sampled["__validator_row_idx"] = int(row_idx)
        sampled_rows.append(sampled)

    if not sampled_rows:
        errors.append(f"{record_name} had no readable rows in validation scope")
        return []
    return sampled_rows


def _check_events_sample(
    events_path: Path,
    sample_lines: int,
    errors: List[str],
    *,
    validation_mode: str,
    stratified_seed: int,
) -> None:
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
        validation_mode=validation_mode,
        stratified_seed=stratified_seed,
    )


def _check_episodes_sample(
    episodes_path: Path,
    sample_lines: int,
    errors: List[str],
    *,
    validation_mode: str,
    stratified_seed: int,
) -> None:
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
        validation_mode=validation_mode,
        stratified_seed=stratified_seed,
    )


def _check_segments_sample(
    segments_path: Path,
    sample_lines: int,
    errors: List[str],
    *,
    validation_mode: str,
    stratified_seed: int,
) -> None:
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
        validation_mode=validation_mode,
        stratified_seed=stratified_seed,
    )


def _check_carmack_multigame_events_sample(
    events_path: Path,
    sample_lines: int,
    errors: List[str],
    *,
    validation_mode: str,
    stratified_seed: int,
) -> List[Mapping[str, Any]]:
    return _collect_jsonl_schema_sample(
        events_path,
        sample_lines,
        errors,
        record_name="events.jsonl",
        required_keys=(
            "multi_run_profile",
            "multi_run_schema_version",
            "frame_idx",
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
            "next_policy_action_idx",
            "applied_action_idx_local",
            "applied_ale_action",
            "reward",
            "terminated",
            "truncated",
            "env_terminated",
            "env_truncated",
            "lives",
            "episode_return_so_far",
            "segment_return_so_far",
            "end_of_episode_pulse",
            "boundary_cause",
            "reset_cause",
            "reset_performed",
            "env_termination_reason",
        ),
        field_validators={
            "multi_run_profile": lambda value: str(value) == CARMACK_MULTI_RUN_PROFILE,
            "multi_run_schema_version": lambda value: str(value) == CARMACK_MULTI_RUN_SCHEMA_VERSION,
            "frame_idx": _is_int,
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
            "next_policy_action_idx": _is_int,
            "applied_action_idx_local": _is_int,
            "applied_ale_action": _is_int,
            "reward": _is_number,
            "terminated": lambda value: isinstance(value, bool),
            "truncated": lambda value: isinstance(value, bool),
            "env_terminated": lambda value: isinstance(value, bool),
            "env_truncated": lambda value: isinstance(value, bool),
            "lives": _is_int,
            "episode_return_so_far": _is_number,
            "segment_return_so_far": _is_number,
            "end_of_episode_pulse": lambda value: isinstance(value, bool),
            "boundary_cause": _is_optional_str,
            "reset_cause": _is_optional_str,
            "reset_performed": lambda value: isinstance(value, bool),
            "env_termination_reason": _is_optional_str,
        },
        validation_mode=validation_mode,
        stratified_seed=stratified_seed,
        stratified_selectors=(
            _stratify_boundary_cause_label,
            _stratify_reset_cause_label,
            _stratify_pulse_reset_pair_label,
        ),
    )


def _check_carmack_multigame_episodes_sample(
    episodes_path: Path,
    sample_lines: int,
    errors: List[str],
    *,
    validation_mode: str,
    stratified_seed: int,
) -> List[Mapping[str, Any]]:
    return _collect_jsonl_schema_sample(
        episodes_path,
        sample_lines,
        errors,
        record_name="episodes.jsonl",
        required_keys=(
            "multi_run_profile",
            "multi_run_schema_version",
            "game_id",
            "episode_id",
            "start_global_frame_idx",
            "end_global_frame_idx",
            "length",
            "return",
            "ended_by",
            "boundary_cause",
        ),
        field_validators={
            "multi_run_profile": lambda value: str(value) == CARMACK_MULTI_RUN_PROFILE,
            "multi_run_schema_version": lambda value: str(value) == CARMACK_MULTI_RUN_SCHEMA_VERSION,
            "game_id": lambda value: isinstance(value, str),
            "episode_id": _is_int,
            "start_global_frame_idx": _is_int,
            "end_global_frame_idx": _is_int,
            "length": _is_int,
            "return": _is_number,
            "ended_by": lambda value: str(value) in {"terminated", "truncated"},
            "boundary_cause": _is_optional_str,
        },
        validation_mode=validation_mode,
        stratified_seed=stratified_seed,
    )


def _check_carmack_multigame_segments_sample(
    segments_path: Path,
    sample_lines: int,
    errors: List[str],
    *,
    validation_mode: str,
    stratified_seed: int,
) -> List[Mapping[str, Any]]:
    return _collect_jsonl_schema_sample(
        segments_path,
        sample_lines,
        errors,
        record_name="segments.jsonl",
        required_keys=(
            "multi_run_profile",
            "multi_run_schema_version",
            "game_id",
            "segment_id",
            "start_global_frame_idx",
            "end_global_frame_idx",
            "length",
            "return",
            "ended_by",
            "boundary_cause",
        ),
        field_validators={
            "multi_run_profile": lambda value: str(value) == CARMACK_MULTI_RUN_PROFILE,
            "multi_run_schema_version": lambda value: str(value) == CARMACK_MULTI_RUN_SCHEMA_VERSION,
            "game_id": lambda value: isinstance(value, str),
            "segment_id": _is_int,
            "start_global_frame_idx": _is_int,
            "end_global_frame_idx": _is_int,
            "length": _is_int,
            "return": _is_number,
            "ended_by": lambda value: str(value) in {"terminated", "truncated"},
            "boundary_cause": _is_optional_str,
        },
        validation_mode=validation_mode,
        stratified_seed=stratified_seed,
    )


def _stratify_boundary_cause_label(row: Mapping[str, Any]) -> Optional[str]:
    boundary_cause = row.get("boundary_cause")
    if isinstance(boundary_cause, str):
        return f"boundary:{boundary_cause}"
    return None


def _stratify_reset_cause_label(row: Mapping[str, Any]) -> Optional[str]:
    reset_cause = row.get("reset_cause")
    if isinstance(reset_cause, str):
        return f"reset:{reset_cause}"
    return None


def _stratify_pulse_reset_pair_label(row: Mapping[str, Any]) -> Optional[str]:
    pulse = row.get("end_of_episode_pulse")
    reset_performed = row.get("reset_performed")
    if isinstance(pulse, bool) and isinstance(reset_performed, bool):
        return f"pulse={int(pulse)}|reset={int(reset_performed)}"
    return None


def _check_carmack_events_sample(
    events_path: Path,
    sample_lines: int,
    errors: List[str],
    *,
    validation_mode: str,
    stratified_seed: int,
) -> List[Mapping[str, Any]]:
    return _collect_jsonl_schema_sample(
        events_path,
        sample_lines,
        errors,
        record_name="events.jsonl",
        required_keys=(
            "single_run_profile",
            "single_run_schema_version",
            "frame_idx",
            "applied_action_idx",
            "decided_action_idx",
            "next_policy_action_idx",
            "decided_action_changed",
            "applied_action_changed",
            "decided_applied_mismatch",
            "applied_action_hold_run_length",
            "reward",
            "terminated",
            "truncated",
            "env_terminated",
            "env_truncated",
            "lives",
            "episode_idx",
            "episode_return",
            "episode_length",
            "end_of_episode_pulse",
            "pulse_reason",
            "boundary_cause",
            "reset_cause",
            "reset_performed",
            "frames_without_reward",
        ),
        field_validators={
            "single_run_profile": lambda value: str(value) == CARMACK_SINGLE_RUN_PROFILE,
            "single_run_schema_version": lambda value: str(value) == CARMACK_SINGLE_RUN_SCHEMA_VERSION,
            "frame_idx": _is_int,
            "applied_action_idx": _is_int,
            "decided_action_idx": _is_int,
            "next_policy_action_idx": _is_int,
            "decided_action_changed": lambda value: isinstance(value, bool),
            "applied_action_changed": lambda value: isinstance(value, bool),
            "decided_applied_mismatch": lambda value: isinstance(value, bool),
            "applied_action_hold_run_length": _is_int,
            "reward": _is_number,
            "terminated": lambda value: isinstance(value, bool),
            "truncated": lambda value: isinstance(value, bool),
            "env_terminated": lambda value: isinstance(value, bool),
            "env_truncated": lambda value: isinstance(value, bool),
            "lives": _is_int,
            "episode_idx": _is_int,
            "episode_return": _is_number,
            "episode_length": _is_int,
            "end_of_episode_pulse": lambda value: isinstance(value, bool),
            "pulse_reason": _is_optional_str,
            "boundary_cause": _is_optional_str,
            "reset_cause": _is_optional_str,
            "reset_performed": lambda value: isinstance(value, bool),
            "frames_without_reward": _is_int,
        },
        validation_mode=validation_mode,
        stratified_seed=stratified_seed,
        stratified_selectors=(
            _stratify_boundary_cause_label,
            _stratify_reset_cause_label,
            _stratify_pulse_reset_pair_label,
        ),
    )


def _check_carmack_episodes_sample(
    episodes_path: Path,
    sample_lines: int,
    errors: List[str],
    *,
    validation_mode: str,
    stratified_seed: int,
) -> List[Mapping[str, Any]]:
    return _collect_jsonl_schema_sample(
        episodes_path,
        sample_lines,
        errors,
        record_name="episodes.jsonl",
        required_keys=(
            "single_run_profile",
            "single_run_schema_version",
            "episode_idx",
            "episode_return",
            "length",
            "termination_reason",
            "end_frame_idx",
            "ended_by_reset",
        ),
        field_validators={
            "single_run_profile": lambda value: str(value) == CARMACK_SINGLE_RUN_PROFILE,
            "single_run_schema_version": lambda value: str(value) == CARMACK_SINGLE_RUN_SCHEMA_VERSION,
            "episode_idx": _is_int,
            "episode_return": _is_number,
            "length": _is_int,
            "termination_reason": lambda value: isinstance(value, str),
            "end_frame_idx": _is_int,
            "ended_by_reset": lambda value: isinstance(value, bool),
        },
        validation_mode=validation_mode,
        stratified_seed=stratified_seed,
    )


def _check_carmack_summary_schema(
    summary: Mapping[str, Any],
    errors: List[str],
    *,
    config_frames: Optional[int] = None,
) -> None:
    if summary.get("single_run_profile") != CARMACK_SINGLE_RUN_PROFILE:
        errors.append(f"run_summary.json single_run_profile must be '{CARMACK_SINGLE_RUN_PROFILE}'")
    if summary.get("single_run_schema_version") != CARMACK_SINGLE_RUN_SCHEMA_VERSION:
        errors.append(f"run_summary.json single_run_schema_version must be '{CARMACK_SINGLE_RUN_SCHEMA_VERSION}'")
    if summary.get("runner_mode") != CARMACK_SINGLE_RUN_PROFILE:
        errors.append(f"run_summary.json runner_mode must be '{CARMACK_SINGLE_RUN_PROFILE}'")

    int_keys = (
        "frames",
        "episodes_completed",
        "last_episode_idx",
        "last_episode_length",
        "pulse_count",
        "life_loss_pulses",
        "reset_count",
        "game_over_resets",
        "truncated_resets",
        "timeout_resets",
        "life_loss_resets",
        "decided_action_change_count",
        "applied_action_change_count",
        "decided_applied_mismatch_count",
        "applied_action_hold_run_count",
        "applied_action_hold_run_max",
    )
    for key in int_keys:
        if key not in summary:
            errors.append(f"run_summary.json missing key: {key}")
            continue
        if not _is_int(summary.get(key)):
            errors.append(f"run_summary.json key '{key}' must be int")

    float_keys = (
        "last_episode_return",
        "decided_action_change_rate",
        "applied_action_change_rate",
        "decided_applied_mismatch_rate",
        "applied_action_hold_run_mean",
    )
    for key in float_keys:
        if key not in summary:
            errors.append(f"run_summary.json missing key: {key}")
            continue
        if not _is_number(summary.get(key)):
            errors.append(f"run_summary.json key '{key}' must be numeric")

    dict_count_keys = ("boundary_cause_counts", "reset_cause_counts")
    for key in dict_count_keys:
        value = summary.get(key)
        if not isinstance(value, dict):
            errors.append(f"run_summary.json key '{key}' must be object")
            continue
        for sub_key, sub_value in value.items():
            if not isinstance(sub_key, str):
                errors.append(f"run_summary.json key '{key}' must map string keys")
                break
            if not _is_int(sub_value):
                errors.append(f"run_summary.json key '{key}' must map to int values")
                break

    if config_frames is not None and _is_int(summary.get("frames")) and int(summary["frames"]) != int(config_frames):
        errors.append("run_summary.json frames does not match config.json frames")


def _check_carmack_multigame_summary_schema(
    summary: Mapping[str, Any],
    errors: List[str],
    *,
    config_total_scheduled_frames: Optional[int] = None,
) -> None:
    int_keys = (
        "frames",
        "episodes_completed",
        "segments_completed",
        "last_episode_id",
        "last_segment_id",
        "visits_completed",
        "total_scheduled_frames",
        "reset_count",
    )
    for key in int_keys:
        if key not in summary:
            errors.append(f"run_summary.json missing key: {key}")
            continue
        if not _is_int(summary.get(key)):
            errors.append(f"run_summary.json key '{key}' must be int")

    for key in ("boundary_cause_counts", "reset_cause_counts"):
        value = summary.get(key)
        if not isinstance(value, dict):
            errors.append(f"run_summary.json key '{key}' must be object")
            continue
        for sub_key, sub_value in value.items():
            if not isinstance(sub_key, str):
                errors.append(f"run_summary.json key '{key}' must map string keys")
                break
            if not _is_int(sub_value):
                errors.append(f"run_summary.json key '{key}' must map to int values")
                break

    if config_total_scheduled_frames is not None and _is_int(summary.get("frames")):
        if int(summary["frames"]) != int(config_total_scheduled_frames):
            errors.append("run_summary.json frames does not match config.json total_scheduled_frames")
    if config_total_scheduled_frames is not None and _is_int(summary.get("total_scheduled_frames")):
        if int(summary["total_scheduled_frames"]) != int(config_total_scheduled_frames):
            errors.append("run_summary.json total_scheduled_frames does not match config.json total_scheduled_frames")


def _check_carmack_multigame_event_semantics(rows: Sequence[Mapping[str, Any]], errors: List[str]) -> None:
    allowed_boundary_causes = {"visit_switch", "terminated", "truncated"}
    allowed_reset_causes = {"visit_switch", "terminated", "truncated"}
    for idx, row in enumerate(rows):
        source_idx = row.get("__validator_row_idx")
        prefix = f"events.jsonl semantic error at row {source_idx if _is_int(source_idx) else idx}"

        pulse = bool(row["end_of_episode_pulse"])
        terminated = bool(row["terminated"])
        truncated = bool(row["truncated"])
        env_terminated = bool(row["env_terminated"])
        env_truncated = bool(row["env_truncated"])
        boundary_cause = row.get("boundary_cause")
        reset_cause = row.get("reset_cause")
        reset_performed = bool(row["reset_performed"])

        if not bool(row["is_decision_frame"]):
            errors.append(f"{prefix}: is_decision_frame must be true in Carmack multi-game profile")

        if int(row["frame_idx"]) != int(row["global_frame_idx"]):
            errors.append(f"{prefix}: frame_idx must equal global_frame_idx")

        if terminated != env_terminated:
            errors.append(f"{prefix}: terminated must match env_terminated")
        if truncated != (env_truncated or str(boundary_cause) == "visit_switch"):
            errors.append(f"{prefix}: truncated must equal (env_truncated or boundary_cause=visit_switch)")
        if pulse != (terminated or truncated):
            errors.append(f"{prefix}: end_of_episode_pulse must equal (terminated or truncated)")

        if not pulse and (boundary_cause is not None or reset_cause is not None):
            errors.append(f"{prefix}: non-pulse row must have boundary_cause/reset_cause = null")
        if pulse and boundary_cause is None:
            errors.append(f"{prefix}: pulse row must set boundary_cause")

        if boundary_cause is not None and str(boundary_cause) not in allowed_boundary_causes:
            errors.append(f"{prefix}: boundary_cause must be one of {sorted(allowed_boundary_causes)}")
        if reset_cause is not None and str(reset_cause) not in allowed_reset_causes:
            errors.append(f"{prefix}: reset_cause must be one of {sorted(allowed_reset_causes)}")

        if reset_performed != (reset_cause is not None):
            errors.append(f"{prefix}: reset_performed must match (reset_cause is not null)")

        if str(boundary_cause) == "visit_switch":
            if not truncated:
                errors.append(f"{prefix}: boundary_cause=visit_switch requires truncated=true")
            if str(reset_cause) != "visit_switch":
                errors.append(f"{prefix}: boundary_cause=visit_switch requires reset_cause=visit_switch")
            if not reset_performed:
                errors.append(f"{prefix}: boundary_cause=visit_switch requires reset_performed=true")
        elif str(boundary_cause) == "truncated":
            if not env_truncated:
                errors.append(f"{prefix}: boundary_cause=truncated requires env_truncated=true")
        elif str(boundary_cause) == "terminated":
            if not env_terminated:
                errors.append(f"{prefix}: boundary_cause=terminated requires env_terminated=true")

        if str(reset_cause) == "visit_switch" and str(boundary_cause) != "visit_switch":
            errors.append(f"{prefix}: reset_cause=visit_switch requires boundary_cause=visit_switch")
        if str(reset_cause) == "truncated" and str(boundary_cause) != "truncated":
            errors.append(f"{prefix}: reset_cause=truncated requires boundary_cause=truncated")
        if str(reset_cause) == "terminated" and str(boundary_cause) != "terminated":
            errors.append(f"{prefix}: reset_cause=terminated requires boundary_cause=terminated")

        if pulse and str(boundary_cause) != "visit_switch":
            if env_truncated and str(boundary_cause) != "truncated":
                errors.append(f"{prefix}: env_truncated=true requires boundary_cause=truncated")
            if (not env_truncated) and env_terminated and str(boundary_cause) != "terminated":
                errors.append(f"{prefix}: env_terminated=true without env_truncated requires boundary_cause=terminated")


def _check_carmack_multigame_episode_semantics(rows: Sequence[Mapping[str, Any]], errors: List[str]) -> None:
    allowed_boundary_causes = {"visit_switch", "terminated", "truncated"}
    for idx, row in enumerate(rows):
        source_idx = row.get("__validator_row_idx")
        prefix = f"episodes.jsonl semantic error at row {source_idx if _is_int(source_idx) else idx}"

        length = int(row["length"])
        start_frame = int(row["start_global_frame_idx"])
        end_frame = int(row["end_global_frame_idx"])
        ended_by = str(row["ended_by"])
        boundary_cause = row.get("boundary_cause")

        if length <= 0:
            errors.append(f"{prefix}: length must be > 0")
        if end_frame < start_frame:
            errors.append(f"{prefix}: end_global_frame_idx must be >= start_global_frame_idx")
        if boundary_cause is None:
            errors.append(f"{prefix}: boundary_cause must be non-null")
            continue
        if str(boundary_cause) not in allowed_boundary_causes:
            errors.append(f"{prefix}: boundary_cause must be one of {sorted(allowed_boundary_causes)}")
            continue
        if ended_by == "terminated" and str(boundary_cause) != "terminated":
            errors.append(f"{prefix}: ended_by=terminated requires boundary_cause=terminated")
        if ended_by == "truncated" and str(boundary_cause) not in {"truncated", "visit_switch"}:
            errors.append(f"{prefix}: ended_by=truncated requires boundary_cause in {{truncated,visit_switch}}")


def _check_carmack_multigame_segment_semantics(rows: Sequence[Mapping[str, Any]], errors: List[str]) -> None:
    allowed_boundary_causes = {"visit_switch", "terminated", "truncated"}
    for idx, row in enumerate(rows):
        source_idx = row.get("__validator_row_idx")
        prefix = f"segments.jsonl semantic error at row {source_idx if _is_int(source_idx) else idx}"

        length = int(row["length"])
        start_frame = int(row["start_global_frame_idx"])
        end_frame = int(row["end_global_frame_idx"])
        ended_by = str(row["ended_by"])
        boundary_cause = row.get("boundary_cause")

        if length <= 0:
            errors.append(f"{prefix}: length must be > 0")
        if end_frame < start_frame:
            errors.append(f"{prefix}: end_global_frame_idx must be >= start_global_frame_idx")
        if boundary_cause is None:
            errors.append(f"{prefix}: boundary_cause must be non-null")
            continue
        if str(boundary_cause) not in allowed_boundary_causes:
            errors.append(f"{prefix}: boundary_cause must be one of {sorted(allowed_boundary_causes)}")
            continue
        if ended_by == "terminated" and str(boundary_cause) != "terminated":
            errors.append(f"{prefix}: ended_by=terminated requires boundary_cause=terminated")
        if ended_by == "truncated" and str(boundary_cause) not in {"truncated", "visit_switch"}:
            errors.append(f"{prefix}: ended_by=truncated requires boundary_cause in {{truncated,visit_switch}}")


def _check_carmack_multigame_sample_consistency(
    event_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    errors: List[str],
) -> None:
    pulse_count_sample = sum(1 for row in event_rows if bool(row["end_of_episode_pulse"]))
    reset_count_sample = sum(1 for row in event_rows if bool(row["reset_performed"]))
    if _is_int(summary.get("episodes_completed")) and pulse_count_sample > int(summary["episodes_completed"]):
        errors.append("run_summary.json episodes_completed is smaller than sampled event pulse count")
    if _is_int(summary.get("segments_completed")) and pulse_count_sample > int(summary["segments_completed"]):
        errors.append("run_summary.json segments_completed is smaller than sampled event pulse count")
    if _is_int(summary.get("reset_count")) and reset_count_sample > int(summary["reset_count"]):
        errors.append("run_summary.json reset_count is smaller than sampled event reset count")

    boundary_counts_sample: Dict[str, int] = {}
    reset_counts_sample: Dict[str, int] = {}
    for row in event_rows:
        boundary_cause = row.get("boundary_cause")
        reset_cause = row.get("reset_cause")
        if isinstance(boundary_cause, str):
            boundary_counts_sample[boundary_cause] = int(boundary_counts_sample.get(boundary_cause, 0) + 1)
        if isinstance(reset_cause, str):
            reset_counts_sample[reset_cause] = int(reset_counts_sample.get(reset_cause, 0) + 1)

    summary_boundary = summary.get("boundary_cause_counts")
    if isinstance(summary_boundary, dict):
        for cause, count in boundary_counts_sample.items():
            value = summary_boundary.get(cause)
            if not _is_int(value) or int(value) < int(count):
                errors.append(f"run_summary.json boundary_cause_counts['{cause}'] is smaller than sampled events")

    summary_reset = summary.get("reset_cause_counts")
    if isinstance(summary_reset, dict):
        for cause, count in reset_counts_sample.items():
            value = summary_reset.get(cause)
            if not _is_int(value) or int(value) < int(count):
                errors.append(f"run_summary.json reset_cause_counts['{cause}'] is smaller than sampled events")

    if _is_int(summary.get("episodes_completed")) and len(episode_rows) > int(summary["episodes_completed"]):
        errors.append("run_summary.json episodes_completed is smaller than sampled episode row count")
    if _is_int(summary.get("segments_completed")) and len(segment_rows) > int(summary["segments_completed"]):
        errors.append("run_summary.json segments_completed is smaller than sampled segment row count")


def _check_carmack_event_semantics(rows: Sequence[Mapping[str, Any]], errors: List[str]) -> None:
    allowed_reset_causes = {"no_reward_timeout", "terminated", "truncated", "life_loss_reset"}
    for idx, row in enumerate(rows):
        source_idx = row.get("__validator_row_idx")
        prefix = f"events.jsonl semantic error at row {source_idx if _is_int(source_idx) else idx}"

        pulse = bool(row["end_of_episode_pulse"])
        boundary_cause = row.get("boundary_cause")
        reset_cause = row.get("reset_cause")
        reset_performed = bool(row["reset_performed"])
        terminated = bool(row["terminated"])
        truncated = bool(row["truncated"])

        if not pulse and (boundary_cause is not None or reset_cause is not None):
            errors.append(f"{prefix}: non-pulse row must have boundary_cause/reset_cause = null")
        if pulse and boundary_cause is None:
            errors.append(f"{prefix}: pulse row must set boundary_cause")

        if reset_performed != (reset_cause is not None):
            errors.append(f"{prefix}: reset_performed must match (reset_cause is not null)")

        if reset_cause is not None and str(reset_cause) not in allowed_reset_causes:
            errors.append(f"{prefix}: reset_cause must be one of {sorted(allowed_reset_causes)}")

        if str(reset_cause) == "no_reward_timeout":
            if str(boundary_cause) != "no_reward_timeout":
                errors.append(f"{prefix}: no_reward_timeout reset requires boundary_cause=no_reward_timeout")
            if not truncated:
                errors.append(f"{prefix}: no_reward_timeout reset requires truncated=true")
        if str(reset_cause) == "terminated" and not terminated:
            errors.append(f"{prefix}: terminated reset requires terminated=true")
        if str(reset_cause) == "truncated" and not truncated:
            errors.append(f"{prefix}: truncated reset requires truncated=true")
        if str(reset_cause) == "life_loss_reset":
            if str(boundary_cause) != "life_loss":
                errors.append(f"{prefix}: life_loss_reset requires boundary_cause=life_loss")
            if terminated:
                errors.append(f"{prefix}: life_loss_reset requires terminated=false")
            if not truncated:
                errors.append(f"{prefix}: life_loss_reset requires truncated=true")

        if str(boundary_cause) == "life_loss":
            if terminated:
                errors.append(f"{prefix}: boundary_cause=life_loss requires terminated=false")
            if reset_cause is not None and str(reset_cause) != "life_loss_reset":
                errors.append(f"{prefix}: boundary_cause=life_loss allows only reset_cause=null or life_loss_reset")

        if terminated and reset_cause is not None and str(reset_cause) not in {"no_reward_timeout", "terminated"}:
            errors.append(f"{prefix}: terminated=true allows reset_cause only no_reward_timeout or terminated")
        if (str(reset_cause) in {"no_reward_timeout", "truncated", "life_loss_reset"}) and not truncated:
            errors.append(f"{prefix}: reset_cause={reset_cause} requires truncated=true")


def _check_carmack_episode_semantics(rows: Sequence[Mapping[str, Any]], errors: List[str]) -> None:
    prev_episode_idx: Optional[int] = None
    prev_end_frame_idx: Optional[int] = None
    for idx, row in enumerate(rows):
        source_idx = row.get("__validator_row_idx")
        prefix = f"episodes.jsonl semantic error at row {source_idx if _is_int(source_idx) else idx}"
        episode_idx = int(row["episode_idx"])
        end_frame_idx = int(row["end_frame_idx"])
        length = int(row["length"])

        if not bool(row["ended_by_reset"]):
            errors.append(f"{prefix}: ended_by_reset must be true in Carmack profile")
        if length <= 0:
            errors.append(f"{prefix}: length must be > 0")
        if prev_episode_idx is not None and episode_idx != prev_episode_idx + 1:
            errors.append(f"{prefix}: episode_idx must increase by 1 in sampled order")
        if prev_end_frame_idx is not None and end_frame_idx <= prev_end_frame_idx:
            errors.append(f"{prefix}: end_frame_idx must be strictly increasing in sampled order")

        prev_episode_idx = episode_idx
        prev_end_frame_idx = end_frame_idx


def _check_carmack_sample_consistency(
    event_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    errors: List[str],
) -> None:
    if not event_rows:
        return
    pulse_count_sample = sum(1 for row in event_rows if bool(row["end_of_episode_pulse"]))
    reset_count_sample = sum(1 for row in event_rows if bool(row["reset_performed"]))
    if _is_int(summary.get("pulse_count")) and pulse_count_sample > int(summary["pulse_count"]):
        errors.append("run_summary.json pulse_count is smaller than sampled event pulse count")
    if _is_int(summary.get("reset_count")) and reset_count_sample > int(summary["reset_count"]):
        errors.append("run_summary.json reset_count is smaller than sampled event reset count")

    boundary_counts_sample: Dict[str, int] = {}
    reset_counts_sample: Dict[str, int] = {}
    for row in event_rows:
        boundary_cause = row.get("boundary_cause")
        reset_cause = row.get("reset_cause")
        if isinstance(boundary_cause, str):
            boundary_counts_sample[boundary_cause] = int(boundary_counts_sample.get(boundary_cause, 0) + 1)
        if isinstance(reset_cause, str):
            reset_counts_sample[reset_cause] = int(reset_counts_sample.get(reset_cause, 0) + 1)

    summary_boundary = summary.get("boundary_cause_counts")
    if isinstance(summary_boundary, dict):
        for cause, count in boundary_counts_sample.items():
            value = summary_boundary.get(cause)
            if not _is_int(value) or int(value) < int(count):
                errors.append(f"run_summary.json boundary_cause_counts['{cause}'] is smaller than sampled events")
    summary_reset = summary.get("reset_cause_counts")
    if isinstance(summary_reset, dict):
        for cause, count in reset_counts_sample.items():
            value = summary_reset.get(cause)
            if not _is_int(value) or int(value) < int(count):
                errors.append(f"run_summary.json reset_cause_counts['{cause}'] is smaller than sampled events")

    if episode_rows and _is_int(summary.get("episodes_completed")) and len(episode_rows) > int(summary["episodes_completed"]):
        errors.append("run_summary.json episodes_completed is smaller than sampled episode row count")

def _is_carmack_multigame_config(config: Mapping[str, Any]) -> bool:
    if str(config.get("multi_run_profile")) == CARMACK_MULTI_RUN_PROFILE:
        return True
    if str(config.get("multi_run_schema_version")) == CARMACK_MULTI_RUN_SCHEMA_VERSION:
        return True
    if str(config.get("runner_mode")) != CARMACK_MULTI_RUN_PROFILE:
        return False
    return isinstance(config.get("games"), list) and isinstance(config.get("schedule"), list)


def _is_carmack_single_run_config(config: Mapping[str, Any]) -> bool:
    if str(config.get("single_run_profile")) == CARMACK_SINGLE_RUN_PROFILE:
        return True
    if str(config.get("single_run_schema_version")) == CARMACK_SINGLE_RUN_SCHEMA_VERSION:
        return True
    return (
        str(config.get("runner_mode")) == CARMACK_SINGLE_RUN_PROFILE
        and isinstance(config.get("game"), str)
        and _is_int(config.get("frames"))
        and not isinstance(config.get("games"), list)
    )


def _normalize_validation_mode(raw_mode: Any) -> Optional[str]:
    mode = str(raw_mode or VALIDATION_MODE_SAMPLE).strip().lower()
    if mode in {VALIDATION_MODE_SAMPLE, VALIDATION_MODE_STRATIFIED, VALIDATION_MODE_FULL}:
        return mode
    return None


def validate_contract(
    run_dir: Path,
    sample_event_lines: int = 0,
    *,
    validation_mode: str = VALIDATION_MODE_SAMPLE,
    stratified_seed: int = 0,
    fail_on_warnings: bool = False,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    errors: List[str] = []
    warnings: List[str] = []

    config_path = run_dir / "config.json"
    score_path = run_dir / "score.json"
    run_summary_path = run_dir / "run_summary.json"
    runtime_fingerprint_path = run_dir / "runtime_fingerprint.json"
    events_path = run_dir / "events.jsonl"
    episodes_path = run_dir / "episodes.jsonl"
    segments_path = run_dir / "segments.jsonl"

    if not config_path.exists():
        return {"ok": False, "errors": ["config.json not found"], "warnings": []}

    try:
        config = _load_json(config_path)
    except Exception as exc:  # pragma: no cover - invalid JSON path
        return {"ok": False, "errors": [f"failed to load config.json: {exc}"], "warnings": []}

    mode = _normalize_validation_mode(validation_mode)
    if mode is None:
        return {
            "ok": False,
            "errors": [
                "validation_mode must be one of: sample, stratified, full",
            ],
            "warnings": [],
        }

    sample_lines = int(sample_event_lines)
    seed = int(stratified_seed)

    if _is_carmack_multigame_config(config):
        if not score_path.exists():
            return {"ok": False, "errors": ["score.json not found"], "warnings": []}
        try:
            score = _load_json(score_path)
        except Exception as exc:  # pragma: no cover - invalid JSON path
            return {"ok": False, "errors": [f"failed to load score.json: {exc}"], "warnings": []}

        _check_multigame_config(config, errors)
        _check_carmack_multigame_config(config, errors)
        _check_config_hash_integrity(config, errors)
        _check_score_schema(score, errors)
        _check_score_tags(config, score, errors)

        run_summary: Optional[Mapping[str, Any]] = None
        if not run_summary_path.exists():
            errors.append("run_summary.json not found")
        else:
            try:
                run_summary = _load_json(run_summary_path)
            except Exception as exc:  # pragma: no cover - invalid JSON path
                errors.append(f"failed to load run_summary.json: {exc}")
            else:
                config_total = config.get("total_scheduled_frames")
                _check_carmack_multigame_summary_schema(
                    run_summary,
                    errors,
                    config_total_scheduled_frames=int(config_total) if _is_int(config_total) else None,
                )

        event_rows = _check_carmack_multigame_events_sample(
            events_path,
            sample_lines,
            errors,
            validation_mode=mode,
            stratified_seed=seed,
        )
        episode_rows = _check_carmack_multigame_episodes_sample(
            episodes_path,
            sample_lines,
            errors,
            validation_mode=mode,
            stratified_seed=seed,
        )
        segment_rows = _check_carmack_multigame_segments_sample(
            segments_path,
            sample_lines,
            errors,
            validation_mode=mode,
            stratified_seed=seed,
        )
        _check_carmack_multigame_event_semantics(event_rows, errors)
        _check_carmack_multigame_episode_semantics(episode_rows, errors)
        _check_carmack_multigame_segment_semantics(segment_rows, errors)
        if run_summary is not None:
            _check_carmack_multigame_sample_consistency(
                event_rows,
                episode_rows,
                segment_rows,
                run_summary,
                errors,
            )
    elif _is_carmack_single_run_config(config):
        _check_carmack_single_run_config(config, errors)
        run_summary: Optional[Mapping[str, Any]] = None
        if not run_summary_path.exists():
            errors.append("run_summary.json not found")
        else:
            try:
                run_summary = _load_json(run_summary_path)
            except Exception as exc:  # pragma: no cover - invalid JSON path
                errors.append(f"failed to load run_summary.json: {exc}")
            else:
                config_frames = config.get("frames")
                _check_carmack_summary_schema(
                    run_summary,
                    errors,
                    config_frames=int(config_frames) if _is_int(config_frames) else None,
                )
        embedded_fingerprint_config = config.get("runtime_fingerprint")
        if not isinstance(embedded_fingerprint_config, dict):
            errors.append("config.json missing object key: runtime_fingerprint")
            embedded_fingerprint_config = None
        embedded_fingerprint_summary = None
        if run_summary is not None:
            embedded_fingerprint_summary = run_summary.get("runtime_fingerprint")
            if not isinstance(embedded_fingerprint_summary, dict):
                errors.append("run_summary.json missing object key: runtime_fingerprint")
                embedded_fingerprint_summary = None

        sidecar_fingerprint: Optional[Mapping[str, Any]] = None
        if runtime_fingerprint_path.exists():
            try:
                sidecar_fingerprint = _load_json(runtime_fingerprint_path)
            except Exception as exc:  # pragma: no cover - invalid JSON path
                errors.append(f"failed to load runtime_fingerprint.json: {exc}")

        if isinstance(embedded_fingerprint_config, dict):
            _check_carmack_runtime_fingerprint(
                embedded_fingerprint_config,
                config,
                errors,
                warnings,
                source_name="config.json runtime_fingerprint",
            )
        if isinstance(embedded_fingerprint_summary, dict):
            _check_carmack_runtime_fingerprint(
                embedded_fingerprint_summary,
                config,
                errors,
                warnings,
                source_name="run_summary.json runtime_fingerprint",
            )
        if isinstance(embedded_fingerprint_config, dict) and isinstance(embedded_fingerprint_summary, dict):
            if dict(embedded_fingerprint_config) != dict(embedded_fingerprint_summary):
                errors.append("config.json and run_summary.json runtime_fingerprint payloads do not match")
        if isinstance(sidecar_fingerprint, dict) and isinstance(embedded_fingerprint_config, dict):
            if dict(sidecar_fingerprint) != dict(embedded_fingerprint_config):
                errors.append("runtime_fingerprint.json does not match embedded config runtime_fingerprint")
        event_rows = _check_carmack_events_sample(
            events_path,
            sample_lines,
            errors,
            validation_mode=mode,
            stratified_seed=seed,
        )
        episode_rows = _check_carmack_episodes_sample(
            episodes_path,
            sample_lines,
            errors,
            validation_mode=mode,
            stratified_seed=seed,
        )
        _check_carmack_event_semantics(event_rows, errors)
        _check_carmack_episode_semantics(episode_rows, errors)
        if run_summary is not None:
            _check_carmack_sample_consistency(event_rows, episode_rows, run_summary, errors)
    else:
        if not score_path.exists():
            return {"ok": False, "errors": ["score.json not found"], "warnings": []}
        try:
            score = _load_json(score_path)
        except Exception as exc:  # pragma: no cover - invalid JSON path
            return {"ok": False, "errors": [f"failed to load score.json: {exc}"], "warnings": []}

        _check_multigame_config(config, errors)
        _check_config_hash_integrity(config, errors)
        _check_score_schema(score, errors)
        _check_score_tags(config, score, errors)
        _check_events_sample(events_path, sample_lines, errors, validation_mode=mode, stratified_seed=seed)
        _check_episodes_sample(episodes_path, sample_lines, errors, validation_mode=mode, stratified_seed=seed)
        _check_segments_sample(segments_path, sample_lines, errors, validation_mode=mode, stratified_seed=seed)

    ok = len(errors) == 0 and (not bool(fail_on_warnings) or len(warnings) == 0)
    return {"ok": ok, "errors": errors, "warnings": warnings}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate benchmark run schema for multi-game v1, Carmack multi-game v1, or single-game Carmack profile."
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory path.")
    parser.add_argument(
        "--sample-event-lines",
        type=int,
        default=0,
        help=(
            "Validation row budget for sample/stratified modes; full mode scans all rows. "
            "Stratified mode requires > 0 and may fail if budget is too small for mandatory strata."
        ),
    )
    parser.add_argument(
        "--validation-mode",
        type=str,
        default=VALIDATION_MODE_SAMPLE,
        choices=[VALIDATION_MODE_SAMPLE, VALIDATION_MODE_STRATIFIED, VALIDATION_MODE_FULL],
        help="Validation mode: sample (first N), stratified (coverage + deterministic rank), or full scan.",
    )
    parser.add_argument(
        "--stratified-seed",
        type=int,
        default=0,
        help="Deterministic seed used by stratified mode row ranking.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        type=int,
        choices=[0, 1],
        default=0,
        help="When set to 1, treat validator warnings as failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_contract(
        Path(args.run_dir),
        sample_event_lines=int(args.sample_event_lines),
        validation_mode=str(args.validation_mode),
        stratified_seed=int(args.stratified_seed),
        fail_on_warnings=bool(int(args.fail_on_warnings)),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
