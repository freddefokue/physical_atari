"""Shared helpers for sweep generation, launching, and aggregation."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

HOSTS = {"obsession", "donald"}
STAGES = {"stage0"}
REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEPS_ROOT = REPO_ROOT / "sweeps"
OUTPUT_ROOT = SWEEPS_ROOT / "output"
TEMPLATES_ROOT = SWEEPS_ROOT / "templates"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp_path, path)


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_csv_ints(value: str) -> List[int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected a comma-separated list of integers")
    return [int(part) for part in parts]


def parse_seed_list(value: str) -> List[int]:
    seeds = parse_csv_ints(value)
    seen = set()
    ordered: List[int] = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            ordered.append(seed)
    return ordered


def parse_gpu_list(value: str) -> List[int]:
    gpus = parse_csv_ints(value)
    if any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU ids must be >= 0")
    return gpus


def normalize_for_hash(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): normalize_for_hash(payload[key]) for key in sorted(payload)}
    if isinstance(payload, list):
        return [normalize_for_hash(item) for item in payload]
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ValueError("Sweep payloads must not contain non-finite floats")
        return float(payload)
    return payload


def load_template(name: str) -> Dict[str, Any]:
    return read_json(TEMPLATES_ROOT / name)


def sweep_root(host: str, family: str, stage: str) -> Path:
    return OUTPUT_ROOT / host / family / stage


def sweep_paths(host: str, family: str, stage: str) -> Dict[str, Path]:
    root = sweep_root(host, family, stage)
    return {
        "root": root,
        "generated_trials": root / "generated_trials",
        "queue": root / "queue",
        "queue_pending": root / "queue" / "pending",
        "queue_claimed": root / "queue" / "claimed",
        "queue_completed": root / "queue" / "completed",
        "queue_failed": root / "queue" / "failed",
        "logs": root / "logs",
        "logs_trials": root / "logs" / "trials",
        "runs": root / "runs",
        "results": root / "results",
        "summaries": root / "summaries",
    }


def ensure_sweep_dirs(host: str, family: str, stage: str) -> Dict[str, Path]:
    paths = sweep_paths(host, family, stage)
    for path in paths.values():
        if path.suffix:
            continue
        path.mkdir(parents=True, exist_ok=True)
    return paths


def manifest_path(paths: Dict[str, Path]) -> Path:
    return paths["generated_trials"] / "manifest.json"


def load_manifest(paths: Dict[str, Path]) -> Dict[str, Any]:
    path = manifest_path(paths)
    if not path.exists():
        raise FileNotFoundError(f"Missing sweep manifest: {path}")
    return read_json(path)


def trial_spec_paths(paths: Dict[str, Path], trial_id: str) -> Dict[str, Path]:
    return {
        "generated_trial_path": paths["generated_trials"] / f"{trial_id}.json",
        "queue_pending_path": paths["queue_pending"] / f"{trial_id}.json",
        "queue_claimed_path": paths["queue_claimed"] / f"{trial_id}.json",
        "queue_completed_path": paths["queue_completed"] / f"{trial_id}.json",
        "queue_failed_path": paths["queue_failed"] / f"{trial_id}.json",
        "trial_run_root": paths["runs"] / trial_id,
        "benchmark_config_path": paths["runs"] / trial_id / "benchmark_config.json",
        "stdout_log_path": paths["logs_trials"] / f"{trial_id}.stdout.log",
        "stderr_log_path": paths["logs_trials"] / f"{trial_id}.stderr.log",
        "result_path": paths["results"] / f"{trial_id}.json",
    }


def load_trial_specs(paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    manifest = load_manifest(paths)
    trial_files = manifest.get("trial_files")
    if not isinstance(trial_files, list) or not trial_files:
        raise ValueError(f"Invalid or empty trial_files in {manifest_path(paths)}")
    specs: List[Dict[str, Any]] = []
    for name in trial_files:
        specs.append(read_json(paths["generated_trials"] / str(name)))
    return specs


def iter_result_records(results_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(results_dir.glob("*.json")):
        yield read_json(path)
