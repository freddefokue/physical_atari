"""Calibration suite runner for continual multi-game Atari benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmark.score_run import score_run

EPS = 1e-9


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    config_path: str
    seeds: Tuple[int, ...]
    agents: Tuple[str, ...]
    config_overrides: Dict[str, Any]
    enforce_smoke_expectations: bool = False


SUITES: Dict[str, SuiteSpec] = {
    "smoke": SuiteSpec(
        name="smoke",
        config_path="configs/v1_smoke.json",
        seeds=(0, 1),
        agents=("repeat", "random", "tinydqn"),
        config_overrides={},
        enforce_smoke_expectations=True,
    ),
    "calib": SuiteSpec(
        name="calib",
        config_path="configs/v1_smoke.json",
        seeds=(0, 1, 2),
        agents=("repeat", "random", "tinydqn"),
        config_overrides={
            "num_cycles": 2,
            "base_visit_frames": 5000,
            "min_visit_frames": 1200,
            "delay": 4,
            "jitter_pct": 0.07,
        },
        enforce_smoke_expectations=False,
    ),
    "paper": SuiteSpec(
        name="paper",
        config_path="configs/v1_reference.json",
        seeds=(0, 1, 2),
        agents=("repeat", "random", "tinydqn"),
        config_overrides={},
        enforce_smoke_expectations=False,
    ),
}


def resolve_config_path(config_arg: Optional[str], suite_config_path: str) -> Path:
    """Resolve config file path robustly, independent of caller CWD."""

    if config_arg is not None:
        return Path(config_arg).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / suite_config_path).resolve()


def parse_int_csv(value: str) -> List[int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected a comma-separated list of integers")
    return [int(part) for part in parts]


def parse_str_csv(value: str) -> List[str]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected a comma-separated list of non-empty values")
    return parts


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two config dicts without mutating inputs."""

    result: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _stats(values: Sequence[float]) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = None
    if mean != 0.0:
        cv = float(std / abs(mean))
    return {
        "count": int(arr.shape[0]),
        "mean": mean,
        "median": float(np.median(arr)),
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv": cv,
    }


def aggregate_summary(run_results: Sequence[Dict[str, Any]], agents: Sequence[str]) -> Dict[str, Any]:
    per_agent: Dict[str, Any] = {}

    for agent in agents:
        agent_runs = [row for row in run_results if row.get("agent") == agent]
        success_runs = [row for row in agent_runs if row.get("status") == "success" and isinstance(row.get("score"), dict)]

        def metric_values(metric: str) -> List[float]:
            values: List[float] = []
            for row in success_runs:
                raw = row["score"].get(metric)
                if raw is None:
                    continue
                value = float(raw)
                if math.isfinite(value):
                    values.append(value)
            return values

        final_vals = metric_values("final_score")
        mean_vals = metric_values("mean_score")
        bottom_vals = metric_values("bottom_k_score")
        fps_vals = metric_values("fps")
        frames_vals = metric_values("frames")
        forgetting_vals = metric_values("forgetting_index_mean")
        plasticity_vals = metric_values("plasticity_mean")

        per_agent[agent] = {
            "counts": {
                "requested": int(len(agent_runs)),
                "success": int(len(success_runs)),
                "failed": int(sum(1 for row in agent_runs if row.get("status") == "failed")),
                "skipped": int(sum(1 for row in agent_runs if row.get("status") == "skipped")),
            },
            "final_score_stats": _stats(final_vals),
            "mean_score_stats": _stats(mean_vals),
            "bottom_k_score_stats": _stats(bottom_vals),
            "fps_mean": float(np.mean(fps_vals)) if fps_vals else None,
            "frames_mean": float(np.mean(frames_vals)) if frames_vals else None,
            "forgetting_index_mean": float(np.mean(forgetting_vals)) if forgetting_vals else None,
            "forgetting_index_median": float(np.median(forgetting_vals)) if forgetting_vals else None,
            "plasticity_mean": float(np.mean(plasticity_vals)) if plasticity_vals else None,
            "plasticity_median": float(np.median(plasticity_vals)) if plasticity_vals else None,
        }

    return {
        "per_agent": per_agent,
        "total_runs": int(len(run_results)),
        "success_count": int(sum(1 for row in run_results if row.get("status") == "success")),
        "failed_count": int(sum(1 for row in run_results if row.get("status") == "failed")),
        "skipped_count": int(sum(1 for row in run_results if row.get("status") == "skipped")),
    }


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _expected_total_frames(config: Dict[str, Any]) -> Optional[int]:
    total = config.get("total_scheduled_frames")
    if total is not None:
        return int(total)
    schedule = config.get("schedule")
    if isinstance(schedule, list):
        total_schedule = 0
        for row in schedule:
            if not isinstance(row, dict) or row.get("visit_frames") is None:
                return None
            total_schedule += int(row["visit_frames"])
        return int(total_schedule)
    return None


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _collect_non_finite_numeric_paths(payload: Any, prefix: str = "") -> List[str]:
    paths: List[str] = []
    if isinstance(payload, bool) or payload is None:
        return paths
    if isinstance(payload, (int, float)):
        if not math.isfinite(float(payload)):
            paths.append(prefix or "<root>")
        return paths
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            child_prefix = f"{prefix}.{key_str}" if prefix else key_str
            paths.extend(_collect_non_finite_numeric_paths(value, child_prefix))
        return paths
    if isinstance(payload, (list, tuple)):
        for idx, value in enumerate(payload):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            paths.extend(_collect_non_finite_numeric_paths(value, child_prefix))
        return paths
    return paths


def _parse_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def evaluate_smoke_expectations(run_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    successful = [row for row in run_results if row.get("status") == "success"]
    if not successful:
        return {"passed": False, "errors": ["No successful runs found."], "warnings": warnings}

    for row in successful:
        run_dir_str = row.get("run_dir")
        if not run_dir_str:
            errors.append(f"Missing run_dir for successful run: agent={row.get('agent')} seed={row.get('seed')}")
            continue
        run_dir = Path(run_dir_str)
        events_path = run_dir / "events.jsonl"
        episodes_path = run_dir / "episodes.jsonl"
        score_path = run_dir / "score.json"
        config_path = run_dir / "config.json"

        if not events_path.exists() or not config_path.exists():
            errors.append(f"Missing events/config logs in {run_dir}")
            continue

        config = load_json(config_path)
        expected_frames = _expected_total_frames(config)
        if expected_frames is None:
            errors.append(f"Unable to compute expected scheduled frames in {run_dir}")
        else:
            observed_frames = _line_count(events_path)
            if observed_frames != expected_frames:
                errors.append(
                    f"events.jsonl line count mismatch in {run_dir}: expected={expected_frames} observed={observed_frames}"
                )

        if not episodes_path.exists() or _line_count(episodes_path) == 0:
            errors.append(f"episodes.jsonl missing or empty in {run_dir}")

        if not score_path.exists():
            errors.append(f"score.json missing in {run_dir}")
        else:
            score = load_json(score_path)
            required_keys = ("final_score", "mean_score", "bottom_k_score", "fps", "frames")
            for key in required_keys:
                if key not in score:
                    errors.append(f"score.json missing '{key}' in {run_dir}")
            for key in required_keys:
                if key not in score:
                    continue
                if _finite_or_none(score.get(key)) is None:
                    errors.append(f"score.json non-finite '{key}' in {run_dir}")
            non_finite_paths = _collect_non_finite_numeric_paths(score)
            for path in non_finite_paths:
                errors.append(f"score.json contains non-finite numeric value at '{path}' in {run_dir}")

    # Aggregate outcome checks.
    def _agent_mean_final(agent: str) -> Optional[float]:
        values = []
        for row in successful:
            if row.get("agent") != agent:
                continue
            score = row.get("score")
            if not isinstance(score, dict):
                continue
            value = _finite_or_none(score.get("final_score"))
            if value is not None:
                values.append(value)
        if not values:
            return None
        return float(np.mean(values))

    random_mean = _agent_mean_final("random")
    repeat_mean = _agent_mean_final("repeat")
    if random_mean is None or repeat_mean is None:
        errors.append("Missing random or repeat successful scores needed for ordering expectation.")
    elif random_mean + EPS < repeat_mean:
        errors.append(
            f"Expected RandomAgent mean final_score >= RepeatActionAgent, got random={random_mean:.6f} repeat={repeat_mean:.6f}."
        )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def evaluate_calib_expectations(run_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    successful = [row for row in run_results if row.get("status") == "success"]
    if not successful:
        return {"passed": False, "errors": ["No successful runs found."], "warnings": warnings}

    for row in successful:
        score = row.get("score")
        if not isinstance(score, dict):
            continue
        mean_val = _finite_or_none(score.get("mean_score"))
        bottom_val = _finite_or_none(score.get("bottom_k_score"))
        if mean_val is None or bottom_val is None:
            continue
        if bottom_val > mean_val + EPS:
            errors.append(
                f"Expected bottom_k_score <= mean_score for agent={row.get('agent')} seed={row.get('seed')}, "
                f"got bottom_k={bottom_val:.6f} mean={mean_val:.6f}."
            )

    random_success = [row for row in successful if row.get("agent") == "random" and isinstance(row.get("score"), dict)]
    random_has_forgetting = any(
        _finite_or_none(row["score"].get("forgetting_index_mean")) is not None for row in random_success
    )
    random_has_plasticity = any(
        _finite_or_none(row["score"].get("plasticity_mean")) is not None for row in random_success
    )
    if not random_has_forgetting:
        errors.append("Expected at least one random successful run with finite forgetting_index_mean.")
    if not random_has_plasticity:
        errors.append("Expected at least one random successful run with finite plasticity_mean.")

    tiny_requested = [row for row in run_results if row.get("agent") == "tinydqn"]
    tiny_success = [row for row in successful if row.get("agent") == "tinydqn"]
    if tiny_requested and not tiny_success:
        errors.append("Expected at least one successful tinydqn run in calib suite.")
    elif tiny_success:
        has_training_signal = False
        for row in tiny_success:
            stats = row.get("agent_stats")
            if not isinstance(stats, dict):
                warnings.append(f"Missing agent_stats for tinydqn seed={row.get('seed')}")
                continue

            train_steps = _parse_int_or_none(stats.get("train_steps"))
            replay_size = _parse_int_or_none(stats.get("replay_size"))
            replay_min_size = _parse_int_or_none(stats.get("replay_min_size"))
            if train_steps is None or replay_size is None or replay_min_size is None:
                errors.append(f"Malformed agent_stats for tinydqn seed={row.get('seed')}")
                continue

            if train_steps > 0 and replay_size >= replay_min_size:
                has_training_signal = True
                break

        if not has_training_signal:
            errors.append(
                "TinyDQN training gate failed: require at least one successful tinydqn run with "
                "train_steps > 0 and replay_size >= replay_min_size."
            )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def _success_metric_by_key(run_rows: Sequence[Dict[str, Any]], metric: str) -> Dict[Tuple[str, int], float]:
    out: Dict[Tuple[str, int], float] = {}
    for row in run_rows:
        if row.get("status") != "success":
            continue
        agent = str(row.get("agent"))
        seed_raw = row.get("seed")
        if seed_raw is None:
            continue
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            continue
        score = row.get("score")
        if not isinstance(score, dict):
            continue
        metric_val = _finite_or_none(score.get(metric))
        if metric_val is None:
            continue
        out[(agent, seed)] = float(metric_val)
    return out


def _stable_json_sha256(payload: Mapping[str, Any]) -> str:
    def _normalize_non_finite(value: Any) -> Any:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, list):
            return [_normalize_non_finite(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_normalize_non_finite(item) for item in value)
        if isinstance(value, dict):
            return {str(key): _normalize_non_finite(item) for key, item in value.items()}
        return value

    canonical = json.dumps(
        _normalize_non_finite(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate_rollout_acceptance(
    run_results: Sequence[Dict[str, Any]],
    baseline_summary: Mapping[str, Any],
    *,
    metric: str,
    mean_floor: float,
    worst_floor: float,
    min_overlap: int,
    require_full_baseline_coverage: bool = False,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    baseline_runs_raw = baseline_summary.get("runs")
    if not isinstance(baseline_runs_raw, list):
        return {
            "passed": False,
            "errors": ["Baseline summary missing list key: runs"],
            "warnings": warnings,
        }

    current_map = _success_metric_by_key(run_results, metric=metric)
    baseline_map = _success_metric_by_key(list(baseline_runs_raw), metric=metric)

    current_keys = set(current_map.keys())
    baseline_keys = set(baseline_map.keys())
    overlap_keys = sorted(current_keys & baseline_keys)

    missing_current = sorted(baseline_keys - current_keys)
    if missing_current:
        shown = ", ".join(f"{agent}:{seed}" for agent, seed in missing_current[:8])
        msg = f"Missing successful current runs for {len(missing_current)} baseline keys (first: {shown})"
        if require_full_baseline_coverage:
            errors.append(msg)
        else:
            warnings.append(msg)

    missing_baseline = sorted(current_keys - baseline_keys)
    if missing_baseline:
        shown = ", ".join(f"{agent}:{seed}" for agent, seed in missing_baseline[:8])
        warnings.append(
            f"{len(missing_baseline)} successful current runs are not present in baseline (first: {shown})"
        )

    if len(overlap_keys) < int(min_overlap):
        errors.append(f"Insufficient overlap: need >= {int(min_overlap)} successful run keys, got {len(overlap_keys)}")
        return {
            "passed": False,
            "errors": errors,
            "warnings": warnings,
            "metric": str(metric),
            "thresholds": {
                "mean_floor": float(mean_floor),
                "worst_floor": float(worst_floor),
                "min_overlap": int(min_overlap),
                "require_full_baseline_coverage": bool(require_full_baseline_coverage),
            },
            "overlap_count": int(len(overlap_keys)),
        }

    delta_by_agent: Dict[str, List[float]] = {}
    delta_rows: List[Dict[str, Any]] = []
    for key in overlap_keys:
        agent, seed = key
        delta = float(current_map[key] - baseline_map[key])
        delta_by_agent.setdefault(agent, []).append(delta)
        delta_rows.append(
            {
                "agent": str(agent),
                "seed": int(seed),
                "baseline": float(baseline_map[key]),
                "current": float(current_map[key]),
                "delta": float(delta),
            }
        )

    per_agent: Dict[str, Dict[str, Any]] = {}
    for agent, deltas in sorted(delta_by_agent.items()):
        arr = np.asarray(deltas, dtype=np.float64)
        mean_delta = float(np.mean(arr))
        worst_delta = float(np.min(arr))
        per_agent[str(agent)] = {
            "count": int(arr.shape[0]),
            "mean_delta": mean_delta,
            "worst_delta": worst_delta,
        }
        if mean_delta + EPS < float(mean_floor):
            errors.append(
                f"Rollout gate failed for agent={agent}: mean {metric} delta {mean_delta:.6f} < floor {float(mean_floor):.6f}"
            )
        if worst_delta + EPS < float(worst_floor):
            errors.append(
                f"Rollout gate failed for agent={agent}: worst-seed {metric} delta {worst_delta:.6f} < floor {float(worst_floor):.6f}"
            )

    all_deltas = [row["delta"] for row in delta_rows]
    mean_all = float(np.mean(np.asarray(all_deltas, dtype=np.float64))) if all_deltas else None
    worst_all = float(np.min(np.asarray(all_deltas, dtype=np.float64))) if all_deltas else None

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "metric": str(metric),
        "thresholds": {
            "mean_floor": float(mean_floor),
            "worst_floor": float(worst_floor),
            "min_overlap": int(min_overlap),
            "require_full_baseline_coverage": bool(require_full_baseline_coverage),
        },
        "overlap_count": int(len(overlap_keys)),
        "missing_current_count": int(len(missing_current)),
        "missing_baseline_count": int(len(missing_baseline)),
        "overlap_keys": [{"agent": str(agent), "seed": int(seed)} for agent, seed in overlap_keys],
        "aggregate": {
            "mean_delta": mean_all,
            "worst_delta": worst_all,
        },
        "per_agent": per_agent,
    }


def _parse_run_dir_from_stdout(stdout: str) -> Optional[Path]:
    matches = re.findall(r"Run complete:\s*(.+)", stdout)
    if not matches:
        return None
    return Path(matches[-1].strip())


def _pick_new_run_dir(logdir: Path, before: Iterable[Path]) -> Optional[Path]:
    before_set = {Path(p) for p in before}
    after = [path for path in logdir.iterdir() if path.is_dir()]
    created = [path for path in after if path not in before_set]
    if not created:
        return None
    created.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return created[0]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _should_stream_line(line: str) -> bool:
    stripped = str(line).strip()
    return stripped != "Sending Reset..."


def _run_streaming(cmd: Sequence[str]) -> Tuple[int, str]:
    """
    Run a subprocess while streaming combined stdout/stderr to the parent stdout.

    Returns the subprocess returncode and combined output text.
    """

    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: List[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            lines.append(line)
            if _should_stream_line(line):
                print(line, end="", flush=True)
        proc.stdout.close()
    returncode = proc.wait()
    return int(returncode), "".join(lines)


def _build_scoring_args(config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    scoring_cfg = config.get("scoring_defaults") if isinstance(config.get("scoring_defaults"), dict) else {}

    window_episodes = int(cli_args.window_episodes) if cli_args.window_episodes is not None else int(scoring_cfg.get("window_episodes", 20))
    bottom_k_frac = float(cli_args.bottom_k_frac) if cli_args.bottom_k_frac is not None else float(scoring_cfg.get("bottom_k_frac", 0.25))
    revisit_episodes = int(cli_args.revisit_episodes) if cli_args.revisit_episodes is not None else int(scoring_cfg.get("revisit_episodes", 5))

    weights_raw = scoring_cfg.get("final_score_weights", [0.5, 0.5])
    if isinstance(weights_raw, (list, tuple)) and len(weights_raw) == 2:
        weights = (float(weights_raw[0]), float(weights_raw[1]))
    else:
        weights = (0.5, 0.5)

    return {
        "window_episodes": window_episodes,
        "bottom_k_frac": bottom_k_frac,
        "revisit_episodes": revisit_episodes,
        "final_score_weights": weights,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark calibration suites.")
    parser.add_argument("--suite", type=str, choices=sorted(SUITES.keys()), default="smoke", help="Calibration suite preset.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for calibration artifacts.")
    parser.add_argument("--config", type=str, default=None, help="Optional config path override (JSON).")
    parser.add_argument("--seeds", type=str, default=None, help="Optional comma-separated seed override.")
    parser.add_argument("--agents", type=str, default=None, help="Optional comma-separated agent override.")
    parser.add_argument("--window-episodes", type=int, default=None, help="Override scoring window_episodes.")
    parser.add_argument("--bottom-k-frac", type=float, default=None, help="Override scoring bottom_k_frac.")
    parser.add_argument("--revisit-episodes", type=int, default=None, help="Override scoring revisit_episodes.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for launching runs.")
    parser.add_argument(
        "--rollout-baseline-summary",
        type=str,
        default=None,
        help="Optional baseline summary.json path for rollout acceptance gates.",
    )
    parser.add_argument(
        "--rollout-enforce",
        type=int,
        choices=[0, 1],
        default=0,
        help="When set, fail the suite if rollout_acceptance gates fail.",
    )
    parser.add_argument(
        "--rollout-metric",
        type=str,
        default="final_score",
        help="Score metric key used for rollout delta gates (default: final_score).",
    )
    parser.add_argument(
        "--rollout-mean-floor",
        type=float,
        default=-0.10,
        help="Minimum allowed mean metric delta per agent/current-baseline.",
    )
    parser.add_argument(
        "--rollout-worst-floor",
        type=float,
        default=-0.25,
        help="Minimum allowed worst-seed metric delta per agent/current-baseline.",
    )
    parser.add_argument(
        "--rollout-min-overlap",
        type=int,
        default=1,
        help="Minimum number of successful overlapping (agent, seed) keys required.",
    )
    parser.add_argument(
        "--rollout-require-full-baseline-coverage",
        type=int,
        choices=[0, 1],
        default=0,
        help="If set, fail rollout when any baseline successful (agent, seed) key is missing in current runs.",
    )
    return parser.parse_args(args=argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    suite = SUITES[str(args.suite)]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_config_path(args.config, suite.config_path)
    base_config = load_json(config_path)
    resolved_config = merge_config(base_config, suite.config_overrides)

    resolved_config_path = out_dir / f"{suite.name}_resolved_config.json"
    _write_json(resolved_config_path, resolved_config)

    scoring_args = _build_scoring_args(resolved_config, args)

    seeds = tuple(parse_int_csv(args.seeds)) if args.seeds else suite.seeds
    agents = tuple(parse_str_csv(args.agents)) if args.agents else suite.agents

    torch_available = _torch_available()

    run_results: List[Dict[str, Any]] = []

    for agent in agents:
        for seed in seeds:
            if agent == "tinydqn" and not torch_available:
                print(f"[skip] agent={agent} seed={seed}: torch is not installed")
                run_results.append(
                    {
                        "agent": str(agent),
                        "seed": int(seed),
                        "status": "skipped",
                        "reason": "torch_not_installed",
                        "run_dir": None,
                        "score": None,
                    }
                )
                continue

            run_base = out_dir / str(agent) / f"seed_{seed}"
            run_base.mkdir(parents=True, exist_ok=True)
            before_dirs = [path for path in run_base.iterdir() if path.is_dir()]

            cmd = [
                str(args.python),
                "-u",
                "-m",
                "benchmark.run_multigame",
                "--config",
                str(resolved_config_path),
                "--seed",
                str(seed),
                "--agent",
                str(agent),
                "--logdir",
                str(run_base),
            ]
            if agent == "repeat":
                cmd.extend(["--repeat-action-idx", "0"])

            print(f"[run] agent={agent} seed={seed}")
            returncode, combined_output = _run_streaming(cmd)
            if returncode != 0:
                error_tail = (combined_output or "").strip().splitlines()
                error_msg = error_tail[-1] if error_tail else f"exit_code={returncode}"
                print(f"[fail] agent={agent} seed={seed}: {error_msg}")
                run_results.append(
                    {
                        "agent": str(agent),
                        "seed": int(seed),
                        "status": "failed",
                        "reason": f"run_multigame_failed: {error_msg}",
                        "returncode": int(returncode),
                        "run_dir": None,
                        "score": None,
                        "stdout_tail": "\n".join((combined_output or "").splitlines()[-20:]),
                        "stderr_tail": "",
                    }
                )
                continue

            run_dir = _parse_run_dir_from_stdout(combined_output)
            if run_dir is None or not run_dir.exists():
                run_dir = _pick_new_run_dir(run_base, before_dirs)
            if run_dir is None or not run_dir.exists():
                print(f"[fail] agent={agent} seed={seed}: could not determine run directory")
                run_results.append(
                    {
                        "agent": str(agent),
                        "seed": int(seed),
                        "status": "failed",
                        "reason": "run_directory_not_found",
                        "run_dir": None,
                        "score": None,
                    }
                )
                continue

            try:
                score = score_run(
                    run_dir=run_dir,
                    window_episodes=int(scoring_args["window_episodes"]),
                    bottom_k_frac=float(scoring_args["bottom_k_frac"]),
                    revisit_episodes=int(scoring_args["revisit_episodes"]),
                    final_score_weights=tuple(scoring_args["final_score_weights"]),
                )
                _write_json(run_dir / "score.json", score)
            except Exception as exc:  # pragma: no cover - defensive guard
                print(f"[fail] agent={agent} seed={seed}: score_run failed: {exc}")
                run_results.append(
                    {
                        "agent": str(agent),
                        "seed": int(seed),
                        "status": "failed",
                        "reason": f"score_run_failed: {exc}",
                        "run_dir": str(run_dir),
                        "score": None,
                    }
                )
                continue

            print(f"[ok] agent={agent} seed={seed} final_score={score.get('final_score')}")
            agent_stats_path = run_dir / "agent_stats.json"
            agent_stats = None
            if agent_stats_path.exists():
                try:
                    payload = load_json(agent_stats_path)
                    if isinstance(payload, dict):
                        agent_stats = payload
                except Exception:  # pragma: no cover - defensive
                    agent_stats = None
            run_results.append(
                {
                    "agent": str(agent),
                    "seed": int(seed),
                    "status": "success",
                    "run_dir": str(run_dir),
                    "score_path": str(run_dir / "score.json"),
                    "score": score,
                    "agent_stats": agent_stats,
                }
            )

    aggregate = aggregate_summary(run_results, agents=agents)
    summary: Dict[str, Any] = {
        "suite": suite.name,
        "out_dir": str(out_dir),
        "resolved_config_path": str(resolved_config_path),
        "seeds": [int(seed) for seed in seeds],
        "agents": [str(agent) for agent in agents],
        "scoring": {
            "window_episodes": int(scoring_args["window_episodes"]),
            "bottom_k_frac": float(scoring_args["bottom_k_frac"]),
            "revisit_episodes": int(scoring_args["revisit_episodes"]),
            "final_score_weights": [
                float(scoring_args["final_score_weights"][0]),
                float(scoring_args["final_score_weights"][1]),
            ],
        },
        "runs": run_results,
        **aggregate,
    }

    if suite.enforce_smoke_expectations:
        expectations = evaluate_smoke_expectations(run_results)
        summary["smoke_expectations"] = expectations
    if suite.name == "calib":
        summary["calib_expectations"] = evaluate_calib_expectations(run_results)
    if args.rollout_baseline_summary:
        baseline_path = Path(str(args.rollout_baseline_summary)).expanduser().resolve()
        rollout_result: Dict[str, Any]
        if not baseline_path.exists():
            rollout_result = {
                "passed": False,
                "errors": [f"rollout baseline summary not found: {baseline_path}"],
                "warnings": [],
            }
        else:
            baseline_summary = load_json(baseline_path)
            rollout_result = evaluate_rollout_acceptance(
                run_results,
                baseline_summary,
                metric=str(args.rollout_metric),
                mean_floor=float(args.rollout_mean_floor),
                worst_floor=float(args.rollout_worst_floor),
                min_overlap=int(args.rollout_min_overlap),
                require_full_baseline_coverage=bool(int(args.rollout_require_full_baseline_coverage)),
            )
            rollout_result["baseline_summary_sha256"] = _stable_json_sha256(baseline_summary)
        rollout_result["baseline_summary_path"] = str(baseline_path)
        summary["rollout_acceptance"] = rollout_result

    summary_path = out_dir / "summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))

    failed = False
    if suite.enforce_smoke_expectations:
        failed = failed or (not bool(summary["smoke_expectations"].get("passed", False)))
    if suite.name == "calib":
        failed = failed or (not bool(summary["calib_expectations"].get("passed", False)))
    if bool(int(args.rollout_enforce)) and "rollout_acceptance" in summary:
        failed = failed or (not bool(summary["rollout_acceptance"].get("passed", False)))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
