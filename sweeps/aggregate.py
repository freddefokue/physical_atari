"""Aggregate completed sweep trial results into ranked summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sweeps.common import HOSTS, STAGES, ensure_sweep_dirs, iter_result_records, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed sweep results.")
    parser.add_argument("--host", type=str, required=True, choices=sorted(HOSTS), help="Target host.")
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=["ppo", "delay_target", "bbf", "rainbow_dqn", "sac"],
        help="Agent family.",
    )
    parser.add_argument("--stage", type=str, default="stage0", choices=sorted(STAGES), help="Sweep stage.")
    return parser.parse_args()


def _finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def forgetting_rank_tuple(value: Any) -> Tuple[int, float]:
    parsed = _finite_float(value)
    if parsed is None:
        return (2, math.inf)
    if parsed <= 0.0:
        return (0, 0.0)
    return (1, parsed)


def rank_key(record: Dict[str, Any]) -> Tuple[float, int, float, float, str]:
    final_score = _finite_float(record.get("final_score"))
    plasticity = _finite_float(record.get("plasticity_mean"))
    forgetting_bucket, forgetting_value = forgetting_rank_tuple(record.get("forgetting_index_mean"))
    return (
        -(final_score if final_score is not None else -math.inf),
        forgetting_bucket,
        forgetting_value,
        -(plasticity if plasticity is not None else -math.inf),
        str(record.get("trial_id", "")),
    )


def summarize_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"total": 0, "completed": 0, "failed": 0}
    for record in records:
        counts["total"] += 1
        status = str(record.get("status", ""))
        if status == "completed":
            counts["completed"] += 1
        elif status == "failed":
            counts["failed"] += 1
    return counts


def flatten_ranked_record(record: Dict[str, Any], rank: int) -> Dict[str, Any]:
    sampled = record.get("sampled_hyperparameters")
    row = {
        "rank": int(rank),
        "trial_id": record.get("trial_id"),
        "status": record.get("status"),
        "training_seed": record.get("training_seed"),
        "final_score": record.get("final_score"),
        "forgetting_index_mean": record.get("forgetting_index_mean"),
        "plasticity_mean": record.get("plasticity_mean"),
        "mean_score": record.get("mean_score"),
        "bottom_k_score": record.get("bottom_k_score"),
        "run_dir": record.get("run_dir"),
        "benchmark_config_path": record.get("benchmark_config_path"),
        "gpu": record.get("gpu"),
        "host": record.get("host"),
    }
    if isinstance(sampled, dict):
        for key in sorted(sampled):
            row[key] = sampled[key]
    return row


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    paths = ensure_sweep_dirs(args.host, args.family, args.stage)
    records = list(iter_result_records(paths["results"]))
    counts = summarize_counts(records)
    completed = [record for record in records if record.get("status") == "completed"]
    ranked = sorted(completed, key=rank_key)

    ranked_rows = [flatten_ranked_record(record, index + 1) for index, record in enumerate(ranked)]
    summary = {
        "host": args.host,
        "family": args.family,
        "stage": args.stage,
        "generated_at": utc_now_iso(),
        "counts": counts,
        "ranking_rule": [
            "higher final_score",
            "lower positive forgetting_index_mean",
            "higher plasticity_mean",
            "trial_id",
        ],
        "ranked_trials": ranked_rows,
    }

    summary_json_path = paths["summaries"] / "summary.json"
    summary_csv_path = paths["summaries"] / "summary.csv"
    write_json(summary_json_path, summary)
    write_summary_csv(summary_csv_path, ranked_rows)

    print(json.dumps({"completed": counts["completed"], "failed": counts["failed"], "summary_json": str(summary_json_path), "summary_csv": str(summary_csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
