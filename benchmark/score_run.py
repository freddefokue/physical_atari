"""Score and analyze continual benchmark runs from log artifacts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from benchmark.contract import BENCHMARK_CONTRACT_VERSION, compute_contract_hash


@dataclass
class EpisodeRecord:
    game_id: str
    start_frame: int
    end_frame: int
    return_value: float
    ended_by: str


@dataclass
class SegmentRecord:
    game_id: str
    segment_id: int
    start_frame: int
    end_frame: int
    return_value: float
    ended_by: str


@dataclass
class VisitRecord:
    visit_idx: int
    game_id: str
    start_frame: int
    end_frame: int
    cycle_idx: Optional[int] = None


@dataclass
class FrameRewardTable:
    frame_indices: np.ndarray
    prefix_rewards: np.ndarray


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return iter(())

    def _generator() -> Iterator[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    return _generator()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(_iter_jsonl(path))


def _pick_int(payload: Dict[str, Any], keys: Sequence[str], default: Optional[int] = None) -> Optional[int]:
    for key in keys:
        if key in payload and payload[key] is not None:
            return int(payload[key])
    return default


def _pick_float(payload: Dict[str, Any], keys: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        if key in payload and payload[key] is not None:
            return float(payload[key])
    return default


def _pick_str(payload: Dict[str, Any], keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        if key in payload and payload[key] is not None:
            return str(payload[key])
    return default


def _normalize_ended_by(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"terminated", "game_over", "life_loss"}:
        return "terminated"
    if value in {"truncated", "time_limit"}:
        return "truncated"
    return value


def _infer_episode_ended_by(payload: Dict[str, Any]) -> str:
    ended_by = _normalize_ended_by(_pick_str(payload, ["ended_by", "termination_reason"]))
    if ended_by is not None:
        return ended_by

    terminated = payload.get("terminated")
    truncated = payload.get("truncated")
    if terminated is True:
        return "terminated"
    if truncated is True:
        return "truncated"
    if truncated is False:
        return "terminated"
    return "terminated"


def _parse_segments(rows: Iterable[Dict[str, Any]]) -> List[SegmentRecord]:
    segments: List[SegmentRecord] = []
    for row in rows:
        game_id = _pick_str(row, ["game_id"]) or "unknown"
        start = _pick_int(row, ["start_global_frame_idx", "start_frame_idx", "start_frame", "start"], default=0)
        end = _pick_int(row, ["end_global_frame_idx", "end_frame_idx", "end_frame", "end"], default=start)
        if start is None:
            start = 0
        if end is None:
            end = start
        ended_by = _normalize_ended_by(_pick_str(row, ["ended_by", "termination_reason"])) or "terminated"
        return_value = _pick_float(row, ["return", "segment_return", "episode_return"], default=0.0)
        segment_id = _pick_int(row, ["segment_id"], default=len(segments))
        if segment_id is None:
            segment_id = len(segments)

        segments.append(
            SegmentRecord(
                game_id=game_id,
                segment_id=int(segment_id),
                start_frame=int(start),
                end_frame=int(end),
                return_value=float(return_value or 0.0),
                ended_by=str(ended_by),
            )
        )

    segments.sort(key=lambda seg: (seg.start_frame, seg.end_frame, seg.segment_id))
    return segments


def _parse_episodes(rows: Iterable[Dict[str, Any]]) -> List[EpisodeRecord]:
    episodes: List[EpisodeRecord] = []
    for row in rows:
        game_id = _pick_str(row, ["game_id"]) or "unknown"
        start = _pick_int(row, ["start_global_frame_idx", "start_frame_idx", "start_frame", "start"])
        end = _pick_int(row, ["end_global_frame_idx", "end_frame_idx", "end_frame", "end"])
        length = _pick_int(row, ["length"])

        if start is None and end is None:
            continue
        if start is None and end is not None and length is not None:
            start = int(end) - int(length) + 1
        if end is None and start is not None and length is not None:
            end = int(start) + int(length) - 1
        if start is None:
            start = int(end if end is not None else 0)
        if end is None:
            end = int(start)

        ended_by = _infer_episode_ended_by(row)
        return_value = _pick_float(row, ["return", "episode_return"], default=0.0)
        episodes.append(
            EpisodeRecord(
                game_id=game_id,
                start_frame=int(start),
                end_frame=int(end),
                return_value=float(return_value or 0.0),
                ended_by=str(ended_by),
            )
        )

    episodes.sort(key=lambda ep: (ep.end_frame, ep.start_frame))
    return episodes


def _derive_visits_from_schedule(config: Dict[str, Any]) -> List[VisitRecord]:
    schedule = config.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        return []

    visits: List[VisitRecord] = []
    cursor = 0
    for idx, row in enumerate(schedule):
        game_id = _pick_str(row, ["game_id"])
        visit_frames = _pick_int(row, ["visit_frames"])
        if game_id is None or visit_frames is None or visit_frames <= 0:
            return []

        start = int(cursor)
        end = int(cursor + visit_frames - 1)
        cycle_idx = _pick_int(row, ["cycle_idx"])
        visit_idx = _pick_int(row, ["visit_idx"], default=idx)
        if visit_idx is None:
            visit_idx = idx

        visits.append(
            VisitRecord(
                visit_idx=int(visit_idx),
                game_id=str(game_id),
                start_frame=start,
                end_frame=end,
                cycle_idx=None if cycle_idx is None else int(cycle_idx),
            )
        )
        cursor = end + 1

    return visits


def _derive_visits_from_segments(segments: Sequence[SegmentRecord]) -> List[VisitRecord]:
    if not segments:
        return []

    visits: List[VisitRecord] = []
    current_start = segments[0].start_frame
    current_game = segments[0].game_id
    visit_idx = 0

    for idx, segment in enumerate(segments):
        next_segment = segments[idx + 1] if idx + 1 < len(segments) else None
        boundary = False

        if segment.ended_by == "truncated":
            boundary = True
        if next_segment is None:
            boundary = True
        elif next_segment.game_id != segment.game_id:
            boundary = True

        if boundary:
            visits.append(
                VisitRecord(
                    visit_idx=int(visit_idx),
                    game_id=str(current_game),
                    start_frame=int(current_start),
                    end_frame=int(segment.end_frame),
                    cycle_idx=None,
                )
            )
            visit_idx += 1
            if next_segment is not None:
                current_start = int(next_segment.start_frame)
                current_game = str(next_segment.game_id)

    return visits


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def _frame_in_visit(frame_idx: int, visit: VisitRecord) -> bool:
    return visit.start_frame <= frame_idx <= visit.end_frame


def _assign_episodes_to_visits(
    episodes: Sequence[EpisodeRecord],
    visits: Sequence[VisitRecord],
) -> Tuple[Dict[int, List[EpisodeRecord]], List[EpisodeRecord]]:
    by_visit: Dict[int, List[EpisodeRecord]] = {visit.visit_idx: [] for visit in visits}
    unassigned: List[EpisodeRecord] = []
    for ep in episodes:
        assigned = False
        for visit in visits:
            if ep.game_id != visit.game_id:
                continue
            if _frame_in_visit(ep.end_frame, visit):
                by_visit.setdefault(visit.visit_idx, []).append(ep)
                assigned = True
                break
        if not assigned:
            unassigned.append(ep)

    for visit_eps in by_visit.values():
        visit_eps.sort(key=lambda ep: ep.end_frame)
    return by_visit, unassigned


def _build_frame_reward_table(frame_indices: Sequence[int], rewards: Sequence[float]) -> FrameRewardTable:
    if not frame_indices:
        return FrameRewardTable(
            frame_indices=np.array([], dtype=np.int64),
            prefix_rewards=np.array([0.0], dtype=np.float64),
        )

    frame_array = np.asarray(frame_indices, dtype=np.int64)
    reward_array = np.asarray(rewards, dtype=np.float64)
    order = np.argsort(frame_array, kind="mergesort")
    frame_array = frame_array[order]
    reward_array = reward_array[order]

    unique_frames, first_indices = np.unique(frame_array, return_index=True)
    summed_rewards = np.add.reduceat(reward_array, first_indices)
    prefix = np.empty(len(summed_rewards) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(summed_rewards, out=prefix[1:])
    return FrameRewardTable(frame_indices=unique_frames, prefix_rewards=prefix)


def _sum_reward_range(reward_table: FrameRewardTable, start_frame: int, end_frame: int) -> float:
    if start_frame > end_frame or reward_table.frame_indices.size == 0:
        return 0.0
    left = int(np.searchsorted(reward_table.frame_indices, int(start_frame), side="left"))
    right = int(np.searchsorted(reward_table.frame_indices, int(end_frame), side="right"))
    if right <= left:
        return 0.0
    return float(reward_table.prefix_rewards[right] - reward_table.prefix_rewards[left])


def _visit_frame_count(visit: VisitRecord) -> int:
    return max(0, int(visit.end_frame) - int(visit.start_frame) + 1)


def _tail_rate(
    visit: VisitRecord,
    n: int,
    reward_tables: Dict[str, FrameRewardTable],
) -> float:
    n_eff = min(int(n), _visit_frame_count(visit))
    if n_eff <= 0:
        return 0.0
    start = max(int(visit.end_frame) - int(n) + 1, int(visit.start_frame))
    end = int(visit.end_frame)
    reward_table = reward_tables.get(visit.game_id)
    total = 0.0 if reward_table is None else _sum_reward_range(reward_table, start, end)
    return float(total / n_eff)


def _head_rate(
    visit: VisitRecord,
    n: int,
    reward_tables: Dict[str, FrameRewardTable],
) -> float:
    n_eff = min(int(n), _visit_frame_count(visit))
    if n_eff <= 0:
        return 0.0
    start = int(visit.start_frame)
    end = min(int(visit.start_frame) + int(n) - 1, int(visit.end_frame))
    reward_table = reward_tables.get(visit.game_id)
    total = 0.0 if reward_table is None else _sum_reward_range(reward_table, start, end)
    return float(total / n_eff)


def _load_reward_tables_and_runtime(
    events_path: Path,
    fallback_frames: int,
) -> Tuple[Dict[str, FrameRewardTable], int, Optional[float], Optional[float], str]:
    per_game_frame_indices: Dict[str, List[int]] = {}
    per_game_rewards: Dict[str, List[float]] = {}

    frames = 0
    first_time: Optional[float] = None
    last_time: Optional[float] = None
    prev_time: Optional[float] = None
    total_step_dt = 0.0
    step_dt_count = 0

    for row in _iter_jsonl(events_path):
        frames += 1
        if "wallclock_time" in row and row["wallclock_time"] is not None:
            t = float(row["wallclock_time"])
            if first_time is None:
                first_time = t
            last_time = t
            if prev_time is not None:
                dt = t - prev_time
                if dt >= 0.0:
                    total_step_dt += dt
                    step_dt_count += 1
            prev_time = t

        frame_idx = _pick_int(row, ["global_frame_idx"])
        if frame_idx is None:
            continue
        game_id = _pick_str(row, ["game_id"], default="unknown") or "unknown"
        reward = _pick_float(row, ["reward"], default=0.0)
        per_game_frame_indices.setdefault(game_id, []).append(int(frame_idx))
        per_game_rewards.setdefault(game_id, []).append(float(reward or 0.0))

    source = "events"
    if frames == 0:
        frames = int(fallback_frames)
        source = "fallback_config_total_scheduled_frames"

    fps: Optional[float] = None
    if first_time is not None and last_time is not None and frames > 1:
        denom = last_time - first_time
        if denom > 0.0:
            fps = float((frames - 1) / denom)

    mean_step_time: Optional[float] = None
    if step_dt_count > 0:
        mean_step_time = float(total_step_dt / step_dt_count)

    reward_tables = {
        game_id: _build_frame_reward_table(
            frame_indices=per_game_frame_indices[game_id],
            rewards=per_game_rewards[game_id],
        )
        for game_id in per_game_frame_indices
    }

    return reward_tables, frames, fps, mean_step_time, source


def score_run(
    run_dir: Path,
    window_frames: int = 5000,
    bottom_k_frac: float = 0.25,
    revisit_frames: int = 2000,
    final_score_weights: Tuple[float, float] = (0.5, 0.5),
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    if window_frames <= 0:
        raise ValueError("window_frames must be > 0")
    if revisit_frames <= 0:
        raise ValueError("revisit_frames must be > 0")
    if bottom_k_frac <= 0.0 or bottom_k_frac > 1.0:
        raise ValueError("bottom_k_frac must be in (0.0, 1.0]")

    config = _read_json(run_dir / "config.json")
    contract_version = str(config.get("benchmark_contract_version", BENCHMARK_CONTRACT_VERSION))
    contract_hash = config.get("benchmark_contract_hash")
    if not isinstance(contract_hash, str) or not contract_hash:
        contract_hash = compute_contract_hash(config)

    segments_rows = _read_jsonl(run_dir / "segments.jsonl")
    segments = _parse_segments(segments_rows)
    episodes_rows = _read_jsonl(run_dir / "episodes.jsonl")

    visits = _derive_visits_from_schedule(config)
    notes: Dict[str, Any] = {}
    if visits:
        notes["visit_source"] = "config_schedule"
    else:
        visits = _derive_visits_from_segments(segments)
        notes["visit_source"] = "segments"

    parsed_episodes = _parse_episodes(episodes_rows)
    episodes = [ep for ep in parsed_episodes if ep.ended_by == "terminated"]
    notes["episode_source"] = "episodes_jsonl_terminated_only"
    notes["episodes_jsonl_truncated_entries_detected"] = any(ep.ended_by == "truncated" for ep in parsed_episodes)
    if not episodes_rows:
        notes["episode_source"] = "episodes_jsonl_missing_or_empty"

    games_config = config.get("games")
    if isinstance(games_config, list) and games_config:
        games = [str(game) for game in games_config]
    else:
        inferred_games = {visit.game_id for visit in visits}
        if not inferred_games:
            inferred_games = {ep.game_id for ep in episodes}
        games = sorted(inferred_games)
    notes["games"] = games

    episodes_by_visit, unassigned_episodes = _assign_episodes_to_visits(episodes, visits)
    notes["unassigned_episode_count"] = int(len(unassigned_episodes))
    if unassigned_episodes:
        notes["unassigned_episode_examples"] = [
            {
                "game_id": ep.game_id,
                "start_frame": int(ep.start_frame),
                "end_frame": int(ep.end_frame),
                "return": float(ep.return_value),
            }
            for ep in unassigned_episodes[:5]
        ]

    visits_by_game: Dict[str, List[VisitRecord]] = {game: [] for game in games}
    for visit in visits:
        visits_by_game.setdefault(visit.game_id, []).append(visit)
    for game_visits in visits_by_game.values():
        game_visits.sort(key=lambda v: (v.start_frame, v.end_frame, v.visit_idx))

    fallback_frames = int(config.get("total_scheduled_frames", 0) or 0)
    reward_tables, frames, fps, mean_step_time, fps_source = _load_reward_tables_and_runtime(
        run_dir / "events.jsonl",
        fallback_frames=fallback_frames,
    )
    notes["mean_step_time"] = mean_step_time
    notes["fps_source"] = fps_source

    cycle_values = [visit.cycle_idx for visit in visits if visit.cycle_idx is not None]
    if cycle_values:
        last_cycle = max(cycle_values)
        last_cycle_visits = [visit for visit in visits if visit.cycle_idx == last_cycle]
        notes["cycle_selection"] = f"cycle_idx=={last_cycle}"
    else:
        last_cycle_visits = []
        for game in games:
            game_visits = visits_by_game.get(game, [])
            if game_visits:
                last_cycle_visits.append(game_visits[-1])
        notes["cycle_selection"] = "approx_last_visit_per_game"

    last_cycle_visits_by_game: Dict[str, List[VisitRecord]] = {game: [] for game in games}
    for visit in last_cycle_visits:
        last_cycle_visits_by_game.setdefault(visit.game_id, []).append(visit)
    for game_visits in last_cycle_visits_by_game.values():
        game_visits.sort(key=lambda v: (v.start_frame, v.end_frame, v.visit_idx))

    selected_last_visit_by_game: Dict[str, VisitRecord] = {}
    for game in games:
        game_visits = last_cycle_visits_by_game.get(game, [])
        if game_visits:
            selected_last_visit_by_game[game] = game_visits[-1]

    visit_order = sorted(visits, key=lambda v: (v.start_frame, v.end_frame, v.visit_idx))
    visit_pos = {visit.visit_idx: pos for pos, visit in enumerate(visit_order)}

    per_game_scores: Dict[str, float] = {}
    per_game_episode_counts: Dict[str, int] = {}
    per_game_visit_frames: Dict[str, int] = {}

    for game in games:
        last_visit = selected_last_visit_by_game.get(game)
        if last_visit is None:
            per_game_episode_counts[game] = 0
            per_game_visit_frames[game] = 0
            continue

        per_game_scores[game] = _tail_rate(last_visit, window_frames, reward_tables)
        per_game_visit_frames[game] = _visit_frame_count(last_visit)
        per_game_episode_counts[game] = int(
            sum(1 for ep in episodes_by_visit.get(last_visit.visit_idx, []) if ep.game_id == game)
        )
    notes["games_with_zero_last_cycle_episodes"] = [game for game in games if per_game_episode_counts.get(game, 0) == 0]

    scored_values = [per_game_scores[game] for game in games if game in per_game_scores]
    mean_score = _mean_or_none(scored_values)

    bottom_k_score: Optional[float] = None
    if scored_values:
        k = max(1, int(math.ceil(bottom_k_frac * len(scored_values))))
        bottom_k_score = float(np.mean(sorted(scored_values)[:k]))

    final_score: Optional[float] = None
    mean_w, bottom_w = final_score_weights
    if mean_score is not None and bottom_k_score is not None:
        final_score = float(mean_w * mean_score + bottom_w * bottom_k_score)
    notes["final_score_weights"] = {"mean": mean_w, "bottom_k": bottom_w}

    per_game_forgetting: Dict[str, float] = {}
    forgetting_values: List[float] = []
    for game in games:
        game_visits = visits_by_game.get(game, [])
        drops: List[float] = []
        for idx in range(len(game_visits) - 1):
            prev_visit = game_visits[idx]
            next_visit = game_visits[idx + 1]
            prev_pos = visit_pos.get(prev_visit.visit_idx)
            next_pos = visit_pos.get(next_visit.visit_idx)
            if prev_pos is not None and next_pos is not None and (next_pos - prev_pos) <= 1:
                continue

            pre = _tail_rate(prev_visit, revisit_frames, reward_tables)
            post = _head_rate(next_visit, revisit_frames, reward_tables)
            drops.append(pre - post)

        if drops:
            value = float(np.mean(drops))
            per_game_forgetting[game] = value
            forgetting_values.append(value)

    forgetting_index_mean = _mean_or_none(forgetting_values)
    forgetting_index_median = float(np.median(forgetting_values)) if forgetting_values else None

    per_game_plasticity: Dict[str, float] = {}
    plasticity_values: List[float] = []

    if cycle_values:
        first_cycle = min(cycle_values)
        first_cycle_visits = [visit for visit in visits if visit.cycle_idx == first_cycle]
        notes["plasticity_cycle_selection"] = f"cycle_idx=={first_cycle}"
    else:
        first_count = max(1, len(games))
        first_cycle_visits = list(visits[:first_count])
        notes["plasticity_cycle_selection"] = f"approx_first_{first_count}_visits"

    first_cycle_visits_by_game: Dict[str, List[VisitRecord]] = {game: [] for game in games}
    for visit in first_cycle_visits:
        first_cycle_visits_by_game.setdefault(visit.game_id, []).append(visit)
    for game_visits in first_cycle_visits_by_game.values():
        game_visits.sort(key=lambda v: (v.start_frame, v.end_frame, v.visit_idx))

    for game in games:
        game_visits = first_cycle_visits_by_game.get(game, [])
        if not game_visits:
            continue

        first_visit = game_visits[0]
        early = _head_rate(first_visit, revisit_frames, reward_tables)
        late = _tail_rate(first_visit, revisit_frames, reward_tables)
        value = float(late - early)
        per_game_plasticity[game] = value
        plasticity_values.append(value)

    plasticity_mean = _mean_or_none(plasticity_values)
    plasticity_median = float(np.median(plasticity_values)) if plasticity_values else None

    result: Dict[str, Any] = {
        "final_score": final_score,
        "mean_score": mean_score,
        "bottom_k_score": bottom_k_score,
        "per_game_scores": per_game_scores,
        "per_game_episode_counts": per_game_episode_counts,
        "per_game_visit_frames": per_game_visit_frames,
        "forgetting_index_mean": forgetting_index_mean,
        "forgetting_index_median": forgetting_index_median,
        "per_game_forgetting": per_game_forgetting,
        "plasticity_mean": plasticity_mean,
        "plasticity_median": plasticity_median,
        "per_game_plasticity": per_game_plasticity,
        "fps": fps,
        "frames": int(frames),
        "benchmark_contract_version": contract_version,
        "benchmark_contract_hash": str(contract_hash),
        "notes": notes,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a continual benchmark run directory.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory containing config/events/episodes/segments logs.")
    parser.add_argument("--window-frames", type=int, default=5000, help="Trailing frame window for online per-game scoring.")
    parser.add_argument("--bottom-k-frac", type=float, default=0.25, help="Bottom-k fraction used for robustness aggregation.")
    parser.add_argument("--revisit-frames", type=int, default=2000, help="Frame window for forgetting/plasticity local averages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = score_run(
        run_dir=run_dir,
        window_frames=args.window_frames,
        bottom_k_frac=args.bottom_k_frac,
        revisit_frames=args.revisit_frames,
    )

    score_path = run_dir / "score.json"
    with score_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
