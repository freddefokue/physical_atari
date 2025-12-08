# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared metrics calculation and summary writing utilities.
Single source of truth for continual learning metrics.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch

from .constants import FRAME_SKIP, INVALID_BUFFER_VALUE, PROGRESS_POINTS

if TYPE_CHECKING:
    from benchmark_runner import GameResult


def moving_average(
    scores: Sequence[float],
    end_frames: Sequence[int],
    window: int,
    current_frame: int,
) -> float:
    """
    Calculate the moving average of episode scores within a frame window.
    
    Args:
        scores: List of episode scores
        end_frames: List of frame indices where each episode ended
        window: Window size in frames for the moving average
        current_frame: Current frame index
    
    Returns:
        Moving average score, or INVALID_BUFFER_VALUE if no episodes in window
    """
    if not scores:
        return INVALID_BUFFER_VALUE
    
    cutoff = current_frame - window
    total = 0.0
    count = 0
    
    for score, frame_idx in zip(reversed(scores), reversed(end_frames)):
        if frame_idx < cutoff:
            break
        total += score
        count += 1
    
    if count == 0:
        return INVALID_BUFFER_VALUE
    
    return total / count


def update_progress_graphs(
    episode_graph: torch.Tensor,
    parms_graph: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    frame_offset: int,
    frames_consumed: int,
    frame_budget: int,
    average_frames: int,
    episode_scores: Sequence[float],
    episode_end: Sequence[int],
    graph_total_frames: Optional[int],
) -> None:
    """
    Update progress graphs with current episode statistics.
    
    Args:
        episode_graph: Tensor to store episode score averages
        parms_graph: Tensor to store parameter norms
        params: Model parameters to track
        frame_offset: Starting frame offset for this game
        frames_consumed: Frames consumed so far in this game
        frame_budget: Total frame budget for this game
        average_frames: Window size for moving average
        episode_scores: List of episode scores
        episode_end: List of frame indices where episodes ended
        graph_total_frames: Total frames across all games (for x-axis scaling)
    """
    total_frames = graph_total_frames or (frame_offset + frame_budget)
    total_frames = max(total_frames, frame_offset + frame_budget)
    current_frame = frame_offset + frames_consumed
    previous_frame = max(frame_offset, current_frame - FRAME_SKIP)
    points = episode_graph.shape[0]
    
    prev_bucket = min((previous_frame * points) // total_frames, points - 1)
    curr_bucket = min((current_frame * points) // total_frames, points - 1)
    
    if curr_bucket == prev_bucket:
        return
    
    avg_score = moving_average(episode_scores, episode_end, average_frames, current_frame)
    
    for bucket in range(prev_bucket + 1, curr_bucket + 1):
        episode_graph[bucket] = avg_score
        with torch.no_grad():
            for idx, param in enumerate(params):
                parms_graph[bucket, idx] = torch.norm(param.detach()).item()


def write_continual_summary(results: Sequence["GameResult"], path: str) -> None:
    """
    Write a comprehensive summary of continual learning results.
    
    Includes metrics:
    - AUC (Area Under Curve): Total reward gathered
    - Retention (Backward Transfer): Performance at cycle N start vs cycle N-1 end
    - Savings (Forward Transfer): Normalized improvement over first attempt
    - Plasticity: Performance improvement within a cycle
    
    Args:
        results: Sequence of GameResult objects from BenchmarkRunner
        path: Output path for JSON summary (CSV will be written to path with .csv extension)
    """
    
    def get_stats(scores: Sequence[float]) -> tuple:
        """Calculate AUC, start performance, and end performance from scores."""
        if not scores:
            return 0.0, 0.0, 0.0
        auc = sum(scores)
        # Average of first 10 episodes (start performance)
        start_perf = float(np.mean(scores[:10])) if len(scores) >= 10 else float(np.mean(scores))
        # Average of last 10 episodes (end performance)
        end_perf = float(np.mean(scores[-10:])) if len(scores) >= 10 else float(np.mean(scores))
        return auc, start_perf, end_perf

    # Organize data by game and cycle
    game_map: Dict[str, Dict[int, Dict[str, float]]] = {}
    summary_records: List[Dict] = []

    for res in results:
        auc, start_perf, end_perf = get_stats(res.episode_scores)
        
        if res.spec.name not in game_map:
            game_map[res.spec.name] = {}
        game_map[res.spec.name][res.cycle_index] = {
            "auc": auc,
            "start": start_perf,
            "end": end_perf
        }

        summary_records.append({
            "cycle_index": res.cycle_index,
            "game_index": res.game_index,
            "game": res.spec.name,
            "frame_offset": res.frame_offset,
            "frame_budget": res.frame_budget,
            "episodes": len(res.episode_scores),
            "total_episode_score": auc,
            "mean_episode_score": float(np.mean(res.episode_scores)) if res.episode_scores else 0.0
        })

    # Calculate CL Metrics and write CSV
    metrics_csv_rows = ["game,cycle,auc,start_perf,end_perf,retention,savings,plasticity"]

    for game, cycles in game_map.items():
        max_cycle = max(cycles.keys())
        for i in range(max_cycle + 1):
            if i not in cycles:
                continue
            
            curr = cycles[i]
            retention = ""
            savings = ""
            plasticity = ""

            # Retention: How well did we hold on since the last time we played?
            if i > 0 and (i - 1) in cycles:
                prev = cycles[i - 1]
                retention = f"{curr['start'] - prev['end']:.2f}"

            # Savings: Normalized improvement over the first attempt
            if i > 0 and 0 in cycles:
                c0 = cycles[0]
                if c0["auc"] != 0:
                    savings_val = (curr["auc"] - c0["auc"]) / abs(c0["auc"])
                    savings = f"{savings_val:.4f}"
                else:
                    savings = "0.0000"
            
            # Plasticity: Performance improvement within the cycle
            plasticity = f"{curr['end'] - curr['start']:.2f}"

            metrics_csv_rows.append(
                f"{game},{i},{curr['auc']:.2f},{curr['start']:.2f},{curr['end']:.2f},{retention},{savings},{plasticity}"
            )

    # Write CSV detailed report
    csv_path = path.replace(".json", "_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metrics_csv_rows))
    print(f"Wrote detailed CL metrics to {csv_path}")

    # Write JSON summary
    final_cycle_index = results[-1].cycle_index if results else -1
    final_cycle_total = sum(
        r["total_episode_score"] for r in summary_records if r["cycle_index"] == final_cycle_index
    )

    payload = {
        "final_cycle_index": final_cycle_index,
        "final_cycle_total_episode_score": final_cycle_total,
        "records": summary_records,
    }
    with open(path, "w", encoding="utf-8") as summary_file:
        json.dump(payload, summary_file, indent=2)
    print(f"Wrote continual summary to {path}")
