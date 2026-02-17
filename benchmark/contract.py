"""Benchmark contract versioning and canonical hash utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence

BENCHMARK_CONTRACT_VERSION = "v1"

DEFAULT_WINDOW_EPISODES = 20
DEFAULT_BOTTOM_K_FRAC = 0.25
DEFAULT_REVISIT_EPISODES = 5
DEFAULT_FINAL_SCORE_WEIGHTS = (0.5, 0.5)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def resolve_scoring_defaults(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config.get("scoring_defaults")
    if not isinstance(raw, Mapping):
        raw = {}

    weights_raw = raw.get("final_score_weights", DEFAULT_FINAL_SCORE_WEIGHTS)
    if isinstance(weights_raw, Sequence) and not isinstance(weights_raw, (str, bytes)) and len(weights_raw) == 2:
        weights = (float(weights_raw[0]), float(weights_raw[1]))
    else:
        weights = DEFAULT_FINAL_SCORE_WEIGHTS

    return {
        "window_episodes": _as_int(raw.get("window_episodes"), DEFAULT_WINDOW_EPISODES),
        "bottom_k_frac": _as_float(raw.get("bottom_k_frac"), DEFAULT_BOTTOM_K_FRAC),
        "revisit_episodes": _as_int(raw.get("revisit_episodes"), DEFAULT_REVISIT_EPISODES),
        "final_score_weights": [float(weights[0]), float(weights[1])],
    }


def _extract_games(config: Mapping[str, Any]) -> Sequence[str]:
    games = config.get("games")
    if isinstance(games, list):
        return [str(game) for game in games]
    return []


def _extract_schedule(config: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
    schedule = config.get("schedule")
    if not isinstance(schedule, list):
        return []

    records = []
    for row in schedule:
        if not isinstance(row, Mapping):
            continue
        visit_idx = row.get("visit_idx")
        cycle_idx = row.get("cycle_idx")
        game_id = row.get("game_id")
        visit_frames = row.get("visit_frames")
        if visit_idx is None or cycle_idx is None or game_id is None or visit_frames is None:
            continue
        records.append(
            {
                "visit_idx": int(visit_idx),
                "cycle_idx": int(cycle_idx),
                "game_id": str(game_id),
                "visit_frames": int(visit_frames),
            }
        )
    return records


def _extract_delay_frames(config: Mapping[str, Any]) -> int:
    if config.get("delay_frames") is not None:
        return int(config["delay_frames"])
    if config.get("delay") is not None:
        return int(config["delay"])
    runner_cfg = config.get("runner_config")
    if isinstance(runner_cfg, Mapping) and runner_cfg.get("delay_frames") is not None:
        return int(runner_cfg["delay_frames"])
    return 0


def _extract_decision_interval(config: Mapping[str, Any]) -> int:
    if config.get("decision_interval") is not None:
        return int(config["decision_interval"])
    runner_cfg = config.get("runner_config")
    if isinstance(runner_cfg, Mapping) and runner_cfg.get("decision_interval") is not None:
        return int(runner_cfg["decision_interval"])
    return 0


def _extract_default_action_idx(config: Mapping[str, Any]) -> int:
    if config.get("default_action_idx") is not None:
        return int(config["default_action_idx"])
    runner_cfg = config.get("runner_config")
    if isinstance(runner_cfg, Mapping) and runner_cfg.get("default_action_idx") is not None:
        return int(runner_cfg["default_action_idx"])
    return 0


def _extract_global_action_set(config: Mapping[str, Any]) -> Sequence[int]:
    policy = config.get("action_mapping_policy")
    if isinstance(policy, Mapping):
        raw = policy.get("global_action_set")
        if isinstance(raw, list):
            return [int(action) for action in raw]

    raw_direct = config.get("global_action_set")
    if isinstance(raw_direct, list):
        return [int(action) for action in raw_direct]
    return []


def canonical_contract_input(
    config: Mapping[str, Any],
    scoring_defaults: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_scoring_defaults = (
        resolve_scoring_defaults(config)
        if scoring_defaults is None
        else resolve_scoring_defaults({"scoring_defaults": dict(scoring_defaults)})
    )

    return {
        "games": list(_extract_games(config)),
        "schedule": list(_extract_schedule(config)),
        "decision_interval": _extract_decision_interval(config),
        "delay_frames": _extract_delay_frames(config),
        "sticky": float(config.get("sticky", 0.0)),
        "life_loss_termination": bool(config.get("life_loss_termination", False)),
        "full_action_space": bool(config.get("full_action_space", True)),
        "global_action_set": list(_extract_global_action_set(config)),
        "default_action_idx": _extract_default_action_idx(config),
        "scoring_defaults": {
            "window_episodes": int(resolved_scoring_defaults["window_episodes"]),
            "bottom_k_frac": float(resolved_scoring_defaults["bottom_k_frac"]),
            "revisit_episodes": int(resolved_scoring_defaults["revisit_episodes"]),
            "final_score_weights": [
                float(resolved_scoring_defaults["final_score_weights"][0]),
                float(resolved_scoring_defaults["final_score_weights"][1]),
            ],
        },
    }


def compute_contract_hash(
    config: Mapping[str, Any],
    scoring_defaults: Optional[Mapping[str, Any]] = None,
) -> str:
    payload = canonical_contract_input(config=config, scoring_defaults=scoring_defaults)
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

