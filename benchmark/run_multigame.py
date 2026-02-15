"""CLI entrypoint for the multi-game continual streaming Atari benchmark runner."""

from __future__ import annotations

import argparse
import platform
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from benchmark.agents import RandomAgent, RepeatActionAgent
from benchmark.ale_env import ALEAtariEnv, ALEEnvConfig
from benchmark.logging_utils import JsonlWriter, dump_json, make_run_dir
from benchmark.multigame_runner import MultiGameRunner, MultiGameRunnerConfig
from benchmark.schedule import Schedule, ScheduleConfig


def parse_games_csv(value: str) -> List[str]:
    games = [part.strip() for part in str(value).split(",") if part.strip()]
    if not games:
        raise ValueError("--games must provide at least one game id")
    return games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-game continual Atari benchmark runner (ALE direct).")
    parser.add_argument(
        "--games",
        type=str,
        required=True,
        help="Comma-separated ALE ROM keys, e.g. ms_pacman,centipede,qbert.",
    )
    parser.add_argument("--num-cycles", type=int, default=3, help="Number of schedule cycles.")
    parser.add_argument("--base-visit-frames", type=int, default=200_000, help="Base frames per visit before jitter.")
    parser.add_argument("--jitter-pct", type=float, default=0.07, help="Visit frame jitter in [0.0, 1.0).")
    parser.add_argument("--min-visit-frames", type=int, default=1, help="Lower bound after visit jitter.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument(
        "--decision-interval",
        "--frame-skip",
        dest="decision_interval",
        type=int,
        default=4,
        help="Decision update interval in frames.",
    )
    parser.add_argument("--delay", type=int, default=6, help="Action latency queue length in frames.")
    parser.add_argument("--sticky", type=float, default=0.25, help="ALE repeat_action_probability.")
    parser.add_argument(
        "--full-action-space",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=full legal action set (benchmark default), 0=minimal action set.",
    )
    parser.add_argument(
        "--life-loss-termination",
        type=int,
        choices=[0, 1],
        default=0,
        help="Treat life loss as terminated episode when set to 1.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["random", "repeat"],
        default="random",
        help="Agent type.",
    )
    parser.add_argument("--repeat-action-idx", type=int, default=0, help="Action index for RepeatActionAgent.")
    parser.add_argument(
        "--default-action-idx",
        type=int,
        default=0,
        help="Global action index used for delay-queue initialization and illegal-action fallback.",
    )
    parser.add_argument(
        "--timestamps",
        type=int,
        choices=[0, 1],
        default=1,
        help="Include wallclock timestamps in per-frame events.",
    )
    parser.add_argument("--logdir", type=str, default="./runs/v1", help="Base output directory.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_agent(args: argparse.Namespace, num_actions: int):
    if args.agent == "random":
        return RandomAgent(num_actions=num_actions, seed=args.seed)
    if args.repeat_action_idx < 0 or args.repeat_action_idx >= num_actions:
        raise ValueError(f"--repeat-action-idx must be in [0, {num_actions - 1}]")
    return RepeatActionAgent(action_idx=args.repeat_action_idx)


def resolve_action_sets(env: ALEAtariEnv, games: Sequence[str]) -> Dict[str, List[int]]:
    resolved: Dict[str, List[int]] = {}
    for game_id in games:
        resolved[game_id] = [int(a) for a in env.load_game(game_id)]
    return resolved


def build_action_mappings(
    resolved_action_sets: Dict[str, List[int]],
    global_action_set: Sequence[int],
    default_action_idx: int,
) -> Dict[str, Dict[str, int]]:
    mappings: Dict[str, Dict[str, int]] = {}
    default_ale_action = int(global_action_set[default_action_idx])
    for game_id, local_actions in resolved_action_sets.items():
        local_lookup = {int(action): idx for idx, action in enumerate(local_actions)}
        fallback_idx = local_lookup.get(default_ale_action, 0)
        per_game: Dict[str, int] = {}
        for global_idx, ale_action in enumerate(global_action_set):
            local_idx = local_lookup.get(int(ale_action), int(fallback_idx))
            per_game[str(global_idx)] = int(local_idx)
        mappings[str(game_id)] = per_game
    return mappings


def get_version_metadata() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    try:
        import ale_py  # pylint: disable=import-outside-toplevel

        versions["ale_py"] = getattr(ale_py, "__version__", "unknown")
    except Exception:  # pragma: no cover - only relevant when ale_py missing
        versions["ale_py"] = "unavailable"
    return versions


def resolve_global_action_set() -> Sequence[int]:
    """
    Resolve canonical global Atari action vocabulary.

    Prefer ale_py.Action enum values when available; fallback to standard 18-action ids.
    """
    try:
        from ale_py import Action  # pylint: disable=import-outside-toplevel

        values = [int(action) for action in Action]
        if values:
            return tuple(values)
    except Exception:  # pragma: no cover - optional runtime path
        pass
    return tuple(range(18))


def build_config_payload(
    args: argparse.Namespace,
    games: Sequence[str],
    run_dir: Path,
    schedule: Schedule,
    runner_config: MultiGameRunnerConfig,
    resolved_action_sets: Dict[str, List[int]],
) -> Dict:
    return {
        "games": [str(game) for game in games],
        "num_cycles": int(args.num_cycles),
        "base_visit_frames": int(args.base_visit_frames),
        "jitter_pct": float(args.jitter_pct),
        "min_visit_frames": int(args.min_visit_frames),
        "seed": int(args.seed),
        "decision_interval": int(args.decision_interval),
        "delay": int(args.delay),
        "sticky": float(args.sticky),
        "full_action_space": bool(args.full_action_space),
        "life_loss_termination": bool(args.life_loss_termination),
        "agent": str(args.agent),
        "repeat_action_idx": int(args.repeat_action_idx),
        "default_action_idx": int(args.default_action_idx),
        "timestamps": bool(args.timestamps),
        "logdir": str(Path(args.logdir)),
        "run_dir": str(run_dir),
        "total_scheduled_frames": int(schedule.total_frames),
        "schedule": schedule.as_records(),
        "resolved_action_sets": {game: list(actions) for game, actions in resolved_action_sets.items()},
        "action_mapping_policy": {
            "global_action_set": [int(a) for a in runner_config.global_action_set],
            "agent_action_idx_space": "global_action_set index",
            "runner_resolution": "global ALE action -> current game local action index",
            "illegal_action_fallback": "default_action_idx ALE action if legal else local index 0",
        },
        "runner_config": {
            "decision_interval": int(runner_config.decision_interval),
            "delay_frames": int(runner_config.delay_frames),
            "default_action_idx": int(runner_config.default_action_idx),
            "include_timestamps": bool(runner_config.include_timestamps),
        },
        "versions": get_version_metadata(),
    }


def main() -> None:
    args = parse_args()
    games = parse_games_csv(args.games)
    seed_everything(args.seed)

    schedule = Schedule(
        ScheduleConfig(
            games=games,
            base_visit_frames=args.base_visit_frames,
            num_cycles=args.num_cycles,
            seed=args.seed,
            jitter_pct=args.jitter_pct,
            min_visit_frames=args.min_visit_frames,
        )
    )

    env_config = ALEEnvConfig(
        game=games[0],
        seed=args.seed,
        sticky_action_prob=args.sticky,
        full_action_space=bool(args.full_action_space),
        life_loss_termination=bool(args.life_loss_termination),
    )
    env = ALEAtariEnv(env_config)

    global_action_set = tuple(resolve_global_action_set())
    if args.default_action_idx < 0 or args.default_action_idx >= len(global_action_set):
        raise ValueError(f"--default-action-idx must be in [0, {len(global_action_set) - 1}]")

    runner_config = MultiGameRunnerConfig(
        decision_interval=args.decision_interval,
        delay_frames=args.delay,
        default_action_idx=args.default_action_idx,
        include_timestamps=bool(args.timestamps),
        global_action_set=global_action_set,
    )
    agent = build_agent(args, num_actions=len(global_action_set))

    resolved_action_sets = resolve_action_sets(env, games)
    action_mappings = build_action_mappings(resolved_action_sets, global_action_set, args.default_action_idx)

    run_dir = make_run_dir(Path(args.logdir), "multigame", args.seed)
    event_writer = JsonlWriter(run_dir / "events.jsonl")
    episode_writer = JsonlWriter(run_dir / "episodes.jsonl")
    segment_writer = JsonlWriter(run_dir / "segments.jsonl")

    config_payload = build_config_payload(args, games, run_dir, schedule, runner_config, resolved_action_sets)
    config_payload["resolved_action_mappings"] = action_mappings
    dump_json(run_dir / "config.json", config_payload)

    try:
        summary = MultiGameRunner(
            env=env,
            agent=agent,
            schedule=schedule,
            config=runner_config,
            event_writer=event_writer,
            episode_writer=episode_writer,
            segment_writer=segment_writer,
        ).run()
    finally:
        event_writer.close()
        episode_writer.close()
        segment_writer.close()

    print(f"Run complete: {run_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
