"""CLI entrypoint for the multi-game continual streaming Atari benchmark runner."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from benchmark.agents import RandomAgent, RepeatActionAgent
from benchmark.ale_env import ALEAtariEnv, ALEEnvConfig
from benchmark.contract import BENCHMARK_CONTRACT_VERSION, compute_contract_hash, resolve_scoring_defaults
from benchmark.logging_utils import JsonlWriter, dump_json, make_run_dir
from benchmark.multigame_runner import MultiGameRunner, MultiGameRunnerConfig
from benchmark.schedule import Schedule, ScheduleConfig


def parse_games_csv(value: str) -> List[str]:
    games = [part.strip() for part in str(value).split(",") if part.strip()]
    if not games:
        raise ValueError("--games must provide at least one game id")
    return games


def _load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("--config must contain a JSON object")
    return payload


def _coerce_config_defaults(config_data: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}

    def set_if_present(dst: str, src_keys: Sequence[str], cast=None) -> None:
        for key in src_keys:
            if key in config_data and config_data[key] is not None:
                value = config_data[key]
                defaults[dst] = cast(value) if cast is not None else value
                return

    games = config_data.get("games")
    if isinstance(games, list):
        defaults["games"] = ",".join(str(game) for game in games)
    elif isinstance(games, str):
        defaults["games"] = games

    set_if_present("num_cycles", ["num_cycles"], int)
    set_if_present("base_visit_frames", ["base_visit_frames"], int)
    set_if_present("jitter_pct", ["jitter_pct"], float)
    set_if_present("min_visit_frames", ["min_visit_frames"], int)
    set_if_present("seed", ["seed"], int)
    set_if_present("decision_interval", ["decision_interval", "frame_skip"], int)
    set_if_present("delay", ["delay", "delay_frames"], int)
    set_if_present("sticky", ["sticky", "sticky_prob"], float)
    set_if_present("full_action_space", ["full_action_space"], int)
    set_if_present("life_loss_termination", ["life_loss_termination"], int)
    set_if_present("agent", ["agent"], str)
    set_if_present("repeat_action_idx", ["repeat_action_idx"], int)
    set_if_present("default_action_idx", ["default_action_idx"], int)
    set_if_present("timestamps", ["timestamps", "include_timestamps"], int)
    set_if_present("logdir", ["logdir"], str)

    # TinyDQN keys can be top-level or nested under agent_config.
    agent_cfg = config_data.get("agent_config")
    if isinstance(agent_cfg, dict):
        merged_cfg = dict(agent_cfg)
    else:
        merged_cfg = {}
    for key in [
        "dqn_gamma",
        "dqn_lr",
        "dqn_buffer_size",
        "dqn_batch_size",
        "dqn_train_every",
        "dqn_target_update",
        "dqn_eps_start",
        "dqn_eps_end",
        "dqn_eps_decay_frames",
        "dqn_replay_min",
        "dqn_use_replay",
        "dqn_device",
    ]:
        if key in config_data and config_data[key] is not None:
            merged_cfg[key] = config_data[key]
    # Direct argparse-style TinyDQN keys.
    arg_cast = {
        "dqn_gamma": float,
        "dqn_lr": float,
        "dqn_buffer_size": int,
        "dqn_batch_size": int,
        "dqn_train_every": int,
        "dqn_target_update": int,
        "dqn_eps_start": float,
        "dqn_eps_end": float,
        "dqn_eps_decay_frames": int,
        "dqn_replay_min": int,
        "dqn_use_replay": int,
        "dqn_device": str,
    }
    for arg_name in [
        "dqn_gamma",
        "dqn_lr",
        "dqn_buffer_size",
        "dqn_batch_size",
        "dqn_train_every",
        "dqn_target_update",
        "dqn_eps_start",
        "dqn_eps_end",
        "dqn_eps_decay_frames",
        "dqn_replay_min",
        "dqn_use_replay",
        "dqn_device",
    ]:
        if arg_name in merged_cfg and merged_cfg[arg_name] is not None:
            defaults[arg_name] = arg_cast[arg_name](merged_cfg[arg_name])

    # Backward-compatible mapping from TinyDQNConfig field names.
    field_to_arg = {
        "gamma": "dqn_gamma",
        "lr": "dqn_lr",
        "buffer_size": "dqn_buffer_size",
        "batch_size": "dqn_batch_size",
        "train_every_decisions": "dqn_train_every",
        "target_update_decisions": "dqn_target_update",
        "eps_start": "dqn_eps_start",
        "eps_end": "dqn_eps_end",
        "eps_decay_frames": "dqn_eps_decay_frames",
        "replay_min_size": "dqn_replay_min",
        "use_replay": "dqn_use_replay",
        "device": "dqn_device",
    }
    for field_name, arg_name in field_to_arg.items():
        if field_name in merged_cfg and merged_cfg[field_name] is not None:
            value = merged_cfg[field_name]
            if arg_name == "dqn_use_replay":
                defaults[arg_name] = int(value)
            else:
                defaults[arg_name] = value

    return defaults


def _build_parser(defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    has_games_default = "games" in defaults and str(defaults["games"]).strip() != ""

    parser = argparse.ArgumentParser(description="Multi-game continual Atari benchmark runner (ALE direct).")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    parser.add_argument(
        "--games",
        type=str,
        required=not has_games_default,
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
        choices=["random", "repeat", "tinydqn"],
        default="random",
        help="Agent type.",
    )
    parser.add_argument("--repeat-action-idx", type=int, default=0, help="Action index for RepeatActionAgent.")
    parser.add_argument("--dqn-gamma", type=float, default=0.99, help="TinyDQN discount factor.")
    parser.add_argument("--dqn-lr", type=float, default=1e-4, help="TinyDQN Adam learning rate.")
    parser.add_argument("--dqn-buffer-size", type=int, default=10000, help="TinyDQN replay buffer size.")
    parser.add_argument("--dqn-batch-size", type=int, default=32, help="TinyDQN batch size.")
    parser.add_argument(
        "--dqn-train-every",
        type=int,
        default=4,
        help="Train TinyDQN every N decision frames.",
    )
    parser.add_argument(
        "--dqn-target-update",
        type=int,
        default=250,
        help="Hard target-network sync interval in decision frames.",
    )
    parser.add_argument("--dqn-eps-start", type=float, default=1.0, help="TinyDQN epsilon at frame 0.")
    parser.add_argument("--dqn-eps-end", type=float, default=0.05, help="TinyDQN final epsilon.")
    parser.add_argument(
        "--dqn-eps-decay-frames",
        type=int,
        default=200000,
        help="Frames for linear epsilon decay.",
    )
    parser.add_argument(
        "--dqn-replay-min",
        type=int,
        default=1000,
        help="Minimum replay size before training starts.",
    )
    parser.add_argument(
        "--dqn-use-replay",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable replay buffer for TinyDQN (1=yes, 0=no).",
    )
    parser.add_argument(
        "--dqn-device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="TinyDQN device.",
    )
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
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(args=argv)
    config_data = _load_config_file(pre_args.config)
    defaults = _coerce_config_defaults(config_data)
    parser = _build_parser(defaults=defaults)
    args = parser.parse_args(args=argv)
    setattr(args, "_config_data", config_data)
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_agent(args: argparse.Namespace, num_actions: int):
    if args.agent == "random":
        return RandomAgent(num_actions=num_actions, seed=args.seed), {}
    if args.agent == "repeat":
        if args.repeat_action_idx < 0 or args.repeat_action_idx >= num_actions:
            raise ValueError(f"--repeat-action-idx must be in [0, {num_actions - 1}]")
        return RepeatActionAgent(action_idx=args.repeat_action_idx), {"action_idx": int(args.repeat_action_idx)}
    try:
        from benchmark.agents_tinydqn import TinyDQNAgent, TinyDQNConfig  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on optional torch dependency
        raise ImportError(
            "agent=tinydqn requires torch. Install torch or use --agent random/--agent repeat."
        ) from exc

    dqn_config = TinyDQNConfig(
        gamma=float(args.dqn_gamma),
        lr=float(args.dqn_lr),
        buffer_size=int(args.dqn_buffer_size),
        batch_size=int(args.dqn_batch_size),
        train_every_decisions=int(args.dqn_train_every),
        target_update_decisions=int(args.dqn_target_update),
        replay_min_size=int(args.dqn_replay_min),
        eps_start=float(args.dqn_eps_start),
        eps_end=float(args.dqn_eps_end),
        eps_decay_frames=int(args.dqn_eps_decay_frames),
        use_replay=bool(int(args.dqn_use_replay)),
        device=str(args.dqn_device),
    )
    agent = TinyDQNAgent(action_space_n=num_actions, seed=int(args.seed), config=dqn_config)
    return agent, dqn_config.as_dict()


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
    agent_config: Dict[str, object],
    config_file_data: Dict[str, Any],
) -> Dict:
    scoring_defaults = resolve_scoring_defaults(config_file_data)
    payload = {
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
        "agent_config": dict(agent_config),
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
        "scoring_defaults": scoring_defaults,
    }
    if args.config is not None:
        payload["config_path"] = str(args.config)
    if config_file_data:
        payload["config_file"] = dict(config_file_data)
    payload["benchmark_contract_version"] = BENCHMARK_CONTRACT_VERSION
    payload["benchmark_contract_hash"] = compute_contract_hash(payload, scoring_defaults=scoring_defaults)
    return payload


def collect_agent_stats(agent: object) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for key in ("replay_size", "finalized_transition_counter", "train_steps", "replay_min_size", "decision_steps"):
        if hasattr(agent, key):
            value = getattr(agent, key)
            if callable(value):
                continue
            if isinstance(value, (int, float, bool, str)) or value is None:
                stats[key] = value
            else:
                stats[key] = str(value)
    return stats


def main() -> None:
    args = parse_args()
    config_file_data = dict(getattr(args, "_config_data", {}))
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
    agent, agent_config = build_agent(args, num_actions=len(global_action_set))

    resolved_action_sets = resolve_action_sets(env, games)
    action_mappings = build_action_mappings(resolved_action_sets, global_action_set, args.default_action_idx)

    run_dir = make_run_dir(Path(args.logdir), "multigame", args.seed)
    event_writer = JsonlWriter(run_dir / "events.jsonl")
    episode_writer = JsonlWriter(run_dir / "episodes.jsonl")
    segment_writer = JsonlWriter(run_dir / "segments.jsonl")

    config_payload = build_config_payload(
        args,
        games,
        run_dir,
        schedule,
        runner_config,
        resolved_action_sets,
        agent_config,
        config_file_data,
    )
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

    dump_json(run_dir / "agent_stats.json", collect_agent_stats(agent))
    print(f"Run complete: {run_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
