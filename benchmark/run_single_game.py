"""CLI entrypoint for the single-game streaming Atari benchmark runner."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np

from benchmark.agents import RandomAgent, RepeatActionAgent
from benchmark.ale_env import ALEAtariEnv, ALEEnvConfig
from benchmark.logging_utils import JsonlWriter, dump_json, make_run_dir
from benchmark.runner import BenchmarkRunner, RunnerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-game streaming Atari benchmark runner (ALE direct).")
    parser.add_argument("--game", type=str, required=True, help="ALE ROM key, e.g. ms_pacman, pong, breakout.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument("--frames", type=int, default=200_000, help="Total environment frames to run.")
    parser.add_argument("--frame-skip", type=int, default=4, help="Decision update interval in frames.")
    parser.add_argument("--delay", type=int, default=0, help="Action latency queue length in frames.")
    parser.add_argument("--sticky", type=float, default=0.25, help="ALE repeat_action_probability.")
    parser.add_argument(
        "--full-action-space",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=full legal action set (default), 0=minimal action set.",
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
    parser.add_argument(
        "--repeat-action-idx",
        type=int,
        default=0,
        help="Action index for RepeatActionAgent.",
    )
    parser.add_argument(
        "--default-action-idx",
        type=int,
        default=0,
        help="Default action index used to initialize delay queue on each reset.",
    )
    parser.add_argument(
        "--timestamps",
        type=int,
        choices=[0, 1],
        default=1,
        help="Include wallclock timestamps in per-frame events.",
    )
    parser.add_argument("--logdir", type=str, default="./runs", help="Base output directory.")
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


def build_config_payload(
    args: argparse.Namespace,
    env: ALEAtariEnv,
    runner_config: RunnerConfig,
    run_dir: Path,
) -> Dict:
    return {
        "game": args.game,
        "seed": int(args.seed),
        "frames": int(args.frames),
        "frame_skip": int(args.frame_skip),
        "delay": int(args.delay),
        "sticky": float(args.sticky),
        "full_action_space": bool(args.full_action_space),
        "life_loss_termination": bool(args.life_loss_termination),
        "agent": args.agent,
        "repeat_action_idx": int(args.repeat_action_idx),
        "default_action_idx": int(args.default_action_idx),
        "timestamps": bool(args.timestamps),
        "logdir": str(Path(args.logdir)),
        "run_dir": str(run_dir),
        "resolved_action_set": list(env.action_set),
        "num_actions": len(env.action_set),
        "runner_config": {
            "total_frames": runner_config.total_frames,
            "frame_skip": runner_config.frame_skip,
            "delay_frames": runner_config.delay_frames,
            "default_action_idx": runner_config.default_action_idx,
            "include_timestamps": runner_config.include_timestamps,
        },
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    env_config = ALEEnvConfig(
        game=args.game,
        seed=args.seed,
        sticky_action_prob=args.sticky,
        full_action_space=bool(args.full_action_space),
        life_loss_termination=bool(args.life_loss_termination),
    )
    env = ALEAtariEnv(env_config)

    if args.default_action_idx < 0 or args.default_action_idx >= len(env.action_set):
        raise ValueError(f"--default-action-idx must be in [0, {len(env.action_set) - 1}]")

    agent = build_agent(args, num_actions=len(env.action_set))

    run_dir = make_run_dir(Path(args.logdir), args.game, args.seed)
    event_writer = JsonlWriter(run_dir / "events.jsonl")
    episode_writer = JsonlWriter(run_dir / "episodes.jsonl")

    runner_config = RunnerConfig(
        total_frames=args.frames,
        frame_skip=args.frame_skip,
        delay_frames=args.delay,
        default_action_idx=args.default_action_idx,
        include_timestamps=bool(args.timestamps),
    )
    config_payload = build_config_payload(args, env, runner_config, run_dir)
    dump_json(run_dir / "config.json", config_payload)

    try:
        summary = BenchmarkRunner(
            env=env,
            agent=agent,
            config=runner_config,
            event_writer=event_writer,
            episode_writer=episode_writer,
        ).run()
    finally:
        event_writer.close()
        episode_writer.close()

    print(f"Run complete: {run_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
