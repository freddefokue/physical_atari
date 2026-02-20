"""CLI entrypoint for the single-game streaming Atari benchmark runner."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from benchmark.agents import RandomAgent, RepeatActionAgent
from benchmark.ale_env import ALEAtariEnv, ALEEnvConfig
from benchmark.carmack_runner import CarmackCompatRunner, CarmackRunnerConfig
from benchmark.logging_utils import JsonlWriter, dump_json, make_run_dir
from benchmark.runner import BenchmarkRunner, RunnerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-game streaming Atari benchmark runner (ALE direct).")
    parser.add_argument("--game", type=str, required=True, help="ALE ROM key, e.g. ms_pacman, pong, breakout.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument("--frames", type=int, default=200_000, help="Total environment frames to run.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Decision update interval in frames (carmack_compat requires 1).",
    )
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
        choices=["random", "repeat", "delay_target"],
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
        help="Default action index used to seed delayed-action queue state.",
    )
    parser.add_argument(
        "--runner-mode",
        type=str,
        choices=["standard", "carmack_compat"],
        default="standard",
        help="Runner loop semantics: benchmark standard or Carmack-compatible single-game loop.",
    )
    parser.add_argument(
        "--lives-as-episodes",
        type=int,
        choices=[0, 1],
        default=1,
        help="Carmack compat mode: emit end_of_episode pulses on life loss.",
    )
    parser.add_argument(
        "--max-frames-without-reward",
        type=int,
        default=18000,
        help="Carmack compat mode: force reset after this many consecutive zero-reward frames.",
    )
    parser.add_argument(
        "--reset-on-life-loss",
        type=int,
        choices=[0, 1],
        default=0,
        help="Carmack compat mode: reset environment on life-loss pulse.",
    )
    parser.add_argument(
        "--compat-reset-delay-queue-on-reset",
        type=int,
        choices=[0, 1],
        default=0,
        help="Carmack compat mode: 1 resets delay queue to default on each episode reset, 0 keeps queue persistent.",
    )
    parser.add_argument(
        "--compat-log-every-frames",
        type=int,
        default=0,
        help="Carmack compat mode: print benchmark [train] progress stats every N frames (0 disables).",
    )
    parser.add_argument(
        "--compat-log-pulses-every",
        type=int,
        default=0,
        help="Carmack compat mode: print benchmark [pulse] logs every N pulses (0 disables).",
    )
    parser.add_argument(
        "--compat-log-resets-every",
        type=int,
        default=1,
        help="Carmack compat mode: print episode reset-return logs every N resets (0 disables).",
    )
    parser.add_argument("--delay-target-gpu", type=int, default=0, help="GPU index for agent_delay_target.")
    parser.add_argument(
        "--delay-target-use-cuda-graphs",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable CUDA graphs for agent_delay_target.",
    )
    parser.add_argument(
        "--delay-target-load-file",
        type=str,
        default=None,
        help="Optional model file path for agent_delay_target.",
    )
    parser.add_argument(
        "--delay-target-ring-buffer-size",
        type=int,
        default=None,
        help="Optional override for agent_delay_target ring_buffer_size (frames).",
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


def validate_args(args: argparse.Namespace) -> None:
    if str(args.runner_mode) == "carmack_compat" and int(args.frame_skip) != 1:
        raise ValueError(
            "--runner-mode carmack_compat requires --frame-skip 1 "
            "(agent-owned cadence; runner does not apply frame-skip)."
        )


class _FrameFromStepAdapter:
    def __init__(self, step_agent) -> None:
        self._step_agent = step_agent
        self._frame_idx = 0

    def frame(self, obs_rgb, reward, boundary) -> int:
        if isinstance(boundary, Mapping):
            terminated = bool(boundary.get("terminated", False))
            truncated = bool(boundary.get("truncated", False))
            end_of_episode_pulse = bool(boundary.get("end_of_episode_pulse", False))
            boundary_cause = boundary.get("boundary_cause")
        else:
            terminated = bool(boundary)
            truncated = False
            end_of_episode_pulse = bool(boundary)
            boundary_cause = None
        action = self._step_agent.step(
            obs_rgb=obs_rgb,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info={
                "frame_idx": int(self._frame_idx),
                "end_of_episode_pulse": bool(end_of_episode_pulse),
                "boundary_cause": boundary_cause,
            },
        )
        self._frame_idx += 1
        return int(action)


def build_agent(args: argparse.Namespace, num_actions: int, total_frames: int):
    if args.agent == "random":
        base = RandomAgent(num_actions=num_actions, seed=args.seed)
    elif args.agent == "repeat":
        if args.repeat_action_idx < 0 or args.repeat_action_idx >= num_actions:
            raise ValueError(f"--repeat-action-idx must be in [0, {num_actions - 1}]")
        base = RepeatActionAgent(action_idx=args.repeat_action_idx)
    else:
        from benchmark.agents_delay_target import DelayTargetAdapter  # pylint: disable=import-outside-toplevel

        kwargs = {
            "gpu": int(args.delay_target_gpu),
            "use_cuda_graphs": bool(int(args.delay_target_use_cuda_graphs)),
        }
        if args.delay_target_load_file:
            kwargs["load_file"] = str(args.delay_target_load_file)
        if args.delay_target_ring_buffer_size is not None:
            kwargs["ring_buffer_size"] = int(args.delay_target_ring_buffer_size)
        base = DelayTargetAdapter(
            data_dir=str(Path(args.logdir)),
            seed=int(args.seed),
            num_actions=int(num_actions),
            total_frames=int(total_frames),
            agent_kwargs=kwargs,
        )

    if args.runner_mode == "carmack_compat":
        if hasattr(base, "frame") and callable(getattr(base, "frame")):
            return base
        return _FrameFromStepAdapter(base)
    return base


def build_config_payload(
    args: argparse.Namespace,
    env: ALEAtariEnv,
    runner_config,
    run_dir: Path,
) -> Dict:
    payload = {
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
        "runner_mode": str(args.runner_mode),
        "lives_as_episodes": bool(args.lives_as_episodes),
        "max_frames_without_reward": int(args.max_frames_without_reward),
        "reset_on_life_loss": bool(args.reset_on_life_loss),
        "compat_reset_delay_queue_on_reset": bool(args.compat_reset_delay_queue_on_reset),
        "compat_log_every_frames": int(args.compat_log_every_frames),
        "compat_log_pulses_every": int(args.compat_log_pulses_every),
        "compat_log_resets_every": int(args.compat_log_resets_every),
        "delay_target_ring_buffer_size": (
            None if args.delay_target_ring_buffer_size is None else int(args.delay_target_ring_buffer_size)
        ),
        "timestamps": bool(args.timestamps),
        "logdir": str(Path(args.logdir)),
        "run_dir": str(run_dir),
        "resolved_action_set": list(env.action_set),
        "num_actions": len(env.action_set),
        "runner_config": {
            "total_frames": runner_config.total_frames,
            "delay_frames": runner_config.delay_frames,
            "default_action_idx": runner_config.default_action_idx,
            "include_timestamps": runner_config.include_timestamps,
        },
    }
    if isinstance(runner_config, RunnerConfig):
        payload["runner_config"]["frame_skip"] = int(runner_config.frame_skip)
    if isinstance(runner_config, CarmackRunnerConfig):
        payload["runner_config"].update(
            {
                "runner_mode": "carmack_compat",
                "action_cadence_mode": "agent_owned",
                "frame_skip_enforced": 1,
                "lives_as_episodes": bool(runner_config.lives_as_episodes),
                "max_frames_without_reward": int(runner_config.max_frames_without_reward),
                "reset_on_life_loss": bool(runner_config.reset_on_life_loss),
                "reset_delay_queue_on_reset": bool(runner_config.reset_delay_queue_on_reset),
                "log_rank": int(runner_config.log_rank),
                "log_name": str(runner_config.log_name),
                "rolling_average_frames": int(runner_config.rolling_average_frames),
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    validate_args(args)
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

    agent = build_agent(args, num_actions=len(env.action_set), total_frames=int(args.frames))

    run_dir = make_run_dir(Path(args.logdir), args.game, args.seed)
    event_writer = JsonlWriter(run_dir / "events.jsonl")
    episode_writer = JsonlWriter(run_dir / "episodes.jsonl")

    if args.runner_mode == "carmack_compat":
        if args.agent == "delay_target":
            log_name = f"delay_{args.game}{int(args.delay)}_{int(args.frame_skip)}"
            if args.delay_target_ring_buffer_size is not None:
                log_name += f"_{int(args.delay_target_ring_buffer_size)}"
            log_rank = int(args.seed)
        else:
            log_name = str(args.agent)
            log_rank = int(args.seed)
        runner_config = CarmackRunnerConfig(
            total_frames=int(args.frames),
            delay_frames=int(args.delay),
            default_action_idx=int(args.default_action_idx),
            include_timestamps=bool(args.timestamps),
            lives_as_episodes=bool(args.lives_as_episodes),
            max_frames_without_reward=int(args.max_frames_without_reward),
            reset_on_life_loss=bool(args.reset_on_life_loss),
            reset_delay_queue_on_reset=bool(args.compat_reset_delay_queue_on_reset),
            progress_log_interval_frames=int(args.compat_log_every_frames),
            pulse_log_interval=int(args.compat_log_pulses_every),
            reset_log_interval=int(args.compat_log_resets_every),
            log_rank=log_rank,
            log_name=log_name,
        )
    else:
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
        if args.runner_mode == "carmack_compat":
            summary = CarmackCompatRunner(
                env=env,
                agent=agent,
                config=runner_config,
                event_writer=event_writer,
                episode_writer=episode_writer,
            ).run()
        else:
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
