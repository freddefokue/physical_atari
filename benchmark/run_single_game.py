"""CLI entrypoint for the single-game streaming Atari benchmark runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from benchmark.agents import RandomAgent, RepeatActionAgent
from benchmark.ale_env import ALEAtariEnv, ALEEnvConfig
from benchmark.carmack_runner import (
    CARMACK_SINGLE_RUN_PROFILE,
    CARMACK_SINGLE_RUN_SCHEMA_VERSION,
    CarmackCompatRunner,
    CarmackRunnerConfig,
)
from benchmark.logging_utils import JsonlWriter, dump_json, make_run_dir
from benchmark.runner import BenchmarkRunner, RunnerConfig

RUNTIME_FINGERPRINT_SCHEMA_VERSION = "runtime_fingerprint_v1"


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
        choices=["random", "repeat", "tinydqn", "delay_target"],
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
        "--dqn-log-train-every",
        type=int,
        default=500,
        help="Emit TinyDQN training log every N train steps (0 disables).",
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
        "--dqn-decision-interval",
        type=int,
        default=1,
        help="TinyDQN decision interval in frames (agent-owned action repeat).",
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
    if str(args.agent) == "tinydqn" and str(args.runner_mode) != "carmack_compat":
        raise ValueError(
            "--agent tinydqn currently requires --runner-mode carmack_compat "
            "because TinyDQN expects applied-action labels from the Carmack boundary payload."
        )
    if str(args.agent) == "tinydqn" and int(args.dqn_decision_interval) <= 0:
        raise ValueError("--dqn-decision-interval must be > 0 for --agent tinydqn.")


class _FrameFromStepAdapter:
    def __init__(self, step_agent, decision_interval: int = 1) -> None:
        self._step_agent = step_agent
        self._decision_interval = max(1, int(decision_interval))
        self._frame_counter = 0

    def frame(self, obs_rgb, reward, boundary) -> int:
        if isinstance(boundary, Mapping):
            terminated = bool(boundary.get("terminated", False))
            truncated = bool(boundary.get("truncated", False))
            end_of_episode_pulse = bool(boundary.get("end_of_episode_pulse", False))
            has_prev_applied_action = bool(boundary.get("has_prev_applied_action", False))
            prev_applied_action_idx = int(boundary.get("prev_applied_action_idx", 0))
            if "is_decision_frame" in boundary:
                is_decision_frame = bool(boundary.get("is_decision_frame"))
            else:
                is_decision_frame = bool(self._frame_counter % self._decision_interval == 0)
        else:
            terminated = bool(boundary)
            truncated = False
            end_of_episode_pulse = bool(boundary)
            has_prev_applied_action = False
            prev_applied_action_idx = 0
            is_decision_frame = bool(self._frame_counter % self._decision_interval == 0)
        action = self._step_agent.step(
            obs_rgb=obs_rgb,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info={
                "end_of_episode_pulse": bool(end_of_episode_pulse),
                "has_prev_applied_action": bool(has_prev_applied_action),
                "prev_applied_action_idx": int(prev_applied_action_idx),
                "is_decision_frame": bool(is_decision_frame),
            },
        )
        self._frame_counter += 1
        return int(action)

    def get_stats(self) -> Mapping[str, object]:
        stats_fn = getattr(self._step_agent, "get_stats", None)
        if callable(stats_fn):
            try:
                payload = stats_fn()
            except Exception:  # pragma: no cover - defensive
                return {}
            if isinstance(payload, Mapping):
                return payload
        return {}


def build_agent(args: argparse.Namespace, num_actions: int, total_frames: int):
    if args.agent == "random":
        base = RandomAgent(num_actions=num_actions, seed=args.seed)
    elif args.agent == "repeat":
        if args.repeat_action_idx < 0 or args.repeat_action_idx >= num_actions:
            raise ValueError(f"--repeat-action-idx must be in [0, {num_actions - 1}]")
        base = RepeatActionAgent(action_idx=args.repeat_action_idx)
    elif args.agent == "delay_target":
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
    else:
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
            train_log_interval=int(args.dqn_log_train_every),
            target_update_decisions=int(args.dqn_target_update),
            replay_min_size=int(args.dqn_replay_min),
            eps_start=float(args.dqn_eps_start),
            eps_end=float(args.dqn_eps_end),
            eps_decay_frames=int(args.dqn_eps_decay_frames),
            use_replay=bool(int(args.dqn_use_replay)),
            device=str(args.dqn_device),
            decision_interval=int(args.dqn_decision_interval),
        )
        base = TinyDQNAgent(action_space_n=num_actions, seed=int(args.seed), config=dqn_config)

    if args.runner_mode == "carmack_compat":
        if hasattr(base, "frame") and callable(getattr(base, "frame")):
            return base
        decision_interval = int(args.dqn_decision_interval) if str(args.agent) == "tinydqn" else 1
        return _FrameFromStepAdapter(base, decision_interval=decision_interval)
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
        "dqn_config": {
            "gamma": float(args.dqn_gamma),
            "lr": float(args.dqn_lr),
            "buffer_size": int(args.dqn_buffer_size),
            "batch_size": int(args.dqn_batch_size),
            "train_every_decisions": int(args.dqn_train_every),
            "train_log_interval": int(args.dqn_log_train_every),
            "target_update_decisions": int(args.dqn_target_update),
            "replay_min_size": int(args.dqn_replay_min),
            "eps_start": float(args.dqn_eps_start),
            "eps_end": float(args.dqn_eps_end),
            "eps_decay_frames": int(args.dqn_eps_decay_frames),
            "use_replay": bool(int(args.dqn_use_replay)),
            "device": str(args.dqn_device),
            "decision_interval": int(args.dqn_decision_interval),
        },
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
        payload["single_run_profile"] = CARMACK_SINGLE_RUN_PROFILE
        payload["single_run_schema_version"] = CARMACK_SINGLE_RUN_SCHEMA_VERSION
        payload["runner_config"].update(
            {
                "runner_mode": CARMACK_SINGLE_RUN_PROFILE,
                "action_cadence_mode": "agent_owned",
                "frame_skip_enforced": 1,
                "single_run_schema_version": CARMACK_SINGLE_RUN_SCHEMA_VERSION,
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


def build_run_summary_payload(
    args: argparse.Namespace,
    summary: Mapping[str, object],
    *,
    runtime_fingerprint: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "runner_mode": str(args.runner_mode),
        **dict(summary),
    }
    if str(args.runner_mode) == CARMACK_SINGLE_RUN_PROFILE:
        payload["single_run_profile"] = CARMACK_SINGLE_RUN_PROFILE
        payload["single_run_schema_version"] = CARMACK_SINGLE_RUN_SCHEMA_VERSION
    if runtime_fingerprint is not None:
        payload["runtime_fingerprint"] = dict(runtime_fingerprint)
    return payload


def _stable_payload_sha256(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _try_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        if isinstance(version, str) and version:
            return str(version)
    except Exception:
        pass
    return "unknown"


def build_runtime_fingerprint_payload(
    args: argparse.Namespace,
    config_payload: Mapping[str, object],
) -> Dict[str, object]:
    rom_path: str = "unknown"
    rom_sha256: str = "0" * 64
    try:
        rom_candidate = ALEAtariEnv._resolve_rom_path(str(args.game))
        rom_path_obj = Path(str(rom_candidate))
        rom_path = str(rom_path_obj)
        if rom_path_obj.exists() and rom_path_obj.is_file():
            rom_sha256 = _file_sha256(rom_path_obj)
    except Exception:
        pass

    torch_version = _try_version("torch")
    cuda_available = False
    try:
        import torch  # pylint: disable=import-outside-toplevel

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    return {
        "fingerprint_schema_version": RUNTIME_FINGERPRINT_SCHEMA_VERSION,
        "runner_mode": str(args.runner_mode),
        "single_run_profile": str(config_payload.get("single_run_profile", args.runner_mode)),
        "single_run_schema_version": str(config_payload.get("single_run_schema_version", "n/a")),
        "game": str(args.game),
        "seed": int(args.seed),
        "seed_policy": "global_seed_python_numpy_ale",
        "frames": int(args.frames),
        "config_sha256_algorithm": "sha256",
        "config_sha256_scope": "config_without_runtime_fingerprint",
        "config_sha256": _stable_payload_sha256(config_payload),
        "python_version": str(sys.version.split()[0]),
        "ale_py_version": _try_version("ale_py"),
        "rom_sha256": str(rom_sha256),
        "platform": str(platform.platform()),
        "machine": str(platform.machine()),
        "processor": str(platform.processor() or "unknown"),
        "python_implementation": str(platform.python_implementation()),
        "python_executable": str(sys.executable),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch_version),
        "cuda_available": bool(cuda_available),
        "rom_path": str(rom_path),
    }


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
    runtime_fingerprint = build_runtime_fingerprint_payload(args, config_payload)
    config_payload_with_fingerprint = dict(config_payload)
    config_payload_with_fingerprint["runtime_fingerprint"] = dict(runtime_fingerprint)
    dump_json(run_dir / "config.json", config_payload_with_fingerprint)
    # Sidecar mirror remains useful for external tooling and backwards compatibility.
    dump_json(run_dir / "runtime_fingerprint.json", dict(runtime_fingerprint))

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

    dump_json(run_dir / "run_summary.json", build_run_summary_payload(args, summary, runtime_fingerprint=runtime_fingerprint))
    print(f"Run complete: {run_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
