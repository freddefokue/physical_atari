#!/usr/bin/env python3
import argparse
from pathlib import Path

from upper_bound_benchmark.benchmark import BenchmarkRunner, BenchmarkConfig, DEFAULT_GAMES
from upper_bound_benchmark.agents.random_agent import RandomAgent
from upper_bound_benchmark.agents.dqn_agent import DQNAgent, DQNConfig
from upper_bound_benchmark.agents.bbf_agent import BBFAgent, BBFConfig


def build_agent(name: str, seed: int, args):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "dqn":
        cfg = DQNConfig(
            frame_stack=args.frame_stack,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            buffer_capacity=args.buffer_capacity,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            train_freq=args.train_freq,
            target_update_freq=args.target_update_freq,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            max_grad_norm=args.max_grad_norm,
        )
        return DQNAgent(seed=seed, config=cfg)
    if name == "bbf":
        cfg = BBFConfig(
            frame_stack=args.frame_stack,
            width_scale=args.width_scale,
            learning_rate=args.learning_rate,
            buffer_capacity=args.buffer_capacity,
            batch_size=args.batch_size,
            replay_ratio=args.replay_ratio,
            learning_starts=args.learning_starts,
            spr_weight=args.spr_weight,
            reset_every=args.reset_every,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            update_horizon_min=args.update_horizon_min,
            update_horizon_max=args.update_horizon_max,
        )
        return BBFAgent(seed=seed, config=cfg)
    raise ValueError(f"Unknown agent: {name}")


def parse_args():
    p = argparse.ArgumentParser(description="Run Carmack Upper Bound Atari benchmark (unofficial)")
    p.add_argument("--agent", type=str, default="random", help="Agent name: random, dqn, bbf")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--frames-per-game", type=int, default=400_000)
    p.add_argument("--sticky-prob", type=float, default=0.25)
    p.add_argument("--target-fps", type=float, default=60.0)
    p.add_argument("--latency-frames", type=int, default=0)
    p.add_argument("--no-realtime", action="store_true", help="Disable real-time pacing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--games", type=str, nargs="*", default=DEFAULT_GAMES)

    # DQN-specific options
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer-capacity", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-update-freq", type=int, default=10_000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=1_000_000)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    
    # BBF-specific options
    p.add_argument("--width-scale", type=int, default=4, help="BBF: Width scaling for Impala encoder")
    p.add_argument("--replay-ratio", type=int, default=2, help="BBF: Gradient updates per env step")
    p.add_argument("--spr-weight", type=float, default=5.0, help="BBF: SPR loss weight")
    p.add_argument("--reset-every", type=int, default=20_000, help="BBF: Reset every N gradient steps")
    p.add_argument("--gamma-min", type=float, default=0.97, help="BBF: Starting gamma (anneals to gamma-max)")
    p.add_argument("--gamma-max", type=float, default=0.997, help="BBF: Final gamma")
    p.add_argument("--update-horizon-min", type=int, default=3, help="BBF: Min n-step")
    p.add_argument("--update-horizon-max", type=int, default=10, help="BBF: Max n-step (anneals to min)")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = BenchmarkConfig(
        cycles=args.cycles,
        frames_per_game=args.frames_per_game,
        sticky_prob=args.sticky_prob,
        full_action_space=True,
        target_fps=args.target_fps,
        realtime=not args.no_realtime,
        latency_frames=args.latency_frames,
        seed=args.seed,
        render=args.render,
        output_dir=args.output_dir,
    )

    agent = build_agent(args.agent, seed=args.seed, args=args)

    runner = BenchmarkRunner(games=args.games, config=cfg)
    results = runner.run(agent)

    print("Final cycle per-game scores:")
    for g, s in results["final_cycle_scores"].items():
        print(f"  {g}: {s:.2f}")
    print(f"Final total: {results['final_total']:.2f}")

    print(f"Results saved to {Path(cfg.output_dir) / 'results.json'}")


if __name__ == "__main__":
    main()
