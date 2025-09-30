#!/usr/bin/env python3
"""Run Atari 100K benchmark for BBF validation.

This script runs a single game for 100,000 steps to match the Atari 100K benchmark.
Example:
    python scripts/run_atari100k.py --agent bbf --game Atlantis --seed 7779
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from upper_bound_benchmark.benchmark import BenchmarkRunner, BenchmarkConfig
from upper_bound_benchmark.agents.bbf_agent import BBFAgent, BBFConfig
from atari100k_config import ATARI_100K_GAMES, normalize_score


def main():
    parser = argparse.ArgumentParser(description="Run Atari 100K benchmark")
    parser.add_argument("--agent", type=str, default="bbf", help="Agent type")
    parser.add_argument("--game", type=str, default="Atlantis", help="Game name (e.g., 'Atlantis')")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--replay-ratio", type=int, default=2, choices=[2, 8],
                       help="Replay ratio (2 for RR=2, 8 for RR=8)")
    parser.add_argument("--output-dir", type=str, default="results/atari100k")
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument("--no-realtime", action="store_true", help="Disable real-time pacing")
    
    args = parser.parse_args()
    
    # Construct game ID
    game_id = f"ALE/{args.game}-v5"
    
    # Validate game is in Atari 100K
    if game_id not in ATARI_100K_GAMES:
        print(f"Warning: {game_id} not in standard Atari 100K games")
        print(f"Available games: {[g.split('/')[1].split('-')[0] for g in ATARI_100K_GAMES]}")
    
    # BBF config for Atari 100K
    # Note: The official BBF uses replay_ratio in terms of (batch_size * batches_to_group)
    # BBF.gin has replay_ratio=64, batch_size=32, batches_to_group=2
    # This gives 64/(32*2) = 1 effective replay ratio per transition
    # But we use simpler definition: number of gradient updates per env step
    bbf_config = BBFConfig(
        frame_stack=4,
        width_scale=4,
        learning_rate=1e-4,
        encoder_learning_rate=1e-4,
        weight_decay=0.1,
        gamma_min=0.97,
        gamma_max=0.997,
        update_horizon_min=3,
        update_horizon_max=10,
        buffer_capacity=200_000,
        batch_size=32,
        replay_ratio=args.replay_ratio,
        learning_starts=2_000,
        target_update_tau=0.005,
        target_action_selection=True,
        reset_every=20_000,
        shrink_factor=0.5,
        perturb_factor=0.5,
        cycle_steps=10_000,
        no_resets_after=100_000,
        spr_weight=5.0,
        jumps=5,
        data_augmentation=True,
        dueling=True,
        double_dqn=True,
        distributional=True,
    )
    
    agent = BBFAgent(seed=args.seed, config=bbf_config)
    
    # Atari 100K benchmark config
    benchmark_config = BenchmarkConfig(
        cycles=1,  # Single cycle for Atari 100K
        frames_per_game=100_000,  # 100K steps
        sticky_prob=0.0,  # No sticky actions for Atari 100K
        full_action_space=True,
        target_fps=60.0,
        realtime=not args.no_realtime,
        latency_frames=0,
        seed=args.seed,
        render=args.render,
        output_dir=args.output_dir,
    )
    
    # Run benchmark
    print(f"Running BBF on {args.game} (seed={args.seed}, RR={args.replay_ratio})")
    print(f"Config: 100K steps, no sticky actions, single cycle")
    print(f"Output directory: {args.output_dir}")
    print("-" * 80)
    
    runner = BenchmarkRunner(games=[game_id], config=benchmark_config)
    results = runner.run(agent)
    
    # Print results
    score = results["final_cycle_scores"][game_id]
    normalized_score = normalize_score(game_id, score)
    
    print("-" * 80)
    print(f"\nResults for {args.game}:")
    print(f"  Raw score: {score:.2f}")
    print(f"  Human-normalized score: {normalized_score:.3f}")
    print(f"\nResults saved to {Path(args.output_dir) / 'results.json'}")
    
    # Compare to paper if Atlantis
    if args.game == "Atlantis":
        print(f"\nBBF paper reports (Table A.1):")
        print(f"  Average score (50 seeds, RR=8): 1173.2")
        print(f"  Normalized: 4.19 (seed 7779), 2.38 (seed 7780)")
        if args.seed in [7779, 7780]:
            print(f"  Note: Your seed {args.seed} can be compared to paper results")


if __name__ == "__main__":
    main()

