#!/usr/bin/env python3
"""Monitor training progress by reading CSV metrics files.

Usage:
    # Watch training metrics in real-time:
    python scripts/monitor_training.py results/atari100k/atlantis_test
    
    # Or use watch command for auto-refresh:
    watch -n 5 python scripts/monitor_training.py results/atari100k/atlantis_test
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def format_number(num_str):
    """Format number with commas."""
    try:
        num = float(num_str)
        if num >= 1000:
            return f"{num:,.0f}"
        elif num >= 1:
            return f"{num:.2f}"
        else:
            return f"{num:.4f}"
    except:
        return num_str


def read_last_n_lines(filepath, n=10):
    """Read last N lines from a file efficiently."""
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    return lines[-n:] if len(lines) > n else lines


def monitor_training(output_dir):
    """Display training progress from CSV files."""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return
    
    metrics_file = output_dir / "training_metrics.csv"
    episode_file = output_dir / "episode_metrics.csv"
    
    print("=" * 80)
    print(f"Training Progress Monitor: {output_dir}")
    print("=" * 80)
    
    # Training metrics
    if metrics_file.exists():
        lines = read_last_n_lines(metrics_file, n=11)  # Header + last 10 entries
        if len(lines) > 1:
            print("\n📊 TRAINING METRICS (Last 10 updates):")
            print("-" * 80)
            
            # Parse header
            reader = csv.DictReader(lines)
            rows = list(reader)
            
            if rows:
                # Show latest metrics prominently
                latest = rows[-1]
                print(f"\n🎯 Latest Update (Step {format_number(latest['env_steps'])}):")
                print(f"  Training Steps:  {format_number(latest['learn_steps'])}")
                print(f"  TD Loss:         {float(latest['td_loss']):.4f}")
                print(f"  SPR Loss:        {float(latest['spr_loss']):.4f}")
                print(f"  Total Loss:      {float(latest['total_loss']):.4f}")
                print(f"  Mean Q-value:    {float(latest['mean_q']):.2f}")
                print(f"  Max Q-value:     {float(latest['max_q']):.2f}")
                print(f"  Gamma (γ):       {float(latest['gamma']):.4f}")
                print(f"  N-step:          {latest['n_step']}")
                print(f"  Time Elapsed:    {float(latest['time_elapsed']):.1f}s")
                
                # Show trend (last 5 vs previous 5)
                if len(rows) >= 10:
                    prev_5_loss = sum(float(r['total_loss']) for r in rows[-10:-5]) / 5
                    last_5_loss = sum(float(r['total_loss']) for r in rows[-5:]) / 5
                    trend = "📉 Decreasing" if last_5_loss < prev_5_loss else "📈 Increasing"
                    print(f"\n  Loss Trend:      {trend} ({prev_5_loss:.4f} → {last_5_loss:.4f})")
                
                # Show table
                print(f"\n  Step History (last {len(rows)} updates):")
                print(f"  {'Step':<10} {'TD Loss':<10} {'SPR Loss':<10} {'Mean Q':<10} {'Gamma':<8}")
                for row in rows[-5:]:
                    print(f"  {format_number(row['env_steps']):<10} "
                          f"{float(row['td_loss']):<10.4f} "
                          f"{float(row['spr_loss']):<10.4f} "
                          f"{float(row['mean_q']):<10.2f} "
                          f"{float(row['gamma']):<8.4f}")
    else:
        print(f"\n⚠️  No training metrics file found at {metrics_file}")
    
    # Episode metrics
    if episode_file.exists():
        lines = read_last_n_lines(episode_file, n=11)  # Header + last 10 episodes
        if len(lines) > 1:
            print("\n" + "=" * 80)
            print("📈 EPISODE METRICS (Last 10 episodes):")
            print("-" * 80)
            
            reader = csv.DictReader(lines)
            rows = list(reader)
            
            if rows:
                latest_ep = rows[-1]
                print(f"\n🎮 Latest Episode (#{latest_ep['episode_num']}):")
                print(f"  Return:       {float(latest_ep['episode_return']):.1f}")
                print(f"  Length:       {latest_ep['episode_length']} steps")
                print(f"  Total Steps:  {format_number(latest_ep['env_steps'])}")
                
                # Calculate statistics over last N episodes
                if len(rows) >= 5:
                    recent_returns = [float(r['episode_return']) for r in rows[-5:]]
                    avg_return = sum(recent_returns) / len(recent_returns)
                    max_return = max(recent_returns)
                    min_return = min(recent_returns)
                    print(f"\n  Last 5 Episodes:")
                    print(f"    Average Return: {avg_return:.1f}")
                    print(f"    Best Return:    {max_return:.1f}")
                    print(f"    Worst Return:   {min_return:.1f}")
                
                # Show table
                print(f"\n  Episode History (last {min(len(rows), 10)} episodes):")
                print(f"  {'Ep#':<6} {'Step':<10} {'Return':<12} {'Length':<8}")
                for row in rows[-10:]:
                    print(f"  {row['episode_num']:<6} "
                          f"{format_number(row['env_steps']):<10} "
                          f"{float(row['episode_return']):<12.1f} "
                          f"{row['episode_length']:<8}")
    else:
        print(f"\n⚠️  No episode metrics file found at {episode_file}")
    
    print("\n" + "=" * 80)
    print("💡 Tip: Use 'watch -n 5 python scripts/monitor_training.py <dir>' for auto-refresh")
    print("💡 Or: tail -f results/atari100k/atlantis_test/training_metrics.csv")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Monitor BBF training progress")
    parser.add_argument("output_dir", type=str, help="Output directory with metrics CSV files")
    parser.add_argument("--episodes", type=int, default=10, help="Number of recent episodes to show")
    
    args = parser.parse_args()
    
    try:
        monitor_training(args.output_dir)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
