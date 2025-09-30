# BBF Atari 100K Validation Guide

This guide explains how to validate your benchmarking infrastructure by reproducing BBF results from the paper.

## Overview

The BBF (Bigger, Better, Faster) agent achieves super-human performance on the Atari 100K benchmark. To validate that your implementation is correct, we'll run BBF on **Atlantis** and compare scores to the paper.

### Expected Scores for Atlantis

From the [BBF paper](https://arxiv.org/abs/2305.19452) and official repository:

**Table A.1 (RR=8, 50 seeds average):**
- BBF average score: **1,173.2**
- Human-normalized: varies by seed

**Official scores from `scores/RR2_BBF.csv`:**
- Seed 7779: **80,689** (normalized: 4.19)
- Seed 7780: **51,302** (normalized: 2.38)

**Paper statistics:**
- Human score: 29,028.1
- Random score: 12,850.0

## Quick Start

### 1. Install Dependencies

```bash
cd upper_bound_benchmark
pip install -r requirements.txt
python -m AutoROM.accept-rom-license --accept-license
```

### 2. Run BBF on Atlantis (RR=2)

```bash
python scripts/run_atari100k.py \
  --agent bbf \
  --game Atlantis \
  --seed 7779 \
  --replay-ratio 2 \
  --no-realtime \
  --output-dir results/atari100k/atlantis_rr2_seed7779
```

**Expected runtime:** ~2-4 hours on a modern GPU (depending on hardware)

### 3. Run BBF on Atlantis (RR=8)

```bash
python scripts/run_atari100k.py \
  --agent bbf \
  --game Atlantis \
  --seed 7779 \
  --replay-ratio 8 \
  --no-realtime \
  --output-dir results/atari100k/atlantis_rr8_seed7779
```

**Expected runtime:** ~8-16 hours on a modern GPU

### 4. Compare Results

The script will automatically print comparison to paper results:

```
Results for Atlantis:
  Raw score: XXXXX
  Human-normalized score: X.XX

BBF paper reports (Table A.1):
  Average score (50 seeds, RR=8): 1173.2
  Normalized: 4.19 (seed 7779), 2.38 (seed 7780)
```

## Validation Criteria

Your implementation is **correctly validated** if:

1. **Same Seed Reproduction:** Running seed 7779 or 7780 produces scores within **±20%** of the official scores
2. **Performance Range:** Your scores are in the reasonable range for BBF (above random, approaching/exceeding human)
3. **No Crashes:** Training completes without errors for 100K steps

## Troubleshooting

### Score is Much Lower Than Expected

Possible issues:
- **Wrong replay ratio:** Make sure you're comparing RR=2 to RR=2, RR=8 to RR=8
- **Sticky actions enabled:** Atari 100K uses `sticky_prob=0.0` (disabled)
- **Wrong number of steps:** Should be 100,000 steps, not frames
- **Network initialization:** PyTorch vs JAX may have different default initializations

### Score is Much Higher Than Expected

Possible issues:
- **Using more than 100K steps**
- **Sticky actions disabled when they should be enabled**
- **Training on easy version of the game**

### Out of Memory

BBF uses a large network (width_scale=4). Try:
- Reduce `--batch-size` from 32 to 16
- Reduce `--width-scale` from 4 to 2 (but this changes the agent)
- Use a GPU with more memory

## Key Differences: PyTorch vs JAX Implementation

Our PyTorch implementation aims to match the official JAX implementation, but some differences exist:

| Component | JAX (Official) | PyTorch (Ours) |
|-----------|----------------|----------------|
| Framework | JAX/Dopamine | PyTorch |
| Replay Buffer | Prioritized Subsequence | Prioritized Subsequence |
| Data Augmentation | DrQ-style (JAX) | DrQ-style (PyTorch) |
| Optimizer | Adam with weight decay | AdamW |
| Mixed Precision | Optional | Not implemented |
| Parallelism | JAX pmap | Single GPU |

## Running on Other Games

To test on other Atari 100K games:

```bash
python scripts/run_atari100k.py \
  --agent bbf \
  --game Breakout \
  --seed 0 \
  --replay-ratio 2
```

Available games: See `atari100k_config.py` for the full list of 26 games.

## Full Benchmark

To run the complete Atari 100K benchmark (26 games):

```bash
for game in Alien Amidar Assault Asterix Atlantis BankHeist BattleZone Boxing Breakout Centipede ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown; do
    python scripts/run_atari100k.py \
      --agent bbf \
      --game $game \
      --seed 0 \
      --replay-ratio 2 \
      --no-realtime \
      --output-dir results/atari100k/${game,,}_rr2_seed0
done
```

**Warning:** This will take several days to complete.

## References

- **BBF Paper:** [Schwarzer et al. "Bigger, Better, Faster: Human-level Atari with human-level efficiency" (2023)](https://arxiv.org/abs/2305.19452)
- **Official Code:** https://github.com/google-research/google-research/tree/master/bigger_better_faster
- **Atari 100K Benchmark:** [Kaiser et al. "Model-based reinforcement learning for Atari" (2020)](https://arxiv.org/abs/1903.00374)

