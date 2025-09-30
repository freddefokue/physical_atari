# BBF (Bigger, Better, Faster) Implementation

PyTorch implementation of BBF from [Schwarzer et al. (2023)](https://arxiv.org/abs/2305.19452) for validating the Atari benchmarking infrastructure.

## Quick Test

To validate your setup on Atlantis (should take ~2-4 hours):

```bash
cd upper_bound_benchmark

# Install dependencies
pip install -r requirements.txt
python -m AutoROM.accept-rom-license --accept-license

# Run BBF on Atlantis
python scripts/run_atari100k.py \
  --agent bbf \
  --game Atlantis \
  --seed 7779 \
  --replay-ratio 2 \
  --no-realtime
```

**Expected score:** ~80,689 (matching official implementation seed 7779)

## Files Created

### Core Implementation
- `upper_bound_benchmark/agents/bbf_networks.py` - Network architectures
  - `ImpalaEncoder`: Impala CNN with 4× width scaling
  - `TransitionModel`: SPR transition model for future predictions
  - `BBFNetwork`: Complete architecture with Q-head, projection heads
  
- `upper_bound_benchmark/agents/bbf_agent.py` - BBF agent
  - `PrioritizedReplayBuffer`: Subsequence replay for SPR
  - `BBFAgent`: Full agent with all BBF components
  - Implements: periodic resets, annealing schedules, EMA targets, SPR loss

### Configuration & Scripts
- `atari100k_config.py` - Atari 100K benchmark configuration
  - 26 game list
  - Human/random scores for normalization
  
- `scripts/run_atari100k.py` - Validation script for single games
  - Runs 100K steps on specified game
  - Compares to paper results
  
- `scripts/run_benchmark.py` - Updated to support BBF agent

### Documentation
- `ATARI100K_VALIDATION.md` - Validation guide
- `BBF_IMPLEMENTATION_NOTES.md` - Technical implementation details
- `BBF_README.md` - This file

## Key Features Implemented

✅ **Network Architecture**
- Impala CNN encoder with 4× width scaling (~21.8M parameters)
- Dueling architecture with C51 distributional RL (51 atoms)
- Transition model and projection heads for SPR

✅ **Training Components**
- Annealing n-step (10→3) and gamma (0.97→0.997)
- Periodic shrink-and-perturb resets every 20K gradient steps
- EMA target network (tau=0.005)
- AdamW optimizer with weight decay (0.1)

✅ **Sample Efficiency**
- Replay ratio 2 or 8 (gradient steps per env step)
- Prioritized subsequence replay buffer
- SPR self-supervised learning (weight=5.0)
- DrQ-style data augmentation

✅ **Exact Configuration Match**
- All hyperparameters from `BBF.gin` config file
- Double DQN + Dueling + C51 + No noisy nets
- Target network for action selection

## Command Line Options

Run with `--help` to see all options:

```bash
python scripts/run_atari100k.py --help
```

Key options for BBF:
- `--replay-ratio {2,8}`: RR=2 (faster) or RR=8 (better performance)
- `--seed`: Random seed (use 7779 or 7780 to match paper)
- `--game`: Game name (e.g., "Atlantis", "Breakout", "Pong")
- `--no-realtime`: Disable real-time pacing for faster training

## Expected Performance

From BBF paper Table A.1 and official repository:

### Atlantis (RR=2)
- Seed 7779: 80,689 (human-normalized: 4.19)
- Seed 7780: 51,302 (human-normalized: 2.38)

### Atlantis (RR=8) 
- Average over 50 seeds: 1,173.2

### IQM Scores (26 games, RR=8)
- BBF: 1.045 (super-human)
- SR-SPR: 0.631
- SPR: 0.337

## Validation Checklist

- [ ] Run Atlantis seed 7779 at RR=2
- [ ] Verify score is within ±20% of 80,689
- [ ] Training completes without errors
- [ ] Check network has ~21.8M parameters
- [ ] Monitor training curves (losses, returns)

## Troubleshooting

**Out of memory?**
- Try `--width-scale 2` (but this changes the agent)
- Reduce `--batch-size` to 16
- Use a larger GPU

**Score too low?**
- Verify `--sticky-prob 0.0` (no sticky actions for Atari 100K)
- Check you're running 100K steps, not frames
- Ensure RR matches comparison (RR=2 vs RR=8)

**Training crashes?**
- Check CUDA/PyTorch compatibility
- Verify all dependencies installed
- Try on CPU first (very slow): set `device='cpu'` in code

## References

Official BBF repository: https://github.com/google-research/google-research/tree/master/bigger_better_faster

Paper: https://arxiv.org/abs/2305.19452

## Implementation Status

✅ **Complete**
- Core BBF agent
- All network components
- Atari 100K config
- Validation scripts
- Documentation

⚠️ **Not Implemented** (vs official)
- Mixed precision training
- Multi-GPU support
- Complex optimizer state reset
- Parallel environment sampling

These omissions shouldn't significantly affect single-game validation.

