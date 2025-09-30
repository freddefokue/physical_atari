## Carmack Upper Bound Atari Benchmark (Unofficial)

This repo provides a minimal, reproducible harness to run the Upper Bound 2025 Atari benchmark proposed by John Carmack.

Key rules implemented:
- Multi-game, sequential learning across 8 Atari 2600 games
- 3 cycles per game, 400,000 frames per cycle (frameskip=1)
- Full action set
- Sticky actions enabled
- Real-time pacing and optional control latency
- Score = sum of per-game returns from the final cycle

### Prerequisites
- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
python -m AutoROM.accept-rom-license --accept-license
```

If you need simple on-screen rendering, ensure a display is available or use a virtual framebuffer.

### Run the benchmark

```bash
python scripts/run_benchmark.py \
  --agent random \
  --cycles 3 \
  --frames-per-game 400000 \
  --target-fps 60 \
  --sticky-prob 0.25 \
  --latency-frames 0 \
  --seed 0
```

Switch to the learning baseline (DQN):

```bash
python scripts/run_benchmark.py \
  --agent dqn \
  --cycles 3 \
  --frames-per-game 400000 \
  --sticky-prob 0.25 \
  --target-fps 60 \
  --latency-frames 0 \
  --seed 0 \
  --learning-rate 1e-4 \
  --buffer-capacity 200000 \
  --batch-size 32 \
  --learning-starts 50000
```

Use `--help` to see all tunable DQN hyperparameters (epsilon decay, target update frequency, etc.).

- Default games:
  - ALE/Atlantis-v5, ALE/BattleZone-v5, ALE/Centipede-v5, ALE/Defender-v5,
    ALE/Krull-v5, ALE/MsPacman-v5, ALE/Qbert-v5, ALE/UpNDown-v5

### Outputs
- Prints per-game returns and total (final cycle) to stdout
- Saves a JSON summary to `./results/results.json`

### Notes
- This harness is designed to evaluate learning under a fixed compute/data budget; it is not an evaluation-only suite for pre-trained frozen policies.
- The sticky action probability is applied via ALE (`repeat_action_probability`).
- Real-time pacing uses wall-clock timing to target the specified FPS; control latency delays applied actions by N frames.

### License
MIT
