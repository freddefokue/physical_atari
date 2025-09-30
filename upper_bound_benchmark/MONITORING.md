# Training Progress Monitoring Guide

This guide explains how to monitor your BBF agent training in real-time.

## Features Added

✅ **Enhanced Console Logging** - Detailed metrics printed every 1000 steps  
✅ **CSV Metrics Files** - All training data saved to CSV for analysis  
✅ **Monitoring Script** - Real-time dashboard for training progress  
✅ **Episode Tracking** - Individual episode returns and lengths

---

## Quick Start

### 1. Run Training (with logging enabled)

```bash
cd upper_bound_benchmark
python3 -u scripts/run_atari100k.py \
    --agent bbf \
    --game Atlantis \
    --seed 7779 \
    --replay-ratio 2 \
    --no-realtime \
    --output-dir results/atari100k/atlantis_test 2>&1 | tee training.log
```

### 2. Monitor Progress (in another terminal)

**Option A: Using the monitoring script (recommended)**
```bash
# One-time check
python scripts/monitor_training.py results/atari100k/atlantis_test

# Auto-refresh every 5 seconds
watch -n 5 python scripts/monitor_training.py results/atari100k/atlantis_test
```

**Option B: Tail the CSV files directly**
```bash
# Training metrics
tail -f results/atari100k/atlantis_test/training_metrics.csv

# Episode metrics
tail -f results/atari100k/atlantis_test/episode_metrics.csv
```

**Option C: Follow the console output**
```bash
tail -f training.log
```

---

## Output Files

Training creates these files in your output directory:

### 📊 `training_metrics.csv`
Logged every 1000 environment steps with:
- `env_steps` - Total environment steps
- `learn_steps` - Total gradient updates
- `td_loss` - Temporal difference loss
- `spr_loss` - Self-predictive representation loss
- `total_loss` - Combined loss
- `mean_q` - Average Q-value across actions
- `max_q` - Maximum Q-value
- `gamma` - Current discount factor (anneals from 0.97 to 0.997)
- `n_step` - Current n-step return horizon (anneals from 10 to 3)
- `epsilon` - Exploration rate (BBF uses 0)
- `time_elapsed` - Training time in seconds

### 📈 `episode_metrics.csv`
Logged at the end of each episode with:
- `env_steps` - Steps when episode ended
- `episode_num` - Episode number
- `episode_return` - Total episode reward
- `episode_length` - Episode duration in steps

### 📋 `results.json`
Final benchmark results (created at the end)

---

## Console Output Examples

### Training Progress (every 1000 steps)
```
[Step 5,000] Training Metrics:
  Loss: TD=0.3421, SPR=0.1234, Total=0.9591
  Q-values: Mean=12.34, Max=45.67
  Hyperparams: γ=0.9850, n-step=7, ε=0.000
  Training: Learn steps=10,000, Buffer size=5,000
```

### Episode Completion
```
  Episode finished: Return=125.0, Length=543, Steps=5,432
```

### Network Reset (every 20,000 learn steps)
```
Performing reset at step 20000
```

---

## Customization

### Change Log Frequency

Edit the `log_interval` in `scripts/run_atari100k.py`:

```python
bbf_config = BBFConfig(
    # ...
    log_interval=500,  # Log every 500 steps instead of 1000
    verbose=True,
    # ...
)
```

### Disable Console Logging (CSV only)

```python
bbf_config = BBFConfig(
    # ...
    verbose=False,  # No console output, only CSV
    # ...
)
```

### Disable All Logging

```python
bbf_config = BBFConfig(
    # ...
    output_dir=None,  # No CSV files
    verbose=False,    # No console output
    # ...
)
```

---

## Analysis Tips

### Plot Training Curves

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('results/atari100k/atlantis_test/training_metrics.csv')

# Plot losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df['env_steps'], df['td_loss'], label='TD Loss')
plt.plot(df['env_steps'], df['spr_loss'], label='SPR Loss')
plt.xlabel('Environment Steps')
plt.ylabel('Loss')
plt.legend()

# Plot Q-values
plt.subplot(1, 2, 2)
plt.plot(df['env_steps'], df['mean_q'], label='Mean Q')
plt.plot(df['env_steps'], df['max_q'], label='Max Q')
plt.xlabel('Environment Steps')
plt.ylabel('Q-value')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
```

### Calculate Learning Speed

```python
df = pd.read_csv('results/atari100k/atlantis_test/training_metrics.csv')
total_time = df['time_elapsed'].iloc[-1]
total_steps = df['env_steps'].iloc[-1]
print(f"Steps per second: {total_steps / total_time:.1f}")
```

### Episode Return Statistics

```python
df = pd.read_csv('results/atari100k/atlantis_test/episode_metrics.csv')
print(f"Mean return: {df['episode_return'].mean():.1f}")
print(f"Max return: {df['episode_return'].max():.1f}")
print(f"Std return: {df['episode_return'].std():.1f}")
```

---

## Troubleshooting

**Q: No CSV files are created**  
A: Make sure `output_dir` is set in BBFConfig and the directory exists.

**Q: Console is too verbose**  
A: Set `verbose=False` in BBFConfig or increase `log_interval`.

**Q: Monitoring script shows no data**  
A: Wait until at least 1000 steps have been completed for first metrics.

**Q: CSV files are empty**  
A: Training hasn't reached `learning_starts` (default: 2000 steps).

---

## What to Watch For

🟢 **Good Signs:**
- TD loss decreasing over time
- Mean Q-values increasing and stabilizing
- Episode returns trending upward
- SPR loss decreasing

🔴 **Warning Signs:**
- TD loss exploding (>10)
- Q-values diverging (>1000)
- Loss not decreasing after 20k steps
- Episodes all getting zero reward

---

## Performance Benchmarks

For Atlantis with BBF (RR=2):
- **Expected**: ~1173 average return (BBF paper, 50 seeds)
- **Training time**: ~30-60 min for 100k steps (GPU)
- **Steps/sec**: ~1500-3000 with GPU, ~100-300 with CPU

---

**Happy Training! 🚀**
