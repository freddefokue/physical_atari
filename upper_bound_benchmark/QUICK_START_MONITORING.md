# 🚀 Quick Start: Training Monitoring

## 1️⃣ Start Training

```bash
cd /workspace/physical_atari/upper_bound_benchmark

python3 -u scripts/run_atari100k.py \
    --agent bbf \
    --game Atlantis \
    --seed 7779 \
    --replay-ratio 2 \
    --no-realtime \
    --output-dir results/atari100k/atlantis_test 2>&1 | tee training.log
```

**What to expect:**
- Initial setup output
- Progress every 10,000 frames (from benchmark.py)
- **NEW**: Detailed metrics every 1,000 steps (from BBF agent)
- **NEW**: Episode completions with return and length

---

## 2️⃣ Monitor Progress (Open Another Terminal)

### Option A: Interactive Dashboard (Best)
```bash
cd /workspace/physical_atari/upper_bound_benchmark

# One-time snapshot
python scripts/monitor_training.py results/atari100k/atlantis_test

# Auto-refresh every 5 seconds
watch -n 5 python scripts/monitor_training.py results/atari100k/atlantis_test
```

### Option B: Tail CSV Files
```bash
# Training metrics
tail -f results/atari100k/atlantis_test/training_metrics.csv

# Episode metrics  
tail -f results/atari100k/atlantis_test/episode_metrics.csv
```

### Option C: Follow Console Log
```bash
tail -f training.log | grep -E "Step|Episode|Loss"
```

---

## 3️⃣ What You'll See

### Console Output (Every 1000 Steps)
```
[Step 5,000] Training Metrics:
  Loss: TD=0.3421, SPR=0.1234, Total=0.9591
  Q-values: Mean=12.34, Max=45.67
  Hyperparams: γ=0.9850, n-step=7, ε=0.000
  Training: Learn steps=10,000, Buffer size=5,000

  Episode finished: Return=125.0, Length=543, Steps=5,123
```

### Monitoring Dashboard
```
================================================================================
Training Progress Monitor: results/atari100k/atlantis_test
================================================================================

📊 TRAINING METRICS (Last 10 updates):
--------------------------------------------------------------------------------

🎯 Latest Update (Step 5,000):
  Training Steps:  10,000
  TD Loss:         0.3421
  SPR Loss:        0.1234
  Total Loss:      0.9591
  Mean Q-value:    12.34
  Max Q-value:     45.67
  Gamma (γ):       0.9850
  N-step:          7
  Time Elapsed:    125.3s

  Loss Trend:      📉 Decreasing (0.3856 → 0.3421)

  Step History (last 5 updates):
  Step       TD Loss    SPR Loss   Mean Q     Gamma   
  3,000      0.3856     0.1298     11.23      0.9775   
  4,000      0.3645     0.1256     11.89      0.9812   
  5,000      0.3421     0.1234     12.34      0.9850   

================================================================================
📈 EPISODE METRICS (Last 10 episodes):
--------------------------------------------------------------------------------

🎮 Latest Episode (#12):
  Return:       125.0
  Length:       543 steps
  Total Steps:  5,432

  Last 5 Episodes:
    Average Return: 118.4
    Best Return:    145.0
    Worst Return:   95.0

  Episode History (last 5 episodes):
  Ep#    Step       Return       Length  
  8      3,654      95.0         412     
  9      4,012      105.0        358     
  10     4,523      132.0        511     
  11     4,987      145.0        464     
  12     5,432      125.0        445     
```

---

## 4️⃣ Quick Analysis

### Check if training is working:
```bash
# Should see decreasing loss
tail -20 results/atari100k/atlantis_test/training_metrics.csv | cut -d',' -f5

# Should see some positive returns
tail -20 results/atari100k/atlantis_test/episode_metrics.csv | cut -d',' -f3
```

### Plot training curves (optional):
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/atari100k/atlantis_test/training_metrics.csv')
df.plot(x='env_steps', y=['td_loss', 'spr_loss'], figsize=(10, 4))
plt.savefig('losses.png')

df.plot(x='env_steps', y=['mean_q', 'max_q'], figsize=(10, 4))
plt.savefig('q_values.png')
```

---

## 5️⃣ Adjust Logging (Optional)

### More frequent logging (every 500 steps):
Edit `scripts/run_atari100k.py`, change:
```python
log_interval=500,  # instead of 1000
```

### Less verbose (CSV only, no console spam):
```python
verbose=False,
```

---

## ⚡ Pro Tips

1. **Use `tmux` or `screen`** to run training in one pane and monitoring in another
2. **Redirect output** to avoid log file bloat: `... 2>&1 | tee training.log`
3. **Check metrics before full run** with a small test (--game Pong, 10K steps)
4. **Watch for anomalies**: Loss >10, Q-values >1000, all zero returns
5. **Compare to paper**: Atlantis should reach ~1173 avg return with RR=8

---

## 📋 Files Created

```
results/atari100k/atlantis_test/
├── training_metrics.csv    ← Training step data
├── episode_metrics.csv     ← Episode completion data
└── results.json           ← Final benchmark results (end of run)
```

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| No CSV files | Wait for 1000 steps to complete |
| No episode data | Wait for first episode to finish (~200-1000 steps) |
| monitor_training.py error | Check output_dir path is correct |
| Too much console output | Set `verbose=False` in config |

---

**You're all set! 🎉**

Training will now show detailed progress every 1000 steps, and you can monitor from another terminal!
