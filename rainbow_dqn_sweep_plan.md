# Rainbow DQN Sweep Plan

This document lays out a practical hyperparameter sweep plan for `rainbow_dqn` on the `physical_atari` continual multi-game benchmark.

The goal is to tune Rainbow DQN for:

- continual multi-game learning
- no task ID
- sticky actions
- optional latency stress testing
- long online training
- revisits and forgetting

This is not a standard Rainbow Atari sweep. The continual benchmark changes which knobs matter and how much budget should be spent on them.

## Benchmark Assumptions

Use these benchmark settings as fixed defaults unless explicitly studying benchmark semantics:

- `runner_mode: "carmack_compat"`
- `decision_interval: 1`
- `delay: 0`
- `jitter_pct: 0.0`
- `life_loss_termination: 0`
- `sticky: 0.25`
- `full_action_space: 1`
- `real_time_mode: 0`

Use `delay = 6` only as a secondary latency stress test for finalists, not as the primary sweep benchmark.
Use `jitter_pct > 0` only as a secondary schedule-robustness stress test, not as the primary sweep benchmark.
Use `life_loss_termination = 1` only as a secondary training-friendly variant, not as the primary sweep benchmark.

Use the 3-game subset for screening and ranking:

- `["centipede", "ms_pacman", "qbert"]`

Use the full 8-game benchmark only for finalists:

- `["ms_pacman", "centipede", "qbert", "defender", "krull", "atlantis", "up_n_down", "battle_zone"]`

## Why Rainbow DQN Needs A Narrow Sweep

Rainbow DQN should be one of the narrowest sweep families in this benchmark.

Reasons:

1. It is relatively slow in this setup.
2. It has a substantial warmup before training begins.
3. It is a highly coupled algorithm: replay, target updates, n-step returns, prioritized replay, noisy nets, and distributional value estimation all interact.

Because of that, a broad brute-force sweep is not efficient.

The right approach is:

- narrow
- staged
- strongly pruned
- warmup-aware

## Rainbow DQN Config Surface

The benchmark-local Rainbow DQN implementation exposes the following config:

```python
class RainbowDQNConfig:
    stack_size: int = 4
    obs_height: int = 84
    obs_width: int = 84
    buffer_size: int = 100_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99
    train_start: int = 50_000
    train_freq: int = 1
    target_update_freq: int = 2_000
    epsilon_start: float = 0.0
    epsilon_end: float = 0.0
    epsilon_decay_frames: int = 1_000_000
    grad_clip: Optional[float] = 10.0
    n_step: int = 3
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    priority_alpha: float = 0.5
    priority_beta: float = 0.4
    priority_beta_increment: float = 1e-6
    priority_eps: float = 1e-6
```

One important practical detail:

- `train_start = 50_000` by default

That means short runs can be very misleading because a lot of the budget can disappear into replay fill before meaningful learning begins.

## Current Benchmark Limitation

At the moment, the benchmark CLI only passes:

- `rainbow_dqn_gpu`
- `rainbow_dqn_load_file`

into `RainbowDQNConfig`.

So there are two levels of answer:

1. what the ideal Rainbow sweep should be conceptually
2. what can actually be swept today through the benchmark

Today, the answer is: a meaningful Rainbow sweep requires exposing more config knobs through the benchmark.

## Rainbow DQN Hyperparameter Priorities

### Tier 1: highest priority

These are the most important Rainbow knobs for this benchmark.

1. `learning_rate`
2. `train_start`
3. `batch_size`
4. `buffer_size`
5. `target_update_freq`
6. `n_step`

These should define the first real Rainbow sweep.

### Why these matter

- `learning_rate`: main optimization and stability knob
- `train_start`: critical because early budget is scarce and warmup is expensive
- `batch_size`: major stability / replay / compute interaction
- `buffer_size`: replay horizon and diversity
- `target_update_freq`: Q-target stability
- `n_step`: delayed credit assignment

### Tier 2: second-pass knobs

These are worth opening after a promising base regime has been found.

7. `gamma`
8. `grad_clip`
9. `priority_alpha`
10. `priority_beta`

### Tier 3: keep fixed initially

These are meaningful but lower priority for the first pass.

- `stack_size`
- `obs_height`
- `obs_width`
- `num_atoms`
- `v_min`
- `v_max`
- `epsilon_start`
- `epsilon_end`
- `epsilon_decay_frames`
- `priority_beta_increment`
- `priority_eps`

## Recommended Core Sweep Space

This is the first Rainbow search space I would use once the benchmark exposes the needed knobs.

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `learning_rate` | `1e-4` | `3e-5, 1e-4, 3e-4` | Main optimization knob |
| `train_start` | `50_000` | `10_000, 25_000, 50_000, 100_000` | Warmup and replay-fill threshold |
| `batch_size` | `32` | `32, 64` | Stability / compute balance |
| `buffer_size` | `100_000` | `50_000, 100_000, 200_000, 400_000` | Replay horizon |
| `target_update_freq` | `2_000` | `1_000, 2_000, 4_000, 8_000` | Target-network stability |
| `n_step` | `3` | `1, 3, 5` | Delayed credit assignment |

## Recommended Second-Pass Sweep Space

Once a good base Rainbow regime is found, open these:

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `gamma` | `0.99` | `0.97, 0.99, 0.995` | Effective planning horizon |
| `grad_clip` | `10.0` | `5.0, 10.0, None` | Optimization stability |
| `priority_alpha` | `0.5` | `0.4, 0.5, 0.6, 0.7` | Replay prioritization strength |
| `priority_beta` | `0.4` | `0.2, 0.4, 0.6` | Replay importance correction |

## What To Keep Fixed Initially

I would keep these fixed in the first Rainbow sweep:

- `stack_size = 4`
- `obs_height = 84`
- `obs_width = 84`
- `num_atoms = 51`
- `v_min = -10`
- `v_max = 10`
- `epsilon_start = 0.0`
- `epsilon_end = 0.0`
- `epsilon_decay_frames = 1_000_000`

### Why keep epsilon fixed

This implementation relies on noisy layers and defaults to zero epsilon exploration. I would not change that in the first sweep unless the goal is explicitly to study exploration design.

## Warmup-Aware Sweep Design

Rainbow’s warmup behavior is especially important in this benchmark.

Because `train_start` is large by default, short-budget runs can mostly measure:

- replay filling
- random early behavior
- weak or delayed learning signal

That means Rainbow needs a slightly larger screening budget than one might first expect.

## Suggested Staged Sweep Procedure

Use a narrow, staged random search with strong pruning.

### Stage 0: quick filter

Benchmark:

- 3 games
- 2 cycles
- `150k` base visit frames

Plan:

- 8 to 12 configs from the core sweep space
- keep top 3 to 4

Why `150k` instead of `100k`:

- the default warmup is large enough that very short runs can be misleading

Goal:

- remove obviously bad replay / warmup / LR regimes

### Stage 1: ranking

Benchmark:

- 3 games
- 2 cycles
- `400k` base visit frames

Plan:

- run the 3 to 4 survivors
- keep top 1 to 2

Goal:

- identify ranking-stable configurations over a longer continual-learning horizon

### Stage 2: refinement

Benchmark:

- 3 games
- 2 cycles
- `400k` base visit frames

Plan:

- locally perturb the top 1 to 2 configs
- open:
  - `gamma`
  - `grad_clip`
  - `priority_alpha`
  - `priority_beta`

Goal:

- improve stability and replay behavior around the best regime

### Stage 3: full-benchmark finalist check

Benchmark:

- 8 games
- 3 cycles
- `400k` base visit frames

Plan:

- run the 1 to 2 finalists

Goal:

- verify the Rainbow configuration generalizes from the subset to the full benchmark

### Stage 4: final confirmation

Benchmark:

- 8 games
- 3 cycles
- `1.0M` base visit frames

Plan:

- run the best config
- if affordable, use 2 seeds

Goal:

- final scientific comparison against the other agent families

I would be cautious about going straight to `1.5M` full-benchmark Rainbow runs unless it is clearly promising, because the wall-clock cost grows quickly.

## Suggested Trial Counts

Rainbow should be one of the smallest sweep families.

- Stage 0: 8 to 12 configs
- Stage 1: 3 to 4 configs
- Stage 2: 4 to 6 local variants
- Stage 3: 1 to 2 configs
- Stage 4: 1 config, 2 seeds if affordable

## Promotion Metric

Promote Rainbow configs using the benchmark’s continual-learning metrics, not just early episodic return.

Recommended ranking:

1. primary: `final_score`
2. tie-break: lower positive `forgetting_index_mean`
3. second tie-break: higher `plasticity_mean`

Also reject configs with weak post-warmup learning even if the overall score is not terrible.

## Failure Modes To Watch For

### 1. Warmup dominates the run

The config spends too much of the budget filling replay and never meaningfully enters a good learning regime.

Likely causes:

- `train_start` too high
- run budget too short

### 2. Q-value instability

Targets or value estimates drift badly.

Likely causes:

- bad `learning_rate`
- poor `target_update_freq`
- weak `grad_clip` regime

### 3. Good first-visit behavior but poor revisits

The agent learns something useful within a visit but does not recover well when returning to an old game.

Likely causes:

- replay horizon too short
- replay prioritization regime too narrow
- poor `gamma` / `n_step` setting

### 4. Flat post-warmup learning

The run clearly begins training, but scores do not move much after warmup.

Likely causes:

- learning rate too low
- train-start too conservative
- poor replay or target-update regime

## What This Sweep Is

This sweep should be:

- narrow
- warmup-aware
- strongly pruned
- not exhaustive

That is intentional. Rainbow is too expensive and too coupled for a wide brute-force sweep to be sensible in this benchmark.

## What I Would Not Sweep First

Do not start with:

- image size
- stack size
- distributional support range
- epsilon schedule
- noisy-net architecture details

Those belong later, if Rainbow already looks strong enough to justify additional study.

## Practical Next Step For `physical_atari`

To make this sweep real in the benchmark, the runner should expose at least:

- `learning_rate`
- `train_start`
- `batch_size`
- `buffer_size`
- `target_update_freq`
- `n_step`
- optionally `gamma`
- optionally `grad_clip`
- optionally `priority_alpha`
- optionally `priority_beta`

Without those, Rainbow cannot yet be meaningfully swept through the benchmark interface.

## Final Recommendation

For this benchmark, Rainbow DQN should be treated as:

- one of the slow, expensive families
- one of the narrowest sweeps
- highly sensitive to warmup budgeting

The first Rainbow sweep should focus on:

- LR
- replay warmup
- replay capacity
- batch size
- target update cadence
- n-step horizon

That is the part of Rainbow most likely to determine whether it can compete in the continual multi-game benchmark without wasting large amounts of wall-clock on uninformative runs.
