# SAC Sweep Plan

This document lays out a practical hyperparameter sweep plan for `sac` on the `physical_atari` continual multi-game benchmark.

The goal is to tune SAC for:

- continual multi-game learning
- no task ID
- sticky actions
- optional latency stress testing
- long online training
- revisits and forgetting

This is not a standard discrete SAC sweep for a stationary benchmark. The continual multi-game setting changes which knobs matter most and how aggressively they should be searched.

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

## Why SAC Needs A Tight, Stability-Focused Sweep

SAC should be treated as a narrow-to-medium sweep family in this benchmark.

Reasons:

1. It is much slower than PPO and slower than `delay_target`.
2. It is replay-heavy, so early runs can be deceptive.
3. The entropy-temperature mechanism can go unstable in long continual online learning.
4. It is especially sensitive to update intensity, target smoothing, and learning-start timing.

That means SAC should not be swept broadly. The right approach is:

- stability-focused
- staged
- warmup-aware
- not exhaustive

## SAC Config Surface

The benchmark-local SAC implementation exposes the following config:

```python
class SACAgentConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    feature_dim: int = 512
    actor_hidden_dim: int = 256
    value_hidden_dim: int = 256
    frame_skip: int = 4
    n_stack: int = 4
    obs_height: int = 128
    obs_width: int = 128
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    gradient_steps: int = 1
    train_freq: int = 1
    tau: float = 0.005
    target_entropy_scale: float = 0.5
    eval_mode: bool = False
    load_file: Optional[str] = None
    gpu: int = 0
```

## Current Benchmark Limitation

At the moment, the benchmark CLI only passes:

- `sac_gpu`
- `sac_load_file`
- `sac_eval_mode`

into `SACAgentConfig`.

So, just like with Rainbow and `delay_target`, there are two levels of answer:

1. the conceptual SAC sweep plan
2. what can actually be swept today through the benchmark

Today, a meaningful SAC sweep requires exposing more config fields through the benchmark interface.

## SAC Hyperparameter Priorities

### Tier 1: highest priority

These are the most important SAC knobs for this benchmark.

1. `learning_rate`
2. `learning_starts`
3. `batch_size`
4. `buffer_size`
5. `gradient_steps`
6. `tau`
7. `target_entropy_scale`

These define the first real SAC sweep.

### Why these matter

- `learning_rate`: main optimization and stability knob
- `learning_starts`: determines how quickly replay-driven training begins
- `batch_size`: stability / compute / replay interaction
- `buffer_size`: replay horizon and diversity
- `gradient_steps`: update intensity per training opportunity
- `tau`: target-network smoothing speed
- `target_entropy_scale`: exploration / entropy calibration

### Tier 2: second-pass knobs

These are worth opening only after a stable base regime has been found.

8. `gamma`
9. `train_freq`
10. `frame_skip`
11. `feature_dim`
12. `actor_hidden_dim`
13. `value_hidden_dim`

### Tier 3: keep fixed initially

These are meaningful but lower priority for the first pass.

- `n_stack`
- `obs_height`
- `obs_width`
- `eval_mode`
- GPU choice

## Recommended Core Sweep Space

This is the first SAC search space I would use once the benchmark exposes the needed knobs.

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `learning_rate` | `1e-4` | `3e-5, 1e-4, 3e-4` | Main optimization knob |
| `learning_starts` | `10_000` | `5_000, 10_000, 25_000, 50_000` | Replay warmup |
| `batch_size` | `64` | `32, 64, 128` | Stability vs throughput |
| `buffer_size` | `100_000` | `50_000, 100_000, 200_000, 400_000` | Replay horizon |
| `gradient_steps` | `1` | `1, 2, 4` | Update intensity |
| `tau` | `0.005` | `0.001, 0.003, 0.005, 0.01` | Target smoothing |
| `target_entropy_scale` | `0.5` | `0.25, 0.5, 0.75, 1.0` | Exploration calibration |

## Recommended Second-Pass Sweep Space

Once a good base SAC regime is found, open these:

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `gamma` | `0.99` | `0.97, 0.99, 0.995` | Effective planning horizon |
| `train_freq` | `1` | `1, 2, 4` | Update cadence |
| `frame_skip` | `4` | `2, 4, 6` | Agent-owned temporal abstraction |
| `feature_dim` | `512` | `256, 512, 768` | Shared representation width |
| `actor_hidden_dim` | `256` | `128, 256, 512` | Actor capacity |
| `value_hidden_dim` | `256` | `128, 256, 512` | Critic capacity |

## What To Keep Fixed Initially

I would keep these fixed in the first SAC sweep:

- `n_stack = 4`
- `obs_height = 128`
- `obs_width = 128`
- `eval_mode = 0`

Those are sensible defaults and not the first place to spend budget.

## Why `target_entropy_scale` Matters So Much Here

In this discrete SAC implementation, entropy is not just a small regularization term. It is an actively adapted part of the algorithm.

That makes `target_entropy_scale` especially important in the continual benchmark because it affects:

- how stochastic the agent remains over long training
- whether it becomes too inert or too noisy after revisits
- whether alpha runs into pathological regimes

This is one of the most benchmark-relevant SAC knobs and should be included in the first sweep.

## Suggested Staged Sweep Procedure

Use a narrow, stability-focused random search.

### Stage 0: quick filter

Benchmark:

- 3 games
- 2 cycles
- `150k` base visit frames

Plan:

- 8 to 12 configs from the core sweep space
- keep top 3 to 4

Goal:

- remove obviously unstable or undertrained SAC regimes quickly

### Stage 1: ranking

Benchmark:

- 3 games
- 2 cycles
- `400k` base visit frames

Plan:

- run the 3 to 4 survivors
- keep top 1 to 2

Goal:

- identify ranking-stable SAC regimes over a longer continual horizon

### Stage 2: refinement

Benchmark:

- 3 games
- 2 cycles
- `400k` base visit frames

Plan:

- locally perturb the top 1 to 2 configs
- open:
  - `gamma`
  - `train_freq`
  - `frame_skip`
  - optionally capacity knobs

Goal:

- improve temporal behavior and stabilize the best regime

### Stage 3: full-benchmark finalist check

Benchmark:

- 8 games
- 3 cycles
- `400k` base visit frames

Plan:

- run the 1 to 2 finalists

Goal:

- verify the SAC configuration generalizes from the subset to the full benchmark

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

## Suggested Trial Counts

SAC should be one of the smaller sweep families.

- Stage 0: 8 to 12 configs
- Stage 1: 3 to 4 configs
- Stage 2: 4 to 6 local variants
- Stage 3: 1 to 2 configs
- Stage 4: 1 config, 2 seeds if affordable

## Promotion Metric

Promote SAC configs using the benchmark’s continual-learning metrics, not only early episodic return.

Recommended ranking:

1. primary: `final_score`
2. tie-break: lower positive `forgetting_index_mean`
3. second tie-break: higher `plasticity_mean`

Also reject SAC configs with obvious optimization pathology even if they score tolerably.

## SAC Failure Modes To Watch For

### 1. Alpha blows up

The entropy temperature rises continuously and the policy remains poorly calibrated.

Likely causes:

- bad `target_entropy_scale`
- learning rate too high
- update intensity too aggressive

### 2. Replay-driven drift

The critics keep updating but returns do not improve or degrade after revisits.

Likely causes:

- too-small replay horizon
- poor `learning_starts`
- bad `gradient_steps` / `train_freq` regime

### 3. Too little plasticity

The agent remains very stable but adapts too slowly when revisiting or switching games.

Likely causes:

- learning rate too low
- tau too small
- target-entropy regime too conservative

### 4. Strong short-horizon learning, weak continual retention

The agent improves inside visits but does not preserve useful behavior across revisits.

Likely causes:

- insufficient replay diversity
- overly aggressive updates
- poor temporal abstraction regime

## What This Sweep Is

This sweep should be:

- narrow-to-medium
- stability-focused
- staged
- not exhaustive

That is intentional. SAC is too slow and too sensitive for a broad brute-force sweep to be efficient here.

## What I Would Not Sweep First

Do not start with:

- image size
- stack depth
- large architecture search
- eval-mode variations

Those belong later, if SAC already looks promising enough to justify more study.

## Practical Next Step For `physical_atari`

To make this sweep real in the benchmark, the runner should expose at least:

- `learning_rate`
- `learning_starts`
- `batch_size`
- `buffer_size`
- `gradient_steps`
- `tau`
- `target_entropy_scale`
- optionally `gamma`
- optionally `train_freq`
- optionally `frame_skip`

Without those, SAC cannot yet be meaningfully swept through the benchmark interface.

## Final Recommendation

For this benchmark, SAC should be treated as:

- one of the slower families
- a stability-sensitive replay learner
- a narrow sweep with strong pruning

The first SAC sweep should focus on:

- LR
- replay warmup
- replay capacity
- batch size
- gradient intensity
- target smoothing
- target entropy scale

That is the part of SAC most likely to determine whether it can become a competitive continual multi-game learner without wasting large amounts of wall-clock on unstable or uninformative runs.
