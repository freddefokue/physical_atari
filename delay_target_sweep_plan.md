# Delay Target Sweep Plan

This document lays out a practical hyperparameter sweep plan for the `delay_target` agent on the `physical_atari` continual multi-game benchmark.

The intent is to tune `delay_target` for:

- continual multi-game learning
- no task ID
- sticky actions
- optional latency stress testing
- long online training
- revisits and forgetting

This is not a generic Atari sweep. The `delay_target` agent is highly benchmark-specific and already encodes a number of design assumptions about streaming delayed control.

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

## Why `delay_target` Needs A Benchmark-Specific Sweep

The `delay_target` agent is unusually tied to the assumptions of this benchmark:

- frame-level streaming
- delayed action effects
- repeated action cadence
- long stacked observation history
- replay over a ring buffer
- recent-sample emphasis during training

That means the sweep should not look like a generic PPO or DQN sweep. The most important knobs are the ones that control:

- learning-rate split between representation and output head
- replay horizon / memory
- temporal credit assignment
- model capacity
- exploration softness

## Core `delay_target` Defaults

The root agent has the following important defaults:

```python
class Agent:
    def __init__(...):
        self.reward_discount = 0.9975
        self.multisteps_max = 64
        self.td_lambda = 0.95

        self.frame_skip = 4
        self.input_stack = 16
        self.obs_width = 128
        self.obs_height = 128
        self.obs_channels = 3

        self.greedy_max = 0.99
        self.greedy_ramp = 100_000
        self.temperature_log2 = -7

        self.base_width = 80
        self.use_model = 3

        self.use_softv = 1
        self.use_weight_norm = 1
        self.repeat_train = 1
        self.min_train_frames = 256

        self.base_lr_log2 = -16
        self.lr_log2 = -18

        self.train_batch = 32
        self.online_batch = 4
        self.online_loss_scale = 2
        self.train_steps = 4

        self.ring_buffer_size = 16_384
```

These defaults already define a fairly opinionated agent. The sweep should therefore be selective, not exhaustive.

## Practical Constraint In `physical_atari`

The current benchmark adapter does not expose the full `delay_target` parameter space.

At the moment, the benchmark CLI exposes mainly:

- `delay_target_gpu`
- `delay_target_use_cuda_graphs`
- `delay_target_load_file`
- `delay_target_ring_buffer_size`
- `delay_target_lr_log2`
- `delay_target_base_lr_log2`

So there are really two sweep plans:

1. a practical sweep using only what the benchmark currently exposes
2. a fuller conceptual sweep for when more knobs are exposed

## Delay Target Hyperparameter Priorities

### Tier 1: highest priority

These are the most important `delay_target` knobs for this benchmark.

1. `base_lr_log2`
2. `lr_log2`
3. `ring_buffer_size`
4. `base_width`
5. `multisteps_max`
6. `td_lambda`
7. `temperature_log2`
8. `greedy_ramp`

### Why these matter

- `base_lr_log2`: controls how fast the representation changes
- `lr_log2`: controls how fast the output head changes
- `ring_buffer_size`: controls replay horizon and training diversity
- `base_width`: controls main model capacity
- `multisteps_max`: controls the temporal target horizon
- `td_lambda`: controls the reward/value blending over that horizon
- `temperature_log2`: controls action softness in the policy
- `greedy_ramp`: controls how quickly the agent becomes policy-driven rather than random-action driven

### Tier 2: important, but more coupled

These are worth opening later.

9. `train_batch`
10. `online_batch`
11. `online_loss_scale`
12. `train_steps`
13. `reward_discount`
14. `use_model`
15. `obs_width`
16. `obs_height`

### Tier 3: keep fixed initially

These are meaningful, but lower priority for the first sweep.

- `frame_skip`
- `input_stack`
- `obs_channels`
- `use_softv`
- `use_weight_norm`
- `repeat_train`
- `min_train_frames`
- `kernel_size`
- `use_dirac`
- `use_biases`
- `weight_decay`
- `beta1`
- `beta2`
- `momentum`

## Current Benchmark-Exposed Sweep

This is the first sweep I would actually run today using the existing benchmark interface.

### Sweep these knobs

- `delay_target_lr_log2`
- `delay_target_base_lr_log2`
- `delay_target_ring_buffer_size`

### Recommended values

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `delay_target_lr_log2` | `-18` | `-19, -18, -17, -16` | Output-head learning rate |
| `delay_target_base_lr_log2` | `-16` | `-17, -16, -15, -14` | Backbone learning rate |
| `delay_target_ring_buffer_size` | `16384` | `8192, 16384, 32768, 65536` | Replay horizon and memory |

### Why this is a good first sweep

These three knobs already control a lot:

- short-term adaptation speed
- long-term representation drift
- amount of replay diversity and history

That makes them a very strong first-pass sweep, even though it is not the full conceptual space.

## Full Conceptual `delay_target` Sweep

If more knobs are exposed in the benchmark later, this is the full sweep structure I would recommend.

## Stage A: core sweep

Sweep:

- `lr_log2`
- `base_lr_log2`
- `ring_buffer_size`
- `base_width`
- `temperature_log2`
- `greedy_ramp`

These are the main levers I would use in the first expanded sweep.

### Recommended ranges

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `lr_log2` | `-18` | `-19, -18, -17, -16` | Head adaptation speed |
| `base_lr_log2` | `-16` | `-17, -16, -15, -14` | Backbone adaptation speed |
| `ring_buffer_size` | `16384` | `8192, 16384, 32768, 65536` | Replay horizon |
| `base_width` | `80` | `48, 64, 80, 96, 128` | Model capacity |
| `temperature_log2` | `-7` | `-9, -8, -7, -6, -5` | Softmax policy temperature |
| `greedy_ramp` | `100000` | `25000, 50000, 100000, 200000, 400000` | Exploration schedule |

## Stage B: temporal credit sweep

Open:

- `multisteps_max`
- `td_lambda`
- optionally `reward_discount`

### Recommended ranges

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `multisteps_max` | `64` | `16, 32, 64, 96, 128` | Temporal horizon |
| `td_lambda` | `0.95` | `0.8, 0.9, 0.95, 0.975, 1.0` | Reward/value blending |
| `reward_discount` | `0.9975` | `0.995, 0.9975, 0.999` | Discount per 60 fps frame |

## Stage C: online-vs-replay balance

Open:

- `train_batch`
- `online_batch`
- `online_loss_scale`
- `train_steps`

### Recommended ranges

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `train_batch` | `32` | `16, 32, 64` | Total train batch |
| `online_batch` | `4` | `2, 4, 8` | Recent samples forced into the batch |
| `online_loss_scale` | `2` | `1, 2, 4` | Relative emphasis on recent online samples |
| `train_steps` | `4` | `4, 8, 16` | Training cadence |

### Constraints

- `online_batch >= 1`
- `online_batch <= train_batch / 2`

## Stage D: architecture / representation ablation

Only open this stage if the earlier sweeps show the agent is already competitive.

Consider:

- `use_model`
- `kernel_size`
- `obs_width`
- `obs_height`
- `input_stack`

These are not first-pass sweep knobs.

## Suggested Staged Sweep Procedure

Use a medium-width, staged random search.

### Stage 0: quick filter

Benchmark:

- 3 games
- 2 cycles
- `200k` base visit frames

Plan:

- 16 to 24 configs
- with the current benchmark interface, sweep:
  - `delay_target_lr_log2`
  - `delay_target_base_lr_log2`
  - `delay_target_ring_buffer_size`
- keep top 6 to 8

Goal:

- identify viable learning-rate / replay-horizon regimes quickly

### Stage 1: ranking

Benchmark:

- 3 games
- 2 cycles
- `500k` base visit frames

Plan:

- run the 6 to 8 survivors
- keep top 2 to 3

Goal:

- identify ranking-stable configurations over a longer continual-learning horizon

### Stage 2: refined sweep

If additional knobs are benchmark-exposed, open:

- `base_width`
- `temperature_log2`
- `greedy_ramp`
- `multisteps_max`
- `td_lambda`

Run:

- 6 to 10 local configs around the best Stage 1 region

Goal:

- improve temporal credit assignment, exploration softness, and model capacity

### Stage 3: full-benchmark finalist check

Benchmark:

- 8 games
- 3 cycles
- `500k` base visit frames

Plan:

- run the 1 to 2 finalists

Goal:

- verify the chosen config generalizes from the subset to the full benchmark

### Stage 4: final confirmation

Benchmark:

- 8 games
- 3 cycles
- `1.0M` to `1.5M` base visit frames

Plan:

- run the best config
- use 2 to 3 seeds if affordable

Goal:

- final scientific comparison against the other agent families

## Suggested Trial Counts

Compared to BBF, `delay_target` is cheaper and can support a broader sweep. Compared to PPO, it is still specialized enough that the sweep should remain more focused.

- Stage 0: 16 to 24 configs
- Stage 1: 6 to 8 configs
- Stage 2: 6 to 10 configs if more knobs are exposed
- Stage 3: 1 to 2 configs
- Stage 4: 1 config, multiple seeds if affordable

## Promotion Metric

Promote configs using the benchmark’s continual-learning metrics, not only short-horizon return.

Recommended ranking:

1. primary: `final_score`
2. tie-break: lower positive `forgetting_index_mean`
3. second tie-break: higher `plasticity_mean`

Also reject configs with obvious training pathologies even if they score reasonably.

## Failure Modes To Watch For

### 1. Fast early learning but poor revisit retention

This often points to:

- too-high backbone learning rate
- too-small replay horizon
- overly aggressive short-horizon training

### 2. Stable but under-adaptive policy

This often points to:

- too-low head learning rate
- too-low exploration softness
- too-long greedy ramp

### 3. High short-term plasticity but unstable value targets

This often points to:

- poor `multisteps_max` / `td_lambda` regime
- too-aggressive LR split

### 4. Good benchmark score on subset, poor generalization to full benchmark

This can indicate:

- insufficient replay diversity
- overfit capacity regime
- poor temporal-credit setting

## What This Sweep Is

This sweep should be:

- medium-width
- staged
- benchmark-aware
- not exhaustive

That is intentional. The `delay_target` parameter space is too coupled for an exhaustive search to be efficient.

## What I Would Not Sweep First

Do not start with:

- `frame_skip`
- `input_stack`
- `obs resolution`
- `kernel_size`
- momentum / Adam betas
- weight norm or Dirac toggles

Those may matter, but they are not the highest-yield first-pass knobs for this benchmark.

## Final Recommendation

For this benchmark, the first `delay_target` sweep should be tightly focused on:

- head LR
- backbone LR
- replay horizon

If more knobs are exposed later, the next most valuable additions are:

- model width
- exploration temperature
- greedy ramp
- multistep horizon
- TD-lambda

That is the part of the `delay_target` space most likely to determine whether it is merely strong at online adaptation or genuinely strong at continual multi-game learning.
