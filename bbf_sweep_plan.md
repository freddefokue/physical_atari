# BBF Sweep Plan

This document lays out a practical hyperparameter sweep plan for BBF on the `physical_atari` continual multi-game benchmark.

It is written for the benchmark setting we discussed:

- continual multi-game Atari
- no task ID
- sticky actions
- optional latency stress testing
- long online training
- revisits and forgetting as first-class outcomes

The goal is not only to maximize sample efficiency on a short-horizon benchmark, but to find BBF configurations that remain strong under sequential revisits and catastrophic-forgetting pressure.

## Benchmark Assumptions

Use these benchmark settings as fixed defaults unless explicitly running a benchmark-ablation study:

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

For screening and ranking runs, use the 3-game subset:

- `["centipede", "ms_pacman", "qbert"]`

For finalist evaluation, use the full 8-game benchmark:

- `["ms_pacman", "centipede", "qbert", "defender", "krull", "atlantis", "up_n_down", "battle_zone"]`

## Why BBF Needs Its Own Sweep Strategy

BBF is unusually sample efficient, but in this setup it is also expensive in wall-clock terms. That makes a naive broad sweep impractical.

The sweep strategy here is therefore:

1. Use a staged random search, not a full grid.
2. Keep the search space focused on the highest-value continual-learning knobs.
3. Promote only a small number of configs at each stage.
4. Delay architecture search until BBF already looks competitive.

## Full BBF Hyperparameter Space

The original BBF implementation exposes many knobs. The `BBFAgent` constructor includes:

```python
BBFAgent(
    noisy=False,
    dueling=True,
    double_dqn=True,
    distributional=True,
    data_augmentation=False,
    num_updates_per_train_step=1,
    network=...,
    num_atoms=51,
    vmax=10.0,
    vmin=None,
    jumps=0,
    spr_weight=0,
    batch_size=32,
    replay_ratio=64,
    batches_to_group=1,
    update_horizon=10,
    max_update_horizon=None,
    min_gamma=None,
    epsilon_fn=...,
    replay_scheme="uniform",
    replay_type="deterministic",
    reset_every=-1,
    no_resets_after=-1,
    reset_offset=1,
    encoder_warmup=0,
    head_warmup=0,
    learning_rate=0.0001,
    encoder_learning_rate=0.0001,
    reset_target=True,
    reset_head=True,
    reset_projection=True,
    reset_encoder=False,
    reset_noise=True,
    reset_priorities=False,
    reset_interval_scaling=None,
    shrink_perturb_keys="",
    perturb_factor=0.2,
    shrink_factor=0.8,
    target_update_tau=1.0,
    max_target_update_tau=None,
    cycle_steps=0,
    target_update_period=1,
    target_action_selection=False,
    ...
)
```

The default BBF gin recipe sets a strong baseline:

- `num_atoms = 51`
- `update_horizon = 3`
- `max_update_horizon = 10`
- `min_gamma = 0.97`
- `cycle_steps = 10000`
- `reset_every = 20000`
- `shrink_perturb_keys = "encoder,transition_model"`
- `shrink_factor = 0.5`
- `perturb_factor = 0.5`
- `no_resets_after = 100000`
- `replay_ratio = 64`
- `batches_to_group = 2`
- `batch_size = 32`
- `spr_weight = 5`
- `jumps = 5`
- `data_augmentation = True`
- `replay_scheme = "prioritized"`
- `learning_rate = 1e-4`
- `encoder_learning_rate = 1e-4`
- `target_update_tau = 0.005`
- `target_action_selection = True`

## Ranked BBF Knobs For This Continual Benchmark

The ranking below is specific to the continual multi-game benchmark, not standard Atari 100K.

### Tier 1: highest priority

These should be treated as the primary BBF sweep knobs.

1. `reset_every`
2. `no_resets_after`
3. `replay_ratio`
4. `learning_rate`
5. `encoder_learning_rate`
6. `batch_size`
7. `replay_scheme`
8. `spr_weight`
9. `jumps`
10. `shrink_perturb_keys`
11. `shrink_factor`
12. `perturb_factor`

### Tier 2: very important but more coupled

These matter, but they are more intertwined with scheduling and stability.

13. `update_horizon`
14. `max_update_horizon`
15. `min_gamma`
16. `cycle_steps`
17. `target_update_tau`
18. `learning_starts`
19. `batches_to_group`
20. `target_action_selection`

### Tier 3: medium priority

These are meaningful, but they are not the best first-pass sweep targets.

- `data_augmentation`
- `target_update_period`
- `num_updates_per_train_step`
- `reset_target`
- `reset_head`
- `reset_projection`
- `reset_encoder`
- `reset_priorities`
- `reset_noise`
- `reset_interval_scaling`
- `encoder_warmup`
- `head_warmup`
- `replay_type`

### Tier 4: architecture-level knobs

These are real hyperparameters, but they are closer to architecture search than basic HP tuning.

- `network.hidden_dim`
- `network.width_scale`
- `network.encoder_type`
- `ImpalaCNN.num_blocks`
- `num_atoms`
- `vmax`
- `vmin`

### Tier 5: low initial priority

These are better treated as ingredient ablations than first-pass sweep variables.

- `noisy`
- `dueling`
- `double_dqn`
- `distributional`
- `eval_noise`
- `use_target_network`
- `match_online_target_rngs`
- `target_eval_mode`
- `half_precision`

## Recommended First-Pass Sweep Space

This is the recommended high-value BBF sweep space for the continual benchmark.

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `learning_starts` | `2000` | `1000, 2000, 5000, 10000` | Lower can improve early adaptation, higher can stabilize replay |
| `replay_ratio` | `64` | `16, 32, 64, 96, 128` | Major sample-efficiency and compute lever |
| `batch_size` | `32` | `16, 32, 64` | Strong interaction with replay ratio |
| `learning_rate` | `1e-4` | `3e-5, 1e-4, 3e-4` | Global optimizer scale |
| `encoder_learning_rate` | `1e-4` | `0.25x, 0.5x, 1x, 2x` of `learning_rate` | Prefer sampling as a ratio |
| `reset_every` | `20000` | `10000, 20000, 40000, 80000` | Core continual-learning knob |
| `no_resets_after` | `100000` | `2x, 5x, 10x` `reset_every`, plus effectively disabled | Sample as a multiple of `reset_every` |
| `spr_weight` | `5` | `1, 3, 5, 7.5, 10` | Representation pressure |
| `jumps` | `5` | `3, 5, 7` | Temporal depth of SPR |
| `target_update_tau` | `0.005` | `0.001, 0.003, 0.005, 0.01` | Stability and tracking speed |
| `update_horizon` | `3` | `3, 5, 7` | n-step credit assignment |
| `max_update_horizon` | `10` | `same as update_horizon`, `7`, `10` | Only meaningful if annealing is enabled |
| `min_gamma` | `0.97` | `0.95, 0.97, 0.985, 0.99` | Effective planning horizon |
| `cycle_steps` | `10000` | `5000, 10000, 20000, 40000` | Schedule/annealing timescale |

## Reset / Shrink-Perturb Sub-Sweep

Once a decent core config has been found, open the reset-mechanics subspace.

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `shrink_perturb_keys` | `"encoder,transition_model"` | `"encoder"`, `"encoder,transition_model"`, `"encoder,transition_model,head"` | High-value continual-learning ablation |
| `shrink_factor` | `0.5` | `0.2, 0.35, 0.5, 0.7, 0.9` | More retention vs less reset |
| `perturb_factor` | `0.5` | `0.1, 0.25, 0.5, 0.75` | More injected plasticity |
| `reset_target` | `True` | `True, False` | Useful small ablation |
| `reset_projection` | `True` | `True, False` | Same |
| `reset_encoder` | `False` | `False, True` | Risky but very informative |

Do not open this subspace too early. It creates a large branching factor and should only be explored around already promising base configs.

## Structured Sampling Rules

Do not sample all parameters independently.

### Rule 1: sample `encoder_learning_rate` as a ratio

Sample:

- `encoder_lr_ratio ∈ {0.25, 0.5, 1.0, 2.0}`

Then set:

- `encoder_learning_rate = encoder_lr_ratio * learning_rate`

### Rule 2: sample `no_resets_after` as a multiple

Sample:

- `2 * reset_every`
- `5 * reset_every`
- `10 * reset_every`
- a very large value for "effectively no cutoff"

This is better than sweeping it as an independent absolute value.

### Rule 3: sample `max_update_horizon` conditionally

With moderate probability:

- set `max_update_horizon = update_horizon`

Otherwise:

- sample from `{7, 10}` subject to `max_update_horizon >= update_horizon`

This separates "no annealing" from "annealed horizon."

## Staged Sweep Procedure

Use a staged random search with strong pruning.

### Stage 0: cheap screening

Benchmark:

- 3 games
- 2 cycles
- `100k` base visit frames

Plan:

- 12 random configs from the core sweep space
- keep top 4

Goal:

- remove broken or obviously weak configs quickly

### Stage 1: ranking

Benchmark:

- 3 games
- 2 cycles
- `400k` base visit frames

Plan:

- evaluate the 4 promoted configs
- add 4 local mutations around the best 2
- keep top 2

Goal:

- find configs that are robust enough to survive longer continual exposure

### Stage 2: schedule / horizon refinement

Benchmark:

- 3 games
- 2 cycles
- `400k` base visit frames

Plan:

- 4 to 6 configs varying:
  - `update_horizon`
  - `max_update_horizon`
  - `min_gamma`
  - `cycle_steps`
- keep top 1 to 2

Goal:

- refine temporal credit-assignment and annealing behavior

### Stage 3: reset-mechanics refinement

Benchmark:

- 3 games
- 2 cycles
- `400k` base visit frames

Plan:

- 4 to 6 configs varying:
  - `shrink_perturb_keys`
  - `shrink_factor`
  - `perturb_factor`
  - optionally `reset_encoder`
- keep best 1

Goal:

- optimize the plasticity/retention tradeoff under continual revisits

### Stage 4: full-benchmark finalist check

Benchmark:

- 8 games
- 3 cycles
- `400k` or `500k` base visit frames

Plan:

- run the 1 to 2 finalists

Goal:

- verify the config generalizes from the 3-game subset to the full benchmark

### Stage 5: final confirmation

Benchmark:

- 8 games
- 3 cycles
- `1.0M` base visit frames

Plan:

- run the best config
- if affordable, use 2 to 3 seeds

Goal:

- final scientific comparison against the other agent families

## Suggested Trial Counts

Because BBF is expensive, keep the sweep narrow.

- Stage 0: 12 configs
- Stage 1: 8 configs
- Stage 2: 4 to 6 configs
- Stage 3: 4 to 6 configs
- Stage 4: 1 to 2 configs
- Stage 5: 1 config, multiple seeds if affordable

## Promotion Metric

Promote configs using the same continual-learning objectives that matter for the final benchmark.

Recommended ranking:

1. primary: `final_score`
2. tie-break: lower positive `forgetting_index_mean`
3. second tie-break: higher `plasticity_mean`

In practice, also reject configs with obvious instability even if they have a decent score.

## What To Keep Fixed Initially

I would keep these fixed during the first sweep:

- `noisy = False`
- `dueling = True`
- `double_dqn = True`
- `distributional = True`
- `data_augmentation = True`
- `replay_type = "deterministic"`
- `num_atoms = 51`
- `target_action_selection = True`
- current Impala-style architecture

These are part of the standard BBF recipe and should not be the first things opened up.

## What Not To Sweep Yet

Do not start with:

- architecture width / depth
- encoder type
- value support parameters
- Rainbow ingredient toggles

Those belong in later ablations, not the first continual-learning sweep.

## Practical Note For `physical_atari`

The current `physical_atari` benchmark only exposes a subset of BBF knobs through its adapter. If you want to execute the full sweep plan directly through the benchmark CLI, more BBF parameters will need to be surfaced.

The currently exposed adapter knobs are roughly:

- `learning_starts`
- `buffer_size`
- `batch_size`
- `replay_ratio`
- `reset_interval`
- `no_resets_after`
- `use_per`
- `use_amp`
- `torch_compile`

To fully support the sweep described in this document, the benchmark should eventually expose at least:

- `learning_rate`
- `encoder_learning_rate`
- `spr_weight`
- `jumps`
- `target_update_tau`
- `update_horizon`
- `max_update_horizon`
- `min_gamma`
- `cycle_steps`
- `shrink_factor`
- `perturb_factor`
- `shrink_perturb_keys`

## Final Recommendation

For this benchmark, the most important BBF questions are not just:

- "How sample efficient is it?"

but also:

- "Does reset-based plasticity actually help revisiting old games?"
- "How much reset is too much?"
- "Can the encoder retain useful structure while remaining plastic?"

That is why the sweep should focus heavily on:

- reset cadence
- reset cutoff
- replay intensity
- representation loss strength
- shrink-and-perturb behavior

Those are the knobs most likely to determine whether BBF is merely fast to relearn or genuinely strong at continual multi-game learning.
