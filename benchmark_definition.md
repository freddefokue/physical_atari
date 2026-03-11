# Primary Benchmark Definition

This document defines the primary continual-learning benchmark to use for hyperparameter sweeps and final agent comparisons in `physical_atari`.

The purpose of this benchmark is to measure:

- sequential multi-task online learning
- retention across revisits
- plasticity and transfer
- re-adaptation without task IDs

The benchmark is intentionally designed to isolate continual-learning ability as cleanly as possible, while still avoiding overly gameable deterministic Atari settings.

## Primary Benchmark

### Runner / environment semantics

- `runner_mode = "carmack_compat"`
- `decision_interval = 1`
- `delay = 0`
- `jitter_pct = 0.0`
- `sticky = 0.25`
- `full_action_space = 1`
- `life_loss_termination = 0`
- `real_time_mode = 0`

### Why these settings are locked

#### `runner_mode = "carmack_compat"`

This keeps the streaming, post-step control-loop semantics that best match the project’s intended continual-learning setting.

#### `decision_interval = 1`

This is required by `carmack_compat` mode and keeps the runner itself from imposing extra action cadence. Agent-owned cadence remains possible for agent families that implement it internally.

#### `delay = 0`

The primary benchmark should isolate continual learning rather than combine it with delayed-actuation robustness.

`delay = 6` remains useful as a secondary stress benchmark, but it is not part of the primary benchmark.

#### `jitter_pct = 0.0`

The primary benchmark uses fixed visit lengths to reduce unnecessary noise and make transfer / forgetting easier to interpret.

Nonzero jitter remains useful as a secondary schedule-robustness stress test, but it is not part of the primary benchmark.

#### `sticky = 0.25`

Sticky actions remain enabled in the primary benchmark because they help prevent brittle exploitation of deterministic Atari timing and make the task more robust without fundamentally changing the continual-learning question.

`sticky = 0.0` is reserved for secondary determinism ablations.

#### `full_action_space = 1`

The primary benchmark uses the full action space instead of a reduced minimal action set. This avoids giving the agent a benchmark-side simplification and better matches the project’s goal of evaluating more general continual competence.

`full_action_space = 0` is reserved for secondary ablations only.

#### `life_loss_termination = 0`

The benchmark should reflect a continuous stream rather than injecting extra artificial boundaries on life loss.

`life_loss_termination = 1` is reserved for secondary training-friendly variants, not for the primary benchmark.

#### `real_time_mode = 0`

The primary benchmark compares agents at matched environment frames, not matched wall-clock speed.

`real_time_mode = 1` is a valuable secondary realism and systems benchmark, but not part of the primary benchmark definition.

## Game Sets

### Screening / ranking subset

Use the 3-game subset:

- `centipede`
- `ms_pacman`
- `qbert`

This subset is intended for early-stage sweeps and ranking because it is meaningfully diverse while still being far cheaper than the full suite.

### Full benchmark suite

Use the 8-game suite:

- `ms_pacman`
- `centipede`
- `qbert`
- `defender`
- `krull`
- `atlantis`
- `up_n_down`
- `battle_zone`

This is the final benchmark suite for cross-family comparison and final confirmation runs.

## Schedule Definition

### Screening / ranking runs

- `num_cycles = 2`

The subset benchmark should include revisits, so one cycle is not enough. Two cycles is the minimum useful continual setting for screening and ranking.

### Full benchmark runs

- `num_cycles = 3`

The full benchmark should use three cycles to allow meaningful plasticity, forgetting, and late-cycle recovery analysis.

## Visit-Length Definition

The benchmark semantics are fixed across all sweep stages, but the frame budget per visit can differ by stage.

### Full benchmark confirmation budget

The locked default for final confirmation is:

- `base_visit_frames = 1_000_000`

This is the primary full-benchmark confirmation budget.

### Why `1_000_000`

This budget is large enough to be meaningful for continual-learning comparison while remaining more tractable than `1_500_000`, especially for slow families such as BBF and Rainbow DQN.

### Optional larger confirm budget

- `base_visit_frames = 1_500_000`

This may still be used for especially strong finalists or extra-confirmation studies, but it is not the default primary full-benchmark budget.

## Scoring Defaults

The benchmark uses explicit frame-based scoring defaults:

- `window_frames = 5000`
- `revisit_frames = 2000`
- `bottom_k_frac = 0.25`
- `final_score_weights = [0.5, 0.5]`

### Why frame-based scoring

The current scorer operates naturally in frame space. Using explicit frame-based defaults avoids ambiguity and makes the benchmark definition clearer than older `window_episodes` / `revisit_episodes` style settings.

## Secondary / Optional Benchmarks

The following are useful but are **not** part of the primary benchmark definition.

### Latency stress test

- same primary benchmark, but `delay = 6`

Purpose:

- test robustness to delayed actuation

### Schedule-robustness stress test

- same primary benchmark, but `jitter_pct > 0`

Purpose:

- test whether agents rely too heavily on fixed visit durations

### Training-friendly life-loss variant

- same primary benchmark, but `life_loss_termination = 1`

Purpose:

- measure how much an agent benefits from extra boundary structure

### Determinism ablation

- same primary benchmark, but `sticky = 0.0`

Purpose:

- test whether performance depends on deterministic exploitation

### Minimal-action-space ablation

- same primary benchmark, but `full_action_space = 0`

Purpose:

- measure how much of performance depends on benchmark-side simplification

### Wall-clock / real-time benchmark

- same primary benchmark, but `real_time_mode = 1`

Purpose:

- evaluate actual real-time usability and implementation speed

## Summary

The primary benchmark is therefore:

- `runner_mode = "carmack_compat"`
- `decision_interval = 1`
- `delay = 0`
- `jitter_pct = 0.0`
- `sticky = 0.25`
- `full_action_space = 1`
- `life_loss_termination = 0`
- `real_time_mode = 0`
- screening subset: `["centipede", "ms_pacman", "qbert"]`
- full suite: `["ms_pacman", "centipede", "qbert", "defender", "krull", "atlantis", "up_n_down", "battle_zone"]`
- screening / ranking: `2` cycles
- full benchmark: `3` cycles
- full benchmark default confirmation budget: `1_000_000` base visit frames
- scoring defaults:
  - `window_frames = 5000`
  - `revisit_frames = 2000`
  - `bottom_k_frac = 0.25`
  - `final_score_weights = [0.5, 0.5]`

This is the benchmark definition that should be used as the canonical default for the hyperparameter sweep program and the main leaderboard.
