# Benchmark Contract v1

This document freezes the v1 run/scoring contract for continual multi-game streaming Atari benchmark runs.

## Artifact Set

A v1 run directory contains:
- `config.json`
- `events.jsonl`
- `episodes.jsonl`
- `segments.jsonl`
- `score.json` (after scoring)

## Required Fields

### `events.jsonl` (one row per frame)
Required keys:
- `global_frame_idx` (int)
- `game_id` (str)
- `visit_idx` (int)
- `cycle_idx` (int)
- `visit_frame_idx` (int)
- `episode_id` (int)
- `segment_id` (int)
- `is_decision_frame` (bool)
- `decided_action_idx` (int)
- `applied_action_idx` (int)
- `reward` (number)
- `terminated` (bool)
- `truncated` (bool)

### `episodes.jsonl` (one row per episode boundary)
Required keys:
- `game_id` (str)
- `episode_id` (int)
- `start_global_frame_idx` (int)
- `end_global_frame_idx` (int)
- `length` (int)
- `return` (number)
- `ended_by` (`"terminated"` or `"truncated"`)

### `segments.jsonl` (one row per reset boundary)
Required keys:
- `game_id` (str)
- `segment_id` (int)
- `start_global_frame_idx` (int)
- `end_global_frame_idx` (int)
- `length` (int)
- `return` (number)
- `ended_by` (`"terminated"` or `"truncated"`)

### `config.json`
Required keys:
- `games` (ordered list[str])
- `schedule` (ordered list of `{visit_idx, cycle_idx, game_id, visit_frames}`)
- `decision_interval` (int)
- `delay` or `runner_config.delay_frames` (int)
- `sticky` (number)
- `life_loss_termination` (bool)
- `full_action_space` (bool)
- `action_mapping_policy.global_action_set` (list[int])
- `default_action_idx` (int)
- `scoring_defaults.window_frames` (int)
- `scoring_defaults.bottom_k_frac` (number in `(0,1]`)
- `scoring_defaults.revisit_frames` (int)
- `scoring_defaults.final_score_weights` (`[mean_w, bottom_k_w]`)
- `benchmark_contract_version` (`"v1"`)
- `benchmark_contract_hash` (sha256 hex string)

### `score.json`
Required keys:
- `final_score` (number or null)
- `mean_score` (number or null)
- `bottom_k_score` (number or null)
- `per_game_scores` (object)
- `per_game_episode_counts` (object)
- `per_game_visit_frames` (object)
- `forgetting_index_mean` (number or null)
- `forgetting_index_median` (number or null)
- `per_game_forgetting` (object)
- `plasticity_mean` (number or null)
- `plasticity_median` (number or null)
- `per_game_plasticity` (object)
- `fps` (number or null)
- `frames` (int)
- `benchmark_contract_version` (`"v1"`)
- `benchmark_contract_hash` (must match `config.json`)

## Schedule Semantics

- The realized `schedule` is deterministic for a fixed `(games, base_visit_frames, num_cycles, seed, jitter_pct, min_visit_frames)`.
- Each visit is a contiguous frame block of exactly `visit_frames`.
- Visit boundaries are hard boundaries between games.
- `truncated == true` is valid only on the last frame of a visit.
- Mid-visit resets are `terminated` boundaries, not `truncated` (including any environment-level truncation signals).

## Agent Info Payload Contract

Agent `step(...)` info payload must include:
- `prev_applied_action_idx` (int)
- `has_prev_applied_action` (bool)

Meaning:
- If `has_prev_applied_action == false`, then `prev_applied_action_idx` is `default_action_idx`.
- Otherwise `prev_applied_action_idx` equals the previous frame's applied global action index.

Forbidden keys (must not be exposed to the agent):
- `game_id`
- `visit_idx`
- `cycle_idx`
- `visit_frame_idx`
- `frames_remaining`
- `visit_frames`
- `episode_id`
- `segment_id`

## Scoring Contract

Inputs:
- `window_frames`
- `bottom_k_frac`
- `revisit_frames`
- `final_score_weights = [mean_w, bottom_k_w]`

Definitions:
- Per-game online score: `tail_rate` over the selected last-cycle visit of each game:
  - `tail_return(visit, n) = sum(reward[f])` for frames `f` in
    `[max(visit.end_frame - n + 1, visit.start_frame), visit.end_frame]`
  - `n_eff = min(n, visit.end_frame - visit.start_frame + 1)`
  - `tail_rate(visit, n) = tail_return(visit, n) / n_eff`
- `mean_score`: mean of per-game scores.
- `bottom_k_score`: mean of lowest `ceil(bottom_k_frac * num_scored_games)` per-game scores.
- `final_score = mean_w * mean_score + bottom_k_w * bottom_k_score`.
- Forgetting per game: mean over non-adjacent revisit pairs of `(pre - post)`, where:
  - `pre = tail_rate(S_i, revisit_frames)`
  - `post = head_rate(S_{i+1}, revisit_frames)`
  - `head_return(visit, n) = sum(reward[f])` for frames `f` in
    `[visit.start_frame, min(visit.start_frame + n - 1, visit.end_frame)]`
  - `head_rate(visit, n) = head_return(visit, n) / n_eff`
- Plasticity per game: for the first-cycle visit `S_0(g)`, `late - early`, where:
  - `early = head_rate(S_0(g), revisit_frames)`
  - `late = tail_rate(S_0(g), revisit_frames)`

## Contract Hash Input (v1)

`benchmark_contract_hash` is SHA256 over canonical JSON (sorted keys) of:
- `games` (ordered)
- `schedule` records (ordered)
- `decision_interval`
- `delay_frames`
- `sticky`
- `life_loss_termination`
- `full_action_space`
- `global_action_set`
- `default_action_idx`
- scoring defaults used:
  - `window_frames`
  - `bottom_k_frac`
  - `revisit_frames`
  - `final_score_weights`

Excluded from hash:
- timestamps/wallclock values
- `run_dir`, `logdir`
- platform/python/library version metadata
- any host/runtime environment metadata

## Versioning Rules

Contract version must bump (v1 -> v2, etc.) for any incompatible change to:
- required artifact keys or key meaning
- schedule semantics or boundary semantics
- agent info payload contract or forbidden fields
- scoring definitions or score aggregation formula
- canonical contract-hash input set or normalization rules

Version bump is not required for:
- additive non-required metadata fields
- implementation optimizations that preserve v1 outputs/semantics
- tooling/docs changes that do not alter contract behavior
