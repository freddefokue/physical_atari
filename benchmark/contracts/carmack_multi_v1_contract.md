# Carmack Multi-Game Contract v1

This document freezes the Carmack-compatible multi-game runner profile for reproducible continual-stream runs.

## 1) Profile Identity (Required)

- `runner_mode`: `"carmack_compat"` (in `config.json`)
- `multi_run_profile`: `"carmack_compat"` (in `config.json`, event/episode/segment rows, `run_summary.json`)
- `multi_run_schema_version`: `"carmack_multi_v1"` (in `config.json`, event/episode/segment rows, `run_summary.json`)

## 2) Control-Loop Semantics (Normative)

- Exactly one env step per global frame.
- Post-step control loop:
  1. apply delayed decided action
  2. call `env.step(...)`
  3. compute boundary/reset state
  4. call `agent.frame(obs, reward, boundary_payload)`
  5. returned action becomes next-frame decided action
- Runner-owned frame skip is disabled in this profile:
  - `decision_interval == 1`
  - `runner_config.action_cadence_mode == "agent_owned"`
  - `runner_config.frame_skip_enforced == 1`

## 3) Agent-Facing Payload Contract

`agent.frame(obs, reward, boundary_payload)` receives:

Required keys:
- `terminated` (bool)
- `truncated` (bool)
- `end_of_episode_pulse` (bool)
- `has_prev_applied_action` (bool)
- `prev_applied_action_idx` (int; index in global action set)
- `global_frame_idx` (int)

Forbidden task-identity/schedule fields:
- `game_id`
- `visit_idx`
- `cycle_idx`
- `visit_frame_idx`
- `episode_id`
- `segment_id`
- `visit_frames`

## 4) Boundary/Reset Semantics

Canonical `boundary_cause` enum:
- `visit_switch`
- `truncated`
- `terminated`

Canonical `reset_cause` enum:
- `visit_switch`
- `truncated`
- `terminated`

Cause precedence:
1. `visit_switch` (synthetic boundary at last frame of each visit)
2. env truncated
3. env terminated

Rules:
- `end_of_episode_pulse == (terminated or truncated)`
- `reset_performed == (reset_cause is not null)`
- For `visit_switch`: `truncated=true`, `boundary_cause="visit_switch"`, `reset_cause="visit_switch"`, `reset_performed=true`
- `terminated == env_terminated`
- `truncated == (env_truncated or boundary_cause=="visit_switch")`

## 5) Required Artifacts and Required Fields

### `config.json`
Must include:
- profile keys in section 1
- schedule core: `games`, `schedule`, `total_scheduled_frames`
- mechanics: `decision_interval`, `delay`, `sticky`, `life_loss_termination`, `full_action_space`, `default_action_idx`
- action mapping policy (`action_mapping_policy.global_action_set`)
- runner config with:
  - `runner_mode`
  - `multi_run_schema_version`
  - `action_cadence_mode`
  - `frame_skip_enforced`
  - `decision_interval`
  - `delay_frames`
  - `reset_delay_queue_on_reset`
  - `reset_delay_queue_on_visit_switch`
- benchmark contract tags:
  - `benchmark_contract_version`
  - `benchmark_contract_hash`

### `events.jsonl` (each row)
Required fields:
- `multi_run_profile`, `multi_run_schema_version`
- `frame_idx`, `global_frame_idx`
- `game_id`, `visit_idx`, `cycle_idx`, `visit_frame_idx`
- `episode_id`, `segment_id`
- `is_decision_frame`
- `decided_action_idx`, `applied_action_idx`, `next_policy_action_idx`
- `applied_action_idx_local`, `applied_ale_action`
- `reward`
- `terminated`, `truncated`
- `env_terminated`, `env_truncated`
- `end_of_episode_pulse`
- `boundary_cause`, `reset_cause`, `reset_performed`
- `lives`
- `episode_return_so_far`, `segment_return_so_far`
- `env_termination_reason` (nullable string)

### `episodes.jsonl` (each row)
Required fields:
- `multi_run_profile`, `multi_run_schema_version`
- `game_id`, `episode_id`
- `start_global_frame_idx`, `end_global_frame_idx`
- `length`, `return`
- `ended_by` in `{terminated,truncated}`
- `boundary_cause`

### `segments.jsonl` (each row)
Required fields:
- `multi_run_profile`, `multi_run_schema_version`
- `game_id`, `segment_id`
- `start_global_frame_idx`, `end_global_frame_idx`
- `length`, `return`
- `ended_by` in `{terminated,truncated}`
- `boundary_cause`

### `run_summary.json`
Required fields:
- `runner_mode`, `multi_run_profile`, `multi_run_schema_version`
- `frames`, `episodes_completed`, `segments_completed`
- `last_episode_id`, `last_segment_id`
- `visits_completed`, `total_scheduled_frames`
- `boundary_cause_counts`, `reset_cause_counts`
- `reset_count`

### `score.json` (scoring stage, not runner)
Produced by scorer tooling and must include benchmark contract tags plus score schema required by `benchmark/validate_contract.py`.

## 6) Delay Queue Policy

- Delay queue seeded with `default_action_idx`.
- Reset behavior is explicit via config:
  - `runner_config.reset_delay_queue_on_reset`
  - `runner_config.reset_delay_queue_on_visit_switch`

## 7) Versioning Rules

Bump contract version if any of the following changes:
- required artifact fields added/removed/renamed
- meaning or precedence of boundary/reset causes changes
- agent-facing payload keys or semantics change
- required enums change
- invariants used by validator change
- scoring-stage required contract tags or interpretation change

Non-breaking changes (no bump):
- additional optional logging fields
- new informational summary fields not consumed by validator/scorer
