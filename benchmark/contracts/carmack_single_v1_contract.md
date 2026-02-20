# Carmack Single-Run Contract (`carmack_single_v1`)

This document freezes the single-game Carmack-compatible runner profile contract.

## Profile Identity

- `single_run_profile`: `"carmack_compat"`
- `single_run_schema_version`: `"carmack_single_v1"`

These fields are required in:

- `config.json` (top-level)
- `config.json.runner_config`
- every row of `events.jsonl`
- every row of `episodes.jsonl`

## Agent-Facing Contract

Runner calls:

```python
agent.frame(obs_rgb, reward, boundary_payload)
```

`boundary_payload` keys:

- `terminated` (`bool`)
- `truncated` (`bool`)
- `end_of_episode_pulse` (`bool`)

Non-goal: no textual boundary reason is passed to the agent. `boundary_cause` is log-only.

## Cadence Contract

- Runner mode: agent-owned cadence.
- `--frame-skip` must be `1` for `carmack_compat`.
- Delay queue behavior is controlled by `reset_delay_queue_on_reset`:
  - `0` (default): persist queue across resets
  - `1`: re-seed queue with `default_action_idx` on reset

## Boundary/Reset Precedence

Boundary precedence (high to low):

1. `no_reward_timeout`
2. `terminated`
3. `truncated`
4. `life_loss`

Reset precedence (high to low):

1. `no_reward_timeout`
2. `terminated`
3. `truncated`
4. `life_loss_reset` (only when `reset_on_life_loss=1`)

## `events.jsonl` Required Fields

- profile markers: `single_run_profile`, `single_run_schema_version`
- control: `frame_idx`, `decided_action_idx`, `applied_action_idx`, `next_policy_action_idx`
- cadence diagnostics:
  - `decided_action_changed`
  - `applied_action_changed`
  - `decided_applied_mismatch`
  - `applied_action_hold_run_length`
- transition/boundary:
  - `reward`
  - `terminated`, `truncated`
  - `env_terminated`, `env_truncated`
  - `end_of_episode_pulse`
  - `pulse_reason`, `boundary_cause`, `reset_cause`
  - `reset_performed`
- episode context: `episode_idx`, `episode_return`, `episode_length`, `lives`
- timeout context: `frames_without_reward`

`pulse_reason`, `boundary_cause`, `reset_cause` may be `null`.

## `episodes.jsonl` Required Fields

- profile markers: `single_run_profile`, `single_run_schema_version`
- `episode_idx`
- `episode_return`
- `length`
- `termination_reason`
- `end_frame_idx`
- `ended_by_reset`

## Summary Fields (return value)

The run summary must include:

- frame/episode counters
- reset counters by cause
- pulse counters by cause
- cadence aggregate diagnostics:
  - action change counts/rates
  - decided/applied mismatch counts/rates
  - applied-action hold run count/mean/max

## Versioning Rules

Bump schema version when changing:

- required fields or types in `events.jsonl` / `episodes.jsonl` / config profile markers
- boundary precedence behavior
- agent-facing payload keys/semantics
- cadence invariants (`frame_skip` or delay/reset semantics)

No version bump required for:

- additive optional logging fields
- documentation-only edits
- internal refactors that do not change externally visible behavior
