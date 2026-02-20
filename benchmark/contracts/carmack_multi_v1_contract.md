# Carmack Multi-Game Contract v1

This document defines the Carmack-compatible multi-game runner contract for continual Atari streams.

## Profile Identity

- `runner_mode`: `"carmack_compat"`
- `multi_run_profile`: `"carmack_compat"`
- `multi_run_schema_version`: `"carmack_multi_v1"`

## Loop Semantics

- One environment step per global frame.
- Control loop order is post-step:
  1. apply delayed decided action
  2. call `env.step(...)`
  3. emit boundary payload and call `agent.frame(obs, reward, boundary_payload)`
  4. returned action becomes next-frame decided action
- Runner does not perform policy frame-skip; cadence is agent-owned.

## Agent-Facing Boundary Payload

Required keys passed to `agent.frame(...)`:
- `terminated` (bool)
- `truncated` (bool)
- `end_of_episode_pulse` (bool)
- `has_prev_applied_action` (bool)
- `prev_applied_action_idx` (int, global-action index)

Optional key:
- `global_frame_idx` (int)

Forbidden task-identity fields in agent payload:
- `game_id`, `visit_idx`, `cycle_idx`, `visit_frame_idx`, `episode_id`, `segment_id`, `visit_frames`

## Boundary and Reset Semantics

Synthetic visit switch boundary:
- last frame of every scheduled visit emits an episode/segment boundary
- logged as `truncated=true`, `boundary_cause="visit_switch"`, `reset_cause="visit_switch"`

Environment boundaries:
- env terminated -> `terminated=true`
- env truncated -> `truncated=true`

Precedence for boundary cause:
1. `visit_switch`
2. env `truncated`
3. env `terminated`

`episodes.jsonl` and `segments.jsonl` keep `ended_by` in `{terminated,truncated}` for scoring compatibility.

## Delay Queue Policy

- Delay queue initialization uses `default_action_idx`.
- Policy on visit switches and in-visit resets is controlled by runner config fields and must be logged in `config.json`.

## Required Artifacts

- `config.json`
- `events.jsonl`
- `episodes.jsonl`
- `segments.jsonl`
- `run_summary.json`

Scoring-stage artifact:
- `score.json` is produced by scorer tooling (not by the runner process itself).

All rows must include profile/schema markers:
- `multi_run_profile`
- `multi_run_schema_version`
