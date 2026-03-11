# PPO Sweep Plan

This document lays out a practical hyperparameter sweep plan for PPO on the `physical_atari` continual multi-game benchmark.

The goal is to tune PPO for:

- continual multi-game learning
- no task ID
- sticky actions
- optional latency stress testing
- long online training
- revisits and forgetting

This is not a standard Atari PPO sweep. The benchmark dynamics are different enough that standard defaults should not be assumed to transfer cleanly.

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

## Why PPO Deserves A Broad Sweep

PPO is much cheaper to run than BBF, Rainbow DQN, or SAC in this setup. That means the sweep strategy should be different:

- broader first-pass search
- more configs in early stages
- longer ranking budget only for survivors

However, the sweep should still not be exhaustive. A full Cartesian product over all PPO knobs is too large and unnecessary.

The right framing is:

- high-coverage
- staged
- budget-aware
- not exhaustive

## PPO Config Surface In `physical_atari`

The benchmark-local PPO agent exposes the following main configuration:

```python
class PPOConfig:
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    train_interval: int = 2048
    batch_size: int = 64
    epochs: int = 4
    reward_clip: float = 1.0
    obs_size: int = 84
    frame_stack: int = 4
    grayscale: bool = True
    normalize_advantages: bool = True
    deterministic_actions: bool = False
    device: str = "auto"
```

In the benchmark CLI, PPO also has:

- `ppo_decision_interval`

That benchmark-specific cadence knob matters a lot in the delayed, sticky, continual setting.

## PPO Hyperparameter Priorities

### Tier 1: definitely sweep

These are the main PPO knobs for this benchmark.

1. `ppo_lr`
2. `ppo_clip_range`
3. `ppo_ent_coef`
4. `ppo_vf_coef`
5. `ppo_rollout_steps`
6. `ppo_batch_size`
7. `ppo_epochs`
8. `ppo_decision_interval`

These are the core knobs I would use in the first sweep.

### Tier 2: sweep after narrowing

These are important, but I would delay them until after the first ranking pass.

9. `ppo_gamma`
10. `ppo_gae_lambda`
11. `ppo_max_grad_norm`
12. `ppo_reward_clip`

### Tier 3: usually keep fixed initially

These are real knobs, but I would not prioritize them in the first pass.

- `ppo_obs_size`
- `ppo_frame_stack`
- `ppo_grayscale`
- `ppo_normalize_advantages`
- `ppo_deterministic_actions`
- `ppo_device`

## Recommended First-Pass Search Space

This is the recommended initial PPO search space.

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `ppo_lr` | `2.5e-4` | `1e-5` to `1e-3` log-uniform | Most important optimization knob |
| `ppo_clip_range` | `0.2` | `0.1, 0.15, 0.2, 0.3` | PPO trust-region aggressiveness |
| `ppo_ent_coef` | `0.01` | `1e-4` to `5e-2` log-uniform | Exploration pressure |
| `ppo_vf_coef` | `0.5` | `0.25, 0.5, 0.75, 1.0, 2.0` | Value-loss weight |
| `ppo_rollout_steps` | `2048` | `256, 512, 1024, 2048, 4096` | Effective on-policy batch size |
| `ppo_batch_size` | `64` | `32, 64, 128, 256` | Minibatch size |
| `ppo_epochs` | `4` | `2, 4, 6, 8` | Optimization intensity per batch |
| `ppo_decision_interval` | `4` | `2, 4, 6, 8` | Agent-owned action cadence |

### Constraints

Apply these constraints during sampling:

- `batch_size <= rollout_steps`
- Prefer `rollout_steps % batch_size == 0`
- `train_interval = rollout_steps` in the initial sweep

That last rule is important. Do not open `train_interval` early unless there is a strong reason. It is cleaner to tie it to `rollout_steps` for the first pass.

## Recommended Second-Pass Search Space

Once the first pass has identified promising PPO regimes, open these:

| Knob | Default | Recommended values / range | Notes |
|---|---:|---|---|
| `ppo_gamma` | `0.99` | `0.97, 0.985, 0.99, 0.995, 0.997` | Effective horizon |
| `ppo_gae_lambda` | `0.95` | `0.9, 0.95, 0.97, 0.99` | Bias/variance tradeoff |
| `ppo_max_grad_norm` | `0.5` | `0.3, 0.5, 1.0` | Gradient stability |
| `ppo_reward_clip` | `1.0` | `0.5, 1.0, 2.0, 0.0` | Reward scaling and robustness |

## What To Keep Fixed Initially

I would keep these fixed for the first PPO sweep:

- `ppo_obs_size = 84`
- `ppo_frame_stack = 4`
- `ppo_grayscale = 1`
- `ppo_normalize_advantages = 1`
- `ppo_deterministic_actions = 0`

These are sensible defaults and are not the first place to spend budget.

## Why `ppo_decision_interval` Matters Here

This is one of the most benchmark-specific PPO knobs.

In the primary sweep setup:

- the runner is in `carmack_compat` mode
- the environment has sticky actions
- the PPO agent owns its own action cadence

So `ppo_decision_interval` is not just a minor efficiency setting. It changes:

- how often PPO can react
- the effective temporal granularity of control
- the observation-action mismatch under delayed actuation

Because of that, I would always include it in the first sweep.

## Staged PPO Sweep Procedure

Use a broad staged random search.

### Stage 0: quick filter

Benchmark:

- 3 games
- 2 cycles
- `250k` base visit frames

Plan:

- 32 to 48 random configs from the Tier 1 space
- keep top 8 to 12

Goal:

- identify PPO regimes that are at least viable under the continual benchmark

### Stage 1: ranking

Benchmark:

- 3 games
- 2 cycles
- `750k` base visit frames

Plan:

- run the 8 to 12 promoted configs
- keep top 3 to 4

Goal:

- identify PPO regimes whose ranking persists at a longer continual-learning horizon

### Stage 2: refinement

Benchmark:

- 3 games
- 2 cycles
- `750k` base visit frames

Plan:

- locally perturb the top 3 to 4 configs
- open Tier 2 knobs:
  - `ppo_gamma`
  - `ppo_gae_lambda`
  - `ppo_max_grad_norm`
  - `ppo_reward_clip`
- keep top 1 to 2

Goal:

- stabilize the best PPO region and improve longer-horizon continual behavior

### Stage 3: full-benchmark finalist check

Benchmark:

- 8 games
- 3 cycles
- `500k` base visit frames

Plan:

- run the 1 to 2 finalists

Goal:

- verify the PPO configuration generalizes from the 3-game subset to the full benchmark

### Stage 4: final confirmation

Benchmark:

- 8 games
- 3 cycles
- `1.0M` base visit frames

Plan:

- run the best PPO config
- if affordable, use 2 to 3 seeds

Goal:

- final scientific comparison against the other agent families

## Suggested Trial Counts

Because PPO is relatively cheap, the sweep can be broader than BBF.

- Stage 0: 32 to 48 configs
- Stage 1: 8 to 12 configs
- Stage 2: 6 to 8 configs
- Stage 3: 1 to 2 configs
- Stage 4: 1 config, multiple seeds if affordable

## Promotion Metric

Promote PPO configs using the benchmark’s continual-learning metrics, not just short-horizon return.

Recommended ranking:

1. primary: `final_score`
2. tie-break: lower positive `forgetting_index_mean`
3. second tie-break: higher `plasticity_mean`

Also reject PPO configs with clear training pathology even if the score is superficially okay.

Important PPO pathology signals:

- extreme `approx_kl`
- entropy collapse too early
- unstable or exploding value loss
- clear late-stage degradation after a good early score

## PPO Failure Modes To Watch For

This benchmark is unusual enough that PPO can fail in several distinct ways.

### 1. Early entropy collapse

The policy becomes nearly deterministic too soon and stops adapting effectively.

Likely culprits:

- low `ent_coef`
- too many epochs
- too high learning rate

### 2. Update instability

The PPO update becomes too aggressive and KL spikes.

Likely culprits:

- high `learning_rate`
- too large `epochs`
- too small `clip_range`
- too large `rollout_steps` combined with aggressive optimization

### 3. Slow adaptation after game switches

The policy remains too inert when revisiting a game.

Likely culprits:

- low `learning_rate`
- low `ent_coef`
- too large `decision_interval`
- poor `gamma` / `gae_lambda` regime

### 4. Good plasticity but poor retention

The agent improves quickly inside a visit but loses too much across revisits.

Likely culprits:

- update regime too aggressive
- poor reward clipping regime
- value loss dominating policy shaping

## Sampling Strategy

Do not use a full grid.

Instead:

- use random search or Latin-hypercube style coverage for Tier 1
- apply hard constraints on valid parameter combinations
- use local perturbations around the best configs in later stages

This gives strong coverage without wasting runs.

## What This PPO Sweep Is

This plan is:

- broad
- staged
- high-coverage
- efficient

It is not exhaustive in the literal combinatorial sense.

That is intentional. Exhaustive search is not realistic, and it is not necessary for PPO here.

## What I Would Not Sweep Yet

Do not start by sweeping:

- `obs_size`
- `frame_stack`
- grayscale vs RGB
- deterministic action mode
- device choice

Those are lower-priority choices for this benchmark and should only be opened up if PPO already looks promising.

## Final Recommendation

For this benchmark, PPO should be treated as the broad-search family:

- much wider than BBF, Rainbow, or SAC
- still not exhaustive
- focused on the high-leverage optimization and cadence knobs

The most important PPO sweep cluster for this benchmark is:

- `ppo_lr`
- `ppo_clip_range`
- `ppo_ent_coef`
- `ppo_rollout_steps`
- `ppo_batch_size`
- `ppo_epochs`
- `ppo_decision_interval`

That cluster is where I would expect most of the meaningful variation to come from under delayed, sticky, continual multi-game Atari.
