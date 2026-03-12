# Sweep Tooling

Stage-0 sweep utilities live under `physical_atari/sweeps/` and write artifacts under:

`physical_atari/sweeps/output/<host>/<family>/<stage>/`

Each sweep root contains:

- `generated_trials/`: trial specs plus `manifest.json`
- `queue/`: file-based pending/claimed/completed/failed state
- `logs/`: per-worker logs and per-trial stdout/stderr logs
- `runs/`: per-trial benchmark config plus benchmark run directories
- `results/`: structured per-trial result records
- `summaries/`: aggregated `summary.json` and `summary.csv`

## Generate PPO Stage 0 trials

From the `physical_atari/` repo root:

```bash
python -m sweeps.generate_trials \
  --host donald \
  --family ppo \
  --stage stage0 \
  --rng-seed 123 \
  --count 40 \
  --seed-list 0
```

This writes deterministic JSON trial specs only. It does not launch anything.

## Launch PPO Stage 0 on `donald`

```bash
python -m sweeps.launch \
  --host donald \
  --family ppo \
  --stage stage0 \
  --gpus 0,1,2,3
```

PPO workers use `CUDA_VISIBLE_DEVICES=<gpu>` and force `--ppo-device cuda`.
For `delay_target`, `rainbow_dqn`, and `sac`, the launcher clears any inherited `CUDA_VISIBLE_DEVICES` mask so their explicit `--*-gpu` flags address host GPU ids directly.

## Launch BBF Stage 0 on `obsession`

`obsession` must be launched with exactly 2 GPUs:

```bash
python -m sweeps.generate_trials \
  --host obsession \
  --family bbf \
  --stage stage0 \
  --rng-seed 123 \
  --count 12 \
  --seed-list 0

python -m sweeps.launch \
  --host obsession \
  --family bbf \
  --stage stage0 \
  --gpus 0,1
```

BBF uses `CUDA_VISIBLE_DEVICES=<gpu>`.

## Resume, skip, and rerun behavior

- Generated trial specs are the source of truth for a sweep batch.
- The launcher skips trials that already have a result record with `status = "completed"`.
- Failed trials stay skipped by default.
- To rerun failed trials only, add `--rerun-failed`.
- Only one launcher process may own a given sweep root at a time. `sweeps.launch` takes a lock under `queue/launcher.lock` and refuses a second concurrent invocation for the same host/family/stage.
- Dry-run command preview:

```bash
python -m sweeps.launch \
  --host donald \
  --family ppo \
  --stage stage0 \
  --gpus 0,1,2,3 \
  --dry-run
```

## Re-aggregate results

Aggregation can be rerun any time without relaunching trials:

```bash
python -m sweeps.aggregate \
  --host donald \
  --family ppo \
  --stage stage0
```

Ranking order is:

1. higher `final_score`
2. lower positive `forgetting_index_mean`
3. higher `plasticity_mean`
4. deterministic `trial_id`

Missing `forgetting_index_mean` is treated as worse than any valid non-positive value.
