"""Generate deterministic Stage-0 sweep trial specs."""

from __future__ import annotations

import argparse
import hashlib
from typing import Any, Dict, List

from sweeps.common import HOSTS, STAGES, ensure_sweep_dirs, normalize_for_hash, parse_seed_list, trial_spec_paths, utc_now_iso, write_json
from sweeps.families import DEFAULT_STAGE0_SEEDS, build_benchmark_config, default_count_for, sample_family_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic sweep trial specs.")
    parser.add_argument("--host", type=str, required=True, choices=sorted(HOSTS), help="Target host name.")
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=["ppo", "delay_target", "bbf", "rainbow_dqn", "sac"],
        help="Agent family to generate.",
    )
    parser.add_argument("--stage", type=str, default="stage0", choices=sorted(STAGES), help="Sweep stage.")
    parser.add_argument("--rng-seed", type=int, required=True, help="Sampling RNG seed for hyperparameter generation.")
    parser.add_argument("--count", type=int, default=None, help="Number of sampled configs to generate.")
    parser.add_argument(
        "--seed-list",
        type=str,
        default="0",
        help="Comma-separated benchmark seed list. Default: 0",
    )
    return parser.parse_args()


def build_trial_id(
    *,
    family: str,
    stage: str,
    rng_seed: int,
    config_index: int,
    training_seed: int,
    sampled_hyperparameters: Dict[str, Any],
) -> str:
    digest_payload = {
        "family": family,
        "stage": stage,
        "rng_seed": rng_seed,
        "config_index": config_index,
        "training_seed": training_seed,
        "sampled_hyperparameters": sampled_hyperparameters,
    }
    normalized = repr(normalize_for_hash(digest_payload)).encode("utf-8")
    digest = hashlib.sha1(normalized).hexdigest()[:10]
    return f"{family}-{stage}-c{config_index:03d}-s{training_seed}-{digest}"


def build_trial_specs(
    *,
    host: str,
    family: str,
    stage: str,
    rng_seed: int,
    count: int,
    seed_list: List[int],
) -> List[Dict[str, Any]]:
    sampled_configs = sample_family_configs(family=family, stage=stage, count=count, rng_seed=rng_seed)
    created_at = utc_now_iso()
    specs: List[Dict[str, Any]] = []
    for config_index, sampled_hyperparameters in enumerate(sampled_configs):
        for training_seed in seed_list:
            trial_id = build_trial_id(
                family=family,
                stage=stage,
                rng_seed=rng_seed,
                config_index=config_index,
                training_seed=training_seed,
                sampled_hyperparameters=sampled_hyperparameters,
            )
            benchmark_config = build_benchmark_config(
                family=family,
                stage=stage,
                training_seed=training_seed,
                sampled_hyperparameters=sampled_hyperparameters,
            )
            specs.append(
                {
                    "trial_id": trial_id,
                    "family": family,
                    "stage": stage,
                    "host": host,
                    "rng_seed": int(rng_seed),
                    "config_index": int(config_index),
                    "training_seed": int(training_seed),
                    "sampled_hyperparameters": dict(sampled_hyperparameters),
                    "benchmark_config": benchmark_config,
                    "generated_at": created_at,
                }
            )
    return specs


def main() -> None:
    args = parse_args()
    count = int(args.count) if args.count is not None else default_count_for(args.family, args.stage)
    if count <= 0:
        raise ValueError("--count must be > 0")
    seed_list = DEFAULT_STAGE0_SEEDS if args.seed_list.strip() == "0" else parse_seed_list(args.seed_list)
    if not seed_list:
        raise ValueError("--seed-list must not be empty")

    paths = ensure_sweep_dirs(args.host, args.family, args.stage)
    specs = build_trial_specs(
        host=args.host,
        family=args.family,
        stage=args.stage,
        rng_seed=int(args.rng_seed),
        count=count,
        seed_list=seed_list,
    )

    trial_files: List[str] = []
    trial_ids: List[str] = []
    for spec in specs:
        derived_paths = trial_spec_paths(paths, spec["trial_id"])
        spec["output_paths"] = {key: str(value) for key, value in derived_paths.items()}
        path = derived_paths["generated_trial_path"]
        write_json(path, spec)
        trial_files.append(path.name)
        trial_ids.append(spec["trial_id"])

    manifest = {
        "host": args.host,
        "family": args.family,
        "stage": args.stage,
        "rng_seed": int(args.rng_seed),
        "config_count": int(count),
        "seed_list": [int(seed) for seed in seed_list],
        "trial_count": int(len(specs)),
        "trial_ids": trial_ids,
        "trial_files": trial_files,
        "generated_at": utc_now_iso(),
    }
    write_json(paths["generated_trials"] / "manifest.json", manifest)

    print(f"Wrote {len(specs)} trial specs to {paths['generated_trials']}")
    print(f"Manifest: {paths['generated_trials'] / 'manifest.json'}")


if __name__ == "__main__":
    main()
