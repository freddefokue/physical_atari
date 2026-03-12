"""Stage-specific family definitions and deterministic samplers."""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from sweeps.common import deep_merge, load_template, normalize_for_hash

DEFAULT_STAGE0_SEEDS = [0]
DISABLED_RESET_SENTINEL = 2_147_483_647
FAMILIES = ("ppo", "delay_target", "bbf", "rainbow_dqn", "sac")


@dataclass(frozen=True)
class FamilyDefinition:
    name: str
    stage: str
    default_count: int
    template_file: str


FAMILY_DEFINITIONS = {
    "ppo": FamilyDefinition(name="ppo", stage="stage0", default_count=40, template_file="ppo_stage0.json"),
    "delay_target": FamilyDefinition(
        name="delay_target",
        stage="stage0",
        default_count=20,
        template_file="delay_target_stage0.json",
    ),
    "bbf": FamilyDefinition(name="bbf", stage="stage0", default_count=12, template_file="bbf_stage0.json"),
    "rainbow_dqn": FamilyDefinition(
        name="rainbow_dqn",
        stage="stage0",
        default_count=10,
        template_file="rainbow_dqn_stage0.json",
    ),
    "sac": FamilyDefinition(name="sac", stage="stage0", default_count=10, template_file="sac_stage0.json"),
}

HELPER_ONLY_SAMPLED_FIELDS = {
    "bbf": {"encoder_lr_ratio", "no_reset_multiple"},
}


def validate_family_stage(family: str, stage: str) -> FamilyDefinition:
    if family not in FAMILY_DEFINITIONS:
        raise ValueError(f"Unsupported family: {family}")
    definition = FAMILY_DEFINITIONS[family]
    if stage != definition.stage:
        raise ValueError(f"Unsupported stage for {family}: {stage}")
    return definition


def default_count_for(family: str, stage: str) -> int:
    return validate_family_stage(family, stage).default_count


def build_benchmark_config(
    *,
    family: str,
    stage: str,
    training_seed: int,
    sampled_hyperparameters: Dict[str, Any],
) -> Dict[str, Any]:
    definition = validate_family_stage(family, stage)
    base_template = load_template("stage0_base.json")
    family_template = load_template(definition.template_file)
    config = deep_merge(base_template["benchmark_config"], family_template["benchmark_config"])
    helper_only_fields = HELPER_ONLY_SAMPLED_FIELDS.get(family, set())
    benchmark_overrides = {
        key: value for key, value in sampled_hyperparameters.items() if key not in helper_only_fields
    }
    config.update(benchmark_overrides)
    config["seed"] = int(training_seed)
    return config


def sample_family_configs(
    *,
    family: str,
    stage: str,
    count: int,
    rng_seed: int,
) -> List[Dict[str, Any]]:
    validate_family_stage(family, stage)
    if count <= 0:
        raise ValueError("--count must be > 0")
    rng = random.Random(rng_seed)
    if family == "ppo":
        return _sample_ppo_configs(count=count, rng=rng)
    if family == "delay_target":
        return _sample_discrete_configs(count=count, rng=rng, space=_delay_target_space())
    if family == "bbf":
        return _sample_discrete_configs(count=count, rng=rng, space=_bbf_space())
    if family == "rainbow_dqn":
        return _sample_discrete_configs(count=count, rng=rng, space=_rainbow_space())
    if family == "sac":
        return _sample_discrete_configs(count=count, rng=rng, space=_sac_space())
    raise AssertionError(f"Unhandled family: {family}")


def _choice_with_replacement(rng: random.Random, values: Sequence[Any]) -> Any:
    return values[rng.randrange(len(values))]


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _canonical_payload(payload: Dict[str, Any]) -> str:
    normalized = normalize_for_hash(payload)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _sample_ppo_configs(*, count: int, rng: random.Random) -> List[Dict[str, Any]]:
    rollout_options = [256, 512, 1024, 2048, 4096]
    batch_options = [32, 64, 128, 256]
    clip_options = [0.1, 0.15, 0.2, 0.3]
    vf_coef_options = [0.25, 0.5, 0.75, 1.0, 2.0]
    epoch_options = [2, 4, 6, 8]
    decision_interval_options = [2, 4, 6, 8]

    configs: List[Dict[str, Any]] = []
    seen = set()
    while len(configs) < count:
        rollout_steps = _choice_with_replacement(rng, rollout_options)
        divisor_batches = [batch for batch in batch_options if batch <= rollout_steps and rollout_steps % batch == 0]
        batch_size = _choice_with_replacement(rng, divisor_batches)
        config = {
            "ppo_lr": _log_uniform(rng, 1e-5, 1e-3),
            "ppo_clip_range": _choice_with_replacement(rng, clip_options),
            "ppo_ent_coef": _log_uniform(rng, 1e-4, 5e-2),
            "ppo_vf_coef": _choice_with_replacement(rng, vf_coef_options),
            "ppo_rollout_steps": rollout_steps,
            "ppo_batch_size": batch_size,
            "ppo_epochs": _choice_with_replacement(rng, epoch_options),
            "ppo_decision_interval": _choice_with_replacement(rng, decision_interval_options),
            "ppo_train_interval": rollout_steps,
        }
        key = _canonical_payload(config)
        if key in seen:
            continue
        seen.add(key)
        configs.append(config)
    return configs


def _sample_discrete_configs(*, count: int, rng: random.Random, space: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not space:
        raise ValueError("Discrete search space must not be empty")
    if count <= len(space):
        return [dict(item) for item in rng.sample(list(space), count)]
    configs = [dict(item) for item in space]
    while len(configs) < count:
        configs.append(dict(_choice_with_replacement(rng, space)))
    return configs


def _delay_target_space() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for lr_log2, base_lr_log2, ring_buffer_size in itertools.product(
        [-19, -18, -17, -16],
        [-17, -16, -15, -14],
        [8192, 16384, 32768, 65536],
    ):
        configs.append(
            {
                "delay_target_lr_log2": lr_log2,
                "delay_target_base_lr_log2": base_lr_log2,
                "delay_target_ring_buffer_size": ring_buffer_size,
            }
        )
    return configs


def _bbf_space() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for (
        learning_starts,
        replay_ratio,
        batch_size,
        learning_rate,
        encoder_lr_ratio,
        reset_interval,
        no_reset_multiple,
        spr_weight,
    ) in itertools.product(
        [1000, 2000, 5000, 10000],
        [16, 32, 64, 96, 128],
        [16, 32, 64],
        [3e-5, 1e-4, 3e-4],
        [0.25, 0.5, 1.0, 2.0],
        [10000, 20000, 40000, 80000],
        [2, 5, 10, "disabled"],
        [1.0, 3.0, 5.0, 7.5, 10.0],
    ):
        if no_reset_multiple == "disabled":
            no_resets_after = DISABLED_RESET_SENTINEL
        else:
            no_resets_after = int(no_reset_multiple) * int(reset_interval)
        configs.append(
            {
                "bbf_learning_starts": learning_starts,
                "bbf_replay_ratio": replay_ratio,
                "bbf_batch_size": batch_size,
                "bbf_learning_rate": learning_rate,
                "encoder_lr_ratio": encoder_lr_ratio,
                "bbf_encoder_learning_rate": encoder_lr_ratio * learning_rate,
                "bbf_reset_interval": reset_interval,
                "no_reset_multiple": no_reset_multiple,
                "bbf_no_resets_after": no_resets_after,
                "bbf_spr_weight": spr_weight,
            }
        )
    return configs


def _rainbow_space() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for values in itertools.product(
        [3e-5, 1e-4, 3e-4],
        [10000, 25000, 50000, 100000],
        [32, 64],
        [50000, 100000, 200000, 400000],
        [1000, 2000, 4000, 8000],
        [1, 3, 5],
    ):
        configs.append(
            {
                "rainbow_dqn_learning_rate": values[0],
                "rainbow_dqn_train_start": values[1],
                "rainbow_dqn_batch_size": values[2],
                "rainbow_dqn_buffer_size": values[3],
                "rainbow_dqn_target_update_freq": values[4],
                "rainbow_dqn_n_step": values[5],
            }
        )
    return configs


def _sac_space() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for values in itertools.product(
        [3e-5, 1e-4, 3e-4],
        [5000, 10000, 25000, 50000],
        [32, 64, 128],
        [50000, 100000, 200000, 400000],
        [1, 2, 4],
        [0.001, 0.003, 0.005, 0.01],
        [0.25, 0.5, 0.75, 1.0],
    ):
        configs.append(
            {
                "sac_learning_rate": values[0],
                "sac_learning_starts": values[1],
                "sac_batch_size": values[2],
                "sac_buffer_size": values[3],
                "sac_gradient_steps": values[4],
                "sac_tau": values[5],
                "sac_target_entropy_scale": values[6],
            }
        )
    return configs
