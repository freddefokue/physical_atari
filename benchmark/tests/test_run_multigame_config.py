from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from benchmark.carmack_multigame_runner import CarmackMultiGameRunnerConfig
from benchmark.run_multigame import (
    BBF_MULTIGAME_HEARTBEAT_TRAIN_INTERVAL,
    _resolve_bbf_log_visibility,
    _resolve_bbf_runtime_settings,
    build_config_payload,
    collect_agent_stats,
    parse_args,
    validate_args,
)
from benchmark.schedule import Schedule, ScheduleConfig


def test_run_multigame_config_defaults_and_cli_override(tmp_path):
    config_path = tmp_path / "cfg.json"
    payload = {
        "games": ["ms_pacman", "centipede"],
        "num_cycles": 2,
        "base_visit_frames": 1234,
        "jitter_pct": 0.03,
        "min_visit_frames": 111,
        "seed": 7,
        "decision_interval": 5,
        "delay": 6,
        "sticky": 0.2,
        "full_action_space": 1,
        "life_loss_termination": 1,
        "default_action_idx": 0,
        "log_episode_every": 7,
        "timestamps": 1,
        "real_time_mode": 1,
        "real_time_fps": 75.0,
        "agent": "tinydqn",
        "agent_config": {
            "gamma": 0.95,
            "lr": 0.0003,
            "batch_size": 64,
            "train_log_interval": 42,
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    args = parse_args([
        "--config",
        str(config_path),
        "--delay",
        "2",
        "--seed",
        "99",
    ])

    assert args.games == "ms_pacman,centipede"
    assert args.num_cycles == 2
    assert args.base_visit_frames == 1234
    assert args.jitter_pct == 0.03
    assert args.min_visit_frames == 111
    assert args.decision_interval == 5
    assert args.delay == 2  # CLI override wins
    assert args.seed == 99  # CLI override wins
    assert args.agent == "tinydqn"
    assert args.log_episode_every == 7
    assert args.dqn_gamma == 0.95
    assert args.dqn_lr == 0.0003
    assert args.dqn_batch_size == 64
    assert args.dqn_log_train_every == 42
    assert args.real_time_mode == 1
    assert args.real_time_fps == 75.0
    assert getattr(args, "_config_data") == payload


def test_collect_agent_stats_prefers_get_stats_payload():
    class _DummyAgent:
        replay_size = 3

        def get_stats(self):
            return {
                "decision_steps": 17,
                "current_epsilon": 0.42,
                "extra": "ok",
            }

    stats = collect_agent_stats(_DummyAgent())
    assert stats["decision_steps"] == 17
    assert stats["current_epsilon"] == 0.42
    assert stats["extra"] == "ok"
    assert stats["replay_size"] == 3


def test_run_multigame_config_supports_runner_config_episode_log_interval(tmp_path):
    config_path = tmp_path / "cfg_runner.json"
    payload = {
        "games": ["ms_pacman"],
        "runner_config": {
            "episode_log_interval": 9,
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    args = parse_args(["--config", str(config_path)])
    assert args.log_episode_every == 9


def test_run_multigame_config_parses_delay_target_agent_config(tmp_path):
    config_path = tmp_path / "cfg_delay_target.json"
    payload = {
        "games": ["ms_pacman"],
        "agent": "delay_target",
        "decision_interval": 1,
        "agent_config": {
            "gpu": 2,
            "use_cuda_graphs": 0,
            "load_file": "/tmp/example.model",
            "ring_buffer_size": 32768,
            "base_width": 96,
            "temperature_log2": -5,
            "greedy_ramp": 250000,
            "multisteps_max": 48,
            "td_lambda": 0.9,
            "train_batch": 64,
            "online_batch": 8,
            "online_loss_scale": 1.5,
            "train_steps": 6,
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    args = parse_args(["--config", str(config_path)])
    assert args.agent == "delay_target"
    assert args.decision_interval == 1
    assert args.delay_target_gpu == 2
    assert args.delay_target_use_cuda_graphs == 0
    assert args.delay_target_load_file == "/tmp/example.model"
    assert args.delay_target_ring_buffer_size == 32768
    assert args.delay_target_base_width == 96
    assert args.delay_target_temperature_log2 == -5
    assert args.delay_target_greedy_ramp == 250000
    assert args.delay_target_multisteps_max == 48
    assert args.delay_target_td_lambda == pytest.approx(0.9)
    assert args.delay_target_train_batch == 64
    assert args.delay_target_online_batch == 8
    assert args.delay_target_online_loss_scale == pytest.approx(1.5)
    assert args.delay_target_train_steps == 6


def test_run_multigame_config_parses_runner_mode_and_tinydqn_decision_interval(tmp_path):
    config_path = tmp_path / "cfg_carmack_tiny.json"
    payload = {
        "games": ["ms_pacman"],
        "runner_mode": "carmack_compat",
        "decision_interval": 1,
        "agent": "tinydqn",
        "agent_config": {
            "decision_interval": 7,
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    args = parse_args(["--config", str(config_path)])
    assert args.runner_mode == "carmack_compat"
    assert args.agent == "tinydqn"
    assert args.decision_interval == 1
    assert args.dqn_decision_interval == 7


def test_validate_args_carmack_requires_decision_interval_one():
    args = parse_args(
        [
            "--games",
            "ms_pacman",
            "--runner-mode",
            "carmack_compat",
            "--decision-interval",
            "4",
        ]
    )
    with pytest.raises(ValueError, match=r"carmack_compat requires --decision-interval 1"):
        validate_args(args)


def test_run_multigame_cli_accepts_ppo_agent():
    args = parse_args(["--games", "pong", "--agent", "ppo"])
    assert args.agent == "ppo"
    assert args.ppo_lr == pytest.approx(2.5e-4)
    assert args.ppo_rollout_steps == 128
    assert args.ppo_decision_interval == 4


def test_run_multigame_cli_accepts_dqn_agent():
    args = parse_args(["--games", "pong", "--agent", "dqn"])
    assert args.agent == "dqn"
    assert args.roboatari_dqn_gpu == 0
    assert args.roboatari_dqn_load_file is None


def test_run_multigame_cli_accepts_rainbow_dqn_agent():
    args = parse_args(["--games", "pong", "--agent", "rainbow_dqn"])
    assert args.agent == "rainbow_dqn"
    assert args.rainbow_dqn_gpu == 0
    assert args.rainbow_dqn_load_file is None
    assert args.rainbow_dqn_learning_rate is None
    assert args.rainbow_dqn_grad_clip is not None


def test_run_multigame_cli_accepts_sac_agent():
    args = parse_args(["--games", "pong", "--agent", "sac"])
    assert args.agent == "sac"
    assert args.sac_gpu == 0
    assert args.sac_load_file is None
    assert args.sac_eval_mode == 0
    assert args.sac_learning_rate is None


def test_run_multigame_cli_accepts_bbf_agent():
    args = parse_args(["--games", "pong", "--agent", "bbf"])
    assert args.agent == "bbf"
    assert args.bbf_learning_starts == 2000
    assert args.bbf_buffer_size == 200000
    assert args.bbf_use_per == 1
    assert args.bbf_learning_rate == pytest.approx(1e-4)
    assert args.bbf_shrink_perturb_keys == "encoder,transition_model"


def test_run_multigame_cli_accepts_delay_target_sweep_flags():
    args = parse_args(
        [
            "--games",
            "pong",
            "--agent",
            "delay_target",
            "--delay-target-base-width",
            "112",
            "--delay-target-temperature-log2",
            "-4",
            "--delay-target-greedy-ramp",
            "150000",
            "--delay-target-multisteps-max",
            "32",
            "--delay-target-td-lambda",
            "0.85",
            "--delay-target-train-batch",
            "48",
            "--delay-target-online-batch",
            "6",
            "--delay-target-online-loss-scale",
            "1.75",
            "--delay-target-train-steps",
            "5",
        ]
    )
    assert args.delay_target_base_width == 112
    assert args.delay_target_temperature_log2 == -4
    assert args.delay_target_greedy_ramp == 150000
    assert args.delay_target_multisteps_max == 32
    assert args.delay_target_td_lambda == pytest.approx(0.85)
    assert args.delay_target_train_batch == 48
    assert args.delay_target_online_batch == 6
    assert args.delay_target_online_loss_scale == pytest.approx(1.75)
    assert args.delay_target_train_steps == 5


def test_run_multigame_cli_accepts_rainbow_dqn_sweep_flags():
    args = parse_args(
        [
            "--games",
            "pong",
            "--agent",
            "rainbow_dqn",
            "--rainbow-dqn-learning-rate",
            "0.0003",
            "--rainbow-dqn-train-start",
            "25000",
            "--rainbow-dqn-batch-size",
            "64",
            "--rainbow-dqn-buffer-size",
            "200000",
            "--rainbow-dqn-target-update-freq",
            "4000",
            "--rainbow-dqn-n-step",
            "5",
            "--rainbow-dqn-gamma",
            "0.995",
            "--rainbow-dqn-grad-clip",
            "none",
            "--rainbow-dqn-priority-alpha",
            "0.6",
            "--rainbow-dqn-priority-beta",
            "0.7",
        ]
    )
    assert args.rainbow_dqn_learning_rate == pytest.approx(3e-4)
    assert args.rainbow_dqn_train_start == 25000
    assert args.rainbow_dqn_batch_size == 64
    assert args.rainbow_dqn_buffer_size == 200000
    assert args.rainbow_dqn_target_update_freq == 4000
    assert args.rainbow_dqn_n_step == 5
    assert args.rainbow_dqn_gamma == pytest.approx(0.995)
    assert args.rainbow_dqn_grad_clip is None
    assert args.rainbow_dqn_priority_alpha == pytest.approx(0.6)
    assert args.rainbow_dqn_priority_beta == pytest.approx(0.7)


def test_run_multigame_cli_accepts_sac_sweep_flags():
    args = parse_args(
        [
            "--games",
            "pong",
            "--agent",
            "sac",
            "--sac-learning-rate",
            "0.0003",
            "--sac-learning-starts",
            "5000",
            "--sac-batch-size",
            "128",
            "--sac-buffer-size",
            "200000",
            "--sac-gradient-steps",
            "4",
            "--sac-tau",
            "0.01",
            "--sac-target-entropy-scale",
            "0.75",
            "--sac-gamma",
            "0.995",
            "--sac-train-freq",
            "2",
            "--sac-frame-skip",
            "6",
        ]
    )
    assert args.sac_learning_rate == pytest.approx(3e-4)
    assert args.sac_learning_starts == 5000
    assert args.sac_batch_size == 128
    assert args.sac_buffer_size == 200000
    assert args.sac_gradient_steps == 4
    assert args.sac_tau == pytest.approx(0.01)
    assert args.sac_target_entropy_scale == pytest.approx(0.75)
    assert args.sac_gamma == pytest.approx(0.995)
    assert args.sac_train_freq == 2
    assert args.sac_frame_skip == 6


def test_run_multigame_cli_accepts_bbf_sweep_flags():
    args = parse_args(
        [
            "--games",
            "pong",
            "--agent",
            "bbf",
            "--bbf-learning-rate",
            "0.0003",
            "--bbf-encoder-learning-rate",
            "0.0002",
            "--bbf-spr-weight",
            "7.5",
            "--bbf-jumps",
            "3",
            "--bbf-target-update-tau",
            "0.01",
            "--bbf-update-horizon",
            "5",
            "--bbf-max-update-horizon",
            "9",
            "--bbf-min-gamma",
            "0.98",
            "--bbf-cycle-steps",
            "20000",
            "--bbf-shrink-factor",
            "0.6",
            "--bbf-perturb-factor",
            "0.4",
            "--bbf-shrink-perturb-keys",
            "encoder",
        ]
    )
    assert args.bbf_learning_rate == pytest.approx(3e-4)
    assert args.bbf_encoder_learning_rate == pytest.approx(2e-4)
    assert args.bbf_spr_weight == pytest.approx(7.5)
    assert args.bbf_jumps == 3
    assert args.bbf_target_update_tau == pytest.approx(0.01)
    assert args.bbf_update_horizon == 5
    assert args.bbf_max_update_horizon == 9
    assert args.bbf_min_gamma == pytest.approx(0.98)
    assert args.bbf_cycle_steps == 20000
    assert args.bbf_shrink_factor == pytest.approx(0.6)
    assert args.bbf_perturb_factor == pytest.approx(0.4)
    assert args.bbf_shrink_perturb_keys == "encoder"


def test_run_multigame_cli_accepts_bbf_native_reset_flags():
    args = parse_args(
        [
            "--games",
            "pong",
            "--agent",
            "bbf",
            "--bbf-native-reset-semantics",
            "1",
            "--bbf-noop-reset-max",
            "17",
            "--bbf-fire-reset",
            "0",
        ]
    )
    assert args.agent == "bbf"
    assert args.bbf_native_reset_semantics == 1
    assert args.bbf_noop_reset_max == 17
    assert args.bbf_fire_reset == 0


def test_run_multigame_config_parses_ppo_agent_config(tmp_path):
    config_path = tmp_path / "cfg_ppo.json"
    payload = {
        "games": ["pong"],
        "agent": "ppo",
        "agent_config": {
            "learning_rate": 0.0001,
            "clip_range": 0.15,
            "rollout_steps": 64,
            "train_interval": 64,
            "batch_size": 16,
            "epochs": 3,
            "device": "cpu",
            "grayscale": 1,
            "decision_interval": 6,
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    args = parse_args(["--config", str(config_path)])
    assert args.agent == "ppo"
    assert args.ppo_lr == pytest.approx(1e-4)
    assert args.ppo_clip_range == pytest.approx(0.15)
    assert args.ppo_rollout_steps == 64
    assert args.ppo_train_interval == 64
    assert args.ppo_batch_size == 16
    assert args.ppo_epochs == 3
    assert args.ppo_device == "cpu"
    assert args.ppo_grayscale == 1
    assert args.ppo_decision_interval == 6


def test_run_multigame_config_parses_dqn_agent_config(tmp_path):
    config_path = tmp_path / "cfg_dqn.json"
    payload = {
        "games": ["pong"],
        "agent": "dqn",
        "runner_mode": "carmack_compat",
        "decision_interval": 1,
        "agent_config": {
            "gpu": 2,
            "load_file": "/tmp/dqn.model",
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    args = parse_args(["--config", str(config_path)])
    assert args.agent == "dqn"
    assert args.runner_mode == "carmack_compat"
    assert args.decision_interval == 1
    assert args.roboatari_dqn_gpu == 2
    assert args.roboatari_dqn_load_file == "/tmp/dqn.model"


def test_run_multigame_config_parses_rainbow_dqn_agent_config(tmp_path):
    config_path = tmp_path / "cfg_rainbow.json"
    payload = {
        "games": ["pong"],
        "agent": "rainbow_dqn",
        "runner_mode": "carmack_compat",
        "decision_interval": 1,
        "agent_config": {
            "gpu": 3,
            "load_file": "/tmp/rainbow.model",
            "learning_rate": 0.0003,
            "train_start": 25000,
            "batch_size": 64,
            "buffer_size": 200000,
            "target_update_freq": 4000,
            "n_step": 5,
            "gamma": 0.995,
            "grad_clip": None,
            "priority_alpha": 0.6,
            "priority_beta": 0.7,
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    args = parse_args(["--config", str(config_path)])
    assert args.agent == "rainbow_dqn"
    assert args.runner_mode == "carmack_compat"
    assert args.decision_interval == 1
    assert args.rainbow_dqn_gpu == 3
    assert args.rainbow_dqn_load_file == "/tmp/rainbow.model"
    assert args.rainbow_dqn_learning_rate == pytest.approx(3e-4)
    assert args.rainbow_dqn_train_start == 25000
    assert args.rainbow_dqn_batch_size == 64
    assert args.rainbow_dqn_buffer_size == 200000
    assert args.rainbow_dqn_target_update_freq == 4000
    assert args.rainbow_dqn_n_step == 5
    assert args.rainbow_dqn_gamma == pytest.approx(0.995)
    assert args.rainbow_dqn_grad_clip is None
    assert args.rainbow_dqn_priority_alpha == pytest.approx(0.6)
    assert args.rainbow_dqn_priority_beta == pytest.approx(0.7)


def test_run_multigame_config_parses_sac_agent_config(tmp_path):
    config_path = tmp_path / "cfg_sac.json"
    payload = {
        "games": ["pong"],
        "agent": "sac",
        "runner_mode": "carmack_compat",
        "decision_interval": 1,
        "agent_config": {
            "gpu": 1,
            "load_file": "/tmp/sac.model",
            "eval_mode": 1,
            "learning_rate": 0.0003,
            "learning_starts": 5000,
            "batch_size": 128,
            "buffer_size": 200000,
            "gradient_steps": 4,
            "tau": 0.01,
            "target_entropy_scale": 0.75,
            "gamma": 0.995,
            "train_freq": 2,
            "frame_skip": 6,
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    args = parse_args(["--config", str(config_path)])
    assert args.agent == "sac"
    assert args.runner_mode == "carmack_compat"
    assert args.decision_interval == 1
    assert args.sac_gpu == 1
    assert args.sac_load_file == "/tmp/sac.model"
    assert args.sac_eval_mode == 1
    assert args.sac_learning_rate == pytest.approx(3e-4)
    assert args.sac_learning_starts == 5000
    assert args.sac_batch_size == 128
    assert args.sac_buffer_size == 200000
    assert args.sac_gradient_steps == 4
    assert args.sac_tau == pytest.approx(0.01)
    assert args.sac_target_entropy_scale == pytest.approx(0.75)
    assert args.sac_gamma == pytest.approx(0.995)
    assert args.sac_train_freq == 2
    assert args.sac_frame_skip == 6


def test_run_multigame_config_parses_bbf_agent_config(tmp_path):
    config_path = tmp_path / "cfg_bbf.json"
    payload = {
        "games": ["pong"],
        "agent": "bbf",
        "runner_mode": "carmack_compat",
        "decision_interval": 1,
        "full_action_space": 1,
        "agent_config": {
            "learning_starts": 123,
            "buffer_size": 54321,
            "batch_size": 16,
            "replay_ratio": 8,
            "learning_rate": 0.0003,
            "encoder_learning_rate": 0.0002,
            "spr_weight": 7.5,
            "jumps": 3,
            "target_update_tau": 0.01,
            "update_horizon": 5,
            "max_update_horizon": 9,
            "min_gamma": 0.98,
            "cycle_steps": 20000,
            "shrink_factor": 0.6,
            "perturb_factor": 0.4,
            "shrink_perturb_keys": "encoder",
            "reset_interval": 2500,
            "no_resets_after": 9000,
            "use_per": 0,
            "use_amp": 1,
            "torch_compile": 1,
            "native_reset_semantics": 1,
            "noop_reset_max": 9,
            "fire_reset": 0,
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    args = parse_args(["--config", str(config_path)])
    assert args.agent == "bbf"
    assert args.runner_mode == "carmack_compat"
    assert args.decision_interval == 1
    assert args.bbf_learning_starts == 123
    assert args.bbf_buffer_size == 54321
    assert args.bbf_batch_size == 16
    assert args.bbf_replay_ratio == 8
    assert args.bbf_learning_rate == pytest.approx(3e-4)
    assert args.bbf_encoder_learning_rate == pytest.approx(2e-4)
    assert args.bbf_spr_weight == pytest.approx(7.5)
    assert args.bbf_jumps == 3
    assert args.bbf_target_update_tau == pytest.approx(0.01)
    assert args.bbf_update_horizon == 5
    assert args.bbf_max_update_horizon == 9
    assert args.bbf_min_gamma == pytest.approx(0.98)
    assert args.bbf_cycle_steps == 20000
    assert args.bbf_shrink_factor == pytest.approx(0.6)
    assert args.bbf_perturb_factor == pytest.approx(0.4)
    assert args.bbf_shrink_perturb_keys == "encoder"
    assert args.bbf_reset_interval == 2500
    assert args.bbf_no_resets_after == 9000
    assert args.bbf_use_per == 0
    assert args.bbf_use_amp == 1
    assert args.bbf_torch_compile == 1
    assert args.bbf_native_reset_semantics == 1
    assert args.bbf_noop_reset_max == 9
    assert args.bbf_fire_reset == 0


def test_validate_args_ppo_requires_positive_decision_interval():
    args = parse_args(["--games", "pong", "--agent", "ppo", "--ppo-decision-interval", "0"])
    with pytest.raises(ValueError, match=r"--ppo-decision-interval must be > 0"):
        validate_args(args)


def test_validate_args_dqn_requires_carmack_mode():
    args = parse_args(["--games", "pong", "--agent", "dqn"])
    with pytest.raises(ValueError, match=r"agent=dqn currently requires --runner-mode carmack_compat"):
        validate_args(args)


def test_validate_args_rainbow_dqn_requires_carmack_mode():
    args = parse_args(["--games", "pong", "--agent", "rainbow_dqn"])
    with pytest.raises(ValueError, match=r"agent=rainbow_dqn currently requires --runner-mode carmack_compat"):
        validate_args(args)


def test_validate_args_sac_requires_carmack_mode():
    args = parse_args(["--games", "pong", "--agent", "sac"])
    with pytest.raises(ValueError, match=r"agent=sac currently requires --runner-mode carmack_compat"):
        validate_args(args)


def test_validate_args_bbf_requires_carmack_mode():
    args = parse_args(["--games", "pong", "--agent", "bbf", "--full-action-space", "1"])
    with pytest.raises(ValueError, match=r"agent=bbf currently requires --runner-mode carmack_compat"):
        validate_args(args)


def test_validate_args_bbf_requires_full_action_space():
    args = parse_args(
        [
            "--games",
            "pong",
            "--agent",
            "bbf",
            "--runner-mode",
            "carmack_compat",
            "--decision-interval",
            "1",
            "--full-action-space",
            "0",
        ]
    )
    with pytest.raises(ValueError, match=r"agent=bbf requires --full-action-space 1"):
        validate_args(args)


def test_validate_args_bbf_requires_real_time_mode_off():
    args = parse_args(
        [
            "--games",
            "pong",
            "--agent",
            "bbf",
            "--runner-mode",
            "carmack_compat",
            "--decision-interval",
            "1",
            "--full-action-space",
            "1",
            "--real-time-mode",
            "1",
        ]
    )
    with pytest.raises(ValueError, match=r"agent=bbf currently requires --real-time-mode 0"):
        validate_args(args)


def test_validate_args_bbf_native_reset_semantics_only_valid_for_bbf():
    args = parse_args(["--games", "pong", "--agent", "random", "--bbf-native-reset-semantics", "1"])
    with pytest.raises(ValueError, match=r"--bbf-native-reset-semantics is only valid"):
        validate_args(args)


def test_validate_args_bbf_noop_reset_max_only_valid_for_bbf():
    args = parse_args(["--games", "pong", "--agent", "random", "--bbf-noop-reset-max", "17"])
    with pytest.raises(ValueError, match=r"--bbf-noop-reset-max is only valid"):
        validate_args(args)


def test_validate_args_bbf_fire_reset_only_valid_for_bbf():
    args = parse_args(["--games", "pong", "--agent", "random", "--bbf-fire-reset", "0"])
    with pytest.raises(ValueError, match=r"--bbf-fire-reset is only valid"):
        validate_args(args)


def test_resolve_bbf_runtime_settings_native_reset_semantics_forces_sticky_zero():
    args = Namespace(
        agent="bbf",
        bbf_native_reset_semantics=1,
        bbf_noop_reset_max=17,
        bbf_fire_reset=1,
        sticky=0.25,
        full_action_space=1,
    )
    settings = _resolve_bbf_runtime_settings(args)
    assert settings["runtime_mode"] == "native_reset_semantics"
    assert settings["native_reset_semantics_requested"] is True
    assert settings["native_reset_semantics_enabled"] is True
    assert settings["sticky_effective"] == pytest.approx(0.0)
    assert settings["full_action_space_effective"] is True
    assert settings["action_space_mode"] == "canonical_full"
    assert settings["noop_reset_max"] == 17
    assert settings["fire_reset_enabled"] is True


def test_resolve_bbf_log_visibility_enables_heartbeat_when_both_logs_disabled():
    args = Namespace(agent="bbf", log_episode_every=0, log_train_every=0)
    settings = _resolve_bbf_log_visibility(args)
    assert settings["requested_log_episode_every"] == 0
    assert settings["requested_log_train_every"] == 0
    assert settings["effective_log_episode_every"] == 0
    assert settings["effective_log_train_every"] == BBF_MULTIGAME_HEARTBEAT_TRAIN_INTERVAL
    assert settings["progress_heartbeat_active"] is True
    assert settings["progress_heartbeat_source"] == "bbf_fallback_train_interval"


def test_resolve_bbf_log_visibility_respects_explicit_intervals():
    args = Namespace(agent="bbf", log_episode_every=0, log_train_every=1234)
    settings = _resolve_bbf_log_visibility(args)
    assert settings["effective_log_train_every"] == 1234
    assert settings["progress_heartbeat_active"] is False
    assert settings["progress_heartbeat_source"] == "requested"


def test_resolve_bbf_log_visibility_no_fallback_for_non_bbf():
    args = Namespace(agent="random", log_episode_every=0, log_train_every=0)
    settings = _resolve_bbf_log_visibility(args)
    assert settings["effective_log_episode_every"] == 0
    assert settings["effective_log_train_every"] == 0
    assert settings["progress_heartbeat_active"] is False
    assert settings["progress_heartbeat_source"] == "not_bbf"


def test_build_config_payload_records_bbf_runtime_fields():
    args = parse_args(
        [
            "--games",
            "pong",
            "--agent",
            "bbf",
            "--runner-mode",
            "carmack_compat",
            "--decision-interval",
            "1",
            "--full-action-space",
            "1",
            "--bbf-native-reset-semantics",
            "1",
            "--bbf-noop-reset-max",
            "11",
            "--bbf-fire-reset",
            "1",
        ]
    )
    schedule = Schedule(
        ScheduleConfig(games=["pong"], base_visit_frames=2, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    runner_config = CarmackMultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=int(args.delay),
        default_action_idx=int(args.default_action_idx),
        episode_log_interval=0,
        train_log_interval=int(BBF_MULTIGAME_HEARTBEAT_TRAIN_INTERVAL),
        include_timestamps=bool(args.timestamps),
        global_action_set=tuple(range(18)),
        real_time_mode=bool(args.real_time_mode),
        real_time_fps=float(args.real_time_fps),
    )
    bbf_runtime = _resolve_bbf_runtime_settings(args)
    bbf_runtime.update(_resolve_bbf_log_visibility(args))
    bbf_runtime["fire_reset_supported"] = True
    bbf_runtime["fire_reset_supported_by_game"] = {"pong": True}
    payload = build_config_payload(
        args=args,
        games=["pong"],
        run_dir=Path("runs/test"),
        schedule=schedule,
        runner_config=runner_config,
        resolved_action_sets={"pong": [0, 1, 2]},
        agent_config={"buffer_size": 200000},
        config_file_data={},
        bbf_runtime=bbf_runtime,
    )

    assert payload["bbf_config"]["native_reset_semantics"] is True
    assert payload["bbf_config"]["noop_reset_max"] == 11
    assert payload["bbf_config"]["fire_reset"] is True
    assert payload["bbf_runtime"]["runtime_mode"] == "native_reset_semantics"
    assert payload["bbf_runtime"]["native_reset_semantics_enabled"] is True
    assert payload["bbf_runtime"]["sticky_effective"] == pytest.approx(0.0)
    assert payload["bbf_runtime"]["full_action_space_effective"] is True
    assert payload["bbf_runtime"]["noop_reset_max_effective"] == 11
    assert payload["bbf_runtime"]["fire_reset_enabled"] is True
    assert payload["bbf_runtime"]["fire_reset_supported"] is True
    assert payload["bbf_runtime"]["fire_reset_supported_by_game"] == {"pong": True}
    assert payload["log_episode_every"] == 0
    assert payload["log_train_every"] == 0
    assert payload["runner_config"]["episode_log_interval"] == 0
    assert payload["runner_config"]["train_log_interval"] == BBF_MULTIGAME_HEARTBEAT_TRAIN_INTERVAL
    assert payload["bbf_runtime"]["requested_log_episode_every"] == 0
    assert payload["bbf_runtime"]["requested_log_train_every"] == 0
    assert payload["bbf_runtime"]["effective_log_episode_every"] == 0
    assert payload["bbf_runtime"]["effective_log_train_every"] == BBF_MULTIGAME_HEARTBEAT_TRAIN_INTERVAL
    assert payload["bbf_runtime"]["progress_heartbeat_active"] is True
    assert payload["bbf_runtime"]["progress_heartbeat_source"] == "bbf_fallback_train_interval"


def test_build_config_payload_records_exposed_sweep_family_config_fields():
    args = parse_args(
        [
            "--games",
            "pong",
            "--agent",
            "sac",
            "--runner-mode",
            "carmack_compat",
            "--decision-interval",
            "1",
            "--delay-target-base-width",
            "112",
            "--delay-target-temperature-log2",
            "-4",
            "--delay-target-greedy-ramp",
            "150000",
            "--delay-target-multisteps-max",
            "32",
            "--delay-target-td-lambda",
            "0.85",
            "--delay-target-train-batch",
            "48",
            "--delay-target-online-batch",
            "6",
            "--delay-target-online-loss-scale",
            "1.75",
            "--delay-target-train-steps",
            "5",
            "--rainbow-dqn-learning-rate",
            "0.0003",
            "--rainbow-dqn-train-start",
            "25000",
            "--rainbow-dqn-batch-size",
            "64",
            "--rainbow-dqn-buffer-size",
            "200000",
            "--rainbow-dqn-target-update-freq",
            "4000",
            "--rainbow-dqn-n-step",
            "5",
            "--rainbow-dqn-gamma",
            "0.995",
            "--rainbow-dqn-grad-clip",
            "none",
            "--rainbow-dqn-priority-alpha",
            "0.6",
            "--rainbow-dqn-priority-beta",
            "0.7",
            "--sac-learning-rate",
            "0.0003",
            "--sac-learning-starts",
            "5000",
            "--sac-batch-size",
            "128",
            "--sac-buffer-size",
            "200000",
            "--sac-gradient-steps",
            "4",
            "--sac-tau",
            "0.01",
            "--sac-target-entropy-scale",
            "0.75",
            "--sac-gamma",
            "0.995",
            "--sac-train-freq",
            "2",
            "--sac-frame-skip",
            "6",
            "--bbf-learning-rate",
            "0.0003",
            "--bbf-encoder-learning-rate",
            "0.0002",
            "--bbf-spr-weight",
            "7.5",
            "--bbf-jumps",
            "3",
            "--bbf-target-update-tau",
            "0.01",
            "--bbf-update-horizon",
            "5",
            "--bbf-max-update-horizon",
            "9",
            "--bbf-min-gamma",
            "0.98",
            "--bbf-cycle-steps",
            "20000",
            "--bbf-shrink-factor",
            "0.6",
            "--bbf-perturb-factor",
            "0.4",
            "--bbf-shrink-perturb-keys",
            "encoder",
        ]
    )
    schedule = Schedule(
        ScheduleConfig(games=["pong"], base_visit_frames=2, num_cycles=1, seed=0, jitter_pct=0.0, min_visit_frames=1)
    )
    runner_config = CarmackMultiGameRunnerConfig(
        decision_interval=1,
        delay_frames=int(args.delay),
        default_action_idx=int(args.default_action_idx),
        episode_log_interval=0,
        train_log_interval=0,
        include_timestamps=bool(args.timestamps),
        global_action_set=tuple(range(18)),
        real_time_mode=bool(args.real_time_mode),
        real_time_fps=float(args.real_time_fps),
    )
    payload = build_config_payload(
        args=args,
        games=["pong"],
        run_dir=Path("runs/test"),
        schedule=schedule,
        runner_config=runner_config,
        resolved_action_sets={"pong": [0, 1, 2]},
        agent_config={"base_width": 112},
        config_file_data={},
    )
    assert payload["delay_target_config"]["base_width"] == 112
    assert payload["delay_target_config"]["temperature_log2"] == -4
    assert payload["delay_target_config"]["greedy_ramp"] == 150000
    assert payload["delay_target_config"]["multisteps_max"] == 32
    assert payload["delay_target_config"]["td_lambda"] == pytest.approx(0.85)
    assert payload["delay_target_config"]["train_batch"] == 48
    assert payload["delay_target_config"]["online_batch"] == 6
    assert payload["delay_target_config"]["online_loss_scale"] == pytest.approx(1.75)
    assert payload["delay_target_config"]["train_steps"] == 5
    assert payload["rainbow_dqn_config"]["learning_rate"] == pytest.approx(3e-4)
    assert payload["rainbow_dqn_config"]["grad_clip"] is None
    assert payload["rainbow_dqn_config"]["priority_beta"] == pytest.approx(0.7)
    assert payload["sac_config"]["learning_rate"] == pytest.approx(3e-4)
    assert payload["sac_config"]["target_entropy_scale"] == pytest.approx(0.75)
    assert payload["sac_config"]["frame_skip"] == 6
    assert payload["bbf_config"]["learning_rate"] == pytest.approx(3e-4)
    assert payload["bbf_config"]["encoder_learning_rate"] == pytest.approx(2e-4)
    assert payload["bbf_config"]["spr_weight"] == pytest.approx(7.5)
    assert payload["bbf_config"]["jumps"] == 3
    assert payload["bbf_config"]["target_update_tau"] == pytest.approx(0.01)
    assert payload["bbf_config"]["update_horizon"] == 5
    assert payload["bbf_config"]["max_update_horizon"] == 9
    assert payload["bbf_config"]["min_gamma"] == pytest.approx(0.98)
    assert payload["bbf_config"]["cycle_steps"] == 20000
    assert payload["bbf_config"]["shrink_factor"] == pytest.approx(0.6)
    assert payload["bbf_config"]["perturb_factor"] == pytest.approx(0.4)
    assert payload["bbf_config"]["shrink_perturb_keys"] == "encoder"


def test_validate_args_requires_positive_real_time_fps():
    args = parse_args(["--games", "pong", "--real-time-fps", "0"])
    with pytest.raises(ValueError, match=r"--real-time-fps must be > 0"):
        validate_args(args)


def test_run_multigame_config_ppo_fields_apply_when_agent_overridden_on_cli(tmp_path):
    config_path = tmp_path / "cfg_mixed_agent.json"
    payload = {
        "games": ["pong"],
        "agent": "tinydqn",
        "agent_config": {
            "learning_rate": 0.0002,
            "rollout_steps": 96,
            "train_interval": 96,
            "batch_size": 24,
            "epochs": 2,
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    args = parse_args(["--config", str(config_path), "--agent", "ppo"])
    assert args.agent == "ppo"
    assert args.ppo_lr == pytest.approx(2e-4)
    assert args.ppo_rollout_steps == 96
    assert args.ppo_train_interval == 96
    assert args.ppo_batch_size == 24
    assert args.ppo_epochs == 2
