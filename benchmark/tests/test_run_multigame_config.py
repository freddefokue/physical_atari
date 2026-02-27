from __future__ import annotations

import json

import pytest

from benchmark.run_multigame import collect_agent_stats, parse_args, validate_args


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


def test_validate_args_ppo_requires_positive_decision_interval():
    args = parse_args(["--games", "pong", "--agent", "ppo", "--ppo-decision-interval", "0"])
    with pytest.raises(ValueError, match=r"--ppo-decision-interval must be > 0"):
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
