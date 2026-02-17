from __future__ import annotations

import json

from benchmark.run_multigame import collect_agent_stats, parse_args


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
