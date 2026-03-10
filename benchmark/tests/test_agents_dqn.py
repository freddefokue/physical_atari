from __future__ import annotations

import argparse
import sys
import types

import numpy as np
import pytest

import benchmark.agents_dqn as agents_dqn
from benchmark.run_multigame import build_agent


def _dqn_args() -> argparse.Namespace:
    return argparse.Namespace(
        agent="dqn",
        seed=0,
        roboatari_dqn_gpu=0,
        roboatari_dqn_load_file=None,
        logdir="./runs/v1",
    )


@pytest.mark.skipif(not agents_dqn._TORCH_AVAILABLE, reason="torch unavailable")
def test_build_agent_dqn_returns_agent_and_config():
    agent, cfg = build_agent(_dqn_args(), num_actions=6, total_frames=256)
    assert isinstance(agent, agents_dqn.DQNAgent)
    assert isinstance(cfg, dict)
    assert cfg["gpu"] == 0
    assert cfg["buffer_size"] == 100_000


@pytest.mark.skipif(not agents_dqn._TORCH_AVAILABLE, reason="torch unavailable")
def test_dqn_uses_transition_obs_for_replay_and_reset_obs_for_next_action():
    agent = agents_dqn.DQNAgent(
        data_dir="./runs",
        seed=0,
        num_actions=4,
        total_frames=128,
        config=agents_dqn.DQNAgentConfig(train_start=9999, batch_size=4),
    )

    first_obs = np.zeros((210, 160, 3), dtype=np.uint8)
    transition_obs = np.full((210, 160, 3), fill_value=80, dtype=np.uint8)
    reset_obs = np.full((210, 160, 3), fill_value=200, dtype=np.uint8)

    first_action = agent.frame(first_obs, reward=0.0, boundary={"terminated": False, "truncated": False})
    second_action = agent.frame(
        reset_obs,
        reward=1.0,
        boundary={
            "terminated": True,
            "truncated": False,
            "end_of_episode_pulse": True,
            "transition_obs_rgb": transition_obs,
            "reset_obs_rgb": reset_obs,
        },
    )

    assert 0 <= int(first_action) < 4
    assert 0 <= int(second_action) < 4
    assert agent.replay.size == 1
    assert float(np.mean(agent.replay._next_states[0, -1])) == pytest.approx(80.0, abs=1.0)  # pylint: disable=protected-access
    assert float(np.mean(agent._last_state[-1])) == pytest.approx(200.0, abs=1.0)  # pylint: disable=protected-access


@pytest.mark.skipif(not agents_dqn._TORCH_AVAILABLE, reason="torch unavailable")
def test_dqn_actions_stay_in_bounds():
    agent = agents_dqn.DQNAgent(
        data_dir="./runs",
        seed=3,
        num_actions=5,
        total_frames=128,
        config=agents_dqn.DQNAgentConfig(train_start=9999, batch_size=4),
    )
    obs = np.zeros((210, 160, 3), dtype=np.uint8)
    actions = [agent.frame(obs, reward=0.0, boundary={"terminated": False, "truncated": False}) for _ in range(8)]
    assert all(0 <= int(action) < 5 for action in actions)


def test_dqn_class_raises_clear_error_when_torch_forced_missing(monkeypatch):
    monkeypatch.setattr(agents_dqn, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(agents_dqn, "_TORCH_IMPORT_ERROR", ImportError("forced missing torch"))
    with pytest.raises(ImportError, match="agent=dqn requires torch"):
        agents_dqn.DQNAgent(
            data_dir="./runs",
            seed=0,
            num_actions=4,
            total_frames=128,
            config=agents_dqn.DQNAgentConfig(),
        )


def test_build_agent_dqn_missing_module_import_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "benchmark.agents_dqn", types.ModuleType("benchmark.agents_dqn"))
    with pytest.raises(ImportError):
        build_agent(_dqn_args(), num_actions=6, total_frames=256)
