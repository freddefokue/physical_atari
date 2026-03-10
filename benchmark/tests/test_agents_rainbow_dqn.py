from __future__ import annotations

import argparse

import numpy as np
import pytest

import benchmark.agents_rainbow_dqn as agents_rainbow_dqn
from benchmark.run_multigame import build_agent


def _rainbow_args() -> argparse.Namespace:
    return argparse.Namespace(
        agent="rainbow_dqn",
        seed=0,
        rainbow_dqn_gpu=0,
        rainbow_dqn_load_file=None,
        logdir="./runs/v1",
    )


@pytest.mark.skipif(not agents_rainbow_dqn._TORCH_AVAILABLE, reason="torch unavailable")
def test_build_agent_rainbow_dqn_returns_agent_and_config():
    agent, cfg = build_agent(_rainbow_args(), num_actions=6, total_frames=256)
    assert isinstance(agent, agents_rainbow_dqn.RainbowDQNAgent)
    assert isinstance(cfg, dict)
    assert cfg["gpu"] == 0
    assert cfg["n_step"] == 3


@pytest.mark.skipif(not agents_rainbow_dqn._TORCH_AVAILABLE, reason="torch unavailable")
def test_rainbow_dqn_uses_transition_obs_for_replay_and_reset_obs_for_next_action():
    agent = agents_rainbow_dqn.RainbowDQNAgent(
        data_dir="./runs",
        seed=0,
        num_actions=4,
        total_frames=128,
        config=agents_rainbow_dqn.RainbowDQNConfig(train_start=9999, batch_size=4),
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
    assert float(np.mean(agent.replay.next_states[0, -1])) == pytest.approx(80.0, abs=1.0)
    assert float(np.mean(agent._last_state[-1])) == pytest.approx(200.0, abs=1.0)  # pylint: disable=protected-access


@pytest.mark.skipif(not agents_rainbow_dqn._TORCH_AVAILABLE, reason="torch unavailable")
def test_rainbow_dqn_actions_stay_in_bounds():
    agent = agents_rainbow_dqn.RainbowDQNAgent(
        data_dir="./runs",
        seed=3,
        num_actions=5,
        total_frames=128,
        config=agents_rainbow_dqn.RainbowDQNConfig(train_start=9999, batch_size=4),
    )
    obs = np.zeros((210, 160, 3), dtype=np.uint8)
    actions = [agent.frame(obs, reward=0.0, boundary={"terminated": False, "truncated": False}) for _ in range(8)]
    assert all(0 <= int(action) < 5 for action in actions)


@pytest.mark.skipif(not agents_rainbow_dqn._TORCH_AVAILABLE, reason="torch unavailable")
def test_rainbow_dqn_step_bridge_preserves_explicit_pulse_and_reset_observation():
    agent = agents_rainbow_dqn.RainbowDQNAgent(
        data_dir="./runs",
        seed=1,
        num_actions=4,
        total_frames=128,
        config=agents_rainbow_dqn.RainbowDQNConfig(train_start=9999, batch_size=4),
    )
    first_obs = np.zeros((210, 160, 3), dtype=np.uint8)
    transition_obs = np.full((210, 160, 3), fill_value=60, dtype=np.uint8)
    reset_obs = np.full((210, 160, 3), fill_value=170, dtype=np.uint8)

    agent.frame(first_obs, reward=0.0, boundary={"terminated": False, "truncated": False})
    action = agent.step(
        obs_rgb=reset_obs,
        reward=1.0,
        terminated=False,
        truncated=False,
        info={
            "end_of_episode_pulse": True,
            "transition_obs_rgb": transition_obs,
            "reset_obs_rgb": reset_obs,
            "global_frame_idx": 7,
        },
    )

    assert 0 <= int(action) < 4
    assert agent.replay.size == 1
    assert float(np.mean(agent.replay.next_states[0, -1])) == pytest.approx(60.0, abs=1.0)
    assert float(np.mean(agent._last_state[-1])) == pytest.approx(170.0, abs=1.0)  # pylint: disable=protected-access


@pytest.mark.skipif(not agents_rainbow_dqn._TORCH_AVAILABLE, reason="torch unavailable")
def test_rainbow_dqn_load_model_accepts_wrapped_checkpoint_schema(tmp_path):
    agent = agents_rainbow_dqn.RainbowDQNAgent(
        data_dir="./runs",
        seed=0,
        num_actions=4,
        total_frames=128,
        config=agents_rainbow_dqn.RainbowDQNConfig(train_start=9999, batch_size=4),
    )
    checkpoint_path = tmp_path / "rainbow_wrapped.pth"
    original_weight = next(agent.network.parameters()).detach().clone()
    agents_rainbow_dqn.torch.save({"network": agent.network.state_dict()}, checkpoint_path)

    with agents_rainbow_dqn.torch.no_grad():
        next(agent.network.parameters()).zero_()

    agent.load_model(str(checkpoint_path))
    restored_weight = next(agent.network.parameters()).detach()
    assert agents_rainbow_dqn.torch.allclose(restored_weight, original_weight)


@pytest.mark.skipif(not agents_rainbow_dqn._TORCH_AVAILABLE, reason="torch unavailable")
def test_rainbow_dqn_init_raises_for_missing_checkpoint_path(tmp_path):
    missing_path = tmp_path / "missing_rainbow.pth"
    with pytest.raises(FileNotFoundError, match="Rainbow DQN checkpoint not found"):
        agents_rainbow_dqn.RainbowDQNAgent(
            data_dir="./runs",
            seed=0,
            num_actions=4,
            total_frames=128,
            config=agents_rainbow_dqn.RainbowDQNConfig(load_file=str(missing_path)),
        )


def test_rainbow_dqn_class_raises_clear_error_when_torch_forced_missing(monkeypatch):
    monkeypatch.setattr(agents_rainbow_dqn, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(agents_rainbow_dqn, "_TORCH_IMPORT_ERROR", ImportError("forced missing torch"))
    with pytest.raises(ImportError, match="agent=rainbow_dqn requires torch"):
        agents_rainbow_dqn.RainbowDQNAgent(
            data_dir="./runs",
            seed=0,
            num_actions=4,
            total_frames=128,
            config=agents_rainbow_dqn.RainbowDQNConfig(),
        )
