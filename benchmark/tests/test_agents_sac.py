from __future__ import annotations

import argparse

import numpy as np
import pytest

import benchmark.agents_sac as agents_sac
from benchmark.run_multigame import build_agent


def _sac_args() -> argparse.Namespace:
    return argparse.Namespace(
        agent="sac",
        seed=0,
        sac_gpu=0,
        sac_load_file=None,
        sac_eval_mode=0,
        logdir="./runs/v1",
    )


@pytest.mark.skipif(not agents_sac._TORCH_AVAILABLE, reason="torch unavailable")
def test_build_agent_sac_returns_agent_and_config():
    agent, cfg = build_agent(_sac_args(), num_actions=6, total_frames=256)
    assert isinstance(agent, agents_sac.SACAgent)
    assert isinstance(cfg, dict)
    assert cfg["gpu"] == 0
    assert cfg["frame_skip"] == 4


@pytest.mark.skipif(not agents_sac._TORCH_AVAILABLE, reason="torch unavailable")
def test_sac_uses_transition_obs_for_replay_and_reset_obs_for_next_action():
    agent = agents_sac.SACAgent(
        data_dir="./runs",
        seed=0,
        num_actions=4,
        total_frames=128,
        config=agents_sac.SACAgentConfig(frame_skip=1, n_stack=1, learning_starts=9999, batch_size=4),
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
    assert float(np.mean(agent.replay._next_obs[0, -1])) == pytest.approx(80.0, abs=1.0)  # pylint: disable=protected-access
    assert float(np.mean(agent._last_obs[-1])) == pytest.approx(200.0, abs=1.0)  # pylint: disable=protected-access


@pytest.mark.skipif(not agents_sac._TORCH_AVAILABLE, reason="torch unavailable")
def test_sac_clears_episode_state_on_off_cadence_done():
    agent = agents_sac.SACAgent(
        data_dir="./runs",
        seed=0,
        num_actions=4,
        total_frames=128,
        config=agents_sac.SACAgentConfig(frame_skip=4, n_stack=1, learning_starts=9999, batch_size=4),
    )
    agent._last_obs = np.full((1, 128, 128), fill_value=33, dtype=np.uint8)  # pylint: disable=protected-access
    agent._last_action = 2  # pylint: disable=protected-access
    agent._accumulated_reward = 5.0  # pylint: disable=protected-access
    agent._append_frame(np.full((128, 128), fill_value=11, dtype=np.uint8))  # pylint: disable=protected-access

    reset_obs = np.full((210, 160, 3), fill_value=200, dtype=np.uint8)
    returned_action = agent.frame(
        reset_obs,
        reward=1.0,
        boundary={
            "terminated": True,
            "truncated": False,
            "end_of_episode_pulse": True,
            "transition_obs_rgb": np.full((210, 160, 3), fill_value=80, dtype=np.uint8),
            "reset_obs_rgb": reset_obs,
        },
    )

    assert returned_action == 0
    assert agent._last_obs is None  # pylint: disable=protected-access
    assert agent._last_action == 0  # pylint: disable=protected-access
    assert agent.replay.size == 0
    assert agent._accumulated_reward == pytest.approx(1.0)  # pylint: disable=protected-access
    assert float(np.mean(agent._current_obs_stack()[-1])) == pytest.approx(200.0, abs=1.0)  # pylint: disable=protected-access


@pytest.mark.skipif(not agents_sac._TORCH_AVAILABLE, reason="torch unavailable")
def test_sac_actions_stay_in_bounds():
    agent = agents_sac.SACAgent(
        data_dir="./runs",
        seed=3,
        num_actions=5,
        total_frames=128,
        config=agents_sac.SACAgentConfig(frame_skip=1, n_stack=1, learning_starts=9999, batch_size=4),
    )
    obs = np.zeros((210, 160, 3), dtype=np.uint8)
    actions = [agent.frame(obs, reward=0.0, boundary={"terminated": False, "truncated": False}) for _ in range(8)]
    assert all(0 <= int(action) < 5 for action in actions)


@pytest.mark.skipif(not agents_sac._TORCH_AVAILABLE, reason="torch unavailable")
def test_sac_load_model_accepts_original_cnn_checkpoint_schema(tmp_path):
    agent = agents_sac.SACAgent(
        data_dir="./runs",
        seed=0,
        num_actions=4,
        total_frames=128,
        config=agents_sac.SACAgentConfig(frame_skip=1, n_stack=1, learning_starts=9999, batch_size=4),
    )
    checkpoint_path = tmp_path / "sac_final.pth"
    original_encoder_weight = next(agent.encoder.parameters()).detach().clone()
    checkpoint = {
        "cnn": agent.encoder.state_dict(),
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_target": agent.q1_target.state_dict(),
        "q2_target": agent.q2_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
    }
    agents_sac.torch.save(checkpoint, checkpoint_path)

    with agents_sac.torch.no_grad():
        next(agent.encoder.parameters()).zero_()

    agent.load_model(str(checkpoint_path))
    restored_weight = next(agent.encoder.parameters()).detach()
    assert agents_sac.torch.allclose(restored_weight, original_encoder_weight)


@pytest.mark.skipif(not agents_sac._TORCH_AVAILABLE, reason="torch unavailable")
def test_sac_train_batch_decreases_alpha_when_policy_entropy_exceeds_target():
    agent = agents_sac.SACAgent(
        data_dir="./runs",
        seed=0,
        num_actions=4,
        total_frames=128,
        config=agents_sac.SACAgentConfig(
            frame_skip=1,
            n_stack=1,
            batch_size=4,
            learning_starts=0,
            learning_rate=1e-2,
        ),
    )

    with agents_sac.torch.no_grad():
        for module in (agent.encoder, agent.actor, agent.q1, agent.q2, agent.q1_target, agent.q2_target):
            for param in module.parameters():
                param.zero_()

    obs = np.zeros((1, 128, 128), dtype=np.uint8)
    for _ in range(4):
        agent.replay.add(obs=obs, action=0, reward=0.0, next_obs=obs, done=False)

    initial_alpha = float(agent._alpha(detach=True).item())  # pylint: disable=protected-access
    agent._train_batch()  # pylint: disable=protected-access

    assert agent.last_entropy == pytest.approx(np.log(4.0), rel=1e-4)
    assert agent.last_alpha < initial_alpha


def test_sac_class_raises_clear_error_when_torch_forced_missing(monkeypatch):
    monkeypatch.setattr(agents_sac, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(agents_sac, "_TORCH_IMPORT_ERROR", ImportError("forced missing torch"))
    with pytest.raises(ImportError, match="agent=sac requires torch"):
        agents_sac.SACAgent(
            data_dir="./runs",
            seed=0,
            num_actions=4,
            total_frames=128,
            config=agents_sac.SACAgentConfig(),
        )
