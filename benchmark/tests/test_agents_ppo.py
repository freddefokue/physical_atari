from __future__ import annotations

import argparse
import math
import sys
import types

import numpy as np
import pytest

import benchmark.agents_ppo as agents_ppo
from benchmark.carmack_multigame_runner import FrameFromStepAdapter
from benchmark.run_multigame import build_agent


def _ppo_args() -> argparse.Namespace:
    return argparse.Namespace(
        agent="ppo",
        seed=0,
        ppo_lr=2.5e-4,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_clip_range=0.2,
        ppo_ent_coef=0.01,
        ppo_vf_coef=0.5,
        ppo_max_grad_norm=0.5,
        ppo_rollout_steps=128,
        ppo_train_interval=128,
        ppo_batch_size=32,
        ppo_epochs=2,
        ppo_reward_clip=1.0,
        ppo_obs_size=84,
        ppo_frame_stack=4,
        ppo_grayscale=1,
        ppo_normalize_advantages=1,
        ppo_deterministic_actions=0,
        ppo_device="cpu",
    )


def test_build_agent_ppo_missing_dependency_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "benchmark.agents_ppo", types.ModuleType("benchmark.agents_ppo"))
    with pytest.raises(ImportError, match="agent=ppo requires torch"):
        build_agent(_ppo_args(), num_actions=6, total_frames=256)


@pytest.mark.skipif(not agents_ppo._TORCH_AVAILABLE, reason="torch unavailable")
def test_build_agent_ppo_returns_agent_and_config():
    agent, cfg = build_agent(_ppo_args(), num_actions=6, total_frames=256)
    assert isinstance(agent, agents_ppo.PPOAgent)
    assert isinstance(cfg, dict)
    assert cfg["learning_rate"] == pytest.approx(2.5e-4)
    assert cfg["rollout_steps"] == 128


@pytest.mark.skipif(not agents_ppo._TORCH_AVAILABLE, reason="torch unavailable")
def test_ppo_adapter_handles_carmack_boundary_payload():
    agent = agents_ppo.PPOAgent(action_space_n=5, seed=0, config=agents_ppo.PPOConfig(rollout_steps=9999, train_interval=9999))
    adapter = FrameFromStepAdapter(agent, decision_interval=1)

    obs = np.zeros((210, 160, 3), dtype=np.uint8)
    actions = []
    for idx in range(4):
        boundary = {
            "terminated": idx == 2,
            "truncated": False,
            "end_of_episode_pulse": idx == 2,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 1,
            "global_frame_idx": idx,
        }
        actions.append(adapter.frame(obs_rgb=obs, reward=0.0, boundary=boundary))

    assert len(actions) == 4
    assert all(0 <= int(action) < 5 for action in actions)


@pytest.mark.skipif(not agents_ppo._TORCH_AVAILABLE, reason="torch unavailable")
def test_ppo_actions_stay_in_bounds_under_nan_reward():
    agent = agents_ppo.PPOAgent(action_space_n=4, seed=123, config=agents_ppo.PPOConfig(rollout_steps=9999, train_interval=9999))
    obs = np.zeros((210, 160, 3), dtype=np.uint8)

    actions = [
        agent.step(obs, reward=float("nan"), terminated=False, truncated=False, info={}),
        agent.step(obs, reward=1.0, terminated=False, truncated=False, info={}),
        agent.step(obs, reward=-1.0, terminated=True, truncated=False, info={}),
    ]
    assert all(0 <= int(action) < 4 for action in actions)
    stats = agent.get_stats()
    assert int(stats["nan_guard_trigger_count"]) >= 1


@pytest.mark.skipif(not agents_ppo._TORCH_AVAILABLE, reason="torch unavailable")
def test_ppo_same_seed_same_initial_actions():
    config = agents_ppo.PPOConfig(
        rollout_steps=9999,
        train_interval=9999,
        device="cpu",
    )
    agent_a = agents_ppo.PPOAgent(action_space_n=6, seed=42, config=config)
    agent_b = agents_ppo.PPOAgent(action_space_n=6, seed=42, config=config)

    obs = np.zeros((210, 160, 3), dtype=np.uint8)
    actions_a = [agent_a.step(obs, reward=0.0, terminated=False, truncated=False, info={}) for _ in range(12)]
    actions_b = [agent_b.step(obs, reward=0.0, terminated=False, truncated=False, info={}) for _ in range(12)]
    assert actions_a == actions_b


@pytest.mark.skipif(not agents_ppo._TORCH_AVAILABLE, reason="torch unavailable")
def test_ppo_class_raises_clear_error_when_torch_forced_missing(monkeypatch):
    monkeypatch.setattr(agents_ppo, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(agents_ppo, "_TORCH_IMPORT_ERROR", ImportError("forced missing torch"))
    with pytest.raises(ImportError, match="agent=ppo requires torch"):
        agents_ppo.PPOAgent(action_space_n=4, seed=0, config=agents_ppo.PPOConfig())


@pytest.mark.skipif(not agents_ppo._TORCH_AVAILABLE, reason="torch unavailable")
def test_ppo_uses_prev_applied_action_idx_for_transition_credit():
    agent = agents_ppo.PPOAgent(
        action_space_n=5,
        seed=7,
        config=agents_ppo.PPOConfig(rollout_steps=9999, train_interval=9999, deterministic_actions=False),
    )
    obs = np.zeros((210, 160, 3), dtype=np.uint8)

    first_action = agent.step(
        obs,
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"has_prev_applied_action": False},
    )
    assert 0 <= int(first_action) < 5
    first_probs = np.asarray(agent._prev_probs, dtype=np.float32).copy()  # pylint: disable=protected-access

    agent.step(
        obs,
        reward=1.0,
        terminated=False,
        truncated=False,
        info={"has_prev_applied_action": True, "prev_applied_action_idx": 4},
    )

    assert agent._rollout_actions[0] == 4  # pylint: disable=protected-access
    expected_logprob = math.log(max(float(first_probs[4]), 1e-8))
    assert agent._rollout_logprobs[0] == pytest.approx(expected_logprob, abs=1e-6)  # pylint: disable=protected-access
