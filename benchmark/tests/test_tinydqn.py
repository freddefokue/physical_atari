from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmark.agents_tinydqn import ReplayBuffer, TinyDQNAgent, TinyDQNConfig


def make_obs(value: int) -> np.ndarray:
    return np.full((210, 160, 3), fill_value=int(value) % 255, dtype=np.uint8)


def test_non_decision_frames_return_last_action_without_new_decision():
    agent = TinyDQNAgent(
        action_space_n=6,
        seed=123,
        config=TinyDQNConfig(
            eps_start=1.0,
            eps_end=1.0,
            replay_min_size=10_000,
            use_replay=True,
            device="cpu",
        ),
    )

    first = agent.step(
        make_obs(0),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 0,
            "has_prev_applied_action": False,
            "prev_applied_action_idx": 0,
        },
    )
    repeated = []
    for frame_idx in range(1, 8):
        repeated.append(
            agent.step(
                make_obs(frame_idx),
                reward=0.1,
                terminated=False,
                truncated=False,
                info={
                    "is_decision_frame": False,
                    "global_frame_idx": frame_idx,
                    "has_prev_applied_action": True,
                    "prev_applied_action_idx": first,
                },
            )
        )

    assert repeated
    assert all(action == first for action in repeated)
    assert agent.decision_steps == 1
    assert agent.replay_size == 0

    agent.step(
        make_obs(8),
        reward=0.2,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 8,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": first,
        },
    )
    assert agent.replay_size == 1


def test_replay_buffer_store_and_sample_shapes():
    replay = ReplayBuffer(capacity=3, obs_shape=(1, 8, 8), seed=0)
    for idx in range(4):
        obs = np.full((1, 8, 8), fill_value=idx, dtype=np.uint8)
        nxt = np.full((1, 8, 8), fill_value=idx + 1, dtype=np.uint8)
        replay.add(obs=obs, action=idx % 2, reward=float(idx), next_obs=nxt, done=bool(idx % 2))

    assert len(replay) == 3
    obs_b, act_b, rew_b, nxt_b, done_b = replay.sample(2)
    assert obs_b.shape == (2, 1, 8, 8)
    assert nxt_b.shape == (2, 1, 8, 8)
    assert act_b.shape == (2,)
    assert rew_b.shape == (2,)
    assert done_b.shape == (2,)
    assert obs_b.dtype == np.uint8
    assert nxt_b.dtype == np.uint8
    assert act_b.dtype == np.int64
    assert rew_b.dtype == np.float32
    assert done_b.dtype == np.float32


def test_terminated_and_truncated_are_stored_as_done():
    agent = TinyDQNAgent(
        action_space_n=4,
        seed=7,
        config=TinyDQNConfig(
            eps_start=0.0,
            eps_end=0.0,
            replay_min_size=10_000,
            use_replay=True,
            device="cpu",
        ),
    )

    agent.step(
        make_obs(1),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 0,
            "has_prev_applied_action": False,
            "prev_applied_action_idx": 0,
        },
    )
    agent.step(
        make_obs(2),
        reward=1.25,
        terminated=True,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 1,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 2,
        },
    )
    assert agent.replay_size == 1
    first_idx = (agent._replay._ptr - 1) % agent._replay.capacity  # pylint: disable=protected-access
    assert float(agent._replay._dones[first_idx]) == 1.0  # pylint: disable=protected-access
    assert float(agent._replay._rewards[first_idx]) == 1.25  # pylint: disable=protected-access
    assert int(agent._replay._actions[first_idx]) == 2  # pylint: disable=protected-access

    agent.step(
        make_obs(3),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 2,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 0,
        },
    )
    agent.step(
        make_obs(4),
        reward=2.5,
        terminated=False,
        truncated=True,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 3,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 1,
        },
    )
    assert agent.replay_size == 3
    second_idx = (agent._replay._ptr - 1) % agent._replay.capacity  # pylint: disable=protected-access
    assert float(agent._replay._dones[second_idx]) == 1.0  # pylint: disable=protected-access
    assert float(agent._replay._rewards[second_idx]) == 2.5  # pylint: disable=protected-access
    assert int(agent._replay._actions[second_idx]) == 1  # pylint: disable=protected-access


def test_deterministic_actions_with_fixed_seed_and_eps_zero():
    config = TinyDQNConfig(
        eps_start=0.0,
        eps_end=0.0,
        replay_min_size=10_000,
        use_replay=True,
        device="cpu",
    )
    agent_a = TinyDQNAgent(action_space_n=5, seed=11, config=config)
    agent_b = TinyDQNAgent(action_space_n=5, seed=11, config=config)

    actions_a = []
    actions_b = []
    for frame_idx in range(20):
        info = {
            "is_decision_frame": (frame_idx % 3 == 0),
            "global_frame_idx": frame_idx,
            "has_prev_applied_action": frame_idx > 0,
            "prev_applied_action_idx": frame_idx % 5,
        }
        reward = float(frame_idx % 4) * 0.25
        terminated = frame_idx in (9, 17)
        truncated = frame_idx in (14,)
        obs = make_obs(10 + frame_idx)
        actions_a.append(agent_a.step(obs, reward, terminated, truncated, info))
        actions_b.append(agent_b.step(obs, reward, terminated, truncated, info))

    assert actions_a == actions_b


def test_transition_action_comes_from_prev_applied_action_info():
    agent = TinyDQNAgent(
        action_space_n=6,
        seed=3,
        config=TinyDQNConfig(
            eps_start=0.0,
            eps_end=0.0,
            replay_min_size=10_000,
            use_replay=True,
            device="cpu",
        ),
    )

    agent.step(
        make_obs(1),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 0,
            "has_prev_applied_action": False,
            "prev_applied_action_idx": 0,
        },
    )
    agent.step(
        make_obs(2),
        reward=0.5,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": False,
            "global_frame_idx": 1,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 4,
        },
    )
    assert agent.replay_size == 0
    agent.step(
        make_obs(3),
        reward=0.25,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 2,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 4,
        },
    )
    assert agent.replay_size == 1
    idx = (agent._replay._ptr - 1) % agent._replay.capacity  # pylint: disable=protected-access
    assert int(agent._replay._actions[idx]) == 4  # pylint: disable=protected-access


def test_transition_action_uses_first_seen_applied_action_within_interval():
    agent = TinyDQNAgent(
        action_space_n=6,
        seed=8,
        config=TinyDQNConfig(
            eps_start=0.0,
            eps_end=0.0,
            replay_min_size=10_000,
            use_replay=True,
            device="cpu",
        ),
    )

    # Start interval at decision boundary.
    agent.step(
        make_obs(1),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 0,
            "default_action_idx": 0,
            "has_prev_applied_action": False,
            "prev_applied_action_idx": 0,
        },
    )
    # Interval frame 1 sees applied action 2.
    agent.step(
        make_obs(2),
        reward=0.1,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": False,
            "global_frame_idx": 1,
            "default_action_idx": 0,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 2,
        },
    )
    # Interval frame 2 sees a different applied action (should NOT overwrite).
    agent.step(
        make_obs(3),
        reward=0.2,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": False,
            "global_frame_idx": 2,
            "default_action_idx": 0,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 5,
        },
    )
    # Next decision finalizes interval.
    agent.step(
        make_obs(4),
        reward=0.3,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 3,
            "default_action_idx": 0,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 5,
        },
    )

    assert agent.replay_size == 1
    idx = (agent._replay._ptr - 1) % agent._replay.capacity  # pylint: disable=protected-access
    assert int(agent._replay._actions[idx]) == 2  # pylint: disable=protected-access


def test_replay_disabled_keeps_buffer_empty():
    agent = TinyDQNAgent(
        action_space_n=5,
        seed=0,
        config=TinyDQNConfig(
            eps_start=0.0,
            eps_end=0.0,
            replay_min_size=10_000,
            use_replay=False,
            device="cpu",
        ),
    )

    for frame_idx in range(12):
        agent.step(
            make_obs(30 + frame_idx),
            reward=0.1,
            terminated=(frame_idx == 7),
            truncated=False,
            info={
                "is_decision_frame": (frame_idx % 3 == 0),
                "global_frame_idx": frame_idx,
                "has_prev_applied_action": frame_idx > 0,
                "prev_applied_action_idx": frame_idx % 5,
            },
        )
    assert agent.replay_size == 0


def test_requires_applied_action_feedback_keys_in_info():
    agent = TinyDQNAgent(
        action_space_n=4,
        seed=0,
        config=TinyDQNConfig(use_replay=False, device="cpu"),
    )
    with pytest.raises(KeyError):
        agent.step(
            make_obs(1),
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"is_decision_frame": True, "global_frame_idx": 0},
        )


def test_first_interval_without_action_label_is_skipped():
    agent = TinyDQNAgent(
        action_space_n=5,
        seed=2,
        config=TinyDQNConfig(
            eps_start=0.0,
            eps_end=0.0,
            replay_min_size=10_000,
            use_replay=True,
            device="cpu",
        ),
    )

    agent.step(
        make_obs(1),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 0,
            "has_prev_applied_action": False,
            "prev_applied_action_idx": 0,
        },
    )
    agent.step(
        make_obs(2),
        reward=0.4,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 1,
            "has_prev_applied_action": False,
            "prev_applied_action_idx": 0,
        },
    )
    assert agent.replay_size == 0


def test_no_double_train_on_single_decision_boundary():
    agent = TinyDQNAgent(
        action_space_n=4,
        seed=4,
        config=TinyDQNConfig(
            eps_start=0.0,
            eps_end=0.0,
            replay_min_size=1,
            batch_size=1,
            train_every_decisions=1,
            use_replay=True,
            device="cpu",
        ),
    )

    # Decision 1: initializes pending interval, no transition finalized yet.
    agent.step(
        make_obs(1),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 0,
            "default_action_idx": 0,
            "has_prev_applied_action": False,
            "prev_applied_action_idx": 0,
        },
    )
    assert agent._train_steps == 0  # pylint: disable=protected-access

    # Decision 2: finalizes exactly one transition and should train exactly once.
    agent.step(
        make_obs(2),
        reward=1.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 1,
            "default_action_idx": 0,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 2,
        },
    )
    assert agent.replay_size == 1
    assert agent._train_steps == 1  # pylint: disable=protected-access

    # Decision 3: one more finalized transition => exactly one more train step.
    agent.step(
        make_obs(3),
        reward=0.5,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 2,
            "default_action_idx": 0,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 1,
        },
    )
    assert agent.replay_size == 2
    assert agent._train_steps == 2  # pylint: disable=protected-access


def test_train_every_decisions_uses_boundary_aligned_counter():
    agent = TinyDQNAgent(
        action_space_n=4,
        seed=5,
        config=TinyDQNConfig(
            eps_start=0.0,
            eps_end=0.0,
            replay_min_size=1,
            batch_size=1,
            train_every_decisions=4,
            use_replay=True,
            device="cpu",
        ),
    )

    # Six decision frames (every frame is a decision frame). A transition is
    # finalized on steps 2..6. With train cadence driven by finalized
    # transitions, only finalized transition #4 should trigger a train step.
    for frame_idx in range(6):
        agent.step(
            make_obs(100 + frame_idx),
            reward=1.0,
            terminated=False,
            truncated=False,
            info={
                "is_decision_frame": True,
                "global_frame_idx": frame_idx,
                "default_action_idx": 0,
                "has_prev_applied_action": frame_idx > 0,
                "prev_applied_action_idx": frame_idx % 4,
            },
        )

    assert agent.replay_size == 5
    assert agent._train_steps == 1  # pylint: disable=protected-access


def test_target_sync_uses_decision_counter_even_without_training():
    agent = TinyDQNAgent(
        action_space_n=4,
        seed=6,
        config=TinyDQNConfig(
            eps_start=0.0,
            eps_end=0.0,
            replay_min_size=10_000,
            train_every_decisions=1000,
            target_update_decisions=2,
            use_replay=False,
            device="cpu",
        ),
    )

    def first_weight(module):
        return next(module.parameters()).detach().clone()

    with torch.no_grad():
        next(agent._online.parameters()).fill_(1.0)  # pylint: disable=protected-access

    # Decision 1: no sync yet (counter=1).
    agent.step(
        make_obs(1),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 0,
            "default_action_idx": 0,
            "has_prev_applied_action": False,
            "prev_applied_action_idx": 0,
        },
    )
    assert not torch.allclose(first_weight(agent._online), first_weight(agent._target))  # pylint: disable=protected-access

    # Decision 2: sync should happen (counter=2), even with replay/training disabled.
    agent.step(
        make_obs(2),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 1,
            "default_action_idx": 0,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 0,
        },
    )
    assert torch.allclose(first_weight(agent._online), first_weight(agent._target))  # pylint: disable=protected-access

    with torch.no_grad():
        next(agent._online.parameters()).fill_(2.0)  # pylint: disable=protected-access

    # Decision 3: no sync (counter=3).
    agent.step(
        make_obs(3),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 2,
            "default_action_idx": 0,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 0,
        },
    )
    assert not torch.allclose(first_weight(agent._online), first_weight(agent._target))  # pylint: disable=protected-access

    # Decision 4: sync again (counter=4).
    agent.step(
        make_obs(4),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "is_decision_frame": True,
            "global_frame_idx": 3,
            "default_action_idx": 0,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 0,
        },
    )
    assert torch.allclose(first_weight(agent._online), first_weight(agent._target))  # pylint: disable=protected-access
