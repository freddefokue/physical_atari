from __future__ import annotations

import argparse
import sys
import types

import numpy as np
import pytest

from benchmark.agents_delay_target import DelayTargetAdapter
from benchmark.run_multigame import build_agent


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def test_delay_target_adapter_maps_episode_end_and_stats(monkeypatch):
    class _FakeDelayAgent:
        def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
            self.init = {
                "data_dir": data_dir,
                "seed": seed,
                "num_actions": num_actions,
                "total_frames": total_frames,
                "kwargs": kwargs,
            }
            self.calls = []
            self.frame_count = 12
            self.u = 8
            self.episode_number = 3
            self.train_loss_ema = _Scalar(1.5)
            self.avg_error_ema = _Scalar(2.5)
            self.max_error_ema = _Scalar(3.5)
            self.target_ema = _Scalar(4.5)

        def frame(self, observation_rgb8, reward, end_of_episode):
            self.calls.append((observation_rgb8.shape, reward, end_of_episode))
            return 2

    monkeypatch.setitem(sys.modules, "agent_delay_target", types.SimpleNamespace(Agent=_FakeDelayAgent))

    adapter = DelayTargetAdapter(
        data_dir="/tmp/runs",
        seed=7,
        num_actions=4,
        total_frames=200,
        agent_kwargs={"gpu": 1},
    )

    obs = np.zeros((210, 160, 3), dtype=np.uint8)
    action = adapter.step(obs, reward=1.25, terminated=False, truncated=True, info={})
    assert action == 2
    assert adapter._agent.calls[-1] == ((210, 160, 3), 1.25, 1)  # pylint: disable=protected-access

    stats = adapter.get_stats()
    assert stats["decision_steps"] == 1
    assert stats["last_action_idx"] == 2
    assert stats["frame_count"] == 12
    assert stats["u"] == 8
    assert stats["episode_number"] == 3
    assert stats["train_loss_ema"] == 1.5
    assert stats["avg_error_ema"] == 2.5
    assert stats["max_error_ema"] == 3.5
    assert stats["target_ema"] == 4.5


def test_delay_target_adapter_raises_on_invalid_action(monkeypatch):
    class _BadActionAgent:
        def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
            del data_dir, seed, num_actions, total_frames, kwargs

        def frame(self, observation_rgb8, reward, end_of_episode):
            del observation_rgb8, reward, end_of_episode
            return 999

    monkeypatch.setitem(sys.modules, "agent_delay_target", types.SimpleNamespace(Agent=_BadActionAgent))
    adapter = DelayTargetAdapter(data_dir="/tmp/runs", seed=0, num_actions=3, total_frames=10, agent_kwargs={})
    with pytest.raises(ValueError, match="out-of-bounds action"):
        adapter.step(np.zeros((210, 160, 3), dtype=np.uint8), reward=0.0, terminated=False, truncated=False, info={})


def test_build_agent_delay_target_requires_decision_interval_1():
    args = argparse.Namespace(
        agent="delay_target",
        decision_interval=4,
        delay_target_gpu=0,
        delay_target_use_cuda_graphs=1,
        delay_target_load_file=None,
        logdir="./runs/v1",
        seed=0,
    )
    with pytest.raises(ValueError, match="requires --decision-interval 1"):
        build_agent(args, num_actions=18, total_frames=100)

