from __future__ import annotations

import numpy as np
import pytest

from benchmark.agents_legacy_roboatari import LegacyRoboAtariAdapter


def test_legacy_roboatari_adapter_maps_boundary_payloads_and_stats(monkeypatch):
    class _FakeCore:
        frame_count = 12
        training_steps = 4
        epsilon = 0.25
        loss_ema = 1.5

    class _FakeAgent:
        def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
            self.init = {
                "data_dir": data_dir,
                "seed": seed,
                "num_actions": num_actions,
                "total_frames": total_frames,
                "kwargs": kwargs,
            }
            self.calls = []
            self.core = _FakeCore()

        def frame(self, observation_rgb8, reward, end_of_episode):
            self.calls.append((observation_rgb8.shape, float(reward), int(end_of_episode)))
            return 2

    def _import_module(name: str):
        assert name == "algorithms.dqn.agent_dqn"
        return type("_Module", (), {"Agent": _FakeAgent})()

    monkeypatch.setattr("benchmark.agents_legacy_roboatari.importlib.import_module", _import_module)

    adapter = LegacyRoboAtariAdapter(
        agent_name="dqn",
        module_name="algorithms.dqn.agent_dqn",
        import_error_hint="unused",
        data_dir="/tmp/runs",
        seed=7,
        num_actions=4,
        total_frames=200,
        agent_kwargs={"gpu": 1},
    )

    obs = np.zeros((210, 160, 3), dtype=np.uint8)
    transition_obs = np.full((210, 160, 3), fill_value=17, dtype=np.uint8)
    assert (
        adapter.frame(
            obs,
            reward=1.25,
            boundary={
                "terminated": False,
                "truncated": True,
                "boundary_cause": "no_reward_timeout",
                "transition_obs_rgb": transition_obs,
                "reset_obs_rgb": obs,
            },
        )
        == 2
    )
    assert adapter.step(obs, reward=0.5, terminated=True, truncated=False, info={"boundary_cause": "life_loss"}) == 2
    assert adapter._agent.calls == [((210, 160, 3), 1.25, 3), ((210, 160, 3), 0.5, 1)]  # pylint: disable=protected-access

    stats = adapter.get_stats()
    assert stats["decision_steps"] == 2
    assert stats["last_action_idx"] == 2
    assert stats["frame_count"] == 12
    assert stats["training_steps"] == 4
    assert stats["epsilon"] == 0.25
    assert stats["loss_ema"] == 1.5


def test_legacy_roboatari_adapter_maps_runner_boundary_causes_to_legacy_codes(monkeypatch):
    class _FakeAgent:
        def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
            del data_dir, seed, num_actions, total_frames, kwargs
            self.calls = []

        def frame(self, observation_rgb8, reward, end_of_episode):
            del observation_rgb8, reward
            self.calls.append(int(end_of_episode))
            return 0

    monkeypatch.setattr(
        "benchmark.agents_legacy_roboatari.importlib.import_module",
        lambda name: type("_Module", (), {"Agent": _FakeAgent})(),
    )
    adapter = LegacyRoboAtariAdapter(
        agent_name="sac",
        module_name="algorithms.sac.agent_sac",
        import_error_hint="unused",
        data_dir="/tmp/runs",
        seed=0,
        num_actions=3,
        total_frames=10,
        agent_kwargs={},
    )

    obs = np.zeros((210, 160, 3), dtype=np.uint8)
    adapter.frame(obs, 0.0, {"terminated": False, "truncated": True, "boundary_cause": "life_loss"})
    adapter.frame(obs, 0.0, {"terminated": True, "truncated": False, "termination_reason": "terminated"})
    adapter.frame(obs, 0.0, {"terminated": False, "truncated": True, "boundary_cause": "visit_switch"})

    assert adapter._agent.calls == [1, 2, 3]  # pylint: disable=protected-access


def test_legacy_roboatari_adapter_raises_on_invalid_action(monkeypatch):
    class _BadActionAgent:
        def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
            del data_dir, seed, num_actions, total_frames, kwargs

        def frame(self, observation_rgb8, reward, end_of_episode):
            del observation_rgb8, reward, end_of_episode
            return 999

    monkeypatch.setattr(
        "benchmark.agents_legacy_roboatari.importlib.import_module",
        lambda name: type("_Module", (), {"Agent": _BadActionAgent})(),
    )
    adapter = LegacyRoboAtariAdapter(
        agent_name="dqn",
        module_name="algorithms.dqn.agent_dqn",
        import_error_hint="unused",
        data_dir="/tmp/runs",
        seed=0,
        num_actions=3,
        total_frames=10,
        agent_kwargs={},
    )
    with pytest.raises(ValueError, match="out-of-bounds action"):
        adapter.step(np.zeros((210, 160, 3), dtype=np.uint8), reward=0.0, terminated=False, truncated=False, info={})


def test_legacy_roboatari_adapter_import_error_is_actionable(monkeypatch):
    def _raise_import(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("benchmark.agents_legacy_roboatari.importlib.import_module", _raise_import)
    with pytest.raises(ImportError, match="agent=dqn requires roboatari/algorithms/dqn/agent_dqn.py"):
        LegacyRoboAtariAdapter(
            agent_name="dqn",
            module_name="algorithms.dqn.agent_dqn",
            import_error_hint=(
                "agent=dqn requires roboatari/algorithms/dqn/agent_dqn.py and its dependencies "
                "(torch plus the local roboatari package imports)."
            ),
            data_dir="/tmp/runs",
            seed=0,
            num_actions=3,
            total_frames=10,
            agent_kwargs={},
        )
