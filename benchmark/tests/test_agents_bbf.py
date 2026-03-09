from __future__ import annotations

import argparse
import sys
import types

import numpy as np
import pytest

from benchmark.agents_bbf import BBFAgentAdapter, BBFAdapterConfig
from benchmark.runner import EnvStep
from benchmark.run_multigame import build_agent as build_multigame_agent
from benchmark.run_single_game import build_agent as build_single_game_agent


class _FakeBox:
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class _FakeDiscrete:
    def __init__(self, n: int):
        self.n = int(n)


class _FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeBBFAgent:
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.act_calls = []
        self.step_calls = []
        self._actions = [2, 3, 4, 5]

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.uint8)
        self.act_calls.append(obs.copy())
        idx = min(len(self.act_calls) - 1, len(self._actions) - 1)
        return int(self._actions[idx] % max(1, int(self.action_space.n)))

    def step(self, next_observation, reward, terminated, truncated, info=None):
        del info
        obs = np.asarray(next_observation, dtype=np.uint8)
        self.step_calls.append(
            {
                "next_obs": obs.copy(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        return {"loss": 1.0}

    def get_stats(self):
        return {"train_steps": len(self.step_calls)}


class _FakeImportRouter:
    def __init__(self) -> None:
        self.agent = None

    def __call__(self, name: str):
        if name == "gymnasium":
            return types.SimpleNamespace(spaces=types.SimpleNamespace(Box=_FakeBox, Discrete=_FakeDiscrete))
        if name == "agent_bbf":
            agent_router = self

            class _BoundFakeAgent(_FakeBBFAgent):
                def __init__(self, observation_space, action_space, config):
                    super().__init__(observation_space, action_space, config)
                    agent_router.agent = self

            return types.SimpleNamespace(
                Agent=_BoundFakeAgent,
                AgentConfig=_FakeConfig,
                FRAME_SKIP=4,
                ATARI_CANONICAL_ACTIONS=18,
            )
        raise ModuleNotFoundError(name)


def test_bbf_adapter_reconstructs_cadence_and_reward_clipping(monkeypatch):
    router = _FakeImportRouter()
    monkeypatch.setattr("benchmark.agents_bbf.importlib.import_module", router)

    adapter = BBFAgentAdapter(
        seed=0,
        num_actions=6,
        total_frames=40,
        config=BBFAdapterConfig(),
    )
    assert router.agent is not None

    actions = []
    rewards = [1.0, 1.0, -2.0, 0.0, 0.5, 0.5, 0.5, 0.5]
    for idx, reward in enumerate(rewards):
        obs = np.full((16, 14, 3), fill_value=idx * 10, dtype=np.uint8)
        actions.append(
            adapter.frame(
                obs_rgb=obs,
                reward=float(reward),
                boundary={"terminated": False, "truncated": False, "end_of_episode_pulse": False},
            )
        )

    assert actions == [2, 2, 2, 3, 3, 3, 3, 4]
    assert len(router.agent.step_calls) == 2
    assert router.agent.step_calls[0]["reward"] == pytest.approx(0.0)
    assert router.agent.step_calls[1]["reward"] == pytest.approx(1.0)
    assert float(np.mean(router.agent.step_calls[0]["next_obs"])) == pytest.approx(30.0, abs=1.0)
    assert router.agent.act_calls[0].shape == (84, 84)


def test_bbf_adapter_flushes_interval_on_episode_pulse(monkeypatch):
    router = _FakeImportRouter()
    monkeypatch.setattr("benchmark.agents_bbf.importlib.import_module", router)

    adapter = BBFAgentAdapter(seed=0, num_actions=6, total_frames=20, config=BBFAdapterConfig())

    first = adapter.frame(
        obs_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        reward=0.0,
        boundary={"terminated": False, "truncated": False, "end_of_episode_pulse": False},
    )
    second = adapter.frame(
        obs_rgb=np.full((10, 10, 3), fill_value=80, dtype=np.uint8),
        reward=2.0,
        boundary={"terminated": False, "truncated": False, "end_of_episode_pulse": True},
    )

    assert first == 2
    assert second == 3
    assert len(router.agent.step_calls) == 1
    assert router.agent.step_calls[0]["reward"] == pytest.approx(1.0)
    assert router.agent.step_calls[0]["terminated"] is False
    assert router.agent.step_calls[0]["truncated"] is True


def test_bbf_adapter_max_pools_raw_rgb_before_grayscale_resize(monkeypatch):
    router = _FakeImportRouter()
    monkeypatch.setattr("benchmark.agents_bbf.importlib.import_module", router)

    adapter = BBFAgentAdapter(seed=0, num_actions=6, total_frames=20, config=BBFAdapterConfig())

    red = np.zeros((12, 12, 3), dtype=np.uint8)
    red[..., 0] = 255
    green = np.zeros((12, 12, 3), dtype=np.uint8)
    green[..., 1] = 255
    black = np.zeros((12, 12, 3), dtype=np.uint8)

    adapter.frame(black, 0.0, {"terminated": False, "truncated": False, "end_of_episode_pulse": False})
    adapter.frame(black, 0.0, {"terminated": False, "truncated": False, "end_of_episode_pulse": False})
    adapter.frame(red, 0.0, {"terminated": False, "truncated": False, "end_of_episode_pulse": False})
    adapter.frame(green, 0.0, {"terminated": False, "truncated": False, "end_of_episode_pulse": False})

    assert len(router.agent.step_calls) == 1
    first_transition_obs = router.agent.step_calls[0]["next_obs"]
    # If max-pooling is done on raw RGB first, red+green combine before grayscale:
    # gray ~= 227 instead of max(gray(red)=76, gray(green)=149)=149.
    assert float(np.mean(first_transition_obs)) == pytest.approx(227.0, abs=2.0)


def test_bbf_adapter_uses_transition_obs_for_step_and_reset_obs_for_next_action(monkeypatch):
    router = _FakeImportRouter()
    monkeypatch.setattr("benchmark.agents_bbf.importlib.import_module", router)

    adapter = BBFAgentAdapter(seed=0, num_actions=6, total_frames=20, config=BBFAdapterConfig())

    black = np.zeros((12, 12, 3), dtype=np.uint8)
    red = np.zeros((12, 12, 3), dtype=np.uint8)
    red[..., 0] = 255
    green = np.zeros((12, 12, 3), dtype=np.uint8)
    green[..., 1] = 255

    # Prime interval and held action.
    adapter.frame(black, 0.0, {"terminated": False, "truncated": False, "end_of_episode_pulse": False})

    # Runner-style reset pulse: positional obs is reset frame, boundary carries both.
    adapter.frame(
        green,
        1.0,
        {
            "terminated": True,
            "truncated": False,
            "end_of_episode_pulse": True,
            "transition_obs_rgb": red,
            "reset_obs_rgb": green,
        },
    )

    assert len(router.agent.step_calls) == 1
    # Transition uses terminal observation path (max over black/red => red => gray ~= 76).
    assert float(np.mean(router.agent.step_calls[0]["next_obs"])) == pytest.approx(76.0, abs=2.0)
    assert router.agent.step_calls[0]["terminated"] is True
    assert router.agent.step_calls[0]["truncated"] is False
    # Next action is selected from reset observation (green => gray ~= 149).
    assert float(np.mean(router.agent.act_calls[-1])) == pytest.approx(149.0, abs=2.0)


def test_build_agent_bbf_single_game_uses_adapter(monkeypatch):
    class _Cfg:
        def __init__(self, **kwargs):
            self._kwargs = dict(kwargs)

        def as_dict(self):
            return dict(self._kwargs)

    class _Adapter:
        def __init__(self, *, seed, num_actions, total_frames, config, **kwargs):
            self.seed = int(seed)
            self.num_actions = int(num_actions)
            self.total_frames = int(total_frames)
            self.config = config
            self.extra = dict(kwargs)

        def frame(self, obs_rgb, reward, boundary):
            del obs_rgb, reward, boundary
            return 0

    monkeypatch.setitem(sys.modules, "benchmark.agents_bbf", types.SimpleNamespace(BBFAgentAdapter=_Adapter, BBFAdapterConfig=_Cfg))

    args = argparse.Namespace(
        agent="bbf",
        runner_mode="carmack_compat",
        seed=5,
        bbf_learning_starts=10,
        bbf_buffer_size=100,
        bbf_batch_size=8,
        bbf_replay_ratio=4,
        bbf_reset_interval=0,
        bbf_no_resets_after=50,
        bbf_use_per=1,
        bbf_use_amp=0,
        bbf_torch_compile=0,
        dqn_decision_interval=1,
        ppo_decision_interval=1,
    )

    agent = build_single_game_agent(args, num_actions=18, total_frames=123)
    assert isinstance(agent, _Adapter)
    assert agent.seed == 5
    assert agent.num_actions == 18
    assert agent.total_frames == 123
    assert agent.extra["parity_mode"] is False
    assert agent.extra["action_space_mode"] == "canonical_full"


def test_build_agent_bbf_multigame_import_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "benchmark.agents_bbf", types.ModuleType("benchmark.agents_bbf"))

    args = argparse.Namespace(
        agent="bbf",
        seed=0,
        bbf_learning_starts=2000,
        bbf_buffer_size=200000,
        bbf_batch_size=32,
        bbf_replay_ratio=64,
        bbf_reset_interval=20000,
        bbf_no_resets_after=100000,
        bbf_use_per=1,
        bbf_use_amp=0,
        bbf_torch_compile=0,
    )

    with pytest.raises(ImportError, match=r"agent=bbf requires benchmark.agents_bbf"):
        build_multigame_agent(args, num_actions=18, total_frames=200)


def test_bbf_adapter_get_stats_exposes_stable_bbf_scalars(monkeypatch):
    router = _FakeImportRouter()
    monkeypatch.setattr("benchmark.agents_bbf.importlib.import_module", router)

    adapter = BBFAgentAdapter(
        seed=0,
        num_actions=6,
        total_frames=40,
        config=BBFAdapterConfig(learning_starts=2000),
    )
    assert router.agent is not None

    class _Replay:
        def __init__(self) -> None:
            self.add_count = 2500

        def num_elements(self):
            return 1234

    router.agent.replay_buffer = _Replay()
    router.agent.training_steps = 77
    router.agent.grad_steps = 44
    router.agent.global_step = 301

    adapter._raw_frames = 12
    adapter._decision_steps = 5
    adapter._transition_steps = 3
    adapter._last_train_stats = {
        "loss": np.float32(2.3456),
        "spr_loss": np.float32(0.6789),
        "avg_q": np.float32(1.25),
        "gamma": np.float32(0.97654),
    }

    stats = adapter.get_stats()

    assert stats["phase"] == "training"
    assert stats["bbf_parity_mode"] is False
    assert stats["action_space_mode"] == "canonical_full"
    assert stats["full_action_space"] is True
    assert stats["replay_size"] == 1234
    assert stats["replay_add_count"] == 2500
    assert stats["buffer_size"] == 200000
    assert stats["learning_starts"] == 2000
    assert stats["train_steps"] == 77
    assert stats["grad_steps"] == 44
    assert stats["global_step"] == 301
    assert stats["decision_steps"] == 5
    assert stats["transition_steps"] == 3
    assert stats["last_train_loss"] == pytest.approx(2.3456, rel=1e-4)
    assert stats["last_train_spr_loss"] == pytest.approx(0.6789, rel=1e-4)
    assert stats["last_train_avg_q"] == pytest.approx(1.25, rel=1e-4)
    assert stats["last_train_gamma"] == pytest.approx(0.97654, rel=1e-4)


def test_bbf_adapter_phase_stays_warmup_until_update_gate_or_grad_steps(monkeypatch):
    router = _FakeImportRouter()
    monkeypatch.setattr("benchmark.agents_bbf.importlib.import_module", router)

    adapter = BBFAgentAdapter(
        seed=0,
        num_actions=6,
        total_frames=40,
        config=BBFAdapterConfig(learning_starts=2000, buffer_size=50000),
    )
    assert router.agent is not None

    class _Replay:
        def __init__(self) -> None:
            self.add_count = 2501

        def num_elements(self):
            return 2501

    router.agent.replay_buffer = _Replay()
    router.agent.training_steps = 2001
    router.agent.grad_steps = 0
    router.agent.learning_ready_step = 3000
    router.agent.update_period = 4

    stats = adapter.get_stats()
    assert stats["phase"] == "warmup"


def test_bbf_adapter_evaluate_returns_native_style_summary(monkeypatch):
    router = _FakeImportRouter()
    monkeypatch.setattr("benchmark.agents_bbf.importlib.import_module", router)

    adapter = BBFAgentAdapter(seed=0, num_actions=6, total_frames=40, config=BBFAdapterConfig())

    class _EvalEnv:
        def __init__(self) -> None:
            self.action_set = list(range(6))
            self._steps = 0

        def reset(self):
            self._steps = 0
            return np.zeros((12, 12, 3), dtype=np.uint8)

        def step(self, action_idx: int):
            del action_idx
            self._steps += 1
            terminated = bool(self._steps >= 4)
            return EnvStep(
                obs_rgb=np.full((12, 12, 3), fill_value=self._steps, dtype=np.uint8),
                reward=1.0,
                terminated=terminated,
                truncated=False,
                lives=3,
                termination_reason="scripted_end" if terminated else None,
            )

    summary = adapter.evaluate(_EvalEnv(), episodes=3, epsilon=0.0, seed=1, clip_rewards=False)
    assert summary is not None
    assert summary["episodes"] == 3
    assert summary["mean_return"] == pytest.approx(4.0)
    assert summary["std_return"] == pytest.approx(0.0)
    assert len(summary["episode_returns"]) == 3
    assert len(summary["episode_lengths"]) == 3
