from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from benchmark.carmack_runner import CARMACK_SINGLE_RUN_PROFILE, CARMACK_SINGLE_RUN_SCHEMA_VERSION, CarmackRunnerConfig
from benchmark.runner import EnvStep
from benchmark.run_single_game import _BBFResetSemanticsEnvAdapter
from benchmark.run_single_game import _CanonicalActionSetEnvAdapter
from benchmark.run_single_game import _FrameFromStepAdapter
from benchmark.run_single_game import _resolve_bbf_eval_reset_settings
from benchmark.run_single_game import _resolve_bbf_runtime_settings
from benchmark.run_single_game import build_agent
from benchmark.run_single_game import build_runtime_fingerprint_payload
from benchmark.run_single_game import build_config_payload
from benchmark.run_single_game import build_run_summary_payload
from benchmark.run_single_game import parse_args
from benchmark.run_single_game import validate_args


def test_validate_args_carmack_requires_frame_skip_1():
    args = Namespace(runner_mode="carmack_compat", frame_skip=4, agent="random", real_time_fps=60.0)
    with pytest.raises(ValueError, match=r"carmack_compat requires --frame-skip 1"):
        validate_args(args)


def test_validate_args_standard_allows_frame_skip_not_one():
    args = Namespace(runner_mode="standard", frame_skip=4, agent="random", real_time_fps=60.0)
    validate_args(args)


def test_validate_args_tinydqn_requires_carmack_mode():
    args = Namespace(runner_mode="standard", frame_skip=1, agent="tinydqn", real_time_fps=60.0)
    with pytest.raises(ValueError, match=r"agent tinydqn currently requires --runner-mode carmack_compat"):
        validate_args(args)


def test_validate_args_tinydqn_requires_positive_decision_interval():
    args = Namespace(
        runner_mode="carmack_compat",
        frame_skip=1,
        agent="tinydqn",
        dqn_decision_interval=0,
        real_time_fps=60.0,
    )
    with pytest.raises(ValueError, match=r"--dqn-decision-interval must be > 0"):
        validate_args(args)


def test_validate_args_ppo_requires_carmack_mode():
    args = Namespace(runner_mode="standard", frame_skip=1, agent="ppo", real_time_fps=60.0)
    with pytest.raises(ValueError, match=r"agent ppo currently requires --runner-mode carmack_compat"):
        validate_args(args)


def test_validate_args_dqn_requires_carmack_mode():
    args = Namespace(runner_mode="standard", frame_skip=1, agent="dqn", real_time_fps=60.0)
    with pytest.raises(ValueError, match=r"agent dqn currently requires --runner-mode carmack_compat"):
        validate_args(args)


def test_validate_args_rainbow_dqn_requires_carmack_mode():
    args = Namespace(runner_mode="standard", frame_skip=1, agent="rainbow_dqn", real_time_fps=60.0)
    with pytest.raises(ValueError, match=r"agent rainbow_dqn currently requires --runner-mode carmack_compat"):
        validate_args(args)


def test_validate_args_sac_requires_carmack_mode():
    args = Namespace(runner_mode="standard", frame_skip=1, agent="sac", real_time_fps=60.0)
    with pytest.raises(ValueError, match=r"agent sac currently requires --runner-mode carmack_compat"):
        validate_args(args)


def test_validate_args_ppo_requires_positive_decision_interval():
    args = Namespace(
        runner_mode="carmack_compat",
        frame_skip=1,
        agent="ppo",
        ppo_decision_interval=0,
        real_time_fps=60.0,
    )
    with pytest.raises(ValueError, match=r"--ppo-decision-interval must be > 0"):
        validate_args(args)


def test_validate_args_bbf_requires_carmack_mode():
    args = Namespace(runner_mode="standard", frame_skip=1, agent="bbf", full_action_space=1, real_time_fps=60.0)
    with pytest.raises(ValueError, match=r"agent bbf currently requires --runner-mode carmack_compat"):
        validate_args(args)


def test_validate_args_bbf_allows_minimal_action_space_without_parity():
    args = Namespace(runner_mode="carmack_compat", frame_skip=1, agent="bbf", full_action_space=0, real_time_fps=60.0)
    validate_args(args)


def test_validate_args_bbf_native_parity_allows_minimal_action_space():
    args = Namespace(
        runner_mode="carmack_compat",
        frame_skip=1,
        agent="bbf",
        full_action_space=0,
        bbf_native_parity=1,
        real_time_fps=60.0,
        bbf_noop_reset_max=30,
        bbf_eval_episodes=0,
        bbf_eval_epsilon=0.001,
        bbf_eval_sticky=0.0,
    )
    validate_args(args)


def test_validate_args_bbf_requires_real_time_mode_off():
    args = Namespace(
        runner_mode="carmack_compat",
        frame_skip=1,
        agent="bbf",
        full_action_space=1,
        real_time_mode=1,
        real_time_fps=60.0,
    )
    with pytest.raises(ValueError, match=r"--agent bbf currently requires --real-time-mode 0"):
        validate_args(args)


def test_validate_args_bbf_native_parity_only_valid_for_bbf():
    args = Namespace(
        runner_mode="carmack_compat",
        frame_skip=1,
        agent="random",
        bbf_native_parity=1,
        real_time_fps=60.0,
        bbf_noop_reset_max=30,
        bbf_eval_episodes=0,
        bbf_eval_epsilon=0.001,
        bbf_eval_sticky=0.0,
    )
    with pytest.raises(ValueError, match=r"--bbf-native-parity is only valid"):
        validate_args(args)


def test_validate_args_bbf_native_reset_semantics_only_valid_for_bbf():
    args = Namespace(
        runner_mode="carmack_compat",
        frame_skip=1,
        agent="random",
        bbf_native_reset_semantics=1,
        real_time_fps=60.0,
        bbf_noop_reset_max=30,
        bbf_eval_episodes=0,
        bbf_eval_epsilon=0.001,
        bbf_eval_sticky=0.0,
    )
    with pytest.raises(ValueError, match=r"--bbf-native-reset-semantics is only valid"):
        validate_args(args)


def test_validate_args_requires_positive_real_time_fps():
    args = Namespace(runner_mode="carmack_compat", frame_skip=1, agent="random", real_time_fps=0.0)
    with pytest.raises(ValueError, match=r"--real-time-fps must be > 0"):
        validate_args(args)


def test_parse_args_accepts_ppo_with_carmack_compat(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--runner-mode",
            "carmack_compat",
            "--frame-skip",
            "1",
            "--agent",
            "ppo",
        ],
    )
    args = parse_args()
    assert args.agent == "ppo"
    assert args.runner_mode == "carmack_compat"
    assert args.ppo_decision_interval == 4


def test_parse_args_accepts_dqn_with_carmack_compat(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--runner-mode",
            "carmack_compat",
            "--frame-skip",
            "1",
            "--agent",
            "dqn",
        ],
    )
    args = parse_args()
    assert args.agent == "dqn"
    assert args.runner_mode == "carmack_compat"
    assert args.roboatari_dqn_gpu == 0


def test_parse_args_accepts_rainbow_dqn_with_carmack_compat(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--runner-mode",
            "carmack_compat",
            "--frame-skip",
            "1",
            "--agent",
            "rainbow_dqn",
        ],
    )
    args = parse_args()
    assert args.agent == "rainbow_dqn"
    assert args.runner_mode == "carmack_compat"
    assert args.rainbow_dqn_gpu == 0
    assert args.rainbow_dqn_learning_rate is None


def test_parse_args_accepts_sac_with_carmack_compat(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--runner-mode",
            "carmack_compat",
            "--frame-skip",
            "1",
            "--agent",
            "sac",
        ],
    )
    args = parse_args()
    assert args.agent == "sac"
    assert args.runner_mode == "carmack_compat"
    assert args.sac_gpu == 0
    assert args.sac_eval_mode == 0
    assert args.sac_learning_rate is None


def test_parse_args_accepts_bbf_with_carmack_compat(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--runner-mode",
            "carmack_compat",
            "--frame-skip",
            "1",
            "--agent",
            "bbf",
            "--full-action-space",
            "1",
        ],
    )
    args = parse_args()
    assert args.agent == "bbf"
    assert args.runner_mode == "carmack_compat"
    assert args.bbf_learning_starts == 2000
    assert args.bbf_learning_rate == pytest.approx(1e-4)


def test_parse_args_accepts_delay_target_sweep_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--agent",
            "delay_target",
            "--delay-target-base-width",
            "112",
            "--delay-target-temperature-log2",
            "-4",
            "--delay-target-greedy-ramp",
            "150000",
            "--delay-target-multisteps-max",
            "32",
            "--delay-target-td-lambda",
            "0.85",
            "--delay-target-train-batch",
            "48",
            "--delay-target-online-batch",
            "6",
            "--delay-target-online-loss-scale",
            "1.75",
            "--delay-target-train-steps",
            "5",
        ],
    )
    args = parse_args()
    assert args.delay_target_base_width == 112
    assert args.delay_target_temperature_log2 == -4
    assert args.delay_target_greedy_ramp == 150000
    assert args.delay_target_multisteps_max == 32
    assert args.delay_target_td_lambda == pytest.approx(0.85)
    assert args.delay_target_train_batch == 48
    assert args.delay_target_online_batch == 6
    assert args.delay_target_online_loss_scale == pytest.approx(1.75)
    assert args.delay_target_train_steps == 5


def test_parse_args_accepts_rainbow_dqn_sweep_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--runner-mode",
            "carmack_compat",
            "--frame-skip",
            "1",
            "--agent",
            "rainbow_dqn",
            "--rainbow-dqn-learning-rate",
            "0.0003",
            "--rainbow-dqn-train-start",
            "25000",
            "--rainbow-dqn-batch-size",
            "64",
            "--rainbow-dqn-buffer-size",
            "200000",
            "--rainbow-dqn-target-update-freq",
            "4000",
            "--rainbow-dqn-n-step",
            "5",
            "--rainbow-dqn-gamma",
            "0.995",
            "--rainbow-dqn-grad-clip",
            "none",
            "--rainbow-dqn-priority-alpha",
            "0.6",
            "--rainbow-dqn-priority-beta",
            "0.7",
        ],
    )
    args = parse_args()
    assert args.rainbow_dqn_learning_rate == pytest.approx(3e-4)
    assert args.rainbow_dqn_train_start == 25000
    assert args.rainbow_dqn_batch_size == 64
    assert args.rainbow_dqn_buffer_size == 200000
    assert args.rainbow_dqn_target_update_freq == 4000
    assert args.rainbow_dqn_n_step == 5
    assert args.rainbow_dqn_gamma == pytest.approx(0.995)
    assert args.rainbow_dqn_grad_clip is None
    assert args.rainbow_dqn_priority_alpha == pytest.approx(0.6)
    assert args.rainbow_dqn_priority_beta == pytest.approx(0.7)


def test_parse_args_accepts_sac_sweep_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--runner-mode",
            "carmack_compat",
            "--frame-skip",
            "1",
            "--agent",
            "sac",
            "--sac-learning-rate",
            "0.0003",
            "--sac-learning-starts",
            "5000",
            "--sac-batch-size",
            "128",
            "--sac-buffer-size",
            "200000",
            "--sac-gradient-steps",
            "4",
            "--sac-tau",
            "0.01",
            "--sac-target-entropy-scale",
            "0.75",
            "--sac-gamma",
            "0.995",
            "--sac-train-freq",
            "2",
            "--sac-frame-skip",
            "6",
        ],
    )
    args = parse_args()
    assert args.sac_learning_rate == pytest.approx(3e-4)
    assert args.sac_learning_starts == 5000
    assert args.sac_batch_size == 128
    assert args.sac_buffer_size == 200000
    assert args.sac_gradient_steps == 4
    assert args.sac_tau == pytest.approx(0.01)
    assert args.sac_target_entropy_scale == pytest.approx(0.75)
    assert args.sac_gamma == pytest.approx(0.995)
    assert args.sac_train_freq == 2
    assert args.sac_frame_skip == 6


def test_parse_args_accepts_bbf_sweep_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--runner-mode",
            "carmack_compat",
            "--frame-skip",
            "1",
            "--agent",
            "bbf",
            "--bbf-learning-rate",
            "0.0003",
            "--bbf-encoder-learning-rate",
            "0.0002",
            "--bbf-spr-weight",
            "7.5",
            "--bbf-jumps",
            "3",
            "--bbf-target-update-tau",
            "0.01",
            "--bbf-update-horizon",
            "5",
            "--bbf-max-update-horizon",
            "9",
            "--bbf-min-gamma",
            "0.98",
            "--bbf-cycle-steps",
            "20000",
            "--bbf-shrink-factor",
            "0.6",
            "--bbf-perturb-factor",
            "0.4",
            "--bbf-shrink-perturb-keys",
            "encoder",
        ],
    )
    args = parse_args()
    assert args.bbf_learning_rate == pytest.approx(3e-4)
    assert args.bbf_encoder_learning_rate == pytest.approx(2e-4)
    assert args.bbf_spr_weight == pytest.approx(7.5)
    assert args.bbf_jumps == 3
    assert args.bbf_target_update_tau == pytest.approx(0.01)
    assert args.bbf_update_horizon == 5
    assert args.bbf_max_update_horizon == 9
    assert args.bbf_min_gamma == pytest.approx(0.98)
    assert args.bbf_cycle_steps == 20000
    assert args.bbf_shrink_factor == pytest.approx(0.6)
    assert args.bbf_perturb_factor == pytest.approx(0.4)
    assert args.bbf_shrink_perturb_keys == "encoder"


def test_parse_args_accepts_bbf_native_parity_switch(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--runner-mode",
            "carmack_compat",
            "--frame-skip",
            "1",
            "--agent",
            "bbf",
            "--bbf-native-parity",
            "1",
        ],
    )
    args = parse_args()
    assert args.agent == "bbf"
    assert args.bbf_native_parity == 1
    assert args.bbf_native_reset_semantics == 0
    assert args.bbf_noop_reset_max == 30


def test_parse_args_accepts_bbf_native_reset_semantics_switch(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_game.py",
            "--game",
            "pong",
            "--runner-mode",
            "carmack_compat",
            "--frame-skip",
            "1",
            "--agent",
            "bbf",
            "--bbf-native-reset-semantics",
            "1",
        ],
    )
    args = parse_args()
    assert args.agent == "bbf"
    assert args.bbf_native_parity == 0
    assert args.bbf_native_reset_semantics == 1


def test_canonical_action_adapter_maps_global_indices_to_local_actions():
    class _DummyEnv:
        def __init__(self) -> None:
            self.action_set = [0, 2, 3]
            self.calls = []

        def reset(self):
            return None

        def lives(self):
            return 3

        def step(self, action_idx: int):
            self.calls.append(int(action_idx))
            return {"ok": True}

    env = _DummyEnv()
    wrapped = _CanonicalActionSetEnvAdapter(env, global_action_set=[0, 1, 2, 3], default_action_idx=1)

    # global idx 0 -> ALE action 0 -> local idx 0
    wrapped.step(0)
    # global idx 2 -> ALE action 2 -> local idx 1
    wrapped.step(2)
    # global idx 1 -> ALE action 1 (illegal for this game) -> fallback to default ALE action 1; missing -> local idx 0
    wrapped.step(1)
    assert env.calls == [0, 1, 0]


def test_resolve_bbf_runtime_settings_native_parity_forces_sticky_zero_and_minimal_actions():
    args = Namespace(
        agent="bbf",
        bbf_native_parity=1,
        bbf_native_reset_semantics=0,
        sticky=0.25,
        full_action_space=1,
        bbf_noop_reset_max=30,
        bbf_fire_reset=1,
    )
    settings = _resolve_bbf_runtime_settings(args)
    assert settings["native_parity_mode"] is True
    assert settings["sticky_effective"] == pytest.approx(0.0)
    assert settings["full_action_space_effective"] is False
    assert settings["action_space_mode"] == "local_minimal"
    assert settings["use_canonical_action_adapter"] is False
    assert settings["native_reset_semantics_enabled"] is True
    assert settings["fire_reset_enabled"] is True


def test_resolve_bbf_runtime_settings_canonical_mode_preserves_full_action_space():
    args = Namespace(
        agent="bbf",
        bbf_native_parity=0,
        bbf_native_reset_semantics=0,
        sticky=0.25,
        full_action_space=1,
        bbf_noop_reset_max=30,
        bbf_fire_reset=1,
    )
    settings = _resolve_bbf_runtime_settings(args)
    assert settings["native_parity_mode"] is False
    assert settings["sticky_effective"] == pytest.approx(0.25)
    assert settings["full_action_space_effective"] is True
    assert settings["action_space_mode"] == "canonical_full"
    assert settings["use_canonical_action_adapter"] is True
    assert settings["native_reset_semantics_enabled"] is False
    assert settings["fire_reset_enabled"] is False


def test_resolve_bbf_runtime_settings_native_reset_semantics_keeps_full_actions():
    args = Namespace(
        agent="bbf",
        bbf_native_parity=0,
        bbf_native_reset_semantics=1,
        sticky=0.25,
        full_action_space=1,
        bbf_noop_reset_max=17,
        bbf_fire_reset=1,
    )
    settings = _resolve_bbf_runtime_settings(args)
    assert settings["native_parity_mode"] is False
    assert settings["native_reset_semantics_requested"] is True
    assert settings["native_reset_semantics_enabled"] is True
    assert settings["full_action_space_effective"] is True
    assert settings["action_space_mode"] == "canonical_full"
    assert settings["use_canonical_action_adapter"] is True
    assert settings["sticky_effective"] == pytest.approx(0.0)
    assert settings["noop_reset_max"] == 17
    assert settings["fire_reset_enabled"] is True


def test_resolve_bbf_runtime_settings_minimal_actions_without_native_reset_semantics():
    args = Namespace(
        agent="bbf",
        bbf_native_parity=0,
        bbf_native_reset_semantics=0,
        sticky=0.15,
        full_action_space=0,
        bbf_noop_reset_max=17,
        bbf_fire_reset=1,
    )
    settings = _resolve_bbf_runtime_settings(args)
    assert settings["runtime_mode"] == "benchmark_standard"
    assert settings["native_reset_semantics_enabled"] is False
    assert settings["full_action_space_effective"] is False
    assert settings["action_space_mode"] == "local_minimal"
    assert settings["use_canonical_action_adapter"] is False
    assert settings["sticky_effective"] == pytest.approx(0.15)
    assert settings["noop_reset_max"] == 0
    assert settings["fire_reset_enabled"] is False


def test_resolve_bbf_eval_reset_settings_uses_runtime_effective_values():
    args = Namespace(
        bbf_noop_reset_max=30,
        bbf_fire_reset=1,
    )
    settings = _resolve_bbf_eval_reset_settings(
        args,
        bbf_runtime={
            "native_reset_semantics_enabled": False,
            "noop_reset_max": 0,
            "fire_reset_enabled": False,
        },
    )
    assert settings["native_reset_semantics_enabled"] is False
    assert settings["noop_reset_max"] == 0
    assert settings["fire_reset_enabled"] is False


def test_resolve_bbf_eval_reset_settings_reports_enabled_mode():
    args = Namespace(
        bbf_noop_reset_max=30,
        bbf_fire_reset=1,
    )
    settings = _resolve_bbf_eval_reset_settings(
        args,
        bbf_runtime={
            "native_reset_semantics_enabled": True,
            "noop_reset_max": 9,
            "fire_reset_enabled": True,
        },
    )
    assert settings["native_reset_semantics_enabled"] is True
    assert settings["noop_reset_max"] == 9
    assert settings["fire_reset_enabled"] is True


def test_bbf_reset_adapter_applies_noop_steps_on_reset():
    class _Env:
        def __init__(self) -> None:
            self.action_set = [0, 1, 2]
            self.reset_calls = 0
            self.step_calls = []

        def get_action_meanings(self):
            return ["NOOP", "FIRE", "UP"]

        def reset(self):
            self.reset_calls += 1
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def lives(self):
            return 3

        def step(self, action_idx: int):
            self.step_calls.append(int(action_idx))
            return EnvStep(
                obs_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                reward=0.0,
                terminated=False,
                truncated=False,
                lives=3,
                termination_reason=None,
            )

    env = _Env()
    wrapped = _BBFResetSemanticsEnvAdapter(env, seed=0, noop_max=5, enable_fire_reset=False)
    wrapped.reset()
    assert env.reset_calls == 1
    assert len(env.step_calls) >= 1
    assert all(call == 0 for call in env.step_calls)


def test_bbf_reset_adapter_applies_fire_reset_when_supported():
    class _Env:
        def __init__(self) -> None:
            self.action_set = [0, 1, 2]
            self.step_calls = []

        def get_action_meanings(self):
            return ["NOOP", "FIRE", "UP"]

        def reset(self):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def lives(self):
            return 3

        def step(self, action_idx: int):
            self.step_calls.append(int(action_idx))
            return EnvStep(
                obs_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                reward=0.0,
                terminated=False,
                truncated=False,
                lives=3,
                termination_reason=None,
            )

    env = _Env()
    wrapped = _BBFResetSemanticsEnvAdapter(env, seed=0, noop_max=0, enable_fire_reset=True)
    wrapped.reset()
    assert wrapped.fire_reset_supported is True
    assert env.step_calls[:2] == [1, 2]


def test_build_config_payload_carmack_marks_agent_owned_cadence():
    class _Env:
        action_set = [0, 1, 2]

    args = Namespace(
        game="breakout",
        seed=0,
        frames=100,
        frame_skip=1,
        delay=0,
        sticky=0.0,
        full_action_space=0,
        life_loss_termination=0,
        agent="random",
        repeat_action_idx=0,
        default_action_idx=0,
        runner_mode="carmack_compat",
        lives_as_episodes=1,
        max_frames_without_reward=1000,
        reset_on_life_loss=0,
        compat_reset_delay_queue_on_reset=0,
        compat_log_every_frames=0,
        compat_log_pulses_every=0,
        compat_log_resets_every=1,
        delay_target_ring_buffer_size=None,
        roboatari_dqn_gpu=0,
        roboatari_dqn_load_file=None,
        rainbow_dqn_gpu=2,
        rainbow_dqn_load_file="/tmp/rainbow_ckpt.pth",
        sac_gpu=0,
        sac_load_file=None,
        sac_eval_mode=0,
        dqn_gamma=0.99,
        dqn_lr=1e-4,
        dqn_buffer_size=10000,
        dqn_batch_size=32,
        dqn_train_every=4,
        dqn_log_train_every=500,
        dqn_target_update=250,
        dqn_eps_start=1.0,
        dqn_eps_end=0.05,
        dqn_eps_decay_frames=200000,
        dqn_replay_min=1000,
        dqn_use_replay=1,
        dqn_device="cpu",
        dqn_decision_interval=1,
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
        ppo_epochs=4,
        ppo_reward_clip=1.0,
        ppo_obs_size=84,
        ppo_frame_stack=4,
        ppo_grayscale=1,
        ppo_normalize_advantages=1,
        ppo_deterministic_actions=0,
        ppo_device="auto",
        ppo_decision_interval=4,
        timestamps=0,
        real_time_mode=1,
        real_time_fps=55.0,
        logdir="runs",
    )
    payload = build_config_payload(
        args=args,
        env=_Env(),
        runner_config=CarmackRunnerConfig(
            total_frames=100,
            include_timestamps=False,
            real_time_mode=True,
            real_time_fps=55.0,
        ),
        run_dir=Path("runs/test"),
    )
    assert payload["single_run_profile"] == CARMACK_SINGLE_RUN_PROFILE
    assert payload["single_run_schema_version"] == CARMACK_SINGLE_RUN_SCHEMA_VERSION
    rc = payload["runner_config"]
    assert rc["runner_mode"] == CARMACK_SINGLE_RUN_PROFILE
    assert rc["single_run_schema_version"] == CARMACK_SINGLE_RUN_SCHEMA_VERSION
    assert rc["action_cadence_mode"] == "agent_owned"
    assert rc["frame_skip_enforced"] == 1
    assert rc["real_time_mode"] is True
    assert rc["real_time_fps"] == pytest.approx(55.0)
    rainbow_cfg = payload["rainbow_dqn_config"]
    assert rainbow_cfg["gpu"] == 2
    assert rainbow_cfg["load_file"] == "/tmp/rainbow_ckpt.pth"
    ppo_cfg = payload["ppo_config"]
    assert ppo_cfg["learning_rate"] == pytest.approx(2.5e-4)
    assert ppo_cfg["gamma"] == pytest.approx(0.99)
    assert ppo_cfg["rollout_steps"] == 128
    assert ppo_cfg["device"] == "auto"
    assert ppo_cfg["decision_interval"] == 4


def test_build_config_payload_records_exposed_sweep_family_fields():
    class _Env:
        action_set = [0, 1, 2]

    args = Namespace(
        game="breakout",
        seed=0,
        frames=100,
        frame_skip=1,
        delay=0,
        sticky=0.0,
        full_action_space=1,
        life_loss_termination=0,
        agent="sac",
        repeat_action_idx=0,
        default_action_idx=0,
        runner_mode="carmack_compat",
        lives_as_episodes=1,
        max_frames_without_reward=1000,
        reset_on_life_loss=0,
        compat_reset_delay_queue_on_reset=0,
        compat_log_every_frames=0,
        compat_log_pulses_every=0,
        compat_log_resets_every=1,
        delay_target_gpu=1,
        delay_target_use_cuda_graphs=0,
        delay_target_load_file="/tmp/delay.model",
        delay_target_ring_buffer_size=32768,
        delay_target_lr_log2=-17,
        delay_target_base_lr_log2=-15,
        delay_target_base_width=112,
        delay_target_temperature_log2=-4,
        delay_target_greedy_ramp=150000,
        delay_target_multisteps_max=32,
        delay_target_td_lambda=0.85,
        delay_target_train_batch=48,
        delay_target_online_batch=6,
        delay_target_online_loss_scale=1.75,
        delay_target_train_steps=5,
        roboatari_dqn_gpu=0,
        roboatari_dqn_load_file=None,
        rainbow_dqn_gpu=2,
        rainbow_dqn_load_file="/tmp/rainbow_ckpt.pth",
        rainbow_dqn_learning_rate=3e-4,
        rainbow_dqn_train_start=25000,
        rainbow_dqn_batch_size=64,
        rainbow_dqn_buffer_size=200000,
        rainbow_dqn_target_update_freq=4000,
        rainbow_dqn_n_step=5,
        rainbow_dqn_gamma=0.995,
        rainbow_dqn_grad_clip=None,
        rainbow_dqn_priority_alpha=0.6,
        rainbow_dqn_priority_beta=0.7,
        sac_gpu=0,
        sac_load_file=None,
        sac_eval_mode=0,
        sac_learning_rate=3e-4,
        sac_learning_starts=5000,
        sac_batch_size=128,
        sac_buffer_size=200000,
        sac_gradient_steps=4,
        sac_tau=0.01,
        sac_target_entropy_scale=0.75,
        sac_gamma=0.995,
        sac_train_freq=2,
        sac_frame_skip=6,
        dqn_gamma=0.99,
        dqn_lr=1e-4,
        dqn_buffer_size=10000,
        dqn_batch_size=32,
        dqn_train_every=4,
        dqn_log_train_every=500,
        dqn_target_update=250,
        dqn_eps_start=1.0,
        dqn_eps_end=0.05,
        dqn_eps_decay_frames=200000,
        dqn_replay_min=1000,
        dqn_use_replay=1,
        dqn_device="cpu",
        dqn_decision_interval=1,
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
        ppo_epochs=4,
        ppo_reward_clip=1.0,
        ppo_obs_size=84,
        ppo_frame_stack=4,
        ppo_grayscale=1,
        ppo_normalize_advantages=1,
        ppo_deterministic_actions=0,
        ppo_device="auto",
        ppo_decision_interval=4,
        bbf_learning_starts=2000,
        bbf_buffer_size=200000,
        bbf_batch_size=32,
        bbf_replay_ratio=64,
        bbf_learning_rate=3e-4,
        bbf_encoder_learning_rate=2e-4,
        bbf_spr_weight=7.5,
        bbf_jumps=3,
        bbf_target_update_tau=0.01,
        bbf_update_horizon=5,
        bbf_max_update_horizon=9,
        bbf_min_gamma=0.98,
        bbf_cycle_steps=20000,
        bbf_shrink_factor=0.6,
        bbf_perturb_factor=0.4,
        bbf_shrink_perturb_keys="encoder",
        bbf_reset_interval=20000,
        bbf_no_resets_after=100000,
        bbf_use_per=1,
        bbf_use_amp=0,
        bbf_torch_compile=0,
        bbf_native_parity=0,
        bbf_native_reset_semantics=0,
        bbf_noop_reset_max=30,
        bbf_fire_reset=1,
        bbf_eval_episodes=0,
        bbf_eval_epsilon=0.001,
        bbf_eval_sticky=0.0,
        bbf_eval_clip_rewards=0,
        timestamps=0,
        real_time_mode=0,
        real_time_fps=55.0,
        logdir="runs",
    )
    payload = build_config_payload(
        args=args,
        env=_Env(),
        runner_config=CarmackRunnerConfig(
            total_frames=100,
            include_timestamps=False,
            real_time_mode=False,
            real_time_fps=55.0,
        ),
        run_dir=Path("runs/test"),
    )
    assert payload["delay_target_config"]["base_width"] == 112
    assert payload["delay_target_config"]["temperature_log2"] == -4
    assert payload["delay_target_config"]["train_steps"] == 5
    assert payload["rainbow_dqn_config"]["learning_rate"] == pytest.approx(3e-4)
    assert payload["rainbow_dqn_config"]["grad_clip"] is None
    assert payload["rainbow_dqn_config"]["priority_alpha"] == pytest.approx(0.6)
    assert payload["sac_config"]["learning_rate"] == pytest.approx(3e-4)
    assert payload["sac_config"]["target_entropy_scale"] == pytest.approx(0.75)
    assert payload["sac_config"]["frame_skip"] == 6
    assert payload["bbf_config"]["learning_rate"] == pytest.approx(3e-4)
    assert payload["bbf_config"]["encoder_learning_rate"] == pytest.approx(2e-4)
    assert payload["bbf_config"]["spr_weight"] == pytest.approx(7.5)
    assert payload["bbf_config"]["jumps"] == 3
    assert payload["bbf_config"]["target_update_tau"] == pytest.approx(0.01)
    assert payload["bbf_config"]["update_horizon"] == 5
    assert payload["bbf_config"]["max_update_horizon"] == 9
    assert payload["bbf_config"]["min_gamma"] == pytest.approx(0.98)
    assert payload["bbf_config"]["cycle_steps"] == 20000
    assert payload["bbf_config"]["shrink_factor"] == pytest.approx(0.6)
    assert payload["bbf_config"]["perturb_factor"] == pytest.approx(0.4)
    assert payload["bbf_config"]["shrink_perturb_keys"] == "encoder"


def test_build_agent_ppo_builds_or_raises_actionable_import_error():
    args = Namespace(
        agent="ppo",
        seed=7,
        runner_mode="carmack_compat",
        ppo_lr=2.5e-4,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_clip_range=0.2,
        ppo_ent_coef=0.01,
        ppo_vf_coef=0.5,
        ppo_max_grad_norm=0.5,
        ppo_rollout_steps=8,
        ppo_train_interval=8,
        ppo_batch_size=4,
        ppo_epochs=1,
        ppo_reward_clip=1.0,
        ppo_obs_size=84,
        ppo_frame_stack=2,
        ppo_grayscale=1,
        ppo_normalize_advantages=1,
        ppo_deterministic_actions=0,
        ppo_device="cpu",
        ppo_decision_interval=4,
        dqn_decision_interval=1,
    )
    try:
        agent = build_agent(args, num_actions=3, total_frames=32)
    except ImportError as exc:
        assert "agent=ppo requires torch" in str(exc)
        return
    assert hasattr(agent, "frame")


def test_build_agent_dqn_builds_or_raises_actionable_import_error():
    args = Namespace(
        agent="dqn",
        seed=7,
        runner_mode="carmack_compat",
        roboatari_dqn_gpu=0,
        roboatari_dqn_load_file=None,
        logdir="./runs",
        dqn_decision_interval=1,
        ppo_decision_interval=1,
    )
    try:
        agent = build_agent(args, num_actions=3, total_frames=32)
    except ImportError as exc:
        assert "agent=dqn requires torch" in str(exc)
        return
    assert hasattr(agent, "frame")


def test_build_agent_rainbow_dqn_builds_or_raises_actionable_import_error():
    args = Namespace(
        agent="rainbow_dqn",
        seed=7,
        runner_mode="carmack_compat",
        rainbow_dqn_gpu=0,
        rainbow_dqn_load_file=None,
        logdir="./runs",
        dqn_decision_interval=1,
        ppo_decision_interval=1,
    )
    try:
        agent = build_agent(args, num_actions=3, total_frames=32)
    except ImportError as exc:
        assert "agent=rainbow_dqn requires torch" in str(exc)
        return
    assert hasattr(agent, "frame")


def test_build_agent_sac_builds_or_raises_actionable_import_error():
    args = Namespace(
        agent="sac",
        seed=7,
        runner_mode="carmack_compat",
        sac_gpu=0,
        sac_load_file=None,
        sac_eval_mode=0,
        logdir="./runs",
        dqn_decision_interval=1,
        ppo_decision_interval=1,
    )
    try:
        agent = build_agent(args, num_actions=3, total_frames=32)
    except ImportError as exc:
        assert "agent=sac requires torch" in str(exc)
        return
    assert hasattr(agent, "frame")


def test_frame_from_step_adapter_does_not_leak_frame_idx():
    class _StepAgent:
        def __init__(self) -> None:
            self.calls = []

        def step(self, obs_rgb, reward, terminated, truncated, info):
            del obs_rgb
            self.calls.append(
                {
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": dict(info),
                }
            )
            return 1

    agent = _StepAgent()
    adapter = _FrameFromStepAdapter(agent)
    adapter.frame(
        obs_rgb=None,
        reward=1.0,
        boundary={
            "terminated": False,
            "truncated": True,
            "end_of_episode_pulse": True,
        },
    )

    assert len(agent.calls) == 1
    call = agent.calls[0]
    assert call["terminated"] is False
    assert call["truncated"] is True
    assert call["info"] == {
        "end_of_episode_pulse": True,
        "has_prev_applied_action": False,
        "prev_applied_action_idx": 0,
        "is_decision_frame": True,
    }
    assert "frame_idx" not in call["info"]
    assert "boundary_cause" not in call["info"]


def test_frame_from_step_adapter_forwards_prev_applied_action_fields():
    class _StepAgent:
        def __init__(self) -> None:
            self.calls = []

        def step(self, obs_rgb, reward, terminated, truncated, info):
            del obs_rgb, reward, terminated, truncated
            self.calls.append(dict(info))
            return 1

    agent = _StepAgent()
    adapter = _FrameFromStepAdapter(agent)
    adapter.frame(
        obs_rgb=None,
        reward=0.0,
        boundary={
            "terminated": False,
            "truncated": False,
            "end_of_episode_pulse": False,
            "has_prev_applied_action": True,
            "prev_applied_action_idx": 3,
        },
    )
    assert agent.calls[0]["has_prev_applied_action"] is True
    assert agent.calls[0]["prev_applied_action_idx"] == 3
    assert agent.calls[0]["is_decision_frame"] is True


def test_frame_from_step_adapter_applies_agent_owned_decision_interval():
    class _StepAgent:
        def __init__(self) -> None:
            self.calls = []

        def step(self, obs_rgb, reward, terminated, truncated, info):
            del obs_rgb, reward, terminated, truncated
            self.calls.append(dict(info))
            return 0

    agent = _StepAgent()
    adapter = _FrameFromStepAdapter(agent, decision_interval=3)
    for _ in range(7):
        adapter.frame(obs_rgb=None, reward=0.0, boundary=False)
    flags = [bool(call["is_decision_frame"]) for call in agent.calls]
    assert flags == [True, False, False, True, False, False, True]


def test_build_run_summary_payload_carmack_includes_schema_markers():
    args = Namespace(runner_mode="carmack_compat")
    summary = {"frames": 10}
    payload = build_run_summary_payload(args, summary)
    assert payload["runner_mode"] == "carmack_compat"
    assert payload["frames"] == 10
    assert payload["single_run_profile"] == CARMACK_SINGLE_RUN_PROFILE
    assert payload["single_run_schema_version"] == CARMACK_SINGLE_RUN_SCHEMA_VERSION


def test_build_config_payload_records_bbf_runtime_mode_fields():
    class _Env:
        action_set = [0, 1, 2]

    args = Namespace(
        game="pong",
        seed=1,
        frames=100,
        frame_skip=1,
        delay=0,
        sticky=0.25,
        full_action_space=1,
        life_loss_termination=0,
        agent="bbf",
        repeat_action_idx=0,
        default_action_idx=0,
        runner_mode="carmack_compat",
        lives_as_episodes=1,
        max_frames_without_reward=1000,
        reset_on_life_loss=0,
        compat_reset_delay_queue_on_reset=0,
        compat_log_every_frames=0,
        compat_log_pulses_every=0,
        compat_log_resets_every=0,
        delay_target_ring_buffer_size=None,
        roboatari_dqn_gpu=0,
        roboatari_dqn_load_file=None,
        rainbow_dqn_gpu=0,
        rainbow_dqn_load_file=None,
        sac_gpu=0,
        sac_load_file=None,
        sac_eval_mode=0,
        dqn_gamma=0.99,
        dqn_lr=1e-4,
        dqn_buffer_size=10000,
        dqn_batch_size=32,
        dqn_train_every=4,
        dqn_log_train_every=500,
        dqn_target_update=250,
        dqn_eps_start=1.0,
        dqn_eps_end=0.05,
        dqn_eps_decay_frames=200000,
        dqn_replay_min=1000,
        dqn_use_replay=1,
        dqn_device="cpu",
        dqn_decision_interval=1,
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
        ppo_epochs=4,
        ppo_reward_clip=1.0,
        ppo_obs_size=84,
        ppo_frame_stack=4,
        ppo_grayscale=1,
        ppo_normalize_advantages=1,
        ppo_deterministic_actions=0,
        ppo_device="auto",
        ppo_decision_interval=4,
        bbf_learning_starts=2000,
        bbf_buffer_size=200000,
        bbf_batch_size=32,
        bbf_replay_ratio=64,
        bbf_reset_interval=20000,
        bbf_no_resets_after=100000,
        bbf_use_per=1,
        bbf_use_amp=0,
        bbf_torch_compile=0,
        bbf_native_parity=1,
        bbf_native_reset_semantics=0,
        bbf_noop_reset_max=30,
        bbf_fire_reset=1,
        bbf_eval_episodes=5,
        bbf_eval_epsilon=0.001,
        bbf_eval_sticky=0.0,
        bbf_eval_clip_rewards=0,
        timestamps=0,
        real_time_mode=0,
        real_time_fps=60.0,
        logdir="runs",
    )
    payload = build_config_payload(
        args=args,
        env=_Env(),
        runner_config=CarmackRunnerConfig(total_frames=100, include_timestamps=False),
        run_dir=Path("runs/test"),
        bbf_runtime={
            "runtime_mode": "parity_preset",
            "native_parity_mode": True,
            "native_reset_semantics_requested": False,
            "native_reset_semantics_enabled": True,
            "action_space_mode": "local_minimal",
            "sticky_requested": 0.25,
            "sticky_effective": 0.0,
            "full_action_space_requested": True,
            "full_action_space_effective": False,
            "use_canonical_action_adapter": False,
            "noop_reset_max_requested": 30,
            "noop_reset_max": 30,
            "fire_reset_requested": True,
            "fire_reset_enabled": True,
            "fire_reset_supported": True,
        },
    )
    assert payload["bbf_runtime"]["runtime_mode"] == "parity_preset"
    assert payload["bbf_runtime"]["native_parity_mode"] is True
    assert payload["bbf_runtime"]["native_reset_semantics_enabled"] is True
    assert payload["bbf_runtime"]["sticky_requested"] == pytest.approx(0.25)
    assert payload["bbf_runtime"]["sticky_effective"] == pytest.approx(0.0)
    assert payload["bbf_runtime"]["full_action_space_requested"] is True
    assert payload["bbf_runtime"]["full_action_space_effective"] is False
    assert payload["bbf_runtime"]["action_space_mode"] == "local_minimal"
    assert payload["bbf_runtime"]["noop_reset_max_requested"] == 30
    assert payload["bbf_runtime"]["noop_reset_max_effective"] == 30
    assert payload["bbf_runtime"]["fire_reset_requested"] is True
    assert payload["bbf_runtime"]["fire_reset_enabled"] is True


def test_build_run_summary_payload_includes_bbf_runtime_and_eval():
    args = Namespace(runner_mode="carmack_compat")
    summary = {"frames": 10}
    payload = build_run_summary_payload(
        args,
        summary,
        bbf_runtime={"native_parity_mode": True, "action_space_mode": "local_minimal"},
        bbf_eval={"episodes": 3, "mean_return": 12.0, "std_return": 1.0},
    )
    assert payload["bbf_runtime"]["native_parity_mode"] is True
    assert payload["bbf_runtime"]["action_space_mode"] == "local_minimal"
    assert payload["bbf_eval"]["episodes"] == 3
    assert payload["bbf_eval"]["mean_return"] == pytest.approx(12.0)


def test_frame_from_step_adapter_get_stats_graceful_fallback():
    class _StepAgent:
        def step(self, obs_rgb, reward, terminated, truncated, info):
            del obs_rgb, reward, terminated, truncated, info
            return 0

        def get_stats(self):
            return {"train_steps": "bad", "epsilon": object()}

    adapter = _FrameFromStepAdapter(_StepAgent())
    payload = adapter.get_stats()
    assert isinstance(payload, dict)
    assert payload["train_steps"] == "bad"


def test_frame_from_step_adapter_get_stats_handles_exceptions():
    class _StepAgent:
        def step(self, obs_rgb, reward, terminated, truncated, info):
            del obs_rgb, reward, terminated, truncated, info
            return 0

        def get_stats(self):
            raise RuntimeError("boom")

    adapter = _FrameFromStepAdapter(_StepAgent())
    assert adapter.get_stats() == {}


def test_build_runtime_fingerprint_payload_contains_required_keys():
    class _Args:
        runner_mode = "carmack_compat"
        game = "breakout"
        seed = 3
        frames = 200

    config_payload = {
        "single_run_profile": CARMACK_SINGLE_RUN_PROFILE,
        "single_run_schema_version": CARMACK_SINGLE_RUN_SCHEMA_VERSION,
        "game": "breakout",
        "seed": 3,
        "frames": 200,
    }
    payload = build_runtime_fingerprint_payload(_Args(), config_payload)
    assert payload["fingerprint_schema_version"] == "runtime_fingerprint_v1"
    assert payload["runner_mode"] == "carmack_compat"
    assert payload["single_run_profile"] == CARMACK_SINGLE_RUN_PROFILE
    assert payload["single_run_schema_version"] == CARMACK_SINGLE_RUN_SCHEMA_VERSION
    assert payload["game"] == "breakout"
    assert payload["seed"] == 3
    assert payload["seed_policy"] == "global_seed_python_numpy_ale"
    assert payload["frames"] == 200
    assert payload["config_sha256_algorithm"] == "sha256"
    assert payload["config_sha256_scope"] == "config_without_runtime_fingerprint"
    assert isinstance(payload["config_sha256"], str) and len(payload["config_sha256"]) == 64
    assert isinstance(payload["rom_sha256"], str) and len(payload["rom_sha256"]) == 64
