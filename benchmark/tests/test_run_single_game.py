from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from benchmark.carmack_runner import CARMACK_SINGLE_RUN_PROFILE, CARMACK_SINGLE_RUN_SCHEMA_VERSION, CarmackRunnerConfig
from benchmark.run_single_game import _FrameFromStepAdapter
from benchmark.run_single_game import build_config_payload
from benchmark.run_single_game import validate_args


def test_validate_args_carmack_requires_frame_skip_1():
    args = Namespace(runner_mode="carmack_compat", frame_skip=4)
    with pytest.raises(ValueError, match=r"carmack_compat requires --frame-skip 1"):
        validate_args(args)


def test_validate_args_standard_allows_frame_skip_not_one():
    args = Namespace(runner_mode="standard", frame_skip=4)
    validate_args(args)


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
        timestamps=0,
        logdir="runs",
    )
    payload = build_config_payload(
        args=args,
        env=_Env(),
        runner_config=CarmackRunnerConfig(total_frames=100, include_timestamps=False),
        run_dir=Path("runs/test"),
    )
    assert payload["single_run_profile"] == CARMACK_SINGLE_RUN_PROFILE
    assert payload["single_run_schema_version"] == CARMACK_SINGLE_RUN_SCHEMA_VERSION
    rc = payload["runner_config"]
    assert rc["runner_mode"] == CARMACK_SINGLE_RUN_PROFILE
    assert rc["single_run_schema_version"] == CARMACK_SINGLE_RUN_SCHEMA_VERSION
    assert rc["action_cadence_mode"] == "agent_owned"
    assert rc["frame_skip_enforced"] == 1


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
    }
    assert "frame_idx" not in call["info"]
    assert "boundary_cause" not in call["info"]
