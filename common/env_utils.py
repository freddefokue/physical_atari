# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared environment creation utilities for Atari agents.
Single source of truth for environment wrapping logic.
"""

from __future__ import annotations

import gymnasium as gym

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from .constants import (
    FRAME_SKIP,
    FRAME_STACK,
    NOOP_MAX,
    OBSERVATION_HEIGHT,
    OBSERVATION_WIDTH,
)


def make_atari_env(
    env_id: str,
    seed: int,
    *,
    frame_skip: int = FRAME_SKIP,
    frame_stack: int = FRAME_STACK,
    noop_max: int = NOOP_MAX,
    obs_height: int = OBSERVATION_HEIGHT,
    obs_width: int = OBSERVATION_WIDTH,
    clip_rewards: bool = True,
    episodic_life: bool = True,
) -> gym.Env:
    """
    Create a Gym Atari environment with standard wrappers.
    
    This is the single source of truth for environment creation across all agents.
    Uses CleanRL-style wrappers for consistency with published benchmarks.
    
    Args:
        env_id: Gymnasium environment ID (e.g., "BreakoutNoFrameskip-v4")
        seed: Random seed for the environment
        frame_skip: Number of frames to skip (action repeat)
        frame_stack: Number of frames to stack for temporal info
        noop_max: Maximum random no-op actions at episode start
        obs_height: Height of preprocessed observation
        obs_width: Width of preprocessed observation
        clip_rewards: Whether to clip rewards to {-1, 0, +1}
        episodic_life: Whether to treat life loss as episode end during training
    
    Returns:
        Wrapped Gymnasium environment ready for training
    """
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    
    if episodic_life:
        env = EpisodicLifeEnv(env)
    
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    if clip_rewards:
        env = ClipRewardEnv(env)
    
    env = gym.wrappers.ResizeObservation(env, (obs_height, obs_width))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env.action_space.seed(seed)
    
    return env
